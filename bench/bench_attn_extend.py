"""ABBA benchmark of the SM90 FP8 extend kernel (Gemma-4 global, head_dim 512) against its fallbacks.

  python bench/bench_attn_extend.py
  python bench/bench_attn_extend.py --case 1:8192:8192 4:4096:2048 --rounds 5

A case is bs:prefix:q_len (every sequence attends prefix + q_len keys, causal over its q_len rows).
Sides, timed in rounds of A B C C B A in one process, one CUDA-event pair per call, L2 flushed before
every call:
  A  dequant: ``wide_kv.attend`` (fp8 pages -> bf16, FA4 bf16), what an FP8 KV cache ran before
  B  green:   ``attn_decode.select_extend`` on the fp8 cache
  C  fa4-bf16: FA4 on a bf16 copy of the cache (what --kv-cache-dtype auto serves), for reference
Reports min / p25 / median per side and the A/B and C/B speedups at p25 (the GPU may be shared).
"""

from __future__ import annotations

import argparse
import math

import torch

from green_attention import attn_decode, wide_kv

FP8 = torch.float8_e4m3fn
HD, KVH, QH = 512, 4, 32
DEFAULT_CASES = ["1:0:8192", "1:8192:8192", "1:24576:8192", "4:4096:2048", "8:0:1024", "32:8192:10"]


def build(bs: int, prefix: int, qlen: int, page: int, seed: int = 0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    ctx = prefix + qlen
    pages_per = -(-ctx // page)
    n_pages = bs * pages_per + 1
    k = (torch.randn((n_pages, page, KVH, HD), generator=g, device=dev) * 2).clamp(-400, 400).to(FP8)
    v = (torch.randn((n_pages, page, KVH, HD), generator=g, device=dev) * 2).clamp(-400, 400).to(FP8)
    perm = torch.randperm(n_pages, generator=g, device=dev).to(torch.int32)
    pt = perm[: bs * pages_per].view(bs, pages_per).contiguous()
    q = (torch.randn((bs * qlen, QH, HD), generator=g, device=dev) * HD ** -0.5).to(torch.bfloat16)
    return dict(
        q=q, k_cache=k, v_cache=v, page_table=pt,
        cache_seqlens=torch.full((bs,), ctx, dtype=torch.int32, device=dev),
        cu_seqlens_q=torch.arange(bs + 1, device=dev, dtype=torch.int32) * qlen,
        max_seqlen_q=qlen, softmax_scale=1.0, causal=True, window_size=(-1, -1), softcap=0.0,
        return_softmax_lse=False, num_splits=0,
        k_descale=torch.full((1,), 0.5, device=dev).expand(bs, KVH),
        v_descale=torch.full((1,), 0.5, device=dev).expand(bs, KVH),
    )


def time_calls(fn, n, flush):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]  # us


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="*", default=DEFAULT_CASES, help="bs:prefix:q_len")
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10, help="calls per slot")
    args = ap.parse_args()

    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache as fa4

    attn_decode.ensure_ready()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    print(f"device {torch.cuda.get_device_name()}  page_size={args.page}  head_dim {HD} q_heads {QH} kv_heads {KVH}")
    print(f"{'case':>14} | {'dequant us':>22} | {'green us':>22} | {'fa4-bf16 us':>22} | {'TFLOP/s':>7} | "
          f"{'A/B':>5} {'C/B':>5}")
    print(f"{'':>14} | {'min   p25   med':>22} | {'min   p25   med':>22} | {'min   p25   med':>22} |")
    ab, cb = [], []
    for spec in args.case:
        bs, prefix, qlen = (int(x) for x in spec.split(":"))
        kw = build(bs, prefix, qlen, args.page)
        green = attn_decode.select_extend(kw)
        assert callable(green), green
        kw_c = {k: v for k, v in kw.items() if k not in ("k_descale", "v_descale")}
        kw_c.update(k_cache=kw["k_cache"].to(torch.bfloat16) * 0.5, v_cache=kw["v_cache"].to(torch.bfloat16) * 0.5)
        sides = {
            "A": lambda: wide_kv.attend(fa4, dict(kw), "extend"),
            "B": lambda: green(**kw),
            "C": lambda: fa4(**kw_c),
        }
        for f in sides.values():
            f()
        torch.cuda.synchronize()
        t = {s: [] for s in sides}
        for _ in range(args.rounds):
            for s in "ABCCBA":
                t[s] += time_calls(sides[s], args.iters, flush)
        flops = 4 * bs * qlen * (prefix + (qlen + 1) / 2) * QH * HD
        p25 = {s: pct(v, 0.25) for s, v in t.items()}
        ab.append(p25["A"] / p25["B"])
        cb.append(p25["C"] / p25["B"])
        cols = " | ".join(f"{min(t[s]):>6.0f} {p25[s]:>6.0f} {pct(t[s], .5):>6.0f}" for s in "ABC")
        print(f"{spec:>14} | {cols} | {flops / p25['B'] / 1e6:>7.1f} | {ab[-1]:>5.2f} {cb[-1]:>5.2f}", flush=True)
        del kw, kw_c, sides
        torch.cuda.empty_cache()
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    print(f"geomean p25 speedup: vs dequant {gm(ab):.3f}x, vs fa4-bf16 {gm(cb):.3f}x over {len(ab)} cases")


if __name__ == "__main__":
    main()

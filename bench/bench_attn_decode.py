"""ABBA benchmark of the SM90 FP8 decode/verify kernel against the FA4 fallback.

  python bench/bench_attn_decode.py --abba                      # full grid
  python bench/bench_attn_decode.py --bs 32 --ctx 32768 --qlen 1 --abba

Grid: bs {1,8,32,51} x ctx {2k,8k,32k,128k} x q_len {1,4}. Each config runs rounds of
A B B A (A = FA4 fallback, B = green) in one process, one CUDA-event pair per call with an L2 flush
before every call (cold L2; --warm-l2 disables the flush). Reports the median over all calls of
each side. KV bytes per call (the roofline numerator) = sum(ctx) * 2 KB (K+V, 4 heads, 256, e4m3);
the q/out traffic is < 0.1 % of that and ignored. Configs whose KV pool exceeds --max-gb are skipped.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess

import torch

from green_attention import attn_decode

HD, KVH, QH = 256, 4, 24
FP8 = torch.float8_e4m3fn
HBM_GBPS = 3350.0
BYTES_PER_TOKEN = 2 * KVH * HD  # K + V, e4m3


def smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover
        return f"nvidia-smi unavailable ({exc})"


def build(bs: int, ctx: int, qlen: int, page: int, seed: int = 0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    pages_per = -(-ctx // page)
    n_pages = bs * pages_per + 1
    k = torch.empty((n_pages, page, KVH, HD), dtype=FP8, device=dev)
    v = torch.empty_like(k)
    base = (torch.randn((1 << 24,), generator=g, device=dev) * 1.0).clamp(-400, 400).to(FP8)
    for t in (k, v):
        flat = t.view(torch.uint8).view(-1)
        step = base.numel()
        bu = base.view(torch.uint8)
        for i in range(0, flat.numel(), step):
            n = min(step, flat.numel() - i)
            flat[i:i + n].copy_(bu[:n])
    perm = torch.randperm(n_pages, generator=g, device=dev).to(torch.int32)
    pt = perm[: bs * pages_per].view(bs, pages_per).contiguous()
    q = torch.randn((bs * qlen, QH, HD), generator=g, device=dev).clamp(-8, 8).to(FP8)
    cu = (torch.arange(bs + 1, device=dev, dtype=torch.int32) * qlen)
    return dict(
        q=q, k_cache=k, v_cache=v, page_table=pt,
        cache_seqlens=torch.full((bs,), ctx, dtype=torch.int32, device=dev),
        cu_seqlens_q=cu, max_seqlen_q=qlen, softmax_scale=HD ** -0.5, causal=True,
        window_size=(-1, -1), softcap=0.0, return_softmax_lse=False, num_splits=0,
        k_descale=torch.full((1,), 0.5, device=dev).expand(bs, KVH),
        v_descale=torch.full((1,), 0.5, device=dev).expand(bs, KVH),
        max_seqlen_k=ctx,
    )


def time_calls(fn, kw, n, flush):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn(**kw)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]  # us


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="*", default=[1, 8, 32, 51])
    ap.add_argument("--ctx", type=int, nargs="*", default=[2048, 8192, 32768, 131072])
    ap.add_argument("--qlen", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3, help="ABBA rounds per config")
    ap.add_argument("--iters", type=int, default=20, help="calls per slot")
    ap.add_argument("--warm-l2", action="store_true")
    ap.add_argument("--max-gb", type=float, default=14.0)
    ap.add_argument("--abba", action="store_true", help="accepted for the documented command line; ABBA is always on")
    args = ap.parse_args()

    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache as fa4

    attn_decode.ensure_ready()
    flush = None if args.warm_l2 else torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    print(f"device {torch.cuda.get_device_name()}  page_size={args.page}  L2 {'warm' if args.warm_l2 else 'flushed (256 MB memset)'}")
    print(f"nvidia-smi before: {smi()}")
    hdr = f"{'bs':>3} {'ctx':>7} {'q':>2} {'splits':>6} | {'fa4 us':>9} {'GB/s':>7} {'%BW':>5} | {'green us':>9} {'GB/s':>7} {'%BW':>5} | {'speedup':>7} | {'p10-p90 fa4':>14} {'p10-p90 green':>14}"
    print(hdr)
    speedups = []
    skipped = []
    for qlen in args.qlen:
        for ctx in args.ctx:
            for bs in args.bs:
                gb = bs * ctx * BYTES_PER_TOKEN / 1e9
                if gb > args.max_gb:
                    skipped.append((bs, ctx, qlen, gb))
                    continue
                kw = build(bs, ctx, qlen, args.page)
                green = attn_decode.select(kw)
                assert callable(green), green
                splits = attn_decode.plan_splits(bs, ctx, attn_decode._S.sm_count[0])
                fa4(**kw)
                green(**kw)
                torch.cuda.synchronize()
                a, b = [], []
                for _ in range(args.rounds):  # A B B A
                    a += time_calls(fa4, kw, args.iters, flush)
                    b += time_calls(green, kw, args.iters, flush)
                    b += time_calls(green, kw, args.iters, flush)
                    a += time_calls(fa4, kw, args.iters, flush)
                ta, tb = statistics.median(a), statistics.median(b)
                bytes_ = bs * ctx * BYTES_PER_TOKEN
                sp = ta / tb
                speedups.append(sp)
                print(
                    f"{bs:>3} {ctx:>7} {qlen:>2} {splits:>6} | {ta:>9.1f} {bytes_ / ta / 1e3:>7.0f} {100 * bytes_ / ta / 1e3 / HBM_GBPS:>5.1f} |"
                    f" {tb:>9.1f} {bytes_ / tb / 1e3:>7.0f} {100 * bytes_ / tb / 1e3 / HBM_GBPS:>5.1f} | {sp:>7.3f} |"
                    f" {pct(a, .1):>6.0f}-{pct(a, .9):<6.0f} {pct(b, .1):>6.0f}-{pct(b, .9):<6.0f}",
                    flush=True,
                )
                del kw
                torch.cuda.empty_cache()
    if speedups:
        gm = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"geomean speedup over {len(speedups)} configs: {gm:.3f}x")
    for bs, ctx, qlen, gb in skipped:
        print(f"skipped bs={bs} ctx={ctx} q={qlen}: KV pool {gb:.1f} GB > --max-gb {args.max_gb}")
    print(f"nvidia-smi after: {smi()}")


if __name__ == "__main__":
    main()

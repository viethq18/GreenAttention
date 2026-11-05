"""Gemma-4 sliding-layer FP8 prefill (hd256, 32 q / 16 kv heads, window (1023, 0)):
FA4 SM90 native fp8 vs the fork's SGLANG_FA4_SM90_FP8_PREFILL_VIA_FA3 route (FA3 fp8).

Calls go through the fork's own _route_fp8_prefill_to_fa3 wrapper with the keyword set
FlashAttentionBackend.forward_extend passes, so the route decision is the production one.
ABBA rounds, min/p25 per side; relerr of FA3 vs FA4 output.

  PYTHONPATH=<sglang fork>/python python bench/bench_fa3_sliding_prefill.py
"""
import functools
import statistics

import torch

from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache as fa3
from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache as fa4
from sglang.srt.layers.attention.flashattention_backend import _route_fp8_prefill_to_fa3

H, HK, D, W = 32, 16, 256, (1023, 0)
dev = "cuda"
calls = {"fa3": 0}


@functools.wraps(fa3)  # the route reads FA3's keyword set through inspect.signature
def fa3_counted(**kw):
    calls["fa3"] += 1
    return fa3(**kw)


routed = _route_fp8_prefill_to_fa3(fa4, fa3_counted)


def timeit(fn, iters):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def case(name, q_lens, prefix_lens, page_size, rounds=6):
    torch.manual_seed(0)
    B = len(q_lens)
    seq = [q + p for q, p in zip(q_lens, prefix_lens)]
    pages = [(s + page_size - 1) // page_size for s in seq]
    n_pages = sum(pages) + 8
    kc = torch.empty(n_pages, page_size, HK, D, device=dev, dtype=torch.bfloat16).normal_(0, 0.5).to(torch.float8_e4m3fn)
    vc = torch.empty(n_pages, page_size, HK, D, device=dev, dtype=torch.bfloat16).normal_(0, 0.5).to(torch.float8_e4m3fn)
    perm = torch.randperm(n_pages, device=dev).to(torch.int32)
    pt = torch.zeros(B, max(pages), dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages):
        pt[b, :n] = perm[off:off + n]
        off += n
    T = sum(q_lens)
    q = (torch.randn(T, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.tensor(seq).cumsum(0)), dtype=torch.int32, device=dev)
    seqlens = torch.tensor(seq, dtype=torch.int32, device=dev)
    desc = torch.full((B, HK), 1.0, device=dev)
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    kw = dict(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=seqlens, cu_seqlens_q=cu_q,
              cu_seqlens_k_new=cu_k, max_seqlen_q=max(q_lens), softmax_scale=1.0, causal=True,
              window_size=W, softcap=0.0, return_softmax_lse=False, num_splits=0, out=out,
              k_descale=desc, v_descale=desc)
    sides = {"fa4": lambda: fa4(**kw), "via_fa3": lambda: routed(**kw)}
    o4 = sides["fa4"]().clone()
    before = calls["fa3"]
    o3 = sides["via_fa3"]().clone()
    took_fa3 = calls["fa3"] > before
    err = ((o3.float() - o4.float()).norm() / o4.float().norm()).item()
    for f in sides.values():
        f()
    torch.cuda.synchronize()
    iters = max(3, min(50, int(2e5 / max(T, 1))))
    t = {k: [] for k in sides}
    for _ in range(rounds):
        for k in ("fa4", "via_fa3", "via_fa3", "fa4"):
            t[k].append(timeit(sides[k], iters))
    r = {}
    for k, v in t.items():
        v.sort()
        r[k] = (v[0], statistics.quantiles(v, n=4)[0])
    print(f"{name:28s} ps={page_size:3d} T={T:6d}  fa4 min/p25 {r['fa4'][0]:8.1f}/{r['fa4'][1]:8.1f} us  "
          f"via_fa3 {r['via_fa3'][0]:8.1f}/{r['via_fa3'][1]:8.1f} us  speedup(min) {r['fa4'][0] / r['via_fa3'][0]:5.2f}x  "
          f"routed={took_fa3} relerr={err:.4f}", flush=True)


print(torch.cuda.get_device_name(), torch.__version__)
for ps in (1, 16):
    case("fresh 64", [64], [0], ps)
    case("fresh 512", [512], [0], ps)
    case("fresh 2048", [2048], [0], ps)
    case("fresh 8192", [8192], [0], ps)
    case("chunk 8192 @ 16k prefix", [8192], [16384], ps)
    case("chunk 2048 @ 30k prefix", [2048], [30000], ps)
    case("batch 8x1024 mixed prefix", [1024] * 8, [0, 500, 3000, 0, 9000, 100, 0, 20000], ps)
    case("batch 32x256 (livekit-ish)", [256] * 32, [i * 700 for i in range(32)], ps)

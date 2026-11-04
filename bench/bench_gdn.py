"""ABBA bench: GreenAttention GDN decode/verify vs the stock Triton kernels.

Usage (on the H100 box, from the repo root):
    python bench/bench_gdn.py [--abba] [--reps 40]

CUDA-graph-replayed launches, L2 flushed before each replay, CUDA events around
the replay. Reports p10/median/p90 microseconds, GB/s of the minimum moved
bytes (state read+write for decode; checkpoint read + ring r/w for verify),
% of the 3.35 TB/s HBM spec, speedup and geomean. nvidia-smi is recorded next
to every table (the GPU is shared: numbers under contention).
"""

import argparse
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "op_test", "sm90"))

import gdn_common as gc  # noqa: E402

from green_attention.gdn_recurrent import (  # noqa: E402
    H,
    HV,
    K,
    SCALE,
    V,
    _select_packed,
    _run_verify,
    _verify_can_use,
    _workspace,
    module,
)
import green_attention.gdn as gdn_mod  # noqa: E402

# The runtime context is only published inside a server; force the gate.
gdn_mod._green_on_sm90 = lambda: True

SPECS = (3.35e12,)  # H100 SXM HBM3
FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def gpu_state():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return out


def timed(fn, reps=40):
    """Median/p10/p90 us of a CUDA-graph replay with a flushed L2."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        FLUSH.zero_()
        ev0.record()
        g.replay()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1) * 1e3)
    ts.sort()
    n = len(ts)
    return ts[n // 2], ts[max(0, n // 10)], ts[min(n - 1, (9 * n) // 10)]


def decode_bytes(call, B):
    state = call["ssm_states"]
    qkv = call["mixed_qkv"].numel() * 2 + call["a"].numel() * 2 + call["b"].numel() * 2
    st = state[0].numel() * state.element_size()
    return 2 * st * B + qkv  # state r+w for the batch rows + activations


def verify_bytes(call):
    B = call["replay_indices"].numel()
    st = call["checkpoint_state"][0].numel() * call["checkpoint_state"].element_size()
    ring = 0
    for t in ("d_cache", "rawv_cache", "k_cache", "rawk_cache", "g_cache"):
        ring += call[t][0].numel() * call[t].element_size()
    qkv = sum(call[t].numel() * 2 for t in ("q", "k", "v", "a", "b")) + call["out"].numel() * 2
    return st * B + ring * B + qkv  # state read (h0) + ring r/w + activations


def bench_decode():
    rows = []
    for sd in (torch.float32, torch.bfloat16):
        for B in (1, 8, 32, 51):
            # Each side gets its own buffers and no per-call clone (a pool clone
            # would dominate the timing). Decode updates the state in place across
            # replays; the cost does not depend on the values.
            call_t, call_g = gc.make_decode(B, sd), gc.make_decode(B, sd)
            green_sel = _select_packed(call_g)
            assert callable(green_sel), green_sel
            call = call_t

            def triton(call=call_t):
                return gc.triton_packed_decode(call)

            def green(call=call_g, sel=green_sel):
                return sel(**call)

            t = timed(triton)
            g = timed(green)
            by = decode_bytes(call, B)
            rows.append((str(sd), B, 1, t, g, by))
    return rows


def bench_verify():
    rows = []
    for sd in (torch.float32, torch.bfloat16):
        for B in (1, 8, 32, 51):
            call = gc.make_verify([4] * B, state_dtype=sd)
            reason = _verify_can_use(call)
            if reason is not None:
                print(f"  (verify declines {sd} B={B}: {reason})")
                continue

            call_t, call_g = gc.clone_call(call), gc.clone_call(call)

            def triton(call=call_t):
                gc.triton_verify(call)

            def green(call=call_g):
                _run_verify(call)

            t = timed(triton)
            g = timed(green)
            rows.append((str(sd), B, 4, t, g, verify_bytes(call)))
    return rows


def report(title, rows):
    print(f"\n== {title}   [nvidia-smi: {gpu_state()}]")
    print(f"{'state':<16}{'B':>4}{'qT':>3}   {'Triton us':>10}{'Green us':>10}{'speedup':>8}{'GB/s':>8}{'%BW':>7}")
    sp = []
    for sd, B, qt, t, g, by in rows:
        med_t = t[0]
        med_g = g[0]
        speed = med_t / med_g
        bw = by / med_g / 1e3  # bytes / us -> GB/s
        sp.append(speed)
        print(
            f"{sd:<16}{B:>4}{qt:>3}   "
            f"{med_t:>10.1f}{med_g:>10.1f}{speed:>8.2f}{bw:>8.0f}{bw / 3350 * 100:>6.0f}%"
            f"   (p10/p90 t {t[1]:.1f}/{t[2]:.1f} g {g[1]:.1f}/{g[2]:.1f})"
        )
    if sp:
        gm = statistics.geometric_mean(sp)
        print(f"geomean speedup: {gm:.3f}x over {len(sp)} cases")
        return gm
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0), "SM90 only"
    module()
    _workspace(torch.device("cuda"))
    print(f"gpu: {torch.cuda.get_device_name()}   [nvidia-smi: {gpu_state()}]")
    globals()["timed"].__globals__["FLUSH"] = FLUSH
    gd = report("GDN packed decode (graph-replayed, L2 flushed)", bench_decode())
    gv = report("GDN replayssm verify (fp32 state, B>=16 only; graph-replayed)", bench_verify())
    if gd and gv:
        print(f"overall geomean: {(gd * gv) ** 0.5:.3f}x")


if __name__ == "__main__":
    main()

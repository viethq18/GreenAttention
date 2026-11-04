"""Microbench of the replayssm FLUSH path (capacity fold + cursor commit) at the production shape.

Usage (on the H100 box, from the repo root):
    python bench/bench_gdn_flush.py [--layers 4] [--reps 30] [--batches 1 8 32 51]

What is timed (all CUDA-graph replay, L2 flushed before every replay, CUDA events around it, cases
interleaved ABBA across ``--rounds`` rounds; median / p10 / p90 microseconds):

  commit  production flush path: ``commit_gdn_replayssm_spec`` + ``commit_gdn_replayssm_circular``
          (cursor advance, compact capacity fold of ALL layers in one launch, cursor reset), exactly as called by
          ``HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`` after every DFLASH verify step.
  legacy  per-layer ``gdn_replayssm_spec_decode(launch_mode="flush")`` (exact fold + flush-output); NOT reachable in
          the production configuration (see docs/gdn-contract.md section 7), kept for completeness.

The commit grid is ``(V/BV, B, HV * num_layers)`` so every cost scales linearly with the number of layers; to keep
memory at a few GB the pool is built with ``--layers`` layers (default 4) and the result is scaled by 48 / layers
(linearity is checked with ``--check-linear``). The ``reset`` baseline (three tiny copies that re-arm the cursors,
needed because the commit clears ``is_flush``) is subtracted.

If ``green_attention.gdn_flush`` exists its kernel is benched against Triton in the same ABBA loop.
"""

import argparse
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "op_test", "sm90"))

import gdn_common as gc  # noqa: E402

H, HV, K, V, L = gc.H, gc.HV, gc.K, gc.V, gc.L
PROD_LAYERS = 48
FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
SPEC_LEN = 4
ACCEPT = 2  # accepted tokens per row in the timed step (the fold length is decided by write_pos)
FLUSH_WP = 24  # write_pos before the commit of a flushing row: 24 + 2 = 26 -> new_wp + 8 > 32
IDLE_WP = 8


def gpu_state():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()


def pct(ts, q):
    ts = sorted(ts)
    return ts[min(len(ts) - 1, int(q * (len(ts) - 1) + 0.5))]


class Pool:
    """Production-shaped speculative-state pool restricted to ``layers`` layers and ``B`` rows."""

    def __init__(self, B: int, layers: int, seed: int = 0):
        g = gc._gen(seed)
        slots = B + 1  # pool slot ``B`` is the PAD/scratch row
        self.B, self.layers = B, layers
        self.temporal = gc._randn((layers, slots, HV, V, K), g, torch.float32, 0.05)
        self.d = gc._randn((layers, slots, HV, L, V), g, torch.bfloat16, 0.1)
        self.k = gc._randn((layers, slots, H, L, K), g, torch.bfloat16, 0.1)
        self.g = -torch.rand((layers, slots, HV, L), generator=g, device="cuda") * 0.1
        self.rawv = gc._randn((layers, slots, HV, L, V), g, torch.bfloat16, 0.01)
        self.rawk = gc._randn((layers, slots, H, L, K), g, torch.bfloat16, 0.01)
        self.beta = torch.rand((layers, slots, HV, L), generator=g, device="cuda")
        # request-keyed ring slot and checkpoint slot are different permutations, like the pool
        self.state_idx = gc._distinct(B, slots, g).to(torch.int32)
        self.replay_idx = gc._distinct(B, slots, g).to(torch.int64)
        self.accept = torch.full((B,), ACCEPT, dtype=torch.int64, device="cuda")
        self.write_pos = torch.zeros(slots + 1, dtype=torch.int32, device="cuda")
        self.cache_base = torch.zeros(slots + 1, dtype=torch.int32, device="cuda")
        self.is_flush = torch.zeros(slots + 1, dtype=torch.int8, device="cuda")
        # activations for the legacy per-layer path
        T = B * SPEC_LEN
        self.q = gc._randn((T, H, K), g, torch.bfloat16)
        self.kk = gc._randn((T, H, K), g, torch.bfloat16)
        self.v = gc._randn((T, HV, V), g, torch.bfloat16)
        self.a = gc._randn((T, HV), g, torch.bfloat16)
        self.b = gc._randn((T, HV), g, torch.bfloat16)
        self.A_log = torch.rand(HV, generator=g, device="cuda")
        self.dt_bias = torch.rand(HV, generator=g, device="cuda")
        self.out = torch.empty(T, HV, V, dtype=torch.bfloat16, device="cuda")
        self.qsl = torch.arange(0, T + 1, SPEC_LEN, dtype=torch.int32, device="cuda")

    def cursors(self, n_flush: int):
        """Cursor templates: the first ``n_flush`` rows are one step before a capacity flush."""
        wp = torch.zeros_like(self.write_pos)
        rows = self.replay_idx
        wp[rows] = IDLE_WP
        wp[rows[:n_flush]] = FLUSH_WP
        cb = torch.zeros_like(self.cache_base)
        cb[rows] = (torch.arange(self.B, device="cuda", dtype=torch.int32) * 5) & (L - 1)
        return wp, cb, torch.zeros_like(self.is_flush)


def triton_commit(p: Pool):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_circular,
        commit_gdn_replayssm_spec,
    )

    commit_gdn_replayssm_spec(
        write_pos=p.write_pos, cache_base=p.cache_base, is_flush=p.is_flush, num_accepted=p.accept,
        replay_indices=p.replay_idx, max_cache_len=L, max_spec_len=SPEC_LEN, fold_every_commit=False,
        null_block_id=-1,
    )
    commit_gdn_replayssm_circular(
        checkpoint_state=p.temporal, d_cache=p.d, k_cache=p.k, g_cache=p.g,
        d_residual_cache=p.rawv, k_residual_cache=p.rawk, state_batch_indices=p.state_idx,
        replay_indices=p.replay_idx, write_pos=p.write_pos, cache_base=p.cache_base, is_flush=p.is_flush,
        accept_lens=p.accept, mamba_track_indices=None, mamba_steps_to_track=None, null_block_id=-1,
    )


def green_commit(p: Pool):
    """Same cursor advance as production (stock Triton), then the green replacement of the circular commit."""
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import commit_gdn_replayssm_spec

    from green_attention import gdn_flush

    commit_gdn_replayssm_spec(
        write_pos=p.write_pos, cache_base=p.cache_base, is_flush=p.is_flush, num_accepted=p.accept,
        replay_indices=p.replay_idx, max_cache_len=L, max_spec_len=SPEC_LEN, fold_every_commit=False,
        null_block_id=-1,
    )
    gdn_flush.run(dict(
        checkpoint_state=p.temporal, d_cache=p.d, k_cache=p.k, g_cache=p.g,
        d_residual_cache=p.rawv, k_residual_cache=p.rawk, state_batch_indices=p.state_idx,
        replay_indices=p.replay_idx, write_pos=p.write_pos, cache_base=p.cache_base, is_flush=p.is_flush,
        accept_lens=p.accept, mamba_track_indices=None, mamba_steps_to_track=None, null_block_id=-1,
    ))


def triton_legacy_layer(p: Pool):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import gdn_replayssm_spec_decode

    gdn_replayssm_spec_decode(
        q=p.q, k=p.kk, v=p.v, a=p.a, b=p.b, A_log=p.A_log, dt_bias=p.dt_bias,
        checkpoint_state=p.temporal[0], d_cache=p.d[0], k_cache=p.k[0], g_cache=p.g[0],
        rawv_cache=p.rawv[0], rawk_cache=p.rawk[0], beta_cache=p.beta[0], out=p.out,
        query_start_loc=p.qsl, ssm_state_indices=p.state_idx, replay_indices=p.replay_idx,
        write_pos=p.write_pos, cache_base=p.cache_base, is_flush=p.is_flush, max_cache_len=L,
        max_spec_len=SPEC_LEN, scale=K**-0.5, use_qk_l2norm_in_kernel=True, null_block_id=-1,
        launch_mode="flush",
    )


def make_graph(fn):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    return g


def replay_us(g, ev0, ev1):
    FLUSH.zero_()
    ev0.record()
    g.replay()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1e3


def abba(graphs: dict, rounds: int, reps: int):
    """Interleave A B B A ... over all graphs; returns {name: [us]}."""
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    names = list(graphs)
    res = {n: [] for n in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order + order[::-1]:
            for _ in range(reps):
                res[n].append(replay_us(graphs[n], ev0, ev1))
    return res


def eager_wall_us(fn, reps=20):
    """Wall time of eager calls (CPU launch + GPU), synchronised each rep; also CPU-only enqueue time."""
    import time

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    wall, enq = [], []
    for _ in range(reps):
        FLUSH.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        enq.append((t1 - t0) * 1e6)
        wall.append((t2 - t0) * 1e6)
    return statistics.median(wall), statistics.median(enq)


def bench_batch(B, layers, reps, rounds, fractions, want_legacy, want_green=True):
    p = Pool(B, layers)
    n_cases = {"idle(0)": 0}
    for f in fractions:
        n_cases[f"{f:.0%}"] = max(1, round(f * B))
    n_cases[f"all({B})"] = B
    # de-duplicate equal flush counts
    seen, uniq = set(), {}
    for k, n in n_cases.items():
        if n not in seen:
            seen.add(n)
            uniq[k] = n
    n_cases = uniq
    tmpl = {name: p.cursors(n) for name, n in n_cases.items()}

    def arm(t):
        p.write_pos.copy_(t[0])
        p.cache_base.copy_(t[1])
        p.is_flush.copy_(t[2])

    graphs = {"reset": make_graph(lambda: arm(tmpl["idle(0)"]))}
    for name in n_cases:
        t = tmpl[name]
        graphs[f"commit:{name}"] = make_graph(lambda t=t: (arm(t), triton_commit(p)))
        if want_green:
            graphs[f"green:{name}"] = make_graph(lambda t=t: (arm(t), green_commit(p)))
    if want_legacy:
        # flush rows get is_flush=1 up front (the legacy kernels key on the flag, read-only)
        for name, n in n_cases.items():
            if n in (0,):
                continue
            wp, cb, fl = tmpl[name]
            fl = fl.clone()
            fl[p.replay_idx[:n]] = 1
            graphs[f"legacy:{name}"] = make_graph(
                lambda wp=wp, cb=cb, fl=fl: (p.write_pos.copy_(wp), p.cache_base.copy_(cb), p.is_flush.copy_(fl), triton_legacy_layer(p))
            )
    res = abba(graphs, rounds, reps)
    base = statistics.median(res["reset"])
    scale = PROD_LAYERS / layers
    # Triton's grid is (V/BV, B, HV*layers): every cost, idle CTAs included, is linear in layers -> scale.
    # Green's idle cost (prep + an empty persistent grid) does not depend on layers: only the part above idle scales.
    green_idle = statistics.median(res["green:idle(0)"]) - base if want_green else 0.0
    rows = []
    for name, ts in res.items():
        if name == "reset":
            continue
        med = statistics.median(ts) - base
        lo, hi = pct(ts, 0.1) - base, pct(ts, 0.9) - base
        kind, case = name.split(":")
        if kind == "green":
            med, lo, hi = (green_idle + (x - green_idle) * scale for x in (med, lo, hi))
            rows.append((kind, case, n_cases[case], med, lo, hi))
            continue
        s = scale if kind == "commit" else 1.0  # legacy is a single layer already
        rows.append((kind, case, n_cases[case], med * s, lo * s, hi * s))
    # eager walls (one call with all rows idle, one with all flushing)
    eager = {}
    eager_names = {"idle": "idle(0)", "all": next(k for k, n in n_cases.items() if n == B)}
    fns = {"triton": triton_commit}
    if want_green:
        fns["green"] = green_commit
    for fname, fn in fns.items():
        for label, name in eager_names.items():
            arm(tmpl[name])
            w, e = eager_wall_us(lambda name=name, fn=fn: (arm(tmpl[name]), fn(p)))
            eager[f"{fname}-{label}"] = (w, e)
    return rows, base, eager, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 51])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    ap.add_argument("--check-linear", action="store_true")
    ap.add_argument("--verify-us", type=float, nargs=2, default=[80.7, 115.1], help="green verify us/layer at B=32/51")
    args = ap.parse_args()

    print(f"[nvidia-smi before: {gpu_state()}]")
    summary = {}
    for B in args.batches:
        rows, base, eager, _ = bench_batch(B, args.layers, args.reps, args.rounds, args.fractions, not args.no_legacy, not args.no_green)
        print(f"\n== B={B} layers={args.layers} (commit scaled x{PROD_LAYERS / args.layers:.1f} to 48 layers; green: only the part above its idle "
              f"cost is scaled; reset baseline {base:.1f} us subtracted)   [nvidia-smi: {gpu_state()}]")
        print(f"{'path':8s} {'rows flushing':>14s} {'n':>3s} {'median us':>10s} {'p10':>8s} {'p90':>8s}")
        for kind, case, n, med, lo, hi in rows:
            print(f"{kind:8s} {case:>14s} {n:3d} {med:10.1f} {lo:8.1f} {hi:8.1f}")
        print(f"eager commit wall (sync each call, {args.layers} layers, incl. reset copies): "
              + ", ".join(f"{k}: wall {w:.0f} us / cpu-enqueue {e:.0f} us" for k, (w, e) in eager.items()))
        summary[B] = rows
        if args.check_linear and B == args.batches[-1]:
            for nl in (2, 8):
                r2, b2, _, _ = bench_batch(B, nl, args.reps, 2, [], False, False)
                print(f"  linearity check layers={nl}: " + ", ".join(f"{c}: {m:.0f}us (x48/{nl})" for k, c, n, m, lo, hi in r2 if k == "commit"))

    print("\n== Amortized flush cost per step (commit path, 48 layers = one launch per step) vs verify")
    print("model: step_cost = t_idle + frac * B * (t_all - t_idle) / B ; frac = steady-state flush probability per row")
    print(f"{'path':7s} {'B':>3s} {'t_idle':>8s} {'us/flushing row':>16s} " + " ".join(f"{'frac=' + f'{x:.0%}':>9s}" for x in (0.05, 1 / 11, 0.10)) + "  (us/step, per layer us, % of green verify)")
    for kind in ("commit", "green"):
        for B, rows in summary.items():
            c = {case: med for k, case, n, med, lo, hi in rows if k == kind}
            if not c:
                continue
            t_idle = c["idle(0)"]
            t_all = next(med for k, case, n, med, lo, hi in rows if k == kind and n == B)
            per_row = (t_all - t_idle) / B
            cells = []
            for fr in (0.05, 1 / 11, 0.10):
                step = t_idle + fr * B * per_row
                ver = args.verify_us[0] if B <= 32 else args.verify_us[1]
                cells.append(f"{step:7.0f}/{step / PROD_LAYERS:4.1f}/{100 * step / PROD_LAYERS / ver:4.1f}%")
            print(f"{kind:7s} {B:3d} {t_idle:8.0f} {per_row:16.1f} " + " ".join(cells))


if __name__ == "__main__":
    main()

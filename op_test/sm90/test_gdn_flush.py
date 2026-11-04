"""GreenAttention SM90 Replay-SSM commit (flush) vs the stock Triton ``commit_gdn_replayssm_circular``.

Covers: capacity-fold rows (n_history 1..28 and the full ring 32), non-flush rows, NULL (-1) rows,
mamba-track snapshots (including no-op track rows and track_len <= 0), wrapped ``cache_base``,
with / without the residual rings, strided slot/layer dimensions, int32 / int64 indices,
cursor-reset parity, an fp64 reference for error attribution, determinism, CUDA-graph
capture/replay, and the ``can_use`` decline cases.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gdn_common as gc  # noqa: E402

from green_attention import gdn_flush  # noqa: E402

H, HV, K, V, T = gdn_flush.H, gdn_flush.HV, gdn_flush.K, gdn_flush.V, gdn_flush.T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 GDN kernels need an H100",
)


@pytest.fixture(autouse=True)
def _force_green_sm90(monkeypatch):
    import green_attention.gdn as gdn_mod

    monkeypatch.setattr(gdn_mod, "_green_on_sm90", lambda: True)


def _triton():
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import commit_gdn_replayssm_circular

    return commit_gdn_replayssm_circular


def make_case(B, layers=2, *, seed=0, has_res=True, n_hist=None, flush=None, null_rows=(), track=None,
              wrap=True, strided=False, idx_dtype=torch.int32, scale=0.05):
    """A production-shaped pool. ``n_hist`` / ``flush`` are per-row lists; ``track`` is None or a list of
    (track_step, has_track_slot) per row."""
    g = gc._gen(seed)
    slots = 2 * B + 2 if track is not None else B + 4
    s2 = B + 2

    def alloc(shape, dtype, sc):
        if strided:  # slot and layer strides larger than dense: [L, 2S, ...][:, ::2]
            big = gc._randn((shape[0] + 1, 2 * shape[1]) + tuple(shape[2:]), g, dtype, sc)
            return big[:shape[0], ::2]
        return gc._randn(shape, g, dtype, sc)

    c = dict(
        checkpoint_state=alloc((layers, slots, HV, V, K), torch.float32, scale),
        d_cache=alloc((layers, s2, HV, T, V), torch.bfloat16, 0.1),
        k_cache=alloc((layers, s2, H, T, K), torch.bfloat16, 0.1),
    )
    c["g_cache"] = -torch.rand((layers, s2, HV, T), generator=g, device="cuda") * 0.1
    if has_res:
        c["d_residual_cache"] = alloc((layers, s2, HV, T, V), torch.bfloat16, 0.01)
        c["k_residual_cache"] = alloc((layers, s2, H, T, K), torch.bfloat16, 0.01)
    else:
        c["d_residual_cache"] = c["k_residual_cache"] = None
    perm = lambda n, hi: gc._distinct(n, hi, g)  # noqa: E731
    state_idx = perm(B, slots).to(idx_dtype)
    for r in null_rows:
        state_idx[r] = -1
    replay_idx = perm(B, s2).to(idx_dtype)
    n_hist = n_hist if n_hist is not None else [(7 * i) % 28 + 1 for i in range(B)]
    flush = flush if flush is not None else [i % 2 == 0 for i in range(B)]
    wp = torch.zeros(s2, dtype=torch.int32, device="cuda")
    cb = torch.zeros(s2, dtype=torch.int32, device="cuda")
    fl = torch.zeros(s2, dtype=torch.int8, device="cuda")
    ri = replay_idx.long()
    wp[ri] = torch.tensor(n_hist, dtype=torch.int32, device="cuda")
    fl[ri] = torch.tensor([1 if f else 0 for f in flush], dtype=torch.int8, device="cuda")
    cb[ri] = (torch.arange(B, device="cuda", dtype=torch.int32) * 13 + (21 if wrap else 0)) & (T - 1)
    accept = torch.tensor([(i % 4) + 1 for i in range(B)], dtype=idx_dtype, device="cuda")
    c.update(state_batch_indices=state_idx, replay_indices=replay_idx, write_pos=wp, cache_base=cb,
             is_flush=fl, accept_lens=accept, mamba_track_indices=None, mamba_steps_to_track=None,
             null_block_id=-1)
    if track is not None:
        used = set(state_idx.long().tolist())
        free = [s for s in range(slots) if s not in used]
        tslots, tsteps = [], []
        for i, (step, has_slot) in enumerate(track):
            tslots.append(free.pop() if has_slot else -1)
            tsteps.append(step)
        c["mamba_track_indices"] = torch.tensor(tslots, dtype=idx_dtype, device="cuda")
        c["mamba_steps_to_track"] = torch.tensor(tsteps, dtype=idx_dtype, device="cuda")
    return c


def clone(c):
    out = {}
    for k, v in c.items():
        if isinstance(v, torch.Tensor):
            nv = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)
            nv.copy_(v)
            out[k] = nv
        else:
            out[k] = v
    return out


def reference_f64(c):
    """fp64 reference of the contract (hi/lo split removed): returns {slot: [L,HV,V,K]} for fold and track."""
    st = c["checkpoint_state"].double()
    L = st.shape[0]
    d = c["d_cache"].double() + (c["d_residual_cache"].double() if c["d_residual_cache"] is not None else 0)
    k = c["k_cache"].double() + (c["k_residual_cache"].double() if c["k_residual_cache"] is not None else 0)
    g = c["g_cache"].double()
    out = {}
    B = c["state_batch_indices"].shape[0]
    ti = c["mamba_track_indices"]
    for r in range(B):
        sidx = int(c["state_batch_indices"][r])
        if sidx < 0:
            continue
        rep = int(c["replay_indices"][r])
        n = int(c["write_pos"][rep])
        base = int(c["cache_base"][rep])
        phys = [(base + t) & (T - 1) for t in range(T)]
        plans = []
        if int(c["is_flush"][rep]) and n > 0:
            plans.append((sidx, n))
        if ti is not None:
            tslot, tstep = int(ti[r]), int(c["mamba_steps_to_track"][r])
            if tslot > -1 and tstep >= 0:
                plans.append((tslot, n - int(c["accept_lens"][r]) + tstep + 1))
        for slot, ln in plans:
            res = torch.empty(L, HV, V, K, dtype=torch.float64, device="cuda")
            for hv in range(HV):
                gt = g[:, rep, hv, phys]  # [L, T]
                m = (torch.arange(T, device="cuda") < ln).double()
                gm = gt * m
                prefix = gm.cumsum(-1)
                total = gm.sum(-1, keepdim=True)
                decay = torch.exp(total - prefix) * m
                u = d[:, rep, hv, phys, :] * decay[..., None]  # [L, T, V]
                kk = k[:, rep, hv // 3, phys, :]  # [L, T, K]
                delta = torch.einsum("ltv,ltk->lvk", u, kk)
                res[:, hv] = st[:, sidx, hv] * torch.exp(total)[..., None] + delta
            out[slot] = res
    return out


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def run_green(c):
    reason = gdn_flush.can_use(c)
    assert reason is None, reason
    gdn_flush.run(c)


def check_against_triton(c, *, tol_rel=2e-6):
    ref, grn = clone(c), clone(c)
    orig = clone(c)
    _triton()(**ref)
    run_green(grn)
    torch.cuda.synchronize()
    # cursors: bit-identical
    for name in ("write_pos", "cache_base", "is_flush"):
        assert torch.equal(grn[name], ref[name]), name
    # every slot: untouched slots bit-identical to the input; touched slots close to Triton
    st_ref, st_grn, st_orig = ref["checkpoint_state"], grn["checkpoint_state"], orig["checkpoint_state"]
    slots = st_ref.shape[1]
    touched = 0
    for s in range(slots):
        if torch.equal(st_ref[:, s], st_orig[:, s]):
            assert torch.equal(st_grn[:, s], st_orig[:, s]), f"slot {s} must be untouched"
        else:
            touched += 1
            e = rel(st_grn[:, s], st_ref[:, s])
            assert e <= tol_rel, f"slot {s}: rel err vs Triton {e:.2e}"
    # the rings are inputs only
    for name in ("d_cache", "k_cache", "g_cache"):
        assert torch.equal(grn[name], orig[name]), name
    return touched, orig, ref, grn


@pytest.mark.parametrize("has_res", [True, False])
@pytest.mark.parametrize("B", [1, 8, 32, 51])
def test_fold_vs_triton(B, has_res):
    c = make_case(B, has_res=has_res, null_rows=(1,) if B > 4 else ())
    touched, *_ = check_against_triton(c)
    assert touched >= 1


def test_fold_every_length():
    """n_history 1..28 and the full ring (32), one row each, every row flushing."""
    lens = list(range(1, 29)) + [32]
    c = make_case(len(lens), n_hist=lens, flush=[True] * len(lens))
    touched, *_ = check_against_triton(c)
    assert touched == len(lens)


def test_non_flush_rows_untouched_and_zero_history():
    c = make_case(8, n_hist=[0, 5, 5, 0, 9, 9, 28, 1], flush=[True, False, True, False, True, False, True, False])
    check_against_triton(c)


@pytest.mark.parametrize("wrap", [False, True])
def test_wrapped_base(wrap):
    c = make_case(16, wrap=wrap, n_hist=[26] * 16, flush=[True] * 16)
    check_against_triton(c)


def test_null_rows_and_cursors():
    c = make_case(12, null_rows=(0, 5, 11), flush=[True] * 12)
    _, orig, ref, grn = check_against_triton(c)
    # NULL rows keep flush=1: the stock finish kernel skips them too (cursors are compared inside)
    assert int(grn["is_flush"].sum()) == int(ref["is_flush"].sum())


@pytest.mark.parametrize("has_res", [True, False])
def test_track_rows(has_res):
    B = 10
    track = [(0, True), (1, True), (3, True), (2, False), (-1, True), (3, True), (0, True), (1, True), (2, True), (0, False)]
    c = make_case(B, has_res=has_res, track=track, n_hist=[4, 6, 5, 7, 3, 2, 9, 1, 8, 4],
                  flush=[True, False, True, False, False, True, False, True, False, True])
    touched, *_ = check_against_triton(c)
    assert touched >= B // 2


def test_track_len_nonpositive_copies_old_state():
    # accept 4 (i % 4 + 1 at i=3), n_hist 1, step 0 -> track_len = 1 - 4 + 0 + 1 = -2: a plain copy
    c = make_case(4, n_hist=[1, 1, 1, 1], flush=[False] * 4, track=[(0, True)] * 4)
    _, orig, ref, grn = check_against_triton(c)
    ti = c["mamba_track_indices"].long()
    si = c["state_batch_indices"].long()
    assert torch.equal(grn["checkpoint_state"][:, ti[3]], orig["checkpoint_state"][:, si[3]])


def test_track_and_flush_same_row_reads_old_state():
    c = make_case(6, n_hist=[10] * 6, flush=[True] * 6, track=[(1, True)] * 6)
    check_against_triton(c)


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_dtypes_and_strides(idx_dtype, strided):
    track = [(1, True)] * 6
    c = make_case(6, idx_dtype=idx_dtype, strided=strided, track=track, n_hist=[3, 9, 14, 20, 26, 28],
                  flush=[True] * 6)
    if strided:
        assert c["checkpoint_state"].stride(1) != HV * V * K
    check_against_triton(c)


def test_error_attribution_vs_fp64():
    c = make_case(8, flush=[True] * 8, track=[(1, True)] * 8, n_hist=[2, 5, 9, 13, 17, 21, 25, 28])
    ref64 = reference_f64(c)
    grn, trt = clone(c), clone(c)
    run_green(grn)
    _triton()(**trt)
    worst_g = worst_t = 0.0
    for slot, want in ref64.items():
        worst_g = max(worst_g, rel(grn["checkpoint_state"][:, slot], want))
        worst_t = max(worst_t, rel(trt["checkpoint_state"][:, slot], want))
    print(f"\nrel err vs fp64: green {worst_g:.2e} triton {worst_t:.2e}")
    assert worst_g <= max(2.0 * worst_t, 1e-6)
    assert worst_g <= 1e-5


def test_deterministic():
    c = make_case(16, flush=[True] * 16, track=[(1, True)] * 16)
    a, b = clone(c), clone(c)
    run_green(a)
    run_green(b)
    assert torch.equal(a["checkpoint_state"], b["checkpoint_state"])


def test_empty_job_list_is_a_noop():
    c = make_case(8, flush=[False] * 8)
    orig = clone(c)
    run_green(c)
    assert torch.equal(c["checkpoint_state"], orig["checkpoint_state"])
    for name in ("write_pos", "cache_base", "is_flush"):
        assert torch.equal(c[name], orig[name])


def test_cuda_graph_capture_replay():
    """One captured launch handles different job lists on replay (device-side job count)."""
    B = 16
    c = make_case(B, flush=[True] * B, track=[(1, True)] * B)
    base = clone(c)
    work = clone(c)
    gdn_flush._prewarm()
    run_green(clone(c))  # warm
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            gdn_flush.run(work)
    torch.cuda.synchronize()
    for flush_rows in (B, 3, 0):
        for name in ("checkpoint_state", "write_pos", "cache_base", "is_flush"):
            work[name].copy_(base[name])
        rep = base["replay_indices"].long()
        work["is_flush"].zero_()
        work["is_flush"][rep[:flush_rows]] = 1
        eager = clone(work)
        run_green(eager)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(work["checkpoint_state"], eager["checkpoint_state"]), flush_rows
        assert torch.equal(work["write_pos"], eager["write_pos"])
        assert torch.equal(work["is_flush"], eager["is_flush"])


def test_can_use_declines():
    c = make_case(4)
    assert gdn_flush.can_use(c) is None
    bad = dict(c, checkpoint_state=c["checkpoint_state"].bfloat16())
    assert "checkpoint_state" in gdn_flush.can_use(bad)
    bad = dict(c, d_residual_cache=None)
    assert "residual" in gdn_flush.can_use(bad)
    bad = dict(c, write_pos=c["write_pos"].long())
    assert "write_pos" in gdn_flush.can_use(bad)
    bad = dict(c, null_block_id=0)
    assert "null_block_id" in gdn_flush.can_use(bad)
    bad = dict(c, mamba_track_indices=c["state_batch_indices"])
    assert "track" in gdn_flush.can_use(bad)
    short = make_case(4)
    short["d_cache"] = short["d_cache"][..., :16, :].contiguous()
    assert "d_cache" in gdn_flush.can_use(short)


def test_hook_falls_back_and_toggles(monkeypatch):
    from green_attention import dispatch

    calls = []

    def original(*args, **kwargs):
        calls.append(kwargs)

    c = make_case(4, flush=[True] * 4)
    ref = clone(c)
    gdn_flush._commit_around(original, **c)
    assert not calls  # green served it
    assert not torch.equal(c["checkpoint_state"], ref["checkpoint_state"])
    monkeypatch.setattr(dispatch, "DISABLED", frozenset({"gdn_flush"}))
    gdn_flush._commit_around(original, **clone(c))
    assert len(calls) == 1  # GREEN_ATTN_DISABLE=gdn_flush -> stock path
    monkeypatch.setattr(dispatch, "DISABLED", frozenset())
    gdn_flush._commit_around(original, **dict(clone(c), null_block_id=0))
    assert len(calls) == 2  # declined -> stock path

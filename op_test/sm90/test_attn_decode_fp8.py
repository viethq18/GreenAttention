"""SM90 FP8-KV decode / verify attention vs an fp32 reference on the dequantized cache.

Run on the H100 box:  python -m pytest op_test/sm90 -q
Tolerance: max abs err <= 2e-2 against the fp32 reference (K/V dequantized exactly, Q taken as the
e4m3 value the kernel sees; the remaining error is the e4m3 rounding of P, 2^-4 relative per element)
and no worse than FA4's own error on the same input (``test_not_worse_than_fa4``).
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs an SM90 GPU"
)

from green_attention import attn_decode  # noqa: E402

FP8 = torch.float8_e4m3fn
TOL = 2e-2

# instance presets: (name, HD, KVH, QH, G, window) — window None means (-1,-1)
QWEN = ("qwen", 256, 4, 24, 6, None)
GEMMA_SWA = ("gemma-swa", 256, 16, 32, 2, 1023)   # sglang passes sliding_window_size - 1
GEMMA_FULL = ("gemma-full", 512, 4, 32, 8, None)


@pytest.fixture(scope="module", autouse=True)
def _ready():
    attn_decode.ensure_ready()


def _e4m3(x):
    return x.clamp(-448, 448).to(FP8)


def make_case(lens, qlens, page, *, q_dtype=FP8, seed=0, kscale=1.0, vscale=1.0, descale="rand", kv_stack=False,
              q_strided=False, out_mode="none", extra_pages=2, pt_pad=1, ret_lse=True, inst=QWEN,
              window=None, scale=None, qscale=1.0):
    """Random paged cache + queries. Returns (kwargs for the kernel, dense reference inputs)."""
    name, HD, KVH, QH, G, wdef = inst
    if window is None:
        window = (-1, -1) if wdef is None else (wdef, 0)
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    B = len(lens)
    pages_per = [max(1, -(-L // page)) for L in lens]
    n_pages = sum(pages_per) + extra_pages
    perm = torch.randperm(n_pages, generator=g, device=dev, dtype=torch.int64).to(torch.int32)
    max_p = max(pages_per) + pt_pad
    pt = torch.zeros((B, max_p), dtype=torch.int32, device=dev)
    pos = 0
    for b, npg in enumerate(pages_per):
        pt[b, :npg] = perm[pos:pos + npg]
        pos += npg
    if page == 1:
        pass  # page ids are token slots
    shape = (n_pages, page, KVH, HD)
    kf = torch.randn(shape, generator=g, device=dev) * kscale
    vf = torch.randn(shape, generator=g, device=dev) * vscale
    if kv_stack:  # K and V interleaved in one buffer, as a combined KV pool would
        buf = torch.empty((n_pages, 2, page, KVH, HD), dtype=FP8, device=dev)
        buf[:, 0] = _e4m3(kf)
        buf[:, 1] = _e4m3(vf)
        kc, vc = buf[:, 0], buf[:, 1]
    else:
        kc, vc = _e4m3(kf), _e4m3(vf)
    if descale == "rand":
        kd = (torch.rand((B, KVH), generator=g, device=dev) * 0.5 + 0.5).float()
        vd = (torch.rand((B, KVH), generator=g, device=dev) * 0.5 + 0.5).float()
    elif descale == "scalar":  # expanded scalar, strides (0, 0), as the backend passes it
        kd = torch.tensor([0.37], device=dev).expand(B, KVH)
        vd = torch.tensor([0.61], device=dev).expand(B, KVH)
    else:
        kd = vd = None
    total_q = sum(qlens)
    qf = torch.randn((total_q, QH, HD), generator=g, device=dev) * qscale
    if q_strided:
        qbuf = torch.empty((total_q, QH, 2 * HD), device=dev, dtype=q_dtype if q_dtype != FP8 else FP8)
        qbuf[..., :HD] = _e4m3(qf) if q_dtype == FP8 else qf.to(q_dtype)
        q = qbuf[..., :HD]
    else:
        q = _e4m3(qf) if q_dtype == FP8 else qf.to(q_dtype)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(qlens, device=dev, dtype=torch.int32).cumsum(0)
    kw = dict(
        q=q, k_cache=kc, v_cache=vc, page_table=pt,
        cache_seqlens=torch.tensor(lens, dtype=torch.int32, device=dev),
        cu_seqlens_q=cu, max_seqlen_q=max(qlens), softmax_scale=HD ** -0.5 if scale is None else scale, causal=True,
        window_size=window, softcap=0.0, return_softmax_lse=ret_lse, num_splits=0,
    )
    if kd is not None:
        kw["k_descale"], kw["v_descale"] = kd, vd
    if out_mode == "given":
        kw["out"] = torch.full((total_q, QH, HD), float("nan"), dtype=torch.bfloat16, device=dev)
    elif out_mode == "strided":
        ob = torch.full((total_q, QH, 2 * HD), float("nan"), dtype=torch.bfloat16, device=dev)
        kw["out"] = ob[..., :HD]
    return kw


def reference(kw, floor=False):
    """fp32 attention on the dequantized cache. Returns (out [total_q,QH,HD], lse [QH,total_q]); with floor=True
    also the elementwise e4m3-P rounding floor [total_q,QH,HD] (see check)."""
    q = kw["q"] if kw["q"].dtype == FP8 else _e4m3(kw["q"].float())  # the kernel casts bf16 q to e4m3
    q = q.float()
    kc, vc, pt = kw["k_cache"].float(), kw["v_cache"].float(), kw["page_table"]
    lens = kw["cache_seqlens"].tolist()
    cu = kw["cu_seqlens_q"].tolist()
    page = kc.shape[1]
    KVH, HD = kc.shape[2], kc.shape[3]
    G = q.shape[1] // KVH
    wl = kw.get("window_size", (-1, -1))[0]
    kd, vd = kw.get("k_descale"), kw.get("v_descale")
    out = torch.zeros_like(q)
    flo = torch.zeros_like(q)
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"), device=q.device)
    for b, L in enumerate(lens):
        ql = cu[b + 1] - cu[b]
        if L == 0:
            continue
        n = torch.arange(L, device=q.device)
        pg = pt[b, n // page].long()
        K = kc[pg, n % page]
        V = vc[pg, n % page]
        for h in range(KVH):
            ks = K[:, h] * (kd[b, h] if kd is not None else 1.0)
            vs = V[:, h] * (vd[b, h] if vd is not None else 1.0)
            qq = q[cu[b]:cu[b + 1], h * G:(h + 1) * G]  # [ql, G, HD]
            s = torch.einsum("tgd,nd->tgn", qq, ks) * kw["softmax_scale"]
            pos = L - ql + torch.arange(ql, device=q.device)  # absolute position of each query row
            s = s.masked_fill(n[None, None, :] > pos[:, None, None], float("-inf"))
            if wl is not None and wl >= 0:
                s = s.masked_fill(n[None, None, :] < pos[:, None, None] - wl, float("-inf"))
            m = s.amax(-1, keepdim=True)
            e = torch.exp(s - m)
            l = e.sum(-1, keepdim=True)
            p = e / l
            o = torch.einsum("tgn,nd->tgd", p, vs)
            out[cu[b]:cu[b + 1], h * G:(h + 1) * G] = o
            if floor:
                worst = torch.einsum("tgn,nd->tgd", p, vs.abs())
                rms = torch.einsum("tgn,nd->tgd", p * p, vs * vs).sqrt()
                flo[cu[b]:cu[b + 1], h * G:(h + 1) * G] = 2.0 ** -4 * torch.minimum(worst, 6.0 / 3 ** 0.5 * rms)
            lse[h * G:(h + 1) * G, cu[b]:cu[b + 1]] = (m + l.log()).squeeze(-1).T
    return (out, lse, flo) if floor else (out, lse)


def run(kw, select=None):
    fn = (select or attn_decode.select)(kw)
    assert callable(fn), f"declined: {fn}"
    return fn(**kw)


def _fa4_out(kw):
    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache as fa4

    q = kw["q"] if kw["q"].dtype == FP8 else _e4m3(kw["q"].float())
    args = {k: v for k, v in kw.items() if k not in ("out", "max_seqlen_k")}
    return fa4(**{**args, "q": q, "return_softmax_lse": False})


def check_out(out, kw, tol=TOL):
    """Per-sequence max abs error <= tol. A sequence may only exceed tol if FA4 (the incumbent) exceeds it
    too on that sequence and we are no worse than it: that is the e4m3-P rounding floor (2^-4 relative per
    probability) that both kernels share on very short / sharply peaked rows, not a kernel defect.
    Returns (max error, reference out, reference lse)."""
    ref_o, ref_lse = reference(kw)
    assert out.dtype == torch.bfloat16 and out.shape == kw["q"].shape
    assert not torch.isnan(out.float()).any()
    cu = kw["cu_seqlens_q"].tolist()
    errs = [(out[cu[b]:cu[b + 1]].float() - ref_o[cu[b]:cu[b + 1]]).abs().max().item() if cu[b + 1] > cu[b] else 0.0
            for b in range(len(cu) - 1)]
    err = max(errs)
    if err > tol and kw["q"].shape[2] > 256:
        # FA4-SM90 FP8 has no head_dim 512 path, so there is no incumbent to compare with. A sequence may exceed
        # tol only within the e4m3-P rounding floor: each probability is rounded to e4m3 (relative error <= 2^-4,
        # l summed unrounded), so |err| <= 2^-4 * (P @ |V|). Rounding errors of different keys are independent,
        # so the bound is also capped at 6 sigma of their sum (sigma = 2^-4/sqrt(3) * sqrt(P^2 @ V^2)): loose only
        # on rows dominated by a few keys, tight on long rows (a lost or misweighted split there fails it).
        # Plus bf16 output rounding (2^-8 relative) and 1e-3 slack.
        _, _, flo = reference(kw, floor=True)
        for b, e in enumerate(errs):
            if e <= tol:
                continue
            s = slice(cu[b], cu[b + 1])
            bound = flo[s] + 2.0 ** -8 * ref_o[s].abs() + 1e-3
            excess = ((out[s].float() - ref_o[s]).abs() - bound).max().item()
            assert excess <= 0, f"seq {b}: max abs err {e} exceeds the e4m3-P floor by {excess}"
    elif err > tol:
        fa = _fa4_out(kw)
        for b, e in enumerate(errs):
            if e <= tol:
                continue
            e_fa4 = (fa[cu[b]:cu[b + 1]].float() - ref_o[cu[b]:cu[b + 1]]).abs().max().item()
            assert e_fa4 > tol and e <= 1.1 * e_fa4 + 1e-3, f"seq {b}: max abs err {e}, fa4 {e_fa4}"
    return err, ref_o, ref_lse


def check(kw, tol=TOL, lse_tol=5e-3, select=None):
    """`check_out` on a fresh call, plus out= aliasing and lse."""
    res = run(kw, select)
    out, lse = res if kw["return_softmax_lse"] else (res, None)
    if "out" in kw:
        assert out.data_ptr() == kw["out"].data_ptr(), "out= not honored"
    err, _, ref_lse = check_out(out, kw, tol)
    if lse is not None:
        assert lse.shape == (kw["q"].shape[1], kw["q"].shape[0]) and lse.dtype == torch.float32
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin)
        lerr = (lse[fin] - ref_lse[fin]).abs().max().item() if fin.any() else 0.0
        assert lerr <= lse_tol, f"lse err {lerr}"
    return err


LENS = [1, 2, 127, 128, 129, 255, 1000, 4097]


@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("qlen", [1, 2, 4, 7, 10])
@pytest.mark.parametrize("q_dtype", [FP8, torch.bfloat16])
def test_ragged_lengths(page, qlen, q_dtype):
    lens = [max(L, qlen) for L in LENS]
    check(make_case(lens, [qlen] * len(lens), page, q_dtype=q_dtype, seed=page + qlen))


@pytest.mark.parametrize("page", [1, 16])
def test_ragged_query_lengths(page):
    lens = [5, 300, 129, 40, 2048, 12, 700, 10]
    qlens = [1, 4, 2, 10, 3, 1, 6, 4]
    check(make_case(lens, qlens, page, seed=3))


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("qlen", [1, 4])
def test_long_context_split_kv(batch, qlen):
    lens = [20000 - 777 * i for i in range(batch)]
    kw = make_case(lens, [qlen] * batch, 16, seed=11, extra_pages=8)
    assert attn_decode.plan_splits(batch, kw["page_table"].shape[1] * 16) > 1
    check(kw)


def test_max_seqlen_k_kwarg_sets_static_splits():
    kw = make_case([3000, 900], [1, 1], 16, seed=5)
    kw["max_seqlen_k"] = 3000
    check(kw)


def test_empty_sequence():
    check(make_case([0, 17, 0, 300], [1, 1, 1, 1], 16, seed=6))
    check(make_case([0, 0], [4, 4], 16, seed=6))  # every row empty: defined (zero) output and -inf lse


@pytest.mark.parametrize("qlen", [1, 4])
def test_dynamic_splits_ragged_batch(qlen):
    # one long sequence sets the grid's split bound; the short ones must plan fewer (or one) split(s)
    lens = [40000, 3, 256, 257, 1024, 9000, 128 * 7 + 5, 16384]
    check(make_case(lens, [qlen] * len(lens), 16, seed=40 + qlen, extra_pages=8))


@pytest.mark.parametrize("L", [255, 256, 257, 511, 512, 513, 2048, 2049, 65536, 100003])
def test_dynamic_splits_boundary_lengths(L):
    check(make_case([L], [1], 16, seed=L % 97, extra_pages=8))
    check(make_case([L, 5], [4, 4], 16, seed=L % 89, extra_pages=8))


def test_graph_replay_changing_split_plan():
    # captured at the maximum length; replay with shorter / empty sequences in the same buffers (every
    # sequence owns pages for the maximum length, so lengths below it index real, distinct pages)
    batch, qlen = 4, 4
    kw = make_case([30000] * batch, [qlen] * batch, 16, seed=44, extra_pages=4, pt_pad=16)
    fn = attn_decode.select(kw)
    assert callable(fn)
    fn(**kw)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_out = fn(**kw)
    for new_lens in ([30000, 20000, 9000, 5000], [qlen, 10, 700, 30000], [0, 0, 0, 0], [12345, 4, 4, 29999]):
        kw["cache_seqlens"].copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        out, lse = static_out
        ref_o, ref_lse = reference(kw)
        errs = [(out[qlen * b:qlen * (b + 1)].float() - ref_o[qlen * b:qlen * (b + 1)]).abs().max().item()
                for b in range(batch)]
        assert all(e <= TOL for e, L in zip(errs, new_lens) if L >= 64) and not torch.isnan(out.float()).any(), (new_lens, errs)
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin)
        assert (lse[fin] - ref_lse[fin]).abs().max().item() <= 5e-3 if fin.any() else True


@pytest.mark.parametrize("descale", ["none", "scalar", "rand"])
def test_descale_forms(descale):
    check(make_case([500, 129, 2000], [4, 4, 4], 16, descale=descale, seed=8))


@pytest.mark.parametrize("inst", [QWEN, GEMMA_SWA, GEMMA_FULL])
def test_fp8_saturation_range(inst):
    # K/V near the e4m3 limit (448) with matching descales: effective values stay O(1)
    kw = make_case([1500, 333, 64], [1, 1, 1], 16, kscale=150.0, vscale=150.0, descale="none", seed=9, inst=inst)
    kvh = kw["k_cache"].shape[2]
    kw["k_descale"] = torch.full((3, kvh), 1 / 150.0, device="cuda")
    kw["v_descale"] = torch.full((3, kvh), 1 / 150.0, device="cuda")
    check(kw)


def test_peaked_softmax():
    # a few tokens dominate: stresses the running max / rescale path and P quantization
    kw = make_case([3000, 1024], [4, 4], 16, kscale=4.0, descale="none", seed=10)
    # logits reach ~18 (natural-log lse); FA4 shows the same 6.3e-3 lse error here (fp32 rounding in the
    # shared q.k products at that magnitude), so the 5e-3 default is relaxed to 1e-2 for this case only.
    check(kw, lse_tol=1e-2)


@pytest.mark.parametrize("out_mode", ["given", "strided"])
def test_out_argument(out_mode):
    check(make_case([700, 129, 33], [4, 4, 4], 16, out_mode=out_mode, seed=12))


def test_strided_q_and_stacked_kv():
    check(make_case([700, 129, 33], [1, 1, 1], 16, q_strided=True, kv_stack=True, seed=13))
    check(make_case([700, 129, 33], [4, 4, 4], 1, q_strided=True, kv_stack=True, seed=14, q_dtype=torch.bfloat16))


def test_no_lse():
    kw = make_case([700, 129], [4, 4], 16, ret_lse=False, seed=15)
    check(kw)


def test_deterministic():
    kw = make_case([4000, 129, 900], [4, 4, 4], 16, seed=16)
    a, la = run(kw)
    b, lb = run(kw)
    assert torch.equal(a, b) and torch.equal(la, lb)


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("qlen", [1, 4])
def test_cuda_graph_capture_replay(batch, qlen):
    lens = [2048 + 37 * i for i in range(batch)]
    kw = make_case(lens, [qlen] * batch, 16, seed=17 + batch, extra_pages=4, pt_pad=64)
    fn = attn_decode.select(kw)
    assert callable(fn)
    fn(**kw)  # warm
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_out = fn(**kw)
    # replay with different sequence lengths written into the captured buffers
    new_lens = [max(qlen, L - 5 - 3 * i) for i, L in enumerate(lens)]
    kw["cache_seqlens"].copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    out, lse = static_out
    ref_o, ref_lse = reference(kw)
    assert (out.float() - ref_o).abs().max().item() <= TOL
    assert (lse - ref_lse).abs().max().item() <= 5e-3


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda kw: kw.update(softcap=30.0), "softcap"),
        (lambda kw: kw.update(window_size=(128, 5)), "window"),
        # full-attention instance is compiled without the window mask: a left window must not be ignored
        (lambda kw: kw.update(window_size=(128, 0)), "full-attention instance"),
        (lambda kw: kw.update(sinks=torch.zeros(24, device="cuda")), "sinks"),
        (lambda kw: kw.update(max_seqlen_q=11), "max_seqlen_q"),
        (lambda kw: kw.update(causal=False, max_seqlen_q=4), "non-causal"),
        (lambda kw: kw.update(k_cache=kw["k_cache"].to(torch.bfloat16), v_cache=kw["v_cache"].to(torch.bfloat16)), "e4m3"),
        (lambda kw: kw.update(q=kw["q"][:, :12]), "q shape"),
        (lambda kw: kw.update(cu_seqlens_q=None), "paged"),
    ],
)
def test_declines_out_of_scope(mutate, needle):
    kw = make_case([300, 40], [4, 4], 16, seed=20)
    mutate(kw)
    why = attn_decode.select(kw)
    assert isinstance(why, str) and needle in why, why


def test_declines_non_power_of_two_page():
    kw = make_case([300, 40], [1, 1], 16, seed=21)
    kw["k_cache"] = kw["k_cache"][:, :12]
    kw["v_cache"] = kw["v_cache"][:, :12]
    why = attn_decode.select(kw)
    assert isinstance(why, str) and "power of two" in why


def test_not_worse_than_fa4():
    pytest.importorskip("sglang")
    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache as fa4

    worst = 0.0
    for page, qlen, lens in [(16, 1, [2000, 129, 700, 5000]), (16, 4, [2000, 129, 700, 5000]), (1, 4, [300, 129, 64])]:
        kw = make_case(lens, [qlen] * len(lens), page, seed=30 + qlen, ret_lse=False)
        ref_o, _ = reference(kw)
        ours = run(kw)
        theirs = fa4(**{**kw, "return_softmax_lse": False})
        e_ours = (ours.float() - ref_o).abs().max().item()
        e_fa4 = (theirs.float() - ref_o).abs().max().item()
        print(f"page={page} qlen={qlen}: err ours {e_ours:.3e}  fa4 {e_fa4:.3e}")
        assert e_ours <= TOL
        assert e_ours <= max(e_fa4 * 1.1, 1e-3), (e_ours, e_fa4)
        worst = max(worst, e_ours)
    assert math.isfinite(worst)


# --------------------------------------------------------------------- gemma instances
# Sliding: sglang passes window_size (sliding_window - 1, 0) = (1023, 0); a row at position p sees
# keys [p - 1023, p]. Global: head_dim 512 (DSPLIT 2), qlen <= 8 (GQA 8 x qlen <= 64 rows).


@pytest.mark.parametrize("page", [1, 16])
@pytest.mark.parametrize("qlen", [1, 2, 8, 16])
def test_gemma_swa_ragged(page, qlen):
    lens = [max(L, qlen) for L in LENS]
    check(make_case(lens, [qlen] * len(lens), page, seed=page + qlen, inst=GEMMA_SWA))


@pytest.mark.parametrize("L", [1, 2, 1023, 1024, 1025, 1026, 1027, 1152, 2048, 5000, 16384])
def test_gemma_swa_window_edges(L):
    # lens that straddle the window length and tile boundaries: the skipped-tile path and the
    # partially-masked first tile are both exercised
    check(make_case([L], [1], 16, seed=L % 97, inst=GEMMA_SWA))
    check(make_case([L, 7], [4, 4], 16, seed=L % 89 + 1, inst=GEMMA_SWA))


def test_gemma_swa_window_qlen_row_bounds():
    # qlen > 1: the first row (position L - qlen) has a shorter visible history than the last row;
    # whole 64-column halves must drop out for the early rows
    lens = [3000, 4096, 4097]
    check(make_case(lens, [16] * len(lens), 16, seed=61, inst=GEMMA_SWA))
    check(make_case(lens, [2, 5, 16], 16, seed=62, inst=GEMMA_SWA))


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("qlen", [1, 4])
def test_gemma_swa_split_kv(batch, qlen):
    # context far beyond the window: split planning must count only the ~9 window tiles
    lens = [20000 - 777 * i for i in range(batch)]
    kw = make_case(lens, [qlen] * batch, 16, seed=71 + batch, inst=GEMMA_SWA, extra_pages=8)
    assert attn_decode.plan_splits(batch, kw["page_table"].shape[1] * 16, units=16) > 1
    check(kw)


@pytest.mark.parametrize("page", [1, 16])
@pytest.mark.parametrize("qlen", [1, 4, 8])
def test_gemma_global_ragged(page, qlen):
    lens = [max(L, qlen) for L in LENS]
    check(make_case(lens, [qlen] * len(lens), page, seed=page + 100 + qlen, inst=GEMMA_FULL))


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("qlen", [1, 8])
def test_gemma_global_split_kv(batch, qlen):
    lens = [20000 - 777 * i for i in range(batch)]
    kw = make_case(lens, [qlen] * batch, 16, seed=81 + batch, inst=GEMMA_FULL, extra_pages=8)
    assert attn_decode.plan_splits(batch, kw["page_table"].shape[1] * 16, units=8) > 1
    check(kw)


def test_gemma_global_dsplits_boundary_lengths():
    for L in [255, 256, 257, 511, 512, 513, 2048, 2049]:
        check(make_case([L], [1], 16, seed=L % 83 + 7, inst=GEMMA_FULL))


@pytest.mark.parametrize("descale", ["none", "scalar", "rand"])
def test_gemma_descale_forms(descale):
    check(make_case([500, 129, 2000], [4, 4, 4], 16, descale=descale, seed=90, inst=GEMMA_SWA))
    check(make_case([500, 129, 2000], [8, 8, 8], 16, descale=descale, seed=91, inst=GEMMA_FULL))


@pytest.mark.parametrize("inst", [GEMMA_SWA, GEMMA_FULL])
def test_gemma_unit_softmax_scale(inst):
    # Gemma 4 runs with scaling = 1.0 (q/k are RMS-normed upstream). Pre-scale q by HD^-0.5 so the score
    # distribution matches the default cases: a kernel that ignored the runtime scale would be off by sqrt(HD).
    hd = inst[1]
    check(make_case([1, 300, 1025, 4097], [1, 2, 4, 8], 16, seed=97, inst=inst, scale=1.0, qscale=hd ** -0.5))


def test_gemma_swa_cuda_graph_capture_replay():
    # window + splits: capture at the max length, replay with shorter lens (and all-empty)
    batch, qlen = 4, 4
    kw = make_case([20000] * batch, [qlen] * batch, 16, seed=95, inst=GEMMA_SWA, extra_pages=4, pt_pad=16)
    fn = attn_decode.select(kw)
    assert callable(fn)
    fn(**kw)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_out = fn(**kw)
    for new_lens in ([20000, 12000, 3000, 500], [qlen, 10, 700, 20000], [0, 0, 0, 0], [12345, 4, 4, 19999]):
        kw["cache_seqlens"].copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        out, lse = static_out
        ref_o, ref_lse = reference(kw)
        errs = [(out[qlen * b:qlen * (b + 1)].float() - ref_o[qlen * b:qlen * (b + 1)]).abs().max().item()
                for b in range(batch)]
        assert all(e <= TOL for e, L in zip(errs, new_lens) if L >= 64) and not torch.isnan(out.float()).any(), (new_lens, errs)


def test_gemma_global_cuda_graph_capture_replay():
    batch, qlen = 4, 8
    kw = make_case([20000] * batch, [qlen] * batch, 16, seed=96, inst=GEMMA_FULL, extra_pages=4, pt_pad=16)
    fn = attn_decode.select(kw)
    assert callable(fn)
    fn(**kw)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_out = fn(**kw)
    for new_lens in ([20000, 9000, 3000, 700], [qlen, 10, 700, 20000], [0, 0, 0, 0], [12345, 4, 4, 19999]):
        kw["cache_seqlens"].copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        out, lse = static_out
        ref_o, ref_lse = reference(kw)
        errs = [(out[qlen * b:qlen * (b + 1)].float() - ref_o[qlen * b:qlen * (b + 1)]).abs().max().item()
                for b in range(batch)]
        assert all(e <= TOL for e, L in zip(errs, new_lens) if L >= 64) and not torch.isnan(out.float()).any(), (new_lens, errs)

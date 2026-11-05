"""SM90 FP8-KV extend / prefill kernel (op ``extend``, Gemma-4 global head_dim 512) vs an fp32 reference.

Same tolerance rules as test_attn_decode_fp8 (``check`` / ``check_out``): FA4 has no SM90 FP8 head_dim 512 path,
so a sequence may only exceed TOL within the shared e4m3-P rounding floor (short, sharply peaked rows).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs an SM90 GPU"
)

from green_attention import attn_decode, dispatch, wide_kv  # noqa: E402
from test_attn_decode_fp8 import FP8, GEMMA_FULL, GEMMA_SWA, QWEN, check, check_out, make_case  # noqa: E402

BQ = attn_decode.EXTEND_BLOCK_Q


@pytest.fixture(scope="module", autouse=True)
def _ready():
    attn_decode.ensure_ready()


def case(lens, qlens, page=16, **kw):
    return make_case(lens, qlens, page, inst=GEMMA_FULL, **kw)


def check_extend(kw, **opts):
    return check(kw, select=attn_decode.select_extend, **opts)


@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("q_dtype", [FP8, torch.bfloat16])
def test_prefix_and_extend_ragged(page, q_dtype):
    # full prefill, prefix + extend, q_len below / at / above one block, a single token
    lens = [700, 129, 2048, 33, BQ, 1, 300]
    qlens = [700, 64, 333, 5, BQ, 1, 299]
    check_extend(case(lens, qlens, page, q_dtype=q_dtype, seed=page))


@pytest.mark.parametrize("qlen", [BQ - 1, BQ, BQ + 1, 2 * BQ - 1, 2 * BQ, 2 * BQ + 1, 127, 128, 129])
def test_query_block_boundaries(qlen):
    # every sequence starts at a different offset modulo the block, so blocks straddle sequence starts
    lens = [qlen + p for p in (0, 3, 130, 1000)]
    check_extend(case(lens, [qlen] * len(lens), seed=qlen))


def test_empty_queries_and_sequences():
    # q_len 0 (a request with nothing to extend) and an empty cache next to real ones
    check_extend(case([300, 0, 50, 17], [5, 0, 50, 0], seed=3))


def test_long_prefill():
    check_extend(case([4500, 37], [3000, 37], seed=4, extra_pages=4))


@pytest.mark.parametrize("descale", ["none", "scalar", "rand"])
def test_descale_forms(descale):
    check_extend(case([500, 129, 2000], [500, 40, 77], descale=descale, seed=5))


@pytest.mark.parametrize("out_mode", ["none", "given", "strided"])
@pytest.mark.parametrize("ret_lse", [True, False])
def test_out_and_lse(out_mode, ret_lse):
    check_extend(case([700, 129, 33], [300, 129, 9], out_mode=out_mode, ret_lse=ret_lse, seed=6))


def test_unit_softmax_scale():
    # Gemma 4 runs with scaling = 1.0; q pre-scaled so the scores match the default cases
    check_extend(case([900, 64], [200, 64], scale=1.0, qscale=512 ** -0.5, seed=7))


def test_strided_q_and_stacked_kv():
    check_extend(case([700, 129], [100, 129], q_strided=True, kv_stack=True, seed=8))


def test_deterministic():
    kw = case([3000, 129], [1000, 100], seed=9)
    fn = attn_decode.select_extend(kw)
    a, la = fn(**kw)
    b, lb = fn(**kw)
    assert torch.equal(a, b) and torch.equal(la, lb)


def test_cuda_graph_capture_replay():
    # the grid depends only on total_q and batch: one capture serves any split of total_q over the batch
    batch, total = 4, 256
    lens = [3000] * batch
    kw = case(lens, [total // batch] * batch, seed=10, extra_pages=4, pt_pad=8)
    fn = attn_decode.select_extend(kw)
    assert callable(fn)
    fn(**kw)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            static_out = fn(**kw)
    for qlens, new_lens in (
        ([64, 64, 64, 64], [3000, 900, 64, 2000]),
        ([1, 250, 3, 2], [1, 2999, 3, 1000]),
        ([0, 0, 256, 0], [0, 7, 256, 0]),
        ([BQ - 1, BQ + 1, 2 * BQ, 256 - 4 * BQ], [BQ - 1, 700, 2 * BQ, 3000]),
    ):
        cu = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.tensor(qlens, dtype=torch.int32, device="cuda").cumsum(0)
        kw["cu_seqlens_q"].copy_(cu)
        kw["cache_seqlens"].copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        check_out(static_out[0], kw)


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda kw: kw.update(softcap=30.0), "softcap"),
        (lambda kw: kw.update(causal=False), "non-causal"),
        (lambda kw: kw.update(window_size=(1023, 0)), "window"),
        (lambda kw: kw.update(max_seqlen_q=None), "max_seqlen_q"),
        (lambda kw: kw.update(q_descale=torch.ones(1, device="cuda")), "q_descale"),
        (lambda kw: kw.update(q=kw["q"][:1].expand(BQ * 65535, -1, -1)), "grid.z"),
    ],
)
def test_declines_out_of_scope(mutate, needle):
    kw = case([300, 40], [100, 40], seed=11)
    mutate(kw)
    why = attn_decode.select_extend(kw)
    assert isinstance(why, str) and needle in why, why


@pytest.mark.parametrize("inst", [QWEN, GEMMA_SWA])
def test_declines_shapes_fa4_runs_natively(inst):
    why = attn_decode.select_extend(make_case([300], [100], 16, inst=inst, seed=12))
    assert isinstance(why, str) and "no extend instance" in why


@pytest.mark.parametrize("mq", [BQ + 2, dispatch.MAX_VERIFY_Q + 1])
def test_dispatch_takes_extend_kernel_not_dequant(monkeypatch, mq):
    # a short extend (a "verify"-sized call past the decode instance's max_q) and a prefill both reach the kernel
    def no_dequant(*a, **k):
        raise AssertionError("went to the dequantizing fallback")

    monkeypatch.setattr(wide_kv, "attend", no_dequant)
    kw = case([400, 2 * mq], [mq, mq], seed=13, ret_lse=False)
    assert wide_kv.applies(kw["k_cache"])
    out = dispatch.with_kvcache(no_dequant)(**kw)
    check_out(out, kw)

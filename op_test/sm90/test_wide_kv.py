"""Gemma 4 global layers (head_dim 512) with an fp8 KV cache through the real FA4, on SM90.

FA4 SM90 has no FP8 kernel past head_dim 256; ``dispatch.with_kvcache`` routes the calls green declines (or
``GREEN_ATTN_DISABLE`` turns off) to ``wide_kv`` (FA4 bf16 on dequantized pages). Reference: fp32 attention with
the bf16 query exact and K/V dequantized exactly; the error left is bf16 rounding of the dequantized K/V and of P.
The green extend kernel on the same shapes: test_attn_extend_fp8.py.
Run on the H100 box:  python -m pytest op_test/sm90/test_wide_kv.py -q
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs an SM90 GPU"
)

from green_attention import dispatch, wide_kv  # noqa: E402
from test_attn_decode_fp8 import FP8, GEMMA_FULL, QWEN, make_case  # noqa: E402

TOL = 2e-2


def _fa4():
    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache

    return flash_attn_with_kvcache


def _reference(kw):
    q, kc, vc, pt = kw["q"].float(), kw["k_cache"].float(), kw["v_cache"].float(), kw["page_table"]
    page, kvh = kc.shape[1], kc.shape[2]
    g = q.shape[1] // kvh
    cu = kw["cu_seqlens_q"].tolist()
    out = torch.zeros((*q.shape[:-1], vc.shape[-1]), device=q.device)
    for b, L in enumerate(kw["cache_seqlens"].tolist()):
        n = torch.arange(L, device=q.device)
        pg = pt[b, n // page].long()
        k, v = kc[pg, n % page], vc[pg, n % page]
        if kw.get("k_descale") is not None:
            k = k * kw["k_descale"][b].view(1, kvh, 1)
            v = v * kw["v_descale"][b].view(1, kvh, 1)
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        qq = q[cu[b] : cu[b + 1]]
        s = torch.einsum("thd,nhd->htn", qq, k) * kw["softmax_scale"]
        pos = L - qq.shape[0] + torch.arange(qq.shape[0], device=q.device)
        s = s.masked_fill(n[None, None, :] > pos[None, :, None], float("-inf"))
        out[cu[b] : cu[b + 1]] = torch.einsum("htn,nhd->thd", s.softmax(-1), v)
    return out


def _gemma_global(lens, qlens, page, **kw):
    case = make_case(lens, qlens, page, q_dtype=torch.bfloat16, inst=GEMMA_FULL, ret_lse=False, **kw)
    case.pop("max_seqlen_q")
    return case


@pytest.mark.parametrize("descale", ["rand", "scalar", None])
@pytest.mark.parametrize("page", [1, 64])
def test_declined_prefill_with_prefix_runs_dequantized(monkeypatch, page, descale):
    monkeypatch.setattr(dispatch, "select", lambda op, kw: None)  # green declines (or is disabled for) every call
    kw = _gemma_global([700, 129, 2048], [700, 64, 333], page, descale=descale)
    kw["max_seqlen_q"] = 700
    out = dispatch.with_kvcache(_fa4())(**kw)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), _reference(kw), atol=TOL, rtol=0)


def test_declined_decode_runs_dequantized(monkeypatch):
    monkeypatch.setattr(dispatch, "select", lambda op, kw: None)  # green declines every call
    kw = _gemma_global([1, 300, 4097], [1, 1, 1], 64)
    kw["max_seqlen_q"] = 1
    out = dispatch.with_kvcache(_fa4())(**kw)
    torch.testing.assert_close(out.float(), _reference(kw), atol=TOL, rtol=0)


def test_request_groups_match_one_group(monkeypatch):
    kw = _gemma_global([700, 129, 2048, 5], [700, 64, 333, 5], 64)
    kw["max_seqlen_q"] = 700
    whole = wide_kv.attend(_fa4(), kw, "extend")
    monkeypatch.setattr(wide_kv, "GROUP_BYTES", 1)
    out = torch.full_like(whole, float("nan"))
    got = wide_kv.attend(_fa4(), {**kw, "out": out}, "extend")
    assert got is out
    torch.testing.assert_close(got.float(), _reference(kw), atol=TOL, rtol=0)
    torch.testing.assert_close(got, whole, atol=TOL, rtol=0)


def test_only_wide_fp8_kv_applies():
    wide = _gemma_global([5], [1], 64)
    assert wide_kv.applies(wide["k_cache"])
    assert not wide_kv.applies(wide["k_cache"].to(torch.bfloat16))
    qwen = make_case([5], [1], 64, inst=QWEN)
    assert qwen["k_cache"].dtype == FP8 and not wide_kv.applies(qwen["k_cache"])


def test_refuses_under_graph_capture():
    kw = _gemma_global([300], [1], 64)
    kw["max_seqlen_q"] = 1
    fa4, graph = _fa4(), torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        with torch.cuda.graph(graph):
            wide_kv.attend(fa4, kw, "decode")

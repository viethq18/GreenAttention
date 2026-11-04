"""Routing rules of ``green_attention.dispatch`` (CPU only)."""

import importlib
import logging

import pytest
import torch


@pytest.fixture
def dispatch(monkeypatch):
    def load(disable=""):
        monkeypatch.setenv("GREEN_ATTN_DISABLE", disable)
        import green_attention.dispatch as module

        return importlib.reload(module)

    return load


def _fallback(**kwargs):
    return "fallback"


def _call(wrapped, max_seqlen_q, dtype=torch.bfloat16):
    return wrapped(q=torch.zeros(2, 24, 256, dtype=dtype), max_seqlen_q=max_seqlen_q)


def test_decode_and_verify_route_by_query_length(dispatch):
    d = dispatch()
    d.register("decode", lambda kw: lambda **_: "decode")
    d.register("verify", lambda kw: lambda **_: "verify")
    wrapped = d.with_kvcache(_fallback)
    assert _call(wrapped, 1) == "decode"
    assert _call(wrapped, 4) == "verify"
    # prefill-sized queries and graph-time symbolic lengths never go green
    assert _call(wrapped, d.MAX_VERIFY_Q + 1) == "fallback"
    assert _call(wrapped, None) == "fallback"


def test_disable_env_forces_fallback_per_op(dispatch):
    d = dispatch("decode")
    d.register("decode", lambda kw: lambda **_: "decode")
    d.register("verify", lambda kw: lambda **_: "verify")
    wrapped = d.with_kvcache(_fallback)
    assert _call(wrapped, 1) == "fallback"
    assert _call(wrapped, 4) == "verify"


def test_disable_env_rejects_unknown_op(dispatch):
    with pytest.raises(ValueError, match="unknown ops"):
        dispatch("decode,prefill")


def test_first_accepting_selector_wins(dispatch):
    d = dispatch()
    d.register("decode", lambda kw: "head_dim unsupported")
    d.register("decode", lambda kw: lambda **_: "second")
    assert _call(d.with_kvcache(_fallback), 1) == "second"


def test_decline_logs_once_per_signature(dispatch, caplog):
    d = dispatch()
    d.register("decode", lambda kw: f"q.dtype={kw['q'].dtype}")
    wrapped = d.with_kvcache(_fallback)
    with caplog.at_level(logging.INFO, logger=d.__name__):
        for _ in range(3):
            assert _call(wrapped, 1) == "fallback"
        assert _call(wrapped, 1, torch.float16) == "fallback"
    lines = [r.getMessage() for r in caplog.records if "fallback" in r.getMessage()]
    assert len(lines) == 2
    assert "q.dtype=torch.bfloat16" in lines[0] and "q.dtype=torch.float16" in lines[1]


def test_declined_decode_casts_bf16_query_for_fp8_fallback(dispatch):
    # GreenAttentionBackend leaves the decode query in bf16; FA4 needs it in the KV dtype.
    d = dispatch()
    d.register("decode", lambda kw: "declined")
    seen = {}
    wrapped = d.with_kvcache(lambda **kw: seen.setdefault("q", kw["q"]))
    kc = torch.zeros(1, 16, 4, 256, dtype=torch.float8_e4m3fn)
    q = torch.full((2, 24, 256), 1.5, dtype=torch.bfloat16)
    wrapped(q=q, k_cache=kc, max_seqlen_q=1)
    assert seen["q"].dtype == torch.float8_e4m3fn and seen["q"].float().eq(1.5).all()
    seen.clear()
    wrapped(q=q, k_cache=kc.to(torch.bfloat16), max_seqlen_q=1)  # bf16 KV: query untouched
    assert seen["q"] is q


@pytest.mark.parametrize("disable", ["", "extend"])
def test_wide_fp8_prefill_routes_to_extend_then_dequant(dispatch, monkeypatch, disable):
    # A prefill over an FP8 cache FA4 cannot attend natively is the "extend" op; declined or disabled it
    # takes the dequantizing fallback, never FA4 directly.
    from green_attention import wide_kv

    d = dispatch(disable)
    monkeypatch.setattr(wide_kv, "applies", lambda kc: kc is not None and kc.shape[-1] > 256)
    monkeypatch.setattr(wide_kv, "attend", lambda fallback, kw, op: f"dequant:{op}")
    d.register("extend", lambda kw: lambda **_: "extend")
    wrapped = d.with_kvcache(_fallback)
    wide = torch.zeros(1, 16, 4, 512, dtype=torch.float8_e4m3fn)
    want = "dequant:extend" if disable else "extend"
    for mq in (d.MAX_VERIFY_Q + 1, None):
        assert wrapped(q=torch.zeros(2, 32, 512), k_cache=wide, max_seqlen_q=mq) == want
    narrow = torch.zeros(1, 16, 4, 256, dtype=torch.float8_e4m3fn)  # FA4 runs it natively: never "extend"
    assert wrapped(q=torch.zeros(2, 24, 256), k_cache=narrow, max_seqlen_q=d.MAX_VERIFY_Q + 1) == "fallback"

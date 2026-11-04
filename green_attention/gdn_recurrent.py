"""SM90 GDN recurrent kernels: decode (packed + split) and Replay-SSM verify.

Decode plugs in through :mod:`green_attention.dispatch` (op ``gdn_decode``), the
two call forms of :class:`sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel`.
The production verify path does not go through that class: with
``--enable-linear-replayssm-spec`` the backend calls
:func:`sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode.gdn_replayssm_spec_decode`
directly (keyword args, ``launch_mode="verify"``), so it is wrapped with an
AROUND hook on that function, registered here at import time.

Unsupported inputs are declined (a string reason) and fall back to the stock
Triton kernels; the reason is logged once per signature by the dispatcher.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "AROUND"),
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py": (
        "gdn_replayssm_spec_decode",
    ),
    "srt/layers/attention/linear/kernels/gdn_triton.py": ("TritonGDNKernel",),
}

TARGET = (
    "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
    "gdn_replayssm_spec_decode"
)

# Qwen3.8-27B GDN geometry (the only supported one in Phase 1).
H, HV, K, V = 16, 48, 128, 128
SCALE = K**-0.5
MAX_SPEC = 4
MAX_RING = 32
# Workspace capacity in sequences (fp32): ~118 KB per sequence, ~30 MB total.
WS_BATCH = 256
# Verify runs green for fp32 state at every batch (ABBA: 1.03x at B=1 up to 1.38x
# at B>=12 vs Triton). bf16-state verify is 0.74-0.90x Triton and stays on Triton;
# the kernel stays compilable and covered by op_test.
DUMP_PATH = "/network-volume/rnd-sglang/runs/green/gdn_dump.pt"

_module = None
_ws: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def module():
    """Compile (once) and return the kernel module."""
    global _module
    if _module is None:
        from green_attention import jit

        _module = jit.load("gdn_recurrent", "sm90/gdn_recurrent/launcher.cu")
    return _module


def _workspace(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The verify scratch written by the prep grid and read by the main grid.

    Indexed per (batch, head): safe as a single buffer because the scheduler
    runs one step at a time on one stream. Sized once for WS_BATCH sequences.
    """
    global _ws
    if _ws is None or _ws[0].device != device:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        _ws = (
            torch.empty(WS_BATCH * H * 8 * K, dtype=torch.float32, device=device),
            torch.empty(WS_BATCH * HV * MAX_RING * 8, dtype=torch.float32, device=device),
            torch.empty(WS_BATCH * HV * 32, dtype=torch.float32, device=device),
        )
    return _ws


# --------------------------------------------------------------------------- decode


def _decode_common(q, k, v, a, b, A_log, dt_bias, state, idx, reasons):
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        reasons.append("q/k/v dtype")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        reasons.append("a/b dtype")
    if A_log.dtype is not torch.float32 or A_log.numel() != HV:
        reasons.append("A_log")
    if dt_bias.numel() != HV or dt_bias.dtype not in (torch.float32, torch.bfloat16):
        reasons.append("dt_bias")
    if idx.ndim != 1 or idx.dtype not in (torch.int32, torch.int64):
        reasons.append(f"cache_indices {idx.dtype} {tuple(idx.shape)}")
    if state.dtype not in (torch.float32, torch.bfloat16):
        reasons.append(f"state dtype {state.dtype}")
    if tuple(state.shape[1:]) != (HV, V, K):
        reasons.append(f"state shape {tuple(state.shape)}")
    if state.stride(1) != V * K or state.stride(2) != K or state.stride(3) != 1:
        reasons.append("state strides")
    if not (a.stride(-1) == 1 and b.stride(-1) == 1):
        reasons.append("a/b last stride")
    if a.stride(0) % 8 or b.stride(0) % 8:
        reasons.append("a/b alignment")
    scale_ok = reasons == []
    return scale_ok


def _decode_rows(state, B):
    if state.dtype is torch.bfloat16:
        return 16 if B >= 32 else 8
    return 8


def _aligned16(t) -> bool:
    return t.data_ptr() % 16 == 0


def _select_packed(kwargs):
    from green_attention.gdn import _green_on_sm90

    if not _green_on_sm90():
        return "not on SM90 green backend"
    mq, a, b = kwargs.get("mixed_qkv"), kwargs.get("a"), kwargs.get("b")
    state, idx = kwargs.get("ssm_states"), kwargs.get("cache_indices")
    if not isinstance(mq, torch.Tensor) or not isinstance(state, torch.Tensor):
        return "missing tensor"
    # The Replay-SSM buffered-decode path (ring caches) is out of scope.
    for ring in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_write_pos"):
        if kwargs.get(ring) is not None:
            return f"replayssm buffered decode ({ring}) unsupported"
    reasons = []
    if mq.ndim != 2 or mq.shape[-1] != 2 * H * K + HV * V or mq.stride(-1) != 1:
        reasons.append(f"mixed_qkv {tuple(mq.shape)}")
    if not _decode_common(
        mq, mq, mq, a, b, kwargs.get("A_log"), kwargs.get("dt_bias"), state, idx, reasons
    ):
        return "; ".join(reasons) or "decode unsupported"
    if not (_aligned16(mq) and _aligned16(a) and _aligned16(b)):
        return "16B alignment"
    scale = kwargs.get("scale")
    if scale is not None and scale != SCALE:
        return f"scale {scale}"

    mod = module()
    B = mq.shape[0]
    rows = _decode_rows(state, B)

    def run(mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, **kw):
        _maybe_dump(dict(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            ssm_states=ssm_states, cache_indices=cache_indices, scale=SCALE,
        ), "decode")
        out = mixed_qkv.new_empty(B, 1, HV, V)
        mod.gdn_decode(
            mixed_qkv, mixed_qkv, mixed_qkv, a, b, A_log, dt_bias,
            cache_indices, cache_indices, ssm_states, out,
            mixed_qkv.stride(0), mixed_qkv.stride(0), mixed_qkv.stride(0),
            H * K, 2 * H * K, a.stride(0), b.stride(0),
            0, 1, rows, B, SCALE,
        )
        return out.transpose(0, 1)  # [1, B, HV, V] like Triton

    return run


def _select_split(kwargs):
    from green_attention.gdn import _green_on_sm90

    if not _green_on_sm90():
        return "not on SM90 green backend"
    q, k, v = kwargs.get("q"), kwargs.get("k"), kwargs.get("v")
    a, b = kwargs.get("a"), kwargs.get("b")
    state, idx = kwargs.get("ssm_states"), kwargs.get("cache_indices")
    cu = kwargs.get("query_start_loc")
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, state, cu)):
        return "missing tensor"
    reasons = []
    if not _decode_common(
        q, k, v, a, b, kwargs.get("A_log"), kwargs.get("dt_bias"), state, idx, reasons
    ):
        return "; ".join(reasons) or "decode unsupported"
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != cu.numel() - 1:
        reasons.append(f"q shape {tuple(q.shape)}")
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or v.shape[3] != V:
        reasons.append("k/v shape")
    for name, t, dim in (("q", q, K), ("k", k, K), ("v", v, V)):
        if t.stride(2) != dim or t.stride(3) != 1 or t.stride(1) % 8:
            reasons.append(f"{name} strides {t.stride()}")
    if cu.dtype is not torch.int32:
        reasons.append(f"query_start_loc dtype {cu.dtype}")
    if not (_aligned16(q) and _aligned16(k) and _aligned16(v)):
        reasons.append("16B alignment")
    if reasons:
        return "; ".join(reasons)
    scale = kwargs.get("scale")
    if scale is not None and scale != SCALE:
        return f"scale {scale}"

    mod = module()
    B = q.shape[1]
    rows = _decode_rows(state, B)

    def run(q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, query_start_loc, **kw):
        out = q.new_empty(1, B, HV, V)
        mod.gdn_decode(
            q, k, v, a, b, A_log, dt_bias, cache_indices, query_start_loc,
            ssm_states, out, q.stride(1), k.stride(1), v.stride(1),
            0, 0, a.stride(0), b.stride(0), 1, 0, rows, B, SCALE,
        )
        return out

    return run

def _run_verify(call):
    """Launch the green verify pair on ``call`` (already validated)."""
    mod = module()
    q = call["q"]
    q_t = q.stride(0)  # token stride: [T, H, K] or [T, H*K]
    k_t = call["k"].stride(0)
    v_t = call["v"].stride(0)
    a_t = call["a"].stride(0) if call["a"].ndim == 2 else call["a"].stride(-2)
    b_t = call["b"].stride(0) if call["b"].ndim == 2 else call["b"].stride(-2)
    out = call["out"]
    o_t = out.stride(0) if out.ndim == 2 else out.stride(0)
    ws = _workspace(q.device)
    mod.gdn_verify(
        q, call["k"], call["v"], call["a"], call["b"], call["A_log"],
        call["dt_bias"], call["checkpoint_state"], call["d_cache"], call["rawv_cache"],
        call["k_cache"], call["rawk_cache"], call["g_cache"], call["ssm_state_indices"],
        call["replay_indices"], call["query_start_loc"], call["write_pos"], call["cache_base"],
        call["is_flush"], out, ws[0], ws[1], ws[2],
        q_t, k_t, v_t, a_t, b_t, o_t,
        call["max_cache_len"], int(call["replay_indices"].numel()), SCALE,
    )
    return out


def _verify_can_use(call) -> "str | None":
    """The decline reason, or None when the green verify kernel applies."""
    from green_attention.gdn import _green_on_sm90

    if not _green_on_sm90():
        return "not on SM90 green backend"
    reason = _verify_regime(call)
    if reason is not None:
        return reason
    return _verify_supported(call)




def _verify_regime(call) -> "str | None":
    """Performance gates (measured crossovers), not correctness limits."""
    st = call.get("checkpoint_state")
    if not isinstance(st, torch.Tensor) or st.dtype not in (torch.float32, torch.bfloat16):
        return f"state dtype {getattr(st, 'dtype', st)}"
    if st.dtype is torch.bfloat16:
        return "bf16 state verify: Triton is faster on this regime"
    return None


def _verify_supported(call) -> "str | None":
    """Correctness/support limits: everything the kernel cannot run on."""
    if call.get("launch_mode") != "verify":
        return f"launch_mode {call.get('launch_mode')!r}"
    st = call.get("checkpoint_state")
    if not isinstance(st, torch.Tensor):
        return "no checkpoint_state"
    if st.dtype not in (torch.float32, torch.bfloat16):
        return f"state dtype {st.dtype}"
    B = int(call["replay_indices"].numel())
    for name in ("d_cache", "rawv_cache", "k_cache", "rawk_cache", "g_cache"):
        if not isinstance(call.get(name), torch.Tensor):
            return f"missing {name}"
    L = int(call.get("max_cache_len") or 0)
    if L not in (16, 32) or call["d_cache"].shape[2] != L:
        return f"max_cache_len {L} vs d_cache {tuple(call['d_cache'].shape)}"
    for name, shape, dt in (
        ("d_cache", (HV, L, V), torch.bfloat16),
        ("rawv_cache", (HV, L, V), torch.bfloat16),
        ("k_cache", (H, L, K), torch.bfloat16),
        ("rawk_cache", (H, L, K), torch.bfloat16),
        ("g_cache", (HV, L), torch.float32),
    ):
        t = call[name]
        if t.dtype is not dt:
            return f"{name} dtype {t.dtype}"
        if t.ndim != len(shape) + 1 or tuple(t.shape[1:]) != shape:
            return f"{name} shape {tuple(t.shape)}"
        if t.stride(-1) != 1 or any(
            t.stride(n) != t.shape[n + 1] * t.stride(n + 1) for n in range(1, t.ndim - 1)
        ):
            return f"{name} strides {t.stride()}"
        if t.stride(0) % 8:
            return f"{name} slot stride {t.stride(0)}"
    if call.get("beta_cache") is not None:
        return "beta_cache unsupported"
    if int(call.get("max_spec_len") or 0) > MAX_SPEC:
        return f"max_spec_len {call.get('max_spec_len')}"
    if call.get("use_qk_l2norm_in_kernel") is not True:
        return "use_qk_l2norm_in_kernel must be True"
    if call.get("null_block_id") != -1:
        return f"null_block_id {call.get('null_block_id')}"
    scale = call.get("scale")
    if scale is not None and scale != SCALE:
        return f"scale {scale}"
    reasons = []
    for name, n, dim in (("q", H, K), ("k", H, K), ("v", HV, V)):
        t = call[name]
        if t.dtype is not torch.bfloat16:
            reasons.append(f"{name} dtype {t.dtype}")
        # [T, n, dim] with a dense head dim, or the flattened [T, n*dim]
        if t.ndim == 3:
            if t.shape[1:] != (n, dim) or t.stride(2) != 1 or t.stride(1) != dim or t.stride(0) % 8:
                reasons.append(f"{name} layout {tuple(t.shape)} {t.stride()}")
        elif t.ndim == 2:
            if t.shape[1] != n * dim or t.stride(1) != 1 or t.stride(0) % 8:
                reasons.append(f"{name} layout {tuple(t.shape)} {t.stride()}")
        else:
            reasons.append(f"{name} layout {tuple(t.shape)}")
    for name, dt in (("a", torch.bfloat16), ("b", torch.bfloat16), ("A_log", torch.float32)):
        if call[name].dtype is not dt:
            reasons.append(f"{name} dtype {call[name].dtype}")
    if reasons:
        return "; ".join(reasons)
    for name in ("ssm_state_indices", "replay_indices"):
        if call[name].dtype not in (torch.int32, torch.int64):
            return f"{name} dtype {call[name].dtype}"
    for name in ("write_pos", "cache_base"):
        if call[name].dtype is not torch.int32:
            return f"{name} dtype {call[name].dtype}"
    if call["is_flush"].dtype is not torch.int8:
        return f"is_flush dtype {call['is_flush'].dtype}"
    qsl = call["query_start_loc"]
    if qsl.dtype is not torch.int32 or qsl.ndim != 1:
        return f"query_start_loc {qsl.dtype} {tuple(qsl.shape)}"
    if B > WS_BATCH:
        return f"batch {B} > workspace capacity {WS_BATCH}"
    out = call.get("out")
    if not isinstance(out, torch.Tensor):
        return "out missing"
    # [T, HV, V] (dense heads) or [T, HV*V]
    if out.ndim == 3:
        if out.shape[1:] != (HV, V) or out.stride(2) != 1 or out.stride(1) != V or out.stride(0) % 8:
            return f"out layout {tuple(out.shape)} {out.stride()}"
    elif out.ndim == 2:
        if out.shape[1] != HV * V or out.stride(1) != 1 or out.stride(0) % 8:
            return f"out layout {out.stride()}"
    else:
        return f"out shape {tuple(out.shape)}"


def _replayssm_around(original, *args, **kwargs):
    from green_attention import dispatch
    from green_attention.gdn import _green_on_sm90

    # Graph-safe (pre-allocated workspace, no host sync), so it also runs while the
    # target-verify CUDA graph is captured: that is the production path.
    if kwargs and _green_on_sm90() and dispatch.enabled("gdn_verify"):
        reason = _verify_can_use(kwargs)
        if reason is None:
            _maybe_dump(kwargs, "verify", original)
            return _run_verify(kwargs)
        dispatch.log_fallback("gdn_verify", (reason,), kwargs)
    return original(*args, **kwargs)


def _select_verify(kwargs):
    # The A1 dispatch call is the non-Replay-SSM target_verify tree form.
    return "non-replayssm target_verify unsupported; prod verify served by the replayssm hook"


# --------------------------------------------------------------------------- dump / prewarm


def _maybe_dump(call, kind, reference=None):
    """With GREEN_DUMP_GDN=1, snapshot one real step (inputs + the stock kernel's
    output, computed on copies; the live buffers are not touched). Fires once per
    process and never while a CUDA graph is captured: run that server with
    --disable-cuda-graph. ``reference`` is the unhooked replayssm function."""
    if os.environ.get("GREEN_DUMP_GDN") != "1":
        return
    if torch.cuda.is_current_stream_capturing() or getattr(_maybe_dump, "_done", False):
        return
    _maybe_dump._done = True
    try:
        _dump_step(call, kind, reference)
    except Exception:  # noqa: BLE001 - the dump must never break serving
        logger.exception("green_attention: GREEN_DUMP_GDN dump failed")


def _dump_step(call, kind, reference):
    os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
    clone = lambda c: {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in c.items()}  # noqa: E731
    if kind == "verify":
        inputs, snap = clone(call), clone(call)
        ref = reference(**snap)
        payload = dict(kind=kind, call=inputs, ref_out=ref.clone())
    else:  # packed decode: only the live state rows, re-indexed 0..B-1 (PAD kept)
        st, idx = call["ssm_states"], call["cache_indices"]
        local = torch.where(idx >= 0, torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype), idx)
        inputs = clone(dict(call, ssm_states=None, cache_indices=local))
        inputs["ssm_states"] = st[idx.to(torch.long).clamp(min=0)].clone()
        post = inputs["ssm_states"].clone()
        ref = _dump_decode_reference(dict(inputs, ssm_states=post))
        payload = dict(kind=kind, call=inputs, ref_out=ref.clone(), post_state=post)
    torch.save(payload, DUMP_PATH)
    logger.info("green_attention: dumped one %s step to %s", kind, DUMP_PATH)


def _dump_decode_reference(call):
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    return TritonGDNKernel().packed_decode(**call)


def _prewarm(backend=None):
    """Compile every kernel instance before CUDA-graph capture (A1 calls this)."""
    module()
    if torch.cuda.is_available():
        _workspace(torch.device("cuda"))


# Registration happens at import: gdn.py imports this module, plugin.register()
# imports gdn.py, and the hook must be in place before gdn_backend first calls
# gdn_replayssm_spec_decode (it resolves the function at call time).
def _register_all():
    from green_attention import dispatch
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    dispatch.register("gdn_decode", _select_packed, _prewarm)
    dispatch.register("gdn_decode", _select_split)
    dispatch.register("gdn_verify", _select_verify)
    HookRegistry.register(TARGET, _replayssm_around, HookType.AROUND)


_register_all()

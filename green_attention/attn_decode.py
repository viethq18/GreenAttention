"""SM90 FP8-KV decode / DFLASH-verify attention (ops ``decode`` and ``verify``).

Specialized scope (anything else is declined with a reason and runs on the FA4 fallback):
  * q ``[total_q, 24, 256]`` e4m3 (or bf16, cast in-kernel); K/V cache e4m3
    ``[pages, page_size, 4, 256]``, any power-of-two page_size, paged by ``page_table``;
  * ``max_seqlen_q`` in 1..10 (6 q-heads x q_len rows must fit one 64-row WGMMA tile),
    causal (bottom-right aligned) for q_len > 1;
  * no softcap / window / sinks / score_mod / rotary / extra K,V update.
Output is bf16 ``[total_q, 24, 256]`` (honors ``out=``); with ``return_softmax_lse`` the call returns
``(out, lse)`` with lse fp32 ``[24, total_q]`` in natural log, exactly like FA4.

The grid is sized by ``plan_splits`` (an upper bound on splits that depends only on the bucketed batch
size and the static max context, so it is identical at CUDA graph capture and replay). The number of
splits each sequence actually uses is planned on the device from ``cache_seqlens`` (see
``plan_quota`` in kernel.cuh), which keeps graph replay valid while lengths change.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch

from green_attention import dispatch, jit

HEAD_DIM = 256
KV_HEADS = 4
Q_HEADS = 24
GQA = Q_HEADS // KV_HEADS
MAX_Q = 10
TILE_N = 128
SPLIT_TILE_TOKENS = 256  # grid upper bound: at most one split per 256 tokens of context
# Worst case CTA count with splits > 1: batch*4*ceil(2*132/(4*batch)) < 2*132 + 4*batch, batch < 66.
MAX_SPLIT_CTAS = 2 * 132 + 4 * 66
WS_ROWS = 64

_ALLOWED_EXTRA = frozenset(
    (
        "q", "k_cache", "v_cache", "page_table", "cache_seqlens", "cu_seqlens_q", "max_seqlen_q",
        "softmax_scale", "causal", "window_size", "softcap", "return_softmax_lse", "num_splits", "out",
        "scheduler_metadata", "k_descale", "v_descale", "max_seqlen_k", "cu_seqlens_k_new", "pack_gqa",
        "sm_margin", "rel_bias_prep_cache",
    )
)


class _State:
    mod = None
    sm_count: dict = {}
    ws: dict = {}  # device index -> (ws_o, ws_lse, dummy_f32, ws_eff)


_S = _State()


def _is_sm90(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0)


def ensure_ready(device: Union[torch.device, int, None] = None) -> None:
    """Compile/load the kernel and allocate the split workspace (idempotent). Must run before graph capture."""
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if _S.mod is None:
        mod = jit.load("attn_decode_fp8", "sm90/attn_decode_fp8/launcher.cu")
        mod.init()
        _S.mod = mod
    if dev.index not in _S.ws:
        ws_o = torch.empty((MAX_SPLIT_CTAS, WS_ROWS, HEAD_DIM), dtype=torch.float32, device=dev)
        ws_lse = torch.empty((MAX_SPLIT_CTAS, WS_ROWS), dtype=torch.float32, device=dev)
        dummy = torch.zeros((1, KV_HEADS), dtype=torch.float32, device=dev)
        ws_eff = torch.zeros((MAX_SPLIT_CTAS,), dtype=torch.int32, device=dev)
        _S.ws[dev.index] = (ws_o, ws_lse, dummy, ws_eff)
        _S.sm_count[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count


def _prewarm(backend) -> None:
    if torch.cuda.is_available() and _is_sm90(torch.device("cuda", torch.cuda.current_device())):
        ensure_ready()


def plan_splits(batch: int, max_ctx: int, sm_count: int = 132) -> int:
    """Grid upper bound: clamp(ceil(2*SMs / (batch*4)), 1, ceil(ctx/256))."""
    want = -(-2 * sm_count // (batch * KV_HEADS))
    cap = max(-(-max_ctx // SPLIT_TILE_TOKENS), 1)
    return max(min(want, cap), 1)


def _decline_extras(kw: dict) -> Optional[str]:
    for name, val in kw.items():
        if name not in _ALLOWED_EXTRA and val is not None:
            return f"unsupported arg {name}"
    return None


def _check_descale(t, batch: int, name: str) -> Optional[str]:
    if t is None:
        return None
    if t.dtype != torch.float32 or t.dim() != 2 or t.shape[1] != KV_HEADS or t.shape[0] != batch:
        return f"{name} must be fp32 [batch, {KV_HEADS}]"
    return None


def select(kw: dict) -> Union[Callable, str]:
    why = _decline_extras(kw)
    if why:
        return why
    q, kc, vc = kw.get("q"), kw.get("k_cache"), kw.get("v_cache")
    pt, cs, cu = kw.get("page_table"), kw.get("cache_seqlens"), kw.get("cu_seqlens_q")
    if q is None or kc is None or vc is None or pt is None or cs is None or cu is None:
        return "not a paged varlen call"
    if not isinstance(cs, torch.Tensor):
        return "cache_seqlens must be a tensor"
    mq = kw.get("max_seqlen_q")
    if not isinstance(mq, int) or not 1 <= mq <= MAX_Q:
        return f"max_seqlen_q {mq} outside 1..{MAX_Q}"
    if mq > 1 and not kw.get("causal", False):
        return "non-causal verify"
    if kw.get("softcap", 0.0):
        return "softcap"
    ws = kw.get("window_size", (-1, -1))
    if tuple(ws) != (-1, -1):
        return "window"
    if q.dim() != 3 or q.shape[1] != Q_HEADS or q.shape[2] != HEAD_DIM or q.stride(2) != 1:
        return "q shape/stride"
    if q.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        return f"q dtype {q.dtype}"
    if kc.dtype != torch.float8_e4m3fn or vc.dtype != torch.float8_e4m3fn:
        return "kv dtype is not e4m3"
    for t in (kc, vc):
        if t.dim() != 4 or t.shape[2] != KV_HEADS or t.shape[3] != HEAD_DIM or t.stride(3) != 1:
            return "kv layout"
    if kc.shape[:2] != vc.shape[:2]:
        return "k/v page geometry differ"
    page = kc.shape[1]
    if page & (page - 1):
        return "page_size not a power of two"
    if pt.dtype != torch.int32 or pt.dim() != 2 or cs.dtype != torch.int32 or cs.dim() != 1 or cs.stride(0) != 1:
        return "page_table/cache_seqlens dtype/layout"
    if cu.dtype != torch.int32 or cu.dim() != 1 or cu.stride(0) != 1:
        return "cu_seqlens_q dtype/layout"
    batch = cs.shape[0]
    if batch == 0 or pt.shape[0] != batch or cu.shape[0] != batch + 1:
        return "batch mismatch"
    why = _check_descale(kw.get("k_descale"), batch, "k_descale") or _check_descale(kw.get("v_descale"), batch, "v_descale")
    if why:
        return why
    out = kw.get("out")
    if out is not None and (
        out.dtype != torch.bfloat16 or tuple(out.shape) != tuple(q.shape) or out.stride(2) != 1
    ):
        return "out layout"
    dev = q.device
    if not q.is_cuda or dev.index not in _S.ws:
        return "kernel not prewarmed for this device"
    msk = kw.get("max_seqlen_k")
    max_ctx = msk if isinstance(msk, int) else pt.shape[1] * page
    splits = plan_splits(batch, max_ctx, _S.sm_count[dev.index])
    if splits > 1 and batch * KV_HEADS * splits > MAX_SPLIT_CTAS:
        return "split workspace too small"
    return lambda **call_kw: _run(splits, page, **call_kw)


def _run(splits: int, page: int, *, q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
         softmax_scale=None, return_softmax_lse=False, out=None, k_descale=None, v_descale=None, **_):
    dev = q.device.index
    ws_o, ws_lse, dummy, ws_eff = _S.ws[dev]
    total_q = q.shape[0]
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((Q_HEADS, total_q), dtype=torch.float32, device=q.device) if return_softmax_lse else dummy
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    q_bf16 = q.dtype == torch.bfloat16
    _S.mod.attn_decode(
        q.view(torch.int16) if q_bf16 else q.view(torch.uint8),
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        dummy if k_descale is None else k_descale,
        dummy if v_descale is None else v_descale,
        out.view(torch.int16),
        lse,
        ws_o,
        ws_lse,
        ws_eff,
        int(q_bf16),
        splits,
        page.bit_length() - 1,
        int(return_softmax_lse),
        int(k_descale is not None),
        int(v_descale is not None),
        cache_seqlens.shape[0],
        max_seqlen_q,
        _S.sm_count[dev],
        float(softmax_scale),
    )
    return (out, lse) if return_softmax_lse else out


dispatch.register("decode", select, prewarm=_prewarm)
dispatch.register("verify", select)

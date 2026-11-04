"""SM90 FP8-KV decode / DFLASH-verify / extend attention (ops ``decode``, ``verify`` and ``extend``).

Templated kernel instances (see docs/gemma4.md); kv_heads is a runtime grid dimension:

  =================  ========  ===  ===  ======  ============================================
  model group        head_dim  GQA  kvh  max_q   window
  =================  ========  ===  ===  ======  ============================================
  Qwen3.8-27B full        256    6    4     10  none
  Gemma-4 sliding         256    2   16     16  window_left >= 0: query at position p attends
                                                    keys [p - W, p] (FA left-window semantics);
                                                    sglang decode passes window_size (W, 0)
  Gemma-4 global          512    8    4      8  none (head_dim 512 = DSPLIT 2 x 256-dim CTAs)
  =================  ========  ===  ===  ======  ============================================

The window is a compile-time property of the instance (``Cfg<..., WINDOW>``): the runtime lower-bound
mask alone costs ~5-7 % on full attention, so only the sliding instance carries it. A left window on
another instance is declined (FA4 fallback).

Specialized scope (anything else is declined with a reason and runs on the FA4 fallback):
  * q ``[total_q, q_heads, head_dim]`` e4m3 (or bf16, cast in-kernel); K/V cache e4m3
    ``[pages, page_size, kv_heads, head_dim]``, any power-of-two page_size, paged by ``page_table``;
  * ``max_seqlen_q`` in 1..max_q (GQA x q_len rows must fit one 64-row WGMMA tile), causal
    (bottom-right aligned) for q_len > 1;
  * no softcap / right window / sinks / score_mod / rotary / extra K,V update.
Output is bf16 ``[total_q, q_heads, head_dim]`` (honors ``out=``); with ``return_softmax_lse`` the
call returns ``(out, lse)`` with lse fp32 ``[q_heads, total_q]`` in natural log, exactly like FA4.

The grid is sized by ``plan_splits`` (an upper bound on splits that depends only on shapes and the
static max context, so it is identical at CUDA graph capture and replay). The number of splits each
sequence actually uses is planned on the device from ``cache_seqlens`` (and the window, see
``seq_tiles`` / ``plan_quota`` in kernel.cuh), which keeps graph replay valid while lengths change.

``extend`` (prefill / extend over the FP8 cache, any q_len, causal, no window) runs a separate kernel
(``extend_kernel.cuh``): each CTA takes a block of ``EXTEND_BLOCK_Q`` query tokens of one sequence, split over two
consumer warpgroups that share every gathered K / V tile, over the prefix its last token sees. Only the Gemma-4
global shape has it: FA4 cannot attend an FP8 cache with head_dim 512 on SM90, every other shape stays on FA4.
The grid depends only on ``q.shape[0]`` and the batch size, so it is CUDA-graph safe.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch

from green_attention import dispatch, jit

TILE_N = 128
SPLIT_TILE_TOKENS = 256  # grid upper bound: at most one split per 256 tokens of context
# (head_dim, GQA) -> max_q (= min(64 / GQA, 16)). kv_heads is runtime (k_cache.shape[2]);
# head_dim 512 runs as DSPLIT = 2 CTAs per kv head.
INSTANCES = {
    (256, 6): 10,
    (256, 2): 16,
    (512, 8): 8,
}
# Instances compiled with the sliding-window mask (Cfg WINDOW = true); see the module docstring.
WINDOWED = frozenset({(256, 2)})
MAX_Q = max(INSTANCES.values())
# (head_dim, GQA) with an extend kernel, and its query block (2 consumer warpgroups x 64 rows / GQA 8).
EXTEND_INSTANCES = frozenset({(512, 8)})
EXTEND_BLOCK_Q = 16
MAX_GRID_Z = 65535
WS_ROWS = 64
WS_HD = 256
# Worst case CTA count with splits > 1: batch*units*ceil(2*132/(batch*units)) < 2*132 + units*batch,
# units = kv_heads * DSPLIT; 66 covers kv_heads <= 16 with DSPLIT 1 (Gemma sliding), the largest unit count.
MAX_SPLIT_CTAS = 2 * 132 + 16 * 66
COMBINE_MAX_SPLITS = 192

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
        ws_o = torch.empty((MAX_SPLIT_CTAS, WS_ROWS, WS_HD), dtype=torch.float32, device=dev)
        ws_lse = torch.empty((MAX_SPLIT_CTAS, WS_ROWS), dtype=torch.float32, device=dev)
        dummy = torch.zeros((1, 1), dtype=torch.float32, device=dev)
        ws_eff = torch.zeros((MAX_SPLIT_CTAS,), dtype=torch.int32, device=dev)
        _S.ws[dev.index] = (ws_o, ws_lse, dummy, ws_eff)
        _S.sm_count[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count


def _prewarm(backend) -> None:
    if torch.cuda.is_available() and _is_sm90(torch.device("cuda", torch.cuda.current_device())):
        ensure_ready()


def plan_splits(batch: int, max_ctx: int, sm_count: int = 132, units: int = 4) -> int:
    """Grid upper bound: clamp(ceil(2*SMs / (batch*units)), 1, ceil(ctx/256)).

    ``units`` = kv_heads * DSPLIT (CTAs per sequence in grid.y). The device planner may use fewer
    splits; the bound only has to be stable across graph capture/replay and keep the workspace sane.
    """
    want = -(-2 * sm_count // (batch * units))
    cap = max(-(-max_ctx // SPLIT_TILE_TOKENS), 1)
    return max(min(want, cap), 1)


def _decline_extras(kw: dict) -> Optional[str]:
    for name, val in kw.items():
        if name not in _ALLOWED_EXTRA and val is not None:
            return f"unsupported arg {name}"
    return None


def _check_descale(t, batch: int, kvh: int, name: str) -> Optional[str]:
    if t is None:
        return None
    if t.dtype != torch.float32 or t.dim() != 2 or t.shape[1] != kvh or t.shape[0] not in (batch, 1):
        return f"{name} must be fp32 [batch|1, {kvh}]"
    return None


def _window_left(kw: dict) -> Union[int, str]:
    """FA window semantics: (left, right). sglang decode passes (W, 0) for a causal sliding layer,
    (-1, -1) otherwise. A right window other than none is declined."""
    ws = kw.get("window_size", (-1, -1))
    if ws is None:
        return -1
    if len(ws) != 2:
        return "window"
    wl, wr = ws
    wl = -1 if wl is None else int(wl)
    wr = -1 if wr is None else int(wr)
    if wr not in (-1, 0):
        return f"window right {wr} must be -1 or 0"
    if wr == 0 and not kw.get("causal", False):
        return "window right 0 without causal"
    return wl


def _shape_reason(kw: dict) -> Optional[str]:
    """Checks shared by decode/verify and extend, up to (and excluding) the per-op instance / q_len rules."""
    why = _decline_extras(kw)
    if why:
        return why
    q, kc, vc = kw.get("q"), kw.get("k_cache"), kw.get("v_cache")
    pt, cs, cu = kw.get("page_table"), kw.get("cache_seqlens"), kw.get("cu_seqlens_q")
    if q is None or kc is None or vc is None or pt is None or cs is None or cu is None:
        return "not a paged varlen call"
    if not isinstance(cs, torch.Tensor):
        return "cache_seqlens must be a tensor"
    if q.dim() != 3 or q.stride(2) != 1 or kc.dim() != 4:
        return "q shape/stride"
    return None


def _paged_reason(kw: dict) -> Optional[str]:
    """dtype / layout / batch / descale / out / prewarm checks shared by decode/verify and extend."""
    q, kc, vc = kw["q"], kw["k_cache"], kw["v_cache"]
    pt, cs, cu = kw["page_table"], kw["cache_seqlens"], kw["cu_seqlens_q"]
    hd, kvh = q.shape[2], kc.shape[2]
    if q.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        return f"q dtype {q.dtype}"
    if kc.dtype != torch.float8_e4m3fn or vc.dtype != torch.float8_e4m3fn:
        return "kv dtype is not e4m3"
    for t in (kc, vc):
        if t.dim() != 4 or t.shape[3] != hd or t.stride(3) != 1:
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
    why = _check_descale(kw.get("k_descale"), batch, kvh, "k_descale") or _check_descale(
        kw.get("v_descale"), batch, kvh, "v_descale"
    )
    if why:
        return why
    out = kw.get("out")
    if out is not None and (out.dtype != torch.bfloat16 or tuple(out.shape) != tuple(q.shape) or out.stride(2) != 1):
        return "out layout"
    if not q.is_cuda or q.device.index not in _S.ws:
        return "kernel not prewarmed for this device"
    return None


def select(kw: dict) -> Union[Callable, str]:
    why = _shape_reason(kw)
    if why:
        return why
    q, kc, pt = kw["q"], kw["k_cache"], kw["page_table"]
    mq = kw.get("max_seqlen_q")
    hd, qh, kvh = q.shape[2], q.shape[1], kc.shape[2]
    inst = INSTANCES.get((hd, qh // kvh)) if kvh and qh % kvh == 0 else None
    if inst is None:
        return f"q shape: no instance for (head_dim={hd}, q_heads={qh}, kv_heads={kvh})"
    if not isinstance(mq, int) or not 1 <= mq <= inst:
        return f"max_seqlen_q {mq} outside 1..{inst}"
    if mq > 1 and not kw.get("causal", False):
        return "non-causal verify"
    if kw.get("softcap", 0.0):
        return "softcap"
    wl = _window_left(kw)
    if isinstance(wl, str):
        return wl
    if wl >= 0 and (hd, qh // kvh) not in WINDOWED:
        return f"window on full-attention instance (head_dim={hd}, GQA={qh // kvh})"
    why = _paged_reason(kw)
    if why:
        return why
    page = kc.shape[1]
    batch = kw["cache_seqlens"].shape[0]
    msk = kw.get("max_seqlen_k")
    max_ctx = msk if isinstance(msk, int) else pt.shape[1] * page
    if wl >= 0:
        # A windowed row sees at most W keys below its own position; + mq + one tile for alignment.
        max_ctx = min(max_ctx, wl + mq + TILE_N)
    units = kvh * (2 if hd == 512 else 1)
    splits = plan_splits(batch, max_ctx, _S.sm_count[q.device.index], units)
    if splits > 1 and batch * units * splits > MAX_SPLIT_CTAS:
        return "split workspace too small"
    return lambda **call_kw: _run(splits, page, window_left=wl, kvh=kvh, units=units, **call_kw)


def select_extend(kw: dict) -> Union[Callable, str]:
    """Prefill / extend over an FP8 cache (any q_len, causal, no window) on an EXTEND_INSTANCES instance."""
    why = _shape_reason(kw)
    if why:
        return why
    q, kc = kw["q"], kw["k_cache"]
    mq = kw.get("max_seqlen_q")
    hd, qh, kvh = q.shape[2], q.shape[1], kc.shape[2]
    if not kvh or qh % kvh or (hd, qh // kvh) not in EXTEND_INSTANCES:
        return f"q shape: no extend instance for (head_dim={hd}, q_heads={qh}, kv_heads={kvh})"
    if not isinstance(mq, int) or mq < 1:
        return f"max_seqlen_q {mq} is not a positive int"
    if mq > 1 and not kw.get("causal", False):
        return "non-causal extend"
    if kw.get("softcap", 0.0):
        return "softcap"
    wl = _window_left(kw)
    if isinstance(wl, str):
        return wl
    if wl >= 0:
        return "window on extend"
    why = _paged_reason(kw)
    if why:
        return why
    if q.shape[0] // EXTEND_BLOCK_Q + kw["cache_seqlens"].shape[0] > MAX_GRID_Z:
        return "too many query blocks for grid.z"
    page = kc.shape[1]
    return lambda **call_kw: _run_extend(page, **call_kw)


def _run(
    splits: int,
    page: int,
    *,
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    max_seqlen_q,
    window_left=-1,
    kvh=0,
    units=0,
    softmax_scale=None,
    return_softmax_lse=False,
    out=None,
    k_descale=None,
    v_descale=None,
    **_,
):
    ws_o, ws_lse, dummy, ws_eff = _S.ws[q.device.index]
    total_q, qh, hd = q.shape
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((qh, total_q), dtype=torch.float32, device=q.device) if return_softmax_lse else dummy
    if softmax_scale is None:
        softmax_scale = hd**-0.5
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
        _S.sm_count[q.device.index],
        window_left,
        float(softmax_scale),
    )
    return (out, lse) if return_softmax_lse else out


def _run_extend(
    page: int,
    *,
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    softmax_scale=None,
    return_softmax_lse=False,
    out=None,
    k_descale=None,
    v_descale=None,
    **_,
):
    dummy = _S.ws[q.device.index][2]
    total_q, qh, hd = q.shape
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((qh, total_q), dtype=torch.float32, device=q.device) if return_softmax_lse else dummy
    if softmax_scale is None:
        softmax_scale = hd**-0.5
    q_bf16 = q.dtype == torch.bfloat16
    _S.mod.attn_extend(
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
        int(q_bf16),
        page.bit_length() - 1,
        int(return_softmax_lse),
        int(k_descale is not None),
        int(v_descale is not None),
        cache_seqlens.shape[0],
        float(softmax_scale),
    )
    return (out, lse) if return_softmax_lse else out


dispatch.register("decode", select, prewarm=_prewarm)
dispatch.register("verify", select)
# A short prefill / extend (max_seqlen_q 2..MAX_VERIFY_Q) is a "verify" call; past the instance's max_q it
# still has the extend kernel. Runs only after `select` declined, so it costs the accepted calls nothing.
dispatch.register("verify", select_extend)
dispatch.register("extend", select_extend)

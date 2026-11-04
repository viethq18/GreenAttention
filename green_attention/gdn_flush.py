"""SM90 Replay-SSM commit ("flush"): the all-layer checkpoint fold + track snapshot.

With ``--enable-linear-replayssm-spec`` the verify forward only appends to a compact
ring (D / K / G). Once per step, outside any CUDA graph, the scheduler calls
:func:`sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode.commit_gdn_replayssm_circular`,
which folds the ring into the fp32 checkpoint for the rows whose ring is nearly full
(``is_flush``) and snapshots the checkpoint at the accepted step for ``mamba_track``
rows. This module replaces that call with an AROUND hook that runs two kernels (a one-CTA
job compaction + cursor reset, and a persistent fold over the compacted jobs). Unsupported
inputs, or ``GREEN_ATTN_DISABLE=gdn_flush``, run the stock Triton path. Contract:
``docs/gdn-contract.md`` section 7.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "AROUND"),
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py": (
        "commit_gdn_replayssm_circular",
    ),
}

TARGET = "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode.commit_gdn_replayssm_circular"

H, HV, K, V, T = 16, 48, 128, 128, 32
MAX_ROWS = 1024  # job-list capacity (flush.cuh kMaxRows)
CTAS_PER_SM = 2  # fold_kernel __launch_bounds__(256, 2)

_module = None
_ws: dict = {}
_sms: dict = {}


def module():
    """Compile (once) and return the kernel module."""
    global _module
    if _module is None:
        from green_attention import jit

        _module = jit.load("gdn_flush", "sm90/gdn_flush/launcher.cu")
    return _module


def _device(device) -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if device.index is None else device


def _workspace(device):
    """The compacted job list and the work counters: written by the prep CTA, read by the
    fold grid. One buffer is safe because the scheduler commits one step at a time on one
    stream. Allocated once (never inside graph capture)."""
    device = _device(device)
    ws = _ws.get(device)
    if ws is None:
        ws = (
            torch.empty(MAX_ROWS * 8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
        _ws[device] = ws
    return ws


def _grid(device) -> int:
    device = _device(device)
    n = _sms.get(device)
    if n is None:
        n = _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return n * CTAS_PER_SM


def _idx_ok(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in (torch.int32, torch.int64) and t.ndim == 1


def can_use(kw: dict):
    """None when the call is supported, else the reason it is declined."""
    st = kw.get("checkpoint_state")
    d, k, g = kw.get("d_cache"), kw.get("k_cache"), kw.get("g_cache")
    dr, kr = kw.get("d_residual_cache"), kw.get("k_residual_cache")
    for name in ("checkpoint_state", "d_cache", "k_cache", "g_cache"):
        if not isinstance(kw.get(name), torch.Tensor) or not kw[name].is_cuda:
            return f"{name} missing or not on CUDA"
    if st.dtype is not torch.float32 or st.ndim != 5 or tuple(st.shape[2:]) != (HV, V, K):
        return f"checkpoint_state {st.dtype} {tuple(st.shape)}"
    if st.stride(4) != 1 or st.stride(3) != K or st.stride(2) != V * K or st.stride(1) % 4:
        return f"checkpoint_state strides {st.stride()}"
    L = st.shape[0]
    if d.dtype is not torch.bfloat16 or d.ndim != 5 or tuple(d.shape[0:1] + d.shape[2:]) != (L, HV, T, V):
        return f"d_cache {d.dtype} {tuple(d.shape)}"
    if k.dtype is not torch.bfloat16 or k.ndim != 5 or tuple(k.shape[0:1] + k.shape[2:]) != (L, H, T, K):
        return f"k_cache {k.dtype} {tuple(k.shape)}"
    if g.dtype is not torch.float32 or g.ndim != 4 or tuple(g.shape[0:1] + g.shape[2:]) != (L, HV, T):
        return f"g_cache {g.dtype} {tuple(g.shape)}"
    for name, t, inner in (("d_cache", d, (T * V, V, 1)), ("k_cache", k, (T * K, K, 1)), ("g_cache", g, (T, 1))):
        if t.stride()[2:] != inner or t.stride(0) % 8 or t.stride(1) % 8:
            return f"{name} strides {t.stride()}"
    if (dr is None) != (kr is None):
        return "exactly one of d_residual_cache / k_residual_cache"
    if dr is not None:
        if dr.dtype is not torch.bfloat16 or dr.shape != d.shape or dr.stride()[2:] != d.stride()[2:]:
            return f"d_residual_cache {dr.dtype} {tuple(dr.shape)} {dr.stride()}"
        if kr.dtype is not torch.bfloat16 or kr.shape != k.shape or kr.stride()[2:] != k.stride()[2:]:
            return f"k_residual_cache {kr.dtype} {tuple(kr.shape)} {kr.stride()}"
        if dr.stride(0) % 8 or dr.stride(1) % 8 or kr.stride(0) % 8 or kr.stride(1) % 8:
            return "residual strides"
    for name in ("state_batch_indices", "replay_indices", "accept_lens"):
        if not _idx_ok(kw.get(name)):
            return f"{name} must be a 1-D int32/int64 tensor"
    rows = kw["state_batch_indices"].shape[0]
    if rows > MAX_ROWS:
        return f"batch {rows} > {MAX_ROWS}"
    if kw["replay_indices"].shape[0] < rows or kw["accept_lens"].shape[0] < rows:
        return "index length mismatch"
    ti, ts = kw.get("mamba_track_indices"), kw.get("mamba_steps_to_track")
    if (ti is None) != (ts is None):
        return "exactly one of mamba_track_indices / mamba_steps_to_track"
    if ti is not None:
        if not _idx_ok(ti) or not _idx_ok(ts):
            return "track tensors must be 1-D int32/int64"
        if ti.shape[0] < rows or ts.shape[0] < rows:
            return "track length mismatch"
    wp, cb, fl = kw.get("write_pos"), kw.get("cache_base"), kw.get("is_flush")
    if not all(isinstance(t, torch.Tensor) and t.ndim == 1 and t.is_contiguous() for t in (wp, cb, fl)):
        return "cursors must be 1-D contiguous tensors"
    if wp.dtype is not torch.int32 or cb.dtype is not torch.int32:
        return f"write_pos/cache_base {wp.dtype}/{cb.dtype}"
    if fl.dtype not in (torch.int8, torch.bool, torch.uint8):
        return f"is_flush {fl.dtype}"
    if kw.get("null_block_id", -1) != -1:
        return f"null_block_id {kw.get('null_block_id')}"
    return None


def run(kw: dict) -> None:
    st = kw["checkpoint_state"]
    d, k, g = kw["d_cache"], kw["k_cache"], kw["g_cache"]
    dr, kr = kw.get("d_residual_cache"), kw.get("k_residual_cache")
    has_res = dr is not None
    sidx, ridx, acc = kw["state_batch_indices"], kw["replay_indices"], kw["accept_lens"]
    ti, ts = kw.get("mamba_track_indices"), kw.get("mamba_steps_to_track")
    has_track = ti is not None
    if not has_track:
        ti, ts = sidx, acc  # placeholders, never read
    jobs, counters = _workspace(st.device)
    fl = kw["is_flush"]
    module().gdn_flush(
        st, d, dr if has_res else d, k, kr if has_res else k, g,
        sidx, ridx, acc, ti, ts,
        kw["write_pos"], kw["cache_base"], fl.view(torch.int8) if fl.dtype is not torch.int8 else fl,
        jobs, counters,
        sidx.stride(0), ridx.stride(0), acc.stride(0), ti.stride(0), ts.stride(0),
        int(has_res), int(has_track), st.shape[0], sidx.shape[0], -1, _grid(st.device),
    )  # fmt: skip


def _commit_around(original, *args, **kwargs):
    from green_attention import dispatch
    from green_attention.gdn import _green_on_sm90

    # Eager (the scheduler commits outside the CUDA graph), but also graph-safe: pre-allocated
    # workspace, device-side job count, no host sync.
    if kwargs and not args and _green_on_sm90() and dispatch.enabled("gdn_flush"):
        reason = can_use(kwargs)
        if reason is None:
            run(kwargs)
            return None
        dispatch.log_fallback("gdn_flush", (reason,), kwargs)
    return original(*args, **kwargs)


def _select(kwargs):
    return "the Replay-SSM commit is served by the commit_gdn_replayssm_circular hook"


def _prewarm(backend=None):
    """Compile both kernel instances and allocate the workspace before graph capture."""
    module()
    if torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
        _workspace(dev)
        _grid(dev)


def _register_all():
    from green_attention import dispatch
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    dispatch.register("gdn_flush", _select, _prewarm)
    HookRegistry.register(TARGET, _commit_around, HookType.AROUND)


_register_all()

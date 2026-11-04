"""FA4 fallback for an FP8 KV cache whose head_dim exceeds FA4's SM90 FP8 limit.

FA4 on SM90 runs FP8 KV natively only up to head_dim 256 (``interface.py`` asserts it), but its
bf16 kernel runs head_dim 512 (Gemma 4 global layers). The green ``decode`` / ``verify`` /
``extend`` kernels take these calls; any call they decline (``GREEN_ATTN_DISABLE``, q_descale /
scale factors, softcap, ...) goes through ``attend``: it gathers the pages the call references,
dequantizes them to bf16 (applying ``k_descale`` / ``v_descale``) into a compact paged buffer and
runs FA4 in bf16 on it. ``GreenAttentionBackend`` keeps the query of these layers in bf16; an e4m3
query is widened exactly.

Cost: one read of the referenced fp8 pages and one bf16 write, memory proportional to
rows x width of the page table. Eager calls above ``GROUP_BYTES`` run in request groups (one host
sync for the query offsets). Under CUDA graph capture the page table spans the static max context,
so the fallback refuses: decode on these layers has to stay on the green kernel.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# The FA4 names this module relies on (test/unit/test_contract.py pins them). The interface
# string goes stale once FA4 runs SM90 FP8 past head_dim 256; drop this fallback then.
SGLANG_READS = {
    "kernels/ops/attention/flash_attention_v4.py": (
        "flash_attn_with_kvcache(q)",
        "flash_attn_with_kvcache(k_cache)",
        "flash_attn_with_kvcache(v_cache)",
        "flash_attn_with_kvcache(page_table)",
        "flash_attn_with_kvcache(cache_seqlens)",
        "flash_attn_with_kvcache(cu_seqlens_q)",
        "flash_attn_with_kvcache(cu_seqlens_k_new)",
        "flash_attn_with_kvcache(k_descale)",
        "flash_attn_with_kvcache(v_descale)",
        "flash_attn_with_kvcache(out)",
    ),
    "kernels/ops/attention/flash_attn/cute/interface.py": ('"SM90 FA4 FP8 supports head_dim <= 256"',),
}

FA4_SM90_FP8_MAX_HEAD_DIM = 256
GROUP_BYTES = 1 << 30  # transient bytes (fp8 gather + bf16 K and V) per eager call before splitting

# Arguments whose meaning survives slicing the batch into request groups; anything else set
# (LSE return, mask/score mods, aux tensors, scale factors, ...) runs as one group.
_GROUPABLE = frozenset(
    {
        "q", "k_cache", "v_cache", "page_table", "cache_seqlens", "cu_seqlens_q", "cu_seqlens_k_new",
        "max_seqlen_q", "max_seqlen_k", "softmax_scale", "causal", "window_size", "softcap",
        "num_splits", "pack_gqa", "sm_margin", "sinks", "out",
    }
)  # fmt: skip
_UNSUPPORTED = ("q_descale", "sfq", "sfk", "sfv")

_LOGGED: set = set()
_LOG_LOCK = threading.Lock()


@functools.lru_cache(maxsize=None)
def _is_sm90(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] == 9


def applies(k_cache) -> bool:
    """True when FA4 cannot attend over ``k_cache`` natively on this device."""
    return (
        isinstance(k_cache, torch.Tensor)
        and k_cache.dtype == torch.float8_e4m3fn
        and k_cache.shape[-1] > FA4_SM90_FP8_MAX_HEAD_DIM
        and k_cache.is_cuda
        and _is_sm90(k_cache.device)
    )


def _log_once(head_dim: int, op: str) -> None:
    with _LOG_LOCK:
        if (head_dim, op) in _LOGGED:
            return
        _LOGGED.add((head_dim, op))
    logger.info("green_attention: %s with fp8 KV head_dim=%d -> FA4 bf16 on dequantized pages", op, head_dim)


def _dequant(cache: torch.Tensor, pages: torch.Tensor, descale, rows: int) -> torch.Tensor:
    """``cache[pages]`` in bf16, scaled per (request, kv head) by ``descale`` in fp32."""
    gathered = cache.index_select(0, pages)
    if descale is None:
        return gathered.to(torch.bfloat16)
    heads, dim = gathered.shape[2], gathered.shape[3]
    if descale.dim() != 2 or descale.shape[0] not in (1, rows) or descale.shape[1] != heads:
        raise ValueError(f"green_attention: descale shape {tuple(descale.shape)} vs ({rows}, {heads})")
    scaled = gathered.float().view(rows, -1, heads, dim).mul_(descale.float().reshape(-1, 1, heads, 1))
    return scaled.view(gathered.shape).to(torch.bfloat16)


def _run_group(fallback: Callable, kw: dict, k_cache, v_cache, k_descale, v_descale):
    pt = kw["page_table"]
    rows, width = pt.shape
    pages = pt.reshape(-1).to(torch.int64)
    kw["k_cache"] = _dequant(k_cache, pages, k_descale, rows)
    kw["v_cache"] = _dequant(v_cache, pages, v_descale, rows)  # never aliases K (FA4 reads k is v as MLA)
    kw["page_table"] = torch.arange(rows * width, dtype=torch.int32, device=pt.device).view(rows, width)
    return fallback(**kw)


def _rows_per_group(page_table, k_cache, v_cache) -> int:
    page, heads = k_cache.shape[1], k_cache.shape[2]
    dk, dv = k_cache.shape[3], v_cache.shape[3]
    # bf16 K held while V is gathered (fp8), widened (fp32) and narrowed (bf16)
    per_row = page_table.shape[1] * page * heads * (2 * dk + 7 * dv)
    return max(1, GROUP_BYTES // max(per_row, 1))


def _rows(t, g0: int, g1: int, batch: int):
    return t[g0:g1] if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == batch else t


def attend(fallback: Callable, kwargs: dict, op: str):
    """Run ``fallback`` (FA4 ``flash_attn_with_kvcache``) on bf16 copies of the referenced fp8 pages."""
    k_cache, v_cache = kwargs["k_cache"], kwargs["v_cache"]
    head_dim = k_cache.shape[-1]
    if k_cache.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"green_attention: {op} with fp8 KV head_dim={head_dim} must run on the green kernel under CUDA "
            f"graph capture: FA4 SM90 FP8 stops at head_dim {FA4_SM90_FP8_MAX_HEAD_DIM} and the dequantized "
            "fallback is not graph-safe. The green decline reason is logged above; re-enable the op in "
            "GREEN_ATTN_DISABLE or use --kv-cache-dtype bfloat16."
        )
    unsupported = [n for n in _UNSUPPORTED if kwargs.get(n) is not None]
    if unsupported or v_cache.dtype != k_cache.dtype:
        raise NotImplementedError(
            f"green_attention: fp8 KV head_dim={head_dim} fallback does not handle {unsupported or 'mixed KV dtypes'}"
        )
    _log_once(head_dim, op)

    kw = dict(kwargs)
    q = kw["q"]
    if q.dtype != torch.bfloat16:
        kw["q"] = q = q.to(torch.bfloat16)
    k_descale, v_descale = kw.pop("k_descale", None), kw.pop("v_descale", None)
    pt = kw["page_table"]
    batch = pt.shape[0]
    rows = _rows_per_group(pt, k_cache, v_cache)
    cu = kw.get("cu_seqlens_q")
    groupable = all(v is None or v is False or n in _GROUPABLE for n, v in kw.items())
    if rows >= batch or cu is None or not groupable:
        return _run_group(fallback, kw, k_cache, v_cache, k_descale, v_descale)

    out = kw.get("out")
    if out is None:
        out = torch.empty((*q.shape[:-1], v_cache.shape[-1]), dtype=torch.bfloat16, device=q.device)
    offsets = cu.tolist()
    for g0 in range(0, batch, rows):
        g1 = min(batch, g0 + rows)
        s, e = offsets[g0], offsets[g1]
        sub = dict(
            kw,
            q=q[s:e],
            out=out[s:e],
            page_table=pt[g0:g1],
            cache_seqlens=_rows(kw.get("cache_seqlens"), g0, g1, batch),
            cu_seqlens_q=cu[g0 : g1 + 1] - s,
            cu_seqlens_k_new=None,  # FA4 ignores it; sliced batches would make it stale
        )
        _run_group(
            fallback, sub, k_cache, v_cache, _rows(k_descale, g0, g1, batch), _rows(v_descale, g0, g1, batch)
        )
    return out

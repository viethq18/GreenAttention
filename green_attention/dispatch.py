"""Route each attention op to a green kernel or to the stock fallback.

Ops: ``decode`` / ``verify`` (full-attention ``flash_attn_with_kvcache`` with
``max_seqlen_q == 1`` / ``2..MAX_VERIFY_Q``), ``extend`` (any other
``flash_attn_with_kvcache`` call over an FP8 cache FA4 cannot run natively, see
``wide_kv.applies``) and ``gdn_decode`` / ``gdn_verify`` (the GDN recurrent
kernels). A kernel module registers a *selector* per op: it
receives the call's keyword arguments and returns the callable to run, or a
string saying why it declines. Selection depends only on shapes, dtypes and
Python scalars, never on tensor data, so it is stable across CUDA graph capture
and replay.

``GREEN_ATTN_DISABLE=decode,verify,extend,gdn_decode,gdn_verify`` (any subset)
forces the fallback for those ops: the A/B switch. Every declined call logs once per
signature (op, reasons, argument dtypes/shapes without the batch dimension).
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch

logger = logging.getLogger(__name__)

OPS = ("decode", "verify", "extend", "gdn_decode", "gdn_verify")
MAX_VERIFY_Q = 16

Selector = Callable[[dict], Union[Callable, str]]


def _parse_disabled(raw: str) -> frozenset:
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = names - set(OPS)
    if unknown:
        raise ValueError(f"GREEN_ATTN_DISABLE: unknown ops {sorted(unknown)}; valid: {OPS}")
    return names


DISABLED = _parse_disabled(os.environ.get("GREEN_ATTN_DISABLE", ""))

_SELECTORS: dict = defaultdict(list)
_PREWARMS: list = []
_LOGGED: set = set()
_LOG_LOCK = threading.Lock()


def register(op: str, selector: Selector, prewarm: Optional[Callable[[Any], None]] = None) -> None:
    """Add a kernel for ``op``; ``prewarm(backend)`` compiles it before graph capture."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; valid: {OPS}")
    _SELECTORS[op].append(selector)
    if prewarm is not None:
        _PREWARMS.append(prewarm)


def prewarm(backend: Any) -> None:
    for fn in _PREWARMS:
        fn(backend)


def enabled(op: str) -> bool:
    return op not in DISABLED and bool(_SELECTORS[op])


def _signature(kwargs: dict) -> tuple:
    sig = []
    for name in sorted(kwargs):
        value = kwargs[name]
        if isinstance(value, torch.Tensor):
            sig.append((name, str(value.dtype), tuple(value.shape[1:])))
        elif value is None or isinstance(value, (bool, int, float, str)):
            sig.append((name, value))
        elif isinstance(value, tuple) and all(isinstance(v, (int, float)) for v in value):
            sig.append((name, value))
        else:
            sig.append((name, type(value).__name__))
    return tuple(sig)


def log_fallback(op: str, reasons: tuple, kwargs: dict) -> None:
    key = (op, reasons, _signature(kwargs))
    with _LOG_LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    logger.info("green_attention: %s -> fallback (%s); signature=%s", op, "; ".join(reasons), key[2])


def select(op: str, kwargs: dict) -> Optional[Callable]:
    """The green kernel for this call, or None for the fallback (logged once)."""
    if op in DISABLED or not _SELECTORS[op]:
        return None
    reasons = []
    for selector in _SELECTORS[op]:
        chosen = selector(kwargs)
        if callable(chosen):
            return chosen
        reasons.append(chosen)
    log_fallback(op, tuple(reasons), kwargs)
    return None


def kvcache_op(max_seqlen_q: Any) -> Optional[str]:
    if not isinstance(max_seqlen_q, int):
        return None
    if max_seqlen_q == 1:
        return "decode"
    if 2 <= max_seqlen_q <= MAX_VERIFY_Q:
        return "verify"
    return None


def with_kvcache(fallback: Callable) -> Callable:
    """Wrap FA4's ``flash_attn_with_kvcache`` so decode/verify (and extend over a wide FP8 cache) can go green."""
    from green_attention import wide_kv

    @functools.wraps(fallback)
    def flash_attn_with_kvcache(*args, **kwargs):
        op = kvcache_op(kwargs.get("max_seqlen_q"))
        wide = not args and wide_kv.applies(kwargs.get("k_cache"))
        if op is None and wide:
            op = "extend"
        if op is not None:
            if args:
                log_fallback(op, ("positional arguments",), kwargs)
            else:
                run = select(op, kwargs)
                if run is not None:
                    return run(**kwargs)
        if wide:
            return wide_kv.attend(fallback, kwargs, op)
        q, kc = kwargs.get("q"), kwargs.get("k_cache")
        if q is not None and kc is not None and kc.dtype == torch.float8_e4m3fn and q.dtype != kc.dtype:
            kwargs["q"] = q.to(kc.dtype)  # GreenAttentionBackend left the decode query in bf16
        return fallback(*args, **kwargs)

    return flash_attn_with_kvcache

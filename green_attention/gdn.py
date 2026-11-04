"""GDN recurrent decode/verify through ``dispatch``.

An AROUND hook on ``GDNKernelDispatcher.__init__`` replaces the dispatcher's
``decode_kernel`` and ``verify_kernel`` with ``GreenGDNKernel`` wrappers around
the kernels SGLang chose. A wrapper runs a green kernel when one accepts the
call and otherwise delegates to the wrapped kernel, flags included. The extend
kernel and the tree-verify kernel are left alone (``GDNAttnBackend`` checks
``isinstance(extend_kernel, TritonGDNKernel)``). Only servers whose attention
backend is ``green`` on SM90 are wrapped, so an fa4 run with this plugin
installed is the untouched baseline.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.attention.linear.kernels.kernel_backend import LinearAttnKernelBase

from green_attention import dispatch
from green_attention import gdn_recurrent  # noqa: F401  (registers gdn kernels + replayssm hook)
from green_attention import gdn_flush  # noqa: F401  (registers the replayssm commit hook)

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/attention/linear/kernels/kernel_backend.py": (
        "LinearAttnKernelBase",
        "uses_state_checkpoints",
        "supports_fused_chain_verify",
        "supports_strided_target_verify_qkv",
        "supports_track_state_snapshot",
        "decode",
        "extend",
        "target_verify",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "GDNKernelDispatcher",
        "__init__(decode_backend)",
        "decode_kernel",
        "verify_kernel",
        "packed_decode",
        "supports_packed_decode",
    ),
    "srt/runtime_context.py": ("attention_backends",),
}

TARGET = "sglang.srt.layers.attention.linear.gdn_backend.GDNKernelDispatcher.__init__"

_FLAGS = (
    "uses_state_checkpoints",
    "supports_fused_chain_verify",
    "supports_strided_target_verify_qkv",
    "supports_track_state_snapshot",
)


class GreenGDNKernel(LinearAttnKernelBase):
    def __init__(self, fallback):
        self.fallback = fallback
        for flag in _FLAGS:
            setattr(self, flag, getattr(fallback, flag))
        self.supports_packed_decode = getattr(fallback, "supports_packed_decode", False)

    def decode(self, q, k, v, a, b, **kwargs):
        if dispatch.enabled("gdn_decode"):
            call = dict(q=q, k=k, v=v, a=a, b=b, **kwargs)
            run = dispatch.select("gdn_decode", call)
            if run is not None:
                return run(**call)
        return self.fallback.decode(q, k, v, a, b, **kwargs)

    def packed_decode(self, mixed_qkv, a, b, **kwargs):
        if dispatch.enabled("gdn_decode"):
            call = dict(mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)
            run = dispatch.select("gdn_decode", call)
            if run is not None:
                return run(**call)
        return self.fallback.packed_decode(mixed_qkv, a, b, **kwargs)

    def target_verify(self, A_log, dt_bias, q, k, v, a, b, **kwargs):
        if dispatch.enabled("gdn_verify"):
            call = dict(A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b, **kwargs)
            run = dispatch.select("gdn_verify", call)
            if run is not None:
                return run(**call)
        return self.fallback.target_verify(A_log, dt_bias, q, k, v, a, b, **kwargs)

    def extend(self, q, k, v, g, beta, **kwargs):
        return self.fallback.extend(q, k, v, g, beta, **kwargs)


def _green_on_sm90() -> bool:
    from sglang.srt.runtime_context import attention_backends

    return "green" in attention_backends() and torch.cuda.get_device_capability() == (9, 0)


def gdn_init_around(original, self, *args, **kwargs):
    original(self, *args, **kwargs)
    if not _green_on_sm90():
        return
    self.decode_kernel = GreenGDNKernel(self.decode_kernel)
    self.verify_kernel = GreenGDNKernel(self.verify_kernel)
    logger.info(
        "green_attention: GDN decode=%s verify=%s wrapped (green ops enabled: decode=%s verify=%s)",
        type(self.decode_kernel.fallback).__name__,
        type(self.verify_kernel.fallback).__name__,
        dispatch.enabled("gdn_decode"),
        dispatch.enabled("gdn_verify"),
    )

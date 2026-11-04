"""The ``green`` full-attention backend: SGLang's FA4 backend with its
``flash_attn_with_kvcache`` routed through ``dispatch`` (decode/verify go to the
green kernels when they accept the call, everything else stays on FA4)."""

from __future__ import annotations

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

from green_attention import dispatch

SGLANG_READS = {
    "srt/layers/attention/flashattention_backend.py": (
        "FlashAttentionBackend",
        "__init__(fa_impl_ver)",
        "fa_impl_ver",
        "_init_flash_attention_functions",
        "_init_flash_attention_functions(device_capability)",
        "flash_attn_with_kvcache",
    ),
}


class GreenAttentionBackend(FlashAttentionBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, fa_impl_ver=4, **kwargs)
        # Compile every green kernel now, before CUDA graph capture.
        dispatch.prewarm(self)

    def _init_flash_attention_functions(self, device_capability):
        super()._init_flash_attention_functions(device_capability)
        self.flash_attn_with_kvcache = dispatch.with_kvcache(self.flash_attn_with_kvcache)

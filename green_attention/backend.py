"""The ``green`` full-attention backend: SGLang's FA4 backend with its
``flash_attn_with_kvcache`` routed through ``dispatch`` (decode/verify go to the
green kernels when they accept the call, everything else stays on FA4).

Decode skips the backend's ``q.to(kv_cache_dtype)``: the green kernel quantizes
the bf16 query to e4m3 itself, and ``dispatch.with_kvcache`` casts it for FA4
on every call the kernel declines. Layers wider than FA4's SM90 FP8 limit
(Gemma 4 global, head_dim 512) keep a bf16 query on every call: FA4 runs them
in bf16 on dequantized pages (``wide_kv``)."""

from __future__ import annotations

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

from green_attention import dispatch, wide_kv
from green_attention import attn_decode  # noqa: F401  (registers the decode/verify kernels)

SGLANG_READS = {
    "srt/layers/attention/flashattention_backend.py": (
        "FlashAttentionBackend",
        "__init__(fa_impl_ver)",
        "fa_impl_ver",
        "_init_flash_attention_functions",
        "_init_flash_attention_functions(device_capability)",
        "flash_attn_with_kvcache",
        "prepare_paged_mha_query",
        "prepare_paged_mha_query(q)",
        "prepare_paged_mha_query(q_rope)",
        "prepare_paged_mha_query(k_rope)",
        "prepare_paged_mha_query(layer)",
        "prepare_paged_mha_query(logical_batch_size)",
        "prepare_paged_mha_query(kv_head_num)",
        "prepare_paged_mha_query(is_prefill)",
    ),
    "srt/layers/radix_attention.py": ("head_dim",),
}


class GreenAttentionBackend(FlashAttentionBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, fa_impl_ver=4, **kwargs)
        # Compile every green kernel now, before CUDA graph capture.
        dispatch.prewarm(self)

    def _init_flash_attention_functions(self, device_capability):
        super()._init_flash_attention_functions(device_capability)
        self.flash_attn_with_kvcache = dispatch.with_kvcache(self.flash_attn_with_kvcache)

    def prepare_paged_mha_query(self, q, q_rope, k_rope, layer, *, logical_batch_size, kv_head_num, is_prefill):
        kw = dict(logical_batch_size=logical_batch_size, kv_head_num=kv_head_num, is_prefill=is_prefill)
        wide = layer.head_dim > wide_kv.FA4_SM90_FP8_MAX_HEAD_DIM
        if q_rope is not None or not (wide or (not is_prefill and dispatch.enabled("decode"))):
            return super().prepare_paged_mha_query(q, q_rope, k_rope, layer, **kw)
        # The stock logic on an empty slice yields the descales without casting q.
        _, _, k_rope, k_descale, v_descale = super().prepare_paged_mha_query(q[:0], None, k_rope, layer, **kw)
        return q, None, k_rope, k_descale, v_descale

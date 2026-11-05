"""SGLang plugin entry point: registers the ``green`` attention backend and the
GDN kernel hook. Launch with ``--attention-backend green
--speculative-draft-attention-backend fa4`` (the DFLASH draft stays on FA4: its
fp8 -> bf16 draft KV override and draft backend map key on ``"fa4"``)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BACKEND = "green"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "register", "AROUND"),
    "srt/layers/attention/attention_registry.py": ("register_attention_backend",),
    "srt/arg_groups/choices.py": (
        "add_attention_backend_choices",
        "add_chunked_prefix_cache_attention_backend",
        "add_deterministic_attention_backend_choices",
        "add_radix_supported_deterministic_attention_backend_choices",
    ),
    "srt/runtime_context.py": ("get_spec",),
    "srt/arg_groups/fields/spec.py": (
        "speculative_algorithm",
        "speculative_draft_attention_backend",
    ),
    "srt/arg_groups/resolution_hooks.py": (
        "register_resolution_hook",
        '"handle_model_specific_adjustments"',
    ),
    "srt/arg_groups/model_hook.py": (
        "handle_model_specific_adjustments",
        "attention_backends_of",
        '"Gemma4ForConditionalGeneration"',
    ),
    "srt/arg_groups/model_override_base.py": ("model_config_of",),
}


def _create_green_backend(runner):
    from green_attention.backend import GreenAttentionBackend

    spec = _draft_backend()
    if spec is not None and spec != "fa4":
        logger.warning(
            "green_attention: target backend is green but the speculative draft runs on "
            "%r; pass --speculative-draft-attention-backend fa4 to match production",
            spec,
        )
    return GreenAttentionBackend(runner)


def _draft_backend():
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return None
    return spec.speculative_draft_attention_backend


# Gemma4's backend check in model_hook.handle_model_specific_adjustments is a literal
# whitelist with no plugin extension point. green is the FA4 backend class
# (GreenAttentionBackend(FlashAttentionBackend, fa_impl_ver=4)) with its own decode kernels, so
# that check alone sees green as fa4. Other archs and every later step see the real name.
GEMMA4_ARCHS = frozenset(
    (
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
    )
)


def _gemma4_accepts_green(server_args, previous) -> None:
    import sglang.srt.arg_groups.model_hook as model_hook
    from sglang.srt.arg_groups.model_override_base import model_config_of

    original = model_hook.attention_backends_of

    def as_fa4(cfg):
        backends = original(cfg)
        # Only the GptOss and Gemma4 branches call this, after the step built the model config.
        if model_config_of(server_args).hf_config.architectures[0] not in GEMMA4_ARCHS:
            return backends
        return tuple("fa4" if b == BACKEND else b for b in backends)

    model_hook.attention_backends_of = as_fa4
    try:
        previous(server_args)
    finally:
        model_hook.attention_backends_of = original


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    from sglang.srt.arg_groups.choices import (
        add_attention_backend_choices,
        add_chunked_prefix_cache_attention_backend,
        add_deterministic_attention_backend_choices,
        add_radix_supported_deterministic_attention_backend_choices,
    )
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from green_attention import gdn

    add_attention_backend_choices([BACKEND])
    add_chunked_prefix_cache_attention_backend(BACKEND)
    add_deterministic_attention_backend_choices([BACKEND])
    add_radix_supported_deterministic_attention_backend_choices([BACKEND])
    register_attention_backend(BACKEND)(_create_green_backend)
    HookRegistry.register(gdn.TARGET, gdn.gdn_init_around, HookType.AROUND)
    register_resolution_hook("handle_model_specific_adjustments")(_gemma4_accepts_green)

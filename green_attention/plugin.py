"""SGLang plugin entry point: registers the ``green`` attention backend and the
GDN kernel hook. Launch with ``--attention-backend green``. The speculative draft
backend depends on the algorithm (``EXPECTED_DRAFT_BACKENDS``): DFLASH keeps its
draft on FA4 (``--speculative-draft-attention-backend fa4``: its fp8 -> bf16
draft KV override and draft backend map key on ``"fa4"``); the Gemma 4 MTP draft
(FROZEN_KV_MTP) runs on green, so leave that flag unset."""

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
    "srt/model_executor/model_runner.py": ("is_draft_worker",),
    "srt/arg_groups/speculative_hook.py": ('"FROZEN_KV_MTP"',),
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


# Speculative algorithm -> the draft attention backends that go with a green target (None: the flag is
# unset and the draft runner inherits the target's backend). Any other algorithm expects fa4.
# FROZEN_KV_MTP (Gemma 4 assistant draft) reads the target's KV cache, including the e4m3 head_dim 512
# global layers that FA4 cannot attend on SM90. With topk 1 the draft runs on the target's backend,
# green's decode kernels taking every draft call (q_len 1); topk > 1 needs its own triton backend.
EXPECTED_DRAFT_BACKENDS = {
    "FROZEN_KV_MTP": (None, BACKEND, "triton"),
}
DEFAULT_DRAFT_BACKENDS = ("fa4",)


def _create_green_backend(runner):
    from green_attention.backend import GreenAttentionBackend

    algorithm, draft = _spec_backends()
    expected = EXPECTED_DRAFT_BACKENDS.get(algorithm, DEFAULT_DRAFT_BACKENDS)
    if runner.is_draft_worker:
        logger.info("green_attention: serving the %s draft", algorithm)
    elif draft is not None and draft not in expected:
        logger.warning(
            "green_attention: target backend is green but the %s draft runs on %r; expected %s",
            algorithm,
            draft,
            " or ".join("unset" if b is None else repr(b) for b in expected),
        )
    return GreenAttentionBackend(runner)


def _spec_backends():
    """(speculative algorithm, --speculative-draft-attention-backend), (None, None) without spec."""
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return None, None
    return spec.speculative_algorithm, spec.speculative_draft_attention_backend


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

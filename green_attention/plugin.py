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


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    from sglang.srt.arg_groups.choices import (
        add_attention_backend_choices,
        add_chunked_prefix_cache_attention_backend,
        add_deterministic_attention_backend_choices,
        add_radix_supported_deterministic_attention_backend_choices,
    )
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from green_attention import gdn

    add_attention_backend_choices([BACKEND])
    add_chunked_prefix_cache_attention_backend(BACKEND)
    add_deterministic_attention_backend_choices([BACKEND])
    add_radix_supported_deterministic_attention_backend_choices([BACKEND])
    register_attention_backend(BACKEND)(_create_green_backend)
    HookRegistry.register(gdn.TARGET, gdn.gdn_init_around, HookType.AROUND)

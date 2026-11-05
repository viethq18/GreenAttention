"""Draft-backend check of ``green_attention.plugin._create_green_backend`` (CPU only)."""

import logging
import sys
import types

import pytest

from green_attention import plugin


@pytest.fixture
def create(monkeypatch):
    backend = types.ModuleType("green_attention.backend")
    backend.GreenAttentionBackend = lambda runner: "green"
    monkeypatch.setitem(sys.modules, "green_attention.backend", backend)

    def run(algorithm, draft):
        monkeypatch.setattr(plugin, "_spec_backends", lambda: (algorithm, draft))
        return plugin._create_green_backend(types.SimpleNamespace(is_draft_worker=False))

    return run


@pytest.mark.parametrize(
    "algorithm, draft, warns",
    [
        (None, None, False),
        ("FROZEN_KV_MTP", None, False),
        ("FROZEN_KV_MTP", "green", False),
        ("FROZEN_KV_MTP", "triton", False),  # topk > 1 requires it
        ("FROZEN_KV_MTP", "fa4", True),  # FA4 SM90 cannot attend the e4m3 head_dim 512 global layers
        ("DFLASH", "fa4", False),
        ("DFLASH", None, False),
        ("DFLASH", "flashinfer", True),
        ("DFLASH", "green", True),
    ],
)
def test_target_warns_on_unexpected_draft_backend(create, caplog, algorithm, draft, warns):
    with caplog.at_level(logging.WARNING, logger=plugin.__name__):
        assert create(algorithm, draft) == "green"
    assert any("draft runs on" in r.message for r in caplog.records) == warns

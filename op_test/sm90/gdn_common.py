"""Shared input builders and baseline callers for the SM90 GDN recurrent kernels.

Shapes are the Qwen3.8-27B-FP8 production ones: 16 key heads, 48 value heads,
dk = dv = 128, replay-ring length 32, bf16 activations, ``scale = dk ** -0.5``.
"""

from __future__ import annotations

import torch

H, HV, K, V, L = 16, 48, 128, 128, 32
SCALE = K**-0.5
PAD = -1


def _gen(seed: int, device: str = "cuda") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _randn(shape, gen, dtype, scale=1.0):
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * scale).to(dtype)


def _distinct(n: int, hi: int, gen) -> torch.Tensor:
    return torch.randperm(hi, generator=gen, device="cuda")[:n].to(torch.int64)


def make_decode(
    B: int,
    state_dtype=torch.float32,
    slots: int = 128,
    seed: int = 0,
    pad_rows=(),
    dt_bias_dtype=torch.float32,
    strided_ba: bool = False,
):
    """Inputs for one packed-decode step (``q_len == 1``)."""
    gen = _gen(seed)
    mixed = _randn((B, 2 * H * K + HV * V), gen, torch.bfloat16)
    if strided_ba:
        ba = _randn((B, 2 * HV), gen, torch.bfloat16)
        a, b = ba[:, :HV], ba[:, HV:]
    else:
        a = _randn((B, HV), gen, torch.bfloat16)
        b = _randn((B, HV), gen, torch.bfloat16)
    A_log = torch.log(torch.rand(HV, generator=gen, device="cuda") * 16 + 1e-3)
    dt_bias = _randn((HV,), gen, torch.float32).to(dt_bias_dtype)
    state = _randn((slots, HV, V, K), gen, state_dtype, 0.1)
    idx = _distinct(B, slots, gen).to(torch.int32)
    for r in pad_rows:
        idx[r] = PAD
    return dict(
        mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        scale=SCALE, ssm_states=state, cache_indices=idx, num_v_heads=HV, head_v_dim=V,
    )


def split_decode(call: dict) -> dict:
    """The ``decode`` (q/k/v separate, ``query_start_loc``) form of a packed-decode call."""
    mixed = call["mixed_qkv"]
    B = mixed.shape[0]
    q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
    return dict(
        q=q.view(1, B, H, K), k=k.view(1, B, H, K), v=v.view(1, B, HV, V),
        a=call["a"], b=call["b"], A_log=call["A_log"], dt_bias=call["dt_bias"],
        ssm_states=call["ssm_states"], cache_indices=call["cache_indices"],
        query_start_loc=torch.arange(B + 1, device=mixed.device, dtype=torch.int32),
    )


def make_verify(
    spec_lens,
    slots: int = 96,
    ring_slots: int = 64,
    seed: int = 0,
    state_dtype=torch.float32,
    write_pos=None,
    cache_base=None,
    flush_rows=(),
    pad_rows=(),
    strided_qkv: bool = False,
    max_len: int = L,
):
    """Inputs for one replayssm verify step (``gdn_replayssm_spec_decode``, launch_mode='verify')."""
    gen = _gen(seed)
    B = len(spec_lens)
    qsl = torch.zeros(B + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(list(spec_lens), dtype=torch.int32).cumsum(0)
    T = int(qsl[-1])
    qsl = qsl.cuda()
    if strided_qkv:
        mixed = _randn((T, 2 * H * K + HV * V), gen, torch.bfloat16)
        q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
        q, k, v = q.view(T, H, K), k.view(T, H, K), v.view(T, HV, V)
    else:
        q = _randn((T, H, K), gen, torch.bfloat16)
        k = _randn((T, H, K), gen, torch.bfloat16)
        v = _randn((T, HV, V), gen, torch.bfloat16)
    a = _randn((T, HV), gen, torch.bfloat16)
    b = _randn((T, HV), gen, torch.bfloat16)
    A_log = torch.log(torch.rand(HV, generator=gen, device="cuda") * 16 + 1e-3)
    dt_bias = _randn((HV,), gen, torch.float32)
    ckpt = _randn((slots, HV, V, K), gen, state_dtype, 0.1)

    kh = _randn((ring_slots, H, max_len, K), gen, torch.float32)
    kh = kh / kh.norm(dim=-1, keepdim=True)
    k_cache = kh.to(torch.bfloat16)
    rawk = (kh - k_cache.float()).to(torch.bfloat16)
    d_cache = _randn((ring_slots, HV, max_len, V), gen, torch.bfloat16, 0.1)
    rawv = _randn((ring_slots, HV, max_len, V), gen, torch.bfloat16, 0.1 / 256)
    g_cache = -torch.rand((ring_slots, HV, max_len), generator=gen, device="cuda") * 0.05

    n_cur = ring_slots + 1
    if write_pos is None:
        wp = torch.randint(0, max_len - 3, (n_cur,), generator=gen, device="cuda", dtype=torch.int32)
    else:
        wp = torch.as_tensor(write_pos, dtype=torch.int32, device="cuda").expand(n_cur).clone()
    if cache_base is None:
        cb = torch.randint(0, max_len, (n_cur,), generator=gen, device="cuda", dtype=torch.int32)
    else:
        cb = torch.as_tensor(cache_base, dtype=torch.int32, device="cuda").expand(n_cur).clone()
    flush = torch.zeros(n_cur, dtype=torch.int8, device="cuda")
    ridx = _distinct(B, ring_slots, gen).to(torch.int32)
    sidx = _distinct(B, slots, gen).to(torch.int32)
    for r in flush_rows:
        flush[ridx[r].long()] = 1
    for r in pad_rows:
        sidx[r] = PAD
    out = q.new_empty(T, HV, V)
    return dict(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        checkpoint_state=ckpt, d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
        rawv_cache=rawv, rawk_cache=rawk, beta_cache=None, out=out,
        query_start_loc=qsl, ssm_state_indices=sidx, replay_indices=ridx,
        write_pos=wp, cache_base=cb, is_flush=flush,
        max_cache_len=max_len, max_spec_len=max(spec_lens),
        launch_mode="verify", null_block_id=PAD, scale=SCALE, use_qk_l2norm_in_kernel=True,
    )


def clone_call(call: dict) -> dict:
    """Deep-copy every tensor so a second implementation sees identical pre-state.

    Strided q/k/v views are cloned as contiguous (values identical).
    """
    out = {}
    for name, val in call.items():
        out[name] = val.clone() if isinstance(val, torch.Tensor) else val
    return out


# ----------------------------------------------------------------------------
# Baselines (stock SGLang kernels)
# ----------------------------------------------------------------------------


def triton_packed_decode(call: dict) -> torch.Tensor:
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    return TritonGDNKernel().packed_decode(**call)


def triton_decode(call: dict) -> torch.Tensor:
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    return TritonGDNKernel().decode(**call)


def triton_verify(call: dict) -> torch.Tensor:
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )

    return gdn_replayssm_spec_decode(**call)

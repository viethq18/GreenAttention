"""``green_attention.wide_kv``: the bf16 FA4 fallback for fp8 KV wider than FA4's SM90 FP8 limit (CPU)."""

import pytest
import torch

from green_attention import dispatch, wide_kv

FP8 = torch.float8_e4m3fn
PAGE, KVH, QH, HD = 4, 2, 8, 512


def _paged_attention(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, softmax_scale, out=None, **_):
    """Causal paged attention in fp32 (the stand-in for FA4)."""
    res = torch.zeros((*q.shape[:-1], v_cache.shape[-1]), dtype=torch.float32)
    cu = cu_seqlens_q.tolist()
    g = q.shape[1] // k_cache.shape[2]
    for b, L in enumerate(cache_seqlens.tolist()):
        n = torch.arange(L)
        pg = page_table[b, n // PAGE].long()
        k = k_cache[pg, n % PAGE].float().repeat_interleave(g, 1)  # [L, QH, HD]
        v = v_cache[pg, n % PAGE].float().repeat_interleave(g, 1)
        qq = q[cu[b] : cu[b + 1]].float()
        s = torch.einsum("thd,nhd->htn", qq, k) * softmax_scale
        pos = L - qq.shape[0] + torch.arange(qq.shape[0])
        s = s.masked_fill(n[None, None, :] > pos[None, :, None], float("-inf"))
        res[cu[b] : cu[b + 1]] = torch.einsum("htn,nhd->thd", s.softmax(-1), v)
    if out is None:
        return res.to(torch.bfloat16)
    out.copy_(res)
    return out


def _case(lens, qlens, descale=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    pages_per = [-(-L // PAGE) for L in lens]
    n_pages = sum(pages_per) + 3
    perm = torch.randperm(n_pages, generator=g).to(torch.int32)
    pt = torch.zeros((len(lens), max(pages_per) + 1), dtype=torch.int32)
    pos = 0
    for b, npg in enumerate(pages_per):
        pt[b, :npg] = perm[pos : pos + npg]
        pos += npg
    kc = torch.randn((n_pages, PAGE, KVH, HD), generator=g).to(FP8)
    vc = torch.randn((n_pages, PAGE, KVH, HD), generator=g).to(FP8)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(qlens).cumsum(0)
    kw = dict(
        q=torch.randn((sum(qlens), QH, HD), generator=g).to(torch.bfloat16),
        k_cache=kc,
        v_cache=vc,
        page_table=pt,
        cache_seqlens=torch.tensor(lens, dtype=torch.int32),
        cu_seqlens_q=cu,
        cu_seqlens_k_new=cu.clone(),
        max_seqlen_q=max(qlens),
        softmax_scale=HD**-0.5,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        return_softmax_lse=False,
        num_splits=0,
    )
    if descale:
        kw["k_descale"] = torch.rand((len(lens), KVH), generator=g) + 0.5
        kw["v_descale"] = torch.tensor([0.25]).expand(len(lens), KVH)  # expanded scalar, as SGLang passes it
    return kw


def _expected(kw):
    """The same attention on K/V dequantized up front, in the caller's page layout."""
    b = kw["page_table"].shape[0]
    ones = torch.ones((b, KVH))
    kd, vd = kw.get("k_descale", ones), kw.get("v_descale", ones)
    pid = torch.arange(kw["k_cache"].shape[0])
    k = torch.zeros(kw["k_cache"].shape, dtype=torch.float32)
    v = torch.zeros_like(k)
    for r in range(b):  # each request scales only the pages it owns
        own = kw["page_table"][r, : -(-int(kw["cache_seqlens"][r]) // PAGE)].long()
        mask = torch.isin(pid, own)
        k[mask] = kw["k_cache"][mask].float() * kd[r].view(1, KVH, 1)
        v[mask] = kw["v_cache"][mask].float() * vd[r].view(1, KVH, 1)
    plain = {n: t for n, t in kw.items() if n not in ("k_descale", "v_descale")}
    return _paged_attention(**{**plain, "k_cache": k.to(torch.bfloat16), "v_cache": v.to(torch.bfloat16)})


@pytest.mark.parametrize("descale", [True, False])
def test_matches_attention_on_dequantized_cache(descale):
    kw = _case([9, 21, 4], [9, 5, 1], descale=descale)
    got = wide_kv.attend(_paged_attention, kw, "extend")
    torch.testing.assert_close(got.float(), _expected(kw).float(), atol=1e-2, rtol=0)


def test_request_groups_match_one_group(monkeypatch):
    kw = _case([9, 21, 4, 13, 2], [9, 5, 1, 13, 2], seed=1)
    whole = wide_kv.attend(_paged_attention, kw, "extend")
    out = torch.full_like(whole, float("nan"))
    monkeypatch.setattr(wide_kv, "GROUP_BYTES", 1)  # one request per group
    got = wide_kv.attend(_paged_attention, {**kw, "out": out}, "extend")
    assert got is out
    torch.testing.assert_close(got, whole, atol=0, rtol=0)


def test_dispatch_routes_wide_fp8_kv_without_casting_query(monkeypatch):
    monkeypatch.setattr(wide_kv, "applies", lambda kc: kc is not None and kc.shape[-1] > 256)
    seen = {}

    def fa4(**kw):
        seen.update(kw)
        return _paged_attention(**kw)

    kw = _case([9, 21], [9, 5])
    dispatch.with_kvcache(fa4)(**kw)
    assert seen["q"].dtype == torch.bfloat16 and seen["k_cache"].dtype == torch.bfloat16
    assert seen["v_cache"].data_ptr() != seen["k_cache"].data_ptr()  # FA4 reads k is v as MLA
    assert "k_descale" not in seen and "v_descale" not in seen


@pytest.mark.parametrize("arg", ["q_descale", "sfk"])
def test_rejects_scale_forms_it_cannot_apply(arg):
    kw = {**_case([5], [1]), arg: torch.ones(1, KVH)}
    with pytest.raises(NotImplementedError, match=arg):
        wide_kv.attend(_paged_attention, kw, "decode")

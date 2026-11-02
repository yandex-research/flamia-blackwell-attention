"""Packed rows of compact DocMaps for the routes of group 2 (cov2): no full-map fallback, outputs bitwise the route's
fused / aligned path.

  * the 2-CTA forward (forward_two_cta='on'; lp_docmaps.PACKED_FWD_2CTA): each CTA of the pair builds its SF atoms from
    the dense per-row tables, O by TMA tiles + a document's partial last tile by row copies -- bitwise the fused pair on
    full maps (fp8 / fp4, causal / dense, MHA / GQA, the 1-CTA packed backward after it), and the 1-CTA packed forward
    at capacity N = T;
  * FA3-style E4M3 codes in the caller's packed rows (flash_attn_fp8(q8, k8, v8, q_descale=..., doc_maps=...),
    and training with grad_proxies on compact maps): bitwise the repacked cu_seqlens form, d 64 / 128; refusals
    (window / mods, grad_proxies on full maps);
  * the cross-document leak probe of both.

run: pytest tests/cute/test_lowprec_docmaps_cov2.py -q
"""
import pytest
import torch

from flash_attn.cute import lp_docmaps as DM
from flash_attn.cute.lowprec import DocMaps, FP4AttnOptions as O4, FP8AttnOptions as O8, flash_attn_fp4, flash_attn_fp8

BF16, E4 = torch.bfloat16, torch.float8_e4m3fn
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 0, 45, 700, 1, 128, 257]


def _cu(lens):
    return [0] + torch.tensor(lens).cumsum(0).tolist()


def _bits(a, b):
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def _pair(fn, O, maps, ts, causal, two=True):
    q, k, v, do = ts
    xs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    o, lse = fn(*xs, causal=causal, deterministic=True, return_lse=True, doc_maps=maps,
                options=O(forward_two_cta="on" if two else "off", two_cta="off"))
    return [o, lse] + list(torch.autograd.grad(o, xs, do))


def _bf(lens, h, hk, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return [q, k, v, do]


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
def test_pair_bitwise_fused(fn, O, causal, hk):
    ts = _bf(LENS, 4, hk)
    T = sum(LENS)
    mc = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=12, max_seqlen=1024).to("cuda")
    mf = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=12, max_seqlen=1024,
                                 padded_size=DM.default_padded_size(T, 12)).to("cuda")
    c0 = list(DM.PACKED_CALLS)
    ref = _pair(fn, O, mf, ts, causal)
    assert DM.PACKED_CALLS == c0, "full maps: the fused pair"
    got = _pair(fn, O, mc, ts, causal)
    assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "compact maps: the packed pair"
    assert ("aligned", "eager") not in DM._CACHE.get(mc, {}), "no full-map fallback"
    for i, (a, b) in enumerate(zip(ref, got)):
        assert _bits(a, b), ["O", "LSE", "dQ", "dK", "dV"][i]


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
def test_pair_capacity_t(fn, O):
    """Capacity N = T, one-token and mixed documents: the pair == the 1-CTA packed forward (O / LSE bitwise by design)."""
    for lens in ([1] * 512, [1, 300, 1, 1, 2, 129, 1, 77]):
        ts = _bf(lens, 4, 2)
        T = sum(lens)
        maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=T, max_seqlen=1024).to("cuda")
        a, b = _pair(fn, O, maps, ts, True), _pair(fn, O, maps, ts, True, two=False)
        assert all(_bits(x, y) for x, y in zip(a, b))


def _codes(T, h, hk, d, n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda hh: (torch.randn(T, hh, d, device="cuda", generator=g) * 40).to(E4)  # noqa: E731
    ds = [torch.rand(n, hk, device="cuda", generator=g) * 0.05 + 0.01 for _ in range(3)]
    return mk(h), mk(hk), mk(hk), ds


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("cap", ["small", "T"])
def test_prequant_doc_maps_bitwise(causal, hk, d, cap):
    T, n = sum(LENS), len(LENS)
    N = T if cap == "T" else n + 3
    q8, k8, v8, ds = _codes(T, 4, hk, d, N)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=N, max_seqlen=1024).to("cuda")
    cuc = torch.tensor(_cu(LENS), dtype=torch.int32, device="cuda")
    with torch.no_grad():
        r = flash_attn_fp8(q8, k8, v8, q_descale=ds[0][:n], k_descale=ds[1][:n], v_descale=ds[2][:n],
                           cu_seqlens_q=cuc, cu_seqlens_k=cuc, max_seqlen_q=1024, max_seqlen_k=1024, causal=causal,
                           return_lse=True)
        c0 = DM.PACKED_CALLS[0]
        m = flash_attn_fp8(q8, k8, v8, q_descale=ds[0], k_descale=ds[1], v_descale=ds[2], doc_maps=maps, causal=causal,
                           return_lse=True)
    assert DM.PACKED_CALLS[0] == c0 + 1
    assert _bits(r[0], m[0]) and _bits(r[1], m[1])


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
def test_prequant_doc_maps_backward_bitwise(causal, hk, d):
    """grad_proxies with doc_maps (compact maps): O / LSE / dQ / dK / dV bitwise the repacked cu_seqlens form, the
    packed forward and backward taken, no full maps built."""
    T, n = sum(LENS), len(LENS)
    q8, k8, v8, ds = _codes(T, 4, hk, d, n + 3)
    do = torch.randn(T, 4, d, device="cuda", dtype=BF16, generator=torch.Generator(device="cuda").manual_seed(5))
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=n + 3, max_seqlen=1024).to("cuda")
    cuc = torch.tensor(_cu(LENS), dtype=torch.int32, device="cuda")

    def run(**a):
        px = [t.to(BF16).requires_grad_(True) for t in (q8, k8, v8)]
        o, lse = flash_attn_fp8(q8, k8, v8, grad_proxies=tuple(px), deterministic=True, causal=causal,
                                return_lse=True, **a)
        return [o, lse] + list(torch.autograd.grad(o, px, do))
    r = run(q_descale=ds[0][:n], k_descale=ds[1][:n], v_descale=ds[2][:n], cu_seqlens_q=cuc, cu_seqlens_k=cuc,
            max_seqlen_q=1024, max_seqlen_k=1024)
    c0 = list(DM.PACKED_CALLS)
    m = run(q_descale=ds[0], k_descale=ds[1], v_descale=ds[2], doc_maps=maps)
    assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1]
    assert not any(isinstance(k_, tuple) and k_[0] == "aligned" for k_ in DM._CACHE.get(maps, {}))
    for i, (a, b) in enumerate(zip(r, m)):
        assert _bits(a, b), ["O", "LSE", "dQ", "dK", "dV"][i]


@gpu
def test_prequant_doc_maps_leak_and_refusals():
    T, n = sum(LENS), len(LENS)
    q8, k8, v8, ds = _codes(T, 4, 2, 128, n + 2)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=n + 2, max_seqlen=1024).to("cuda")
    cu = _cu(LENS)

    def run(a, b, c, e):
        with torch.no_grad():
            return flash_attn_fp8(a, b, c, q_descale=e[0], k_descale=e[1], v_descale=e[2], doc_maps=maps, causal=True,
                                  return_lse=True)
    o0, l0 = run(q8, k8, v8, ds)
    for j in (0, 4, 7):
        sl = slice(cu[j], cu[j + 1])
        q2, k2, v2 = (t.clone() for t in (q8, k8, v8))
        for t in (q2, k2, v2):
            t[sl] = torch.full_like(t[sl].float(), 448.0).to(E4)
        e2 = [x.clone() for x in ds]
        for x in e2:
            x[j] *= 1000.0
        o1, l1 = run(q2, k2, v2, e2)
        for i in range(n):
            if LENS[i] and i != j:
                assert torch.equal(o0[cu[i]:cu[i + 1]], o1[cu[i]:cu[i + 1]]), (j, i)
                assert torch.equal(l0[:, cu[i]:cu[i + 1]], l1[:, cu[i]:cu[i + 1]]), (j, i)
    with pytest.raises(NotImplementedError, match="window"):
        flash_attn_fp8(q8, k8, v8, q_descale=ds[0], k_descale=ds[1], v_descale=ds[2], doc_maps=maps,
                       window_size=(100, 0))
    px = tuple(t.to(BF16).requires_grad_(True) for t in (q8, k8, v8))
    full = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=n + 2, max_seqlen=1024,
                                   padded_size=DM.default_padded_size(T, n + 2)).to("cuda")
    with pytest.raises(NotImplementedError, match="compact maps"):
        flash_attn_fp8(q8, k8, v8, q_descale=ds[0], k_descale=ds[1], v_descale=ds[2], doc_maps=full, grad_proxies=px)
    with pytest.raises(ValueError, match="max_num_documents|\\(12, 2\\)|shape"):
        flash_attn_fp8(q8, k8, v8, q_descale=ds[0][:n], k_descale=ds[1][:n], v_descale=ds[2][:n], doc_maps=maps)

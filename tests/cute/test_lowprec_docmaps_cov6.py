"""The all-FP4 routes on documents (cov6).

  * k_channel 'all' / 'open' on documents (full maps, the aligned fused path): every document's gradients as accurate
    as k_channel 'fold' against an FP32 reference -- before cov6 the backward read r_k ('all') at a document-length
    offset and the forward's document-major bias / weight ('open') at the aligned key, so dQ ('all') and dQ / dK / dV
    ('open') of every document but the first were wrong;
  * the all-FP4 route on the packed rows of compact DocMaps (no full-map fallback): the causal backward with k_channel
    'all' / 'open' and post 'epi', the dense (non-causal) forward and backward -- O / LSE / dK / dV bitwise the fused path
    (full maps), dQ within its fp16 atomic accumulation spread; capacity N = T; the leak probe; no host sync / graph.

run: pytest tests/cute/test_lowprec_docmaps_cov6.py -q
"""
import math

import pytest
import torch

from flash_attn.cute import lp_docmaps as DM
from flash_attn.cute.lowprec import DocMaps, FP4AttnOptions as O4, flash_attn_fp4

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 0, 45, 700, 1, 128, 257]


def _bits(a, b):
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def _dq_close(a, b):
    """dQ within the fp16 atomic accumulation's order spread: |a - b| <= 2^-6 max |a| of its (token, head) row."""
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    af, bf = a.float(), b.float()
    return bool(((af - bf).abs() <= af.abs().amax(-1, keepdim=True) * 2.0 ** -6).all().item())


def _same4(ref, got, sl=None):
    for t, (a, b) in enumerate(zip(ref, got)):
        if sl is not None:
            a, b = a[sl(t)], b[sl(t)]
        if not (_dq_close(a, b) if t == 2 else _bits(a, b)):
            return False
    return True


def _cu(lens):
    return [0] + torch.tensor(lens).cumsum(0).tolist()


def _bf(lens, h, hk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, 128, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, 128, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return [q, k, v, do]


def _maps(lens, cap):
    mc = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap, max_seqlen=4096).to("cuda")
    return mc, _full(lens, cap)


def _full(lens, cap):
    return DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap, max_seqlen=4096,
                                   padded_size=DM.default_padded_size(sum(lens), cap)).to("cuda")


def _all4(maps, ts, causal=True, **okw):
    from flash_attn.cute.interface_bwd_fp4q import reset_sr_counter
    reset_sr_counter()
    q, k, v, do = ts
    xs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    okw = dict(okw)
    if "post" in okw:
        from flash_attn.cute.lowprec import KernelTuning
        okw["tuning"] = KernelTuning(bwd_fp4q_post=okw.pop("post"))
    o, lse = flash_attn_fp4(*xs, causal=causal, doc_maps=maps, return_lse=True, pv_precision="fp4", backward="fp4",
                            options=O4(**okw))
    return [o, lse] + list(torch.autograd.grad(o, xs, do))


def _ref(q, k, v, do, causal=True):
    q, k, v = (t.float().requires_grad_(True) for t in (q, k, v))
    r = q.shape[1] // k.shape[1]
    kk, vv = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    s = torch.einsum("qhd,khd->hqk", q, kk) / math.sqrt(q.shape[-1])
    if causal:
        L = q.shape[0]
        s = s.masked_fill(~torch.ones(L, L, device=q.device, dtype=torch.bool).tril(), float("-inf"))
    o = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
    return [o, None] + list(torch.autograd.grad(o, (q, k, v), do.float()))   # (the _all4 order: O, LSE, dQ, dK, dV)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


@gpu
@pytest.mark.parametrize("hk", [4, 1], ids=["mha", "gqa"])
def test_all4_k_channel_documents_full_maps(hk):
    """k_channel 'all' / 'open' with documents (full maps): per document, dQ / dK / dV as close to the FP32 reference
    as k_channel 'fold' ('all': the same arithmetic on every pair; 'open': no channel closing, finite)."""
    lens = LENS
    ts = _bf(lens, 4, hk, seed=3)
    mf = _full(lens, len(lens) + 3)
    cu = _cu(lens)
    res = {kc: _all4(mf, ts, k_channel=kc, all_fp4_stabiliser="rtn") for kc in ("fold", "all", "open")}
    for i, L in enumerate(lens):
        if L < 16:
            continue
        sl = slice(cu[i], cu[i + 1])
        ref = _ref(*(t[sl] for t in ts))
        for j, nm in ((2, "dQ"), (3, "dK"), (4, "dV")):
            ef = _rel(res["fold"][j][sl], ref[j])
            ea = _rel(res["all"][j][sl], ref[j])
            eo = _rel(res["open"][j][sl], ref[j])
            assert ef < 0.35, (nm, i, ef)
            assert ea < ef + 0.02, ("all", nm, i, ea, ef)
            assert math.isfinite(eo) and eo < ef + 0.02, ("open", nm, i, eo, ef)


# ---------------------------------------------------------------------------------------------------------------------
# the packed rows (compact maps) against the fused path (full maps)
# ---------------------------------------------------------------------------------------------------------------------
VARIANTS = {
    "all": (True, dict(k_channel="all")), "all_rtn": (True, dict(k_channel="all", all_fp4_stabiliser="rtn")),
    "open": (True, dict(k_channel="open")), "open_rtn": (True, dict(k_channel="open", all_fp4_stabiliser="rtn")),
    "epi": (True, dict(post="epi")), "dense": (False, {}), "dense_rtn": (False, dict(all_fp4_stabiliser="rtn")),
    "dense_exp2": (False, dict(softmax_exp="exp2")), "dense_pt": (False, dict(operand_scaling="per_tensor")),
}


@gpu
@pytest.mark.parametrize("hk", [4, 1], ids=["mha", "gqa"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_all4_packed_bitwise_cov6(hk, variant):
    causal, okw = VARIANTS[variant]
    for lens in (LENS, [5, 0, 130, 1, 2, 77]):
        ts = _bf(lens, 4, hk)
        mc, mf = _maps(lens, len(lens) + 3)
        c0 = list(DM.PACKED_CALLS)
        ref = _all4(mf, ts, causal, **okw)
        assert DM.PACKED_CALLS == c0, "full maps: the fused path"
        got = _all4(mc, ts, causal, **okw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "compact maps: the packed rows, forward and backward"
        assert _same4(ref, got), (variant, lens)


@gpu
@pytest.mark.parametrize("variant", ["all", "open", "dense"])
def test_all4_capacity_t_cov6(variant):
    causal, okw = VARIANTS[variant]
    for lens in ([1] * 512, [1, 300, 1, 1, 2, 129, 1, 77]):
        ts = _bf(lens, 8, 2)
        mc, mf = _maps(lens, sum(lens))
        assert _same4(_all4(mf, ts, causal, **okw), _all4(mc, ts, causal, **okw)), (variant, lens)


@gpu
@pytest.mark.parametrize("variant", ["all", "open", "dense"])
def test_all4_no_leak_cov6(variant):
    """A document's Q / K / V / dO scaled by 1000 (or its edge V / K rows): every other document's O / LSE / dQ / dK /
    dV bitwise unchanged (dQ: within the atomic-order spread)."""
    causal, okw = VARIANTS[variant]
    lens = [300, 1000, 0, 45, 700, 1, 128, 257, 3, 17]
    cu = _cu(lens)
    mc, _ = _maps(lens, len(lens) + 2)
    base = _bf(lens, 4, 2, seed=3)
    ref = _all4(mc, base, causal, **okw)
    for j in (0, 3, 5, 7, 9):
        for pm in ("edges", "all"):
            ts = [t.clone() for t in base]
            if pm == "all":
                for t in ts:
                    t[cu[j]:cu[j + 1]] *= 1000.0
            else:
                for t in ts[1:3]:
                    t[cu[j]] *= 1000.0
                    t[cu[j + 1] - 1] *= 1000.0
            got = _all4(mc, ts, causal, **okw)
            for i in range(len(lens)):
                if i == j or lens[i] == 0:
                    continue
                rows = slice(cu[i], cu[i + 1])
                assert _same4(ref, got, lambda t: (slice(None), rows) if t == 1 else (rows,)), (variant, j, pm, i)


@gpu
@pytest.mark.parametrize("variant", ["all", "open", "dense"])
def test_all4_graph_no_sync_cov6(variant):
    causal, okw = VARIANTS[variant]
    lens = [931, 1777, 3, 61, 0, 700, 1, 255]
    T = sum(lens)
    cu = torch.tensor(_cu(lens), dtype=torch.int32).cuda()
    ts = _bf(lens, 4, 2)
    torch.cuda.synchronize()
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=T, total_q=T, total_k=T, max_seqlen=2048)
    c0 = list(DM.PACKED_CALLS)
    _all4(maps, ts, causal, **okw)   # (compile)
    assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ref = _all4(maps, ts, causal, **okw)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert _same4(ref, _all4(maps, ts, causal, **okw))

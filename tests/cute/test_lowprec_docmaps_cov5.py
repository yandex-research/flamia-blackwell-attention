"""Packed rows of compact DocMaps for the 4-bit P.V routes (cov5): no full-map fallback, outputs bitwise the route's
fused path (full maps).

  * option A (pv_precision='fp4', inference): the all-FP4 forward with V^T realigned by the load warp from the
    key-parity copies (flash_attn.cute.v4_packed), Q / K / V quantised in the packed rows (quant.fp4 packed_rows=,
    quant.fp4_diagonal documents["packed"], the rotated Q / K of quant.fp8), O / LSE at the packed rows -- causal (the
    diagonal-only key-local V) / dense, MHA / GQA, the configuration variants, capacity N = T;
  * the cross-document leak probe (V groups at the documents' packed edges);
  * all-FP4 (backward='fp4', causal, key channel 'fold'): the forward plus flash_bwd_sm100_fp4q codes_packed (the K^T
    / SF atoms built by the load warp from the packed rows, statistics windows, dK / dV straight to the packed key
    rows, dQ through a (h, T') accumulator) -- variants, capacity N = T, the leak probe, no host sync.

run: pytest tests/cute/test_lowprec_docmaps_cov5.py -q
"""
import pytest
import torch

from flash_attn.cute import lp_docmaps as DM
from flash_attn.cute.lowprec import DocMaps, FP4AttnOptions as O4, flash_attn_fp4

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 0, 45, 700, 1, 128, 257]


def _cu(lens):
    return [0] + torch.tensor(lens).cumsum(0).tolist()


def _bits(a, b):
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def _bf(lens, h, hk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, 128, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, 128, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return [q, k, v, do]


def _opt_a(maps, ts, causal, **kw):
    q, k, v, _ = ts
    with torch.no_grad():
        o, lse = flash_attn_fp4(q, k, v, causal=causal, doc_maps=maps, return_lse=True, pv_precision="fp4", **kw)
    return [o, lse]


def _maps(lens, cap):
    T = sum(lens)
    mc = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap, max_seqlen=4096).to("cuda")
    mf = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap, max_seqlen=4096,
                                 padded_size=DM.default_padded_size(T, cap)).to("cuda")
    return mc, mf


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 1], ids=["mha", "gqa"])
@pytest.mark.parametrize("variant", ["default", "exp2", "mix_M", "single_level", "pcons_none", "cs_off", "had_off"])
def test_option_a_packed_bitwise(causal, hk, variant):
    okw = dict(exp2=dict(softmax_exp="exp2"), mix_M=dict(exp_mix="MMMMMMMM"), single_level=dict(p_frame="single_level"),
               pcons_none=dict(p_consistency="none")).get(variant, {})
    akw = dict(cs_off=dict(causal_safety="off"), had_off=dict(qk_hadamard=False)).get(variant, {})
    for lens in (LENS, [5, 0, 130, 1, 2, 77]):
        ts = _bf(lens, 4, hk)
        mc, mf = _maps(lens, len(lens) + 3)
        c0 = list(DM.PACKED_CALLS)
        ref = _opt_a(mf, ts, causal, options=O4(**okw), **akw)
        assert DM.PACKED_CALLS == c0, "full maps: the fused path"
        got = _opt_a(mc, ts, causal, options=O4(**okw), **akw)
        assert DM.PACKED_CALLS[0] == c0[0] + 1, "compact maps: the packed rows"
        assert ("aligned", "eager") not in DM._CACHE.get(mc, {}), "no full-map fallback"
        for i, (a, b) in enumerate(zip(ref, got)):
            assert _bits(a, b), (["O", "LSE"][i], lens)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_option_a_capacity_t(causal):
    """Capacity N = T (one-token and mixed documents): bitwise the fused path of the same documents."""
    for lens in ([1] * 512, [1, 300, 1, 1, 2, 129, 1, 77]):
        ts = _bf(lens, 8, 2)
        mc, mf = _maps(lens, sum(lens))
        assert all(_bits(a, b) for a, b in zip(_opt_a(mf, ts, causal), _opt_a(mc, ts, causal)))


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_option_a_no_leak(causal):
    """A document's V (its first / last key -- the 16-key groups at its packed edges) or all of its rows scaled by 1000:
    every other document's O / LSE bitwise unchanged."""
    lens = [300, 1000, 0, 45, 700, 1, 128, 257, 3, 17]
    cu = _cu(lens)
    mc, _ = _maps(lens, len(lens) + 2)
    base = _bf(lens, 4, 2, seed=3)
    ref = _opt_a(mc, base, causal)
    for j in (0, 3, 5, 7, 9):
        for pm in ("v_edges", "all"):
            ts = [t.clone() for t in base]
            if pm == "all":
                for t in ts[:3]:
                    t[cu[j]:cu[j + 1]] *= 1000.0
            else:
                ts[2][cu[j]] *= 1000.0
                ts[2][cu[j + 1] - 1] *= 1000.0
            got = _opt_a(mc, ts, causal)
            for i in range(len(lens)):
                if i == j or lens[i] == 0:
                    continue
                o_same = torch.equal(ref[0][cu[i]:cu[i + 1]], got[0][cu[i]:cu[i + 1]])
                l_same = torch.equal(ref[1][:, cu[i]:cu[i + 1]], got[1][:, cu[i]:cu[i + 1]])
                assert o_same and l_same, (j, pm, i)


@gpu
def test_option_a_graph_no_sync():
    """No host sync (maps built from CUDA cu_seqlens), and a CUDA-graph replay is bitwise the eager call."""
    lens = [931, 1777, 3, 61, 0, 700, 1, 255]
    T = sum(lens)
    cu = torch.tensor(_cu(lens), dtype=torch.int32).cuda()
    ts = _bf(lens, 4, 2)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        maps = DocMaps.from_cu_seqlens(cu, max_num_documents=T, total_q=T, total_k=T, max_seqlen=2048)
        ref = _opt_a(maps, ts, True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _opt_a(maps, ts, True)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = _opt_a(maps, ts, True)
    gr.replay()
    torch.cuda.synchronize()
    assert all(_bits(a, b) for a, b in zip(ref, out))


# ---------------------------------------------------------------------------------------------------------------------
# all-FP4 training (backward='fp4'): the causal diagonal route on the packed rows -- O / LSE / dK / dV bitwise the fused
# path's, dQ within its fp16 atomic accumulation spread (the SR counter reset before each call: the same dither)
# ---------------------------------------------------------------------------------------------------------------------

def _dq_close(a, b):
    """(the all-4-bit backward accumulates dQ with fp16 atomics: the fused path itself is not run-to-run bitwise
    there) dQ within the fp16 atomic accumulation's order spread: |a - b| <= 2^-6 * max |a| of its (token, head) row."""
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    af, bf = a.float(), b.float()
    tol = af.abs().amax(-1, keepdim=True) * 2.0 ** -6   # (the fused path against itself: up to 2^-7)
    return bool(((af - bf).abs() <= tol).all().item())


def _same4(ref, got, sl=None):
    """O / LSE / dK / dV bitwise, dQ within the atomic-order spread."""
    for t, (a, b) in enumerate(zip(ref, got)):
        if sl is not None:
            a, b = a[sl(t)], b[sl(t)]
        if t == 2:
            if not _dq_close(a, b):
                return False
        elif not _bits(a, b):
            return False
    return True


def _all4(maps, ts, causal=True, **okw):
    from flash_attn.cute.interface_bwd_fp4q import reset_sr_counter
    q, k, v, do = ts
    reset_sr_counter()
    px = [t.clone().requires_grad_(True) for t in (q, k, v)]
    o, lse = flash_attn_fp4(*px, causal=causal, doc_maps=maps, return_lse=True, pv_precision="fp4", backward="fp4",
                            options=O4(**okw))
    return [o, lse] + list(torch.autograd.grad(o, px, do))


ALL4_VARIANTS = dict(default={}, rtn=dict(all_fp4_stabiliser="rtn"), no_srdk=dict(sr_dk_operand=False),
                     srdv=dict(sr_dv_operand=True), exp2=dict(softmax_exp="exp2"), mix=dict(exp_mix="EMMEMMEM"),
                     pt=dict(operand_scaling="per_tensor"))


@gpu
@pytest.mark.parametrize("hk", [4, 1], ids=["mha", "gqa"])
@pytest.mark.parametrize("variant", list(ALL4_VARIANTS))
def test_all4_packed_bitwise(hk, variant):
    for lens in (LENS, [5, 0, 130, 1, 2, 77]):
        ts = _bf(lens, 4, hk)
        mc, mf = _maps(lens, len(lens) + 3)
        c0 = list(DM.PACKED_CALLS)
        ref = _all4(mf, ts, **ALL4_VARIANTS[variant])
        assert DM.PACKED_CALLS == c0, "full maps: the fused path"
        got = _all4(mc, ts, **ALL4_VARIANTS[variant])
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "compact maps: the packed rows, forward and backward"
        assert _same4(ref, got), lens


@gpu
def test_all4_capacity_t():
    for lens in ([1] * 512, [1, 300, 1, 1, 2, 129, 1, 77]):
        ts = _bf(lens, 8, 2)
        mc, mf = _maps(lens, sum(lens))
        assert _same4(_all4(mf, ts), _all4(mc, ts))


@gpu
def test_all4_no_leak():
    """A document's Q / K / V / dO scaled by 1000 (or its edge V rows): every other document's O / LSE / dQ / dK / dV
    bitwise unchanged."""
    lens = [300, 1000, 0, 45, 700, 1, 128, 257, 3, 17]
    cu = _cu(lens)
    mc, _ = _maps(lens, len(lens) + 2)
    base = _bf(lens, 4, 2, seed=3)
    ref = _all4(mc, base)
    for j in (0, 3, 5, 7, 9):
        for pm in ("v_edges", "all"):
            ts = [t.clone() for t in base]
            if pm == "all":
                for t in ts:
                    t[cu[j]:cu[j + 1]] *= 1000.0
            else:
                ts[2][cu[j]] *= 1000.0
                ts[2][cu[j + 1] - 1] *= 1000.0
            got = _all4(mc, ts)
            for i in range(len(lens)):
                if i == j or lens[i] == 0:
                    continue
                rows = slice(cu[i], cu[i + 1])
                assert _same4(ref, got, lambda t: (slice(None), rows) if t == 1 else (rows,)), (j, pm, i)


@gpu
def test_all4_graph_no_sync():
    lens = [931, 1777, 3, 61, 0, 700, 1, 255]
    T = sum(lens)
    cu = torch.tensor(_cu(lens), dtype=torch.int32).cuda()
    ts = _bf(lens, 4, 2)
    torch.cuda.synchronize()
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=T, total_q=T, total_k=T, max_seqlen=2048)
    _all4(maps, ts)   # (compile)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ref = _all4(maps, ts)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert _same4(ref, _all4(maps, ts))

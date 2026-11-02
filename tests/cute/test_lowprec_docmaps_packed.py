"""Packed rows for doc_maps calls (lp_docmaps.PACKED_ROWS / PACKED_BWD, the default): no aligned (S-row) buffer -- the
codes, O, the LSE, the dQ accumulator stay at the packed T rows; only the per-tile scale tables and the backward's
statistics are per aligned tile / row.

  * outputs bitwise (bit patterns, sign of zero included) the fused doc_maps path's (57cab39): fp8 / fp4, causal /
    dense, MHA / GQA, d 64 / 128, qk_hadamard
    off and default, zero-length / length-1 / non-multiple-of-128 documents, deterministic (O, LSE, dQ, dK, dV) and not
    (O, LSE, dK, dV); the 1-CTA backward on both sides (two_cta="off");
  * return_lse = the fused path's (h, T) rows;
  * a perturbed document leaves every other document's outputs bitwise unchanged (the cross-document leak probe);
  * no host sync; CUDA-graph capture with static maps, replayed on another microbatch (copy_).

run: pytest tests/cute/test_lowprec_docmaps_packed.py -q
"""
import functools

import pytest
import torch

from flash_attn.cute import lp_docmaps as DM
from flash_attn.cute.lowprec import DocMaps, FP4AttnOptions as O4, FP8AttnOptions as O8, flash_attn_fp4, flash_attn_fp8
from flash_attn.cute.lowprec.options import KernelTuning

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 0, 45, 700, 1, 128, 257]
CAP = 12


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _packed(lens, h, hk, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return q, k, v, do


@pytest.fixture
def packed_mode(monkeypatch):
    def set_(on: bool):
        monkeypatch.setattr(DM, "PACKED_ROWS", on)
        monkeypatch.setattr(DM, "PACKED_BWD", on)
    return set_


def _run(fn, q, k, v, do, **kw):
    x = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o, lse = fn(*x, return_lse=True, **kw)
    return [o, lse] + list(torch.autograd.grad(o, x, do))


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("had", [None, False], ids=["had_default", "had_off"])
def test_bitwise_fused(fn, O, causal, hk, d, had, packed_mode):
    if d == 64 and had is None:
        pytest.skip("qk_hadamard is d = 128 only (default off at d = 64: the had_off case)")
    q, k, v, do = _packed(LENS, 4, hk, d)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP, max_seqlen=1024).to("cuda")
    for det in (True, False):
        kw = dict(causal=causal, deterministic=det, doc_maps=maps, qk_hadamard=had, options=O(two_cta="off"))
        packed_mode(False)
        ref = _run(fn, q, k, v, do, **kw)
        packed_mode(True)
        c0 = list(DM.PACKED_CALLS)
        got = _run(fn, q, k, v, do, **kw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:   # (dQ: atomics without deterministic)
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


def _bits_equal(a, b):
    it = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


@gpu
def test_default_and_uncovered_routes():
    """Packed rows are the default for doc_maps calls; a backward setting the packed path does not serve (no ds_fold on
    FULL maps: their packed padding statistics rely on the dO launch's D fold; compact maps serve it, cov3) keeps the
    fused aligned path, with the same outputs."""
    assert DM.PACKED_ROWS and DM.PACKED_BWD
    q, k, v, do = _packed(LENS, 4, 2)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP, max_seqlen=1024).to("cuda")
    c0 = list(DM.PACKED_CALLS)
    _run(flash_attn_fp8, q, k, v, do, causal=True, doc_maps=maps)
    assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1]
    full = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP, max_seqlen=1024, padded_size=128 * 32).to("cuda")
    c0 = list(DM.PACKED_CALLS)
    _run(flash_attn_fp8, q, k, v, do, causal=True, doc_maps=full, options=O8(tuning=KernelTuning(bwd_fp4_ds_fold=False)))
    assert DM.PACKED_CALLS == c0, "ds_fold=False must keep the fused path"


@gpu
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
def test_inference_lse(fn, packed_mode):
    q, k, v, _ = _packed(LENS, 4, 2)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP).to("cuda")
    packed_mode(False)
    with torch.no_grad():
        o0, l0 = fn(q, k, v, causal=True, return_lse=True, doc_maps=maps)
    packed_mode(True)
    with torch.no_grad():
        o1, l1 = fn(q, k, v, causal=True, return_lse=True, doc_maps=maps)
    assert l1.shape == (4, sum(LENS)) and torch.equal(o0, o1) and torch.equal(l0, l1)


@gpu
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_leak_probe(fn, causal, packed_mode):
    """Perturb one document (noise; a x1000 blow-up): every other document's O / LSE / dQ / dK / dV bitwise unchanged.
    (Under the blow-up dQ is held to the fused path's dQ of the same call instead: the f16 dQ accumulator's pre-scale is
    one per call, from the max |K| / ||V|| / ||dO|| over all documents -- a channel of the aligned path too.)"""
    packed_mode(True)
    lens = [300, 1000, 0, 45, 700, 1, 128, 257, 3, 999]
    cu = _cu(lens).tolist()
    q, k, v, do = _packed(lens, 4, 2, 64, seed=3)
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=12, max_seqlen=2048).to("cuda")
    kw = dict(causal=causal, deterministic=True, doc_maps=maps)

    def outs(ts, pk=True):
        packed_mode(pk)
        r = _run(fn, *ts, **kw)
        r[1] = r[1].t()
        return r
    ref = outs((q, k, v, do))
    g = torch.Generator(device="cuda").manual_seed(7)
    for j in (0, 6, 7, 9):
        for mode in ("noise", "blowup"):
            ts = [t.clone() for t in (q, k, v, do)]
            for t in ts:
                if mode == "noise":
                    t[cu[j]:cu[j + 1]] += torch.randn(t[cu[j]:cu[j + 1]].shape, device="cuda", generator=g).to(t.dtype)
                else:
                    t[cu[j]:cu[j + 1]] *= 1000.0
            got = outs(ts)
            fused_dq = outs(ts, False)[2] if mode == "blowup" else None
            for nm, a, b in zip(("O", "LSE", "dQ", "dK", "dV"), ref, got):
                if nm == "dQ" and fused_dq is not None:
                    assert torch.equal(fused_dq, b), ("dQ differs from the fused path's", j, mode)
                    continue
                for i in range(len(lens)):
                    same = torch.equal(a[cu[i]:cu[i + 1]], b[cu[i]:cu[i + 1]])
                    assert same or i == j, (nm, "leak into document", i, "from", j, mode)


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("d", [64, 128])
def test_no_sync_and_graph(fn, O, d, packed_mode):
    packed_mode(True)
    lens_a, lens_b = LENS, [700, 3, 300, 129, 0, 1000, 299]
    assert sum(lens_a) == sum(lens_b)
    q, k, v, do = _packed(lens_a, 4, 2, d)
    q2, k2, v2, do2 = _packed(lens_b, 4, 2, d, seed=1)
    ma = DocMaps.from_cu_seqlens(_cu(lens_a), max_num_documents=CAP, max_seqlen=1024)
    mb = DocMaps.from_cu_seqlens(_cu(lens_b), max_num_documents=CAP, max_seqlen=1024)
    static = ma.to("cuda")
    kw = dict(causal=True, deterministic=True)
    ref_a = _run(fn, q, k, v, do, doc_maps=ma.to("cuda"), **kw)
    ref_b = _run(fn, q2, k2, v2, do2, doc_maps=mb.to("cuda"), **kw)
    xs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    gdo = do.clone()
    c0 = list(DM.PACKED_CALLS)
    torch.cuda.set_sync_debug_mode("error")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for x in xs:
                    x.grad = None
                fn(*xs, doc_maps=static, **kw).backward(gdo)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        for x in xs:
            x.grad = None
        with torch.cuda.graph(g):
            o = fn(*xs, doc_maps=static, **kw)
            o.backward(gdo)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert DM.PACKED_CALLS[0] - c0[0] == 3 and DM.PACKED_CALLS[1] - c0[1] == 3
    for a, b in zip((q, k, v, do), (*xs, gdo)):
        b.data.copy_(a)
    g.replay()
    assert torch.equal(o, ref_a[0]) and all(torch.equal(x.grad, r) for x, r in zip(xs, ref_a[2:]))
    static.copy_(mb.to("cuda"))
    for a, b in zip((q2, k2, v2, do2), (*xs, gdo)):
        b.data.copy_(a)
    g.replay()
    assert torch.equal(o, ref_b[0]) and all(torch.equal(x.grad, r) for x, r in zip(xs, ref_b[2:]))


# ---------------------------------------------------------------------------------------------------------------------
# cov1: routes the packed rows serve on compact maps with no full-map fallback (lp_docmaps.packed_only_ok / packed_ok)
# ---------------------------------------------------------------------------------------------------------------------
LENS_N1 = [1] * 200 + [2, 300, 1, 129] + [1] * 40   # (a capacity N = T worst case: mostly one-token documents)


def _no_full_maps(monkeypatch):
    """Fail if the call builds the full (padded) maps of compact maps (DocMaps.aligned_maps)."""
    def refuse(self):
        raise AssertionError("aligned_maps() called: the full-map fallback")
    monkeypatch.setattr(DM.DocMaps, "aligned_maps", refuse)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [96, 80, 32])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_padded_head_dim_packed(causal, hk, d, lens, cap, packed_mode, monkeypatch):
    """8-bit token-block route at a padded head dim on compact maps: the packed rows (codes padded to 64 / 128 columns
    by the quantisers), no full maps; O / LSE / dK / dV (and dQ, deterministic) bitwise the fused path's."""
    q, k, v, do = _packed(lens, 4, hk, d)
    T = sum(lens)
    maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap or T, max_seqlen=1024).to("cuda")
    assert maps.compact
    for det in (True, False):
        kw = dict(causal=causal, deterministic=det, doc_maps=maps)
        packed_mode(False)
        ref = _run(flash_attn_fp8, q, k, v, do, **kw)
        packed_mode(True)
        with monkeypatch.context() as m:
            _no_full_maps(m)
            c0 = list(DM.PACKED_CALLS)
            got = _run(flash_attn_fp8, q, k, v, do, **kw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_per_tensor_packed(fn, O, causal, hk, d, lens, cap, packed_mode, monkeypatch):
    """operand_scaling='per_tensor' (one E4M3 scale per (document, kv head)) on compact maps: the packed rows (the
    per-tensor quantisers' packed documents mode, the 4-bit route's NVFP4 S words dense per row, the 1-CTA backward's
    dense statistics), no full maps; O / LSE / dK / dV (and dQ, deterministic) bitwise the fused path's."""
    if fn is flash_attn_fp4 and d == 64:
        pytest.skip("per-tensor operands on documents: the 4-bit route is d = 128 only")
    q, k, v, do = _packed(lens, 4, hk, d)
    T = sum(lens)
    maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap or T, max_seqlen=1024).to("cuda")
    for det in (True, False):
        kw = dict(causal=causal, deterministic=det, doc_maps=maps, options=O(operand_scaling="per_tensor"))
        packed_mode(False)
        ref = _run(fn, q, k, v, do, **kw)
        packed_mode(True)
        with monkeypatch.context() as m:
            _no_full_maps(m)
            c0 = list(DM.PACKED_CALLS)
            got = _run(fn, q, k, v, do, **kw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


# ---------------------------------------------------------------------------------------------------------------------
# cov3: more setups on compact maps' packed rows (no full-map fallback)
# ---------------------------------------------------------------------------------------------------------------------
def _packed_vs_fused(fn, lens, cap, hk, d, causal, packed_mode, monkeypatch, **kw):
    """O / LSE / dK / dV (and dQ, deterministic) of a compact-maps call: the packed rows, no full maps, bitwise the
    fused path's (PACKED_ROWS off: the full maps of the same documents)."""
    q, k, v, do = _packed(lens, 4, hk, d)
    T = sum(lens)
    maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap or T, max_seqlen=1024).to("cuda")
    assert maps.compact
    for det in (True, False):
        a = dict(causal=causal, deterministic=det, doc_maps=maps, **kw)
        packed_mode(False)
        ref = _run(fn, q, k, v, do, **a)
        packed_mode(True)
        with monkeypatch.context() as m:
            _no_full_maps(m)
            c0 = list(DM.PACKED_CALLS)
            got = _run(fn, q, k, v, do, **a)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("rowlocal,causal", [("diagonal", True), ("all", True), ("all", False)],
                         ids=["diag-causal", "all-causal", "all-dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_ds_dq_rowlocal_packed(fn, O, rowlocal, causal, hk, d, lens, cap, packed_mode, monkeypatch):
    """ds_dq_rowlocal (the dQ side's per-query dS exponents) on compact maps' sf_dense backward."""
    _packed_vs_fused(fn, lens, cap, hk, d, causal, packed_mode, monkeypatch, options=O(ds_dq_rowlocal=rowlocal))


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_no_ds_fold_packed(fn, O, causal, hk, d, lens, cap, packed_mode, monkeypatch):
    """KernelTuning.bwd_fp4_ds_fold=False (D unfolded: the sf_dense load warp writes D = -2^100 into a padding row)."""
    _packed_vs_fused(fn, lens, cap, hk, d, causal, packed_mode, monkeypatch,
                     options=O(tuning=KernelTuning(bwd_fp4_ds_fold=False)))


@gpu
@pytest.mark.parametrize("operand_scaling", ["auto", "per_tensor"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_q_block_2d_packed(operand_scaling, hk, d, lens, cap, packed_mode, monkeypatch):
    """q_block_2d (Q's NVFP4 16 x 16 blocks; dense documents only) from quant.fp4's packed rows."""
    if operand_scaling == "per_tensor" and d != 128:
        pytest.skip("per-tensor operands on documents: the 4-bit route is d = 128 only")
    _packed_vs_fused(flash_attn_fp4, lens, cap, hk, d, False, packed_mode, monkeypatch, qk_hadamard=None,
                     options=O4(q_block_2d=True, operand_scaling=operand_scaling))


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_tcgen05_rotation_packed(fn, O, causal, hk, lens, cap, packed_mode, monkeypatch):
    """qk_hadamard with the tcgen05 rotation (quant._fp8_tcgen05's compact form: the packed rows, dense tables)."""
    _packed_vs_fused(fn, lens, cap, hk, 128, causal, packed_mode, monkeypatch, qk_hadamard=True,
                     options=O(tuning=KernelTuning(qk_hadamard_rotation_method="tcgen05")))


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_fp8_qk_sf_vec_packed(causal, d, lens, cap, packed_mode, monkeypatch):
    """KernelTuning.fp4_qk_sf_vec=32 on the 8-bit route (which reads no 4-bit S operand): the packed rows."""
    _packed_vs_fused(flash_attn_fp8, lens, cap, 2, d, causal, packed_mode, monkeypatch,
                     options=O8(tuning=KernelTuning(fp4_qk_sf_vec=32)))


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_token_local_packed(fn, O, causal, hk, d, lens, cap, monkeypatch):
    """operand_scaling='token_local' on compact maps: the fold forward and the token table per packed row, no full maps;
    O / LSE / dK / dV (and dQ, deterministic) bitwise the aligned path's (the documents at 128-aligned starts in a
    zero-padded buffer, cu_seqlens + seqused -- the doc_maps fused path of token-local operands does not run)."""
    q, k, v, do = _packed(lens, 4, hk, d)
    T = sum(lens)
    maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap or T, max_seqlen=1024).to("cuda")
    cp = [0]
    for L in lens:
        cp.append(cp[-1] + (L + 127) // 128 * 128)
    idx = torch.cat([torch.arange(cp[i], cp[i] + L) for i, L in enumerate(lens)]).cuda()

    def pad(x):
        y = torch.zeros((cp[-1],) + x.shape[1:], dtype=x.dtype, device=x.device)
        y[idx] = x
        return y
    cpt = torch.tensor(cp, dtype=torch.int32, device="cuda")
    used = torch.tensor(lens, dtype=torch.int32, device="cuda")
    for det in (False,):   # (the deterministic backward keeps token-local operands off the packed rows)
        kw = dict(causal=causal, deterministic=det, qk_hadamard=None,
                  options=O(operand_scaling="token_local", ds_dq_rowlocal="all"))
        xa = [pad(t).requires_grad_(True) for t in (q, k, v)]
        o, lse = fn(*xa, cu_seqlens_q=cpt, cu_seqlens_k=cpt, seqused_q=used, seqused_k=used, max_seqlen_q=max(lens),
                    max_seqlen_k=max(lens), return_lse=True, **kw)
        ref = [o[idx], lse[:, idx]] + [t[idx] for t in torch.autograd.grad(o, xa, pad(do))]
        with monkeypatch.context() as m:
            _no_full_maps(m)
            c0 = list(DM.PACKED_CALLS)
            got = _run(fn, q, k, v, do, doc_maps=maps, **kw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


# ---------------------------------------------------------------------------------------------------------------------
# cov4: window / softcap / score_mod / mask_mod on the packed rows (compact maps, no full-map fallback)
# ---------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _mods():
    """score_mod / mask_mod that read the (document-local) indices: a coordinate mismatch shows."""
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def sm_idx(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return tSrS_ssa + (q_idx % 7).to(cutlass.Float32) * 0.25 - (kv_idx % 5).to(cutlass.Float32) * 0.5

    @cute.jit
    def sm_idx_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad

    @cute.jit
    def sm_sq(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return tSrS_ssa * tSrS_ssa

    @cute.jit
    def sm_sq_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad * cute.full_like(grad, 2.0) * score

    @cute.jit
    def mm_stripe(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        three, one = utils.scalar_to_ssa(3, cutlass.Int32), utils.scalar_to_ssa(1, cutlass.Int32)
        return ((n_idx % three) != one) | (n_idx == m_idx)

    @cute.jit
    def mm_band(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        w = utils.scalar_to_ssa(64, cutlass.Int32)
        return (n_idx >= m_idx - w) & (n_idx <= m_idx + w)

    return dict(sm_idx_c=dict(causal=True, score_mod=sm_idx, score_mod_bwd=sm_idx_bwd),
                sm_sq=dict(causal=False, score_mod=sm_sq, score_mod_bwd=sm_sq_bwd),
                mm_stripe=dict(causal=False, mask_mod=mm_stripe), mm_band=dict(causal=False, mask_mod=mm_band))


FEATS4 = ["w64c", "w300c", "w128x128", "w17x200", "cap30c", "cap5", "sm_idx_c", "sm_sq", "mm_stripe", "mm_band"]


def _feat4(name):
    if name.startswith("w"):
        return {"w64c": dict(causal=True, window_size=(64, 0)), "w300c": dict(causal=True, window_size=(300, 0)),
                "w128x128": dict(causal=False, window_size=(128, 128)),
                "w17x200": dict(causal=False, window_size=(17, 200))}[name]
    if name.startswith("cap"):
        return dict(causal=True, softcap=30.0) if name == "cap30c" else dict(causal=False, softcap=5.0)
    return _mods()[name]


@gpu
@pytest.mark.parametrize("feat", FEATS4)
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), (LENS_N1, None)], ids=["docs", "n_eq_t"])
def test_masks_packed(feat, fn, hk, d, lens, cap, packed_mode, monkeypatch):
    """window / softcap / score_mod / mask_mod on compact maps: the packed rows (a window / mask_mod: the packed dK / dV
    start zero for key tiles with no query), no full maps; O / LSE / dK / dV (and dQ, deterministic) bitwise the fused
    path's."""
    q, k, v, do = _packed(lens, 4, hk, d)
    T = sum(lens)
    maps = DocMaps.from_cu_seqlens(_cu(lens), max_num_documents=cap or T, max_seqlen=1024).to("cuda")
    for det in (True, False):
        kw = dict(deterministic=det, doc_maps=maps, **_feat4(feat))
        packed_mode(False)
        ref = _run(fn, q, k, v, do, **kw)
        packed_mode(True)
        with monkeypatch.context() as m:
            _no_full_maps(m)
            c0 = list(DM.PACKED_CALLS)
            got = _run(fn, q, k, v, do, **kw)
        assert DM.PACKED_CALLS == [c0[0] + 1, c0[1] + 1], "the packed rows were not taken"
        for i, (x, y) in enumerate(zip(ref, got)):
            if i == 2 and not det:
                continue
            assert _bits_equal(x, y), (det, ["O", "LSE", "dQ", "dK", "dV"][i])


@gpu
@pytest.mark.parametrize("feat", ["w64c", "w17x200", "cap30c", "sm_idx_c", "mm_stripe"])
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
def test_masks_leak_probe(feat, fn, packed_mode):
    """The cross-document leak probe with a window / softcap / mods: a perturbed document (noise, x1000) leaves every
    other document's outputs bitwise unchanged (dQ under the blow-up: the fused path's dQ of the same call)."""
    packed_mode(True)
    lens = [300, 1000, 0, 45, 700, 1, 128, 257, 3, 999]
    cu = _cu(lens).tolist()
    q, k, v, do = _packed(lens, 4, 2, 64, seed=3)
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=12, max_seqlen=2048).to("cuda")
    kw = dict(deterministic=True, doc_maps=maps, **_feat4(feat))

    def outs(ts, pk=True):
        packed_mode(pk)
        r = _run(fn, *ts, **kw)
        r[1] = r[1].t()
        return r
    ref = outs((q, k, v, do))
    g = torch.Generator(device="cuda").manual_seed(7)
    for j in (0, 6, 9):
        for mode in ("noise", "blowup"):
            ts = [t.clone() for t in (q, k, v, do)]
            for t in ts:
                if mode == "noise":
                    t[cu[j]:cu[j + 1]] += torch.randn(t[cu[j]:cu[j + 1]].shape, device="cuda", generator=g).to(t.dtype)
                else:
                    t[cu[j]:cu[j + 1]] *= 1000.0
            got = outs(ts)
            fused_dq = outs(ts, False)[2] if mode == "blowup" else None
            for nm, a, b in zip(("O", "LSE", "dQ", "dK", "dV"), ref, got):
                if nm == "dQ" and fused_dq is not None:
                    assert torch.equal(fused_dq, b), ("dQ differs from the fused path's", j, mode)
                    continue
                for i in range(len(lens)):
                    same = torch.equal(a[cu[i]:cu[i + 1]], b[cu[i]:cu[i + 1]])
                    assert same or i == j, (nm, "leak into document", i, "from", j, mode)

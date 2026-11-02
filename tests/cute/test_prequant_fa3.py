"""FA3-style pre-quantised inputs (flash_attn_fp8 with E4M3 q / k / v + q_descale / k_descale / v_descale).

  * test_rule_*           the plan's rule-32 refusals (meta tensors, no CUDA work) and the API's host validation;
  * test_equivalence_*    codes + scales from our own per-tensor quantiser handed over as pre-quantised input give
                          outputs BITWISE equal to the internal per-tensor path: batched (one scale per tensor,
                          broadcast to (b, h_kv)) and aligned varlen (one per (document, kv head)), causal / dense,
                          MHA / GQA; the backward through grad_proxies on aligned varlen (deterministic dQ) too;
  * test_batched_grad     the batched training path: bitwise the inference forward, the documents form of the
                          call and the internal per-tensor aligned varlen path on the rows as documents (same
                          granularity); gradients as accurate vs fp32 as the internal batched per-tensor path's;
  * test_no_host_sync / test_graph_capture   inference and training steps.

run: pytest tests/cute/test_prequant_fa3.py -q
"""
import math

import pytest
import torch

from flash_attn.cute.lowprec import FP8AttnOptions as O8, flash_attn_fp4, flash_attn_fp8, resolve_fp8

E4 = torch.float8_e4m3fn
BF16 = torch.bfloat16
PT = O8(operand_scaling="per_tensor")
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")


# ================================================================================================================
# CPU: rule 32 on the plan (meta tensors)
# ================================================================================================================
def M(*shape, dtype=E4):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _qkv8(b=1, s=1024, h=4, hk=4, d=128, s_k=None):
    return M(b, s, h, d), M(b, s_k or s, hk, d), M(b, s_k or s, hk, d)


_I32 = lambda n: torch.empty(n, dtype=torch.int32, device="meta")  # noqa: E731

RULES = [
    ("seqused_k", dict(seqused_k=_I32(1)), {}, NotImplementedError, "pre-quantised E4M3 inputs with seqused_k"),
    ("d96", {}, dict(d=96), NotImplementedError, "pre-quantised E4M3 inputs: head_dim 64 or 128"),
    ("ragged", {}, dict(s=1000), NotImplementedError, "pre-quantised E4M3 inputs on a batched call need s_q % 128"),
    ("op_tb", dict(options=O8(operand_scaling="token_block")), {}, ValueError, "operand_scaling='token_block' with"),
    ("upstream", dict(options=O8(forward_kernel="upstream")), {}, ValueError, "forward_kernel='upstream' with"),
    ("e5m2", dict(options=O8(grad_format="e5m2", operand_scaling="per_tensor")), {}, ValueError,
     "grad_format='e5m2' with pre-quantised"),
    ("both", dict(q_scale_fused=True), {}, ValueError, "flash_attn_fp8: q_descale / k_descale / v_descale (E4M3"),
    ("hadamard", dict(qk_hadamard=True), {}, ValueError, "pre-quantised Q/K are unrotated codes"),
]


@pytest.mark.parametrize("name,kw,shape,exc,msg", RULES, ids=[r[0] for r in RULES])
def test_rule_32_plan(name, kw, shape, exc, msg):
    with pytest.raises(exc) as ei:
        resolve_fp8(*_qkv8(**shape), prequant_fa3=True, **kw)
    assert msg in str(ei.value), str(ei.value)


def test_rule_32_repacked_varlen():
    q, k, v = M(1024, 4, 128), M(1024, 4, 128), M(1024, 4, 128)
    with pytest.raises(NotImplementedError, match="repacked varlen"):
        resolve_fp8(q, k, v, cu_seqlens_q=_I32(3), cu_seqlens_k=_I32(3), prequant_fa3=True)


@pytest.mark.parametrize("kw", [dict(window_size=(64, 0), causal=True), dict(softcap=30.0),
                                dict(mask_mod=lambda *a: None)], ids=["window", "softcap", "mask_mod"])
def test_plan_fa3_masks_resolve(kw):
    """window / softcap / score_mod / mask_mod run on the pre-quantised path (the forward's score_mod reads the
    per-(row or document, kv head) score scale before the mod; the doc_scaled backward applies masks and mods)."""
    p = resolve_fp8(*_qkv8(), prequant_fa3=True, **kw)
    assert p.fwd.operand_scaling == "per_tensor"


def test_plan_fa3_resolves():
    p = resolve_fp8(*_qkv8(), causal=True, prequant_fa3=True)
    assert p.fwd.operand_scaling == "per_tensor" and not p.fwd.qk_hadamard and p.bwd.kernel == "none"
    assert p.shape.dtype == "bf16" and p.prequant
    assert ("prequant", "fa3") == p.resolution_log[[r[0] for r in p.resolution_log].index("prequant")][:2]


# ================================================================================================================
# GPU helpers
# ================================================================================================================
def _docs_layout(lens, hq, hk, d, seed=0):
    starts, tot = [], 0
    for n in lens:
        starts.append(tot)
        tot += -(-n // 128) * 128
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(tot, hq, d, device="cuda", dtype=BF16, generator=g)
    k, v = (torch.randn(tot, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    msk = torch.zeros(tot, dtype=torch.bool, device="cuda")
    for s0, n in zip(starts, lens):
        msk[s0:s0 + n] = True
    q, k, v = (t * msk.view(-1, 1, 1) for t in (q, k, v))   # padding rows: zero (the padding contract)
    cu = torch.tensor(starts + [tot], dtype=torch.int32, device="cuda")
    su = torch.tensor(lens, dtype=torch.int32, device="cuda")
    kw = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, seqused_q=su, seqused_k=su, max_seqlen_q=max(lens),
              max_seqlen_k=max(lens))
    return q, k, v, msk, kw


def _our_codes_varlen(q, k, v, kw, pow2=False):
    """Our own per-(document, kv head) quantiser (the internal per-tensor varlen path's call): codes + float64 tables."""
    from flash_attn.cute.lp_varlen import LPDocs
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    cu, su = kw["cu_seqlens_q"], kw["seqused_q"]
    docs = LPDocs.aligned_from(cu, cu, su, su, q.shape[0], k.shape[0])
    r = quantize_fp8_per_tensor([q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)], power_of_two_scale=pow2,
                                documents=dict(docs.doc_quant_args(("q", "k", "k"), k.shape[1])))
    n = su.shape[0]
    return [c[0] for c in r["codes"]], [r["doc_scales"][i][:n] for i in range(3)]


def _our_codes_batched(q, k, v, pow2=False):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    r = quantize_fp8_per_tensor([q, k, v], power_of_two_scale=pow2)
    b, hk = q.shape[0], k.shape[2]
    return list(r["codes"]), [r["scales"][i].reshape(1, 1).expand(b, hk) for i in range(3)]


def _batched(b, s, hq, hk, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, s, hq, d, device="cuda", dtype=BF16, generator=g)
    k, v = (torch.randn(b, s, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return q, k, v


def _d(ds):
    return dict(q_descale=ds[0], k_descale=ds[1], v_descale=ds[2])


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


# ================================================================================================================
# GPU: host validation
# ================================================================================================================
@gpu
def test_rule_32_api_validation():
    q, k, v = _batched(1, 256, 4, 2, 128)
    (q8, k8, v8), ds = _our_codes_batched(q, k, v)
    ok = _d([x.float().contiguous() for x in ds])
    with pytest.raises(ValueError, match="need q_descale, k_descale and v_descale"):
        flash_attn_fp8(q8, k8, v8, q_descale=ok["q_descale"])
    with pytest.raises(ValueError, match=r"must be a float32 \(or float64\) CUDA tensor of shape \(b, h_kv\) = \(1, 2\)"):
        flash_attn_fp8(q8, k8, v8, **dict(ok, k_descale=torch.ones(1, 4, device="cuda")))
    with pytest.raises(ValueError, match="must be a float32"):
        flash_attn_fp8(q8, k8, v8, **dict(ok, v_descale=torch.ones(1, 2, device="cuda", dtype=BF16)))
    with pytest.raises(ValueError, match="must all be torch.float8_e4m3fn"):
        flash_attn_fp8(q, k, v, **ok)
    with pytest.raises(ValueError, match="grad_proxies"):
        flash_attn_fp8(q8.clone().requires_grad_(True), k8, v8, **ok)
    with pytest.raises(ValueError, match=r"grad_proxies\[1\] must be a bf16 tensor"):
        flash_attn_fp8(q8, k8, v8, **ok, grad_proxies=(q.clone().requires_grad_(), q.clone(), v.clone()))
    with pytest.raises(NotImplementedError, match="flash_attn_fp4 takes no pre-quantised E4M3"):
        flash_attn_fp4(q8, k8, v8, **ok)
    with pytest.raises(NotImplementedError, match="flash_attn_fp4 takes no pre-quantised E4M3"):
        flash_attn_fp4(q, k, v, **ok)


# ================================================================================================================
# GPU: bitwise equivalence with the internal per-tensor path
# ================================================================================================================
@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hq,hk,d", [(4, 4, 128), (8, 2, 128), (4, 4, 64)], ids=["mha", "gqa", "d64"])
def test_equivalence_batched(causal, hq, hk, d):
    q, k, v = _batched(2, 1024, hq, hk, d)
    ref, lse_ref = flash_attn_fp8(q, k, v, causal=causal, qk_hadamard=False, options=PT, return_lse=True)
    (q8, k8, v8), ds = _our_codes_batched(q, k, v)
    out, lse = flash_attn_fp8(q8, k8, v8, causal=causal, return_lse=True, **_d(ds))
    assert out.dtype == BF16 and out.shape == ref.shape
    assert torch.equal(out, ref) and torch.equal(lse, lse_ref)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_equivalence_batched_two_cta_forward(causal):
    q, k, v = _batched(1, 4096, 4, 4, 128)
    o2 = O8(operand_scaling="per_tensor", forward_two_cta="on")
    ref = flash_attn_fp8(q, k, v, causal=causal, qk_hadamard=False, options=o2)
    (q8, k8, v8), ds = _our_codes_batched(q, k, v)
    out = flash_attn_fp8(q8, k8, v8, causal=causal, options=O8(forward_two_cta="on"), **_d(ds))
    assert torch.equal(out, ref)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hq,hk,d", [(4, 4, 128), (8, 2, 128), (4, 4, 64)], ids=["mha", "gqa", "d64"])
def test_equivalence_varlen(causal, hq, hk, d):
    q, k, v, msk, kw = _docs_layout([300, 700, 128, 1000], hq, hk, d)
    ref, lse_ref = flash_attn_fp8(q, k, v, causal=causal, qk_hadamard=False, options=PT, return_lse=True, **kw)
    (q8, k8, v8), ds = _our_codes_varlen(q, k, v, kw)
    out, lse = flash_attn_fp8(q8, k8, v8, causal=causal, return_lse=True, **_d(ds), **kw)
    assert torch.equal(out[msk], ref[msk]) and torch.equal(lse[:, msk], lse_ref[:, msk])


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hq,hk", [(4, 4), (8, 2)], ids=["mha", "gqa"])
def test_equivalence_varlen_backward(causal, hq, hk):
    """grad_proxies on aligned varlen: O, dQ, dK, dV bitwise the internal per-tensor varlen path's (deterministic dQ;
    the f16 dQ pre-scale from the dequantised V codes' norm lands on the same power of two here)."""
    q, k, v, msk, kw = _docs_layout([300, 700, 128, 1000], hq, hk, 128)
    g = torch.randn(q.shape, device="cuda", dtype=BF16, generator=torch.Generator(device="cuda").manual_seed(1))
    g = g * msk.view(-1, 1, 1)
    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    o_ref = flash_attn_fp8(qr, kr, vr, causal=causal, qk_hadamard=False, options=PT, deterministic=True, **kw)
    o_ref.backward(g)
    (q8, k8, v8), ds = _our_codes_varlen(q, k, v, kw)
    qp, kp, vp = (t.clone().requires_grad_(True) for t in (q, k, v))
    o = flash_attn_fp8(q8, k8, v8, causal=causal, deterministic=True, grad_proxies=(qp, kp, vp), **_d(ds), **kw)
    o.backward(g)
    for nm, a, b in (("o", o, o_ref), ("dq", qp.grad, qr.grad), ("dk", kp.grad, kr.grad), ("dv", vp.grad, vr.grad)):
        assert torch.equal(a[msk], b[msk]), nm


@gpu
@pytest.mark.parametrize("layout", ["batched", "varlen"])
def test_fp32_descales(layout):
    """FA3's float32 descales: bitwise where the scales are powers of two (exact in float32), and within the last bits
    of the score scale otherwise."""
    if layout == "batched":
        q, k, v = _batched(2, 1024, 8, 2, 128)
        kw, msk = {}, None
        mk = lambda pow2: _our_codes_batched(q, k, v, pow2)  # noqa: E731
    else:
        q, k, v, msk, kw = _docs_layout([300, 700, 1000], 8, 2, 128)
        mk = lambda pow2: _our_codes_varlen(q, k, v, kw, pow2)  # noqa: E731
    sel = (lambda t: t) if msk is None else (lambda t: t[msk])  # noqa: E731
    ref2 = flash_attn_fp8(q, k, v, causal=True, qk_hadamard=False,
                          options=O8(operand_scaling="per_tensor", per_tensor_scale="power_of_two"), **kw)
    (q8, k8, v8), ds = mk(True)
    out2 = flash_attn_fp8(q8, k8, v8, causal=True, **_d([x.float() for x in ds]), **kw)
    assert torch.equal(sel(out2), sel(ref2))
    (q8, k8, v8), ds = mk(False)
    ref = flash_attn_fp8(q, k, v, causal=True, qk_hadamard=False, options=PT, **kw)
    out = flash_attn_fp8(q8, k8, v8, causal=True, **_d([x.float() for x in ds]), **kw)
    assert _cos(sel(out), sel(ref)) > 0.99999


def _same(a, b, causal, what):
    """O / LSE bitwise; gradients bitwise on dense calls. Causal: the batched kernel walks its tiles in another order
    than the documents kernel (LPT longest-key-tile-first, and the m-major walk of deterministic causal GQA; doc_varlen
    keeps the plain order), so dK / dV accumulate the same products in another order: the same math to fp32 rounding."""
    for nm, x, y in zip(("o", "lse", "dq", "dk", "dv"), a, b):
        if nm in ("o", "lse") or not causal:
            assert torch.equal(x, y), (what, nm)
        else:
            assert _cos(x, y) > 0.99999 and (x.float() - y.float()).abs().max() <= 1e-2 * y.float().abs().max(), \
                (what, nm, _cos(x, y))


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hq,hk,d", [(8, 2, 128), (4, 4, 64)], ids=["gqa", "d64"])
def test_batched_grad(causal, hq, hk, d):
    """The batched training path (batched forward, batched dO launch, the 1-CTA kernel with per-(row, kv head) tables):
    O / LSE bitwise the inference forward; vs (a) the documents form of the same call and (b) the INTERNAL per-tensor
    aligned varlen path on the same rows as equal-length documents (the same granularity: one scale per (row, kv head)
    of Q / K / V / dO): O, LSE bitwise, dQ, dK, dV bitwise on dense calls (deterministic), see _same for causal; and as
    accurate vs fp32 as the internal batched path."""
    from flash_attn.cute.lowprec import prequant_fa3 as PF
    b, s = 2, 1024
    q, k, v = _batched(b, s, hq, hk, d)
    g = torch.randn(q.shape, device="cuda", dtype=BF16, generator=torch.Generator(device="cuda").manual_seed(1))
    # our per-(row, kv head) quantiser: the internal per-tensor varlen path's codes and scales, the rows as documents
    ar = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    kw = dict(cu_seqlens_q=ar * s, cu_seqlens_k=ar * s, seqused_q=torch.full((b,), s, dtype=torch.int32, device="cuda"),
              seqused_k=torch.full((b,), s, dtype=torch.int32, device="cuda"), max_seqlen_q=s, max_seqlen_k=s)
    flat = lambda t: t.reshape(b * s, t.shape[2], t.shape[3])  # noqa: E731
    codes, ds = _our_codes_varlen(flat(q), flat(k), flat(v), kw)
    q8, k8, v8 = (c.view(b, s, c.shape[1], c.shape[2]) for c in codes)
    o_inf, lse_inf = flash_attn_fp8(q8, k8, v8, causal=causal, return_lse=True, **_d(ds))

    def run_pq():
        qp, kp, vp = (t.clone().requires_grad_(True) for t in (q, k, v))
        o, lse = flash_attn_fp8(q8, k8, v8, causal=causal, grad_proxies=(qp, kp, vp), return_lse=True,
                                deterministic=True, **_d(ds))
        o.backward(g)
        return o.detach(), lse, qp.grad, kp.grad, vp.grad
    a = run_pq()
    assert torch.equal(a[0], o_inf) and torch.equal(a[1], lse_inf)
    PF.BATCHED_AS_DOCUMENTS = True
    try:
        c = run_pq()
    finally:
        PF.BATCHED_AS_DOCUMENTS = False
    _same(a, c, causal, "documents form")
    # the internal per-tensor aligned varlen path on the rows as documents (its own quantisers)
    fl = lambda t: t.reshape(b * s, t.shape[2], t.shape[3]).clone().requires_grad_(True)  # noqa: E731
    qr, kr, vr = fl(q), fl(k), fl(v)
    o_r, lse_r = flash_attn_fp8(qr, kr, vr, causal=causal, qk_hadamard=False, options=PT, deterministic=True,
                                return_lse=True, **kw)
    o_r.backward(g.reshape(b * s, hq, d))
    ref = (o_r.detach().view(b, s, hq, d), lse_r.view(hq, b, s).permute(1, 0, 2), qr.grad.view(q.shape),
           kr.grad.view(k.shape), vr.grad.view(v.shape))
    _same(a, ref, causal, "internal per-tensor varlen")
    # accuracy vs fp32, against the internal batched per-tensor path (per-tensor dO)
    qb, kb, vb = (t.clone().requires_grad_(True) for t in (q, k, v))
    flash_attn_fp8(qb, kb, vb, causal=causal, qk_hadamard=False, options=PT).backward(g)
    qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
    of = torch.nn.functional.scaled_dot_product_attention(
        qf.transpose(1, 2), kf.repeat_interleave(hq // hk, 2).transpose(1, 2),
        vf.repeat_interleave(hq // hk, 2).transpose(1, 2), is_causal=causal).transpose(1, 2)
    of.backward(g.float())
    for nm, x, y, f in (("dq", a[2], qb.grad, qf.grad), ("dk", a[3], kb.grad, kf.grad), ("dv", a[4], vb.grad, vf.grad)):
        assert _cos(x, f) > 0.995 and _cos(x, f) > _cos(y, f) - 2e-4, (nm, _cos(x, f), _cos(y, f))


# ================================================================================================================
# GPU: no host sync, CUDA graphs
# ================================================================================================================
def _cases():
    q, k, v, msk, kw = _docs_layout([300, 700, 1000], 8, 2, 128)
    (q8, k8, v8), ds = _our_codes_varlen(q, k, v, kw)
    qb, kb, vb = _batched(2, 1024, 8, 2, 128)
    (qb8, kb8, vb8), dsb = _our_codes_batched(qb, kb, vb)
    return [((q8, k8, v8), (q, k, v), dict(**_d(ds), **kw)),
            ((qb8, kb8, vb8), (qb, kb, vb), _d([x.contiguous() for x in dsb]))]


@gpu
def test_no_host_sync():
    for codes, bf, kw in _cases():
        g = torch.ones_like(bf[0])
        prox = [t.clone().requires_grad_(True) for t in bf]

        def step():
            with torch.no_grad():
                flash_attn_fp8(*codes, causal=True, **kw)
            o = flash_attn_fp8(*codes, causal=True, grad_proxies=tuple(prox), **kw)
            o.backward(g)
        step()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            step()
            step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()


@gpu
def test_graph_capture():
    for codes, bf, kw in _cases():
        g = torch.randn_like(bf[0])
        ep = [t.clone().requires_grad_(True) for t in bf]
        eager_inf = flash_attn_fp8(*codes, causal=True, **kw)
        e = flash_attn_fp8(*codes, causal=True, grad_proxies=tuple(ep), deterministic=True, **kw)
        e.backward(g)
        eager_g = [p.grad.clone() for p in ep]
        e = e.detach()
        prox = [t.clone().requires_grad_(True) for t in bf]   # fresh leaves for the capture (as test_lowprec_api)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for p in prox:
                    p.grad = None
                flash_attn_fp8(*codes, causal=True, **kw)
                flash_attn_fp8(*codes, causal=True, grad_proxies=tuple(prox), deterministic=True, **kw).backward(g)
        torch.cuda.current_stream().wait_stream(s)
        for p in prox:
            p.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gi = flash_attn_fp8(*codes, causal=True, **kw)
            go = flash_attn_fp8(*codes, causal=True, grad_proxies=tuple(prox), deterministic=True, **kw)
            go.backward(g)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(gi, eager_inf) and torch.equal(go, e)
        for a, b in zip(prox, eager_g):
            assert torch.equal(a.grad, b)

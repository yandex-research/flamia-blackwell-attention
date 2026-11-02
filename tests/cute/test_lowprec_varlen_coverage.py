"""Varlen everywhere: the combinations the varlen-coverage step enabled, one compact gate each (docs/lowprec.md,
'Varlen coverage').

  * per-tensor operands and return_lse on the repack fallback == the aligned path on the zero-padded documents;
  * token-local operands on documents: exact causality (a future key changed: nothing before it moves);
  * q_block_2d on dense documents: no 16-query block spans two documents (other documents bitwise unchanged);
  * the 2-CTA backward ('on') deterministic and with total_q != total_k on aligned documents, packed at 256 and at 128
    (the key pair straddling documents), MHA and GQA;
  * the 2-CTA forward ('on') on documents: bitwise the 1-CTA forward;
  * pre-quantised E4M3 input with a window on documents, and on the repack fallback.

run: pytest tests/cute/test_lowprec_varlen_coverage.py -q
"""
import math

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions as O4, FP8AttnOptions as O8, flash_attn_fp4, flash_attn_fp8

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 45, 700, 1, 128]


def _packed(lens, h, hk, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    return q, k, v, do, cu


def _aligned(lens, xs, align=128):
    st = [0]
    for n in lens:
        st.append(st[-1] + max(-(-n // align) * align, align))
    out = []
    for x in xs:
        y = torch.zeros((st[-1],) + tuple(x.shape[1:]), dtype=x.dtype, device="cuda")
        o = 0
        for n, s0 in zip(lens, st[:-1]):
            y[s0:s0 + n] = x[o:o + n]
            o += n
        out.append(y)
    msk = torch.zeros(st[-1], dtype=torch.bool, device="cuda")
    for n, s0 in zip(lens, st[:-1]):
        msk[s0:s0 + n] = True
    kw = dict(cu_seqlens_q=torch.tensor(st, dtype=torch.int32, device="cuda"),
              cu_seqlens_k=torch.tensor(st, dtype=torch.int32, device="cuda"),
              seqused_q=torch.tensor(lens, dtype=torch.int32, device="cuda"),
              seqused_k=torch.tensor(lens, dtype=torch.int32, device="cuda"),
              max_seqlen_q=max(lens), max_seqlen_k=max(lens))
    return out, kw, msk


def _run(fn, q, k, v, do, **kw):
    x = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v))
    r = fn(*x, **kw)
    o, lse = r if isinstance(r, tuple) else (r, None)
    return [o, lse] + list(torch.autograd.grad(o, x, do))


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
def test_repack_per_tensor_and_lse(fn, O):
    q, k, v, do, cu = _packed(LENS, 8, 2)
    kw = dict(causal=True, deterministic=True, return_lse=True, options=O(operand_scaling="per_tensor"))
    r = _run(fn, q, k, v, do, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1000, max_seqlen_k=1000, **kw)
    (qa, ka, va, doa), akw, msk = _aligned(LENS, (q, k, v, do))
    a = _run(fn, qa, ka, va, doa, **kw, **akw)
    assert torch.equal(r[0], a[0][msk]) and torch.equal(r[1], a[1][:, msk])
    for x, y in zip(r[2:], a[2:]):
        assert torch.equal(x, y[msk])
    # return_lse on the default (token-block) repack fallback: the gathered LSE rows
    o, lse = fn(q, k, v, causal=True, return_lse=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1000,
                max_seqlen_k=1000)
    assert lse.shape == (8, sum(LENS)) and torch.isfinite(lse).all()


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
def test_token_local_documents_exactly_causal(fn, O):
    q, k, v, do, cu = _packed(LENS, 4, 4)
    kw = dict(causal=True, deterministic=True, qk_hadamard=False, cu_seqlens_q=cu, cu_seqlens_k=cu,
              max_seqlen_q=1000, max_seqlen_k=1000,
              options=O(operand_scaling="token_local", ds_dq_rowlocal="all", two_cta="off"))
    a = _run(fn, q, k, v, do, **kw)
    j = 300 + 517   # one future key of document 1
    k2, v2 = k.clone(), v.clone()
    k2[j], v2[j] = -k2[j] * 3 + 1, v2[j] * 5 - 2
    b = _run(fn, q, k2, v2, do, **kw)
    assert torch.equal(a[0][300:j], b[0][300:j]) and torch.equal(a[2][300:j], b[2][300:j])
    assert torch.equal(a[0][:300], b[0][:300]) and torch.equal(a[0][1300:], b[0][1300:])
    assert not torch.equal(a[0][j:1300], b[0][j:1300])


@gpu
def test_q_block_2d_dense_documents():
    q, k, v, do, cu = _packed(LENS, 4, 4)
    kw = dict(qk_hadamard=False, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1000, max_seqlen_k=1000,
              options=O4(q_block_2d=True))
    q2 = q.clone()
    q2[1290:1300] *= 7   # the last 16-query block of document 1 with real rows
    q2[1300] = -q2[1300] * 5
    with torch.no_grad():
        o1, o2 = flash_attn_fp4(q, k, v, **kw), flash_attn_fp4(q2, k, v, **kw)
    assert torch.equal(o1[:300], o2[:300]) and torch.equal(o1[1345:], o2[1345:])
    assert not torch.equal(o1[300:1345], o2[300:1345])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
def test_two_cta_backward_documents(fn, O):
    q, k, v, do, _ = _packed(LENS, 4, 4)
    (qa, ka, va, doa), akw, msk = _aligned(LENS, (q, k, v, do), align=256)
    kw = dict(causal=True, deterministic=True, **akw)
    a1 = _run(fn, qa, ka, va, doa, options=O(two_cta="on"), **kw)
    a2 = _run(fn, qa, ka, va, doa, options=O(two_cta="on"), **kw)
    c = _run(fn, qa, ka, va, doa, options=O(two_cta="off"), **kw)
    for x, y in zip(a1[2:], a2[2:]):
        assert torch.equal(x[msk], y[msk])            # deterministic
    assert torch.equal(a1[0][msk], c[0][msk])
    for x, y in zip(a1[3:], c[3:]):
        assert torch.equal(x[msk], y[msk])            # dK / dV bitwise the 1-CTA kernel's
    # total_q != total_k (bottom-right causal per document)
    lk = [700, 1000, 300, 900, 64, 128]
    _, k3, v3, _, _ = _packed(lk, 4, 4, seed=1)
    (ka3, va3), kkw, mk = _aligned(lk, (k3, v3), align=256)
    xkw = dict(akw, cu_seqlens_k=kkw["cu_seqlens_k"], seqused_k=kkw["seqused_k"], max_seqlen_k=1000)
    x2 = _run(fn, qa, ka3, va3, doa, causal=True, options=O(two_cta="on"), **xkw)
    x1 = _run(fn, qa, ka3, va3, doa, causal=True, options=O(two_cta="off"), **xkw)
    assert torch.equal(x2[0][msk], x1[0][msk]) and torch.equal(x2[3][mk], x1[3][mk]) and torch.equal(x2[4][mk], x1[4][mk])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
def test_two_cta_backward_documents_128(fn, O, causal, hk):
    """Documents packed at 128 (the caller's contract): the 2-CTA key pair straddles documents (doc_t128). MHA: dK / dV
    bitwise the 1-CTA kernel's; GQA: dK / dV within rounding of it; deterministic: bitwise run to run."""
    lens = [300, 1000, 45, 700, 1, 128, 257]
    q, k, v, do, _ = _packed(lens, 4, hk)
    (qa, ka, va, doa), akw, msk = _aligned(lens, (q, k, v, do), align=128)
    for det in (False, True):
        kw = dict(causal=causal, deterministic=det, **akw)
        a1 = _run(fn, qa, ka, va, doa, options=O(two_cta="on"), **kw)
        c = _run(fn, qa, ka, va, doa, options=O(two_cta="off"), **kw)
        assert torch.equal(a1[0][msk], c[0][msk])
        for x, y in zip(a1[2:], c[2:]):
            x, y = x[msk].float(), y[msk].float()
            assert x.isfinite().all() and (x * y).sum() / (x.norm() * y.norm()) > 0.9999
        if hk == 4:
            for x, y in zip(a1[3:], c[3:]):
                assert torch.equal(x[msk], y[msk])
        if det:
            a2 = _run(fn, qa, ka, va, doa, options=O(two_cta="on"), **kw)
            for x, y in zip(a1[2:], a2[2:]):
                assert torch.equal(x[msk], y[msk])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_two_cta_forward_documents(fn, O, causal):
    q, k, v, do, cu = _packed(LENS, 4, 4)
    kw = dict(causal=causal, return_lse=True, deterministic=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
              max_seqlen_q=1000, max_seqlen_k=1000)
    a = _run(fn, q, k, v, do, options=O(forward_two_cta="on", two_cta="off"), **kw)
    b = _run(fn, q, k, v, do, options=O(forward_two_cta="off", two_cta="off"), **kw)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_prequant_window_and_repacked(causal):
    from flash_attn.cute.lp_varlen import LPDocs
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    q, k, v, do, cu = _packed(LENS, 4, 4)
    (qa, ka, va, doa), akw, msk = _aligned(LENS, (q, k, v, do))
    docs = LPDocs.aligned_from(akw["cu_seqlens_q"], akw["cu_seqlens_k"], akw["seqused_q"], akw["seqused_k"],
                               qa.shape[0], ka.shape[0])
    r = quantize_fp8_per_tensor([qa.unsqueeze(0), ka.unsqueeze(0), va.unsqueeze(0)],
                                documents=dict(docs.doc_quant_args(("q", "k", "k"), 4)))
    c8 = [c[0] for c in r["codes"]]
    ds = dict(zip(("q_descale", "k_descale", "v_descale"), (r["doc_scales"][i][:len(LENS)] for i in range(3))))
    win = dict(window_size=(200, 0) if causal else (150, 150))
    ref = flash_attn_fp8(qa, ka, va, causal=causal, qk_hadamard=False, options=O8(operand_scaling="per_tensor"),
                         **win, **akw)
    out = flash_attn_fp8(*c8, causal=causal, **ds, **win, **akw)
    assert torch.equal(out[msk], ref[msk])
    # the repack fallback: the same codes packed back to back
    rep = flash_attn_fp8(*(x[msk] for x in c8), causal=causal, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1000,
                         max_seqlen_k=1000, **ds)
    assert torch.equal(rep, flash_attn_fp8(*c8, causal=causal, **ds, **akw)[msk])

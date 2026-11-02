"""The 2-CTA backward everywhere it is feasible, and the long-sequence default (cta2, owner decision 2026-10-07).

  * CPU: two_cta="auto" takes the 2-CTA backward at key length >= 32768 (s_k batched, max_seqlen_k on varlen) on every
    supported route -- GQA, deterministic, aligned / repacked documents, s_q != s_k, s_k % 256 == 128 -- and keeps the
    earlier rules below it;
  * GPU: deterministic GQA on the pair (dK / dV ordered by q head: bitwise run to run), batched s_q != s_k and
    s_k % 256 == 128 tails (MHA dK / dV bitwise the 1-CTA kernel's).

run: pytest tests/cute/test_lowprec_2cta_long.py -q
"""
import pytest
import torch

from flash_attn.cute.lowprec import (FP4AttnOptions as O4, FP8AttnOptions as O8, KernelTuning, flash_attn_fp4,
                                     flash_attn_fp8, resolve_fp4, resolve_fp8)

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LONG = 32768


def M(b, s, h, d=128, rg=True):
    return torch.empty(b, s, h, d, dtype=BF16, device="meta").requires_grad_(rg)


def _i32(n):
    return torch.empty(n, dtype=torch.int32, device="meta")


@pytest.mark.parametrize("res", [resolve_fp4, resolve_fp8], ids=["fp4", "fp8"])
def test_long_default_resolves_2cta(res):
    for causal in (True, False):
        for det in (False, True):
            for h, hk in ((16, 16), (16, 4), (32, 8)):
                p = res(M(1, LONG, h), M(1, LONG, hk), M(1, LONG, hk), causal=causal, deterministic=det)
                # deterministic keeps the 1-CTA kernel (the 2-CTA ordered dQ reduce measured no faster)
                assert p.bwd.kernel == ("1cta" if det else "2cta"), (causal, det, h, hk, p.log())
                assert p.specs.host.bwd_2cta_gqa == (h != hk and not det)
                assert any(f == "two_cta" and ("long-sequence rule" in why or "deterministic at key length" in why)
                           for f, _, why in p.resolution_log)
                if det:   # two_cta='on' takes the pair (GQA ordered by q head)
                    p = res(M(1, LONG, h), M(1, LONG, hk), M(1, LONG, hk), causal=causal, deterministic=det,
                            options=(O4 if res is resolve_fp4 else O8)(two_cta="on"))
                    assert p.bwd.kernel == "2cta" and p.specs.host.bwd_2cta_gqa == (h != hk)
        # s_q != s_k and the s_k % 256 == 128 tail
        p = res(M(1, 16384, 16), M(1, LONG + 128, 16), M(1, LONG + 128, 16), causal=causal)
        assert p.bwd.kernel == "2cta", p.log()
        # aligned varlen (documents packed at 128): the key length is max_seqlen_k
        q = torch.empty(LONG * 2, 16, 128, dtype=BF16, device="meta", requires_grad=True)
        al = dict(cu_seqlens_q=_i32(3), cu_seqlens_k=_i32(3), seqused_q=_i32(2), seqused_k=_i32(2),
                  max_seqlen_q=LONG, max_seqlen_k=LONG)
        assert res(q, q, q, causal=causal, **al).bwd.kernel == "2cta"
        assert res(q, q, q, causal=causal, **dict(al, max_seqlen_q=LONG - 128, max_seqlen_k=LONG - 128)).bwd.kernel \
            == "1cta"
        rp = dict(cu_seqlens_q=_i32(3), cu_seqlens_k=_i32(3), max_seqlen_q=LONG, max_seqlen_k=LONG)
        assert res(q, q, q, causal=causal, **rp).bwd.kernel == "2cta"


@pytest.mark.parametrize("res,O", [(resolve_fp4, O4), (resolve_fp8, O8)], ids=["fp4", "fp8"])
def test_below_long_keeps_the_earlier_rules(res, O):
    s = LONG // 2
    for h, hk in ((16, 4), (32, 8)):
        assert res(M(1, s, h), M(1, s, hk), M(1, s, hk), causal=True).bwd.kernel == "1cta"
    assert res(M(1, s, 16), M(1, s, 16), M(1, s, 16), causal=True, deterministic=True).bwd.kernel == "1cta"
    assert res(M(1, 1024, 16), M(1, 1152, 16), M(1, 1152, 16), causal=True).bwd.kernel == "1cta"
    # explicit: 'off' wins at any length, the tuning field's False keeps GQA off the pair
    assert res(M(1, LONG, 16), M(1, LONG, 16), M(1, LONG, 16), options=O(two_cta="off")).bwd.kernel == "1cta"
    p = res(M(1, LONG, 16), M(1, LONG, 4), M(1, LONG, 4), options=O(tuning=KernelTuning(bwd_2cta_gqa=False)))
    assert p.bwd.kernel == "1cta"
    # 'on' serves GQA and the batched tails without the tuning field
    assert res(M(1, 1024, 8), M(1, 1152, 2), M(1, 1152, 2), options=O(two_cta="on")).bwd.kernel == "2cta"


def _qkvg(b, sq, sk, h, hk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(b, sq, h, 128, device="cuda", generator=g, dtype=BF16),
            torch.randn(b, sk, hk, 128, device="cuda", generator=g, dtype=BF16),
            torch.randn(b, sk, hk, 128, device="cuda", generator=g, dtype=BF16),
            torch.randn(b, sq, h, 128, device="cuda", generator=g, dtype=BF16))


def _run(fn, q, k, v, do, **kw):
    x = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = fn(*x, **kw)
    return [o] + list(torch.autograd.grad(o, x, do))


def _cos(x, y):
    x, y = x.float().flatten(), y.float().flatten()
    return ((x @ y) / (x.norm() * y.norm())).item()


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_deterministic_gqa_pair(fn, O, causal):
    q, k, v, do = _qkvg(2, 2048, 2048, 8, 2)
    kw = dict(causal=causal, deterministic=True)
    a = [_run(fn, q, k, v, do, options=O(two_cta="on"), **kw) for _ in range(3)]
    c = _run(fn, q, k, v, do, options=O(two_cta="off"), **kw)
    for r in a[1:]:
        assert all(torch.equal(x, y) for x, y in zip(a[0], r))      # dK / dV ordered by q head
    assert torch.equal(a[0][0], c[0])
    for x, y in zip(a[0][1:], c[1:]):
        assert _cos(x, y) > 0.99999


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("sq,sk", [(1024, 1408), (1408, 640), (1152, 1152)])
def test_batched_tails(fn, O, sq, sk):
    q, k, v, do = _qkvg(2, sq, sk, 4, 4)
    for causal in (True, False):
        a = _run(fn, q, k, v, do, causal=causal, options=O(two_cta="on"))
        c = _run(fn, q, k, v, do, causal=causal, options=O(two_cta="off"))
        assert torch.equal(a[0], c[0]) and torch.equal(a[2], c[2]) and torch.equal(a[3], c[3])
        assert _cos(a[1], c[1]) > 0.99999


@pytest.mark.parametrize("res,O", [(resolve_fp4, O4), (resolve_fp8, O8)], ids=["fp4", "fp8"])
def test_documents_at_128_with_mods_stay_1cta(res, O):
    """score_mod / mask_mod / softcap on aligned documents (packed at 128): the 1-CTA kernel, auto and refused 'on'."""
    q = torch.empty(LONG * 2, 16, 128, dtype=BF16, device="meta", requires_grad=True)
    al = dict(cu_seqlens_q=_i32(3), cu_seqlens_k=_i32(3), seqused_q=_i32(2), seqused_k=_i32(2),
              max_seqlen_q=LONG, max_seqlen_k=LONG)
    assert res(q, q, q, causal=True, softcap=30.0, **al).bwd.kernel == "1cta"
    with pytest.raises(ValueError, match="documents packed at 128"):
        res(q, q, q, causal=True, softcap=30.0, options=O(two_cta="on"), **al)

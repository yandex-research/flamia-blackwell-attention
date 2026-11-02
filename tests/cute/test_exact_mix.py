"""The exact classifier: the clamped Mitchell classifier is the only one, and the per-16-key-group-position value-function mix
(`p_mix`: M Mitchell, E ex2.approx, 2 / 3 the clamped FMA-pipe exp2) is reproduced bit for bit by the backward that
rebuilds P-hat (the all-FP4 route's query-row backward). Option A has no backward (it is inference-only)."""

import pytest
import torch

from flash_attn.cute.flash_fwd_sm100_fp4 import parse_p_mix
from flash_attn.cute.lowprec import FP4AttnOptions as O4, flash_attn_fp4, resolve_fp4

BF16 = torch.bfloat16


def M(s=1024, h=4, rg=True):
    return torch.empty(1, s, h, 128, dtype=BF16, device="meta").requires_grad_(rg)


def test_parse_p_mix():
    assert parse_p_mix(None) is None
    assert parse_p_mix("me2m3mem") == tuple("ME2M3MEM")
    for bad in ("MEM", "MEMEMEMEM", "MEMEMEMX", ""):
        with pytest.raises(ValueError):
            parse_p_mix(bad)


def test_plan_exp_mix():
    q = M()
    p = resolve_fp4(q, q, q, pv_precision="fp4", backward="fp4", options=O4(exp_mix="emmemmem"))
    assert p.bwd.exp_mix == "EMMEMMEM"
    assert resolve_fp4(q, q, q, pv_precision="fp4", backward="fp4").bwd.exp_mix == "none"      # auto: training route
    qi = M(rg=False)
    assert resolve_fp4(qi, qi, qi, pv_precision="fp4").bwd.exp_mix == "EMMEMMEM"          # auto: option A inference
    assert resolve_fp4(qi, qi, qi, pv_precision="fp4", options=O4(exp_mix="none")).bwd.exp_mix == "none"
    with pytest.raises(ValueError):
        resolve_fp4(q, q, q, options=O4(exp_mix="EMMEMMEM"))          # E4M3 P-hat: nothing to mix
    with pytest.raises(ValueError):
        resolve_fp4(q, q, q, pv_precision="fp4", backward="fp4", options=O4(exp_mix="EMMEMMEX"))


def test_directp_knobs_refused():
    from flash_attn.cute.flash_fwd_sm100_fp4 import FlashAttentionForwardSm100
    import inspect
    assert "directp_clamp" not in inspect.signature(FlashAttentionForwardSm100.__init__).parameters
    assert "shiftless" not in inspect.signature(FlashAttentionForwardSm100.__init__).parameters


gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="SM100 kernels")


def _inputs(s=1024, h=2, hk=2):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, s, h, 128, generator=g).to(BF16).cuda()
    k = torch.randn(1, s, hk, 128, generator=g).to(BF16).cuda()
    v = torch.randn(1, s, hk, 128, generator=g).to(BF16).cuda()
    do = torch.randn(1, s, h, 128, generator=torch.Generator().manual_seed(7)).to(BF16).cuda()
    return q, k, v, do


def _fp4all(q, k, v, do, causal, mix, dbg=True):
    """The public all-4-bit step (flash_attn_fp4(pv_precision="fp4", backward="fp4"); options.exp_mix = mix) with the
    route's private debug hook lowprec._routes._ALL4_DBG set: the forward's E2M1 codes / SF bytes and the backward's
    reproduced codes land in the two buffers."""
    from flash_attn.cute.lowprec import _routes
    b, s, h, _ = q.shape
    s_k = k.shape[1]
    if dbg:
        _routes._ALL4_DBG = dict(fwd=torch.zeros(b, h, s, s_k + 9 * (s_k // 16), dtype=torch.uint8, device="cuda"),
                                 bwd=torch.zeros(b, h, s, s_k, dtype=torch.uint8, device="cuda"))
    try:
        qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
        o = flash_attn_fp4(qg, kg, vg, causal=causal, pv_precision="fp4", backward="fp4",
                           options=O4(exp_mix=mix) if mix is not None else None)
        grads = torch.autograd.grad(o, (qg, kg, vg), do)
        torch.cuda.synchronize()
        d = _routes._ALL4_DBG
        codes = (d["fwd"][..., :s_k].clone(), d["bwd"].clone()) if dbg else None
    finally:
        _routes._ALL4_DBG = None
    return o, grads, codes


@gpu
@pytest.mark.parametrize("causal,h,hk", [(True, 2, 2), (False, 8, 2)])
def test_fp4all_mix_codes_bitwise(causal, h, hk):
    q, k, v, do = _inputs(h=h, hk=hk)
    _o, _g, (cf, cb) = _fp4all(q, k, v, do, causal, "ME2M3MEM")
    assert torch.equal(cf, cb), int((cf != cb).sum())
    assert int((cf > 0).sum()) > 0


@gpu
def test_fp4all_all_mitchell_mix_is_the_default():
    q, k, v, do = _inputs()
    from flash_attn.cute import interface_bwd_fp4q as iq
    iq.reset_sr_counter()   # the same SR seed sequence for both steps
    o0, _g0, c0 = _fp4all(q, k, v, do, True, None)
    iq.reset_sr_counter()
    o1, _g1, c1 = _fp4all(q, k, v, do, True, "MMMMMMMM")
    assert torch.equal(o0, o1) and torch.equal(c0[0], c1[0]) and torch.equal(c0[1], c1[1])

"""The 2-CTA dk8 backward (flash_bwd_sm100_2cta8, nvfp4_s) parks the next tile's SFK / SFQ in the dP bank's WG-1 stage-0
block [288, 320) of BOTH CTAs (tcgen05.cp cta_group::2, issued by the leader's MMA warp). The MMA warp must therefore wait
until the compute warps of both CTAs have loaded stage 0 of dP(t) (dP_loaded_pair_mbar_ptr, a cluster-scope count-16
barrier on the leader); waiting on the leader's own count-8 barrier let a slow WG 1 of the peer CTA read SF bytes as dP for
its query block 1 (dK / dQ wrong on the odd 128-key tiles, the dense option A row identity off by 3-20 %; since fixed).

The race is timing-dependent, so these tests provoke it deterministically: the kernel's private constructor argument
`_test_skew_peer=True` (compile-time; keyed by the driver only when set) delays the peer CTA's WG 1 by ~20k cycles before
it loads dP(t). The arithmetic is unchanged by the delay, so every output must be bitwise the unskewed call's (dQ: within
its own atomics spread). (Option A is inference-only, so this test has no option A rows.)
"""
import contextlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM100 only")


def _inputs(b, s, h, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v, do = (torch.randn(b, s, h, 128, generator=g).to(torch.bfloat16).cuda() for _ in range(4))
    return q, k, v, do


class _Count2Cta:
    """Counts constructions of the 2-CTA backward kernel object (one per 2-CTA driver call)."""

    def __enter__(self):
        from flash_attn.cute.flash_bwd_sm100_2cta8 import FlashAttentionBackwardSm100TwoCta8 as K2
        self.K2, self.orig, self.n = K2, K2.__init__, 0

        def init(obj, *a, **kw):
            self.n += 1
            self.orig(obj, *a, **kw)
        K2.__init__ = init
        return self

    def __exit__(self, *exc):
        self.K2.__init__ = self.orig


@contextlib.contextmanager
def _skew(on: bool):
    """Construct every 2-CTA backward kernel with `_test_skew_peer=on` (the test-only hook)."""
    from flash_attn.cute.flash_bwd_sm100_2cta8 import FlashAttentionBackwardSm100TwoCta8 as K2
    orig = K2.__init__

    def init(obj, *a, **kw):
        orig(obj, *a, **dict(kw, _test_skew_peer=on))
    K2.__init__ = init
    try:
        yield
    finally:
        K2.__init__ = orig


def _fp4_default(q, k, v, do, causal):
    """The public 4-bit step at its defaults (E4M3 P.V forward, the dk8 backward: 2-CTA where the plan makes it so)."""
    from flash_attn.cute.lowprec import flash_attn_fp4
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    o = flash_attn_fp4(qg, kg, vg, causal=causal)
    return (o,) + torch.autograd.grad(o, (qg, kg, vg), do)


@pytest.mark.parametrize("causal", [False, True])
def test_2cta_sf_park_waits_for_peer(causal):
    fn = _fp4_default
    q, k, v, do = _inputs(1, 1024, 2)
    with _Count2Cta() as c, _skew(False):
        o0, dq0, dk0, dv0 = fn(q, k, v, do, causal)
        torch.cuda.synchronize()
    assert c.n >= 1, "the 2-CTA dk8 backward did not run (the test needs it)"
    for _ in range(2):
        with _skew(True):
            o1, dq1, dk1, dv1 = fn(q, k, v, do, causal)
            torch.cuda.synchronize()
        assert torch.equal(o0, o1)
        assert torch.equal(dv0, dv1)
        assert torch.equal(dk0, dk1), (dk0.float() - dk1.float()).abs().max().item()
        assert (dq0.float() - dq1.float()).abs().max().item() <= 1e-2 * dq0.float().abs().max().item()

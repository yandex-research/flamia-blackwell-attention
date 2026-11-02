"""`blk-backward-mutates-caller-D`: the block-scaled backward drivers must not write into the operands a caller hands them.

The dS fold scales D in place; with caller-supplied fp8_scales["mx"] the driver used to fold the CALLER's
mx["do"]["D"], so a second backward on the same operands read the folded D (A1: dQ cos 0.9906 -> 0.910, norm ratio
0.997 -> 1.134). Each case runs the backward twice on the same operands and checks (a) the two results are bitwise
equal and (b) every caller tensor is bitwise unchanged. Routes: dk8 (1-CTA / 2-CTA) and FP8 (1-CTA / 2-CTA), token
blocks (blk32x128) and 32 x 32 (blk32, 1-CTA only). The drivers run with the specs the plan builds for the same call
(flash_attn_fp4 / flash_attn_fp8 with options.operand_scaling and options.two_cta; resolve_*(...).specs)."""
import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, FP8AttnOptions, resolve_fp4, resolve_fp8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM100 GPU required")

SCALING = {"blk32x128": "token_block", "blk32": "block_32x32"}   # the spec value -> options.operand_scaling


def _specs(route, scaling, two_cta, causal, deterministic, q):
    """The call's specs: the 4-bit (E4M3 P.V, dk8 backward) or the 8-bit route at this operand scaling and 2-CTA choice.
    qk_hadamard=False: the operands below are unrotated, as the earlier default handed them."""
    m = torch.empty(q.shape, dtype=q.dtype, device="meta").requires_grad_(True)
    kw = dict(causal=causal, deterministic=deterministic, qk_hadamard=False)
    if route == "fp4":
        plan = resolve_fp4(m, m, m, options=FP4AttnOptions(operand_scaling=SCALING[scaling], two_cta=two_cta), **kw)
    else:
        plan = resolve_fp8(m, m, m, options=FP8AttnOptions(operand_scaling=SCALING[scaling], two_cta=two_cta), **kw)
    assert plan.bwd.kernel == ("2cta" if two_cta == "on" else "1cta"), plan.bwd.kernel
    return plan.specs


def _tensors(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _tensors(v, out)
    return out


@pytest.mark.parametrize("route", ["fp4", "fp8"])
@pytest.mark.parametrize("two_cta", ["off", "on"])
@pytest.mark.parametrize("scaling", ["blk32x128", "blk32"])
@pytest.mark.parametrize("causal", [True, False])
def test_backward_twice_same_operands(route, two_cta, scaling, causal):
    if two_cta == "on" and scaling != "blk32x128":
        pytest.skip("the 2-CTA kernel has the token-block variant only")
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4, mx_operands
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v, do = (torch.randn(1, 1024, 2, 128, generator=g).to("cuda", torch.bfloat16) for _ in range(4))
    out, lse = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)[:2]
    det = two_cta == "off"  # the 2-CTA kernel is not deterministic (dQ reduce order, ~1e-3): dK / dV bitwise, dQ close
    sp = _specs(route, scaling, two_cta, causal, det, q)
    mx = mx_operands(q=q, k=k, v=v, dout=do, out=out, quant=sp.quant, host=sp.host)
    snap = [t.clone() for t in _tensors(mx, [])]
    g1 = _flash_attn_bwd_fp4(q, k, v, out, do, lse, causal=causal, deterministic=det, fp8_scales={"mx": mx}, specs=sp)
    torch.cuda.synchronize()
    for a, b in zip(_tensors(mx, []), snap):
        assert torch.equal(a, b), "the backward wrote into a caller operand"
    g2 = _flash_attn_bwd_fp4(q, k, v, out, do, lse, causal=causal, deterministic=det, fp8_scales={"mx": mx}, specs=sp)
    for i, (a, b) in enumerate(zip(g1, g2)):
        if det or i > 0:
            assert torch.equal(a, b), f"grad {i} differs on the second call"
        else:
            rel = (a.float() - b.float()).norm() / b.float().norm()
            assert rel < 5e-3, f"dQ differs on the second call: rel {rel:.2e}"  # reduce-order floor ~1e-3; the bug: ~0.4

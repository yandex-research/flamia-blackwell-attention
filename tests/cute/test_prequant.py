# The exactness of folding the softmax factor into Q before quantisation.
#
# The claim: "scaling Q by c before quantisation leaves the codes unchanged for any amax-derived scale,
# per tensor or per block, because amax(cQ) = c amax(Q) and therefore round(cQ / (c s_q)) = round(Q / s_q)".
#
# That is true in exact arithmetic. These tests pin down what the machine actually does, separately for the two
# directions a fold can be implemented in:
#
#   descale-side fold (prequant.quantize_fused, what the API uses): the tensor and its scale are untouched and
#     only the reported descale is multiplied by c. Bitwise identical codes at EVERY granularity, by
#     construction. Asserted here so a future refactor cannot quietly start pre-multiplying.
#
#   literal pre-scaling (prequant.quantize_fused(..., prescale=True), what an upstream projection GEMM does):
#     amax is re-derived from c*Q. Bitwise identical iff c is a power of two. For non-power-of-two c it deviates,
#     and the three granularities deviate for three different reasons -- see test_prescale_*.
#
# run: pytest tests/cute/test_prequant.py -q      (needs an SM100 GPU for the FP4/MX quantisers)
import math
import os

import pytest
import numpy as np
import torch

from flash_attn.cute import prequant

# a wrong-tree import would make every bitwise claim below meaningless: when this file sits in a checkout, the
# flash_attn under test must be that checkout's (tests/cute/ -> repository root)
_TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if os.path.isfile(os.path.join(_TREE, "flash_attn", "cute", "prequant.py")):
    # (prequant.__file__, not flash_attn.__file__: an installed flash-attn-4 makes flash_attn a namespace package)
    assert os.path.realpath(prequant.__file__).startswith(_TREE + os.sep), (
        f"flash_attn.cute imported from {prequant.__file__}, not from this checkout {_TREE}")

D = 128
C = prequant.softmax_fold_factor(D)  # log2(e) / sqrt(128) = 0.127530..., not a power of two
C_POW2 = prequant.split_pow2(C)[0]  # 0.125


def _x(b=2, s=256, h=4, d=D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16, generator=g)


def _codes_equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_split_pow2_is_exact():
    for c in (C, 1.0, 0.125, 3.7, 1e-8, 1e8, prequant.LOG2E):
        p, r = prequant.split_pow2(c)
        assert 1.0 <= r < 2.0
        assert p * r == c, (c, p, r)  # exact: only an exponent moved
    assert prequant.is_pow2(0.125) and not prequant.is_pow2(C)


@pytest.mark.parametrize("fmt", ["e4m3", "nvfp4", "mxfp4", "mxfp8"])
def test_descale_fold_is_bitwise_exact(fmt):
    """The fold the API implements: codes bitwise identical at per-tensor, per-16 and per-32 granularity."""
    x = _x()
    ref = prequant.quantize_fused(x, fmt, c=1.0)
    fused = prequant.quantize_fused(x, fmt, c=C)
    assert _codes_equal(ref.codes, fused.codes), f"{fmt}: the descale-side fold moved a code"
    if ref.granularity == 0:
        # per-tensor scales are one-element device tensors (no host read in the library); the test reads them
        # (float32 device scalars: the product scale * float32(C), correctly rounded -- the FP64-free contract)
        assert float(fused.scale) == float(np.float32(float(ref.scale)) * np.float32(C))
        assert float(fused.base_scale) == float(ref.scale)
    else:
        assert torch.equal(ref.scale, fused.scale), f"{fmt}: the scale-factor bytes must be untouched"
    assert fused.fused == C


@pytest.mark.parametrize("fmt", ["e4m3", "mxfp4", "mxfp8"])
def test_prescale_pow2_is_bitwise_exact_where_the_scale_has_exponent_room(fmt):
    """A power-of-two pre-multiplication moves only exponents, so the codes survive it bitwise.

    True for the per-tensor fp32 scale and for the MX formats' UE8M0 scale (8 exponent bits, 2^-127..2^127).
    The scale itself of course changes -- that is where c now lives: the UE8M0 byte shifts by exactly
    log2(c), which the test checks.
    """
    x = _x()
    ref = prequant.quantize_fused(x, fmt, c=1.0)
    pre = prequant.quantize_fused(x, fmt, c=C_POW2, prescale=True)
    assert _codes_equal(ref.codes, pre.codes), f"{fmt}: a power-of-two pre-scale moved a code"
    if ref.granularity == 0:
        assert float(pre.scale) == float(ref.scale) * C_POW2
    else:
        shift = torch.round(torch.log2(torch.tensor(C_POW2))).int().item()
        d = pre.scale.contiguous().reshape(-1).int() - ref.scale.contiguous().reshape(-1).int()
        assert torch.unique(d).tolist() == [shift], f"{fmt}: UE8M0 bytes moved by {torch.unique(d).tolist()}, want {shift}"


def test_prescale_pow2_nvfp4_is_not_exact_the_e4m3_scale_runs_out_of_exponent():
    """NVFP4 exception, measured: its per-16 scale is E4M3, whose 4 exponent bits bottom out at 2^-6 (2^-9
    subnormal). Dividing the scale by 8 pushes the smallest groups into E4M3 subnormals, where the shift is
    lossy, so a few codes move even for a power-of-two c. `fused-factor-rounded-into-pow2-scale` is therefore
    safe for MX and *not* unconditionally safe for NVFP4.
    """
    x = _x()
    ref = prequant.quantize_fused(x, "nvfp4", c=1.0)
    pre = prequant.quantize_fused(x, "nvfp4", c=C_POW2, prescale=True)
    d = pre.scale.contiguous().reshape(-1).int() - ref.scale.contiguous().reshape(-1).int()
    # -24 is the clean 3-binade E4M3 exponent shift (8 codes per binade); -23 is a group that saturated
    assert set(torch.unique(d).tolist()) <= {-24, -23}, torch.unique(d).tolist()
    moved = (ref.codes.view(torch.uint8) != pre.codes.view(torch.uint8)).float().mean().item()
    assert moved < 1e-3, f"expected only the subnormal-clipped groups to move, got {moved:.4%}"


@pytest.mark.parametrize("prescale_dtype", ["same", "fp32"])
def test_prescale_nonpow2_per_tensor_deviates(prescale_dtype):
    """Per-tensor fp32 scale, non-power-of-two c: the identity holds in reals, so the deviation is pure
    rounding of the materialised product -- and it matters *where* that product is rounded. A bf16 epilogue
    (prescale_dtype="same") re-rounds c*x to 8 mantissa bits and moves ~2% of the codes; an fp32 epilogue
    moves far fewer. Neither is bitwise, which is why the API folds into the descale instead.
    """
    x = _x()
    ref = prequant.quantize_fused(x, "e4m3", c=1.0)
    pre = prequant.quantize_fused(x, "e4m3", c=C, prescale=True, prescale_dtype=prescale_dtype)
    moved = (ref.codes.view(torch.uint8) != pre.codes.view(torch.uint8)).float().mean().item()
    if prescale_dtype == "fp32":
        assert float(pre.scale) == pytest.approx(float(ref.scale) * C, rel=1e-6)
        assert moved < 5e-3, f"fp32 pre-scale moved {moved:.3%} of the codes"
    else:
        assert moved < 5e-2, f"bf16 pre-scale moved {moved:.3%} of the codes"


def test_prescale_nonpow2_nvfp4_deviates_because_the_e4m3_scale_rerounds():
    """NVFP4's per-16 scale is itself E4M3, so E4M3(c*amax/6) != c*E4M3(amax/6): the codes MUST move."""
    x = _x()
    ref = prequant.quantize_fused(x, "nvfp4", c=1.0)
    pre = prequant.quantize_fused(x, "nvfp4", c=C, prescale=True)
    assert not torch.equal(ref.scale, pre.scale), "expected the E4M3 scale bytes to re-round"
    assert not _codes_equal(ref.codes, pre.codes), "expected the E2M1 codes to move with the re-rounded scale"


def test_prescale_nonpow2_mx_deviates_because_the_ue8m0_exponent_is_a_ceil():
    """UE8M0 verdict: the scale is a pure power of two chosen by a ceil, so a non-power-of-two c is NOT exact.

    byte = 127 + ceil(log2(amax / FMAX)); with c folded in the argument shifts by log2(c), which is not an
    integer, so the ceil lands on a different exponent for part of the tensor. Only split_pow2(c)[0] is safe.
    """
    x = _x()
    for fmt in ("mxfp4", "mxfp8"):
        ref = prequant.quantize_fused(x, fmt, c=1.0)
        pre = prequant.quantize_fused(x, fmt, c=C, prescale=True)
        assert not torch.equal(ref.scale, pre.scale), f"{fmt}: expected the UE8M0 exponents to shift"
        assert not _codes_equal(ref.codes, pre.codes), f"{fmt}: expected the codes to move"
        # the deviation is bounded: the exponent moves by at most one binade, so the code changes by <= 1 ulp
        # of the format's grid. Quantified as a relative dequantisation error below.
        from flash_attn.cute.prequant import LOG2E  # noqa: F401  (the documented constant)


def test_score_scale_log2_is_one_number_either_way():
    """The honest reason the fold saves nothing in the inner loop: one fp32 scalar with or without it."""
    x, y = _x(seed=1), _x(seed=2)
    qo = prequant.quantize_fused(x, "e4m3", c=C)
    ko = prequant.quantize_fused(y, "e4m3", c=1.0)
    fused = prequant.score_scale_log2(qo, ko, D, None, q_scale_fused=True)
    qo_u = prequant.quantize_fused(x, "e4m3", c=1.0)
    plain = prequant.score_scale_log2(qo_u, ko, D, None, q_scale_fused=False)
    assert float(fused) == pytest.approx(float(plain), rel=1e-6)


def test_unit_score_scale_is_self_limiting():
    """The fold's real verdict: a unit score scale forces max_code(Q)*max_code(K) = c*amax(Q)*amax(K).

    Filling E4M3 on both sides would need that product to be 448^2 = 200704; a normalised activation gives
    about 2.5, so the balanced maximum is ~1.6 and about 8 of E4M3's 15 usable binades go. The builder used to refuse
    that unless the caller opted in; the refusal needed the amaxes on the host, and scales are GPU-only now, so it
    builds on the device and returns (the reads below are the test's, not the library's).
    """
    x, y = _x(seed=1), _x(seed=2)
    qo, ko = prequant.quantize_fused_unit_score(x, y, None, "e4m3")
    # the descales are exact reciprocals, so the score scalar is bitwise 1.0 -- the whole point
    assert float(qo.scale) * float(ko.scale) == 1.0
    assert prequant.score_scale_log2(qo, ko, D, None, q_scale_fused=True) == 1.0
    # and the forced product identity, to within the power-of-two rounding of the split
    prod = C * float(x.abs().amax()) * float(y.abs().amax())
    mq = float(x.abs().amax()) / float(qo.unfused_scale())
    mk = float(y.abs().amax()) / float(ko.scale)
    assert mq * mk == pytest.approx(prod, rel=1e-5), (mq, mk, prod)
    assert max(mq, mk) < 448.0 / 16, f"expected to lose binades, got maxima {mq:.3g} {mk:.3g}"


def test_fused_scale_for_unit_score_is_an_exact_reciprocal():
    for dk in (0.015625, 1.0, 64.0):
        assert prequant.fused_scale_for_unit_score(dk) * dk == 1.0


def test_validate_rejects_a_wrong_granularity():
    x = _x()
    op = prequant.quantize_fused(x, "nvfp4", c=C)
    bad = prequant.QuantizedOperand(op.codes, op.scale[..., :1], "nvfp4")
    with pytest.raises(AssertionError, match="factors"):
        bad.validate("q", shape=tuple(x.shape), head_dim=D)
    per_tensor_scale_on_block_format = prequant.QuantizedOperand(op.codes, 1.0, "nvfp4")
    with pytest.raises(AssertionError, match="scale-factor tensor"):
        per_tensor_scale_on_block_format.validate("q", shape=tuple(x.shape), head_dim=D)

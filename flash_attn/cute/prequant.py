# Pre-quantised Q / K operands with the softmax factor folded in.
#
# Two separable things live here.
#
# 1. `QuantizedOperand` -- a (codes, scale, format, granularity) carrier the attention entry points accept in
#    place of a bf16 tensor, so the step's quantiser pass does not run at all. That is the feature's real value:
#    it removes one HBM round trip per tensor per step (302 MB at b4 s4096 d128 h16, 604 MB at b1 s32768) and
#    the launches with it, and it is what lets a projection GEMM emit E4M3 / E2M1 directly.
#
# 2. The *fold*: the forward's score scale c = log2(e) * softmax_scale carried by the Q operand instead of by
#    the kernel. The algebra is that scaling Q by c before an amax-derived quantisation leaves the
#    codes unchanged and only multiplies the descale by c, because amax(cQ) = c amax(Q) and therefore
#    round(cQ / (c s_q)) = round(Q / s_q).
#
#    That identity holds in exact arithmetic. In the machine it holds *exactly* only for the fold this module
#    implements -- folding c into the reported **descale** and quantising the untouched tensor with the untouched
#    scale, which by construction cannot move a single code. It does NOT hold in general for the other
#    direction, literally pre-multiplying the tensor and re-deriving the scale from amax(cQ):
#
#      * per-tensor fp32 scale: exact iff c is a power of two. Otherwise fl(c*x) and fl(1/fl(c*s)) each round,
#        so the fp32 product differs in its last bits; the E4M3 code almost always survives the difference but
#        not provably (measured in tests/cute/test_prequant.py).
#      * NVFP4, E4M3 scale per 16: never exact for non-power-of-two c, because the scale itself is rounded --
#        E4M3(c * amax/6) != c * E4M3(amax/6).
#      * MX (UE8M0) scale per 32: never exact for non-power-of-two c, because the exponent is a ceil --
#        127 + ceil(log2(c*amax/FMAX)) != 127 + ceil(log2(amax/FMAX)) + log2(c).
#
#    So: a caller who wants the fold gets it for free and exactly through `fold_into_descale`. A caller whose
#    upstream GEMM has *already* multiplied Q by something (`prescaled=`) gets a bitwise-identical result only
#    for the power-of-two part, which `split_pow2` isolates; the residual is reported and stays in the kernel's
#    runtime scalar. Nothing here silently changes a code.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

LOG2E = 1.4426950408889634  # math.log2(math.e), the exp2 softmax's change of base
E4M3_MAX = 448.0
E5M2_MAX = 57344.0
E2M1_MAX = 6.0

# format -> (torch code dtype or None for packed E2M1, scale granularity along the contracted dim, code amax)
#   granularity 0 = one fp32 scalar for the whole tensor; 16 = NVFP4 (E4M3 scale factors); 32 = MX (UE8M0).
_FORMATS = {
    "e4m3": (torch.float8_e4m3fn, 0, E4M3_MAX),
    "e5m2": (torch.float8_e5m2, 0, E5M2_MAX),
    "nvfp4": (torch.uint8, 16, E2M1_MAX),
    "mxfp4": (torch.uint8, 32, E2M1_MAX),
    "mxfp8": (torch.float8_e4m3fn, 32, E4M3_MAX),
}


def format_info(fmt: str) -> Tuple[Optional[torch.dtype], int, float]:
    if fmt not in _FORMATS:
        raise ValueError(f"unknown quantised format {fmt!r}; known: {sorted(_FORMATS)}")
    return _FORMATS[fmt]


# ---------------------------------------------------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------------------------------------------------
def softmax_fold_factor(head_dim: int, softmax_scale: Optional[float] = None) -> float:
    """c = log2(e) * softmax_scale, the whole constant the exp2 forward applies to every score.

    softmax_scale defaults to 1/sqrt(head_dim), exactly as the entry points do. This is the factor
    `q_scale_fused=True` promises the Q operand already carries.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    return LOG2E * float(softmax_scale)


def split_pow2(c: float) -> Tuple[float, float]:
    """c = pow2 * residual with pow2 a power of two and residual in [1, 2).

    The power-of-two part can be folded into *any* of the quantised formats with bitwise-identical codes,
    including the UE8M0 and E4M3 block scales, because it only shifts an exponent. The residual cannot.
    """
    if not (c > 0.0 and math.isfinite(c)):
        raise ValueError(f"fold factor must be finite and positive, got {c}")
    m, e = math.frexp(float(c))  # c = m * 2**e, m in [0.5, 1)
    return math.ldexp(1.0, e - 1), m * 2.0


def is_pow2(c: float) -> bool:
    return c > 0.0 and math.isfinite(c) and split_pow2(c)[1] == 1.0


def is_scalar_scale(scale) -> bool:
    """A per-tensor descale held on the GPU: a floating CUDA tensor with one element (device_scales). Block scale
    factors are uint8 tensors with one byte per group, so the two never collide."""
    return isinstance(scale, torch.Tensor) and scale.is_floating_point() and scale.numel() == 1


def fold_into_descale(scale: Union[float, torch.Tensor], c: float) -> Union[float, torch.Tensor]:
    """The exact fold: multiply the descale by c and leave every code alone.

    For a per-tensor scale this is one host multiply. For a block-scale *tensor* (E4M3 per 16, UE8M0 per 32)
    the factor cannot be absorbed into the bytes without rounding, so it stays a separate scalar -- which is
    what the kernel's single runtime score scalar already is. Callers therefore keep the SF tensor untouched
    and carry c in `QuantizedOperand.fused`.
    """
    if is_scalar_scale(scale):
        return scale * float(c)  # a device descale: the float32 product (scale * float32(c)), on the GPU
    if isinstance(scale, torch.Tensor):
        raise TypeError("a block-scale tensor cannot absorb an arbitrary factor exactly; keep it in `fused`")
    return float(scale) * float(c)


# ---------------------------------------------------------------------------------------------------------------------
# the operand carrier
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantizedOperand:
    """A caller-supplied quantised Q or K.

    codes        the quantised tensor. (b, s, h, d) for FP8 formats; (b, s, h, d//2) uint8 for the E2M1 ones.
    scale        per-tensor formats: the *descale* (x ~= codes * scale), a Python float or a one-element float
                 CUDA tensor (kept on the GPU, never read back). Block-scaled formats: the swizzled scale-factor
                 tensor the SM100 MMA reads, in the layout the matching quantiser emits.
    fmt          one of prequant._FORMATS.
    fused        the constant already carried by these codes' descale (1.0 = none). For Q this is
                 `softmax_fold_factor(...)` when `q_scale_fused=True`.
    prescaled    True when the *tensor* was multiplied by `fused` before quantisation (a projection GEMM
                 epilogue) rather than `fused` being folded into the descale afterwards. Only the
                 power-of-two part of such a fold is bitwise-neutral; see the module docstring.
    """

    codes: torch.Tensor
    scale: Union[float, torch.Tensor]
    fmt: str
    fused: float = 1.0
    prescaled: bool = False
    # the descale before the fold, kept verbatim so a consumer that wants the unfolded number (the FP8
    # backward re-derives S from the same codes) never has to divide `scale` by `fused` and round.
    base_scale: Optional[Union[float, torch.Tensor]] = None
    # quantize_fused_unit_score's pair: descale_q * descale_k is exactly 1.0 by construction (reciprocal powers of
    # two), so the kernel can specialise the score scalar away without anyone reading the values
    unit_score: bool = False

    def unfused_scale(self) -> Union[float, torch.Tensor]:
        if self.base_scale is not None:
            return self.base_scale
        if is_scalar_scale(self.scale):
            return self.scale / float(self.fused)
        if isinstance(self.scale, torch.Tensor):
            return self.scale
        return float(self.scale) / float(self.fused)

    @property
    def granularity(self) -> int:
        return format_info(self.fmt)[1]

    def validate(self, name: str, *, shape: Tuple[int, ...], head_dim: int) -> None:
        """Aggressive validation: dtype, shape, and the scale's shape against the declared granularity."""
        code_dtype, gran, _ = format_info(self.fmt)
        b, s, h, d = shape
        assert d == head_dim, f"{name}: head_dim {head_dim} != shape {shape}"
        assert isinstance(self.codes, torch.Tensor) and self.codes.is_cuda, f"{name}.codes must be a CUDA tensor"
        packed = gran and code_dtype is torch.uint8
        want = (b, s, h, d // 2) if packed else (b, s, h, d)
        assert tuple(self.codes.shape) == want, f"{name}.codes shape {tuple(self.codes.shape)} != {want} for {self.fmt}"
        assert self.codes.dtype == code_dtype, f"{name}.codes dtype {self.codes.dtype} != {code_dtype} for {self.fmt}"
        assert self.codes.stride(-1) == 1, f"{name}.codes must be contiguous in the last dim"
        assert math.isfinite(self.fused) and self.fused > 0.0, f"{name}.fused must be finite positive, got {self.fused}"
        if gran == 0:
            if isinstance(self.scale, torch.Tensor):  # a device descale: validated by shape only (no host read)
                assert is_scalar_scale(self.scale) and self.scale.is_cuda, (
                    f"{name}: {self.fmt} is per-tensor, scale must be a float or a one-element float CUDA tensor"
                )
            else:
                assert math.isfinite(float(self.scale)) and float(self.scale) > 0.0, f"{name}.scale = {self.scale}"
        else:
            assert isinstance(self.scale, torch.Tensor), f"{name}: {self.fmt} needs a scale-factor tensor"
            assert self.scale.dtype == torch.uint8, f"{name}.scale dtype {self.scale.dtype} != uint8"
            assert self.scale.is_cuda, f"{name}.scale must be a CUDA tensor"
            # the swizzled SM100 layout is (32, 4, s/128, 4, d/(gran*4), h, b) -- one byte per `gran` elements
            n_sf = b * h * s * (d // gran)
            assert self.scale.dim() == 7 and self.scale.permute(6, 5, 2, 4, 0, 1, 3).is_contiguous(), (
                f"{name}.scale must be the (32, 4, s/128, 4, d/{gran * 4}, h, b) view of a contiguous (b, h, s/128, "
                f"d/{gran * 4}, 32, 4, 4) tensor, got shape {tuple(self.scale.shape)} strides {self.scale.stride()}; "
                f"a raw quantiser output needs .permute(4, 5, 2, 6, 3, 1, 0)"
            )
            assert self.scale.numel() == n_sf, (
                f"{name}.scale has {self.scale.numel()} factors, expected {n_sf} for {self.fmt} "
                f"(one per {gran} elements of a {shape} tensor). A wrong granularity here is silent, so it asserts."
            )


# ---------------------------------------------------------------------------------------------------------------------
# quantiser entries with the factor folded in
# ---------------------------------------------------------------------------------------------------------------------
def quantize_fused(
    x: torch.Tensor,
    fmt: str = "e4m3",
    c: float = 1.0,
    scale: Optional[float] = None,
    *,
    prescale: bool = False,
    prescale_dtype: str = "same",
) -> QuantizedOperand:
    """Quantise x for the attention kernels with the factor c folded into the descale.

    Returns codes that are **bitwise identical** to `quantize_fused(x, fmt, c=1.0)`'s, at every granularity,
    and a descale multiplied by c -- the exact fold. `prescale=True` instead materialises `c * x` and
    re-derives the scale from its amax, which is what an upstream GEMM epilogue does; that is bitwise
    identical only for a power-of-two c (see the module docstring) and is provided so the deviation can be
    measured rather than assumed. `prescale_dtype="same"` rounds `c * x` back to x's dtype, as a bf16 GEMM
    epilogue would; `"fp32"` keeps the product in fp32, as a GEMM that casts straight to the quantised format
    would. The two differ materially for a non-power-of-two c, and the tests report both.
    """
    format_info(fmt)
    c = float(c)
    if prescale:
        keep = prescale_dtype == "same" and x.dtype in (torch.bfloat16, torch.float16)
        xs = (x.float() * c).to(x.dtype) if keep else x.float() * c
        op = _quantize(xs, fmt, scale=None if scale is None else scale * c)
        if is_scalar_scale(op[1]):
            base = op[1] / c
        else:
            base = None if isinstance(op[1], torch.Tensor) else float(op[1]) / c
        return QuantizedOperand(op[0], op[1], fmt, fused=c, prescaled=True, base_scale=base)
    codes, s = _quantize(x, fmt, scale=scale)
    if isinstance(s, torch.Tensor) and not is_scalar_scale(s):
        return QuantizedOperand(codes, s, fmt, fused=c)
    return QuantizedOperand(codes, fold_into_descale(s, c), fmt, fused=c, base_scale=s)


def _quantize(x: torch.Tensor, fmt: str, scale: Optional[float] = None):
    """The plain, unfolded quantiser per format; reuses the tree's existing kernels verbatim."""
    if fmt in ("e4m3", "e5m2"):
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        from flash_attn.cute.device_scales import dev32

        r = quantize_fp8_per_tensor(x, scales=scale, fp8_format=fmt)
        return r["codes"], dev32(r["scales"][0], x.device)   # a float32 device scalar, given or computed
    if fmt in ("nvfp4", "mxfp4"):
        from flash_attn.cute.quant.fp4 import quantize_fp4_rows

        r = quantize_fp4_rows(x, scale_group_size=16 if fmt == "nvfp4" else 32)
        return r["codes"], r["scale_factors"]
    if fmt == "mxfp8":
        from flash_attn.cute.quant.fp8 import quantize_fp8_blocks

        # MXFP8 = E4M3 with a UE8M0 scale per 32 along d; the operand carries the (32, 4, s/128, 4, d/128, h, b) SF view
        # the kernels take, as the NVFP4 / MXFP4 quantisers return it
        r = quantize_fp8_blocks(x, block_rows=1, block_columns=32, output_exponent_table=False)
        return r["codes"], r["sf_tok"]
    raise AssertionError(fmt)


def quantize_qk_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    fmt: str = "e4m3",
    softmax_scale: Optional[float] = None,
    scales: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[QuantizedOperand, QuantizedOperand, float]:
    """(q_op with c folded in, k_op, c). c = log2(e) * softmax_scale, ready for `q_scale_fused=True`."""
    c = softmax_fold_factor(q.shape[-1], softmax_scale)
    sq, sk = scales if scales is not None else (None, None)
    return quantize_fused(q, fmt, c=c, scale=sq), quantize_fused(k, fmt, c=1.0, scale=sk), c


# ---------------------------------------------------------------------------------------------------------------------
# the score scalar the kernel is left with
# ---------------------------------------------------------------------------------------------------------------------
def score_scale_log2(
    q_op: Optional[QuantizedOperand],
    k_op: Optional[QuantizedOperand],
    head_dim: int,
    softmax_scale: Optional[float] = None,
    q_scale_fused: bool = False,
) -> float:
    """The single fp32 scalar the kernel multiplies every score by, i.e. `softmax_scale_log2`.

    Unfused: log2(e) * softmax_scale * descale_q * descale_k.
    Fused:   descale_q * descale_k, because descale_q already carries log2(e) * softmax_scale.

    The two are the same number to within fp32 rounding of the same product, which is the honest reason the
    fold saves nothing in the inner loop: there is exactly one scalar either way.
    """
    c = softmax_fold_factor(head_dim, softmax_scale)
    block_q = q_op is not None and q_op.granularity != 0
    # per-tensor descales are used as they are: a Python float, or a device scalar (the product then stays on the GPU)
    dq = 1.0 if (q_op is None or block_q) else q_op.scale
    dk = 1.0 if (k_op is None or k_op.granularity != 0) else k_op.scale
    if not q_scale_fused:
        return c * dq * dk
    if block_q:
        # A block-scaled operand has no per-tensor descale for the fold to hide in: the hardware applies the
        # per-16 / per-32 factors and the accumulator is already the dequantised Q.K. The only place c can live
        # is inside the scale-factor bytes, which means the tensor was genuinely pre-multiplied. Demand that
        # rather than silently dropping the factor.
        if not q_op.prescaled:
            raise ValueError(
                f"q_scale_fused=True with a block-scaled Q ({q_op.fmt}) needs the tensor to have been "
                f"pre-multiplied by c (quantize_fused(..., prescale=True)); a block-scale tensor cannot absorb "
                f"an arbitrary factor after the fact (a power-of-two block scale would round the fused factor)."
            )
        return 1.0
    return dq * dk


def quantize_fused_unit_score(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: Optional[float] = None,
    fmt: str = "e4m3",
    *,
    allow_range_loss: bool = False,
) -> Tuple[QuantizedOperand, QuantizedOperand]:
    """Per-tensor operands whose descales multiply to bitwise 1.0 -- the `unit-score-scale` construction.

    This is the only configuration in which folding buys a kernel-side instruction: with the score scalar a
    compile-time 1.0 the kernel drops every multiplication by it. It is built by putting K on a power-of-two
    descale (a standard FP8 delayed-scaling choice) and giving Q the exact reciprocal, so the product is 1.0
    in binary with no rounding at all, then balancing the two so neither operand is worse off than the other.

    **The construction is arithmetically self-limiting, and this is the headline result of the fold.** With a
    unit score scale the two operands' largest codes are forced to multiply to a constant:

        max_code(Q) * max_code(K) = c * amax(Q) * amax(K)

    because the descales must be reciprocals. Filling the E4M3 range on both sides would need
    c * amax(Q) * amax(K) = 448^2 = 200704; a normalised activation gives about 2.5, five orders of magnitude
    too *small*. So the balanced choice puts both maxima at sqrt(2.5) ~ 1.6 instead of 448, which throws away
    about 8 of E4M3's 15 usable binades at the bottom of the distribution -- the small values fall into E4M3
    subnormals. `allow_range_loss` used to refuse that by default; the refusal read the amaxes on the host, so the
    construction is now built on the GPU and always returned (the argument is accepted and ignored).

    There is no way around it by choosing the scales differently: the product is fixed by c and the two amaxes.
    """
    if format_info(fmt)[1] != 0:
        raise ValueError(f"unit-score-scale needs a per-tensor format, got {fmt}")
    # Built on the GPU (no host read of any amax), in float32 (torch float32 ops; no FP64).
    # The host version REFUSED a construction that loses more than one binade (allow_range_loss) and one whose Q would
    # overflow; both refusals need the amax on the host, so they are gone -- the quantiser clamps at the format's
    # maximum, and the range loss is the documented property of the construction (see above).
    from flash_attn.cute.device_scales import amax_abs
    c = softmax_fold_factor(q.shape[-1], softmax_scale)
    from flash_attn.cute.device_scales import div32
    amax_q = amax_abs(q).float().clamp_min(1e-30)
    amax_k = amax_abs(k).float().clamp_min(1e-30)
    # the forced product of the two operands' largest codes, and the balanced split of it: sqrt(c amax_q amax_k) as
    # sqrt(c amax_q) sqrt(amax_k), so no float32 intermediate overflows (the product of two amaxes could)
    target = torch.sqrt(c * amax_q) * torch.sqrt(amax_k)
    # K on the power of two nearest the balanced target, Q on its exact reciprocal
    dk = torch.exp2(torch.round(torch.log2(amax_k / target)))
    dq = 1.0 / dk  # exact: both are powers of two
    s_q = div32(dq, c)  # the Q codes' true per-tensor descale (one rounding)
    k_codes, _ = _quantize(k, fmt, scale=dk.reshape(1))
    k_op = QuantizedOperand(k_codes, dk.reshape(1), fmt, fused=1.0, base_scale=dk.reshape(1), unit_score=True)
    q_codes, _ = _quantize(q, fmt, scale=s_q.reshape(1))
    q_op = QuantizedOperand(q_codes, dq.reshape(1), fmt, fused=c, base_scale=s_q.reshape(1), unit_score=True)
    return q_op, k_op


def fused_scale_for_unit_score(k_descale: float) -> float:
    """The Q descale that makes the kernel's score scalar exactly 1.0, given K's descale.

    This is the only way the fold buys a kernel-side instruction: with `descale_q * descale_k == 1` the
    kernel can specialise `softmax_scale_log2` to the compile-time constant 1.0 and every multiplication by
    it folds away. It is reachable in practice -- the E4M3 headroom needed is
    c * amax(Q) * amax(K) <= 448^2 = 200704, five orders of magnitude above a normalised activation -- but the
    Q codes then sit on a *shifted* grid, so they are NOT the amax-derived ones and the accuracy must be
    measured, not assumed.
    """
    if is_scalar_scale(k_descale):
        return 1.0 / k_descale  # a device descale stays on the GPU
    return 1.0 / float(k_descale)

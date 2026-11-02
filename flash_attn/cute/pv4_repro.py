# `reproduce-forward-p` (option A, p_consistency="reproduce_forward_p"): the forward side of option A's P^ consistency
# (option A ships as inference only; its research backward is not part of this library). The all-FP4 forward exports, per query
# row, its final frame, its row sum and every frame change (REPRO_WORDS words; the inference contract fixed that kernel
# variant's O / LSE), and the forward / all-4-bit kernels share the exact-arithmetic helpers below (f32c, div_rn, fma_rn,
# the emulated exp2 pairs). The backward-side P-hat rebuild is not shipped.
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T as _T, dsl_user_op as _dsl_user_op
from cutlass._mlir.dialects import llvm as _llvm

REPRO_RECORDS = 5                 # = FlashAttentionForwardSm100FP4.REPRO_RECORDS
REPRO_WORDS = 3 + REPRO_RECORDS   # [F, 1 / Z bits, change count, records...] per query row
# "reproduce_forward_p" (the default): the forward variant with the P-hat export; "none": the plain forward (its training
# arm, exp(S - LSE) against the forward's O, diverged by step 1000 and is archived with the backward)
P_CONSISTENCY_MODES = ("reproduce_forward_p", "none")


def f32c(x: float) -> float:
    """x rounded to the nearest fp32 value, as a Python float (exactly representable: any conversion keeps it)."""
    import struct
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


@_dsl_user_op
def div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    out = _llvm.inline_asm(_T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                           "div.rn.f32 $0, $1, $2;", "=f,f,f",
                           has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
    return Float32(out)


@_dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    out = _llvm.inline_asm(_T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
                                      Float32(c).ir_value(loc=loc, ip=ip)],
                           "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                           has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
    return Float32(out)


@cute.jit
def ex2_emu_pair2(x: Float32, y: Float32, clamp: cutlass.Constexpr[bool]):
    """flash_fwd_sm100_fp4._ex2_emu_pair with the degree-2 polynomial: the same packed ops, element for element."""
    from flash_attn.cute import utils
    if const_expr(clamp):
        x = cute.arch.fmax(x, Float32(-126.0))
        y = cute.arch.fmax(y, Float32(-126.0))
    MAGIC = 12582912.0
    C0, C1, C2 = 1.0004426738244065, 0.7034358772399556, 0.23842222937876753
    t0, t1 = cute.arch.add_packed_f32x2((x, y), (Float32(MAGIC), Float32(MAGIC)))
    u0, u1 = cute.arch.add_packed_f32x2((t0, t1), (Float32(-MAGIC), Float32(-MAGIC)))
    f0, f1 = cute.arch.fma_packed_f32x2((u0, u1), (Float32(-1.0), Float32(-1.0)), (x, y))
    p0, p1 = cute.arch.fma_packed_f32x2((f0, f1), (Float32(C2), Float32(C2)), (Float32(C1), Float32(C1)))
    p0, p1 = cute.arch.fma_packed_f32x2((p0, p1), (f0, f1), (Float32(C0), Float32(C0)))
    return utils.combine_int_frac_ex2(t0, p0), utils.combine_int_frac_ex2(t1, p1)


@cute.jit
def ex2_emu_pair3(x: Float32, y: Float32):
    """flash_fwd_sm100_fp4._ex2_emu_pair(clamp=True, poly=3): the -126 clamp, then utils.ex2_emulation_2 (degree 3)."""
    from flash_attn.cute import utils
    x = cute.arch.fmax(x, Float32(-126.0))
    y = cute.arch.fmax(y, Float32(-126.0))
    return utils.ex2_emulation_2(x, y)


def p_consistency_mode(p_consistency) -> str:
    """Option A's P consistency (explicit, compile-time; FP4AttnOptions.p_consistency): None = "reproduce_forward_p"."""
    if p_consistency is None:
        return "reproduce_forward_p"
    if p_consistency not in P_CONSISTENCY_MODES:
        raise ValueError(f"p_consistency must be one of {P_CONSISTENCY_MODES}, got {p_consistency!r}")
    return p_consistency


SOFTMAX_EXP_MODES = ("mitchell", "exp2")


def softmax_exp_mode(softmax_exp) -> str:
    if softmax_exp is None:
        return "mitchell"
    if softmax_exp not in SOFTMAX_EXP_MODES:
        raise ValueError(f"softmax_exp must be one of {SOFTMAX_EXP_MODES}, got {softmax_exp!r}")
    return softmax_exp

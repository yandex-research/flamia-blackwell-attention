# The NVFP4 side-output encoders of the FP8 set (quantize_fp8_blocks' also_make_nvfp4_copy): option A's dO copy with the
# legacy zero-scale rule of nvfp4_2d_triton._q_groups ("dp_operand"), the S operand's rule ("qk_operand"), and the
# stochastic-rounding stream bitwise the old Triton quantiser (incremental splitmix64 states). Written as a local
# copy of the FP4 quantiser's helpers; the primitive conversions are now quant/fp4.py's own
# (packed_float_to_e2m1 / packed_float_to_ue4m3 / div_full / e4m3_byte_value: the same PTX), and the group encoders that
# carry these two encodings and the old SR stream stay here.
#
# The encoding is the arithmetic of nvfp4_2d_triton._q_groups (the Triton quantiser option A ran), op for op:
#   amax  = max |y| over the 16-group
#   sf8   = E4M3_rn(amax * k)        k = fp32(1/6) (round to nearest) | fp32(17/96) (stochastic: never clips)
#   inv   = sf8 > 0 ? div.full(1, sf8) : 0
#   code  = E2M1_rn(y * inv)                         (nearest: cvt.rn.satfinite.e2m1x2)
#         = E2M1 grid point chosen with u = splitmix64(idx * 16 + n, seed) >> 40 / 2^24   (stochastic)
# and the decoded value of a code nibble (sign bit 3; magnitudes 0 .5 1 1.5 2 3 4 6), exactly _e2m1_code_val.
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flash_attn.cute.quant.fp4 import (packed_float_to_e2m1, packed_float_to_ue4m3, e4m3_byte_value,  # noqa: F401
                                       div_full as div_full_f32)



@dsl_user_op
def floor_ftz_f32(a, *, loc=None, ip=None) -> Float32:
    """cvt.rmi.ftz.f32.f32: Triton's tl.floor."""
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "cvt.rmi.ftz.f32.f32 $0, $1;", "=f,f",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


_GOLDEN = 0x9E3779B97F4A7C15


@dsl_user_op
def splitmix_start(idx, seed, *, loc=None, ip=None) -> Int64:
    """z = idx * golden + seed (mod 2^64): the splitmix64 state of element idx; element idx + n is z + n * golden."""
    return Int64(llvm.inline_asm(
        T.i64(), [Int64(idx).ir_value(loc=loc, ip=ip), Int64(seed).ir_value(loc=loc, ip=ip)],
        "{\n\tmul.lo.u64 $0, $1, 0x9E3779B97F4A7C15;\n\tadd.u64 $0, $0, $2;\n\t}\n",
        "=l,l,l", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def splitmix_u01_at(z0, n: cutlass.Constexpr[int], *, loc=None, ip=None) -> Float32:
    """nvfp4_2d_triton._hash_u01 of the element n places after the one whose state is z0: z ^= z >>> 30; z *= c1;
    z ^= z >>> 27; z *= c2; (z >>> 40) / 2^24 (the masked arithmetic shifts of the Triton code are logical shifts; its
    last xor-shift cannot reach bits 40..63)."""
    step = (n * _GOLDEN) & 0xFFFFFFFFFFFFFFFF
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(z0).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .u64 z, t;\n\t.reg .u32 hi;\n\t"
        f"add.u64 z, $1, 0x{step:016X};\n\t"
        "shr.u64 t, z, 30;\n\txor.b64 z, z, t;\n\tmul.lo.u64 z, z, 0xBF58476D1CE4E5B9;\n\t"
        "shr.u64 t, z, 27;\n\txor.b64 z, z, t;\n\t"
        "mul.lo.u64 z, z, 0x94D049BB133111EB;\n\t"
        "shr.u64 t, z, 40;\n\tcvt.u32.u64 hi, t;\n\tcvt.rn.f32.u32 $0, hi;\n\tmul.rn.f32 $0, $0, 0f33800000;\n\t}\n",
        "=f,l", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def e2m1_value(word, k: cutlass.Constexpr[int], *, loc=None, ip=None) -> Float32:
    """Nibble k of a code word (element 2j: low nibble of byte j) as its fp32 value, branch-free: the magnitude's bf16
    bytes from two byte-permute tables, the sign from bit 3 (a negative-zero code gives -0.0, as _e2m1_code_val)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 c, m, hi, lo, sg;\n\t"
        f"shr.u32 c, $1, {4 * k};\n\tand.b32 m, c, 7;\n\t"
        "prmt.b32 hi, 0x3F3F3F00, 0x40404040, m;\n\tprmt.b32 lo, 0xC0800000, 0xC0804000, m;\n\t"
        "and.b32 hi, hi, 0xFF;\n\tand.b32 lo, lo, 0xFF;\n\tshl.b32 hi, hi, 24;\n\tshl.b32 lo, lo, 16;\n\t"
        "and.b32 sg, c, 8;\n\tshl.b32 sg, sg, 28;\n\tor.b32 hi, hi, lo;\n\tor.b32 $0, hi, sg;\n\t}\n",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def e2m1_stochastic_value(y, u, *, loc=None, ip=None) -> Float32:
    """nvfp4_2d_triton._e2m1_sr, branch-free: y onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (|y| clamped to 6), rounding
    up when u * step < |y| - lo (lo the grid point below, step the grid spacing there), the sign of y kept."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(y).ir_value(loc=loc, ip=ip), Float32(u).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 a, t, l2, l4, lo, st, d, us, hh, v, vn;\n\t.reg .pred p2, p4, q, n;\n\t"
        "abs.f32 a, $1;\n\tmin.f32 a, a, 0f40C00000;\n\t"
        "setp.lt.f32 p2, a, 0f40000000;\n\tsetp.lt.f32 p4, a, 0f40800000;\n\t"
        "mul.rn.f32 t, a, 0f40000000;\n\tcvt.rmi.ftz.f32.f32 t, t;\n\tmul.rn.f32 l2, t, 0f3F000000;\n\t"
        "cvt.rmi.ftz.f32.f32 l4, a;\n\t"
        "selp.f32 lo, l4, 0f40800000, p4;\n\tselp.f32 lo, l2, lo, p2;\n\t"
        "selp.f32 st, 0f3F800000, 0f40000000, p4;\n\tselp.f32 st, 0f3F000000, st, p2;\n\t"
        "sub.rn.f32 d, a, lo;\n\tmul.rn.f32 us, $2, st;\n\tsetp.lt.f32 q, us, d;\n\t"
        "add.rn.f32 hh, lo, st;\n\tselp.f32 v, hh, lo, q;\n\t"
        "setp.lt.f32 n, $1, 0f00000000;\n\tneg.f32 vn, v;\n\tselp.f32 $0, vn, v, n;\n\t}\n",
        "=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def e2m1_nibble_value(c: Int32) -> Float32:
    """nvfp4_quant_triton._e2m1_code_val."""
    m = c & 7
    mf = Float32(m)
    v = Float32(6.0)
    if m < 4:
        v = mf * Float32(0.5)
    elif m < 6:
        v = mf - Float32(2.0)
    elif m == 6:
        v = Float32(4.0)
    if (c & 8) != 0:
        v = -v
    return v


@dsl_user_op
def _f32_op(op: cutlass.Constexpr, *args, loc=None, ip=None) -> Float32:
    cons = "=f," + ",".join("f" for _ in args)
    body = f"{op} $0, " + ", ".join(f"${i + 1}" for i in range(len(args))) + ";"
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in args], body, cons,
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def nvfp4_group16(v: cute.Tensor, stochastic: cutlass.Constexpr, idx_group: Int64, seed: Int64,
                  encoding: cutlass.Constexpr = "dp_operand"):
    """NVFP4 of one 16-group v (fp32, already scaled) -> (E4M3 SF byte in the low byte of an Int32, the SF value, the two
    code words); v is overwritten. Two encodings:
      "dp_operand"  option A's dO copy, nvfp4_2d_triton._q_groups: SF = E4M3(amax * k) (k = 1/6, or 17/96 under
                    stochastic rounding), codes of v * (SF > 0 ? div.full(1, SF) : 0), round to nearest or stochastic
                    (u = splitmix(idx * 16 + n, seed));
      "qk_operand"  the S operand's Q / K codes, FlashInfer's rule (bitwise FlashInfer's nvfp4_quantize):
                    subnormal inputs count as signed zeros (abs.ftz / mul.ftz), SF = E4M3(amax / 6), multiplier
                    SF > 0 ? 1 / SF : (amax / 6 > 0 ? +inf : 0), round to nearest."""
    a = Float32(0.0)
    for j in cutlass.range_constexpr(16):
        if const_expr(encoding == "qk_operand"):
            a = cute.arch.fmax(a, _f32_op("abs.ftz.f32", v[j]))
        else:
            a = cute.arch.fmax(a, cute.math.absf(v[j]))
    sfu = a * Float32(17.0 / 96.0) if const_expr(stochastic) else a * Float32(1.0 / 6.0)
    sfw = packed_float_to_ue4m3(sfu, Float32(0.0), Float32(0.0), Float32(0.0))
    sfr = e4m3_byte_value(sfw, 0)
    inv = Float32(0.0)
    if const_expr(encoding == "qk_operand"):
        if sfu > Float32(0.0):
            inv = Float32(float("inf"))
    if sfr > Float32(0.0):
        inv = div_full_f32(Float32(1.0), sfr)
    for j in cutlass.range_constexpr(0, 16, 2):
        if const_expr(encoding == "qk_operand"):
            v[j], v[j + 1] = cute.arch.mul_packed_f32x2((v[j], v[j + 1]), (inv, inv), ftz=True)
        else:
            v[j] = v[j] * inv
            v[j + 1] = v[j + 1] * inv
    if const_expr(stochastic):
        z0 = splitmix_start(idx_group * Int64(16), seed)
        for j in cutlass.range_constexpr(16):
            v[j] = e2m1_stochastic_value(v[j], splitmix_u01_at(z0, j))
    w0 = packed_float_to_e2m1(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
    w1 = packed_float_to_e2m1(v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])
    return sfw & 0xFF, sfr, w0, w1


@cute.jit
def nvfp4_half_group8(v: cute.Tensor, stochastic: cutlass.Constexpr, idx_group: Int64, element_offset, seed: Int64,
                      encoding: cutlass.Constexpr = "dp_operand"):
    """nvfp4_group16 for a 16-group split over a lane pair (partner = lane ^ 1): this lane holds elements
    element_offset .. element_offset + 7 (0 or 8) in v[0:8] (fp32, already scaled). The group amax is the max of the
    two lanes' maxima (one shuffle; a max is exact, so the SF is bitwise nvfp4_group16's); every other step is the same
    per-element arithmetic, the stochastic stream starting at element idx_group * 16 + element_offset. Returns (SF byte,
    the code word of the 8 elements); v is overwritten. Both lanes of the pair must call it (the shuffle)."""
    a = Float32(0.0)
    for j in cutlass.range_constexpr(8):
        if const_expr(encoding == "qk_operand"):
            a = cute.arch.fmax(a, _f32_op("abs.ftz.f32", v[j]))
        else:
            a = cute.arch.fmax(a, cute.math.absf(v[j]))
    a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=1))
    sfu = a * Float32(17.0 / 96.0) if const_expr(stochastic) else a * Float32(1.0 / 6.0)
    sfw = packed_float_to_ue4m3(sfu, Float32(0.0), Float32(0.0), Float32(0.0))
    sfr = e4m3_byte_value(sfw, 0)
    inv = Float32(0.0)
    if const_expr(encoding == "qk_operand"):
        if sfu > Float32(0.0):
            inv = Float32(float("inf"))
    if sfr > Float32(0.0):
        inv = div_full_f32(Float32(1.0), sfr)
    for j in cutlass.range_constexpr(0, 8, 2):
        if const_expr(encoding == "qk_operand"):
            v[j], v[j + 1] = cute.arch.mul_packed_f32x2((v[j], v[j + 1]), (inv, inv), ftz=True)
        else:
            v[j] = v[j] * inv
            v[j + 1] = v[j + 1] * inv
    if const_expr(stochastic):
        z0 = splitmix_start(idx_group * Int64(16) + Int64(element_offset), seed)
        for j in cutlass.range_constexpr(8):
            v[j] = e2m1_stochastic_value(v[j], splitmix_u01_at(z0, j))
    w0 = packed_float_to_e2m1(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
    return sfw & 0xFF, w0


@cute.jit
def nvfp4_encode_row128(x: cute.Tensor, mult: cute.Tensor, words: cute.Tensor, stochastic: cutlass.Constexpr,
                        idx_row: Int64, seed: Int64, encoding: cutlass.Constexpr = "dp_operand"):
    """y = x * mult[g] (one fp32 multiplier per 16-group) -> words: 16 Int32 of E2M1 codes (element 2k low nibble of
    byte k); returns the two packed E4M3 SF words (groups 0-3, 4-7) and the 8 SF values. idx_row: the row's
    random-stream base (group g uses idx_row + g)."""
    sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
    sfw[0] = Int32(0)
    sfw[1] = Int32(0)
    sfv = cute.make_rmem_tensor(cute.make_layout(8), Float32)
    for g in cutlass.range_constexpr(8):
        v = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        for j in cutlass.range_constexpr(16):
            v[j] = x[16 * g + j] * mult[g]
        byte, sfr, w0, w1 = nvfp4_group16(v, stochastic, idx_row + Int64(g), seed, encoding)
        words[2 * g] = w0
        words[2 * g + 1] = w1
        sfw[g // 4] = sfw[g // 4] | (byte << (8 * (g % 4)))
        sfv[g] = sfr
    return sfw[0], sfw[1], sfv

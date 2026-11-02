# The FP4 quantisers of the low-precision attention routes (CuTe DSL, SM100): NVFP4 (E2M1 codes, an E4M3 scale per 16)
# and MXFP4 (E2M1 codes, a UE8M0 scale per 32), in the SF layouts the SM100 block-scaled MMAs read.
#
#   quantize_fp4_rows             one read of a (b, s, h, 128) tensor -> scale groups along d (the row copy) and / or along
#                                 the tokens (the transposed copy), plus every side product (E4M3 copy, D, norms, amax)
#   quantize_fp4_blocks_2d        the same read, one scale per 16 x 16 (NVFP4) or 32 x 32 (MXFP4) block, shared by the row
#                                 and the transposed copy (one set of codes serves both GEMMs)
#   quantize_fp4_causal_diagonal  (fp4_diagonal.py) the all-FP4 / option-A causal diagonal operands in ONE launch
#
# Every option is a compile-time constant: each combination is its own specialised kernel, and an option that is off
# costs nothing. One CTA per (128-token tile, head, batch), 128 threads, thread r owning token row r of the tile (the
# whole 128-wide row in registers), so every transform (RoPE, Hadamard, scales) and every per-row reduction is
# in-register arithmetic. The transposed copy and the 16-row / 32-row block reductions go through shared memory or warp
# shuffles. FlashInfer is not used anywhere.
#
# THE ENCODING (one definition for every quantiser and every copy; importable for the FP8 set's NVFP4 side outputs):
#   NVFP4 scale    SF = E4M3_rn_satfinite(amax * f32(1/6)) (rounding "stochastic": amax * f32(17/96), so the grid never
#                  clips; "ms_eden": amax * f32(0.93 * 17/96); "unit": SF = 1), amax over the group after subnormal inputs
#                  are flushed to zero.
#   MXFP4 scale    SF = 2^e, the smallest power of two with 6 * 2^e >= amax; byte e + 127; byte 0 for an all-zero group or
#                  when amax * f32(1/6) is subnormal (flushed).
#   codes          E2M1 = cvt.rn.satfinite.e2m1(x * m) (stochastic: the E2M1 grid point chosen with the counter hash),
#                  x flushed to zero when subnormal, m = 1 / SF for SF > 0; for NVFP4 m = +inf when the SF rounds to 0 but
#                  amax > 0 (nonzero x -> +-6, zero x -> +6), m = 0 for an all-zero group (codes keep the sign of zero).
#                  This is FlashInfer's rule, adopted by these quantisers; with SF = 0 every code decodes to 0, so it
#                  only fixes the code bytes.
#   Four-over-Six  NVFP4 + nearest only: per group also the grid scaled to max 4 (SF4 = E4M3(amax / 4), codes clamped to
#                  +-4); the candidate with the lower squared error wins (ties keep 6).
#   stochastic     uniform u = splitmix64(element_index * golden + seed) >> 40, / 2^24; the element index of a group is
#                  group_index * group_elements + position (see _sr_* below), seed & (2^62 - 1). Bitwise the
#                  nvfp4_2d_triton streams.
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda
from flash_attn.cute.quant._tcgen05 import RowRotation
from cutlass.cute.nvgpu import cpasync

try:
    from . import _rotations as rot
except ImportError:  # loaded by file path (benchmark leaf loading): the sibling by path, no package import
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_rotations.py")
    _spec = _ilu.spec_from_file_location("_quant_rotations_leaf", _p)
    rot = _ilu.module_from_spec(_spec)
    _sys.modules["_quant_rotations_leaf"] = rot
    _spec.loader.exec_module(rot)


# This module imports only torch and cutlass-dsl (leaf-importable): the two PTX conversions below are local copies of
# blackwell_helpers' packed_float_to_e2m1 / packed_float_to_ue4m3.
@dsl_user_op
def packed_float_to_e2m1(f0, f1, f2, f3, f4, f5, f6, f7, *, loc=None, ip=None) -> Int32:
    """8 fp32 -> 8 E2M1 codes packed in an Int32 (cvt.rn.satfinite.e2m1x2.f32, element 0 in the low nibble)."""
    out = llvm.inline_asm(
        T.i32(), [Float32(v).ir_value(loc=loc, ip=ip) for v in (f0, f1, f2, f3, f4, f5, f6, f7)],
        "{\n\t.reg .b8 byte0;\n\t.reg .b8 byte1;\n\t.reg .b8 byte2;\n\t.reg .b8 byte3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;\n\tcvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;\n\tcvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;\n\t"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t}\n",
        "=r,f,f,f,f,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(out)


@dsl_user_op
def packed_float_to_ue4m3(f0, f1, f2, f3, *, loc=None, ip=None) -> Int32:
    """4 fp32 -> 4 E4M3 bytes packed in an Int32 (cvt.rn.satfinite.e4m3x2.f32, element 0 in the low byte)."""
    out = llvm.inline_asm(
        T.i32(), [Float32(v).ir_value(loc=loc, ip=ip) for v in (f0, f1, f2, f3)],
        "{\n\t.reg .b16 lo;\n\t.reg .b16 hi;\n\tcvt.rn.satfinite.e4m3x2.f32   lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32   hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}\n",
        "=r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(out)

E2M1_MAX = 6.0
SR_SCALE_DIVISOR = 6.0 * 16.0 / 17.0          # rounding="stochastic": SF = amax / (6 * 16 / 17)
EDEN_SCALE_DIVISOR = 6.0 * 16.0 / 17.0 / 0.93  # rounding="ms_eden"
ROUNDINGS = ("nearest", "stochastic", "unit", "ms_eden")
F32_MIN_NORMAL = 1.1754943508222875e-38


# ======================================================================================================================
# Register-level helpers (importable: the FP8 set's NVFP4 side outputs use exactly these)
# ======================================================================================================================
def _asm_f32(op: str, *args) -> Float32:
    n = len(args)
    out = llvm.inline_asm(
        T.f32(), [Float32(a).ir_value() for a in args],
        f"{op} $0, " + ", ".join(f"${i + 1}" for i in range(n)) + ";",
        "=f," + ",".join(["f"] * n), has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(out)


def abs_ftz(a) -> Float32:
    """|a| with a subnormal a flushed to +0 (abs.ftz.f32)."""
    return _asm_f32("abs.ftz.f32", a)


def mul_ftz(a, b) -> Float32:
    """a * b with subnormal operands / result flushed to a signed zero (mul.ftz.f32: the same FMUL)."""
    return _asm_f32("mul.ftz.f32", a, b)


def div_full(a, b) -> Float32:
    """div.full.f32: Triton's fp32 '/' (2 ulp), used where a value must equal the Triton kernels' bitwise."""
    return _asm_f32("div.full.f32", a, b)


def mul_rn(a, b) -> Float32:
    """a * b as its own instruction (opaque to the compiler: never contracted into an FMA)."""
    return _asm_f32("mul.rn.f32", a, b)


def fma_rn(a, b, c) -> Float32:
    """fma(a, b, c), one rounding, explicitly."""
    return _asm_f32("fma.rn.f32", a, b, c)


def div_rn_f32(a, b) -> Float32:
    """a / b correctly rounded (div.rn.f32: Triton's tl.math.div_rn, torch's fp32 '/')."""
    return _asm_f32("div.rn.f32", a, b)


@dsl_user_op
def st_relaxed_f32(ptr: cute.Pointer, val, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), Float32(val).ir_value(loc=loc, ip=ip)],
                    "st.relaxed.gpu.global.f32 [$0], $1;", "l,f", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_relaxed_f32(ptr: cute.Pointer, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [ptr.toint(loc=loc, ip=ip).ir_value()],
                                   "ld.relaxed.gpu.global.f32 $0, [$1];", "=f,l", has_side_effects=True,
                                   is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


def round_to_bf16(a) -> Float32:
    """f32(bf16_rn(a)): cvt.rn.bf16.f32 and back (torch's .to(torch.bfloat16))."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value()],
        "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}\n",
        "=f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def add_rn(a, b) -> Float32:
    return _asm_f32("add.rn.f32", a, b)


def sub_rn(a, b) -> Float32:
    return _asm_f32("sub.rn.f32", a, b)


def rcp_rn(a) -> Float32:
    """1 / a correctly rounded (torch.reciprocal's 1.0f / a)."""
    return _asm_f32("rcp.rn.f32", a)


def lg2_approx(a) -> Float32:
    return _asm_f32("lg2.approx.f32", a)


def ex2_approx(a) -> Float32:
    return _asm_f32("ex2.approx.f32", a)


def f32_bits(f) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Float32(f).ir_value()], "mov.b32 $0, $1;", "=r,f",
                                 has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def bits_f32(i) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Int32(i).ir_value()], "mov.b32 $0, $1;", "=f,r",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def e4m3_byte_value(word: Int32, k: cutlass.Constexpr[int]) -> Float32:
    """Byte k (0..3) of a packed E4M3 word -> its fp32 value (e4m3 -> f16 -> f32, both exact)."""
    out = llvm.inline_asm(
        T.f32(), [Int32(word).ir_value()],
        "{\n\t.reg .b32 w;\n\t.reg .b16 x;\n\t.reg .b32 p;\n\t.reg .b16 h0, h1;\n\t"
        f"shr.b32 w, $1, {8 * k};\n\t"
        "cvt.u16.u32 x, w;\n\tcvt.rn.f16x2.e4m3x2 p, x;\n\tmov.b32 {h0, h1}, p;\n\tcvt.f32.f16 $0, h0;\n\t}\n",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(out)


def e4m3_rn(f) -> Int32:
    """fp32 -> E4M3 byte (cvt.rn.satfinite), in the low byte of an Int32."""
    return packed_float_to_ue4m3(f, Float32(0.0), Float32(0.0), Float32(0.0)) & 0xFF


def nvfp4_code_multiplier(sf_value: Float32, amax_over_scale_divisor: Float32) -> Float32:
    """The multiplier of a group's codes (the encoding's zero-scale rule): 1 / SF for SF > 0; +inf when the SF rounded
    to 0 but the unrounded scale > 0; 0 for an all-zero group."""
    out = llvm.inline_asm(
        T.f32(), [Float32(sf_value).ir_value(), Float32(amax_over_scale_divisor).ir_value()],
        "{\n\t.reg .pred pr, pu;\n\t.reg .f32 t, r;\n\t"
        "setp.gt.f32 pr, $1, 0f00000000;\n\tsetp.gt.f32 pu, $2, 0f00000000;\n\t"
        "rcp.rn.f32 r, $1;\n\tselp.f32 t, 0f7F800000, 0f00000000, pu;\n\tselp.f32 $0, r, t, pr;\n\t}\n",
        "=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(out)


@cute.jit
def mxfp4_scale(amax: Float32):
    """MXFP4 (OCP MX) scale of a group: (UE8M0 byte as Int32, fp32 multiplier of the codes). The exponent is estimated
    from the bits of amax * f32(1/6) and corrected by two exact comparisons (6 * 2^e is exact), so the divide's rounding
    never moves it. A subnormal amax / 6 (or an all-zero group) gives byte 0 and multiplier 0."""
    r = amax * Float32(1.0 / 6.0)
    bits = f32_bits(r)
    e = ((bits >> 23) & 0xFF) - 127
    if (bits & 0x7FFFFF) != 0:
        e = e + 1
    if Float32(6.0) * cute.math.exp2(Float32(e)) < amax:
        e = e + 1
    if Float32(6.0) * cute.math.exp2(Float32(e - 1)) >= amax:
        e = e - 1
    byte = Int32(0)
    mult = Float32(0.0)
    if r >= Float32(F32_MIN_NORMAL):
        byte = e + 127
        mult = Float32(1.0) / cute.math.exp2(Float32(e))
    return byte, mult


SPLITMIX_GOLDEN = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64


def _wrap64(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= (1 << 63) else v


@cute.jit
def uniform_from_mixed(z: Int64) -> Float32:
    """The splitmix64 finaliser of z = index * golden + seed (see uniform_from_counter), split out so a group's elements
    (consecutive indices) step z by a compile-time constant instead of a 64-bit multiply each."""
    z = (z ^ ((z >> 30) & Int64(0x3FFFFFFFF))) * Int64(-4658895280553007687)
    z = (z ^ ((z >> 27) & Int64(0x1FFFFFFFFF))) * Int64(-7723592293110705685)
    z = z ^ ((z >> 31) & Int64(0x1FFFFFFFF))
    return Float32(Int32((z >> 40) & Int64(0xFFFFFF))) * Float32(1.0 / 16777216.0)


@cute.jit
def uniform_from_counter(index: Int64, seed: Int64) -> Float32:
    """u in [0, 1) from a 64-bit element index and seed: the splitmix64 finaliser (logical shifts as masked arithmetic
    shifts, wrapping 64-bit products) -- bitwise nvfp4_2d_triton._hash_u01."""
    z = index * Int64(-7046029254386353131) + seed
    z = (z ^ ((z >> 30) & Int64(0x3FFFFFFFF))) * Int64(-4658895280553007687)
    z = (z ^ ((z >> 27) & Int64(0x1FFFFFFFFF))) * Int64(-7723592293110705685)
    z = z ^ ((z >> 31) & Int64(0x1FFFFFFFF))
    return Float32(Int32((z >> 40) & Int64(0xFFFFFF))) * Float32(1.0 / 16777216.0)


@cute.jit
def stochastic_round_e2m1(y: Float32, u: Float32) -> Float32:
    """Stochastic rounding of y onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (|y| clamped to 6, NaN -> 6) with uniform
    u: every step exact. Bitwise nvfp4_2d_triton._e2m1_sr."""
    a = cute.arch.fmin(cute.math.absf(y), Float32(6.0))
    lo = Float32(4.0)
    step = Float32(2.0)
    if a < Float32(2.0):
        lo = cute.math.floor(a * Float32(2.0)) * Float32(0.5)
        step = Float32(0.5)
    elif a < Float32(4.0):
        lo = cute.math.floor(a)
        step = Float32(1.0)
    v = lo
    if u * step < a - lo:
        v = lo + step
    if y < Float32(0.0):
        v = -v
    return v


@cute.jit
def e2m1_value(y: Float32) -> Float32:
    """The E2M1 round-to-nearest value of y (sign kept, |y| saturating at 6, NaN -> +6): the grid the cvt rounds to.
    Bitwise nvfp4_quant_triton._e2m1_val (used by Four-over-Six's error)."""
    a = cute.arch.fmin(cute.math.absf(y), Float32(6.0))
    v = Float32(6.0)
    if a < Float32(0.25):
        v = Float32(0.0)
    elif a < Float32(0.75):
        v = Float32(0.5)
    elif a < Float32(1.25):
        v = Float32(1.0)
    elif a < Float32(1.75):
        v = Float32(1.5)
    elif a < Float32(2.5):
        v = Float32(2.0)
    elif a < Float32(3.5):
        v = Float32(3.0)
    elif a < Float32(5.0):
        v = Float32(4.0)
    if y < Float32(0.0):
        v = -v
    return v


@cute.jit
def e2m1_pair_values(byte_word: Int32, k: cutlass.Constexpr[int]):
    """Byte k of a packed E2M1 word -> the two fp32 values (low nibble first), via cvt.rn.f16x2.e2m1x2 (exact)."""
    out = llvm.inline_asm(
        T.i32(), [Int32(byte_word).ir_value()],
        "{\n\t.reg .b32 w;\n\t.reg .b8 c;\n\t"
        f"shr.b32 w, $1, {8 * k};\n\t"
        "cvt.u8.u32 c, w;\n\tcvt.rn.f16x2.e2m1x2 $0, c;\n\t}\n",
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    h = Int32(out)
    lo = llvm.inline_asm(T.f32(), [h.ir_value()], "{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, $1;\n\tcvt.f32.f16 $0, a;\n\t}\n",
                         "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    hi = llvm.inline_asm(T.f32(), [h.ir_value()], "{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, $1;\n\tcvt.f32.f16 $0, b;\n\t}\n",
                         "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(lo), Float32(hi)


@cute.jit
def e2m1_code_value(code: Int32) -> Float32:
    """E2M1 code (bit 3 sign, magnitude codes 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6) -> fp32."""
    m = code & 7
    mf = Float32(m)
    v = Float32(6.0)
    if m < 4:
        v = mf * Float32(0.5)
    elif m < 6:
        v = mf - Float32(2.0)
    elif m == 6:
        v = Float32(4.0)
    if (code & 8) != 0:
        v = -v
    return v


@cute.jit
def encode_e2m1_words(y: cute.Tensor, base: cutlass.Constexpr[int], n: cutlass.Constexpr[int], words: cute.Tensor,
                      word_base: cutlass.Constexpr[int]):
    """Pack y[base : base + n] (already multiplied by the code multiplier, or SR grid values) into n / 8 E2M1 words
    (cvt.rn.satfinite, element 0 in the low nibble)."""
    for w in cutlass.range_constexpr(n // 8):
        b = base + 8 * w
        words[word_base + w] = packed_float_to_e2m1(y[b], y[b + 1], y[b + 2], y[b + 3],
                                                    y[b + 4], y[b + 5], y[b + 6], y[b + 7])


@cute.jit
def nvfp4_scale_from_amax(amax: Float32, rounding: cutlass.Constexpr[str]):
    """NVFP4 group scale from the (FTZ) group amax: (E4M3 byte as Int32, SF value, code multiplier)."""
    if const_expr(rounding == "unit"):
        return Int32(0x38), Float32(1.0), Float32(1.0)
    if const_expr(rounding == "nearest"):
        u = amax * Float32(1.0 / 6.0)
    elif const_expr(rounding == "stochastic"):
        u = amax * Float32(17.0 / 96.0)
    else:
        u = amax * Float32(0.93 * 17.0 / 96.0)
    byte = e4m3_rn(u)
    sfv = e4m3_byte_value(byte, 0)
    return byte, sfv, nvfp4_code_multiplier(sfv, u)


@cute.jit
def e4m3_stochastic(x: Float32, u: Float32) -> Float32:
    """Stochastic rounding of x >= 0 onto the E4M3 grid (MS-EDEN's scale), exact arithmetic: e = floor(log2(max(x, 2^-6)))
    from the exponent bits, step 2^(e - 3), lo = floor(x / step) * step. (The old Triton form used lg2/ex2.approx and
    div.full; it gave the same bytes on 4M groups, probe_eden.)"""
    xc = cute.arch.fmin(cute.arch.fmax(x, Float32(0.0)), Float32(448.0))
    e = ((f32_bits(cute.arch.fmax(xc, Float32(0.015625))) >> 23) & 0xFF) - 127
    step = bits_f32((e - 3 + 127) << 23)
    lo = cute.math.floor(xc * bits_f32((127 - (e - 3)) << 23)) * step
    v = lo
    if u * step < xc - lo:
        v = lo + step
    return cute.arch.fmin(v, Float32(448.0))


# ----------------------------------------------------------------------------------------------------------------------
# Summation orders. Where a quantiser output is a fp32 sum (D, the Four-over-Six errors, MS-EDEN's factor, row norms) the
# order is part of the definition (the old Triton kernels' orders, recovered by numerical probes), so the
# reference and the old kernels agree bitwise. An order is a tuple of column-bit offsets: level k adds the partial sums
# of indices differing in that bit (an xor butterfly); the products are exact where it matters (D: E2M1 x E4M3 x bf16).
# ----------------------------------------------------------------------------------------------------------------------
LOG2_E_F32 = math.log2(math.e)                 # rounded to fp32 where used: torch's fp32 tensor x Python float
D_SUM_ORDER = (2, 4, 1, 64, 32, 16, 8)        # nvfp4_2d_triton D (tl.sum over 128, recovered: probe7)


def xor_tree_pairs(n: int, order: tuple):
    """[(dst, src), ...] per level for an xor-butterfly sum of n values in the given offset order; the root lands in 0."""
    levels = []
    done = 0
    for off in order:
        pairs = []
        for c in range(n):
            if (c & (done | off)) == 0:
                pairs.append((c, c | off))
        levels.append(pairs)
        done |= off
    return levels


@cute.jit
def xor_tree_sum(v: cute.Tensor, n: cutlass.Constexpr[int], order: cutlass.Constexpr):
    """In-place xor-butterfly sum of v[0:n] in `order`; returns v[0]."""
    for pairs in cutlass.range_constexpr(len(xor_tree_pairs(n, order))):
        for p in cutlass.range_constexpr(len(xor_tree_pairs(n, order)[pairs])):
            d_, s_ = xor_tree_pairs(n, order)[pairs][p]
            v[d_] = v[d_] + v[s_]
    return v[0]


TORCH_LAST_DIM_ORDER = (16, 8, 4, 2, 1)   # torch's 128-wide fp32 last-dim reduce: 32 lanes x 4, then this xor tree


@cute.jit
def torch_row_sum(p: cute.Tensor) -> Float32:
    """sum of a 128-value row exactly as torch's x.sum(-1) reduces a contiguous (..., 128) fp32 tensor
    (recovered by probing torch: 32 lanes of 4 consecutive values summed in order, then an xor butterfly
    over the lanes 16, 8, 4, 2, 1)."""
    t = cute.make_rmem_tensor(cute.make_layout(32), Float32)
    for g in cutlass.range_constexpr(32):
        a = add_rn(add_rn(p[4 * g], p[4 * g + 1]), p[4 * g + 2])
        t[g] = add_rn(a, p[4 * g + 3])
    return xor_tree_sum(t, 32, TORCH_LAST_DIM_ORDER)


@cute.jit
def torch_row_sumsq(x: cute.Tensor) -> Float32:
    """sum of squares of a 128-value row in torch's norm(dim=-1) order (32 lanes x 4, xor tree 16..1), before the sqrt."""
    t = cute.make_rmem_tensor(cute.make_layout(32), Float32)
    for g in cutlass.range_constexpr(32):
        a = mul_rn(x[4 * g], x[4 * g])
        for j in cutlass.range_constexpr(1, 4):
            a = fma_rn(x[4 * g + j], x[4 * g + j], a)
        t[g] = a
    return xor_tree_sum(t, 32, TORCH_LAST_DIM_ORDER)


@cute.jit
def torch_row_norm2(x: cute.Tensor) -> Float32:
    """||x||_2 of a 128-value row as torch's x.float().norm(dim=-1) (the same lanes and tree; squares exact), sqrt.rn."""
    t = cute.make_rmem_tensor(cute.make_layout(32), Float32)
    for g in cutlass.range_constexpr(32):
        a = mul_rn(x[4 * g], x[4 * g])
        for j in cutlass.range_constexpr(1, 4):
            a = fma_rn(x[4 * g + j], x[4 * g + j], a)
        t[g] = a
    return _asm_f32("sqrt.rn.f32", xor_tree_sum(t, 32, TORCH_LAST_DIM_ORDER))


@cute.jit
def pow2_floor_as_torch(f: Float32) -> Float32:
    """2^floor(log2 f) for normal positive f, exactly: f's exponent field with the mantissa cleared (the torch side,
    interface_bwd_fp4q.pow2_floor_bits, takes the same bits). (The earlier form rounded log2 to float32 first, as
    torch.floor(torch.log2(f)) does, so a value within a float32 rounding below 2^k gave 2^k; that needed a float64
    log2. Now such a value gives 2^(k-1), the exact floor.)"""
    return bits_f32(f32_bits(f) & Int32(0x7F800000))


# ======================================================================================================================
# The tile kernel
# ======================================================================================================================
@dataclass(frozen=True)
class _TileSpec:
    input_dtype: object                    # cutlass dtype of x
    scale_group_size: int = 16             # 16 (NVFP4) | 32 (MXFP4)
    row_copy: bool = True
    transposed_copy: bool = False
    transposed_key_order: bool = True      # transposed copy: 32-token blocks of each tile in the kernels' order [0,2,1,3]
    blocks_2d: bool = False                # one scale per 16x16 (NVFP4) / 32x32 (MXFP4) block, shared by both copies
    four_over_six: bool = False
    global_scale: str = "none"             # none | host | device
    rounding: str = "nearest"
    row_factors: bool = False              # per-32-row power-of-two factors
    e4m3_copy: str = "none"                # none | rotated | unrotated (the values before the Hadamard)
    row_norm_max: bool = False
    tile_amax: bool = False
    hadamard_signs: Optional[tuple] = None
    rotation_tc: bool = False              # hadamard_rotation_method="tensor_core": the warps rotate the staged tile on
    #                                        mma.sync into an fp32 tile, each thread reads its rotated row back
    rotation_tc5: bool = False             # hadamard_rotation_method="tcgen05": the row the butterfly would rotate
    #                                        (RoPE'd, zero-padded) rotated on tcgen05 (quant._tcgen05.RowRotation)
    rope: str = "none"                     # none | transposed_tables | row_tables
    d_output: bool = False
    output_dtype_o: object = None
    transposed_head_major: bool = False    # the transposed codes buffer is (b, h, d, s/2) (a layout of the argument only)
    head_dim_in: int = 128                 # x's head dim (a multiple of the group size)
    head_dim_out: int = 128                # the quantised operands' head dim: head_dim_in, or 128 (zero padding in registers)
    staged_load: bool = True               # the tile goes through shared memory with coalesced 16-byte loads
    min_ctas_per_sm: int = 0               # occupancy hint (launch bounds): 0 = none
    head_dim_chunks: int = 1               # head_dim 128 * chunks (row copy only): each chunk is viewed as a head
    # the all-FP4 backward's causal dS row scale (interface_bwd_fp4q.fp4q_prepare's former torch chain), from the dO read:
    # per query row F_q (a power of two), x -> bf16(x F_q) before the codes, uinv, partial maxima and the dQ zero rows
    causal_row_scale: bool = False
    row_scale_fold: bool = False           # the K fold: the D estimate of the bound times the prefix max of r_k
    row_scale_top: float = 0.0             # the dO-side cap: F_q max|dO| <= top (smax x 448, or x 256 for MS-EDEN)
    lse_log2_offset: float = 0.0           # uinv = 2^min(frame - lse log2 e - offset, 1000)
    zero_dq_accum: bool = False            # zero the (b, h, s, 128) f16 / fp32 dQ accumulator rows of the tile
    # a value reduced over the whole tensor in the same launch (the last tile to finish reduces the per-tile partials and
    # resets a persistent counter: no zero-fill launch, no second pass):
    #   "row_norm2"   max over rows of ||x||^2 in torch's x.float().norm(dim=-1) order (out[0]; the norm max is its sqrt.rn)
    #   "abs"         max |x| (out[0]); with global_scale_divisor > 0 also out64[0] = max(amax / divisor, 1e-30), float32
    #   "dense_bound" the all-FP4 backward's global dS bound: max over rows of dense_bound_row's bound_q
    global_max: str = "none"
    global_scale_divisor: float = 0.0
    # NVFP4 transposed copy: also its SF atoms with the two head-dim halves swapped (rows 64-127 in the words of rows
    # 0-63 and back: the dQ GEMM's second d-half SF tile), in the second half of the same SF buffer
    transposed_sf_halves_swapped: bool = False
    row_amax_out: bool = False             # per-row max |x| of the raw row -> (b, h, s)
    # the all-FP4 backward's dense (global dS multiplier) per-row chain from the dO launch with the device global scale
    dense_row_bound: bool = False
    lse_log2_offset_dense: float = 0.0
    # the stochastic-rounding seed read from a device counter c (an int64 (1,) tensor): seed = (c * 0x9E3779B1) mod 2^31,
    # the host formula of the seed sequence, evaluated on the GPU (a CUDA-graph replay reads the counter's new value)
    device_seed: bool = False
    # row_end (the aligned varlen padding): an int32 (b, s / 128) tensor; rows q0 + r >= row_end[bb, t] are zero rows
    row_end: bool = False
    # per-(document, kv head) E4M3 copy scale (per-tensor operands on aligned varlen documents): the scale argument is an
    # (n_docs + 1, h_kv) float32 table, indexed by the document of the row's 128-row tile (mE4Doc) and its kv head
    e4m3_doc: bool = False
    # aligned varlen documents (b = 1): the launch's reductions into slots (causal_row_scale's max slots) are per document
    # of the row's 128-row tile (mE4Doc), max_slots then (n_docs + 1) x 2
    doc_slots: bool = False
    # lp_docmaps gather (b = 1): x is a shape carrier; the staged tile's row r is row mSrc[r] of the packed mX (a zero
    # row where mSrc[r] < 0); the tiles are the codes' (mQ)
    gather: bool = False
    # cov1, lp_docmaps compact maps (b = 1): x is the caller's PACKED (1, T, h, d) rows; the tiles are the maps' aligned
    # 128-row tiles (persistent over the device tile count), tile t's rows the packed rows (vt_tile row 0) + r for r <
    # its real rows (row 1), zero rows after; the codes / E4M3 copy at the packed rows, the SF bytes as the dense
    # per-row words (b h, R d / 64) int32 (the sf_dense kernels' NVFP4 S words), no per-tile atom
    packed: bool = False

    @property
    def persistent(self) -> bool:
        return self.transposed_copy

    @property
    def fp32_transposed_tile(self) -> bool:
        return (self.hadamard_signs is not None or self.global_scale != "none" or self.row_factors
                or self.input_dtype == Float32)

    @property
    def stage_row_elems(self) -> int:
        """Elements per padded smem row of the staged tile (16 bytes of padding: conflict-free row reads)."""
        return self.head_dim_in + 16 // (self.input_dtype.width // 8)

    @property
    def smem_bytes(self) -> int:
        n = 256 + 16                         # sRed (64 floats) + alignment
        n += (2 if self.persistent else 1) * 128 * self.stage_row_elems * (self.input_dtype.width // 8) + 16
        if self.transposed_copy and not self.blocks_2d:
            esize = 4 if self.fp32_transposed_tile else self.input_dtype.width // 8
            pad = 16 // esize
            n += self.head_dim_out * (128 + pad) * esize + 16
        if self.transposed_copy and self.blocks_2d:
            n += self.head_dim_out * 80 + 16 + 128 * 2 * 4 + 16   # transposed code bytes (padded rows) + SF words
        if self.rotation_tc and not self.rotated_tile_in_transposed_tile:
            n += 128 * 132 * 4 + 16                                # the rotated rows (fp32, padded)
        if self.rotation_tc5:
            n += RowRotation.smem_bytes                            # the UMMA A / B images, mbarrier, TMEM address
        return n

    @property
    def rotated_tile_in_transposed_tile(self) -> bool:
        """The tensor-core rotation's fp32 rows share the fp32 transposed tile (both 128 x 132 fp32; the rows are read
        back before the transposed copy writes it)."""
        return self.rotation_tc and self.transposed_copy and not self.blocks_2d and self.fp32_transposed_tile \
            and self.head_dim_out == 128


@cute.jit
def _aligned16(dtype: cutlass.Constexpr, it, n: cutlass.Constexpr[int], space: cutlass.Constexpr):
    """A length-n view at iterator `it`, asserted 16-byte aligned (the DSL cannot prove it through a dynamic offset; without
    it the copy falls back to element-wide accesses)."""
    return cute.make_tensor(cute.make_ptr(dtype, it.toint(), space, assumed_align=16), cute.make_layout(n))


def _perm_block(blk):
    return (blk % 2) * 2 + blk // 2      # [0, 2, 1, 3] (an involution)


@dsl_user_op
def _cp_async_16_zfill(smem_addr, gmem_ptr: cute.Pointer, nbytes, *, loc=None, ip=None) -> None:
    """16-byte cp.async reading nbytes (0 or 16) of gmem, zero-filling the rest (the lp_docmaps gather)."""
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
                           Int32(nbytes).ir_value(loc=loc, ip=ip)],
                    "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _shf_r(lo, hi, sh, *, loc=None, ip=None) -> Int32:
    """(hi:lo) >> sh, 0 <= sh < 32."""
    return Int32(llvm.inline_asm(T.i32(), [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip),
                                           Int32(sh).ir_value(loc=loc, ip=ip)],
                                 "shf.r.clamp.b32 $0, $1, $2, $3;", "=r,r,r,r", has_side_effects=False,
                                 is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def store_bytes_unaligned(dst_i32_ptr, b0, nb, src_smem_u8_ptr):
    """(cov5) bytes src[0 .. nb) (a 16-byte aligned shared row of 64 bytes, nb <= 64) to bytes b0 .. b0 + nb of the
    global row at dst_i32_ptr (4-byte aligned): the bytes before the first whole word and after the last one as byte
    stores, the whole words between as 32-bit stores of funnel-shifted source words -- no byte outside [b0, b0 + nb) is
    written (the neighbouring bytes are other documents' codes)."""
    src = cute.make_tensor(cute.recast_ptr(src_smem_u8_ptr, dtype=Int32), cute.make_layout(16))
    w = cute.make_rmem_tensor(cute.make_layout(17), Int32)
    for i in cutlass.range_constexpr(16):
        w[i] = src[i]
    w[16] = Int32(0)
    d32 = cute.make_tensor(dst_i32_ptr, cute.make_layout(1 << 30))
    d8 = cute.make_tensor(cute.recast_ptr(dst_i32_ptr, dtype=cutlass.Uint8), cute.make_layout(1 << 30))
    s = b0 & 3
    hd = (4 - s) & 3                          # head bytes before the first whole destination word
    wa = (b0 + 3) >> 2                        # the first whole destination word
    we = (b0 + nb) >> 2                       # past the last whole destination word
    sh = hd * 8                               # source byte offset of word wa: hd
    # source word j of the shifted stream: bytes hd + 4 j .. -> (w[j'] , w[j'+1]) with j' = j + (hd >> 2) = j
    for j in cutlass.range_constexpr(16):
        if wa + j < we:
            d32[wa + j] = _shf_r(w[j], w[j + 1], sh)   # (sh = 0: w[j])
    sb = cute.make_tensor(src_smem_u8_ptr, cute.make_layout(64))
    for j in cutlass.range_constexpr(3):
        if j < hd:
            if j < nb:
                d8[b0 + j] = sb[j]
    tl = (b0 + nb) & 3                        # tail bytes after the last whole word
    for j in cutlass.range_constexpr(3):
        if j < tl:
            if (b0 + nb - tl + j) >= b0 + hd:     # (not a head byte: a row shorter than its first word)
                d8[b0 + nb - tl + j] = sb[nb - tl + j]


class _FP4TileKernel:
    def __init__(self, spec: _TileSpec):
        self.s = spec
        self.threads = 128

    @cute.jit
    def __call__(self, mX: cute.Tensor, mQ: cute.Tensor, mSF: cute.Tensor, mQt: cute.Tensor, mSFt: cute.Tensor,
                 mX8: cute.Tensor, mE4Scale: cute.Tensor, mOuter: cute.Tensor, inv_outer: Float32, outer: Float32,
                 mRowFac: cute.Tensor, mNorm: cute.Tensor, mTileAmax: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
                 mO: cute.Tensor, mD: cute.Tensor, mLse: cute.Tensor, mFrame: cute.Tensor, mVmax: cute.Tensor, mRmax: cute.Tensor, mQAmax: cute.Tensor,
                 mRowF: cute.Tensor, mUinv: cute.Tensor, mAmaxSlots: cute.Tensor, mDqAcc: cute.Tensor,
                 mExtraRow: cute.Tensor, mPartials: cute.Tensor, mCounter: cute.Tensor, mGlobalOut: cute.Tensor,
                 mGlobalOut64: cute.Tensor,
                 seed: Int64, num_ctas: Int32, mWork: cute.Tensor, mSeed: Optional[cute.Tensor],
                 mRowEnd: Optional[cute.Tensor], mE4Doc: Optional[cute.Tensor], stream: cuda.CUstream,
                 mSrc: Optional[cute.Tensor] = None, mVt: Optional[cute.Tensor] = None,
                 mNT: Optional[cute.Tensor] = None, hrows=None):
        self.kernel(mX, mQ, mSF, mQt, mSFt, mX8, mE4Scale, mOuter, inv_outer, outer, mRowFac, mNorm, mTileAmax,
                    mCos, mSin, mO, mD, mLse, mFrame, mVmax, mRmax, mQAmax, mRowF, mUinv, mAmaxSlots, mDqAcc, mExtraRow, mPartials, mCounter, mGlobalOut, mGlobalOut64, seed, num_ctas, mWork, mSeed, mRowEnd, mE4Doc, mSrc,
                    *((mVt, mNT, hrows) if const_expr(self.s.packed) else ())).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1],
                                                               smem=self.s.smem_bytes, stream=stream,
                                                               min_blocks_per_mp=self.s.min_ctas_per_sm)

    @cute.jit
    def prefetch(self, mX: cute.Tensor, sXb: cute.Tensor, tile, tiles, nheads, mSrc=None, pk=None):
        """cp.async the tile's 128 rows into the staging buffer: each warp instruction 512 contiguous bytes. pk (packed,
        cov1): (t, head, packed start, real rows) -- rows past the real rows staged as zeros."""
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        t = tile % tiles
        hh = (tile // tiles) % nheads
        bb = tile // (tiles * nheads)
        if const_expr(pk is not None):
            t, hh, bb = pk[0], pk[1], Int32(0)
        q0 = t * 128
        vec = 128 // sp.input_dtype.width                 # elements per 16 bytes
        cpr = sp.head_dim_in // vec                        # chunks per row
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), sp.input_dtype,
                                   num_bits_per_copy=128)
        for i in cutlass.range_constexpr(cpr):
            k = i * 128 + tidx
            rr = k // cpr
            ch = k % cpr
            if const_expr(pk is not None):   # packed: row pb + rr, a zero row past the real rows (address clamped)
                _cp_async_16_zfill(Int32(sXb[rr, None].iterator.toint()) + ch * 16,
                                   mX[0, cutlass.min(pk[2] + rr, mX.shape[1] - 1), hh, None].iterator + ch * vec,
                                   cutlass.select_(rr < pk[3], Int32(16), Int32(0)))
            elif const_expr(mSrc is not None):   # lp_docmaps gather: the packed row (a zero row where mSrc < 0)
                sr = mSrc[q0 + rr]
                _cp_async_16_zfill(Int32(sXb[rr, None].iterator.toint()) + ch * 16,
                                   mX[bb, cutlass.max(sr, Int32(0)), hh, None].iterator + ch * vec,
                                   cutlass.select_(sr >= Int32(0), Int32(16), Int32(0)))
            else:
                cute.copy(atom, _aligned16(sp.input_dtype, mX[bb, q0 + rr, hh, None].iterator + ch * vec, vec,
                                           cute.AddressSpace.gmem),
                          _aligned16(sp.input_dtype, sXb[rr, None].iterator + ch * vec, vec, cute.AddressSpace.smem))

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mQ: cute.Tensor, mSF: cute.Tensor, mQt: cute.Tensor, mSFt: cute.Tensor,
               mX8: cute.Tensor, mE4Scale: cute.Tensor, mOuter: cute.Tensor, inv_outer: Float32, outer: Float32,
               mRowFac: cute.Tensor, mNorm: cute.Tensor, mTileAmax: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
               mO: cute.Tensor, mD: cute.Tensor, mLse: cute.Tensor, mFrame: cute.Tensor, mVmax: cute.Tensor, mRmax: cute.Tensor, mQAmax: cute.Tensor,
               mRowF: cute.Tensor, mUinv: cute.Tensor, mAmaxSlots: cute.Tensor, mDqAcc: cute.Tensor,
               mExtraRow: cute.Tensor, mPartials: cute.Tensor, mCounter: cute.Tensor, mGlobalOut: cute.Tensor,
               mGlobalOut64: cute.Tensor,
               seed: Int64, num_ctas: Int32, mWork: cute.Tensor, mSeed: Optional[cute.Tensor] = None,
               mRowEnd: Optional[cute.Tensor] = None, mE4Doc: Optional[cute.Tensor] = None,
               mSrc: Optional[cute.Tensor] = None, mVt: Optional[cute.Tensor] = None, mNT: Optional[cute.Tensor] = None,
               hrows=None):
        sp = self.s
        if const_expr(sp.device_seed):
            seed = (mSeed[0] * Int64(0x9E3779B1)) & Int64(0x7FFFFFFF)
        if const_expr(sp.e4m3_doc or sp.doc_slots):
            self._e4doc = mE4Doc   # the document of each 128-row tile (store_e4m3's per-document scale / doc_slots)
        tidx, _, _ = cute.arch.thread_idx()
        seqlen = mX.shape[1]
        if const_expr(mSrc is not None):   # (lp_docmaps gather: mX is the packed source, the tiles are the codes')
            seqlen = mQ.shape[1]
        tiles = seqlen // 128
        nheads = mX.shape[2]
        total = tiles * nheads * mX.shape[0]
        smem = cutlass.utils.SmemAllocator()
        sRed = smem.allocate_tensor(Float32, cute.make_layout(64), 16)
        RS = sp.stage_row_elems
        nbuf = 2 if const_expr(sp.persistent) else 1
        sX = smem.allocate_tensor(sp.input_dtype, cute.make_layout((128, RS, nbuf), stride=(RS, 1, 128 * RS)), 16)
        sT = sRed
        sB = sRed
        sS = sRed
        DO = sp.head_dim_out
        if const_expr(sp.transposed_copy and not sp.blocks_2d):
            edt = Float32 if const_expr(sp.fp32_transposed_tile) else sp.input_dtype
            pad = 16 // (edt.width // 8)
            sT = smem.allocate_tensor(edt, cute.make_layout((DO, 128 + pad), stride=(128 + pad, 1)), 16)
        if const_expr(sp.transposed_copy and sp.blocks_2d):
            sB = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((DO, 80), stride=(80, 1)), 16)
            sS = smem.allocate_tensor(Int32, cute.make_layout((128, 2), stride=(2, 1)), 16)
        sR32 = sRed
        if const_expr(sp.rotation_tc):
            if const_expr(sp.rotated_tile_in_transposed_tile):
                sR32 = cute.make_tensor(sT.iterator, cute.make_layout((128, 132), stride=(132, 1)))
            else:
                sR32 = smem.allocate_tensor(Float32, cute.make_layout((128, 132), stride=(132, 1)), 16)
        st5 = None
        if const_expr(sp.rotation_tc5):
            st5 = RowRotation(sp.input_dtype, sp.hadamard_signs).allocate(smem)
            RowRotation(sp.input_dtype, sp.hadamard_signs).setup(st5, tidx)
        bid, _, _ = cute.arch.block_idx()
        if const_expr(sp.packed):
            # (cov1, compact maps) persistent over the device tile count: tile = (head, aligned tile t), its rows the
            # packed rows pb + r, r < nvr (zero rows after); mSF is the dense per-row SF word table
            tiles = mVt.shape[1]
            # (cov5) the stochastic rounding's row hash counts the aligned rows of the FULL maps' layout (hrows: their
            # padded size), so the dither is bitwise the fused path's
            seqlen = hrows
            nt = mNT[0]
            for tile in cutlass.range(bid, nt * nheads, num_ctas, unroll=1):
                t = tile % nt
                hh = tile // nt
                pb = mVt[0, t]
                nvr = mVt[1, t] & Int32(0xFFFF)
                self.prefetch(mX, sX[None, None, 0], tile, tiles, nheads, None, (t, hh, pb, nvr))
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                self.tile_body(mX, mQ, mSF, mQt, mSFt, mX8, mE4Scale, mOuter, inv_outer, outer, mRowFac, mNorm,
                               mTileAmax, mCos, mSin, mO, mD, mLse, mFrame, mVmax, mRmax, mQAmax, mRowF, mUinv,
                               mAmaxSlots, mDqAcc, mExtraRow, mPartials, mCounter, mGlobalOut, mGlobalOut64, seed,
                               sX[None, None, 0], sRed, sT, sB, sS, Int32(0), hh, t, tiles, nheads, seqlen, sR32, st5,
                               Int32(0), mRowEnd, pk=(pb, nvr, nt))
                cute.arch.sync_threads()
        elif const_expr(sp.persistent):
            self.prefetch(mX, sX[None, None, 0], bid, tiles, nheads, mSrc)
            cute.arch.cp_async_commit_group()
            n_iter = (total - bid + num_ctas - 1) // num_ctas
            for it in cutlass.range(n_iter, unroll=1):
                tile = bid + it * num_ctas
                nxt = tile + num_ctas
                if nxt < total:
                    self.prefetch(mX, sX[None, None, (it + 1) % 2], nxt, tiles, nheads, mSrc)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
                t = tile % tiles
                hh = (tile // tiles) % nheads
                bb = tile // (tiles * nheads)
                self.tile_body(mX, mQ, mSF, mQt, mSFt, mX8, mE4Scale, mOuter, inv_outer, outer, mRowFac, mNorm,
                               mTileAmax, mCos, mSin, mO, mD, mLse, mFrame, mVmax, mRmax, mQAmax, mRowF, mUinv, mAmaxSlots, mDqAcc, mExtraRow, mPartials, mCounter, mGlobalOut, mGlobalOut64, seed, sX[None, None, it % 2], sRed, sT, sB, sS, bb, hh, t,
                               tiles, nheads, seqlen, sR32, st5, Int32(it & 1), mRowEnd)
                cute.arch.sync_threads()
        else:
            self.prefetch(mX, sX[None, None, 0], bid, tiles, nheads, mSrc)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            t = bid % tiles
            hh = (bid // tiles) % nheads
            bb = bid // (tiles * nheads)
            self.tile_body(mX, mQ, mSF, mQt, mSFt, mX8, mE4Scale, mOuter, inv_outer, outer, mRowFac, mNorm, mTileAmax,
                           mCos, mSin, mO, mD, mLse, mFrame, mVmax, mRmax, mQAmax, mRowF, mUinv, mAmaxSlots, mDqAcc, mExtraRow, mPartials, mCounter, mGlobalOut, mGlobalOut64, seed, sX[None, None, 0], sRed, sT, sB, sS, bb, hh, t, tiles, nheads,
                           seqlen, sR32, st5, Int32(0), mRowEnd)
        if const_expr(sp.rotation_tc5):
            RowRotation(sp.input_dtype, sp.hadamard_signs).teardown(st5, tidx)

    @cute.jit
    def tile_body(self, mX: cute.Tensor, mQ: cute.Tensor, mSF: cute.Tensor, mQt: cute.Tensor, mSFt: cute.Tensor,
                  mX8: cute.Tensor, mE4Scale: cute.Tensor, mOuter: cute.Tensor, inv_outer: Float32, outer: Float32,
                  mRowFac: cute.Tensor, mNorm: cute.Tensor, mTileAmax: cute.Tensor, mCos: cute.Tensor,
                  mSin: cute.Tensor, mO: cute.Tensor, mD: cute.Tensor, mLse: cute.Tensor, mFrame: cute.Tensor,
                  mVmax: cute.Tensor, mRmax: cute.Tensor, mQAmax: cute.Tensor, mRowF: cute.Tensor, mUinv: cute.Tensor,
                  mAmaxSlots: cute.Tensor, mDqAcc: cute.Tensor, mExtraRow: cute.Tensor, mPartials: cute.Tensor,
                  mCounter: cute.Tensor, mGlobalOut: cute.Tensor, mGlobalOut64: cute.Tensor, seed: Int64, sXb: cute.Tensor, sRed: cute.Tensor,
                  sT: cute.Tensor, sB: cute.Tensor, sS: cute.Tensor, bb, hh, t, tiles, nheads, seqlen,
                  sR32: cute.Tensor, st5, rot_phase, mRowEnd=None, pk=None):
        sp = self.s
        G = sp.scale_group_size
        tidx, _, _ = cute.arch.thread_idx()
        q0 = t * 128
        r = tidx
        # (packed, cov1: this thread's packed row and whether the tile's row r is a real row)
        prow = None
        prow_ok = None
        prow_c = None
        if const_expr(pk is not None):
            prow = pk[0] + r
            prow_ok = r < pk[1]
            prow_c = pk[0] + cutlass.min(r, pk[1] - 1)   # (cov5: a real row to read the per-row inputs at)
        lane = tidx % 32
        warp = tidx // 32
        if const_expr(sp.rotation_tc):
            self.rotate_tile_tc(sXb, sR32, mCos, mSin, q0)
        # ------------------------------------------------------------------ load row r, transforms
        DI = sp.head_dim_in
        DO = sp.head_dim_out
        # the row-scale modes' per-row scalars, loaded before the row work so their latency is hidden:
        # [lse, frame, vmax_q, rmax_q, max|Q|, max||V||^2]
        rsc = cute.make_rmem_tensor(cute.make_layout(6), Float32)
        if const_expr(sp.causal_row_scale or sp.dense_row_bound):
            ridx0 = (bb * nheads + hh) * seqlen + q0 + r
            if const_expr(pk is not None):   # (cov5: the per-row tables per packed row, (h, T))
                ridx0 = hh * (mLse.shape[0] // nheads) + prow_c
            rsc[0] = mLse[ridx0]
            rsc[1] = mFrame[ridx0]
            if const_expr(sp.causal_row_scale):
                rsc[2] = mVmax[ridx0]
                rsc[4] = mQAmax[ridx0]
                if const_expr(sp.row_scale_fold):
                    rsc[3] = mRmax[ridx0]
            else:
                if const_expr(sp.doc_slots):
                    rsc[5] = mVmax[self._e4doc[t]]   # the row's document's max ||V||^2
                else:
                    rsc[5] = mVmax[0]
        rowb = cute.make_rmem_tensor(cute.make_layout(DI), sp.input_dtype)
        cute.autovec_copy(cute.make_tensor(sXb[r, None].iterator, cute.make_layout(DI)), rowb)
        xin = cute.make_rmem_tensor(cute.make_layout(DI), Float32)
        xin.store(rowb.load().to(Float32))
        if const_expr(mRowEnd is not None):
            # the aligned varlen padding (row_end): a row past its document's true length is a zero row
            keep = q0 + r < mRowEnd[bb, t]
            for j in cutlass.range_constexpr(DI):
                xin[j] = cutlass.select_(keep, xin[j], Float32(0.0))
            if const_expr(sp.causal_row_scale):
                rsc[4] = cutlass.select_(keep, rsc[4], Float32(0.0))   # a padding row adds nothing to max |Q| / F_q
            if const_expr(sp.dense_row_bound):
                rsc[0] = cutlass.select_(keep, rsc[0], -Float32.inf)   # a padding row: uinv 0, bound 0
        if const_expr(pk is not None and sp.causal_row_scale):
            rsc[4] = cutlass.select_(prow_ok, rsc[4], Float32(0.0))   # (a zero row: nothing to max |Q| / F_q)
        if const_expr(sp.rope != "none"):
            rot.rope_rotate_row(xin, mCos, mSin, q0 + r, sp.rope == "transposed_tables", DI)
            # the rotated row in the input dtype: what a standalone RoPE kernel writes (both copies use it)
            rowb.store(xin.load().to(sp.input_dtype))
            xin.store(rowb.load().to(Float32))
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)    # columns >= DI: the zero padding
        for j in cutlass.range_constexpr(128):
            if const_expr(j < DI):
                x[j] = xin[j]
            else:
                x[j] = Float32(0.0)
        if const_expr(sp.causal_row_scale):
            if const_expr(pk is not None):
                self.causal_row_scale(x, rowb, mO, rsc, mRowF, mUinv, mDqAcc, sRed, bb, hh, prow_c, nheads, seqlen, lane,
                                      warp, pk_ok=prow_ok)
            else:
                self.causal_row_scale(x, rowb, mO, rsc, mRowF, mUinv, mDqAcc, sRed, bb, hh, q0 + r, nheads, seqlen, lane,
                                      warp)
        ridx = (bb * nheads + hh) * seqlen + q0 + r
        if const_expr(pk is not None):   # (cov6: the per-row outputs per packed row, (1, h, T); real rows written)
            ridx = hh * mX.shape[1] + prow_c
        gpart = Float32(0.0)                         # this row's contribution to the launch-wide max
        if const_expr(sp.row_amax_out or sp.global_max == "abs" or sp.dense_row_bound):
            ram = Float32(0.0)
            for j in cutlass.range_constexpr(128):
                ram = cute.arch.fmax(ram, cute.math.absf(x[j]))
            if const_expr(sp.row_amax_out or sp.dense_row_bound):
                if const_expr(pk is not None):
                    if prow_ok:
                        mExtraRow[ridx] = ram
                else:
                    mExtraRow[ridx] = ram
            if const_expr(sp.global_max == "abs"):
                gpart = ram
        if const_expr(sp.global_max == "row_norm2"):
            gpart = torch_row_sumsq(x)
        dense_don = Float32(0.0)
        if const_expr(sp.dense_row_bound):
            dense_don = torch_row_norm2(x)               # ||dO|| of the raw row (before the global scale)
            if const_expr(sp.zero_dq_accum and pk is not None):
                if prow_ok:   # (cov6 packed: the real rows' accumulator rows (1, h, T', 128), each thread its own)
                    self.zero_accumulator_row(mDqAcc, hh, prow)
            elif const_expr(sp.zero_dq_accum):
                self.zero_accumulator_tile(mDqAcc, bb, hh, q0)
        if const_expr(sp.e4m3_copy == "unrotated" and pk is not None):
            if prow_ok:
                self.store_e4m3(x, mX8, mE4Scale, bb, q0 + r, hh, prow)
        elif const_expr(sp.e4m3_copy == "unrotated"):
            self.store_e4m3(x, mX8, mE4Scale, bb, q0 + r, hh)
        if const_expr(sp.hadamard_signs is not None and sp.rotation_tc):
            cute.autovec_copy(cute.make_tensor(sR32[r, None].iterator, cute.make_layout(128)), x)
        elif const_expr(sp.hadamard_signs is not None and sp.rotation_tc5):
            # the row the butterfly rotates (exact in the input dtype: RoPE'd and rounded, zero-padded), on tcgen05
            rw5 = cute.make_rmem_tensor(cute.make_layout(64), Int32)
            av5 = cute.make_tensor(cute.recast_ptr(rw5.iterator, dtype=sp.input_dtype), cute.make_layout(128))
            av5.store(x.load().to(sp.input_dtype))
            rot5 = RowRotation(sp.input_dtype, sp.hadamard_signs)
            rot5.write_row(st5, r, rw5)
            rot5.rotate(st5, r, x, rot_phase)
        elif const_expr(sp.hadamard_signs is not None):
            rot.hadamard_rotate_row(x, sp.hadamard_signs)
        if const_expr(sp.e4m3_copy == "rotated" and pk is not None):
            if prow_ok:
                self.store_e4m3(x, mX8, mE4Scale, bb, q0 + r, hh, prow)
        elif const_expr(sp.e4m3_copy == "rotated"):
            self.store_e4m3(x, mX8, mE4Scale, bb, q0 + r, hh)
        if const_expr(sp.row_norm_max):
            # max over rows of ||x||^2 (the Direct-P guard's norms), of the values the codes are made from: the sum of
            # squares in qk_hadamard_quant's two-lane order (sequential fma per 64-column half, then the halves)
            lo = Float32(0.0)
            hi = Float32(0.0)
            for j in cutlass.range_constexpr(min(64, DO)):
                lo = fma_rn(x[j], x[j], lo)
            for j in cutlass.range_constexpr(64, DO):
                hi = fma_rn(x[j], x[j], hi)
            nrm = lo + hi
            nrm = cute.arch.warp_reduction_max(nrm)
            if lane == 0:
                sRed[warp] = nrm
        if const_expr(sp.tile_amax):
            m = Float32(0.0)
            for j in cutlass.range_constexpr(DO):
                m = cute.arch.fmax(m, cute.math.absf(x[j]))
            m = cute.arch.warp_reduction_max(m)
            if lane == 0:
                sRed[4 + warp] = m
        # global (outer) scale and the per-32-row factors: x / outer / f (f a power of two: exact)
        if const_expr(sp.global_scale == "device"):
            if const_expr(sp.doc_slots):
                od = mOuter[self._e4doc[t]]     # the row's document's global scale (float32)
            else:
                od = mOuter[0]
            inv_outer = rcp_rn(od)
            outer = od
        if const_expr(sp.global_scale != "none"):
            for j in cutlass.range_constexpr(DO):
                x[j] = x[j] * inv_outer
        rfac = Float32(1.0)
        if const_expr(sp.row_factors):
            rfac = mRowFac[(bb * nheads + hh) * (seqlen // 32) + (q0 + r) // 32]
            rinv = Float32(1.0) / rfac
            for j in cutlass.range_constexpr(DO):
                x[j] = x[j] * rinv

        # ------------------------------------------------------------------ the row copy (and the 2D block codes)
        base_idx = (Int64(bb * nheads + hh) * Int64(seqlen) + Int64(q0)) * Int64(4096)
        words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        need_codes = sp.row_copy or sp.d_output or (sp.transposed_copy and sp.blocks_2d)
        if const_expr(need_codes):
            self.encode_row(x, words, sfw, r, lane, base_idx, seed)
        if const_expr(sp.row_copy and sp.packed):
            # (packed, cov1) the codes at the packed row; the SF words dense per packed row (word k = atom k's row word)
            katoms = DO // (4 * G)
            if prow_ok:
                if const_expr(DO == 128):
                    cute.autovec_copy(words, cute.make_tensor(mQ[0, prow, hh, None].iterator, cute.make_layout(16)))
                else:
                    wo = cute.make_rmem_tensor(cute.make_layout(DO // 8), Int32)
                    for w in cutlass.range_constexpr(DO // 8):
                        wo[w] = words[w]
                    cute.autovec_copy(wo, cute.make_tensor(mQ[0, prow, hh, None].iterator, cute.make_layout(DO // 8)))
                for k in cutlass.range_constexpr(katoms):
                    mSF[hh, prow * katoms + k] = sfw[k]
        elif const_expr(sp.row_copy):
            if const_expr(DO == 128):
                cute.autovec_copy(words, cute.make_tensor(mQ[bb, q0 + r, hh, None].iterator, cute.make_layout(16)))
            else:
                wo = cute.make_rmem_tensor(cute.make_layout(DO // 8), Int32)
                for w in cutlass.range_constexpr(DO // 8):
                    wo[w] = words[w]
                cute.autovec_copy(wo, cute.make_tensor(mQ[bb, q0 + r, hh, None].iterator, cute.make_layout(DO // 8)))
            i = r % 32
            j = r // 32
            katoms = DO // (4 * G)
            NC = sp.head_dim_chunks
            base_sf = ((((bb * (nheads // NC) + hh // NC) * tiles + t) * NC + hh % NC) * (128 * katoms))
            for k in cutlass.range_constexpr(katoms):
                mSF[base_sf + 128 * k + i * 4 + j] = sfw[k]
        if const_expr(sp.d_output and pk is not None):
            dsum_row = self.d_row(words, sfw, mO, mD, bb, hh, prow_c, nheads, seqlen, outer, rfac, pk_ok=prow_ok)
            if const_expr(sp.dense_row_bound):   # (cov6 packed)
                gpart = self.dense_bound_row(dsum_row, dense_don, rsc, inv_outer, mRowF, mUinv, mQAmax, ridx,
                                             pk_ok=prow_ok)
        elif const_expr(sp.d_output):
            dsum_row = self.d_row(words, sfw, mO, mD, bb, hh, q0 + r, nheads, seqlen, outer, rfac)
            if const_expr(sp.dense_row_bound):
                # mQAmax / mRowF are this mode's D' and bound_q outputs (the causal mode's inputs / F_q)
                gpart = self.dense_bound_row(dsum_row, dense_don, rsc, inv_outer, mRowF, mUinv, mQAmax, ridx)

        # ------------------------------------------------------------------ the transposed copy
        if const_expr(sp.transposed_copy and not sp.blocks_2d):
            if const_expr(sp.rotated_tile_in_transposed_tile):
                cute.arch.sync_threads()        # every thread has read its rotated row back from the shared tile
            self.transposed_from_values(sT, x, rowb, mQt, mSFt, bb, hh, t, tiles, nheads, base_idx, seed, pk)
        if const_expr(sp.transposed_copy and sp.blocks_2d):
            self.transposed_from_codes(sB, sS, words, sfw, mQt, mSFt, bb, hh, t, tiles, nheads, pk)

        # ------------------------------------------------------------------ CTA reductions
        if const_expr(sp.global_max != "none"):
            gw = cute.arch.warp_reduction_max(gpart)
            if lane == 0:
                sRed[16 + warp] = gw
            cute.arch.sync_threads()
            if const_expr(pk is not None):   # (packed: the device tile count's dense partial index)
                self.global_max_tile(sRed, mPartials, mCounter, mGlobalOut, mGlobalOut64, hh * pk[2] + t,
                                     pk[2] * nheads, pk[2])
            else:
                self.global_max_tile(sRed, mPartials, mCounter, mGlobalOut, mGlobalOut64,
                                     (bb * nheads + hh) * tiles + t, tiles * nheads * mX.shape[0], tiles)
        if const_expr(sp.causal_row_scale):
            cute.arch.sync_threads()
            if tidx == 0:
                qa4 = cute.arch.fmax(cute.arch.fmax(sRed[8], sRed[9]), cute.arch.fmax(sRed[10], sRed[11]))
                da4 = cute.arch.fmax(cute.arch.fmax(sRed[12], sRed[13]), cute.arch.fmax(sRed[14], sRed[15]))
                if const_expr(sp.doc_slots):
                    # this tile's document's pair of slots
                    dslot = 2 * self._e4doc[t]
                    red_max_nonneg_f32(qa4, mAmaxSlots.iterator + dslot)
                    red_max_nonneg_f32(da4, mAmaxSlots.iterator + dslot + 1)
                else:
                    red_max_nonneg_f32(qa4, mAmaxSlots.iterator)
                    red_max_nonneg_f32(da4, mAmaxSlots.iterator + 1)
        if const_expr(sp.row_norm_max or sp.tile_amax):
            cute.arch.sync_threads()
            if tidx == 0:
                if const_expr(sp.row_norm_max):
                    n4 = cute.arch.fmax(cute.arch.fmax(sRed[0], sRed[1]), cute.arch.fmax(sRed[2], sRed[3]))
                    red_max_nonneg_f32(n4, mNorm.iterator)
                if const_expr(sp.tile_amax):
                    a4 = cute.arch.fmax(cute.arch.fmax(sRed[4], sRed[5]), cute.arch.fmax(sRed[6], sRed[7]))
                    mTileAmax[(bb * nheads + hh) * tiles + t] = a4

    # ------------------------------------------------------------------------------------------------------------------
    @cute.jit
    def rotate_tile_tc(self, sXb: cute.Tensor, sR32: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, q0):
        """hadamard_rotation_method="tensor_core": warp w rotates the staged rows 32 w .. 32 w + 31 as two 16-row quad
        groups (_rotations.quad_hadamard_rotate; RoPE first, in the lane, as the thread-per-row path; columns past the
        input head dim are the zero padding) into the fp32 tile sR32, whose row r thread r reads back in place of
        hadamard_rotate_row. The rows are the warp's own: a warp barrier suffices."""
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp = tidx // 32
        gq = lane // 4
        tq = lane % 4
        DI = sp.head_dim_in
        dt = sp.input_dtype
        sw = rot.quad_sign_words(sp.hadamard_signs)
        for k in cutlass.range_constexpr(2):
            a = cute.make_rmem_tensor(cute.make_layout(32), Int32)
            for i in cutlass.range_constexpr(2):
                row = 32 * warp + 16 * k + gq + 8 * i
                for m in cutlass.range_constexpr(4):
                    if const_expr(32 * m + 32 <= DI):
                        cute.autovec_copy(_aligned16(Int32, sXb[row, None].iterator + 32 * m + 8 * tq, 4,
                                                     cute.AddressSpace.smem),
                                          cute.make_tensor(a.iterator + 16 * i + 4 * m, cute.make_layout(4)))
                    else:
                        for e in cutlass.range_constexpr(4):
                            a[16 * i + 4 * m + e] = Int32(0)
            if const_expr(sp.rope != "none"):
                rot.quad_rope(a, tq, q0 + 32 * warp + 16 * k + gq, q0 + 32 * warp + 16 * k + gq + 8, mCos, mSin,
                              sp.rope == "transposed_tables", dt)
            v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
            rot.quad_hadamard_rotate(a, v, lane, sw, dt is cutlass.Float16)
            for i in cutlass.range_constexpr(2):
                row = 32 * warp + 16 * k + gq + 8 * i
                for m in cutlass.range_constexpr(4):
                    cute.autovec_copy(cute.make_tensor(v.iterator + 32 * i + 8 * m, cute.make_layout(8)),
                                      _aligned16(Float32, sR32[row, None].iterator + 32 * m + 8 * tq, 8,
                                                 cute.AddressSpace.smem))
        cute.arch.sync_warp()

    @cute.jit
    def zero_accumulator_row(self, mDqAcc: cute.Tensor, hh, prow):
        """(cov6 packed) Zero accumulator row prow of head hh ((1, h, T', 128)): the thread's own row."""
        et = mDqAcc.element_type
        vec = 128 // et.width
        z = cute.make_rmem_tensor(cute.make_layout(vec), et)
        z.fill(0)
        for k in cutlass.range_constexpr(128 // vec):
            cute.autovec_copy(z, _aligned16(et, mDqAcc[0, hh, prow, None].iterator + k * vec, vec, cute.AddressSpace.gmem))

    @cute.jit
    def zero_accumulator_tile(self, mDqAcc: cute.Tensor, bb, hh, q0):
        """Zero the tile's 128 accumulator rows (b, h, q0:q0 + 128, 128): one contiguous block, written by the CTA in
        consecutive 16-byte chunks (coalesced)."""
        tidx, _, _ = cute.arch.thread_idx()
        et = mDqAcc.element_type
        vec = 128 // et.width
        base = mDqAcc[bb, hh, q0, None].iterator
        z = cute.make_rmem_tensor(cute.make_layout(vec), et)
        z.fill(0)
        for k in cutlass.range_constexpr(128 * 128 // vec // 128):
            cute.autovec_copy(z, _aligned16(et, base + (k * 128 + tidx) * vec, vec, cute.AddressSpace.gmem))

    @cute.jit
    def global_max_tile(self, sRed: cute.Tensor, mPartials: cute.Tensor, mCounter: cute.Tensor, mGlobalOut: cute.Tensor,
                        mGlobalOut64: cute.Tensor, tile, total, tiles=None):
        """This tile's max into mPartials[tile]; the tile that finishes last reduces every partial (max is exact in any
        order), writes the result and resets the persistent counter to 0 for the next launch on this stream."""
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        # (spd4) packed rows: the caller pre-fills the document slots (0) and their scales (1e-30, the zero max's) on the
        # stream, so the last tile neither zeroes all N + 1 slots nor derives all N + 1 scales -- a serial tail of ~0.2 ms
        # at the fork's capacity N = T (T = 65536) -- only the scales of the tiles' documents
        pk_docs = const_expr(sp.doc_slots and sp.packed)
        if tidx == 0:
            m4 = cute.arch.fmax(cute.arch.fmax(sRed[16], sRed[17]), cute.arch.fmax(sRed[18], sRed[19]))
            st_relaxed_f32(mPartials.iterator + tile, m4)
            done = cute.arch.atomic_add(mCounter.iterator, Int32(1), sem="acq_rel", scope="gpu")
            sRed[20] = Float32(0.0)
            if done == total - 1:
                sRed[20] = Float32(1.0)
        cute.arch.sync_threads()
        if const_expr(sp.doc_slots):
            # documents (b = 1): one max per document of the tile (mE4Doc); the last tile zeroes the (n_docs + 1) outputs,
            # max-reduces every partial into its document's slot, then derives the float32 scales
            if sRed[20] == Float32(1.0):
                fence_acq_rel_gpu()
                nd1 = mGlobalOut.shape[0]
                if const_expr(not pk_docs):
                    for i in cutlass.range((nd1 - tidx + 127) // 128, unroll=1):
                        st_relaxed_f32(mGlobalOut.iterator + (tidx + i * 128), Float32(0.0))
                    fence_acq_rel_gpu()
                    cute.arch.sync_threads()
                for i in cutlass.range((total - tidx + 127) // 128, unroll=1):
                    k = tidx + i * 128
                    red_max_nonneg_f32(ld_relaxed_f32(mPartials.iterator + k), mGlobalOut.iterator + self._e4doc[k % tiles])
                fence_acq_rel_gpu()
                cute.arch.sync_threads()
                if const_expr(sp.global_max == "abs" and sp.global_scale_divisor > 0.0):
                    if const_expr(pk_docs):
                        # (spd4) the scales of the tiles' documents only (the others keep the pre-filled zero max's; a
                        # document of several tiles is written once per tile, with the same value)
                        for i in cutlass.range((tiles - tidx + 127) // 128, unroll=1):
                            dslot = self._e4doc[tidx + i * 128]
                            gslot = ld_relaxed_f32(mGlobalOut.iterator + dslot)
                            mGlobalOut64[dslot] = cute.arch.fmax(div_rn_f32(gslot, Float32(sp.global_scale_divisor)),
                                                                 Float32(1e-30))
                    else:
                        for i in cutlass.range((nd1 - tidx + 127) // 128, unroll=1):
                            g = ld_relaxed_f32(mGlobalOut.iterator + (tidx + i * 128))
                            mGlobalOut64[tidx + i * 128] = cute.arch.fmax(
                                div_rn_f32(g, Float32(sp.global_scale_divisor)), Float32(1e-30))
                if tidx == 0:
                    mCounter[0] = Int32(0)
        elif sRed[20] == Float32(1.0):
            fence_acq_rel_gpu()
            m = Float32(0.0)
            for i in cutlass.range((total - tidx + 127) // 128, unroll=1):
                m = cute.arch.fmax(m, ld_relaxed_f32(mPartials.iterator + (tidx + i * 128)))
            m = cute.arch.warp_reduction_max(m)
            if tidx % 32 == 0:
                sRed[24 + tidx // 32] = m
            cute.arch.sync_threads()
            if tidx == 0:
                g = cute.arch.fmax(cute.arch.fmax(sRed[24], sRed[25]), cute.arch.fmax(sRed[26], sRed[27]))
                mGlobalOut[0] = g
                if const_expr(sp.global_max == "abs" and sp.global_scale_divisor > 0.0):
                    # max(amax / divisor, 1e-30) in float32 (div.rn; device_scales.div32(amax, divisor).clamp_min)
                    # (mGlobalOut64: the global scale output, float32 since the FP64 removal)
                    mGlobalOut64[0] = cute.arch.fmax(div_rn_f32(g, Float32(sp.global_scale_divisor)), Float32(1e-30))
                mCounter[0] = Int32(0)

    @cute.jit
    def dense_bound_row(self, dsum: Float32, don: Float32, rsc: cute.Tensor, rinv: Float32, mBoundRow: cute.Tensor,
                        mUinv: cute.Tensor, mDp: cute.Tensor, ridx, pk_ok=None) -> Float32:
        """fp4q_prepare's global-mode per-row chain in torch's fp32 order: lse2 = lse f32(log2 e);
        P_max = 2^min(frame + 8 - lse2, 0); v_max = sqrt.rn(max ||V||^2) (from the forward's V quantiser);
        bound_q = (P_max ((||dO|| v_max) + |D|)) * rinv, rinv = rcp.rn(s_dO) (torch's t * reciprocal(s_dO));
        D' = D * rinv; uinv = 2^min(frame - lse2 - offset, 1000) (0 where lse2 is not finite). Returns bound_q."""
        sp = self.s
        lse = rsc[0]
        frame = rsc[1]
        lse2 = mul_rn(lse, Float32(LOG2_E_F32))
        pexp = cute.arch.fmin(sub_rn(add_rn(frame, Float32(8.0)), lse2), Float32(0.0))
        pmax_b = cute.math.pow(Float32(2.0), pexp)
        vmax = _asm_f32("sqrt.rn.f32", rsc[5])
        bq = mul_rn(mul_rn(pmax_b, add_rn(mul_rn(don, vmax), cute.math.absf(dsum))), rinv)
        if const_expr(pk_ok is None):
            mBoundRow[ridx] = bq
            mDp[ridx] = mul_rn(dsum, rinv)
        uexp = cute.arch.fmin(sub_rn(sub_rn(frame, lse2), Float32(sp.lse_log2_offset_dense)), Float32(1000.0))
        upow = cute.math.pow(Float32(2.0), uexp)
        uinv = Float32(0.0)
        if cute.math.isfinite(lse2):
            uinv = upow
        if const_expr(pk_ok is not None):   # (cov6 packed: the real rows only)
            if pk_ok:
                mBoundRow[ridx] = bq
                mDp[ridx] = mul_rn(dsum, rinv)
                mUinv[ridx] = uinv
        else:
            mUinv[ridx] = uinv
        return bq

    # ------------------------------------------------------------------------------------------------------------------
    @cute.jit
    def causal_row_scale(self, x: cute.Tensor, rowb: cute.Tensor, mO: cute.Tensor, rsc: cute.Tensor,
                         mRowF: cute.Tensor, mUinv: cute.Tensor, mDqAcc: cute.Tensor, sRed: cute.Tensor, bb, hh, q,
                         nheads, seqlen, lane, warp, pk_ok=None):
        """The causal dS scale of row q (fp4q_prepare's former torch chain, op by op in its fp32 order):
          lse2 = lse * f32(log2 e); P_max bound 2^min(frame + 8 - lse2, 0); bound = P_max (||dO|| vmax_q + |D_est|)
          (fold: D_est x rmax_q), D_est = sum O dO and ||dO|| in torch's reduction order;
          f = min(448 / max(bound, 1e-30), top / max(max|dO|, 1e-30)) (1 where bound or max|dO| is 0), clamped to
          [2^-100, 2^100]; F_q = 2^floor(log2 f); uinv = 2^min(frame - lse2 - offset, 1000) (0 where lse2 is not finite).
        x <- f32(bf16(x F_q)) in place (exact for a power of two outside the subnormal range, rounded as torch's
        .to(dtype)). Also the row's max|Q| / F_q (bf16-rounded, as (q / F_q).to(bf16)) and max|dO| into sRed[8..15], and
        the row's dQ accumulator zeroed."""
        sp = self.s
        ridx = (bb * nheads + hh) * seqlen + q
        if const_expr(pk_ok is not None):   # (cov5 packed: q is the packed row (h, T) tables; written for real rows)
            ridx = hh * (mRowF.shape[0] // nheads) + q
        # sum O dO in torch's order, from 16-byte chunks of the O row (lane g of the 32 torch lanes = values 4g..4g+3):
        # no 128-wide product array (register pressure of the one-thread-per-row tile)
        lanes = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        o_it = mO[bb, q, hh, None].iterator
        for c in cutlass.range_constexpr(16):
            ob = cute.make_rmem_tensor(cute.make_layout(8), sp.output_dtype_o)
            cute.autovec_copy(cute.make_tensor(o_it + 8 * c, cute.make_layout(8)), ob)
            for hg in cutlass.range_constexpr(2):
                g = 2 * c + hg
                p0 = mul_rn(x[4 * g], Float32(ob[4 * hg]))               # exact: bf16 x bf16
                p1 = mul_rn(x[4 * g + 1], Float32(ob[4 * hg + 1]))
                p2 = mul_rn(x[4 * g + 2], Float32(ob[4 * hg + 2]))
                p3 = mul_rn(x[4 * g + 3], Float32(ob[4 * hg + 3]))
                lanes[g] = add_rn(add_rn(add_rn(p0, p1), p2), p3)
        d_est = xor_tree_sum(lanes, 32, TORCH_LAST_DIM_ORDER)
        dmax = Float32(0.0)
        for j in cutlass.range_constexpr(128):
            dmax = cute.arch.fmax(dmax, cute.math.absf(x[j]))
        do_n = torch_row_norm2(x)
        lse = rsc[0]
        frame = rsc[1]
        lse2 = mul_rn(lse, Float32(LOG2_E_F32))
        pexp = cute.arch.fmin(sub_rn(add_rn(frame, Float32(8.0)), lse2), Float32(0.0))
        pmax_b = cute.math.pow(Float32(2.0), pexp)
        if const_expr(sp.row_scale_fold):
            d_est = mul_rn(d_est, rsc[3])
        bound = mul_rn(pmax_b, add_rn(mul_rn(do_n, rsc[2]), cute.math.absf(d_est)))
        f1 = mul_rn(rcp_rn(cute.arch.fmax(bound, Float32(1e-30))), Float32(448.0))
        f2 = mul_rn(rcp_rn(cute.arch.fmax(dmax, Float32(1e-30))), Float32(sp.row_scale_top))
        fmin12 = cute.arch.fmin(f1, f2)
        f = Float32(1.0)
        if bound > Float32(0.0):
            if dmax > Float32(0.0):
                f = fmin12
        f = cute.arch.fmin(cute.arch.fmax(f, Float32(2.0 ** -100)), Float32(2.0 ** 100))
        fq = pow2_floor_as_torch(f)
        if const_expr(pk_ok is not None):
            if pk_ok:
                mRowF[ridx] = fq
        else:
            mRowF[ridx] = fq
        uexp = cute.arch.fmin(sub_rn(sub_rn(frame, lse2), Float32(sp.lse_log2_offset)), Float32(1000.0))
        upow = cute.math.pow(Float32(2.0), uexp)
        uinv = Float32(0.0)
        if cute.math.isfinite(lse2):
            uinv = upow
        if const_expr(pk_ok is not None):
            if pk_ok:
                mUinv[ridx] = uinv
        else:
            mUinv[ridx] = uinv
        # dO row x F_q, rounded to the input dtype (torch: (dout.float() * F).to(dout.dtype))
        for j in cutlass.range_constexpr(128):
            x[j] = mul_rn(x[j], fq)
        rowb.store(x.load().to(sp.input_dtype))
        x.store(rowb.load().to(Float32))
        # the per-tensor E4M3 amaxes of the dK / dV operands: max |Q| / F_q (bf16, as (q / F_q).to(bf16)) and max |dO|
        qf = round_to_bf16(div_rn_f32(rsc[4], fq))
        qf = cute.arch.warp_reduction_max(qf)
        dm = cute.arch.warp_reduction_max(dmax)
        if lane == 0:
            sRed[8 + warp] = qf
            sRed[12 + warp] = dm
        if const_expr(sp.zero_dq_accum and pk_ok is not None):
            # (cov5 packed: the real rows' accumulator rows (1, h, T', 128), each thread its own row)
            if pk_ok:
                et = mDqAcc.element_type
                vec = 128 // et.width
                z = cute.make_rmem_tensor(cute.make_layout(vec), et)
                z.fill(0)
                for k in cutlass.range_constexpr(128 // vec):
                    cute.autovec_copy(z, _aligned16(et, mDqAcc[0, hh, q, None].iterator + k * vec, vec,
                                                    cute.AddressSpace.gmem))
        elif const_expr(sp.zero_dq_accum):
            tz, _, _ = cute.arch.thread_idx()
            self.zero_accumulator_tile(mDqAcc, bb, hh, q - tz)

    # ------------------------------------------------------------------------------------------------------------------
    @cute.jit
    def store_e4m3(self, x: cute.Tensor, mX8: cute.Tensor, mSc: cute.Tensor, bb, q, hh, qw=None):
        # 1 / s = rcp.rn.f32 of the float32 scale (the Triton quantisers' multiplier, div_rn(1, s)); the
        # cvt.rn.satfinite equals Triton's clamp-to-448-then-convert
        if const_expr(self.s.e4m3_doc):
            # per-(document, kv head) scale: the row's document (its 128-row tile's) and kv head
            inv = rcp_rn(mSc[self._e4doc[q // 128], hh // (mX8.shape[2] // mSc.shape[1])])
        else:
            inv = rcp_rn(mSc[0])
        nw = self.s.head_dim_out // 4
        w8 = cute.make_rmem_tensor(cute.make_layout(nw), Int32)
        for w in cutlass.range_constexpr(nw):
            if const_expr((self.s.hadamard_signs is None or self.s.e4m3_copy == "unrotated") and 4 * w >= self.s.head_dim_in):
                w8[w] = Int32(0)                    # the zero padding (written explicitly)
            else:
                w8[w] = packed_float_to_ue4m3(x[4 * w] * inv, x[4 * w + 1] * inv, x[4 * w + 2] * inv,
                                              x[4 * w + 3] * inv)
        cute.autovec_copy(w8, cute.make_tensor(mX8[bb, q if const_expr(qw is None) else qw, hh, None].iterator,
                                               cute.make_layout(nw)))

    @cute.jit
    def encode_row(self, x: cute.Tensor, words: cute.Tensor, sfw: cute.Tensor, r, lane, base_idx: Int64, seed: Int64):
        """Codes of row r (16 words) and its SF bytes packed into sfw (NVFP4: 2 words of 4 groups; MXFP4: 1 word)."""
        sp = self.s
        G = sp.scale_group_size
        DO = sp.head_dim_out
        ng = DO // G
        xf = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for j in cutlass.range_constexpr(128):
            xf[j] = x[j]
        for g in cutlass.range_constexpr(ng):
            m = Float32(0.0)
            for j in cutlass.range_constexpr(G):
                m = cute.arch.fmax(m, abs_ftz(xf[G * g + j]))
            amax[g] = m
        if const_expr(sp.blocks_2d):
            # the block amax: rows r .. r + G - 1 are G consecutive lanes of one warp
            for g in cutlass.range_constexpr(ng):
                for k in cutlass.range_constexpr(int(math.log2(G))):
                    amax[g] = cute.arch.fmax(amax[g], cute.arch.shuffle_sync_bfly(amax[g], offset=1 << k))
        sfw[0] = Int32(0)
        sfw[1] = Int32(0)
        y = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        for g in cutlass.range_constexpr(ng):
            byte = Int32(0)
            mult = Float32(0.0)
            sfv = Float32(0.0)
            if const_expr(G == 32):
                byte, mult = mxfp4_scale(amax[g])
            else:
                byte, sfv, mult = nvfp4_scale_from_amax(amax[g], sp.rounding)
            for j in cutlass.range_constexpr(G):
                y[G * g + j] = mul_ftz(xf[G * g + j], mult)
            if const_expr(sp.rounding == "stochastic"):
                if const_expr(sp.blocks_2d):
                    el0 = (base_idx + Int64((r // G) * ng + g)) * Int64(G * G) + Int64((r % G) * G)
                else:
                    el0 = (base_idx + Int64(r * ng + g)) * Int64(G)
                z0 = el0 * Int64(SPLITMIX_GOLDEN) + seed
                for j in cutlass.range_constexpr(G):
                    z = z0 + Int64(_wrap64(j * SPLITMIX_GOLDEN))
                    y[G * g + j] = stochastic_round_e2m1(y[G * g + j], uniform_from_mixed(z))
            if const_expr(sp.four_over_six):
                byte = self.four_over_six(xf, y, g, amax[g], sfv, byte, FO6_ROW_SUM_ORDER)
            if const_expr(sp.rounding == "ms_eden"):
                byte = self.eden_scale(xf, y, g, sfv, byte, base_idx + Int64(r * ng + g), seed)
            if const_expr(G == 16):
                sfw[g // 4] = sfw[g // 4] | ((byte & 0xFF) << (8 * (g % 4)))
            else:
                sfw[0] = sfw[0] | ((byte & 0xFF) << (8 * g))
        encode_e2m1_words(y, 0, DO, words, 0)

    @cute.jit
    def four_over_six(self, xf: cute.Tensor, y: cute.Tensor, g: cutlass.Constexpr[int], amax: Float32, sfv: Float32,
                      byte: Int32, order: cutlass.Constexpr) -> Int32:
        """Four-over-Six on group g: the 4-max candidate (SF4 = E4M3(amax / 4), codes clamped to +-4) replaces the 6-max one
        when its squared error is lower. Returns the chosen SF byte; y[16 g:] updated in place."""
        b4 = e4m3_rn(amax * Float32(0.25))
        sf4 = e4m3_byte_value(b4, 0)
        inv4 = Float32(0.0)
        if sf4 > Float32(0.0):
            inv4 = Float32(1.0) / sf4
        y4 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        d6 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        d4 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        for j in cutlass.range_constexpr(16):
            y4[j] = cute.arch.fmin(cute.arch.fmax(mul_ftz(xf[16 * g + j], inv4), Float32(-4.0)), Float32(4.0))
            a6 = e2m1_value(y[16 * g + j]) * sfv - abs_ftz_signed(xf[16 * g + j])
            a4 = e2m1_value(y4[j]) * sf4 - abs_ftz_signed(xf[16 * g + j])
            d6[j] = mul_rn(a6, a6)
            d4[j] = mul_rn(a4, a4)
        e6 = xor_tree_sum(d6, 16, order)
        e4 = xor_tree_sum(d4, 16, order)
        out = byte
        if e4 < e6:
            out = b4
            for j in cutlass.range_constexpr(16):
                y[16 * g + j] = y4[j]
        return out

    @cute.jit
    def eden_scale(self, xf: cute.Tensor, y: cute.Tensor, g: cutlass.Constexpr[int], sfv: Float32, byte: Int32,
                   idx: Int64, seed: Int64) -> Int32:
        """MS-EDEN (Quartet II): RTN codes at the clipping scale, then the group's factor <x, x> / <x, x^> folded into the
        E4M3 scale by stochastic rounding (its own stream: index + 2^44)."""
        num_t = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        den_t = cute.make_rmem_tensor(cute.make_layout(16), Float32)
        cw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        encode_e2m1_words(y, 16 * g, 16, cw, 0)          # the codes the cvt makes (ties to even), decoded below
        for j in cutlass.range_constexpr(16):
            xv = abs_ftz_signed(xf[16 * g + j])
            num_t[j] = mul_rn(xv, xv)
            den_t[j] = mul_rn(xv, e2m1_code_value((cw[j // 8] >> (4 * (j % 8))) & 0xF))
        num = xor_tree_sum(num_t, 16, EDEN_SUM_ORDER)
        den = xor_tree_sum(den_t, 16, EDEN_SUM_ORDER) * sfv
        fac = Float32(1.0)
        if den > Float32(0.0):
            fac = div_full(num, den)
        v = e4m3_stochastic(sfv * fac, uniform_from_counter(idx + Int64(1 << 44), seed))
        return e4m3_rn(v)

    @cute.jit
    def d_row(self, words: cute.Tensor, sfw: cute.Tensor, mO: cute.Tensor, mD: cute.Tensor, bb, hh, q, nheads, seqlen,
              outer: Float32, rfac: Float32, pk_ok=None):
        """D_q = sum_d o * dequant(codes) (* outer * f): the products are exact (E2M1 x E4M3 x o), the sum in D_SUM_ORDER."""
        sp = self.s
        DI = sp.head_dim_in
        ob = cute.make_rmem_tensor(cute.make_layout(DI), sp.output_dtype_o)
        cute.autovec_copy(cute.make_tensor(mO[bb, q, hh, None].iterator, cute.make_layout(DI)), ob)
        prod = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        G = sp.scale_group_size
        for c in cutlass.range_constexpr(DI, 128):
            prod[c] = Float32(0.0)
        sfs = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for g in cutlass.range_constexpr(DI // G):
            if const_expr(G == 16):
                sfs[g] = e4m3_byte_value(sfw[g // 4], g % 4)
            else:
                sfs[g] = bits_f32(((sfw[0] >> (8 * g)) & 0xFF) << 23)
        for c in cutlass.range_constexpr(0, DI, 2):
            v0, v1 = e2m1_pair_values(words[c // 8], (c % 8) // 2)
            prod[c] = mul_rn(v0 * sfs[c // G], Float32(ob[c]))
            prod[c + 1] = mul_rn(v1 * sfs[c // G], Float32(ob[c + 1]))
        dsum = xor_tree_sum(prod, 128, D_SUM_ORDER)
        if const_expr(sp.global_scale != "none"):
            dsum = dsum * outer
        if const_expr(sp.row_factors):
            dsum = dsum * rfac
        if const_expr(pk_ok is not None):   # (cov5 packed: q the packed row of (h, T) D)
            if pk_ok:
                mD[hh * (mD.shape[0] // nheads) + q] = dsum
        else:
            mD[(bb * nheads + hh) * seqlen + q] = dsum
        return dsum

    @cute.jit
    def transposed_from_values(self, sT: cute.Tensor, x: cute.Tensor, rowb: cute.Tensor, mQt: cute.Tensor,
                               mSFt: cute.Tensor, bb, hh, t, tiles, nheads, base_idx: Int64, seed: Int64, pk=None):
        """1 x G groups along the tokens: the tile's values go through a padded smem transpose; thread dd quantises d-row
        dd (the tokens in the kernels' block order when transposed_key_order)."""
        sp = self.s
        G = sp.scale_group_size
        tidx, _, _ = cute.arch.thread_idx()
        r = tidx
        fp32 = sp.fp32_transposed_tile
        edt = Float32 if const_expr(fp32) else sp.input_dtype
        pad = 16 // (edt.width // 8)
        DO = sp.head_dim_out
        for c in cutlass.range_constexpr(DO):
            if const_expr(fp32):
                sT[c, r] = x[c]
            elif const_expr(c < sp.head_dim_in):
                sT[c, r] = rowb[c]
            else:
                sT[c, r] = edt(0.0)
        cute.arch.sync_threads()
        if tidx < DO:
            self.transposed_row_from_smem(sT, mQt, mSFt, bb, hh, t, tiles, nheads, base_idx, seed, pk)
        elif const_expr(DO < 128):
            self.store_transposed_sf(mSFt, Int32(0), Int32(0), bb, hh, t, tiles, nheads, tidx)

    @cute.jit
    def transposed_row_from_smem(self, sT: cute.Tensor, mQt: cute.Tensor, mSFt: cute.Tensor, bb, hh, t, tiles, nheads,
                                 base_idx: Int64, seed: Int64, pk=None):
        sp = self.s
        G = sp.scale_group_size
        tidx, _, _ = cute.arch.thread_idx()
        edt = sT.element_type
        colb = cute.make_rmem_tensor(cute.make_layout(128), edt)
        row_s8 = cute.logical_divide(sT[tidx, None], cute.make_layout(8))
        colb8 = cute.logical_divide(colb, cute.make_layout(8))
        for blk in cutlass.range_constexpr(4):
            src_blk = _perm_block(blk) if const_expr(sp.transposed_key_order) else blk
            for v in cutlass.range_constexpr(4):
                cute.autovec_copy(row_s8[None, src_blk * 4 + v], colb8[None, blk * 4 + v])
        xt = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        xt.store(colb.load().to(Float32))
        ng = 128 // G
        amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for g in cutlass.range_constexpr(ng):
            m = Float32(0.0)
            for j in cutlass.range_constexpr(G):
                m = cute.arch.fmax(m, abs_ftz(xt[G * g + j]))
            amax[g] = m
        y = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        sfw0 = Int32(0)
        sfw1 = Int32(0)
        dd = tidx
        for g in cutlass.range_constexpr(ng):
            byte = Int32(0)
            mult = Float32(0.0)
            sfv = Float32(0.0)
            if const_expr(G == 32):
                byte, mult = mxfp4_scale(amax[g])
            else:
                byte, sfv, mult = nvfp4_scale_from_amax(amax[g], sp.rounding)
            for j in cutlass.range_constexpr(G):
                y[G * g + j] = mul_ftz(xt[G * g + j], mult)
            if const_expr(sp.rounding == "stochastic"):
                idx = base_idx + Int64(1 << 40) + Int64(dd * ng + g)
                z0 = (idx * Int64(G)) * Int64(SPLITMIX_GOLDEN) + seed
                for j in cutlass.range_constexpr(G):
                    z = z0 + Int64(_wrap64(j * SPLITMIX_GOLDEN))
                    y[G * g + j] = stochastic_round_e2m1(y[G * g + j], uniform_from_mixed(z))
            if const_expr(sp.four_over_six):
                byte = self.four_over_six(xt, y, g, amax[g], sfv, byte, FO6_TRANSPOSED_SUM_ORDER)
            if const_expr(sp.rounding == "ms_eden"):
                byte = self.eden_scale(xt, y, g, sfv, byte, base_idx + Int64(1 << 40) + Int64(dd * ng + g), seed)
            if const_expr(g < 4):
                sfw0 = sfw0 | ((byte & 0xFF) << (8 * g))
            else:
                sfw1 = sfw1 | ((byte & 0xFF) << (8 * (g - 4)))
        words_t = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        encode_e2m1_words(y, 0, 128, words_t, 0)
        if const_expr(pk is not None):
            self.transposed_row_packed_values(sT, words_t, sfw0, sfw1, mQt, mSFt, hh, t, nheads, pk)
        else:
            cute.autovec_copy(words_t, cute.make_tensor(mQt[bb, dd, hh, None].iterator + t * 16, cute.make_layout(16)))
            self.store_transposed_sf(mSFt, sfw0, sfw1, bb, hh, t, tiles, nheads, dd)

    @cute.jit
    def transposed_row_packed_values(self, sT: cute.Tensor, words_t: cute.Tensor, sfw0, sfw1, mQt: cute.Tensor,
                                     mSFt: cute.Tensor, hh, t, nheads, pk):
        """(cov6 packed, the 1 x 16 copy) d row dd's 64 code bytes (natural key order) into the key-parity copies as
        transposed_row_packed (staged through the thread's own row of the transposed tile: read already), its 8 group
        SF bytes into the group table (h, G_b, 128) at slots (pb >> 4) + t + g -- the groups with a real key only (a
        later slot is the next tile's)."""
        tidx, _, _ = cute.arch.thread_idx()
        dd = tidx
        pb, nvr = pk[0], pk[1]
        c = pb & 1
        stg = cute.recast_ptr(sT[dd, None].iterator, dtype=Int32)
        cute.autovec_copy(words_t, cute.make_tensor(stg, cute.make_layout(16)))
        store_bytes_unaligned(mQt[c, dd, hh, None].iterator, (pb + c) >> 1, (nvr + 1) >> 1,
                              cute.recast_ptr(stg, dtype=cutlass.Uint8))
        gb = cute.size(mSFt.shape) // (nheads * 32)
        g8 = cute.make_tensor(cute.recast_ptr(mSFt.iterator, dtype=cutlass.Uint8), cute.make_layout(1 << 30))
        base = (hh * gb + (pb >> 4) + t) * 128 + dd
        for g in cutlass.range_constexpr(8):
            if Int32(16 * g) < nvr:
                w = sfw0 if const_expr(g < 4) else sfw1
                g8[base + 128 * g] = cutlass.Uint8((w >> (8 * (g % 4))) & 0xFF)

    @cute.jit
    def store_transposed_sf(self, mSFt: cute.Tensor, sfw0: Int32, sfw1: Int32, bb, hh, t, tiles, nheads, dd):
        sp = self.s
        if const_expr(sp.scale_group_size == 16):
            # atoms (b, h, 1, s / 64, 32, 4, 4): token group q0 / 16 + g -> atom 2t + g // 4, byte g % 4
            base_t = ((bb * nheads + hh) * (tiles * 2) + t * 2) * 128 + (dd % 32) * 4 + dd // 32
            mSFt[base_t] = sfw0
            mSFt[base_t + 128] = sfw1
            if const_expr(sp.transposed_sf_halves_swapped):
                half = cute.size(mSFt.shape) // 2
                base_s = ((bb * nheads + hh) * (tiles * 2) + t * 2) * 128 + (dd % 32) * 4 + (dd // 32 + 2) % 4
                mSFt[half + base_s] = sfw0
                mSFt[half + base_s + 128] = sfw1
        else:
            # atoms (b, h, 1, s / 128, 32, 4, 4): token group q0 / 32 + g -> atom t, byte g
            base_t = ((bb * nheads + hh) * tiles + t) * 128 + (dd % 32) * 4 + dd // 32
            mSFt[base_t] = sfw0

    @cute.jit
    def transposed_from_codes(self, sB: cute.Tensor, sS: cute.Tensor, words: cute.Tensor, sfw: cute.Tensor,
                              mQt: cute.Tensor, mSFt: cute.Tensor, bb, hh, t, tiles, nheads, pk=None):
        """2D blocks: the transposed copy carries the SAME codes (re-packed in (d, token) pairs) and the block scales,
        replicated into the transposed SF layout. pk (packed, cov5): the codes into the key-parity copies, no SF atom
        (transposed_row_from_codes)."""
        sp = self.s
        G = sp.scale_group_size
        tidx, _, _ = cute.arch.thread_idx()
        r = tidx
        DO = sp.head_dim_out
        # sB[c, k]: the code byte of d column c and token pair k = (row 2k low nibble, row 2k + 1 high nibble)
        nh = DO // 16                                   # words of half a row
        odd = r % 2
        k = r // 2
        for w in cutlass.range_constexpr(nh):
            send = words[w]
            if odd == 0:
                send = words[w + nh]
            recv = cute.arch.shuffle_sync_bfly(send, offset=1)
            mine = words[w + nh]
            if odd == 0:
                mine = words[w]
            lo_w = mine
            hi_w = recv
            if odd == 1:
                lo_w = recv
                hi_w = mine
            for e in cutlass.range_constexpr(8):
                byte = ((lo_w >> (4 * e)) & 0xF) | (((hi_w >> (4 * e)) & 0xF) << 4)
                c = 8 * w + e + odd * (DO // 2)          # the even lane writes columns [0, DO/2), the odd lane the rest
                sB[c, k] = cutlass.Uint8(byte)
        sS[r, 0] = sfw[0]
        sS[r, 1] = sfw[1]
        cute.arch.sync_threads()
        if const_expr(pk is not None):
            if tidx < DO:
                self.transposed_row_packed(sB, mQt, hh, pk)
        elif tidx < DO:
            self.transposed_row_from_codes(sB, sS, mQt, mSFt, bb, hh, t, tiles, nheads)
        elif const_expr(DO < 128):
            self.store_transposed_sf(mSFt, Int32(0), Int32(0), bb, hh, t, tiles, nheads, tidx)

    @cute.jit
    def transposed_row_packed(self, sB: cute.Tensor, mQt: cute.Tensor, hh, pk):
        """(packed, cov5) d row dd of the tile's transposed codes into the KEY-PARITY copies (2, h, d, TB) bytes: the
        tile's document starts at packed key ps (= pb - 128 j), its codes go to copy c = ps & 1 at key column k + c, so
        every document starts on a byte boundary of its copy and no byte holds two documents' codes (the next document
        of the same copy starts past this one's last byte). Only the ceil(nvr / 2) bytes of the real keys are written
        (their last high nibble, past an odd nvr, is a zero row's code). The consumer (flash_fwd_sm100_fp4 v4_packed)
        reads the tile from byte (pb + c) / 2 of copy c. The SF bytes: the row copy's dense per-row words."""
        tidx, _, _ = cute.arch.thread_idx()
        dd = tidx
        pb, nvr = pk[0], pk[1]
        c = pb & 1
        store_bytes_unaligned(mQt[c, dd, hh, None].iterator, (pb + c) >> 1, (nvr + 1) >> 1, sB[dd, None].iterator)

    @cute.jit
    def transposed_row_from_codes(self, sB: cute.Tensor, sS: cute.Tensor, mQt: cute.Tensor, mSFt: cute.Tensor, bb, hh, t,
                                  tiles, nheads):
        sp = self.s
        G = sp.scale_group_size
        tidx, _, _ = cute.arch.thread_idx()
        dd = tidx
        words_t = cute.make_rmem_tensor(cute.make_layout((4, 4)), Int32)
        for blk in cutlass.range_constexpr(4):             # 32 tokens = 16 bytes per block
            src_blk = _perm_block(blk) if const_expr(sp.transposed_key_order) else blk
            cute.autovec_copy(_aligned16(Int32, sB[dd, None].iterator + 16 * src_blk, 4, cute.AddressSpace.smem),
                              words_t[None, blk])
        cute.autovec_copy(words_t, cute.make_tensor(mQt[bb, dd, hh, None].iterator + t * 16, cute.make_layout((4, 4))))
        # SF of (d row dd, token group): the block scale of (row block, d group dd // G)
        g = dd // G
        sfw0 = Int32(0)
        sfw1 = Int32(0)
        ngt = 128 // G
        for gt in cutlass.range_constexpr(ngt):
            blk_rows = G * gt                                   # first output token of the group
            pblk = blk_rows // 32
            src_blk = _perm_block(pblk) if const_expr(sp.transposed_key_order) else pblk
            src = src_blk * 32 + blk_rows % 32
            if const_expr(G == 16):
                byte = (sS[src, g // 4] >> (8 * (g % 4))) & 0xFF
            else:
                byte = (sS[src, 0] >> (8 * g)) & 0xFF
            if const_expr(gt < 4):
                sfw0 = sfw0 | (byte << (8 * gt))
            else:
                sfw1 = sfw1 | (byte << (8 * (gt - 4)))
        self.store_transposed_sf(mSFt, sfw0, sfw1, bb, hh, t, tiles, nheads, dd)


@cute.jit
def abs_ftz_signed(a):
    """a with a subnormal value flushed to a signed zero (mul.ftz by 1)."""
    return mul_ftz(a, Float32(1.0))


FO6_ROW_SUM_ORDER = (2, 4, 1, 8)         # nvfp4_quant_triton FO6, row copy (recovered: probe_fo6r, 6 seeds x 1M groups)
FO6_TRANSPOSED_SUM_ORDER = (1, 8, 4, 2)  # ... transposed copy (probe_fo6t3: the only order with 0 mismatches)
EDEN_SUM_ORDER = (2, 4, 1, 8)            # MS-EDEN <x, x>, <x, x^> (any order matched the old SF bytes: probe_eden)


@dsl_user_op
def red_max_nonneg_f32(val: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Max-reduce a non-negative fp32 into a gmem slot that starts at 0 (non-negative floats order as their u32 bits),
    issued only when it beats the slot's current value (one address for the whole grid)."""
    llvm.inline_asm(
        None, [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), Float32(val).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 v, c;\n\t.reg .pred q;\n\tmov.b32 v, $1;\n\tld.volatile.global.b32 c, [$0];\n\t"
        "setp.gt.u32 q, v, c;\n\t@q red.global.max.u32 [$0], v;\n\t}\n",
        "l,f", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


# ======================================================================================================================
# Host side
# ======================================================================================================================
@dataclass
class CausalRowScale:
    """Inputs of quantize_fp4_rows' causal dS row scale (the all-FP4 backward's dO launch; fp4q_prepare). Per query row
    (b, h, s) float32, contiguous: lse (nats) and frame (log2) from the forward, key_norm_prefix_max (vmax_q) and, for the
    K fold, key_factor_prefix_max (rmax_q) from the forward's diagonal quantiser, query_row_amax (max |Q| per row, ditto).
    max_slots: (2,) float32 zeroed by the forward, max-reduced with max |Q| / F_q and max |dO|. top: the dO-side cap
    smax x 448 (x 256 for MS-EDEN). lse_log2_offset: uinv's offset. zero_accumulator: the (b, h, s, 128) dQ accumulator
    to zero (or None). Outputs: "row_factor" F_q and "uinv", (b, h, s) float32."""
    lse: torch.Tensor
    frame: torch.Tensor
    key_norm_prefix_max: torch.Tensor
    query_row_amax: torch.Tensor
    max_slots: torch.Tensor
    top: float
    lse_log2_offset: float
    key_factor_prefix_max: Optional[torch.Tensor] = None
    zero_accumulator: Optional[torch.Tensor] = None


@dataclass
class DenseRowBound:
    """Inputs of quantize_fp4_rows' dense (global dS multiplier) row chain (the all-FP4 backward's non-causal dO launch,
    which runs with global_scale = s_dO as a (1,) float32 device tensor). lse / frame (b, h, s) float32 contiguous from
    the forward; value_norm2_max (1,) float32 = max_k ||V_k||^2 (the forward's V quantiser, output_tensor_max="row_norm2").
    Outputs: "bound_rows", "uinv", "d_scaled" (D / s_dO), "row_amax" (max |dO| per row), all (b, h, s) float32, and
    "tensor_max" (1,) float32 = max over rows of bound_rows (the global dS bound). zero_accumulator as CausalRowScale."""
    lse: torch.Tensor
    frame: torch.Tensor
    value_norm2_max: torch.Tensor
    lse_log2_offset: float
    zero_accumulator: Optional[torch.Tensor] = None


_TORCH2CUTE = None


def _cute_dtype(dt):
    global _TORCH2CUTE
    if _TORCH2CUTE is None:
        _TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: Float32}
    return _TORCH2CUTE[dt]


def _sf_view(raw: torch.Tensor) -> torch.Tensor:
    """Contiguous SF atoms (b, h, R / 128, K / (4 G), 32, 4, 4) -> the (32, 4, R / 128, 4, K / (4 G), h, b) view the SM100
    block-scaled kernels take."""
    return raw.permute(4, 5, 2, 6, 3, 1, 0)


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, divisibility=1):
    """torch tensor -> cute tensor for TVM FFI (interface_fwd_fp4.to_cute_tensor's construction, local for leaf imports)."""
    from cutlass.cute.runtime import from_dlpack
    if divisibility > 1 and 1 in t.shape:
        # a size-1 mode's stride is free; give it the compact value of its place in dim_order (the layout check of
        # mark_compact_shape_dynamic rejects e.g. a (1, 1, d, s) head-major view whose h stride equals the d stride)
        st, prev = list(t.stride()), None
        for m in reversed(t.dim_order()):
            if t.shape[m] == 1 and prev is not None:
                st[m] = st[prev] * t.shape[prev]
            prev = m
        t = t.as_strided(t.shape, st)
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility > 1:
        tensor = tensor.mark_compact_shape_dynamic(mode=leading_dim, stride_order=t.dim_order(), divisibility=divisibility)
    return tensor


_CACHE = None


def _jit_cache():
    global _CACHE
    if _CACHE is None:
        try:  # the library's persistent JIT cache when the package imports; a dict when loaded as a leaf
            from flash_attn.cute.cache_utils import get_jit_cache
            _CACHE = get_jit_cache("quant_fp4")
        except Exception:  # noqa: BLE001
            _CACHE = {}
    return _CACHE


def _compile_tile(spec: _TileSpec, x, q_w, sf_w, qt_w, sft_w, x8_w, e4_scale, outer_t, rowfac, norm, tamax, cos, sin, o,
                  d, crs, stream, seed_t=None, row_end=None, e4_doc=None, src=None, packed=None):
    """Compile (or fetch from the in-memory / persistent cache) the kernel of `spec`; the tensors only give dtypes and
    layouts (CPU tensors work: compile-only runs set CUTE_DSL_ARCH=sm_100a)."""
    key = ("fp4_tile", repr(spec))
    cache = _jit_cache()
    if key not in cache:
        cargs = [to_cute_tensor(x, divisibility=8), to_cute_tensor(q_w, divisibility=4),
                 to_cute_tensor(sf_w, assumed_align=4) if spec.packed else to_cute_tensor(sf_w, assumed_align=4, leading_dim=0),
                 to_cute_tensor(qt_w, divisibility=4),
                 to_cute_tensor(sft_w, assumed_align=4, leading_dim=0), to_cute_tensor(x8_w, divisibility=4),
                 to_cute_tensor(e4_scale, assumed_align=4, leading_dim=e4_scale.dim() - 1),   # (float32 scales: 4-byte)
                 to_cute_tensor(outer_t, assumed_align=4, leading_dim=0), Float32(1.0), Float32(1.0),
                 to_cute_tensor(rowfac, assumed_align=4, leading_dim=0), to_cute_tensor(norm, assumed_align=4, leading_dim=0),
                 to_cute_tensor(tamax, assumed_align=4, leading_dim=0), to_cute_tensor(cos, assumed_align=4),
                 to_cute_tensor(sin, assumed_align=4), to_cute_tensor(o, divisibility=8),
                 to_cute_tensor(d, assumed_align=4, leading_dim=0),
                 *[to_cute_tensor(t_, assumed_align=4, leading_dim=0) for t_ in crs[:8]],
                 to_cute_tensor(crs[8], divisibility=8),
                 *[to_cute_tensor(t_, assumed_align=4, leading_dim=0) for t_ in crs[9:13]],
                 to_cute_tensor(crs[13], assumed_align=4, leading_dim=0),
                 Int64(0), Int32(1),
                 to_cute_tensor(_workspace(x.device), assumed_align=4, leading_dim=0),
                 to_cute_tensor(seed_t, assumed_align=8, leading_dim=0) if spec.device_seed else None,
                 to_cute_tensor(row_end, assumed_align=4) if spec.row_end else None,
                 to_cute_tensor(e4_doc.reshape(-1), assumed_align=4, leading_dim=0) if (spec.e4m3_doc or spec.doc_slots)
                 else None,
                 cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)]
        if spec.gather:
            cargs.append(to_cute_tensor(src, assumed_align=4, leading_dim=0))
        if spec.packed:
            cargs += [None, to_cute_tensor(packed[0], assumed_align=4), to_cute_tensor(packed[1], assumed_align=4,
                                                                                         leading_dim=0), Int32(0)]
        cache[key] = cute.compile(_FP4TileKernel(spec), *cargs, options="--enable-tvm-ffi")
    return cache[key]


COMPILE_ONLY = False
_MIN_CTAS = 0          # tuning switch (benchmarks): launch-bounds occupancy hint
_STAGED_LOAD = True    # tuning switch (benchmarks)


def _launch(spec: _TileSpec, x, q_w, sf_w, qt_w, sft_w, x8_w, e4_scale, outer_t, inv_outer, outer, rowfac, norm, tamax,
            cos, sin, o, d, crs, seed, fn=None, seed_t=None, row_end=None, e4_doc=None, gather=None, packed=None):
    total = (x.shape[1] // 128) * x.shape[2] * x.shape[0]   # (x: the carrier under a gather -- the tiles)
    if packed is not None:   # (packed: the static tile bound; the kernel loops over the device count)
        total = min(packed[0].shape[1] * x.shape[2], _num_sms(x.device) * 4)
    src = None
    if gather is not None:   # lp_docmaps: the packed source rows through the row map
        x, src = gather
    if fn is None:
        fn = _compile_tile(spec, x, q_w, sf_w, qt_w, sft_w, x8_w, e4_scale, outer_t, rowfac, norm, tamax, cos, sin, o, d,
                           crs, None, seed_t, row_end, e4_doc, src, packed)
    if COMPILE_ONLY:
        return fn
    fn(x, q_w, sf_w, qt_w, sft_w, x8_w, e4_scale, outer_t, Float32(inv_outer), Float32(outer), rowfac, norm, tamax, cos,
       sin, o, d, *crs, Int64(seed), Int32(min(total, _num_sms(x.device) * 2) if spec.persistent else total),
       _workspace(x.device), seed_t if spec.device_seed else None, row_end if spec.row_end else None,
       e4_doc.reshape(-1) if (spec.e4m3_doc or spec.doc_slots) else None, src,
       *((packed[0], packed[1], Int32(packed[2] if len(packed) > 2 else packed[0].shape[1] * 128))
         if packed is not None else (None, None, None)))
    return fn


def _head_dims(d: int, G: int, padded_head_dim, name: str = "x"):
    """(head_dim_in, head_dim_out): the quantised operands are head_dim_out wide (NVFP4: 64 or 128; MXFP4: 128); a
    narrower x is zero-padded in registers when padded_head_dim=128 is asked for."""
    if d % G or d > 128 or d < G:
        raise NotImplementedError(f"{name}: head_dim {d}: a multiple of the scale group ({G}) up to 128 is supported "
                                  "(wider heads: not built)")
    dout = d if padded_head_dim is None else int(padded_head_dim)
    if dout not in ((64, 128) if G == 16 else (128,)) or dout < d:
        raise ValueError(f"{name}: head_dim {d} -> quantised width {dout}: the block-scaled operands are "
                         f"{'64 or 128' if G == 16 else '128'} wide; pass padded_head_dim=128 to zero-pad in registers")
    return d, dout


_DUMMIES = {}
_WORKSPACE = {}


def _workspace(dev):
    """The tile scheduler's two int32 counters, zero between launches (the kernel's last CTA resets them). One per
    (device, stream): kernels on one stream run in order; two streams get two workspaces."""
    if COMPILE_ONLY:
        return torch.empty(2, dtype=torch.int32, device=dev)
    key = (dev.index, torch._C._cuda_getCurrentRawStream(dev.index))   # cheap per call (host overhead)
    w = _WORKSPACE.get(key)
    if w is None:
        w = _WORKSPACE[key] = torch.zeros(2, dtype=torch.int32, device=dev)
    return w
_SMS = {}


def _num_sms(dev) -> int:
    key = str(dev)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[key]


def _ctas_per_sm(spec: "_TileSpec") -> int:
    """Resident CTAs per SM for the persistent grid: registers allow 3 (about 160 / thread); shared memory may allow
    fewer (228 KB per SM, 1 KB reserved per CTA)."""
    by_smem = max(1, (228 * 1024) // (spec.smem_bytes + 1024))
    return max(1, min(4, by_smem))


def _dummies(dev):
    """Placeholders for the arguments a specialisation does not read: allocated once per device (no launch per call)."""
    key = str(dev)
    if key not in _DUMMIES or COMPILE_ONLY:
        mk = torch.empty if COMPILE_ONLY else torch.zeros
        d = torch.empty(1, dtype=torch.float32, device=dev) if COMPILE_ONLY else torch.ones(1, dtype=torch.float32, device=dev)
        out = (mk(16, dtype=torch.int32, device=dev), mk(1, dtype=torch.float32, device=dev), d,
               mk(1, 64, dtype=torch.float32, device=dev))
        if COMPILE_ONLY:
            return out
        _DUMMIES[key] = out
    return _DUMMIES[key]


_DUMMY_ACC = {}
_MAX_COUNTER = {}


def _max_counter(dev):
    """The launch-wide max's tile counter: one int32 per (device, stream), zero at allocation, reset to 0 by the last
    tile of every launch that uses it (kernels on one stream run in order)."""
    if COMPILE_ONLY:
        return torch.empty(1, dtype=torch.int32, device=dev)
    key = (dev.index, torch._C._cuda_getCurrentRawStream(dev.index))
    c = _MAX_COUNTER.get(key)
    if c is None:
        c = _MAX_COUNTER[key] = torch.zeros(1, dtype=torch.int32, device=dev)
    return c


def _dummy_accumulator(dev):
    """A (1, 1, 1, 128) float16 placeholder for the unused dQ-accumulator argument (allocated once per device)."""
    key = str(dev)
    if key not in _DUMMY_ACC or COMPILE_ONLY:
        t = torch.empty(1, 1, 1, 128, dtype=torch.float16, device=dev)
        if COMPILE_ONLY:
            return t
        _DUMMY_ACC[key] = t
    return _DUMMY_ACC[key]


def _chunk_view(x: torch.Tensor):
    """(b, s, h, 128 c) -> ((b, s, h c, 128) view, c)."""
    b, s, h, d = x.shape
    if d <= 128:
        return x, 1
    if d % 128 or x.stride(2) != d or x.stride(3) != 1:
        raise NotImplementedError(f"head_dim {d}: above 128 only multiples of 128 with contiguous heads")
    return x.view(b, s, h * (d // 128), 128), d // 128


def _check_input(x: torch.Tensor, name: str = "x"):
    if x.dim() != 4:
        raise ValueError(f"{name} must be (batch, seqlen, heads, head_dim), got shape {tuple(x.shape)}")
    b, s, h, d = x.shape
    if s % 128:
        raise NotImplementedError(f"{name}: seqlen {s} is not a multiple of 128")
    if x.stride(3) != 1:
        raise ValueError(f"{name}: the head dim must be contiguous")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"{name}: dtype {x.dtype}; bf16, fp16 or fp32")


def packed_kt_groups(total: int, tiles: int) -> int:
    """(cov6) The 16-key group slots of the packed 1 x 16 transposed copy's SF table for T packed keys in `tiles` aligned
    tiles: group g of tile t (first packed key pb) at (pb >> 4) + t + g < T / 16 + tiles + 8."""
    return int(total) // 16 + int(tiles) + 8


def packed_vt_bytes(total: int) -> int:
    """(cov5) The byte columns of one key-parity copy of the packed transposed codes for T packed keys: the T + 1 key
    columns of a copy, the realigning consumer's 16-byte-aligned 80-byte window past the last one, rounded to 16."""
    return (int(total) // 2 + 1 + 96 + 15) // 16 * 16


def _quantize_tile(x, *, blocks_2d, scale_group_size, make_row_copy, make_transposed_copy,
                   transposed_copy_in_kernel_key_order, transposed_copy_layout, use_four_over_six, global_scale,
                   rounding, stochastic_rounding_seed, per_32_row_power_of_two_factors, also_make_e4m3_per_tensor_copy,
                   e4m3_copy_scale, e4m3_copy_of_unrotated_values, output_row_norm_max, output_tile_amax,
                   apply_hadamard_rotation, hadamard_sign_seed, rope_cos, rope_sin, compute_d_with_output,
                   padded_head_dim=None, causal_row_scale: Optional[CausalRowScale] = None,
                   output_row_amax: bool = False, output_tensor_max: Optional[str] = None,
                   fp4_global_scale_divisor: Optional[float] = None, dense_row_bound: Optional[DenseRowBound] = None,
                   make_transposed_sf_halves_swapped: bool = False, hadamard_rotation_method: Optional[str] = None,
                   row_end: Optional[torch.Tensor] = None, e4m3_copy_documents: Optional[torch.Tensor] = None,
                   row_documents: Optional[torch.Tensor] = None, n_documents: Optional[int] = None,
                   gather_rows=None, packed_rows=None):
    if packed_rows is not None:
        # (cov1) the packed rows of compact maps: x (1, T, h, d) at any T; the plain row copy (+ E4M3 copy, rotation)
        # (cov5) or the 2D-block NVFP4 copy with its transposed codes in the key-parity copies (V of the 4-bit P.V)
        _b2 = (blocks_2d and int(scale_group_size) == 16 and (not make_transposed_copy or (
            not transposed_copy_in_kernel_key_order and transposed_copy_layout == "batch_head_dim_seq"))
            and output_tensor_max in (None, "row_norm2")
            and (x.shape[3] == 128 or (x.shape[3] == 64 and not make_transposed_copy))   # (cov3: the row copy at d 64 too)
            and apply_hadamard_rotation is False)
        # (cov5) or the all-4-bit backward's causal dO launch: the row copy (nearest / stochastic), D, the causal row
        # scale with its per-document slots, the zeroed dQ accumulator rows -- per packed row
        _cr = (causal_row_scale is not None and compute_d_with_output is not None and make_row_copy
               and not make_transposed_copy and not blocks_2d and x.shape[3] == 128 and int(scale_group_size) == 16
               and rounding in ("nearest", "stochastic") and not apply_hadamard_rotation)
        # (cov6) or the NVFP4 1 x 16 transposed copy (K^T, the all-4-bit backward's dQ operand): the codes in the
        # key-parity copies in natural key order, the SF bytes per 16-key group in the group table (packed_kt_groups)
        _t1 = (make_transposed_copy and not blocks_2d and int(scale_group_size) == 16 and x.shape[3] == 128
               and rounding == "nearest" and not apply_hadamard_rotation and not transposed_copy_in_kernel_key_order
               and compute_d_with_output is None and causal_row_scale is None and output_tensor_max is None
               and e4m3_copy_documents is None and not also_make_e4m3_per_tensor_copy)
        # (cov6) or the dense all-4-bit backward's dO launches: the per-document max |dO| pass (no codes) and the row
        # copy with D, the dense row bound under the per-document device global scale -- per packed row
        _dn = (dense_row_bound is not None and compute_d_with_output is not None and make_row_copy
               and not make_transposed_copy and not blocks_2d and x.shape[3] == 128 and int(scale_group_size) == 16
               and rounding in ("nearest", "stochastic") and not apply_hadamard_rotation
               and isinstance(global_scale, torch.Tensor) and causal_row_scale is None)
        _am = (not make_row_copy and not make_transposed_copy and not blocks_2d and output_tensor_max == "abs"
               and x.shape[3] == 128 and not apply_hadamard_rotation and compute_d_with_output is None
               and causal_row_scale is None and dense_row_bound is None and global_scale is None)
        if (x.dim() != 4 or x.shape[0] != 1 or x.stride(3) != 1 or (blocks_2d and not _b2)
                or (make_transposed_copy and not (_b2 or _t1))
                or not (make_row_copy or _t1 or _am) or use_four_over_six or (global_scale is not None and not _dn)
                or (rounding != "nearest" and not (_cr or _dn))
                or per_32_row_power_of_two_factors is not None or output_row_norm_max is not None
                or output_tile_amax is not None or rope_cos is not None
                or (compute_d_with_output is not None and not (_cr or _dn))
                or padded_head_dim is not None or (causal_row_scale is not None and not _cr)
                or (output_row_amax and not (make_row_copy and not blocks_2d and x.shape[3] == 128))
                or (output_tensor_max is not None and not (_b2 or _am)) or (dense_row_bound is not None and not _dn)
                or (make_transposed_sf_halves_swapped and not _t1) or row_end is not None
                or (row_documents is not None and not (_b2 or _cr or _dn or _am))
                or x.shape[3] not in (64, 128)
                or x.dtype not in (torch.bfloat16, torch.float16)):
            raise ValueError("quantize_fp4_rows(packed_rows=): the plain row copy of (1, T, h, 64 / 128) 16-bit rows "
                             "(+ the E4M3 copy, the rotation), nearest rounding")
        vt_pk, nt_pk = packed_rows[:2]   # (cov5: [2] the full maps' padded size, the stochastic rounding's row hash)
        assert vt_pk.dtype == torch.int32 and vt_pk.dim() == 2 and vt_pk.shape[0] == 2 \
            and nt_pk.dtype == torch.int32 and nt_pk.numel() == 1
        if e4m3_copy_documents is not None and e4m3_copy_documents.numel() != vt_pk.shape[1]:
            raise ValueError("packed_rows: e4m3_copy_documents is one document per aligned tile of the maps")
        if row_documents is not None and row_documents.numel() != vt_pk.shape[1]:
            raise ValueError("packed_rows: row_documents is one document per aligned tile of the maps")
    else:
        _check_input(x)
    if make_transposed_sf_halves_swapped and (not make_transposed_copy or blocks_2d or int(scale_group_size) != 16
                                              or x.shape[3] != 128):
        raise ValueError("make_transposed_sf_halves_swapped: the NVFP4 1 x 16 transposed copy at head_dim 128")
    b0, s0, h0, d0 = x.shape
    x, chunks = _chunk_view(x)
    if chunks > 1 and (make_transposed_copy or blocks_2d or output_row_norm_max is not None or output_tile_amax is not None
                       or apply_hadamard_rotation or rope_cos is not None or compute_d_with_output is not None
                       or per_32_row_power_of_two_factors is not None or padded_head_dim is not None):
        raise NotImplementedError(f"head_dim {d0}: the row copy and the E4M3 copy only (no transposed / 2D / norms / "
                                  "rotation / RoPE / D / row factors above 128)")
    b, s, h, din = x.shape
    dev = x.device
    if row_end is not None:
        assert row_end.dtype == torch.int32 and row_end.shape == (b, s // 128) and row_end.is_cuda \
            and row_end.stride(-1) == 1, (row_end.dtype, row_end.shape, row_end.stride())
        if (blocks_2d or make_transposed_copy or apply_hadamard_rotation or rope_cos is not None
                or (compute_d_with_output is not None and causal_row_scale is None and dense_row_bound is None)
                or chunks > 1):
            raise NotImplementedError("row_end (the aligned varlen padding mask) is built for the plain row copy (and "
                                      "its E4M3 copy): no 2D blocks, transposed copy, rotation, RoPE, D or row scales")
    G = int(scale_group_size)
    if G not in (16, 32):
        raise ValueError(f"scale_group_size must be 16 (NVFP4) or 32 (MXFP4), got {scale_group_size}")
    din, d = _head_dims(din, G, padded_head_dim)
    if apply_hadamard_rotation and d != 128:
        raise NotImplementedError("the Hadamard rotation is 128-point: pass padded_head_dim=128 for a narrower head")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}, got {rounding!r}")
    # a device counter (int64 (1,)) as the seed: the kernel derives the seed from it on the GPU (graph-replay safe)
    device_seed = isinstance(stochastic_rounding_seed, torch.Tensor)
    if device_seed and (stochastic_rounding_seed.dtype != torch.int64 or stochastic_rounding_seed.numel() != 1
                        or stochastic_rounding_seed.device != dev):
        raise ValueError("stochastic_rounding_seed as a tensor: an int64 (1,) device counter on x's device")
    if G == 32 and (use_four_over_six or rounding in ("ms_eden", "unit")):
        raise ValueError("Four-over-Six, MS-EDEN and unit scales are NVFP4 options (an MXFP4 scale is a power of two)")
    if use_four_over_six and rounding != "nearest":
        raise ValueError("Four-over-Six is defined with rounding='nearest'")
    if use_four_over_six and blocks_2d:
        raise NotImplementedError("Four-over-Six on 2D blocks has no definition here (no caller)")
    if rounding == "ms_eden" and (blocks_2d or make_transposed_copy):
        if blocks_2d:
            raise NotImplementedError("MS-EDEN on 2D blocks has no definition here (no caller)")
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("RoPE needs both rope_cos and rope_sin")
    if transposed_copy_layout not in ("batch_dim_head_seq", "batch_head_dim_seq"):
        raise ValueError(f"transposed_copy_layout: 'batch_dim_head_seq' or 'batch_head_dim_seq', got {transposed_copy_layout!r}")
    rope = "none"
    if rope_cos is not None:
        if rope_cos.shape != rope_sin.shape or rope_cos.dtype != torch.float32 or rope_sin.dtype != torch.float32 \
                or not rope_cos.is_contiguous() or not rope_sin.is_contiguous():
            raise ValueError("rope_cos / rope_sin: contiguous float32 tables of the same shape, (64, s) or (s, 64)")
        if tuple(rope_cos.shape) == (din // 2, s):
            rope = "transposed_tables"
        elif tuple(rope_cos.shape) == (s, din // 2):
            rope = "row_tables"
        else:
            raise ValueError(f"rope tables must be ({din // 2}, {s}) or ({s}, {din // 2}), got {tuple(rope_cos.shape)}")
    if global_scale is None:
        gs_mode, inv_outer, outer = "none", 1.0, 1.0
    elif isinstance(global_scale, torch.Tensor):
        if global_scale.dtype == torch.float64:   # (accepted: converted to float32 once)
            global_scale = global_scale.float()
        if global_scale.dtype != torch.float32 or (global_scale.numel() != 1 and row_documents is None) \
                or not (global_scale.is_cuda or COMPILE_ONLY):
            raise ValueError("a device global_scale must be a (1,) float32 CUDA tensor")
        gs_mode, inv_outer, outer = "device", 1.0, 1.0
    else:
        gs_mode, inv_outer, outer = "host", 1.0 / float(global_scale), float(global_scale)
        if float(global_scale) == 1.0:
            gs_mode = "none"
    e4 = "none"
    if also_make_e4m3_per_tensor_copy:
        if e4m3_copy_documents is not None:
            # per-(document, kv head) scales: an (n_docs + 1, h_kv) float32 table and the (tiles,) int32 document of
            # every 128-row tile (b = 1; lp_varlen.LPDocs)
            if isinstance(e4m3_copy_scale, torch.Tensor) and e4m3_copy_scale.dtype == torch.float64:
                e4m3_copy_scale = e4m3_copy_scale.float()   # (accepted: converted once)
            if not isinstance(e4m3_copy_scale, torch.Tensor) or e4m3_copy_scale.dtype != torch.float32 \
                    or e4m3_copy_scale.dim() != 2 or not e4m3_copy_scale.is_contiguous() or x.shape[0] != 1 \
                    or x.shape[2] % e4m3_copy_scale.shape[1] or e4m3_copy_documents.dtype != torch.int32 \
                    or (e4m3_copy_documents.numel() != x.shape[1] // 128 and packed_rows is None):
                raise ValueError("e4m3_copy_documents: an (n_docs + 1, h_kv) float32 e4m3_copy_scale, b = 1, and one "
                                 "int32 document per 128-row tile")
        elif isinstance(e4m3_copy_scale, torch.Tensor) and e4m3_copy_scale.dtype == torch.float64 \
                and e4m3_copy_scale.numel() == 1:
            e4m3_copy_scale = e4m3_copy_scale.float()   # (accepted: converted once)
        if e4m3_copy_documents is None and (not isinstance(e4m3_copy_scale, torch.Tensor)
                                            or e4m3_copy_scale.dtype != torch.float32 or e4m3_copy_scale.numel() != 1):
            raise ValueError("e4m3_copy_scale must be a (1,) float32 device tensor (the per-tensor scale s; codes x / s)")
        e4 = "unrotated" if (e4m3_copy_of_unrotated_values and apply_hadamard_rotation) else "rotated"
    if compute_d_with_output is not None:
        o = compute_d_with_output
        if o.shape != x.shape or o.stride(3) != 1:
            raise ValueError("compute_d_with_output must have x's shape with a contiguous head dim")
        if apply_hadamard_rotation:
            raise NotImplementedError("D with a Hadamard-rotated x needs the rotated O; rotate O first (no caller)")
    crs_key = None
    if causal_row_scale is not None:
        c_ = causal_row_scale
        if compute_d_with_output is None or not make_row_copy or make_transposed_copy or blocks_2d or chunks > 1:
            raise ValueError("causal_row_scale: the row copy with compute_d_with_output (the dO launch), no transposed / 2D")
        if apply_hadamard_rotation or rope_cos is not None or global_scale is not None or e4 != "none" \
                or per_32_row_power_of_two_factors is not None or din != 128 or d != 128 or G != 16:
            raise ValueError("causal_row_scale: NVFP4, head_dim 128, no rotation / RoPE / global scale / E4M3 copy / row "
                             "factors (it sets its own per-row factor)")
        bhs = (x.shape[0], x.shape[2], x.shape[1])
        for nm in ("lse", "frame", "key_norm_prefix_max", "query_row_amax", "key_factor_prefix_max"):
            t_ = getattr(c_, nm)
            if t_ is None:
                continue
            if t_.dtype != torch.float32 or tuple(t_.shape) != bhs or not t_.is_contiguous():
                raise ValueError(f"causal_row_scale.{nm}: contiguous float32 {bhs}, got {t_.dtype} {tuple(t_.shape)}")
        if row_documents is not None:
            if x.shape[0] != 1 or row_documents.dtype != torch.int32 or row_documents.numel() != (
                    x.shape[1] // 128 if packed_rows is None else packed_rows[0].shape[1]) \
                    or c_.max_slots.dtype != torch.float32 or c_.max_slots.numel() % 2:
                raise ValueError("row_documents: b = 1, one int32 document per 128-row tile, max_slots (n_docs + 1) x 2 "
                                 "float32")
        elif c_.max_slots.dtype != torch.float32 or c_.max_slots.numel() != 2:
            raise ValueError("causal_row_scale.max_slots: (2,) float32, zeroed (by the forward)")
        za = c_.zero_accumulator
        if za is not None and (za.dtype != torch.float16 or za.shape[3] != 128 or not za.is_contiguous() or (
                tuple(za.shape[:3]) != bhs if packed_rows is None
                else (tuple(za.shape[:2]) != bhs[:2] or za.shape[2] < x.shape[1]))):
            raise ValueError(f"causal_row_scale.zero_accumulator: contiguous float16 {bhs + (128,)}")
        crs_key = (c_.key_factor_prefix_max is not None, float(c_.top), float(c_.lse_log2_offset), za is not None)
    gmax = "none"
    if output_tensor_max is not None:
        if output_tensor_max not in ("row_norm2", "abs"):
            raise ValueError(f"output_tensor_max: 'row_norm2' or 'abs', got {output_tensor_max!r}")
        if apply_hadamard_rotation or rope_cos is not None or din != 128 or chunks > 1:
            raise NotImplementedError("output_tensor_max: of the raw rows (no rotation / RoPE), head_dim 128")
        gmax = output_tensor_max
    if fp4_global_scale_divisor is not None and gmax != "abs":
        raise ValueError("fp4_global_scale_divisor needs output_tensor_max='abs'")
    if output_row_amax and (din != 128 or chunks > 1 or rope_cos is not None):
        raise NotImplementedError("output_row_amax: head_dim 128, the raw rows")
    if dense_row_bound is not None:
        if causal_row_scale is not None or gmax != "none" or compute_d_with_output is None or gs_mode != "device" \
                or not make_row_copy or make_transposed_copy or blocks_2d or din != 128 or d != 128 or G != 16 \
                or apply_hadamard_rotation or rope_cos is not None or per_32_row_power_of_two_factors is not None:
            raise ValueError("dense_row_bound: the NVFP4 row copy with compute_d_with_output and a device global_scale")
        bhs = (x.shape[0], x.shape[2], x.shape[1])
        for nm in ("lse", "frame"):
            t_ = getattr(dense_row_bound, nm)
            if t_.dtype != torch.float32 or tuple(t_.shape) != bhs or not t_.is_contiguous():
                raise ValueError(f"dense_row_bound.{nm}: contiguous float32 {bhs}")
        za = dense_row_bound.zero_accumulator
        if za is not None and (za.dtype != torch.float16 or za.shape[3] != 128 or not za.is_contiguous() or (
                tuple(za.shape[:3]) != bhs if packed_rows is None
                else (tuple(za.shape[:2]) != bhs[:2] or za.shape[2] < x.shape[1]))):
            raise ValueError(f"dense_row_bound.zero_accumulator: contiguous float16 {bhs + (128,)}")
        gmax = "dense_bound"
        crs_key = ("dense", float(dense_row_bound.lse_log2_offset), za is not None)
    crs_key = (crs_key, gmax, None if fp4_global_scale_divisor is None else float(fp4_global_scale_divisor),
               bool(output_row_amax), bool(make_transposed_sf_halves_swapped))
    rotation_tc = False
    rotation_tc5 = False
    if apply_hadamard_rotation:
        why = _tc_rows_unsupported(x, din, rope)
        method = rot.resolve_rotation_method(
            hadamard_rotation_method, "quantize_fp4_" + ("blocks_2d" if blocks_2d else "rows"), why,
            "the thread-per-row tile kernel: the rotated rows through a shared fp32 tile measured slower than the "
            "in-register butterfly (A1 row copy 22.9 vs 32.7 us, + E4M3 copy 32.6 vs 44.1 us)",
            tcgen05_unsupported=_tc5_rows_unsupported(x, causal_row_scale, dense_row_bound))
        rotation_tc = method == "tensor_core"
        rotation_tc5 = method == "tcgen05"
    elif hadamard_rotation_method is not None:
        rot.resolve_rotation_method(hadamard_rotation_method, "quantize_fp4", tcgen05_unsupported=None)
    # host overhead: the specialisation and its compiled function memoised on the plain values that determine them
    fkey = (x.dtype, G, bool(make_row_copy), bool(make_transposed_copy), bool(transposed_copy_in_kernel_key_order),
            bool(blocks_2d), bool(use_four_over_six), gs_mode, rounding, per_32_row_power_of_two_factors is not None, e4,
            output_row_norm_max is not None, output_tile_amax is not None,
            hadamard_sign_seed if apply_hadamard_rotation else None, rope,
            None if compute_d_with_output is None else compute_d_with_output.dtype, transposed_copy_layout, din, d, chunks,
            _STAGED_LOAD, _MIN_CTAS, crs_key, rotation_tc, rotation_tc5, device_seed) \
        + ((("row_end", True),) if row_end is not None else ()) \
        + ((("e4m3_doc", True),) if e4m3_copy_documents is not None else ()) \
        + ((("doc_slots", True),) if row_documents is not None else ()) \
        + ((("gather", True),) if gather_rows is not None else ()) \
        + ((("packed", True),) if packed_rows is not None else ())
    if row_documents is not None and causal_row_scale is None and gmax == "none" and dense_row_bound is None:
        raise NotImplementedError("row_documents: the per-document slots of causal_row_scale / output_tensor_max / "
                                  "dense_row_bound")
    if row_documents is not None and (x.shape[0] != 1 or row_documents.dtype != torch.int32
                                      or row_documents.numel() != (x.shape[1] // 128 if packed_rows is None
                                                                   else packed_rows[0].shape[1])
                                      or (make_transposed_copy and not blocks_2d)):
        raise ValueError("row_documents: b = 1, one int32 document per 128-row tile (no 1 x 16 transposed copy)")
    hit = None if COMPILE_ONLY else _FAST.get(fkey)
    spec, fn = hit if hit is not None else (_tile_spec(**locals()), None)
    if gather_rows is not None and hit is None:
        spec = dataclasses.replace(spec, gather=True)
    if packed_rows is not None and hit is None:
        spec = dataclasses.replace(spec, packed=True)
    u8 = dict(dtype=torch.uint8, device=dev)
    dummy_i, dummy_f, dummy_d, dummy_cos = _dummies(dev)
    out = {}
    di4 = dummy_i.view(1, 1, 1, 16)
    if make_row_copy and packed_rows is not None:
        # (packed: the codes at the packed rows; the SF words dense per packed row; a one-tile placeholder atom view)
        from flash_attn.cute.quant.fp8 import dense_row_stride
        codes = torch.empty(b, s, h, d // 2, **u8)
        sf_raw = torch.empty(b0, h0, 1, d // (4 * G), 32, 4, 4, **u8)
        sf4r = torch.empty(b0 * h0, dense_row_stride(s) * (d // (4 * G)), dtype=torch.int32, device=dev)
        q_w, sf_w = codes.view(torch.int32), sf4r
        out["codes"], out["scale_factors"], out["sf4_rows"] = codes.view(b0, s0, h0, -1), _sf_view(sf_raw), sf4r
    elif make_row_copy:
        codes = torch.empty(b, s, h, d // 2, **u8)
        sf_raw = torch.empty(b0, h0, s // 128, chunks * d // (4 * G), 32, 4, 4, **u8)
        q_w, sf_w = codes.view(torch.int32), sf_raw.view(torch.int32).view(-1)
        out["codes"], out["scale_factors"] = codes.view(b0, s0, h0, -1), _sf_view(sf_raw)
    else:
        q_w, sf_w = di4, dummy_i
    if make_transposed_copy and packed_rows is not None:
        # (packed, cov5) the key-parity copies (2, h, d, TB) bytes: copy c holds the documents starting at a key of
        # parity c, key k at column k + c (transposed_row_packed); TB = packed_vt_bytes(T)
        tcodes = torch.empty(2, h, d, packed_vt_bytes(s), **u8)
        qt_w = tcodes.view(torch.int32).permute(0, 2, 1, 3)
        sft_raw = torch.empty(1, h, 1, 2, 32, 4, 4, **u8)
        sft_w = sft_raw.view(torch.int32).view(-1)
        out["transposed_codes"], out["transposed_scale_factors"] = tcodes, _sf_view(sft_raw)
        if not blocks_2d:
            # (cov6, the 1 x 16 copy) the SF bytes per 16-key group: (h, G_b, 128) bytes (one per d row), group g of
            # aligned tile t (first packed key pb) at slot (pb >> 4) + t + g -- disjoint across tiles and documents
            gt = packed_kt_groups(s, packed_rows[0].shape[1])
            sfg = torch.empty(h, gt * 32, dtype=torch.int32, device=dev)
            sft_w = sfg.view(-1)
            out["transposed_sf_groups"] = sfg
            if make_transposed_sf_halves_swapped:   # (the atom views of the fused path's names; never read packed)
                out["transposed_scale_factors_halves_swapped"] = _sf_view(sft_raw)
    elif make_transposed_copy:
        if transposed_copy_layout == "batch_dim_head_seq":
            tcodes = torch.empty(b, d, h, s // 2, **u8)
            qt_w = tcodes.view(torch.int32)
        else:
            tcodes = torch.empty(b, h, d, s // 2, **u8)
            qt_w = tcodes.view(torch.int32).permute(0, 2, 1, 3)
        if make_transposed_sf_halves_swapped:
            sft_both = torch.empty(2, b, h, 1, s // (4 * G), 32, 4, 4, **u8)
            sft_raw = sft_both[0]
            sft_w = sft_both.view(torch.int32).view(-1)
            out["transposed_scale_factors_halves_swapped"] = _sf_view(sft_both[1])
        else:
            sft_raw = torch.empty(b, h, 1, s // (4 * G), 32, 4, 4, **u8)
            sft_w = sft_raw.view(torch.int32).view(-1)
        out["transposed_codes"], out["transposed_scale_factors"] = tcodes, _sf_view(sft_raw)
    else:
        qt_w, sft_w = di4, dummy_i
    if e4 != "none":
        x8 = torch.empty(b, s, h, d, dtype=torch.float8_e4m3fn, device=dev)
        x8_w, e4s = x8.view(torch.int32), (e4m3_copy_scale if e4m3_copy_documents is not None
                                           else e4m3_copy_scale.reshape(1))
        out["e4m3_copy"] = x8.view(b0, s0, h0, -1)
    else:
        x8_w, e4s = di4, dummy_d
    outer_t = global_scale.reshape(-1) if gs_mode == "device" else dummy_d
    rowfac = per_32_row_power_of_two_factors
    if rowfac is not None:
        if rowfac.dtype != torch.float32 or tuple(rowfac.shape) != (b, h, s // 32) or not rowfac.is_contiguous():
            raise ValueError(f"per_32_row_power_of_two_factors: contiguous float32 ({b}, {h}, {s // 32})")
        rowfac = rowfac.view(-1)
    else:
        rowfac = dummy_f
    norm = output_row_norm_max.view(-1) if output_row_norm_max is not None else dummy_f
    if output_row_norm_max is not None and (output_row_norm_max.dtype != torch.float32 or output_row_norm_max.numel() != 1):
        raise ValueError("output_row_norm_max: a zeroed (1,) float32 device slot (max-reduced into)")
    tamax = output_tile_amax.view(-1) if output_tile_amax is not None else dummy_f
    if output_tile_amax is not None and (output_tile_amax.dtype != torch.float32
                                         or output_tile_amax.numel() != b * h * (s // 128)):
        raise ValueError(f"output_tile_amax: float32 with {b * h * (s // 128)} elements (one per 128-row tile)")
    cos = rope_cos if rope_cos is not None else dummy_cos
    sin = rope_sin if rope_sin is not None else cos
    if compute_d_with_output is not None:
        o = compute_d_with_output
        dout = torch.empty(b, h, s, dtype=torch.float32, device=dev)
        out["d"] = dout
        dout = dout.view(-1)
    else:
        o, dout = x, dummy_f
    if causal_row_scale is not None:
        c_ = causal_row_scale
        rowf = torch.empty(b, h, s, dtype=torch.float32, device=dev)
        uinv = torch.empty(b, h, s, dtype=torch.float32, device=dev)
        out["row_factor"], out["uinv"] = rowf, uinv
        crs = [c_.lse.view(-1), c_.frame.view(-1), c_.key_norm_prefix_max.view(-1),
               c_.key_factor_prefix_max.view(-1) if c_.key_factor_prefix_max is not None else dummy_f,
               c_.query_row_amax.view(-1), rowf.view(-1), uinv.view(-1), c_.max_slots.view(-1),
               c_.zero_accumulator if c_.zero_accumulator is not None else _dummy_accumulator(dev)]
    elif dense_row_bound is not None:
        db = dense_row_bound
        bound_rows, uinv, d_scaled = (torch.empty(b, h, s, dtype=torch.float32, device=dev) for _ in range(3))
        out["bound_rows"], out["uinv"], out["d_scaled"] = bound_rows, uinv, d_scaled
        # slots: 0 lse, 1 frame, 2 value norm^2 max, 3 -, 4 D' (out), 5 bound_rows (out), 6 uinv (out), 7 -, 8 zero acc
        crs = [db.lse.view(-1), db.frame.view(-1), db.value_norm2_max.view(-1), dummy_f, d_scaled.view(-1),
               bound_rows.view(-1), uinv.view(-1), dummy_f,
               db.zero_accumulator if db.zero_accumulator is not None else _dummy_accumulator(dev)]
    else:
        crs = [dummy_f] * 8 + [_dummy_accumulator(dev)]
    extra_row = dummy_f
    if output_row_amax or dense_row_bound is not None:
        extra_row = torch.empty(b, h, s, dtype=torch.float32, device=dev)
        out["row_amax"] = extra_row
        extra_row = extra_row.view(-1)
    if gmax != "none":
        n_tiles = b * h * (s // 128) if packed_rows is None else h * packed_rows[0].shape[1]
        partials = torch.empty(n_tiles, dtype=torch.float32, device=dev)
        if row_documents is not None and packed_rows is not None:
            # (spd4) packed rows: the slots pre-filled here (the kernel's last tile only touches the tiles' documents)
            gout = torch.zeros(int(n_documents), dtype=torch.float32, device=dev)
            gout64 = torch.full((int(n_documents),), 1e-30, dtype=torch.float32, device=dev)
            out["tensor_max"] = gout
        elif row_documents is not None:   # per document (n_documents = documents + the sentinel)
            gout = torch.empty(int(n_documents), dtype=torch.float32, device=dev)
            gout64 = torch.empty(int(n_documents), dtype=torch.float32, device=dev)
            out["tensor_max"] = gout
        else:
            gout = torch.empty(2, dtype=torch.float32, device=dev)
            gout64 = torch.empty(1, dtype=torch.float32, device=dev)
            out["tensor_max"] = gout[:1]
        if fp4_global_scale_divisor is not None:
            out["fp4_global_scale"] = gout64
        crs += [extra_row, partials, _max_counter(dev), gout, gout64]
    else:
        crs += [extra_row, dummy_f, dummy_i, dummy_f, dummy_d]
    seed = 0 if device_seed else int(stochastic_rounding_seed) & 0x3FFFFFFFFFFFFFFF
    fn = _launch(spec, x, q_w, sf_w, qt_w, sft_w, x8_w, e4s, outer_t, inv_outer, outer, rowfac, norm, tamax, cos, sin, o,
                 dout, crs, seed, fn, seed_t=stochastic_rounding_seed if device_seed else None, row_end=row_end,
                 e4_doc=e4m3_copy_documents if e4m3_copy_documents is not None else row_documents, gather=gather_rows,
                 packed=packed_rows)
    if hit is None and not COMPILE_ONLY:
        _FAST[fkey] = (spec, fn)
    return out


_FAST = {}


def _tc5_rows_unsupported(x, causal_row_scale, dense_row_bound):
    """Why the tile kernel cannot rotate with hadamard_rotation_method="tcgen05" (None: it can)."""
    if x.dtype == torch.float32:
        return ("needs a bf16 / fp16 input: an fp32 row would be rounded to a 16-bit UMMA operand (the tensor_core "
                "method rotates fp32 inputs exactly, with the fp32 butterfly)")
    if causal_row_scale is not None or dense_row_bound is not None:
        return "with causal_row_scale / dense_row_bound (rows rescaled before the rotation are not exact UMMA operands)"
    return None


def _tc_rows_unsupported(x, din, rope):
    """Why the tile kernel's tensor-core rotation cannot run on this call (None: it can)."""
    if x.dtype == torch.float32:
        return "fp32 input (the mma.sync operands are 16-bit: bf16 / f16 rows are exact in them, fp32 rows are not)"
    if rope != "none" and din != 128:
        return "RoPE on a head dim below 128 (the quad layout pairs columns c and c + 64)"
    return None


def _tile_spec(*, x, G, make_row_copy, make_transposed_copy, transposed_copy_in_kernel_key_order, blocks_2d,
               use_four_over_six, gs_mode, rounding, per_32_row_power_of_two_factors, e4, output_row_norm_max,
               output_tile_amax, apply_hadamard_rotation, hadamard_sign_seed, rope, compute_d_with_output,
               transposed_copy_layout, din, d, chunks, causal_row_scale=None, gmax="none",
               fp4_global_scale_divisor=None, output_row_amax=False, dense_row_bound=None,
               make_transposed_sf_halves_swapped=False, rotation_tc=False, rotation_tc5=False, device_seed=False,
               row_end=None, e4m3_copy_documents=None, row_documents=None, **_):
    signs = rot.hadamard_signs(hadamard_sign_seed) if apply_hadamard_rotation else None
    o_dt = None if compute_d_with_output is None else _cute_dtype(compute_d_with_output.dtype)
    return _TileSpec(
        input_dtype=_cute_dtype(x.dtype), scale_group_size=G, row_copy=bool(make_row_copy),
        transposed_copy=bool(make_transposed_copy), transposed_key_order=bool(transposed_copy_in_kernel_key_order),
        blocks_2d=bool(blocks_2d), four_over_six=bool(use_four_over_six), global_scale=gs_mode, rounding=rounding,
        row_factors=per_32_row_power_of_two_factors is not None, e4m3_copy=e4, row_norm_max=output_row_norm_max is not None,
        tile_amax=output_tile_amax is not None, hadamard_signs=signs, rope=rope, d_output=compute_d_with_output is not None,
        output_dtype_o=o_dt, transposed_head_major=bool(make_transposed_copy) and transposed_copy_layout == "batch_head_dim_seq",
        head_dim_in=din, head_dim_out=d, staged_load=_STAGED_LOAD, min_ctas_per_sm=_MIN_CTAS, head_dim_chunks=chunks,
        causal_row_scale=causal_row_scale is not None,
        row_scale_fold=causal_row_scale is not None and causal_row_scale.key_factor_prefix_max is not None,
        row_scale_top=float(causal_row_scale.top) if causal_row_scale is not None else 0.0,
        lse_log2_offset=float(causal_row_scale.lse_log2_offset) if causal_row_scale is not None else 0.0,
        zero_dq_accum=(causal_row_scale is not None and causal_row_scale.zero_accumulator is not None)
        or (dense_row_bound is not None and dense_row_bound.zero_accumulator is not None),
        global_max=gmax, global_scale_divisor=float(fp4_global_scale_divisor or 0.0),
        row_amax_out=bool(output_row_amax), dense_row_bound=dense_row_bound is not None,
        transposed_sf_halves_swapped=bool(make_transposed_sf_halves_swapped),
        lse_log2_offset_dense=float(dense_row_bound.lse_log2_offset) if dense_row_bound is not None else 0.0,
        rotation_tc=bool(rotation_tc), rotation_tc5=bool(rotation_tc5), device_seed=bool(device_seed),
        row_end=row_end is not None, e4m3_doc=e4m3_copy_documents is not None, doc_slots=row_documents is not None)


def _docmaps_gather_plain(x, plain: bool):
    """lp_docmaps: (packed source, row map) of a registered shape carrier x when this launch is the plain row copy at
    head_dim 64 / 128 (the tile kernel reads it through the map); filled first (materialised) otherwise; None for any
    other x."""
    from flash_attn.cute import lp_docmaps
    if not lp_docmaps._LAZY or not isinstance(x, torch.Tensor):
        return None
    g = lp_docmaps.lazy_source(x)
    if g is None:
        return None
    if plain and x.dim() == 4 and x.shape[0] == 1 and x.shape[3] in (64, 128) and x.dtype in (torch.bfloat16,
                                                                                            torch.float16):
        return g
    lp_docmaps.materialise(x, g)
    return None


def _docmaps_fill(x):
    """lp_docmaps: a registered shape carrier is filled with its gathered rows before this quantiser reads it."""
    from flash_attn.cute import lp_docmaps
    if lp_docmaps._LAZY and isinstance(x, torch.Tensor):
        lp_docmaps.materialise(x)


def quantize_fp4_rows(x: torch.Tensor, *, scale_group_size: int = 16, make_row_copy: bool = True,
                      make_transposed_copy: bool = False, transposed_copy_in_kernel_key_order: bool = True,
                      transposed_copy_layout: str = "batch_dim_head_seq", use_four_over_six: bool = False,
                      global_scale=None, rounding: str = "nearest", stochastic_rounding_seed: int = 0,
                      per_32_row_power_of_two_factors: Optional[torch.Tensor] = None,
                      also_make_e4m3_per_tensor_copy: bool = False, e4m3_copy_scale: Optional[torch.Tensor] = None,
                      e4m3_copy_of_unrotated_values: bool = False, output_row_norm_max: Optional[torch.Tensor] = None,
                      output_tile_amax: Optional[torch.Tensor] = None, apply_hadamard_rotation: bool = False,
                      hadamard_sign_seed: int = rot.DEFAULT_HADAMARD_SIGN_SEED, rope_cos: Optional[torch.Tensor] = None,
                      rope_sin: Optional[torch.Tensor] = None,
                      compute_d_with_output: Optional[torch.Tensor] = None,
                      padded_head_dim: Optional[int] = None,
                      causal_row_scale: Optional[CausalRowScale] = None, output_row_amax: bool = False,
                      output_tensor_max: Optional[str] = None, fp4_global_scale_divisor: Optional[float] = None,
                      dense_row_bound: Optional[DenseRowBound] = None,
                      make_transposed_sf_halves_swapped: bool = False,
                      hadamard_rotation_method: Optional[str] = None,
                      row_end: Optional[torch.Tensor] = None,
                      e4m3_copy_documents: Optional[torch.Tensor] = None,
                      row_documents: Optional[torch.Tensor] = None, n_documents: Optional[int] = None,
                      packed_rows: Optional[tuple] = None) -> dict:
    """One-read row-wise NVFP4 / MXFP4 quantiser of x (batch, seqlen, heads, head_dim), bf16 / fp16 / fp32, seqlen % 128 ==
    0, head_dim a multiple of the group size up to 128. The quantised operands are head_dim wide (NVFP4: 64 or 128; MXFP4:
    128), or padded_head_dim=128 wide with the extra columns zero-padded in registers (the zero_pad_128 routes). The shapes
    below are written for 128.

    Copies (each optional, from the same read):
      row copy         scale groups of `scale_group_size` along the head dim: codes (b, s, h, 64) uint8 (two E2M1 per
                       byte, low nibble first) + scale_factors (the 7-D (32, 4, s/128, 4, 128/(4G), h, b) SF view).
      transposed copy  groups along the tokens: transposed_codes ("batch_dim_head_seq": (b, 128, h, s/2); or
                       "batch_head_dim_seq": (b, h, 128, s/2)) + transposed_scale_factors (32, 4, 1, 4, s/(4G), h, b).
                       transposed_copy_in_kernel_key_order: the 32-token blocks of every 128-token tile in the order
                       [0, 2, 1, 3] that the backward kernels' K operand reads (False: natural order).
    Transforms, applied in this order to the values the copies are made from:
      apply_rope (rope_cos / rope_sin given): GPT-NeoX halved RoPE at each token's position, cos / sin float32 (64, s)
                       (coalesced) or (s, 64); the rotated row is rounded to x's dtype (a standalone RoPE's output).
      apply_hadamard_rotation: x -> (x * signs) H / sqrt(128), signs from hadamard_sign_seed.
                       hadamard_rotation_method: "tensor_core" (quant._rotations: the warps rotate the staged tile on
                       mma.sync into a shared fp32 tile; equal to the butterfly up to the fp32 summation order) |
                       "butterfly" (bitwise the earlier kernels) | None: the butterfly (this thread-per-row kernel
                       measured faster with it; logged); fp32 input and RoPE below head dim 128 run the butterfly |
                       "tcgen05" (OPT-IN; NOT bitwise the other methods: the row the butterfly would rotate -- RoPE'd,
                       rounded, zero-padded -- rotated on tcgen05 UMMA, quant._tcgen05.RowRotation; every option of
                       this kernel incl. RoPE below head dim 128; refused with fp32 input and with causal_row_scale /
                       dense_row_bound). Speed: a drop-in into this thread-per-row kernel (+64 KB of shared memory per
                       CTA), not tuned.
      global_scale     the codes are of x / global_scale (a Python float or a (1,) float32 device tensor); dequantised =
                       code * SF * global_scale.
      per_32_row_power_of_two_factors  (b, h, s/32) float32 powers of two f: the codes are of x / f.
    Rounding: "nearest" | "stochastic" (stochastic_rounding_seed; SF = amax / (6 16/17)) | "unit" (SF = 1, RTN
      saturating at 6) | "ms_eden" (MS-EDEN, Quartet II). use_four_over_six: NVFP4 + nearest only.
    Side outputs:
      also_make_e4m3_per_tensor_copy: E4M3(x / e4m3_copy_scale) (b, s, h, 128) ("e4m3_copy"), e4m3_copy_scale a (1,)
                       float32 device scalar; of the rotated values unless e4m3_copy_of_unrotated_values.
      output_row_norm_max: a zeroed (1,) float32 slot, max-reduced with max_row ||x||^2 (the quantised values, rotated).
      output_tile_amax: float32 (b * h * s/128,) receiving each 128-row tile's max |x| (rotated); with no copy
                       requested this is the amax-only mode.
      compute_d_with_output=O: D (b, h, s) float32 = sum_d O * dequant(row codes) (x global_scale x f), "d".
      causal_row_scale (CausalRowScale; the all-FP4 backward's dO launch): each row's power-of-two factor F_q from the
                       row's own statistics, the codes and D of bf16(x F_q), "row_factor" F_q, "uinv", the E4M3 amaxes
                       max-reduced into its max_slots, and its dQ accumulator rows zeroed.
      output_row_amax: "row_amax" (b, h, s) float32, max |x| of each raw row.
      output_tensor_max: "row_norm2" | "abs": "tensor_max" (1,) float32, max over rows of ||x||^2 (torch's norm order) or
                       max |x|, reduced in the same launch by its last tile (no fill, no second pass);
                       fp4_global_scale_divisor (with "abs"): also "fp4_global_scale" (1,) float32 = max(amax / divisor,
                       1e-30), the device global scale of a later quantise of x.
      dense_row_bound (DenseRowBound; the all-FP4 backward's non-causal dO launch): see DenseRowBound.
      make_transposed_sf_halves_swapped (NVFP4 transposed copy, d 128): "transposed_scale_factors_halves_swapped", the
                       transposed SF atoms with the head-dim halves exchanged (the all-FP4 backward's second dQ SF tile).
      row_end (int32 (b, s / 128) on the GPU; the aligned varlen padding mask): row 128 t + r is quantised as a zero
                       row when 128 t + r >= row_end[bb, t] (plain row copy / E4M3 copy only).
      e4m3_copy_documents (int32 (s / 128,), b = 1; per-tensor operands on aligned varlen documents): the E4M3 copy's
                       scale is per (document, kv head): e4m3_copy_scale an (n_docs + 1, h_kv) float32 table, row r of
                       head h coded with the scale of the document of its 128-row tile and kv head h / (h / h_kv).
      row_documents (int32 (s / 128,), b = 1; aligned varlen documents): every per-launch statistic per document of
                       the row's 128-row tile -- causal_row_scale's max slots ((n_docs + 1) x 2 floats), output_tensor_max
                       ("tensor_max" / "fp4_global_scale" (n_documents,), n_documents = documents + the sentinel), a
                       device global_scale and dense_row_bound's value_norm2_max read per document ((n_documents,)); with
                       row_end, the padding rows also add nothing to them.
      packed_rows (lp_docmaps compact maps, cov1; b = 1): (vt_tile (2, tiles) int32, device tile count (1,) int32) --
                       x is the caller's PACKED (1, T, h, d) rows (any T): the tiles are the maps' aligned 128-row tiles
                       (tile t = packed rows vt_tile[0, t] + r, r < vt_tile[1, t]; the rows past them zero rows, as the
                       aligned layout's padding), the codes and the E4M3 copy at the packed rows, the SF bytes as
                       "sf4_rows" (b h, R d / 64) int32, R = quant.fp8.dense_row_stride(T) (word k of a row = its
                       64-column atom k's word); "scale_factors" a one-tile placeholder. The plain NVFP4 row copy (+ the
                       per-document E4M3 copy, the rotation), nearest rounding.
    Returns a dict with the requested outputs. One kernel launch; no host synchronisation."""
    if packed_rows is not None:
        return _quantize_tile(
            x, packed_rows=tuple(packed_rows), blocks_2d=False, scale_group_size=scale_group_size,
            make_row_copy=make_row_copy, make_transposed_copy=make_transposed_copy,
            transposed_copy_in_kernel_key_order=transposed_copy_in_kernel_key_order,
            transposed_copy_layout=transposed_copy_layout, use_four_over_six=use_four_over_six,
            global_scale=global_scale, rounding=rounding, stochastic_rounding_seed=stochastic_rounding_seed,
            per_32_row_power_of_two_factors=per_32_row_power_of_two_factors,
            also_make_e4m3_per_tensor_copy=also_make_e4m3_per_tensor_copy, e4m3_copy_scale=e4m3_copy_scale,
            e4m3_copy_of_unrotated_values=e4m3_copy_of_unrotated_values, output_row_norm_max=output_row_norm_max,
            output_tile_amax=output_tile_amax, apply_hadamard_rotation=apply_hadamard_rotation,
            hadamard_sign_seed=hadamard_sign_seed, rope_cos=rope_cos, rope_sin=rope_sin,
            compute_d_with_output=compute_d_with_output, padded_head_dim=padded_head_dim,
            causal_row_scale=causal_row_scale, output_row_amax=output_row_amax, output_tensor_max=output_tensor_max,
            fp4_global_scale_divisor=fp4_global_scale_divisor, dense_row_bound=dense_row_bound,
            make_transposed_sf_halves_swapped=make_transposed_sf_halves_swapped,
            hadamard_rotation_method=hadamard_rotation_method, row_end=row_end,
            e4m3_copy_documents=e4m3_copy_documents, row_documents=row_documents, n_documents=n_documents)
    gather = _docmaps_gather_plain(x, scale_group_size == 16 and make_row_copy and not make_transposed_copy
                                   and not use_four_over_six and global_scale is None and rounding == "nearest"
                                   and per_32_row_power_of_two_factors is None and not also_make_e4m3_per_tensor_copy
                                   and output_row_norm_max is None and output_tile_amax is None
                                   and not apply_hadamard_rotation and rope_cos is None and compute_d_with_output is None
                                   and padded_head_dim is None and causal_row_scale is None and not output_row_amax
                                   and output_tensor_max is None and dense_row_bound is None
                                   and not make_transposed_sf_halves_swapped and row_end is None
                                   and e4m3_copy_documents is None and row_documents is None
                                   and isinstance(stochastic_rounding_seed, int))
    return _quantize_tile(
        x, gather_rows=gather, blocks_2d=False, scale_group_size=scale_group_size, make_row_copy=make_row_copy,
        make_transposed_copy=make_transposed_copy, transposed_copy_in_kernel_key_order=transposed_copy_in_kernel_key_order,
        transposed_copy_layout=transposed_copy_layout, use_four_over_six=use_four_over_six, global_scale=global_scale,
        rounding=rounding, stochastic_rounding_seed=stochastic_rounding_seed,
        per_32_row_power_of_two_factors=per_32_row_power_of_two_factors,
        also_make_e4m3_per_tensor_copy=also_make_e4m3_per_tensor_copy, e4m3_copy_scale=e4m3_copy_scale,
        e4m3_copy_of_unrotated_values=e4m3_copy_of_unrotated_values, output_row_norm_max=output_row_norm_max,
        output_tile_amax=output_tile_amax, apply_hadamard_rotation=apply_hadamard_rotation,
        hadamard_sign_seed=hadamard_sign_seed, rope_cos=rope_cos, rope_sin=rope_sin,
        compute_d_with_output=compute_d_with_output, padded_head_dim=padded_head_dim,
        causal_row_scale=causal_row_scale, output_row_amax=output_row_amax, output_tensor_max=output_tensor_max,
        fp4_global_scale_divisor=fp4_global_scale_divisor, dense_row_bound=dense_row_bound,
        make_transposed_sf_halves_swapped=make_transposed_sf_halves_swapped,
        hadamard_rotation_method=hadamard_rotation_method, row_end=row_end, e4m3_copy_documents=e4m3_copy_documents,
        row_documents=row_documents, n_documents=n_documents)


def quantize_fp4_blocks_2d(x: torch.Tensor, *, scale_group_size: int = 16, make_row_copy: bool = True,
                           make_transposed_copy: bool = False, transposed_copy_in_kernel_key_order: bool = True,
                           transposed_copy_layout: str = "batch_dim_head_seq", global_scale=None,
                           rounding: str = "nearest", stochastic_rounding_seed: int = 0,
                           per_32_row_power_of_two_factors: Optional[torch.Tensor] = None,
                           also_make_e4m3_per_tensor_copy: bool = False, e4m3_copy_scale: Optional[torch.Tensor] = None,
                           e4m3_copy_of_unrotated_values: bool = False,
                           output_row_norm_max: Optional[torch.Tensor] = None, apply_hadamard_rotation: bool = False,
                           hadamard_sign_seed: int = rot.DEFAULT_HADAMARD_SIGN_SEED,
                           rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None,
                           compute_d_with_output: Optional[torch.Tensor] = None,
                           padded_head_dim: Optional[int] = None, output_tensor_max: Optional[str] = None,
                           hadamard_rotation_method: Optional[str] = None,
                           row_documents: Optional[torch.Tensor] = None, n_documents: Optional[int] = None,
                           packed_rows: Optional[tuple] = None) -> dict:
    """2D-block NVFP4 / MXFP4: one scale per 16 tokens x 16 d (scale_group_size 16, E4M3) or 32 x 32 (32, UE8M0), valid
    along both dims, so the row copy and the transposed copy carry the SAME codes and the same block scale (replicated
    into each copy's SF layout): one quantisation serves the GEMM that contracts d and the one that contracts the tokens.
    Arguments, transforms, side outputs and outputs as quantize_fp4_rows. One kernel launch.
    packed_rows (cov5; lp_docmaps compact maps, NVFP4 16 x 16, d 128): as quantize_fp4_rows' -- the row codes at the
    packed rows, the block SF bytes as the dense per-row words "sf4_rows" (every row of a 16-row block carries its
    block's 8 bytes), the transposed codes ("batch_head_dim_seq", tile key order) as the KEY-PARITY copies (2, h, d,
    packed_vt_bytes(T)) uint8 (transposed_row_packed); output_tensor_max="row_norm2" with row_documents (one document
    per aligned tile of the maps) as before."""
    if packed_rows is None:
        _docmaps_fill(x)
    return _quantize_tile(
        x, blocks_2d=True, scale_group_size=scale_group_size, make_row_copy=make_row_copy,
        make_transposed_copy=make_transposed_copy, transposed_copy_in_kernel_key_order=transposed_copy_in_kernel_key_order,
        transposed_copy_layout=transposed_copy_layout, use_four_over_six=False, global_scale=global_scale,
        rounding=rounding, stochastic_rounding_seed=stochastic_rounding_seed,
        per_32_row_power_of_two_factors=per_32_row_power_of_two_factors,
        also_make_e4m3_per_tensor_copy=also_make_e4m3_per_tensor_copy, e4m3_copy_scale=e4m3_copy_scale,
        e4m3_copy_of_unrotated_values=e4m3_copy_of_unrotated_values, output_row_norm_max=output_row_norm_max,
        output_tile_amax=None, apply_hadamard_rotation=apply_hadamard_rotation, hadamard_sign_seed=hadamard_sign_seed,
        rope_cos=rope_cos, rope_sin=rope_sin, compute_d_with_output=compute_d_with_output,
        padded_head_dim=padded_head_dim, output_tensor_max=output_tensor_max,
        hadamard_rotation_method=hadamard_rotation_method, row_documents=row_documents, n_documents=n_documents,
        **({} if packed_rows is None else dict(packed_rows=tuple(packed_rows))))


# ======================================================================================================================
# Dequantiser (the inverse of the row copy): code * SF in the requested dtype (exact in bf16 / fp16 / fp32)
# ======================================================================================================================
class _DequantRowsKernel:
    def __init__(self, out_dtype, scale_group_size: int):
        self.out_dtype = out_dtype
        self.G = scale_group_size

    @cute.jit
    def __call__(self, mC: cute.Tensor, mSF: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        grid = [mOut.shape[1] // 128, mOut.shape[2], mOut.shape[0]]
        self.kernel(mC, mSF, mOut).launch(grid=grid, block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mC: cute.Tensor, mSF: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        tiles = mOut.shape[1] // 128
        nheads = mOut.shape[2]
        r = tidx
        q = t * 128 + r
        words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        cute.autovec_copy(cute.make_tensor(mC[bb, q, hh, None].iterator, cute.make_layout(16)), words)
        i = r % 32
        j = r // 32
        G = self.G
        sfs = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        if const_expr(G == 16):
            base = ((bb * nheads + hh) * tiles + t) * 256
            w0 = mSF[base + i * 4 + j]
            w1 = mSF[base + 128 + i * 4 + j]
            for g in cutlass.range_constexpr(8):
                if const_expr(g < 4):
                    sfs[g] = e4m3_byte_value(w0, g)
                else:
                    sfs[g] = e4m3_byte_value(w1, g - 4)
        else:
            base = ((bb * nheads + hh) * tiles + t) * 128
            w0 = mSF[base + i * 4 + j]
            for g in cutlass.range_constexpr(4):
                sfs[g] = bits_f32(((w0 >> (8 * g)) & 0xFF) << 23)
        vals = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        for c in cutlass.range_constexpr(0, 128, 2):
            v0, v1 = e2m1_pair_values(words[c // 8], (c % 8) // 2)
            vals[c] = v0 * sfs[c // G]
            vals[c + 1] = v1 * sfs[c // G]
        ob = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
        ob.store(vals.load().to(self.out_dtype))
        cute.autovec_copy(ob, cute.make_tensor(mOut[bb, q, hh, None].iterator, cute.make_layout(128)))


def dequantize_fp4_rows(codes: torch.Tensor, scale_factors: torch.Tensor, dtype=torch.bfloat16,
                        scale_group_size: int = 16) -> torch.Tensor:
    """The row copy of quantize_fp4_rows (codes (b, s, h, 64) uint8, scale_factors its 7-D SF view) -> (b, s, h, 128) values
    code * SF in `dtype` (exact for bf16 / fp16 / fp32). One launch."""
    b, s, h, dh = codes.shape
    if dh != 64 or s % 128:
        raise NotImplementedError(f"dequantize_fp4_rows: codes (b, s, h, 64), s % 128 == 0; got {tuple(codes.shape)}")
    raw = scale_factors.permute(6, 5, 2, 4, 0, 1, 3)
    if not raw.is_contiguous():
        raw = raw.contiguous()
    out = torch.empty(b, s, h, 128, dtype=dtype, device=codes.device)
    cw = codes.contiguous().view(torch.int32)
    sw = raw.view(torch.int32).view(-1)
    key = ("fp4_dequant_rows", str(dtype), int(scale_group_size))
    cache = _jit_cache()
    if key not in cache:
        cache[key] = cute.compile(_DequantRowsKernel(_cute_dtype(dtype), int(scale_group_size)),
                                  to_cute_tensor(cw, divisibility=4), to_cute_tensor(sw, assumed_align=4, leading_dim=0),
                                  to_cute_tensor(out, divisibility=8),
                                  cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    if not COMPILE_ONLY:
        cache[key](cw, sw, out)
    return out

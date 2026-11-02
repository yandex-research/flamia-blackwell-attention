# FP8 quantisers of the low-precision attention routes (CuTe DSL). Two functions:
#
#   quantize_fp8_blocks      block-scaled FP8 with UE8M0 (power-of-two) scales: every block shape the routes and the
#                            benchmarks use (block_rows 1..128 tokens x block_columns 1 / 32 / 64 / 128 head-dim elements),
#                            E4M3 / E5M2, token- and channel-major codes from one read, the exponent table, the token-block
#                            factors, D = sum_d O dequant(code), the shifted log2 LSE, Hadamard, RoPE, option A's fused
#                            NVFP4 copy. THE default-path quantiser (Q, K, V, dO of the 4-bit and 8-bit routes).
#   quantize_fp8_per_tensor  per-tensor FP8 with the scale on the GPU (given, or from the amax: never read to the host),
#                            up to four tensors per launch, the backward prologue's side outputs (D, shifted LSE, zeroed
#                            dQ accumulator, dS bound, row norms), row divisors, Hadamard, RoPE.
#
# Every option is a compile-time constant: one specialised kernel per combination, nothing is paid for an option that is
# off. Both kernels are thread-per-row: one CTA per (128-token tile, head, batch), thread r owns token row r (all 128
# head-dim values in registers), so RoPE, the Hadamard rotation, the per-row scale groups and the dot products are
# register arithmetic.
#
# SCALING-CONSISTENCY RULE (a design invariant): the forward and the backward read the SAME quantised
# operands (same format, granularity and scale values); these functions are called once per operand and their codes
# are shared by both passes.
#
# Bitwise contract with the quantisers they replace (Triton mx2d_quant / mxfp8_quant / e4m3_quant_triton): the same
# instructions in the same order -- the UE8M0 selector's ex2.approx.ftz / div.full, the clamp then cvt.rn.satfinite,
# 1 / s as rcp.rn.f32 of the float32 scale, and the old reduction order of every float sum (`_old_triton_row_dot`).
# No FP64 anywhere: every scale is a float32 value derived with correctly rounded float32 operations (div.rn / rcp.rn /
# mul.rn), bitwise a numpy float32 evaluation of the same formula (tests/cute/test_device_scales_fp32.py).
import math
from typing import Optional, Sequence

import numpy as np
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda

from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.quant._rotations import (INV_SQRT_128, DEFAULT_HADAMARD_SIGN_SEED, rope_rotate_row,
                                              hadamard_rotate_row as hadamard_rotate_row128,
                                              hadamard_signs as hadamard_signs_tuple, quad_hadamard_rotate,
                                              quad_sign_words, resolve_rotation_method, quad_rope)
from flash_attn.cute.quant.fp4 import div_full as div_full_f32, e4m3_byte_value
from flash_attn.cute.quant._fp8_nvfp4 import e2m1_value, nvfp4_group16, nvfp4_encode_row128, nvfp4_half_group8
from flash_attn.cute.quant._tcgen05 import RowRotation


@cute.jit
def _rope_packed_row(words: cute.Tensor, mCos, mSin, pos, rope_t: cutlass.Constexpr, dt: cutlass.Constexpr):
    """RoPE on a row held as 64 packed 16-bit words, in place: fp32 pairs (rope_rotate_row's fma forms), rounded back to
    the storage dtype (bitwise quad_rope / apply_rope_row128 on the same row)."""
    av = cute.make_tensor(cute.recast_ptr(words.iterator, dtype=dt), cute.make_layout(128))
    xr = cute.make_rmem_tensor(cute.make_layout(128), Float32)
    xr.store(av.load().to(Float32))
    rope_rotate_row(xr, mCos, mSin, pos, rope_t)
    av.store(xr.load().to(dt))


@cute.jit
def apply_rope_row128(x: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, pos, cos_sin_transposed: cutlass.Constexpr,
                      storage_dtype: cutlass.Constexpr):
    """quant/_rotations.rope_rotate_row (GPT-NeoX halves, the defined fma forms), then the row rounded through the storage
    dtype (a no-op for fp32): the row a standalone RoPE kernel would have written."""
    rope_rotate_row(x, mCos, mSin, pos, cos_sin_transposed)
    if const_expr(storage_dtype is not Float32):
        rowb = cute.make_rmem_tensor(cute.make_layout(128), storage_dtype)
        rowb.store(x.load().to(storage_dtype))
        x.store(rowb.load().to(Float32))

__all__ = ["quantize_fp8_blocks", "quantize_fp8_per_tensor", "FP8_MAX", "TORCH_FP8"]

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
TORCH_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
LOG2E = 1.4426950408889634
FLT_MIN = 1.1754943508222875e-38   # 2^-126
_COMPILE_OPTIONS = "--enable-tvm-ffi"  # profiling scripts append --keep-cubin
# The quant package imports only torch / cutlass and its own files, so it also loads where the rest of the package
# cannot (the benchmarks' cuDNN environment loads it with stub parent packages).
_TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: cutlass.Float32}


def _cute_dtype(dt):
    return _TORCH2CUTE[dt]


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, divisibility=1):
    """torch -> cute tensor for TVM FFI with a dynamic layout (interface_fwd_fp4.to_cute_tensor)."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility > 1:
        tensor = tensor.mark_compact_shape_dynamic(mode=leading_dim, stride_order=t.dim_order(), divisibility=divisibility)
    return tensor


# ---------------------------------------------------------------------------------------------------- PTX primitives
def _asm_f32(asm, *args, loc=None, ip=None):
    cons = "=f," + ",".join("f" for _ in args)
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip) for a in args], asm, cons,
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def ex2_ftz(a, *, loc=None, ip=None) -> Float32:
    """ex2.approx.ftz.f32: Triton's tl.exp2 (exact powers of two for integer inputs in the normal range, 0 below)."""
    return _asm_f32("ex2.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    """mul.rn.f32: a product that can never be contracted into an FMA."""
    return _asm_f32("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def fmin_f32(a, b, *, loc=None, ip=None) -> Float32:
    """min.f32 (a NaN operand yields the other one): cute.arch.fmin's instruction, also on cutlass-dsl versions without
    it (the cuDNN environment's 4.5)."""
    return _asm_f32("min.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _asm_f32("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _asm_f32("sub.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    return _asm_f32("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def sqrt_rn(a, *, loc=None, ip=None) -> Float32:
    return _asm_f32("sqrt.rn.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def f32_bits(a, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Float32(a).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f",
                                 has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None) -> Float32:
    """div.rn.f32: a / b correctly rounded."""
    return _asm_f32("div.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def rcp_rn(a, *, loc=None, ip=None) -> Float32:
    """rcp.rn.f32: 1 / a correctly rounded (bitwise div.rn.f32(1, a))."""
    return _asm_f32("rcp.rn.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def pow2_ceil_f32(a, *, loc=None, ip=None) -> Float32:
    """The smallest power of two >= a (a > 0, a normal float32 below 2^127), exactly: the exponent field, plus one when
    the mantissa is nonzero (power_of_two_scale)."""
    return _asm_f32(
        "{\n\t.reg .b32 b, m, e;\n\t.reg .pred p;\n\tmov.b32 b, $1;\n\tand.b32 m, b, 8388607;\n\t"
        "shr.u32 e, b, 23;\n\tsetp.ne.u32 p, m, 0;\n\t@p add.u32 e, e, 1;\n\tshl.b32 b, e, 23;\n\t"
        "mov.b32 $0, b;\n\t}\n", a, loc=loc, ip=ip)


@dsl_user_op
def bits_f32(a, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Int32(a).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def prmt(a, b, sel, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
                                           Int32(sel).ir_value(loc=loc, ip=ip)],
                                 "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def e5m2_byte_value(word, k: cutlass.Constexpr[int], *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 w;\n\t.reg .b16 x;\n\t.reg .b32 p;\n\t.reg .b16 h0, h1;\n\t"
        f"shr.b32 w, $1, {8 * k};\n\tcvt.u16.u32 x, w;\n\tcvt.rn.f16x2.e5m2x2 p, x;\n\tmov.b32 {{h0, h1}}, p;\n\t"
        "cvt.f32.f16 $0, h0;\n\t}\n",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(out)


@dsl_user_op
def fp8_pair_f16x2(fmt: cutlass.Constexpr, word, half: cutlass.Constexpr[int], *, loc=None, ip=None) -> Int32:
    """Bytes 2 half, 2 half + 1 of a packed FP8 word -> f16x2 (one cvt.rn.f16x2.{e4m3,e5m2}x2, exact)."""
    sel = "l" if half == 0 else "h"
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\t"
        f"cvt.rn.f16x2.{fmt}x2 $0, {sel};\n\t}}\n",
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def atomic_max_u32(ptr: cute.Pointer, val, *, loc=None, ip=None) -> None:
    """red.global.max.u32 of a non-negative float's bits (non-negative floats order as their bit patterns)."""
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(val).ir_value(loc=loc, ip=ip)],
                    "red.relaxed.gpu.global.max.u32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atomic_max_if_larger_f32(ptr: cute.Pointer, val, *, loc=None, ip=None) -> None:
    """Max-reduce a non-negative fp32 into a gmem slot, issued only when it beats the slot's current value (the old
    Triton quantisers' `if v > load(volatile): atomic_max`)."""
    llvm.inline_asm(
        None, [ptr.toint(loc=loc, ip=ip).ir_value(), Float32(val).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 v, c;\n\t.reg .pred q;\n\tmov.b32 v, $1;\n\tld.volatile.global.b32 c, [$0];\n\t"
        "setp.gt.u32 q, v, c;\n\t@q red.global.max.u32 [$0], v;\n\t}\n",
        "l,f", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atomic_add_acq_rel_i32(ptr: cute.Pointer, val, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(val).ir_value(loc=loc, ip=ip)],
                                 "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
                                 is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def ld_acquire_i32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value()], "ld.acquire.gpu.global.b32 $0, [$1];",
                                 "=r,l", has_side_effects=True, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def ld_cg_i32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """A plain L2 load (ld.global.cg): for values a PREVIOUS launch wrote."""
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value()], "ld.global.cg.b32 $0, [$1];",
                                 "=r,l", has_side_effects=True, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def st_relaxed_i32(ptr: cute.Pointer, val, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(val).ir_value(loc=loc, ip=ip)],
                    "st.relaxed.gpu.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_16(smem_addr, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), gmem_ptr.toint(loc=loc, ip=ip).ir_value()],
                    "cp.async.cg.shared.global [$0], [$1], 16;", "r,l", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_16_zfill(smem_addr, gmem_ptr: cute.Pointer, nbytes, *, loc=None, ip=None) -> None:
    """16-byte cp.async reading nbytes (0 or 16) of gmem, zero-filling the rest."""
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
                           Int32(nbytes).ir_value(loc=loc, ip=ip)],
                    "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_4_zfill(smem_addr, gmem_ptr: cute.Pointer, nbytes, *, loc=None, ip=None) -> None:
    """4-byte cp.async (through L1) reading nbytes (0 or 4) of gmem, zero-filling the rest."""
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
                           Int32(nbytes).ir_value(loc=loc, ip=ip)],
                    "cp.async.ca.shared.global [$0], [$1], 4, $2;", "r,l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "cp.async.commit_group;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_wait_1(*, loc=None, ip=None) -> None:
    """Wait until at most one committed cp.async group of this thread is pending."""
    llvm.inline_asm(None, [], "cp.async.wait_group 1;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "cp.async.commit_group;\n\tcp.async.wait_group 0;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def absmax_x2(acc, w, half: cutlass.Constexpr, *, loc=None, ip=None) -> Int32:
    """Packed running max of |x| over two 16-bit floats (bf16x2 / f16x2): exact, a max needs no conversion."""
    op = "bf16x2" if half == "bf16" else "f16x2"
    return Int32(llvm.inline_asm(T.i32(), [Int32(acc).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
                                 f"{{\n\t.reg .b32 a;\n\tabs.{op} a, $2;\n\tmax.{op} $0, $1, a;\n\t}}\n", "=r,r,r",
                                 has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def half_lo_f32(w, half: cutlass.Constexpr, *, loc=None, ip=None) -> Float32:
    if half == "bf16":
        return bits_f32(Int32(w) << 16, loc=loc, ip=ip)
    return Float32(llvm.inline_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)],
                                   "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, l;\n\t}\n", "=f,r",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def half_hi_f32(w, half: cutlass.Constexpr, *, loc=None, ip=None) -> Float32:
    if half == "bf16":
        return bits_f32(Int32(w) & Int32(-65536), loc=loc, ip=ip)
    return Float32(llvm.inline_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)],
                                   "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, h;\n\t}\n", "=f,r",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


# ---------------------------------------------------------------------------------------------------- shared pieces
@dsl_user_op
def cvt_fp8x4(fmt: cutlass.Constexpr, f0, f1, f2, f3, *, loc=None, ip=None) -> Int32:
    """4 fp32 -> 4 FP8 bytes (element 0 in the low byte): two cvt.rn.satfinite.{e4m3,e5m2}x2.f32."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(f).ir_value(loc=loc, ip=ip) for f in (f0, f1, f2, f3)],
        "{\n\t.reg .b16 lo, hi;\n\t"
        f"cvt.rn.satfinite.{fmt}x2.f32 lo, $2, $1;\n\tcvt.rn.satfinite.{fmt}x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t}\n",
        "=r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _pack4(fmt: cutlass.Constexpr, f0, f1, f2, f3):
    return cvt_fp8x4(fmt, f0, f1, f2, f3)


@cute.jit
def _code_values4(fmt: cutlass.Constexpr, word):
    """The four FP8 values of a packed word (byte 0 first) as fp32: two f16x2 conversions, exact."""
    v = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    for hf in cutlass.range_constexpr(2):
        p = fp8_pair_f16x2(fmt, word, hf)
        v[2 * hf] = half_lo_f32(p, "f16")
        v[2 * hf + 1] = half_hi_f32(p, "f16")
    return v


@cute.jit
def _code_value(fmt: cutlass.Constexpr, word, k: cutlass.Constexpr):
    if const_expr(fmt == "e4m3"):
        return e4m3_byte_value(word, k)
    return e5m2_byte_value(word, k)


@cute.jit
def _clamp(y, fmax: cutlass.Constexpr):
    """tl.minimum(tl.maximum(y, -FMAX), FMAX): max.f32 / min.f32 (a NaN operand yields the other one)."""
    return fmin_f32(cute.arch.fmax(y, Float32(-fmax)), Float32(fmax))


@cute.jit
def _ue8m0_select(amax: Float32, fmax: cutlass.Constexpr, flush_subnormals: cutlass.Constexpr = False):
    """The round-UP UE8M0 selector of mx2d_quant / mxfp8_quant: e = ceil(log2(amax / FMAX)) from the float bits, then two
    exact corrections (FMAX 2^e is exact in fp32). Returns (byte, scale 2^e, 1 / scale); 0 / 0 / 0 for an all-zero
    group, and with flush_subnormals also for a subnormal amax / FMAX (FlashInfer's rule). Same
    instructions as the Triton kernels (ex2.approx.ftz, div.full)."""
    rr = amax * Float32(1.0 / fmax)
    bits = f32_bits(rr)
    e = ((bits >> 23) & 0xFF) - 127
    if (bits & 0x7FFFFF) != 0:
        e = e + 1
    if Float32(fmax) * ex2_ftz(Float32(e)) < amax:
        e = e + 1
    if Float32(fmax) * ex2_ftz(Float32(e - 1)) >= amax:
        e = e - 1
    sfr = Float32(0.0)
    byte = Int32(0)
    pos = amax > Float32(0.0)
    if const_expr(flush_subnormals):
        pos = rr >= Float32(FLT_MIN)
    if pos:
        sfr = ex2_ftz(Float32(e))
        byte = (e + 127) & 0xFF
    inv = Float32(0.0)
    if sfr > Float32(0.0):
        inv = div_full_f32(Float32(1.0), sfr)
    return byte, sfr, inv


def _old_tree_bits(hd: int) -> tuple:
    """The column bits the old reduction combines after bit 1, in order (probed at head dim 64, 128, 256):
    2, 3, 0, then the high bits from the top down to 4."""
    top = hd.bit_length() - 1
    return (2, 3, 0) + tuple(range(top - 1, 3, -1))


def _old_tree_plan(hd: int = 128):
    """(first-level columns, add steps (dst, src) over the hd/2 partials, final partial) of the old reduction tree."""
    cols = [c for c in range(hd) if (c >> 1) & 1 == 0]
    live = list(range(hd // 2))
    steps = []
    for bit in _old_tree_bits(hd):
        pos = {cols[k]: k for k in live}
        nxt = []
        for k in live:
            if (cols[k] >> bit) & 1 == 0:
                steps.append((k, pos[cols[k] | (1 << bit)]))
                nxt.append(k)
        live = nxt
    return tuple(cols), tuple(steps), live[0]


_TREE = {hd: _old_tree_plan(hd) for hd in (64, 128, 256)}
_TREE_COLS, _TREE_STEPS, _TREE_ROOT = _TREE[128]


def _partial_index(c: int) -> int:
    """Index of column c (bit 1 clear) among the first-level partials."""
    return (c // 4) * 2 + c % 2


@cute.jit
def _old_triton_row_dot(a: cute.Tensor, o: cute.Tensor) -> Float32:
    """sum_c a[c] o[c] over 128 columns in the exact order of the old Triton kernels' tl.sum(axis=1) on a [128, 128]
    tile (8 warps; decoded with cancellation probes): a butterfly over the column index bits
    1, 2, 3, 0, 6, 5, 4. The first level is an FMA onto the partner's rounded product,
        u_c = fma(a_c, o_c, fl(a_{c+2} o_{c+2}))      for every c with bit 1 clear,
    and every later level a plain fp32 add of the two partials whose columns differ in that bit."""
    v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
    for k in cutlass.range_constexpr(64):
        c = _TREE_COLS[k]
        v[k] = fma_rn(a[c], o[c], mul_rn(a[c + 2], o[c + 2]))
    for i in cutlass.range_constexpr(len(_TREE_STEPS)):
        v[_TREE_STEPS[i][0]] = add_rn(v[_TREE_STEPS[i][0]], v[_TREE_STEPS[i][1]])
    return v[_TREE_ROOT]


@cute.jit
def _transpose_store_codes(words: cute.Tensor, mQ8T: cute.Tensor, bb, hh, q0, r, hd: cutlass.Constexpr = 128):
    """Channel-major copy (b, h, d, s) of the thread's token row: a 4 x 4 byte transpose inside each quad of lanes
    (tokens r0..r0+3), so lane r0 + j holds column 4w + j of the four tokens as one word, then one 4-byte store per word
    (a warp writes 32 contiguous bytes of each of 4 columns per instruction)."""
    j = r % 4
    r0 = r - j
    sel_a = Int32(0x5410)
    if j >= 2:
        sel_a = Int32(0x3276)
    sel_b = Int32(0x6240)
    if (j % 2) == 1:
        sel_b = Int32(0x3715)
    for w in cutlass.range_constexpr(hd // 4):
        mine = words[w]
        other = cute.arch.shuffle_sync_bfly(mine, offset=2)
        a = prmt(mine, other, sel_a)
        other_b = cute.arch.shuffle_sync_bfly(a, offset=1)
        tw = prmt(a, other_b, sel_b)
        mQ8T[bb, hh, 4 * w + j, (q0 + r0) // 4] = tw


@cute.jit
def _transpose_store_codes_packed(words: cute.Tensor, mQ8T: cute.Tensor, bb, hh, vt0, nv, r,
                                  hd: cutlass.Constexpr = 128):
    """_transpose_store_codes into the PACKED channel-major copy (lp_docmaps packed rows): the tile's rows are the
    columns vt0 .. vt0 + nv - 1 of the copy (vt0: a multiple of 16 -- its documents start at multiples of 16 keys,
    DocMaps.cu_vt), and the columns up to the next multiple of 16 are written as zero codes (no document's: the TMA
    tile reads them, so never leave them unwritten). Whole aligned words only: no byte store, no next document's
    column touched."""
    j = r % 4
    r0 = r - j
    sel_a = Int32(0x5410)
    if j >= 2:
        sel_a = Int32(0x3276)
    sel_b = Int32(0x6240)
    if (j % 2) == 1:
        sel_b = Int32(0x3715)
    nz = (nv + 15) // 16 * 16
    if nv >= (1 << 16):   # (DocMaps.VT_LAST: the last real tile also zeroes its columns past cu_vt[-1])
        nv = nv - (1 << 16)
        nz = Int32(128)
    kq = cutlass.min(cutlass.max(nv - r0, Int32(0)), Int32(4))       # this quad's real tokens
    keep = Int32(-1)
    if kq < 4:
        keep = (Int32(1) << (8 * kq)) - 1
    for w in cutlass.range_constexpr(hd // 4):
        mine = words[w]
        other = cute.arch.shuffle_sync_bfly(mine, offset=2)
        a = prmt(mine, other, sel_a)
        other_b = cute.arch.shuffle_sync_bfly(a, offset=1)
        tw = prmt(a, other_b, sel_b)
        if r0 < nz:
            mQ8T[bb, hh, 4 * w + j, (vt0 + r0) // 4] = tw & keep


def _row_pad(dtype, hd: int = 128) -> int:
    """Elements per staged shared-memory row: hd plus 16 bytes of padding (conflict-free 16-byte row reads)."""
    return hd + 16 // (dtype.width // 8)


@cute.jit
def _stage_tile(mX: cute.Tensor, sX: cute.Tensor, bb, q0, hh, dtype: cutlass.Constexpr, hd: cutlass.Constexpr = 128,
                x_cols: cutlass.Constexpr = 0, nvalid=None, row_src=None, row_cap=None):
    """Issue coalesced 16-byte cp.async copies of the (128 tokens, hd) tile of head hh into sX (row stride padded):
    consecutive threads take consecutive 16-byte chunks, so a warp reads whole rows. x_cols (0 = hd): the input rows'
    width when narrower than hd (the head-dim padding of the token-block fallback shapes): columns x_cols .. hd - 1 are staged as zeros.
    nvalid (row_end, the aligned varlen padding): rows >= nvalid of the tile are staged as zeros (never read).
    row_src (lp_docmaps gather, (s,) int32): row q0 + r of the tile is row row_src[q0 + r] of mX (the packed caller
    tensor), or a zero row where row_src < 0 (the aligned layout's padding rows)."""
    tidx, _, _ = cute.arch.thread_idx()
    eb = dtype.width // 8
    cpr = hd * eb // 16                     # 16-byte chunks per row
    rs = _row_pad(dtype, hd)
    base = Int32(sX.iterator.toint())
    for it in cutlass.range_constexpr(cpr):
        cid = it * 128 + tidx
        row = cid // cpr
        c16 = cid % cpr
        if const_expr(row_src is not None):
            sr = row_src[q0 + row]
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16,
                              mX[bb, cutlass.max(sr, Int32(0)), hh, None].iterator + c16 * (16 // eb),
                              cutlass.select_(sr >= Int32(0), Int32(16), Int32(0)))
        elif const_expr(row_cap is not None and x_cols != 0 and x_cols != hd):
            # (compact maps, a padded head dim) the packed rows' x_cols columns, zeros after them and past nvalid
            cin = x_cols * eb // 16
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16,
                              mX[bb, cutlass.min(q0 + row, row_cap - 1), hh, None].iterator
                              + cutlass.min(c16, cin - 1) * (16 // eb),
                              cutlass.select_(c16 < cin, cutlass.select_(row < nvalid, Int32(16), Int32(0)), Int32(0)))
        elif const_expr(row_cap is not None):
            # (sfd compact maps) packed rows q0 + row for row < nvalid, zeros after (the address clamped in range)
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16,
                              mX[bb, cutlass.min(q0 + row, row_cap - 1), hh, None].iterator + c16 * (16 // eb),
                              cutlass.select_(row < nvalid, Int32(16), Int32(0)))
        elif const_expr(nvalid is not None and (x_cols == 0 or x_cols == hd)):
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16, mX[bb, q0 + row, hh, None].iterator + c16 * (16 // eb),
                              cutlass.select_(row < nvalid, Int32(16), Int32(0)))
        elif const_expr(nvalid is not None):
            cin = x_cols * eb // 16
            src = mX[bb, q0 + row, hh, None].iterator + cutlass.min(c16, cin - 1) * (16 // eb)
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16, src,
                              cutlass.select_(c16 < cin, cutlass.select_(row < nvalid, Int32(16), Int32(0)), Int32(0)))
        elif const_expr(x_cols == 0 or x_cols == hd):
            cp_async_16(base + (row * rs) * eb + c16 * 16, mX[bb, q0 + row, hh, None].iterator + c16 * (16 // eb))
        else:
            cin = x_cols * eb // 16         # 16-byte chunks the input row holds
            inside = c16 < cin
            src = mX[bb, q0 + row, hh, None].iterator + cutlass.min(c16, cin - 1) * (16 // eb)
            cp_async_16_zfill(base + (row * rs) * eb + c16 * 16, src, cutlass.select_(inside, Int32(16), Int32(0)))


@cute.jit
def _smem_run(sX: cute.Tensor, r, col, n: cutlass.Constexpr, dtype: cutlass.Constexpr, hd: cutlass.Constexpr = 128):
    """Elements [col, col + n) of staged row r as a shared-memory view the compiler knows is 16-byte aligned (col is a
    multiple of 16 bytes), so the copy is vectorised."""
    eb = dtype.width // 8
    addr = sX.iterator.toint() + (r * _row_pad(dtype, hd) + col) * eb
    return cute.make_tensor(cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(n))


@cute.jit
def _load_staged_words(sX: cute.Tensor, r, col, n: cutlass.Constexpr, dtype: cutlass.Constexpr,
                       hd: cutlass.Constexpr = 128):
    """n 32-bit words of staged row r starting at element col (16-byte aligned), raw."""
    eb = dtype.width // 8
    addr = sX.iterator.toint() + (r * _row_pad(dtype, hd) + col) * eb
    w = cute.make_rmem_tensor(cute.make_layout(n), Int32)
    cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, addr, cute.AddressSpace.smem, assumed_align=16),
                                       cute.make_layout(n)), w)
    return w


@cute.jit
def _load_staged(sX: cute.Tensor, r, col, n: cutlass.Constexpr, dtype: cutlass.Constexpr, hd: cutlass.Constexpr = 128):
    rowb = cute.make_rmem_tensor(cute.make_layout(n), dtype)
    cute.autovec_copy(_smem_run(sX, r, col, n, dtype, hd), rowb)
    x = cute.make_rmem_tensor(cute.make_layout(n), Float32)
    x.store(rowb.load().to(Float32))
    return x


@cute.jit
def _load_staged_row(sX: cute.Tensor, r, dtype: cutlass.Constexpr):
    return _load_staged(sX, r, 0, 128, dtype)


@cute.jit
def _store_staged_words(sX: cute.Tensor, r, word, w0, w1, dtype: cutlass.Constexpr, hd: cutlass.Constexpr = 128):
    """Two code words into the thread's own staged row at word offset `word` (the row's input there is consumed)."""
    eb = dtype.width // 8
    addr = sX.iterator.toint() + r * _row_pad(dtype, hd) * eb + word * 4
    t = cute.make_tensor(cute.make_ptr(Int32, addr, cute.AddressSpace.smem, assumed_align=8), cute.make_layout(2))
    t[0] = w0
    t[1] = w1


@cute.jit
def _copy_out_codes(sX: cute.Tensor, mQ: cute.Tensor, bb, q0, hh, dtype: cutlass.Constexpr, hd: cutlass.Constexpr = 128):
    """The tile's codes (written into the staged rows) to global memory as coalesced 16-byte stores. mQ: the int32
    view (b, s, h, hd / 4) of the codes."""
    tidx, _, _ = cute.arch.thread_idx()
    eb = dtype.width // 8
    row_bytes = _row_pad(dtype, hd) * eb
    cpr = hd // 16                           # 16-byte code chunks per row
    for it in cutlass.range_constexpr(cpr):
        cid = it * 128 + tidx
        row = cid // cpr
        c4 = (cid % cpr) * 4
        ch = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        src = cute.make_tensor(cute.make_ptr(Int32, sX.iterator.toint() + row * row_bytes + c4 * 4,
                                             cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4))
        cute.autovec_copy(src, ch)
        dst = cute.make_tensor(cute.make_ptr(Int32, (mQ[bb, q0 + row, hh, None].iterator + c4).toint(),
                                             cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(4))
        cute.autovec_copy(ch, dst)


def _sign_words(signs) -> tuple:
    """The rotation's sign bits as 4 words: lane L negates element e of its 8 columns when bit 8 L + e is set."""
    bits = 0
    for c, v in enumerate(signs):
        if v < 0:
            bits |= 1 << c
    return tuple((bits >> (32 * w)) & 0xFFFFFFFF for w in range(4))


@cute.jit
def _lane_transform(v: cute.Tensor, lane, pos, mCos, mSin, rope: cutlass.Constexpr, rope_t: cutlass.Constexpr,
                    sign_words: cutlass.Constexpr, dt: cutlass.Constexpr):
    """RoPE then the Hadamard rotation of a 128-wide row held by 16 lanes (lane L: columns 8 L .. 8 L + 7), partners
    through lane shuffles. RoPE: column c pairs with c + 64 (lane ^ 8), rope_rotate_pair's fma forms, the row rounded
    through the storage dtype. Rotation: signs, butterfly levels H = 1, 2, 4 in the thread and H = 8 .. 64
    across lanes (the lower member keeps a + b, the upper a - b), then * fp32(1 / sqrt(128)) -- fwht_rows' op order."""
    if const_expr(rope):
        j0 = (lane * 8) % 64
        cs = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        sn = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for e in cutlass.range_constexpr(8):
            if const_expr(rope_t):
                cs[e] = mCos[j0 + e, pos]
                sn[e] = mSin[j0 + e, pos]
            else:
                cs[e] = mCos[pos, j0 + e]
                sn[e] = mSin[pos, j0 + e]
        hi = lane >= 8
        for e in cutlass.range_constexpr(8):
            # lane < 8 holds x0 (column j), lane >= 8 holds x1 (column j + 64): rope_rotate_pair's arithmetic
            pv = cute.arch.shuffle_sync_bfly(v[e], offset=8)
            r_ = fma_rn(v[e], cs[e], -mul_rn(pv, sn[e]))
            if hi:
                if const_expr(rope_t):
                    r_ = fma_rn(v[e], cs[e], mul_rn(pv, sn[e]))
                else:
                    r_ = fma_rn(pv, sn[e], mul_rn(v[e], cs[e]))
            v[e] = r_
        if const_expr(dt is not Float32):
            rb = cute.make_rmem_tensor(cute.make_layout(8), dt)
            rb.store(v.load().to(dt))
            v.store(rb.load().to(Float32))
    if const_expr(sign_words is not None):
        word = Int32(sign_words[0])
        if lane // 4 == 1:
            word = Int32(sign_words[1])
        if lane // 4 == 2:
            word = Int32(sign_words[2])
        if lane // 4 == 3:
            word = Int32(sign_words[3])
        bits = word >> ((lane % 4) * 8)
        for e in cutlass.range_constexpr(8):
            if ((bits >> e) & 1) != 0:
                v[e] = -v[e]
        for lv in cutlass.range_constexpr(3):
            hs = 1 << lv
            for e in cutlass.range_constexpr(8):
                if const_expr((e & hs) == 0):
                    a_, b_ = v[e], v[e + hs]
                    v[e] = add_rn(a_, b_)
                    v[e + hs] = sub_rn(a_, b_)
        for lv in cutlass.range_constexpr(4):
            m = 1 << lv
            lower = (lane & m) == 0
            for e in cutlass.range_constexpr(8):
                pv = cute.arch.shuffle_sync_bfly(v[e], offset=m)
                r_ = sub_rn(pv, v[e])
                if lower:
                    r_ = add_rn(v[e], pv)
                v[e] = r_
        for e in cutlass.range_constexpr(8):
            v[e] = mul_rn(v[e], Float32(INV_SQRT_128))
    return v


# byte offset of the staged raw tile inside the fp32 tile's buffer, per input width: pass 0's fp32 rows 0..63 end
# below pass 1's raw rows for 16-bit inputs; fp32 inputs are transformed in place
_RAW_OFFSET = {16: 128 * (128 + 4) * 4 - 128 * (128 + 8) * 2, 32: 0}
assert 64 * (128 + 4) * 4 <= _RAW_OFFSET[16] + 64 * (128 + 8) * 2

# the lower members of the in-lane butterfly level H = 2^lv over 64 elements
_PAIR_LOWS = tuple(tuple(e for e in range(64) if (e & (1 << lv)) == 0) for lv in range(6))


@cute.jit
def _pair_transform(v: cute.Tensor, h, pos, mCos, mSin, rope: cutlass.Constexpr, rope_t: cutlass.Constexpr,
                    signs: cutlass.Constexpr, dt: cutlass.Constexpr):
    """RoPE then the Hadamard rotation of a 128-wide row held by a lane pair (lane parity h: columns 64 h .. 64 h + 63,
    v[e] = column 64 h + e), the same arithmetic as _lane_transform in fewer instructions: RoPE pairs column e with e + 64
    (the partner lane); the rotation's levels H = 1 .. 32 run inside the lane (packed f32x2 adds / subtracts: lower
    member a + b, upper a - b), level 64 is one shuffle folded as fma(x, +-1, partner) -- exact products, so the same
    single rounding as the add / subtract -- then * fp32(1 / sqrt(128)) (fwht_rows' op order)."""
    if const_expr(rope):
        cs = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        sns = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        if const_expr(rope_t):
            for e in cutlass.range_constexpr(64):     # (64, s): a warp's 16 consecutive positions per load
                cs[e] = mCos[e, pos]
                sns[e] = mSin[e, pos]
        else:                                         # (s, 64) contiguous rows: 16-byte loads
            cute.autovec_copy(cute.make_tensor(cute.make_ptr(Float32, mCos[pos, None].iterator.toint(),
                                                             cute.AddressSpace.gmem, assumed_align=16),
                                               cute.make_layout(64)), cs)
            cute.autovec_copy(cute.make_tensor(cute.make_ptr(Float32, mSin[pos, None].iterator.toint(),
                                                             cute.AddressSpace.gmem, assumed_align=16),
                                               cute.make_layout(64)), sns)
        for e in cutlass.range_constexpr(64):
            c = cs[e]
            sn = sns[e]
            pv = cute.arch.shuffle_sync_bfly(v[e], offset=1)
            r_ = fma_rn(v[e], c, -mul_rn(pv, sn))
            if h == 1:
                if const_expr(rope_t):
                    r_ = fma_rn(v[e], c, mul_rn(pv, sn))
                else:
                    r_ = fma_rn(pv, sn, mul_rn(v[e], c))
            v[e] = r_
        if const_expr(dt is not Float32):
            rb = cute.make_rmem_tensor(cute.make_layout(64), dt)
            rb.store(v.load().to(dt))
            v.store(rb.load().to(Float32))
    if const_expr(signs is not None):
        # the sign of column 64 h + e: compile-time where both halves agree, else a +-1 multiply chosen by h (exact)
        m_lo_neg = Float32(1.0)        # the factor when only the low half's sign is negative
        if h == 0:
            m_lo_neg = Float32(-1.0)
        for e in cutlass.range_constexpr(64):
            lo_neg = signs[e] < 0
            hi_neg = signs[e + 64] < 0
            if const_expr(lo_neg and hi_neg):
                v[e] = -v[e]
            elif const_expr(lo_neg):
                v[e] = v[e] * m_lo_neg
            elif const_expr(hi_neg):
                v[e] = v[e] * (-m_lo_neg)
        for k in cutlass.range_constexpr(0, 64, 2):
            # level H = 1 scalar: its pair is one register pair (packing (x0, x2) with (x1, x3) costs a MOV per value)
            a_ = v[k]
            b_ = v[k + 1]
            v[k] = a_ + b_
            v[k + 1] = a_ - b_
        for lv in cutlass.range_constexpr(1, 6):
            hs = 1 << lv
            for k in cutlass.range_constexpr(16):
                e0, e1 = _PAIR_LOWS[lv][2 * k], _PAIR_LOWS[lv][2 * k + 1]
                a0, a1 = v[e0], v[e1]
                b0, b1 = v[e0 + hs], v[e1 + hs]
                v[e0], v[e1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                v[e0 + hs], v[e1 + hs] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
        sg = Float32(1.0)
        if h == 1:
            sg = Float32(-1.0)
        for e in cutlass.range_constexpr(0, 64, 2):
            p0 = cute.arch.shuffle_sync_bfly(v[e], offset=1)
            p1 = cute.arch.shuffle_sync_bfly(v[e + 1], offset=1)
            v[e], v[e + 1] = cute.arch.fma_packed_f32x2((v[e], v[e + 1]), (sg, sg), (p0, p1))
        for e in cutlass.range_constexpr(0, 64, 2):
            v[e], v[e + 1] = cute.arch.mul_packed_f32x2((v[e], v[e + 1]), (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))
    return v


@cute.jit
def _quad_scatter_max(am: cute.Tensor, tq):
    """am[0..3] of every lane of a quad (lanes 4 g .. 4 g + 3, tq = lane % 4) -> the max over the quad of am[tq], in lane
    tq (a reduce-scatter: 3 shuffles instead of 8). Exact (fmax)."""
    hb = (tq >> 1) & 1
    s0 = am[2]
    s1 = am[3]
    k0 = am[0]
    k1 = am[1]
    if hb == 1:
        s0 = am[0]
        s1 = am[1]
        k0 = am[2]
        k1 = am[3]
    k0 = cute.arch.fmax(k0, cute.arch.shuffle_sync_bfly(s0, offset=2))
    k1 = cute.arch.fmax(k1, cute.arch.shuffle_sync_bfly(s1, offset=2))
    snd = k1
    keep = k0
    if (tq & 1) == 1:
        snd = k0
        keep = k1
    return cute.arch.fmax(keep, cute.arch.shuffle_sync_bfly(snd, offset=1))


# ---------------------------------------------------------------------------------------------------- block kernel
class _BlockQuantKernel:
    """One CTA per (128-token tile, head, batch), 128 threads, thread r = token row r. See quantize_fp8_blocks."""

    def __init__(self, in_dtype, out_dtype, fmt, rows, cols, channel_major, exponent_table, factors, d_mode,
                 lse, signs, rope, rope_transposed, nvfp4, nvfp4_scaling, nvfp4_stochastic, flush_subnormals,
                 nvfp4_encoding="dp_operand", zero_acc=False, hd=128, d_mul=None, stat_amax=False, stat_norm=False,
                 token_local=False, token_axis_unit=False, token_log2=False, token_inverse=False, x_cols=0):
        self.stat_amax = stat_amax      # per-tile max |x| of the quantised values (output_amax)
        self.stat_norm = stat_norm      # per-tile max row ||x||^2 of the quantised values (output_row_norm2_max)
        # token_local (token_local_exponents): codes, sf_tok and D take each token's own exponent; sf_dim / the table /
        # the factors stay the 32-token block's. token_axis_unit: sf_dim is the unit byte 127 (the token-contraction GEMM
        # gets the per-token factor folded into its other operand instead).
        self.token_local = token_local
        self.token_axis_unit = token_axis_unit
        self.token_log2 = token_log2          # also log2 of the per-token factor (an integer, as fp32)
        self.token_inverse = token_inverse    # also 1 / the per-token factor
        self.hd = hd                    # head dim: 128, or 64 (kernel_stream, no RoPE / rotation)
        self.x_cols = x_cols            # input row width when narrower than hd (zero-padded; kernel_stream), 0 = hd
        self.d_mul = d_mul              # None, or a float32 constant D is multiplied by before its store (the D fold)
        self.ngd = hd // 32             # 32-column groups along d (the SF atom has 4 K-groups: d = 64 repeats 0, 1)
        self.flush_subnormals = flush_subnormals
        self.nvfp4_encoding = nvfp4_encoding
        self.zero_acc = zero_acc
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype      # dtype of O (D), or None
        self.fmt = fmt
        self.fmax = FP8_MAX[fmt]
        self.rows = rows
        self.cols = cols
        self.channel_major = channel_major
        self.exponent_table = exponent_table
        self.factors = factors
        self.d_mode = d_mode            # None | "fp8" | "nvfp4"
        self.lse = lse
        self.signs = signs
        self.rope = rope
        self.rope_transposed = rope_transposed
        self.nvfp4 = nvfp4
        self.nvfp4_scaling = nvfp4_scaling
        self.nvfp4_stochastic = nvfp4_stochastic
        ng = 128 if cols == 1 else 4
        self.ng = ng
        # every row with block_columns >= 32 takes the low-register streaming form (kernel_stream); RoPE / rotation rows are
        # transformed by 16 lanes per row on the way in, into an fp32 tile
        self.streaming = cols >= 32
        self.lane_transform = rope or signs is not None
        self.sign_words = None if signs is None else _sign_words(signs)
        # RoPE / rotation token blocks without the row side outputs: the register-resident lane-pair form (kernel_pair)
        self.pair = (self.lane_transform and rows == 32 and cols >= 32 and not channel_major and d_mode is None
                     and not lse and not zero_acc)
        self.pair_ctas = 0  # > 0: kernel_pair_persistent with this many CTAs (set by the host; 0 = one CTA per tile)
        self.pair_named = False  # token-block named barriers instead of CTA barriers in kernel_pair (set by the host)
        self.pair_coalesced = False  # sector-complete half-row loads / code stores via a lane-pair exchange (host)
        self.pair_min_ctas = 0  # kernel_pair's register budget as a residency target (nvvm.minctasm; 0 = compiler's)
        # hadamard_rotation_method="tensor_core": the rotation on mma.sync (kernel_tc for the kernel_pair rows); set by the
        # host only where it applies (a rotation, 16-bit input)
        self.tc = False
        self.tc_min_ctas = 0    # kernel_tc's residency target (nvvm.minctasm; 0 = compiler's)
        # hadamard_rotation_method="tcgen05" in the thread-per-row kernels (kernel_stream, block_columns=1): the staged
        # (RoPE'd) rows rotated on tcgen05 (quant._tcgen05.RowRotation); set by the host, which also adds its SMEM
        self.tc5 = False
        # hadamard_rotation_scope="nvfp4_copy" (kernel_tc only; set by the host): the FP8 codes, tables, factors and amax
        # of the raw rows, the NVFP4 copy of the rotated rows
        self.rotate_nvfp4_copy_only = False
        # the aligned varlen row end (mRowEnd, set by the host): row_end_input stages rows past it as zero rows
        # (kernel_stream; kernel_pair / kernel_tc mask whenever mRowEnd is given), row_end_lse writes +inf as their
        # shifted LSE (the backward's P = 0 on padding rows)
        self.row_end_input = False
        self.row_end_lse = False
        # lp_docmaps packed rows (set by the host, with mSrc): the per-row outputs (codes, channel-major codes, the NVFP4
        # copy, D, the shifted LSE, the zeroed accumulator) are the PACKED rows src[q0 + r] (rows without one: none);
        # the per-tile outputs (SFs, table, factors, statistics) stay per aligned tile
        self.packed_out = False
        # sfd (SF redesign, with packed_out; set by the host): the per-row DENSE tables next to the per-tile ones --
        # mEvR (b h, R) u8: each real row's 32-token block's effective exponent (the factor rule applied: an all-zero
        # block takes its tile's largest, 0 = an all-zero tile = factor 1), mSF4R (b h, R d / 64) int32: the NVFP4
        # copy's SF word(s) of each real row, mDR / mLseR (b h, R) f32: D'' / the shifted LSE of each real row. R =
        # dense_row_stride(T); the tail rows [T, T + 128) of mEvR are zeroed (by tile 0)
        self.dense_out = False
        # sfd compact maps (with dense_out; set by the host): no row map -- the tile table mVt holds each tile's packed
        # start and real rows, the grid is persistent over the device tile count mNT, and no per-tile atom / table
        # / aligned statistic is written (only the dense rows and the per-tile statistics)
        self.compact = False
        self.quad_signs = None if signs is None else quad_sign_words(signs)
        self.smem_bytes = 128 * _row_pad(in_dtype, hd) * (in_dtype.width // 8) + 16
        if self.streaming and self.lane_transform:   # the fp32 tile (holding the raw one) + the rows' group maxima
            self.smem_bytes += 128 * _row_pad(Float32) * 4 - 128 * _row_pad(in_dtype) * (in_dtype.width // 8) + 128 * 16 + 16
        if d_mode is not None:
            self.smem_bytes += 128 * _row_pad(out_dtype, hd) * (out_dtype.width // 8) + 16
        if cols == 1:
            self.smem_bytes += 128 * 132 * 4 + 3 * 4 * 128 * 4 + 64
        else:
            if rows > 32:
                self.smem_bytes += 4 * ng * 4 + 16
            if rows >= 32:
                self.smem_bytes += 4 * ng * 4 + 16
        self.smem_bytes += 64
        if stat_amax or stat_norm:
            self.smem_bytes += 32 + 16

    @cute.jit
    def __call__(self, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mCos, mSin, mX4, mSF4, mNvS,
                 mAcc, mStats, mTok, mTokL, mTokI, mRowEnd, seed: Int64, lse_shift: Float32, stream: cuda.CUstream,
                 mSrc=None, mVt=None, mEvR=None, mSF4R=None, mDR=None, mLseR=None, mNT=None):
        grid = [mX.shape[1] // 128, mX.shape[2], mX.shape[0]]
        if const_expr(mSrc is not None):   # lp_docmaps gather: mX is the packed source, the tiles are the codes'
            grid = [mQ8.shape[1] // 128, mX.shape[2], mX.shape[0]]
            if const_expr(self.packed_out):   # (packed rows: the tiles are the row map's)
                grid = [mSrc.shape[0] // 128, mX.shape[2], mX.shape[0]]
        if const_expr(self.compact):
            # persistent over the device tile count: min(tile bound, 2 ceil(T / 128)) CTAs a head -- one tile a CTA
            # unless N > T / 128 (N = T: about 64 tiles a CTA, not T x heads CTAs that exit)
            grid = [cutlass.min(mVt.shape[1], 2 * ((mX.shape[1] + 127) // 128)), mX.shape[2], mX.shape[0]]
        if const_expr(self.token_local):
            assert self.streaming and not self.pair, "token_local_exponents: the streaming block kernel"
        if const_expr(self.pair and self.tc):
            self.kernel_tc(mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd, seed,
                           mSrc, *((mEvR, mSF4R) if const_expr(self.dense_out) else ()),
                           *((mVt, mNT) if const_expr(self.compact) else ())).launch(
                grid=grid, block=[256, 1, 1], smem=256, stream=stream, min_blocks_per_mp=self.tc_min_ctas)
        elif const_expr(self.pair and self.pair_ctas > 0):
            self.kernel_pair_persistent(mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd,
                                        seed).launch(
                grid=[self.pair_ctas, 1, 1], block=[256, 1, 1], smem=2 * 256 * 128 + 1024, stream=stream)
        elif const_expr(self.pair):
            self.kernel_pair(mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd, seed).launch(
                grid=grid, block=[256, 1, 1], smem=512, stream=stream, min_blocks_per_mp=self.pair_min_ctas)
        elif const_expr(self.streaming):
            self.kernel_stream(mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mX4, mSF4, mNvS, mAcc,
                               mCos, mSin, mStats, mTok, mTokL, mTokI, mRowEnd, seed, lse_shift, mSrc,
                               *((mVt,) if const_expr(mVt is not None) else ()),
                               *((mEvR, mSF4R, mDR, mLseR) if const_expr(self.dense_out) else ()),
                               *((mNT,) if const_expr(self.compact) else ())).launch(
                grid=grid, block=[128, 1, 1], smem=self.smem_bytes, stream=stream)
        else:
            self.kernel(mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mCos, mSin, mX4, mSF4, mNvS,
                        mAcc, seed, lse_shift).launch(grid=grid, block=[128, 1, 1], smem=self.smem_bytes, stream=stream)

    @cute.kernel
    def kernel_pair(self, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd,
                    seed: Int64):
        """RoPE / rotation token blocks (block_rows 32), 256 threads: lane pair (2 r, 2 r + 1) owns token row r, lane
        parity h its columns 64 h .. 64 h + 63 -- the half row is loaded once (eight 16-byte loads), transformed
        (_pair_transform) and quantised from registers. Block maxima: the lane's two 32-column groups, a butterfly over
        the warp's 16 rows (xor 2 .. 16), the two warps of a 32-row block through shared memory; then kernel_stream's
        selector, codes, NVFP4 copy and tables. One CTA per tile (kernel_pair_persistent is the default schedule)."""
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        r = tidx // 2
        h = tidx % 2
        dt = self.in_dtype
        smem = cutlass.utils.SmemAllocator()
        sa = smem.allocate_tensor(Float32, cute.make_layout(32), 16)     # [warp][column group]
        sb = smem.allocate_tensor(Int32, cute.make_layout(16), 16)       # [token block][column group]: the bytes
        sfr_s = smem.allocate_tensor(Float32, cute.make_layout(16), 16)  # ... the scales 2^e
        inv_s = smem.allocate_tensor(Float32, cute.make_layout(16), 16)  # ... 1 / scale
        if const_expr(self.pair_coalesced):
            rbw = self._pair_load_coalesced(mX, mX[bb, t * 128 + r, hh, None].iterator, h)
            rb = cute.make_tensor(cute.recast_ptr(rbw.iterator, dtype=dt), cute.make_layout(64))
        else:
            rb = cute.make_rmem_tensor(cute.make_layout(64), dt)
            cute.autovec_copy(cute.make_tensor(cute.make_ptr(dt, (mX[bb, t * 128 + r, hh, None].iterator + 64 * h).toint(),
                                                             cute.AddressSpace.gmem, assumed_align=16),
                                               cute.make_layout(64)), rb)
        self._pair_tile(rb, t, hh, bb, sa, sb, sfr_s, inv_s, Int32(0), mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4,
                        mNvS, mCos, mSin, mStats, seed, mRowEnd)

    @cute.jit
    def _pair_load_coalesced(self, mX, row_it, h):
        """The lane's half row (columns 64 h .. 64 h + 63) as 32 words, loaded sector-complete: instruction m (0..7)
        reads row chunk 2 m + h (16 bytes), so a lane pair reads 32 contiguous bytes (the plain half-row load reads 16
        of every 32 per instruction: ncu 16.0 / 32 bytes per sector). Lane 0 then holds chunks 0, 2, .., 14 and lane 1
        chunks 1, 3, .., 15; one xor-1 exchange of 4 chunks (16 words) puts chunks 8 h .. 8 h + 7 in order."""
        dt = self.in_dtype
        per = 16 // (dt.width // 8)  # elements per 16-byte chunk
        tmp = cute.make_rmem_tensor(cute.make_layout(32), Int32)
        for m in cutlass.range_constexpr(8):
            src = cute.make_tensor(cute.make_ptr(Int32, (row_it + (2 * m + h) * per).toint(), cute.AddressSpace.gmem,
                                                 assumed_align=16), cute.make_layout(4))
            cute.autovec_copy(src, cute.make_tensor(tmp.iterator + 4 * m, cute.make_layout(4)))
        out = cute.make_rmem_tensor(cute.make_layout(32), Int32)
        for i in cutlass.range_constexpr(4):
            for w in cutlass.range_constexpr(4):
                lo = tmp[4 * i + w]          # chunk 2 i + h
                hi = tmp[4 * (4 + i) + w]    # chunk 8 + 2 i + h
                send = Int32(cutlass.select_(h == 1, lo, hi))  # lane 0 gives chunk 8 + 2 i, lane 1 chunk 2 i + 1
                keep = Int32(cutlass.select_(h == 1, hi, lo))  # lane 0 keeps chunk 2 i, lane 1 chunk 9 + 2 i
                got = cute.arch.shuffle_sync_bfly(send, offset=1)
                out[4 * (2 * i) + w] = Int32(cutlass.select_(h == 1, got, keep))
                out[4 * (2 * i + 1) + w] = Int32(cutlass.select_(h == 1, keep, got))
        return out

    @cute.kernel
    def kernel_pair_persistent(self, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats,
                               mRowEnd, seed: Int64):
        """kernel_pair's tiles on a persistent schedule: kernel_pair was
        latency-bound -- one tile per CTA at 2-3 CTAs / SM, load -> rotate -> barriers -> store with nothing in flight
        meanwhile (ncu: 25 % DRAM, 37 % issue). Here grid.x CTAs loop over the (t, head, batch) tiles with stride grid.x,
        and while one tile is rotated and quantised the NEXT tile's half rows are already on their way: each thread
        cp.asyncs its own 128 bytes into its own shared-memory slot (double-buffered; the 16-byte chunk k of thread T at
        slot position (k + T) % 8, so a quarter warp's 16-byte reads hit distinct banks), waits for its own group only
        (no barrier: a thread reads back only what it copied), and runs kernel_pair's body (_pair_tile, the same
        arithmetic: bitwise). The block-maximum scratch is double-buffered by tile parity, so no extra barrier either."""
        tidx, _, _ = cute.arch.thread_idx()
        lin0, _, _ = cute.arch.block_idx()
        nct, _, _ = cute.arch.grid_dim()
        tiles = mX.shape[1] // 128
        nheads = mX.shape[2]
        total = tiles * nheads * mX.shape[0]
        r = tidx // 2
        h = tidx % 2
        dt = self.in_dtype
        smem = cutlass.utils.SmemAllocator()
        sa = smem.allocate_tensor(Float32, cute.make_layout(64), 16)
        sb = smem.allocate_tensor(Int32, cute.make_layout(32), 16)
        sfr_s = smem.allocate_tensor(Float32, cute.make_layout(32), 16)
        inv_s = smem.allocate_tensor(Float32, cute.make_layout(32), 16)
        sbuf = smem.allocate_tensor(Int32, cute.make_layout(2 * 256 * 32), 128)  # 2 x (256 threads x 128 bytes)
        base = Int32(sbuf.iterator.toint()) + tidx * 128
        rot = tidx % 8
        if lin0 < total:
            self._pair_prefetch(mX, lin0, Int32(0), base, rot, tiles, nheads, r, h)
        cp_async_commit()
        for idx in cutlass.range(lin0, total, nct, unroll=1):
            buf = ((idx - lin0) // nct) % 2
            nxt = idx + nct
            if nxt < total:
                self._pair_prefetch(mX, nxt, 1 - buf, base, rot, tiles, nheads, r, h)
            cp_async_commit()
            cp_async_wait_1()
            rb = cute.make_rmem_tensor(cute.make_layout(64), dt)
            for k in cutlass.range_constexpr(8):
                src = cute.make_tensor(cute.make_ptr(dt, base + buf * 32768 + ((k + rot) % 8) * 16, cute.AddressSpace.smem,
                                                     assumed_align=16), cute.make_layout(8))
                cute.autovec_copy(src, cute.make_tensor(rb.iterator + 8 * k, cute.make_layout(8)))
            t = idx % tiles
            bh = idx // tiles
            self._pair_tile(rb, t, bh % nheads, bh // nheads, sa, sb, sfr_s, inv_s, buf, mX, mQ8, mSFTok, mSFDim, mExp,
                            mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, seed, mRowEnd)

    @cute.jit
    def _pair_prefetch(self, mX, idx, buf, base, rot, tiles, nheads, r, h):
        """cp.async this thread's 128-byte half row of tile idx into its slot of buffer buf (chunk k at (k + rot) % 8)."""
        t_ = idx % tiles
        bh_ = idx // tiles
        g = mX[bh_ // nheads, t_ * 128 + r, bh_ % nheads, None].iterator + 64 * h
        for k in cutlass.range_constexpr(8):
            cp_async_16(base + buf * 32768 + ((k + rot) % 8) * 16, g + 8 * k)

    @cute.jit
    def _pair_tile(self, rb, t, hh, bb, sa, sb, sfr_s, inv_s, par, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS,
                   mCos, mSin, mStats, seed: Int64, mRowEnd=None):
        """kernel_pair's body for one tile, from the thread's raw half row rb. par selects the block-maximum scratch
        (sa 32 floats, sb / sfr_s / inv_s 16 words each, at offset par x size): 0 for one tile per CTA."""
        tidx, _, _ = cute.arch.thread_idx()
        s = mX.shape[1]
        tiles = s // 128
        nheads = mX.shape[2]
        bh = bb * nheads + hh
        q0 = t * 128
        r = tidx // 2
        h = tidx % 2
        lane = tidx % 32
        warp = tidx // 32
        rl = r % 32                     # the row's place in the 128x4 SF atom
        rw = r // 32                    # = its 32-row token block
        fmax = self.fmax
        dt = self.in_dtype
        pa = par * 32
        pb = par * 16
        if const_expr(mRowEnd is not None):
            # aligned varlen padding (row_end): a row past its document's true length is a zero row
            keep = q0 + r < mRowEnd[bb, t]
            for e in cutlass.range_constexpr(64):
                rb[e] = cutlass.select_(keep, rb[e], dt(0.0))
        v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        for e in cutlass.range_constexpr(64):
            v[e] = Float32(rb[e])
        v = _pair_transform(v, h, q0 + r, mCos, mSin, self.rope, self.rope_transposed, self.signs, dt)

        if const_expr(self.nvfp4 and self.nvfp4_scaling != "token_block"):
            # raw / per_tensor: no block scale needed, so before the barrier (its registers free again for the codes)
            self._pair_nvfp4(v, None, h, r, rl, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed)
        # ---- block maxima (pass 1's values: fmax of |x| is order-free)
        am = cute.make_rmem_tensor(cute.make_layout(2), Float32)
        for j in cutlass.range_constexpr(2):
            m = Float32(0.0)
            for e in cutlass.range_constexpr(32):
                m = cute.arch.fmax(m, cute.math.absf(v[32 * j + e]))
            am[j] = m
        if const_expr(self.cols >= 64):
            m = cute.arch.fmax(am[0], am[1])
            if const_expr(self.cols == 128):
                m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1))
            am[0] = m
            am[1] = m
        for i in cutlass.range_constexpr(1, 5):       # the warp's 16 rows with the same column half
            for j in cutlass.range_constexpr(2):
                am[j] = cute.arch.fmax(am[j], cute.arch.shuffle_sync_bfly(am[j], offset=1 << i))
        if lane < 2:
            sa[pa + warp * 4 + 2 * lane] = am[0]
            sa[pa + warp * 4 + 2 * lane + 1] = am[1]
        inv2 = cute.make_rmem_tensor(cute.make_layout(2), Float32)
        byte4 = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        if const_expr(self.pair_named):
            # the token block's two warps only (named barrier 1 + rw, 64 threads): every thread selects its block's
            # exponents itself from the same maxima (the same _ue8m0_select inputs as the 16 selecting threads below)
            cute.arch.barrier(barrier_id=1 + rw, number_of_threads=64)
            inv4 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for g in cutlass.range_constexpr(4):
                if const_expr(g == 0 or self.cols < 128):
                    a = cute.arch.fmax(sa[pa + 8 * rw + g], sa[pa + 8 * rw + 4 + g])
                    byte_g, _sfr, inv_g = _ue8m0_select(a, fmax, self.flush_subnormals)
                    byte4[g] = byte_g
                    inv4[g] = inv_g
                else:
                    byte4[g] = byte4[0]
                    inv4[g] = inv4[0]
            for j in cutlass.range_constexpr(2):
                inv2[j] = inv4[j]
                if h == 1:
                    inv2[j] = inv4[2 + j]
        else:
            cute.arch.sync_threads()
            if const_expr(self.stat_amax):
                if warp == 1:  # output_amax: the tile's max |x R|, a butterfly over the 32 exchanged maxima (warp 1:
                    # idle in the selection below)
                    tmax = sa[pa + lane]
                    for lv in cutlass.range_constexpr(5):
                        tmax = cute.arch.fmax(tmax, cute.arch.shuffle_sync_bfly(tmax, offset=1 << lv))
                    if lane == 0:
                        mStats[bh * tiles + t] = tmax
            if tidx < 16:
                tb = tidx // 4
                g = tidx % 4
                a = cute.arch.fmax(sa[pa + 8 * tb + g], sa[pa + 8 * tb + 4 + g])
                byte, sfr, inv = _ue8m0_select(a, fmax, self.flush_subnormals)
                sb[pb + tidx] = byte
                sfr_s[pb + tidx] = sfr
                inv_s[pb + tidx] = inv
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr(2):
                inv2[j] = inv_s[pb + rw * 4 + 2 * h + j]

        # ---- codes from registers
        w8 = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        for w in cutlass.range_constexpr(16):
            j = w // 8
            y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for e in cutlass.range_constexpr(0, 4, 2):
                y[e], y[e + 1] = cute.arch.mul_packed_f32x2((v[4 * w + e], v[4 * w + e + 1]), (inv2[j], inv2[j]),
                                                            ftz=True if const_expr(self.flush_subnormals) else None)
                y[e] = _clamp(y[e], fmax)
                y[e + 1] = _clamp(y[e + 1], fmax)
            w8[w] = _pack4(self.fmt, y[0], y[1], y[2], y[3])
        if const_expr(self.pair_coalesced):
            # sector-complete code stores: instruction m writes row chunk 2 m + h (16 bytes); lane 0 owns chunks 0..3
            # (w8 words 4 q .. 4 q + 3 = chunk q), lane 1 chunks 4..7 -- one xor-1 exchange of 2 chunks (8 words)
            qrow = mQ8[bb, q0 + r, hh, None].iterator
            got = cute.make_rmem_tensor(cute.make_layout(8), Int32)
            for i in cutlass.range_constexpr(2):
                for w in cutlass.range_constexpr(4):
                    send = Int32(cutlass.select_(h == 1, w8[4 * (2 * i) + w], w8[4 * (2 * i + 1) + w]))
                    got[4 * i + w] = cute.arch.shuffle_sync_bfly(send, offset=1)
            for m in cutlass.range_constexpr(4):
                st = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                for w in cutlass.range_constexpr(4):
                    if const_expr(m < 2):
                        st[w] = Int32(cutlass.select_(h == 1, got[4 * m + w], w8[4 * (2 * m) + w]))
                    else:
                        st[w] = Int32(cutlass.select_(h == 1, w8[4 * (2 * (m - 2) + 1) + w], got[4 * (m - 2) + w]))
                cute.autovec_copy(st, cute.make_tensor(cute.make_ptr(Int32, (qrow + 4 * (2 * m + h)).toint(),
                                                                     cute.AddressSpace.gmem, assumed_align=16),
                                                       cute.make_layout(4)))
        else:
            cute.autovec_copy(w8, cute.make_tensor(cute.make_ptr(Int32, (mQ8[bb, q0 + r, hh, None].iterator + 16 * h).toint(),
                                                                 cute.AddressSpace.gmem, assumed_align=16),
                                                   cute.make_layout(16)))
        if const_expr(self.nvfp4 and self.nvfp4_scaling == "token_block"):
            self._pair_nvfp4(v, inv2, h, r, rl, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed)

        # ---- tables (kernel_stream's words)
        if const_expr(self.pair_named):
            # per token block: sf_tok words (the 4 bytes of the row's block), sf_dim as this block's byte of each word
            # (byte rw of word (tile, row, group)), the block's exponent bytes and factor
            if h == 0:
                mSFTok[(bh * tiles + t) * 128 + rl * 4 + rw] = (byte4[0] | (byte4[1] << 8) | (byte4[2] << 16)
                                                               | (byte4[3] << 24))
            u = tidx % 64
            for jj in cutlass.range_constexpr(2):
                gc = 2 * (u // 32) + jj
                bv = byte4[jj]
                if u >= 32:
                    bv = byte4[2 + jj]
                widx = (bh * tiles + t) * 128 + (u % 32) * 4 + gc
                bptr = cute.make_ptr(Uint8, mSFDim.iterator.toint() + Int64(widx) * Int64(4) + Int64(rw),
                                     cute.AddressSpace.gmem, assumed_align=1)
                cute.make_tensor(bptr, cute.make_layout(1))[0] = Uint8(bv)
            if const_expr(self.exponent_table):
                if u < 4:
                    bv = byte4[0]
                    for g in cutlass.range_constexpr(1, 4):
                        if u == g:
                            bv = byte4[g]
                    mExp[(bh * (tiles * 4) + 4 * t + rw) * 4 + u] = Uint8(bv)
            if const_expr(self.factors or self.stat_amax):
                # every block's maxima are in sa (double-buffered by par, so the next tile cannot overwrite them)
                cute.arch.sync_threads()
            if const_expr(self.stat_amax):
                if warp == 1:  # output_amax: the tile's max |x R| over the 32 exchanged maxima
                    tmax = sa[pa + lane]
                    for lv in cutlass.range_constexpr(5):
                        tmax = cute.arch.fmax(tmax, cute.arch.shuffle_sync_bfly(tmax, offset=1 << lv))
                    if lane == 0:
                        mStats[bh * tiles + t] = tmax
            if const_expr(self.factors):
                if tidx < 4:
                    # the 4 blocks' group-0 bytes (the same _ue8m0_select inputs as the sync branch's sb[4 tb])
                    eb = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                    for tb in cutlass.range_constexpr(4):
                        a = cute.arch.fmax(sa[pa + 8 * tb], sa[pa + 8 * tb + 4])
                        byte_t, _sfr_t, _inv_t = _ue8m0_select(a, fmax, self.flush_subnormals)
                        eb[tb] = byte_t
                    ev = eb[0]
                    for tb in cutlass.range_constexpr(1, 4):
                        if tidx == tb:
                            ev = eb[tb]
                    if ev == 0:  # an all-zero block: its tile's largest factor, 1 for an all-zero tile
                        ev = cutlass.max(cutlass.max(eb[0], eb[1]), cutlass.max(eb[2], eb[3]))
                    f = Float32(1.0)
                    if ev > 0:
                        f = ex2_ftz(Float32(ev - 127))
                    mFac[bb, hh, 4 * t + tidx] = f
        else:
            if h == 0:
                mSFTok[(bh * tiles + t) * 128 + rl * 4 + rw] = (sb[pb + rw * 4] | (sb[pb + rw * 4 + 1] << 8)
                                                               | (sb[pb + rw * 4 + 2] << 16) | (sb[pb + rw * 4 + 3] << 24))
            if tidx < 128:
                gc = tidx // 32
                mSFDim[(bh * tiles + t) * 128 + (tidx % 32) * 4 + gc] = (
                    sb[pb + gc] | (sb[pb + 4 + gc] << 8) | (sb[pb + 8 + gc] << 16) | (sb[pb + 12 + gc] << 24))
            if const_expr(self.exponent_table):
                if tidx < 16:
                    mExp[(bh * (tiles * 4) + 4 * t + tidx // 4) * 4 + tidx % 4] = Uint8(sb[pb + tidx])
            if const_expr(self.factors):
                if tidx < 4:
                    ev = sb[pb + tidx * 4]
                    if ev == 0:  # an all-zero block: its tile's largest factor, 1 for an all-zero tile
                        ev = cutlass.max(cutlass.max(sb[pb], sb[pb + 4]), cutlass.max(sb[pb + 8], sb[pb + 12]))
                    f = Float32(1.0)
                    if ev > 0:
                        f = ex2_ftz(Float32(ev - 127))
                    mFac[bb, hh, 4 * t + tidx] = f


    @cute.jit
    def _pair_nvfp4(self, v, inv2, h, r, rl, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed: Int64):
        # option A / the S operand: the NVFP4 1x16 copy of this half row (groups 4 h .. 4 h + 3 of the row)
        sinv = Float32(1.0)
        if const_expr(self.nvfp4_scaling == "per_tensor"):
            sinv = div_full_f32(Float32(1.0), mNvS[0])
        idx_row = ((Int64(bb) * Int64(nheads) + Int64(hh)) * Int64(s) + Int64(q0)) * Int64(4096) + Int64(r) * Int64(8)
        w4 = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        sfw = Int32(0)
        for k in cutlass.range_constexpr(4):
            v16 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            for e in cutlass.range_constexpr(16):
                xv = v[16 * k + e]
                # (qk_operand on raw values flushes by itself: abs.ftz / mul.ftz)
                if const_expr(self.flush_subnormals and not (self.nvfp4_encoding == "qk_operand"
                                                             and self.nvfp4_scaling == "raw")):
                    if cute.math.absf(xv) < Float32(FLT_MIN):
                        xv = mul_rn(xv, Float32(0.0))
                if const_expr(self.nvfp4_scaling == "token_block"):
                    xv = xv * inv2[k // 2]
                elif const_expr(self.nvfp4_scaling == "per_tensor"):
                    xv = xv * sinv
                v16[e] = xv
            byte4, _sfr4, c0, c1 = nvfp4_group16(v16, self.nvfp4_stochastic, idx_row + Int64(4 * h + k), seed,
                                                 self.nvfp4_encoding)
            w4[2 * k] = c0
            w4[2 * k + 1] = c1
            sfw = sfw | (byte4 << (8 * k))
        cute.autovec_copy(w4, cute.make_tensor(cute.make_ptr(Int32, (mX4[bb, q0 + r, hh, None].iterator + 8 * h).toint(),
                                                             cute.AddressSpace.gmem, assumed_align=16),
                                               cute.make_layout(8)))
        mSF4[(bh * tiles + t) * 256 + 128 * h + rl * 4 + rw] = sfw

    # ------------------------------------------------------------------------------------- tensor-core rotation form
    @cute.kernel
    def kernel_tc(self, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd, seed: Int64,
                  mSrc=None, mEvR=None, mSF4R=None, mVt=None, mNT=None):
        """kernel_pair with the rotation on the tensor cores (hadamard_rotation_method="tensor_core"), one
        CTA per tile: 256 threads, warp w owns rows 16 w .. 16 w + 15 of the tile in the quad layout of
        _rotations.quad_hadamard_rotate (lane (g, tq): rows 16 w + g and 16 w + g + 8, 8-column chunks 4 m + tq), loaded
        straight into the mma A fragments (a quad reads 64 contiguous bytes per instruction); _tc_tile does the rest."""
        tidx0, _, _ = cute.arch.thread_idx()   # (read before the block index: the other kernels' PTX unchanged)
        t, hh, bb = cute.arch.block_idx()
        if const_expr(self.compact):
            # sfd compact maps: persistent over the device tile count; the tile's packed rows from the tile table
            if t < mNT[0]:
                self._tc_one(t, hh, bb, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats,
                             mRowEnd, seed, (mVt[0, t], mVt[1, t]), mEvR, mSF4R, mVt.shape[1] * 128, tidx0=tidx0)
            for tt in cutlass.range(t + cute.arch.grid_dim()[0], mNT[0], cute.arch.grid_dim()[0], unroll=1):
                cute.arch.sync_threads()
                self._tc_one(tt, hh, bb, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats,
                             mRowEnd, seed, (mVt[0, tt], mVt[1, tt]), mEvR, mSF4R, mVt.shape[1] * 128, tidx0=tidx0)
        else:
            self._tc_one(t, hh, bb, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd,
                         seed, mSrc, mEvR, mSF4R, None, tidx0=tidx0)

    @cute.jit
    def _psrc(self, mSrc, q0, x):
        """The packed row of row q0 + x of the tile (< 0: none): the row map, or (sfd compact) the tile's
        (packed start, real rows) pair."""
        if const_expr(isinstance(mSrc, tuple)):
            return cutlass.select_(x < mSrc[1], mSrc[0] + x, Int32(-1))
        return mSrc[q0 + x]

    @cute.jit
    def _tc_one(self, t, hh, bb, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats, mRowEnd,
                seed: Int64, mSrc, mEvR, mSF4R, s_tiles, tidx0=None):
        """kernel_tc's body for tile t of head hh, batch row bb (mSrc: the row map, or the compact tile pair)."""
        tidx = tidx0
        lane = tidx % 32
        warp = tidx // 32
        smem = cutlass.utils.SmemAllocator()
        sa = smem.allocate_tensor(Float32, cute.make_layout(32), 16)     # [warp][column group]
        a = cute.make_rmem_tensor(cute.make_layout(32), Int32)
        if const_expr(mSrc is not None):
            # lp_docmaps gather: row r of the tile is row mSrc[r] of the packed mX, a zero row where mSrc[r] < 0
            for i in cutlass.range_constexpr(2):
                sr = self._psrc(mSrc, t * 128, 16 * warp + lane // 4 + 8 * i)
                it = mX[bb, cutlass.max(sr, Int32(0)), hh, None].iterator
                for m in cutlass.range_constexpr(4):
                    cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, (it + 32 * m + 8 * (lane % 4)).toint(),
                                                                     cute.AddressSpace.gmem, assumed_align=16),
                                                       cute.make_layout(4)),
                                      cute.make_tensor(a.iterator + 16 * i + 4 * m, cute.make_layout(4)))
                for j in cutlass.range_constexpr(16):
                    a[16 * i + j] = cutlass.select_(sr >= Int32(0), a[16 * i + j], Int32(0))
            self._tc_tile(a, t, hh, bb, sa, Int32(0), mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos,
                          mSin, mStats, seed, gather=True, mSrc=mSrc if const_expr(self.packed_out) else None,
                          mEvR=mEvR, mSF4R=mSF4R, s_tiles=s_tiles)
            return
        for i in cutlass.range_constexpr(2):
            it = mX[bb, t * 128 + 16 * warp + lane // 4 + 8 * i, hh, None].iterator
            for m in cutlass.range_constexpr(4):
                cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, (it + 32 * m + 8 * (lane % 4)).toint(),
                                                                 cute.AddressSpace.gmem, assumed_align=16),
                                                   cute.make_layout(4)),
                                  cute.make_tensor(a.iterator + 16 * i + 4 * m, cute.make_layout(4)))
        if const_expr(mRowEnd is not None):
            # aligned varlen padding (row_end): a row past its document's true length is a zero row
            for i in cutlass.range_constexpr(2):
                keep = t * 128 + 16 * warp + lane // 4 + 8 * i < mRowEnd[bb, t]
                for j in cutlass.range_constexpr(16):
                    a[16 * i + j] = cutlass.select_(keep, a[16 * i + j], Int32(0))
        self._tc_tile(a, t, hh, bb, sa, Int32(0), mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin,
                      mStats, seed)

    @cute.jit
    def _tc_tile(self, a, t, hh, bb, sa, par, mX, mQ8, mSFTok, mSFDim, mExp, mFac, mX4, mSF4, mNvS, mCos, mSin, mStats,
                 seed: Int64, gather: cutlass.Constexpr = False, mSrc=None, mEvR=None, mSF4R=None, s_tiles=None):
        """kernel_tc's body for one tile from the lane's packed 16-bit inputs a (quad layout): RoPE in the lane (column
        c + 64 is chunk m + 2), 16 mma.sync.m16n8k8 + a 4-level butterfly per warp, then kernel_pair's arithmetic on the
        same values (block maxima, the token block's named barrier and selector, codes, the NVFP4 copy, the tables) -- only
        the layout of the work differs, so the outputs equal kernel_pair's wherever the rotated values do. par selects
        the block-maximum scratch (sa, 32 floats at offset 32 par)."""
        tidx, _, _ = cute.arch.thread_idx()
        s = mX.shape[1]
        if const_expr(gather):   # (lp_docmaps gather: mX is the packed source, the tiles are the codes')
            s = mQ8.shape[1]
            if const_expr(isinstance(mSrc, tuple)):   # (sfd compact: the tile table's)
                s = s_tiles
            elif const_expr(mSrc is not None):   # (packed rows: the tiles are the row map's)
                s = mSrc.shape[0]
        tiles = s // 128
        nheads = mX.shape[2]
        bh = bb * nheads + hh
        q0 = t * 128
        lane = tidx % 32
        warp = tidx // 32
        g = lane // 4
        tq = lane % 4
        rw = warp // 2                  # the warp's 32-token block
        fmax = self.fmax
        dt = self.in_dtype
        pa = par * 32
        if const_expr(self.rope):
            quad_rope(a, tq, q0 + 16 * warp + g, q0 + 16 * warp + g + 8, mCos, mSin, self.rope_transposed, dt)
        v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        inv4 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        byte4 = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        # the values the FP8 codes / tables / factors / amax are of: the rotated rows (v), or, with
        # rotate_nvfp4_copy_only, the raw rows: block maxima on the packed 16-bit inputs, the codes from their exact fp32
        # values (a's element 32 i + 8 m + e is v's), and the rotation only afterwards, for the NVFP4 copy
        raw = self.rotate_nvfp4_copy_only
        half = "f16" if const_expr(dt is cutlass.Float16) else "bf16"
        if const_expr(not self.rotate_nvfp4_copy_only):
            quad_hadamard_rotate(a, v, lane, self.quad_signs, dt is cutlass.Float16)
            if const_expr(self.nvfp4 and self.nvfp4_scaling != "token_block"):
                self._tc_nvfp4(v, inv4, tq, warp, g, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed,
                               mSrc, mSF4R)
        # ---- block maxima: the lane's chunk m is in 32-column group m; a reduce-scatter over the quad leaves group tq's
        # maximum in lane tq, then the 8 row pairs of the warp (xor 4, 8, 16)
        am = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for m in cutlass.range_constexpr(4):
            if const_expr(raw):  # a packed |x| max of the 16-bit inputs (exact), as kernel_stream's
                acc = Int32(0)
                for i in cutlass.range_constexpr(2):
                    for p in cutlass.range_constexpr(4):
                        acc = absmax_x2(acc, a[16 * i + 4 * m + p], half)
                am[m] = cute.arch.fmax(half_lo_f32(acc, half), half_hi_f32(acc, half))
            else:
                mx = Float32(0.0)
                for i in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(8):
                        mx = cute.arch.fmax(mx, cute.math.absf(v[32 * i + 8 * m + e]))
                am[m] = mx
        gmax = Float32(0.0)
        if const_expr(self.cols == 128):
            gmax = cute.arch.fmax(cute.arch.fmax(am[0], am[1]), cute.arch.fmax(am[2], am[3]))
            for lv in cutlass.range_constexpr(5):
                gmax = cute.arch.fmax(gmax, cute.arch.shuffle_sync_bfly(gmax, offset=1 << lv))
        else:
            if const_expr(self.cols == 64):
                am[0] = cute.arch.fmax(am[0], am[1])
                am[1] = am[0]
                am[2] = cute.arch.fmax(am[2], am[3])
                am[3] = am[2]
            gmax = _quad_scatter_max(am, tq)
            for lv in cutlass.range_constexpr(2, 5):
                gmax = cute.arch.fmax(gmax, cute.arch.shuffle_sync_bfly(gmax, offset=1 << lv))
        if lane < 4:
            sa[pa + warp * 4 + lane] = gmax
        # the token block's two warps (named barrier 1 + rw, 64 threads); every thread selects its block's exponents
        cute.arch.barrier(barrier_id=1 + rw, number_of_threads=64)
        for gq in cutlass.range_constexpr(4):
            if const_expr(gq == 0 or self.cols < 128):
                amx = cute.arch.fmax(sa[pa + 8 * rw + gq], sa[pa + 8 * rw + 4 + gq])
                byte_g, _sfr, inv_g = _ue8m0_select(amx, fmax, self.flush_subnormals)
                byte4[gq] = byte_g
                inv4[gq] = inv_g
            else:
                byte4[gq] = byte4[0]
                inv4[gq] = inv4[0]

        # ---- codes from registers: chunk (i, m) -> 8 bytes at row byte 8 (4 m + tq) (a quad writes 32 contiguous bytes)
        for i in cutlass.range_constexpr(2):
            if const_expr(mSrc is not None):   # packed rows: the row's packed row (< 0: none, nothing written)
                prow_i = self._psrc(mSrc, q0, 16 * warp + g + 8 * i)
                qrow = mQ8[bb, cutlass.max(prow_i, Int32(0)), hh, None].iterator
            else:
                qrow = mQ8[bb, q0 + 16 * warp + g + 8 * i, hh, None].iterator
            for m in cutlass.range_constexpr(4):
                w2 = cute.make_rmem_tensor(cute.make_layout(2), Int32)
                for hw in cutlass.range_constexpr(2):
                    y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                    for e in cutlass.range_constexpr(0, 4, 2):
                        y[e], y[e + 1] = cute.arch.mul_packed_f32x2(
                            (half_lo_f32(a[16 * i + 4 * m + 2 * hw + e // 2], half) if raw
                             else v[32 * i + 8 * m + 4 * hw + e],
                             half_hi_f32(a[16 * i + 4 * m + 2 * hw + e // 2], half) if raw
                             else v[32 * i + 8 * m + 4 * hw + e + 1]),
                            (inv4[m], inv4[m]),
                            ftz=True if const_expr(self.flush_subnormals) else None)
                        y[e] = _clamp(y[e], fmax)
                        y[e + 1] = _clamp(y[e + 1], fmax)
                    w2[hw] = _pack4(self.fmt, y[0], y[1], y[2], y[3])
                if const_expr(mSrc is not None):
                    if prow_i >= 0:
                        cute.autovec_copy(w2, cute.make_tensor(cute.make_ptr(Int32, (qrow + 2 * (4 * m + tq)).toint(),
                                                                             cute.AddressSpace.gmem, assumed_align=8),
                                                               cute.make_layout(2)))
                else:
                    cute.autovec_copy(w2, cute.make_tensor(cute.make_ptr(Int32, (qrow + 2 * (4 * m + tq)).toint(),
                                                                         cute.AddressSpace.gmem, assumed_align=8),
                                                           cute.make_layout(2)))
        if const_expr(self.rotate_nvfp4_copy_only):
            # the NVFP4 copy of the ROTATED rows (scaling "raw": independent of the FP8 blocks above)
            quad_hadamard_rotate(a, v, lane, self.quad_signs, dt is cutlass.Float16)
            self._tc_nvfp4(v, inv4, tq, warp, g, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed,
                           mSrc, mSF4R)
        elif const_expr(self.nvfp4 and self.nvfp4_scaling == "token_block"):
            self._tc_nvfp4(v, inv4, tq, warp, g, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed,
                           mSrc, mSF4R)

        # ---- tables: kernel_pair's named-barrier words (sf_tok: lane tq < 2 writes row g + 8 tq of the warp)
        if tq < 2:
            rl = (16 * warp + g + 8 * tq) % 32
            if const_expr(not self.compact):
                mSFTok[(bh * tiles + t) * 128 + rl * 4 + rw] = (byte4[0] | (byte4[1] << 8) | (byte4[2] << 16)
                                                               | (byte4[3] << 24))
        u = tidx % 64
        for jj in cutlass.range_constexpr(2):
            gc = 2 * (u // 32) + jj
            bv = byte4[jj]
            if u >= 32:
                bv = byte4[2 + jj]
            widx = (bh * tiles + t) * 128 + (u % 32) * 4 + gc
            if const_expr(not self.compact):
                bptr = cute.make_ptr(Uint8, mSFDim.iterator.toint() + Int64(widx) * Int64(4) + Int64(rw),
                                     cute.AddressSpace.gmem, assumed_align=1)
                cute.make_tensor(bptr, cute.make_layout(1))[0] = Uint8(bv)
        if const_expr(self.exponent_table and not self.compact):
            if u < 4:
                bv = byte4[0]
                for gq in cutlass.range_constexpr(1, 4):
                    if u == gq:
                        bv = byte4[gq]
                mExp[(bh * (tiles * 4) + 4 * t + rw) * 4 + u] = Uint8(bv)
        if const_expr(self.factors or self.stat_amax):
            cute.arch.sync_threads()
        if const_expr(self.stat_amax):
            if warp == 1:  # output_amax: the tile's max |x R| over the 32 exchanged maxima
                tmax = sa[pa + lane]
                for lv in cutlass.range_constexpr(5):
                    tmax = cute.arch.fmax(tmax, cute.arch.shuffle_sync_bfly(tmax, offset=1 << lv))
                if lane == 0:
                    mStats[bh * tiles + t] = tmax
        if const_expr(self.factors):
            if tidx < 4:
                eb = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                for tb in cutlass.range_constexpr(4):
                    amx = cute.arch.fmax(sa[pa + 8 * tb], sa[pa + 8 * tb + 4])
                    byte_t, _sfr_t, _inv_t = _ue8m0_select(amx, fmax, self.flush_subnormals)
                    eb[tb] = byte_t
                ev = eb[0]
                for tb in cutlass.range_constexpr(1, 4):
                    if tidx == tb:
                        ev = eb[tb]
                if ev == 0:  # an all-zero block: its tile's largest factor, 1 for an all-zero tile
                    ev = cutlass.max(cutlass.max(eb[0], eb[1]), cutlass.max(eb[2], eb[3]))
                f = Float32(1.0)
                if ev > 0:
                    f = ex2_ftz(Float32(ev - 127))
                if const_expr(not self.compact):
                    mFac[bb, hh, 4 * t + tidx] = f
        if const_expr(self.dense_out):
            # sfd: each real row's block's effective exponent (the factor rule), at its packed row; tile 0 zeroes the
            # tail rows [T, T + 128)
            if const_expr(not (self.factors or self.stat_amax)):
                cute.arch.sync_threads()
            if tidx < 128:
                ebd = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                for tb in cutlass.range_constexpr(4):
                    amx = cute.arch.fmax(sa[pa + 8 * tb], sa[pa + 8 * tb + 4])
                    byte_t, _sfr_t, _inv_t = _ue8m0_select(amx, fmax, self.flush_subnormals)
                    ebd[tb] = byte_t
                evd = ebd[0]
                for tb in cutlass.range_constexpr(1, 4):
                    if tidx // 32 == tb:
                        evd = ebd[tb]
                if evd == 0:
                    evd = cutlass.max(cutlass.max(ebd[0], ebd[1]), cutlass.max(ebd[2], ebd[3]))
                prow_e = self._psrc(mSrc, q0, tidx)
                if prow_e >= 0:
                    mEvR[bh, prow_e] = Uint8(evd)
                if t == 0:
                    mEvR[bh, mQ8.shape[1] + tidx] = Uint8(0)

    @cute.jit
    def _tc_nvfp4(self, v, inv4, tq, warp, g, rw, bb, hh, bh, q0, t, tiles, s, nheads, mX4, mSF4, mNvS, seed: Int64,
                  mSrc=None, mSF4R=None):
        """kernel_tc's NVFP4 1x16 copy: chunk (i, m) is half of 16-group 2 m + tq / 2 of row 16 w + g + 8 i (elements
        8 (tq % 2) ..), the lane pair (tq, tq ^ 1) holding the two halves (nvfp4_half_group8: the same per-element
        arithmetic as _pair_nvfp4's nvfp4_group16, and the same stochastic-stream indices). Code word 4 m + tq of the
        row; the SF words assembled from both lane pairs of the quad (shfl.xor 2)."""
        sinv = Float32(1.0)
        if const_expr(self.nvfp4_scaling == "per_tensor"):
            sinv = div_full_f32(Float32(1.0), mNvS[0])
        par = tq >> 1
        sfb = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        for i in cutlass.range_constexpr(2):
            r = 16 * warp + g + 8 * i
            idx_row = ((Int64(bb) * Int64(nheads) + Int64(hh)) * Int64(s) + Int64(q0)) * Int64(4096) + Int64(r) * Int64(8)
            if const_expr(mSrc is not None):   # packed rows (< 0: nothing written)
                prow_r = self._psrc(mSrc, q0, r)
                xrow = mX4[bb, cutlass.max(prow_r, Int32(0)), hh, None].iterator
            else:
                xrow = mX4[bb, q0 + r, hh, None].iterator
            for m in cutlass.range_constexpr(4):
                v8 = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                for e in cutlass.range_constexpr(8):
                    xv = v[32 * i + 8 * m + e]
                    if const_expr(self.flush_subnormals and not (self.nvfp4_encoding == "qk_operand"
                                                                 and self.nvfp4_scaling == "raw")):
                        if cute.math.absf(xv) < Float32(FLT_MIN):
                            xv = mul_rn(xv, Float32(0.0))
                    if const_expr(self.nvfp4_scaling == "token_block"):
                        xv = xv * inv4[m]
                    elif const_expr(self.nvfp4_scaling == "per_tensor"):
                        xv = xv * sinv
                    v8[e] = xv
                byte, w = nvfp4_half_group8(v8, self.nvfp4_stochastic, idx_row + Int64(2 * m) + Int64(par),
                                            8 * (tq & 1), seed, self.nvfp4_encoding)
                if const_expr(mSrc is not None):
                    if prow_r >= 0:
                        cute.make_tensor(cute.make_ptr(Int32, (xrow + 4 * m + tq).toint(), cute.AddressSpace.gmem,
                                                       assumed_align=4), cute.make_layout(1))[0] = w
                else:
                    cute.make_tensor(cute.make_ptr(Int32, (xrow + 4 * m + tq).toint(), cute.AddressSpace.gmem,
                                                   assumed_align=4), cute.make_layout(1))[0] = w
                sfb[4 * i + m] = byte
        # word h of row i holds groups 4 h .. 4 h + 3 = (m = 2 h, 2 h + 1) x (parity 0, 1); lane tq writes (i, h) =
        # (tq % 2, tq / 2)
        mine = Int32(0)
        for i in cutlass.range_constexpr(2):
            for hw in cutlass.range_constexpr(2):
                own = (sfb[4 * i + 2 * hw] << (8 * par)) | (sfb[4 * i + 2 * hw + 1] << (8 * (2 + par)))
                full = own | cute.arch.shuffle_sync_bfly(own, offset=2)
                if tq == i + 2 * hw:
                    mine = full
        rl = (16 * warp + g + 8 * (tq & 1)) % 32
        if const_expr(not self.compact):
            mSF4[(bh * tiles + t) * 256 + 128 * (tq >> 1) + rl * 4 + rw] = mine
        if const_expr(self.dense_out):   # sfd: the row's SF word of atom tq / 2, at its packed row
            prow_d = self._psrc(mSrc, q0, 32 * rw + rl)
            if prow_d >= 0:
                mSF4R[bh, prow_d * 2 + (tq >> 1)] = mine


    @cute.kernel
    def kernel_stream(self, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mX4, mSF4, mNvS,
                      mAcc, mCos, mSin, mStats, mTok, mTokL, mTokI, mRowEnd, seed: Int64, lse_shift: Float32,
                      mSrc=None, mVt=None, mEvR=None, mSF4R=None, mDR=None, mLseR=None, mNT=None):
        """The low-register form for plain rows (no RoPE / rotation / NVFP4 copy, block_columns >= 32): the row is read
        from the staged tile twice (group maxima, then codes) instead of being held in registers, the codes are written
        back into the thread's own staged row and leave the CTA as coalesced 16-byte stores. mSrc (lp_docmaps gather):
        the staged rows come from the packed mX through the row map; the tiles are the codes' (mQ8's). compact (sfd):
        persistent over the device tile count mNT[0] (tile t of a CTA: blockIdx.x + k gridDim.x)."""
        tidx0, _, _ = cute.arch.thread_idx()   # (read before the block index: the other kernels' PTX unchanged)
        t, hh, bb = cute.arch.block_idx()
        if const_expr(self.compact):
            for tt in cutlass.range(t, mNT[0], cute.arch.grid_dim()[0], unroll=1):
                self._stream_tile(tt, hh, bb, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mX4, mSF4, mNvS, mAcc, mCos, mSin, mStats, mTok, mTokL, mTokI, mRowEnd, seed, lse_shift, mSrc, mVt, mEvR, mSF4R, mDR, mLseR, tidx0=tidx0)
                cute.arch.sync_threads()
        else:
            self._stream_tile(t, hh, bb, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mX4, mSF4, mNvS, mAcc, mCos, mSin, mStats, mTok, mTokL, mTokI, mRowEnd, seed, lse_shift, mSrc, mVt, mEvR, mSF4R, mDR, mLseR, tidx0=tidx0)

    @cute.jit
    def _stream_tile(self, t, hh, bb, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mX4, mSF4,
                     mNvS, mAcc, mCos, mSin, mStats, mTok, mTokL, mTokI, mRowEnd, seed: Int64, lse_shift: Float32,
                     mSrc, mVt, mEvR, mSF4R, mDR, mLseR, tidx0=None):
        """kernel_stream's body for tile t of head hh, batch row bb."""
        tidx = tidx0
        s = mX.shape[1]
        if const_expr(mSrc is not None):
            s = mQ8.shape[1]
            if const_expr(self.packed_out):
                s = mSrc.shape[0]
        if const_expr(self.compact):   # (sfd: the tile table's tiles: 128 x the static tile bound)
            s = mVt.shape[1] * 128
        tiles = s // 128
        nheads = mX.shape[2]
        bh = bb * nheads + hh
        q0 = t * 128
        r = tidx
        lane = r % 32
        warp = r // 32
        fmax = self.fmax
        dt = self.in_dtype
        hd = self.hd
        ngd = self.ngd
        # aligned varlen padding (row_end): rows >= nv of this tile are staged as zeros (input and O)
        nv = None
        if const_expr(mRowEnd is not None and self.row_end_input):
            nv = mRowEnd[bb, t] - q0
        if const_expr(self.compact):
            # sfd compact maps: the tile's packed start and real rows from the tile table (no row map)
            pb = mVt[0, t]
            nvr = mVt[1, t]
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.lane_transform):
            # RoPE / rotation: the raw tile is staged (cp.async) at the END of the fp32 tile's buffer, then transformed by
            # a lane pair per row into the fp32 tile (below, after the wait)
            dt = Float32
            rsx = _row_pad(dt)
            sX = smem.allocate_tensor(dt, cute.make_layout((128, rsx), stride=(rsx, 1)), 16)
            rsr = _row_pad(self.in_dtype)
            sR = cute.make_tensor(cute.make_ptr(self.in_dtype, sX.iterator.toint() + _RAW_OFFSET[self.in_dtype.width],
                                                cute.AddressSpace.smem, assumed_align=16),
                                  cute.make_layout((128, rsr), stride=(rsr, 1)))
            _stage_tile(mX, sR, bb, q0, hh, self.in_dtype, nvalid=nv)
            sG = smem.allocate_tensor(Float32, cute.make_layout((128, 4), stride=(4, 1)), 16)   # the rows' group maxima
        else:
            rsx = _row_pad(dt, hd)
            sX = smem.allocate_tensor(dt, cute.make_layout((128, rsx), stride=(rsx, 1)), 16)
            if const_expr(self.compact):
                _stage_tile(mX, sX, bb, pb, hh, dt, hd, self.x_cols, nvalid=nvr, row_cap=mX.shape[1])
            elif const_expr(mSrc is not None):
                _stage_tile(mX, sX, bb, q0, hh, dt, hd, self.x_cols, row_src=mSrc)
            else:
                _stage_tile(mX, sX, bb, q0, hh, dt, hd, self.x_cols, nvalid=nv)
        if const_expr(self.d_mode is not None):
            rso = _row_pad(self.out_dtype, hd)
            sO = smem.allocate_tensor(self.out_dtype, cute.make_layout((128, rso), stride=(rso, 1)), 16)
            if const_expr(self.compact):
                _stage_tile(mO, sO, bb, pb, hh, self.out_dtype, hd, nvalid=nvr, row_cap=mO.shape[1])
            elif const_expr(mSrc is not None and self.o_gather):   # (O in the caller's packed rows too)
                _stage_tile(mO, sO, bb, q0, hh, self.out_dtype, hd, row_src=mSrc)
            else:
                _stage_tile(mO, sO, bb, q0, hh, self.out_dtype, hd, nvalid=nv)
        cp_async_wait_all()
        cute.arch.sync_threads()
        if const_expr(self.packed_out):
            # packed rows: a tile lies in one document, so its real rows are a prefix mapping to consecutive packed rows
            # -- row r of the tile is packed row pb + r for r < nvr, none past it (DocMaps: src[q0 + r] = src[q0] + r
            # there, -1 after). Two uniform loads per tile instead of one dependent load per stored row / chunk.
            if const_expr(not self.compact):
                pb = mSrc[q0]
                nvr = mVt[1, t] & Int32(0xFFFF)   # (vt_tile[_q] row 1: the tile's real rows, VT_LAST flag above bit 16)
            prow = cutlass.select_(r < nvr, pb + r, Int32(-1))   # this thread's packed row (< 0: none)
        if const_expr(self.lane_transform and self.tc5):
            # hadamard_rotation_method="tcgen05": thread r takes its staged raw row (RoPE in fp32, rounded to the input
            # dtype, as every method), the tile is rotated on tcgen05 and thread r writes its rotated row to the fp32
            # tile and its four 32-column maxima to sG (the tensor_core branch's outputs)
            rot5 = RowRotation(self.in_dtype, self.signs)
            st5 = rot5.allocate(smem)
            rot5.setup(st5, tidx)
            words = cute.make_rmem_tensor(cute.make_layout(64), Int32)
            cute.autovec_copy(cute.make_tensor(cute.recast_ptr(_smem_run(sR, r, 0, 128, self.in_dtype).iterator,
                                                               dtype=Int32), cute.make_layout(64)), words)
            if const_expr(self.rope):
                _rope_packed_row(words, mCos, mSin, q0 + r, self.rope_transposed, self.in_dtype)
            rot5.write_row(st5, r, words)
            v = cute.make_rmem_tensor(cute.make_layout(128), Float32)
            rot5.rotate(st5, r, v, Int32(0))
            for c8 in cutlass.range_constexpr(16):
                cute.autovec_copy(cute.make_tensor(v.iterator + 8 * c8, cute.make_layout(8)),
                                  _smem_run(sX, r, 8 * c8, 8, Float32))
            for g in cutlass.range_constexpr(4):
                mg = Float32(0.0)
                for e in cutlass.range_constexpr(32):
                    mg = cute.arch.fmax(mg, cute.math.absf(v[32 * g + e]))
                sG[r, g] = mg
            rot5.teardown(st5, tidx)
        elif const_expr(self.lane_transform and self.tc):
            # the tensor-core rotation (hadamard_rotation_method="tensor_core"): pass p, warp w rotates rows
            # 64 p + 16 w .. + 15 in the quad layout (quad_hadamard_rotate, A fragments read from the staged raw rows),
            # writes them to the fp32 tile and the rows' 32-column maxima to sG; the same pass / barrier order as below
            lane = tidx % 32
            gq = lane // 4
            tq = lane % 4
            for ps in cutlass.range_constexpr(2):
                a = cute.make_rmem_tensor(cute.make_layout(32), Int32)
                for i in cutlass.range_constexpr(2):
                    row = ps * 64 + 16 * warp + gq + 8 * i
                    for m in cutlass.range_constexpr(4):
                        cute.autovec_copy(
                            cute.make_tensor(cute.recast_ptr(_smem_run(sR, row, 32 * m + 8 * tq, 8, self.in_dtype).iterator,
                                                             dtype=Int32), cute.make_layout(4)),
                            cute.make_tensor(a.iterator + 16 * i + 4 * m, cute.make_layout(4)))
                cute.arch.sync_threads()
                if const_expr(self.rope):
                    quad_rope(a, tq, q0 + ps * 64 + 16 * warp + gq, q0 + ps * 64 + 16 * warp + gq + 8, mCos, mSin,
                               self.rope_transposed, self.in_dtype)
                v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
                quad_hadamard_rotate(a, v, lane, self.quad_signs, self.in_dtype is cutlass.Float16)
                for i in cutlass.range_constexpr(2):
                    row = ps * 64 + 16 * warp + gq + 8 * i
                    am = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                    for m in cutlass.range_constexpr(4):
                        cute.autovec_copy(cute.make_tensor(v.iterator + 32 * i + 8 * m, cute.make_layout(8)),
                                          _smem_run(sX, row, 32 * m + 8 * tq, 8, Float32))
                        mx = Float32(0.0)
                        for e in cutlass.range_constexpr(8):
                            mx = cute.arch.fmax(mx, cute.math.absf(v[32 * i + 8 * m + e]))
                        am[m] = mx
                    sG[row, tq] = _quad_scatter_max(am, tq)
            cute.arch.sync_threads()
        elif const_expr(self.lane_transform):
            # pass p transforms rows 64 p .. 64 p + 63 (a lane pair per row): all of its raw rows are read before any of
            # its fp32 rows is written (pass 0's writes end below the raw rows of pass 1)
            hp = tidx % 2
            for ps in cutlass.range_constexpr(2):
                row = ps * 64 + tidx // 2
                rb = cute.make_rmem_tensor(cute.make_layout(64), self.in_dtype)
                cute.autovec_copy(_smem_run(sR, row, hp * 64, 64, self.in_dtype), rb)
                cute.arch.sync_threads()
                v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
                for e in cutlass.range_constexpr(64):
                    v[e] = Float32(rb[e])
                v = _pair_transform(v, hp, q0 + row, mCos, mSin, self.rope, self.rope_transposed, self.signs,
                                    self.in_dtype)
                for c8 in cutlass.range_constexpr(8):
                    t8 = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                    for e in cutlass.range_constexpr(8):
                        t8[e] = v[c8 * 8 + e]
                    cute.autovec_copy(t8, _smem_run(sX, row, hp * 64 + c8 * 8, 8, Float32))
                for j in cutlass.range_constexpr(2):
                    m = Float32(0.0)
                    for e in cutlass.range_constexpr(32):
                        m = cute.arch.fmax(m, cute.math.absf(v[32 * j + e]))
                    sG[row, 2 * hp + j] = m
            cute.arch.sync_threads()

        # ---- pass 1: the row's four 32-column maxima (a subnormal counts as 0 under the flush rule)
        # (no flush needed here: a block whose max is subnormal gets byte 0 either way, and a subnormal never is the max
        # of a block with a normal element)
        amax = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for g in cutlass.range_constexpr(ngd):
            if const_expr(self.lane_transform):
                amax[g] = sG[r, g]
            elif const_expr(dt.width == 16):
                half = "bf16" if const_expr(dt is cutlass.BFloat16) else "f16"
                acc = Int32(0)
                for k in cutlass.range_constexpr(4):
                    wv = _load_staged_words(sX, r, 32 * g + 8 * k, 4, dt, hd)
                    for e in cutlass.range_constexpr(4):
                        acc = absmax_x2(acc, wv[e], half)
                amax[g] = cute.arch.fmax(half_lo_f32(acc, half), half_hi_f32(acc, half))
            else:
                m = Float32(0.0)
                for k in cutlass.range_constexpr(4):
                    v = _load_staged(sX, r, 32 * g + 8 * k, 8, dt, hd)
                    for e in cutlass.range_constexpr(8):
                        m = cute.arch.fmax(m, cute.math.absf(v[e]))
                amax[g] = m
        cols_eff = min(self.cols, hd)    # block_columns >= d: one exponent per token block
        if const_expr(cols_eff == 64 and hd == 128):
            amax[0] = cute.arch.fmax(amax[0], amax[1])
            amax[1] = amax[0]
            amax[2] = cute.arch.fmax(amax[2], amax[3])
            amax[3] = amax[2]
        if const_expr(cols_eff == hd and hd > 32):
            m = amax[0]
            for g in cutlass.range_constexpr(1, ngd):
                m = cute.arch.fmax(m, amax[g])
            for g in cutlass.range_constexpr(ngd):
                amax[g] = m
        for g in cutlass.range_constexpr(ngd, 4):   # d = 64: the SF atom's K-groups 2, 3 repeat 0, 1
            amax[g] = amax[g % ngd]
        tok_amax = amax[0]   # token_local: this token's own amax (block_columns = d: every group holds it)
        if const_expr(self.rows > 1):
            for i in cutlass.range_constexpr(int(math.log2(min(self.rows, 32)))):
                for g in cutlass.range_constexpr(4):
                    amax[g] = cute.arch.fmax(amax[g], cute.arch.shuffle_sync_bfly(amax[g], offset=1 << i))
        if const_expr(self.rows > 32):
            s_amax = smem.allocate_tensor(Float32, cute.make_layout((4, 4), stride=(4, 1)), 16)
            if lane == 0:
                for g in cutlass.range_constexpr(4):
                    s_amax[warp, g] = amax[g]
            cute.arch.sync_threads()
            nb = self.rows // 32
            w0 = (warp // nb) * nb
            for g in cutlass.range_constexpr(4):
                m = s_amax[w0, g]
                for k in cutlass.range_constexpr(1, nb):
                    m = cute.arch.fmax(m, s_amax[w0 + k, g])
                amax[g] = m
        sfb = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        sfr = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        inv = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for g in cutlass.range_constexpr(4):
            sfb[g], sfr[g], inv[g] = _ue8m0_select(amax[g], fmax, self.flush_subnormals)
        sfb_blk = sfb      # the token block's bytes: sf_dim, the exponent table, the factors
        if const_expr(self.token_local):
            # codes, sf_tok and D from this token's own exponent: no token's codes depend on another token (token-local operands)
            sfb_blk = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for g in cutlass.range_constexpr(4):
                sfb_blk[g] = sfb[g]
            tb_, tr_, ti_ = _ue8m0_select(tok_amax, fmax, self.flush_subnormals)
            for g in cutlass.range_constexpr(4):
                sfb[g] = tb_
                sfr[g] = tr_
                inv[g] = ti_
            # the per-token factor 2^(e_t - 127); 1 for an all-zero token (its codes are 0 under any scale, and a fold
            # by 0 would drop it from a row sum)
            tf_ = tr_ if tr_ > Float32(0.0) else Float32(1.0)
            if const_expr(self.compact):
                # (cov3, compact maps: the token table per PACKED row, real rows only; tile 0 writes the 128-row tail
                # past T as zero tokens' factors -- the packed kernels read a document's last tile past T)
                if r < nvr:
                    mTok[bb, hh, pb + r] = tf_
                    if const_expr(self.token_log2):
                        mTokL[bb, hh, pb + r] = Float32((f32_bits(tf_) >> 23) - 127)
                    if const_expr(self.token_inverse):
                        mTokI[bb, hh, pb + r] = bits_f32(Int32(254 << 23) - f32_bits(tf_))
                if t == 0:
                    mTok[bb, hh, mQ8.shape[1] + r] = Float32(1.0)
                    if const_expr(self.token_log2):
                        mTokL[bb, hh, mQ8.shape[1] + r] = Float32(0.0)
                    if const_expr(self.token_inverse):
                        mTokI[bb, hh, mQ8.shape[1] + r] = Float32(1.0)
            else:
                mTok[bb, hh, q0 + r] = tf_
                if const_expr(self.token_log2):
                    mTokL[bb, hh, q0 + r] = Float32((f32_bits(tf_) >> 23) - 127)
                if const_expr(self.token_inverse):
                    mTokI[bb, hh, q0 + r] = bits_f32(Int32(254 << 23) - f32_bits(tf_))

        # ---- pass 2: codes (and D's first-level partials) 8 columns at a time, written back into the own row
        eb = dt.width // 8
        row_words = rsx * eb // 4
        sW = cute.make_tensor(cute.recast_ptr(sX.iterator, dtype=Int32),
                              cute.make_layout((128, row_words), stride=(row_words, 1)))
        dpart = cute.make_rmem_tensor(cute.make_layout(hd // 2), Float32)
        if const_expr(self.nvfp4):
            # option A: the NVFP4 1x16 copy of x / f (token_block), x (raw) or x / s (per_tensor) from the same reads
            w4 = cute.make_rmem_tensor(cute.make_layout(hd // 8), Int32)
            sf4 = cute.make_rmem_tensor(cute.make_layout(2), Int32)
            sf4[0] = Int32(0)
            sf4[1] = Int32(0)
            sfv4 = cute.make_rmem_tensor(cute.make_layout(hd // 16), Float32)
            sinv = Float32(1.0)
            if const_expr(self.nvfp4_scaling == "per_tensor"):
                sinv = div_full_f32(Float32(1.0), mNvS[0])
            idx_row = ((Int64(bb) * Int64(nheads) + Int64(hh)) * Int64(s) + Int64(q0)) * Int64(4096) + Int64(r) * Int64(8)
        n2a = cute.make_rmem_tensor(cute.make_layout(8), Float32)   # output_row_norm2_max: ||x||^2, 8 partial chains
        for e in cutlass.range_constexpr(8):
            n2a[e] = Float32(0.0)
        for j in cutlass.range_constexpr(hd // 8):
            g = j // 4
            v = _load_staged(sX, r, 8 * j, 8, dt, hd)
            if const_expr(self.stat_norm):
                for e in cutlass.range_constexpr(8):
                    n2a[e] = n2a[e] + v[e] * v[e]
            y = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            for e in cutlass.range_constexpr(0, 8, 2):
                # flush rule: mul.ftz treats a subnormal x as a signed zero (and a subnormal product rounds to a zero code
                # either way), so it IS the flush, at no cost
                y[e], y[e + 1] = cute.arch.mul_packed_f32x2((v[e], v[e + 1]), (inv[g], inv[g]),
                                                            ftz=True if const_expr(self.flush_subnormals) else None)
                y[e] = _clamp(y[e], fmax)
                y[e + 1] = _clamp(y[e + 1], fmax)
            w0 = _pack4(self.fmt, y[0], y[1], y[2], y[3])
            w1 = _pack4(self.fmt, y[4], y[5], y[6], y[7])
            if const_expr(self.nvfp4):
                if const_expr(j % 2 == 0):
                    v16 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
                for e in cutlass.range_constexpr(8):
                    xv = v[e]
                    if const_expr(self.flush_subnormals):
                        if cute.math.absf(xv) < Float32(FLT_MIN):
                            xv = mul_rn(xv, Float32(0.0))
                    if const_expr(self.nvfp4_scaling == "token_block"):
                        xv = xv * inv[g]
                    elif const_expr(self.nvfp4_scaling == "per_tensor"):
                        xv = xv * sinv
                    v16[8 * (j % 2) + e] = xv
                if const_expr(j % 2 == 1):
                    g16 = j // 2
                    byte, sfr4, c0, c1 = nvfp4_group16(v16, self.nvfp4_stochastic, idx_row + Int64(g16), seed,
                                                       self.nvfp4_encoding)
                    w4[2 * g16] = c0
                    w4[2 * g16 + 1] = c1
                    sf4[g16 // 4] = sf4[g16 // 4] | (byte << (8 * (g16 % 4)))
                    sfv4[g16] = sfr4
                    if const_expr(self.d_mode == "nvfp4"):
                        # D's first-level partials of these 16 columns from the NVFP4 values: under stochastic rounding
                        # the grid values the codes were packed from, else the decoded codes
                        for hlf in cutlass.range_constexpr(2):
                            ov = _load_staged(sO, r, 8 * (j - 1 + hlf), 8, self.out_dtype, hd)
                            dq = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                            for e in cutlass.range_constexpr(8):
                                if const_expr(self.nvfp4_stochastic):
                                    dq[e] = v16[8 * hlf + e] * sfr4
                                else:
                                    dq[e] = e2m1_value(c0 if const_expr(hlf == 0) else c1, e) * sfr4
                            for kk in cutlass.range_constexpr(4):
                                e = (0, 1, 4, 5)[kk]
                                c = 8 * (j - 1 + hlf) + e
                                dpart[(c // 4) * 2 + c % 2] = fma_rn(dq[e], ov[e], mul_rn(dq[e + 2], ov[e + 2]))
            if const_expr(self.d_mode == "fp8"):
                ov = _load_staged(sO, r, 8 * j, 8, self.out_dtype, hd)
                dq = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                c0v = _code_values4(self.fmt, w0)
                c1v = _code_values4(self.fmt, w1)
                for e in cutlass.range_constexpr(4):
                    dq[e] = c0v[e] * sfr[g]
                    dq[4 + e] = c1v[e] * sfr[g]
                for kk in cutlass.range_constexpr(4):   # columns with bit 1 clear (0, 1, 4, 5); partner + 2
                    e = (0, 1, 4, 5)[kk]
                    c = 8 * j + e
                    dpart[(c // 4) * 2 + c % 2] = fma_rn(dq[e], ov[e], mul_rn(dq[e + 2], ov[e + 2]))
            sW[r, 2 * j] = w0
            sW[r, 2 * j + 1] = w1
        if const_expr(self.nvfp4):
            if const_expr(self.packed_out):
                if prow >= 0:
                    cute.autovec_copy(w4, cute.make_tensor(mX4[bb, prow, hh, None].iterator, cute.make_layout(hd // 8)))
            else:
                cute.autovec_copy(w4, cute.make_tensor(mX4[bb, q0 + r, hh, None].iterator, cute.make_layout(hd // 8)))
            base4 = (bh * tiles + t) * (128 * (hd // 64)) + lane * 4 + warp     # one SF atom per 64 columns
            if const_expr(not self.compact):   # (sfd compact: no per-tile atom anywhere)
                for k in cutlass.range_constexpr(hd // 64):
                    mSF4[base4 + 128 * k] = sf4[k]
            if const_expr(self.dense_out):
                if prow >= 0:
                    for k in cutlass.range_constexpr(hd // 64):
                        mSF4R[bh, prow * (hd // 64) + k] = sf4[k]
        if const_expr(self.d_mode is not None):
            steps = _TREE[hd][1]
            for i in cutlass.range_constexpr(len(steps)):
                dpart[steps[i][0]] = add_rn(dpart[steps[i][0]], dpart[steps[i][1]])
            dsum = dpart[_TREE[hd][2]]
            if const_expr(self.d_mode == "nvfp4"):
                if const_expr(self.nvfp4_scaling == "token_block"):
                    dsum = dsum * cute.arch.fmax(cute.arch.fmax(sfr[0], sfr[1]), cute.arch.fmax(sfr[2], sfr[3]))
                elif const_expr(self.nvfp4_scaling == "per_tensor"):
                    dsum = dsum * mNvS[0]
            if const_expr(self.d_mul is not None):
                dsum = mul_rn(dsum, Float32(self.d_mul))
            if const_expr(self.packed_out and (self.d_mul is not None or self.compact)):
                # packed rows: D is per ALIGNED row (s = S), every row written. A padding row's folded D'' is +2^120,
                # not the aligned layout's -0: there the row's dP is 0 and dS = +0 x (b 0 + -0) = +0; here its dP is
                # the next document's, and dS = +0 x (b dP + 2^120) = +0 again (the backward's emulation-free form)
                if const_expr(self.dense_out):
                    if prow >= 0:
                        mDR[bh, prow] = dsum
                if const_expr(not self.compact):
                    if prow < 0:
                        dsum = Float32(2.0 ** 120)
                    mD[bh * s + q0 + r] = dsum
            else:
                mD[bh * s + q0 + r] = dsum
        if const_expr(self.channel_major):
            words = cute.make_rmem_tensor(cute.make_layout(hd // 4), Int32)
            for w in cutlass.range_constexpr(hd // 4):
                words[w] = sW[r, w]
            if const_expr(self.packed_out):
                vt0 = mVt[0, t]
                if vt0 >= 0:   # (a tile past the documents: nothing)
                    _transpose_store_codes_packed(words, mQ8T, bb, hh, vt0, mVt[1, t], r, hd)
            else:
                _transpose_store_codes(words, mQ8T, bb, hh, q0, r, hd)
        cute.arch.sync_threads()
        # ---- coalesced copy-out of the codes: consecutive threads store consecutive 16-byte chunks
        cpr = hd // 16                   # 16-byte chunks per row of codes
        for it in cutlass.range_constexpr(cpr):
            cid = it * 128 + tidx
            row = cid // cpr
            c4 = (cid % cpr) * 4
            ch = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            src = cute.make_tensor(cute.make_ptr(Int32, sW.iterator.toint() + (row * row_words + c4) * 4,
                                                 cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4))
            cute.autovec_copy(src, ch)
            if const_expr(self.packed_out):
                if row < nvr:
                    dst = cute.make_tensor(cute.make_ptr(Int32, (mQ8[bb, pb + row, hh, None].iterator + c4).toint(),
                                                         cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(4))
                    cute.autovec_copy(ch, dst)
            else:
                dst = cute.make_tensor(cute.make_ptr(Int32, (mQ8[bb, q0 + row, hh, None].iterator + c4).toint(),
                                                     cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(4))
                cute.autovec_copy(ch, dst)

        if const_expr(not self.compact):
            mSFTok[(bh * tiles + t) * 128 + lane * 4 + warp] = sfb[0] | (sfb[1] << 8) | (sfb[2] << 16) | (sfb[3] << 24)
        if const_expr(self.rows >= 32):
            s_e = smem.allocate_tensor(Int32, cute.make_layout((4, 4), stride=(4, 1)), 16)
            if const_expr(self.stat_amax or self.stat_norm):
                # output_amax / output_row_norm2_max: the warps' maxima through the SF exchange's barrier below
                s_st = smem.allocate_tensor(Float32, cute.make_layout(8), 16)
                am = cute.arch.fmax(cute.arch.fmax(amax[0], amax[1]), cute.arch.fmax(amax[2], amax[3]))  # block maxima
                n2 = (n2a[0] + n2a[1]) + (n2a[2] + n2a[3]) + ((n2a[4] + n2a[5]) + (n2a[6] + n2a[7]))
                for i in cutlass.range_constexpr(5):
                    n2 = cute.arch.fmax(n2, cute.arch.shuffle_sync_bfly(n2, offset=1 << i))
                if lane == 0:
                    s_st[warp] = am
                    s_st[4 + warp] = n2
            if lane == 0:
                for g in cutlass.range_constexpr(4):
                    s_e[warp, g] = sfb_blk[g]
            cute.arch.sync_threads()
            if const_expr(self.stat_amax or self.stat_norm):
                if tidx == 0:   # one store per CTA and statistic (device_scales._dq_prescale reduces the tiles)
                    cta = bh * tiles + t
                    if const_expr(self.stat_amax):
                        mStats[cta] = cute.arch.fmax(cute.arch.fmax(s_st[0], s_st[1]), cute.arch.fmax(s_st[2], s_st[3]))
                    if const_expr(self.stat_norm):
                        mStats[mX.shape[0] * nheads * tiles + cta] = cute.arch.fmax(
                            cute.arch.fmax(s_st[4], s_st[5]), cute.arch.fmax(s_st[6], s_st[7]))
            c = tidx
            gc = (c % hd) // 32              # d = 64: atom rows 64..127 repeat rows 0..63
            if const_expr(self.compact):
                pass
            elif const_expr(self.token_axis_unit):
                mSFDim[(bh * tiles + t) * 128 + (c % 32) * 4 + c // 32] = Int32(0x7F7F7F7F)
            else:
                mSFDim[(bh * tiles + t) * 128 + (c % 32) * 4 + c // 32] = (
                    s_e[0, gc] | (s_e[1, gc] << 8) | (s_e[2, gc] << 16) | (s_e[3, gc] << 24))
            if const_expr(self.exponent_table and not self.compact):
                if tidx < 4 * ngd:
                    mExp[(bh * (tiles * 4) + 4 * t + tidx // ngd) * ngd + tidx % ngd] = Uint8(s_e[tidx // ngd, tidx % ngd])
            if const_expr(self.factors and not self.compact):
                if tidx < 4:
                    ev = s_e[tidx, 0]
                    if ev == 0:  # an all-zero block: its tile's largest factor, 1 for an all-zero tile
                        ev = cutlass.max(cutlass.max(s_e[0, 0], s_e[1, 0]), cutlass.max(s_e[2, 0], s_e[3, 0]))
                    f = Float32(1.0)
                    if ev > 0:
                        f = ex2_ftz(Float32(ev - 127))
                    mFac[bb, hh, 4 * t + tidx] = f
            if const_expr(self.dense_out):
                # the row's block's effective exponent (the factor rule above; 0 = factor 1), per real row
                evw = s_e[warp, 0]
                if evw == 0:
                    evw = cutlass.max(cutlass.max(s_e[0, 0], s_e[1, 0]), cutlass.max(s_e[2, 0], s_e[3, 0]))
                if const_expr(self.token_local and self.token_axis_unit):
                    evw = Int32(127)   # (cov3: token-local, unit token-axis scale -- the aligned d-rows atoms' 0x7F)
                if r < nvr:
                    mEvR[bh, pb + r] = Uint8(evw)
                if t == 0:
                    mEvR[bh, mQ8.shape[1] + r] = Uint8(0)
        if const_expr(self.lse and self.packed_out):
            # packed rows: the input LSE is the packed rows', the shifted LSE per ALIGNED row (s = S); a padding row
            # gets +inf (P = exp2(S - inf) = +0 in the backward, whatever its S), as the aligned layout's
            ll = Float32(math.inf)
            if prow >= 0:
                lv = mLse[bh * (mLse.shape[0] // (mX.shape[0] * nheads)) + prow]
                ll = Float32(0.0)
                if lv != Float32(-math.inf):
                    ll = mul_rn(lv, Float32(LOG2E))
                if const_expr(self.dense_out):
                    mLseR[bh, prow] = ll - lse_shift
            if const_expr(not self.compact):
                mLseOut[bh * s + q0 + r] = ll - lse_shift
        elif const_expr(self.lse):
            lv = mLse[bh * s + q0 + r]
            ll = Float32(0.0)
            if lv != Float32(-math.inf):
                ll = mul_rn(lv, Float32(LOG2E))
            if const_expr(mRowEnd is not None and self.row_end_lse):
                if q0 + r >= mRowEnd[bb, t]:
                    ll = Float32(math.inf)   # a padding row: P = exp2(S - inf) = 0 in the backward
            mLseOut[bh * s + q0 + r] = ll - lse_shift
        if const_expr(self.zero_acc and self.packed_out):
            # packed rows: the accumulator's rows are the packed rows (each tile's valid rows)
            sacc = mAcc.shape[0] // (mX.shape[0] * nheads * (self.hd // 2))
            z = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for c in cutlass.range_constexpr(4):
                z[c] = Int32(0)
            for it in cutlass.range_constexpr(self.hd // 8):     # 128 rows x d / 8 chunks of 16 bytes
                cid = it * 128 + tidx
                if cid // (self.hd // 8) < nvr:
                    cute.autovec_copy(z, cute.make_tensor(
                        cute.make_ptr(Int32, (mAcc.iterator + (bh * sacc + pb + cid // (self.hd // 8)) * (self.hd // 2)
                                              + (cid % (self.hd // 8)) * 4).toint(), cute.AddressSpace.gmem,
                                      assumed_align=16), cute.make_layout(4)))
        elif const_expr(self.zero_acc):
            # the backward's f16 dQ accumulator, flat (b, h, s * d): the tile's 128 rows are one contiguous run of
            # 128 d halves -- consecutive threads zero consecutive 16-byte chunks (coalesced)
            z = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for c in cutlass.range_constexpr(4):
                z[c] = Int32(0)
            for it in cutlass.range_constexpr(self.hd // 8):     # 128 rows x d / 2 words, 512 words per step
                cute.autovec_copy(z, cute.make_tensor(cute.make_ptr(Int32, (mAcc.iterator + (bh * s + q0) * (self.hd // 2)
                                                                            + (it * 128 + tidx) * 4).toint(),
                                                                    cute.AddressSpace.gmem, assumed_align=16),
                                                       cute.make_layout(4)))

    @cute.kernel
    def kernel(self, mX, mO, mQ8, mQ8T, mSFTok, mSFDim, mExp, mFac, mD, mLse, mLseOut, mCos, mSin, mX4, mSF4, mNvS,
               mAcc, seed: Int64, lse_shift: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        s = mX.shape[1]
        tiles = s // 128
        nheads = mX.shape[2]
        bh = bb * nheads + hh
        q0 = t * 128
        r = tidx
        lane = r % 32
        warp = r // 32
        ng = self.ng
        fmax = self.fmax

        smem = cutlass.utils.SmemAllocator()
        rsx = _row_pad(self.in_dtype)
        sX = smem.allocate_tensor(self.in_dtype, cute.make_layout((128, rsx), stride=(rsx, 1)), 16)
        _stage_tile(mX, sX, bb, q0, hh, self.in_dtype)
        if const_expr(self.d_mode is not None):
            rso = _row_pad(self.out_dtype)
            sO = smem.allocate_tensor(self.out_dtype, cute.make_layout((128, rso), stride=(rso, 1)), 16)
            _stage_tile(mO, sO, bb, q0, hh, self.out_dtype)
        cp_async_wait_all()
        cute.arch.sync_threads()
        rsf = 132
        tc_rows = (self.tc or self.tc5) and self.cols == 1   # the tensor-core rotation straight into the fp32 tile sT
        sT = sX
        if const_expr(self.cols == 1):
            sT = smem.allocate_tensor(Float32, cute.make_layout((128, rsf), stride=(rsf, 1)), 16)
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        if const_expr(tc_rows and self.tc5):
            # hadamard_rotation_method="tcgen05": thread r's staged row (RoPE'd and rounded as above) rotated on
            # tcgen05, flushed as the tensor_core branch, into its sT row
            rot5 = RowRotation(self.in_dtype, self.signs)
            st5 = rot5.allocate(smem)
            rot5.setup(st5, tidx)
            words = cute.make_rmem_tensor(cute.make_layout(64), Int32)
            cute.autovec_copy(cute.make_tensor(cute.recast_ptr(_smem_run(sX, r, 0, 128, self.in_dtype).iterator,
                                                               dtype=Int32), cute.make_layout(64)), words)
            if const_expr(self.rope):
                _rope_packed_row(words, mCos, mSin, q0 + r, self.rope_transposed, self.in_dtype)
            rot5.write_row(st5, r, words)
            rot5.rotate(st5, r, x, Int32(0))
            rot5.teardown(st5, tidx)
            if const_expr(self.flush_subnormals):
                for c in cutlass.range_constexpr(128):
                    if cute.math.absf(x[c]) < Float32(FLT_MIN):
                        x[c] = mul_rn(x[c], Float32(0.0))
            cute.autovec_copy(x, cute.make_tensor(sT[r, None].iterator, cute.make_layout(128)))
        elif const_expr(tc_rows):
            # warp w rotates its rows 32 w .. 32 w + 31 as two 16-row quad groups (quad_hadamard_rotate, A fragments from
            # the staged rows; RoPE in the lane), flushes as below and writes them to sT; thread r reads its row back
            gq = lane // 4
            tq = lane % 4
            for k in cutlass.range_constexpr(2):
                a = cute.make_rmem_tensor(cute.make_layout(32), Int32)
                for i in cutlass.range_constexpr(2):
                    row = 32 * warp + 16 * k + gq + 8 * i
                    for m in cutlass.range_constexpr(4):
                        cute.autovec_copy(
                            cute.make_tensor(cute.recast_ptr(_smem_run(sX, row, 32 * m + 8 * tq, 8, self.in_dtype).iterator,
                                                             dtype=Int32), cute.make_layout(4)),
                            cute.make_tensor(a.iterator + 16 * i + 4 * m, cute.make_layout(4)))
                if const_expr(self.rope):
                    quad_rope(a, tq, q0 + 32 * warp + 16 * k + gq, q0 + 32 * warp + 16 * k + gq + 8, mCos, mSin,
                               self.rope_transposed, self.in_dtype)
                v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
                quad_hadamard_rotate(a, v, lane, self.quad_signs, self.in_dtype is cutlass.Float16)
                if const_expr(self.flush_subnormals):
                    for c in cutlass.range_constexpr(64):
                        if cute.math.absf(v[c]) < Float32(FLT_MIN):
                            v[c] = mul_rn(v[c], Float32(0.0))
                for i in cutlass.range_constexpr(2):
                    row = 32 * warp + 16 * k + gq + 8 * i
                    for m in cutlass.range_constexpr(4):
                        cute.autovec_copy(cute.make_tensor(v.iterator + 32 * i + 8 * m, cute.make_layout(8)),
                                          cute.make_tensor(cute.make_ptr(Float32, (sT[row, None].iterator + 32 * m
                                                                                   + 8 * tq).toint(),
                                                                         cute.AddressSpace.smem, assumed_align=16),
                                                           cute.make_layout(8)))
            cute.arch.sync_warp()
            cute.autovec_copy(cute.make_tensor(sT[r, None].iterator, cute.make_layout(128)), x)
        else:
            x = _load_staged_row(sX, r, self.in_dtype)
            if const_expr(self.rope):
                apply_rope_row128(x, mCos, mSin, q0 + r, self.rope_transposed, self.in_dtype)
            if const_expr(self.signs is not None):
                hadamard_rotate_row128(x, self.signs)
            if const_expr(self.flush_subnormals):
                # subnormal inputs count as zeros of the same sign (FlashInfer's FTZ)
                for c in cutlass.range_constexpr(128):
                    if cute.math.absf(x[c]) < Float32(FLT_MIN):
                        x[c] = mul_rn(x[c], Float32(0.0))

        # ---- amax of every scale group this row takes part in
        if const_expr(self.cols == 1):
            # blocks along the sequence (block_rows tokens x 1 column): the transformed rows go through shared memory
            # once, thread c reduces column c over each block, and the per-(token group, column) scales come back as
            # broadcast shared-memory reads
            if const_expr(not tc_rows):
                cute.autovec_copy(x, cute.make_tensor(sT[r, None].iterator, cute.make_layout(128)))
            s_inv = smem.allocate_tensor(Float32, cute.make_layout((4, 128), stride=(128, 1)), 16)
            s_sfr = smem.allocate_tensor(Float32, cute.make_layout((4, 128), stride=(128, 1)), 16)
            s_e = smem.allocate_tensor(Int32, cute.make_layout((4, 128), stride=(128, 1)), 16)
            cute.arch.sync_threads()
            colmax = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for kk in cutlass.range_constexpr(4):
                m = Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    m = cute.arch.fmax(m, cute.math.absf(sT[32 * kk + i, tidx]))
                colmax[kk] = m
            nb = self.rows // 32
            for kk in cutlass.range_constexpr(4):
                m = colmax[(kk // nb) * nb]
                for k in cutlass.range_constexpr(1, nb):
                    m = cute.arch.fmax(m, colmax[(kk // nb) * nb + k])
                e_b, e_s, e_i = _ue8m0_select(m, fmax, self.flush_subnormals)
                s_inv[kk, tidx] = e_i
                s_sfr[kk, tidx] = e_s
                s_e[kk, tidx] = e_b
            cute.arch.sync_threads()
            words = cute.make_rmem_tensor(cute.make_layout(32), Int32)
            for w in cutlass.range_constexpr(32):
                y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                for k in cutlass.range_constexpr(4):
                    y[k] = _clamp(x[4 * w + k] * s_inv[warp, 4 * w + k], fmax)
                words[w] = _pack4(self.fmt, y[0], y[1], y[2], y[3])
            cute.autovec_copy(words, cute.make_tensor(mQ8[bb, q0 + r, hh, None].iterator, cute.make_layout(32)))
        else:
            amax = cute.make_rmem_tensor(cute.make_layout(ng), Float32)
            for g in cutlass.range_constexpr(4):
                m = Float32(0.0)
                for j in cutlass.range_constexpr(32):
                    m = cute.arch.fmax(m, cute.math.absf(x[32 * g + j]))
                amax[g] = m
            if const_expr(self.cols == 64):
                amax[0] = cute.arch.fmax(amax[0], amax[1])
                amax[1] = amax[0]
                amax[2] = cute.arch.fmax(amax[2], amax[3])
                amax[3] = amax[2]
            if const_expr(self.cols == 128):
                m = cute.arch.fmax(cute.arch.fmax(amax[0], amax[1]), cute.arch.fmax(amax[2], amax[3]))
                for g in cutlass.range_constexpr(4):
                    amax[g] = m
            if const_expr(self.rows > 1):
                for i in cutlass.range_constexpr(int(math.log2(min(self.rows, 32)))):
                    for g in cutlass.range_constexpr(ng):
                        amax[g] = cute.arch.fmax(amax[g], cute.arch.shuffle_sync_bfly(amax[g], offset=1 << i))
            if const_expr(self.rows > 32):
                s_amax = smem.allocate_tensor(Float32, cute.make_layout((4, ng), stride=(ng, 1)), 16)
                if lane == 0:
                    for g in cutlass.range_constexpr(ng):
                        s_amax[warp, g] = amax[g]
                cute.arch.sync_threads()
                nb = self.rows // 32
                w0 = (warp // nb) * nb
                for g in cutlass.range_constexpr(ng):
                    m = s_amax[w0, g]
                    for k in cutlass.range_constexpr(1, nb):
                        m = cute.arch.fmax(m, s_amax[w0 + k, g])
                    amax[g] = m

            # ---- UE8M0 scales and the codes
            sfb = cute.make_rmem_tensor(cute.make_layout(ng), Int32)
            sfr = cute.make_rmem_tensor(cute.make_layout(ng), Float32)
            inv = cute.make_rmem_tensor(cute.make_layout(ng), Float32)
            for g in cutlass.range_constexpr(ng):
                sfb[g], sfr[g], inv[g] = _ue8m0_select(amax[g], fmax, self.flush_subnormals)
            words = cute.make_rmem_tensor(cute.make_layout(32), Int32)
            for w in cutlass.range_constexpr(32):
                y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                for k in cutlass.range_constexpr(4):
                    y[k] = _clamp(x[4 * w + k] * inv[(4 * w + k) // 32], fmax)
                words[w] = _pack4(self.fmt, y[0], y[1], y[2], y[3])
            cute.autovec_copy(words, cute.make_tensor(mQ8[bb, q0 + r, hh, None].iterator, cute.make_layout(32)))

        if const_expr(self.cols >= 32):
            # SF of the (rows = tokens, K = head dim) operand: the 128 x 4 atom, row r -> word (r % 32) * 4 + r // 32
            mSFTok[(bh * tiles + t) * 128 + lane * 4 + warp] = sfb[0] | (sfb[1] << 8) | (sfb[2] << 16) | (sfb[3] << 24)

        if const_expr(self.rows >= 32):
            if const_expr(self.cols > 1):
                # every lane of warp kk holds token group kk's exponents: share them through shared memory
                s_e = smem.allocate_tensor(Int32, cute.make_layout((4, ng), stride=(ng, 1)), 16)
                if lane == 0:
                    for g in cutlass.range_constexpr(ng):
                        s_e[warp, g] = sfb[g]
                cute.arch.sync_threads()
            # SF of the (rows = head dim, K = tokens) operand: row c = tidx, K-groups = this tile's 4 token groups
            c = tidx
            gc = c if const_expr(self.cols == 1) else c // 32
            mSFDim[(bh * tiles + t) * 128 + (c % 32) * 4 + c // 32] = (
                s_e[0, gc] | (s_e[1, gc] << 8) | (s_e[2, gc] << 16) | (s_e[3, gc] << 24))
            if const_expr(self.exponent_table):
                if tidx < 16:
                    mExp[(bh * (tiles * 4) + 4 * t + tidx // 4) * 4 + tidx % 4] = Uint8(s_e[tidx // 4, tidx % 4])
            if const_expr(self.factors):
                if tidx < 4:
                    ev = s_e[tidx, 0]
                    if ev == 0:  # an all-zero block: its tile's largest factor, 1 for an all-zero tile
                        ev = cutlass.max(cutlass.max(s_e[0, 0], s_e[1, 0]), cutlass.max(s_e[2, 0], s_e[3, 0]))
                    f = Float32(1.0)
                    if ev > 0:
                        f = ex2_ftz(Float32(ev - 127))
                    mFac[bb, hh, 4 * t + tidx] = f

        if const_expr(self.channel_major):
            _transpose_store_codes(words, mQ8T, bb, hh, q0, r)

        if const_expr(self.lse):
            lv = mLse[bh * s + q0 + r]
            ll = Float32(0.0)
            if lv != Float32(-math.inf):
                ll = mul_rn(lv, Float32(LOG2E))
            mLseOut[bh * s + q0 + r] = ll - lse_shift
        if const_expr(self.zero_acc):
            # the backward's f16 dQ accumulator, flat (b, h, s * d): the tile's 128 rows are one contiguous run of
            # 128 d halves -- consecutive threads zero consecutive 16-byte chunks (coalesced)
            z = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for c in cutlass.range_constexpr(4):
                z[c] = Int32(0)
            for it in cutlass.range_constexpr(self.hd // 8):     # 128 rows x d / 2 words, 512 words per step
                cute.autovec_copy(z, cute.make_tensor(cute.make_ptr(Int32, (mAcc.iterator + (bh * s + q0) * (self.hd // 2)
                                                                            + (it * 128 + tidx) * 4).toint(),
                                                                    cute.AddressSpace.gmem, assumed_align=16),
                                                       cute.make_layout(4)))

        if const_expr(self.nvfp4):
            # option A: the NVFP4 1x16 codes of x / f (token_block: f = this token block's factor, a power of two), of x
            # (raw) or of x / s (per_tensor: s a float32 power of two on the device), from the same registers
            mult = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            for g in cutlass.range_constexpr(8):
                if const_expr(self.nvfp4_scaling == "token_block"):
                    mult[g] = inv[g // 2]
                elif const_expr(self.nvfp4_scaling == "raw"):
                    mult[g] = Float32(1.0)
                else:
                    mult[g] = div_full_f32(Float32(1.0), mNvS[0])
            w4 = cute.make_rmem_tensor(cute.make_layout(16), Int32)
            idx_row = ((Int64(bb) * Int64(nheads) + Int64(hh)) * Int64(s) + Int64(q0)) * Int64(4096) + Int64(r) * Int64(8)
            sf40, sf41, sfv4 = nvfp4_encode_row128(x, mult, w4, self.nvfp4_stochastic, idx_row, seed,
                                                   self.nvfp4_encoding)
            cute.autovec_copy(w4, cute.make_tensor(mX4[bb, q0 + r, hh, None].iterator, cute.make_layout(16)))
            base4 = (bh * tiles + t) * 256 + lane * 4 + warp
            mSF4[base4] = sf40
            mSF4[base4 + 128] = sf41
            if const_expr(self.d_mode == "nvfp4"):
                of = _load_staged_row(sO, r, self.out_dtype)
                deq = cute.make_rmem_tensor(cute.make_layout(128), Float32)
                for w in cutlass.range_constexpr(16):
                    for e in cutlass.range_constexpr(8):
                        deq[8 * w + e] = e2m1_value(w4[w], e) * sfv4[(8 * w + e) // 16]
                dsum = _old_triton_row_dot(deq, of)
                if const_expr(self.nvfp4_scaling == "token_block"):
                    m = cute.arch.fmax(cute.arch.fmax(sfr[0], sfr[1]), cute.arch.fmax(sfr[2], sfr[3]))
                    dsum = dsum * m
                elif const_expr(self.nvfp4_scaling == "per_tensor"):
                    dsum = dsum * mNvS[0]
                if const_expr(self.d_mul is not None):
                    dsum = mul_rn(dsum, Float32(self.d_mul))
                mD[bh * s + q0 + r] = dsum

        if const_expr(self.d_mode == "fp8"):
            # D = sum_d O * dequant(code): the softmax-consistent D of the dP GEMM's own operand
            of = _load_staged_row(sO, r, self.out_dtype)
            deq = cute.make_rmem_tensor(cute.make_layout(128), Float32)
            for c in cutlass.range_constexpr(128):
                if const_expr(self.cols == 1):
                    deq[c] = _code_value(self.fmt, words[c // 4], c % 4) * s_sfr[warp, c]
                else:
                    deq[c] = _code_value(self.fmt, words[c // 4], c % 4) * sfr[c // 32]
            dsum = _old_triton_row_dot(deq, of)
            if const_expr(self.d_mul is not None):
                dsum = mul_rn(dsum, Float32(self.d_mul))
            mD[bh * s + q0 + r] = dsum

    @cute.jit
    def _load_o(self, mO, bb, q, hh):
        ob = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
        cute.autovec_copy(cute.make_tensor(mO[bb, q, hh, None].iterator, cute.make_layout(128)), ob)
        of = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        of.store(ob.load().to(Float32))
        return of


_BLOCK_CACHE = {}


def _sf_view(raw: torch.Tensor) -> torch.Tensor:
    """(b, h, R/128, K/128, 32, 4, 4) contiguous -> the (32, 4, R/128, 4, K/128, h, b) view the SM100 kernels take."""
    return raw.permute(4, 5, 2, 6, 3, 1, 0)


_DUMMIES = {}


def _dummy(device, dtype=torch.int32, n=4):
    """A cached placeholder for an output a specialisation does not touch (no allocation or fill launch per call)."""
    key = (str(device), dtype, n)
    if key not in _DUMMIES:
        _DUMMIES[key] = torch.zeros(n, dtype=dtype, device=device)
    return _DUMMIES[key]


# kernel_pair_persistent's grid: CTAs per SM by variant (the register-limited residency: 80 registers x 256 threads
# without the NVFP4 copy, 106 with it). A module attribute so a benchmark can sweep it; 0 selects one CTA per tile.
PAIR_CTAS_PER_SM = {False: 0, True: 0}
# kernel_pair: the 32-token block's exponents behind a 64-thread named barrier per block (no CTA barrier)
PAIR_NAMED_BARRIERS = True
# kernel_pair: sector-complete global loads / FP8 code stores through a lane-pair register exchange
PAIR_COALESCED = True
# kernel_pair: residency target per SM (nvvm.minctasm, i.e. the register cap), without / with the NVFP4 copy; 0 = none
PAIR_MIN_CTAS = {False: 0, True: 3}  # A1 fp4 + S codes 39.1 -> 36.8 us; fp8 unchanged at 3, spills at >= 4 (t_qmin)
# kernel_tc (hadamard_rotation_method="tensor_core"): residency target per SM without / with the NVFP4 copy; 0 = none
# (3 = <= 80 registers: 81 already rounds to 88 per thread = 2 CTAs / SM, 23.3 -> 27.4 us at A1; 4 spills: 40 / 57 us)
TC_MIN_CTAS = {False: 3, True: 3}
_SM_COUNT = {}


def _pair_ctas(device, nvfp4: bool) -> int:
    per = PAIR_CTAS_PER_SM[bool(nvfp4)]
    if per <= 0:
        return 0
    key = str(device)
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return per * _SM_COUNT[key]


def _docmaps_gather(x):
    """The lp_docmaps registry's (packed source, row map) of a registered carrier x, else None."""
    from flash_attn.cute import lp_docmaps
    return lp_docmaps.lazy_source(x) if lp_docmaps._LAZY else None


def _docmaps_packed(x):
    """The lp_docmaps registry's (row map, S) of a registered packed operand x, else None."""
    from flash_attn.cute import lp_docmaps
    return lp_docmaps.packed_rows_of(x) if lp_docmaps._PACKED else None


def quantize_fp8_blocks(
    x: torch.Tensor,
    *,
    fp8_format: str = "e4m3",
    block_rows: int = 32,
    block_columns: int = 128,
    flush_subnormal_inputs: bool = True,
    scale_factor_layout: str = "sm100",
    make_channel_major_copy: bool = False,
    output_exponent_table: bool = True,
    output_token_block_factors=None,
    compute_d_with_output: Optional[torch.Tensor] = None,
    shifted_log2_lse_from: Optional[torch.Tensor] = None,
    log2_lse_shift: float = 0.0,
    apply_hadamard_rotation: bool = False,
    hadamard_sign_seed: int = DEFAULT_HADAMARD_SIGN_SEED,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    also_make_nvfp4_copy: bool = False,
    nvfp4_copy_scaling: str = "token_block",
    nvfp4_copy_rounding: str = "nearest",
    nvfp4_copy_seed: int = 0,
    nvfp4_copy_tensor_scale: Optional[torch.Tensor] = None,
    nvfp4_copy_encoding: str = "dp_operand",
    zero_accumulator: Optional[torch.Tensor] = None,
    d_multiplier: Optional[float] = None,
    output_amax: bool = False,
    output_row_norm2_max: bool = False,
    token_local_exponents: bool = False,
    output_token_factors=None,
    output_token_log2_factors=None,
    output_token_inverse_factors=None,
    token_axis_scale: str = "block",
    hadamard_rotation_method: Optional[str] = None,
    pad_head_dim_to: Optional[int] = None,
    hadamard_rotation_scope: str = "all",
    row_end: Optional[torch.Tensor] = None,
    lse_row_end: Optional[torch.Tensor] = None,
    gather_rows: Optional[tuple] = None,
    packed_rows: Optional[tuple] = None,
    packed_dense: bool = False,
) -> dict:
    """Block-scaled FP8 of x (b, s, h, 128) bf16 / fp16 / fp32 (last dim contiguous, s % 128 == 0), one read.

    Scale: one UE8M0 exponent per block of `block_rows` tokens x `block_columns` head-dim elements, the round-up selector
    e = 127 + ceil(log2(amax / FP8_MAX)) (0 for an all-zero block), so no element saturates. The MMA reads one exponent per
    32 elements along its K dimension, so coarser blocks are written replicated.
      block_rows     1, 2, 4, ..., 128 (tokens per exponent)
      block_columns  1 (one exponent per head-dim column and block_rows >= 32 tokens: blocks along the sequence),
                     32 (MXFP8), 64, 128 (a token block when block_rows = 32: the default path)
    Defaults: 32 x 128 token blocks with the exponent table -- the default path's operands.

    Options (all compile-time; off unless given):
      fp8_format                 "e4m3" | "e5m2"
      flush_subnormal_inputs     True (default, the library's rule = FlashInfer's): inputs with
                                 |x| < 2^-126 count as signed zeros and a subnormal amax / FP8_MAX gives SF byte 0 (codes
                                 +-0); False = the earlier mx2d_quant arithmetic (no flush of subnormal inputs)
      scale_factor_layout        "sm100": scale factors as the (32, 4, s/128, 4, 1, h, b) views the SM100 kernels take;
                                 "cudnn": flat uint8 in cuDNN's F8_128x4 order ((b, h, s/128) atoms along d,
                                 (d/128, b, h, s/128) atoms along the sequence; the same bytes at d = 128)
      make_channel_major_copy    also the codes as (b, h, d, s), tokens contiguous (the forward's K-major V)
      output_exponent_table      (b, h, s/32, 4) uint8 exponent table (block_rows >= 32, block_columns >= 32)
      output_token_block_factors True (allocate) or a (b, h, s/32) fp32 tensor written in place: 2^(e - 127) of column
                                 group 0 per 32-token group; an all-zero block takes the largest factor of its 128-token
                                 tile (1 for an all-zero tile). Its codes are 0, so every consumer that multiplies by the
                                 factor is unchanged; the fold forward's log2 f_V / 1 / f_V then keep the block's keys in
                                 the softmax row sum (a factor 0 dropped them: O too large by 1 / (1 - their probability))
                                 without lifting the row's max frame above its neighbours' (a factor 1 next to f_V ~ 2^-6
                                 cost the other keys' P-hat ~6 bits); block_rows >= 32
      compute_d_with_output=O    D (b, h, s) fp32 = sum_d O * dequant(codes) (or of the NVFP4 copy's values, times its
                                 factor, when also_make_nvfp4_copy), in the old kernels' summation order
      shifted_log2_lse_from=lse  (b, h, s) fp32: also lse_log2 = where(lse == -inf, 0, lse * log2 e) - log2_lse_shift
      apply_hadamard_rotation    quantise (x * signs) H / sqrt(128) (signs from hadamard_sign_seed)
      rope_cos / rope_sin        RoPE (GPT-NeoX halves) before quantising; fp32 (s, 64) or (64, s)
      also_make_nvfp4_copy       option A's fused dual output: also the NVFP4 1x16-along-d codes of x / f
                                 (nvfp4_copy_scaling "token_block": f = this block's factor; "raw": x; "per_tensor":
                                 x / nvfp4_copy_tensor_scale, a (1,) fp32 device power of two), nvfp4_copy_rounding
                                 "nearest" | "stochastic" (nvfp4_copy_seed); nvfp4_copy_encoding "dp_operand"
                                 (option A's dO copy: a zero-SF group gives zero codes) | "qk_operand" (the S operand's
                                 Q / K codes, FlashInfer's rule; nearest only) -- with apply_hadamard_rotation this is
                                 the one-read rotated Q / K (NVFP4 + token blocks), with "raw" the token-block route's S codes
                                 from the token-block launch; token blocks (32 x 128) only
      zero_accumulator=acc       a contiguous f16 tensor of b * h * s * 128 elements, zeroed in the same launch (the
                                 backward's dQ accumulator)
      d_multiplier=c             (with compute_d_with_output) D times the host float32 constant c in the same launch, one
                                 fp32 multiply after the sum: the backward drivers' D fold dpsum.mul_(c) (
                                 out["d_folded"] = c)
      output_amax                also out["amax"]: per 128-token tile max |x| of the values quantised (after the
                                 rotation / RoPE), an fp32 (b * h * s/128,) view; its maximum is the tensor's max |x|
      output_row_norm2_max       also out["row_norm2_max"]: per tile max over rows of ||x||^2, likewise (not with the
                                 rotation / RoPE). One store per CTA through the kernel's existing exchange, no
                                 atomics and no fill; the consumer reduces (the backward's f16 dQ range,
                                 device_scales._dq_prescale). block_rows >= 32, block_columns >= 32
      token_local_exponents      (token-local operands; 32 x d token blocks, no rotation / RoPE / NVFP4 copy) the codes, sf_tok and D use
                                 each TOKEN's own exponent (1 x d), so no token's codes depend on any other token; sf_dim,
                                 the exponent table and the token-block factors stay the 32-token block's (bitwise the
                                 default's). Writes the per-token factors 2^(e_t - 127) (1 for an all-zero token) to
                                 output_token_factors (a (b, h, s) fp32 view, e.g. a row of a factor table) or a new
                                 tensor out["token_factors"]; output_token_log2_factors / output_token_inverse_factors
                                 (views, same shape) also receive log2 f_t (an integer, fp32) and 1 / f_t (the fold
                                 forward's per-key bias and row-sum weight)
      token_axis_scale           "block" (default) | "unit": sf_dim holds the unit byte 127 instead of the block exponent
                                 (with token_local_exponents: the token-contraction GEMM folds the per-token factor into
                                 its other operand)
      hadamard_rotation_method   (with apply_hadamard_rotation) "tensor_core" (the rotation on mma.sync, equal to the
                                 butterfly up to the fp32 summation order) | "butterfly" (the register butterfly, bitwise
                                 the earlier kernels) | None: tensor_core where it measured faster -- every kernel but
                                 block_columns=1 (thread-per-row); an fp32 input runs the butterfly (logged:
                                 quant._rotations.resolve_rotation_method) | "tcgen05" (OPT-IN, never the default:
                                 the rotation on tcgen05 UMMA, quant._tcgen05 / quant._fp8_tcgen05 -- NOT bitwise the
                                 other methods: the fp32 rotated values are the tensor core's single 128-term
                                 accumulation, ~1e-7 .. 3e-6 of the codes differ from tensor_core; every other step is
                                 the same arithmetic). Every block option of the 16-bit-input kernels: the token-block
                                 kernel (persistent TMA / UMMA / TMEM, one CTA per SM) and the streaming /
                                 block_columns=1 kernels (the staged rows rotated in place). Refused (ValueError naming
                                 "tensor_core", no fallback): fp32 input (tensor_core rotates fp32 inputs exactly with the
                                 fp32 butterfly; a UMMA operand would round them), a token-block input TMA cannot tile.
                                 Speed: measured per launch (B200): 17-28 % faster than tensor_core on the default routes' Q / K calls from 512 tiles (b h s / 128) up, e.g. A1 b4 s4096 h16 8-bit 16.7 vs 23.1 us, 4-bit 26.4 vs 32.7 us; SLOWER on small launches -- about +0.8 us (+23 %) at 64 tiles (b1 s2048 h4) and below (a fixed cost: the UMMA operand images, TMEM, the extra TMA / UMMA / TMEM hops of a CTA's first tile)
      pad_head_dim_to            64 | 128 (the token-block fallback shapes): x's head dim d_x (a multiple of 16, below it) is quantised as if
                                 zero-padded to this width -- the input rows are read once and the missing columns staged as
                                 zeros in the same launch (no padded copy of x). Every output is the padded operand's
                                 (codes (b, s, h, pad), SFs, table, factors, statistics: zero columns move no exponent);
                                 compute_d_with_output takes the padded O (b, s, h, pad). Plain token / 32 x 32 blocks only
                                 (no rotation / RoPE / NVFP4 copy / token-local exponents)
      hadamard_rotation_scope    (with apply_hadamard_rotation) "all" (default: every output is of the rotated rows) |
                                 "nvfp4_copy": only the NVFP4 copy is of the rotated rows; the FP8 codes, the SF tables,
                                 the exponent table, the factors and amax are of x itself -- bitwise the unrotated
                                 launch's, and the NVFP4 copy bitwise the "all" launch's. The 4-bit routes' Q / K: the
                                 S operand rotated, the dQ / dK GEMMs' E4M3 operands not (no back-rotation). Needs
                                 also_make_nvfp4_copy with nvfp4_copy_scaling "raw" or "per_tensor", 16-bit input and
                                 the tensor-core rotation (the token-block kernel_tc; no D / LSE / channel-major copy)
      row_end                    (b, s / 128) int32 on the GPU, or None: the aligned varlen padding mask. Row q0 + r of the
                                 128-token tile t (q0 = 128 t) of batch bb is quantised as a zero row when
                                 q0 + r >= row_end[bb, t] -- its input (and, with compute_d_with_output, its O) is never
                                 read, so padding rows holding anything (NaN / inf included) give exactly the outputs of
                                 zero rows: zero codes, D = 0, and no effect on any block exponent, factor or statistic.
                                 Every token-block kernel (block_columns >= 32) but the opt-in tcgen05 rotation
      lse_row_end                the same table (or None), for shifted_log2_lse_from only: rows past it get lse_log2 = +inf
                                 (the backward's P = 0 there), the input rows are read as they are
      gather_rows=(xs, src)      (lp_docmaps; b = 1) x is a shape carrier: row r of x is row src[r] of xs (1, T, h, d) (a zero
                                 row where src[r] < 0), read by the streaming kernel through the row map (no gathered
                                 copy of x). Taken from the lp_docmaps registry when x is a registered carrier. Kernels
                                 without the row map (rotation / RoPE, padded head dim, block_columns < 32) gather x first
      packed_rows=(src, S[, vt_tile, vt_cols])
                                 (lp_docmaps packed rows; b = 1) x is the caller's PACKED (1, T, h, d) rows and the
                                 tiles are the S / 128 tiles of the aligned layout src (S,) maps to x (src[r] < 0: a
                                 padding row, quantised as a zero row): every per-row output (codes, codes_t, x4, the
                                 zeroed accumulator; lse / O inputs) is the packed (T) rows, written only for the rows
                                 src maps (a document's last tile never writes the next document's rows); a tile's
                                 real rows are a prefix (vt_tile row 1) at consecutive packed rows from src[q0]. D and
                                 lse_log2 are per ALIGNED row (b, h, S), padding rows +inf (lse_log2) and, with
                                 d_multiplier, D'' = +2^120 (the packed backward's dS = +0 there); the per-tile outputs
                                 (SFs, the table, the factors, the statistics) stay per aligned tile. codes_t (needs
                                 vt_tile / vt_cols, DocMaps.vt_tile and a static column count
                                 holding cu_vt[-1]) is (1, h, d, vt_cols): the documents' keys at multiples of 16
                                 columns (DocMaps.cu_vt), the gap columns zero codes. The streaming kernel only: plain
                                 16-bit rows, block_columns >= 32

    Returns a dict: codes (b, s, h, d) fp8; sf_tok / sf_tok_raw (block_columns >= 32: the (tokens, d) operand's scale
    factors, view / contiguous (b, h, s/128, 1, 32, 4, 4)); sf_dim / sf_dim_raw (block_rows >= 32: the (d, tokens)
    operand's); codes_t (b, h, d, s); e; fac; D; lse_log2; x4 (b, s, h, 64) uint8 / sf4 (the NVFP4 copy)."""
    assert x.dim() == 4 and x.stride(-1) == 1, (x.shape, x.stride())
    b, s, h, d = x.shape
    if packed_rows is None and gather_rows is None:
        packed_rows = _docmaps_packed(x)
    # sfd compact maps: packed_rows = (None, S, vt_tile, T, ntiles): no row map; dense per-row tables only
    compact = packed_rows is not None and packed_rows[0] is None
    if compact:
        packed_dense = True
    sr = s     # rows of the per-row outputs (packed_rows: the packed length T; s is then the aligned tiles' S)
    if packed_rows is not None:
        src_pk, s_pk = packed_rows[:2]
        vt_pk = packed_rows[2:]
        if not vt_pk:
            raise ValueError("quantize_fp8_blocks(packed_rows=): needs (vt_tile, vt_cols) (DocMaps.vt_tile[_q])")
        _rot_ok = (apply_hadamard_rotation and d == 128 and x.dtype != torch.float32 and not make_channel_major_copy
                   and compute_d_with_output is None and shifted_log2_lse_from is None and zero_accumulator is None
                   and (hadamard_rotation_method in (None, "tensor_core")   # (kernel_tc, the tensor-core rotation)
                        or (compact and hadamard_rotation_method == "tcgen05")))   # (cov3: the tcgen05 kernel, compact)
        if (b != 1 or gather_rows is not None or (apply_hadamard_rotation and not _rot_ok) or rope_cos is not None
                or (token_local_exponents and not compact)
                or (pad_head_dim_to is not None and pad_head_dim_to != d and not compact) or x.dtype == torch.float32
                or block_columns < 32 or block_rows != 32 or row_end is not None or lse_row_end is not None
                or not x.is_contiguous()):
            raise ValueError("quantize_fp8_blocks(packed_rows=): 16-bit packed rows (b = 1, contiguous), 32-row "
                             "token blocks, block_columns >= 32, no RoPE / token-local / padded head dim / row_end, a "
                             "rotation only on the tensor-core token-block kernel")
        if compact:
            assert make_channel_major_copy is False, "compact maps: no channel-major copy (the forward reads V's rows)"
            nt_pk = packed_rows[4]
            assert nt_pk.dtype == torch.int32 and nt_pk.numel() == 1 and vt_pk[0].shape == (2, int(s_pk) // 128)
        else:
            assert src_pk.dtype == torch.int32 and src_pk.shape == (int(s_pk),) and src_pk.is_cuda, (src_pk.shape,)
        sr, s = s, int(s_pk)
    if gather_rows is None and packed_rows is None and _docmaps_gather is not None:
        gather_rows = _docmaps_gather(x)
    if gather_rows is not None:
        # the row map rides the streaming kernel's staging; elsewhere the carrier is filled first (one gather)
        # (the rotation: the tensor-core token-block kernel, kernel_tc, takes the map; tcgen05 / butterfly do not)
        rot_tc = (apply_hadamard_rotation and rope_cos is None and block_rows == 32 and not make_channel_major_copy
                  and compute_d_with_output is None and shifted_log2_lse_from is None and zero_accumulator is None
                  and d == 128 and hadamard_rotation_method in (None, "tensor_core"))
        if (block_columns < 32 or (apply_hadamard_rotation and not rot_tc) or rope_cos is not None
                or token_local_exponents or (pad_head_dim_to is not None and pad_head_dim_to != d)
                or x.dtype == torch.float32 or row_end is not None):
            from flash_attn.cute.lp_docmaps import materialise
            materialise(x, gather_rows)
            gather_rows = None
    if lse_row_end is not None:
        if shifted_log2_lse_from is None or (row_end is not None and row_end is not lse_row_end):
            raise ValueError("quantize_fp8_blocks: lse_row_end needs shifted_log2_lse_from (and is row_end's table)")
    rend_t = row_end if row_end is not None else lse_row_end
    if rend_t is not None:
        assert rend_t.dtype == torch.int32 and rend_t.shape == (b, s // 128) and rend_t.is_cuda \
            and rend_t.stride(-1) == 1, (rend_t.dtype, rend_t.shape, rend_t.stride())
    if row_end is not None:
        assert row_end.dtype == torch.int32 and row_end.shape == (b, s // 128) and row_end.is_cuda \
            and row_end.stride(-1) == 1, (row_end.dtype, row_end.shape, row_end.stride())
        if block_columns == 1 or block_rows < 32:
            raise ValueError("quantize_fp8_blocks: row_end takes the token-block kernels (block_rows >= 32, "
                             "block_columns >= 32)")
    x_cols = 0                  # 0: the input rows are the staged width
    if pad_head_dim_to is not None and pad_head_dim_to != d:
        if pad_head_dim_to not in (64, 128) or d % 16 != 0 or not 0 < d < pad_head_dim_to:
            raise ValueError(f"quantize_fp8_blocks: pad_head_dim_to={pad_head_dim_to} needs 64 or 128 above a head_dim that is a "
                             f"multiple of 16; got head_dim {d}")
        if (apply_hadamard_rotation or rope_cos is not None or also_make_nvfp4_copy or token_local_exponents
                or block_columns < 32 or block_rows < 32):
            raise ValueError("quantize_fp8_blocks: pad_head_dim_to takes plain 32 x 32 / token blocks (no rotation, RoPE, "
                             "NVFP4 copy or token-local exponents)")
        x_cols, d = d, pad_head_dim_to
    if d not in (64, 128) or s % 128 != 0:
        raise ValueError(f"quantize_fp8_blocks: head_dim must be 64 or 128 and s % 128 == 0, got {tuple(x.shape)}")
    if d != 128 and (apply_hadamard_rotation or rope_cos is not None or block_columns == 1):
        raise ValueError("quantize_fp8_blocks: RoPE, the Hadamard rotation and block_columns=1 are built for head_dim 128")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"quantize_fp8_blocks: input dtype {x.dtype}")
    if fp8_format not in FP8_MAX:
        raise ValueError(f"fp8_format must be 'e4m3' or 'e5m2', got {fp8_format!r}")
    if block_rows not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError(f"block_rows must be a power of two <= 128, got {block_rows}")
    if block_columns not in (1, 32, 64, 128):
        raise ValueError(f"block_columns must be 1, 32, 64 or 128, got {block_columns}")
    if block_columns == 1 and block_rows < 32:
        raise ValueError("block_columns=1 (scales along the sequence) needs block_rows >= 32 (the MMA's 32-element group)")
    if scale_factor_layout not in ("sm100", "cudnn"):
        raise ValueError(f"scale_factor_layout must be 'sm100' or 'cudnn', got {scale_factor_layout!r}")
    want_fac = output_token_block_factors is not None and output_token_block_factors is not False
    if (output_exponent_table or want_fac) and (block_rows < 32 or block_columns == 1):
        output_exponent_table = output_exponent_table and block_rows >= 32 and block_columns >= 32
        if want_fac:
            raise ValueError("output_token_block_factors needs block_rows >= 32 and block_columns >= 32")
    if also_make_nvfp4_copy:
        if (block_rows, block_columns) != (32, 128):
            raise ValueError("also_make_nvfp4_copy (option A) is defined on token blocks: block_rows=32, block_columns=128 "
                             "(the whole head dim)")
        if nvfp4_copy_scaling not in ("token_block", "raw", "per_tensor"):
            raise ValueError(f"nvfp4_copy_scaling must be 'token_block', 'raw' or 'per_tensor', got {nvfp4_copy_scaling!r}")
        if nvfp4_copy_rounding not in ("nearest", "stochastic"):
            raise ValueError(f"nvfp4_copy_rounding must be 'nearest' or 'stochastic', got {nvfp4_copy_rounding!r}")
        if nvfp4_copy_encoding not in ("dp_operand", "qk_operand"):
            raise ValueError(f"nvfp4_copy_encoding must be 'dp_operand' or 'qk_operand', got {nvfp4_copy_encoding!r}")
        if nvfp4_copy_encoding == "qk_operand" and nvfp4_copy_rounding != "nearest":
            raise ValueError("nvfp4_copy_encoding='qk_operand' rounds to nearest")
        if nvfp4_copy_scaling == "per_tensor":
            assert nvfp4_copy_tensor_scale is not None and nvfp4_copy_tensor_scale.dtype == torch.float32 \
                and nvfp4_copy_tensor_scale.numel() == 1, "nvfp4_copy_scaling='per_tensor' needs a (1,) fp32 device scale"
    if token_axis_scale not in ("block", "unit"):
        raise ValueError(f"token_axis_scale must be 'block' or 'unit', got {token_axis_scale!r}")
    if token_local_exponents:
        if block_rows != 32 or block_columns < d or apply_hadamard_rotation or rope_cos is not None \
                or (also_make_nvfp4_copy and nvfp4_copy_scaling != "raw"):
            raise ValueError("token_local_exponents: 32 x d token blocks without rotation / RoPE, NVFP4 copy of the raw "
                             "rows only")
    elif output_token_factors is not None or output_token_log2_factors is not None \
            or output_token_inverse_factors is not None:
        raise ValueError("output_token_factors / _log2_factors / _inverse_factors need token_local_exponents=True")
    rope = rope_cos is not None
    if rope:
        assert rope_sin is not None and rope_cos.shape == rope_sin.shape, (rope_cos.shape, rope_sin.shape)
        assert rope_cos.shape in ((s, d // 2), (d // 2, s)), (rope_cos.shape, s, d)
        assert rope_cos.dtype == rope_sin.dtype == torch.float32 and rope_cos.is_contiguous() and rope_sin.is_contiguous()
    rope_t = rope and rope_cos.shape == (d // 2, s) and s != d // 2
    dev = x.device
    tiles = s // 128
    if (output_amax or output_row_norm2_max) and (block_columns == 1 or block_rows < 32):
        raise ValueError("output_amax / output_row_norm2_max need block_rows >= 32 and block_columns >= 32")
    codes = torch.empty(b, sr, h, d, dtype=TORCH_FP8[fp8_format], device=dev)
    q8_w = codes.view(torch.int32)
    out = dict(codes=codes)
    # (2, b * h * s/128) per-tile maxima: plane 0 max |x|, plane 1 max row ||x||^2 (only the requested planes written)
    pair = ((apply_hadamard_rotation or rope_cos is not None) and block_rows == 32 and block_columns >= 32
            and not make_channel_major_copy and compute_d_with_output is None and shifted_log2_lse_from is None
            and zero_accumulator is None)  # _BlockQuantKernel.pair
    if output_row_norm2_max and pair:
        raise ValueError("output_row_norm2_max is not built on the rotation / RoPE token-block kernel (output_amax is)")
    stats = torch.empty(2, b * h * tiles, dtype=torch.float32, device=dev) \
        if (output_amax or output_row_norm2_max) else _dummy(dev, torch.float32).view(1, 4)
    if make_channel_major_copy:
        codes_t = torch.empty(b, h, d, s if packed_rows is None else int(vt_pk[1]), dtype=TORCH_FP8[fp8_format],
                              device=dev)
        q8t_w = codes_t.view(torch.int32)
        out["codes_t"] = codes_t
    else:
        q8t_w = _dummy(dev).view(1, 1, 1, 4)
    # (compact maps: no per-tile atom / table is written -- one-tile placeholders keep the kernels' views shaped)
    tiles_a = 1 if compact else tiles
    if block_columns >= 32:     # one SF atom along d (d = 64: K-groups 2, 3 repeat 0, 1)
        sft = torch.empty(b, h, tiles_a, 1, 32, 4, 4, dtype=torch.uint8, device=dev)
        sft_w = sft.view(torch.int32).view(-1)
    else:
        sft, sft_w = None, _dummy(dev)
    if block_rows >= 32:        # one SF atom along d (d = 64: rows 64..127 repeat rows 0..63)
        sfd = torch.empty(b, h, 1, tiles_a, 32, 4, 4, dtype=torch.uint8, device=dev)
        sfd_w = sfd.view(torch.int32).view(-1)
    else:
        sfd, sfd_w = None, _dummy(dev)
    if output_exponent_table and compact:
        etab = _dummy(dev, torch.uint8, 16)
        out["e"] = None
    elif output_exponent_table:
        etab = torch.empty(b, h, s // 32, d // 32, dtype=torch.uint8, device=dev)
        out["e"] = etab
    else:
        etab = _dummy(dev, torch.uint8, 16)
    if want_fac and compact:   # (compact: the factors come from the dense exponent rows)
        want_fac = False
        fac = _dummy(dev, torch.float32).view(1, 1, 4)
        out["fac"] = None
    elif want_fac:
        if isinstance(output_token_block_factors, torch.Tensor):
            fac = output_token_block_factors
            assert fac.shape == (b, h, s // 32) and fac.dtype == torch.float32 and fac.stride(-1) == 1, (fac.shape, fac.stride())
        else:
            fac = torch.empty(b, h, s // 32, dtype=torch.float32, device=dev)
        out["fac"] = fac
    else:
        fac = _dummy(dev, torch.float32).view(1, 1, 4)
    d_mode = None
    if compute_d_with_output is not None:
        o = compute_d_with_output
        # (lp_docmaps gather: O may be the caller's packed rows, read through the same row map as x)
        assert (o.shape == (b, sr, h, d) or (gather_rows is not None and o.shape == gather_rows[0].shape)) \
            and o.stride(-1) == 1 and o.dtype in (torch.bfloat16, torch.float16, torch.float32)
        dsum = torch.empty(b, h, s if packed_rows is not None else sr, dtype=torch.float32, device=dev) \
            if not compact else _dummy(dev, torch.float32)
        out["D"] = dsum
        d_mode = "nvfp4" if also_make_nvfp4_copy else "fp8"
    else:
        o, dsum = x, _dummy(dev, torch.float32)
    if d_multiplier is not None:
        if compute_d_with_output is None:
            raise ValueError("d_multiplier scales D: it needs compute_d_with_output")
        d_multiplier = float(np.float32(d_multiplier))
        out["d_folded"] = d_multiplier
    if shifted_log2_lse_from is not None:
        lse = shifted_log2_lse_from
        assert lse.shape == (b, h, sr) and lse.dtype == torch.float32 and lse.is_contiguous(), (lse.shape, lse.dtype)
        lse_out = torch.empty(b, h, s if packed_rows is not None else sr, dtype=torch.float32, device=dev) \
            if not compact else _dummy(dev, torch.float32)
        out["lse_log2"] = lse_out
    else:
        lse, lse_out = _dummy(dev, torch.float32), _dummy(dev, torch.float32)
    if also_make_nvfp4_copy:
        x4 = torch.empty(b, sr, h, d // 2, dtype=torch.uint8, device=dev)
        sf4 = torch.empty(b, h, tiles_a, d // 64, 32, 4, 4, dtype=torch.uint8, device=dev)
        x4_w, sf4_w = x4.view(torch.int32), sf4.view(torch.int32).view(-1)
        out["x4"], out["sf4"] = x4, _sf_view(sf4)
        nvs = nvfp4_copy_tensor_scale.reshape(1) if nvfp4_copy_scaling == "per_tensor" else _dummy(dev, torch.float32)
    else:
        x4_w, sf4_w, nvs = _dummy(dev).view(1, 1, 1, 4), _dummy(dev), _dummy(dev, torch.float32)
    cos, sin = (rope_cos, rope_sin) if rope else (_dummy(dev, torch.float32).view(1, 4), _dummy(dev, torch.float32).view(1, 4))
    if zero_accumulator is not None:
        assert zero_accumulator.dtype == torch.float16 and zero_accumulator.is_contiguous() \
            and zero_accumulator.numel() == b * h * sr * d, (zero_accumulator.shape, zero_accumulator.dtype)
        acc_w = zero_accumulator.view(torch.int32).view(-1)
    else:
        acc_w = _dummy(dev)
    if token_local_exponents:
        if output_token_factors is not None:
            tokf = output_token_factors
            assert (tokf.shape == (b, h, s) or (compact and tokf.shape == (b, h, sr + 128))) \
                and tokf.dtype == torch.float32 and tokf.stride(-1) == 1, \
                (tokf.shape, tokf.stride())
        else:
            tokf = torch.empty(b, h, s, dtype=torch.float32, device=dev)
        out["token_factors"] = tokf
    else:
        tokf = _dummy(dev, torch.float32).view(1, 1, 4)
    tokl = output_token_log2_factors if output_token_log2_factors is not None else _dummy(dev, torch.float32).view(1, 1, 4)
    toki = output_token_inverse_factors if output_token_inverse_factors is not None else \
        _dummy(dev, torch.float32).view(1, 1, 4)
    for t_ in (output_token_log2_factors, output_token_inverse_factors):
        if t_ is not None:
            assert (t_.shape == (b, h, s) or (compact and t_.shape == (b, h, sr + 128))) and t_.dtype == torch.float32 \
                and t_.stride(-1) == 1, (t_.shape, t_.stride())
    signs = hadamard_signs_tuple(hadamard_sign_seed) if apply_hadamard_rotation else None
    key = (x.dtype, o.dtype if d_mode else None, fp8_format, block_rows, block_columns, make_channel_major_copy,
           bool(output_exponent_table), want_fac, d_mode, shifted_log2_lse_from is not None, signs, rope, rope_t,
           also_make_nvfp4_copy, nvfp4_copy_scaling, nvfp4_copy_rounding, bool(flush_subnormal_inputs),
           nvfp4_copy_encoding, zero_accumulator is not None, d, d_multiplier, bool(output_amax),
           bool(output_row_norm2_max), bool(token_local_exponents), token_axis_scale,
           output_token_log2_factors is not None, output_token_inverse_factors is not None) + \
        ((("x_cols", x_cols),) if x_cols else ())
    pair_ctas = _pair_ctas(dev, also_make_nvfp4_copy) if (signs is not None or rope) else 0
    pair_named = bool(PAIR_NAMED_BARRIERS) and (signs is not None or rope)
    # (16-bit rows only: _pair_load_coalesced moves 32 words = 64 16-bit elements, half an fp32 half row -- with fp32
    # input it left half of every lane's values unloaded: nondeterministic rotated codes, since fixed)
    pair_coal = bool(PAIR_COALESCED) and (signs is not None or rope) and x.dtype != torch.float32
    pair_min = int(PAIR_MIN_CTAS[bool(also_make_nvfp4_copy)]) if (signs is not None or rope) else 0
    tc = False
    tc5 = False     # hadamard_rotation_method="tcgen05" (opt-in, quant._fp8_tcgen05 / quant._tcgen05)
    if signs is not None:
        why = None
        if x.dtype == torch.float32:
            why = "fp32 input (the mma.sync operands are 16-bit: bf16 / f16 rows are exact in them, fp32 rows are not)"
        slower = None
        if block_columns == 1:
            slower = ("block_columns=1, the thread-per-row kernel: rotated rows through its fp32 tile measured slower "
                      "(A1 42.9 us butterfly vs 46.3 us tensor cores)")
        method = resolve_rotation_method(hadamard_rotation_method, "quantize_fp8_blocks", why, slower,
                                         tcgen05_unsupported=_tcgen05_unsupported_blocks(x, pair))
        tc = method == "tensor_core"
        tc5 = method == "tcgen05"
    elif hadamard_rotation_method is not None:
        resolve_rotation_method(hadamard_rotation_method, "quantize_fp8_blocks",
                                tcgen05_unsupported=None)   # validates the name (nothing is rotated)
    if gather_rows is not None and apply_hadamard_rotation and not (pair and tc and not tc5):
        from flash_attn.cute.lp_docmaps import materialise   # (only kernel_tc of the rotated kernels takes the map)
        materialise(x, gather_rows)
        gather_rows = None
    tc_min = int(TC_MIN_CTAS[bool(also_make_nvfp4_copy)]) if tc else 0
    if hadamard_rotation_scope not in ("all", "nvfp4_copy"):
        raise ValueError(f"hadamard_rotation_scope must be 'all' or 'nvfp4_copy', got {hadamard_rotation_scope!r}")
    nvfp4_only = hadamard_rotation_scope == "nvfp4_copy"
    if nvfp4_only and not (signs is not None and also_make_nvfp4_copy and nvfp4_copy_scaling in ("raw", "per_tensor")
                           and pair and (tc or tc5)):
        raise ValueError("hadamard_rotation_scope='nvfp4_copy' needs apply_hadamard_rotation, also_make_nvfp4_copy with "
                         "nvfp4_copy_scaling 'raw' or 'per_tensor', a 16-bit input and the tensor-core (tensor_core / "
                         "tcgen05) token-block kernel (block_rows 32, no D / LSE / channel-major copy / zeroing)")
    if tc5 and pair and row_end is not None:
        raise ValueError("quantize_fp8_blocks: row_end is not built on the tcgen05 rotation kernel; use "
                         "hadamard_rotation_method='tensor_core' (the default)")
    if tc5 and pair:
        pk5 = None
        if packed_rows is not None:   # (cov3: compact maps only -- the dense per-row exponent bytes / NVFP4 SF words)
            assert compact and block_columns == 128, "tcgen05 packed rows: compact maps, token blocks"
            R = dense_row_stride(sr)
            evr = torch.empty(b * h, R, dtype=torch.uint8, device=dev)
            out["ev_rows"] = evr
            sf4r = torch.empty(b * h, R * (d // 64), dtype=torch.int32, device=dev) if also_make_nvfp4_copy \
                else _dummy(dev).view(1, 4)
            if also_make_nvfp4_copy:
                out["sf4_rows"] = sf4r
            pk5 = (vt_pk[0], packed_rows[4], evr, sf4r)
        _launch_token_block_tcgen05(
            x, codes, sft_w, sfd_w, etab, fac, stats, out, signs=signs, fp8_format=fp8_format, block_columns=block_columns,
            exponent_table=bool(output_exponent_table), factors=want_fac, flush=bool(flush_subnormal_inputs), rope=rope,
            rope_t=rope_t, cos=cos, sin=sin, nvfp4=also_make_nvfp4_copy, nvfp4_scaling=nvfp4_copy_scaling,
            nvfp4_stochastic=nvfp4_copy_rounding == "stochastic", nvfp4_encoding=nvfp4_copy_encoding, nvs=nvs,
            sf4_w=sf4_w, x4=out.get("x4"), stat_amax=bool(output_amax), nvfp4_only=nvfp4_only,
            seed=int(nvfp4_copy_seed) & 0x3FFFFFFFFFFFFFFF, packed=pk5)
        return _finish_block_outputs(out, sft, sfd, stats, scale_factor_layout, output_amax, output_row_norm2_max)
    key = key + (pair_ctas, pair_named, pair_coal, pair_min, tc, tc_min, nvfp4_only, tc5) \
        + ((("row_end", row_end is not None, lse_row_end is not None),) if rend_t is not None else ())
    gx = ()
    o_g = False
    if packed_rows is not None:
        assert (not pair) or (tc and not tc5), "packed_rows: the streaming kernel or the tensor-core rotation kernel"
        o_g = compute_d_with_output is not None   # (O: the caller's packed rows, read through the same map)
        key = key + ("packed", o_g)
        gx = (src_pk, vt_pk[0])
        if packed_dense:
            assert (not pair) or tc, "packed_dense: the streaming kernel or the tensor-core rotation kernel"
            R = dense_row_stride(sr)
            evr = torch.empty(b * h, R, dtype=torch.uint8, device=dev)
            out["ev_rows"] = evr
            sf4r = torch.empty(b * h, R * (d // 64), dtype=torch.int32, device=dev) if also_make_nvfp4_copy \
                else _dummy(dev).view(1, 4)
            if also_make_nvfp4_copy:
                out["sf4_rows"] = sf4r
            dr = torch.empty(b * h, R, dtype=torch.float32, device=dev) if compute_d_with_output is not None \
                else _dummy(dev, torch.float32).view(1, 4)
            if compute_d_with_output is not None:
                out["D_rows"] = dr
            lr = torch.empty(b * h, R, dtype=torch.float32, device=dev) if shifted_log2_lse_from is not None \
                else _dummy(dev, torch.float32).view(1, 4)
            if shifted_log2_lse_from is not None:
                out["lse_rows"] = lr
            gx = gx + (evr, sf4r, dr, lr)
            key = key + ("dense",)
            if compact:   # (the dense rows ARE the statistics; the tile table and the device tile count)
                if compute_d_with_output is not None:
                    out["D"] = dr.view(b, h, R)
                if shifted_log2_lse_from is not None:
                    out["lse_log2"] = lr.view(b, h, R)
                gx = (None, vt_pk[0]) + gx[2:] + (packed_rows[4],)
                key = key + ("compact",)
    if gather_rows is not None:
        xs, src = gather_rows
        assert b == 1 and xs.dim() == 4 and xs.shape[0] == 1 and tuple(xs.shape[2:]) == (h, d) and xs.dtype == x.dtype \
            and xs.stride(-1) == 1 and src.dtype == torch.int32 and src.shape == (s,), (xs.shape, src.shape)
        x = xs
        o_g = compute_d_with_output is not None and o.shape[1] != s   # O in the caller's packed rows (lp_docmaps)
        assert not o_g or tuple(o.shape) == tuple(xs.shape), (o.shape, xs.shape)
        key = key + ("gather", o_g)
        gx = (src,)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    etab_f, dsum_f, lse_f, lse_out_f = etab.view(-1), dsum.view(-1), lse.view(-1), lse_out.view(-1)
    args = (x, o, q8_w, q8t_w, sft_w, sfd_w, etab_f, fac, dsum_f, lse_f, lse_out_f, cos, sin, x4_w, sf4_w, nvs, acc_w,
            stats.view(-1), tokf, tokl, toki, rend_t)
    if key not in _BLOCK_CACHE:
        kern = _BlockQuantKernel(_cute_dtype(x.dtype), _cute_dtype(o.dtype) if d_mode else None, fp8_format, block_rows,
                                 block_columns, make_channel_major_copy, bool(output_exponent_table), want_fac, d_mode,
                                 shifted_log2_lse_from is not None, signs, rope, rope_t, also_make_nvfp4_copy,
                                 nvfp4_copy_scaling, nvfp4_copy_rounding == "stochastic", bool(flush_subnormal_inputs),
                                 nvfp4_copy_encoding, zero_accumulator is not None, d, d_mul=d_multiplier,
                                 stat_amax=bool(output_amax), stat_norm=bool(output_row_norm2_max),
                                 token_local=bool(token_local_exponents), token_axis_unit=token_axis_scale == "unit",
                                 token_log2=output_token_log2_factors is not None,
                                 token_inverse=output_token_inverse_factors is not None, x_cols=x_cols)
        kern.pair_ctas = pair_ctas
        kern.pair_named = pair_named
        kern.pair_coalesced = pair_coal
        kern.pair_min_ctas = pair_min
        kern.tc = tc
        kern.tc_min_ctas = tc_min
        kern.tc5 = tc5
        if tc5:
            kern.smem_bytes += RowRotation.smem_bytes
        kern.rotate_nvfp4_copy_only = nvfp4_only
        kern.row_end_input = row_end is not None
        kern.o_gather = o_g
        kern.packed_out = packed_rows is not None
        kern.dense_out = bool(packed_rows is not None and packed_dense)
        kern.compact = bool(compact)
        kern.row_end_lse = lse_row_end is not None
        cargs = [to_cute_tensor(x, divisibility=8), to_cute_tensor(o, divisibility=8),
                 to_cute_tensor(q8_w, divisibility=4), to_cute_tensor(q8t_w, divisibility=1, assumed_align=4),
                 to_cute_tensor(sft_w, assumed_align=4), to_cute_tensor(sfd_w, assumed_align=4),
                 to_cute_tensor(etab_f, assumed_align=1), to_cute_tensor(fac, assumed_align=4),
                 to_cute_tensor(dsum_f, assumed_align=4), to_cute_tensor(lse_f, assumed_align=4),
                 to_cute_tensor(lse_out_f, assumed_align=4), to_cute_tensor(cos, assumed_align=4),
                 to_cute_tensor(sin, assumed_align=4), to_cute_tensor(x4_w, divisibility=4),
                 to_cute_tensor(sf4_w, assumed_align=4), to_cute_tensor(nvs, assumed_align=4),
                 to_cute_tensor(acc_w, assumed_align=16), to_cute_tensor(stats.view(-1), assumed_align=4),
                 to_cute_tensor(tokf, assumed_align=4), to_cute_tensor(tokl, assumed_align=4),
                 to_cute_tensor(toki, assumed_align=4),
                 None if rend_t is None else to_cute_tensor(rend_t, assumed_align=4)]
        _BLOCK_CACHE[key] = cute.compile(kern, *cargs, Int64(0), Float32(0.0), stream,
                                         *(None if t_ is None else to_cute_tensor(t_, assumed_align=4) for t_ in gx),
                                         options=_COMPILE_OPTIONS)
    _BLOCK_CACHE[key](*args, int(nvfp4_copy_seed) & 0x3FFFFFFFFFFFFFFF, float(log2_lse_shift), stream, *gx)
    return _finish_block_outputs(out, sft, sfd, stats, scale_factor_layout, output_amax, output_row_norm2_max)


def dense_row_stride(t: int) -> int:
    """(sfd) The per-head row stride R of the dense per-row tables (exponents, NVFP4 SF words, D'', shifted LSE) of
    T packed rows: a multiple of 16 (16-byte aligned u8 / f32 head rows) with at least 272 rows past T -- a 132-row
    f32 bulk copy from (row & ~3) and a 144-byte u8 one from (row & ~15) of any row < T + 128 stay inside the head
    (the forward's second Q tile of a work tile starts up to T + 127 when the last document is short, and
    sfd_atoms.load_sf4 / load_bytes read it unpredicated: with a 144-row tail that read ran up to ~110 rows past the
    table -- an intermittent illegal address on the last head) -- and rows [T, T + 128) of the exponent table are a
    zero tail."""
    return (int(t) + 15) // 16 * 16 + 272


def _finish_block_outputs(out, sft, sfd, stats, scale_factor_layout, output_amax, output_row_norm2_max):
    if sft is not None:
        out["sf_tok_raw"] = sft
        out["sf_tok"] = _sf_view(sft) if scale_factor_layout == "sm100" else sft.reshape(-1)
    if sfd is not None:
        out["sf_dim_raw"] = sfd
        out["sf_dim"] = _sf_view(sfd) if scale_factor_layout == "sm100" else sfd.reshape(-1)
    if output_amax:
        out["amax"] = stats[0]
    if output_row_norm2_max:
        out["row_norm2_max"] = stats[1]
    return out


def _tma_ready(x: torch.Tensor) -> bool:
    """A (b, s, h, 128) 16-bit tensor TMA can tile: 16-byte aligned base and strides, last dim contiguous."""
    es = x.element_size()
    return x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and all((st * es) % 16 == 0 for st in x.stride()[:-1])


def _tcgen05_unsupported_blocks(x: torch.Tensor, pair: bool):
    """Why quantize_fp8_blocks cannot rotate this call with hadamard_rotation_method="tcgen05" (None: it can)."""
    if x.dtype == torch.float32:
        return ("needs a bf16 / fp16 input: an fp32 row would be rounded to a 16-bit UMMA operand (the tensor_core "
                "method rotates fp32 inputs exactly, with the fp32 butterfly)")
    if pair and not _tma_ready(x):
        return "needs a TMA-tileable input (16-byte aligned base and strides, contiguous head dim)"
    return None


TCGEN05_PER_TENSOR = ("is not built for the per-tensor kernel (lane-pair rows of 64-token tiles over a persistent "
                      "multi-tensor grid, rotated once per pass)")
_TC5_CACHE = {}
_TC5_DUMMY4 = {}


def _launch_token_block_tcgen05(x, codes, sft_w, sfd_w, etab, fac, stats, out, *, signs, fp8_format, block_columns,
                                exponent_table, factors, flush, rope, rope_t, cos, sin, nvfp4, nvfp4_scaling,
                                nvfp4_stochastic, nvfp4_encoding, nvs, sf4_w, x4, stat_amax, nvfp4_only, seed,
                                packed=None):
    """The token-block kernel with the rotation on tcgen05 (quant._fp8_tcgen05): kernel_tc's outputs, one launch,
    persistent (min(#SM, tiles) CTAs)."""
    from flash_attn.cute.quant import _fp8_tcgen05 as K5
    K5._bind()
    b, s, h, d = x.shape
    dev = x.device
    tiles = s // 128
    if x4 is None:
        k4 = str(dev)
        if k4 not in _TC5_DUMMY4:
            _TC5_DUMMY4[k4] = torch.empty(1, 128, 1, 64, dtype=torch.uint8, device=dev)
        x4 = _TC5_DUMMY4[k4]
    key = (x.dtype, fp8_format, block_columns, exponent_table, factors, flush, signs, rope, rope_t, nvfp4,
           nvfp4_scaling, nvfp4_stochastic, nvfp4_encoding, stat_amax, nvfp4_only) + (("compact",) if packed else ())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    q8b = codes.view(torch.uint8)
    etab_f = etab.view(-1)
    st0 = stats[0] if stats.dim() == 2 and stats.shape[0] == 2 else stats.view(-1)
    args = (x, q8b, x4, sft_w, sfd_w, etab_f, fac, sf4_w, nvs, cos, sin, st0)
    if key not in _TC5_CACHE:
        kern = K5.TokenBlockTcgen05Kernel(_cute_dtype(x.dtype), fp8_format, FP8_MAX[fp8_format], block_columns,
                                          exponent_table, factors, flush, signs, rope, rope_t, nvfp4, nvfp4_scaling,
                                          nvfp4_stochastic, nvfp4_encoding, stat_amax, nvfp4_only)
        cargs = [to_cute_tensor(x, divisibility=8), to_cute_tensor(q8b, divisibility=16),
                 to_cute_tensor(x4, divisibility=16), to_cute_tensor(sft_w, assumed_align=4),
                 to_cute_tensor(sfd_w, assumed_align=4), to_cute_tensor(etab_f, assumed_align=1),
                 to_cute_tensor(fac, assumed_align=4), to_cute_tensor(sf4_w, assumed_align=4),
                 to_cute_tensor(nvs, assumed_align=4), to_cute_tensor(cos, assumed_align=4),
                 to_cute_tensor(sin, assumed_align=4), to_cute_tensor(st0, assumed_align=4)]
        pargs = ()
        if packed is not None:   # (cov3 compact: the tile table, the device tile count, the dense rows)
            kern.compact = True
            pargs = (to_cute_tensor(packed[0], assumed_align=4), to_cute_tensor(packed[1], assumed_align=4),
                     to_cute_tensor(packed[2], assumed_align=4), to_cute_tensor(packed[3], assumed_align=4))
        _TC5_CACHE[key] = cute.compile(kern, *cargs, Int64(0), Int32(1), stream, *pargs, options=_COMPILE_OPTIONS)
    grid = min(_sm_count(dev), b * h * tiles)
    _TC5_CACHE[key](*args, seed, grid, stream, *(packed if packed is not None else ()))


# ---------------------------------------------------------------------------------------------------- per-tensor kernel
# Workspace (int32, per device and stream, zero at rest -- the last CTA of every launch restores it):
#   [0] grid-barrier arrivals   [1] grid-barrier generation   [2] finished CTAs   [3] unused
#   [4 + i] amax bits of tensor i   [8 + i] max row ||x_i||^2 bits   [12] dS bound bits
_WS = {}


_DOC_WS = {}


def _doc_workspace(device, numel: int) -> torch.Tensor:
    """quantize_fp8_per_tensor(documents=...): the persistent zero int32 slots of the per-(document, kv head) amax
    reduction, per (device, stream) like _workspace; each launch's last CTA re-zeroes the slots it used. Grows (a fresh
    zero allocation) only when a call needs more slots than any before it on the stream."""
    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    t = _DOC_WS.get(key)
    if t is None or t.numel() < numel:
        t = _DOC_WS[key] = torch.zeros(max(numel, 1024), dtype=torch.int32, device=device)
    return t[:numel]


def _workspace(device) -> torch.Tensor:
    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    if key not in _WS:
        _WS[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return _WS[key]


@dsl_user_op
def _st_byte(ptr: cute.Pointer, idx, val, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [(ptr.toint(loc=loc, ip=ip) + Int64(idx)).ir_value(), Int32(val).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .b16 t;\n\tcvt.u16.u32 t, $1;\n\tst.global.u8 [$0], t;\n\t}\n", "l,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def _scale_of(source: cutlass.Constexpr, fmax: cutlass.Constexpr, i: cutlass.Constexpr, mScales, mWs, hs: Float32,
              acquire: cutlass.Constexpr = True):
    """(1 / s, s, s) of tensor i, float32: s given (a host float32 argument / a device float32) or
    s = max(amax / FP8_MAX, 1e-12) (div.rn.f32, then the clamp) from the reduced amax; 1 / s = rcp.rn.f32(s).
    acquire=False: the amax was reduced by a previous launch (a plain L2 load; an acquire would hold every later load of
    the CTA back)."""
    sd = Float32(1.0)
    if const_expr(source == "host"):
        sd = hs
    elif const_expr(source == "device"):
        sd = mScales[i]
    else:
        a = Float32(0.0)
        if const_expr(acquire):
            a = bits_f32(ld_acquire_i32(mWs.iterator + 4 + i))
        else:
            a = bits_f32(ld_cg_i32(mWs.iterator + 4 + i))
        sd = div_rn(a, Float32(fmax))
        if sd < Float32(1e-12):
            sd = Float32(1e-12)
        if const_expr(source == "computed_pow2"):
            # power_of_two_scale: a later token moves the scale only across a power-of-two boundary, and a power of
            # two divides exactly (no change to any code that stays normal)
            sd = pow2_ceil_f32(sd)
    return rcp_rn(sd), sd, sd


@cute.jit
def _grid_barrier(mWs, gdim):
    """Grid-wide barrier of a persistent launch (every CTA resident): arrival counter ws[0], generation ws[1]."""
    tidx, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if tidx == 0:
        fence_acq_rel_gpu()
        gen = ld_acquire_i32(mWs.iterator + 1)
        arrived = atomic_add_acq_rel_i32(mWs.iterator, Int32(1))
        if arrived == gdim - 1:
            st_relaxed_i32(mWs.iterator, Int32(0))
            fence_acq_rel_gpu()
            atomic_add_acq_rel_i32(mWs.iterator + 1, Int32(1))
        else:
            while ld_acquire_i32(mWs.iterator + 1) == gen:
                pass
        fence_acq_rel_gpu()
    cute.arch.sync_threads()


@cute.jit
def _last_cta_epilogue(mWs, mScales, gdim, source: cutlass.Constexpr, fmax: cutlass.Constexpr, n: cutlass.Constexpr,
                       hss, mNormOut, norms: cutlass.Constexpr, bound: cutlass.Constexpr, mBoundOut,
                       derived: cutlass.Constexpr = (), mDerived=None, dms=(), dds=()):
    """The last CTA to finish publishes the computed scales and the reduced side outputs, then zeroes the workspace."""
    tidx, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if tidx == 0:
        fence_acq_rel_gpu()
        done = atomic_add_acq_rel_i32(mWs.iterator + 2, Int32(1))
        if done == gdim - 1:
            fence_acq_rel_gpu()
            if const_expr(source in ("computed", "computed_pow2")):
                for i in cutlass.range_constexpr(n):
                    _inv, _sc, sd = _scale_of(source, fmax, i, mScales, mWs, hss[i])
                    mScales[i] = sd
            for i in cutlass.range_constexpr(n):
                if const_expr(norms[i]):
                    mNormOut[i] = bits_f32(ld_acquire_i32(mWs.iterator + 8 + i))
            if const_expr(bound):
                mBoundOut[0] = bits_f32(ld_acquire_i32(mWs.iterator + 12))
            # derived fp32 scalars (m_k s_i) / d_k of the scales, float32 mul.rn then div.rn (the kernels' epilogue
            # scalars)
            for k in cutlass.range_constexpr(len(derived)):
                if const_expr(derived[k] < 0):
                    mDerived[k] = div_rn(dms[k], dds[k])
                else:
                    _inv, _sc, sd = _scale_of(source, fmax, derived[k], mScales, mWs, hss[derived[k]])
                    mDerived[k] = div_rn(mul_rn(dms[k], sd), dds[k])
            for k in cutlass.range_constexpr(2, 16):
                st_relaxed_i32(mWs.iterator + k, Int32(0))


@cute.jit
def _doc_scale_of(fmax: cutlass.Constexpr, pow2: cutlass.Constexpr, a32: Int32):
    """The per-(document, kv head) scale from a reduced amax's bits: _scale_of's computed expression, s = max(amax /
    FP8_MAX, 1e-12) in float32 (rounded up to a power of two with pow2) -> (1 / s, s, s)."""
    sd = div_rn(bits_f32(a32), Float32(fmax))
    if sd < Float32(1e-12):
        sd = Float32(1e-12)
    if const_expr(pow2):
        sd = pow2_ceil_f32(sd)
    return rcp_rn(sd), sd, sd


@cute.jit
def _doc_epilogue(mWs, gdim, sred, fmax: cutlass.Constexpr, pow2: cutlass.Constexpr, n: cutlass.Constexpr,
                  derived: cutlass.Constexpr, fold: cutlass.Constexpr, mDocAmax, mDocScale, mDocDer, mDocFold, mDocQ):
    """Per-document mode, the tail of the AMAX launch: the last CTA to finish (its own counter, ws[3]) turns every
    reduced (tensor, document, kv head) amax into its scale -- the float32 table (mDocScale), the derived float32
    tables (float32((s_a m) s_b), the kernels' per-document scalars) and the quantise launch's per-slot float32
    (1 / s, s, D fold multiplier) (mDocQ) -- and re-zeroes the amax slots (a persistent zero workspace, like ws). The
    quantise launch then reads one float32 triple per tile, prefetched with the tile's staging: no scale arithmetic and no
    dependent load chain on its hot path."""
    tidx, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if tidx == 0:
        fence_acq_rel_gpu()
        done = atomic_add_acq_rel_i32(mWs.iterator + 3, Int32(1))
        last = Float32(0.0)
        if done == gdim - 1:
            fence_acq_rel_gpu()
            st_relaxed_i32(mWs.iterator + 3, Int32(0))
            last = Float32(1.0)
        sred[0] = last
    cute.arch.sync_threads()
    if sred[0] != Float32(0.0):
        nd = mDocAmax.shape[1] * mDocAmax.shape[2]
        e = Int32(tidx)
        while e < nd:
            sds = [Float32(1.0)] * n
            # every slot's amax load first (L2: the red.max results; thread 0's acquire fence + the barrier order them),
            # then the arithmetic: one load latency per slot group, not one per tensor
            av = [Int32(0)] * n
            for i in cutlass.range_constexpr(n):
                av[i] = ld_cg_i32(mDocAmax.iterator + i * nd + e)
            fv = Float32(0.0)
            if const_expr(fold is not None):
                fv = mDocFold[e]
            for i in cutlass.range_constexpr(n):
                a = av[i]
                inv, sc, sd = _doc_scale_of(fmax, pow2, a)
                mDocScale[i * nd + e] = sd
                sds[i] = sd
                mDocQ[(i * nd + e) * 4] = inv
                mDocQ[(i * nd + e) * 4 + 1] = sc
                if const_expr(i == 0 and fold is not None):
                    # the D fold of the backward, this document's m, in float32 as the drivers' scalar kernels form
                    # it (device_scales): "fp8" m = -(c / (s_v s_dO)) (c = ds_boost / 2^shift), "dk8"
                    # m = -((s_v s_dO) / 2^shift) (1 / (s_v s_dO)) (the division by s_ds = 1 is exact)
                    sv = fv
                    fk, fc = fold
                    t = mul_rn(sv, sd)
                    if const_expr(fk == "fp8"):
                        mDocQ[(i * nd + e) * 4 + 2] = -div_rn(Float32(fc), t)
                    else:
                        mDocQ[(i * nd + e) * 4 + 2] = -mul_rn(div_rn(t, Float32(fc)), rcp_rn(t))
                st_relaxed_i32(mDocAmax.iterator + i * nd + e, Int32(0))
            for k in cutlass.range_constexpr(len(derived)):
                ia, ib, mult = derived[k]
                if const_expr(ib < 0):
                    mDocDer[k * nd + e] = mul_rn(sds[ia], Float32(mult))
                else:
                    mDocDer[k * nd + e] = mul_rn(mul_rn(sds[ia], Float32(mult)), sds[ib])
            e = e + 128
    cute.arch.sync_threads()


# the lower members of the lane-pair row norm's in-lane levels (bits 2, 3, 0 after the first level's bit 1)
_PAIR_NORM_LOWS = (tuple(e for e in range(64) if e & 6 == 0),
                   tuple(e for e in range(64) if e & 14 == 0),
                   tuple(e for e in range(64) if e & 15 == 0))
_PT_CTAS_PER_SM = 6      # the rows kernel's resident CTAs per SM (launch minctasm: <= 80 registers at 128 threads)
# with lane-pair rows (RoPE / Hadamard: 64 values of a row per lane) 80 registers spill (16-272 B of local memory per
# thread, worst with several rotated tensors on the tensor cores); 4 CTAs (<= 128 registers) spill 0-56 B. Rotated
# Q + K + V with V norms at b4 s4096 h16 d128: 169 -> 126 us (tensor_core), 144 -> 120 us (butterfly), bitwise the
# same outputs; 3 CTAs measured no better, 2 worse
_PT_CTAS_PER_SM_PAIR = 4
_PT_CTAS = {}            # compiled rows kernel -> its resident CTAs per SM
_PT_ITERS = 8            # row steps per tile: 8 x 16 bytes of 16-bit input per thread per stage (two stages)


class _PerTensorKernel:
    """Per-tensor FP8 of up to four (b_i, s_i, h_i, 64 / 128 / 256) tensors with row options. A tile is (tensor, b, h,
    block of 64 tokens at head dim 128); a persistent grid of _PT_CTAS_PER_SM resident CTAs per SM walks the tiles with a
    two-stage cp.async pipeline (tile p + grid in flight while tile p is processed; per-row scalar inputs -- LSE,
    P_max, row divisors -- staged with it). Two row layouts, chosen per tensor at compile time: hd / 8 lanes per row,
    8 elements each (row trees across lanes in the old reduction order); or, for RoPE / Hadamard rows at head dim 128
    without the backward-prologue outputs, a lane pair per row (_pair_transform in registers). Per-row scalar work
    (D, LSE, dS bound) runs once per row. `mode`:
      "quant"  quantise with scales that are given (host float32 arguments or a float32 device vector) or that the
               previous "amax" launch reduced into the workspace;
      "amax"   only the amax of every tensor (of the transformed rows: RoPE / Hadamard / row divisors applied);
      "fused"  single launch: the resident grid reduces the amax, meets at a grid-wide barrier, then quantises.
    The last CTA to finish publishes the computed scales and the reduced side outputs (row norms, dS bound, derived
    scalars) and zeroes the workspace; with nothing to publish there is no epilogue."""

    def __init__(self, n, in_dtypes, fmt, rot, rope, rope_t, divide, norms, codes, scale_source, out_dtype, lse,
                 zero_acc, bound, mode, hd=128, derived=(), docs=None):
        self.n = n
        # per-document mode (aligned varlen, per-tensor operands): None, or dict(side=('q' | 'k' per tensor),
        # derived=((a, b, multiplier), ...), fold=c or None, pow2=bool) -- every (document, kv head) of every tensor
        # gets its own scale, reduced over the document's real rows only (rows past its true length are staged as
        # zeros); see quantize_fp8_per_tensor(documents=...)
        self.docs = docs
        self.derived = derived          # per derived scalar: the scale index, or -1 for a constant
        self.hd = hd                    # head dim: 64 / 128 / 256 (RoPE / Hadamard: 128)
        self.codes = codes              # per tensor bool: write the codes (False: the scale only)
        self.in_dtypes = in_dtypes
        self.fmt = fmt
        self.fmax = FP8_MAX[fmt]
        self.rot = rot                  # per tensor: signs tuple or None
        self.rope = rope                # per tensor bool
        self.rope_t = rope_t
        self.divide = divide            # per tensor bool: x / f_row, rounded through the input dtype
        self.norms = norms              # per tensor bool: max row ||x||^2
        self.scale_source = scale_source  # "host" | "device" | "computed" | "computed_pow2" (power_of_two_scale)
        self.out_dtype = out_dtype      # D (tensor 0) with this O dtype, or None
        self.lse = lse
        self.zero_acc = zero_acc
        self.bound = bound
        self.mode = mode
        # row layout: LANES threads per row, 8 contiguous elements each (one 16-byte load of a 16-bit input); a CTA holds
        # 128 / LANES rows per step and ITERS steps per tile, so its loads of a tile are all in flight together
        self.lanes = hd // 8
        self.rows_step = 128 // self.lanes
        self.iters = _PT_ITERS
        self.rows_tile = self.iters * self.rows_step
        # staging: thread-private slots (step k, thread, 16-byte chunk) of the x tile and of O
        self.x_chunks = max(dt_.width for dt_ in in_dtypes) // 16        # 16-byte chunks per lane step (8 elements)
        self.o_chunks = 0 if out_dtype is None else out_dtype.width // 16
        self.x_buf = 128 * self.iters * 16 * self.x_chunks               # bytes of one x stage
        self.o_buf = 128 * self.iters * 16 * self.o_chunks
        # per-row scalar inputs staged with the tile: slot a of LSE (tensor 0), P_max (tensor 0), each tensor's F
        arrs = []
        self.a_lse = len(arrs) if lse else None
        arrs += ["lse"] if lse else []
        self.a_pm = len(arrs) if bound else None
        arrs += ["pmax"] if bound else []
        self.a_f = []
        for i_ in range(4):
            self.a_f.append(len(arrs) if divide[i_] else None)
            arrs += ["f%d" % i_] if divide[i_] else []
        self.a_f = tuple(self.a_f)
        self.n_arr = len(arrs)
        self.smem_bytes = 64 + 2 * (self.x_buf + self.o_buf) + 2 * self.iters * 128 * 4 * max(self.n_arr, 1) + 64
        self.sign_words = tuple(None if r_ is None else _sign_words(r_) for r_ in rot)
        # lane-pair tensors: RoPE / Hadamard rows at head dim 128 without D / LSE / accumulator / bound / divisor
        self.pair = tuple((rot[i_] is not None or rope[i_]) and hd == 128
                          and not (i_ == 0 and (out_dtype is not None or lse or zero_acc or bound))
                          for i_ in range(4))
        # resident CTAs per SM (launch minctasm, so the register budget): fewer with lane-pair rows (no spills)
        self.ctas_per_sm = _PT_CTAS_PER_SM_PAIR if any(self.pair) else _PT_CTAS_PER_SM
        # hadamard_rotation_method="tensor_core": per tensor, the lane-pair rows take the quad form (_quad_row /
        # _quad_quant: quad_hadamard_rotate on the staged rows); set by the host for rotated 16-bit lane-pair tensors
        self.tc = (False,) * 4
        self.quad_signs = tuple(None if r_ is None else quad_sign_words(r_) for r_ in rot)

    @cute.jit
    def __call__(self, mX0, mX1, mX2, mX3, mF0, mF1, mF2, mF3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, mNormOut, mO, mD,
                 mLse, mLseOut, mAcc, mPmax, mVn2, mBoundOut, mCos, mSin, mDerived, hs0: Float32, hs1: Float32, hs2: Float32,
                 hs3: Float32, off1: Int32, off2: Int32, off3: Int32, total: Int32, qoff1: Int32, qoff2: Int32,
                 qoff3: Int32, qtotal: Int32, lse_shift: Float32, dm0: Float32, dm1: Float32, dm2: Float32,
                 dd0: Float32, dd1: Float32, dd2: Float32, grid_x: Int32, stream: cuda.CUstream,
                 mReQ=None, mReK=None, mTdQ=None, mTdK=None, mDocAmax=None, mDocScale=None, mDocDer=None,
                 mDocFold=None, mDocQ=None, mVtQ=None, mVtK=None, mNT=None):
        self.kernel(mX0, mX1, mX2, mX3, mF0, mF1, mF2, mF3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, mNormOut, mO, mD, mLse,
                    mLseOut, mAcc, mPmax, mVn2, mBoundOut, mCos, mSin, mDerived, hs0, hs1, hs2, hs3, off1, off2, off3,
                    total, qoff1, qoff2, qoff3, qtotal, lse_shift, dm0, dm1, dm2, dd0, dd1, dd2,
                    mReQ, mReK, mTdQ, mTdK, mDocAmax, mDocScale, mDocDer, mDocFold, mDocQ,
                    *((mVtQ, mVtK, mNT) if const_expr(self._packed()) else ())).launch(
            grid=[grid_x, 1, 1], block=[128, 1, 1], smem=self.smem_bytes, stream=stream, min_blocks_per_mp=self.ctas_per_sm,
            preferred_smem_carveout=100)

    def _packed(self) -> bool:
        """(lp_docmaps compact maps) the tensors are the caller's PACKED (1, T, h, d) rows: tile p = (h, aligned tile t
        of the side's device tile count, token block); its rows are packed rows from the tile table's (packed start,
        real rows) -- see quantize_fp8_per_tensor(documents=dict(packed=...))."""
        return self.docs is not None and bool(self.docs.get("packed", False))

    @cute.jit
    def _tilepos(self, mX, p, i=None):
        """Tile p = (b, h, token block): (b, h, its first token). Packed (_packed, tensor i): (0, h, its first PACKED
        row, the end of its tile's real packed rows)."""
        if const_expr(self._packed()):
            mVtQ, mVtK, ntq, ntk = self._pkv
            mVt, nt = (mVtQ, ntq) if const_expr(self.docs["side"][i] == "q") else (mVtK, ntk)
            bpt = 128 // self.rows_tile
            nts = nt * bpt
            hh = p // nts
            rem = p - hh * nts
            t = rem // bpt
            pb = mVt[0, t]
            return Int32(0), hh, pb + (rem - t * bpt) * self.rows_tile, pb + (mVt[1, t] & Int32(0xFFFF))
        ns, nh = mX.shape[1], mX.shape[2]
        nts = (ns + self.rows_tile - 1) // self.rows_tile
        bh = p // nts
        return bh // nh, bh % nh, (p - bh * nts) * self.rows_tile

    @cute.jit
    def _rowpos(self, mX, tp, k):
        """Row (b, s, h) of this thread at step k of the tile at tp = _tilepos(...), its validity and the thread's first
        column."""
        tidx, _, _ = cute.arch.thread_idx()
        bb, hh, s0 = tp[0], tp[1], tp[2]
        ss = s0 + k * self.rows_step + tidx // self.lanes
        valid = ss < (tp[3] if const_expr(len(tp) > 3) else mX.shape[1])
        if not valid:
            ss = Int32(0)
        return bb, ss, hh, valid, (tidx % self.lanes) * 8

    @cute.jit
    def _load8(self, mX, bb, ss, hh, col, valid, dt: cutlass.Constexpr):
        v = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        rb = cute.make_rmem_tensor(cute.make_layout(8), dt)
        for e in cutlass.range_constexpr(8):
            rb[e] = dt(0.0)
        if valid:
            src = cute.make_tensor(cute.make_ptr(dt, (mX[bb, ss, hh, None].iterator + col).toint(), cute.AddressSpace.gmem,
                                                 assumed_align=16), cute.make_layout(8))
            cute.autovec_copy(src, rb)
        v.store(rb.load().to(Float32))
        return v

    @cute.jit
    def _stage(self, mX, p, dt: cutlass.Constexpr, sb, chunks: cutlass.Constexpr, buf, mShape=None, row_end=None,
               i=None):
        """cp.async this thread's 8 elements of every row step of tile p into its private slots of stage buf of sb.
        row_end (per-document mode): rows at or past it are staged as zeros (a document's padding rows)."""
        tidx, _, _ = cute.arch.thread_idx()
        base = Int32(sb.iterator.toint()) + buf * (128 * self.iters * 16 * chunks)
        eb = dt.width // 8
        mS = mX if const_expr(mShape is None) else mShape
        tp = self._tilepos(mS, p, i)
        for k in cutlass.range_constexpr(self.iters):
            bb, ss, hh, valid, col = self._rowpos(mS, tp, k)
            nbytes = Int32(16)
            if not valid:
                nbytes = Int32(0)          # a row past s: zeros (its address is clamped to row 0)
            if const_expr(row_end is not None):
                if ss >= row_end:
                    nbytes = Int32(0)      # a document's padding row: zeros
            for c in cutlass.range_constexpr(8 * eb // 16):
                cp_async_16_zfill(base + ((k * 128 + tidx) * chunks + c) * 16,
                                  mX[bb, ss, hh, None].iterator + col + c * (16 // eb), nbytes)

    @cute.jit
    def _staged_f32(self, sb, k: cutlass.Constexpr, valid, dt: cutlass.Constexpr, chunks: cutlass.Constexpr, buf):
        """The thread's 8 staged elements of step k (stage buf) as fp32 (zeros for a row past the end)."""
        tidx, _, _ = cute.arch.thread_idx()
        w = cute.make_rmem_tensor(cute.make_layout(4 * chunks), Int32)
        cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, sb.iterator.toint() + buf * (128 * self.iters * 16 * chunks)
                                                         + ((k * 128 + tidx) * chunks) * 16,
                                                         cute.AddressSpace.smem, assumed_align=16),
                                           cute.make_layout(4 * chunks)), w)
        v = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        if const_expr(dt.width == 32):
            for e in cutlass.range_constexpr(8):
                v[e] = bits_f32(w[e])
        else:
            half = "bf16" if const_expr(dt is cutlass.BFloat16) else "f16"
            for e in cutlass.range_constexpr(4):
                v[2 * e] = half_lo_f32(w[e], half)
                v[2 * e + 1] = half_hi_f32(w[e], half)
        return v

    @cute.jit
    def _load_raw(self, mX, bb, ss, hh, col, valid, dt: cutlass.Constexpr, raw, k: cutlass.Constexpr):
        for e in cutlass.range_constexpr(8):
            raw[e, k] = dt(0.0)
        if valid:
            src = cute.make_tensor(cute.make_ptr(dt, (mX[bb, ss, hh, None].iterator + col).toint(), cute.AddressSpace.gmem,
                                                 assumed_align=16), cute.make_layout(8))
            cute.autovec_copy(src, raw[None, k])

    @cute.jit
    def _raw_f32(self, raw, k: cutlass.Constexpr):
        v = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for e in cutlass.range_constexpr(8):
            v[e] = Float32(raw[e, k])
        return v

    @cute.jit
    def _transform(self, i: cutlass.Constexpr, v, mF, mCos, mSin, bb, ss, hh, col, mX, sS, buf, k: cutlass.Constexpr):
        """Row factor, RoPE, Hadamard on the thread's 8 elements of the row (partners through lane shuffles)."""
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.in_dtypes[i]
        ns, nh = mX.shape[1], mX.shape[2]
        lane = tidx % self.lanes
        if const_expr(self.divide[i]):
            # (x / f_row).to(storage dtype): fp4q_diag_triton's Q / F_q (a power of two, so both steps are exact)
            f = self._rowval(sS, self.a_f[i], buf, k)
            rb = cute.make_rmem_tensor(cute.make_layout(8), dt)
            for e in cutlass.range_constexpr(8):
                v[e] = _asm_f32("div.rn.f32 $0, $1, $2;", v[e], f)
            rb.store(v.load().to(dt))
            v.store(rb.load().to(Float32))
        if const_expr(self.rope[i] or self.rot[i] is not None):
            v = _lane_transform(v, lane, ss, mCos, mSin, self.rope[i], self.rope_t, self.sign_words[i], dt)
        return v

    @cute.jit
    def _row_tree(self, part):
        """The old reduction order over a row from this thread's 4 first-level partials (columns 8 L + {0, 1, 4, 5},
        partner + 2): bit 2 in the thread, bit 3 across lane ^ 1, bit 0 in the thread, then the high column bits from
        the top down across lanes. Every lane of the row ends with the row's value."""
        q0 = add_rn(part[0], part[2])
        q1 = add_rn(part[1], part[3])
        q0 = add_rn(q0, cute.arch.shuffle_sync_bfly(q0, offset=1))
        q1 = add_rn(q1, cute.arch.shuffle_sync_bfly(q1, offset=1))
        t = add_rn(q0, q1)
        top = self.hd.bit_length() - 1
        for bit in cutlass.range_constexpr(top - 1, 3, -1):
            t = add_rn(t, cute.arch.shuffle_sync_bfly(t, offset=1 << (bit - 3)))
        return t

    @cute.jit
    def _amax_tile(self, i: cutlass.Constexpr, mX, mF, mCos, mSin, p, run: Float32, sbuf, sS, buf) -> Float32:
        m = run
        dt = self.in_dtypes[i]
        tp = self._tilepos(mX, p, i)
        for k in cutlass.range_constexpr(self.iters):
            bb, ss, hh, valid, col = self._rowpos(mX, tp, k)
            v = self._staged_f32(sbuf, k, valid, dt, self.x_chunks, buf)
            v = self._transform(i, v, mF, mCos, mSin, bb, ss, hh, col, mX, sS, buf, k)
            for e in cutlass.range_constexpr(8):
                m = cute.arch.fmax(m, cute.math.absf(v[e]))
        return m

    @cute.jit
    def _stage_pair(self, mX, p, dt: cutlass.Constexpr, sb, chunks: cutlass.Constexpr, buf, row_end=None, i=None):
        """Lane-pair staging: thread t copies the 64 elements of half row t % 2 of row t // 2 of tile p into its slots
        (chunk c in slot c; zeros past s)."""
        tidx, _, _ = cute.arch.thread_idx()
        base = Int32(sb.iterator.toint()) + buf * (128 * self.iters * 16 * chunks)
        eb = dt.width // 8
        units = 16 * chunks                    # 16-byte units per row (128 elements)
        tp = self._tilepos(mX, p, i)
        bb, hh, s0 = tp[0], tp[1], tp[2]
        s_end = tp[3] if const_expr(len(tp) > 3) else mX.shape[1]
        for k in cutlass.range_constexpr(8 * chunks):
            q = k * 128 + tidx
            r = q // units
            cu = q % units
            ss = s0 + r
            nbytes = Int32(16)
            if const_expr(row_end is not None):
                if ss >= row_end:
                    nbytes = Int32(0)      # a document's padding row: zeros
            if ss >= s_end:
                ss = Int32(0)
                nbytes = Int32(0)
            pos = r * units + (cu ^ ((2 * r + cu // (units // 2)) & 7))
            cp_async_16_zfill(base + pos * 16, mX[bb, ss, hh, None].iterator + cu * (16 // eb), nbytes)

    @cute.jit
    def _pair_row(self, i: cutlass.Constexpr, mX, p, mCos, mSin, sbuf, buf, mF):
        """Lane-pair form: (the transformed half row (64 fp32), b, s, h, valid, h-half)."""
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.in_dtypes[i]
        tp = self._tilepos(mX, p, i)
        bb, hh, s0 = tp[0], tp[1], tp[2]
        ss = s0 + tidx // 2
        valid = ss < (tp[3] if const_expr(len(tp) > 3) else mX.shape[1])
        if not valid:
            ss = Int32(0)
        hp = tidx % 2
        v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        chunks = self.x_chunks
        units = 16 * chunks
        r = tidx // 2
        base = sbuf.iterator.toint() + buf * (128 * self.iters * 16 * chunks)
        sw = (2 * r + hp) & 7
        for g in cutlass.range_constexpr(8):
            w = cute.make_rmem_tensor(cute.make_layout(4 * chunks), Int32)
            for j in cutlass.range_constexpr(chunks):
                cu = hp * (units // 2) + g * chunks + j
                pos = r * units + (cu ^ sw)
                wj = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, base + pos * 16, cute.AddressSpace.smem,
                                                                 assumed_align=16), cute.make_layout(4)), wj)
                for e in cutlass.range_constexpr(4):
                    w[4 * j + e] = wj[e]
            if const_expr(dt.width == 32):
                for e in cutlass.range_constexpr(8):
                    v[8 * g + e] = bits_f32(w[e])
            else:
                half = "bf16" if const_expr(dt is cutlass.BFloat16) else "f16"
                for e in cutlass.range_constexpr(4):
                    v[8 * g + 2 * e] = half_lo_f32(w[e], half)
                    v[8 * g + 2 * e + 1] = half_hi_f32(w[e], half)
        if const_expr(self.divide[i]):
            # (x / f_row).to(storage dtype), as _transform (a row past s reads row 0's factor: its zeros stay zeros)
            f = mF[(bb * mX.shape[2] + hh) * mX.shape[1] + ss]
            rb = cute.make_rmem_tensor(cute.make_layout(64), dt)
            for e in cutlass.range_constexpr(64):
                v[e] = _asm_f32("div.rn.f32 $0, $1, $2;", v[e], f)
            rb.store(v.load().to(dt))
            v.store(rb.load().to(Float32))
        v = _pair_transform(v, hp, ss, mCos, mSin, self.rope[i], self.rope_t, self.rot[i], dt)
        return v, bb, ss, hh, valid, hp

    @cute.jit
    def _pair_norm2(self, v) -> Float32:
        """||row||^2 of a lane-pair row in the old order: first level fma(x_c, x_c, fl(x_{c+2}^2)) (bit 1), bits 2, 3, 0
        in the lane, bit 6 across the pair, bits 5, 4 in the lane; both lanes end with the value."""
        t = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        for e in cutlass.range_constexpr(64):
            if const_expr((e & 2) == 0):
                t[e] = fma_rn(v[e], v[e], mul_rn(v[e + 2], v[e + 2]))
        for lv in cutlass.range_constexpr(3):
            hb = (4, 8, 1)[lv]
            for j in cutlass.range_constexpr(len(_PAIR_NORM_LOWS[lv])):
                e = _PAIR_NORM_LOWS[lv][j]
                t[e] = add_rn(t[e], t[e + hb])
        for j in cutlass.range_constexpr(4):          # the survivors: e = 16 j (bits 4, 5)
            t[16 * j] = add_rn(t[16 * j], cute.arch.shuffle_sync_bfly(t[16 * j], offset=1))
        t[0] = add_rn(t[0], t[32])
        t[16] = add_rn(t[16], t[48])
        return add_rn(t[0], t[16])

    @cute.jit
    def _stage_rowvals(self, mS, p, src, a: cutlass.Constexpr, sS, buf, lane0_only: cutlass.Constexpr = False,
                       i=None):
        """4-byte cp.async of the (b, h, s)-indexed row value of every row step of tile p into slot a (0 past s); with
        lane0_only only the row's first lane (the one that reads it) copies."""
        tidx, _, _ = cute.arch.thread_idx()
        ns, nh = mS.shape[1], mS.shape[2]
        tp = self._tilepos(mS, p, i)
        base = Int32(sS.iterator.toint()) + ((a * 2 + buf) * self.iters * 128 + tidx) * 4
        if const_expr(lane0_only):
            # row-scalar inputs: lane L < 8 of a row copies row step L's value (it alone reads it)
            lr = tidx % self.lanes
            if lr < self.iters:
                bb, ss, hh, valid, col = self._rowpos(mS, tp, lr)
                nbytes = Int32(4)
                if not valid:
                    nbytes = Int32(0)
                cp_async_4_zfill(base + lr * 128 * 4, src.iterator + (bb * nh + hh) * ns + ss, nbytes)
        else:
            for k in cutlass.range_constexpr(self.iters):
                bb, ss, hh, valid, col = self._rowpos(mS, tp, k)
                nbytes = Int32(4)
                if not valid:
                    nbytes = Int32(0)
                cp_async_4_zfill(base + k * 128 * 4, src.iterator + (bb * nh + hh) * ns + ss, nbytes)

    @cute.jit
    def _rowval_dyn(self, sS, a: cutlass.Constexpr, buf, k) -> Float32:
        tidx, _, _ = cute.arch.thread_idx()
        return sS[((a * 2 + buf) * self.iters + k) * 128 + tidx]

    @cute.jit
    def _rowval(self, sS, a: cutlass.Constexpr, buf, k: cutlass.Constexpr) -> Float32:
        tidx, _, _ = cute.arch.thread_idx()
        return sS[((a * 2 + buf) * self.iters + k) * 128 + tidx]

    @cute.jit
    def _stage_tile_of(self, p, ofs, quant: cutlass.Constexpr, xs, fs, mO, mLse, mPmax, sbuf, sO, sS, buf,
                       row_end=None):
        """Stage tile p (of whichever tensor owns it; nothing past the end) into stage buf; the caller commits.
        row_end (per-document mode): the tile's document's row end (_doc_meta, loaded a step ahead) -- every
        document's padding rows are staged as zeros."""
        for i in cutlass.range_constexpr(self.n):
            if const_expr(self.codes[i] or not quant):
                if p >= ofs[i]:
                    if p < ofs[i + 1]:
                        # (packed: a tile's rows past its document are not rows of it -- its real-row end bounds it)
                        re = None if const_expr(self._packed()) else row_end
                        if const_expr(self.pair[i]):
                            self._stage_pair(xs[i], p - ofs[i], self.in_dtypes[i], sbuf, self.x_chunks, buf, re, i)
                        else:
                            self._stage(xs[i], p - ofs[i], self.in_dtypes[i], sbuf, self.x_chunks, buf, None, re, i)
                        if const_expr(self.divide[i] and not self.pair[i]):
                            self._stage_rowvals(xs[i], p - ofs[i], fs[i], self.a_f[i], sS, buf, False, i)
                        if const_expr(quant and i == 0 and self.out_dtype is not None):
                            self._stage(mO, p - ofs[0], self.out_dtype, sO, self.o_chunks, buf, xs[0], re, 0)
                        if const_expr(quant and i == 0 and self.lse):
                            self._stage_rowvals(xs[0], p - ofs[0], mLse, self.a_lse, sS, buf, True, 0)
                        if const_expr(quant and i == 0 and self.bound and self.out_dtype is not None):
                            self._stage_rowvals(xs[0], p - ofs[0], mPmax, self.a_pm, sS, buf, True, 0)

    @cute.jit
    def _doc_tile(self, i: cutlass.Constexpr, mX, p, tabs):
        """Per-document mode: (document, kv head, row end) of tile p of tensor i. A tile lies inside one 128-row tile,
        so inside one document (aligned starts); the document tables are per 128-row tile of the Q side (Q, dO) or the K
        side (K, V): the document (the sentinel past the last real row) and the end of its real rows."""
        mReQ, mReK, mTdQ, mTdK, mDocAmax = tabs
        # b = 1 and s % 128 == 0 (the host checks both): tile p = (head, token block) with one division; the 128-row
        # tile and the kv head (the group size is a compile-time constant) by shifts
        nts = mX.shape[1] // self.rows_tile
        if const_expr(self._packed()):   # (packed: the side's device tile count of 128-row tiles)
            nts = (self._pkv[2] if const_expr(self.docs["side"][i] == "q") else self._pkv[3]) * (128 // self.rows_tile)
        hh = p // nts
        t = ((p - hh * nts) * self.rows_tile) // 128
        if const_expr(self.docs.get("batched", False)):
            # batched documents (b, s, h, d), b > 1: tile p = (b, h, token block) as _tilepos; the tables are per 128-row
            # tile of the flattened (b s) rows, the row ends local to the batch row
            bh = p // nts
            hh = bh % mX.shape[2]
            t = ((p - bh * nts) * self.rows_tile) // 128 + (bh // mX.shape[2]) * (mX.shape[1] // 128)
        doc = Int32(0)
        re = Int32(0)
        if const_expr(self.docs["side"][i] == "q"):
            doc = mTdQ[t]
            re = mReQ[t]
        else:
            doc = mTdK[t]
            re = mReK[t]
        kvh = hh // self.docs["group"][i]
        return doc, kvh, re

    @cute.jit
    def _doc_meta(self, p, ofs, quant: cutlass.Constexpr, xs, tabs):
        """Per-document mode: (row end, slot) of tile p -- its document's real-row end and its (tensor, document, kv
        head) slot index -- for whichever tensor owns it ((0, 0) past the end). The loops load it two tiles ahead, so
        the staging of a tile never waits on these table loads."""
        mDocAmax = tabs[4]
        re = Int32(0)
        slot = Int32(0)
        for i in cutlass.range_constexpr(self.n):
            if const_expr(self.codes[i] or not quant):
                if p >= ofs[i]:
                    if p < ofs[i + 1]:
                        _dd, _hk, _re = self._doc_tile(i, xs[i], p - ofs[i], tabs)
                        re = _re
                        slot = self._doc_idx(i, mDocAmax, _dd, _hk)
        return re, slot

    @cute.jit
    def _doc_idx(self, i: cutlass.Constexpr, mDocAmax, doc, kvh):
        return (i * mDocAmax.shape[1] + doc) * mDocAmax.shape[2] + kvh

    @cute.jit
    def _doc_amax_red(self, m: Float32, mDocAmax, slot):
        """One tile's max |x| into its (tensor, document, kv head) slot: a warp max, then one red.max per warp."""
        tidx, _, _ = cute.arch.thread_idx()
        for k in cutlass.range_constexpr(5):
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << k))
        if tidx % 32 == 0:
            if m > Float32(0.0):
                atomic_max_u32(mDocAmax.iterator + slot, f32_bits(m))

    @cute.jit
    def _scales(self, i: cutlass.Constexpr, mScales, mWs, hs: Float32):
        return _scale_of(self.scale_source, self.fmax, i, mScales, mWs, hs, self.mode == "fused")

    @cute.jit
    def _pair_quant(self, i: cutlass.Constexpr, mX, mQ, mCos, mSin, p, inv: Float32, sbuf, buf, mF) -> Float32:
        """Lane-pair quantisation of a tile: the half row's 64 codes (four 16-byte stores) and its row norm."""
        fmax = self.fmax
        v, bb, ss, hh, valid, hp = self._pair_row(i, mX, p, mCos, mSin, sbuf, buf, mF)
        w = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        for q in cutlass.range_constexpr(16):
            y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for e in cutlass.range_constexpr(0, 4, 2):
                y[e], y[e + 1] = cute.arch.mul_packed_f32x2((v[4 * q + e], v[4 * q + e + 1]), (inv, inv))
            w[q] = _pack4(self.fmt, _clamp(y[0], fmax), _clamp(y[1], fmax), _clamp(y[2], fmax), _clamp(y[3], fmax))
        if valid:
            cute.autovec_copy(w, cute.make_tensor(cute.make_ptr(Int32, (mQ[bb, ss, hh, None].iterator + 16 * hp).toint(),
                                                                cute.AddressSpace.gmem, assumed_align=16),
                                                  cute.make_layout(16)))
        nmax = Float32(0.0)
        if const_expr(self.norms[i]):
            n2 = self._pair_norm2(v)
            if valid:
                nmax = n2
        return nmax

    @cute.jit
    def _quad_row(self, i: cutlass.Constexpr, mX, p, mCos, mSin, sbuf, buf, mF):
        """The tensor-core form of _pair_row: warp w rotates rows 16 w .. 16 w + 15 of the 64-row tile in the quad layout
        (lane (g, tq): rows 16 w + g, + 8, chunks 4 m + tq), its A fragments read from the lane-pair staging (chunk cu
        of row r at slot r 16 + (cu ^ ((2 r + cu / 8) & 7))); row divisor, RoPE and the signs as _pair_row, then
        quad_hadamard_rotate. Returns (v (64 fp32, v[32 i + 8 m + e]), b, h, (s_0, s_1), (valid_0, valid_1))."""
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.in_dtypes[i]
        lane = tidx % 32
        warp = tidx // 32
        gq = lane // 4
        tq = lane % 4
        tp = self._tilepos(mX, p, i)
        bb, hh, s0 = tp[0], tp[1], tp[2]
        s_end = tp[3] if const_expr(len(tp) > 3) else mX.shape[1]
        base = sbuf.iterator.toint() + buf * (128 * self.iters * 16 * self.x_chunks)
        ssv = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        okv = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        a = cute.make_rmem_tensor(cute.make_layout(32), Int32)
        for i_ in cutlass.range_constexpr(2):
            r = 16 * warp + gq + 8 * i_
            ss = s0 + r
            ok = Int32(1)
            if ss >= s_end:
                ss = Int32(0)
                ok = Int32(0)
            ssv[i_] = ss
            okv[i_] = ok
            for m in cutlass.range_constexpr(4):
                cu = 4 * m + tq
                pos = r * 16 + (cu ^ ((2 * r + (m // 2)) & 7))
                cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, base + pos * 16, cute.AddressSpace.smem,
                                                                 assumed_align=16), cute.make_layout(4)),
                                  cute.make_tensor(a.iterator + 16 * i_ + 4 * m, cute.make_layout(4)))
        if const_expr(self.divide[i]):
            # (x / f_row).to(storage dtype), as _pair_row (a row past s reads row 0's factor: its zeros stay zeros)
            av = cute.make_tensor(cute.recast_ptr(a.iterator, dtype=dt), cute.make_layout(64))
            xf = cute.make_rmem_tensor(cute.make_layout(64), Float32)
            xf.store(av.load().to(Float32))
            for i_ in cutlass.range_constexpr(2):
                f = mF[(bb * mX.shape[2] + hh) * mX.shape[1] + ssv[i_]]
                for e in cutlass.range_constexpr(32):
                    xf[32 * i_ + e] = _asm_f32("div.rn.f32 $0, $1, $2;", xf[32 * i_ + e], f)
            av.store(xf.load().to(dt))
        if const_expr(self.rope[i]):
            quad_rope(a, tq, ssv[0], ssv[1], mCos, mSin, self.rope_t, dt)
        v = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        quad_hadamard_rotate(a, v, lane, self.quad_signs[i], dt is cutlass.Float16)
        return v, bb, hh, ssv, okv

    @cute.jit
    def _quad_quant(self, i: cutlass.Constexpr, mX, mQ, mCos, mSin, p, inv: Float32, sbuf, buf, mF) -> Float32:
        """The tensor-core form of _pair_quant: the quad rows' codes (chunk (i, m) -> 8 bytes at row byte 8 (4 m + tq))
        and the max row ||x||^2 of the valid rows (an fma chain per lane over its 32 values, then across the quad: an
        order of its own, not _pair_norm2's)."""
        tidx, _, _ = cute.arch.thread_idx()
        fmax = self.fmax
        lane = tidx % 32
        tq = tidx % 4
        v, bb, hh, ssv, okv = self._quad_row(i, mX, p, mCos, mSin, sbuf, buf, mF)
        for i_ in cutlass.range_constexpr(2):
            qrow = mQ[bb, ssv[i_], hh, None].iterator
            for m in cutlass.range_constexpr(4):
                w2 = cute.make_rmem_tensor(cute.make_layout(2), Int32)
                for hw in cutlass.range_constexpr(2):
                    y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                    for e in cutlass.range_constexpr(0, 4, 2):
                        y[e], y[e + 1] = cute.arch.mul_packed_f32x2(
                            (v[32 * i_ + 8 * m + 4 * hw + e], v[32 * i_ + 8 * m + 4 * hw + e + 1]), (inv, inv))
                    w2[hw] = _pack4(self.fmt, _clamp(y[0], fmax), _clamp(y[1], fmax), _clamp(y[2], fmax),
                                    _clamp(y[3], fmax))
                if okv[i_] != 0:
                    cute.autovec_copy(w2, cute.make_tensor(cute.make_ptr(Int32, (qrow + 2 * (4 * m + tq)).toint(),
                                                                         cute.AddressSpace.gmem, assumed_align=8),
                                                           cute.make_layout(2)))
        nmax = Float32(0.0)
        if const_expr(self.norms[i]):
            for i_ in cutlass.range_constexpr(2):
                acc = Float32(0.0)
                for k in cutlass.range_constexpr(32):
                    acc = fma_rn(v[32 * i_ + k], v[32 * i_ + k], acc)
                n2 = add_rn(acc, cute.arch.shuffle_sync_bfly(acc, offset=1))
                n2 = add_rn(n2, cute.arch.shuffle_sync_bfly(n2, offset=2))
                if okv[i_] != 0:
                    nmax = cute.arch.fmax(nmax, n2)
        return nmax

    @cute.jit
    def _quant_tile(self, i: cutlass.Constexpr, mX, mF, mQ, mCos, mSin, p, inv: Float32, sc32: Float32, mWs, mO, mD,
                    mLse, mLseOut, mAcc, mPmax, mVn2, lse_shift: Float32, sbuf, sO, sS, buf, row_end=None,
                    d_mul=None):
        tidx, _, _ = cute.arch.thread_idx()
        hd = self.hd
        with_o = i == 0 and self.out_dtype is not None
        want_norm = self.norms[i] or (i == 0 and self.bound)
        fmax = self.fmax
        dt = self.in_dtypes[i]
        ns, nh = mX.shape[1], mX.shape[2]
        lane0 = (tidx % self.lanes) == 0
        nmax = Float32(0.0)
        bmax = Float32(0.0)
        vmax = Float32(0.0)
        if const_expr(with_o and self.bound):
            vmax = sqrt_rn(mVn2[0])
        tp = self._tilepos(mX, p, i)
        lr = tidx % self.lanes
        nso = ns   # (the per-row outputs' row stride: packed, the dense rows' R)
        if const_expr(self._packed() and i == 0 and (self.out_dtype is not None or self.lse)):
            nso = (cute.size(mD.shape) if const_expr(self.out_dtype is not None) else cute.size(mLseOut.shape)) // nh
        d_keep = Float32(0.0)
        n_keep = Float32(0.0)
        for k in cutlass.range_constexpr(self.iters):
            bb, ss, hh, valid, col = self._rowpos(mX, tp, k)
            v = self._staged_f32(sbuf, k, valid, dt, self.x_chunks, buf)
            v = self._transform(i, v, mF, mCos, mSin, bb, ss, hh, col, mX, sS, buf, k)
            y = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            for e in cutlass.range_constexpr(0, 8, 2):
                y[e], y[e + 1] = cute.arch.mul_packed_f32x2((v[e], v[e + 1]), (inv, inv))
            w0 = _pack4(self.fmt, _clamp(y[0], fmax), _clamp(y[1], fmax), _clamp(y[2], fmax), _clamp(y[3], fmax))
            w1 = _pack4(self.fmt, _clamp(y[4], fmax), _clamp(y[5], fmax), _clamp(y[6], fmax), _clamp(y[7], fmax))
            if valid:
                dst = cute.make_tensor(cute.make_ptr(Int32, (mQ[bb, ss, hh, None].iterator + col // 4).toint(),
                                                     cute.AddressSpace.gmem, assumed_align=8), cute.make_layout(2))
                dst[0] = w0
                dst[1] = w1
            ridx = (bb * nh + hh) * ns + ss          # the (b, h, s) index of the row
            nrm2 = Float32(0.0)
            if const_expr(want_norm):
                part = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                for kk in cutlass.range_constexpr(4):
                    e = (0, 1, 4, 5)[kk]
                    part[kk] = fma_rn(v[e], v[e], mul_rn(v[e + 2], v[e + 2]))
                nrm2 = self._row_tree(part)
                if valid:
                    nmax = cute.arch.fmax(nmax, nrm2)
            if const_expr(with_o):
                ov = self._staged_f32(sO, k, valid, self.out_dtype, self.o_chunks, buf)
                dq = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                c0v = _code_values4(self.fmt, w0)
                c1v = _code_values4(self.fmt, w1)
                for e in cutlass.range_constexpr(4):
                    dq[e] = mul_rn(c0v[e], sc32)
                    dq[4 + e] = mul_rn(c1v[e], sc32)
                part = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                for kk in cutlass.range_constexpr(4):
                    e = (0, 1, 4, 5)[kk]
                    part[kk] = fma_rn(dq[e], ov[e], mul_rn(dq[e + 2], ov[e + 2]))
                dsum = self._row_tree(part)
                if lr == k:                          # this lane's row step for the scalar work below
                    d_keep = dsum
                    n_keep = nrm2
            if const_expr(i == 0 and self.zero_acc):
                if valid:
                    z = cute.make_tensor(cute.make_ptr(Int32, (mAcc.iterator + (ridx * hd + col) // 2).toint(),
                                                       cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(4))
                    for c in cutlass.range_constexpr(4):
                        z[c] = Int32(0)
        if const_expr(i == 0 and (self.out_dtype is not None or self.lse)):
            # ---- per-row scalar work, once per row: lane L < 8 of the row takes row step L
            if lr < self.iters:
                bb, ss, hh, valid, col = self._rowpos(mX, tp, lr)
                ridx = (bb * nh + hh) * nso + ss
                if valid:
                    if const_expr(with_o):
                        if const_expr(d_mul is not None):
                            mD[ridx] = mul_rn(d_keep, d_mul)   # per-document D fold D'' = m D (m of this document)
                        else:
                            mD[ridx] = d_keep
                    if const_expr(self.lse):
                        lv = self._rowval_dyn(sS, self.a_lse, buf, lr)
                        ll = Float32(0.0)
                        if lv != Float32(-math.inf):
                            ll = mul_rn(lv, Float32(LOG2E))
                        if const_expr(row_end is not None):
                            if ss >= row_end:
                                ll = Float32(math.inf)   # a document's padding row: P = exp2(S - inf) = 0
                        mLseOut[ridx] = ll - lse_shift
                if const_expr(with_o and self.bound):
                    # the dS scale bound min(P_max_q, 1) (||dO_q|| max_k ||V_k|| + |D_q|), each product rounded on its own
                    t1 = mul_rn(sqrt_rn(n_keep), vmax)
                    yq = t1 + cute.math.absf(d_keep)
                    pm = Float32(0.0)
                    if valid:
                        pm = fmin_f32(self._rowval_dyn(sS, self.a_pm, buf, lr), Float32(1.0))
                    bmax = cute.arch.fmax(bmax, mul_rn(pm, yq))
        return nmax, bmax

    @cute.kernel
    def kernel(self, mX0, mX1, mX2, mX3, mF0, mF1, mF2, mF3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, mNormOut, mO, mD, mLse,
               mLseOut, mAcc, mPmax, mVn2, mBoundOut, mCos, mSin, mDerived, hs0: Float32, hs1: Float32, hs2: Float32,
               hs3: Float32, off1: Int32, off2: Int32, off3: Int32, total: Int32, qoff1: Int32, qoff2: Int32,
               qoff3: Int32, qtotal: Int32, lse_shift: Float32, dm0: Float32, dm1: Float32, dm2: Float32,
               dd0: Float32, dd1: Float32, dd2: Float32, mReQ=None, mReK=None, mTdQ=None, mTdK=None, mDocAmax=None,
               mDocScale=None, mDocDer=None, mDocFold=None, mDocQ=None, mVtQ=None, mVtK=None, mNT=None):
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        n = self.n
        xs = (mX0, mX1, mX2, mX3)
        fs = (mF0, mF1, mF2, mF3)
        qs = (mQ0, mQ1, mQ2, mQ3)
        hss = (hs0, hs1, hs2, hs3)
        offs = (Int32(0), off1, off2, off3, total)
        qoffs = (Int32(0), qoff1, qoff2, qoff3, qtotal)   # the quantisation pass: tensors with codes only
        if const_expr(self._packed()):
            # packed rows (compact maps): the tiles of tensor i are (its heads) x (its side's DEVICE tile count) x
            # (token blocks per 128-row tile) -- the offsets from the device counts, not the host's static bound
            ntq, ntk = mNT[0], mNT[1]
            self._pkv = (mVtQ, mVtK, ntq, ntk)
            o_ = [Int32(0)]
            q_ = [Int32(0)]
            for i in cutlass.range_constexpr(n):
                c_ = xs[i].shape[2] * (ntq if const_expr(self.docs["side"][i] == "q") else ntk) * (128 // self.rows_tile)
                o_.append(o_[-1] + c_)
                q_.append(q_[-1] + (c_ if const_expr(self.codes[i]) else Int32(0)))
            offs = tuple(o_ + [o_[-1]] * (4 - n))
            qoffs = tuple(q_ + [q_[-1]] * (4 - n))
            total = o_[-1]
            qtotal = q_[-1]
        smem = cutlass.utils.SmemAllocator()
        sred = smem.allocate_tensor(Float32, cute.make_layout(4), 16)
        sbuf = smem.allocate_tensor(Int32, cute.make_layout(2 * self.x_buf // 4), 16)       # two stages
        sO = sbuf
        if const_expr(self.o_chunks > 0):
            sO = smem.allocate_tensor(Int32, cute.make_layout(2 * self.o_buf // 4), 16)
        sS = smem.allocate_tensor(Float32, cute.make_layout(2 * self.iters * 128 * max(self.n_arr, 1)), 16)
        tabs = None
        if const_expr(self.docs is not None):
            tabs = (mReQ, mReK, mTdQ, mTdK, mDocAmax)
        if const_expr(self.mode in ("amax", "fused")):
            # ---- phase 1: amax of every tensor over this CTA's tiles
            run = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for i in cutlass.range_constexpr(4):
                run[i] = Float32(0.0)
            p = pid
            buf = Int32(0)
            m0_re, m0_slot, m1_re, m1_slot = None, None, None, None
            if const_expr(self.docs is not None):
                m0_re, m0_slot = self._doc_meta(p, offs, False, xs, tabs)
                m1_re, m1_slot = self._doc_meta(p + gdim, offs, False, xs, tabs)
            self._stage_tile_of(p, offs, False, xs, fs, mO, mLse, mPmax, sbuf, sO, sS, buf, m0_re)
            cp_async_commit()
            while p < total:
                pn = p + gdim
                if const_expr(any(self.pair)):
                    cute.arch.sync_threads()        # every lane pair is done reading the stage written next
                self._stage_tile_of(pn, offs, False, xs, fs, mO, mLse, mPmax, sbuf, sO, sS, buf ^ 1, m1_re)   # the next tile in flight
                m2_re, m2_slot = None, None
                if const_expr(self.docs is not None):
                    m2_re, m2_slot = self._doc_meta(pn + gdim, offs, False, xs, tabs)
                cp_async_commit()
                cp_async_wait_1()
                if const_expr(any(self.pair)):
                    cute.arch.sync_threads()        # lane-pair rows were staged by other threads
                for i in cutlass.range_constexpr(n):
                    if p >= offs[i]:
                        if p < offs[i + 1]:
                            if const_expr(self.docs is not None):
                                # per-document mode: this tile's own max goes to its (document, kv head) slot; the
                                # running max still makes the tensor's global scale (the max over its documents)
                                tm = Float32(0.0)
                                if const_expr(self.pair[i] and self.tc[i]):
                                    v, _b, _h, _ss, _ok = self._quad_row(i, xs[i], p - offs[i], mCos, mSin, sbuf, buf,
                                                                         fs[i])
                                    for e in cutlass.range_constexpr(64):
                                        tm = cute.arch.fmax(tm, cute.math.absf(v[e]))
                                elif const_expr(self.pair[i]):
                                    v, _b, _s, _h, _valid, _hp = self._pair_row(i, xs[i], p - offs[i], mCos, mSin, sbuf,
                                                                                buf, fs[i])
                                    for e in cutlass.range_constexpr(64):
                                        tm = cute.arch.fmax(tm, cute.math.absf(v[e]))
                                else:
                                    tm = self._amax_tile(i, xs[i], fs[i], mCos, mSin, p - offs[i], tm, sbuf, sS, buf)
                                self._doc_amax_red(tm, mDocAmax, m0_slot)
                                run[i] = cute.arch.fmax(run[i], tm)
                            elif const_expr(self.pair[i] and self.tc[i]):
                                v, _b, _h, _ss, _ok = self._quad_row(i, xs[i], p - offs[i], mCos, mSin, sbuf, buf, fs[i])
                                for e in cutlass.range_constexpr(64):
                                    run[i] = cute.arch.fmax(run[i], cute.math.absf(v[e]))
                            elif const_expr(self.pair[i]):
                                v, _b, _s, _h, _valid, _hp = self._pair_row(i, xs[i], p - offs[i], mCos, mSin, sbuf, buf,
                                                                            fs[i])
                                for e in cutlass.range_constexpr(64):
                                    run[i] = cute.arch.fmax(run[i], cute.math.absf(v[e]))
                            else:
                                run[i] = self._amax_tile(i, xs[i], fs[i], mCos, mSin, p - offs[i], run[i], sbuf, sS,
                                                         buf)
                buf = buf ^ 1
                p = pn
                if const_expr(self.docs is not None):
                    m0_re, m0_slot, m1_re, m1_slot = m1_re, m1_slot, m2_re, m2_slot
            cp_async_wait_all()
            for i in cutlass.range_constexpr(n):
                m = _cta_max(run[i], sred)
                if tidx == 0:
                    if m > Float32(0.0):
                        atomic_max_u32(mWs.iterator + 4 + i, f32_bits(m))
            if const_expr(self.docs is not None):
                assert self.mode == "amax", "per-document mode: the amax launch publishes the tables"
                _doc_epilogue(mWs, gdim, sred, self.fmax, self.docs["pow2"], n, self.docs["derived"],
                              self.docs["fold"], mDocAmax, mDocScale, mDocDer, mDocFold, mDocQ)
        if const_expr(self.mode == "fused"):
            _grid_barrier(mWs, gdim)   # every CTA is resident: grid_x <= the occupancy bound
        if const_expr(self.mode in ("quant", "fused")):
            # ---- phase 2: quantise (the scales once per CTA)
            invs = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            sc32s = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for i in cutlass.range_constexpr(n):
                invs[i], sc32s[i], _sd = self._scales(i, mScales, mWs, hss[i])
            nm = cute.make_rmem_tensor(cute.make_layout(4), Float32)   # this CTA's max row ||x||^2 per tensor
            for i in cutlass.range_constexpr(4):
                nm[i] = Float32(0.0)
            bm = Float32(0.0)                                          # ... and its dS bound
            p = pid
            buf = Int32(0)
            m0_re, m0_slot, m1_re, m1_slot = None, None, None, None
            dq_c = None
            if const_expr(self.docs is not None):
                # per-document mode: tile metadata two tiles ahead; the tile's (1 / s, s, D fold) triple loaded with
                # its staging (one step ahead, in flight while the previous tile is quantised)
                m0_re, m0_slot = self._doc_meta(p, qoffs, True, xs, tabs)
                m1_re, m1_slot = self._doc_meta(p + gdim, qoffs, True, xs, tabs)
                dq_c = cute.make_rmem_tensor(cute.make_layout(3), Float32)
                for c in cutlass.range_constexpr(3):
                    dq_c[c] = mDocQ[m0_slot * 4 + c]
            self._stage_tile_of(p, qoffs, True, xs, fs, mO, mLse, mPmax, sbuf, sO, sS, buf, m0_re)
            cp_async_commit()
            while p < qtotal:
                pn = p + gdim
                if const_expr(any(self.pair)):
                    cute.arch.sync_threads()        # every lane pair is done reading the stage written next
                self._stage_tile_of(pn, qoffs, True, xs, fs, mO, mLse, mPmax, sbuf, sO, sS, buf ^ 1, m1_re)   # the next tile in flight
                dq_n = None
                m2_re, m2_slot = None, None
                if const_expr(self.docs is not None):
                    dq_n = cute.make_rmem_tensor(cute.make_layout(3), Float32)
                    for c in cutlass.range_constexpr(3):
                        dq_n[c] = mDocQ[m1_slot * 4 + c]
                    m2_re, m2_slot = self._doc_meta(pn + gdim, qoffs, True, xs, tabs)
                cp_async_commit()
                cp_async_wait_1()
                if const_expr(any(self.pair)):
                    cute.arch.sync_threads()        # lane-pair rows were staged by other threads
                for i in cutlass.range_constexpr(n):
                    if const_expr(self.codes[i]):
                        if p >= qoffs[i]:
                            if p < qoffs[i + 1]:
                                inv, sc32 = invs[i], sc32s[i]
                                d_re = None
                                d_mul = None
                                if const_expr(self.docs is not None):
                                    # this tile's (document, kv head) scale (and D fold), published by the amax
                                    # launch's tail and prefetched one tile ahead
                                    inv, sc32 = dq_c[0], dq_c[1]
                                    if const_expr(i == 0 and self.lse and not self._packed()):
                                        d_re = m0_re
                                    if const_expr(i == 0 and self.out_dtype is not None
                                                  and self.docs["fold"] is not None):
                                        d_mul = dq_c[2]
                                if const_expr(self.pair[i] and self.tc[i]):
                                    tn = self._quad_quant(i, xs[i], qs[i], mCos, mSin, p - qoffs[i], inv, sbuf, buf,
                                                          fs[i])
                                    nm[i] = cute.arch.fmax(nm[i], tn)
                                elif const_expr(self.pair[i]):
                                    tn = self._pair_quant(i, xs[i], qs[i], mCos, mSin, p - qoffs[i], inv, sbuf, buf,
                                                          fs[i])
                                    nm[i] = cute.arch.fmax(nm[i], tn)
                                else:
                                    tn, tb = self._quant_tile(i, xs[i], fs[i], qs[i], mCos, mSin, p - qoffs[i], inv,
                                                              sc32, mWs, mO, mD, mLse, mLseOut, mAcc, mPmax, mVn2,
                                                              lse_shift, sbuf, sO, sS, buf, d_re, d_mul)
                                    nm[i] = cute.arch.fmax(nm[i], tn)
                                    if const_expr(i == 0):
                                        bm = cute.arch.fmax(bm, tb)
                buf = buf ^ 1
                p = pn
                if const_expr(self.docs is not None):
                    for c in cutlass.range_constexpr(3):
                        dq_c[c] = dq_n[c]
                    m0_re, m0_slot, m1_re, m1_slot = m1_re, m1_slot, m2_re, m2_slot
            cp_async_wait_all()
            for i in cutlass.range_constexpr(n):
                if const_expr(self.norms[i] and self.codes[i]):
                    tmax = _cta_max(nm[i], sred)
                    if tidx == 0:
                        atomic_max_if_larger_f32(mWs.iterator + 8 + i, tmax)
            if const_expr(self.out_dtype is not None and self.bound):
                tmax = _cta_max(bm, sred)
                if tidx == 0:
                    atomic_max_u32(mWs.iterator + 12, f32_bits(tmax))
            if const_expr(self.scale_source in ("computed", "computed_pow2") or any(self.norms) or self.bound
                          or len(self.derived) > 0):
                _last_cta_epilogue(mWs, mScales, gdim, self.scale_source, self.fmax, n, hss, mNormOut, self.norms,
                                   self.bound, mBoundOut, self.derived, mDerived, (dm0, dm1, dm2), (dd0, dd1, dd2))


@cute.jit
def _cta_max(v: Float32, sred: cute.Tensor) -> Float32:
    """max over the CTA's 128 threads (warp shuffles, then 4 floats through shared memory)."""
    tidx, _, _ = cute.arch.thread_idx()
    m = v
    for i in cutlass.range_constexpr(5):
        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << i))
    cute.arch.sync_threads()
    if tidx % 32 == 0:
        sred[tidx // 32] = m
    cute.arch.sync_threads()
    return cute.arch.fmax(cute.arch.fmax(sred[0], sred[1]), cute.arch.fmax(sred[2], sred[3]))


class _FlatPerTensorKernel:
    """Per-tensor FP8 of up to four contiguous tensors of ANY shape, elementwise: codes = FP8(clamp(x * float32(1 / s))).
    The fast path of plain per-tensor quantisation (no row outputs), and the fallback for shapes the token-tile kernel
    cannot take (d not 64 / 128 / 256, non-contiguous rows, varlen-packed operands). 8192-element chunks walked by a
    grid of at most 8 CTAs per SM: 128 threads x 8 coalesced 16-byte loads per chunk (all in flight before any
    arithmetic); the same modes and workspace as _PerTensorKernel, the last-CTA epilogue only for computed scales."""

    CHUNK = 8192
    LOADS = CHUNK // 1024              # 16-byte loads per thread, all in flight together

    def __init__(self, n, in_dtypes, fmt, codes, scale_source, mode):
        self.n = n
        self.in_dtypes = in_dtypes
        self.fmt = fmt
        self.fmax = FP8_MAX[fmt]
        self.codes = codes
        self.scale_source = scale_source
        self.mode = mode

    @cute.jit
    def __call__(self, mX0, mX1, mX2, mX3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, hs0: Float32, hs1: Float32, hs2: Float32,
                 hs3: Float32, off1: Int32, off2: Int32, off3: Int32, total: Int32, qoff1: Int32, qoff2: Int32,
                 qoff3: Int32, qtotal: Int32, grid_x: Int32, stream: cuda.CUstream):
        self.kernel(mX0, mX1, mX2, mX3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, hs0, hs1, hs2, hs3, off1, off2, off3, total,
                    qoff1, qoff2, qoff3, qtotal).launch(grid=[grid_x, 1, 1], block=[128, 1, 1], smem=64, stream=stream)

    @cute.jit
    def _load(self, i: cutlass.Constexpr, mX, p):
        """The thread's 8 x 8 elements of chunk p of tensor i (sub-chunk k: elements p * 8192 + 1024 k + 8 tidx + [0, 8))
        as raw input-dtype registers (out-of-range elements -> 0), and the tensor's element count."""
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.in_dtypes[i]
        nel = mX.shape[0]
        rb = cute.make_rmem_tensor(cute.make_layout((8, self.LOADS), stride=(1, 8)), dt)
        for k in cutlass.range_constexpr(self.LOADS):
            e0 = p * self.CHUNK + k * 1024 + tidx * 8
            if e0 + 8 <= nel:
                src = cute.make_tensor(cute.make_ptr(dt, (mX.iterator + e0).toint(), cute.AddressSpace.gmem,
                                                     assumed_align=16), cute.make_layout(8))
                cute.autovec_copy(src, rb[None, k])
            else:
                for j in cutlass.range_constexpr(8):
                    rb[j, k] = dt(0.0)
                    if e0 + j < nel:
                        rb[j, k] = mX[e0 + j]
        return rb, nel

    @cute.kernel
    def kernel(self, mX0, mX1, mX2, mX3, mQ0, mQ1, mQ2, mQ3, mScales, mWs, hs0: Float32, hs1: Float32, hs2: Float32,
               hs3: Float32, off1: Int32, off2: Int32, off3: Int32, total: Int32, qoff1: Int32, qoff2: Int32,
               qoff3: Int32, qtotal: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        n = self.n
        xs = (mX0, mX1, mX2, mX3)
        qs = (mQ0, mQ1, mQ2, mQ3)
        hss = (hs0, hs1, hs2, hs3)
        offs = (Int32(0), off1, off2, off3, total)
        qoffs = (Int32(0), qoff1, qoff2, qoff3, qtotal)
        fmax = self.fmax
        smem = cutlass.utils.SmemAllocator()
        sred = smem.allocate_tensor(Float32, cute.make_layout(4), 16)
        if const_expr(self.mode in ("amax", "fused")):
            run = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for i in cutlass.range_constexpr(4):
                run[i] = Float32(0.0)
            p = pid
            while p < total:
                for i in cutlass.range_constexpr(n):
                    if p >= offs[i]:
                        if p < offs[i + 1]:
                            rb, _nel = self._load(i, xs[i], p - offs[i])
                            v = cute.make_rmem_tensor(cute.make_layout((8, self.LOADS), stride=(1, 8)), Float32)
                            v.store(rb.load().to(Float32))
                            for k in cutlass.range_constexpr(self.LOADS):
                                for e in cutlass.range_constexpr(8):
                                    run[i] = cute.arch.fmax(run[i], cute.math.absf(v[e, k]))
                p = p + gdim
            for i in cutlass.range_constexpr(n):
                m = _cta_max(run[i], sred)
                if tidx == 0:
                    if m > Float32(0.0):
                        atomic_max_u32(mWs.iterator + 4 + i, f32_bits(m))
        if const_expr(self.mode == "fused"):
            _grid_barrier(mWs, gdim)
        if const_expr(self.mode in ("quant", "fused")):
            invs = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for i in cutlass.range_constexpr(n):
                invs[i], _sc, _sd = _scale_of(self.scale_source, self.fmax, i, mScales, mWs, hss[i], self.mode == "fused")
            p = pid
            while p < qtotal:
                for i in cutlass.range_constexpr(n):
                    if const_expr(self.codes[i]):
                        if p >= qoffs[i]:
                            if p < qoffs[i + 1]:
                                pp = p - qoffs[i]
                                rb, nel = self._load(i, xs[i], pp)
                                v = cute.make_rmem_tensor(cute.make_layout((8, self.LOADS), stride=(1, 8)), Float32)
                                v.store(rb.load().to(Float32))
                                for k in cutlass.range_constexpr(self.LOADS):
                                    y = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                                    for e in cutlass.range_constexpr(0, 8, 2):
                                        y[e], y[e + 1] = cute.arch.mul_packed_f32x2((v[e, k], v[e + 1, k]),
                                                                                    (invs[i], invs[i]))
                                    w0 = _pack4(self.fmt, _clamp(y[0], fmax), _clamp(y[1], fmax), _clamp(y[2], fmax),
                                                _clamp(y[3], fmax))
                                    w1 = _pack4(self.fmt, _clamp(y[4], fmax), _clamp(y[5], fmax), _clamp(y[6], fmax),
                                                _clamp(y[7], fmax))
                                    e0 = pp * self.CHUNK + k * 1024 + tidx * 8
                                    if e0 + 8 <= nel:
                                        dst = cute.make_tensor(cute.make_ptr(Int32, (qs[i].iterator + e0 // 4).toint(),
                                                                             cute.AddressSpace.gmem, assumed_align=8),
                                                               cute.make_layout(2))
                                        dst[0] = w0
                                        dst[1] = w1
                                    else:
                                        for j in cutlass.range_constexpr(8):
                                            if e0 + j < nel:
                                                wsel = w0 if const_expr(j < 4) else w1
                                                _st_byte(qs[i].iterator, e0 + j, (wsel >> (8 * (j % 4))) & 0xFF)
                p = p + gdim
            if const_expr(self.scale_source in ("computed", "computed_pow2")):   # given scales: nothing to publish
                _last_cta_epilogue(mWs, mScales, gdim, self.scale_source, self.fmax, n, hss, mScales, (False,) * 4,
                                   False, mScales)


_PT_CACHE = {}
_SM_COUNT = {}


def _normalise_scales(scales, n, dev, power_of_two_scale=False):
    """(source, float32 device scales or a placeholder, 4 host float32 values) for the scales argument: None (computed
    on the GPU; power_of_two_scale rounds each up to a power of two), a device tensor of n scales (float32; another
    dtype is converted once), a Python float (one tensor) or a sequence of Python floats."""
    if power_of_two_scale and scales is not None:
        raise ValueError("power_of_two_scale applies to computed scales (scales=None)")
    if scales is None:
        return ("computed_pow2" if power_of_two_scale else "computed"), torch.empty(n, dtype=torch.float32, device=dev), \
            [1.0] * 4
    if isinstance(scales, torch.Tensor):
        assert scales.numel() == n and scales.is_cuda, (scales.shape, scales.device)
        sc = scales.reshape(n)
        sc = (sc if sc.dtype == torch.float32 else sc.float()).contiguous()   # (float64 accepted: converted once)
        return "device", sc, [1.0] * 4
    if isinstance(scales, (int, float)):
        scales = [scales]
    host = [float(np.float32(v)) for v in scales]   # the kernel's Float32 arguments (rounded to float32 once)
    assert len(host) == n, (host, n)
    return "host", _dummy(dev, torch.float32), host + [1.0] * (4 - n)


def per_bool(v, n):
    if v is None or isinstance(v, bool):
        return (bool(v),) * n
    return tuple(bool(e) for e in v)


def _per_tensor_flat(ts, single, scales, fp8_format, output_codes, single_launch, amax_only=False,
                     power_of_two_scale=False):
    """quantize_fp8_per_tensor for tensors of any shape (elementwise; the flat kernel). amax_only: just reduce the amaxes
    into the workspace (for a rows-kernel quantisation pass that follows and publishes the scales)."""
    n = len(ts)
    dev = ts[0].device
    want = per_bool(True if output_codes is None else output_codes, n)
    if scales is not None and not all(want):
        raise ValueError("output_codes=False only makes sense when the scales are computed (scales=None)")
    flats = [x.contiguous().view(-1) for x in ts]
    stores = [torch.empty((x.numel() + 3) // 4 * 4, dtype=torch.uint8, device=dev) if w else None
              for x, w in zip(flats, want)]
    source, sc_out, host = _normalise_scales(scales, n, dev, power_of_two_scale)
    chunk = _FlatPerTensorKernel.CHUNK
    counts = [(f.numel() + chunk - 1) // chunk for f in flats]
    offs, qoffs = [0], [0]
    for c, w in zip(counts, want):
        offs.append(offs[-1] + c)
        qoffs.append(qoffs[-1] + (c if w else 0))
    total, qtotal = offs[-1], qoffs[-1]
    offs = offs[1:] + [total] * (4 - n)
    qoffs = qoffs[1:] + [qtotal] * (4 - n)
    pad = flats + [flats[0]] * (4 - n)
    qd = _dummy(dev)
    qpad = [st.view(torch.int32) if st is not None else qd for st in stores] + [qd] * (4 - n)
    ws = _workspace(dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (*pad, *qpad, sc_out, ws, *host, offs[0], offs[1], offs[2], total, qoffs[0], qoffs[1], qoffs[2], qtotal)

    def launch(mode, grid_x):
        key = ("flat", n, tuple(x.dtype for x in pad), fp8_format, tuple(want) + (False,) * (4 - n), source, mode)
        if key not in _PT_CACHE:
            kern = _FlatPerTensorKernel(n, tuple(_cute_dtype(x.dtype) for x in pad), fp8_format,
                                        tuple(want) + (False,) * (4 - n), source, mode)
            cargs = ([to_cute_tensor(x, assumed_align=16) for x in pad] + [to_cute_tensor(q, assumed_align=16) for q in qpad]
                     + [to_cute_tensor(sc_out, assumed_align=8), to_cute_tensor(ws, assumed_align=16)])
            _PT_CACHE[key] = cute.compile(kern, *cargs, Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0),
                                          Int32(0), Int32(0), Int32(0), Int32(1), Int32(0), Int32(0), Int32(0),
                                          Int32(1), Int32(1), stream, options=_COMPILE_OPTIONS)
        _PT_CACHE[key](*args, grid_x, stream)

    if amax_only:
        launch("amax", min(total, 8 * _sm_count(dev)))
        return None
    if source.startswith("computed") and single_launch:
        launch("fused", min(total, 4 * _sm_count(dev)))
    elif source.startswith("computed"):
        launch("amax", min(total, 8 * _sm_count(dev)))   # grid-stride: few CTAs, few same-address atomics
        launch("quant", max(min(qtotal, 8 * _sm_count(dev)), 1))   # one scale read / epilogue per CTA
    else:
        launch("quant", max(min(qtotal, 8 * _sm_count(dev)), 1))
    codes = [st[:x.numel()].view(TORCH_FP8[fp8_format]).view(x.shape) if st is not None else None
             for st, x in zip(stores, ts)]
    return dict(codes=codes[0] if single else codes,
                scales=sc_out if source != "host" else tuple(host[:n]))


def _sm_count(device) -> int:
    key = str(device)
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[key]


def quantize_fp8_per_tensor(
    tensors,
    *,
    scales=None,
    fp8_format: str = "e4m3",
    apply_hadamard_rotation=False,
    hadamard_sign_seed: int = DEFAULT_HADAMARD_SIGN_SEED,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_applies_to=None,
    row_divisors=None,
    output_row_norm2_max=False,
    compute_d_with_output: Optional[torch.Tensor] = None,
    shifted_log2_lse_from: Optional[torch.Tensor] = None,
    log2_lse_shift: float = 0.0,
    zero_accumulator: Optional[torch.Tensor] = None,
    ds_bound_from=None,
    output_codes=True,
    output_scale_scalars=None,
    single_launch: bool = False,
    power_of_two_scale: bool = False,
    hadamard_rotation_method: Optional[str] = None,
    documents: Optional[dict] = None,
) -> dict:
    """Per-tensor FP8: codes_i = FP8(x_i / s_i) (clamp to +-FP8_MAX, round to nearest even) for up to four tensors x_i
    in one launch per pass: (b_i, s_i, h_i, 64 / 128 / 256) bf16 / fp16 / fp32 with a contiguous last dim (GQA shapes
    may differ; any s) for the row options, any shape when contiguous for plain quantisation (the flat kernel).

    scales: None -> s_i = max(amax_i / FP8_MAX, 1e-12) in float32 (div.rn) from the amax of the (transformed) tensor, reduced on the
            GPU (no host read, ever): by default in two launches (amax, then quantise -- measured faster than the single
            launch at A1 / A3: 21.6 vs 27.2 us for one 64 MB tensor); single_launch=True does it in ONE launch (a
            persistent grid, resident by construction, with a grid-wide barrier between the amax and the quantisation).
            A float32 device tensor (n,) (float64 is accepted and converted once) or a sequence of Python floats
            (rounded to float32): given scales (one launch).
    fp8_format: "e4m3" | "e5m2".
    power_of_two_scale: computed scales (scales=None) rounded UP to a power of two, s_i = 2^ceil(log2(max(amax_i /
            FP8_MAX, 1e-12))) (compile-time): no code saturates, a power-of-two scale divides exactly, and a change of
            the amax that stays inside its power-of-two binade changes no code of any token (the per-tensor
            causal channel is then exactly 0 unless a later token moves the amax across a binade boundary).
    apply_hadamard_rotation: bool or a per-tensor tuple: quantise (x * signs) H / sqrt(128) (the amax is the rotated one).
    hadamard_rotation_method: "tensor_core" | "butterfly" | None | "tcgen05" (refused here: the per-tensor kernel has
            no tcgen05 form; quant._rotations.resolve_rotation_method) (quant._rotations): the rotation of the lane-pair rows
            on mma.sync (equal to the butterfly up to the fp32 summation order; the row norms then take an order of their
            own) or the register butterfly (bitwise the earlier kernels). None = tensor_core, except when only the amax of
            the rotated tensors is wanted (output_codes=False: the butterfly measured faster there). The butterfly runs,
            logged, for fp32 rows and for a rotated tensor 0 with the backward-prologue outputs (16-lane row layout).
    rope_cos / rope_sin (+ rope_applies_to, per-tensor bools, default all): RoPE before the rotation.
    row_divisors: per-tensor (b, h, s) fp32 power-of-two row factors or None: x / f_row rounded through the input dtype
            (the all-FP4 route's Q / F_q), before everything else.
    output_row_norm2_max: bool or per-tensor tuple: max over rows of ||x_i||^2 (fp32, the old kernels' sum order).
    output_scale_scalars: up to three (tensor index or None, multiplier, divisor): fp32 scalars (multiplier * s_i) /
            divisor (multiplier / divisor for None), float32 (mul.rn, then div.rn), written by the same launch
            (the epilogue scalars a backward kernel reads from the device, e.g. softmax_scale s_Q and s_dO / 448).
    output_codes: bool or per-tensor tuple (default True): False computes only that tensor's scale (the per-tensor
            E4M3 scale of a tensor another quantiser encodes, e.g. the NVFP4 Q / K with their fused E4M3 copies).
    Backward-prologue side outputs (single tensor: dO):
      compute_d_with_output=O       D (b, h, s) fp32 = sum_d O * (code * float32(s)), the dP GEMM's own operand
      shifted_log2_lse_from=lse     lse_log2 = where(lse == -inf, 0, lse * log2 e) - log2_lse_shift
      zero_accumulator=acc          a contiguous f16 tensor of b * h * s * 128 elements, zeroed in the same launch
      ds_bound_from=(pmax, vnorm2)  the dS scale bound max_q min(P_max_q, 1)(||dO_q|| sqrt(vnorm2) + |D_q|) (needs D)

    documents (aligned varlen documents, lp_varlen.LPDocs.doc_quant_args): one scale PER (DOCUMENT, KV HEAD) of every
            tensor instead of one per tensor -- s_i[d, h] = max(amax / FP8_MAX, 1e-12) over the real rows of document d
            at kv head h (the q heads of a GQA group share their kv head's), computed scales only, (1, total, h, d)
            tensors. Each document's padding rows (at or past its true length) are read as zero rows: they reach no
            scale, code, row norm or D, and the shifted LSE is +inf there. The same launches as the per-tensor form (the
            amax pass red.max-es each tile into its (document, kv head) slot of a persistent zero workspace, the quantise
            pass reads it back; the last CTA publishes the tables and re-zeroes the slots). dict keys: side (per tensor
            "q" / "k": which 128-row tile tables index it), row_end_q / row_end_k, tile_doc_q / tile_doc_k ((tiles,)
            int32: the end of the real rows / the document of every 128-row tile, the sentinel past the last real
            row), n_rows (documents + the sentinel), h_kv, derived (up to two (a, b, multiplier): float32((s_a m) s_b)
            per (document, kv head); b = -1 for float32(s_a m)), fold ((n_rows, h_kv) float32 s_V table, c[, kind]) or
            None: D'' = float32(m) D written in place of D (the backward's D fold), kind "fp8" (default; the 8-bit
            control's m = -c / (s_V s_dO)) or "dk8" (the 4-bit route's m = -((s_V s_dO / c) / 1) / (s_V s_dO), c =
            2^shift).
            Outputs: doc_scales (n, n_rows, h_kv) float32, doc_derived (len(derived), n_rows, h_kv) float32; "scales"
            stays the per-tensor max over the documents (the global scale).
            batched=True (key): (b, s, h, d) tensors with every batch row a document (lowprec.prequant_fa3's batched
            training path): the tables are per 128-row tile of the flattened (b s) rows, the row ends LOCAL to the batch
            row (s for a full row); D / the shifted LSE keep their (b, h, s) layout.

    Returns dict(codes: list of fp8 tensors (a tensor when one was given), scales: (n,) float32 device tensor (or the
    given Python floats), row_norm2_max: (n,) fp32 device tensor, D, lse_log2, ds_bound: (1,) fp32)."""
    from flash_attn.cute import lp_docmaps as _DM   # (a registered lp_docmaps carrier is filled first)
    if _DM._LAZY:
        for t_ in (tensors if isinstance(tensors, (list, tuple)) else (tensors,)):
            _DM.materialise(t_)
    single = isinstance(tensors, torch.Tensor)
    ts = [tensors] if single else list(tensors)
    n = len(ts)
    if not 1 <= n <= 4:
        raise ValueError("quantize_fp8_per_tensor takes 1 to 4 tensors")
    if fp8_format not in FP8_MAX:
        raise ValueError(f"fp8_format must be 'e4m3' or 'e5m2', got {fp8_format!r}")
    dev = ts[0].device
    for x in ts:
        assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), x.dtype
    hd = ts[0].shape[-1] if ts[0].dim() == 4 else 0
    tile_ok = all(x.dim() == 4 and x.stride(-1) == 1 and x.shape[-1] == hd for x in ts) and hd in (64, 128, 256) \
        and all(x.stride(0) % 8 == 0 and x.stride(1) % 8 == 0 and x.stride(2) % 8 == 0 for x in ts)
    row_opts = (apply_hadamard_rotation not in (False, None) and any(per_bool(apply_hadamard_rotation, n))) or \
        rope_cos is not None or row_divisors is not None or any(per_bool(output_row_norm2_max, n)) or \
        compute_d_with_output is not None or shifted_log2_lse_from is not None or zero_accumulator is not None \
        or ds_bound_from is not None or bool(output_scale_scalars)
    _pk_docs = None   # (lp_docmaps compact maps) the tensors are the caller's packed rows: (vt_tile_q, vt_tile_k, ntiles)
    if documents is not None and documents.get("packed") is not None:
        _pk_docs = tuple(documents["packed"])
        assert len(_pk_docs) == 3 and _pk_docs[2].dtype == torch.int32 and _pk_docs[2].numel() == 2
    if documents is not None:
        _bdocs = bool(documents.get("batched", False))   # batched documents: (b, s, h, d), one per batch row
        if not tile_ok or scales is not None or ds_bound_from is not None or single_launch or rope_cos is not None \
                or any((x.shape[0] != 1 and not _bdocs) or (x.shape[1] % 128 and _pk_docs is None) for x in ts):
            raise ValueError("quantize_fp8_per_tensor(documents=...): (1, total, h, 64 / 128 / 256) tensors, total % 128 "
                             "== 0, computed scales, no dS bound / RoPE / single launch")
        row_opts = True   # the tile kernel (the flat kernel has no rows)
    # plain quantisation is elementwise: the flat kernel (coalesced, no staging) whenever the tensors are contiguous
    if not row_opts and (not tile_ok or all(x.is_contiguous() for x in ts)):
        return _per_tensor_flat(ts, single, scales, fp8_format, output_codes, single_launch,
                                power_of_two_scale=power_of_two_scale)
    if not tile_ok:
        if row_opts:
            raise ValueError("quantize_fp8_per_tensor: the row options (Hadamard, RoPE, row divisors, row norms, D, LSE, "
                             "accumulator, dS bound, scale scalars) need (b, s, h, 64 / 128 / 256) tensors with "
                             f"s % 128 == 0 and a contiguous last dim; got {[tuple(x.shape) for x in ts]}")
        return _per_tensor_flat(ts, single, scales, fp8_format, output_codes, single_launch,
                                power_of_two_scale=power_of_two_scale)

    def per(v, default):
        if v is None or isinstance(v, bool):
            return tuple([bool(v) if v is not None else default] * n)
        v = tuple(v)
        assert len(v) == n, (v, n)
        return v

    rot = per(apply_hadamard_rotation, False)
    if (any(rot) or rope_cos is not None) and hd != 128:
        raise ValueError(f"quantize_fp8_per_tensor: RoPE and the Hadamard rotation are built for head_dim 128, got {hd}")
    signs = hadamard_signs_tuple(hadamard_sign_seed) if any(rot) else None
    rot = tuple(signs if r else None for r in rot) + (None,) * (4 - n)
    rope = rope_cos is not None
    rope_on = per(rope_applies_to, True) if rope else (False,) * n
    if rope:
        assert rope_sin is not None and rope_cos.shape == rope_sin.shape
        assert rope_cos.dtype == rope_sin.dtype == torch.float32 and rope_cos.is_contiguous() and rope_sin.is_contiguous()
        for x, on in zip(ts, rope_on):
            if on:
                assert rope_cos.shape in ((x.shape[1], 64), (64, x.shape[1])), (rope_cos.shape, x.shape)
    rope_t = rope and rope_cos.shape[0] == 64 and rope_cos.shape[1] != 64
    divs = list(row_divisors) if row_divisors is not None else [None] * n
    assert len(divs) == n
    norms = per(output_row_norm2_max, False)
    want_codes = per(output_codes, True)
    if scales is not None and not all(want_codes):
        raise ValueError("output_codes=False only makes sense when the scales are computed (scales=None)")
    for i, (wc, nm) in enumerate(zip(want_codes, norms)):
        if nm and not wc:
            raise ValueError("output_row_norm2_max needs the tensor's codes pass (output_codes=True)")
    side = (compute_d_with_output is not None or shifted_log2_lse_from is not None or zero_accumulator is not None
            or ds_bound_from is not None)
    if side and n != 1:
        raise ValueError("the backward-prologue side outputs (D, shifted LSE, zeroed accumulator, dS bound) take one tensor")
    if ds_bound_from is not None and compute_d_with_output is None:
        raise ValueError("ds_bound_from needs compute_d_with_output (the bound uses |D_q|)")
    x0 = ts[0]
    b0, s0, h0, _ = x0.shape
    out = {}
    codes = [torch.empty(x.shape, dtype=TORCH_FP8[fp8_format], device=dev) if wc else None
             for x, wc in zip(ts, want_codes)]
    dmy = _dummy(dev, torch.float32)
    source, sc_out, host = _normalise_scales(scales, n, dev, power_of_two_scale)
    ws = _workspace(dev)
    nrm_out = torch.empty(n, dtype=torch.float32, device=dev) if any(norms) else dmy
    out_dt = None
    # (packed rows: D / the shifted LSE are the dense per-row tables of the packed backward, row stride
    # dense_row_stride(T); only a document's real rows are written -- the backward fills the rest per tile)
    s_out = dense_row_stride(s0) if _pk_docs is not None else s0
    if compute_d_with_output is not None:
        o = compute_d_with_output
        assert o.shape == x0.shape and o.stride(-1) == 1, (o.shape, o.stride())
        dsum = torch.empty(b0, h0, s_out, dtype=torch.float32, device=dev)
        out["D"] = dsum
        out_dt = o.dtype
    else:
        o, dsum = x0, dmy
    if shifted_log2_lse_from is not None:
        lse = shifted_log2_lse_from
        assert lse.shape == (b0, h0, s0) and lse.dtype == torch.float32 and lse.is_contiguous()
        lse_out = torch.empty(b0, h0, s_out, dtype=torch.float32, device=dev)
        out["lse_log2"] = lse_out
    else:
        lse, lse_out = dmy, dmy
    if zero_accumulator is not None:
        acc = zero_accumulator
        assert acc.dtype == torch.float16 and acc.is_contiguous() and acc.numel() == b0 * h0 * s0 * x0.shape[-1]
        acc_w = acc.view(torch.int32).view(-1)
    else:
        acc_w = _dummy(dev)
    if ds_bound_from is not None:
        pmax, vn2 = ds_bound_from
        assert pmax.shape == (b0, h0, s0) and pmax.dtype == torch.float32 and pmax.is_contiguous()
        assert vn2.dtype == torch.float32 and vn2.is_cuda
        bound_out = torch.empty(1, dtype=torch.float32, device=dev)
        out["ds_bound"] = bound_out
    else:
        pmax, vn2, bound_out = dmy, dmy, dmy
    cos, sin = (rope_cos, rope_sin) if rope else (dmy.view(1, 4), dmy.view(1, 4))
    derived = tuple(output_scale_scalars or ())
    assert len(derived) <= 3, "output_scale_scalars: at most three"
    d_idx = tuple(-1 if e[0] is None else int(e[0]) for e in derived)
    d_mul = [float(np.float32(e[1])) for e in derived] + [1.0] * (3 - len(derived))   # Float32 arguments
    d_div = [float(np.float32(e[2])) for e in derived] + [1.0] * (3 - len(derived))
    derived_out = torch.empty(max(len(derived), 1), dtype=torch.float32, device=dev) if derived else dmy
    rows_tile = _PT_ITERS * (128 // (hd // 8))   # _PerTensorKernel.rows_tile; tiles = (b, h, token block)
    counts = [x.shape[0] * x.shape[2] * ((x.shape[1] + rows_tile - 1) // rows_tile) for x in ts]
    if _pk_docs is not None:   # (packed: the static tile bound -- the kernel takes the device counts; grid sizing only)
        _sd = tuple(documents["side"])
        counts = [x.shape[2] * _pk_docs[0 if sd_ == "q" else 1].shape[1] * (128 // rows_tile) for x, sd_ in zip(ts, _sd)]
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    total = offs[-1]
    offs = offs[1:] + [total] * (4 - n)   # off1, off2, off3, total
    qoffs = [0]
    for c, wc in zip(counts, want_codes):
        qoffs.append(qoffs[-1] + (c if wc else 0))
    qtotal = qoffs[-1]
    qoffs = qoffs[1:] + [qtotal] * (4 - n)
    pad = ts + [ts[0]] * (4 - n)
    fpad = [(f.reshape(-1) if f is not None else dmy) for f in divs] + [dmy] * (4 - n)
    for f, x in zip(divs, ts):
        if f is not None:
            assert f.dtype == torch.float32 and f.numel() == x.shape[0] * x.shape[2] * x.shape[1] and f.is_contiguous()
    qdummy = _dummy(dev).view(1, 1, 1, 4)
    qpad = [c.view(torch.int32) if c is not None else qdummy for c in codes] + [qdummy] * (4 - n)
    in_dts = tuple(_cute_dtype(x.dtype) for x in pad)
    common = (tuple(rot), tuple(rope_on) + (False,) * (4 - n), rope_t, tuple(f is not None for f in divs) + (False,) * (4 - n),
              tuple(norms) + (False,) * (4 - n), tuple(want_codes) + (False,) * (4 - n))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (*pad, *fpad, *qpad, sc_out, ws, nrm_out, o, dsum.view(-1), lse.view(-1), lse_out.view(-1), acc_w,
            pmax.view(-1), vn2.reshape(-1), bound_out, cos, sin, derived_out, *host, offs[0], offs[1], offs[2], total,
            qoffs[0], qoffs[1], qoffs[2], qtotal, float(log2_lse_shift), *d_mul, *d_div)

    # the rotation method per tensor (the lane-pair rows: rotated, head dim 128, not tensor 0 with the prologue outputs)
    tc = [False] * 4
    if any(r is not None for r in rot):
        prologue = (out_dt is not None or shifted_log2_lse_from is not None or zero_accumulator is not None
                    or ds_bound_from is not None)
        whys = []
        for i_ in range(n):
            if rot[i_] is None:
                continue
            if any(x.dtype == torch.float32 for x in ts):
                whys.append(f"tensor {i_}: an fp32 tensor in the launch (the mma.sync operands are 16-bit, and the "
                            "staging slots are sized for the widest input)")
            elif i_ == 0 and prologue:
                whys.append("tensor 0 with D / LSE / accumulator / dS bound: its 16-lane row layout")
            else:
                tc[i_] = True
        slower = None
        if not any(want_codes[i_] for i_ in range(n) if rot[i_] is not None):
            slower = ("only the amax of the rotated tensors (output_codes=False): the amax launch measured slower on the "
                      "tensor cores (Q + K at A1: 42.2 us butterfly vs 49.0 us)")
        method = resolve_rotation_method(hadamard_rotation_method, "quantize_fp8_per_tensor",
                                         "; ".join(whys) if (whys and not any(tc)) else None, slower,
                                         tcgen05_unsupported=TCGEN05_PER_TENSOR)
        if method != "tensor_core":
            tc = [False] * 4
        elif whys:   # some tensors on the tensor cores, the others (logged) on the butterfly
            resolve_rotation_method(hadamard_rotation_method, "quantize_fp8_per_tensor (partial)", "; ".join(whys))
    elif hadamard_rotation_method is not None:
        resolve_rotation_method(hadamard_rotation_method, "quantize_fp8_per_tensor",
                                tcgen05_unsupported=None)   # validates the name (nothing is rotated)
    tc = tuple(tc)
    dspec, dargs = None, ()
    if documents is not None:
        dk = documents
        side = tuple(dk["side"])
        assert len(side) == n and all(x in ("q", "k") for x in side), side
        nr, hkv = int(dk["n_rows"]), int(dk["h_kv"])
        for x in ts:
            assert x.shape[2] % hkv == 0, (x.shape, hkv)
        der = tuple((int(a_), int(b_), float(m_)) for a_, b_, m_ in (dk.get("derived") or ()))
        assert len(der) <= 2
        fold = dk.get("fold")
        if fold is not None:
            assert out_dt is not None and n == 1, "documents fold: the D fold of the dO launch (compute_d_with_output)"
        dspec = dict(side=side, group=tuple(int(x.shape[2] // hkv) for x in ts), derived=der,
                     fold=None if fold is None else
                     ((str(fold[2]), float(fold[1])) if len(fold) > 2 else ("fp8", float(fold[1]))),
                     pow2=bool(power_of_two_scale))
        if dk.get("batched", False):
            dspec["batched"] = True   # (only when set: the aligned varlen keys and code stay as they were)
        if _pk_docs is not None:
            dspec["packed"] = True    # (only when set, as batched)
        amax_ws = _doc_workspace(dev, n * nr * hkv).view(n, nr, hkv)
        doc_scales = torch.empty(n, nr, hkv, dtype=torch.float32, device=dev)
        doc_der = torch.empty(max(len(der), 1), nr, hkv, dtype=torch.float32, device=dev)
        doc_q = torch.empty(n * nr * hkv * 4, dtype=torch.float32, device=dev)   # (1 / s, s, D fold, -) per slot
        fold_t = fold[0].reshape(-1) if fold is not None else doc_scales.view(-1)[:nr * hkv]  # (unread without a fold)
        assert fold_t.dtype == torch.float32 and fold_t.numel() == nr * hkv and fold_t.is_contiguous()
        dargs = (dk["row_end_q"].reshape(-1), dk["row_end_k"].reshape(-1), dk["tile_doc_q"].reshape(-1),
                 dk["tile_doc_k"].reshape(-1), amax_ws, doc_scales.view(-1), doc_der.view(-1), fold_t, doc_q)
        for t_ in dargs[:4]:
            assert t_.dtype == torch.int32 and t_.is_contiguous()
        if _pk_docs is not None:
            dargs = dargs + _pk_docs
        out["doc_scales"] = doc_scales
        if der:
            out["doc_derived"] = doc_der[:len(der)]

    def launch(mode, grid_x):
        key = (n, tuple(x.dtype for x in pad), fp8_format, common, source, out_dt, shifted_log2_lse_from is not None,
               zero_accumulator is not None, ds_bound_from is not None, mode, hd, d_idx, tc)
        if dspec is not None:
            key = key + (("docs",) + tuple(sorted((k_, str(v_)) for k_, v_ in dspec.items())),)
        if key not in _PT_CACHE:
            kern = _PerTensorKernel(n, in_dts, fp8_format, *common, source, _cute_dtype(out_dt) if out_dt else None,
                                    shifted_log2_lse_from is not None, zero_accumulator is not None,
                                    ds_bound_from is not None, mode, hd, d_idx, docs=dspec)
            kern.tc = tuple(t_ and kern.pair[i_] for i_, t_ in enumerate(tc))
            cargs = ([to_cute_tensor(x, divisibility=8) for x in pad] + [to_cute_tensor(f, assumed_align=4) for f in fpad]
                     + [to_cute_tensor(q, divisibility=4) for q in qpad]
                     + [to_cute_tensor(sc_out, assumed_align=8), to_cute_tensor(ws, assumed_align=16),
                        to_cute_tensor(nrm_out, assumed_align=4), to_cute_tensor(o, divisibility=8)]
                     + [to_cute_tensor(t_, assumed_align=4) for t_ in (dsum.view(-1), lse.view(-1), lse_out.view(-1))]
                     + [to_cute_tensor(acc_w, assumed_align=16)]
                     + [to_cute_tensor(t_, assumed_align=4) for t_ in (pmax.view(-1), vn2.reshape(-1), bound_out, cos, sin,
                                                                       derived_out)])
            dcargs = ()
            if dspec is not None:
                dcargs = tuple([to_cute_tensor(t_, assumed_align=4, leading_dim=0) for t_ in dargs[:4]]
                               + [to_cute_tensor(dargs[4], assumed_align=4, leading_dim=2)]
                               + [to_cute_tensor(t_, assumed_align=t_.element_size(), leading_dim=0)
                                  for t_ in dargs[5:9]])   # (float32 tables: 4-byte aligned)
                if _pk_docs is not None:   # (packed: the tile tables (2, tiles) and the device tile counts (2,))
                    dcargs = dcargs + (to_cute_tensor(dargs[9], assumed_align=4), to_cute_tensor(dargs[10], assumed_align=4),
                                       to_cute_tensor(dargs[11], assumed_align=4, leading_dim=0))
            _PT_CACHE[key] = cute.compile(kern, *cargs, Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0),
                                          Int32(0), Int32(0), Int32(0), Int32(1), Int32(0), Int32(0), Int32(0),
                                          Int32(1), Float32(0.0), *(Float32(1.0),) * 6, Int32(1), stream, *dcargs,
                                          options=_COMPILE_OPTIONS)
            # resident CTAs per SM: minctasm bounds the registers, shared memory (227 KB, 1 KB reserved per CTA) the rest
            _PT_CTAS[key] = max(1, min(kern.ctas_per_sm, (227 * 1024) // (kern.smem_bytes + 1024)))
        # persistent grid of resident CTAs: per-CTA reductions, atomics and the last-CTA epilogue once per CTA
        _PT_CACHE[key](*args, max(1, min(grid_x, _PT_CTAS[key] * _sm_count(dev))), stream, *dargs)

    if source.startswith("computed") and single_launch:
        launch("fused", total)
    elif source.startswith("computed"):
        if not any(r is not None for r in rot) and not rope and all(f is None for f in divs) \
                and all(x.is_contiguous() for x in ts) and documents is None:
            # untransformed rows: the amax is elementwise -- the flat kernel's amax launch (same workspace)
            _per_tensor_flat(ts, single, None, fp8_format, False, False, amax_only=True)
        else:
            launch("amax", total)
        launch("quant", qtotal)   # >= 1 CTA: its last-CTA epilogue publishes the scales
    else:
        launch("quant", qtotal)
    out["codes"] = codes[0] if single else codes   # None for a tensor with output_codes=False
    out["scales"] = sc_out if source != "host" else tuple(host[:n])   # given Python floats come back as given
    if any(norms):
        out["row_norm2_max"] = nrm_out
    if derived:
        out["scale_scalars"] = derived_out
    return out

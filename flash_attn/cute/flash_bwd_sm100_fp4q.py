# All-FP4 attention backward on Blackwell (SM100), query-row compute.
# Derived from flash_bwd_sm100_fp4.py (the C3 NVFP4 backward). Every GEMM is an NVFP4 block-scaled tcgen05 MMA:
#   S  = Q^ K^T      (A = Q4 row-wise, staged per query tile; B = K4 row-wise)        -> TMEM, lanes = query rows
#   dP = dO^ V^T     (A = dO4 row-wise, staged; B = V4 = the row-wise copy of the forward's 16x16 V^)
#   dQ = dS^ K^T     (A = dS^ E2M1 in TMEM (query rows, keys in the C3 32-block order); B = K^T per d-half)
#   dK += dS^^T Q^T  (A = dS^^T E2M1 in SMEM (key rows), SFs S2T; B = Q^T staged)
#   dV += P^v^T dO^T (A = P^v^T E2M1 in SMEM, SFs S2T; B = dO^T staged)
# Work tiles are key-stationary (dK / dV accumulate in TMEM and leave through the epilogue, dQ is reduced per step), but
# every compute thread holds one QUERY row: it reproduces the forward's Direct-P P^ exactly (same S, same 16-key group maxes,
# the forward's classifier constants, the exported final integer frame F) and computes the straight-through gradient of the
# forward O = sum_k P^_k V_k / Z:   dS_k = w_k dP_k - D m_k,  w = P^ / Z (E2M1 codes x E4M3 SFs), m = Mitchell / Z (the forward's
# row sum Z = sum Mitchell x exact SF, so sum_k m_k = 1 and dS rows sum to 0). Per query only (F, 1/Z) and D are read.
# dS^ (dQ and dK) and P^v (dV) use 16x16 block scales (16 queries x 16 keys): in-thread 16-key maxes, a butterfly max over the
# 16 query threads, E4M3 scales, RTN or stochastic rounding (dS^). The SMEM A operands pair adjacent queries in one byte (4-bit
# operands must be K-major): even threads merge the partner's code words (one butterfly shuffle) and store the bytes.
import math
import os
from typing import Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr, Float8E4M3FN, Float4E2M1FN, Float8E8M0FNU
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cutlass_dsl import T as _T, dsl_user_op as _dsl_user_op
from cutlass._mlir.dialects import llvm as _llvm

from quack import layout_utils
from flash_attn.cute import utils
from flash_attn.cute.lowprec.specs import BwdQSpec
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.blackwell_helpers import (
    gemm_w_idx,
    make_smem_layout_sfa,
    make_smem_layout_sfb,
)
from flash_attn.cute.flash_fwd_sm100_fp4 import (directp_window_consts, mitchell_value_w, DIRECTP_DELTA, DIRECTP_WINDOW,
                                                E2E_POLY_E2M1_PV, SPLIT_MASKED_GM, FP4_PV_RESCALE_THRESHOLD)
from flash_attn.cute.pv4_repro import fma_rn
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, SingleTileLPTBwdScheduler
from flash_attn.cute.tile_scheduler import PackedOverflowScheduler, FixedTileScheduler
from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100

# FP4_PV_RESCALE_THRESHOLD (imported above from the forward, re-exported here): the all-4-bit forward's lazy P-frame rescale
# threshold (log2 units). The forward's frame bias max_offset = log2 6 + log2 448 - this, which the backward's P^
# reproduction (self.max_offset) and the host's LSE offset (interface_bwd_fp4q.LSE_LOG2_OFFSET) must match: one module
# constant, so the forward, this kernel and the driver cannot disagree.


def _e2m1_bytes_to_f32x4(w, shift: int, loc=None, ip=None):
    """bytes shift / 8 and shift / 8 + 1 of w (2 E2M1 codes each, low nibble first) -> 4 FP32 values."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 t0, t1, p0, p1;\n\t"
        ".reg .b8 c0, c1;\n\t"
        ".reg .b16 h0, h1, h2, h3;\n\t"
        f"shr.b32 t0, $4, {shift};\n\t"
        "cvt.u8.u32 c0, t0;\n\t"
        "shr.b32 t1, t0, 8;\n\t"
        "cvt.u8.u32 c1, t1;\n\t"
        "cvt.rn.f16x2.e2m1x2 p0, c0;\n\t"
        "cvt.rn.f16x2.e2m1x2 p1, c1;\n\t"
        "cvt.u16.u32 h0, p0;\n\t"
        "shr.b32 p0, p0, 16;\n\t"
        "cvt.u16.u32 h1, p0;\n\t"
        "cvt.u16.u32 h2, p1;\n\t"
        "shr.b32 p1, p1, 16;\n\t"
        "cvt.u16.u32 h3, p1;\n\t"
        "cvt.f32.f16 $0, h0;\n\t"
        "cvt.f32.f16 $1, h1;\n\t"
        "cvt.f32.f16 $2, h2;\n\t"
        "cvt.f32.f16 $3, h3;\n\t"
        "}\n",
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@_dsl_user_op
def packed_e2m1_to_float8(w: Int32, *, loc=None, ip=None):
    """8 E2M1 codes packed in w (byte i = codes 2i (low nibble), 2i + 1) -> 8 FP32 values (e2m1x2 -> f16x2 -> f32)."""
    return _e2m1_bytes_to_f32x4(w, 0, loc=loc, ip=ip) + _e2m1_bytes_to_f32x4(w, 16, loc=loc, ip=ip)


@_dsl_user_op
def packed_float_to_e2m1x4_rs(f0, f1, f2, f3, f4, f5, f6, f7, r0: Int32, r1: Int32, *, loc=None, ip=None) -> Int32:
    """8 FP32 values -> 8 packed E2M1 codes (element 0 in the low nibble) with hardware stochastic rounding: two 4-wide
    cvt.rs (8 random bits per element are added to the discarded mantissa bits; rounding away from zero on carry), random
    words r0 (elements 0-3) and r1 (elements 4-7). Exactly stochastic rounding onto the E2M1 grid at 8-bit dither resolution."""
    out = _llvm.inline_asm(
        _T.i32(),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (f0, f1, f2, f3, f4, f5, f6, f7)]
        + [Int32(r0).ir_value(loc=loc, ip=ip), Int32(r1).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 h0, h1;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 h0, {$4, $3, $2, $1}, $9;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 h1, {$8, $7, $6, $5}, $10;\n\t"
        "mov.b32 $0, {h0, h1};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


@_dsl_user_op
def packed_float_to_e4m3x4_rs(f0, f1, f2, f3, r: Int32, *, loc=None, ip=None) -> Int32:
    """4 FP32 values -> 4 packed E4M3 bytes (element 0 in the low byte) with hardware stochastic rounding: one 4-wide cvt.rs
    (8 random bits per element added to the discarded mantissa bits), random word r."""
    out = _llvm.inline_asm(
        _T.i32(),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (f0, f1, f2, f3)] + [Int32(r).ir_value(loc=loc, ip=ip)],
        "cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        "=r,f,f,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


def _i32_wrap(x: int) -> int:
    """A Python int wrapped to signed 32 bits (two's complement), for products of compile-time hash constants that the
    hash means modulo 2^32. Same value the DSL would truncate to, without its out-of-range literal warning."""
    return ((x + (1 << 31)) % (1 << 32)) - (1 << 31)


@_dsl_user_op
def sr_seed_from_counter(ctr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """The stochastic-rounding seed (c * 0x9E3779B1) mod 2^31 of the step's int64 SR counter c (the host formula: only c's
    low word reaches the low 31 bits). An opaque load (side-effecting asm): an ordinary load placed in the tile loop is
    hoisted back to the kernel entry by the compiler (measured: the GQA kernel's spills grow further)."""
    lo = _llvm.inline_asm(_T.i32(), [ctr.toint(loc=loc, ip=ip).ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l",
                          has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
    return (Int32(lo) * Int32(-1640531535)) & Int32(0x7FFFFFFF)


def e4m3w_to_f16x2(w: Int32, *, loc=None, ip=None):
    """4 E4M3 bytes of w -> two f16x2 registers (bytes 0-1, 2-3), exact."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.i32(), _T.i32()]),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.u16.u32 lo, $2;\n\t"
        "shr.b32 t, $2, 16;\n\t"
        "cvt.u16.u32 hi, t;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(_llvm.extractvalue(_T.i32(), out, [0], loc=loc, ip=ip)), Int32(_llvm.extractvalue(_T.i32(), out, [1], loc=loc, ip=ip))


def f16x2_to_e4m3w(p0: Int32, p1: Int32, *, loc=None, ip=None) -> Int32:
    """inverse of e4m3w_to_f16x2 (exact for E4M3-representable values)."""
    out = _llvm.inline_asm(
        _T.i32(),
        [Int32(p0).ir_value(loc=loc, ip=ip), Int32(p1).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 lo, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 hi, $2;\n\t"
        "mov.b32 $0, {lo, hi};\n\t"
        "}\n",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


def e4m3x2_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """E4M3(a), E4M3(b) (round to nearest) as one exact f16x2 register (a in the low half)."""
    out = _llvm.inline_asm(
        _T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 e;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 e, $2, $1;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, e;\n\t"
        "}\n",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


def f16x2_e4m3_out(p: Int32, *, loc=None, ip=None):
    """exact f16x2 register of E4M3 values -> (the 2 E4M3 bytes as a u32 (low half), value a, value b)."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.i32(), _T.f32(), _T.f32()]),
        [Int32(p).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 e, h0, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e, $3;\n\t"
        "cvt.u32.u16 $0, e;\n\t"
        "mov.b32 {h0, h1}, $3;\n\t"
        "cvt.f32.f16 $1, h0;\n\t"
        "cvt.f32.f16 $2, h1;\n\t"
        "}\n",
        "=r,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return (Int32(_llvm.extractvalue(_T.i32(), out, [0], loc=loc, ip=ip)),
            Float32(_llvm.extractvalue(_T.f32(), out, [1], loc=loc, ip=ip)),
            Float32(_llvm.extractvalue(_T.f32(), out, [2], loc=loc, ip=ip)))


def max_f16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    out = _llvm.inline_asm(
        _T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "{\n\tmax.f16x2 $0, $1, $2;\n\t}\n",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


def pack_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """2 FP32 values -> one 32-bit register of 2 IEEE half floats (a in the low 16 bits), round to nearest."""
    out = _llvm.inline_asm(
        _T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 h0, h1;\n\t"
        "cvt.rn.satfinite.f16.f32 h0, $1;\n\t"
        "cvt.rn.satfinite.f16.f32 h1, $2;\n\t"
        "mov.b32 $0, {h0, h1};\n\t"
        "}\n",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


def unpack_f16x2(w: Int32, *, loc=None, ip=None):
    """inverse of pack_f16x2 -> (a, b) FP32."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 2),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .b16 h0, h1;\n\t"
        "cvt.u16.u32 h0, $2;\n\t"
        "shr.b32 t, $2, 16;\n\t"
        "cvt.u16.u32 h1, t;\n\t"
        "cvt.f32.f16 $0, h0;\n\t"
        "cvt.f32.f16 $1, h1;\n\t"
        "}\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return (Float32(_llvm.extractvalue(_T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(_llvm.extractvalue(_T.f32(), out, [1], loc=loc, ip=ip)))


def merge_nibble_words(own: Int32, par: Int32, *, loc=None, ip=None):
    """own, par: 8 packed E2M1 codes each (code i in nibble i). Returns two 32-bit words holding 8 bytes, byte i =
    own code i (low nibble) | partner code i (high nibble): the nibble spread of both words interleaved."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.i32(), _T.i32()]),
        [Int32(own).ir_value(loc=loc, ip=ip), Int32(par).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b64 x, y, t;\n\t"
        "cvt.u64.u32 x, $2;\n\t"
        "cvt.u64.u32 y, $3;\n\t"
        "shl.b64 t, x, 16;\n\t"
        "or.b64 x, x, t;\n\t"
        "and.b64 x, x, 0x0000FFFF0000FFFF;\n\t"
        "shl.b64 t, x, 8;\n\t"
        "or.b64 x, x, t;\n\t"
        "and.b64 x, x, 0x00FF00FF00FF00FF;\n\t"
        "shl.b64 t, x, 4;\n\t"
        "or.b64 x, x, t;\n\t"
        "and.b64 x, x, 0x0F0F0F0F0F0F0F0F;\n\t"
        "shl.b64 t, y, 16;\n\t"
        "or.b64 y, y, t;\n\t"
        "and.b64 y, y, 0x0000FFFF0000FFFF;\n\t"
        "shl.b64 t, y, 8;\n\t"
        "or.b64 y, y, t;\n\t"
        "and.b64 y, y, 0x00FF00FF00FF00FF;\n\t"
        "shl.b64 t, y, 4;\n\t"
        "or.b64 y, y, t;\n\t"
        "and.b64 y, y, 0x0F0F0F0F0F0F0F0F;\n\t"
        "shl.b64 y, y, 4;\n\t"
        "or.b64 x, x, y;\n\t"
        "mov.b64 {$0, $1}, x;\n\t"
        "}\n",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(_llvm.extractvalue(_T.i32(), out, [0], loc=loc, ip=ip)), Int32(_llvm.extractvalue(_T.i32(), out, [1], loc=loc, ip=ip))


# cov6: the packed dK / dV epilogue's 16-byte row stores (False: cov5's 2-byte element stores, the same values)
PACKED_EPI_VEC = True


class FlashAttentionBackwardSm100FP4Q:
    arch = 100

    def __init__(
        self,
        head_dim: int = 128,
        is_causal: bool = False,
        deterministic: bool = False,
        qhead_per_kvhead: int = 1,
        directp_clamp: int = 2,
        ds_sr: bool = True,
        # exact classifier: the forward's p_mix (flash_fwd_sm100_fp4): 8 characters over M / E / 2 / 3, the value function of each
        # 16-key group position of the 128-key tile; the P^ codes, Z and m are rebuilt with the same function per group
        p_mix: Optional[str] = None,
        dkv: str = "e4m3",
        # diagonal-only-key-local-v-scale: the forward applied the per-key bias / row-sum weight ONLY
        # on the causal masked (diagonal) tiles and read the key-local V payload there, so the backward (a) reproduces
        # the bias on exactly those (m_block, n_block) pairs and nowhere else -- otherwise the two passes compute
        # different P^ -- and (b) reads the DIAGONAL payload as its dP = dO V^T operand on the diagonal pair. (b) is
        # `key-grouped-v-scale-leaks-through-dp-too`: with the key-grouped payload as the dP operand everywhere, dQ_t
        # still depends on the group scale of keys k' > t (1.95e-02 measured, leaky 2.54e-02); with the select, dQ
        # leaks exactly 0. dkv8 (E4M3 dK / dV) backward only.
        vtok_diag: bool = False,
        # diagonal-only, host-proven: every key tile has exactly ONE diagonal pair (MHA and (s_k - s_q) % 128 == 0), so
        # it is statically iteration 0 of the m loop and the MMA warp needs no runtime dP-operand select
        diag_single: bool = False,
        # diagonal-only: the FORWARD's diagonal granularity in 128-query tiles (`diagonal-pairs-are-the-forwards-cta`).
        # The forward classifies a step as masked per CTA, and its CTA spans q_tiles x 128 packed rows = 256 queries
        # for MHA at the default Q_STAGE = 2, so it biases key tile 2j and reads the diagonal payload there for query
        # tiles 2j AND 2j + 1. The backward's 128-query tiles must reproduce that set pair for pair: its diagonal
        # test runs on m rounded down to a multiple of this. 1 = the backward's own tile (GQA packs r >= 2 rows).
        diag_fwd_tiles: int = 1,
        # `k-operand-of-dq-leaks`, diagonal-only (host-proven: MHA, (s_k - s_q) % 128 == 0, one work tile per CTA). The dQ
        # GEMM's B operand K^T carries E4M3 scales per 16 KEYS, so on the causally cut pair a query's dQ reads the scale of
        # keys after it. With diag_k that pair reads a key-local K^T payload instead -- K / 2^e_k with the key tile's
        # constant 2^c as its scale -- and the dQ copy of dS' carries r_k = 2^(e_k - c) there, so dS^ K^T is unchanged in
        # exact arithmetic. No SMEM is added: the payload is loaded into the plain V tile (not read before the first
        # non-diagonal dP, which follows dQ(0)) and plain V is reloaded after dQ(0).
        diag_k: bool = False,
        # `k-operand-of-dq-leaks`, the other form: the key-local K^T IS the dQ operand on every pair of the key tile (the
        # host passes it in K^T's place) and the dQ copy of dS' carries r_k on every pair -- no second tile, no alias, no
        # reload; the compute warps' r_k multiply is the only kernel change.
        diag_kr_all: bool = False,
        # `k-operand-of-dq-leaks` (the K operand of dQ carried a causal leak), folded (QuantSpec.vtok_diag_k = "fold"): r_k rides in operands the host builds -- the dP
        # operand's row SF bytes (V r_k), the per-key u gather ([r | u r]) -- so dS' = r_k dS on every pair with no
        # multiply here; the key-local K^T is the dQ operand on every pair (host), the dV operand's 448 becomes the
        # tile's 448 2^-A (m r <= 2^A), and the host scales the dK / dV rows by 1 / r_k and 2^A / r_k after the kernel
        diag_fold: bool = False,
        # aligned varlen documents (lp_varlen.LPDocs; b = 1, every document at a 128-aligned start): each key tile's
        # query range, mask and diagonal classification in its document's coordinates (lp_bwd_tiles.LPTileInfo), the
        # per-document scalars [dk_scale, dv_scale, ds_mul] an (n_docs + 1, 3) table (mScl) read by the tile's document
        doc_varlen: bool = False,
        # cov5 (lp_docmaps compact maps, with doc_varlen): the caller's PACKED rows -- every code / E4M3 operand / per-row
        # table / gradient row at the document's packed start (the tiles keep the aligned numbering of the maps), the SF
        # atoms built by the load warp from the dense per-row words, K^T realigned from the key-parity copies, the
        # statistics from 16-byte-aligned windows, padding rows / keys given the aligned layout's values
        codes_packed: bool = False,
        *,
        # the compile-time settings (lowprec.specs.BwdQSpec; the driver builds it): required, so no construction site
        # silently runs the default configuration
        spec: BwdQSpec,
    ):
        assert isinstance(spec, BwdQSpec), spec
        self.spec = spec
        assert head_dim == 128, "all-FP4 backward: head_dim 128 (pad on the host)"
        self.diag_single = bool(diag_single) and bool(vtok_diag) and qhead_per_kvhead == 1
        self.diag_fwd_tiles = int(diag_fwd_tiles) if vtok_diag else 1
        self.diag_kr_all = bool(diag_kr_all) and bool(vtok_diag)
        self.diag_k = bool(diag_k) and bool(vtok_diag) and qhead_per_kvhead == 1 and not self.diag_kr_all
        self.diag_kr = self.diag_k or self.diag_kr_all   # the compute warps multiply r_k into the dQ copy of dS'
        self.diag_fold = bool(diag_fold) and bool(vtok_diag) and not self.diag_kr
        self.doc_varlen = bool(doc_varlen)
        self.codes_packed = bool(codes_packed)
        # (cov6) the packed K^T is the plain NVFP4 1 x 16 copy (SF per 16-key group from quant.fp4's packed group table)
        # unless a key channel route makes it the key-local payload (tile-constant SF words)
        self.pk_kt_groups = self.codes_packed and not (self.diag_fold or self.diag_kr_all)
        assert not self.codes_packed or (self.doc_varlen and not diag_k), \
            "codes_packed: documents (cov6: the causal diagonal route or the dense route)"
        if self.doc_varlen:
            assert dkv == "e4m3" and not self.diag_k and not self.diag_single and not deterministic, (
                "documents: the dkv8 backward, no diag_k / diag_single (per-document offsets), non-deterministic")
        assert self.diag_fwd_tiles >= 1
        self.vtok_diag_arg = bool(vtok_diag)
        self.vtok_dp_diag = bool(vtok_diag)   # the diagonal payload is also the dP operand on the diagonal pair
        self.tile_m = 128
        self.tile_n = 128
        self.tile_hdim = 128
        self.is_causal = is_causal
        self.is_local = False
        self.deterministic = deterministic
        self.qhead_per_kvhead = int(qhead_per_kvhead)
        self.sf_dtype = Float8E4M3FN
        self.sf_vec_size = 16
        self.dq_reduce_dtype = cutlass.Float16
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)
        self.acc_dtype = Float32
        self.mma_inst_bits_k = 256
        self.mma_inst_tile_k = 2  # K = 128 elements -> 2 E2M1 instructions of 64
        # tilers (M, N, K)
        self.mma_tiler_S = (128, 128, 128)      # queries x keys x d
        self.mma_tiler_dP = (128, 128, 128)     # queries x keys x dv
        self.mma_tiler_dQh = (128, 64, 128)     # queries x (d half) x keys
        self.mma_tiler_dK = (128, 128, 128)     # keys x d x queries
        self.mma_tiler_dV = (128, 128, 128)     # keys x dv x queries
        self.num_compute_wgs = 2
        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = tuple(range(4, 12))
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.empty_warp_ids = (14, 15)
        self.threads_per_cta = 16 * cute.arch.WARP_SIZE
        self.epi_barrier_base = int(NamedBarrierBwdSm100.EpilogueWG1)
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute), num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # Direct-P classifier constants: exactly the forward's (its window / delta constants, clamp)
        self.directp_clamp = int(directp_clamp)
        assert self.directp_clamp in (1, 2)
        self.directp_delta = DIRECTP_DELTA
        self.directp_window = DIRECTP_WINDOW
        self.directp_mag, self.directp_lea = directp_window_consts(self.directp_window)
        # BwdQSpec.exp2: reproduce a forward built with FwdKernelSpec.fp4all_exp2 -- the same lean structure (group max,
        # group SF, u-weighted m), but P^ = ex2.approx(s*scale + log2 6 - gm*scale) on the MUFU pipe with one pair in
        # every KernelTuning.fp4_exp2_emulation_every emulated on the FMA pipe, instead of Mitchell's magic-constant bit trick. The two
        # kernels MUST agree: P^, Z and m all change, so a forward and a backward on different settings are inconsistent.
        # The plan sets this and the forward's FwdKernelSpec.fp4all_exp2 from one value, so a step's forward and
        # backward agree on P^ by construction. Overriding one of these without the other makes the step
        # inconsistent.
        self.fp4q_exp2 = bool(spec.exp2)
        _e2e = spec.exp2_emulation_every
        if _e2e is None:
            _e2e = BwdQSpec.rule_exp2_emulation_every()
        self.e2e_freq = int(_e2e)
        self.e2e_poly = E2E_POLY_E2M1_PV   # the forward's polynomial on this (E2M1 P V) route
        from flash_attn.cute.flash_fwd_sm100_fp4 import parse_p_mix
        self.p_mix = parse_p_mix(p_mix)
        if self.p_mix is not None:
            assert not self.fp4q_exp2, "p_mix replaces the single exp2 value function (BwdQSpec.exp2=False)"
            self.mix_kinds = tuple(sorted(set(self.p_mix), key="ME23".index))
            self.mix_table = sum(self.mix_kinds.index(c) << (2 * p) for p, c in enumerate(self.p_mix))
        self.p_mix_key = None if self.p_mix is None else "".join(self.p_mix)
        # the forward's P range: max_offset = log2 6 + log2 448 - rescale threshold (the all-4-bit forward's lazy threshold)
        self.max_offset = math.log2(6) + math.log2(448) - FP4_PV_RESCALE_THRESHOLD
        self.gm_floor = SPLIT_MASKED_GM
        # dS^ stochastic rounding (default on: the emulated training needs it)
        self.ds_sr = bool(ds_sr)
        # dK / dV operands: "e4m3" (per-tensor E4M3 GEMMs, A = E4M3 dS^ / P rows the query threads write contiguously to
        # SMEM, MN-major; dS^ of dQ 1x16 in-thread) or "fp4" (K-major E2M1 SMEM scatter, dS^ / P^v 16x16)
        assert dkv in ("e4m3", "fp4"), dkv
        self.dkv8 = dkv == "e4m3"
        # 8-bit dS for the dQ GEMM: A = the E4M3 codes the dK operand already holds (TMEM), B = per-tensor E4M3 K^ (SMEM),
        # kind::f8f6f4 with no scale factors on either side (read in __init__: __call__ needs it before the MMAs are built)
        self.dq8 = bool(self.dkv8 and spec.dq8)
        # A of dQ from SMEM (the dK operand's own tile), B = K^ K-major. (Other dq8 operand placements -- TMEM A, MN-major
        # B, a generic GEMM -- were probed during development and are not shipped.)
        # SR dither: "weyl" (default: one hash per 16-element group, v_j = frac(u0 + phi_j) - 1/2, exactly uniform per element)
        # or "hash" (an independent hash per element)
        self.sr_dither = spec.sr_dither  # "hw": cvt.rs.e2m1x4 with 16 random words per thread
        # P^v scale boost: w <= 1, E4M3 SF = amax * boost / 6 in [2^-9, 448] for amax >= 2^-20
        self.pv_boost = float(2 ** 11)
        self.Q_stage = 2
        self.dO_stage = 2
        # TMEM map (columns): S [0,128) | dV acc [128,256) | dP [256,384) | dK acc [384,512)
        #   S bank after S loaded:  SFdO (A of dP) [64,72) | SFV (B of dP) [72,80)
        #   dP bank after dP loaded: dQ half acc [256,320) | dS^ codes (A of dQ) [320,336) | SFdS^ [336,344) |
        #                            SFQ (A of S) [344,352) | SFK (B of S) [352,360) -> then SF dS^T / SFQ^T / SF P^v / SFdO^T
        #                            [344,376) | SFK^T half (B of dQ) [376,384)
        self.tmem_alloc_cols = 512
        self.tmem_S_offset = 0
        self.tmem_dV_offset = 128
        self.tmem_dP_offset = 256
        self.tmem_dK_offset = 384
        self.tmem_dQ_offset = 256
        self.tmem_dSq_offset = 320
        self.tmem_SFdSq_offset = 336
        self.tmem_SFdO_offset = 64
        self.tmem_SFV_offset = 72
        # dq8: the E4M3 dS operand spans 320..352, so the S GEMM's SF tiles move into the 352..376 hole
        self.tmem_SFQ_offset = 352 if spec.dq8 else 344
        self.tmem_SFK_offset = 360 if spec.dq8 else 352
        # dK / dV run after dP(t+1) is in TMEM [256, 384): their SFs park in the S bank (S(t+1) loaded, SFdO / SFV consumed)
        # 80/88 are 8- but not 32-aligned; moving these parks (a former debug override) showed that TMEM alignment
        # was NOT the cause of the MXFP8 dS fault (error 716 identically at (64, 96), (96, 64), (32, 96), (128, 160)).
        self.tmem_SFdSt_offset = 80
        self.tmem_SFQt_offset = 88
        self.tmem_SFPv_offset = 96
        self.tmem_SFdOt_offset = 104
        # diagonal-only: the diagonal payload's SFV (B of dP on the diagonal pair) in the S bank's last hole
        self.tmem_SFVb_offset = 112
        self.tmem_SFKt_offset = 376
        self.num_regs_reduce = 80
        self.num_regs_compute = 168
        self.num_regs_load = 88
        self.num_regs_mma = 88
        self.num_regs_empty = 88
        assert 4 * self.num_regs_reduce + 8 * self.num_regs_compute + self.num_regs_load + self.num_regs_mma + 2 * self.num_regs_empty <= 2048
        self.buffer_align_bytes = 1024
        self.dQ_reduce_ncol = 64
        self.sdQaccum_stage = 2
        self.sdKVaccum_stage = 2
        self.dK_reduce_ncol = 32

    # ------------------------------------------------------------------ MMAs and layouts
    def _mxds_init(self):
        """BwdQSpec.mxds: UE8M0 block scales per 32 queries on the dK path's E4M3 dS (mxf8f6f4 scale_vec::1X, which keeps the
        MN-major operands the hybrid path writes contiguously -- PTX 9.7.16: transpose A/B is supported for this kind)."""
        # token-block scales: 32 x 32 by default -- except on the diagonal r_k routes (diag_k / diag_kr_all), whose dQ copy
        # of dS' assumes the plain dK scale path (asserted below): there the default stays the per-tensor / row scale, and
        # only an explicit BwdQSpec.mxds reaches the assert
        _mxds = self.spec.mxds if self.spec.mxds is not None else BwdQSpec.rule_mxds(self.diag_kr)
        _mxds = {"per_tensor": "0", "mx32": "1", "blk32": "blk32"}[_mxds]   # the internal spelling the checks below read
        assert _mxds in ("0", "1", "blk32"), _mxds
        self.mxds = bool(self.dkv8 and _mxds in ("1", "blk32"))
        self.mxds_blk32 = bool(self.mxds and _mxds == "blk32")
        # stochastic rounding of the hybrid path's E4M3 operands. On by default for dS^ of dK: the analytic bound's ~5x slack
        # makes round-to-nearest flush every softmax row's tail to zero, and SR is worth 0.022 / 0.024 of val loss on two
        # seeds (0.9434 -> 0.9212, 0.9460 -> 0.9220) for +2.5-3.5 % on the backward. P^ of dV is neutral (0.9265), so off.
        # dS^ grid target (swept during development; these are the measured values)
        self.ds_smax_t = 96.0 / 17.0 if self.ds_sr else 6.0
        self.dk_sr = bool(self.dkv8 and self.spec.dk_sr)
        self.dv_sr = bool(self.dkv8 and self.spec.dv_sr)
        # KernelTuning.bwd_fp4q_dk_sr_blk32 (opt-in): SR of the dK side's E4M3 dS^ under the default blk32 dS (whose pack
        # below is round-to-nearest whatever dk_sr says), with the group's dither words (needs ds_sr or dk_sr)
        self.dk_sr_blk32 = bool(self.mxds_blk32 and self.spec.dk_sr_blk32 and (self.ds_sr or self.dk_sr))
        self.sf_vec_mx = 32
        self.sf_dtype_mx = Float8E8M0FNU

    def _get_tiled_mma(self):
        def bs(tiler, a_src=None):
            if a_src is None:
                return sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                    Float4E2M1FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.sf_dtype, self.sf_vec_size,
                    self.cta_group, tiler[:2])
            return sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float4E2M1FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.sf_dtype, self.sf_vec_size,
                self.cta_group, tiler[:2], a_src)

        self._mxds_init()
        self.tiled_mma_S = bs(self.mma_tiler_S)
        self.tiled_mma_dP = bs(self.mma_tiler_dP)
        self.tiled_mma_dQh = bs(self.mma_tiler_dQh, tcgen05.OperandSource.TMEM)
        if self.dq8:
            self.tiled_mma_dQh = sm100_utils_basic.make_trivial_tiled_mma(
                Float8E4M3FN, Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.acc_dtype,
                self.cta_group, self.mma_tiler_dQh[:2], tcgen05.OperandSource.SMEM)
        self.tiled_mma_dK = bs(self.mma_tiler_dK)
        self.tiled_mma_dV = bs(self.mma_tiler_dV)
        if self.dkv8:
            # dK = dS^^T Q, dV = P^T dO: A from the (queries, keys) E4M3 SMEM rows (MN-major), B = E4M3 Q / dO rows (MN-major)
            mn = tcgen05.OperandMajorMode.MN
            if self.mxds:
                self.tiled_mma_dK_mx = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                    Float8E4M3FN, mn, mn, self.sf_dtype_mx, self.sf_vec_mx, self.cta_group, self.mma_tiler_dK[:2])
                self.tiled_mma_dK = self.tiled_mma_dK_mx
            else:
                self.tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(Float8E4M3FN, mn, mn, self.acc_dtype, self.cta_group,
                                                                             self.mma_tiler_dK[:2])
            self.tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(Float8E4M3FN, mn, mn, self.acc_dtype, self.cta_group,
                                                                         self.mma_tiler_dV[:2])
        # SFB helper MMA for the TMA of B-side scale factors whose N rows are 128 (N = keys / d)
        self.tiled_mma_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            Float4E2M1FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.sf_dtype, self.sf_vec_size,
            tcgen05.CtaGroup.ONE, (128, 128))
        self.mma_tiler_sfb = (128, 128, 128)

    def _smem_layout_a_inter(self, tiled_mma, tiler):
        """make_smem_layout_a with the non-swizzled K_INTER atom: logical byte offsets are physical, so the compute warps can
        store scattered bytes of the K-major A operand directly (4-bit row-major A accepts every swizzling mode)."""
        a_shape = tiled_mma.partition_shape_A(cute.dice(tiler, (1, None, 1)))
        atom = sm100_utils_basic.make_smem_layout_atom(sm100_utils_basic.SmemLayoutAtomKind.K_INTER, Float4E2M1FN)
        return sm100_utils_basic.tile_to_mma_shape(atom, cute.append(a_shape, 1), order=(1, 2, 3))

    def _setup_smem_layout(self):
        e2m1 = Float4E2M1FN
        k = {"mma_tile_inst_k": self.mma_inst_tile_k}
        # S = Q4 K4^T: A staged (query tiles), B static (KV tile)
        self.sQ4_layout = sm100_utils_basic.make_smem_layout_a(self.tiled_mma_S, self.mma_tiler_S, e2m1, self.Q_stage)
        sK4 = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_S, self.mma_tiler_S, e2m1, 1)
        self.sK4_layout = cute.slice_(sK4, (None, None, None, 0))
        self.sfq_layout = make_smem_layout_sfa(self.tiled_mma_S, self.mma_tiler_S, self.sf_vec_size, self.Q_stage, **k)
        self.sfk_layout = make_smem_layout_sfb(self.tiled_mma_S, self.mma_tiler_S, self.sf_vec_size, 1, **k)
        # dP = dO4 V4^T
        self.sdO4_layout = sm100_utils_basic.make_smem_layout_a(self.tiled_mma_dP, self.mma_tiler_dP, e2m1, self.dO_stage)
        sV4 = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dP, self.mma_tiler_dP, e2m1, 1)
        self.sV4_layout = cute.slice_(sV4, (None, None, None, 0))
        self.sfdo_layout = make_smem_layout_sfa(self.tiled_mma_dP, self.mma_tiler_dP, self.sf_vec_size, self.dO_stage, **k)
        self.sfv_layout = make_smem_layout_sfb(self.tiled_mma_dP, self.mma_tiler_dP, self.sf_vec_size, 1, **k)
        # dQ halves: A = TMEM dS^, B = K^T rows of one d half (two static stages: half 0 / half 1)
        tdSq = sm100_utils_basic.make_smem_layout_a(self.tiled_mma_dQh, self.mma_tiler_dQh,
                                                    Float8E4M3FN if self.dq8 else e2m1, 1)
        if self.dq8:
            self.sdSq_a_layout = cute.slice_(tdSq, (None, None, None, 0))
        self.tdSq_layout = cute.slice_(tdSq, (None, None, None, 0))
        self.sKt4_layout = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dQh, self.mma_tiler_dQh,
                                                                Float8E4M3FN if self.dq8 else e2m1, 2)
        # K^T SFs, one stage per d half: stage 1 is the host's copy of rows 64-127 moved to columns 0, 1 of the SF atom
        self.sfkt_layout = make_smem_layout_sfb(self.tiled_mma_S if self.dq8 else self.tiled_mma_dQh, self.mma_tiler_dQh,
                                                self.sf_vec_size, 2, **k)
        # diag_k: the key-local K^T payload's scale is the key tile's constant, so both d halves share one SF stage
        self.sfkt1_layout = make_smem_layout_sfb(self.tiled_mma_S if self.dq8 else self.tiled_mma_dQh, self.mma_tiler_dQh,
                                                 self.sf_vec_size, 1, **k)
        sf_mma_q = self.tiled_mma_S if self.dq8 else self.tiled_mma_dQh  # dq8: unused placeholders of the same shape
        self.sfdsq_layout = make_smem_layout_sfa(sf_mma_q, self.mma_tiler_dQh, self.sf_vec_size, 1, **k)
        # dK: A = dS^^T (SMEM, written by the compute warps), B = Q^T staged
        sdSt4 = self._smem_layout_a_inter(self.tiled_mma_dK, self.mma_tiler_dK)  # (a swizzled alternative layout was measured and is not shipped)
        self.sdSt4_layout = cute.slice_(sdSt4, (None, None, None, 0))
        self.sQt4_layout = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dK, self.mma_tiler_dK, e2m1, self.Q_stage)
        sf_mma_k = self.tiled_mma_S if self.dkv8 else self.tiled_mma_dK  # dkv8: unused placeholders of the same shape
        self.sfdst_layout = make_smem_layout_sfa(sf_mma_k, self.mma_tiler_dK, self.sf_vec_size, 1, **k)
        self.sfqt_layout = make_smem_layout_sfb(sf_mma_k, self.mma_tiler_dK, self.sf_vec_size, self.Q_stage, **k)
        # the SFQt TMA atom is built from this E2M1-shaped layout on every route (dkv8 never issues it; MXDS re-shapes
        # sfqt_layout for the E4M3 K-instructions below, which the E2M1 SFB helper's V-map does not match)
        self.sfqt_tma_layout = self.sfqt_layout
        # dV: A = P^v^T (SMEM), B = dO^T staged
        sPt4 = self._smem_layout_a_inter(self.tiled_mma_dV, self.mma_tiler_dV)
        self.sPt4_layout = cute.slice_(sPt4, (None, None, None, 0))
        self.sdOt4_layout = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dV, self.mma_tiler_dV, e2m1, self.dO_stage)
        sf_mma_v = self.tiled_mma_S if self.dkv8 else self.tiled_mma_dV
        self.sfpt_layout = make_smem_layout_sfa(sf_mma_v, self.mma_tiler_dV, self.sf_vec_size, 1, **k)
        self.sfdot_layout = make_smem_layout_sfb(sf_mma_v, self.mma_tiler_dV, self.sf_vec_size, self.dO_stage, **k)
        if self.dkv8:
            e4 = Float8E4M3FN
            self.sdSt4_layout = cute.slice_(sm100_utils_basic.make_smem_layout_a(self.tiled_mma_dK, self.mma_tiler_dK, e4, 1),
                                            (None, None, None, 0))
            self.sPt4_layout = cute.slice_(sm100_utils_basic.make_smem_layout_a(self.tiled_mma_dV, self.mma_tiler_dV, e4, 1),
                                           (None, None, None, 0))
            self.sQt4_layout = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dK, self.mma_tiler_dK, e4, self.Q_stage)
            self.sdOt4_layout = sm100_utils_basic.make_smem_layout_b(self.tiled_mma_dV, self.mma_tiler_dV, e4, self.dO_stage)
            # the same bytes as written by the query-row compute warps: (queries, keys) row-major E4M3
            self.sKV8_rows_layout = sm100_utils_basic.make_smem_layout_epi(e4, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n), 1)
            if self.mxds:
                # real SF tiles for the dK GEMM: A = dS (one UE8M0 per key row x 32 queries), B = Q^ (constant 2^0).
                # token-block scales: the E4M3 GEMM issues tile_m // 32 K-instructions of 32 queries -- the E2M1 GEMMs' `k`
                # (tile_hdim // 64 = 2 instructions of 64) described half of the K extent and cudaErrorMisalignedAddress
                # followed (the fault the mx32 dS first showed)
                kmx = {"mma_tile_inst_k": self.tile_m // 32}
                self.sfdst_layout = make_smem_layout_sfa(self.tiled_mma_dK_mx, self.mma_tiler_dK, self.sf_vec_mx, 1, **kmx)
                self.sfqt_layout = make_smem_layout_sfb(self.tiled_mma_dK_mx, self.mma_tiler_dK, self.sf_vec_mx, self.Q_stage, **kmx)
        # plain (keys, 64 bytes) row-major byte views written by the compute warps (the A operands' K-major bytes) and the
        # (keys, 8 groups) SF bytes; the MMA reads the same memory through the layouts above
        # K_INTER A layout (strides in nibbles): (row, byte) -> (byte // 16) * 2048 + row * 16 + byte % 16 (four 16-byte atoms
        # per row, atom-major; verified with single-byte probes)
        self.sAbytes_layout = cute.make_layout(8192)
        self.sAsf_layout = cute.make_layout((128, 8))
        # per-query stats (2 floats per query: frame F, 1 / Z) and D, staged per query tile
        self.sFU_layout = cute.make_layout((self.tile_m, 2, self.Q_stage), stride=(2, 1, 2 * cute.round_up(self.tile_m, 64)))
        self.sD_layout = cute.make_layout((self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64)))
        # dQ reduce staging (tile_m rows, 64 cols) f16; dK / dV epilogue
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dq_reduce_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.dQ_reduce_ncol), self.sdQaccum_stage)
        nwg = self.num_compute_wgs
        self.sdK_epi_tile = (self.tile_n, math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // nwg))
        self.sdV_epi_tile = (self.tile_n, math.gcd(128 // (self.dv_dtype.width // 8), self.tile_hdim // nwg))
        if self.dkv8:  # SMEM budget (232448 bytes): the E4M3 dK / dV operands are 2x the E2M1 ones -> 16-column epilogue chunks
            self.sdK_epi_tile = (self.tile_n, 16)
            self.sdV_epi_tile = (self.tile_n, 16)
            self.dK_reduce_ncol = 16
        self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(self.dk_dtype, LayoutEnum.ROW_MAJOR, self.sdK_epi_tile, nwg)
        self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(self.dv_dtype, LayoutEnum.ROW_MAJOR, self.sdV_epi_tile, nwg)

    # ------------------------------------------------------------------ entry
    @cute.jit
    def __call__(
        self,
        mQ4, mK4, mV4, mdO4,          # cute.Pointer, E2M1 row-wise (b, s, h, 64)
        mQt4, mKt4, mdOt4,            # cute.Pointer, E2M1 transposed (b, 128, h, s/2); K^T keys in the C3 32-block order
        mSFQ: cute.Tensor, mSFK: cute.Tensor, mSFV: cute.Tensor, mSFdO: cute.Tensor,
        mSFQt: cute.Tensor, mSFKt: cute.Tensor, mSFdOt: cute.Tensor, mSFKt1: cute.Tensor,
        mFrame: cute.Tensor,          # (b, h, s_q) f32: the forward's final integer log2 frame F
        mUinv: cute.Tensor,           # (b, h, s_q) f32: 1 / Z (the forward's row sum in its frame)
        mD: cute.Tensor,              # (b, h, s_q) f32: sum_d O dO^ / (s_dO s_V)
        mdQaccum: cute.Tensor,        # (b, h, s_q, d) f16
        mdK: cute.Tensor,             # (b, s_k, h_k, d) bf16
        mdV: cute.Tensor,             # (b, s_k, h_k, d) bf16
        softmax_scale: Float32,
        ds_mul: Float32,              # dS' multiplier before the dS^ quantiser
        dk_scale: Float32,
        dv_scale: Float32,
        dq_shift: Float32,   # dq8: pow2 pre-scale of the dQ partials so their f16 sum cannot overflow (1.0 otherwise)
        sr_seed: Int32,
        q4_shape: tuple = (), k4_shape: tuple = (), v4_shape: tuple = (), do4_shape: tuple = (),
        qt4_shape: tuple = (), kt4_shape: tuple = (), dot4_shape: tuple = (),
        mQ8: Optional[cute.Tensor] = None,    # dkv8: per-tensor E4M3 Q (b, s_q, h, d) (B of dK)
        mdO8: Optional[cute.Tensor] = None,   # dkv8: per-tensor E4M3 dO (b, s_q, h, d) (B of dV)
        mK8: Optional[cute.Tensor] = None,    # dq8: per-tensor E4M3 K^, C3-permuted keys (b, s_k, h_kv, d) (B of dQ)
        # vtok (causal-safe V^, flash_fwd_sm100_fp4): per-key bias beta and weight u, (b, h_k, s_k) f32 -- S += beta before the
        # P^ reproduction, Mitchell values x u in m (sum_k m = 1 with the forward's u-weighted Z)
        mVBias: Optional[cute.Tensor] = None,
        mVU: Optional[cute.Tensor] = None,
        # diagonal-only: the DIAGONAL payload's row-wise V codes and their scale factors, the dP operand on the
        # diagonal pair. Without them the dP operand is the key-grouped payload on every pair and dQ keeps a
        # measured dependence on future V even though the forward has none.
        mV4b=None,                            # cute.Pointer, E2M1 row-wise, same shape as mV4
        mSFVb: Optional[cute.Tensor] = None,
        # diag_k: the key-local K^T payload (codes in mKt4's layout, SF tile constant per key tile) and the doubled
        # per-key residual [1 | r_k] (b, h_k, 2 s_k) f32 that the dQ copy of dS' carries on the cut pair
        mKt4b=None,
        mSFKtb: Optional[cute.Tensor] = None,
        mKR: Optional[cute.Tensor] = None,
        # dK / dV epilogue scales and the dS multiplier [dk_scale, dv_scale, ds_mul] as device fp32 (the host never reads
        # the per-tensor E4M3 amax nor the global dS bound); None: the dk_scale / dv_scale / ds_mul arguments
        mScl: Optional[cute.Tensor] = None,
        # diag_fold: the dV operand's per-tile constant (b, h_k, s_k / 128); the dK / dV row factors are applied after the
        # kernel (fp4q_run) -- in the epilogue they cost the backward ~7 % at A3, a register-cliff effect (bwdk: the opt-in
        # mFdK / mFdV below stage them at the work-tile start instead: A1 -6.2 %, A3 -0.1 % on the route)
        mDvA: Optional[cute.Tensor] = None,
        # debug build (the all-FP4 P^ gate): (b, h, s_q, s_k) uint8, every reproduced E2M1 code
        mCodeDbg: Optional[cute.Tensor] = None,
        # the step's stochastic-rounding counter (int64 (1,)): the seed is derived from it in the kernel (sr_seed unused)
        mSeed: Optional[cute.Tensor] = None,
        # bwdk (opt-in, KernelTuning.bwd_fp4q_post = 'epi'): the K fold's dK / dV row factors (b, h_k, s_k) f32 powers of
        # two, applied in the dK / dV epilogue as a per-row epilogue scale (scale * f exact, and the bf16 rounding commutes
        # with a power of two outside the subnormal range: bitwise the post launch's in-place multiply)
        mFdK: Optional[cute.Tensor] = None,
        mFdV: Optional[cute.Tensor] = None,
        # documents (doc_varlen): lp_varlen.LPDocs.doc_tab (6, n_docs + 1) and the document of each key tile
        mDocTab: Optional[cute.Tensor] = None,
        mNDoc: Optional[cute.Tensor] = None,
        # cov5 codes_packed: [cu_q packed (n + 1,), cu_k packed (n + 1,), SF4Q (h, Rq 2), SF4K (h_k, R 2), SF4dO (h, Rq 2),
        # SFV words (h_k, R 2), SFVb words (h_k, R 2) | tile words, K^T key-parity copies (2, h_k, d, TB) u8, K^T's tile
        # words (h_k, tw), the padding-key table (h_k, tiles, 4) f32]; the maps' aligned row counts of the queries / keys
        # and the per-key tables' half
        stream: cuda.CUstream = None,
        mPk: Optional[list] = None,
        pk_sq=None,
        pk_sk=None,
        pk_tp=None,
        # (spd4) the capped grid: the key side's device tile count (1,) int32 and the main launch's key tiles (cap); with
        # them the backward is two launches -- key tiles [0, cap) one per CTA, the rest persistent (PackedOverflowScheduler)
        pk_nt: Optional[cute.Tensor] = None,
        pk_cap=None,
    ):
        assert (mDocTab is not None) == self.doc_varlen and (mNDoc is not None) == self.doc_varlen
        self.dev_scl = mScl is not None
        self.fold_epi = mFdK is not None
        # the SR seed from the device counter: GQA kernels read it once per work tile (dev_seed_tile), MHA kernels at the
        # kernel entry (the c8a0af15 form, measured faster there)
        self.dev_seed_tile = mSeed is not None and self.qhead_per_kvhead > 1
        self.vtok = const_expr(mVBias is not None)
        # `query-row-key-local-v-scale-dv-halved`, folded into the kernel. With the
        # key-local V the all-FP4 dV GEMM's A operand used to be the BIASED w' = P'^ / Z, which carries P s_k, so the
        # host divided dV by s_k afterwards. Here the A operand is built from the UNBIASED m = Mitchell u SF / Z (~ P)
        # the dkv8 branch already multiplies, so the kernel's dV is dL/dV directly. m is not an E2M1 code, so the
        # P^v block scale is re-derived from m's own 16-key group maximum (E4M3(amax / 6), reduced over the 16 query
        # threads exactly as before) instead of assuming the codes' maximum of 6.
        self.vtok_dv_u = const_expr(self.vtok and not self.dkv8)
        # diagonal-only: the bias / weight apply on the forward's masked steps. The forward classifies per CTA
        # (q_tiles x 128 packed rows), so the classification is the same block_info call on the forward CTA's first
        # query tile (_diag_m), not on the backward's own 128-query tile (`diagonal-pairs-are-the-forwards-cta`).
        self.vtok_diag = const_expr(self.vtok_diag_arg and self.vtok)
        if const_expr(self.vtok_diag):
            assert self.is_causal, "vtok_diag: causal only"
            assert self.dkv8, "vtok_diag: the dkv8 (e4m3 dK / dV) backward only"
            assert self.tile_m == 128 and self.tile_n == 128, "vtok_diag: the forward's tiling"
        if const_expr(self.diag_k):
            assert self.vtok_diag and self.qhead_per_kvhead == 1 and not self.dq8, "diag_k: diagonal-only MHA and the E2M1 dQ"
            assert mKt4b is not None and mSFKtb is not None and mKR is not None, "diag_k: the key-local K^T payload and r_k"
        if const_expr(self.diag_kr_all):
            assert mKR is not None and not self.dq8, "diag_kr_all: r_k, and the E2M1 dQ (K^T in its place is key-local)"
        if const_expr(self.diag_fold):
            assert mDvA is not None and not self.dq8 and self.dkv8, (
                "diag_fold: the per-tile dV constant, the E2M1 dQ, the dkv8 backward")
        self.lse_dtype = Float32
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        assert mdQaccum.element_type == self.dq_reduce_dtype

        def _fp4_tensor(ptr, shape):
            return cute.make_tensor(ptr, cute.make_ordered_layout(shape, order=tuple(range(len(shape) - 1, -1, -1))))

        mQ4, mK4, mV4, mdO4 = [_fp4_tensor(p, s) for p, s in ((mQ4, q4_shape), (mK4, k4_shape), (mV4, v4_shape), (mdO4, do4_shape))]
        mQt4, mKt4, mdOt4 = [_fp4_tensor(p, s) for p, s in ((mQt4, qt4_shape), (mKt4, kt4_shape), (mdOt4, dot4_shape))]
        if const_expr(mV4b is not None):
            mV4b = _fp4_tensor(mV4b, v4_shape)
        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]
        T = [1, 3, 2, 0]  # (b, s, h, d) -> (s, d, h, b); (b, d, h, s) -> (d, s, h, b)
        mQ4, mK4, mV4, mdO4, mQt4, mKt4, mdOt4 = [layout_utils.select(t, mode=T) for t in (mQ4, mK4, mV4, mdO4, mQt4, mKt4, mdOt4)]
        # The DSL's AST pass flags ANY name assigned inside an `if` body as defined in control flow and refuses
        # to read it from another region (the load / mma warp blocks), static predicate or not. So no b-name
        # below is assigned inside an if: without the diagonal payload every b-name IS the ordinary V name, so
        # the prologue's dP issues against exactly the same buffers as before and the default kernel is bitwise.
        assert not self.vtok_dp_diag or (mV4b is not None and mSFVb is not None), (
            "vtok_diag: the diagonal payload's row-wise V and SF")
        mV4b = layout_utils.select(mV4b, mode=T) if self.vtok_dp_diag else mV4
        mKt4b = layout_utils.select(_fp4_tensor(mKt4b, kt4_shape), mode=T) if self.diag_k else mKt4
        mFrame, mUinv, mD = [layout_utils.select(t, mode=[2, 1, 0]) for t in (mFrame, mUinv, mD)]
        mdQaccum = layout_utils.select(mdQaccum, mode=[2, 3, 1, 0])
        mdK, mdV = [layout_utils.select(t, mode=T) for t in (mdK, mdV)]
        if const_expr(self.codes_packed):
            mPk = list(mPk) + [mdK, mdV]   # (cov5: the epilogue's direct stores to the packed key rows)

        def _sf_tensor(sf, op_shape):
            lay = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, op_shape, (2, 1, 3, 4))
            return cute.make_tensor(cute.recast_ptr(sf.iterator, dtype=self.sf_dtype), lay)

        mSFQ, mSFK, mSFV, mSFdO = [_sf_tensor(s, t.shape) for s, t in ((mSFQ, mQ4), (mSFK, mK4), (mSFV, mV4), (mSFdO, mdO4))]
        mSFQt, mSFKt, mSFdOt, mSFKt1 = [_sf_tensor(s, t.shape) for s, t in ((mSFQt, mQt4), (mSFKt, mKt4), (mSFdOt, mdOt4), (mSFKt1, mKt4))]
        mSFVb = _sf_tensor(mSFVb, mV4b.shape) if self.vtok_dp_diag else mSFV
        mSFKtb = _sf_tensor(mSFKtb, mKt4b.shape) if self.diag_k else mSFKt
        if const_expr(self.dkv8):
            # E4M3 Q / dO in the (d, s, h, b) view of the transposed operands (the dK / dV B loaders tile them unchanged)
            mQt4 = layout_utils.select(assume_tensor_aligned(mQ8), mode=[3, 1, 2, 0])
            mdOt4 = layout_utils.select(assume_tensor_aligned(mdO8), mode=[3, 1, 2, 0])
            if const_expr(self.dq8):
                mKt4 = layout_utils.select(assume_tensor_aligned(mK8), mode=[1, 3, 2, 0])

        self._get_tiled_mma()
        self._setup_smem_layout()
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_S.thr_id.shape,))

        cl = self.cluster_layout_vmnk.shape
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        sel3 = lambda lay: cute.select(lay, mode=[0, 1, 2])
        mA = cute.nvgpu.make_tiled_tma_atom_A
        mB = cute.nvgpu.make_tiled_tma_atom_B
        tma_atom_Q4, tma_tensor_Q4 = mA(tma_load_op, mQ4, sel3(self.sQ4_layout), self.mma_tiler_S, self.tiled_mma_S, cl)
        tma_atom_K4, tma_tensor_K4 = mB(tma_load_op, mK4, sel3(self.sK4_layout), self.mma_tiler_S, self.tiled_mma_S, cl)
        tma_atom_dO4, tma_tensor_dO4 = mA(tma_load_op, mdO4, sel3(self.sdO4_layout), self.mma_tiler_dP, self.tiled_mma_dP, cl)
        tma_atom_V4, tma_tensor_V4 = mB(tma_load_op, mV4, sel3(self.sV4_layout), self.mma_tiler_dP, self.tiled_mma_dP, cl)
        tma_atom_V4b, tma_tensor_V4b = mB(tma_load_op, mV4b, sel3(self.sV4_layout), self.mma_tiler_dP, self.tiled_mma_dP, cl)
        tma_atom_Kt4, tma_tensor_Kt4 = mB(tma_load_op, mKt4, sel3(self.sKt4_layout), self.mma_tiler_dQh, self.tiled_mma_dQh, cl)
        tma_atom_Kt4b, tma_tensor_Kt4b = mB(tma_load_op, mKt4b, sel3(self.sKt4_layout), self.mma_tiler_dQh, self.tiled_mma_dQh, cl)
        tma_atom_Qt4, tma_tensor_Qt4 = mB(tma_load_op, mQt4, sel3(self.sQt4_layout), self.mma_tiler_dK, self.tiled_mma_dK, cl)
        tma_atom_dOt4, tma_tensor_dOt4 = mB(tma_load_op, mdOt4, sel3(self.sdOt4_layout), self.mma_tiler_dV, self.tiled_mma_dV, cl)
        sfa_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
        sfb_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
        cl_sfb = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_sfb.thr_id.shape,)).shape
        i16 = dict(internal_type=cutlass.Int16)
        tma_atom_SFQ, tma_tensor_SFQ = mA(sfa_op, mSFQ, sel3(self.sfq_layout), self.mma_tiler_S, self.tiled_mma_S, cl, **i16)
        tma_atom_SFdO, tma_tensor_SFdO = mA(sfa_op, mSFdO, sel3(self.sfdo_layout), self.mma_tiler_dP, self.tiled_mma_dP, cl, **i16)
        tma_atom_SFK, tma_tensor_SFK = mB(sfb_op, mSFK, sel3(self.sfk_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFV, tma_tensor_SFV = mB(sfb_op, mSFV, sel3(self.sfv_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFVb, tma_tensor_SFVb = mB(sfb_op, mSFVb, sel3(self.sfv_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFKt, tma_tensor_SFKt = mB(sfb_op, mSFKt, sel3(self.sfkt_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFKt1, tma_tensor_SFKt1 = mB(sfb_op, mSFKt1, sel3(self.sfkt_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFKtb, tma_tensor_SFKtb = mB(sfb_op, mSFKtb, sel3(self.sfkt_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFQt, tma_tensor_SFQt = mB(sfb_op, mSFQt, sel3(self.sfqt_tma_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_SFdOt, tma_tensor_SFdOt = mB(sfb_op, mSFdOt, sel3(self.sfdot_layout), self.mma_tiler_sfb, self.tiled_mma_sfb, cl_sfb, **i16)
        tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(tma_store_op, mdK, cute.select(self.sdK_layout, mode=[0, 1]), self.sdK_epi_tile, 1)
        tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(tma_store_op, mdV, cute.select(self.sdV_layout, mode=[0, 1]), self.sdV_epi_tile, 1)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        tma_atom_dQ, mdQaccum_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_reduce_op, mdQaccum, cute.select(self.sdQ_layout, mode=[0, 1]), (self.tile_m, self.dQ_reduce_ncol), 1)
        thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))
        val_layout_r2s_dKV = cute.make_ordered_layout((1, 128 // self.dk_dtype.width), order=(1, 0))
        copy_atom_r2s_dKV = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dk_dtype, num_bits_per_copy=128)
        tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV)

        def _bytes(dtype, layout):
            return cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1, 2]))

        e2m1 = Float4E2M1FN
        self.tma_copy_bytes = {
            "Q": _bytes(e2m1, self.sQ4_layout) + _bytes(self.sf_dtype, self.sfq_layout) + (
                _bytes(Float8E4M3FN, self.sQt4_layout) if self.dkv8 else _bytes(e2m1, self.sQt4_layout) + _bytes(self.sf_dtype, self.sfqt_layout)),
            "dO": _bytes(e2m1, self.sdO4_layout) + _bytes(self.sf_dtype, self.sfdo_layout) + (
                _bytes(Float8E4M3FN, self.sdOt4_layout) if self.dkv8 else _bytes(e2m1, self.sdOt4_layout) + _bytes(self.sf_dtype, self.sfdot_layout)),
            "K": _bytes(e2m1, self.sK4_layout) + _bytes(self.sf_dtype, self.sfk_layout)
            + 2 * _bytes(Float8E4M3FN if self.dq8 else e2m1, self.sKt4_layout)   # dq8: the K^ halves are E4M3, twice the bytes
            + 2 * _bytes(self.sf_dtype, self.sfkt_layout),
            "V": (_bytes(e2m1, self.sV4_layout) + _bytes(self.sf_dtype, self.sfv_layout))
                 * (2 if self.vtok_dp_diag else 1),   # diagonal-only: both payloads on one barrier
            "V1": _bytes(e2m1, self.sV4_layout) + _bytes(self.sf_dtype, self.sfv_layout),
            "LSE": 2 * self.tile_m * 4,
            "dPsum": self.tile_m * 4,
        }
        # cov5 codes_packed: the SF atoms and K^T are the load warp's generic stores (no TMA bytes); the statistics come in
        # 16-byte-aligned 132-float windows
        self.stats_rows = self.tile_m + (4 if self.codes_packed else 0)
        if const_expr(self.codes_packed):
            sfb = lambda lay: _bytes(self.sf_dtype, lay)  # noqa: E731
            self.tma_copy_bytes["Q"] -= sfb(self.sfq_layout)
            self.tma_copy_bytes["dO"] -= sfb(self.sfdo_layout)
            self.tma_copy_bytes["K"] -= sfb(self.sfk_layout) + 2 * _bytes(e2m1, self.sKt4_layout) + 2 * sfb(self.sfkt_layout)
            self.tma_copy_bytes["V"] -= sfb(self.sfv_layout) * (2 if self.vtok_dp_diag else 1)
            self.tma_copy_bytes["V1"] -= sfb(self.sfv_layout)
            self.tma_copy_bytes["LSE"] = 2 * self.stats_rows * 4
            self.tma_copy_bytes["dPsum"] = self.stats_rows * 4

        TileScheduler = SingleTileScheduler
        if const_expr(self.diag_kr):
            assert not self.mxds, "diag_k / diag_kr_all: the dQ copy's r_k assumes the plain (non-MXDS) dK scale path"
        if const_expr(self.diag_k):
            # the aliased K^T payload rides the V tile's barrier and bytes, and its V reload uses phase 0 of one barrier pair:
            # both hold because every CTA runs exactly one work tile
            kb = 2 * _bytes(e2m1, self.sKt4_layout) + _bytes(self.sf_dtype, self.sfkt_layout)
            assert kb == self.tma_copy_bytes["V1"], ("diag_k: the K^T payload must fill the V tile exactly", kb, self.tma_copy_bytes["V1"])
            assert cute.cosize(self.sKt4_layout) <= cute.cosize(self.sV4_layout), "diag_k: K^T payload does not fit the V tile"
            assert cute.cosize(self.sfkt1_layout) <= cute.cosize(self.sfv_layout), "diag_k: K^T SF does not fit the V SF tile"
        pk_split = const_expr(self.codes_packed and pk_cap is not None)
        tile_sched_args = TileSchedulerArguments(
            pk_cap if const_expr(pk_split) else
            cute.ceil_div(pk_sk if const_expr(self.codes_packed) else cute.size(mK4.shape[0]), self.tile_n),
            cute.size(mK4.shape[2]) if const_expr(self.qhead_per_kvhead > 1) else cute.size(mQ4.shape[2]),
            cute.size(mK4.shape[3]),
            1,
            cute.size(mQ4.shape[0]),
            mQ4.shape[1],
            mdO4.shape[1],
            total_q=cute.size(mK4.shape[0]) * cute.size(mK4.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=1,
            is_persistent=False,
            lpt=False,
            head_swizzle=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        sched_over = None
        if const_expr(pk_split):
            # (spd4) the capped grid: at the fork's capacity N = T the static key tile bound min(T, ceil(T / 128) + N) is
            # T, and one CTA per bound tile launched ~T x heads mostly empty CTAs (+3.7 ms at T = 65536 h16/4). The main
            # launch takes [0, cap) (cap = 5 / 4 ceil(T / 128): every tile while the documents average >= 512 keys),
            # the overflow launch the device count's rest, persistent (a CTA per tile of the rest would need the device
            # count on the host; one persistent kernel for all tiles keeps loop-carried state in the main code)
            _nh = cute.size(mK4.shape[2]) if const_expr(self.qhead_per_kvhead > 1) else cute.size(mQ4.shape[2])
            sched_over = PackedOverflowScheduler.Params(pk_nt, _nh, pk_cap,
                                                        (cute.ceil_div(pk_sk, self.tile_n) - pk_cap) * _nh)
        epi_bytes = max(cute.size_in_bytes(self.dk_dtype, self.sdK_layout), cute.size_in_bytes(self.dv_dtype, self.sdV_layout))

        kv_dtype = Float8E4M3FN if self.dkv8 else e2m1
        if const_expr(self.dkv8):
            assert cute.size_in_bytes(Float8E4M3FN, self.sdSt4_layout) == cute.size_in_bytes(Float8E4M3FN, self.sKV8_rows_layout)

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dSkv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            KtB_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]   # diag_k: [K^T payload read by dQ(0) | V reloaded]
            S_loaded_mbar_ptr: cutlass.Int64
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: Int64
            sQ4: cute.struct.Align[cute.struct.MemRange[e2m1, cute.cosize(self.sQ4_layout)], self.buffer_align_bytes]
            sK4: cute.struct.Align[cute.struct.MemRange[e2m1, cute.cosize(self.sK4_layout)], self.buffer_align_bytes]
            sdO4: cute.struct.Align[cute.struct.MemRange[e2m1, cute.cosize(self.sdO4_layout)], self.buffer_align_bytes]
            sV4: cute.struct.Align[cute.struct.MemRange[e2m1, cute.cosize(self.sV4_layout)], self.buffer_align_bytes]
            # diagonal-only: the diagonal payload's V tile, the dP operand on the diagonal pair (1 element when off)
            sV4b: cute.struct.Align[cute.struct.MemRange[e2m1, cute.cosize(self.sV4_layout) if self.vtok_dp_diag else 1],
                                    self.buffer_align_bytes]
            # dq8: the dQ GEMM's K^ halves are E4M3, so the buffer is elements-in-bytes, not nibbles
            sKt4: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN if self.dq8 else e2m1, cute.cosize(self.sKt4_layout)],
                                    self.buffer_align_bytes]
            sQt4: cute.struct.Align[cute.struct.MemRange[kv_dtype, cute.cosize(self.sQt4_layout)], self.buffer_align_bytes]
            sdOt4: cute.struct.Align[cute.struct.MemRange[kv_dtype, cute.cosize(self.sdOt4_layout)], self.buffer_align_bytes]
            sdSt4: cute.struct.Align[cute.struct.MemRange[kv_dtype, cute.cosize(self.sdSt4_layout)], self.buffer_align_bytes]
            sPt4: cute.struct.Align[cute.struct.MemRange[kv_dtype, cute.cosize(self.sPt4_layout)], self.buffer_align_bytes]
            sSFQ: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfq_layout)], self.buffer_align_bytes]
            sSFK: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfk_layout)], self.buffer_align_bytes]
            sSFdO: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfdo_layout)], self.buffer_align_bytes]
            sSFV: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfv_layout)], self.buffer_align_bytes]
            sSFVb: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfv_layout) if self.vtok_dp_diag else 1],
                                     self.buffer_align_bytes]
            sSFKt: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfkt_layout)], self.buffer_align_bytes]
            sSFQt: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, 64 if (self.dkv8 and not self.mxds) else cute.cosize(self.sfqt_layout)],
                self.buffer_align_bytes]
            sSFdOt: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, 64 if self.dkv8 else cute.cosize(self.sfdot_layout)], self.buffer_align_bytes]
            sSFdSt: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, 64 if (self.dkv8 and not self.mxds) else cute.cosize(self.sfdst_layout)],
                self.buffer_align_bytes]
            sSFPt: cute.struct.Align[cute.struct.MemRange[cute.Uint8, 64 if self.dkv8 else cute.cosize(self.sfpt_layout)], self.buffer_align_bytes]
            sFrame: cute.struct.Align[cute.struct.MemRange[Float32, self.stats_rows * self.Q_stage], 128]
            sUinv: cute.struct.Align[cute.struct.MemRange[Float32, self.stats_rows * self.Q_stage], 128]
            sD: cute.struct.Align[cute.struct.MemRange[Float32, self.stats_rows * self.dO_stage], 128]
            sEpiK: cute.struct.Align[cute.struct.MemRange[cute.Uint8, epi_bytes], self.buffer_align_bytes]
            sEpiV: cute.struct.Align[cute.struct.MemRange[cute.Uint8, epi_bytes], self.buffer_align_bytes]
            sdQaccum: cute.struct.Align[cute.struct.MemRange[self.dq_reduce_dtype, cute.cosize(self.sdQ_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        for launch in cutlass.range_constexpr(2 if const_expr(pk_split) else 1):
            self._pk_ovf = launch == 1
            if const_expr(launch == 1):
                # (trace time) the overflow launch: the persistent scheduler
                self.tile_scheduler_cls = PackedOverflowScheduler
                tile_sched_params = sched_over
                grid_dim = PackedOverflowScheduler.get_grid_shape(sched_over)
            self.kernel(
                tma_tensor_Q4, tma_tensor_K4, tma_tensor_dO4, tma_tensor_V4, tma_tensor_Kt4, tma_tensor_Qt4, tma_tensor_dOt4,
                tma_tensor_SFQ, tma_tensor_SFK, tma_tensor_SFdO, tma_tensor_SFV, tma_tensor_SFKt, tma_tensor_SFQt, tma_tensor_SFdOt,
                tma_tensor_SFKt1, mFrame, mUinv, mD,
                mdQaccum_tma_tensor, mdK_tma_tensor, mdV_tma_tensor,
                tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Kt4, tma_atom_Qt4, tma_atom_dOt4,
                tma_atom_SFQ, tma_atom_SFK, tma_atom_SFdO, tma_atom_SFV, tma_atom_SFKt, tma_atom_SFQt, tma_atom_SFdOt,
                tma_atom_SFKt1, tma_atom_dQ, tma_atom_dK, tma_atom_dV,
                tiled_copy_r2s_dKV,
                softmax_scale_log2, ds_mul, dk_scale, dv_scale, dq_shift, sr_seed,
                tile_sched_params, mVBias, mVU,
                tma_tensor_V4b, tma_tensor_SFVb, tma_atom_V4b, tma_atom_SFVb,
                tma_tensor_Kt4b, tma_tensor_SFKtb, tma_atom_Kt4b, tma_atom_SFKtb, mKR, mScl, mDvA, mCodeDbg, mSeed,
                mFdK, mFdV, mDocTab, mNDoc,
                *((mPk, pk_sq, pk_sk, pk_tp) if const_expr(self.codes_packed) else ()),
            ).launch(grid=grid_dim, block=[self.threads_per_cta, 1, 1], cluster=None,
                     smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)
        self.tile_scheduler_cls = TileScheduler
        self._pk_ovf = False

    @cute.kernel
    def kernel(
        self,
        mQ4, mK4, mdO4, mV4, mKt4, mQt4, mdOt4,
        mSFQ, mSFK, mSFdO, mSFV, mSFKt, mSFQt, mSFdOt, mSFKt1,
        mFrame: cute.Tensor, mUinv: cute.Tensor, mD: cute.Tensor,
        mdQaccum: cute.Tensor, mdK_tma_tensor: cute.Tensor, mdV_tma_tensor: cute.Tensor,
        tma_atom_Q4: cute.CopyAtom, tma_atom_K4: cute.CopyAtom, tma_atom_dO4: cute.CopyAtom, tma_atom_V4: cute.CopyAtom,
        tma_atom_Kt4: cute.CopyAtom, tma_atom_Qt4: cute.CopyAtom, tma_atom_dOt4: cute.CopyAtom,
        tma_atom_SFQ: cute.CopyAtom, tma_atom_SFK: cute.CopyAtom, tma_atom_SFdO: cute.CopyAtom, tma_atom_SFV: cute.CopyAtom,
        tma_atom_SFKt: cute.CopyAtom, tma_atom_SFQt: cute.CopyAtom, tma_atom_SFdOt: cute.CopyAtom, tma_atom_SFKt1: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom, tma_atom_dK: cute.CopyAtom, tma_atom_dV: cute.CopyAtom,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale_log2: Float32, ds_mul: Float32, dk_scale: Float32, dv_scale: Float32, dq_shift: Float32, sr_seed: Int32,
        tile_sched_params: ParamsBase,
        mVBias: Optional[cute.Tensor] = None,
        mVU: Optional[cute.Tensor] = None,
        # diagonal-only: the diagonal payload's row-wise V / SF coordinate tensors and their TMA atoms. When the
        # payload is absent these ARE V's own, so the kernel is bitwise the default one.
        mV4b=None, mSFVb=None, tma_atom_V4b=None, tma_atom_SFVb=None,
        # diag_k: the key-local K^T payload's coordinate tensors / atoms (K^T's own when off) and the doubled r_k
        mKt4b=None, mSFKtb=None, tma_atom_Kt4b=None, tma_atom_SFKtb=None, mKR=None, mScl=None,
        mDvA=None,
        mCodeDbg=None,
        mSeed=None,
        mFdK=None,
        mFdV=None,
        mDocTab=None,
        mNDoc=None,
        mPk=None,
        pk_sq=None,
        pk_sk=None,
        pk_tp=None,
    ):
        if const_expr(mSeed is not None):
            # the stochastic-rounding seed from the step's device counter c: (c * 0x9E3779B1) mod 2^31, the host formula,
            # evaluated on the GPU so a CUDA-graph replay draws a fresh dither (the sr_seed argument is then 0).
            if const_expr(self.dev_seed_tile):
                # GQA: compute_loop reads c once per work tile. Read here, the seed held a register from the kernel
                # entry through every warp's loop and the GQA kernel spilled (LDL 54 -> 107, +3.7 %)
                self._sr_ctr = mSeed
            else:
                sr_seed = Int32((mSeed[0] * Int64(0x9E3779B1)) & Int64(0x7FFFFFFF))
        if const_expr(mDvA is not None):
            self._dva = mDvA
        if const_expr(mFdK is not None):
            self._fdk = mFdK
            self._fdv = mFdV
        self.code_dbg = const_expr(mCodeDbg is not None)
        if const_expr(mCodeDbg is not None):
            self._code_dbg = mCodeDbg
        if const_expr(mScl is not None):
            self._scl = mScl
            if const_expr(not self.doc_varlen):
                ds_mul = mScl[2]  # [dk_scale, dv_scale, ds_mul] on the device: no host read of the global dS scale
        if const_expr(self.doc_varlen):
            from flash_attn.cute.lp_bwd_tiles import LPTileInfo
            self._ndoc = mNDoc
            self._sk_alloc = mK4.shape[0]   # the allocation's key count (the doubled per-key tables' half)
            if const_expr(self.codes_packed):
                self._sk_alloc = pk_tp      # (cov5: the packed per-key tables' half T')
                self._pk = mPk
                self._doc_tab = mDocTab
            self._lpt = LPTileInfo(BlockInfo(self.tile_m, self.tile_n, self.is_causal, self.is_local, False, None, None,
                                             qhead_per_kvhead_packgqa=1),
                                   pk_sq if const_expr(self.codes_packed) else mQ4.shape[0],
                                   pk_sk if const_expr(self.codes_packed) else mK4.shape[0], self.tile_m, self.tile_n,
                                   mDocTab=mDocTab, mNDoc=mNDoc, docs=True)
        if const_expr(mKR is not None):
            self._diag_kr = mKR
        if const_expr(mVBias is not None):
            self._vtok_bias = mVBias  # values of the kernel region, read by compute_loop
            self._vtok_u = mVU
        # MMAs and layouts rebuilt inside the kernel region (host-side layouts cannot be used here)
        self._get_tiled_mma()
        self._setup_smem_layout()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                for atom in (tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Kt4, tma_atom_Qt4, tma_atom_dOt4,
                             tma_atom_SFQ, tma_atom_SFK, tma_atom_SFdO, tma_atom_SFV, tma_atom_SFKt, tma_atom_SFQt, tma_atom_SFdOt,
                             tma_atom_SFKt1, tma_atom_dQ, tma_atom_dK, tma_atom_dV):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        S_loaded_mbar_ptr = storage.S_loaded_mbar_ptr
        ktb_mbar = storage.KtB_mbar_ptr.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(S_loaded_mbar_ptr, len(self.compute_warp_ids))
                if const_expr(self.diag_k):
                    cute.arch.mbarrier_init(ktb_mbar, 1)       # one tcgen05.commit after dQ(0)'s second half
                    cute.arch.mbarrier_init(ktb_mbar + 1, 1)   # one arrive_and_expect_tx of the V reload
        cute.arch.mbarrier_init_fence()

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier, allocator_warp_id=self.mma_warp_id,
            is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_S.thr_id.shape,))
        producer_group_MMA = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_compute = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids))
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.S_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        consumer_group_reduce = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids))
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=producer_group_MMA, consumer_group=consumer_group_reduce,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)),
            consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
            barrier_storage=storage.dS_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        # dK / dV A operands (SMEM) of query tile t are committed one step later, after the compute warps loaded S(t+1)
        pipeline_dSkv = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)),
            consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
            barrier_storage=storage.dSkv_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        producer_group_load = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_mma = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_compute_tma = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids))
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(), num_stages=self.Q_stage, producer_group=producer_group_load,
            consumer_group=consumer_group_compute_tma, tx_count=self.tma_copy_bytes["LSE"], defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(), num_stages=self.dO_stage, producer_group=producer_group_load,
            consumer_group=consumer_group_compute_tma, tx_count=self.tma_copy_bytes["dPsum"], defer_sync=True,
        )
        # (cov5 codes_packed: the Q / dO full barriers take a second arrival -- the load warp's, after its SF atom stores)
        producer_group_qdo = (cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 2)
                              if const_expr(self.codes_packed) else producer_group_load)
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(), num_stages=self.Q_stage, producer_group=producer_group_qdo,
            consumer_group=consumer_group_mma, tx_count=self.tma_copy_bytes["Q"], cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(), num_stages=self.dO_stage, producer_group=producer_group_qdo,
            consumer_group=consumer_group_mma, tx_count=self.tma_copy_bytes["dO"], cta_layout_vmnk=cluster_layout_vmnk, defer_sync=False,
        )

        def gt(buf, lay):
            return buf.get_tensor(lay.outer, swizzle=lay.inner)

        sQ4, sK4, sdO4, sV4 = gt(storage.sQ4, self.sQ4_layout), gt(storage.sK4, self.sK4_layout), gt(storage.sdO4, self.sdO4_layout), gt(storage.sV4, self.sV4_layout)
        sV4b = gt(storage.sV4b, self.sV4_layout) if self.vtok_dp_diag else sV4
        sKt4, sQt4, sdOt4 = gt(storage.sKt4, self.sKt4_layout), gt(storage.sQt4, self.sQt4_layout), gt(storage.sdOt4, self.sdOt4_layout)
        sdSt4, sPt4 = gt(storage.sdSt4, self.sdSt4_layout), gt(storage.sPt4, self.sPt4_layout)
        sSFQ, sSFK, sSFdO, sSFV = (storage.sSFQ.get_tensor(self.sfq_layout), storage.sSFK.get_tensor(self.sfk_layout),
                                   storage.sSFdO.get_tensor(self.sfdo_layout), storage.sSFV.get_tensor(self.sfv_layout))
        sSFVb = storage.sSFVb.get_tensor(self.sfv_layout) if self.vtok_dp_diag else sSFV
        sSFKt = storage.sSFKt.get_tensor(self.sfkt_layout)
        # diag_k: the key-local K^T payload lives in the plain V tile and its SF in the V SF tile (aliases)
        sKtb = gt(storage.sV4, self.sKt4_layout) if self.diag_k else sKt4
        sSFKtb = storage.sSFV.get_tensor(self.sfkt1_layout) if self.diag_k else sSFKt
        if const_expr(self.dkv8):  # dK / dV have no block scales: 64-byte placeholders
            sSFQt, sSFdOt = storage.sSFQt.get_tensor(cute.make_layout(64)), storage.sSFdOt.get_tensor(cute.make_layout(64))
            if const_expr(self.mxds):
                sSFQt = storage.sSFQt.get_tensor(self.sfqt_layout, dtype=self.sf_dtype_mx)
            sSFdSt = (storage.sSFdSt.get_tensor(self.sfdst_layout, dtype=self.sf_dtype_mx) if const_expr(self.mxds)
                      else storage.sSFdSt.get_tensor(cute.make_layout(64), dtype=self.sf_dtype))
            sSFPt = storage.sSFPt.get_tensor(cute.make_layout(64), dtype=self.sf_dtype)
        else:
            sSFQt, sSFdOt = storage.sSFQt.get_tensor(self.sfqt_layout), storage.sSFdOt.get_tensor(self.sfdot_layout)
            sSFdSt = storage.sSFdSt.get_tensor(self.sfdst_layout, dtype=self.sf_dtype)
            sSFPt = storage.sSFPt.get_tensor(self.sfpt_layout, dtype=self.sf_dtype)
        # raw byte views of the compute-written A operands / SF buffers
        if const_expr(self.dkv8):
            sdSt4_u8, sPt4_u8 = sdSt4, sPt4  # dkv8: the compute warps rebuild the (queries, keys) row view from the iterators
        else:
            sdSt4_u8 = cute.make_tensor(cute.recast_ptr(sdSt4.iterator, dtype=cute.Uint8), self.sAbytes_layout)
            sPt4_u8 = cute.make_tensor(cute.recast_ptr(sPt4.iterator, dtype=cute.Uint8), self.sAbytes_layout)
        n_sfd = 64 if const_expr(self.dkv8 and not self.mxds) else cute.cosize(self.sfdst_layout)
        n_sfp = 64 if const_expr(self.dkv8) else cute.cosize(self.sfpt_layout)
        sSFdSt_u8 = storage.sSFdSt.get_tensor(cute.make_layout(n_sfd))
        sSFPt_u8 = storage.sSFPt.get_tensor(cute.make_layout(n_sfp))
        sSFdSt_mx = storage.sSFdSt.get_tensor(cute.make_layout(n_sfd))
        sSFQt_mx = cute.make_tensor(cute.recast_ptr(storage.sSFQt.get_tensor(cute.make_layout(4)).iterator, dtype=Int32),
                                    cute.make_layout(cute.cosize(self.sfqt_layout) // 4 if const_expr(self.mxds) else 1))
        sSFdSt_u8 = cute.make_tensor(cute.recast_ptr(sSFdSt_u8.iterator, dtype=cute.Uint16), cute.make_layout(n_sfd // 2))
        sSFPt_u8 = cute.make_tensor(cute.recast_ptr(sSFPt_u8.iterator, dtype=cute.Uint16), cute.make_layout(n_sfp // 2))
        sFrame = storage.sFrame.get_tensor(cute.make_layout((self.stats_rows, self.Q_stage), stride=(1, self.stats_rows)))
        sUinv = storage.sUinv.get_tensor(cute.make_layout((self.stats_rows, self.Q_stage), stride=(1, self.stats_rows)))
        sD = storage.sD.get_tensor(cute.make_layout((self.stats_rows, self.dO_stage), stride=(1, self.stats_rows)))
        sdV = storage.sEpiV.get_tensor(self.sdV_layout.outer, swizzle=self.sdV_layout.inner, dtype=self.dv_dtype)
        sdK = storage.sEpiK.get_tensor(self.sdK_layout.outer, swizzle=self.sdK_layout.inner, dtype=self.dk_dtype)
        if const_expr(self.fold_epi):
            # fold_epi: the K fold's row factors of each compute thread's key row, parked in the dK epilogue's staging
            # buffer (idle until the dK epilogue): warpgroup w's 128 threads x (f_K, f_V) inside w's own slice
            nwg_e = self.num_compute_wgs
            slice_f = cute.size_in_bytes(self.dk_dtype, self.sdK_layout) // 4 // nwg_e
            self._sfkv = cute.make_tensor(cute.recast_ptr(storage.sEpiK.data_ptr(), dtype=Float32),
                                          cute.make_layout((nwg_e, 128, 2), stride=(slice_f, 2, 1)))
        sdQ = gt(storage.sdQaccum, self.sdQ_layout)

        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

        def _acc(tiled_mma, tiler, off):
            thr = tiled_mma.get_slice(0)
            return thr, cute.make_tensor(tmem_ptr + off, thr.make_fragment_C(thr.partition_shape_C(tiler[:2])).layout)

        thr_mma_S, tStS = _acc(self.tiled_mma_S, self.mma_tiler_S, self.tmem_S_offset)
        thr_mma_dP, tdPtdP = _acc(self.tiled_mma_dP, self.mma_tiler_dP, self.tmem_dP_offset)
        thr_mma_dV, tdVtdV = _acc(self.tiled_mma_dV, self.mma_tiler_dV, self.tmem_dV_offset)
        thr_mma_dK, tdKtdK = _acc(self.tiled_mma_dK, self.mma_tiler_dK, self.tmem_dK_offset)
        thr_mma_dQh, tdQtdQ = _acc(self.tiled_mma_dQh, self.mma_tiler_dQh, self.tmem_dQ_offset)
        tdSq = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_dSq_offset,
                                                dtype=Float8E4M3FN if const_expr(self.dq8) else Float4E2M1FN),
                                self.tdSq_layout.outer)
        if const_expr(self.dq8):
            # the same bytes the dK GEMM reads, viewed K-major as A of dQ (queries x keys, keys contiguous)
            tdSq = gt(storage.sdSt4, self.sdSq_a_layout)

        def _sf_tmem(col, layout):
            return cute.make_tensor(cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=16),
                                                    dtype=self.sf_dtype), layout)

        def sl(lay):
            return cute.slice_(lay, (None, None, None, 0))

        sfa_t = blockscaled_utils.make_tmem_layout_sfa
        sfb_t = blockscaled_utils.make_tmem_layout_sfb
        tCtSFQ = _sf_tmem(self.tmem_SFQ_offset, sfa_t(self.tiled_mma_S, self.mma_tiler_S, self.sf_vec_size, sl(self.sfq_layout)))
        tCtSFK = _sf_tmem(self.tmem_SFK_offset, sfb_t(self.tiled_mma_S, self.mma_tiler_S, self.sf_vec_size, sl(self.sfk_layout)))
        tCtSFdO = _sf_tmem(self.tmem_SFdO_offset, sfa_t(self.tiled_mma_dP, self.mma_tiler_dP, self.sf_vec_size, sl(self.sfdo_layout)))
        tCtSFV = _sf_tmem(self.tmem_SFV_offset, sfb_t(self.tiled_mma_dP, self.mma_tiler_dP, self.sf_vec_size, sl(self.sfv_layout)))
        tCtSFdSq = _sf_tmem(self.tmem_SFdSq_offset, sfa_t(self.tiled_mma_dQh, self.mma_tiler_dQh, self.sf_vec_size, sl(self.sfdsq_layout)))
        tCtSFKt = _sf_tmem(self.tmem_SFKt_offset, sfb_t(self.tiled_mma_dQh, self.mma_tiler_dQh, self.sf_vec_size, sl(self.sfkt_layout)))
        # the S2T of the 128-row K^T SF tile goes through a 128-row TMEM layout (rows 64-127 at +2 columns); the half GEMMs read
        # the same memory through the 64-row layout (half 1 with sfb_col_offset 2)
        tCtSFKt128 = _sf_tmem(self.tmem_SFKt_offset, sfb_t(self.tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, sl(self.sfkt_layout)))
        tCtSFVb = (_sf_tmem(self.tmem_SFVb_offset, sfb_t(self.tiled_mma_dP, self.mma_tiler_dP, self.sf_vec_size, sl(self.sfv_layout)))
                   if self.vtok_dp_diag else tCtSFV)
        if const_expr(self.mxds):
            _sf_tmem_mx = lambda col, layout: cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=16),
                                dtype=self.sf_dtype_mx), layout)
            tCtSFdSt = _sf_tmem_mx(self.tmem_SFdSt_offset,
                                   sfa_t(self.tiled_mma_dK_mx, self.mma_tiler_dK, self.sf_vec_mx, sl(self.sfdst_layout)))
            tCtSFQt = _sf_tmem_mx(self.tmem_SFQt_offset,
                                  sfb_t(self.tiled_mma_dK_mx, self.mma_tiler_dK, self.sf_vec_mx, sl(self.sfqt_layout)))
        else:
            tCtSFdSt = _sf_tmem(self.tmem_SFdSt_offset, sfa_t(self.tiled_mma_dK, self.mma_tiler_dK, self.sf_vec_size, sl(self.sfdst_layout)))
            tCtSFQt = _sf_tmem(self.tmem_SFQt_offset, sfb_t(self.tiled_mma_dK, self.mma_tiler_dK, self.sf_vec_size, sl(self.sfqt_layout)))
        tCtSFPt = _sf_tmem(self.tmem_SFPv_offset, sfa_t(self.tiled_mma_dV, self.mma_tiler_dV, self.sf_vec_size, sl(self.sfpt_layout)))
        tCtSFdOt = _sf_tmem(self.tmem_SFdOt_offset, sfb_t(self.tiled_mma_dV, self.mma_tiler_dV, self.sf_vec_size, sl(self.sfdot_layout)))

        block_info = BlockInfo(self.tile_m, self.tile_n, self.is_causal, self.is_local, False, None, None, qhead_per_kvhead_packgqa=1)
        SeqlenInfoCls = partial(SeqlenInfoQK.create, seqlen_q_static=mQ4.shape[0], seqlen_k_static=mK4.shape[0], mCuSeqlensQ=None,
                                mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=None, tile_m=self.tile_m, tile_n=self.tile_n)
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        if const_expr(getattr(self, "_pk_ovf", False)):
            # (spd4) the overflow launch of the capped packed grid: persistent over the tiles past the main grid, each tile
            # run by the one-tile code on freshly initialised barriers and fresh pipeline states (the one-tile kernel's
            # pipelines are not balanced across work tiles). Between tiles: the MMA warp's commit (every UMMA and its
            # barrier arrivals done), a sync of the role warps, warp 0 re-initialises every mbarrier, a second sync.
            role_bar = cutlass.pipeline.NamedBarrier(barrier_id=int(NamedBarrierBwdSm100.TmemPtr) + 1,
                                                     num_threads=cute.arch.WARP_SIZE * self.empty_warp_ids[0])
            if warp_idx >= self.empty_warp_ids[0]:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)
            if warp_idx == self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
                tmem.allocate(self.tmem_alloc_cols)
            if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_reduce)
            if warp_idx < self.empty_warp_ids[0] and warp_idx != self.load_warp_id:
                tmem.wait_for_alloc()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ktb_mbar, 1)   # (the tile-done commit; diag_k is off on this path)
            cute.arch.mbarrier_init_fence()

            ovf_pipes = (pipeline_S_P, pipeline_dP, pipeline_dKV, pipeline_dQ, pipeline_dS, pipeline_dSkv,
                         pipeline_LSE, pipeline_dPsum, pipeline_Q, pipeline_dO)

            # one tile loop per role (each carries only its own role's state across tiles)
            if warp_idx == self.load_warp_id:
                role_bar.arrive_and_wait()
                sch_l = PackedOverflowScheduler.create(tile_sched_params)
                wt_l = sch_l.initial_work_tile_info()
                while wt_l.is_valid_tile:
                    nb_l, hh_l, _, _ = wt_l.tile_idx
                    self.load_packed(
                        thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, mQ4, mK4, mdO4, mV4, mQt4, mdOt4, mFrame, mUinv,
                        mD, sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sSFQ, sSFK, sSFdO, sSFV, sSFKt, sFrame, sUinv, sD,
                        tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Qt4, tma_atom_dOt4,
                        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum, block_info, SeqlenInfoCls,
                        partial(FixedTileScheduler.create, FixedTileScheduler.Params(nb_l, hh_l)),
                        mV4b, sV4b, sSFVb, tma_atom_V4b)
                    self._ovf_tile_end(role_bar, ovf_pipes, S_loaded_mbar_ptr, ktb_mbar)
                    sch_l.advance_to_next_work()
                    wt_l = sch_l.get_current_work()
            if warp_idx == self.mma_warp_id:
                role_bar.arrive_and_wait()
                sch_m = PackedOverflowScheduler.create(tile_sched_params)
                wt_m = sch_m.initial_work_tile_info()
                while wt_m.is_valid_tile:
                    nb_m, hh_m, _, _ = wt_m.tile_idx
                    self.mma_kv8(
                        sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sdSt4, sPt4,
                        sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sSFdSt, sSFPt,
                        tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ, tdSq,
                        tCtSFQ, tCtSFK, tCtSFdO, tCtSFV, tCtSFdSq, tCtSFKt, tCtSFdSt, tCtSFQt, tCtSFPt, tCtSFdOt,
                        pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP,
                        pipeline_dQ, S_loaded_mbar_ptr, block_info, SeqlenInfoCls,
                        partial(FixedTileScheduler.create, FixedTileScheduler.Params(nb_m, hh_m)),
                        tCtSFKt128=tCtSFKt128, sV4b=sV4b, sSFVb=sSFVb, tCtSFVb=tCtSFVb, sKtb=sKtb, sSFKtb=sSFKtb,
                        ktb_mbar=ktb_mbar)
                    # every UMMA of the tile (and the barrier arrivals of its commits) complete
                    with cute.arch.elect_one():
                        tcgen05.commit(ktb_mbar)
                    cute.arch.mbarrier_wait(ktb_mbar, 0)
                    self._ovf_tile_end(role_bar, ovf_pipes, S_loaded_mbar_ptr, ktb_mbar)
                    sch_m.advance_to_next_work()
                    wt_m = sch_m.get_current_work()
            if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
                role_bar.arrive_and_wait()
                sch_c = PackedOverflowScheduler.create(tile_sched_params)
                wt_c = sch_c.initial_work_tile_info()
                while wt_c.is_valid_tile:
                    nb_c, hh_c, _, _ = wt_c.tile_idx
                    self.compute_loop(
                        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
                        tStS, tdPtdP, tdVtdV, tdKtdK, sFrame, sUinv, sD, sdSt4_u8, sPt4_u8, sSFdSt_u8, sSFPt_u8,
                        sSFdSt_mx, sSFQt_mx, S_loaded_mbar_ptr,
                        pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV,
                        pipeline_dP, softmax_scale_log2, ds_mul, sr_seed, block_info, SeqlenInfoCls,
                        partial(FixedTileScheduler.create, FixedTileScheduler.Params(nb_c, hh_c)),
                        sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK, tiled_copy_r2s_dKV,
                        dk_scale, dv_scale,
                    )
                    self._ovf_tile_end(role_bar, ovf_pipes, S_loaded_mbar_ptr, ktb_mbar)
                    sch_c.advance_to_next_work()
                    wt_c = sch_c.get_current_work()
            if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
                role_bar.arrive_and_wait()
                sch_r = PackedOverflowScheduler.create(tile_sched_params)
                wt_r = sch_r.initial_work_tile_info()
                while wt_r.is_valid_tile:
                    nb_r, hh_r, _, _ = wt_r.tile_idx
                    self.dQacc_reduce(mdQaccum, sdQ, tma_atom_dQ, thr_mma_dQh, tdQtdQ, pipeline_dQ, block_info,
                                      SeqlenInfoCls, partial(FixedTileScheduler.create, FixedTileScheduler.Params(nb_r, hh_r)),
                                      dq_shift)
                    self._ovf_tile_end(role_bar, ovf_pipes, S_loaded_mbar_ptr, ktb_mbar)
                    sch_r.advance_to_next_work()
                    wt_r = sch_r.get_current_work()
            if warp_idx == self.mma_warp_id:
                tmem.relinquish_alloc_permit()
                tmem_alloc_barrier.arrive_and_wait()
                tmem._num_allocated_columns = self.tmem_alloc_cols   # (the allocate ran in another branch: trace-time count)
                tmem.free(tmem.retrieve_ptr(Float32))
            if warp_idx < self.mma_warp_id:
                tmem_alloc_barrier.arrive()
        else:
            if warp_idx >= self.empty_warp_ids[0]:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)
            if const_expr(self.codes_packed):
                if warp_idx == self.load_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_load)
                    self.load_packed(
                        thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, mQ4, mK4, mdO4, mV4, mQt4, mdOt4, mFrame, mUinv, mD,
                        sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sSFQ, sSFK, sSFdO, sSFV, sSFKt, sFrame, sUinv, sD,
                        tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Qt4, tma_atom_dOt4,
                        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum, block_info, SeqlenInfoCls, TileSchedulerCls,
                        mV4b, sV4b, sSFVb, tma_atom_V4b)
            elif warp_idx == self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.load(
                    thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_dQh,
                    mQ4, mK4, mdO4, mV4, mKt4, mQt4, mdOt4, mSFQ, mSFK, mSFdO, mSFV, mSFKt, mSFQt, mSFdOt, mFrame, mUinv, mD,
                    mSFKt1, tma_atom_SFKt1, sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sFrame, sUinv, sD,
                    tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Kt4, tma_atom_Qt4, tma_atom_dOt4,
                    tma_atom_SFQ, tma_atom_SFK, tma_atom_SFdO, tma_atom_SFV, tma_atom_SFKt, tma_atom_SFQt, tma_atom_SFdOt,
                    pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum, block_info, SeqlenInfoCls, TileSchedulerCls,
                    mV4b=mV4b, mSFVb=mSFVb, sV4b=sV4b, sSFVb=sSFVb,
                    tma_atom_V4b=tma_atom_V4b, tma_atom_SFVb=tma_atom_SFVb,
                    mKt4b=mKt4b, mSFKtb=mSFKtb, sKtb=sKtb, sSFKtb=sSFKtb, tma_atom_Kt4b=tma_atom_Kt4b,
                    tma_atom_SFKtb=tma_atom_SFKtb, ktb_mbar=ktb_mbar,
                )
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
                tmem.allocate(self.tmem_alloc_cols)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                mma_fn = self.mma_kv8 if const_expr(self.dkv8) else self.mma
                mma_fn(
                    sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sdSt4, sPt4,
                    sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sSFdSt, sSFPt,
                    tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ, tdSq,
                    tCtSFQ, tCtSFK, tCtSFdO, tCtSFV, tCtSFdSq, tCtSFKt, tCtSFdSt, tCtSFQt, tCtSFPt, tCtSFdOt,
                    pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP, pipeline_dQ,
                    S_loaded_mbar_ptr, block_info, SeqlenInfoCls, TileSchedulerCls, tCtSFKt128=tCtSFKt128,
                    sV4b=sV4b, sSFVb=sSFVb, tCtSFVb=tCtSFVb, sKtb=sKtb, sSFKtb=sSFKtb, ktb_mbar=ktb_mbar,
                )
                tmem.relinquish_alloc_permit()
                tmem_alloc_barrier.arrive_and_wait()
                tmem.free(tmem_ptr)
            if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_compute)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                self.compute_loop(
                    thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
                    tStS, tdPtdP, tdVtdV, tdKtdK, sFrame, sUinv, sD, sdSt4_u8, sPt4_u8, sSFdSt_u8, sSFPt_u8, sSFdSt_mx, sSFQt_mx,
                    S_loaded_mbar_ptr,
                    pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP,
                    softmax_scale_log2, ds_mul, sr_seed, block_info, SeqlenInfoCls, TileSchedulerCls,
                    sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK, tiled_copy_r2s_dKV, dk_scale, dv_scale,
                )
                tmem_alloc_barrier.arrive()
            if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_reduce)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                self.dQacc_reduce(mdQaccum, sdQ, tma_atom_dQ, thr_mma_dQh, tdQtdQ, pipeline_dQ, block_info, SeqlenInfoCls,
                                  TileSchedulerCls, dq_shift)
                tmem_alloc_barrier.arrive()
        return

    @cute.jit
    def _ovf_tile_end(self, role_bar, pipes, S_loaded_mbar_ptr, ktb_mbar):
        """(spd4, the packed overflow launch, trace-time inline) the tile is done in every role: fresh barriers for the
        next one -- a role-warp sync, warp 0 re-initialises every mbarrier, a second sync."""
        role_bar.arrive_and_wait()
        for p_ in pipes:
            p_.sync_object_full.mbarrier_init()
            p_.sync_object_empty.mbarrier_init()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(S_loaded_mbar_ptr, len(self.compute_warp_ids))
                cute.arch.mbarrier_init(ktb_mbar, 1)
        cute.arch.mbarrier_init_fence()
        role_bar.arrive_and_wait()

    # ------------------------------------------------------------------ load warp
    @cute.jit
    def _load_noop(self, **kwargs):
        pass

    @cute.jit
    def _load_sf_stage(self, tma_atom, g, s, src_idx, dst_idx, tma_bar_ptr):
        cute.copy(tma_atom, g[None, src_idx], s[None, dst_idx], tma_bar_ptr=tma_bar_ptr)

    def _sfb_loader(self, tma_atom, thr_mma_sfb, mSF_cur, sSF, offsets):
        """B-side scale factors (128-row SF tiles): gmem tile at the dynamic index -> smem stage."""
        one = cute.make_layout(1)
        g = cute.local_tile(mSF_cur, (128, 128), offsets)
        ts, tg = cpasync.tma_partition(tma_atom, 0, one, cute.group_modes(sSF, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(g), 0, 3))
        return partial(self._load_sf_stage, tma_atom, cute.filter_zeros(tg), cute.filter_zeros(ts))

    def _q_loaders(self, hq, batch_idx, thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_sfb,
                   mQ4, mdO4, mQt4, mdOt4, mSFQ, mSFdO, mSFQt, mSFdOt, mFrame, mUinv, mD,
                   sQ4, sdO4, sQt4, sdOt4, sSFQ, sSFdO, sSFQt, sSFdOt,
                   tma_atom_Q4, tma_atom_dO4, tma_atom_Qt4, tma_atom_dOt4, tma_atom_SFQ, tma_atom_SFdO, tma_atom_SFQt, tma_atom_SFdOt):
        """query-side loaders of q head hq (trace-time Python): A operands Q4 / dO4 (+ A-side SFs) of S / dP, B operands Q^T / dO^T
        (+ B-side SFs) of dK / dV, and the per-query stats tiles."""
        one = cute.make_layout(1)
        gQ4 = cute.local_tile(mQ4[None, None, hq, batch_idx], cute.select(self.mma_tiler_S, mode=[0, 2]), (None, 0))
        lQ4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q4, 0, one, thr_mma_S.partition_A(gQ4), sQ4)
        gSFQ = cute.local_tile(mSFQ[None, None, hq, batch_idx], cute.select(self.mma_tiler_S, mode=[0, 2]), (None, 0))
        lSFQ, _, _ = copy_utils.tma_get_copy_fn(tma_atom_SFQ, 0, one, thr_mma_S.partition_A(gSFQ), sSFQ, filter_zeros=True)
        gdO4 = cute.local_tile(mdO4[None, None, hq, batch_idx], cute.select(self.mma_tiler_dP, mode=[0, 2]), (None, 0))
        ldO4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO4, 0, one, thr_mma_dP.partition_A(gdO4), sdO4)
        gSFdO = cute.local_tile(mSFdO[None, None, hq, batch_idx], cute.select(self.mma_tiler_dP, mode=[0, 2]), (None, 0))
        lSFdO, _, _ = copy_utils.tma_get_copy_fn(tma_atom_SFdO, 0, one, thr_mma_dP.partition_A(gSFdO), sSFdO, filter_zeros=True)
        gQt = cute.local_tile(mQt4[None, None, hq, batch_idx], cute.select(self.mma_tiler_dK, mode=[1, 2]), (0, None))
        lQt4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Qt4, 0, one, thr_mma_dK.partition_B(gQt), sQt4)
        lSFQt = self._load_noop if self.dkv8 else self._sfb_loader(tma_atom_SFQt, thr_mma_sfb, mSFQt[None, None, hq, batch_idx], sSFQt, (0, None))
        gdOt = cute.local_tile(mdOt4[None, None, hq, batch_idx], cute.select(self.mma_tiler_dV, mode=[1, 2]), (0, None))
        ldOt4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dOt4, 0, one, thr_mma_dV.partition_B(gdOt), sdOt4)
        lSFdOt = self._load_noop if self.dkv8 else self._sfb_loader(tma_atom_SFdOt, thr_mma_sfb, mSFdOt[None, None, hq, batch_idx], sSFdOt, (0, None))
        gF = cute.local_tile(mFrame[None, hq, batch_idx], (self.tile_m,), (None,))
        gU = cute.local_tile(mUinv[None, hq, batch_idx], (self.tile_m,), (None,))
        gD = cute.local_tile(mD[None, hq, batch_idx], (self.tile_m,), (None,))
        return (lQ4, lSFQ, lQt4, lSFQt), (ldO4, lSFdO, ldOt4, lSFdOt), gF, gU, gD

    @cute.jit
    def load(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_dQh,
        mQ4, mK4, mdO4, mV4, mKt4, mQt4, mdOt4, mSFQ, mSFK, mSFdO, mSFV, mSFKt, mSFQt, mSFdOt, mFrame, mUinv, mD,
        mSFKt1, tma_atom_SFKt1, sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sFrame, sUinv, sD,
        tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Kt4, tma_atom_Qt4, tma_atom_dOt4,
        tma_atom_SFQ, tma_atom_SFK, tma_atom_SFdO, tma_atom_SFV, tma_atom_SFKt, tma_atom_SFQt, tma_atom_SFdOt,
        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum, block_info, SeqlenInfoCls, TileSchedulerCls,
        # diagonal-only: the diagonal payload's V tile and SF, loaded on the same barrier as V
        mV4b=None, mSFVb=None, sV4b=None, sSFVb=None, tma_atom_V4b=None, tma_atom_SFVb=None,
        mKt4b=None, mSFKtb=None, sKtb=None, sSFKtb=None, tma_atom_Kt4b=None, tma_atom_SFKtb=None, ktb_mbar=None,
    ):
        producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        one = cute.make_layout(1)
        thr_mma_sfb = self.tiled_mma_sfb.get_slice(0)
        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        copy_stats = partial(cute.copy, copy_atom_stats)
        r_q = self.qhead_per_kvhead
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            if const_expr(not self.doc_varlen) or m_block_min < m_block_max:   # documents: a key tile with no query (padding / an empty document) does nothing
                # diag_k: iteration 0 (m_block_min) is the pair the causal boundary cuts -- unless s_k > s_q and this key tile
                # lies wholly before the diagonal, in which case every query sees it and the plain K^T is sound
                cut0_i = Int32(n_loc >= block_info.get_n_block_min_causal_local_mask(seqlen, self._loc_m(m_block_min, m_off), Int32(0)))
                # KV-side (static per work tile): K4 / V4 (B of S / dP), K^T halves (B of dQ), their B-side SFs
                gK4 = cute.local_tile(mK4[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_S, mode=[1, 2]), (n_block, 0))
                load_K4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K4, 0, one, thr_mma_S.partition_B(gK4), sK4, single_stage=True)
                gV4 = cute.local_tile(mV4[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dP, mode=[1, 2]), (n_block, 0))
                load_V4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4, 0, one, thr_mma_dP.partition_B(gV4), sV4, single_stage=True)
                gV4b = cute.local_tile(mV4b[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dP, mode=[1, 2]), (n_block, 0))
                load_V4b, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4b, 0, one, thr_mma_dP.partition_B(gV4b), sV4b, single_stage=True)
                load_SFVb = self._sfb_loader(tma_atom_SFVb, thr_mma_sfb, mSFVb[None, None, head_idx, batch_idx], sSFVb, (None, 0))
                gKt0 = cute.local_tile(mKt4[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dQh, mode=[1, 2]), (0, None))
                gKt1 = cute.local_tile(mKt4[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dQh, mode=[1, 2]), (1, None))
                load_Kt4_0, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Kt4, 0, one, thr_mma_dQh.partition_B(gKt0), sKt4)
                load_Kt4_1, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Kt4, 0, one, thr_mma_dQh.partition_B(gKt1), sKt4)
                load_SFK = self._sfb_loader(tma_atom_SFK, thr_mma_sfb, mSFK[None, None, head_idx, batch_idx], sSFK, (None, 0))
                load_SFV = self._sfb_loader(tma_atom_SFV, thr_mma_sfb, mSFV[None, None, head_idx, batch_idx], sSFV, (None, 0))
                load_SFKt = self._sfb_loader(tma_atom_SFKt, thr_mma_sfb, mSFKt[None, None, head_idx, batch_idx], sSFKt, (0, None))
                load_SFKt1 = self._sfb_loader(tma_atom_SFKt1, thr_mma_sfb, mSFKt1[None, None, head_idx, batch_idx], sSFKt, (0, None))
                gKtb0 = cute.local_tile(mKt4b[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dQh, mode=[1, 2]), (0, None))
                gKtb1 = cute.local_tile(mKt4b[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_dQh, mode=[1, 2]), (1, None))
                load_Ktb_0, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Kt4b, 0, one, thr_mma_dQh.partition_B(gKtb0), sKtb)
                load_Ktb_1, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Kt4b, 0, one, thr_mma_dQh.partition_B(gKtb1), sKtb)
                load_SFKtb = self._sfb_loader(tma_atom_SFKtb, thr_mma_sfb, mSFKtb[None, None, head_idx, batch_idx], sSFKtb, (0, None))
                qa = (thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_sfb, mQ4, mdO4, mQt4, mdOt4, mSFQ, mSFdO, mSFQt, mSFdOt,
                      mFrame, mUinv, mD, sQ4, sdO4, sQt4, sdOt4, sSFQ, sSFdO, sSFQt, sSFdOt,
                      tma_atom_Q4, tma_atom_dO4, tma_atom_Qt4, tma_atom_dOt4, tma_atom_SFQ, tma_atom_SFdO, tma_atom_SFQt, tma_atom_SFdOt)
                nm = m_block_max - m_block_min
                m0 = m_block_min
                (lQ4, lSFQ, lQt4, lSFQt), (ldO4, lSFdO, ldOt4, lSFdOt), gF, gU, gD = self._q_loaders(head_idx * r_q, batch_idx, *qa)
                pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                load_K4(tma_bar_ptr=bar)
                load_SFK(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                load_Kt4_0(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                load_Kt4_1(src_idx=n_block, dst_idx=1, tma_bar_ptr=bar)
                load_SFKt(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                load_SFKt1(src_idx=n_block, dst_idx=1, tma_bar_ptr=bar)
                lQ4(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                lSFQ(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                lQt4(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                lSFQt(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                pipeline_Q.producer_commit(producer_state_Q)
                pipeline_LSE.producer_acquire(producer_state_Q)
                b_lse = pipeline_LSE.producer_get_barrier(producer_state_Q)
                with cute.arch.elect_one():
                    copy_stats(gF[None, m0], sFrame[None, producer_state_Q.index], mbar_ptr=b_lse)
                    copy_stats(gU[None, m0], sUinv[None, producer_state_Q.index], mbar_ptr=b_lse)
                producer_state_Q.advance()
                pipeline_dO.producer_acquire(producer_state_dO, extra_tx_count=self.tma_copy_bytes["V"])
                bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                if const_expr(self.diag_k):
                    # the cut pair's key-local K^T payload into the V tile (same bytes); plain V follows after dQ(0)
                    if cut0_i != 0:
                        load_Ktb_0(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                        load_Ktb_1(src_idx=n_block, dst_idx=1, tma_bar_ptr=bar)
                        load_SFKtb(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                    else:
                        load_V4(tma_bar_ptr=bar)
                        load_SFV(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                else:
                    load_V4(tma_bar_ptr=bar)
                    load_SFV(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                if const_expr(self.vtok_dp_diag):
                    load_V4b(tma_bar_ptr=bar)
                    load_SFVb(src_idx=n_block, dst_idx=0, tma_bar_ptr=bar)
                ldO4(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                lSFdO(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                ldOt4(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                lSFdOt(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                pipeline_dO.producer_commit(producer_state_dO)
                pipeline_dPsum.producer_acquire(producer_state_dO)
                with cute.arch.elect_one():
                    copy_stats(gD[None, m0], sD[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                producer_state_dO.advance()
                # diag_k: the V reload goes in after query tile min(2, nm - 1)'s loads -- early enough for the first plain dP
                # (iteration 1 at the soonest) and late enough that the wait for dQ(0) cannot block a load dQ(0) depends on
                jr = cutlass.min(Int32(2), nm - 1)
                for j in cutlass.range(1, r_q * nm, unroll=1):
                    qh = j // nm
                    m_blk = m_block_min + (j - qh * nm)
                    (jQ4, jSFQ, jQt4, jSFQt), (jdO4, jSFdO, jdOt4, jSFdOt), jF, jU, jD = self._q_loaders(head_idx * r_q + qh, batch_idx, *qa)
                    pipeline_Q.producer_acquire(producer_state_Q)
                    bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                    jQ4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    jSFQ(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    jQt4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    jSFQt(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    pipeline_Q.producer_commit(producer_state_Q)
                    pipeline_LSE.producer_acquire(producer_state_Q)
                    b_lse = pipeline_LSE.producer_get_barrier(producer_state_Q)
                    with cute.arch.elect_one():
                        copy_stats(jF[None, m_blk], sFrame[None, producer_state_Q.index], mbar_ptr=b_lse)
                        copy_stats(jU[None, m_blk], sUinv[None, producer_state_Q.index], mbar_ptr=b_lse)
                    producer_state_Q.advance()
                    pipeline_dO.producer_acquire(producer_state_dO)
                    bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                    jdO4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    jSFdO(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    jdOt4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    jSFdOt(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    pipeline_dO.producer_commit(producer_state_dO)
                    pipeline_dPsum.producer_acquire(producer_state_dO)
                    with cute.arch.elect_one():
                        copy_stats(jD[None, m_blk], sD[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                    producer_state_dO.advance()
                    if const_expr(self.diag_k):
                        if cut0_i != 0:
                            if j == jr:
                                cute.arch.mbarrier_wait(ktb_mbar, 0)          # dQ(0) has read the K^T payload
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(ktb_mbar + 1, self.tma_copy_bytes["V1"])
                                load_V4(tma_bar_ptr=ktb_mbar + 1)
                                load_SFV(src_idx=n_block, dst_idx=0, tma_bar_ptr=ktb_mbar + 1)
                pipeline_Q.producer_tail(producer_state_Q.clone())
                pipeline_LSE.producer_tail(producer_state_Q)
                pipeline_dO.producer_tail(producer_state_dO.clone())
                pipeline_dPsum.producer_tail(producer_state_dO)
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ load warp, packed rows (cov5)
    @cute.jit
    def _pk_rows(self, m, off, tiler_mn, tile_idx):
        """The (rows, cols) gmem tile of an operand whose rows are the packed rows: rows off + 128 t of tile index t."""
        return cute.local_tile(cute.domain_offset((off,) + (0,) * (cute.rank(m) - 1), m), tiler_mn, tile_idx)

    @cute.jit
    def _pk_publish(self, full):
        """The load warp's second arrival on a stage it wrote SF atoms / K^T into: proxy fence, warp sync, one arrive."""
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(full)

    @cute.jit
    def load_packed(
        self, thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, mQ4, mK4, mdO4, mV4, mQt4, mdOt4, mFrame, mUinv, mD,
        sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sSFQ, sSFK, sSFdO, sSFV, sSFKt, sFrame, sUinv, sD,
        tma_atom_Q4, tma_atom_K4, tma_atom_dO4, tma_atom_V4, tma_atom_Qt4, tma_atom_dOt4,
        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum, block_info, SeqlenInfoCls, TileSchedulerCls,
        mV4b, sV4b, sSFVb, tma_atom_V4b,
    ):
        """(cov5 codes_packed) load(): every row tile at the document's packed rows (the aligned tile numbering of the maps:
        tile t of a document whose first aligned tile is t0 starts at its packed start + 128 (t - t0)); the SF atoms of Q /
        K / V / dO built here from the dense per-row words (rows past the document's real ones 0), K^T realigned here from
        its key-parity copies, its SF and the diagonal payload's (route 'all') the tiles' constant words; the statistics
        from 16-byte-aligned 132-float windows (the compute warps read them at the window's offset). The SF / K^T stores
        are generic writes: the stage's full barrier takes a second arrival after them (_pk_publish)."""
        from flash_attn.cute import sfd_atoms as SA
        from flash_attn.cute import v4_packed as V4
        pk = self._pk
        mSF4Q, mSF4K, mSF4dO, mSFVw, mSFVbw, mKT, mKTw = pk[2], pk[3], pk[4], pk[5], pk[6], pk[7], pk[8]
        producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        one = cute.make_layout(1)
        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        copy_stats = partial(cute.copy, copy_atom_stats)
        r_q = self.qhead_per_kvhead
        SR = self.stats_rows
        nk4 = 2
        sfq_b = cute.size_in_bytes(self.sf_dtype, cute.select(self.sfq_layout, mode=[0, 1, 2]))
        sfdo_b = cute.size_in_bytes(self.sf_dtype, cute.select(self.sfdo_layout, mode=[0, 1, 2]))
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            if m_block_min < m_block_max:
                psq = seqlen.offset_q                          # (documents: the packed starts)
                psk = seqlen.offset_k
                qoff = psq - m_off * self.tile_m               # packed row of aligned query tile m: qoff + 128 m
                kps = psk + n_loc * self.tile_n                # this key tile's first packed row
                koff = kps - n_block * self.tile_n
                nr_k = seqlen.seqlen_k - n_loc * self.tile_n
                gK4 = self._pk_rows(mK4[None, None, head_idx, batch_idx], koff, cute.select(self.mma_tiler_S, mode=[1, 2]),
                                    (n_block, 0))
                load_K4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K4, 0, one, thr_mma_S.partition_B(gK4), sK4,
                                                           single_stage=True)
                gV4 = self._pk_rows(mV4[None, None, head_idx, batch_idx], koff, cute.select(self.mma_tiler_dP, mode=[1, 2]),
                                    (n_block, 0))
                load_V4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4, 0, one, thr_mma_dP.partition_B(gV4), sV4,
                                                           single_stage=True)
                gV4b = self._pk_rows(mV4b[None, None, head_idx, batch_idx], koff,
                                     cute.select(self.mma_tiler_dP, mode=[1, 2]), (n_block, 0))
                load_V4b, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4b, 0, one, thr_mma_dP.partition_B(gV4b), sV4b,
                                                            single_stage=True)
                nm = m_block_max - m_block_min
                m0 = m_block_min
                c_par = psk & 1
                for j in cutlass.range(r_q * nm, unroll=1):
                    qh = j // nm
                    m_blk = m_block_min + (j - qh * nm)
                    hq = head_idx * r_q + qh
                    mrow = qoff + m_blk * self.tile_m          # this query tile's first packed row
                    nr_q = seqlen.seqlen_q - (m_blk - m_off) * self.tile_m
                    gQ4 = self._pk_rows(mQ4[None, None, hq, batch_idx], qoff, cute.select(self.mma_tiler_S, mode=[0, 2]),
                                        (None, 0))
                    lQ4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q4, 0, one, thr_mma_S.partition_A(gQ4), sQ4)
                    gdO4 = self._pk_rows(mdO4[None, None, hq, batch_idx], qoff,
                                         cute.select(self.mma_tiler_dP, mode=[0, 2]), (None, 0))
                    ldO4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO4, 0, one, thr_mma_dP.partition_A(gdO4), sdO4)
                    # (dkv8: the E4M3 Q / dO rows, (d, s) views: the row offset on mode 1)
                    gQt = cute.local_tile(cute.domain_offset((0, qoff), mQt4[None, None, hq, batch_idx]),
                                          cute.select(self.mma_tiler_dK, mode=[1, 2]), (0, None))
                    lQt4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Qt4, 0, one, thr_mma_dK.partition_B(gQt), sQt4)
                    gdOt = cute.local_tile(cute.domain_offset((0, qoff), mdOt4[None, None, hq, batch_idx]),
                                           cute.select(self.mma_tiler_dV, mode=[1, 2]), (0, None))
                    ldOt4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dOt4, 0, one, thr_mma_dV.partition_B(gdOt), sdOt4)
                    # ---- Q stage (+ K, K^T, their SF on the work tile's first)
                    rq = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
                    SA.load_sf4(mSF4Q, hq, mrow, nk4, rq)
                    if j == 0:
                        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                    else:
                        pipeline_Q.producer_acquire(producer_state_Q)
                    bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                    if j == 0:
                        load_K4(tma_bar_ptr=bar)
                    lQ4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lQt4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    if j == 0:
                        rk = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
                        SA.load_sf4(mSF4K, head_idx, kps, nk4, rk)
                        SA.store_sf4_atoms(rk, nr_k, nk4, sSFK.iterator.toint())
                        # K^T (the dQ operand): realigned from the key-parity copy, its SF the tile's constant word
                        V4.realign_kt_gmem(mKT[c_par, head_idx, 0, None].iterator, mKT.shape[3], (kps + c_par) >> 1,
                                           nr_k, sKt4.iterator.toint())
                        if const_expr(self.pk_kt_groups):
                            # (cov6, key channel 'open') the plain K^T's SF atoms from the packed group table
                            V4.store_kt_group_sf(mKTw[head_idx, None].iterator, (kps >> 4) + n_block, nr_k,
                                                 sSFKt.iterator.toint())
                        else:
                            V4.fill_words(mKTw[head_idx, n_block], sSFKt.iterator.toint(),
                                          cute.size_in_bytes(self.sf_dtype, self.sfkt_layout))
                    SA.store_sf4_atoms(rq, nr_q, nk4, sSFQ.iterator.toint() + producer_state_Q.index * sfq_b)
                    self._pk_publish(bar)
                    pipeline_LSE.producer_acquire(producer_state_Q)
                    b_lse = pipeline_LSE.producer_get_barrier(producer_state_Q)
                    w0 = mrow & ~3
                    with cute.arch.elect_one():
                        copy_stats(cute.make_tensor(mFrame[None, hq, batch_idx].iterator + w0, cute.make_layout(SR)),
                                   sFrame[None, producer_state_Q.index], mbar_ptr=b_lse)
                        copy_stats(cute.make_tensor(mUinv[None, hq, batch_idx].iterator + w0, cute.make_layout(SR)),
                                   sUinv[None, producer_state_Q.index], mbar_ptr=b_lse)
                    producer_state_Q.advance()
                    # ---- dO stage (+ V, the diagonal payload, their SF on the work tile's first)
                    rd = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
                    SA.load_sf4(mSF4dO, hq, mrow, nk4, rd)
                    if j == 0:
                        pipeline_dO.producer_acquire(producer_state_dO, extra_tx_count=self.tma_copy_bytes["V"])
                    else:
                        pipeline_dO.producer_acquire(producer_state_dO)
                    bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                    if j == 0:
                        load_V4(tma_bar_ptr=bar)
                        if const_expr(self.vtok_dp_diag):
                            load_V4b(tma_bar_ptr=bar)
                    ldO4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    ldOt4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    if j == 0:
                        rv = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
                        SA.load_sf4(mSFVw, head_idx, kps, nk4, rv)
                        SA.store_sf4_atoms(rv, nr_k, nk4, sSFV.iterator.toint())
                        if const_expr(self.vtok_dp_diag):
                            if const_expr(self.diag_fold):
                                rb = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
                                SA.load_sf4(mSFVbw, head_idx, kps, nk4, rb)
                                SA.store_sf4_atoms(rb, nr_k, nk4, sSFVb.iterator.toint())
                            else:
                                # (the diagonal payload's SF: the tile's constant 2^c_n in every byte -- rows past the
                                # document too, as the aligned layout's: its zero rows keep the tile constant)
                                V4.fill_words(mSFVbw[head_idx, n_block], sSFVb.iterator.toint(),
                                              cute.size_in_bytes(self.sf_dtype, self.sfv_layout))
                    SA.store_sf4_atoms(rd, nr_q, nk4, sSFdO.iterator.toint() + producer_state_dO.index * sfdo_b)
                    self._pk_publish(bar)
                    pipeline_dPsum.producer_acquire(producer_state_dO)
                    with cute.arch.elect_one():
                        copy_stats(cute.make_tensor(mD[None, hq, batch_idx].iterator + w0, cute.make_layout(SR)),
                                   sD[None, producer_state_dO.index],
                                   mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                    producer_state_dO.advance()
                pipeline_Q.producer_tail(producer_state_Q.clone())
                pipeline_LSE.producer_tail(producer_state_Q)
                pipeline_dO.producer_tail(producer_state_dO.clone())
                pipeline_dPsum.producer_tail(producer_state_dO)
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ MMA warp
    @cute.jit
    def s2t(self, tiled_copy, src, dst):
        cute.copy(tiled_copy, src, dst)

    def s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def _loc_m(self, m, m_off):
        """A query tile index in the coordinates of the mask: the document's under documents (m_off its first tile),
        the index itself batched (m_off None: no instruction)."""
        return m if m_off is None else m - m_off

    @cute.jit
    def _tile_range(self, block_info, SeqlenInfoCls, batch_idx, n_block):
        """(seqlen, m_block_min, m_block_max, m_off, n_loc) of key tile n_block: m_block_min / _max global query tiles;
        seqlen, n_loc in the mask's coordinates (lp_bwd_tiles.LPTileInfo under documents). Batched: m_off None and n_loc
        n_block (the former code exactly)."""
        if const_expr(self.doc_varlen):
            return self._lpt.get(batch_idx, n_block)
        else:
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            return seqlen, m_block_min, m_block_max, None, n_block

    def _diag_m(self, m):
        """diagonal-only: the first query tile of the forward CTA that holds query tile m -- the m the forward's masked-step
        test ran with (`diagonal-pairs-are-the-forwards-cta`). Plain Python, inlined at trace; the identity when the
        forward's CTA is the backward's tile, so that kernel is unchanged."""
        if self.diag_fwd_tiles == 1:
            return m
        return m - m % Int32(self.diag_fwd_tiles)

    def _s2t_fns(self, sSF, tCtSF, num_stages):
        cp, src, dst = self.s2t_copy_and_partition(sSF, tCtSF)
        return [partial(self.s2t, cp, src[None, None, None, None, s], dst) for s in range(num_stages)]

    @cute.jit
    def gemm_fp4(self, op, acc_offset: cutlass.Constexpr[int], tCrA, tCrB, sA, sB, tScaleA, tScaleB, zero_init,
                 ta_addr: cutlass.Constexpr[int] = -1, sfb_col_offset: cutlass.Constexpr[int] = 0, gname: cutlass.Constexpr[str] = ""):
        sm100_utils.gemm_ptx_partial_fp4(
            op, acc_offset, tCrA, tCrB, sA=sA, sB=sB, tScaleA=tScaleA, tScaleB=tScaleB, zero_init=zero_init,
            tA_addr=(ta_addr if const_expr(ta_addr >= 0) else None),
            sfb_col_offset=(Int32(sfb_col_offset) if const_expr(sfb_col_offset != 0) else None),
        )

    @cute.jit
    def mma(
        self,
        sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sdSt4, sPt4,
        sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sSFdSt, sSFPt,
        tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ, tdSq,
        tCtSFQ, tCtSFK, tCtSFdO, tCtSFV, tCtSFdSq, tCtSFKt, tCtSFdSt, tCtSFQt, tCtSFPt, tCtSFdOt,
        pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP, pipeline_dQ,
        S_loaded_mbar_ptr, block_info, SeqlenInfoCls, TileSchedulerCls, tCtSFKt128=None,
        # diagonal-only is dkv8-only (mma_kv8 takes the diagonal payload); accepted here so one call site serves both
        sV4b=None, sSFVb=None, tCtSFVb=None, sKtb=None, sSFKtb=None, ktb_mbar=None,
    ):
        tSrQ4 = self.tiled_mma_S.make_fragment_A(sQ4)
        tSrK4 = self.tiled_mma_S.make_fragment_B(sK4)
        tPrdO4 = self.tiled_mma_dP.make_fragment_A(sdO4)
        tPrV4 = self.tiled_mma_dP.make_fragment_B(sV4)
        tQrdS = self.tiled_mma_dQh.make_fragment_A(tdSq)
        tQrKt = self.tiled_mma_dQh.make_fragment_B(sKt4)
        tKrdS = self.tiled_mma_dK.make_fragment_A(sdSt4)
        tKrQt = self.tiled_mma_dK.make_fragment_B(sQt4)
        tVrP = self.tiled_mma_dV.make_fragment_A(sPt4)
        tVrdOt = self.tiled_mma_dV.make_fragment_B(sdOt4)
        s2t_Q = self._s2t_fns(sSFQ, tCtSFQ, self.Q_stage)
        s2t_K = self._s2t_fns(sSFK, tCtSFK, 1)[0]
        s2t_dO = self._s2t_fns(sSFdO, tCtSFdO, self.dO_stage)
        s2t_V = self._s2t_fns(sSFV, tCtSFV, 1)[0]
        s2t_Kt = self._s2t_fns(sSFKt, tCtSFKt, 2)
        s2t_Qt = self._s2t_fns(sSFQt, tCtSFQt, self.Q_stage)
        s2t_dOt = self._s2t_fns(sSFdOt, tCtSFdOt, self.dO_stage)
        s2t_dSt = self._s2t_fns(sSFdSt, tCtSFdSt, 1)[0]
        s2t_Pt = self._s2t_fns(sSFPt, tCtSFPt, 1)[0]
        op_S, op_dP, op_dQ, op_dK, op_dV = (self.tiled_mma_S.op, self.tiled_mma_dP.op, self.tiled_mma_dQh.op,
                                             self.tiled_mma_dK.op, self.tiled_mma_dV.op)
        st = lambda t, i: t[None, None, None, i]
        mma_S = [partial(self.gemm_fp4, op_S, self.tmem_S_offset, st(tSrQ4, i), tSrK4, st(sQ4, i), sK4, tCtSFQ, tCtSFK, True, gname="S")
                 for i in range(self.Q_stage)]
        mma_dP = [partial(self.gemm_fp4, op_dP, self.tmem_dP_offset, st(tPrdO4, i), tPrV4, st(sdO4, i), sV4, tCtSFdO, tCtSFV, True, gname="dP")
                  for i in range(self.dO_stage)]
        if const_expr(self.dq8):
            mma_dQ = [partial(self.gemm_e4m3, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), tdSq, st(sKt4, h), True,
                              gname="dQ") for h in range(2)]
        elif const_expr(self.dq8):
            mma_dQ = [partial(self.gemm_e4m3, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), None, st(sKt4, h), True,
                              ta_addr=self.tmem_dSq_offset, gname="dQ") for h in range(2)]
        else:
            mma_dQ = [partial(self.gemm_fp4, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), None, st(sKt4, h), tCtSFdSq, tCtSFKt, True,
                              ta_addr=self.tmem_dSq_offset, gname="dQ") for h in range(2)]
        mma_dK = [partial(self.gemm_fp4, op_dK, self.tmem_dK_offset, tKrdS, st(tKrQt, i), sdSt4, st(sQt4, i), tCtSFdSt, tCtSFQt, gname="dK")
                  for i in range(self.Q_stage)]
        mma_dV = [partial(self.gemm_fp4, op_dV, self.tmem_dV_offset, tVrP, st(tVrdOt, i), sPt4, st(sdOt4, i), tCtSFPt, tCtSFdOt, gname="dV")
                  for i in range(self.dO_stage)]

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        pipeline_dO_consumer = pipeline_dO.make_consumer()
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        consumer_state_dSkv = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_phase_acc = Int32(1)
        producer_phase_dKV = Int32(1)
        phase_dPe = Int32(0)
        phase_dQe = Int32(0)
        phase_Sl = Int32(0)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            accumulate = False
            # ---- prologue: S(0); the compute warps' first dS_kv commit is a placeholder (no query tile before 0); dP(0)
            handle_Q = pipeline_Q_consumer.wait_and_advance()
            sm100_utils.tcgen05_after_thread_sync()
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            s2t_K()
            if handle_Q.index == 0:
                s2t_Q[0]()
                mma_S[0]()
            else:
                s2t_Q[1]()
                mma_S[1]()
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            handle_dO = pipeline_dO_consumer.wait_and_advance()
            cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
            phase_Sl ^= 1
            pipeline_dSkv.consumer_wait(consumer_state_dSkv)
            pipeline_dSkv.consumer_release(consumer_state_dSkv)
            consumer_state_dSkv.advance()
            sm100_utils.tcgen05_after_thread_sync()
            s2t_V()
            if handle_dO.index == 0:
                s2t_dO[0]()
                mma_dP[0]()
            else:
                s2t_dO[1]()
                mma_dP[1]()
            pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
            producer_phase_acc ^= 1
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S phase of query tile 0 done
            main_loop_iters = (m_block_max - m_block_min) * self.qhead_per_kvhead - 1
            handle_Q_next = handle_Q
            handle_dO_next = handle_dO
            for it in cutlass.range(main_loop_iters, unroll=1):
                # (1) S(t+1) once the compute warps have loaded dP(t)
                handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                pipeline_dP.sync_object_empty.wait(0, phase_dPe)
                phase_dPe ^= 1
                sm100_utils.tcgen05_after_thread_sync()
                s2t_K()
                if handle_Q_next.index == 0:
                    s2t_Q[0]()
                    mma_S[0]()
                else:
                    s2t_Q[1]()
                    mma_S[1]()
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                # (2) dQ(t) from the TMEM dS^ codes: two d halves into the same 64 columns, drained in between
                pipeline_dS.consumer_wait(consumer_state_dS)
                sm100_utils.tcgen05_after_thread_sync()
                s2t_Kt[0]()
                mma_dQ[0]()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                sm100_utils.tcgen05_after_thread_sync()
                s2t_Kt[1]()
                mma_dQ[1]()
                pipeline_dQ.sync_object_full.arrive(1, pipeline_dQ.producer_mask, cta_group)
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()
                # (3) dP(t+1)
                handle_dO_next = pipeline_dO_consumer.wait_and_advance()
                cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                phase_Sl ^= 1
                pipeline_dQ.sync_object_empty.wait(1, phase_dQe)
                phase_dQe ^= 1
                sm100_utils.tcgen05_after_thread_sync()
                s2t_V()
                if handle_dO_next.index == 0:
                    s2t_dO[0]()
                    mma_dP[0]()
                else:
                    s2t_dO[1]()
                    mma_dP[1]()
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                # (4) dK(t), dV(t) from the SMEM operands the compute warps committed after loading S(t+1)
                pipeline_dSkv.consumer_wait(consumer_state_dSkv)
                sm100_utils.tcgen05_after_thread_sync()
                s2t_dSt()
                if handle_Q.index == 0:
                    s2t_Qt[0]()
                    mma_dK[0](zero_init=not accumulate)
                else:
                    s2t_Qt[1]()
                    mma_dK[1](zero_init=not accumulate)
                s2t_Pt()
                if handle_dO.index == 0:
                    s2t_dOt[0]()
                    mma_dV[0](zero_init=not accumulate)
                else:
                    s2t_dOt[1]()
                    mma_dV[1](zero_init=not accumulate)
                accumulate = True
                handle_Q.release()
                handle_dO.release()
                pipeline_dSkv.consumer_release(consumer_state_dSkv)
                consumer_state_dSkv.advance()
                # (5) the S phase of query tile t+1 done
                producer_phase_acc ^= 1
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                handle_Q = handle_Q_next
                handle_dO = handle_dO_next

            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            # ---- tail: dQ(last), then dV(last), dK(last) with the epilogue hand-offs
            pipeline_dP.sync_object_empty.wait(0, phase_dPe)
            phase_dPe ^= 1
            pipeline_dS.consumer_wait(consumer_state_dS)
            sm100_utils.tcgen05_after_thread_sync()
            s2t_Kt[0]()
            mma_dQ[0]()
            pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
            pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
            sm100_utils.tcgen05_after_thread_sync()
            s2t_Kt[1]()
            mma_dQ[1]()
            pipeline_dQ.sync_object_full.arrive(1, pipeline_dQ.producer_mask, cta_group)
            pipeline_dS.consumer_release(consumer_state_dS)
            consumer_state_dS.advance()
            pipeline_dQ.sync_object_empty.wait(1, phase_dQe)
            phase_dQe ^= 1
            pipeline_dSkv.consumer_wait(consumer_state_dSkv)
            sm100_utils.tcgen05_after_thread_sync()
            s2t_Pt()
            if handle_dO.index == 0:
                s2t_dOt[0]()
                mma_dV[0](zero_init=not accumulate)
            else:
                s2t_dOt[1]()
                mma_dV[1](zero_init=not accumulate)
            handle_dO.release()
            pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
            pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
            pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
            sm100_utils.tcgen05_after_thread_sync()
            s2t_dSt()
            if handle_Q.index == 0:
                s2t_Qt[0]()
                mma_dK[0](zero_init=not accumulate)
            else:
                s2t_Qt[1]()
                mma_dK[1](zero_init=not accumulate)
            pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
            producer_phase_dKV ^= 1
            handle_Q.release()
            pipeline_dSkv.consumer_release(consumer_state_dSkv)
            consumer_state_dSkv.advance()
            producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def gemm_e4m3(self, op, acc_offset: cutlass.Constexpr[int], tCrA, tCrB, sA, sB, zero_init,
                  ta_addr: cutlass.Constexpr[int] = -1, gname: cutlass.Constexpr[str] = ""):
        sm100_utils.gemm_ptx_partial(op, acc_offset, tCrA, tCrB, sA=sA, sB=sB, zero_init=zero_init,
                                     tA_addr=(ta_addr if const_expr(ta_addr >= 0) else None))

    @cute.jit
    def mma_kv8(
        self,
        sQ4, sK4, sdO4, sV4, sKt4, sQt4, sdOt4, sdSt4, sPt4,
        sSFQ, sSFK, sSFdO, sSFV, sSFKt, sSFQt, sSFdOt, sSFdSt, sSFPt,
        tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ, tdSq,
        tCtSFQ, tCtSFK, tCtSFdO, tCtSFV, tCtSFdSq, tCtSFKt, tCtSFdSt, tCtSFQt, tCtSFPt, tCtSFdOt,
        pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP, pipeline_dQ,
        S_loaded_mbar_ptr, block_info, SeqlenInfoCls, TileSchedulerCls, tCtSFKt128=None,
        # diagonal-only: the diagonal payload's V tile / SF / TMEM SF slot -- the dP operand on the diagonal pair
        sV4b=None, sSFVb=None, tCtSFVb=None,
        # diag_k: the aliased key-local K^T tile / SF and its barrier pair
        sKtb=None, sSFKtb=None, ktb_mbar=None,
    ):
        tSrQ4 = self.tiled_mma_S.make_fragment_A(sQ4)
        tSrK4 = self.tiled_mma_S.make_fragment_B(sK4)
        tPrdO4 = self.tiled_mma_dP.make_fragment_A(sdO4)
        tPrV4 = self.tiled_mma_dP.make_fragment_B(sV4)
        tQrdS = self.tiled_mma_dQh.make_fragment_A(tdSq)
        tQrKt = self.tiled_mma_dQh.make_fragment_B(sKt4)
        tKrdS = self.tiled_mma_dK.make_fragment_A(sdSt4)
        tKrQt = self.tiled_mma_dK.make_fragment_B(sQt4)
        tVrP = self.tiled_mma_dV.make_fragment_A(sPt4)
        tVrdOt = self.tiled_mma_dV.make_fragment_B(sdOt4)
        s2t_Q = self._s2t_fns(sSFQ, tCtSFQ, self.Q_stage)
        s2t_K = self._s2t_fns(sSFK, tCtSFK, 1)[0]
        s2t_dO = self._s2t_fns(sSFdO, tCtSFdO, self.dO_stage)
        s2t_V = self._s2t_fns(sSFV, tCtSFV, 1)[0]
        tPrV4b = self.tiled_mma_dP.make_fragment_B(sV4b)
        s2t_Vb = self._s2t_fns(sSFVb, tCtSFVb, 1)[0]
        s2t_Kt = self._s2t_fns(sSFKt, tCtSFKt, 2)
        op_S, op_dP, op_dQ, op_dK, op_dV = (self.tiled_mma_S.op, self.tiled_mma_dP.op, self.tiled_mma_dQh.op,
                                             self.tiled_mma_dK.op, self.tiled_mma_dV.op)
        st = lambda t, i: t[None, None, None, i]
        # diag_k: dQ(0) of the cut pair against the key-local K^T payload (aliased V tile); one SF stage serves both halves
        tQrKtb = self.tiled_mma_dQh.make_fragment_B(sKtb)
        s2t_Ktb = self._s2t_fns(sSFKtb, tCtSFKt, 1)[0]
        mma_dQb = [partial(self.gemm_fp4, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKtb, h), None, st(sKtb, h), tCtSFdSq, tCtSFKt, True,
                           ta_addr=self.tmem_dSq_offset, gname="dQb") for h in range(2)]
        mma_S = [partial(self.gemm_fp4, op_S, self.tmem_S_offset, st(tSrQ4, i), tSrK4, st(sQ4, i), sK4, tCtSFQ, tCtSFK, True, gname="S")
                 for i in range(self.Q_stage)]
        mma_dP = [partial(self.gemm_fp4, op_dP, self.tmem_dP_offset, st(tPrdO4, i), tPrV4, st(sdO4, i), sV4, tCtSFdO, tCtSFV, True, gname="dP")
                  for i in range(self.dO_stage)]
        # diagonal-only: the same GEMM against the diagonal payload's tile and its own TMEM SF slot
        mma_dP_b = [partial(self.gemm_fp4, op_dP, self.tmem_dP_offset, st(tPrdO4, i), tPrV4b, st(sdO4, i), sV4b,
                            tCtSFdO, tCtSFVb, True, gname="dPb") for i in range(self.dO_stage)]
        if const_expr(self.dq8):
            mma_dQ = [partial(self.gemm_e4m3, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), tdSq, st(sKt4, h), True,
                              gname="dQ") for h in range(2)]
        elif const_expr(self.dq8):
            mma_dQ = [partial(self.gemm_e4m3, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), None, st(sKt4, h), True,
                              ta_addr=self.tmem_dSq_offset, gname="dQ") for h in range(2)]
        else:
            mma_dQ = [partial(self.gemm_fp4, op_dQ, self.tmem_dQ_offset, tQrdS, st(tQrKt, h), None, st(sKt4, h), tCtSFdSq, tCtSFKt, True,
                              ta_addr=self.tmem_dSq_offset, gname="dQ") for h in range(2)]
        if const_expr(self.mxds):
            mma_dK = [partial(sm100_utils.gemm_blockscaled_generic, self.tiled_mma_dK, tdKtdK, tKrdS, st(tKrQt, i),
                              tCtSFdSt, tCtSFQt) for i in range(self.Q_stage)]
            s2t_dSt_mx = self._s2t_fns(sSFdSt, tCtSFdSt, 1)[0]
            s2t_Qt_mx = self._s2t_fns(sSFQt, tCtSFQt, self.Q_stage)
        else:
            mma_dK = [partial(self.gemm_e4m3, op_dK, self.tmem_dK_offset, tKrdS, st(tKrQt, i), sdSt4, st(sQt4, i), gname="dK")
                      for i in range(self.Q_stage)]
        mma_dV = [partial(self.gemm_e4m3, op_dV, self.tmem_dV_offset, tVrP, st(tVrdOt, i), sPt4, st(sdOt4, i), gname="dV")
                  for i in range(self.dO_stage)]

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_phase_acc = Int32(1)
        producer_phase_dKV = Int32(1)
        phase_dPe = Int32(0)
        phase_dQe = Int32(0)
        phase_Sl = Int32(0)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            if const_expr(not self.doc_varlen) or m_block_min < m_block_max:   # documents: a key tile with no query (padding / an empty document) does nothing
                cut0_i = Int32(n_loc >= block_info.get_n_block_min_causal_local_mask(seqlen, self._loc_m(m_block_min, m_off), Int32(0)))  # diag_k
                vwait = cut0_i   # diag_k: the V reload is still to be waited for
                accumulate = False
                # ---- prologue: S(0) [SFQ(0) / SFK parked in the dP bank], dP(0) [SFdO(0) / SFV parked in the S bank]
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                sm100_utils.tcgen05_after_thread_sync()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                s2t_K()
                if handle_Q.index == 0:
                    s2t_Q[0]()
                    mma_S[0]()
                else:
                    s2t_Q[1]()
                    mma_S[1]()
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                pipeline_dO.consumer_wait(consumer_state_dO)
                cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                phase_Sl ^= 1
                sm100_utils.tcgen05_after_thread_sync()
                # diagonal-only: iteration 0 is m_block_min, the smallest m this n_block visits. With s_k == s_q it is always
                # a diagonal pair, but with s_k > s_q the first (s_k - s_q) / 128 key tiles are visible to every query and the
                # forward never biased them, so the pair is tested like every other (one branch per key tile).
                if const_expr(self.vtok_dp_diag):
                    if n_loc >= block_info.get_n_block_min_causal_local_mask(seqlen, self._diag_m(self._loc_m(m_block_min, m_off)), Int32(0)):
                        s2t_Vb()
                        if consumer_state_dO.index == 0:
                            s2t_dO[0]()
                            mma_dP_b[0]()
                        else:
                            s2t_dO[1]()
                            mma_dP_b[1]()
                    else:
                        s2t_V()
                        if consumer_state_dO.index == 0:
                            s2t_dO[0]()
                            mma_dP[0]()
                        else:
                            s2t_dO[1]()
                            mma_dP[1]()
                else:
                    s2t_Vb()
                    if consumer_state_dO.index == 0:
                        s2t_dO[0]()
                        mma_dP_b[0]()
                    else:
                        s2t_dO[1]()
                        mma_dP_b[1]()
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                producer_phase_acc ^= 1
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S phase of query tile 0 done (C3: P(0) stored)
                nm_m = m_block_max - m_block_min   # diagonal-only: the m cycle length per query head
                m_cur = m_block_min                # diagonal-only: the query tile of the iteration just issued (iteration 0)
                main_loop_iters = (m_block_max - m_block_min) * self.qhead_per_kvhead - 1
                handle_Q_next = handle_Q
                dO_idx = consumer_state_dO.index
                for it in cutlass.range(main_loop_iters, unroll=1):
                    # (1) S(t+1) once the compute warps have loaded dP(t)
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    pipeline_dP.sync_object_empty.wait(0, phase_dPe)
                    phase_dPe ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    s2t_K()
                    if handle_Q_next.index == 0:
                        s2t_Q[0]()
                        mma_S[0]()
                    else:
                        s2t_Q[1]()
                        mma_S[1]()
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    # (2) dK(t), dV(t) from the dS^^T / P^v^T SMEM operands the compute warps committed
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    if const_expr(self.mxds):
                        # the MXDS SF parks (SFdSt 80, SFQt 88) sit in the S bank, which holds S(t+1) from the GEMM just
                        # issued: the S2Ts must wait until the compute warps have loaded it (the wait dP(t+1) takes at (4) for
                        # its own S-bank parks, moved up). Without it the S2T overwrote S(t+1) columns 80..95 and dQ / dV / dK
                        # all came back NaN (skipping the S2T kept dQ / dV exact, which located the fault).
                        cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                        phase_Sl ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    if const_expr(self.mxds):
                        s2t_dSt_mx()
                    if handle_Q.index == 0:
                        if const_expr(self.mxds):
                            s2t_Qt_mx[0]()
                        mma_dK[0](zero_init=not accumulate)
                    else:
                        if const_expr(self.mxds):
                            s2t_Qt_mx[1 if const_expr(self.Q_stage > 1) else 0]()
                        mma_dK[1](zero_init=not accumulate)
                    if dO_idx == 0:
                        mma_dV[0](zero_init=not accumulate)
                    else:
                        mma_dV[1](zero_init=not accumulate)
                    accumulate = True
                    handle_Q.release()
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()
                    # (3) dQ(t): two d halves into the same 64 columns, drained by the reduce warps in between
                    # diag_k: dQ(0) of the cut pair reads the key-local K^T payload, then frees the V tile for the reload
                    kb_now = cut0_i * Int32(it == 0) if const_expr(self.diag_k) else Int32(0)
                    if kb_now != 0:
                        s2t_Ktb()
                        mma_dQb[0]()
                    else:
                        s2t_Kt[0]()
                        mma_dQ[0]()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dQ.sync_object_empty.wait(0, phase_dQe)  # half 1 reuses the window: the reduce warps loaded half 0
                    sm100_utils.tcgen05_after_thread_sync()
                    if kb_now != 0:
                        s2t_Ktb()
                        mma_dQb[1]()
                        with cute.arch.elect_one():
                            tcgen05.commit(ktb_mbar)
                    else:
                        s2t_Kt[1]()
                        mma_dQ[1]()
                    pipeline_dQ.sync_object_full.arrive(1, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()
                    # (4) dP(t+1)
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    dO_idx = consumer_state_dO.index
                    if const_expr(not self.mxds):  # MXDS waited at (2)
                        cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                        phase_Sl ^= 1
                    pipeline_dQ.sync_object_empty.wait(1, phase_dQe)
                    phase_dQe ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    if const_expr(self.vtok_dp_diag and (not self.diag_single or self.diag_fwd_tiles > 1 or self.diag_k)):
                        # this iteration's query tile: the compute warps walk m_block_min + iter % nm, and this loop
                        # runs iter = it + 1. With GQA the diagonal pair recurs once per query head, and with an
                        # unaligned diagonal offset a second key tile is masked too, so the predicate is dynamic --
                        # one branch per key tile in the MMA warp, not per element.
                        # tracked incrementally (m_block_min .. m_block_max - 1, wrapping per query head): no runtime
                        # integer division on the MMA warp's issue path
                        m_nx = m_cur + 1
                        m_cur = m_nx - nm_m * Int32(m_nx == m_block_max)
                        if n_loc >= block_info.get_n_block_min_causal_local_mask(seqlen, self._diag_m(self._loc_m(m_cur, m_off)), Int32(0)):
                            s2t_Vb()
                            if dO_idx == 0:
                                s2t_dO[0]()
                                mma_dP_b[0]()
                            else:
                                s2t_dO[1]()
                                mma_dP_b[1]()
                        else:
                            if const_expr(self.diag_k):
                                if vwait != 0:
                                    # plain V is back in its tile (after dQ(0)): waited ONCE, at the first plain dP -- a wait on
                                    # every plain dP, even on a completed phase, cost the backward ~17 %
                                    cute.arch.mbarrier_wait(ktb_mbar + 1, 0)
                                    vwait = Int32(0)
                            s2t_V()
                            if dO_idx == 0:
                                s2t_dO[0]()
                                mma_dP[0]()
                            else:
                                s2t_dO[1]()
                                mma_dP[1]()
                    else:
                        s2t_V()
                        if dO_idx == 0:
                            s2t_dO[0]()
                            mma_dP[0]()
                        else:
                            s2t_dO[1]()
                            mma_dP[1]()
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                    # (5) the S phase of query tile t+1 done
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    handle_Q = handle_Q_next

                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                # ---- tail: dV(last), dK(last), dQ(last)
                pipeline_dP.sync_object_empty.wait(0, phase_dPe)
                phase_dPe ^= 1
                pipeline_dS.consumer_wait(consumer_state_dS)
                sm100_utils.tcgen05_after_thread_sync()
                if dO_idx == 0:
                    mma_dV[0](zero_init=not accumulate)
                else:
                    mma_dV[1](zero_init=not accumulate)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                sm100_utils.tcgen05_after_thread_sync()
                if const_expr(self.mxds):
                    s2t_dSt_mx()
                if handle_Q.index == 0:
                    if const_expr(self.mxds):
                        s2t_Qt_mx[0]()
                    mma_dK[0](zero_init=not accumulate)
                else:
                    if const_expr(self.mxds):
                        s2t_Qt_mx[1 if const_expr(self.Q_stage > 1) else 0]()
                    mma_dK[1](zero_init=not accumulate)
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                producer_phase_dKV ^= 1
                # diag_k: with a single query tile the cut pair's dQ(0) is this tail's
                kb_tail = cut0_i * Int32(main_loop_iters == 0) if const_expr(self.diag_k) else Int32(0)
                if kb_tail != 0:
                    s2t_Ktb()
                    mma_dQb[0]()
                else:
                    s2t_Kt[0]()
                    mma_dQ[0]()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                sm100_utils.tcgen05_after_thread_sync()
                if kb_tail != 0:
                    s2t_Ktb()
                    mma_dQb[1]()
                    with cute.arch.elect_one():
                        tcgen05.commit(ktb_mbar)
                else:
                    s2t_Kt[1]()
                    mma_dQ[1]()
                pipeline_dQ.sync_object_full.arrive(1, pipeline_dQ.producer_mask, cta_group)
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()
                producer_phase_acc ^= 1
                pipeline_dQ.sync_object_empty.wait(1, phase_dQe)
                phase_dQe ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ compute warps (query rows)
    @cute.jit
    def _gmax16(self, x):
        r = cute.make_rmem_tensor(x.shape, Float32)
        r.store(x)
        p0 = utils.fmax(r[0], r[1], r[2])
        p1 = utils.fmax(r[3], r[4], r[5])
        p2 = utils.fmax(r[6], r[7], r[8])
        p3 = utils.fmax(r[9], r[10], r[11])
        p4 = utils.fmax(r[12], r[13], r[14])
        q0 = utils.fmax(p0, p1, p2)
        q1 = utils.fmax(p3, p4, r[15])
        return utils.fmax(q0, q1)

    @cute.jit
    def _ex2_emu_pair(self, x: Float32, y: Float32) -> Tuple[Float32, Float32]:
        """FMA-pipe exp2 of a pair -- bit-identical to the forward's FlashAttentionForwardSm100Fp4._ex2_emu_pair
        (degree-2 polynomial). The masked entries reaching here are already floored to the
        finite group-max placeholder, so no -inf clamp is needed (unlike the forward's masked steps)."""
        if const_expr(self.e2e_poly == 2):
            MAGIC = 12582912.0  # 1.5 * 2^23: round-to-nearest integer in the low mantissa bits
            C0, C1, C2 = 1.0004426738244065, 0.7034358772399556, 0.23842222937876753
            x = cute.arch.fmax(x, Float32(-126.0))
            y = cute.arch.fmax(y, Float32(-126.0))
            t0, t1 = cute.arch.add_packed_f32x2((x, y), (Float32(MAGIC), Float32(MAGIC)))
            u0, u1 = cute.arch.add_packed_f32x2((t0, t1), (Float32(-MAGIC), Float32(-MAGIC)))
            f0, f1 = cute.arch.fma_packed_f32x2((u0, u1), (Float32(-1.0), Float32(-1.0)), (x, y))
            p0, p1 = cute.arch.fma_packed_f32x2((f0, f1), (Float32(C2), Float32(C2)), (Float32(C1), Float32(C1)))
            p0, p1 = cute.arch.fma_packed_f32x2((p0, p1), (f0, f1), (Float32(C0), Float32(C0)))
            return utils.combine_int_frac_ex2(t0, p0), utils.combine_int_frac_ex2(t1, p1)
        else:
            return utils.ex2_emulation_2(x, y)

    @cute.jit
    def _p_groups(self, half: cutlass.Constexpr, tSrS, tScS_t2r, sfx, gmu, cw, mit16, kb, scale, scale2, c_dp, GM_FLOOR,
                  uinv, kbase, batch_idx, head_idx, upad=None):
        """The compute loop's P^ reproduction of one query tile's 4 groups per thread (codes cw, m words mit16, group SFs sfx);
        half: the warp group (0 / 1) -- its 32-key blocks of the 128-key tile are 2 stage + half (0 without a mix)."""
        for stage in cutlass.range_constexpr(2):
            for g in cutlass.range_constexpr(2):
                gi = 2 * stage + g
                sg16 = cute.make_rmem_tensor(16, Float32)
                for j in cutlass.range_constexpr(16):
                    sg16[j] = tSrS[16 * g + j, stage, 0, 0]
                gm = cute.arch.fmax(self._gmax16(sg16.load()), GM_FLOOR)
                sfx[gi] = cute.arch.exp2(gm * scale + kb)
                # one rounding, pinned (the forward's fma_rn): written as c_dp - gm scale2 the contraction was ptxas's
                # choice per site and one code in ~10^6 differed between the passes
                c2 = fma_rn(-gm, scale2, c_dp)
                if const_expr(self.p_mix is not None):
                    # the forward's group position of (warp group `half`, stage, g) is 4 stage + 2 half + g (each warp
                    # group holds the 32-key blocks 2 stage + half; measured with a position-map probe): a
                    # compile-time value function per group
                    if const_expr(half is not None):
                        self._mix_values(sg16, gm, scale, scale2, c2, self.p_mix[4 * stage + 2 * half + g])
                    else:
                        # the two warp groups differ: the position from the key coordinate, one code path per function
                        # (a per-group branch; duplicating the whole group loop per warp group measured slower)
                        gpos = (tScS_t2r[16 * g, stage, 0, 0][1] % 128) // 16
                        fid = (Int32(self.mix_table) >> (2 * gpos)) & Int32(3)
                        for ki in cutlass.range_constexpr(len(self.mix_kinds)):
                            if fid == Int32(ki):
                                self._mix_values(sg16, gm, scale, scale2, c2, self.mix_kinds[ki])
                    cw[2 * gi] = sm100_utils.packed_float_to_e2m1_relu(
                        sg16[0], sg16[1], sg16[2], sg16[3], sg16[4], sg16[5], sg16[6], sg16[7])
                    cw[2 * gi + 1] = sm100_utils.packed_float_to_e2m1_relu(
                        sg16[8], sg16[9], sg16[10], sg16[11], sg16[12], sg16[13], sg16[14], sg16[15])
                else:
                    for j in cutlass.range_constexpr(8):
                        a0, a1 = utils.fma_packed_f32x2((sg16[2 * j], sg16[2 * j + 1]), (scale2, scale2), (c2, c2))
                        if const_expr(self.fp4q_exp2):
                            # the forward's exp2 path, pair for pair: the same pairs land on the FMA pipe
                            # (_pair index runs over the 8 pairs of the group, as in softmax_step_split)
                            if const_expr(self.e2e_freq > 0 and (gi * 8 + j) % self.e2e_freq == 0):
                                sg16[2 * j], sg16[2 * j + 1] = self._ex2_emu_pair(a0, a1)
                            else:
                                sg16[2 * j] = cute.arch.exp2(a0)
                                sg16[2 * j + 1] = cute.arch.exp2(a1)
                        else:
                            sg16[2 * j] = mitchell_value_w(a0, self.directp_lea, 1)
                            sg16[2 * j + 1] = mitchell_value_w(a1, self.directp_lea, 1)
                    cw[2 * gi] = sm100_utils.packed_float_to_e2m1_relu(
                        sg16[0], sg16[1], sg16[2], sg16[3], sg16[4], sg16[5], sg16[6], sg16[7])
                    cw[2 * gi + 1] = sm100_utils.packed_float_to_e2m1_relu(
                        sg16[8], sg16[9], sg16[10], sg16[11], sg16[12], sg16[13], sg16[14], sg16[15])
                if const_expr(self.vtok):
                    # m_k = Mitchell_k u_k SF / Z (the forward's u-weighted row sum): unbiased weights, sum_k m = 1
                    # diagonal-only: kbase selects u on a diagonal pair and the unit weights elsewhere (see above)
                    for j in cutlass.range_constexpr(8):
                        ca = tScS_t2r[16 * g + 2 * j, stage, 0, 0][1]
                        cb = tScS_t2r[16 * g + 2 * j + 1, stage, 0, 0][1]
                        ua = self._vtok_u[batch_idx, head_idx, kbase + ca]
                        ub = self._vtok_u[batch_idx, head_idx, kbase + cb]
                        if const_expr(upad is not None):
                            # (cov5 packed) a key past the document's end: the aligned layout's zero row's weight (the
                            # tile's padding value -- here the next document's key)
                            ua = cutlass.select_(ca < upad[0], ua, upad[1])
                            ub = cutlass.select_(cb < upad[0], ub, upad[1])
                        sg16[2 * j], sg16[2 * j + 1] = utils.mul_packed_f32x2((sg16[2 * j], sg16[2 * j + 1]), (ua, ub))
                mg = sfx[gi] * uinv
                if const_expr(self.vtok_dv_u):
                    gmu[gi] = self._gmax16(sg16.load()) * mg
                for j in cutlass.range_constexpr(8):
                    m0, m1 = utils.mul_packed_f32x2((sg16[2 * j], sg16[2 * j + 1]), (mg, mg))
                    mit16[8 * gi + j] = pack_f16x2(m0, m1)

    @cute.jit
    def _mix_values(self, sg16, gm, scale, scale2, c2, kind: cutlass.Constexpr[str]):
        """p_mix: the 16 values of one group in place, with the forward's value function `kind` (flash_fwd_sm100_fp4
        _directp_group): 'M' the clamped Mitchell classifier on the 2^16-scaled argument (c2 = c_dp - gm scale2, the
        forward's own expression); 'E' / '2' / '3' on the exp2 argument s scale + fma_rn(-gm, scale, log2 6): ex2.approx,
        or the clamped degree-2 / degree-3 FMA-pipe exp2, pair for pair."""
        from flash_attn.cute.pv4_repro import fma_rn, ex2_emu_pair2, ex2_emu_pair3
        if const_expr(kind == "M"):
            for j in cutlass.range_constexpr(8):
                a0, a1 = utils.fma_packed_f32x2((sg16[2 * j], sg16[2 * j + 1]), (scale2, scale2), (c2, c2))
                sg16[2 * j] = mitchell_value_w(a0, self.directp_lea, 1)
                sg16[2 * j + 1] = mitchell_value_w(a1, self.directp_lea, 1)
        else:
            ce = fma_rn(-gm, scale, Float32(math.log2(6.0)))
            for j in cutlass.range_constexpr(8):
                a0, a1 = utils.fma_packed_f32x2((sg16[2 * j], sg16[2 * j + 1]), (scale, scale), (ce, ce))
                if const_expr(kind == "E"):
                    sg16[2 * j] = cute.arch.exp2(a0)
                    sg16[2 * j + 1] = cute.arch.exp2(a1)
                elif const_expr(kind == "2"):
                    sg16[2 * j], sg16[2 * j + 1] = ex2_emu_pair2(a0, a1, True)
                else:
                    sg16[2 * j], sg16[2 * j + 1] = ex2_emu_pair3(a0, a1)

    @cute.jit
    def _srw(self, rng: Int32, salt: cutlass.Constexpr[int]) -> Int32:
        """one random word for cvt.rs, the dQ operand's mix with its own salt."""
        xr = (rng + Int32(_i32_wrap(salt * -1640531527))) * Int32(-2048144789)
        xr = xr ^ ((xr >> 13) & Int32(0x7FFFF))
        xr = xr * Int32(-1028477387)
        return xr ^ ((xr >> 16) & Int32(0xFFFF))

    @cute.jit
    def _gamax16(self, x):
        r = cute.make_rmem_tensor(x.shape, Float32)
        for j in cutlass.range_constexpr(16):
            r[j] = cute.math.absf(x[j])
        return self._gmax16(r.load())

    @cute.jit
    def _kv_flush(self, cqw, cpw, c_ssw: Int32, c_svw: Int32, abase: Int32, sfbase2: Int32, is_even, is_warp_leader,
                  sdSt4_u8, sPt4_u8, sSFdSt_u16, sSFPt_u16):
        """A operands of dK (dS^^T) and dV (P^v^T) in SMEM, key rows: byte (key, query // 2) = even query's code | odd partner's
        code << 4 (one butterfly shuffle per code word); SF bytes: the warp's lane-0 thread takes the lane-16 query group's
        block-SF words and stores both groups' bytes of a key row as one u16 (SF IDs 0-1 or 2-3)."""
        pds = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        ppv = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        for wi in cutlass.range_constexpr(8):
            pds[wi] = cute.arch.shuffle_sync_bfly(cqw[wi], offset=1)
            ppv[wi] = cute.arch.shuffle_sync_bfly(cpw[wi], offset=1)
        if is_even:
            for wi in cutlass.range_constexpr(8):
                md0, md1 = merge_nibble_words(cqw[wi], pds[wi])
                mp0, mp1 = merge_nibble_words(cpw[wi], ppv[wi])
                for bi in cutlass.range_constexpr(8):
                    eb = 8 * wi + bi
                    kc = eb + (32 if eb >= 32 else 0)  # key = kc + 32 wg (C3 [0, 2, 1, 3] 32-key blocks per WG)
                    mdw = md0 if const_expr(bi < 4) else md1
                    mpw = mp0 if const_expr(bi < 4) else mp1
                    sh = 8 * (bi % 4)
                    aoff = abase + Int32(kc * 16)
                    sdSt4_u8[aoff] = cute.Uint8((mdw >> sh) & Int32(0xFF))
                    sPt4_u8[aoff] = cute.Uint8((mpw >> sh) & Int32(0xFF))
        ssw16 = cute.arch.shuffle_sync_bfly(c_ssw, offset=16)
        svw16 = cute.arch.shuffle_sync_bfly(c_svw, offset=16)
        if is_warp_leader:
            for gi in cutlass.range_constexpr(4):
                sh8 = 8 * gi
                v_ds = ((c_ssw >> sh8) & Int32(0xFF)) | (((ssw16 >> sh8) & Int32(0xFF)) << 8)
                v_pv = ((c_svw >> sh8) & Int32(0xFF)) | (((svw16 >> sh8) & Int32(0xFF)) << 8)
                for j in cutlass.range_constexpr(16):
                    el = 16 * gi + j
                    kc = el + (32 if el >= 32 else 0)
                    off2 = sfbase2 + Int32(((kc % 32) * 16 + (kc // 32) * 4) // 2)
                    sSFdSt_u16[off2] = cute.Uint16(v_ds)
                    sSFPt_u16[off2] = cute.Uint16(v_pv)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _reduce_e4m3w(self, w: Int32) -> Int32:
        """max over the 16 threads of a 16-thread group of 4 E4M3 bytes (a 4-level butterfly on two f16x2 registers, exact:
        E4M3 rounding is monotone, so this is E4M3 of the group max)."""
        p0, p1 = e4m3w_to_f16x2(w)
        for lvl in cutlass.range_constexpr(4):
            p0 = max_f16x2(p0, cute.arch.shuffle_sync_bfly(p0, offset=1 << lvl))
            p1 = max_f16x2(p1, cute.arch.shuffle_sync_bfly(p1, offset=1 << lvl))
        return f16x2_to_e4m3w(p0, p1)

    @cute.jit
    def _clamp(self, x: Float32, c: float) -> Float32:
        return -cute.arch.fmax(-cute.arch.fmax(x, Float32(-c)), Float32(-c))

    @cute.jit
    def _sr_e2m1_warp(self, y: Float32, v: Float32) -> Float32:
        """stochastic rounding onto the E2M1 grid, branch-free: the grid {0, .5, 1, 1.5, 2, 3, 4, 6} (odd-extended) is warped onto
        the integers by the piecewise-linear w(y) = 2y - (y - clamp(y, 2)) - (y - clamp(y, 4)) / 2; t = w(y) + v with v = u - 1/2
        uniform in [-1/2, 1/2) and the inverse warp g(t) = t/2 + (t - clamp(t, 4)) / 2 + (t - clamp(t, 6)) hand the value to
        the round-to-nearest E2M1 cvt, whose nearest grid point is g(round(t)) = g(floor(w + u)) (grid midpoints lie inside the
        linear pieces): exactly stochastic rounding."""
        c1 = y - self._clamp(y, 2.0)
        c2 = y - self._clamp(y, 4.0)
        t = (y + y) - c1 - Float32(0.5) * c2 + v
        d1 = t - self._clamp(t, 4.0)
        d2 = t - self._clamp(t, 6.0)
        return Float32(0.5) * (t + d1) + d2

    @cute.jit
    def _sr_e2m1(self, y: Float32, u: Float32) -> Float32:
        """stochastic rounding of y onto the E2M1 grid (|y| <= 6) with u in [0, 1)."""
        a = -cute.arch.fmax(-cute.math.absf(y), Float32(-6.0))
        small = a < Float32(2.0)
        mid = a < Float32(4.0)
        step = Float32(0.5) if small else (Float32(1.0) if mid else Float32(2.0))
        istep = Float32(2.0) if small else (Float32(1.0) if mid else Float32(0.5))
        qv = a * istep
        MG = Float32(12582912.0)
        fl = (qv - Float32(0.5) + MG) - MG  # floor for non-integer qv (1.5 * 2^23 magic round)
        up = (qv - fl) > u
        val = (fl + Float32(1.0)) * step if up else fl * step
        return -val if y < Float32(0.0) else val

    @cute.jit
    def compute_loop(
        self, thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
        tStS, tdPtdP, tdVtdV, tdKtdK, sFrame, sUinv, sD, sdSt4_u8, sPt4_u8, sSFdSt_u16, sSFPt_u16, sSFdSt_mx, sSFQt_mx,
        S_loaded_mbar_ptr,
        pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dSkv, pipeline_dKV, pipeline_dP,
        softmax_scale_log2, ds_mul, sr_seed, block_info, SeqlenInfoCls, TileSchedulerCls,
        sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK, tiled_copy_r2s_dKV, dk_scale, dv_scale,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        num_wg = self.num_compute_wgs
        lane = tidx % 32
        is_even = (lane % 2) == 0
        is_leader = (lane % 16) == 0
        is_warp_leader = lane == 0
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_S[:2]))
        tStS_t2r = thr_copy_t2r.partition_S(tStS)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)
        if const_expr(self.dkv8):
            copy_atom_r2s8 = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, Float8E4M3FN, Float32, thr_copy_t2r)
            thr_copy_r2s8 = cute.make_tiled_copy_D(copy_atom_r2s8, thr_copy_t2r).get_slice(tidx)
            rows_l = cute.slice_(self.sKV8_rows_layout.outer, (None, None, 0))
            rows_l = cute.make_layout((rows_l.shape,), stride=(rows_l.stride,))
            tRS_dS8 = thr_copy_r2s8.partition_D(cute.make_tensor(cute.recast_ptr(sdSt4_u8.iterator, self.sKV8_rows_layout.inner), rows_l))
            tRS_P8 = thr_copy_r2s8.partition_D(cute.make_tensor(cute.recast_ptr(sPt4_u8.iterator, self.sKV8_rows_layout.inner), rows_l))
        p4_cols, sf_cols = (32 if const_expr(self.dq8) else 16), 8
        atom_p4 = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p4_cols // num_wg)), Float32)
        atom_sf = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(sf_cols // num_wg)), Float32)
        thr_r2t_p4 = copy_utils.make_tmem_copy(atom_p4, num_wg).get_slice(tidx)
        thr_r2t_sf = copy_utils.make_tmem_copy(atom_sf, num_wg).get_slice(tidx)
        tScP4 = cute.composition(tScS, (cute.make_layout((self.tile_m, p4_cols)), 1, 1))
        tScSF = cute.composition(tScS, (cute.make_layout((self.tile_m, sf_cols)), 1, 1))
        tScP4_r2t = thr_r2t_p4.partition_S(tScP4)
        tScSF_r2t = thr_r2t_sf.partition_S(tScSF)
        tdSq4 = cute.make_tensor(tdPtdP.iterator + (self.tmem_dSq_offset - self.tmem_dP_offset),
                                 cute.composition(tdPtdP, (cute.make_layout((self.tile_m, p4_cols)), 1, 1)).layout)
        tdSqSF = cute.make_tensor(tdPtdP.iterator + (self.tmem_SFdSq_offset - self.tmem_dP_offset),
                                  cute.composition(tdPtdP, (cute.make_layout((self.tile_m, sf_cols)), 1, 1)).layout)
        tdSq4_r2t = thr_r2t_p4.partition_D(tdSq4)
        tdSqSF_r2t = thr_r2t_sf.partition_D(tdSqSF)

        # classifier constants (the forward's softmax_step_split, Direct-P lean)
        LOG2_PMAX = math.log2(6.0)
        scale = softmax_scale_log2
        # exp2 reproduction (BwdQSpec.exp2): DP_K = 1 and no magic constant, so c2 = log2 6 - gm*scale is the exp2 offset
        DP_K = 1.0 if const_expr(self.fp4q_exp2) else float(1 << sm100_utils.DIRECTP_FRAC_BITS)
        scale2 = scale * DP_K
        c_dp = (Float32(LOG2_PMAX) if const_expr(self.fp4q_exp2) else
                Float32(LOG2_PMAX * DP_K + self.directp_mag - self.directp_delta * DP_K))
        GM_FLOOR = Float32(self.gm_floor)
        # 1x16 scale target of the dQ operand's E2M1 dS^: sf = amax / T, so the E2M1 top code 6 represents 6 amax / T.
        # T = 5.647 (the shipped SR value) leaves headroom; T > 6 clips the group's largest values in exchange for a finer
        # grid under the bulk of them (Quartet II's four-over-six argument).
        ds_smax_inv = Float32(1.0 / self.ds_smax_t)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        phase_dPe_c = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dPsum = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)

        producer_state_dSkv = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        # dK / dV A-operand addressing (constant per thread): byte (key, query // 2), key = kc + 32 wg
        qrow = tScS_t2r[0, 0, 0, 0][0]
        qcol = qrow // 2
        wg_c = tidx // 128
        abase = (qcol // 16) * 2048 + qcol % 16 + wg_c * 512
        qgrp = qrow // 16
        sfbase2 = ((qgrp // 4) * 512 + wg_c * 4 + qgrp % 4) // 2
        if const_expr(self.mxds):
            # B of dK is plain E4M3 Q^ (its scale is folded into dk_scale): a constant 2^0 SFB tile
            for w in cutlass.range_constexpr(cute.size(sSFQt_mx.layout) // 256 + 1):
                wi = Int32(256 * w) + tidx
                if wi < Int32(cute.size(sSFQt_mx.layout)):
                    sSFQt_mx[wi] = Int32(0x7F7F7F7F)
            cute.arch.fence_view_async_shared()
        # the previous query tile's code / SF words, written to SMEM after the next S load
        cqw = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        cpw = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        for wi in cutlass.range_constexpr(8):
            cqw[wi] = Int32(0)
            cpw[wi] = Int32(0)
        c_ssw = Int32(0)
        c_svw = Int32(0)

        mCodeDbg = self._code_dbg if const_expr(self.code_dbg) else None   # a local: not written through self in the loop
        sfkv = self._sfkv if const_expr(self.fold_epi) else None   # (a local, as mCodeDbg)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            if const_expr(not self.doc_varlen) or m_block_min < m_block_max:   # documents: a key tile with no query (padding / an empty document) does nothing
                nm_c = m_block_max - m_block_min
                dsc = None
                pk_kp = pk_qsh = pk_cend = pk_pad0 = pk_pad1 = pk_pad2 = None
                if const_expr(self.codes_packed):
                    # (cov5) the key tile's first packed key, the statistics windows' row offset, the real keys, the
                    # padding-key weights of this tile (the per-key tables' first / second half)
                    pk_kp = seqlen.offset_k + n_loc * self.tile_n
                    pk_qsh = seqlen.offset_q & 3
                    pk_cend = seqlen.seqlen_k - n_loc * self.tile_n
                    if const_expr(self.vtok):   # (cov6: the dense route has no per-key weight)
                        pk_pad0 = self._pk[9][0, head_idx, n_block, 0]
                        pk_pad1 = self._pk[9][0, head_idx, n_block, 1]
                    if const_expr(self.diag_kr):   # (cov6 'all': the padding key's r_k)
                        pk_pad2 = self._pk[9][0, head_idx, n_block, 2]
                if const_expr(self.doc_varlen):
                    # this key tile's document: its row of the per-document scalars
                    dsc = self._ndoc[n_block]
                    if const_expr(self.dev_scl):
                        ds_mul = self._scl[dsc, 2]
                # diag_fold: the dV operand's scale for this key tile (448 2^-A keeps m r <= 448); 448 otherwise
                a448 = (self._dva[batch_idx, head_idx, n_block] if const_expr(self.diag_fold)
                        else Float32(448.0))
                if const_expr(self.fold_epi):
                    # the K fold's row factors of this thread's key row (dp_idx), loaded now -- the load's latency sits under
                    # the tile's pipeline start, not on the epilogue at its end -- into the thread's own SMEM slots (it reads
                    # them back itself: no sync; read before the dV epilogue, whose per-warpgroup barriers precede any dK
                    # epilogue store into this slice)
                    if const_expr(self.codes_packed):   # (cov6: the factors per packed key, (1, h_k, T'))
                        sfkv[tidx // 128, dp_idx, 0] = self._fdk[0, head_idx, pk_kp + dp_idx]
                        sfkv[tidx // 128, dp_idx, 1] = self._fdv[0, head_idx, pk_kp + dp_idx]
                    else:
                        sfkv[tidx // 128, dp_idx, 0] = self._fdk[batch_idx, head_idx, n_block * self.tile_n + dp_idx]
                        sfkv[tidx // 128, dp_idx, 1] = self._fdv[batch_idx, head_idx, n_block * self.tile_n + dp_idx]
                loop_count = nm_c * self.qhead_per_kvhead
                if const_expr(self.dev_seed_tile and (self.ds_sr or self.dk_sr or self.dv_sr)):
                    # the seed's key-tile term, once per work tile (what the compiler hoists from the scalar-seed hash)
                    sr_seed_nb = sr_seed_from_counter(self._sr_ctr.iterator) ^ (n_block * Int32(-1640531535))
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    if const_expr(self.qhead_per_kvhead > 1):
                        m_block = m_block_min + (iter_idx - (iter_idx // nm_c) * nm_c)
                    else:
                        m_block = m_block_min + iter_idx
                    # ======== S phase
                    pipeline_LSE.consumer_wait(consumer_state_LSE)
                    pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                    tSrS = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                    cute.copy(thr_copy_t2r, tStS_t2r, tSrS)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                    if const_expr(not self.dkv8):
                        # dK / dV operands of the previous query tile (a placeholder commit at iter 0), while the MMA runs dQ and dP
                        pipeline_dSkv.producer_acquire(producer_state_dSkv)
                        self._kv_flush(cqw, cpw, c_ssw, c_svw, abase, sfbase2, is_even, is_warp_leader, sdSt4_u8, sPt4_u8, sSFdSt_u16, sSFPt_u16)
                        with cute.arch.elect_one():
                            pipeline_dSkv.producer_commit(producer_state_dSkv)
                        producer_state_dSkv.advance()
                    if const_expr(self.codes_packed):
                        # (cov5) the window's row; a row past the document: the aligned layout's padding row (frame 0,
                        # uinv 0 -- LSE -inf)
                        pk_rok = dp_idx < seqlen.seqlen_q - (m_block - m_off) * self.tile_m
                        fr = cutlass.select_(pk_rok, sFrame[dp_idx + pk_qsh, consumer_state_LSE.index], Float32(0.0))
                        uinv = cutlass.select_(pk_rok, sUinv[dp_idx + pk_qsh, consumer_state_LSE.index], Float32(0.0))
                    else:
                        fr = sFrame[dp_idx, consumer_state_LSE.index]
                        uinv = sUinv[dp_idx, consumer_state_LSE.index]
                    if const_expr(self.doc_varlen):
                        # documents: the mask in the document's coordinates -- causal (bottom-right) and the document's
                        # true lengths (its padding rows / keys)
                        m_loc = m_block - m_off
                        row_end = seqlen.seqlen_q - m_loc * self.tile_m
                        col_end = seqlen.seqlen_k - n_loc * self.tile_n
                        needs_mask = (row_end < self.tile_m) or (col_end < self.tile_n)
                        if const_expr(self.is_causal):
                            needs_mask = needs_mask or (m_loc * self.tile_m + (seqlen.seqlen_k - seqlen.seqlen_q)
                                                        < (n_loc + 1) * self.tile_n - 1)
                        if needs_mask:
                            causal_off = seqlen.seqlen_k - n_loc * self.tile_n - seqlen.seqlen_q
                            for stage in cutlass.range_constexpr(2):
                                for i in cutlass.range_constexpr(32):
                                    crd = tScS_t2r[i, stage, 0, 0]
                                    lim = col_end
                                    if const_expr(self.is_causal):
                                        lim = cutlass.min(lim, crd[0] + m_loc * self.tile_m + causal_off + 1)
                                    if crd[0] >= row_end:
                                        lim = Int32(0)
                                    tSrS[i, stage, 0, 0] = -Float32.inf if crd[1] >= lim else tSrS[i, stage, 0, 0]
                    elif const_expr(self.is_causal):
                        check_m_boundary = (m_block + 1) * self.tile_m > seqlen.seqlen_q
                        needs_mask = (m_block * self.tile_m + (seqlen.seqlen_k - seqlen.seqlen_q)
                                      < (n_block + 1) * self.tile_n - 1) or check_m_boundary
                        if needs_mask:
                            causal_off = seqlen.seqlen_k - n_block * self.tile_n - seqlen.seqlen_q
                            for stage in cutlass.range_constexpr(2):
                                for i in cutlass.range_constexpr(32):
                                    crd = tScS_t2r[i, stage, 0, 0]
                                    lim = crd[0] + m_block * self.tile_m + causal_off + 1
                                    tSrS[i, stage, 0, 0] = -Float32.inf if crd[1] >= lim else tSrS[i, stage, 0, 0]
                    kbase = None
                    if const_expr(self.vtok):
                        # the forward's per-key bias (same raw-score add before the group maxes / classifier).
                        # diagonal-only: the forward biased this pair only if n_block is in its masked region -- the block_info
                        # call its load warp makes, on the first query tile of the FORWARD's CTA (_diag_m). The host
                        # hands over DOUBLED per-key tensors -- [zero bias | bias] and [unit weight | u] along keys -- and a
                        # diagonal pair indexes the second half, so the gathers are the full fix's unconditional ones with
                        # no 0 / 1 multipliers (`diagonal-backward-spills-under-dsl48`: the multiplier form cost the
                        # backward 16 % under cutlass-dsl 4.8, the full fix's plain gathers 3.5 %).
                        # (documents: the doubled tables' second half starts at the allocation's key count)
                        kdiag = (Int32(n_loc >= block_info.get_n_block_min_causal_local_mask(
                                    seqlen, self._diag_m(self._loc_m(m_block, m_off)), Int32(0)))
                                 if self.vtok_diag else Int32(0))
                        if const_expr(self.codes_packed):
                            # (cov5) the packed per-key tables (1, h_k, 2 T'): the tile's packed keys, the second half
                            kbase = pk_kp + self._sk_alloc * kdiag
                        else:
                            kbase = n_block * self.tile_n + (
                                (self._sk_alloc if const_expr(self.doc_varlen) else seqlen.seqlen_k) * kdiag
                                if self.vtok_diag else Int32(0))
                        for stage in cutlass.range_constexpr(2):
                            for i in cutlass.range_constexpr(32):
                                tSrS[i, stage, 0, 0] = tSrS[i, stage, 0, 0] + self._vtok_bias[batch_idx, head_idx, kbase + tScS_t2r[i, stage, 0, 0][1]]
                    kb = (Float32(self.max_offset) - fr) - Float32(LOG2_PMAX)
                    sfx = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                    # vtok_dv_u: per group, max_k m_k (the all-FP4 dV operand's group amax); off, no tensor at all
                    gmu = cute.make_rmem_tensor(cute.make_layout(4), Float32) if self.vtok_dv_u else None
                    cw = cute.make_rmem_tensor(cute.make_layout(8), Int32)   # P^ code words (2 per group)
                    mit16 = cute.make_rmem_tensor(cute.make_layout(32), Int32)   # m = Mitchell x exact SF / Z, f16 pairs
                    # the Direct-P classifier and Mitchell values in place in the S fragment (register budget: no second 64-float
                    # fragment lives across the dP wait); clamp 1 = the forward's clamp 2 arithmetic (max.s32 + shl / add)
                    if const_expr(self.p_mix is None or (tuple(self.p_mix[0:2]) == tuple(self.p_mix[2:4])
                                                         and tuple(self.p_mix[4:6]) == tuple(self.p_mix[6:8]))):
                        self._p_groups(0, tSrS, tScS_t2r, sfx, gmu, cw, mit16, kb, scale, scale2, c_dp, GM_FLOOR, uinv, kbase, batch_idx, head_idx,
                                       (pk_cend, cutlass.select_(kdiag != 0, pk_pad1, pk_pad0))
                                       if const_expr(self.codes_packed and self.vtok) else None)
                    else:
                        # p_mix whose two warp groups hold different functions: per-group dispatch
                        self._p_groups(None, tSrS, tScS_t2r, sfx, gmu, cw, mit16, kb, scale, scale2, c_dp, GM_FLOOR, uinv,
                                       kbase, batch_idx, head_idx,
                                       (pk_cend, cutlass.select_(kdiag != 0, pk_pad1, pk_pad0))
                                       if const_expr(self.codes_packed and self.vtok) else None)
                    if const_expr(self.code_dbg):
                        # debug build: every reproduced code, (query, key) indexed in the forward dump's layout
                        _hq_dbg = (head_idx * self.qhead_per_kvhead + iter_idx // nm_c) if const_expr(self.qhead_per_kvhead > 1) \
                            else head_idx
                        for stage in cutlass.range_constexpr(2):
                            for g in cutlass.range_constexpr(2):
                                gi = 2 * stage + g
                                for j in cutlass.range_constexpr(16):
                                    crd = tScS_t2r[16 * g + j, stage, 0, 0]
                                    mCodeDbg[batch_idx, _hq_dbg, m_block * self.tile_m + crd[0], n_block * self.tile_n + crd[1]] = (
                                        (cw[2 * gi + j // 8] >> (4 * (j % 8))) & Int32(0xF)).to(cutlass.Uint8)
                    sfw = sm100_utils.packed_float_to_ue4m3(sfx[0], sfx[1], sfx[2], sfx[3])
                    sfr0, sfr1, sfr2, sfr3 = sm100_utils.packed_ue4m3_to_float(sfw)
                    sfr = [sfr0, sfr1, sfr2, sfr3]
                    # P^v block scales: the max P^ code of every group is 6 (its max score's Direct-P value is Mitchell(log2 6 -
                    # delta) = 6.05; fully masked groups have SF 0), so max |w boost| over the group = 6 wg and the block SF
                    # E4M3(amax / 6) = max over the 16 query threads of E4M3(wg): an exact reduction of one E4M3 word
                    if const_expr(self.dkv8):
                        svr = sfr
                        kpv = sfr
                    else:
                        # vtok_dv_u: the SF word is E4M3(max_k m_k boost / 6) and the codes are m boost / SF (see __call__)
                        wg = ([gmu[gi] * Float32(self.pv_boost / 6.0) for gi in range(4)] if self.vtok_dv_u
                              else [sfr[gi] * uinv * Float32(self.pv_boost) for gi in range(4)])
                        svw = self._reduce_e4m3w(sm100_utils.packed_float_to_ue4m3(wg[0], wg[1], wg[2], wg[3]))
                        svr0, svr1, svr2, svr3 = sm100_utils.packed_ue4m3_to_float(svw)
                        svr = [svr0, svr1, svr2, svr3]
                        kpv = ([Float32(self.pv_boost) * cute.arch.rcp_approx(cute.arch.fmax(svr[gi], Float32(1e-30)))
                                for gi in range(4)] if self.vtok_dv_u
                               else [wg[gi] * cute.arch.rcp_approx(cute.arch.fmax(svr[gi], Float32(1e-30))) for gi in range(4)])
                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    pipeline_LSE.consumer_release(consumer_state_LSE)
                    consumer_state_LSE.advance()

                    if const_expr(self.dkv8):
                        # ======== dP phase (dkv8): per load stage -- dS' in place; E4M3 rows of dS' (A of dK) and P (A of dV) to
                        # SMEM; dS^ of dQ with in-thread 1x16 E4M3 scales and hardware SR
                        pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                        # the stochastic quantisers' random words depend only on the seed and this tile's indices, so they are
                        # derived here, under the dP wait: consumed later they would sit on the dS store's hand-off chain, which
                        # measured +2 % on the backward even though cvt.rs itself is free (constant-rbits ablation: baseline)
                        # `sr-dither-missing-batch-head`: batch_idx, head_idx and the q-head all belong in this
                        # hash. Each rounding is individually unbiased either way, but without them the rounding error
                        # at a given (row-in-tile, key-tile, element) position is IDENTICAL across every batch and head
                        # of the step. Under GQA that is the coherent case: m_block = m_block_min + iter_idx mod nm_c
                        # repeats for each of the r q-heads sharing a kv head, and their dK / dV are summed inside the
                        # kernel, so the errors add in phase and the variance grows as r^2 instead of r -- which negates
                        # most of what `stochastic-rounding-ds` buys (the 0.9419 -> 0.9248 result). The host quantiser
                        # always got this right. Three more XORs, derived here under the dP wait, off the store's
                        # hand-off chain like the rest of the hash.
                        _qh_sr = (iter_idx // nm_c) if const_expr(self.qhead_per_kvhead > 1) else Int32(0)
                        if const_expr(self.ds_sr or self.dk_sr or self.dv_sr):
                            if const_expr(self.dev_seed_tile):
                                rng = sr_seed_nb ^ (m_block * Int32(-2048144777)) ^ (tidx * Int32(-1028477379))\
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))\
                                      ^ (_qh_sr * Int32(-1509342103))
                            else:
                                rng = sr_seed ^ (n_block * Int32(-1640531535)) ^ (m_block * Int32(-2048144777)) ^ (tidx * Int32(-1028477379))\
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))\
                                      ^ (_qh_sr * Int32(-1509342103))
                        pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                        if const_expr(self.codes_packed):   # (cov5: the window's row, a padding row's D'' 0)
                            dsum = cutlass.select_(pk_rok, sD[dp_idx + pk_qsh, consumer_state_dPsum.index], Float32(0.0))
                        else:
                            dsum = sD[dp_idx, consumer_state_dPsum.index]
                        nd = -(dsum * ds_mul)
                        # diag_k: the doubled [1 | r_k] gather base -- the second half on the causally cut pair (the backward's
                        # own 128-query tile: the K scale leak is a property of the cut, not of the forward's V tiling)
                        if const_expr(self.doc_varlen and self.diag_kr):
                            # (cov6) documents: the doubled table's second half starts at the allocation's key count
                            # (the per-key tables' half, as kbase above), the cut test in the document's coordinates;
                            # the batched form's seqlen_k half and global tile indices read another key's r_k
                            kcut = Int32(n_loc >= block_info.get_n_block_min_causal_local_mask(
                                seqlen, self._loc_m(m_block, m_off), Int32(0)))
                            if const_expr(self.codes_packed):
                                kbase_k = pk_kp + self._sk_alloc * kcut
                            else:
                                kbase_k = n_block * self.tile_n + self._sk_alloc * kcut
                        else:
                            kbase_k = n_block * self.tile_n + (
                                seqlen.seqlen_k * Int32(n_block >= block_info.get_n_block_min_causal_local_mask(seqlen, m_block, Int32(0)))
                                if self.diag_kr else Int32(0))
                        qw = cute.make_rmem_tensor(cute.make_layout(8), Int32)
                        ssw = Int32(0)
                        for st in cutlass.range_constexpr(2):
                            tdPs = cute.make_rmem_tensor(tScS_t2r[None, st, None, None].shape, Float32)
                            cute.copy(thr_copy_t2r, tdPtdP_t2r[None, st, None, None], tdPs)
                            if const_expr(st == 1):
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    pipeline_dP.consumer_release(consumer_state_S_P_dP)
                            rws_all = [None, None, None, None]
                            p8w = cute.make_rmem_tensor(8, Int32)
                            d8w = cute.make_rmem_tensor(8, Int32)
                            lma = []
                            lam = []  # raw per-group amax (the 32 x 32 variants reduce both groups, then the warp)
                            for g in cutlass.range_constexpr(2):
                                gi = 2 * st + g
                                ag = sfr[gi] * uinv * ds_mul
                                a8 = sfr[gi] * uinv * Float32(448.0)
                                # The dS weight is vv * ag on BOTH pair kinds once the dP operand matches the pair:
                                # on a diagonal pair the codes carry r_k and the dP operand (the diagonal payload)
                                # dequantises to V / r_k, so the two cancel; on a non-diagonal pair both are
                                # unbiased. So no per-key factor is needed in the dP phase, and the forward's
                                # E2M1-rounded P^ is what the backward uses, as in the leaky route.
                                for wj in cutlass.range_constexpr(2):
                                    v0, v1, v2, v3, v4, v5, v6, v7 = packed_e2m1_to_float8(cw[2 * gi + wj])
                                    vv = (v0, v1, v2, v3, v4, v5, v6, v7)
                                    pv = []
                                    for j in cutlass.range_constexpr(4):
                                        e0 = 16 * g + 8 * wj + 2 * j
                                        u0, u1 = unpack_f16x2(mit16[(16 * gi + 8 * wj + 2 * j) // 2])
                                        if const_expr(self.vtok):
                                            # vtok: dV from the unbiased m = Mitchell u SF / Z (~ P) -- dL/dV directly. The codes
                                            # w' = P'^ / Z carry P s_k rounded against the group's largest s, which loses low-norm
                                            # keys once divided by s_k. dV is outside the dS row-sum identity, so either is consistent.
                                            p0, p1 = utils.mul_packed_f32x2((u0, u1), (a448, a448))
                                        else:
                                            p0, p1 = utils.mul_packed_f32x2((vv[2 * j], vv[2 * j + 1]), (a8, a8))
                                        pv.append(p0)
                                        pv.append(p1)
                                        w0, w1 = utils.mul_packed_f32x2((vv[2 * j], vv[2 * j + 1]), (ag, ag))
                                        c0, c1 = utils.mul_packed_f32x2((u0, u1), (nd, nd))
                                        tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0] = utils.fma_packed_f32x2((tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0]), (w0, w1), (c0, c1))
                                    if const_expr(self.dv_sr):
                                        p8w[4 * g + 2 * wj] = packed_float_to_e4m3x4_rs(pv[0], pv[1], pv[2], pv[3], self._srw(rng, 128 + 8 * gi + 2 * wj))
                                        p8w[4 * g + 2 * wj + 1] = packed_float_to_e4m3x4_rs(pv[4], pv[5], pv[6], pv[7], self._srw(rng, 160 + 8 * gi + 2 * wj))
                                    else:
                                        p8w[4 * g + 2 * wj] = sm100_utils.packed_float_to_ue4m3(pv[0], pv[1], pv[2], pv[3])
                                        p8w[4 * g + 2 * wj + 1] = sm100_utils.packed_float_to_ue4m3(pv[4], pv[5], pv[6], pv[7])
                                g16 = cute.make_rmem_tensor(16, Float32)
                                for j in cutlass.range_constexpr(16):
                                    if const_expr(self.diag_kr and self.codes_packed):
                                        # (cov6) a key past the document's end (here the next document's) takes the
                                        # aligned layout's padding-key r_k (the tile's pad table): the masked dS' is
                                        # not an exact zero, and the next document's r_k would carry it into dQ
                                        ck_ = tScS_t2r[16 * g + j, st, 0, 0][1]
                                        rk_ = cutlass.select_(ck_ < pk_cend, self._diag_kr[batch_idx, head_idx, kbase_k + ck_], pk_pad2)
                                        g16[j] = tdPs[16 * g + j, 0, 0] * rk_
                                    elif const_expr(self.diag_kr):
                                        # the dQ copy carries r_k on the cut pair (the dK copy below still reads the unscaled tdPs)
                                        g16[j] = tdPs[16 * g + j, 0, 0] * self._diag_kr[batch_idx, head_idx, kbase_k + tScS_t2r[16 * g + j, st, 0, 0][1]]
                                    else:
                                        g16[j] = tdPs[16 * g + j, 0, 0]
                                am = self._gamax16(g16.load())
                                lma.append(am * ds_smax_inv)
                                lam.append(am)
                                if const_expr(self.mxds and not self.mxds_blk32):
                                    # UE8M0 scale of this (16 keys x 32 queries) block: the in-thread 16-key amax reduced over the
                                    # warp's 32 query rows. The MMA re-applies 2^e, so dk_scale is unchanged.
                                    for lvl in cutlass.range_constexpr(5):
                                        am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, offset=1 << lvl))
                                    ef = sm100_utils.ceil_f32(utils.log2f(cute.arch.fmax(am, Float32(1e-30)) * Float32(1.0 / 448.0)))
                                    ef = cute.arch.fmax(ef, Float32(-126.0))
                                    mxs = cute.arch.exp2(-ef)
                                    if lane < 16:  # one lane per key row of the group
                                        mrow = Int32(16 * gi + (32 if const_expr(gi >= 2) else 0)) + lane + 32 * wg_c
                                        sSFdSt_mx[(mrow % 32) * 16 + (mrow // 32) * 4 + qrow // 32] = cute.Uint8(Int32(ef) + Int32(127))
                                if const_expr(self.ds_sr or self.dk_sr):
                                    # four random words per 16-value group, derived once here and shared by the group's two
                                    # stochastic quantisers (dS^ of dK below, dS^ of dQ further down), each cvt.rs taking one word
                                    # per four values. cvt.rs itself is free -- a constant-rbits ablation sits exactly on baseline
                                    # -- so the whole cost is this dither's dependency on the dS store's hand-off chain: deriving
                                    # the words per stage (+3 %), all sixteen up front (+8 %), from a serial LCG (+6 %) or with
                                    # separate salts per quantiser (+6 %) are all worse than sharing them from here (+2.5 %).
                                    rw_g = []
                                    for wj in cutlass.range_constexpr(4):
                                        xr = (rng + Int32(_i32_wrap((8 * gi + 2 * wj + 1) * -1640531527))) * Int32(-2048144789)
                                        xr = xr ^ ((xr >> 13) & Int32(0x7FFFF))
                                        xr = xr * Int32(-1028477387)
                                        rw_g.append(xr ^ ((xr >> 16) & Int32(0xFFFF)))
                                    rws_all[gi] = rw_g
                                for k4 in cutlass.range_constexpr(4):
                                    o4 = 16 * g + 4 * k4
                                    if const_expr(self.mxds_blk32):
                                        pass  # packed after both groups, with the 32 x 32 block exponent (below)
                                    elif const_expr(self.mxds):
                                        y0, y1 = utils.mul_packed_f32x2((tdPs[o4, 0, 0], tdPs[o4 + 1, 0, 0]), (mxs, mxs))
                                        y2, y3 = utils.mul_packed_f32x2((tdPs[o4 + 2, 0, 0], tdPs[o4 + 3, 0, 0]), (mxs, mxs))
                                        d8w[4 * g + k4] = sm100_utils.packed_float_to_ue4m3(y0, y1, y2, y3)
                                    elif const_expr(self.dk_sr):
                                        d8w[4 * g + k4] = packed_float_to_e4m3x4_rs(tdPs[o4, 0, 0], tdPs[o4 + 1, 0, 0], tdPs[o4 + 2, 0, 0],
                                                                                    tdPs[o4 + 3, 0, 0], rws_all[gi][k4])
                                    else:
                                        d8w[4 * g + k4] = sm100_utils.packed_float_to_ue4m3(tdPs[o4, 0, 0], tdPs[o4 + 1, 0, 0], tdPs[o4 + 2, 0, 0], tdPs[o4 + 3, 0, 0])
                                if const_expr(self.diag_kr):
                                    # dK's words are packed: from here on tdPs is the dQ copy, r_k on the cut pair
                                    for j in cutlass.range_constexpr(16):
                                        tdPs[16 * g + j, 0, 0] = g16[j]
                            if const_expr(self.mxds_blk32):
                                # BwdQSpec.mxds = "blk32": one UE8M0 exponent per (32 keys of this stage x the warp's 32 query
                                # rows) -- the two groups' in-thread amax, then the warp; round-up (the causal-safety selector)
                                assert not self.diag_kr, "BwdQSpec.mxds='blk32' with the diagonal K route is not built"
                                amb = cute.arch.fmax(lam[0], lam[1])
                                for lvl in cutlass.range_constexpr(5):
                                    amb = cute.arch.fmax(amb, cute.arch.shuffle_sync_bfly(amb, offset=1 << lvl))
                                efb = sm100_utils.ceil_f32(utils.log2f(cute.arch.fmax(amb, Float32(1e-30)) * Float32(1.0 / 448.0)))
                                efb = cute.arch.fmax(efb, Float32(-126.0))
                                mxb = cute.arch.exp2(-efb)
                                mrow = Int32(64 * st) + lane + 32 * wg_c
                                sSFdSt_mx[(mrow % 32) * 16 + (mrow // 32) * 4 + qrow // 32] = cute.Uint8(Int32(efb) + Int32(127))
                                for g in cutlass.range_constexpr(2):
                                    for k4 in cutlass.range_constexpr(4):
                                        o4 = 16 * g + 4 * k4
                                        y0, y1 = utils.mul_packed_f32x2((tdPs[o4, 0, 0], tdPs[o4 + 1, 0, 0]), (mxb, mxb))
                                        y2, y3 = utils.mul_packed_f32x2((tdPs[o4 + 2, 0, 0], tdPs[o4 + 3, 0, 0]), (mxb, mxb))
                                        if const_expr(self.dk_sr_blk32):
                                            d8w[4 * g + k4] = packed_float_to_e4m3x4_rs(y0, y1, y2, y3, rws_all[2 * st + g][k4])
                                        else:
                                            d8w[4 * g + k4] = sm100_utils.packed_float_to_ue4m3(y0, y1, y2, y3)
                            cute.autovec_copy(cute.recast_tensor(d8w, Float8E4M3FN), tRS_dS8[None, st])
                            cute.autovec_copy(cute.recast_tensor(p8w, Float8E4M3FN), tRS_P8[None, st])
                            if const_expr(self.dq8):
                                # dS^ of dQ IS dS^ of dK: with A from SMEM the dQ GEMM reads the tile just stored, so there is
                                # nothing left to pack here at all
                                continue
                            # dS^ of dQ: the 1x16 (per query row, 16 keys) E4M3 scales of this stage's 2 groups, in-thread
                            sb, ssr_a, ssr_b = f16x2_e4m3_out(e4m3x2_f16x2(lma[0], lma[1]))
                            ssw = ssw | (sb << (16 * st))
                            ssr_s = [ssr_a, ssr_b]
                            for g in cutlass.range_constexpr(2):
                                gi = 2 * st + g
                                iv = cute.arch.rcp_approx(cute.arch.fmax(ssr_s[g], Float32(1e-30)))
                                tq = []
                                for j in cutlass.range_constexpr(8):
                                    e0 = 16 * g + 2 * j
                                    q0, q1 = utils.mul_packed_f32x2((tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0]), (iv, iv))
                                    tq.append(q0)
                                    tq.append(q1)
                                if const_expr(self.ds_sr):
                                    rws = rws_all[gi]
                                    qw[2 * gi] = packed_float_to_e2m1x4_rs(tq[0], tq[1], tq[2], tq[3], tq[4], tq[5], tq[6], tq[7], rws[0], rws[1])
                                    qw[2 * gi + 1] = packed_float_to_e2m1x4_rs(tq[8], tq[9], tq[10], tq[11], tq[12], tq[13], tq[14], tq[15], rws[2], rws[3])
                                else:
                                    qw[2 * gi] = sm100_utils.packed_float_to_e2m1(tq[0], tq[1], tq[2], tq[3], tq[4], tq[5], tq[6], tq[7])
                                    qw[2 * gi + 1] = sm100_utils.packed_float_to_e2m1(tq[8], tq[9], tq[10], tq[11], tq[12], tq[13], tq[14], tq[15])
                        cute.arch.fence_view_async_shared()
                        pw = qw
                        svw = ssw
                    else:
                        # ======== dP phase: one TMEM load stage (32 elements, 2 SF groups) at a time, end to end (register budget)
                        pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                        pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                        dsum = sD[dp_idx, consumer_state_dPsum.index]
                        nd = -(dsum * ds_mul)
                        pw = cute.make_rmem_tensor(cute.make_layout(8), Int32)
                        qw = cute.make_rmem_tensor(cute.make_layout(16 if const_expr(self.dq8) else 8), Int32)
                        # `sr-dither-missing-batch-head`: batch_idx, head_idx and the q-head all belong in this
                        # hash. Each rounding is individually unbiased either way, but without them the rounding error
                        # at a given (row-in-tile, key-tile, element) position is IDENTICAL across every batch and head
                        # of the step. Under GQA that is the coherent case: m_block = m_block_min + iter_idx mod nm_c
                        # repeats for each of the r q-heads sharing a kv head, and their dK / dV are summed inside the
                        # kernel, so the errors add in phase and the variance grows as r^2 instead of r -- which negates
                        # most of what `stochastic-rounding-ds` buys (the 0.9419 -> 0.9248 result). The host quantiser
                        # always got this right. Three more XORs, derived here under the dP wait, off the store's
                        # hand-off chain like the rest of the hash.
                        _qh_sr = (iter_idx // nm_c) if const_expr(self.qhead_per_kvhead > 1) else Int32(0)
                        if const_expr(self.ds_sr or self.dk_sr or self.dv_sr):
                            if const_expr(self.dev_seed_tile):
                                rng = sr_seed_nb ^ (m_block * Int32(-2048144777)) ^ (tidx * Int32(-1028477379))\
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))\
                                      ^ (_qh_sr * Int32(-1509342103))
                            else:
                                rng = sr_seed ^ (n_block * Int32(-1640531535)) ^ (m_block * Int32(-2048144777)) ^ (tidx * Int32(-1028477379))\
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))\
                                      ^ (_qh_sr * Int32(-1509342103))
                        ssw = Int32(0)
                        for st in cutlass.range_constexpr(2):
                            tdPs = cute.make_rmem_tensor(tScS_t2r[None, st, None, None].shape, Float32)
                            cute.copy(thr_copy_t2r, tdPtdP_t2r[None, st, None, None], tdPs)
                            if const_expr(st == 1):
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    pipeline_dP.consumer_release(consumer_state_S_P_dP)
                            # one decode of the P^ codes serves P^v (RTN codes of vals kpv) and dS' = w dP' - D' m (in place)
                            lma = []
                            for g in cutlass.range_constexpr(2):
                                gi = 2 * st + g
                                ag = sfr[gi] * uinv * ds_mul
                                for wj in cutlass.range_constexpr(2):
                                    v0, v1, v2, v3, v4, v5, v6, v7 = packed_e2m1_to_float8(cw[2 * gi + wj])
                                    vv = (v0, v1, v2, v3, v4, v5, v6, v7)
                                    pv = []
                                    for j in cutlass.range_constexpr(4):
                                        e0 = 16 * g + 8 * wj + 2 * j
                                        if const_expr(self.vtok_dv_u):
                                            # the unbiased m (u_k folded in), not the biased code: dV comes out as dL/dV
                                            um0, um1 = unpack_f16x2(mit16[(16 * gi + 8 * wj + 2 * j) // 2])
                                            p0, p1 = utils.mul_packed_f32x2((um0, um1), (kpv[gi], kpv[gi]))
                                        else:
                                            p0, p1 = utils.mul_packed_f32x2((vv[2 * j], vv[2 * j + 1]), (kpv[gi], kpv[gi]))
                                        pv.append(p0)
                                        pv.append(p1)
                                        w0, w1 = utils.mul_packed_f32x2((vv[2 * j], vv[2 * j + 1]), (ag, ag))
                                        u0, u1 = unpack_f16x2(mit16[(16 * gi + 8 * wj + 2 * j) // 2])
                                        c0, c1 = utils.mul_packed_f32x2((u0, u1), (nd, nd))
                                        tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0] = utils.fma_packed_f32x2((tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0]), (w0, w1), (c0, c1))
                                    pw[2 * gi + wj] = sm100_utils.packed_float_to_e2m1_relu(pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], pv[7])
                                g16 = cute.make_rmem_tensor(16, Float32)
                                for j in cutlass.range_constexpr(16):
                                    g16[j] = tdPs[16 * g + j, 0, 0]
                                lma.append(self._gamax16(g16.load()) * ds_smax_inv)
                            # dS^ 16x16 block scales of this stage's 2 groups: exact reduction of one f16x2 register of E4M3 values
                            sp = e4m3x2_f16x2(lma[0], lma[1])
                            for lvl in cutlass.range_constexpr(4):
                                sp = max_f16x2(sp, cute.arch.shuffle_sync_bfly(sp, offset=1 << lvl))
                            sb, ssr_a, ssr_b = f16x2_e4m3_out(sp)
                            ssw = ssw | (sb << (16 * st))
                            ssr_s = [ssr_a, ssr_b]
                            for g in cutlass.range_constexpr(2):
                                gi = 2 * st + g
                                iv = cute.arch.rcp_approx(cute.arch.fmax(ssr_s[g], Float32(1e-30)))
                                tq = []
                                for j in cutlass.range_constexpr(8):
                                    e0 = 16 * g + 2 * j
                                    q0, q1 = utils.mul_packed_f32x2((tdPs[e0, 0, 0], tdPs[e0 + 1, 0, 0]), (iv, iv))
                                    tq.append(q0)
                                    tq.append(q1)
                                if const_expr(self.ds_sr):
                                    # hardware SR: 2 random words per 8 elements (murmur3 finaliser of a per-thread counter)
                                    rws = []
                                    for wj in cutlass.range_constexpr(4):
                                        xr = (rng + Int32(_i32_wrap((8 * gi + 2 * wj + 1) * -1640531527))) * Int32(-2048144789)
                                        xr = xr ^ ((xr >> 13) & Int32(0x7FFFF))
                                        xr = xr * Int32(-1028477387)
                                        rws.append(xr ^ ((xr >> 16) & Int32(0xFFFF)))
                                    qw[2 * gi] = packed_float_to_e2m1x4_rs(tq[0], tq[1], tq[2], tq[3], tq[4], tq[5], tq[6], tq[7], rws[0], rws[1])
                                    qw[2 * gi + 1] = packed_float_to_e2m1x4_rs(tq[8], tq[9], tq[10], tq[11], tq[12], tq[13], tq[14], tq[15], rws[2], rws[3])
                                else:
                                    qw[2 * gi] = sm100_utils.packed_float_to_e2m1(tq[0], tq[1], tq[2], tq[3], tq[4], tq[5], tq[6], tq[7])
                                    qw[2 * gi + 1] = sm100_utils.packed_float_to_e2m1(tq[8], tq[9], tq[10], tq[11], tq[12], tq[13], tq[14], tq[15])
                    pipeline_dS.producer_acquire(producer_state_dS)
                    # A of dQ (TMEM, K-tile of this WG) + its SF word, once every compute warp has loaded dP(t)
                    pipeline_dP.sync_object_empty.wait(0, phase_dPe_c)
                    phase_dPe_c ^= 1
                    if const_expr(not self.dq8):   # A from SMEM: nothing goes to TMEM at all
                        tQ4 = cute.make_rmem_tensor(tScP4_r2t.shape, Float32)
                        tQ4_u32 = cute.recast_tensor(tQ4, Int32)
                        for wi in cutlass.range_constexpr(16 if self.dq8 else 8):
                            tQ4_u32[wi] = qw[wi]
                        if const_expr(not self.dq8):
                            tQSF = cute.make_rmem_tensor(tScSF_r2t.shape, Float32)
                            tQSF_u32 = cute.recast_tensor(tQSF, Int32)
                            for j in cutlass.range_constexpr(cute.size(tQSF_u32)):
                                tQSF_u32[j] = ssw
                        cute.copy(thr_r2t_p4, tQ4, tdSq4_r2t)
                        if const_expr(not self.dq8):
                            cute.copy(thr_r2t_sf, tQSF, tdSqSF_r2t)
                    cute.arch.fence_view_async_tmem_store()
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()
                    if const_expr(not self.dkv8):
                        for wi in cutlass.range_constexpr(8):
                            cqw[wi] = qw[wi]
                            cpw[wi] = pw[wi]
                        c_ssw = ssw
                        c_svw = svw
                    consumer_state_S_P_dP.advance()
                    pipeline_dPsum.consumer_release(consumer_state_dPsum)
                    consumer_state_dPsum.advance()

                if const_expr(not self.dkv8):
                    # the last query tile's dK / dV operands
                    pipeline_dSkv.producer_acquire(producer_state_dSkv)
                    self._kv_flush(cqw, cpw, c_ssw, c_svw, abase, sfbase2, is_even, is_warp_leader, sdSt4_u8, sPt4_u8, sSFdSt_u16, sSFPt_u16)
                    with cute.arch.elect_one():
                        pipeline_dSkv.producer_commit(producer_state_dSkv)
                    producer_state_dSkv.advance()

                thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                f_k = f_v = None
                if const_expr(self.fold_epi):
                    f_k = sfkv[tidx // 128, dp_idx, 0]
                    f_v = sfkv[tidx // 128, dp_idx, 1]
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx, batch_idx, head_idx, n_block, thr_mma_dV, tdVtdV, mdV_tma_tensor, sdV, tma_atom_dV,
                    thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV,
                    (self._scl[dsc, 1] if const_expr(self.doc_varlen) else self._scl[1]) if const_expr(self.dev_scl)
                    else dv_scale, self.epi_barrier_base, "V",
                    f_v, (pk_kp, pk_cend) if const_expr(self.codes_packed) else None,
                )
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx, batch_idx, head_idx, n_block, thr_mma_dK, tdKtdK, mdK_tma_tensor, sdK, tma_atom_dK,
                    thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV,
                    (self._scl[dsc, 0] if const_expr(self.doc_varlen) else self._scl[0]) if const_expr(self.dev_scl)
                    else dk_scale, self.epi_barrier_base, "K",
                    f_k, (pk_kp, pk_cend) if const_expr(self.codes_packed) else None,
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ dK / dV epilogue (TMA store; as the C3 kernel)
    @cute.jit
    def split_wg(self, t: cute.Tensor, wg_idx: cutlass.Int32, num_wg: cutlass.Constexpr[int]):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            if const_expr(rank == 3):
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg))
                coord = (None, None, (None, wg_idx)) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2], reduced_shape[3] // num_wg))
                coord = (None, None, None, (None, wg_idx)) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self, tidx, batch_idx, head_idx, n_block, thr_mma, tdKVtdKV, mdKV, sdKV, tma_atom_dKV,
        thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, scale, barrier_id, K_or_V: cutlass.Constexpr[str],
        f_row=None, pk_rows=None,
    ):
        if const_expr(pk_rows is not None):
            return self._epilogue_dKV_packed(tidx, head_idx, thr_mma, tdKVtdKV, pipeline_dKV, consumer_state_dKV, scale,
                                             K_or_V, pk_rows, f_row)
        tile_hdim = self.tile_hdim
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
        sdKV = sdKV[None, None, wg_idx]
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)
        mdKV_cur = mdKV[None, None, head_idx, batch_idx]
        gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0))
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(gdKV, epi_tile, (0, None))
        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV, 0, cute.make_layout(1), cute.group_modes(sdKV, 0, 2), cute.group_modes(gdKV_epi, 0, 2)
        )
        num_epi_stages = cute.size(tdKVgdKV.shape[1])
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32)
        pipeline_dKV.consumer_wait(consumer_state_dKV)
        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r = self.split_wg(thr_copy_t2r.partition_S(tdKVtdKV), wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]
            cdKV = cute.make_identity_tensor((self.tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r = self.split_wg(thr_copy_t2r.partition_D(tdKVcdKV), wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]
            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()
            scale_r = scale
            if const_expr(f_row is not None):
                # one TMEM lane = one key row per thread (32x32b): the row's power of two folded into the scale (exact)
                scale_r = scale * f_row
            for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale_r, scale_r)
                )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            if leader_warp:
                cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def _epilogue_dKV_packed(self, tidx, head_idx, thr_mma, tdKVtdKV, pipeline_dKV, consumer_state_dKV, scale,
                             K_or_V: cutlass.Constexpr[str], pk_rows, f_row=None):
        """(cov5 codes_packed) dK / dV of the key tile straight from TMEM to the packed key rows: each thread's TMEM lane
        is one key row, stored only for the document's real keys (a TMA tile would overwrite the next document's rows);
        the values as the TMA epilogue's (scale, then the output dtype)."""
        tile_hdim = self.tile_hdim
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        mG = self._pk[10] if const_expr(K_or_V == "K") else self._pk[11]   # (s, d, h_k, b) views of the packed rows
        kp, nr = pk_rows
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32)
        pipeline_dKV.consumer_wait(consumer_state_dKV)
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
        src = self.split_wg(thr_copy_t2r.partition_S(tdKVtdKV), wg_idx, num_wg)
        cdKV = cute.make_identity_tensor((self.tile_n, tile_hdim))
        crd = self.split_wg(thr_copy_t2r.partition_D(thr_mma.partition_C(cdKV)), wg_idx, num_wg)
        r32 = cute.make_rmem_tensor(crd.shape, Float32)
        cute.copy(thr_copy_t2r, src, r32)
        cute.arch.fence_view_async_tmem_load()
        if const_expr(f_row is not None):
            # (cov6 fold_epi) the thread's key row's power of two folded into the scale (exact), as the TMA epilogue
            scale = scale * f_row
        for i in cutlass.range(cute.size(r32.shape) // 2, unroll_full=True):
            r32[2 * i], r32[2 * i + 1] = cute.arch.mul_packed_f32x2((r32[2 * i], r32[2 * i + 1]), (scale, scale))
        rv = cute.make_rmem_tensor(r32.shape, dtype)
        rv.store(r32.load().to(dtype))
        if const_expr(PACKED_EPI_VEC and cute.size(rv.shape) % 8 == 0):
            # (cov6) 16-byte stores: the thread's TMEM lane is one key row and its values run along d in runs of 8
            # (the 32x32b load); one vector store per 8 columns instead of eight 2-byte stores
            for ch in cutlass.range_constexpr(cute.size(rv.shape) // 8):
                c = crd[8 * ch]
                if c[0] < nr:
                    v8 = cute.make_rmem_tensor(cute.make_layout(8), dtype)
                    for j in cutlass.range_constexpr(8):
                        v8[j] = rv[8 * ch + j]
                    gp = cute.make_ptr(dtype, mG[kp + c[0], None, head_idx, 0].iterator.toint() + c[1] * (dtype.width // 8),
                                       cute.AddressSpace.gmem, assumed_align=16)
                    cute.autovec_copy(v8, cute.make_tensor(gp, cute.make_layout(8)))
        else:
            for i in cutlass.range_constexpr(cute.size(rv.shape)):
                c = crd[i]
                if c[0] < nr:
                    mG[kp + c[0], c[1], head_idx, 0] = rv[i]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    # ------------------------------------------------------------------ reduce warps (dQ, two d halves in one TMEM window)
    @cute.jit
    def dQacc_reduce(self, mdQaccum, sdQ, tma_atom_dQ, thr_mma_dQh, tdQtdQ, pipeline_dQ, block_info, SeqlenInfoCls,
                     TileSchedulerCls, dq_shift=None):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQh.partition_C(cute.make_identity_tensor(self.mma_tiler_dQh[:2]))
        tdQrdQ_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        atom_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dq_reduce_dtype, num_bits_per_copy=128)
        thr_layout_r2s = cute.make_ordered_layout((num_reduce_threads, 1), order=(1, 0))
        val_layout_r2s = cute.make_ordered_layout((1, 128 // self.dq_reduce_dtype.width), order=(1, 0))
        thr_copy_r2s = cute.make_tiled_copy_tv(atom_r2s, thr_layout_r2s, val_layout_r2s).get_slice(tidx)
        one = cute.make_layout(1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        smem_stage = Int32(0)
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_off, n_loc = self._tile_range(block_info, SeqlenInfoCls, batch_idx, n_block)
            nm_r = m_block_max - m_block_min
            loop_count = nm_r * self.qhead_per_kvhead
            for iter_idx in cutlass.range(loop_count, unroll=1):
                if const_expr(self.qhead_per_kvhead > 1):
                    qh_r = iter_idx // nm_r
                    m_block = m_block_min + (iter_idx - qh_r * nm_r)
                    hq_r = head_idx * self.qhead_per_kvhead + qh_r
                else:
                    m_block = m_block_min + iter_idx
                    hq_r = head_idx
                if const_expr(self.codes_packed):
                    # (cov5) the query tile's packed rows (the accumulator (1, h, T', 128)): adding a tile past the
                    # document adds the padding rows' dQ (their dS is 0) to the next document's rows
                    gdQ = cute.local_tile(cute.domain_offset((seqlen.offset_q - m_off * self.tile_m, 0),
                                                             mdQaccum[None, None, hq_r, batch_idx]),
                                          (self.tile_m, self.dQ_reduce_ncol), (None, None))
                else:
                    gdQ = cute.local_tile(mdQaccum[None, None, hq_r, batch_idx], (self.tile_m, self.dQ_reduce_ncol), (None, None))
                tdQsdQ, tdQgdQ = cpasync.tma_partition(tma_atom_dQ, 0, one, cute.group_modes(sdQ, 0, 2), cute.group_modes(gdQ, 0, 2))
                for half in cutlass.range_constexpr(2):
                    sdQ_cur = sdQ[None, None, smem_stage]
                    tdQsdQ_r2s = thr_copy_r2s.partition_D(sdQ_cur)
                    pipeline_dQ.consumer_wait(dQ_consumer_state)
                    tdQrdQ = cute.make_rmem_tensor(tdQrdQ_shape, Float32)
                    cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipeline_dQ.consumer_release(dQ_consumer_state)
                    dQ_consumer_state.advance()
                    tdQrdQ_f32 = cute.make_tensor(tdQrdQ.iterator, tdQsdQ_r2s.shape)
                    if const_expr(self.dq8):
                        # dq8: both operands carry the full E4M3 range (the 4-bit path's block scales were applied by the
                        # MMA, so its accumulator held dequantised products), and these partials are reduce-added in f16
                        # across every key tile -- so they are pre-scaled by a pow2 that the host sizes from the key count,
                        # and dq_scale carries it back
                        for i in cutlass.range_constexpr(cute.size(tdQrdQ_f32) // 2):
                            a, b = utils.mul_packed_f32x2((tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1]), (dq_shift, dq_shift))
                            tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1] = a, b
                    tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dq_reduce_dtype)
                    utils.cvt_f16(tdQrdQ_f32, tdQrdQ_r2s)
                    cute.copy(thr_copy_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    cute.arch.fence_view_async_shared()
                    self.reduce_sync_barrier.arrive_and_wait()
                    if is_tma_warp:
                        cute.copy(tma_atom_dQ, tdQsdQ[None, smem_stage], tdQgdQ[None, m_block, half])
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=True)
                    self.reduce_sync_barrier.arrive_and_wait()
                    smem_stage = (smem_stage + 1) % self.sdQaccum_stage
            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            self.reduce_sync_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        if is_tma_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=False)

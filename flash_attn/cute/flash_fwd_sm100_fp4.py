# Portions of this file are adapted from the FP4 FlashAttention-4 forward of
# hao-ai-lab/flash-attention-fp4 (https://github.com/hao-ai-lab/flash-attention-fp4,
# branch fp4, commit 5aa37a96), BSD-3-Clause.
# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
import os
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial
from dataclasses import dataclass
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Boolean, const_expr, Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Float4E2M1FN
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from flash_attn.cute.blackwell_helpers import make_smem_layout_sfa, make_smem_layout_sfb, BlockScaledBasicChunk
import cutlass.utils.blockscaled_layout as blockscaled_utils
from flash_attn.cute.paged_kv import PagedKVManager
import flash_attn.cute.utils as utils
from flash_attn.cute.lowprec.specs import FwdKernelSpec
from flash_attn.cute import copy_utils
import flash_attn.cute.pipeline as pipeline
from flash_attn.cute.mask import AttentionMask
# from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.softmax import SoftmaxSm100, apply_learnable_sink, apply_score_mod_inner
from flash_attn.cute.utils import AuxData


def _as_aux_data(aux_tensors):
    """Upstream replaced the bare `aux_tensors` list with an AuxData(tensors, scalars) tuple in
    apply_mask_sm100 / apply_score_mod_inner. This kernel still threads the list internally, so it
    is converted at the two call boundaries rather than rewritten throughout."""
    if aux_tensors is None:
        return AuxData()
    if isinstance(aux_tensors, AuxData):
        return aux_tensors
    return AuxData(tensors=aux_tensors)
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.blackwell_helpers import packed_float_to_ue4m3, packed_float_to_e2m1
from flash_attn.cute.fast_math import FastDivmod
from cutlass.cutlass_dsl import T as _T, dsl_user_op as _dsl_user_op
from cutlass.cute.arch import llvm as _llvm
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    PersistentLPTScheduler,
    PersistentZigzagLPTScheduler,
    SingleTileVarlenScheduler,
    VarlenWorkListScheduler,
    VarlenWorkListDynScheduler,
    ParamsBase,
)

# Direct-P magic-number window (DIRECTP_WINDOW below). The bit trick of the Mitchell classifier
# (blackwell_helpers.DIRECTP_*) is exact only while t = (x - delta) * 2^16 + MAG stays in [2^23, 2^24), where
# x = (s - group max) * scale + log2 6 <= log2 6. The shared constants centre the window (MAG = 1.5 * 2^23):
# x - delta in [-64, 64), so half of it (x > 2.6) is never used and a score more than ~64 log2 units (44 nats)
# below its 16-key group max leaves it; the shifted pattern then wraps into garbage (x = -100 -> -1.2e36,
# x = -191 -> 5.6e13, some patterns inf / NaN): the Mitchell row sum goes non-finite / negative -> LSE NaN,
# O row 0 (toy-transformer GQA inputs, a row with an 84-nat logit range), and the codes of those keys can be
# wrong. Re-centring (MAG = 2^23 + L * 2^16, LEA_ADD = (127 << 23) - (bits(MAG) << 7)) moves the window to
# x - delta in [-L, 128 - L): L = 124 covers x in [-123.9, log2 6] (86 nats below the group max) with the same
# instructions and the same in-window values (max rel err vs 2^x 4.69 % for every L, measured by a sweep over x).
# A window of 64 would restore the shared constants.
# The 4-bit P^ steps always clamp the bit pattern of t from below at bits(2^23) = 0x4B000000 (one max.s32 per element,
# ALU pipe; clamp 2 = max.s32 + mad.lo): exact for any range. The window-only form (clamp 0) is gone (the 8-bit
# P^ is exp2 only).
_DIRECTP_T_FLOOR_BITS = 0x4B000000  # bits of 2^23 as f32

# The values the forward resolves and every kernel / reference that reproduces its P^ must share (the all-4-bit backward
# flash_bwd_sm100_fp4q, interface_bwd_fp4q's LSE offset, tests/cute/_fp4q_ref). They used to be tunable (the Direct-P
# delta and window, the emulation polynomial and clamp, the masked-group placeholder, the E2M1-PV rescale threshold's
# default); they are fixed at the values the default path always used.
DIRECTP_DELTA = math.log2(1.25) - 0.25   # the Mitchell classifier's bias delta
DIRECTP_WINDOW = 124                      # the classifier window's low end (directp_window_consts): 86 nats exact
E2E_POLY_E2M1_PV = 2                      # FMA-pipe exp2 polynomial on the 4-bit-PV routes (packed degree-2 minimax)
E2E_POLY_E4M3_PV = 3                      # ... elsewhere (utils.ex2_emulation_2, degree 3)
E2E_CLAMP = True                          # the emulation's -126 clamp: without it the emulation wraps to NaN
SPLIT_MASKED_GM = -1e9                    # group max of an all-masked 16-key group (finite placeholder; the backward's gm_floor)
FP4_PV_RESCALE_THRESHOLD = 8.0            # the E2M1-PV lazy P-frame rescale threshold (log2 units); the default of the
                                          # FwdKernelSpec.fp4_pv_rescale_threshold, which the plan sets (to 0) only for the
                                          # 4-bit-PV route's per-tile frame, never on the all-4-bit route


def directp_window_consts(window_low: int):
    """(MAG, signed LEA add) of the Direct-P classifier with the window x - delta in [-window_low, 128 - window_low)."""
    import struct
    assert 3 <= window_low <= 125, window_low
    mag = float(2 ** 23 + window_low * 2 ** sm100_utils.DIRECTP_FRAC_BITS)
    bits = struct.unpack("<I", struct.pack("<f", mag))[0]
    lea = ((127 << 23) - ((bits << (23 - sm100_utils.DIRECTP_FRAC_BITS)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return mag, (lea if lea < 2**31 else lea - 2**32)


P_MIX_CODES = ("M", "E", "2", "3")


def parse_p_mix(p_mix):
    """p_mix (8 characters over M / E / 2 / 3, one per 16-key group position of a 128-key tile) -> tuple, or None."""
    if p_mix is None:
        return None
    t = tuple(str(p_mix).upper())
    if len(t) != 8 or any(c not in P_MIX_CODES for c in t):
        raise ValueError(f"p_mix must be 8 characters over {P_MIX_CODES} (one per 16-key group position), got {p_mix!r}")
    return t


def _directp_lea_asm(a: str, b: str, lea: int, clamp) -> str:
    """t pattern a -> Mitchell pattern b: [max.s32 clamp] + shl/add (clamp 0 / True) or [max.s32] + mad.lo (clamp 2)."""
    sh = 23 - sm100_utils.DIRECTP_FRAC_BITS
    cl = f"max.s32 {a}, {a}, {_DIRECTP_T_FLOOR_BITS};\n\t" if clamp else ""
    if clamp == 2:
        return cl + f"mad.lo.s32 {b}, {a}, {1 << sh}, {lea};\n\t"
    return cl + f"shl.b32 {b}, {a}, {sh};\n\tadd.s32 {b}, {b}, {lea};\n\t"


def mitchell_value_w(t: Float32, lea: int, clamp, *, loc=None, ip=None) -> Float32:
    """blackwell_helpers.mitchell_value with a chosen LEA constant (window) and an optional domain clamp."""
    out = _llvm.inline_asm(
        _T.f32(),
        [Float32(t).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, b;\n\t"
        "mov.b32 a, $1;\n\t" + _directp_lea_asm("a", "b", lea, clamp) + "mov.b32 $0, b;\n\t"
        "}\n",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """fma.rn.f32: a*b + c with one rounding, pinned. Written out as a*b + c the contraction is ptxas's
    choice per site and changes with unrelated code nearby (the NVFP4 split step's group constants
    came out as FFMA in one build and FMUL + FADD in another: 1-ulp SF / row-sum / LSE differences)."""
    out = _llvm.inline_asm(
        _T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


def e2m1x8_values2(w: Int32, *, loc=None, ip=None):
    """8 packed E2M1 codes (element 0 in the low nibble, non-negative) -> 8 f32 = 2 x their values (PRMT LUT, BFE, I2F)."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 8),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 lo, hi, p0, p1, s, t;\n\t"
        "mov.b32 lo, 0x03020100;\n\tmov.b32 hi, 0x0C080604;\n\t"
        "prmt.b32 p0, lo, hi, $8;\n\tshr.u32 s, $8, 16;\n\tprmt.b32 p1, lo, hi, s;\n\t"
        + "".join(f"bfe.u32 t, p0, {8 * i}, 8;\n\tcvt.rn.f32.u32 ${i}, t;\n\t" for i in range(4))
        + "".join(f"bfe.u32 t, p1, {8 * i}, 8;\n\tcvt.rn.f32.u32 ${4 + i}, t;\n\t" for i in range(4))
        + "}\n",
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(8))


def f32_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    """The bit pattern of an f32 as an i32 (mov.b32)."""
    out = _llvm.inline_asm(_T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f",
                           has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
    return Int32(out)


def bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    """An i32 bit pattern as an f32 (mov.b32)."""
    out = _llvm.inline_asm(_T.f32(), [Int32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r",
                           has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
    return Float32(out)


def ceil_pow2_f32(a: Float32, *, loc=None, ip=None):
    """fwdgap "sfi": (ceil(a), 2^ceil(a)) without the XU pipe -- a magic add rounded up (ceil in the low mantissa bits,
    exact for |a| < 2^22), the exact back-subtract, and 2^k assembled as the exponent field ((k << 23) + (127 << 23),
    one LEA). Equals (cvt.rpi.f32.f32, ex2.approx(ceil)) wherever ex2.approx is exact on integers."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32(), _T.f32()]),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 t;\n\t.reg .b32 tb, e;\n\t"
        "add.rp.f32 t, $2, 0f4B400000;\n\t"
        "sub.rn.f32 $0, t, 0f4B400000;\n\t"
        "mov.b32 tb, t;\n\t"
        "shl.b32 e, tb, 23;\n\t"
        "add.s32 e, e, 1065353216;\n\t"
        "mov.b32 $1, e;\n\t}\n",
        "=f,=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return (Float32(_llvm.extractvalue(_T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(_llvm.extractvalue(_T.f32(), out, [1], loc=loc, ip=ip)))


def sch_bf16_pair(t0: Float32, t1: Float32, *, loc=None, ip=None):
    """fwdgap "sch": a Schraudolph exp2 read out of the fixed-point FFMA t = x 128 + (127 128 + 1.5 2^23): the low 16
    bits of t are the bf16 pattern of ~2^x (linear mantissa), placed as the high half of an f32 (shl 16) and clamped at 0
    (x < -127 gives a negative pattern). Valid for x >= -383 (below that the 16-bit field wraps)."""
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32(), _T.f32()]),
        [Float32(t0).ir_value(loc=loc, ip=ip), Float32(t1).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, b;\n\t.reg .f32 fa, fb;\n\t"
        "mov.b32 a, $2;\n\tmov.b32 b, $3;\n\t"
        "shl.b32 a, a, 16;\n\tshl.b32 b, b, 16;\n\t"
        "mov.b32 fa, a;\n\tmov.b32 fb, b;\n\t"
        "max.f32 $0, fa, 0f00000000;\n\tmax.f32 $1, fb, 0f00000000;\n\t}\n",
        "=f,=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return (Float32(_llvm.extractvalue(_T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(_llvm.extractvalue(_T.f32(), out, [1], loc=loc, ip=ip)))


# fwdgap "sch": t = x * 128 + SCH_C, SCH_C = 1.5 2^23 + 128 (127 - 0.0573): the -0.0573 octave shift makes the linear
# mantissa's ratio to 2^x average 1 over the fraction (range -3.9 % .. +2.0 %), so the row sum / LSE carry no mean bias
SCH_C = 12582912.0 + 128.0 * (127.0 - 0.0573)


def packed_mitchell_to_e2m1_w(t0, t1, t2, t3, t4, t5, t6, t7, lea: int, clamp, *, loc=None, ip=None) -> Int32:
    """blackwell_helpers.packed_mitchell_to_e2m1 (LEA form) with a chosen LEA constant and an optional domain clamp."""
    movs = "".join(f"mov.b32 a{i}, ${i + 1};\n\t" for i in range(8))
    lea_s = "".join(_directp_lea_asm(f"a{i}", f"b{i}", lea, clamp) for i in range(8))
    out_uint32 = _llvm.inline_asm(
        _T.i32(),
        [Float32(t).ir_value(loc=loc, ip=ip) for t in (t0, t1, t2, t3, t4, t5, t6, t7)],
        "{\n\t"
        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
        ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        + movs
        + lea_s
        + "cvt.rn.satfinite.relu.e2m1x2.f32   byte0, b1, b0;\n\t"
        "cvt.rn.satfinite.relu.e2m1x2.f32   byte1, b3, b2;\n\t"
        "cvt.rn.satfinite.relu.e2m1x2.f32   byte2, b5, b4;\n\t"
        "cvt.rn.satfinite.relu.e2m1x2.f32   byte3, b7, b6;\n\t"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(out_uint32)


# === TUNING KNOBS ===
# Keys: (is_causal: bool, head_dim_padded: int)
# Values:
#   e2e_freq: int -- exp2 emulation frequency (0=all hardware exp2)
#   e2e_start_frg: int -- fragment index to start emulation from
#   num_regs_softmax: int -- register count for softmax warps (multiple of 8)
#   num_regs_correction: int -- register count for correction warps (multiple of 8)
#   num_regs_other: derived as 512 - num_regs_softmax * 2 - num_regs_correction

def _is_sm103():
    """B300 / GB300 (SM 10.3: 2x MUFU.EX2 throughput, tcgen05.ld.red) as the target of a compile started now: the
    library's one architecture query (interface._get_device_arch: the arch override for CPU-only compiles, else the
    current CUDA device). The interface resolves it once per call and passes it to the kernel (is_sm103=, in the compile
    key); the kernel calls this only when constructed directly."""
    try:
        from flash_attn.cute.interface import _get_device_arch   # (lazy: interface imports the kernel modules)
        return _get_device_arch() == 103
    except Exception:
        return False

# SM100 (B200/GB200): e2e emulation helps overlap FP32 FMA with MMA.
_FP4_TUNING_CONFIG_SM100 = {
    # BF16 PV: e2e_freq=16 verified via bench_fp4.py (1921 TF peak on B200).
    (False, 128): {"e2e_freq": 16, "e2e_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 80, "enable_e2e": True},
    (True, 128):  {"e2e_freq": 16, "e2e_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 80, "enable_e2e": True},
}
# SM103 (B300/GB300): 2x MUFU throughput -- e2e emulation hurts because hw exp2
# already finishes in 512 cycles (was 1024 on SM100). Disabling e2e: 2100 TF
# (vs 1905 TF with e2e on) for BF16 PV on GB300 at 2070 MHz.
_FP4_TUNING_CONFIG_SM103 = {
    (False, 128): {"e2e_freq": 16, "e2e_start_frg": 0, "num_regs_softmax": 192, "num_regs_correction": 80, "enable_e2e": False},
    (True, 128):  {"e2e_freq": 16, "e2e_start_frg": 0, "num_regs_softmax": 192, "num_regs_correction": 80, "enable_e2e": False},
}

def _fp4_tuning_config(is_sm103: bool) -> dict:
    """The per-shape table of the target architecture (picked per kernel, not once at import)."""
    return _FP4_TUNING_CONFIG_SM103 if is_sm103 else _FP4_TUNING_CONFIG_SM100

# FP8 PV overrides: when v_dtype.width == 8 and quant_pv == False
_FP4_FP8PV_TUNING_CONFIG = {
    # FP8 PV: e2e_freq=9 verified on B200 (2018 TF peak). On B300, e2e is
    # irrelevant (~1910 TF regardless) -- F2FP packing is the softmax bottleneck.
    (False, 128): {"e2e_freq": 9, "e2e_start_frg": 0},
    (True, 128):  {"e2e_freq": 9, "e2e_start_frg": 0},
}
# === END TUNING KNOBS ===


# Rows covered by one 32x4 block-scaled scale-factor atom. Under cu_seqlens the SF tensor's mode 0 is
# ((32, 4), n_atoms) and sub-mode 1 counts ATOMS, so document offsets are divided by this.
_SF_ATOM_ROWS = 128

class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


@_dsl_user_op
def mbar_try_wait_hint(mbar_ptr, phase, hint_ns: int, *, loc=None, ip=None) -> Boolean:
    """mbarrier.try_wait.parity with an explicit suspend-time hint (ns): the thread
    sleeps at most `hint_ns` before returning False, so a warp can poll two barriers
    without spinning at full issue rate."""
    r = _llvm.inline_asm(
        _T.i32(),
        [mbar_ptr.toint().ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2, %d;\n\tselp.u32 $0, 1, 0, p;\n}" % int(hint_ns),
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Int32(r) != 0


def _nvvm_scope_cluster():
    """nvvm MemScopeKind.CLUSTER (cov2: the pair's peer arrive after its SF atom stores -- a DSMEM data hand-off)."""
    from cutlass._mlir.dialects import nvvm
    return nvvm.MemScopeKind.CLUSTER


V4_SF_BYTES = 1040   # cov5 v4_packed: a key tile's dense V SF words in its staging slot (rows (ps & ~1) .. + 130)
SFD_KV_SLOTS = 4   # sf_dense: the key tiles' staging ring (bulk copies issued SFD_KV_SLOTS - 1 key tiles ahead; a power of 2)


class FlashAttentionForwardSm100:
    # The MXFP4 split path's integer log2 frame offset: max_offset of the UE8M0 group SFs (which need an INTEGER
    # fold) AND the offset _lse_log2_offset subtracts, one constant for both. 3 ~ log2 6 + log2 448 - 8, rounded.
    # It does NOT depend on the MXFP4 rescale threshold, and must not: a suspected defect
    # (`mxfp4-threshold-invisible-to-lse`) assumed max_offset subtracts that threshold on this path, but only the NVFP4
    # two-level branch does; here max_offset was always this constant, so the LSE was never out of step. Measured:
    # the exported LSE matches the exact log-sum-exp of the dequantised operands to
    # 1e-4 nats at thresholds 0, 4, 8 and 12, dense and causal; coupling the offset to the threshold (3 -> 7 -> 11)
    # moves neither O nor LSE, because a power-of-two frame cancels in UE8M0-scaled E2M1 codes and in P^ / Z.
    MX_INT_FRAME_OFFSET = 3

    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        sf_dtype: Optional[Type[cutlass.Numeric]] = None,
        sf_vec_size: Optional[int] = None,
        # PV-side scale-factor format; None mirrors the QK side. Set to
        # (Float8E8M0FNU, 32) for MXFP8 P/V with NVFP4 QK.
        sf_dtype_pv: Optional[Type[cutlass.Numeric]] = None,
        sf_vec_size_pv: Optional[int] = None,
        pv_elem_width: Optional[int] = None,  # 4 (E2M1 P/V) or 8 (E4M3 P/V); None -> infer from sf_vec_size_pv
        qk_elem_width: Optional[int] = None,  # 4 (E2M1 Q/K: NVFP4 or MXFP4) or 8 (MXFP8); None -> infer from sf_vec_size
        qk_split_pipe: Optional[bool] = None,  # split-QK cross-tile pipeline (interface decides; env overrides)
        qk_split_mask_seqlen: bool = False,  # split pipeline: the first KV tile needs the seqlen_k mask (ragged)
        # optional per-query log P_max = max_k S_qk * softmax_scale - LSE_q (nats, float32, LSE layout) stored next to
        # LSE (mPmax); the softmax warps track the true running row max (the rescale threshold keeps a stale one)
        return_pmax: bool = False,
        # ---- prequantised-qk-fused-scale (prequant.py) -----------------------------------------------------------------
        # The caller folded log2(e) * softmax_scale (and the operand descales) into Q before quantisation, so the
        # kernel's single score scalar is exactly 1.0 and every multiplication by it can be dropped. The interface
        # sets this only when the scalar came out bitwise 1.0; it is in the compile key.
        score_scale_unit: bool = False,
        # ---- diagonal-only-key-local-v-scale ---------------------------------------------------------------------------
        # Apply the key-local (vtok) V scale ONLY on the causal masked steps, the only tiles in which a V scale group
        # can straddle the causal boundary. The caller then supplies V (and SFV) as TWO payloads concatenated along the
        # key axis: keys [0, s_k) carry the fast key-grouped quantisation read by the fully unmasked tiles, keys
        # [s_k, 2 s_k) the key-local one (mVBias / mVU) read by the masked tiles. Causal only, no varlen / paged /
        # split KV. The host setting is QuantSpec.vtok_diag (interface_bwd_fp4q.fp4all_vtok_mode); it is in the key.
        vtok_diag: bool = False,
        # ---- mixed exponent paths on the exact all-FP4 step (exact classifier; reproduce-forward-p only) ------------------
        # p_mix: None (the value function the spec selects: Mitchell, or FwdKernelSpec.fp4all_exp2) or 8 characters, one per
        # 16-key group POSITION g = (key mod 128) / 16 of every key tile: 'M' the clamped Mitchell classifier (ALU: max.s32 +
        # mad.lo per element), 'E' ex2.approx (MUFU) + the cvt pack, '2' / '3' the clamped degree-2 / degree-3 FMA-pipe
        # exp2 (_ex2_emu_pair). A pure compile-time function of the group position, so the backward rebuilds P^ bit for bit
        # (the reproduce-forward-p contract). Mitchell groups keep the 2^16-scaled classifier constants; the others the exp2
        # argument s scale + (log2 6 - gm scale). "MMMMMMMM" is the Mitchell step itself.
        p_mix: Optional[str] = None,
        # FwdKernelSpec.o_regstore's default, resolved by the interface before its compile key (it depends on the route,
        # the mask and seqlen_k, which the constructor does not see). None = "0". An explicit spec value still wins.
        o_regstore_default: Optional[bool] = None,
        # the target is SM103 (B300): the interface resolves it (FLASH_ATTENTION_ARCH, else the device) and keys on it;
        # None = _is_sm103() now (direct construction). Every SM103 rule of this kernel reads self.is_sm103.
        is_sm103: Optional[bool] = None,
        # FwdKernelSpec.fp8_generic_e2e's default (pure-E4M3 generic step), resolved by the interface for the same reason. None = 6.
        fp8_generic_e2e_default: Optional[int] = None,
        # ---- mxfp8-pv-row-frame-p ------------------------------------------------------------------------------------
        # MXFP8 P.V (block-scaled E4M3 V, UE8M0 SFV per 32 keys) with the ROW-FRAME E4M3 P^ of the per-tensor 8-bit step:
        # P^ = E4M3(P * 2^offset) against the (lazy) row max, exactly the pure-FP8 generic step's P^ (fused softmax loop,
        # FMA-pipe exp2 share, (4, 4) offset / threshold). The P scale factors are a CONSTANT tile of 1.0 (UE8M0 127),
        # written once per CTA, so the softmax computes no group maxes / SF exponents and stores no SF bytes; the PV GEMM
        # stays block-scaled for V's token-block factors (no V factor is folded into P, so no in-block V -> O leak).
        # The row sum carries 2^offset exactly as on the per-tensor route (LSE subtracts it).
        mxfp8_pv_row_p: bool = False,
        # ---- token-block-qk-fold --------------------------------------------------------------------------------------
        # With token-block E4M3 Q / K (one UE8M0 factor per 32 tokens x the whole head dim, mTbFac rows 0 / 1), the S GEMM
        # is a plain kind::f8f6f4 GEMM on the codes and f_Q(row) f_K(32-key group) is folded into the softmax's one
        # scale-subtract FFMA2 per pair (a_g * scale_log2) and its row max (max_g a_g max_{j in g} S_j, exact: a_g is a
        # power of two). No SFQ / SFK tiles, TMA loads or S2T copies. Needs mxfp8_pv_row_p (the V factors stay in the
        # block-scaled P.V GEMM).
        token_block_qk_fold: bool = False,
        # ---- two-cta-forward ------------------------------------------------------------------------------------------
        # A CTA pair (cluster 2 x 1) runs every QK / PV GEMM as one cta_group::2 MMA of M = 256: each CTA keeps its own
        # 128 query rows per Q stage (S / P / O in its own TMEM, its own softmax and correction warps) and loads HALF of
        # every K / V tile (64 keys of K, 64 head-dim columns of V^T), so per CTA the K / V SMEM traffic and footprint
        # halve. The leader CTA's MMA warp issues every MMA and scale-factor copy for both; the B-side scale factors
        # (SFK, SFV) are multicast to both CTAs' SMEM. The cluster tile is 2 x q_stage x 128 = 512 query rows. Under
        # causal masking each softmax warpgroup processes exactly the KV tiles the 1-CTA kernel gives its rows (the
        # pair's extra leading tiles, fully masked for that warpgroup, are bookkept without computation), so O and LSE
        # are bitwise those of the 1-CTA forward. Generic softmax step only (no split pipeline / SplitKV /
        # varlen / paged KV / pack_gqa / persistent causal). Compile-time; keyed by the interface.
        use_2cta_forward: bool = False,
        # causal on a persistent grid (is_persistent): PersistentZigzagLPTScheduler's balanced assignment (global
        # longest-first order, boustrophedon over the CTAs) instead of PersistentLPTScheduler's stripes. The same work
        # tiles, permuted: O and LSE bitwise. Compile-time; keyed by the interface.
        causal_zigzag_order: bool = False,
        # causal, 1-CTA: Q stage 0 (the first 128 rows of the 256-row tile) skips the tile's last KV block, which lies
        # wholly above its diagonal -- no QK(0) / PV(0) / softmax / correction step for it (mma_pair's stage-0 skip,
        # two-cta-forward, on one CTA), and each stage masks only its own diagonal KV tile. O and LSE bitwise.
        # Compile-time; keyed by the interface.
        causal_stage_skip: bool = False,
        # aligned-start documents (cu_seqlens_aligned: every document starts on a 128-row tile and its slot holds the
        # whole last tile): the epilogue writes the WHOLE last Q tile of a document, the rows past seqlen_q as zeros
        # (their scale is 0), so the padding rows of O stay neutral at no extra pass. Compile-time; keyed by the driver.
        zero_pad_rows: bool = False,
        # aligned-start documents only (with zero_pad_rows): the batched epilogue -- O through sO and the epilogue warp's
        # TMA store instead of the correction warps' predicated copies. Every 128-row Q tile that starts inside a
        # document lies inside its slot (128-aligned starts, the slot holds the whole last tile), so the tile is stored
        # whole (its rows past seqlen_q are the zeros zero_pad_rows writes, as the correction-warp copy does); a tile
        # that starts at or past seqlen_q is not stored (it may lie in the next document's slot). It also lifts the
        # correction-warp-epilogue condition of causal_stage_skip. O and LSE bitwise. Compile-time; keyed by the driver.
        varlen_tma_o: bool = False,
        # lp_docmaps (aligned documents, b = 1): O is written to the caller's PACKED rows -- document b's row r goes to
        # row mCuSeqlensO[b] + r of mO (T, h, dv) -- by predicated copies of the real rows only (no TMA store, no zero
        # rows: the packed buffer has no padding). LSE stays at the aligned rows. Compile-time; keyed by the driver.
        o_rows_packed: bool = False,
        # lp_docmaps packed rows (with o_rows_packed): the Q / K / V codes and the LSE are the caller's PACKED rows too --
        # document b's Q rows (and LSE) from mCuSeqlensO[b], its K / V rows (V: the channel-major (d, T) codes' keys)
        # from mCuSeqlensKc[b]; the scale factors stay per aligned 128-row tile (mCuSeqlensQ / mCuSeqlensK, the aligned
        # starts). A tile's rows past its document's end are the next document's: finite codes, masked by seqlen_q /
        # seqlen_k as the aligned layout's padding rows (P = 0; their O / LSE rows are not stored). Compile-time; keyed.
        codes_packed: bool = False,
        # sfd (SF redesign; with codes_packed, token-block E4M3 / NVFP4 S operands): the scale-factor atoms are built
        # in SMEM by the load warp from the dense per-row tables (mSfdF: EvQ, EvK, EvV (b h, R) uint8 [, SF4Q, SF4K
        # (b h, R d / 64) int32 on the NVFP4 S route]; flash_attn.cute.sfd_atoms) instead of TMA'd per-aligned-tile
        # atoms, and V is the caller's row codes (an MN-major P.V operand, mCuSeqlensVt = the packed key starts).
        # Compile-time; keyed by the driver.
        sf_dense: bool = False,
        # per-(document, kv head) score scale (per-tensor E4M3 operands on aligned varlen documents): mSoftmaxScale is a
        # (n_docs, h_kv) float32 table (softmax_scale s_q s_k of each document and kv head), read by the softmax and
        # correction warps at the start of every work tile instead of once per kernel. Compile-time; keyed by the driver.
        score_scale_doc: bool = False,
        # aligned varlen work list (mVarlenWork) claiming (opt-in; HostSpec.varlen_fwd_sched): 'static' =
        # VarlenWorkListScheduler's boustrophedon deal (the default, untouched), 'atomic' / 'clc' =
        # VarlenWorkListDynScheduler (dynamic claiming in list order by a global counter / Cluster Launch Control).
        # Scheduling only: O and LSE bitwise. Compile-time; keyed by the driver. varlen_sched_stages: its SMEM ring depth.
        varlen_sched: str = "static",
        varlen_sched_stages: int = 2,
        varlen_sched_jit: bool = False,
        varlen_sched_early: bool = False,
        # ---- every compile-time setting of the constructor (lowprec/specs.py FwdKernelSpec). Required: the driver
        # builds it from the call's plan (no default, so a construction site cannot silently fall back to defaults). causal_zigzag_order / causal_stage_skip above stay
        # the resolved driver values.
        *,
        spec: FwdKernelSpec,
    ):
        self.spec = spec
        # fwdgap: opt-in experimental switches, FwdKernelSpec.exp ("" = the certified kernel)
        self.xf = frozenset(x for x in spec.exp.split(",") if x)
        self.zero_pad_rows = bool(zero_pad_rows)
        self.varlen_tma_o = bool(varlen_tma_o and is_varlen_q)
        assert not self.varlen_tma_o or self.zero_pad_rows, "varlen_tma_o: aligned documents (zero_pad_rows) only"
        self.o_rows_packed = bool(o_rows_packed)
        assert not self.o_rows_packed or (self.zero_pad_rows and is_varlen_q), "o_rows_packed: aligned documents only"
        self.codes_packed = bool(codes_packed)
        assert not self.codes_packed or self.o_rows_packed, "codes_packed: with o_rows_packed"
        self.sf_dense = bool(sf_dense)
        assert not self.sf_dense or self.codes_packed, "sf_dense: with codes_packed"
        self.score_scale_doc = bool(score_scale_doc)
        assert varlen_sched in ("static", "atomic", "clc"), varlen_sched
        self.varlen_sched = varlen_sched
        self.varlen_sched_stages = int(varlen_sched_stages)
        self.varlen_sched_jit = bool(varlen_sched_jit)
        self.varlen_sched_early = bool(varlen_sched_early and varlen_sched_jit)
        self.use_2cta_forward = bool(use_2cta_forward)
        # cov2: the pair on lp_docmaps packed rows (sf_dense; O by TMA tiles + the partial tiles' row copies)
        self.o_rows_tma = bool(self.o_rows_packed and self.use_2cta_forward)
        # spd3 (o_rows_tma1): the 1-CTA forward on packed rows with the sO epilogue warp (varlen_tma_o) -- a 128-row tile
        # inside its document by the TMA store at its packed start, a document's partial last tile by the epilogue
        # warp's row copies (a rolled loop); the pair's o_rows_tma epilogue on one CTA (o_rows_tile: either)
        self.o_rows_tma1 = bool(self.o_rows_packed and self.varlen_tma_o and not self.use_2cta_forward)
        self.o_rows_tile = bool(self.o_rows_tma or self.o_rows_tma1)
        self.causal_zigzag_order = bool(causal_zigzag_order)
        self.causal_stage_skip = bool(causal_stage_skip)
        self.cta_group_size = 2 if self.use_2cta_forward else 1
        # arrivals per warp on the MMA-side barriers (P full / O rescaled, P full 2, S read): every thread on the
        # 1-CTA kernel, one elected thread per warp on the pair (remote arrives, see _arrive_mma)
        self._mma_arrive_threads = 1 if self.use_2cta_forward else 32
        self.mxfp8_pv_row_p_arg = bool(mxfp8_pv_row_p)
        self.tbqk_fold_arg = bool(token_block_qk_fold)
        self.o_regstore_default = bool(o_regstore_default)
        self.is_sm103 = bool(_is_sm103() if is_sm103 is None else is_sm103)
        self.fp8_generic_e2e_default = (FwdKernelSpec.rule_fp8_generic_e2e(self.is_sm103, False)
                                        if fp8_generic_e2e_default is None else int(fp8_generic_e2e_default))
        self.score_scale_unit = bool(score_scale_unit)
        self.vtok_diag_arg = bool(vtok_diag)
        self.p_mix = parse_p_mix(p_mix)
        self.return_pmax = bool(return_pmax)
        self._qk_elem_width = qk_elem_width
        self._pv_elem_width = pv_elem_width
        assert (sf_dtype, sf_vec_size) in {
            (cutlass.Float8E4M3FN, 16),
            (cutlass.Float8E8M0FNU, 32),
        }, f"Unsupported block-scaled configuration: sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}"
        if sf_dtype_pv is not None:
            assert (sf_dtype_pv, sf_vec_size_pv) in {
                (cutlass.Float8E4M3FN, 16),
                (cutlass.Float8E8M0FNU, 32),
            }, f"Unsupported PV block-scaled configuration: {sf_dtype_pv}, {sf_vec_size_pv}"
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        # Single Q stage (as split-KV does) whenever head_dim_v > 128: the only way
        # it fits the 512-column TMEM budget without 2-CTA (S 2x128 + O 1x256 = 512),
        # at the cost of the softmax ping-pong. FwdKernelSpec.q_stage overrides.
        self.q_stage = int(spec.q_stage if spec.q_stage is not None
                           else FwdKernelSpec.rule_q_stage(is_split_kv, self.head_dim_v_padded))
        assert self.q_stage in [1, 2]
        self.q_tiles = self.q_stage  # Q tiles (row blocks) per CTA
        # 2 Q tile per CTA
        self.cta_tiler = (self.q_tiles * m_block_size, n_block_size, self.head_dim_padded)
        # two-cta-forward: the MMA tiler's M covers both CTAs of the pair (each CTA owns m_block_size rows of it)
        self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (self.cta_group_size, 1)
        if self.use_2cta_forward:
            assert m_block_size == 128 and n_block_size == 128, "use_2cta_forward: 128 x 128 tiles"
            assert self.head_dim_padded == 128 and self.head_dim_v_padded == 128, "use_2cta_forward: head_dim 128"
            assert self.q_stage == 2, "use_2cta_forward: two Q stages"
            # (varlen: aligned documents with the batched TMA O epilogue only -- every 128-row Q tile of a pair that
            # starts inside a document lies in its slot; the tiles past seqlen_q are not stored)
            assert not is_split_kv and not pack_gqa and not paged_kv_non_tma and (
                not is_varlen_q or self.varlen_tma_o), \
                "use_2cta_forward: no SplitKV, pack_gqa, paged KV or varlen (aligned documents: varlen_tma_o)"
            assert not (is_persistent and (is_causal or is_local)), \
                "use_2cta_forward: causal / local run one cluster per tile (no persistent LPT grid)"
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q and not self.varlen_tma_o
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 2
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3) # stage 0
        self.softmax1_warp_ids = (4, 5, 6, 7) # stage 1
        # self.correction_warp_ids = (8, 9)
        self.correction_warp_ids = (8, 9, 10, 11)
        # self.mma_warp_id = 10
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15, )
        # Single Q stage (split-KV, or q_stage=1 for head_dim 256): the
        # second softmax warpgroup has no S tile to process. It idles with its
        # registers deallocated instead of running the stage-1 loop against
        # barriers that only exist for q_stage == 2 (same as flash_fwd_sm100.py).
        self.idle_warp_ids = ()
        if self.q_stage == 1:
            self.idle_warp_ids = self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.idle_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q and not self.varlen_tma_o: # fallback
            self.epilogue_warp_ids = (13, 14)

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        # self.tmem_o_offset = self.tmem_s_offset
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_to_p_offset = self.n_block_size // 2

        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # e.g., 64, 192

        # vec buffer for the rescale factor (FwdKernelSpec.max_tmem): tmem_vec_rel / tmem_vec_offset, set in
        # _setup_attributes once the pipeline (generic / split) is known
        assert self.n_block_size == 128

        _tune_key = (self.is_causal, self.head_dim_padded)
        self._tune = _fp4_tuning_config(self.is_sm103).get(_tune_key, {})
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        elif "num_regs_softmax" in self._tune:
            self.num_regs_softmax = self._tune["num_regs_softmax"]
            self.num_regs_correction = self._tune["num_regs_correction"]
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        else:
            self.num_regs_softmax = 192
            self.num_regs_correction = 80
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        self.num_regs_empty = 24
        self.buffer_align_bytes = 1024
        
        # Scale factor parameters for block-scaled quantization (FP4)
        self.sf_dtype = sf_dtype
        self.sf_vec_size = sf_vec_size
        self.sf_dtype_pv = sf_dtype_pv if sf_dtype_pv is not None else sf_dtype
        self._qk_split_pipe_req = qk_split_pipe  # interface's applicability decision (see _setup_attributes)
        self.qk_split_mask_seqlen = bool(qk_split_mask_seqlen)
        self.sf_vec_size_pv = sf_vec_size_pv if sf_vec_size_pv is not None else sf_vec_size
        # Correction-warp O rescale: number of 16-column TMEM fragments kept in
        # flight (loads issued back to back before the multiplies/stores) so the
        # tcgen05.ld latency is paid once per batch instead of once per fragment.
        self.corr_rescale_batch = int(spec.corr_rescale_batch)
        # fp4_group_max_first=True (log2 quant path only): take the per-group
        # maxes on the RAW scores, derive the row max from them (8 values instead
        # of a 128-wide tree) and fold scale, row max and group bias into ONE
        # packed FMA per pair (drops the separate scale_subtract_rowmax pass).
        # The measured B200 defaults apply to the E2M1-P modes
        # only (NVFP4 / MXFP4 PV); the E4M3-P (MXFP8 PV) path keeps the old behaviour.
        _e2m1_pv = (pv_elem_width if pv_elem_width is not None
                    else (4 if (sf_vec_size_pv if sf_vec_size_pv is not None else sf_vec_size) == 16 else 8)) == 4
        # the construction-time answer to "is the PV operand 4-bit": self.v_dtype only exists in __call__
        self._e2m1_pv = bool(_e2m1_pv)
        self.fp4_group_max_first = bool(spec.fp4_group_max_first if spec.fp4_group_max_first is not None
                                        else FwdKernelSpec.rule_group_max_first(_e2m1_pv))
        # exp2_emulation_every=k (log2 quant path): every k-th pair of exps is
        # computed on the FMA pipe (packed degree-3 emulation) instead of MUFU.
        # 0 = all MUFU; 2 = 50 % emulated; 4 = 25 %.
        # The fraction of exp2 pairs emulated on the FMA pipe: one pair in every N (0 = all MUFU).
        # FwdKernelSpec.exp2_emulation_every (KernelTuning.fp4_exp2_emulation_every) overrides the per-shape default.
        _e2e = spec.exp2_emulation_every
        if _e2e is None:
            _e2e = FwdKernelSpec.rule_exp2_emulation_every(_e2m1_pv)
        self.fp4_log2_e2e_freq = int(_e2e)  # 1/3 on the FMA pipe: optimum on the split pipeline (E30)
        # The FMA-pipe exp2 emulation's polynomial (fixed, E2E_POLY_*): 2 = the packed degree-2 minimax
        # (2 FADD2 + 3 FFMA2 + 2 ALU per pair) on the 4-bit-PV routes -- the value the all-4-bit backward reproduces
        # (flash_bwd_sm100_fp4q e2e_poly, pv4_repro) -- and 3 = the fork's ex2_emulation_2 (degree 3, ~12 instr per pair)
        # elsewhere. Always clamped (E2E_CLAMP): without the clamp the emulation wraps to NaN.
        self.fp4_e2e_poly = E2E_POLY_E2M1_PV if _e2m1_pv else E2E_POLY_E4M3_PV
        self.fp4_e2e_clamp = E2E_CLAMP   # see _ex2_emu_pair
        self.fp4_rescale_threshold_default = str(FP4_PV_RESCALE_THRESHOLD) if _e2m1_pv else "0.0"
        # Force MXFP8 QK through the inline-PTX helper for perf A/B. Measured
        # after the tmem_s_offset SFQ collision fix:
        #   MXFP8+BF16 (1,32768,24,128): inline 1718 TF, generic 1670 TF (+2.9%)
        #   MXFP8+BF16 (1,32768,12,128): inline 1643 TF, generic 1650 TF (tie)
        # Mixed results; keep default generic, toggleable for per-shape tuning.
        self.debug_mxfp8_use_inline_ptx = bool(spec.mxfp8_use_inline_ptx)
        self.fp8_pv_use_explicit_pack = bool(spec.fp8_pv_use_explicit_pack)
        # e2e (exp2 emulation) interleaving config -- env vars read here,
        # applied in __call__ where v_dtype/quant_pv are known.
        self._e2e_freq_override = int(spec.e2e_freq) if spec.e2e_freq is not None else None
        self.e2e_freq = self._e2e_freq_override if self._e2e_freq_override is not None else self._tune.get("e2e_freq", 16)
        self.e2e_start_frg = int(spec.e2e_start_frg if spec.e2e_start_frg is not None
                                 else self._tune.get("e2e_start_frg", 0))
        # e2e helps NVFP4 (sf_vec_size=16) but hurts MXFP8 (sf_vec_size=32) for BF16 PV.
        # On SM103 (B300), hardware exp2 is 2x faster -- e2e may not be needed.
        _force_e2e_env = spec.force_e2e
        self._force_e2e_from_env = _force_e2e_env is not None
        if _force_e2e_env is not None:
            self.force_e2e = bool(_force_e2e_env)
        else:
            self.force_e2e = self._tune.get("enable_e2e", False) and self.sf_vec_size == 16
        # softmax_fused=1 (generic step, pure-E4M3 PV, explicit pack): one loop over pairs
        # in FlashInfer's shape -- the scale-subtract FFMA2 of pair p+4 issued between the two exp2
        # of pair p, the E4M3 cvt every 4 elements and the packed row-sum accumulators lagging 4
        # pairs behind -- instead of the four passes (scale-subtract / exp2 / pack / row sum after
        # the P store). Same emulated pairs, same row-sum accumulation order.
        # =2: FlashInfer's issue order proper -- the MUFU pairs first (element order, FFMA2 lookahead,
        # cvt as groups complete), the 20 emulated pairs as a tail, the P store, then the row sum
        # (fadd_reduce, after the store as before): the P store no longer waits for the row-sum FADD2s.
        # defaults (row S4): the fused loop and the TMEM factor are +1.6..+2.8 % / +0.8..+1.5 % on
        # causal and -3..-5 % on non-causal, so they default on for causal only; the mask-free
        # unmasked loop is bitwise and +11..+14 % causal, 0 non-causal, so it defaults on everywhere
        self.softmax_fused_mode = int(spec.softmax_fused if spec.softmax_fused is not None
                                      else FwdKernelSpec.rule_softmax_fused(self.is_causal))
        # mask_loops=1: no exp2 emulation on masked steps (FlashInfer: EXP2_EMULATION_COUNT=0
        # when whether_apply_mask). The loop split itself (first masked step / causal-masked steps /
        # unmasked steps with mask_fn=None / local-left steps) is what softmax_loop already does.
        # =1: the unmasked loop (below the causal diagonal / every non-first step on nc) gets
        # mask_fn=None instead of the step partial's default mask (which on causal re-applies the
        # causal compare/select to every tile); =2: additionally no emulation on the masked steps.
        self.mask_loops = int(spec.mask_loops)
        # `tmem-rescale-factor` (max_tmem=True): the per-step rescale factor goes softmax ->
        # correction through the TMEM vec slot (column 0 of the stage's S bank, tcgen05.st/ld, as
        # upstream FA4 and FlashInfer) instead of the sScale SMEM slot; the softmax_corr full/empty
        # handshake is unchanged. `max-tmem-two-read-styles`: the setting is READ once, here,
        # and only as a value -- "0", "1" or unset (None), anything else refused -- and self.max_tmem is RESOLVED
        # once, in _resolve_max_tmem() (after the split-pipe and key-local decisions its default depends on), in one
        # expression with one default. It used to be assigned here with the generic step's default and then re-assigned under a
        # presence test on the split pipeline, which is the shape that once lost a guard
        # (`chunked-step-tmem-guard-lost`). Nothing reads self.max_tmem before that resolution.
        self._max_tmem_knob = None if spec.max_tmem is None else ("1" if spec.max_tmem else "0")
        if self._max_tmem_knob not in (None, "0", "1"):
            raise ValueError(f"FwdKernelSpec.max_tmem must be False / True (or None), got {self._max_tmem_knob!r}")
        self.softmax_fused = self.softmax_fused_mode > 0
        self.fp8_pv_zero_fill_regs = bool(spec.fp8_pv_zero_fill_regs)
        # Log-domain NVFP4 P quantization: group max on pre-exp scores, the
        # 1/group_max scale folded into the exp2 argument -- removes all
        # per-group divisions and the post-exp scaling pass. Default ON:
        # +8-15% FP4 PV on GB300 (1388 -> 1595 TF at s=32768) at identical
        # measured accuracy vs the FP32 reference.
        self.fp4_pv_log2_quant = bool(spec.fp4_pv_log2_quant)
        # Two-level P scaling for NVFP4 PV (SageAttention3 sec.3.2): place the
        # row max at 6*448 instead of 6, so the per-16 UE4M3 group SF
        # (= group_max/6) spans (0, 448] instead of (0, 1]. Groups whose max is
        # below 2^-9 of the row max no longer flush to SF=0 and sub-normal SFs
        # (below 2^-6 of the row max) keep their mantissa bits. The P codes are
        # unchanged (group max still maps to 6.0) and row_sum is accumulated
        # from the same scaled P in every quant path, so O/row_sum is unchanged.
        self.fp4_pv_two_level = bool(spec.fp4_pv_two_level)
        # NOTE on the comment above: with the default lazy threshold (fp4_pv_rescale_threshold 8) the bias is
        # lowered by the threshold, so a FRESH row max lands at 6*448/2^8 = 10.5 (group SF 1.75), and only a max that
        # is 8 log2 units stale reaches 6*448. Threshold 0 is SageAttention3's released code exactly (constant 2688
        # on the exact running max, softmax_fused.h).
        # `per-tile-row-frame` (fp4_pv_tile_frame=True, off by default): SageAttention3's PAPER form (Alg. 1 line 10,
        # s_P1 = rowmax(P~_ij)/(448*6)): every KV tile scales P against ITS OWN row max, so each tile's largest group
        # SF is 448 however far the tile sits below the running max. The O accumulator (fp32 TMEM) is kept in the frame
        # of the current tile's max: acc_scale = 2^((ref_old - ref) * scale) may exceed 1 (the correction then rescales
        # O up). The reference is floored at running max - fp4_pv_tile_frame_drop log2 units (default 64) so the
        # accumulator cannot overflow; tiles further down carry < 2^-64 of the row's mass. Split NVFP4 step only
        # (softmax_step_split, UE4M3 SFs, not the integer frames of the all-4-bit forward / MXFP4, whose backward
        # rebuilds P^ from one frame per row); requires fp4_pv_rescale_threshold 0 and fp4_pv_two_level.
        self.fp4_pv_tile_frame = bool(spec.fp4_pv_tile_frame)
        self.fp4_pv_tile_frame_drop = float(spec.fp4_pv_tile_frame_drop) if self.fp4_pv_tile_frame else 0.0
        if self.fp4_pv_tile_frame:
            _tf_thr = self._fp4_pv_rescale_threshold()
            if _tf_thr != 0.0 or not self.fp4_pv_two_level:
                raise ValueError("fp4_pv_tile_frame=True needs fp4_pv_rescale_threshold=0 and fp4_pv_two_level=True "
                                 f"(got threshold {_tf_thr}, two-level {self.fp4_pv_two_level})")
            if not (0.0 < self.fp4_pv_tile_frame_drop <= 80.0):
                raise ValueError(f"fp4_pv_tile_frame_drop must be in (0, 80] log2 units, got {self.fp4_pv_tile_frame_drop}")
        # SM103 tcgen05.ld.red: fused TMEM S load + per-x32-tile max -- row max
        # nearly free for all PV modes (and free MXFP8 P group maxes).
        self.use_ldred_rowmax = bool(spec.ldred_rowmax if spec.ldred_rowmax is not None
                                     else FwdKernelSpec.rule_ldred_rowmax(self.is_sm103))
        self.mma_inst_bits_k = 256
        _qk_w = self._qk_elem_width if self._qk_elem_width is not None else (4 if self.sf_vec_size == 16 else 8)
        if _qk_w == 4:
            # NVFP4 (sf 16) / MXFP4 (sf 32, UE8M0): 256-bit operand tile covers 64 logical K elements.
            self.mma_inst_tile_k = self.head_dim_padded // (self.mma_inst_bits_k // 8 * 2)
        elif _qk_w == 8:
            # MXFP8: 256-bit operand tile covers 32 logical K elements.
            self.mma_inst_tile_k = self.head_dim_padded // (self.mma_inst_bits_k // 8)
        else:
            raise ValueError(f"Unsupported QK element width {_qk_w}")
        # The QK scale-factor tiles span whole 128 x 4 SF atoms along K: at d = 64 with 32-element groups (MXFP8 /
        # MXFP4) the operand K holds 2 groups, so the SF tiles (TMA, smem, TMEM) are laid out for K = 4 groups -- the
        # quantiser fills the atom -- and the MMA's two K blocks read groups 0 and 1 (native head_dim 64). Identity elsewhere.
        self.qk_sf_k = max(self.head_dim_padded, 4 * self.sf_vec_size)
        self.mma_tiler_qk_sf = (self.cta_group_size * m_block_size, n_block_size, self.qk_sf_k)  # as mma_tiler_qk (2-CTA: M = 256)
        self.mma_inst_tile_k_sf = self.mma_inst_tile_k * self.qk_sf_k // self.head_dim_padded
        # PV GEMM: K dimension is the n_block (seqlen tile), and the operand
        # element width follows the PV format.
        # K per PV instruction follows the P/V element width: 4-bit E2M1 -> K64
        # for both NVFP4 (sf_vec 16) and MXFP4 (sf_vec 32); 8-bit E4M3 -> K32.
        _pv_w = pv_elem_width if pv_elem_width is not None else (4 if self.sf_vec_size_pv == 16 else 8)
        if _pv_w == 4:
            self.mma_inst_tile_k_pv = self.n_block_size // (self.mma_inst_bits_k // 8 * 2)
        else:
            self.mma_inst_tile_k_pv = self.n_block_size // (self.mma_inst_bits_k // 8)

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """
        self.acc_stage = 1
        self.epi_stage = 2
        # Compute kv_stage from SMEM budget.
        # Blackwell: 228KB per SM, 227KB optin per block.
        # K and V alias when same dtype or when K is smaller (FP4 K in BF16 V).
        smem_budget = 227 * 1024
        align = self.buffer_align_bytes  # 128B struct field alignment
        def align_up(x, a): return (x + a - 1) // a * a
        # Fixed fields (not scaled by kv_stage): mbar, tmem_holding_buf, sScale, sO, sQ, SFQ, SFP
        # mbar_total depends on kv_stage but is small (~40 barriers * 8B = 320B); use upper bound
        smem_mbar = 512  # generous upper bound for mbarrier storage
        smem_tmem = 4  # Int32
        # corr_empty_early=True: the correction releases a stage's acc_scale slot
        # (corr_empty[stage]) right after reading it, not the other stage's slot after
        # this stage's O rescale (the original cross-stage release = WG ping-pong lock).
        # The acc_scale slot is then double-buffered by tile parity (n_block & 1):
        # [acc_scale p0: stage*m][row_sum: 2m+stage*m][acc_scale p1: 4m+stage*m][row_max: 6m+stage*m]
        self.corr_empty_early = bool(self.spec.corr_empty_early)
        # Split-QK cross-tile pipeline (see mma_split / softmax_step_split): the interface
        # requests it where it applies (E2M1 PV, no masking, seqlen_k % n_block == 0);
        # FwdKernelSpec.qk_split_pipe False/True overrides -- through resolve_qk_split_pipe, the one reader of that field,
        # which the driver is to call BEFORE it builds the compile key (`qk-split-pipe-overrides-keyed-flag`).
        self.qk_split_pipe = self.resolve_qk_split_pipe(self._qk_split_pipe_req, self.spec)
        # max_tmem: the rescale factor's TMEM vec slot = one column (two by tile parity under
        # corr_empty_early) of the stage's S bank, one f32 per row, tcgen05.st by the softmax warps /
        # tcgen05.ld by the correction warps. The column must be dead from the softmax's S load until
        # the correction's read one KV step later:
        #   generic pipeline: the whole bank is dead from the S load to the next QK (P^ at +64); 96
        #   split pipeline: columns 0..63 are the next tile's QK_a target from sfqk_load (arrived right
        #     after the b-half load, BEFORE the row max and the slot store), P^ sits at +64, the FP4
        #     route's SFP / SFV at +80..95 and its SFQ_a / SFK_a (s2t-copied before QK_a(t+1)) at
        #     +96..111; 112..127 are written only by QK_b(t+1), which follows PV(t) and therefore the
        #     correction's read. 112 for both split routes (E4M3 has nothing above +95).
        self.tmem_vec_rel = 112 if self.qk_split_pipe else 96
        self.tmem_vec_offset = [o + self.tmem_vec_rel for o in self.tmem_s_offset]
        # split pipe: the b-half load is issued right after the a-half maxes (overlaps the a exps)
        self.qk_split_bprefetch = bool(self.spec.qk_split_bprefetch)
        # split pipe: the P k-block-0 store's wait::st + P_full arrive deferred to the k-block-1
        # store (one wait covers both), removing a serial stall from the middle of the exp region
        # (E38, default on since Direct-P: +1.9 % at 32768; measured 0 on the exp2 step)
        self.qk_split_late_p0 = bool(self.spec.qk_split_late_p0)
        # E46 (default on, +2.4 % at 32768): 8-op FMNMX3 group-max trees, a 4-op row max over
        # the group maxes and one-FFMA SF arguments in the Direct-P split step
        self.split_lean = bool(self.spec.split_lean)
        # Softmax path. The default is FA4's own: exp2 on the MUFU pipe with a tunable fraction
        # emulated on the FMA pipe (fp4_log2_e2e_freq below), plus the correction warpgroup.
        #
        # Direct-P is the alternative -- no 2^x at all: a magic-number FFMA2 plus one LEA per element
        # reproduce Mitchell's 2^x bit pattern, F2FP.RELU rounds it to E2M1, and the row sum is
        # rebuilt from the packed codes. It is materially faster, but it replaces the paper's softmax
        # and its ~1 % value error biases the row sum (measured: LSE max error 0.0407 vs 0.0013 nats
        # for exp2). FwdKernelSpec.direct_p selects it; the 4-bit P^ steps use it by default.
        self.qk_split_directp = bool(self.spec.direct_p)
        # Direct-P classifier constants (DIRECTP_DELTA / DIRECTP_WINDOW, fixed; the all-4-bit backward and
        # pv4_repro use the same values): the Mitchell bias delta = log2(1.25) - 0.25, and the magic window (see
        # directp_window_consts): 124 = scores down to 86 nats below their group max are exact at no cost (the shared
        # 1.5 * 2^23 centring = 64: 44 nats, NaN LSE beyond)
        self.directp_delta = DIRECTP_DELTA
        self.directp_window = DIRECTP_WINDOW
        # The Mitchell classifier is always the clamped one (max.s32 + mad.lo per element: exact for any score range). The
        # window-only classifier (clamp 0), the Direct-P guard that chose it (host or GPU: select pair / runtime branch) and
        # their settings are gone (by design: on trained activations the guard always picked the clamped one, and the
        # forward stays one launch per pass).
        self.directp_clamp = 2
        self.directp_mag, self.directp_lea = directp_window_consts(self.directp_window)
        # row sum source: "mitchell" (FADD2 tree over the Mitchell values); the all-FP4 step below may take "codes"
        # (exact sum of the quantized values via PRMT LUT + DP4A), which option A's reproduce-forward-p forward sets
        self.directp_rowsum = "mitchell"
        # fp4all=True: the all-FP4 training forward whose P^ the query-row FP4 backward (flash_bwd_sm100_fp4q.py) reproduces
        # bit for bit. Row frames are integers F (log2 units; O rescales by exact powers of two, updated when the snapped row max
        # grows past the rescale threshold), masked split steps use the Direct-P classifier too (finite group-max placeholders),
        # O is normalised by an exact reciprocal of the row sum, and the pmax output carries the final frame F (float, log2
        # units) instead of log P_max. LSE = (F + log2(row sum) - offset) ln 2. The row sum Z stays the fast Mitchell sum with the
        # exact (unrounded) group SFs: the backward takes the straight-through gradient of O = sum_k P^_k V_k / Z,
        # dS_k = w_k dP_k - D m_k with w = P^ / Z (codes x E4M3 SFs) and m = Mitchell / Z (sum_k m_k = 1), whose rows sum to 0.
        self.fp4all = bool(self.spec.fp4all)
        # fp4all + per-key V scale (causal-safe V^, set in __call__ when mVBias is given): the PV operand is v~ = V / s_k
        # with every group SF = 1 (no scale spans sequence positions, so no future key reaches a past output), s_k folds into
        # P^ as the per-key bias beta_k = log2(s_k) / scale on S, and the Mitchell row sum weights each value by u_k = 1 / s_k
        # (Z = sum_k P_k, LSE unbiased). Set in __call__ right after this method.
        # fp4all_exp2=True: keep the whole lean Direct-P structure (group maxes only, group SFs from the maxes, vtok's
        # u-weighted row sum, one E2M1 pack per 8) but compute the value with FA4's exp2 path instead of Mitchell's bit
        # trick: P^_i = ex2.approx(s_i*scale + (log2 6 - gm*scale)) on the MUFU pipe, with one pair in every
        # exp2_emulation_every emulated on the FMA pipe (_ex2_emu_pair) so the two pipes can be balanced. The classifier
        # constant is the Direct-P one at DP_K = 1 (no 2^23 frame, no LEA, no window/clamp). The fp4q backward has the
        # matching setting (BwdQSpec.exp2) so P^, Z and m stay bit-consistent across the step.
        self.fp4all_exp2 = bool(self.spec.fp4all_exp2)
        if self.fp4all:
            assert self.qk_split_pipe and self.split_lean and (self.qk_split_directp or self.fp4all_exp2), \
                "fp4all: split lean pipeline with either the exp2 path or Direct-P"
            assert self.return_pmax, "fp4all: construct with return_pmax=True (the frame export)"
            self.directp_rowsum = str(self.spec.fp4all_rowsum)  # "codes": sum of the quantised values (+11 %)
        # MXFP4 PV in the split pipeline: the UE8M0 group SF is exponent-only, so both the per-group constant and the
        # row-max fold must be integers -- cg is floored (codes then peak in (3, 6] instead of exactly 6) and the fold uses
        # the integer frame F = round(row_max * scale + 1/2) that the all-4-bit forward already keeps (k_row = max_offset - F,
        # with an integer max_offset). mxfp4_split=False falls back to the generic (non-split) MXFP4 step.
        self.mx_pv = bool(self.sf_dtype_pv == Float8E8M0FNU and self.sf_vec_size_pv == 32
                          and (self._pv_elem_width or 4) == 4)
        self.mx_pv_split = bool(self.mx_pv and self.qk_split_pipe and self.spec.mxfp4_split)
        # Direct-P for MXFP4 only paid under the window guard (now removed): with the exact clamp its Mitchell classifier
        # costs more than the exp2 it removes (4.91 vs 4.53 ms at s32768 c). "auto" = exp2 (what the exact guard resolved
        # to); 1 forces Direct-P (clamped).
        self.mx_directp = self.spec.mxfp4_directp is True   # None ("auto") = exp2
        self.int_frame = bool(self.fp4all or self.mx_pv_split)
        if self.fp4_pv_tile_frame and self.int_frame:
            raise ValueError("fp4_pv_tile_frame=True is not defined on the integer-frame steps (fp4all, MXFP4 split)")
        self.fp4all_sf_decode = bool(self.spec.fp4all_sf_decode)  # 1: row sum from the E4M3-rounded SFs (+2-3 %)
        # Non-block-scaled E4M3 P^ scaling: P^ = E4M3(P * 2^offset), row max refreshed only past
        # `threshold` (log2 units); offset + threshold <= log2 448. ONE source of truth for the softmax
        # object and the LSE epilogue (row_sum carries 2^offset, LSE must subtract it): reading the knob
        # twice with different defaults shipped an LSE off by exactly 6 ln 2 when the pure-FP8 default
        # moved to (6, 2). NVFP4-QK + FP8-PV hybrid: (0, 8).
        # Pure E4M3 Q/K/V: (4, 4), cuDNN FROST's own window (`deferred-rescale-window-4-4`):
        # O rescales (FMUL2) 4.03 M -> 0.54 M at b4-s4096-dense; paired locked-clock medians of 3 against (6, 2):
        # +2.9 / +4.1 / -0.3 (noise, spread 2.34-2.46 ms) / +1.6 % at b4-s4096 causal / dense, b1-s32768 causal / dense.
        # Numerics: O cosine -8e-6 on N(0,1), -5e-5 at logit scale 3; LSE max error unchanged (<1e-4 nats); the shipped
        # step's dQ / dK / dV cosines vs fp32 unchanged to 6 digits; 4-seed 6000-step LM convergence 0.8493 vs 0.8505.
        # (6, 2) stays one setting pair away: KernelTuning fp8_pv_p_log2_offset=6, fp8_pv_rescale_threshold=2.
        _pure_fp8_pv = self.p_rowframe8 and ((not self.quant_qk) or self.pv_row_p)
        self.fp8_pv_log2_offset = float(self.spec.fp8_pv_log2_offset if self.spec.fp8_pv_log2_offset is not None
                                        else FwdKernelSpec.rule_fp8_pv_log2_offset(_pure_fp8_pv))
        self.fp8_pv_rescale_threshold = float(
            self.spec.fp8_pv_rescale_threshold if self.spec.fp8_pv_rescale_threshold is not None
            else FwdKernelSpec.rule_fp8_pv_rescale_threshold(_pure_fp8_pv)
        )
        # `p-hat-saturation-unasserted`: the comment above states the invariant -- offset + threshold <= log2 448 --
        # and nothing enforced it. \hat{P} = E4M3(P * 2^offset) with the row max refreshed only past `threshold`, so the
        # largest value the pack can see is 2^(offset + threshold), and E4M3's finite maximum is 448 = 2^8.807.
        # Exceed it and the convert saturates: O is wrong with no NaN and no inf to notice. Both terms are swept
        # knobs (`row-max-offset-threshold-dial`), so (8, 2) -- a pair that looks exactly as legal as the measured
        # (6, 2) -- is one sweep away. Asserted on the resolved values, not on the knobs.
        assert self.fp8_pv_log2_offset + self.fp8_pv_rescale_threshold <= math.log2(448.0) + 1e-9, (
            f"fp8_pv_p_log2_offset ({self.fp8_pv_log2_offset}) + fp8_pv_rescale_threshold "
            f"({self.fp8_pv_rescale_threshold}) = {self.fp8_pv_log2_offset + self.fp8_pv_rescale_threshold} > "
            f"log2(448) = {math.log2(448.0):.3f}: P-hat = E4M3(P * 2^offset) saturates and O is silently wrong"
        )
        self.sscale_floats = (8 if self.corr_empty_early else self.q_stage * 2) * self.m_block_size
        # return_pmax: the true row max per (stage, row) after the regular sScale slots
        self.sscale_pmax_off = self.sscale_floats
        if self.return_pmax:
            self.sscale_floats += self.q_stage * self.m_block_size
        # sXchg: the 1 KB SMEM slot of the removed column-split pipeline's partial-max exchange (512 floats there, 4
        # here, read by nothing). It stays in the struct so the SMEM layout and the kv_stage budget below are the
        # measured ones; dropping it frees 1 KB and may change kv_stage (re-measure first).
        self.sxchg_floats = 4
        smem_sScale = align_up(self.sscale_floats * 4, align) + align_up(self.sxchg_floats * 4, align)  # Float32
        smem_q_per_stage = self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_o_per_stage = self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        # o_regstore=True (generic route): the correction warps store the normalised O tile straight
        # from the tmem-load registers to gmem (autovec 16-byte stores, one row per thread as
        # FlashInfer's non-TMA epilogue does); no sO round trip, no corr<->epi barrier, the epilogue
        # warp idles. sO stays allocated (SMEM layout unchanged) so the change is isolated. Not for
        # pack_gqa (rows map to (seq, head) pairs) -- those tiles keep the sO + TMA store.
        # Default on only where the interface's rule says so (`register-direct-o-store`): pure E4M3 Q/K/V, dense, no
        # varlen, seqlen_k >= 8192 -- bitwise (O, LSE) on every gate cell, and +2.1 to +3.1 % over the TMA store at
        # s_k 8192 / 16384 / 32768, but -4 / -5 % at s_k 2048 / 1024, a wash at 4096, and -1.9 / -5.8 % causal.
        self.o_regstore = (bool(self.spec.o_regstore if self.spec.o_regstore is not None else self.o_regstore_default)
                           and not self.overlap_sO_sQ
                           and not self.use_correction_warps_for_epi and not self.pack_gqa
                           and not self.use_2cta_forward)
        smem_sO = align_up(smem_o_per_stage * self.epi_stage, align) if not self.overlap_sO_sQ else 0
        # q_dbuf=True: Q + SFQ double-buffered across work tiles (split pipeline only):
        # the load warp prefetches the next tile's Q while this tile runs
        # Default on for the pure-E4M3 split route: +4.4 % at s4096 non-causal, 0 elsewhere, no
        # numerics change; the FP4 routes measured 0 / -2 %.
        _q_dbuf = bool(self.spec.q_dbuf if self.spec.q_dbuf is not None
                       else FwdKernelSpec.rule_q_dbuf(self.quant_qk, self.quant_pv))
        self.q_bufs = 2 if (_q_dbuf and self.qk_split_pipe
                            and not self.overlap_sO_sQ) else 1
        smem_sQ = align_up(smem_q_per_stage * self.q_tiles * self.q_bufs, align)
        # SFQ/SFP are per q_stage, SF layout cosize: m_block * head_dim / sf_vec_size
        sfq_per_stage = self.m_block_size * self.qk_sf_k // self.sf_vec_size
        sfp_per_stage = self.m_block_size * self.head_dim_v_padded // self.sf_vec_size_pv
        smem_sSFQ = align_up(sfq_per_stage * self.q_tiles * self.q_bufs, align) if self.quant_qk else 0
        smem_sSFP = align_up(sfp_per_stage * self.q_stage, align) if self.quant_pv else 0
        smem_fixed = smem_mbar + smem_tmem + smem_sScale + smem_sO + smem_sQ + smem_sSFQ + smem_sSFP
        # sf_dense: the key tiles' staging ring (SFD_KV_SLOTS slots: K / V exponent bytes, the NVFP4 S route's K SF
        # words) and its mbarriers
        self.sfd_kv_slot = 0
        if self.sf_dense:
            _sf4 = 528 * (self.head_dim_padded // 64) if self.sf_vec_size == 16 else 0
            # (cov5 v4_packed: + V's dense SF rows (130 rows x 8 bytes) and the diagonal payload's tile word)
            _v4 = (V4_SF_BYTES + 16) if getattr(self, "v4_packed", False) else 0
            self.sfd_kv_slot = (288 + _sf4 + _v4 + 127) // 128 * 128
            smem_fixed += SFD_KV_SLOTS * self.sfd_kv_slot + 64
            if getattr(self, "v4_packed", False):
                from flash_attn.cute import v4_packed as V4
                smem_fixed += V4.STAGES * V4.STAGE_BYTES + 128
        # Per-kv_stage fields: sK (or aliased), sV, SFK, SFV
        # two-cta-forward: each CTA holds half of every K tile (n_block / 2 keys) and of every V tile (head_dim_v / 2)
        smem_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8 // self.cta_group_size
        smem_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8 // self.cta_group_size
        if self.v_dtype == self.k_dtype or self.k_dtype.width < self.v_dtype.width:
            smem_kv_per_stage = max(smem_k_per_stage, smem_v_per_stage)
        else:
            smem_kv_per_stage = smem_k_per_stage + smem_v_per_stage
        # SF layout cosize per stage: n_block * head_dim / sf_vec_size (MMA-tiled)
        sfk_per_stage = self.n_block_size * self.qk_sf_k // self.sf_vec_size
        sfv_per_stage = self.n_block_size * self.head_dim_v_padded // self.sf_vec_size_pv
        if self.quant_qk:
            smem_kv_per_stage += sfk_per_stage
        if self.quant_pv:
            smem_kv_per_stage += sfv_per_stage
        # Add per-stage padding for swizzle/layout cosize inflation (~128B per staged field)
        num_kv_staged_fields = 2  # sK + sV (or 1 if aliased, but sV still has cosize overhead)
        if self.quant_qk:
            num_kv_staged_fields += 1  # sSFK
        if self.quant_pv:
            num_kv_staged_fields += 1  # sSFV
        smem_kv_per_stage += num_kv_staged_fields * 128
        self.kv_stage = (smem_budget - smem_fixed) // smem_kv_per_stage
        self._kv_smem_room = (smem_budget, smem_fixed, smem_kv_per_stage)   # (token-local operands: _fit_tok_slot)
        # Pure FP8 PV should not keep increasing KV pipeline depth just because V is smaller.
        # Keep a local env override so we can sweep end-to-end performance without
        # editing the scheduling code for every experiment.
        fp8_pv_kv_stage_cap_env = self.spec.fp8_pv_kv_stage_cap
        fp8_pv_kv_stage_cap = (
            int(fp8_pv_kv_stage_cap_env)
            if fp8_pv_kv_stage_cap_env is not None
            else FwdKernelSpec.rule_fp8_pv_kv_stage_cap(self.head_dim_v_padded)
        )
        if const_expr(self.p_rowframe8 and fp8_pv_kv_stage_cap > 0):
            self.kv_stage = min(self.kv_stage, fp8_pv_kv_stage_cap)
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        # self.uneven_kv_smem = (
            # self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        # )
        self.uneven_kv_smem = False
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

        # Split-pipe rule for `tmem-rescale-factor`. The blanket "off on every split pipeline"
        # (bitwise, but 1.3-1.9 % slower) was measured while `mitchell-direct-p` was accidentally
        # off, i.e. against the exp2 value function (`direct-p-default-inverted`). Re-measured with
        # Direct-P live, free clock, guard pinned to the window-only classifier, NVFP4 / MXFP4 forward, TFLOP/s
        # off -> on:
        #   NVFP4  b4-s4096-causal 1523->1581  dense 1964->2102   b1-s32768-causal 2099->2231  dense 2074->2232
        #   MXFP4  b4-s4096-causal 1679->1661  dense 2285->2312   b1-s32768-causal 2484->2428  dense 2447->2453
        # So it is worth +3.8 to +7.6 % on the NVFP4 split step and -2.3 to +1.2 % (noise) on the
        # MXFP4 one, whose integer frame keeps the correction warps busy differently. Default: on
        # for a 4-bit-PV split step that is NOT integer-framed, off otherwise, which is the
        # measured optimum per cell. FwdKernelSpec.max_tmem still overrides, and it is bitwise either way.
        # The value is resolved by _resolve_max_tmem(), called from __call__ once self.vtok is known.
        if self.use_2cta_forward:
            # two-cta-forward: the generic MMA program and softmax step only
            assert not (self.qk_split_pipe or self.overlap_sO_sQ
                        or self.o_regstore or self.epi_stage != 2
                        or self.q_bufs != 1 or self.use_correction_warps_for_epi), (
                "use_2cta_forward: the generic MMA program / softmax step with the sO + TMA epilogue only")

    def _resolve_stage_skip(self, absent: tuple) -> None:
        """causal_stage_skip (1-CTA) / pair_stage_skip (two-cta-forward) -> self.stage_skip, self.stage_skip_wg_mask.
        The 1-CTA skip runs the pair's programs (mma_pair, the stage-skip correction loop, the softmax loop's n_zero),
        so only where those are complete: the generic MMA program, softmax step and correction loop."""
        generic = not any((self.qk_split_pipe,
                           self.use_correction_warps_for_epi, self.fp4_pv_tile_frame, self.vtok,
                           self.vtok_diag, self.tb_fwd, self.repro))
        one_cta = all((self.causal_stage_skip, self.is_causal, not self.is_local, self.corr_empty_early,
                       self.q_stage == 2, self.q_bufs == 1, not self.is_split_kv, self.mask_mod is None,
                       not self.pack_gqa,
                       all(t is None for t in absent), generic))
        self.stage_skip = bool(self.pair_stage_skip or one_cta)
        # 1-CTA: the softmax steps' masked / unmasked boundary is the stage's own 128-row group's too (stage 1's second
        # KV tile from the diagonal lies wholly below its diagonal: an unmasked step), as on the pair
        self.stage_skip_wg_mask = bool(self.stage_skip and not self.use_2cta_forward)

    def _resolve_max_tmem(self) -> None:
        """THE resolution of self.max_tmem (`max-tmem-two-read-styles`), called once from __call__ after
        _setup_attributes (which read the setting) and after self.vtok is set; max_tmem is read only by traced kernel
        methods, so this is early enough. The spec's value if it was given, else ONE default.

        Default on the split step: the measured rule in _setup_attributes (4-bit PV, not integer-framed, Direct-P),
        except on the all-FP4 route (`fp4-gap-outside-the-compiled-kernel`), where the TMEM factor makes
        the softmax loop spill once per KV tile and turning it off is bitwise and faster -- dense key-local -4.8 /
        -5.8 %, causal plain all-FP4 -0.8 / -1.6 % (b4-s4096 / b1-s32768) -- UNLESS the step is causal and
        key-local (full or diagonal-only), where turning it off measured SLOWER (full fix forward +3.8 / +4.5 %,
        diagonal-only 0.0 / +0.5 %). On the generic step: is_causal. Every input is a keyed specialisation and the
        explicit setting is in the compile key through spec.key().
        """
        if self._max_tmem_knob is not None:
            _max_tmem = self._max_tmem_knob == "1"
        elif self.qk_split_pipe:
            _max_tmem = bool(self.quant_pv and not self.mx_pv_split and self.qk_split_directp
                             and (not self.fp4all or (self.vtok and self.is_causal)))
        else:
            _max_tmem = bool(self.is_causal)
        self.max_tmem = bool(_max_tmem)

    @staticmethod
    def resolve_qk_split_pipe(requested: bool, spec: FwdKernelSpec) -> bool:
        """The split-QK pipeline decision: the driver's applicability `requested`, overridden by spec.qk_split_pipe
        (KernelTuning.qk_split_pipe).

        `qk-split-pipe-overrides-keyed-flag`: the override used to be applied inside
        _setup_attributes, AFTER the driver had keyed its own `_qk_split_ok`, so the key item and the pipeline that
        ran could disagree (the spec in the key kept the two from sharing a slot, but the item lied). This is
        now the ONE reader of the setting, a pure function of the request and the spec, so the driver can key
        exactly the value the constructor will compute: `_qk_split_ok = FlashAttentionForwardSm100.
        resolve_qk_split_pipe(_qk_split_ok)` before the key, after which the constructor's call is idempotent
        (`qk-split-resolved-before-key`). One name, one default (unset = the request), values False / True only.
        """
        if spec.qk_split_pipe is None:
            return bool(requested)
        return bool(spec.qk_split_pipe)

    def _fp4_pv_rescale_threshold(self) -> float:
        """The E2M1 P.V lazy rescale threshold (log2 units, Python float): spec.fp4_pv_rescale_threshold, else
        the route default fp4_rescale_threshold_default. The one source for the
        softmax step, the LSE offset and the reproduce-forward-p offset."""
        if self.spec.fp4_pv_rescale_threshold is not None:
            return float(self.spec.fp4_pv_rescale_threshold)
        return float(self.fp4_rescale_threshold_default)

    def _qk_sf_shape(self, shape):
        """The QK operand shape with K widened to the SF tiles' K (qk_sf_k): the gmem SF view covers whole atoms."""
        if self.qk_sf_k == self.head_dim_padded:
            return shape
        return (shape[0], self.qk_sf_k) + tuple(shape[2:])

    @cute.jit
    def __call__(
        self,
        mQ,  # cute.Tensor or cute.Pointer (b, s_q, h, d)
        mK,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, d)
        mV,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        # Per-(batch, kv-head) V dequant scale for externally-quantized FP8 V
        # (FA3 semantics); folded into the output normalization.
        v_descale: Optional[cute.Tensor] = None,
        # optional log P_max output (same shape / dtype as mLSE); needs return_pmax=True at construction
        mPmax: Optional[cute.Tensor] = None,
        # For pointer-based Q/K: separate shapes to handle cross-attention (seqlen_q != seqlen_k)
        q_ptr_shape: tuple = (),
        k_ptr_shape: tuple = (),
        # For pointer-based V (FP4 K-major): full headdim shape (b, s, h, d)
        v_ptr_shape: tuple = (),
        # fp4all causal-safe V: per-key bias beta (b, h_k, s_k) float32 (log2 s_k / scale_log2) and weight u = 1 / s_k
        mVBias: Optional[cute.Tensor] = None,
        mVU: Optional[cute.Tensor] = None,
        # vtok_w: per key tile 256 f32 [A_k = e_k + 8, u_k = 2^-e_k] (b, h_k, s_k/128 x 256), staged into SMEM by the MMA warp
        mVTokW: Optional[cute.Tensor] = None,
        # vtok_tsf: the same 256 floats per key tile as bytes in an SFV-shaped uint8 tensor, staged into SMEM by the load
        # warp's async TMA copy next to SFV (no instructions on the load / MMA / softmax chains)
        mVTokSF: Optional[cute.Tensor] = None,
        # GPU-only score scale (device_scales): float32 [softmax_scale] in device memory, e.g. softmax_scale * s_q * s_k of
        # per-tensor FP8 Q / K computed on the GPU. When given, the kernel reads it instead of `softmax_scale` (ignored).
        mSoftmaxScale: Optional[cute.Tensor] = None,
        # token-block operands (token-block scales): (4, b, h, s / 32) float32 = 2^(e - 127) per 32-token block of
        # Q, K, V (and dO, unused here) from quantize_fp8_blocks(block_columns=128). The factors are folded into the scores (tb_fwd):
        # S' = f_Q f_K S^ + log2(f_V) / scale_log2 per 32-key group, so the QK / PV GEMMs stay plain E4M3 / NVFP4,
        # P' = P f_V goes through the unchanged max / exp2 / pack (P' <= 2^(max_offset + threshold): the online max runs
        # over the biased scores), and the row sum weights each group by 1 / f_V.
        mTbFac: Optional[cute.Tensor] = None,
        # `reproduce-forward-p` (option A, p_consistency="reproduce_forward_p"): the per-query export the backward needs to
        # rebuild this forward's P^ bit for bit, (b, h, s_q / 128, REPRO_WORDS, 128) int32 per query row: [0] the final
        # integer frame F, [1] 1 / Z (float bits; Z the row sum in the final frame), [2] the number of frame changes, [3, 3 + R) the
        # changes in processing order, (n_block << 16) | int16(frame before the change). Written by the softmax threads.
        mRepro: Optional[cute.Tensor] = None,
        # debug build only: (b, h, s_q, s_k + s_k / 16) uint8 -- the E2M1 code of every (query, key) and the E4M3 SF byte of
        # every (query, 16-key group) exactly as this forward produced them (bitwise reproduction gate)
        mReproDbg: Optional[cute.Tensor] = None,
        # token-local operands (operand_scaling="token_local"; tb_fwd only): the (6, b, h, s) fp32 token table,
        # rows f_Q, f_K, f_V, f_dO (f = 2^(e_t - 127), each token's OWN exponent of its token-local E4M3 codes),
        # log2 f_V, 1 / f_V; K / V rows on the kv heads. V: P' = P f_V(k) per key (the per-32-key
        # group factor's role), the row sum weighted by 1 / f_V(k). E4M3 Q / K (tb_qk): S' = f_Q(row) f_K(k) S. No
        # token's scale depends on any other token. None = today's group factors (bitwise).
        mTbTokV: Optional[cute.Tensor] = None,
        # aligned varlen: the precomputed work list (lp_varlen.varlen_worklist; tile_scheduler.VarlenWorkListScheduler):
        # the persistent longest-first scheduler instead of SingleTileVarlenScheduler. Needs cu_seqlens_q.
        mVarlenWork: Optional[cute.Tensor] = None,
        # o_rows_packed (lp_docmaps): the packed start of each document's O rows, int32 (b + 1,)
        mCuSeqlensO: Optional[cute.Tensor] = None,
        # codes_packed (lp_docmaps): the packed start of each document's K / V rows, int32 (b + 1,)
        mCuSeqlensKc: Optional[cute.Tensor] = None,
        # codes_packed: the channel-major V's document starts (multiples of 16 keys), int32 (b + 1,)
        mCuSeqlensVt: Optional[cute.Tensor] = None,
        # sf_dense: [EvQ, EvK, EvV (b h, R) uint8 (+ SF4Q, SF4K (b h, R d / 64) int32 on the NVFP4 S route)]
        mSfdF: Optional[list] = None,
        compute_sp1: cutlass.Constexpr[bool] = False,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        For FP4, mQ/mK can be cute.Pointer with q/k_ptr_shape providing (b, s, h, d).
        The kernel builds tensors from the pointer using make_ordered_layout.
        """
        # Build Q/K tensors from pointer/tensor + shape
        # For pointers: mQ is a Pointer, .iterator not needed
        # For tensors: mQ is a Tensor, use .iterator to extract pointer
        q_iter = mQ.iterator if hasattr(mQ, 'iterator') else mQ
        k_iter = mK.iterator if hasattr(mK, 'iterator') else mK
        mQ = cute.make_tensor(q_iter, cute.make_ordered_layout(
            q_ptr_shape, order=tuple(range(len(q_ptr_shape) - 1, -1, -1))
        ))
        mK = cute.make_tensor(k_iter, cute.make_ordered_layout(
            k_ptr_shape, order=tuple(range(len(k_ptr_shape) - 1, -1, -1))
        ))
        # FP4 K-major V: build from pointer with explicit (b, s, h, d) shape and
        # K-major strides (S*H*D, 1, S, S*H). The host transposes V's underlying
        # buffer so that seqlen has stride 1 in the FP4 byte buffer.
        if const_expr(len(v_ptr_shape) > 0):
            v_iter = mV.iterator if hasattr(mV, 'iterator') else mV
            # K-major V: nvfp4_quantize on `v.permute(0,2,3,1).reshape(b*h*d, s)`
            # produces an FP4 byte buffer of physical shape (b, h, d, s/2) row-major,
            # where each int8 byte holds two seqlen-adjacent FP4 in the high/low
            # nibble. Logically the V tensor has FP4 shape (b, s, h, d) with element
            # strides (h*d*s, 1, h*d, d) -- order=(3, 0, 2, 1):
            #   s (dim 1) order 0 -> stride 1
            #   d (dim 3) order 1 -> stride s
            #   h (dim 2) order 2 -> stride s*d
            #   b (dim 0) order 3 -> stride s*d*h
            # Int64 shape so make_ordered_layout produces Int64 strides
            # (tma_partition for SFV requires Int64).
            from cutlass import Int64
            if const_expr(len(v_ptr_shape) == 3):
                # aligned varlen documents: the packed keys (total_k, h, d) of the b = 1 buffer (1, h, d, total_k / 2)
                v_s, v_h, v_d = v_ptr_shape
                mV = cute.make_tensor(v_iter, cute.make_ordered_layout(
                    (Int64(v_s), Int64(v_h), Int64(v_d)),
                    order=(0, 2, 1),
                ))
            else:
                v_b, v_s, v_h, v_d = v_ptr_shape
                mV = cute.make_tensor(v_iter, cute.make_ordered_layout(
                    (Int64(v_b), Int64(v_s), Int64(v_h), Int64(v_d)),
                    order=(3, 0, 2, 1),
                ))
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.compute_sp1 = const_expr(compute_sp1)
        if const_expr(mSFQ is None):
            assert self.q_dtype.width >= 8
            assert const_expr(mSFK is None), "Must provide both QK sfs or None"
        self.quant_qk = const_expr(mSFQ is not None)
        self.quant_pv = const_expr(mSFV is not None)
        # mxfp8-pv-row-frame-p: P^ is row-framed E4M3 (no in-kernel P SF); p_sf_inkernel = the log2 group-quant P^ path
        self.pv_row_p = const_expr(self.mxfp8_pv_row_p_arg and self.quant_pv and self.v_dtype == Float8E4M3FN
                                   and self.sf_vec_size_pv == 32)
        if const_expr(self.mxfp8_pv_row_p_arg):
            assert self.pv_row_p, "mxfp8_pv_row_p: needs MXFP8 P.V (E4M3 V with UE8M0 SFV per 32 keys)"
        self.p_sf_inkernel = const_expr(self.quant_pv and not self.pv_row_p)
        # the P^ of the per-tensor 8-bit step: E4M3, row-framed (no block scale computed for it)
        self.p_rowframe8 = const_expr((not self.p_sf_inkernel) and self.v_dtype.width == 8)
        if const_expr(mSFV is not None and self.v_dtype == Float8E4M3FN):
            assert self.fp4_pv_log2_quant, (
                "MXFP8 PV requires the log-domain quant path"
            )
        # per-tensor E4M3 Q/K with block-scaled P^/V (MXFP8 PV) is not wired (token_block_qk_fold aside)
        if const_expr(not self.tbqk_fold_arg):
            assert not (not self.quant_qk and self.quant_pv)

        # Assume all strides are divisible by 128 bits except the last stride
        def _assume_strides(t):
            divby = 128 // t.element_type.width
            return tuple(
                s if isinstance(s, int) else cute.assume(s, divby=divby)
                for s in t.stride[:-1]
            ) + (t.stride[-1],)
        mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=_assume_strides(t)))
            for t in (mV, mO)
        ]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)) # (s_q, d, h, b)
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        assert const_expr((mPmax is not None) == self.return_pmax), "mPmax must be passed iff built with return_pmax"
        if const_expr(mPmax is not None):
            assert const_expr(mLSE is not None and not self.is_split_kv and not self.pack_gqa)
            mPmax = cute.make_tensor(mPmax.iterator, cute.select(mPmax.layout, mode=LSE_layout_transpose))
        # (s, d, h, b) -> (d, s, h, b)
        # For FP4 block-scaled MMA, B (V) must be K-major (K=seqlen contiguous).
        # Skip V transpose when V is already K-major (headdim contiguous = N contiguous after transpose).
        # Without transpose: mV = (s, d, h, b) -> mode 0=s(K), mode 1=d(N) with d contiguous -> K-major: NO!
        # With transpose:    mV = (d, s, h, b) -> mode 0=d(N), mode 1=s(K) with d contiguous -> MN-major
        # For FP4 we need K-major: mode 1 (K=seqlen) contiguous. Need V physically transposed on host.
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        # FP4 block-scaled MMA requires K-major B; standard MMA uses MN-major
        if const_expr(mSFV is not None):
            # (sfd: an E4M3 V -- kind::mxf8f6f4 -- may be MN-major, the caller's row codes (lp_docmaps packed rows))
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K
                          and not (self.codes_packed and mV.element_type is cutlass.Float8E4M3FN)):
                raise RuntimeError("The layout of mV must be K-major for FP4 block-scaled MMA")
        else:
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
                raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype == cutlass.Int8):
            assert self.q_dtype == self.k_dtype
            self.q_dtype = self.k_dtype = cutlass.Float4E2M1FN
            if const_expr(mSFV is not None):
                self.v_dtype = cutlass.Float4E2M1FN

        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        # sf_dense: the S operands are E4M3 token blocks (exponent tables) or NVFP4 (per-row SF words)
        self.fp8_sfd = self.q_dtype is not cutlass.Float4E2M1FN
        if const_expr(
            mSFV is not None
            and self.q_dtype != self.v_dtype
            and not (self.v_dtype == Float8E4M3FN and self.sf_vec_size_pv == 32)
        ):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype} (V quantization requires matching dtype or MXFP8 PV)")
        # cov5 (v4_packed): the sf_dense forward with an NVFP4 V -- the transposed codes from the key-parity copies,
        # realigned by the load warp (flash_attn.cute.v4_packed); mV only shapes the views
        self.v4_packed = const_expr(self.sf_dense and mSFV is not None and self.v_dtype == cutlass.Float4E2M1FN)
        self._setup_attributes()
        self.vtok_tma = const_expr(mVTokSF is not None)
        self.vtok_w = const_expr(mVTokW is not None or mVTokSF is not None)
        self.vtok = const_expr(mVBias is not None or mVTokW is not None or mVTokSF is not None)
        # token-block-qk-fold: the factors' QK half only (the V half stays in the block-scaled P.V GEMM)
        self.tbqk_fold = const_expr(mTbFac is not None and self.tbqk_fold_arg)
        if const_expr(self.tbqk_fold_arg):
            assert self.tbqk_fold and not self.quant_qk and self.pv_row_p and self.q_dtype == Float8E4M3FN, \
                "token_block_qk_fold: plain E4M3 Q / K codes + mTbFac, with mxfp8_pv_row_p"
            assert not self.qk_split_pipe and not self.pack_gqa and self.score_mod is None and self.n_block_size % 32 == 0
            # (the max runs per 32-key group on the raw S: ld.red's x32 tile maxes on unmasked tiles, _tbqk_row_max)
        self.tb_fwd = const_expr(mTbFac is not None and not self.tbqk_fold)
        # documents (aligned starts, in place in a b = 1 buffer): the fold forward's factor / token tables are the
        # buffer's, (rows, 1, h, s): read at batch 0 and the document's offset + the local token index
        self.tb_varlen = const_expr(self.tb_fwd and mCuSeqlensQ is not None)
        self.tb_tok_v = const_expr(mTbTokV is not None)
        if const_expr(self.tb_tok_v):
            assert self.tb_fwd and self.n_block_size % 32 == 0, "token-local V: the tb_fwd fold forward"
        self.tb_qk = const_expr(mTbFac is not None and not self.quant_qk)
        self.tb_tok_qk = const_expr(mTbTokV is not None and self.tb_qk)
        if const_expr(self.tb_tok_v and self.head_dim_padded < 128):
            # token-local operands: the softmax warps' per-key slot (8 warps x n_block x 2 / 3 rows of fp32) is not in
            # _setup_attributes' fixed budget (it is known only here); without it the head_dim-64 8-bit fold forward
            # took 21 KV stages and overflowed SMEM (234 KB > 227 KB). Fewer KV stages where the slot does not fit
            # (head_dim 128 kernels keep their measured stage count: they fit, the stage cap / wider tiles bound them).
            _b, _f, _per = self._kv_smem_room
            _tok = 8 * self.n_block_size * (3 if self.tb_tok_qk else 2) * 4 + self.buffer_align_bytes
            self.kv_stage = min(self.kv_stage, (_b - _f - _tok) // _per)
        if const_expr(self.tb_fwd):
            assert not self.quant_pv and self.v_dtype.width == 8, "tb_fwd: plain 8-bit V (E4M3 PV)"
            assert not self.qk_split_pipe and not self.vtok and not self.pack_gqa and self.score_mod is None, \
                "tb_fwd: the generic softmax step, no vtok / pack_gqa / score_mod"
            assert self.n_block_size % 32 == 0
            # the row max is the max of the transformed scores, from the raw per-group maxes (ld.red's x32 tile maxes
            # on unmasked tiles, _tb_row_max); the group-weighted row sum replaces the fused / fadd_reduce row sums
            if const_expr(self.tb_qk):
                # with Q.K factors (a_g S + c_g): no fused load -- O with it was not bitwise the plain load's build
                # (B300 fwdreg/chk: one digest whichever max was used, another with the plain load)
                self.use_ldred_rowmax = False
            self.softmax_fused = False
            self.softmax_fused_mode = 0
        self._resolve_max_tmem()
        if const_expr(mVTokW is not None or mVTokSF is not None):
            assert self.fp4all and self.directp_rowsum == "mitchell" and self.qk_split_pipe
        # `reproduce-forward-p`: the export (mRepro) is the fp4all integer-frame split step with the codes row sum
        # (Z = sum of the dequantised P^ x u, so sum_k P^_k u_k / Z = 1 and D = dO . O is the backward's exact D) and the
        # E4M3-decoded group SFs in that sum
        self.repro = const_expr(mRepro is not None)
        self.repro_dbg = const_expr(mReproDbg is not None)
        if const_expr(self.repro):
            assert self.fp4all and self.qk_split_pipe and self.int_frame and self.directp_rowsum == "codes" \
                and self.fp4all_sf_decode, "repro: fp4all split step, fp4all_rowsum='codes', fp4all_sf_decode=True"
            assert not self.mx_pv_split, "repro: the NVFP4 P^ with the lazy integer frame (no MXFP4 P^)"
        if const_expr(self.p_mix is not None):
            assert self.fp4all and self.split_lean and self.qk_split_directp and not self.fp4all_exp2, \
                "p_mix: the all-FP4 split step (fp4all) on the Mitchell configuration (fp4all_exp2=False)"
        # the debug dump (codes, SF bytes, group maxes) also serves the all-FP4 route's backward gate (no mRepro there)
        assert not self.repro_dbg or self.fp4all
        if const_expr(mVBias is not None):
            assert self.fp4all and mVU is not None and (self.directp_rowsum == "mitchell" or self.repro), \
                "vtok: fp4all, Mitchell row sum (the codes row sum with the reproduce-forward-p export)"
        # diagonal-only-key-local-v-scale: the key-local V scale on the causal masked steps only, which read the SECOND
        # half of a doubled-along-keys V / SFV payload. A V scale group spans sf_vec_size_pv consecutive keys starting at
        # key 0, so with n_block_size % sf_vec_size_pv == 0 no group straddles a key-tile boundary and the masked steps
        # are exactly the tiles a group can straddle the causal boundary in. Without the per-key tensors this is the
        # `tile_const` payload (a per-tile constant V scale, which cannot represent V): only the V tile index changes.
        self.vtok_diag = const_expr(self.vtok_diag_arg)
        # vtok_bulk: the per-key bias gathers of a key-local step as 64-float vector loads instead of 64 scalar global
        # loads, on the diagonal-only route (`diagonal-forward-per-key-gathers`): its only per-key work is on the masked
        # steps, whose scalar gathers are nearly all of its forward gap to leaky (loads ablated: +4.0 / +0.5 %). Forward
        # 0.2321 -> 0.2238 ms at b4-s4096-causal, 2.3740 -> 2.2287 ms at b1-s32768-causal; O / LSE / frame bitwise.
        # The row-sum weight gathers stay scalar: bulk there buys nothing more and is not bitwise (the row sum's fused
        # multiply-adds change, LSE by <= 5.7e-6). Every other route keeps scalar gathers.
        self.vtok_bulk = const_expr(bool(self.vtok_diag))
        if const_expr(self.vtok_diag):
            assert not self.vtok_w and not self.vtok_tma, (
                "vtok_diag: the mVBias / mVU key-local form, or no per-key tensors at all (tile_const)"
            )
            assert self.is_causal and not self.is_local, "vtok_diag: causal masking only"
            assert ((mCuSeqlensK is None and mSeqUsedK is None) or self.zero_pad_rows) and mPageTable is None, (
                "vtok_diag: no paged KV, varlen only as aligned documents (the second payload is indexed by whole key "
                "tiles)"
            )
            assert self.n_block_size % self.sf_vec_size_pv == 0, (
                "vtok_diag: the V scale group must divide the key tile, else a group straddles a tile boundary"
            )
            assert not self.is_split_kv, "vtok_diag: no SplitKV"
        self.use_tma_O = self.arch >= 90 and ((mCuSeqlensQ is None and mSeqUsedQ is None) or self.varlen_tma_o)
        # cov2 (o_rows_tma): the 2-CTA forward on packed rows keeps the sO epilogue warp -- a 128-row tile wholly inside
        # its document by the TMA store at its packed start, a document's partial last tile by the epilogue warp's
        # predicated row copies of its real rows (a TMA tile would overwrite the next document's rows)
        if const_expr(self.o_rows_packed):   # (lp_docmaps packed O: predicated row copies, never a TMA tile)
            assert (not self.use_2cta_forward or (self.sf_dense and self.varlen_tma_o)) and not self.pack_gqa \
                and not self.is_split_kv and not self.o_regstore and mCuSeqlensO is not None, \
                "o_rows_packed: the 1-CTA forward (or the sf_dense pair) without pack_gqa / SplitKV / o_regstore"
        if const_expr(self.codes_packed):
            # (cov3: the fold forward on documents too -- its tables per PACKED row, read at the packed starts)
            assert mCuSeqlensKc is not None and mCuSeqlensVt is not None and mPageTable is None \
                and (not self.vtok or (self.v4_packed and mVBias is not None and not self.vtok_w)) \
                and (not self.tb_fwd or self.tb_varlen) \
                and (not self.use_2cta_forward or (self.sf_dense and not self.v4_packed)), \
                "codes_packed: the 1-CTA forward, no paged KV; vtok only the v4_packed mVBias form; fold on documents"
            if const_expr(not self.o_rows_tile):
                self.use_tma_O = False
        if const_expr(self.o_regstore):
            # E41 / o_regstore: the correction warps store O from registers; the kernel needs
            # the gmem tensor, not the TMA coordinate tensor, and no epilogue TMA store
            self.use_tma_O = False
        # Quantized PV (NVFP4 / MXFP4 / MXFP8): the log2-domain group quant is
        # only active with exp2 emulation off, and on B200 its instruction
        # elimination is worth more than the MUFU offload the SM100 tuning
        # table's enable_e2e buys for BF16 PV: NVFP4+NVFP4 1009 -> 1152 TF at
        # (1,4096,24,128) and 1203 -> 1388 TF at s=16384, identical output;
        # MXFP4 PV 983 -> 1163 TF. MXFP8 PV needs the log2 path regardless.
        # An explicit force_e2e still wins.
        if const_expr(self.p_sf_inkernel and not self._force_e2e_from_env):
            self.force_e2e = False
        # Apply FP8 PV tuning overrides when v_dtype is FP8
        if const_expr(self.p_rowframe8):
            _fp8_tune = _FP4_FP8PV_TUNING_CONFIG.get(
                (self.is_causal, self.head_dim_padded), {}
            )
            if const_expr(self._e2e_freq_override is None and "e2e_freq" in _fp8_tune):
                _fp8_default = _fp8_tune["e2e_freq"] if self.sf_vec_size == 16 else 10
                self.e2e_freq = _fp8_default
                self.e2e_start_frg = _fp8_tune.get("e2e_start_frg", self.e2e_start_frg)
            pass  # force_e2e already set in __init__ (respects force_e2e env)
            if const_expr((not self.quant_qk or self.pv_row_p) and not self.qk_split_pipe):
                # pure E4M3 Q/K/V on the generic step (the FP8 route's non-causal forward): FA4's
                # FMA-pipe exp2 emulation every 6th pair is the MUFU/FMA balance point here --
                # +6 / +9 % at s4096 / s32768 non-causal, parity causal;
                # an explicit force_e2e / e2e_freq still wins
                if const_expr(not self._force_e2e_from_env):
                    self.force_e2e = True
                if const_expr(self._e2e_freq_override is None):
                    # default from the interface (`emulation-share-short-dense`): 8 (16 emulated pairs per 128)
                    # on dense 2048 <= s_k < 8192, else 6 (20 pairs)
                    self.e2e_freq = int(self.spec.fp8_generic_e2e if self.spec.fp8_generic_e2e is not None
                                        else self.fp8_generic_e2e_default)
                    # the configuration that was measured (env override) keeps the SM100 table's
                    # start fragment; the FP8-PV tune above had just replaced it
                    self.e2e_start_frg = int(self.spec.e2e_start_frg if self.spec.e2e_start_frg is not None
                                             else self._tune.get("e2e_start_frg", 0))
        elif const_expr(self._e2e_freq_override is None):
            if const_expr(
                self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa
            ):
                self.e2e_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else self._tune.get("e2e_freq", 10)

        use_2cta_instrs = self.mma_tiler_qk[0] == 256
        assert use_2cta_instrs == self.use_2cta_forward
        # two-cta-forward, causal, plain-E4M3 S GEMM: Q stage 0 skips the pair's leading tiles above its diagonal in the
        # MMA / softmax / correction programs (mma_pair); with SFQ / SFK copies (4-bit QK) the cross-stage S-read
        # handshake keeps both stages in lockstep, so those run the bookkept zero steps instead
        self.pair_stage_skip = bool(self.use_2cta_forward and self.is_causal and self.corr_empty_early)
        # causal_stage_skip: the same program on the 1-CTA kernel (Q stage = a 128-row group); resolved in plain Python
        # (a long boolean expression traced by the AST preprocessor took minutes to compile)
        # (aligned documents with the TMA epilogue: the stage skip reads only the per-tile seqlen info)
        self._resolve_stage_skip((blocksparse_tensors,) if const_expr(self.varlen_tma_o)
                                 else (mCuSeqlensQ, mSeqUsedQ, blocksparse_tensors))
        if const_expr(self.use_2cta_forward):
            assert self.varlen_tma_o or (mCuSeqlensQ is None and mCuSeqlensK is None and mSeqUsedQ is None
                                         and mSeqUsedK is None), \
                "use_2cta_forward: no varlen / seqused (aligned documents: varlen_tma_o)"
            assert mPageTable is None and blocksparse_tensors is None and aux_tensors is None, \
                "use_2cta_forward: no paged KV, block sparsity or aux tensors"
            assert self.score_mod is None and self.mask_mod is None and learnable_sink is None, \
                "use_2cta_forward: no score_mod / mask_mod / learnable sink"
            assert not self.vtok and not self.vtok_diag and not self.tb_fwd, \
                "use_2cta_forward: no key-local V or tb_fwd"
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        
        # Use block-scaled MMA for PV only if V is being quantized (mSFV is provided)
        if const_expr(self.quant_qk):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.qk_acc_dtype,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )

        if const_expr(self.quant_pv):
            tiled_mma_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype_pv,
                self.sf_vec_size_pv,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
        else:
            tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.pv_acc_dtype,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # per-CTA epilogue tile (two-cta-forward: each CTA stores its own m_block_size rows)
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)
        
        # ((Atom_Inst_M, Atom_Inst_K), MMA_M, MMA_K, STAGE)
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_tiles * self.q_bufs,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )
        
        sfv_smem_layout_staged = None
        sfp_smem_layout_staged = None
        # # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
        # Always thread our computed mma_inst_tile_k through. Default is 4 in the
        # helper, which only happens to match MXFP8 d=128. MXFP8 d=64 needs 2 and
        # was failing to compile via TMA SFQ atom shape mismatch.
        sf_layout_kwargs = {"mma_tile_inst_k": self.mma_inst_tile_k_sf}
        sfq_smem_layout_staged = make_smem_layout_sfa(
            tiled_mma_qk,
            self.mma_tiler_qk_sf,
            self.sf_vec_size,
            self.q_stage * self.q_bufs,
            **sf_layout_kwargs,
        )
        sfk_smem_layout_staged = make_smem_layout_sfb(
            tiled_mma_qk,
            self.mma_tiler_qk_sf,
            self.sf_vec_size,
            self.kv_stage,
            **sf_layout_kwargs,
        )
        # Create P scale factor layout for P*V operation (P is the A matrix)
        if const_expr(self.quant_pv):
            sf_layout_kwargs_pv = {"mma_tile_inst_k": self.mma_inst_tile_k_pv}
            sfp_smem_layout_staged = make_smem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size_pv,
                self.q_stage,
                **sf_layout_kwargs_pv,
            )

            sfv_smem_layout_staged = make_smem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size_pv,
                self.kv_stage,
                **sf_layout_kwargs_pv,
            )
        
        if const_expr(self.v4_packed):
            assert self.v_dtype == self.k_dtype and self.same_hdim_kv_padded
            self._v4_stage_bytes = const_expr(max(sV_layout.outer.stride[-1], 0)) * self.v_dtype.width // 8
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )


        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]), # (qhead_per_kvhead, sq)
                mQ.shape[1], # d
                mK.shape[2], # h_k
                *mQ.shape[3:], # b
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # Add scale factor copy bytes to Q/K/V since they use the same barrier
        if const_expr(self.quant_qk):
            self.tma_copy_bytes["Q"] += cute.size_in_bytes(mSFQ.element_type, cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]))
            self.tma_copy_bytes["K"] += cute.size_in_bytes(mSFK.element_type, cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]))
        if const_expr(self.quant_pv):
            self.tma_copy_bytes["V"] += cute.size_in_bytes(mSFV.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))
        if const_expr(self.vtok_tma):
            self.tma_copy_bytes["V"] += cute.size_in_bytes(mVTokSF.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))
            self.vtok_sf_slot_bytes = cute.size_in_bytes(mVTokSF.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))
        if const_expr(self.sf_dense):
            # the SF atoms are the load warp's generic stores (sfd_atoms): no tx bytes. cov1: or no V atom at all (a
            # per-tensor E4M3 V, the NVFP4 S operand's atoms only)
            assert mSfdF is not None and self.quant_qk and (self.quant_pv or not self.fp8_sfd) and not self.vtok_tma
            self.tma_copy_bytes["Q"] -= cute.size_in_bytes(mSFQ.element_type, cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]))
            self.tma_copy_bytes["K"] -= cute.size_in_bytes(mSFK.element_type, cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]))
            if const_expr(self.quant_pv):
                self.tma_copy_bytes["V"] -= cute.size_in_bytes(mSFV.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))
            self.sfd_bytes = tuple(cute.size_in_bytes(t.element_type, cute.select(l, mode=[0, 1, 2]))
                                   for t, l in ((mSFQ, sfq_smem_layout_staged), (mSFK, sfk_smem_layout_staged),
                                                *(((mSFV, sfv_smem_layout_staged),) if self.quant_pv else ())))
            if const_expr(self.v4_packed):   # (cov5: V's codes are the load warp's realigned stores, no TMA)
                self.tma_copy_bytes["V"] = 0
                # the V^T staging windows by ONE tensor TMA per key tile (128 d rows x 80 bytes from a 16-byte-aligned
                # column; 128 single-row bulk copies measured 3.4x the forward): an atom per payload's parity copies,
                # appended to the sf_dense list (mSfdF[9:13])
                from flash_attn.cute import v4_packed as V4
                _v4ta = []
                for _m in (mSfdF[5], mSfdF[7]):
                    _tb = _m.shape[3]
                    _m2 = cute.make_tensor(_m.iterator, cute.make_layout((_m.shape[0] * _m.shape[1] * _m.shape[2], _tb),
                                                                         stride=(_tb, 1)))
                    _a, _t = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), _m2,
                                                         cute.make_ordered_layout((128, V4.ROW_WINDOW), order=(1, 0)),
                                                         (128, V4.ROW_WINDOW))
                    _v4ta += [_a, _t]
                self._v4_tma_args = tuple(_v4ta)
        if const_expr(self.use_2cta_forward):
            # two-cta-forward: the leader's full barrier counts both CTAs' bytes -- each CTA's own Q / SFQ / K half / V half,
            # and the multicast SFK / SFV tile that lands in both CTAs' SMEM
            for _n in ("Q", "K", "V"):
                self.tma_copy_bytes[_n] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        mQ_shape = mQ.shape
        # Scale-factor tile order. Dense tensors are (s, d, h, b); under cu_seqlens the batch mode is
        # gone and they are (total, d, h), so the order tuple must match the rank -- d fastest, then the
        # sequence, then the head. The FP4 scale factor of a token is a max WITHIN that token's row along
        # d (the quantiser reduces over the group axis), so packing documents together
        # never mixes them: a 128-row SF atom straddling a document boundary is numerically identical to
        # quantising each document alone (verified bitwise).
        sf_order = (2, 1, 3, 4) if const_expr(cute.rank(mQ_shape) == 4) else (2, 1, 3)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_K = None
        tma_atom_V = None
        mK_shape = mK.shape
        mV_shape = mV.shape
        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )

        # codes_packed: the codes are T packed rows, the scale factors S aligned rows (the SF views' tile counts)
        mQ_sf_shape, mK_sf_shape, mV_sf_shape = mQ_shape, mK_shape, mV_shape
        if const_expr(self.codes_packed):
            if const_expr(mSFQ is not None):
                mQ_sf_shape = (mSFQ.shape[2] * 128,) + tuple(mQ_shape[1:])
            if const_expr(mSFK is not None):
                mK_sf_shape = (mSFK.shape[2] * 128,) + tuple(mK_shape[1:])
            if const_expr(mSFV is not None):
                mV_sf_shape = (mV_shape[0], mSFV.shape[4] * 128) + tuple(mV_shape[2:])
        # TMA load for scale factors
        tma_atom_sfq = None
        tma_tensor_sfq = None
        tma_atom_sfk = None
        tma_tensor_sfk = None
        tma_atom_sfv = None
        tma_tensor_sfv = None
        tma_atom_vt = None
        tma_tensor_vt = None
        tiled_mma_sfb_qk = None
        tiled_mma_sfb_pv = None
        # Setup TMA load for SFQ
        if const_expr(self.quant_qk):
            sfq_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout,
                                            self._qk_sf_shape(mQ_sf_shape), sf_order)
            sfq_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A(
                sfq_op,
                mSFQ,
                cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]),
                self.mma_tiler_qk_sf,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # Setup TMA load for SFK (scale factor for K, like SFB)
            sfk_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )

        if const_expr(self.quant_qk):
            sfk_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout,
                                            self._qk_sf_shape(mK_sf_shape), sf_order)
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)

            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_qk = (
                self.mma_tiler_qk[0],
                self.mma_tiler_qk[1],
                self.mma_inst_bits_k // self.k_dtype.width,
            )

            mma_inst_shape_mnk_sfb_qk = (
                mma_inst_shape_mnk_qk[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_qk[1], 128),
                mma_inst_shape_mnk_qk[2],
            )
        
            mma_tiler_sfb_qk = (
                mma_inst_shape_mnk_sfb_qk[0],
                mma_inst_shape_mnk_sfb_qk[1],
                mma_inst_shape_mnk_sfb_qk[2] * self.mma_inst_tile_k_sf,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_qk[:2],
            )
            cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_qk.thr_id.shape,),
            )
            tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_B(
                sfk_op,
                mSFK,
                cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_qk,
                tiled_mma_sfb_qk,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
    
        if const_expr(self.quant_pv):
            # Setup TMA load for SFV (scale factor for V, like SFB)
            sfv_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_pv.thr_id
            )
            # Setup scale factor tensor layout
            # d = 64: the SF B tiles of V (rows = d) are whole 128-row atoms (the quantiser fills rows d..127; the N = 64
            # MMA reads rows 0..63), so the gmem SF view and the load's tile are padded to 128 rows (native head_dim 64)
            self.sfv_n_pad = cute.round_up(self.mma_tiler_pv[1], 128) != self.mma_tiler_pv[1]
            _mV_sf_shape = ((cute.round_up(self.mma_tiler_pv[1], 128),) + tuple(mV_sf_shape[1:])
                            if self.sfv_n_pad else mV_sf_shape)
            sfv_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size_pv).layout,
                                            _mV_sf_shape, sf_order)
            mSFV = cute.make_tensor(mSFV.iterator, sfv_layout)
            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_pv = ( # the same processed by one tcgen05.mma instruction
                self.mma_tiler_pv[0],
                self.mma_tiler_pv[1],
                self.mma_inst_bits_k // self.v_dtype.width,
            )
            use_2cta_instrs = self.mma_tiler_pv[0] == 256
            mma_inst_shape_mnk_sfb_pv = (
                mma_inst_shape_mnk_pv[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_pv[1], 128),
                mma_inst_shape_mnk_pv[2],
            )
            mma_tiler_sfb_pv = (
                mma_inst_shape_mnk_sfb_pv[0],
                mma_inst_shape_mnk_sfb_pv[1],
                mma_inst_shape_mnk_sfb_pv[2] * self.mma_inst_tile_k_pv,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype_pv,
                self.sf_vec_size_pv,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_pv[:2],
                p_source,
            )
            cluster_layout_sfb_pv_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_pv.thr_id.shape,),
            )
            tma_atom_sfv, tma_tensor_sfv = cute.nvgpu.make_tiled_tma_atom_B(
                sfv_op,
                mSFV,
                cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_pv,
                tiled_mma_sfb_pv,
                cluster_layout_sfb_pv_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # rebuilt inside the kernel (an MMA object built here cannot be used in the kernel's region)
            _sfb_pv_args = (self.v_dtype, p_major_mode, self.v_major_mode, self.sf_dtype_pv, self.sf_vec_size_pv,
                            cute.nvgpu.tcgen05.CtaGroup.ONE, mma_inst_shape_mnk_sfb_pv[:2], p_source)
            self.make_tiled_mma_sfb_pv = lambda: sm100_utils_basic.make_blockscaled_trivial_tiled_mma(*_sfb_pv_args)
            self.mma_tiler_sfb_pv = mma_tiler_sfb_pv
            if const_expr(self.vtok_tma):
                mVTokSF = cute.make_tensor(mVTokSF.iterator, sfv_layout)
                tma_atom_vt, tma_tensor_vt = cute.nvgpu.make_tiled_tma_atom_B(
                    sfv_op,
                    mVTokSF,
                    cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]),
                    mma_tiler_sfb_pv,
                    tiled_mma_sfb_pv,
                    cluster_layout_sfb_pv_vmnk.shape,
                    internal_type=cutlass.Int16,
                )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        mO_rows = mO   # (o_rows_tma: the gmem O of the partial tiles' row copies)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        if const_expr(self.use_tma_O and not self.o_rows_tile):
            pass
        else:
            if const_expr(not self.use_tma_O):
                tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(mVarlenWork is not None):
            assert mCuSeqlensQ is not None and not self.is_split_kv and not self.pack_gqa and not self.use_2cta_forward
            TileScheduler = VarlenWorkListScheduler
            if const_expr(self.varlen_sched != "static"):
                TileScheduler = VarlenWorkListDynScheduler
        elif const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                # persistent causal (fwd_causal_persistent=1 at the interface): the LPT order on a persistent grid
                TileScheduler = PersistentLPTScheduler if const_expr(self.is_persistent) else SingleTileLPTScheduler
                if const_expr(self.is_persistent and self.causal_zigzag_order):
                    TileScheduler = PersistentZigzagLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        # two-cta-forward: the scheduler's tile is the cluster's 2 x cta_tiler[0] rows (the static persistent scheduler
        # divides by the cluster size itself, so it is given CTA tiles)
        _num_block_rows = self.cta_tiler[0] * (
            self.cta_group_size if const_expr(TileScheduler is not StaticPersistentTileScheduler) else 1)
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_rows),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            # For sub-byte dtypes (FP4), width=4 rounds to 0 under integer division;
            # clamp so size_one_head * element_size in the scheduler's L2 swizzle calc
            # doesn't become 0 and divide by zero in the else branch of the ifexp.
            element_size=max(self.k_dtype.width // 8, 1),
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=self.use_2cta_forward,
        )
        if const_expr(mVarlenWork is not None and self.varlen_sched != "static"):
            # list words: [header (2), n_max entries (2 n_max), counters (4)] (lp_varlen.varlen_worklist ctr=True)
            _n_max = (cute.size(mVarlenWork.shape[0]) - 6) // 2
            _bound = _n_max
            if const_expr(self.varlen_sched == "clc"):
                # entries <= sum over documents of ceil(slot / 2) <= (n_max + b) / 2: CTAs past it would only be drained
                _bound = cutlass.min(_n_max, (_n_max + cute.size(mCuSeqlensQ.shape[0]) - 1) // 2)
            # consumer warps of the ring: every warp running a role loop (the producer is the first empty warp)
            _cons = (*self.softmax0_warp_ids, *self.softmax1_warp_ids, *self.correction_warp_ids, self.mma_warp_id,
                     *self.load_warp_ids)
            if const_expr(not self.use_correction_warps_for_epi):
                _cons = _cons + tuple(self.epilogue_warp_ids)
            tile_sched_params = VarlenWorkListDynScheduler.Params(
                mVarlenWork, cute.size(mQ.shape[2]), _bound * cute.size(mQ.shape[2]), Int32(2 + 2 * _n_max),
                self.varlen_sched, self.varlen_sched_stages, len(_cons), self.varlen_sched_jit)
        elif const_expr(mVarlenWork is not None):
            # (n_entries upper bound = (len - 2) / 2) x heads: the grid's tile bound, from the list's host-sized shape
            tile_sched_params = VarlenWorkListScheduler.Params(
                mVarlenWork, cute.size(mQ.shape[2]),
                (cute.size(mVarlenWork.shape[0]) - 2) // 2 * cute.size(mQ.shape[2]), True, Int32(0), False)
        else:
            tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage * self.q_bufs
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage * self.q_bufs
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.q_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.q_stage
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + self.q_stage
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + self.q_stage
        # softmax_corr_* are per Q stage, corr_epi_* per sO stage (E42: they may differ)
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.q_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.epi_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + self.epi_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        # QK and PV SF tmem load wait for softmax t2r store
        self.mbar_sfqk_load_offset = self.mbar_P_full_2_offset + self.q_stage 
        self.mbar_sfpv_load_offset = self.mbar_sfqk_load_offset + self.q_stage
        # (3 barriers of the removed column-split pipeline: kept so the barrier offsets stay as measured)
        self.mbar_corr_final_offset = self.mbar_sfpv_load_offset + self.q_stage
        self.mbar_p_empty_offset = self.mbar_corr_final_offset + 1
        self.mbar_sa_full_offset = self.mbar_p_empty_offset + 2  # split pipe: S a-half full [2]
        self.mbar_sa_loaded_offset = self.mbar_sa_full_offset + 2  # split pipe: softmax loaded S a-half [2]
        self.mbar_vtok_offset = self.mbar_sa_loaded_offset + 2  # vtok SMEM staging: the step's beta / u written [2]
        # (8 barriers of the removed P-cross / S-ahead pipelines -- s_empty [2], inplace [2], pfree [4] -- stay
        # allocated so the barrier array keeps the measured layout)
        self.mbar_total = self.mbar_vtok_offset + 3 + 2 + 2 + 4
        # two-cta-forward: the peer MMA warp's "done with TMEM" arrival before the cta_group::2 dealloc
        self.mbar_tmem_dealloc_peer_offset = self.mbar_total
        self.mbar_total += 1
        # dynamic work-list claiming (varlen_sched != 'static'): the scheduler ring (VarlenWorkListDynScheduler.ring_words),
        # 16-byte aligned (its CLC response word); absent on every other build
        self.mbar_sched_ring_offset = self.mbar_total
        if const_expr(mVarlenWork is not None and self.varlen_sched != "static"):
            assert not self.vtok_w and len(self.empty_warp_ids) > 0 and not self.use_2cta_forward
            self.mbar_sched_ring_offset = self.mbar_total + (self.mbar_total % 2)
            self.mbar_total = self.mbar_sched_ring_offset + VarlenWorkListDynScheduler.ring_words(self.varlen_sched_stages)
        # vtok_w: ring of vtok_ring key tiles x 256 floats, staged by the load warp when it loads K(t) (slot t mod ring), read in
        # bulk by the softmax threads
        self.vtok_ring = 12  # SMEM 12 x 256 floats; the stager may run ring - 1 key tiles ahead of the slower softmax stage
        if const_expr(self.vtok_w and not self.vtok_tma):
            self.vtok_smem_floats = self.vtok_ring * 256
        else:
            self.vtok_smem_floats = 1
        # self.mbar_total = self.mbar_P_full_2_offset + self.q_stage
        # P handoff split: fraction of P store chunks (and matching PV MMA
        # K-tiles) released at the first P_full signal; the rest go through
        # P_full_2's embedded wait inside the PV GEMM.
        # 3/4 measured best for FP8 PV on GB300: 1909 -> 2366 TF at s=32768
        # (the old 1/2 default came from B200 tuning). BF16 PV already used
        # 3/4; FP4 PV split is num/den of its store chunks (k//2 when only 2).
        fp8_pv_split_num = int(self.spec.fp8_pv_p_split_num)  # B200: 1/2 (3/4 was a B300 tuning; 0.178 -> 0.142 ms s4096 h24)
        fp8_pv_split_den = int(self.spec.fp8_pv_p_split_den)
        fp4_pv_split_num = int(self.spec.fp4_pv_p_split_num)
        fp4_pv_split_den = int(self.spec.fp4_pv_p_split_den)
        fp8_pv_small_d = self.head_dim_v_padded <= 64
        self.mbar_p_split = lambda k: (
            max(
                1,
                min(
                    k - 1,
                    k
                    * (3 if cutlass.const_expr(fp8_pv_small_d) else fp8_pv_split_num)
                    // (4 if cutlass.const_expr(fp8_pv_small_d) else fp8_pv_split_den),
                ),
            )
            if cutlass.const_expr(self.v_dtype.width == 8 and k > 1) else
            (
                (k // 4 * 3) if cutlass.const_expr(self.v_dtype.width > 8) else
                max(1, min(k - 1, k * fp4_pv_split_num // fp4_pv_split_den))
            )
        )
        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 1
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )
        
        # Calculate scale factor shared memory sizes
        # Use size 1 as minimum to avoid alignment issues when size is 0
        sfq_smem_size = cute.cosize(sfq_smem_layout_staged) if const_expr(self.quant_qk) else 1
        sfk_smem_size = cute.cosize(sfk_smem_layout_staged) if const_expr(self.quant_qk) else 1
        sfp_smem_size = cute.cosize(sfp_smem_layout_staged) if const_expr(self.quant_pv) else 1
        sfv_smem_size = cute.cosize(sfv_smem_layout_staged) if const_expr(self.quant_pv) else 1

        # token-local operands: per softmax warp, the key tile's per-key rows (log2 f_V, 1 / f_V[, f_K])
        self.tok_rows = (3 if self.tb_tok_qk else 2) if self.tb_tok_v else 0
        self.tok_slot_floats = 8 * self.n_block_size * self.tok_rows if self.tb_tok_v else 4
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, self.sscale_floats], self.buffer_align_bytes]
            sXchg: cute.struct.Align[cute.struct.MemRange[Float32, self.sxchg_floats], self.buffer_align_bytes]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            # K reuses V's buffer when K is smaller (FP4 K in BF16 V), or same dtype
            k_aliases_v = self.k_dtype.width < self.v_dtype.width
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, 1] if const_expr(k_aliases_v) else cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                1 if const_expr(k_aliases_v) else self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)] if not const_expr(self.v_dtype == self.k_dtype) else cute.struct.MemRange[self.k_dtype, 1],
                self.buffer_align_bytes if not const_expr(self.v_dtype == self.k_dtype) else 1,
            ]
            # Scale factor shared memory (if block-scaled quantization is used)
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfq_smem_size],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfk_smem_size],
                self.buffer_align_bytes,
            ]
            sSFP: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype_pv, sfp_smem_size],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype_pv, sfv_smem_size],
                self.buffer_align_bytes,
            ]
            sVtok: cute.struct.Align[cute.struct.MemRange[Float32, self.vtok_smem_floats], self.buffer_align_bytes]
            sVtokSF: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype_pv, sfv_smem_size if const_expr(self.vtok_tma) else 1],
                self.buffer_align_bytes,
            ]
            # token-local operands: each softmax warp's per-key slot of the current key tile
            sTok: cute.struct.Align[cute.struct.MemRange[Float32, self.tok_slot_floats], 16]

        if const_expr(self.sf_dense):
            # sf_dense only (the other kernels' storage is unchanged): the key tiles' staging ring and its mbarriers
            SharedStorage.__annotations__["sKvStg"] = cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, SFD_KV_SLOTS * self.sfd_kv_slot], 128]
            SharedStorage.__annotations__["sKvStg_mbar"] = cute.struct.MemRange[cutlass.Int64, SFD_KV_SLOTS]
            if const_expr(self.v4_packed):
                from flash_attn.cute import v4_packed as V4
                SharedStorage.__annotations__["sV4Stg"] = cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, V4.STAGES * V4.STAGE_BYTES], 128]
                SharedStorage.__annotations__["sV4Stg_mbar"] = cute.struct.MemRange[cutlass.Int64, V4.STAGES]
        SharedStorage = cute.struct(SharedStorage)

        # Remove scale factors to avoid OOM. Seems I can't set their size to 0
        @cute.struct
        class SharedStorageBF16:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, self.sscale_floats], self.buffer_align_bytes]
            sXchg: cute.struct.Align[cute.struct.MemRange[Float32, self.sxchg_floats], self.buffer_align_bytes]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            # token-local operands: each softmax warp's per-key slot of the current key tile
            sTok: cute.struct.Align[cute.struct.MemRange[Float32, self.tok_slot_floats], 16]
        self.shared_storage = SharedStorage if const_expr(self.quant_qk) or const_expr(self.quant_pv) else SharedStorageBF16
        
        # Verify shared memory fits within budget
        total_smem_bytes = self.shared_storage.size_in_bytes()
        assert total_smem_bytes <= 227 * 1024, (
            f"SharedStorage {total_smem_bytes // 1024}KB exceeds 227KB limit. "
            f"Reduce kv_stage (currently {self.kv_stage})."
        )
        
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmod.create(seqlen_q)
            seqlen_k_divmod = FastDivmod.create(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            tma_atom_sfq,
            tma_tensor_sfq,
            tma_atom_sfk,
            tma_tensor_sfk,
            tma_atom_sfv,
            tma_tensor_sfv,
            sfq_smem_layout_staged,
            sfk_smem_layout_staged,
            sfp_smem_layout_staged,
            sfv_smem_layout_staged,
            v_descale,
            mPmax,
            mVBias,
            mVU,
            mVTokW,
            tma_atom_vt,
            tma_tensor_vt,
            mSoftmaxScale,
            mTbFac,
            mRepro,
            mReproDbg,
            mTbTokV,
            tiled_mma_sfb_qk if const_expr(self.use_2cta_forward) else None,
            tiled_mma_sfb_pv if const_expr(self.use_2cta_forward) else None,
            mCuSeqlensO,
            mCuSeqlensKc,
            mCuSeqlensVt,
            *(((list(mSfdF) + [mO_rows]) if const_expr(self.o_rows_tma) else mSfdF,)
              if const_expr(self.sf_dense) else ()),
            *(self._v4_tma_args if const_expr(self.v4_packed) else ()),
            **({"mORows": mO_rows} if const_expr(self.o_rows_tma1) else {}),
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        tma_atom_sfq: Optional[cute.CopyAtom] = None,
        tma_tensor_sfq: Optional[cute.Tensor] = None,
        tma_atom_sfk: Optional[cute.CopyAtom] = None,
        tma_tensor_sfk: Optional[cute.Tensor] = None,
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfq_smem_layout_staged: Optional[cute.Layout] = None,
        sfk_smem_layout_staged: Optional[cute.Layout] = None,
        sfp_smem_layout_staged: Optional[cute.Layout] = None,
        sfv_smem_layout_staged: Optional[cute.Layout] = None,
        v_descale: Optional[cute.Tensor] = None,
        mPmax: Optional[cute.Tensor] = None,
        mVBias: Optional[cute.Tensor] = None,
        mVU: Optional[cute.Tensor] = None,
        mVTokW: Optional[cute.Tensor] = None,
        tma_atom_vt: Optional[cute.CopyAtom] = None,
        tma_tensor_vt: Optional[cute.Tensor] = None,
        mSoftmaxScale: Optional[cute.Tensor] = None,
        mTbFac: Optional[cute.Tensor] = None,
        mRepro: Optional[cute.Tensor] = None,
        mReproDbg: Optional[cute.Tensor] = None,
        mTbTokV: Optional[cute.Tensor] = None,
        tiled_mma_sfb_qk: Optional[cute.TiledMma] = None,   # two-cta-forward: the multicast SFK / SFV tiles' MMAs
        tiled_mma_sfb_pv: Optional[cute.TiledMma] = None,
        mCuSeqlensO: Optional[cute.Tensor] = None,          # o_rows_packed: the packed O row starts
        mCuSeqlensKc: Optional[cute.Tensor] = None,         # codes_packed: the packed K row starts
        mCuSeqlensVt: Optional[cute.Tensor] = None,         # codes_packed: the channel-major V's starts
        mSfdF: Optional[list] = None,                       # sf_dense: the dense per-row tables
        # cov5 v4_packed: the V^T staging TMA atoms / coordinate tensors of the two payloads' parity copies
        tma_atom_v4a: Optional[cute.CopyAtom] = None,
        tma_tensor_v4a: Optional[cute.Tensor] = None,
        tma_atom_v4b: Optional[cute.CopyAtom] = None,
        tma_tensor_v4b: Optional[cute.Tensor] = None,
        mORows: Optional[cute.Tensor] = None,               # o_rows_tma1: the gmem O of the partial tiles' row copies
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """
        # ---- prequantised-qk-fused-scale BEGIN (prequant.py; one guarded rebind, nothing else in this file) ---------------
        # score_scale_unit: the caller folded log2(e) * softmax_scale and the operand descales into Q's descale, so
        # the score scalar is exactly 1.0. softmax_scale_log2 arrives as a dynamic kernel parameter, which ptxas
        # cannot fold; rebinding it to a compile-time constant here is what lets every `* scale_log2` disappear --
        # the row-max scaling, the correction factor, the LSE finalisation and the masked-tile constants. The
        # steady-state exp2 is `fma(x, scale, -rowmax*scale)`, so there it only turns an FFMA into an FADD.
        if const_expr(self.score_scale_unit):
            softmax_scale_log2 = Float32(1.0)
        # ---- prequantised-qk-fused-scale END --------------------------------------------------------------------------------
        if const_expr(mSoftmaxScale is not None):
            # GPU-only score scale: the float32 the host would have passed, from device memory; the log2 scale derived
            # exactly as __call__ derives it (softmax_scale * LOG2_E, one fp32 multiply)
            assert not self.score_scale_unit
            if const_expr(self.score_scale_doc):
                self._score_tab = mSoftmaxScale   # (n_docs, h_kv): read per work tile (_doc_score_log2)
            elif const_expr(self.score_mod is not None):
                # score_mod: the scale applied before the mod (softmax_scale_log2 stays the change of base, as __call__)
                softmax_scale = mSoftmaxScale[0]
            else:
                softmax_scale_log2 = mSoftmaxScale[0] * math.log2(math.e)
        if const_expr(mTbFac is not None):
            self._tb_fac = mTbFac   # token-block factors, read by the softmax threads (_tb_transform)
        if const_expr(mRepro is not None):
            self._repro = mRepro    # `reproduce-forward-p` export, written by the softmax threads
            if const_expr(self.codes_packed):
                self._cu_q = mCuSeqlensO   # (cov5: the export's packed query rows)
        if const_expr(mReproDbg is not None):
            self._repro_dbg = mReproDbg
        if const_expr(mTbTokV is not None):
            self._tb_tok_v = mTbTokV  # token-local operands: per-key V factors
        if const_expr(self.codes_packed and self.tb_varlen):
            self._tb_pk = (mCuSeqlensO, mCuSeqlensKc)   # (cov3: the fold tables' packed row starts)
        if const_expr(mVBias is not None):
            # vtok per-key tensors, read by the softmax threads (_vtok_add_bias, _directp_group); bound here so they are
            # values of the kernel region
            self._vtok_bias = mVBias
            self._vtok_u = mVU
            if const_expr(self.codes_packed):
                self._cu_kc = mCuSeqlensKc   # (cov5: the per-key tables' packed key starts)
        if const_expr(mVTokW is not None):
            self._vtok_w = mVTokW

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # two-cta-forward: this CTA's index in the pair (its 128-row half of every M = 256 MMA); the leader (0) issues the
        # MMAs, the scale-factor copies and the commits for both
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        if const_expr(self.use_2cta_forward):
            mma_tile_coord_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) % self.cta_group_size
        else:
            mma_tile_coord_v = Int32(0)
        is_leader_cta = mma_tile_coord_v == 0
        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(tma_atom_sfq is not None):
                cpasync.prefetch_descriptor(tma_atom_sfq)
            if const_expr(tma_atom_sfk is not None):
                cpasync.prefetch_descriptor(tma_atom_sfk)
            if const_expr(tma_atom_sfv is not None):
                cpasync.prefetch_descriptor(tma_atom_sfv)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()

        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage * self.q_bufs):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_full_offset + i, 2 if const_expr(self.o_rows_tma) else 1
                )   # (o_rows_tma: the pair's two load warps arrive -- each built its own SF atoms)
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id])
                )
        n_sm_warps = len(self.softmax0_warp_ids)
        if warp_idx == 2:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i, cute.arch.WARP_SIZE * 4
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i, cute.arch.WARP_SIZE * n_sm_warps
                )
            if const_expr(self.qk_split_pipe):
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_sa_full_offset + i, 1)
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_sa_loaded_offset + i, cute.arch.WARP_SIZE * 4)
                    if const_expr(self.vtok_w and not self.vtok_tma):
                        if const_expr(i == 0):
                            cute.arch.mbarrier_init(mbar_ptr + self.mbar_vtok_offset, 1)  # the stager warp's elected thread
                        cute.arch.mbarrier_init(mbar_ptr + self.mbar_vtok_offset + 1 + i, len(self.softmax0_warp_ids))
        # kept: removing it changes CuTe DSL codegen (measured: different PTX, ~1 % slower all-4-bit forward): a dynamic `if` whose
        # const-false body is the former exp-pingpong / s0-s1 barrier init (verbatim; a stub body does not reproduce the
        # PTX) still shapes the kernel PTX.
        if warp_idx == 3:
            if const_expr(False):
                if cute.arch.lane_idx() == 0:
                    _wg_pp_zero_t = storage.wg_pp_zero.get_tensor(cute.make_layout(4))
                    _wg_pp_zero_t[0] = Int32(0)
            if const_expr(False):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE
                    )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(self.epi_stage):   # one pair per sO stage (E42)
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(self.q_stage):
                # two-cta-forward: the leader's barrier collects both CTAs' softmax + correction arrivals
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    self._mma_arrive_threads
                    * (n_sm_warps + len(self.correction_warp_ids)) * self.cta_group_size,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    self._mma_arrive_threads * n_sm_warps * self.cta_group_size,
                )
        if warp_idx == 7:
            if const_expr(self.use_2cta_forward):
                # the peer MMA warp's 32 threads arrive here before this CTA deallocates (cta_group::2 TMEM)
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_tmem_dealloc_peer_offset, cute.arch.WARP_SIZE)
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        if warp_idx == 8:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_sfqk_load_offset + i,
                    self._mma_arrive_threads * n_sm_warps * self.cta_group_size
                )
        if const_expr(self.tile_scheduler_cls is VarlenWorkListDynScheduler):
            if warp_idx == 9:
                VarlenWorkListDynScheduler.init_barriers(tile_sched_params, mbar_ptr + self.mbar_sched_ring_offset)
        if const_expr(self.sf_dense):
            if warp_idx == 10:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(SFD_KV_SLOTS):
                        cute.arch.mbarrier_init(storage.sKvStg_mbar.data_ptr() + i, 1)
                    if const_expr(self.v4_packed):
                        from flash_attn.cute import v4_packed as V4
                        for i in cutlass.range_constexpr(V4.STAGES):
                            cute.arch.mbarrier_init(storage.sV4Stg_mbar.data_ptr() + i, 1)
            # the load warp's sf_dense operands (an attribute, not load() arguments: the other kernels' PTX unchanged)
            if const_expr(self.o_rows_tma):
                self._o_rows = mSfdF[len(mSfdF) - 1]   # (the gmem O: the partial tiles' row copies)
            self._sfd_ld = (mSfdF, storage.sKvStg.get_tensor(cute.make_layout(SFD_KV_SLOTS * self.sfd_kv_slot)),
                            storage.sKvStg_mbar.data_ptr())
            if const_expr(self.v4_packed):
                self._v4_ctile = mSfdF[8]   # (the diagonal payload's per-tile SF word: 2^c_n)
                # (cov5) the V staging slots (smem byte address) and their barriers, the staging TMA atoms
                self._v4_stg = (storage.sV4Stg.data_ptr().toint(), storage.sV4Stg_mbar.data_ptr())
                self._v4_tma = (tma_atom_v4a, tma_tensor_v4a, tma_atom_v4b, tma_tensor_v4b)
        if const_expr(self.o_rows_tma1):
            self._o_rows = mORows   # (spd3: the gmem O of the partial tiles' row copies)
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        # (two-cta-forward: created over the pair -- multicast K / V release, leader-only expect_tx -- and its barrier
        # init sync is cluster-wide, which every remote arrive / multicast below relies on)
        pipeline_kv = self.make_and_init_load_kv_pipeline(
            mbar_ptr + self.mbar_load_kv_full_offset,
            cta_layout_vmnk=cta_layout_vmnk if const_expr(self.use_2cta_forward) else None,
        )

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # K and V share physical SMEM:
        # - same dtype: V aliases K (V uses K's base pointer)
        # - FP4 K + BF16 V: K aliases V (K uses V's base pointer, stride scaled by dtype ratio)
        if const_expr(self.v_dtype == self.k_dtype):
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
            if const_expr(self.v4_packed):   # (cov5: the V stages' byte base for the load warp's realigned stores)
                self._v4_sv = storage.sK.data_ptr().toint()
        elif const_expr(self.k_dtype.width < self.v_dtype.width):
            # K aliases V's buffer -- K is smaller, fits inside V's stage.
            # K's stage stride must match V's stage stride in bytes so they align.
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stride_sK_aligned = const_expr(stride_sV * self.v_dtype.width // self.k_dtype.width)
            sK_outer_aligned = cute.make_layout(
                sK_layout.outer.shape,
                stride=(*sK_layout.outer.stride[:-1], stride_sK_aligned),
            )
            sK = storage.sV.get_tensor(sK_outer_aligned, swizzle=sK_layout.inner, dtype=self.k_dtype)
        else:
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            # aliased onto sQ (overlap mode) or never written (O direct store)
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.sscale_floats))
        sXchg = storage.sXchg.get_tensor(cute.make_layout(self.sxchg_floats))
        if const_expr(self.vtok_w):
            self._vtok_smem_t = storage.sVtok.get_tensor(cute.make_layout(self.vtok_smem_floats))
        if const_expr(self.tb_tok_v):
            # [key n_block_size][row tok_rows][softmax warp 8]
            self._tok_slot = storage.sTok.get_tensor(cute.make_layout(
                (self.n_block_size, self.tok_rows, 8), stride=(1, self.n_block_size, self.n_block_size * self.tok_rows)))
        sVT = None
        if const_expr(self.vtok_tma):
            sVT = storage.sVtokSF.get_tensor(sfv_smem_layout_staged)
            self._vtok_sf_base = storage.sVtokSF.get_tensor(cute.make_layout(1)).iterator  # slot t = base + t * slot bytes

        # Get scale factor shared memory tensors if they exist
        sSFV = None
        sSFP = None
        sSFQ = None
        sSFK = None
        if const_expr(self.quant_qk):
            sSFQ = storage.sSFQ.get_tensor(sfq_smem_layout_staged)
            sSFK = storage.sSFK.get_tensor(sfk_smem_layout_staged)
        if const_expr(sfp_smem_layout_staged is not None):
            sSFP = storage.sSFP.get_tensor(sfp_smem_layout_staged)
        if const_expr(sfv_smem_layout_staged is not None):
            sSFV = storage.sSFV.get_tensor(sfv_smem_layout_staged)

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)  # 1-CTA: slice 0; 2-CTA: this CTA's half
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)


        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(2)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(self.q_stage)
        )

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator
                + self.qk_acc_dtype.width // tOrP._dtype.width * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(2)
        ]
        # Setup scale factor TMEM tensors and S2T copy operations.
        # For QK, park the block-scaled metadata on the opposite S stage so it
        # doesn't alias the accumulator tile currently being produced.
        align = 16 # required for tcgen05.cp 4x32dp128bit TODO: may be just 4, 4 * 32 = 128 bits
        tCtSFQs = [None] * self.q_stage
        tCtSFKs = [None] * self.q_stage
        tCtSF_split = None
        if const_expr(self.quant_qk):
            # SFQK tmem placement (an earlier experiment compared these slots):
            #   "s"  (default) -- use tmem_s_offset (staggered): SF for stage k
            #                    lives where S for stage (q_stage-1-k) lives.
            #                    Safe by construction; current baseline.
            #   "o"            -- use tmem_o_offset slots. Frees S's tmem from
            #                    SFQK traffic so softmax warp + SFQK MMA copy
            #                    don't compete. Risk: collides with O acc.
            #   "o_stagger"    -- o offsets with opposite-stage staggering
            #                    (stage k -> tmem_o_offset[q_stage-1-k]).
            sfq_base_offsets = self.tmem_s_offset
            if const_expr(self.q_stage == 1):
                # Single stage: the S1 slot (cols [128, 256)) is unused, so
                # SFQ/SFK live there rather than aliasing the S0 accumulator
                # that the same QK MMA is writing.
                sfq_stage_order = (1,)
            else:
                sfq_stage_order = tuple(
                    self.q_stage - 1 - stage for stage in range(self.q_stage)
                )
            sfq_tmem_ptrs_f32 = [
                cute.make_ptr(
                    Float32,
                    sfq_base_offsets[sfq_stage_order[stage]],
                    mem_space=cute.AddressSpace.tmem,
                    assumed_align=align,
                )
                for stage in range(self.q_stage)
            ]
            sfq_tmem_ptrs = [
                cute.recast_ptr(sfq_tmem_ptrs_f32[stage], dtype=self.sf_dtype)
                for stage in range(self.q_stage)
            ]

            # (MMA, MMA_M, MMA_K)
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk_sf,
                self.sf_vec_size,
                cute.slice_(sfq_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFQs = [cute.make_tensor(sfq_tmem_ptrs[stage], tCtSFQ_layout) for stage in range(self.q_stage)]

            # Make SFK tmem tensor
            # TMEM offsets are expressed in u32 columns. Match the dense
            # blockscaled kernels instead of deriving a smaller offset from the
            # recast FP8 tensor view.
            sf_atom_mn = 32
            sfq_tmem_cols = (self.m_block_size // sf_atom_mn) * self.mma_inst_tile_k_sf  # per CTA (its own SFA rows)
            sfk_tmem_ptrs = [
                cute.recast_ptr(sfq_tmem_ptrs_f32[stage] + sfq_tmem_cols, dtype=self.sf_dtype)
                for stage in range(self.q_stage)
            ]

            # (MMA, MMA_N, MMA_K)
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk_sf,
                self.sf_vec_size,
                cute.slice_(sfk_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFKs = [cute.make_tensor(sfk_tmem_ptrs[stage], tCtSFK_layout) for stage in range(self.q_stage)]
            if const_expr(self.qk_split_pipe):
                # per stage: SFQ_a @ s+96, SFK_a @ s+104, SFQ_b @ s+48, SFK_b @ s+56 (own bank)
                def _sf_tensor(col, dt, lay):
                    return cute.make_tensor(
                        cute.recast_ptr(
                            cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=align),
                            dtype=dt,
                        ),
                        lay,
                    )
                tCtSF_split = (
                    [_sf_tensor(self.tmem_s_offset[st] + 96, self.sf_dtype, tCtSFQ_layout) for st in range(2)],
                    [_sf_tensor(self.tmem_s_offset[st] + 104, self.sf_dtype, tCtSFK_layout) for st in range(2)],
                    [_sf_tensor(self.tmem_s_offset[st] + 48, self.sf_dtype, tCtSFQ_layout) for st in range(2)],
                    [_sf_tensor(self.tmem_s_offset[st] + 56, self.sf_dtype, tCtSFK_layout) for st in range(2)],
                )
        
        # Setup SFP and SFV TMEM tensors
        # Reuse the TMEM of S
        tCtSFPs = [None] * self.q_stage
        tCtSFVs = [None] * self.q_stage
        if const_expr(self.quant_pv):
            _sfp_col = 80 if self.qk_split_pipe else 0
            sfp_tmem_ptrs = [cute.recast_ptr(
                            cute.make_ptr(Float32, self.tmem_s_offset[stage] + _sfp_col,
                            mem_space=cute.AddressSpace.tmem, assumed_align=align),
                            dtype=self.sf_dtype_pv) for stage in range(self.q_stage)]
            # (MMA, MMA_M, MMA_K) 
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size_pv,
                cute.slice_(sfp_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFPs = [cute.make_tensor(sfp_tmem_ptrs[stage], tCtSFP_layout) for stage in range(self.q_stage)]
            
            # Make SFV tmem tensor.
            # find_tmem_tensor_col_offset returns u32 cols; sfp_tmem_ptrs are
            # typed as `self.sf_dtype` (1 byte for E4M3/E8M0), so we convert
            # u32 cols -> SF-element offset via `sf_dtype_per_u32 = 32 / sf_width`.
            # Both E4M3 and E8M0 are 8 bits -> 4 SFs per u32 col.
            sf_dtype_per_u32 = 32 // self.sf_dtype_pv.width
            sfp_offset = math.ceil(tcgen05.find_tmem_tensor_col_offset(tCtSFPs[0]) * sf_dtype_per_u32 / align) * align
            sfv_tmem_ptrs = [sfp_tmem_ptrs[stage] + sfp_offset for stage in range(self.q_stage)]

            # (MMA, MMA_N, MMA_K) for P*V operation (V is the B matrix)
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size_pv,
                cute.slice_(sfv_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFVs = [cute.make_tensor(sfv_tmem_ptrs[stage], tCtSFV_layout) for stage in range(self.q_stage)]

        # two-cta-forward: the work tile (and so the KV range every role walks) is the cluster's 2 x cta_tiler[0] rows;
        # block_info_wg gives a softmax warpgroup the range and the causal-mask boundary of its own 256-row group
        # (the rows Q stage `stage` covers across the pair), exactly the 1-CTA kernel's per-CTA values
        block_info_wg = None
        if const_expr(self.use_2cta_forward or self.stage_skip or self.stage_skip_wg_mask):
            block_info_wg = BlockInfo(
                self.m_block_size * self.cta_group_size,
                self.cta_tiler[1],
                self.is_causal,
                self.is_local,
                self.is_split_kv,
                window_size_left,
                window_size_right,
                qhead_per_kvhead_packgqa=1,
            )
        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0] * self.cta_group_size,
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        if const_expr(self.tile_scheduler_cls is VarlenWorkListDynScheduler):
            TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params,
                                       mbar_ptr + self.mbar_sched_ring_offset)
        else:
            TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
                if const_expr(self.vtok_w and not self.vtok_tma):
                    self.vtok_stager(mbar_ptr, block_info, num_splits, SeqlenInfoCls, TileSchedulerCls)
                if const_expr(self.tile_scheduler_cls is VarlenWorkListDynScheduler and not self.varlen_sched_jit):
                    # dynamic work-list claiming: this warp is the ring's producer
                    TileSchedulerCls().produce()

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # Idle softmax WG (q_stage == 1): release registers and fall through to
        # the kernel end. Its threads are excluded from every barrier count.
        if const_expr(len(self.idle_warp_ids) > 0):
            if warp_idx >= self.idle_warp_ids[0] and warp_idx <= self.idle_warp_ids[-1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_sfq,
                tma_tensor_sfq,
                tma_atom_sfk,
                tma_tensor_sfk,
                sSFQ,
                sSFK,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                tma_atom_sfv,
                tma_tensor_sfv,
                sfv_smem_layout_staged,
                sSFV,
                tma_atom_vt,
                tma_tensor_vt,
                sVT,
                mma_tile_coord_v=mma_tile_coord_v if const_expr(self.use_2cta_forward) else None,
                tiled_mma_sfb_qk=tiled_mma_sfb_qk,
                tiled_mma_sfb_pv=tiled_mma_sfb_pv,
                **({"TileSchedulerProd": partial(TileSchedulerCls, producer=True)}
                   if const_expr(self.tile_scheduler_cls is VarlenWorkListDynScheduler and self.varlen_sched_jit) else {}),
                **({"mCuSeqlensO": mCuSeqlensO, "mCuSeqlensKc": mCuSeqlensKc, "mCuSeqlensVt": mCuSeqlensVt}
                   if const_expr(self.codes_packed) else {}),
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                # two-cta-forward: tcgen05.alloc.cta_group::2, issued by the MMA warp of each CTA of the pair
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf,
                                     is_two_cta=True if const_expr(self.use_2cta_forward) else None)
                cute.arch.sync_warp()

            mma_fn = self.mma_pair if const_expr(self.stage_skip) else (self.mma_split if const_expr(self.qk_split_pipe) else self.mma)
            mma_fn(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                sSFQ,
                sSFK,
                sSFV,
                sSFP,
                tCtSFQs,
                tCtSFKs,
                tCtSFPs,
                tCtSFVs,
                tCtSF_split=tCtSF_split,
                **({"is_leader_cta": is_leader_cta} if const_expr(self.use_2cta_forward) else {}),
                **({"block_info_wg": block_info_wg} if const_expr(self.stage_skip) else {}),
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True if const_expr(self.use_2cta_forward) else None)
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            if const_expr(self.use_2cta_forward):
                # both CTAs' softmax / correction warps are done with their TMEM (the leader's MMAs wrote into the
                # peer's): arrive on the peer's barrier, wait for the peer's arrival on ours, then deallocate
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_peer_offset,
                                          peer_cta_rank_in_cluster=mma_tile_coord_v ^ 1)
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_peer_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols,
                                   is_two_cta=True if const_expr(self.use_2cta_forward) else None)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mma_tile_coord_v=mma_tile_coord_v,
                    **({"mCuSeqlensO": mCuSeqlensO} if const_expr(self.o_rows_packed) else {}),
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        last_softmax_warp = (
            self.softmax0_warp_ids[-1] if self.q_stage == 1 else self.softmax1_warp_ids[-1]
        )
        if warp_idx <= last_softmax_warp:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                blocksparse_tensors=blocksparse_tensors,
                sSFP=sSFP,
                **({"block_info_wg": block_info_wg, "mma_tile_coord_v": mma_tile_coord_v}
                   if const_expr(self.use_2cta_forward) else {}),
                **({"block_info_wg": block_info_wg}
                   if const_expr((self.stage_skip or self.stage_skip_wg_mask) and not self.use_2cta_forward) else {}),
            )

            if const_expr(self.q_stage == 1):
                # Single stage: only softmax WG0 runs, always on stage 0.
                softmax_loop(
                    stage=0,
                    tStSi=cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout),
                    tCtSFP=tCtSFPs[0] if const_expr(self.quant_pv) else None,
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # Dynamic-stage dispatch
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                if const_expr(self.quant_pv):
                    tCtSFP = tCtSFPs[0] if stage == 0 else tCtSFPs[1]
                else:
                    tCtSFP = None
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(
                        tStS.iterator
                        + (self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]),
                        tStS.layout,
                    ),
                    tCtSFP=tCtSFP, # need to copy P sf to tmem after exp
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                sO,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                v_descale=v_descale,
                mPmax=mPmax,
                **({"mma_tile_coord_v": mma_tile_coord_v, "block_info_wg": block_info_wg}
                   if const_expr(self.use_2cta_forward) else {}),
                **({"block_info_wg": block_info_wg} if const_expr(self.stage_skip and not self.use_2cta_forward) else {}),
                **({"mCuSeqlensO": mCuSeqlensO} if const_expr(self.o_rows_packed) else {}),
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        if const_expr(self.use_2cta_forward):
            # no CTA of the pair leaves while the other may still signal its barriers (the leader's multicast
            # commits, the peer's remote arrives) or read its SMEM (the cta_group::2 scale-factor copies)
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.atom.ThrMma,
        thr_mma_pv: cute.atom.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_sfq: Optional[cute.CopyAtom],
        tma_tensor_sfq: Optional[cute.Tensor],
        tma_atom_sfk: Optional[cute.CopyAtom],
        tma_tensor_sfk: Optional[cute.Tensor],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfv_smem_layout_staged: Optional[cute.ComposedLayout] = None,
        sSFV: Optional[cute.Tensor] = None,
        tma_atom_vt: Optional[cute.CopyAtom] = None,
        tma_tensor_vt: Optional[cute.Tensor] = None,
        sVT: Optional[cute.Tensor] = None,
        mma_tile_coord_v: Optional[Int32] = None,   # two-cta-forward: this CTA's index in the pair
        tiled_mma_sfb_qk: Optional[cute.TiledMma] = None,   # two-cta-forward: the 1-CTA SFB MMAs (multicast SF tiles)
        tiled_mma_sfb_pv: Optional[cute.TiledMma] = None,
        # jit claim (varlen_sched_jit): the load warp's scheduler is the claiming one
        TileSchedulerProd: Optional[Callable] = None,
        mCuSeqlensO: Optional[cute.Tensor] = None,    # codes_packed: the packed Q row starts
        mCuSeqlensKc: Optional[cute.Tensor] = None,   # codes_packed: the packed K row starts
        mCuSeqlensVt: Optional[cute.Tensor] = None,   # codes_packed: the channel-major V's starts (16-aligned)
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        is_leader_cta = True if const_expr(not self.use_2cta_forward) else (mma_tile_coord_v == 0)
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        q_tile_count = Int32(0)   # E40: Q buffer = count & 1, its phase flips every 2nd tile
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        vt_ph = Int32(0)  # vtok_w: vtok barrier phase (one stager arrival per key tile, waited before K(t))
        # sf_dense: key-tile staging copies issued / consumed (ring slot = n % SFD_KV_SLOTS)
        kv_n_issue = Int32(0) if const_expr(self.sf_dense) else None
        kv_n_cons = Int32(0) if const_expr(self.sf_dense) else None
        # cov5 v4_packed: V staging copies issued / consumed (slot = count & 1)
        v4_iss = Int32(0) if const_expr(self.v4_packed) else None
        v4_con = Int32(0) if const_expr(self.v4_packed) else None
        if const_expr(self.o_rows_tma):
            self._pair_v = mma_tile_coord_v   # (_sfd_load_Q / _sfd_publish: this CTA's index in the pair)
        tile_scheduler = TileSchedulerCls() if const_expr(TileSchedulerProd is None) else TileSchedulerProd()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # mQ: [s, d, h, b]
            if const_expr(self.codes_packed):   # (the packed Q rows; the SFs below stay at the aligned tiles)
                mQ_cur = cute.domain_offset((mCuSeqlensO[batch_idx],) + (None,) * (cute.rank(mQ) - 1),
                                            mQ)[None, None, head_idx]
            else:
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)) # (bM, hdim/bK, RestM)

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                elif const_expr(self.codes_packed):   # (the packed K / V rows)
                    mK_cur = cute.domain_offset((mCuSeqlensKc[batch_idx], 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, mCuSeqlensVt[batch_idx]), mV[None, None, head_idx_kv])
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmod.create(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            # Partition SFQ similar to Q - index batch and head first, then use select like gQ
            load_SFQ_fn = None
            if const_expr(self.quant_qk):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    tma_tensor_sfq_cur = tma_tensor_sfq[None, None, head_idx, batch_idx]
                else:
                    # varlen: mode 0 of a scale-factor tensor is ((32, 4), n_atoms) -- sub-mode 1 counts
                    # 128-row SF ATOMS, not rows, so the document offset must be divided by the atom
                    # height. (The within-atom map r -> 16*(r%32) + 4*(r//32) is not linear in r, so a
                    # non-atom-aligned offset is not expressible at all; hence the 128-alignment
                    # requirement on cu_seqlens, checked on the host.)
                    tma_tensor_sfq_cur = cute.domain_offset(
                        ((0, seqlen.offset_q // _SF_ATOM_ROWS), 0), tma_tensor_sfq[None, None, head_idx]
                    )
                gSFQ = cute.local_tile(tma_tensor_sfq_cur, cute.select(self.mma_tiler_qk_sf, mode=[0, 2]), (None, 0))
                tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                load_SFQ_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_sfq, 0, cute.make_layout(1), tSgSFQ, sSFQ, filter_zeros=True
                )
            
            # Partition SFK similar to K - index batch and head first like mK_cur
            tKsSFK = None
            tKgSFK = None
            sfk_mcast_mask = None
            sfv_mcast_mask = None
            if const_expr(self.quant_qk):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, batch_idx]
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk_sf, mode=[1, 2]), (None, 0))
                    else:
                        # see the SFQ comment: the offset is in 128-row SF atoms, not rows
                        tma_tensor_sfk_cur = cute.domain_offset(
                            ((0, seqlen.offset_k // _SF_ATOM_ROWS), 0), tma_tensor_sfk[None, None, head_idx_kv]
                        )
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk_sf, mode=[1, 2]), (None, 0))
                else:
                    tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, None]
                    gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk_sf, mode=[1, 2]), (None, 0, None))
                if const_expr(self.use_2cta_forward):
                    # the SFK tile covers the pair's whole N (all 128 keys) and lands in both CTAs' SMEM: each CTA loads
                    # half of it and multicasts to both (the 2-CTA backward's SFQ pattern)
                    cl_sfb_qk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_sfb_qk.thr_id.shape,))
                    coord_sfb_qk = cl_sfb_qk.get_flat_coord(mma_tile_coord_v)
                    tSgSFK = tiled_mma_sfb_qk.get_slice(0).partition_B(gSFK)
                    tKsSFK, tKgSFK = cpasync.tma_partition(
                        tma_atom_sfk,
                        coord_sfb_qk[1],
                        cute.make_layout(cute.slice_(cl_sfb_qk, (0, None, 0, 0)).shape),
                        cute.group_modes(sSFK, 0, 3),
                        cute.group_modes(tSgSFK, 0, 3),
                    )
                    sfk_mcast_mask = cpasync.create_tma_multicast_mask(cl_sfb_qk, coord_sfb_qk, mcast_mode=1)
                else:
                    tSgSFK = thr_mma_qk.partition_B(gSFK)
                    # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                    tKsSFK, tKgSFK = cpasync.tma_partition(
                        tma_atom_sfk,
                        0, 
                        cute.make_layout(1),
                        cute.group_modes(sSFK, 0, 3),
                        cute.group_modes(tSgSFK, 0, 3),
                    )
                tKsSFK = cute.filter_zeros(tKsSFK)
                tKgSFK = cute.filter_zeros(tKgSFK)
            
            # Partition SFV similar to V
            tVsSFV, tVgSFV = None, None
            if const_expr(self.quant_pv):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, batch_idx]
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                    else:
                        tma_tensor_sfv_cur = cute.domain_offset((0, seqlen.offset_k), tma_tensor_sfv[None, None, head_idx_kv])
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                else:
                    tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, None]
                    gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
                if const_expr(self.use_2cta_forward):
                    # SFV: the whole head_dim_v of the pair in both CTAs (multicast, as SFK)
                    cl_sfb_pv = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_sfb_pv.thr_id.shape,))
                    coord_sfb_pv = cl_sfb_pv.get_flat_coord(mma_tile_coord_v)
                    tOgSFV = tiled_mma_sfb_pv.get_slice(0).partition_B(gSFV)
                    tVsSFV, tVgSFV = cpasync.tma_partition(
                        tma_atom_sfv,
                        coord_sfb_pv[1],
                        cute.make_layout(cute.slice_(cl_sfb_pv, (0, None, 0, 0)).shape),
                        cute.group_modes(sSFV, 0, 3),
                        cute.group_modes(tOgSFV, 0, 3),
                    )
                    sfv_mcast_mask = cpasync.create_tma_multicast_mask(cl_sfb_pv, coord_sfb_pv, mcast_mode=1)
                else:
                    if const_expr(self.sfv_n_pad):
                        # d = 64: the 128-row SF tile, partitioned as the SFB helper MMA's B (the TMA atom's own view)
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_sfb_pv, mode=[1, 2]), (0, None))
                        tOgSFV = self.make_tiled_mma_sfb_pv().get_slice(0).partition_B(gSFV)
                    else:
                        tOgSFV = thr_mma_pv.partition_B(gSFV)
                    # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                    tVsSFV, tVgSFV = cpasync.tma_partition(
                        tma_atom_sfv,
                        0,  
                        cute.make_layout(1),
                        cute.group_modes(sSFV, 0, 3),
                        cute.group_modes(tOgSFV, 0, 3),
                    )
                tVsSFV = cute.filter_zeros(tVsSFV)
                tVgSFV = cute.filter_zeros(tVgSFV)
            tVsVT, tVgVT = None, None
            if const_expr(self.vtok_tma):
                gVT = cute.local_tile(tma_tensor_vt[None, None, head_idx_kv, batch_idx], cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                tVsVT, tVgVT = cpasync.tma_partition(
                    tma_atom_vt, 0, cute.make_layout(1), cute.group_modes(sVT, 0, 3),
                    cute.group_modes(thr_mma_pv.partition_B(gVT), 0, 3),
                )
                tVsVT = cute.filter_zeros(tVsVT)
                tVgVT = cute.filter_zeros(tVgVT)

            q_phase_eff = (((q_tile_count >> 1) & 1) ^ 1) if self.q_bufs == 2 else q_producer_phase
            q_slot0 = (q_tile_count & 1) * self.q_tiles if self.q_bufs == 2 else 0
            q_slot1 = q_slot0 + 1
            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_phase_eff,
                load_SFQ_fn=load_SFQ_fn if const_expr(tma_atom_sfq is not None) else None,
                is_leader_cta=is_leader_cta,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
                tma_atom_sf=tma_atom_sfk,
                tXgSF=tKgSFK,
                tXsSF=tKsSFK,
                is_leader_cta=is_leader_cta,
                sf_mcast_mask=sfk_mcast_mask,
            )

            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
                tma_atom_sf=tma_atom_sfv,
                tXgSF=tVgSFV,
                tXsSF=tVsSFV,
                tma_atom_sf2=tma_atom_vt,
                tXgSF2=tVgVT,
                tXsSF2=tVsVT,
                is_leader_cta=is_leader_cta,
                sf_mcast_mask=sfv_mcast_mask,
            )

            if const_expr(self.sf_dense):
                # sfd: Q / K / V by TMA, their SF atoms by this warp from the dense per-row tables (each stage: TMA
                # issue, atom stores, fence, the arrive with the tx bytes). A key tile's table bytes arrive by bulk
                # copies into a staging slot issued SFD_KV_SLOTS - 1 key tiles ahead (the producer never waits on a
                # global load inside the key loop); Q's two tiles' by loads at the work-tile start. The first staging
                # copies are issued after the work tile's first Q / K / V TMAs (their packed key start is a uniform
                # register: its move waits for the load, which before the TMAs held them a global round trip). The
                # code is kept small: this path's extra instructions cost mainly instruction-cache misses of the whole
                # kernel (spfwd: one staging-issue site in the key loop, K's and V's words read after one slot wait,
                # the first key tile's atoms written by the key loop's first step)
                from flash_attn.cute import sfd_atoms as SA
                mSfdF, sKvStg, kvstg_mbar = self._sfd_ld
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                nk4 = self.head_dim_padded // 64
                mtq = mSfdF[0] if const_expr(self.fp8_sfd) else mSfdF[3]
                mtk = mSfdF[1] if const_expr(self.fp8_sfd) else mSfdF[4]
                mEvV = mSfdF[2]
                if const_expr(self.o_rows_tma):
                    # (the pair: this CTA's 128-row tile of each of the work tile's q_tiles 256-row groups)
                    psq0 = mCuSeqlensO[batch_idx] + (self.q_tiles * m_block * self.cta_group_size
                                                     + mma_tile_coord_v) * self.m_block_size
                else:
                    psq0 = mCuSeqlensO[batch_idx] + self.q_tiles * m_block * self.m_block_size
                psk0 = mCuSeqlensKc[batch_idx]
                rq = cute.make_rmem_tensor(cute.make_layout(4 * nk4 * self.q_tiles), Int32)
                if const_expr(self.o_rows_tma):
                    # the pair: this CTA's tile qt starts at psq0 + 256 qt; a tile may start up to 3 tiles past its
                    # document's end (the last document: beyond the dense tables' tail rows) -- no table read there
                    # (its rows are never stored)
                    if const_expr(not self.fp8_sfd):
                        for qt in cutlass.range_constexpr(self.q_tiles):
                            _rq = cute.make_tensor(rq.iterator + 4 * nk4 * qt, cute.make_layout(4 * nk4))
                            for _i in cutlass.range_constexpr(4 * nk4):
                                _rq[_i] = Int32(0)
                            if psq0 + qt * self.m_block_size * self.cta_group_size - mCuSeqlensO[batch_idx] \
                                    < seqlen.seqlen_q:
                                SA.load_sf4(mtq, head_idx, psq0 + qt * self.m_block_size * self.cta_group_size, nk4,
                                            _rq)
                    else:
                        ql = cute.arch.lane_idx() & (4 * self.q_tiles - 1)
                        _pr = psq0 + (ql >> 2) * self.m_block_size * self.cta_group_size
                        rq[0] = Int32(0)
                        if _pr - mCuSeqlensO[batch_idx] < seqlen.seqlen_q:
                            rq[0] = Int32(mtq[head_idx, _pr + 32 * (ql & 3)])
                else:
                    # Q's scale words at the work-tile start, as lane-varying loads (a warp-uniform load's value moves to a
                    # uniform register, and that move would hold the first TMAs a global round trip): NVFP4 S its SF words
                    # (rows lane + 32 j), E4M3 S lane 4 qt + i Q tile qt's block byte i (gathered at the atom)
                    if const_expr(not self.fp8_sfd):
                        for qt in cutlass.range_constexpr(self.q_tiles):
                            SA.load_sf4(mtq, head_idx, psq0 + qt * self.m_block_size, nk4,
                                        cute.make_tensor(rq.iterator + 4 * nk4 * qt, cute.make_layout(4 * nk4)))
                    else:
                        ql = cute.arch.lane_idx() & (4 * self.q_tiles - 1)
                        rq[0] = Int32(mtq[head_idx, psq0 + (ql >> 2) * self.m_block_size + 32 * (ql & 3)])
                sfd_q = (rq, psq0, seqlen.seqlen_q)
                n_kt = n_block_max - n_block_min
                # (the tables' kv-head rows as base pointers, formed once a work tile)
                kvs = (mtk[head_idx_kv, None].iterator, mEvV[head_idx_kv, None].iterator, head_idx_kv, psk0,
                       n_block_max, sKvStg, kvstg_mbar)
                v4s = None
                if const_expr(self.v4_packed):
                    # (cov5) V^T from the key-parity copies: the document's copy c = psk0 & 1, its tile n_block's window
                    # at byte (psk0 + c) / 2 + 64 n_block; the diagonal payload (vtok_diag) on n_block >= n_diag_min
                    n_diag_min = (block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
                                  if const_expr(self.vtok_diag) else n_block_max)
                    c_par = psk0 & 1
                    v4s = (mSfdF[5][c_par, head_idx_kv, 0, None].iterator, mSfdF[7][c_par, head_idx_kv, 0, None].iterator,
                           mSfdF[5].shape[3], (psk0 + c_par) >> 1, n_diag_min, seqlen.seqlen_k,
                           mSfdF[6][head_idx_kv, None].iterator, mSfdF[8][head_idx_kv, None].iterator,
                           seqlen.offset_k // self.n_block_size,
                           (c_par * mSfdF[5].shape[1] + head_idx_kv) * mSfdF[5].shape[2],   # [9] the staging rows
                           *self._v4_tma)                                                      # [10:14] the TMA atoms
                    kvs = kvs + (v4s,)
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    # the first key tile's TMAs here (Q0, K0, Q1, V0 issued before any atom); its atoms in the key
                    # loop's first step, which skips their TMAs
                    st_k0 = kv_producer_state.clone()
                    self._sfd_load_Q(load_Q_fn, mbar_ptr, self.q_tiles * m_block + 0, q_slot0, q_phase_eff, sSFQ,
                                     sfd_q, 0, part=1)
                    self._sfd_load_K(tma_atom_K, tKgK, tKsK, mbar_ptr, n_block_max - 1, st_k0, sSFK, kvs,
                                     seqlen.seqlen_k, kv_n_cons, part=1)
                    st_v0 = st_k0.clone()
                    st_v0.advance()
                    if const_expr(self.q_tiles == 2):
                        self._sfd_load_Q(load_Q_fn, mbar_ptr, self.q_tiles * m_block + 1, q_slot1, q_phase_eff, sSFQ,
                                         sfd_q, 1, part=1)
                    if const_expr(self.v4_packed):
                        # (cov5) the first two key tiles' V windows; V0's stage waited empty here
                        from flash_attn.cute import v4_packed as V4
                        n_v0 = cutlass.min(n_kt, Int32(V4.STAGES))
                        for t0 in cutlass.range(n_v0, unroll=1):
                            self._v4_issue(v4s, n_block_max - 1 - t0, v4_iss + t0)
                        v4_iss += n_v0
                        self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + st_v0.index, st_v0.phase)
                    else:
                        self._sfd_load_V(tma_atom_V, tVgV, tVsV, mbar_ptr, n_block_max - 1, st_v0, sSFV, Int32(0),
                                         part=1)
                    # (after the TMAs) the first SFD_KV_SLOTS - 1 key tiles' staging copies
                    n_st0 = cutlass.min(n_kt, Int32(SFD_KV_SLOTS - 1))
                    for t0 in cutlass.range(n_st0, unroll=1):
                        self._sfd_kv_issue(kvs, t0, kv_n_issue + t0)
                    kv_n_issue += n_st0
                    self._sfd_load_Q(load_Q_fn, mbar_ptr, self.q_tiles * m_block + 0, q_slot0, q_phase_eff, sSFQ,
                                     sfd_q, 0, part=2)
                    if const_expr(self.q_tiles == 2):
                        self._sfd_load_Q(load_Q_fn, mbar_ptr, self.q_tiles * m_block + 1, q_slot1, q_phase_eff, sSFQ,
                                         sfd_q, 1, part=2)
                    q_producer_phase ^= 1
                    q_tile_count += 1
                    for t in cutlass.range(n_kt, unroll=1):
                        n_block = n_block_max - 1 - t
                        if t > 0:
                            if const_expr(TileSchedulerProd is not None and self.varlen_sched_early):
                                if n_block == n_block_min:
                                    tile_scheduler.prefetch_next_work()
                            # key tile t + SFD_KV_SLOTS - 2's staging copy (its slot's previous tile is consumed)
                            if t + (SFD_KV_SLOTS - 2) < n_kt:
                                self._sfd_kv_issue(kvs, t + (SFD_KV_SLOTS - 2), kv_n_issue)
                                kv_n_issue += 1
                        wv = self._sfd_load_K(tma_atom_K, tKgK, tKsK, mbar_ptr, n_block, kv_producer_state, sSFK,
                                              kvs, seqlen.seqlen_k, kv_n_cons, do_tma=t > 0)
                        kv_producer_state.advance()
                        if const_expr(self.v4_packed):
                            self._v4_load_V(mbar_ptr, n_block, kv_producer_state, sSFV, sV, kvs, kv_n_cons, v4_con,
                                            t > 0)
                            v4_con += 1
                            if t + V4.STAGES < n_kt:
                                self._v4_issue(v4s, n_block - V4.STAGES, v4_iss)
                                v4_iss += 1
                        else:
                            self._sfd_load_V(tma_atom_V, tVgV, tVsV, mbar_ptr, n_block, kv_producer_state, sSFV, wv,
                                             do_tma=t > 0)
                        kv_producer_state.advance()
                        kv_n_cons += 1
            elif const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                # diagonal-only-key-local-v-scale: the masked steps (n_block >= n_diag_min, the tiles the causal boundary
                # cuts) read V / SFV from the second payload, which starts n_v_half key tiles in. Off, both are 0 and
                # unused, so the default kernel's V tile index is the plain n_block.
                n_diag_min = (block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
                              if self.vtok_diag else Int32(0))
                # (aligned documents: the second payload follows ALL keys of the allocation; V is domain-offset by the
                # document's start, so the tile offset is the allocation's tile count)
                n_v_half = (cute.ceil_div(mK.shape[0] if const_expr(seqlen.has_cu_seqlens_k) else seqlen.seqlen_k,
                                          self.n_block_size) if self.vtok_diag else Int32(0))
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_tiles * m_block + 0, stage=q_slot0)  # Q0 + SFQ0
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    if const_expr(self.vtok_w and not self.vtok_tma):
                        self._mbar_wait_h(mbar_ptr + self.mbar_vtok_offset, vt_ph)
                        vt_ph ^= 1
                    load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0 + SFK0
                    kv_producer_state.advance()
                    if const_expr(self.q_tiles == 2) and (const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE):
                        load_Q(block=self.q_tiles * m_block + 1, stage=q_slot1)  # Q1 + SFQ1
                    q_producer_phase ^= 1
                    q_tile_count += 1
                    v_block_first = (n_block_max - 1 + n_v_half * Int32(n_block_max - 1 >= n_diag_min)
                                     if self.vtok_diag else n_block_max - 1)
                    load_V(block=v_block_first, producer_state=kv_producer_state, page_idx=page_idx)  # V0 + SFV0
                    kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        if const_expr(TileSchedulerProd is not None and self.varlen_sched_early):
                            # early jit claim: before the last K / V pair's loads (their slot waits hide its latency)
                            if n_block == n_block_min:
                                tile_scheduler.prefetch_next_work()
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        if const_expr(self.vtok_w and not self.vtok_tma):
                            self._mbar_wait_h(mbar_ptr + self.mbar_vtok_offset, vt_ph)
                            vt_ph ^= 1
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki + SFKi
                        kv_producer_state.advance()
                        v_block = (n_block + n_v_half * Int32(n_block >= n_diag_min)
                                   if self.vtok_diag else n_block)
                        load_V(block=v_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi + SFVi
                        kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )


            if const_expr(TileSchedulerProd is not None and self.varlen_sched_early):
                # early jit claim: made inside the K / V loop when the tile has >= 2 key blocks
                if n_block_max - n_block_min < 2:
                    tile_scheduler.prefetch_next_work()
            else:
                tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop
        if const_expr(TileSchedulerProd is not None):
            tile_scheduler.producer_tail()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.atom.ThrMma,
        tiled_mma_pv: cute.atom.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        # In smem
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        # In tmem - per-stage scale factors
        tCtSFQs: Tuple[cute.Tensor, ...],
        tCtSFKs: Tuple[cute.Tensor, ...],
        tCtSFPs: Tuple[cute.Tensor, ...],
        tCtSFVs: Tuple[cute.Tensor, ...],
        tCtSF_split=None,
        is_leader_cta=True,  # two-cta-forward: only the leader CTA issues (its MMAs cover both CTAs' rows)
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        if const_expr(self.quant_qk):
            # For MXFP8 QK the generic cute.gemm path is slightly faster than
            # our inline-PTX helper (+1-5% across shapes, +4.5% on
            # (1,32768,24,128)) because the helper's per-K SF-address
            # plumbing adds register pressure. Use generic by default for
            # block-scaled FP8; keep inline-PTX for NVFP4 where it wins.
            if const_expr(
                qk_mma_op.a_dtype in (Float8E4M3FN, Float8E5M2)
                and not self.debug_mxfp8_use_inline_ptx
            ):
                gemm_Si = [
                    partial(
                        sm100_utils.gemm_blockscaled_generic,
                        tiled_mma_qk,
                        tStSs[stage],
                        tSrQs[stage],
                        tScaleA=tCtSFQs[stage],
                        tScaleB=tCtSFKs[stage],
                        zero_init=True,
                    )
                    for stage in range(self.q_stage)
                ]
            else:
                gemm_Si = [
                    partial(
                        sm100_utils.gemm_ptx_partial_fp4,
                        qk_mma_op,
                        self.tmem_s_offset[stage],
                        tSrQs[stage],
                        sA=sQ[None, None, None, stage],
                        zero_init=True,
                        tScaleA=tCtSFQs[stage],
                        tScaleB=tCtSFKs[stage],
                    )
                    for stage in range(self.q_stage)
                ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    tSrQs[stage],
                    sA=sQ[None, None, None, stage],
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial_fp4,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                tScaleA=tCtSFPs[stage],
                tScaleB=tCtSFVs[stage],
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
                tA_addr=self.tmem_p_offset[stage],
            ) if const_expr(self.quant_pv) else
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        # Partition for S2T copy of SFQ/SFK - change addr per q_stage to avoid overwriting
        if const_expr(self.quant_qk):
            tiled_copy_s2t_sfq_staged = [
                self.mainloop_s2t_copy_and_partition(sSFQ, tCtSFQs[stage])
                for stage in range(self.q_stage)
            ] 
            tiled_copy_s2t_sfk_staged = [
                self.mainloop_s2t_copy_and_partition(sSFK, tCtSFKs[stage])
                for stage in range(self.q_stage)
            ] 
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, _ = tiled_copy_s2t_sfq_staged[0]
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, _ = tiled_copy_s2t_sfk_staged[0]
        else:
            # Dummy values when quant_qk is False - these won't be used
            tiled_copy_s2t_sfq_staged = []
            tiled_copy_s2t_sfk_staged = []
            tiled_copy_s2t_sfq = None
            tCsSFQ_compact_s2t = None
            tiled_copy_s2t_sfk = None
            tCsSFK_compact_s2t = None

        if const_expr(self.quant_pv):
            tiled_copy_s2t_sfv_staged = [
                self.mainloop_s2t_copy_and_partition(sSFV, tCtSFVs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, _ = tiled_copy_s2t_sfv_staged[0]
            # S2T copy setup for SFP (P scale factors, computed by softmax warp via R2S)
            tiled_copy_s2t_sfp_staged = [
                self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, _ = tiled_copy_s2t_sfp_staged[0]
        else:
            tiled_copy_s2t_sfv_staged = []
            tiled_copy_s2t_sfv = None
            tCsSFV_compact_s2t = None
            tiled_copy_s2t_sfp_staged = []
            tiled_copy_s2t_sfp = None
            tCsSFP_compact_s2t = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        
        
        mma_sfqk_producer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase
                    )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    
                    # Copy SFQ 
                    # only tmem changes per q_stage.
                    if const_expr(self.quant_qk):
                        sm100_utils.tcgen05_after_thread_sync()
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                        _, _, tCtSFQ_compact_s2t = tiled_copy_s2t_sfq_staged[stage]
                        tCsSFQ_compact_s2t_staged = tCsSFQ_compact_s2t[None, None, None, None, stage]
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_compact_s2t_staged,
                            tCtSFQ_compact_s2t,
                        )

                        _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                        tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_compact_s2t_staged,
                            tCtSFK_compact_s2t,
                        )

                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(
                            sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        )

                    gemm_Si[stage](
                        tCrB=tSrKi,  # tCrB
                        sB=sK_cur,  # sB
                    )

                    # 4. release S0 / S1
                    self._umma_arrive(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                if const_expr(self.quant_qk):
                    mma_sfqk_producer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        # P (and SFP in direct mode) were written by the softmax
                        # warps' tcgen05.st; the PTX model requires
                        # fence::after_thread_sync on the consumer side before
                        # the tcgen05 ops that read them (the embedded P_full_2
                        # wait in the PV GEMM already has one).
                        sm100_utils.tcgen05_after_thread_sync()
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        
                        # No need for mbar.wait because it depends on the same Si as the prev qk mma
                        if const_expr(self.quant_pv):
                            # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                            # (skipped when the softmax warps store SFP directly)
                            _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                            tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t_cur,
                                tCtSFP_compact_s2t,
                            )
                            # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                            _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                            tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_compact_s2t_staged,
                                tCtSFV_compact_s2t,
                            )

                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)

                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )

                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1) after the last stage's PV GEMM
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)

                        if const_expr(self.quant_qk):
                            _, _, tCtSFQ_compact_s2t = tiled_copy_s2t_sfq_staged[stage]
                            tCsSFQ_compact_s2t_staged = tCsSFQ_compact_s2t[None, None, None, None, stage]
                            sm100_utils.tcgen05_after_thread_sync()
                            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                            cute.copy(
                                tiled_copy_s2t_sfq,
                                tCsSFQ_compact_s2t_staged,
                                tCtSFQ_compact_s2t,
                            )
                            _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                            tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t_staged,
                                tCtSFK_compact_s2t,
                            )

                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)

                        gemm_Si[stage](
                            tCrB=tSrK[None, None, None, Ki_index],  # tCrB
                            sB=sK_cur,  # sB
                        )
                        # 3. release S0
                        self._umma_arrive(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    if const_expr(self.quant_qk):
                        mma_sfqk_producer_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range_constexpr(self.q_stage):
                    self._umma_arrive(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase
                    )
                    # See the matching fence above: order the PV MMA's TMEM
                    # reads of P/SFP after the softmax warps' tcgen05.st.
                    sm100_utils.tcgen05_after_thread_sync()
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)

                    if const_expr(self.quant_pv):
                        # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                        # (skipped when the softmax warps store SFP directly)
                        _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                        tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                        cute.copy(
                            tiled_copy_s2t_sfp,
                            tCsSFP_compact_s2t_cur,
                            tCtSFP_compact_s2t,
                        )
                        # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                        _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                        tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                        cute.copy(
                            tiled_copy_s2t_sfv,
                            tCsSFV_compact_s2t_staged,
                            tCtSFV_compact_s2t,
                        )
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    _zi_post = not O_should_accumulate
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=_zi_post,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    self._umma_arrive(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    def _pair_stage_skip(self, block_info_wg, seqlen, m_block, stage, split_idx, num_splits, n_block_max):
        """two-cta-forward: the KV tiles at the front of the pair's range that lie wholly above the causal diagonal of Q
        stage `stage`'s 256-row group (the pair's range minus the group's own, kept >= 1 step). 0 off causal."""
        if const_expr(not self.is_causal):
            return Int32(0)
        nmax_wg = block_info_wg.get_n_block_min_max(seqlen, self.q_stage * m_block + stage, split_idx, num_splits)[1]
        return cutlass.max(cutlass.min(n_block_max - nmax_wg, n_block_max - 1), Int32(0))

    @cute.jit
    def mma_pair(
        self,
        tiled_mma_qk: cute.atom.ThrMma,
        tiled_mma_pv: cute.atom.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        tCtSFQs: Tuple[cute.Tensor, ...],
        tCtSFKs: Tuple[cute.Tensor, ...],
        tCtSFPs: Tuple[cute.Tensor, ...],
        tCtSFVs: Tuple[cute.Tensor, ...],
        tCtSF_split=None,
        is_leader_cta=True,
        block_info_wg: Optional[BlockInfo] = None,
    ):
        """two-cta-forward, causal, plain-E4M3 S GEMM (no SFQ / SFK): the generic MMA program (mma) with Q stage 0
        starting z0 KV tiles late -- the tiles above its group's diagonal get no QK(0) / PV(0) at all (the softmax and
        correction warps of stage 0 skip them too), so the pair runs exactly the 1-CTA kernel's GEMMs for each 256-row
        group. Stage 1's group ends the pair's range (z1 = 0). Per-stage P-full phases and O-accumulate flags."""
        assert self.stage_skip and self.q_stage == 2
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        if const_expr(not self.quant_qk):
            gemm_Si = [
                partial(sm100_utils.gemm_ptx_partial, qk_mma_op, self.tmem_s_offset[stage], tSrQs[stage],
                        sA=sQ[None, None, None, stage], zero_init=True, cta_group=self.cta_group_size)
                for stage in range(2)
            ]
            s2t_sfq = s2t_sfk = None
        else:
            # block-scaled S GEMM (as mma): MXFP8 through the generic cute.gemm, NVFP4 through the inline PTX
            if const_expr(qk_mma_op.a_dtype in (Float8E4M3FN, Float8E5M2)):
                gemm_Si = [
                    partial(sm100_utils.gemm_blockscaled_generic, tiled_mma_qk, tStSs[stage], tSrQs[stage],
                            tScaleA=tCtSFQs[stage], tScaleB=tCtSFKs[stage], zero_init=True)
                    for stage in range(2)
                ]
            else:
                gemm_Si = [
                    partial(sm100_utils.gemm_ptx_partial_fp4, qk_mma_op, self.tmem_s_offset[stage], tSrQs[stage],
                            sA=sQ[None, None, None, stage], zero_init=True, tScaleA=tCtSFQs[stage],
                            tScaleB=tCtSFKs[stage])
                    for stage in range(2)
                ]
            s2t_sfq = [self.mainloop_s2t_copy_and_partition(sSFQ, tCtSFQs[stage]) for stage in range(2)]
            s2t_sfk = [self.mainloop_s2t_copy_and_partition(sSFK, tCtSFKs[stage]) for stage in range(2)]
        gemm_Pi = [
            partial(sm100_utils.gemm_ptx_partial_fp4, pv_mma_op, self.tmem_o_offset[stage], tOrPs[stage], sA=None,
                    tScaleA=tCtSFPs[stage], tScaleB=tCtSFVs[stage],
                    pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])), tA_addr=self.tmem_p_offset[stage])
            if const_expr(self.quant_pv) else
            partial(sm100_utils.gemm_ptx_partial, pv_mma_op, self.tmem_o_offset[stage], tOrPs[stage], sA=None,
                    pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])), cta_group=self.cta_group_size)
            for stage in range(2)
        ]
        if const_expr(self.quant_pv):
            s2t_sfv = [self.mainloop_s2t_copy_and_partition(sSFV, tCtSFVs[stage]) for stage in range(2)]
            s2t_sfp = [self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[stage]) for stage in range(2)]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        p_phase0 = Int32(0)
        p_phase1 = Int32(0)
        # block-scaled S GEMM: QK(s)'s scale factors go into the OTHER stage's S bank once that stage's softmax has loaded
        # its S (sfqk_load[s]). [0] is arrived by stage 1 every step (and once at kernel start), so its phase advances
        # every iteration whether or not QK(0) is issued; [1] is arrived by stage 0 only for the steps it runs -- while
        # stage 0 skips, S bank 0 is idle and QK(1) needs no wait.
        sf_ph0 = Int32(0)
        sf_ph1 = Int32(0)

        if const_expr(not self.quant_pv):
            s2t_sfp = s2t_sfv = None
        pv = partial(self._pair_pv, gemm_Pi=gemm_Pi, s2t_sfp=s2t_sfp, s2t_sfv=s2t_sfv, tOrV=tOrV, sV=sV, mbar_ptr=mbar_ptr)
        qk = partial(self._pair_qk, gemm_Si=gemm_Si, tSrK=tSrK, sK=sK, mbar_ptr=mbar_ptr, s2t_sfq=s2t_sfq,
                     s2t_sfk=s2t_sfk)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            block_iter_count = n_block_max - n_block_min
            z0 = self._pair_stage_skip(block_info_wg, seqlen, m_block, 0, split_idx, num_splits, n_block_max)
            if is_leader_cta:
                # QK of the first KV tile: stage 0 only if its group needs that tile
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + 0, mma_q_consumer_phase)
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                # sfqk_load[0]'s phase is waited even when QK(0) of this tile is skipped (z0 > 0): the next wait (before
                # QK(0, z0), parity sf_ph0 ^ 1) cannot tell "phase 0 not yet completed" from "phase 1 completed", so
                # flipping without waiting let QK(0, z0)'s SF copy into S bank 1 pass while a stage-1 softmax warp had not
                # made its kernel-start pre-arrive -- and so had not yet loaded S(1, 0), which the copy then overwrote
                # (stage-1 rows of a CTA's first tile; ~0.4 % of launches on the persistent grid, 2.6 % one CTA per tile)
                if const_expr(self.quant_qk):
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + 0, sf_ph0)
                if z0 == 0:
                    qk(0, mma_kv_consumer_state.index)
                if const_expr(self.quant_qk):
                    sf_ph0 ^= 1
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + 1, mma_q_consumer_phase)
                if const_expr(self.quant_qk):
                    if z0 == 0:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + 1, sf_ph1)
                        sf_ph1 ^= 1
                qk(1, mma_kv_consumer_state.index)
                mma_q_consumer_phase ^= 1
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                O_acc0 = False
                O_acc1 = False
                for i in cutlass.range(block_iter_count - 1, unroll=1):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index = mma_kv_consumer_state.index
                    if i >= z0:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, p_phase0)
                        pv(0, Vi_index, not O_acc0, p_phase0)
                        p_phase0 ^= 1
                        O_acc0 = True
                    mma_kv_consumer_state.advance()
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index = mma_kv_consumer_state.index
                    if i + 1 >= z0:
                        if const_expr(self.quant_qk):
                            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + 0, sf_ph0)
                        qk(0, Ki_index)
                    if const_expr(self.quant_qk):
                        sf_ph0 ^= 1
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, p_phase1)
                    pv(1, Vi_index, not O_acc1, p_phase1)
                    p_phase1 ^= 1
                    O_acc1 = True
                    pipeline_kv.consumer_release(mma_kv_release_state)
                    if const_expr(self.quant_qk):
                        if i + 1 >= z0:  # QK(0, i+1) was issued into S bank 0: wait for its load
                            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + 1, sf_ph1)
                            sf_ph1 ^= 1
                    qk(1, Ki_index)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                for stage in cutlass.range_constexpr(2):
                    self._umma_arrive(mbar_ptr + self.mbar_load_q_empty_offset + stage)
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index = mma_kv_consumer_state.index
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, p_phase0)
                pv(0, Vi_index, not O_acc0, p_phase0)
                p_phase0 ^= 1
                self._umma_arrive(mbar_ptr + self.mbar_O_full_offset + 0)
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, p_phase1)
                pv(1, Vi_index, not O_acc1, p_phase1)
                p_phase1 ^= 1
                self._umma_arrive(mbar_ptr + self.mbar_O_full_offset + 1)
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def _pair_pv(self, stage, Vi_index, zero_init, phase, gemm_Pi, s2t_sfp, s2t_sfv, tOrV, sV, mbar_ptr):
        """mma_pair: PV(stage) of KV tile Vi_index; P(stage) (and its SF tile) were written by both CTAs' softmax warps."""
        sm100_utils.tcgen05_after_thread_sync()
        if const_expr(self.quant_pv):
            cute.copy(s2t_sfp[stage][0], s2t_sfp[stage][1][None, None, None, None, stage], s2t_sfp[stage][2])
            cute.copy(s2t_sfv[stage][0], s2t_sfv[stage][1][None, None, None, None, Vi_index], s2t_sfv[stage][2])
        gemm_Pi[stage](tCrB=tOrV[None, None, None, Vi_index], sB=sV[None, None, None, Vi_index], zero_init=zero_init,
                       mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase=phase)

    def _pair_qk(self, stage, Ki_index, gemm_Si, tSrK, sK, mbar_ptr, s2t_sfq=None, s2t_sfk=None):
        """mma_pair: QK(stage) of KV tile Ki_index (after its scale-factor copies, block-scaled S GEMM), S full multicast
        to both CTAs."""
        if const_expr(self.quant_qk):
            sm100_utils.tcgen05_after_thread_sync()
            cute.copy(s2t_sfq[stage][0], s2t_sfq[stage][1][None, None, None, None, stage], s2t_sfq[stage][2])
            cute.copy(s2t_sfk[stage][0], s2t_sfk[stage][1][None, None, None, None, Ki_index], s2t_sfk[stage][2])
        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK[None, None, None, Ki_index])
        self._umma_arrive(mbar_ptr + self.mbar_S_full_offset + stage)

    def _arrive_mma(self, mbar: cute.Pointer):
        """Arrive on a barrier the MMA warp waits on (P full, O rescaled, S read for the SF copies). two-cta-forward:
        the leader CTA's MMA warp issues for both CTAs, so both CTAs' threads arrive on the LEADER's barrier (rank 0 of
        the pair; its count is doubled)."""
        if const_expr(self.use_2cta_forward):
            # one remote arrive per warp (after the warp's own tcgen05.wait::st / SMEM fences): 32x fewer DSMEM arrives
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mbar, peer_cta_rank_in_cluster=Int32(0))
        else:
            cute.arch.mbarrier_arrive(mbar)

    def _umma_arrive(self, mbar: cute.Pointer):
        """tcgen05.commit of the MMA warp's issued ops onto `mbar`: two-cta-forward multicasts it to the same barrier
        of both CTAs of the pair (cta_group::2), the 1-CTA kernel commits locally."""
        with cute.arch.elect_one():
            if const_expr(self.use_2cta_forward):
                tcgen05.commit(mbar, (1 << self.cta_group_size) - 1, tcgen05.CtaGroup.TWO)
            else:
                tcgen05.commit(mbar)

    # =====================================================================
    #  Split-QK cross-tile pipeline (qk_split_pipe=1)
    # =====================================================================
    @cute.jit
    def _mbar_wait_h(self, mbar_ptr: cute.Pointer, phase: Int32):
        """mbarrier wait (the split pipeline's waits)."""
        cute.arch.mbarrier_wait(mbar_ptr, phase)

    @cute.jit
    def mma_split(
        self,
        tiled_mma_qk: cute.atom.ThrMma,
        tiled_mma_pv: cute.atom.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        tCtSFQs: Tuple[cute.Tensor, ...],
        tCtSFKs: Tuple[cute.Tensor, ...],
        tCtSFPs: Tuple[cute.Tensor, ...],
        tCtSFVs: Tuple[cute.Tensor, ...],
        tCtSF_split=None,
    ):
        """Per KV step t and stage s, in program order: QK_a(s,t+1) [after softmax(s)
        loaded S(s,t)], PV(s,t) [after P(s,t)], QK_b(s,t+1) [after softmax(s) loaded
        S_a(s,t+1)]. The a-half of S(t+1) is therefore ready while the softmax still
        works on tile t; the b-half follows the usual turnaround, hidden by the
        softmax's a-half work of t+1."""
        assert self.q_stage == 2 and self.quant_qk == self.quant_pv and self.n_block_size == 128
        assert not self.use_block_sparsity and not self.uneven_kv_smem and not self.is_split_kv
        assert self.kv_stage >= 3
        # Block-scaled operands only (NVFP4 / MXFP4 Q, K, P^, V with SF s2t copies before every GEMM).
        assert self.quant_qk and self.quant_pv, "mma_split: block-scaled Q / K / P^ / V only"
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        pv_mma_op = tiled_mma_pv.op
        sfq_a, sfk_a, sfq_b, sfk_b = tCtSF_split
        tiled_mma_qk64 = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.q_dtype, self.q_major_mode, self.k_major_mode, self.sf_dtype, self.sf_vec_size,
            self.cta_group, (self.m_block_size, 64),
        )
        _sf_hi_off = 2 * (32 // self.sf_dtype.width)   # SF-B rows 64-127: +2 columns (same lanes)
        qk64_op = tiled_mma_qk64.op
        _k_half_elems = 64 * self.head_dim_padded       # K rows 64-127 in smem (a whole SW128 atom at E4M3 too)
        gemm_P = []
        for st in cutlass.range_constexpr(2):
            gemm_P.append(partial(
                sm100_utils.gemm_ptx_partial_fp4, pv_mma_op, self.tmem_o_offset[st], tOrPs[st],
                sA=None, tScaleA=tCtSFPs[st], tScaleB=tCtSFVs[st],
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[st].shape[2])),
                tA_addr=self.tmem_p_offset[st],
            ))
        # s2t copies: (tiled_copy, smem partition, tmem partition) per target
        cp_qa = [self.mainloop_s2t_copy_and_partition(sSFQ, sfq_a[st]) for st in range(2)]
        cp_ka = [self.mainloop_s2t_copy_and_partition(sSFK, sfk_a[st]) for st in range(2)]
        cp_qb = [self.mainloop_s2t_copy_and_partition(sSFQ, sfq_b[st]) for st in range(2)]
        cp_kb = [self.mainloop_s2t_copy_and_partition(sSFK, sfk_b[st]) for st in range(2)]
        cp_p = [self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[st]) for st in range(2)]
        cp_v = [self.mainloop_s2t_copy_and_partition(sSFV, tCtSFVs[st]) for st in range(2)]
        tc_sfq, tCs_sfq, _ = cp_qa[0]
        tc_sfk, tCs_sfk, _ = cp_ka[0]
        tc_sfp, tCs_sfp, _ = cp_p[0]
        tc_sfv, tCs_sfv, _ = cp_v[0]
        tCt_qa0, tCt_qa1 = cp_qa[0][2], cp_qa[1][2]
        tCt_ka0, tCt_ka1 = cp_ka[0][2], cp_ka[1][2]
        tCt_qb0, tCt_qb1 = cp_qb[0][2], cp_qb[1][2]
        tCt_kb0, tCt_kb1 = cp_kb[0][2], cp_kb[1][2]
        tCt_p0, tCt_p1 = cp_p[0][2], cp_p[1][2]
        tCt_v0, tCt_v1 = cp_v[0][2], cp_v[1][2]

        # SF s2t copies as partial objects (a Python closure inside the staged loop is rejected by
        # the DSL: SCOPE_CLOSURE_CAPTURE)
        _sf_sfq = partial(self._s2t_or_skip, tc_sfq, tCs_sfq)
        _sf_sfk = partial(self._s2t_or_skip, tc_sfk, tCs_sfk)
        _sf_sfp = partial(self._s2t_or_skip, tc_sfp, tCs_sfp)
        _sf_sfv = partial(self._s2t_or_skip, tc_sfv, tCs_sfv)

        KVS = self.kv_stage

        q_ph = Int32(0)
        qcount = Int32(0)      # E40: Q buffer = qcount & 1 (q_bufs == 2)
        pf = Int32(0)          # P_full_O_rescaled parity (both stages, per tile)
        sl0 = Int32(0)         # sfqk_load[s] = softmax(s) loaded S(s,t) fully (pre-arrived once)
        sl1 = Int32(0)
        sa0 = Int32(0)         # sa_loaded[s] = softmax(s) loaded the a-half of the current tile
        sa1 = Int32(0)
        kidx = Int32(0)        # K(t) ring position (index, phase); V(t) = K(t) + 1
        kph = Int32(0)
        vidx = Int32(1)
        vph = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            n = n_block_max - n_block_min
            # E40: Q smem slots of this tile (double-buffered across tiles when q_bufs == 2)
            qs0 = (qcount & 1) * 2 if self.q_bufs == 2 else 0
            qs1 = qs0 + 1
            if const_expr(self.q_bufs == 2):
                q_ph = (qcount >> 1) & 1
            gemm_a = []
            gemm_b = []
            for st in cutlass.range_constexpr(2):
                _qs = (qs0, qs1)[st]
                gemm_a.append(partial(
                    sm100_utils.gemm_ptx_partial_fp4, qk64_op, self.tmem_s_offset[st],
                    tSrQ[None, None, None, _qs], sA=sQ[None, None, None, _qs], zero_init=True,
                    tScaleA=sfq_a[st], tScaleB=sfk_a[st],
                ))
                gemm_b.append(partial(
                    sm100_utils.gemm_ptx_partial_fp4, qk64_op, self.tmem_s_offset[st] + 64,
                    tSrQ[None, None, None, _qs], sA=sQ[None, None, None, _qs], zero_init=True,
                    tScaleA=sfq_b[st],
                    tScaleB=cute.make_tensor(sfk_b[st].iterator + _sf_hi_off, sfk_b[st].layout),
                ))
            self._mbar_wait_h(mbar_ptr + self.mbar_load_q_full_offset + qs0, q_ph)
            self._mbar_wait_h(mbar_ptr + self.mbar_load_q_full_offset + qs1, q_ph)
            if const_expr(self.q_bufs == 1):
                q_ph ^= 1
            # ---- prologue: QK_a(s,0) for both stages, then QK_b(s,0) after the a-half loads
            self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + kidx, kph)
            sm100_utils.tcgen05_after_thread_sync()
            self._mbar_wait_h(mbar_ptr + self.mbar_sfqk_load_offset + 0, sl0)
            sl0 ^= 1
            _sf_sfq(qs0, tCt_qa0)
            _sf_sfk(kidx, tCt_ka0)
            gemm_a[0](tCrB=tSrK[None, None, None, kidx], sB=sK[None, None, None, kidx])
            if const_expr(self.vtok_tma):  # V(t) (carrying the vtok bytes) is staged before softmax(t) starts
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + vidx, vph)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_sa_full_offset + 0)
            self._mbar_wait_h(mbar_ptr + self.mbar_sfqk_load_offset + 1, sl1)
            sl1 ^= 1
            _sf_sfq(qs1, tCt_qa1)
            _sf_sfk(kidx, tCt_ka1)
            gemm_a[1](tCrB=tSrK[None, None, None, kidx], sB=sK[None, None, None, kidx])
            if const_expr(self.vtok_tma):  # V(t) (carrying the vtok bytes) is staged before softmax(t) starts
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + vidx, vph)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_sa_full_offset + 1)
            sK_hi = cute.make_tensor(sK[None, None, None, kidx].iterator + _k_half_elems, sK[None, None, None, kidx].layout)
            self._mbar_wait_h(mbar_ptr + self.mbar_sa_loaded_offset + 0, sa0)
            sa0 ^= 1
            sm100_utils.tcgen05_after_thread_sync()
            _sf_sfq(qs0, tCt_qb0)
            _sf_sfk(kidx, tCt_kb0)
            gemm_b[0](tCrB=tSrK[None, None, None, kidx], sB=sK_hi)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 0)
            self._mbar_wait_h(mbar_ptr + self.mbar_sa_loaded_offset + 1, sa1)
            sa1 ^= 1
            sm100_utils.tcgen05_after_thread_sync()
            _sf_sfq(qs1, tCt_qb1)
            _sf_sfk(kidx, tCt_kb1)
            gemm_b[1](tCrB=tSrK[None, None, None, kidx], sB=sK_hi)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 1)
                tcgen05.commit(mbar_ptr + self.mbar_load_kv_empty_offset + kidx)  # K(0) consumed
            kidx += 2
            if kidx >= KVS:
                kidx -= KVS
                kph ^= 1
            # ---- steady state
            for t in cutlass.range(n, unroll=1):
                has_next = t + 1 < n
                # stage 0
                if has_next:
                    self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + kidx, kph)
                    self._mbar_wait_h(mbar_ptr + self.mbar_sfqk_load_offset + 0, sl0)
                    sl0 ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    _sf_sfq(qs0, tCt_qa0)
                    _sf_sfk(kidx, tCt_ka0)
                    gemm_a[0](tCrB=tSrK[None, None, None, kidx], sB=sK[None, None, None, kidx])
                    if const_expr(self.vtok_tma):  # V(t+1) (carrying the vtok bytes) is staged before softmax(t+1) starts
                        self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + (vidx + 2) % KVS, vph ^ Int32((vidx + 2) >= KVS))
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_sa_full_offset + 0)
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + vidx, vph)
                self._mbar_wait_h(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, pf)
                sm100_utils.tcgen05_after_thread_sync()
                _sf_sfp(0, tCt_p0)
                _sf_sfv(vidx, tCt_v0)
                gemm_P[0](
                    tCrB=tOrV[None, None, None, vidx], sB=sV[None, None, None, vidx],
                    zero_init=t == 0, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 0, mbar_phase=pf,
                )
                if has_next:
                    self._mbar_wait_h(mbar_ptr + self.mbar_sa_loaded_offset + 0, sa0)
                    sa0 ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    _sf_sfq(qs0, tCt_qb0)
                    _sf_sfk(kidx, tCt_kb0)
                    gemm_b[0](
                        tCrB=tSrK[None, None, None, kidx],
                        sB=cute.make_tensor(sK[None, None, None, kidx].iterator + _k_half_elems, sK[None, None, None, kidx].layout),
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 0)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + 0)
                # stage 1
                if has_next:
                    self._mbar_wait_h(mbar_ptr + self.mbar_sfqk_load_offset + 1, sl1)
                    sl1 ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    _sf_sfq(qs1, tCt_qa1)
                    _sf_sfk(kidx, tCt_ka1)
                    gemm_a[1](tCrB=tSrK[None, None, None, kidx], sB=sK[None, None, None, kidx])
                    if const_expr(self.vtok_tma):  # V(t+1) (carrying the vtok bytes) is staged before softmax(t+1) starts
                        self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_full_offset + (vidx + 2) % KVS, vph ^ Int32((vidx + 2) >= KVS))
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_sa_full_offset + 1)
                self._mbar_wait_h(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, pf)
                sm100_utils.tcgen05_after_thread_sync()
                _sf_sfp(1, tCt_p1)
                _sf_sfv(vidx, tCt_v1)
                gemm_P[1](
                    tCrB=tOrV[None, None, None, vidx], sB=sV[None, None, None, vidx],
                    zero_init=t == 0, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 1, mbar_phase=pf,
                )
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_load_kv_empty_offset + vidx)  # V(t) consumed
                if has_next:
                    self._mbar_wait_h(mbar_ptr + self.mbar_sa_loaded_offset + 1, sa1)
                    sa1 ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    _sf_sfq(qs1, tCt_qb1)
                    _sf_sfk(kidx, tCt_kb1)
                    gemm_b[1](
                        tCrB=tSrK[None, None, None, kidx],
                        sB=cute.make_tensor(sK[None, None, None, kidx].iterator + _k_half_elems, sK[None, None, None, kidx].layout),
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 1)
                        tcgen05.commit(mbar_ptr + self.mbar_load_kv_empty_offset + kidx)  # K(t+1) consumed
                    kidx += 2
                    if kidx >= KVS:
                        kidx -= KVS
                        kph ^= 1
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + 1)
                pf ^= 1
                vidx += 2
                if vidx >= KVS:
                    vidx -= KVS
                    vph ^= 1
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + qs0)
                tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + qs1)
            qcount += 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _doc_score(self, batch_idx: Int32, head_idx: Int32) -> Float32:
        """score_scale_doc with a score_mod: this work tile's score scale softmax_scale s_q s_k (natural units)."""
        kvh = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        return self._score_tab[batch_idx, kvh]

    @cute.jit
    def _doc_score_log2(self, batch_idx: Int32, head_idx: Int32) -> Float32:
        """score_scale_doc: this work tile's log2 score scale, softmax_scale s_q s_k of its (document, kv head) times
        log2 e -- one fp32 multiply, as the per-kernel scalar is derived."""
        kvh = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        return self._score_tab[batch_idx, kvh] * math.log2(math.e)

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.atom.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tCtSFP: Optional[Tuple[cute.Tensor, ...]] = None,
        sSFP: Optional[cute.Tensor] = None,
        block_info_wg: Optional[BlockInfo] = None,  # two-cta-forward: this warpgroup's 256-row group's KV range
        mma_tile_coord_v: Optional[Int32] = None,   # two-cta-forward: this CTA's index in the pair
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tStScale = cute.composition(
            cute.make_tensor(tStSi.iterator + self.tmem_vec_rel, tStSi.layout), cute.make_layout((self.m_block_size, 1))
        )
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        # Build the TMEM copy from the rank-2 accumulator view, as upstream's forward does
        # (`tSAcc = tStS[(None, None), 0, 0, stage]`). Keeping the rank-3 layout here makes
        # make_tmem_copy produce a rank-3 tiler, and upstream's apply_mask_sm100 slices its
        # coordinate tensor down to rank 2 before partitioning with it -- which cutlass-dsl >= 4.6
        # rejects ("expects rank(input) >= rank(tiler)"). Older DSLs tolerated the mismatch.
        tStSi_r2 = tStSi[(None, None), 0, 0]
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi_r2).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi_r2)
        thr_tmem_load_h = None
        tStS_t2r_a = None
        tStS_t2r_b = None
        tScSh = None
        if const_expr(self.qk_split_pipe):
            _half_layout = cute.composition(tStSi.layout, cute.make_layout((self.m_block_size, 64)))
            _tStSh_a = cute.make_tensor(tStSi.iterator, _half_layout)
            _tStSh_b = cute.make_tensor(tStSi.iterator + 64, _half_layout)
            tScSh = cute.composition(tScS, cute.make_layout((self.m_block_size, 64)))
            thr_tmem_load_h = tcgen05.make_tmem_copy(tmem_load_atom, _tStSh_a).get_slice(tidx)
            tStS_t2r_a = thr_tmem_load_h.partition_S(_tStSh_a)
            tStS_t2r_b = thr_tmem_load_h.partition_S(_tStSh_b)
            tSrS_a_persist = cute.make_rmem_tensor(thr_tmem_load_h.partition_D(tScSh).shape, Float32)
        else:
            tSrS_a_persist = None
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
            Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        # `tmem-store-rep-presence-test`: ONE read with ONE default. The setting's PRESENCE selects the P store
        # atom's repetition on the 8-bit non-block-scaled PV path, so an earlier presence test picked a different
        # store atom under an unchanged cache key. Reading FwdKernelSpec.fp8_pv_tmem_store_rep once with a None
        # default makes presence and value alike visible to spec.key(), which keys it.
        _fp8_pv_store_rep_knob = self.spec.fp8_pv_tmem_store_rep   # presence-significant (None = the default below)
        fp8_pv_store_rep = int(_fp8_pv_store_rep_knob) if _fp8_pv_store_rep_knob is not None else 8
        fp8_pv_default_store_rep = 4 if self.head_dim_v_padded <= 64 else 8
        # FP4 PV: smaller rep = more P store chunks = finer-grained P handoff
        # to the PV MMA (combined with fp4_pv_p_split_num / den).
        fp4_pv_store_rep = int(self.spec.fp4_pv_tmem_store_rep)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(
                    fp8_pv_store_rep if const_expr(
                        self.p_rowframe8 and _fp8_pv_store_rep_knob is not None
                    ) else
                    fp8_pv_default_store_rep if const_expr(self.p_rowframe8) else
                    fp4_pv_store_rep if const_expr(self.quant_pv) else
                    16
                )
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # First iter: no need for wait correction for sfqk1
        if const_expr(self.qk_split_pipe):
            # sfqk_load[stage] now means "this WG loaded S(stage, t)": one pre-arrive so the
            # MMA's first QK_a of the kernel passes
            self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset + stage)
        elif const_expr(self.q_stage == 2):
            if stage == 1 and const_expr(self.quant_qk):
                sfqk_stage = 0
                self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)
        elif const_expr(self.quant_qk):
            # Single stage: the MMA warp's first SFQ/SFK copy waits on
            # sfqk_load[0] phase 0, which the (absent) stage-1 WG used to
            # pre-arrive. Stage 0 pre-arrives for itself instead.
            self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset)
        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        vt_scount = Int32(0)  # vtok_w: key tiles processed so far in this CTA (mirrors the load warp's count)

        if const_expr(self.pv_row_p):
            # mxfp8-pv-row-frame-p: this stage's P scale factors are the constant 1.0 (UE8M0 127) -- the row-frame P^
            # carries its scale in the row sum. Written once per CTA at the thread's slots of the SF atom (the layout
            # softmax_step's R2S uses: 4 K-groups per row, lane * 16 + (warp % 4) * 4); the MMA warp S2T-copies the
            # tile before every PV. Fenced for the async proxy before any P_full arrive of this WG.
            _base = (tidx % 32) * 16 + ((tidx // 32) % 4) * 4
            _sfp_w = cute.make_tensor(
                cute.recast_ptr(sSFP[None, None, None, stage].iterator + _base, dtype=cutlass.Uint32),
                cute.make_layout(1),
            )
            _sfp_w[0] = cutlass.Uint32(0x7F7F7F7F)
            cute.arch.fence_proxy("async.shared", space="cta")

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(self.score_scale_doc):
                if const_expr(self.score_mod is not None):
                    # score_mod: this tile's (document, kv head) scale is applied before the mod; the log2 scale stays
                    # the change of base (LOG2_E)
                    softmax_scale = self._doc_score(batch_idx, head_idx)
                else:
                    softmax_scale_log2 = self._doc_score_log2(batch_idx, head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            vt_c0 = vt_scount + n_block_max - 1
            vt_nt = n_block_max - n_block_min

            # two-cta-forward: Q stage `stage` of the pair is the 256-row group m_wg (this CTA its half
            # mma_tile_coord_v, whose row offset thr_mma_qk's partition carries); its own KV range ends n_zero tiles
            # before the pair's (the causal tiles that only the pair's later rows need)
            m_wg = self.q_stage * m_block + stage
            n_zero = None
            if const_expr((self.use_2cta_forward and self.is_causal) or self.stage_skip):
                n_zero = self._pair_stage_skip(block_info_wg, seqlen, m_block, stage, split_idx, num_splits, n_block_max)
            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=m_wg * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=_as_aux_data(aux_tensors),
            )
            # masks / varlen: a mask_mod without block sparsity is evaluated on every tile (as the backward does)
            block_mask_mod = self.mask_mod
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=block_mask_mod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            import math as _math
            if const_expr(self.p_sf_inkernel):
                # P range targets the quantized dtype's max: E2M1 max = 6,
                # E4M3 max = 448 (MXFP8 PV). NVFP4 PV with two-level scaling
                # (see fp4_pv_two_level) raises the row max to 6*448 so the
                # UE4M3 group SFs use the full E4M3 range.
                if const_expr(self.mx_pv_split):
                    # UE8M0 SFs: the fold must be an integer (see mx_pv_split). The value comes from the one
                    # constant _lse_log2_offset also subtracts (MX_INT_FRAME_OFFSET), so the frame the group SFs
                    # carry and the frame the LSE removes are the same number by construction.
                    _max_offset = float(self.MX_INT_FRAME_OFFSET)
                elif const_expr(self.v_dtype.width == 4):
                    _max_offset = _math.log2(6) + (_math.log2(448) if self.fp4_pv_two_level else 0.0)
                else:
                    _max_offset = _math.log2(448)
                # Lazy O rescaling (FA4): 0.0 = correction warps rescale O every
                # KV block. Default 8 (measured on B200): together with the two-level bias
                # lowered by the threshold below, group-max-first and the 1/4
                # FMA-pipe exp2 it is +17-21 % at identical output. (Threshold 8
                # alone, without the bias lowering, saturated E4M3 SFs / was slower.)
                _rescale_threshold = self._fp4_pv_rescale_threshold()
                if const_expr(self.mx_pv_split):
                    _rescale_threshold = 8.0   # the MXFP4 split step's integer-frame threshold (log2 units)
                elif _rescale_threshold > 0.0 and self.fp4_pv_two_level:
                    # A stale row max lets the group SF grow by up to 2^threshold:
                    # lower the two-level bias by the same amount so SF stays
                    # <= 448 (E4M3 max) instead of saturating (cos 0.98 -> 0.946).
                    _max_offset -= _rescale_threshold
            elif const_expr(self.p_rowframe8):
                # FP8 PV: P is already per-tile normalized to [0, 1] before the
                # E4M3 cast, which sits in E4M3's dense region -- no extra offset
                # needed. A nonzero offset (the old d>64 default of 8.0) only
                # pushes P up into coarser-mantissa E4M3 codes, dropping FP8-PV
                # cos_sim from ~0.99 to ~0.977 (0.9985->0.984 for MXFP8 QK) with
                # no throughput gain. Keep 0.0; override via env if ever needed.
                # (offset, threshold) must stay below log2 448 = 8.81 together. Upstream's FP8 forward
                # runs (8, 0): P^ fills the top of E4M3 with an exact row max every tile; this path
                # ran (0, 8), whose fresh-max tiles leave the small probabilities in the subnormal
                # band (O cosine 3e-5 below upstream on every gate cell). The
                # "offset 8 hurts" measurement above was taken with threshold 8 as well (saturation).
                # Measured: (8, 0) reproduces upstream's O to 1e-6 (cos vs
                # fp32 0.999794 / 0.999687 = upstream's); (0, 8) 0.999764 / 0.999649; every lazy pair
                # in between keeps most of the deficit ((7, 1): 0.999782 / 0.999668). Exactness costs
                # the O rescale on fresh-max tiles: -4 % causal, -8.5 % non-causal at s4096.
                # Scoped to pure E4M3 Q/K/V: the NVFP4-QK + FP8-PV hybrid (the FP4 training step's
                # forward) keeps its measured (0, 8) -- the exact-max trade was not measured there
                # (the rescale alone cost it 3-7 % in the matrix).
                # Pure FP8 default (6, 2): the tuned split of E4M3's 8.8 log2 between small-P precision
                # (offset) and lazy-max headroom (threshold). vs the exact (8, 0): +1 / +1 / +7 / +6 %
                # at the four report cells, O cos 0.999774 / 0.999660 (60 % of the (0, 8) deficit gone),
                # and not worse on the 4-seed convergence gate (0.8526 vs 0.8567; upstream 0.8503).
                # (values resolved once in _setup_attributes; the LSE epilogue subtracts the same offset)
                _max_offset = self.fp8_pv_log2_offset
                _rescale_threshold = self.fp8_pv_rescale_threshold
            else:
                _max_offset = 0
                _rescale_threshold = 8.0
            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=_rescale_threshold,
                softmax_scale=softmax_scale,
                max_offset=_max_offset,
            )
            softmax.reset()
            # return_pmax: true running row max (softmax.row_max is kept stale within the rescale threshold)
            # token-block-qk-fold: the next step's f_K (and the tile's f_Q), loaded one step ahead
            tbqk_pref = cute.make_rmem_tensor(cute.make_layout(self.n_block_size // 32 + 1), Float32) \
                if const_expr(self.tbqk_fold) else None
            pmax_rm = None
            if const_expr(self.return_pmax):
                pmax_rm = cute.make_rmem_tensor(cute.make_layout(1), Float32)
                pmax_rm[0] = -Float32.inf
            frame_rm = None
            if const_expr(self.fp4all and self.repro):
                # `reproduce-forward-p`: [0] the frame, [1] the number of frame changes recorded so far (mRepro)
                frame_rm = cute.make_rmem_tensor(cute.make_layout(2), Float32)
                frame_rm[0] = -Float32.inf
                frame_rm[1] = Float32(0.0)
                pmax_rm = frame_rm
            elif const_expr(self.fp4all):
                # fp4all: the row's integer log2 frame (exported through the pmax slot)
                frame_rm = pmax_rm
            elif const_expr(self.int_frame):
                frame_rm = cute.make_rmem_tensor(cute.make_layout(1), Float32)
                frame_rm[0] = -Float32.inf
            elif const_expr(self.fp4_pv_tile_frame):
                # `per-tile-row-frame`: frame_rm holds the TRUE running raw max (the floor of the tile reference);
                # softmax.row_max holds the reference the current tile was scaled against (what the LSE uses)
                assert self.qk_split_pipe and self.quant_pv and self.v_dtype.width == 4 and not self.mx_pv, \
                    "fp4_pv_tile_frame: split NVFP4-PV step only"
                frame_rm = cute.make_rmem_tensor(cute.make_layout(1), Float32)
                frame_rm[0] = -Float32.inf
            # the tile reference's floor distance in raw score units, once per work tile (MUFU.RCP: the floor's exact
            # position does not matter, only that it sits ~drop log2 units under the running max)
            tf_drop_raw = (Float32(self.fp4_pv_tile_frame_drop) * cute.arch.rcp_approx(softmax_scale_log2)
                           if const_expr(self.fp4_pv_tile_frame) else None)

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            _step_fn = self._softmax_step_fn()
            softmax_step = partial(
                _step_fn,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_wg * self.cta_group_size,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                mask_fn=partial(mask_fn, mask_seqlen=False),
                tCtSFP=tCtSFP,
                sSFP=sSFP,
                thr_tmem_load_h=thr_tmem_load_h,
                tStS_t2r_a=tStS_t2r_a,
                tStS_t2r_b=tStS_t2r_b,
                tScSh=tScSh,
                tSrS_a_persist=tSrS_a_persist,
                pmax_rm=pmax_rm,
                **({"frame_rm": frame_rm} if self.qk_split_pipe else {}),  # fp4all frames: split pipeline only
                **({"tf_drop_raw": tf_drop_raw} if self.fp4_pv_tile_frame else {}),  # `per-tile-row-frame`
                **({"vt_c0": vt_c0} if self.vtok_w else {}),
                **({"tbqk_pref": tbqk_pref} if self.tbqk_fold else {}),
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
                )
                si_corr_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                )
                if not empty_tile:
                    _rs_off = (2 * self.m_block_size) if const_expr(self.corr_empty_early) else 0
                    _rm_off = (6 * self.m_block_size) if const_expr(self.corr_empty_early) else (2 * self.m_block_size)
                    sScale[tidx + stage * self.m_block_size + _rs_off] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + _rm_off
                        ] = frame_rm[0] if const_expr(self.int_frame) else softmax.row_max[0]
                    if const_expr(self.return_pmax):
                        sScale[tidx + stage * self.m_block_size + self.sscale_pmax_off] = pmax_rm[0]
                    if const_expr(self.repro):
                        # (only a query tile that holds rows: with an odd tile count the CTA's second tile lies past
                        # the sequence -- past the export buffer batched, in the next document's tiles under documents)
                        if (self.q_stage * m_block + stage) * self.m_block_size < seqlen.seqlen_q:
                            self._repro_finish(*self._rp_bm(batch_idx, self.q_stage * m_block + stage, seqlen), head_idx,
                                               tidx, frame_rm, softmax.row_sum[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    if const_expr(self.use_2cta_forward and self.is_causal and not self.pair_stage_skip):
                        # the pair's leading tiles above this warpgroup's diagonal: bookkept, not computed
                        for z in cutlass.range(n_zero, unroll=1):
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = self.softmax_zero_step(
                                mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase,
                                n_block_max - 1 - z, stage, tidx, mbar_ptr, thr_mma_qk, thr_tmem_store,
                                thr_tmem_store_scale, tStScale_r2t, tStP_r2t, sScale, sSFP,
                            )
                    if const_expr((self.use_2cta_forward and self.is_causal) or self.stage_skip):
                        # (pair_stage_skip: the MMA issues no QK / PV of this group for these tiles at all)
                        n_block_max -= n_zero
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                        split_masked=True,
                        split_mask_seqlen=True,
                        **({"unit_scale_first": True} if const_expr(self.use_2cta_forward or self.stage_skip) else {}),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        if const_expr(self.use_2cta_forward or self.stage_skip_wg_mask):
                            n_block_min_causal_local_mask = block_info_wg.get_n_block_min_causal_local_mask(
                                seqlen, m_wg, n_block_min
                            )
                        else:
                            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                    split_masked=True,
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking
                    if const_expr(self.use_2cta_forward or self.stage_skip_wg_mask):
                        n_block_min_before_local_mask = block_info_wg.get_n_block_min_before_local_mask(
                            seqlen, m_wg, n_block_min
                        )
                    else:
                        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    # The default mask partial here has mask_seqlen=False; with
                    # no causal/local/mask_mod it is a compile-time no-op, so
                    # the ld.red hardware row max is valid.
                    unmasked = cutlass.const_expr(
                        not self.is_causal
                        and not self.is_local
                        and self.mask_mod is None
                        and not self.use_block_sparsity
                    )
                    unmasked_step = softmax_step
                    # mask_loops: these tiles need no mask at all -- drop the partial's default
                    # mask_fn (on causal it is the causal compare/select on every element)
                    _unmasked_kw = {"mask_fn": None} if const_expr(self.mask_loops >= 1 and not self.use_block_sparsity and self.mask_mod is None) else {}
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = unmasked_step(
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                            # (a dropped mask is a no-op too: interior causal / window tiles take the fused ld.red
                            # row max and group maxes like the dense kernel's; only SM103 builds differ)
                            mask_is_noop=unmasked or const_expr(self.use_ldred_rowmax and len(_unmasked_kw) > 0),
                            **_unmasked_kw,
                        )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    _rs_off = (2 * self.m_block_size) if const_expr(self.corr_empty_early) else 0
                    _rm_off = (6 * self.m_block_size) if const_expr(self.corr_empty_early) else (2 * self.m_block_size)
                    sScale[tidx + stage * self.m_block_size + _rs_off] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + _rm_off
                        ] = frame_rm[0] if const_expr(self.int_frame) else softmax.row_max[0]
                    if const_expr(self.return_pmax):
                        sScale[tidx + stage * self.m_block_size + self.sscale_pmax_off] = pmax_rm[0]
                    if const_expr(self.repro):
                        # (only a query tile that holds rows: with an odd tile count the CTA's second tile lies past
                        # the sequence -- past the export buffer batched, in the next document's tiles under documents)
                        if (self.q_stage * m_block + stage) * self.m_block_size < seqlen.seqlen_q:
                            self._repro_finish(*self._rp_bm(batch_idx, self.q_stage * m_block + stage, seqlen), head_idx,
                                               tidx, frame_rm, softmax.row_sum[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + utils.log2f(softmax.row_sum[0])) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            if const_expr(self.vtok_tma):
                vt_scount = (vt_scount + vt_nt) % (self.kv_stage // 2)
            elif const_expr(self.vtok_w and not self.vtok_tma):
                vt_scount = (vt_scount + vt_nt) % self.vtok_ring
            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop
        
    @cute.jit
    def _quant_fp4(self, 
                   # src
                   tSrP_f32: cute.Tensor,
                   tSrPSF_f32: cute.Tensor,
                   # dst
                   tSrP: cute.Tensor,
                   tSrPSF: cute.Tensor,
                   ):
        tSrP_f32_frag = cute.logical_divide(tSrP_f32, cute.make_layout(self.sf_vec_size_pv))
        assert cute.size(tSrP_f32_frag, mode=[1]) == cute.size(tSrPSF_f32)
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(self.sf_vec_size_pv))
        tSrPSF_u32_view = cute.recast_tensor(tSrPSF, cute.Int32)

        # Process in groups of 4 for UE4M3 conversion
        assert cute.size(tSrPSF_f32) % 4 == 0
        for i in cutlass.range_constexpr(0, cute.size(tSrPSF_f32) // 4):
        # for i in cutlass.range_constexpr(0, 2):
            # Pack 4 FP32 values into UE4M3 format
            packed_ue4m3 = packed_float_to_ue4m3(
                tSrPSF_f32[i * 4],
                tSrPSF_f32[i * 4 + 1], 
                tSrPSF_f32[i * 4 + 2],
                tSrPSF_f32[i * 4 + 3]
            )
            tSrPSF_u32_view[i] = packed_ue4m3
    
        # Quantize main tensor to E2M1 format (8 values per uint32_t)
        # Process in groups of 8 for E2M1 conversion
        for i in cutlass.range_constexpr(0, cute.size(tSrP_frag, mode=[1])):
            tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, i], cute.Int32)
            for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                packed_e2m1 = packed_float_to_e2m1(
                    tSrP_f32_frag[k * 8, i],
                    tSrP_f32_frag[k * 8 + 1, i],
                    tSrP_f32_frag[k * 8 + 2, i],
                    tSrP_f32_frag[k * 8 + 3, i],
                    tSrP_f32_frag[k * 8 + 4, i],
                    tSrP_f32_frag[k * 8 + 5, i],
                    tSrP_f32_frag[k * 8 + 6, i],
                    tSrP_f32_frag[k * 8 + 7, i]
                )
                tSrP_u32_view[k] = packed_e2m1

    @cute.jit
    def _fused_group_max_scale_quant(
        self,
        softmax: SoftmaxSm100,
        tSrP_f32: cute.Tensor,
        tSrP: cute.Tensor,
    ):
        """Fused group_max + scale + quantize to reduce register pressure.

        Instead of 3 separate passes over the full fragment (128 FP32 values),
        processes each sf_vec_size group (16 elements) sequentially:
        group_max -> scale -> quant_e2m1 before moving to the next group.
        Peak live regs: 16 (group) + 1 (max) + 2 (packed) vs ~150 in the split path.
        """
        inv6 = Float32(1.0 / 6.0)
        # E8M0 SFs (MXFP4 PV) must be powers of two: round the group scale UP so
        # P/sf stays <= 6, and use the rounded value for both scaling and packing.
        is_e8m0_sf = const_expr(self.sf_dtype_pv == Float8E8M0FNU)
        acc_S_frag = cute.logical_divide(tSrP_f32, cute.make_layout(self.sf_vec_size_pv))
        num_groups = cute.size(acc_S_frag, mode=[1])
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(self.sf_vec_size_pv))
        tSrPSF_f32 = cute.make_rmem_tensor(cute.make_layout(num_groups), Float32)
        tSrPSF = cute.make_rmem_tensor(tSrPSF_f32.layout, self.sf_dtype_pv)

        for g in cutlass.range_constexpr(num_groups):
            # 1. Group max (reduction over sf_vec_size elements)
            gmax = softmax._compute_row_max(acc_S_frag[None, g].load()) * inv6
            if const_expr(is_e8m0_sf):
                gmax = cute.arch.exp2(sm100_utils.ceil_f32(cute.math.log2(gmax)))
            tSrPSF_f32[g] = gmax

            # 2. Scale by 1/gmax
            inv_gmax = Float32(1.0) / cute.arch.fmax(gmax, 1e-20)
            for j in cutlass.range(0, self.sf_vec_size_pv, 2, unroll_full=True):
                acc_S_frag[j, g], acc_S_frag[j + 1, g] = utils.fma_packed_f32x2(
                    (acc_S_frag[j, g], acc_S_frag[j + 1, g]),
                    (inv_gmax, inv_gmax),
                    (Float32(0.0), Float32(0.0)),
                )

            # 3. Quantize this group to E2M1
            tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, g], cute.Int32)
            for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                tSrP_u32_view[k] = packed_float_to_e2m1(
                    acc_S_frag[k * 8, g],
                    acc_S_frag[k * 8 + 1, g],
                    acc_S_frag[k * 8 + 2, g],
                    acc_S_frag[k * 8 + 3, g],
                    acc_S_frag[k * 8 + 4, g],
                    acc_S_frag[k * 8 + 5, g],
                    acc_S_frag[k * 8 + 6, g],
                    acc_S_frag[k * 8 + 7, g],
                )

        # 4. Pack SF values to UE4M3 / UE8M0 (separate loop, small tensor)
        tSrPSF_u32_view = cute.recast_tensor(tSrPSF, cute.Int32)
        for i in cutlass.range_constexpr(0, cute.size(tSrPSF_f32) // 4):
            if const_expr(is_e8m0_sf):
                tSrPSF_u32_view[i] = sm100_utils.packed_float_to_ue8m0(
                    tSrPSF_f32[i * 4],
                    tSrPSF_f32[i * 4 + 1],
                    tSrPSF_f32[i * 4 + 2],
                    tSrPSF_f32[i * 4 + 3],
                )
            else:
                tSrPSF_u32_view[i] = packed_float_to_ue4m3(
                    tSrPSF_f32[i * 4],
                    tSrPSF_f32[i * 4 + 1],
                    tSrPSF_f32[i * 4 + 2],
                    tSrPSF_f32[i * 4 + 3],
                )

        return tSrPSF_f32, tSrPSF

    @cute.jit
    def _fused_log2_group_quant(
        self,
        softmax: SoftmaxSm100,
        tSrS: cute.Tensor,   # PRE-exp scaled scores (scale_subtract_rowmax done)
        tSrP: cute.Tensor,   # fp4 view of the P r2t backing storage
        acc_scale: Float32,
        is_first: cutlass.Constexpr[bool],
        # SM103 ld.red per-x32-tile maxes of the RAW S values (pre
        # scale_subtract). When sf_vec_size == 32 each tile max IS the group
        # max: m'_g = m_raw_g * scale_log2 + (max_offset - row_max*scale_log2)
        # replaces the whole FMNMX group reduce with one FMA per group.
        hw_group_maxes: Optional[cute.Tensor] = None,
        row_max: Optional[Float32] = None,
        # fp4_group_max_first: per-group maxes of the RAW (unscaled, not
        # row-max-subtracted) scores; tSrS then holds raw scores and the
        # scale/row-max/group-bias fold into one packed FMA per pair.
        raw_group_maxes: Optional[cute.Tensor] = None,
        clamp_neg_inf: cutlass.Constexpr[bool] = False,
        # fwdgap "gmx": the group maxes of the SCALED scores, already computed (exactly) by the caller
        scaled_group_maxes: Optional[cute.Tensor] = None,
    ):
        """Log-domain NVFP4 P quantization (fp4_pv_log2_quant=1).

        Exploits exp2 monotonicity: max(exp2(s)) = exp2(max(s)). The group max
        is taken on the PRE-exp scores and the 1/group_max scaling folds into
        the exp2 argument:
            m_g  = max_{i in g} s_i
            P_i  = exp2(s_i - m_g + log2 6)   in (0, 6], group max exactly 6.0
            SF_g = exp2(m_g - log2 6)         (== max(exp2 s)/6, as baseline)
        vs the baseline exp-all -> group_max -> 1/x divide -> scale pass, this
        removes all per-group divisions and the post-exp scaling pass, and
        shortens the per-group dependency chain to max -> subtract -> exp2.
        row_sum is rebuilt as sum_g SF_g * sum_{i in g} P_i -- equal to
        sum exp2(s_i) up to FP32 rounding (summation order differs, so output
        is accuracy-equivalent but not bitwise vs baseline).
        """
        is_e8m0_sf = const_expr(self.sf_dtype_pv == Float8E8M0FNU)
        # P range targets the quantized dtype's max (E2M1: 6, E4M3: 448).
        LOG2_PMAX = math.log2(6.0) if const_expr(self.v_dtype.width == 4) else math.log2(448.0)
        acc_S_frag = cute.logical_divide(tSrS, cute.make_layout(self.sf_vec_size_pv))
        num_groups = cute.size(acc_S_frag, mode=[1])
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(self.sf_vec_size_pv))
        tSrPSF_f32 = cute.make_rmem_tensor(cute.make_layout(num_groups), Float32)
        tSrPSF = cute.make_rmem_tensor(tSrPSF_f32.layout, self.sf_dtype_pv)

        scores_raw = const_expr(raw_group_maxes is not None)
        if const_expr(hw_group_maxes is not None):
            assert cute.size(hw_group_maxes) == num_groups, (
                "ld.red tile count must equal SF group count (sf_vec_size==32)"
            )
        if const_expr(hw_group_maxes is not None or scores_raw):
            # Transform raw-S group maxes into post-scale_subtract group maxes.
            hw_bias = softmax.max_offset - row_max * softmax.scale_log2

        row_sum_acc = Float32(0.0)
        for g in cutlass.range_constexpr(num_groups):
            # Group max in log space; clamp so fully-masked groups (all -inf)
            # don't produce NaN from (-inf) - (-inf) below.
            if const_expr(hw_group_maxes is not None):
                m = hw_group_maxes[g] * softmax.scale_log2 + hw_bias
            elif const_expr(scores_raw):
                m = raw_group_maxes[g] * softmax.scale_log2 + hw_bias
            elif const_expr(scaled_group_maxes is not None):
                m = scaled_group_maxes[g]
            else:
                m = softmax._compute_row_max(acc_S_frag[None, g].load())
            bias = cute.arch.fmax(m, Float32(-100.0)) - LOG2_PMAX
            sf_int = None
            if const_expr(is_e8m0_sf and "sfi" in self.xf):
                # fwdgap "sfi": ceil and 2^ceil on the FMA / ALU pipes (no FRND, no MUFU per group)
                bias, sf_int = ceil_pow2_f32(bias)
            elif const_expr(is_e8m0_sf):
                # E8M0 SF is exponent-only: round the log-domain bias UP to an
                # integer so P stays <= the dtype max. SF = 2^ceil(bias).
                bias = sm100_utils.ceil_f32(bias)

            # exp2 with the group scale folded into the argument
            if const_expr(scores_raw):
                # raw scores: P_i = exp2(s_i*scale + (max_offset - row_max*scale - bias))
                c_g = hw_bias - bias
                for j in cutlass.range(0, self.sf_vec_size_pv, 2, unroll_full=True):
                    acc_S_frag[j, g], acc_S_frag[j + 1, g] = utils.fma_packed_f32x2(
                        (acc_S_frag[j, g], acc_S_frag[j + 1, g]),
                        (softmax.scale_log2, softmax.scale_log2),
                        (c_g, c_g),
                    )
            elif const_expr("schg" in self.xf and not scores_raw):
                # fwdgap "schg": the emulated pairs take the Schraudolph read-out of the merged fixed-point FFMA2
                # (t = (x - bias) 128 + SCH_C); the other pairs subtract the bias as below
                sbias_g = Float32(SCH_C) - bias * Float32(128.0)
                for j in cutlass.range_constexpr(0, self.sf_vec_size_pv, 2):
                    _pair = g * (self.sf_vec_size_pv // 2) + j // 2
                    if const_expr(self.fp4_log2_e2e_freq > 0 and _pair % self.fp4_log2_e2e_freq == 0):
                        acc_S_frag[j, g], acc_S_frag[j + 1, g] = utils.fma_packed_f32x2(
                            (acc_S_frag[j, g], acc_S_frag[j + 1, g]), (Float32(128.0), Float32(128.0)),
                            (sbias_g, sbias_g))
                    else:
                        acc_S_frag[j, g], acc_S_frag[j + 1, g] = utils.fma_packed_f32x2(
                            (acc_S_frag[j, g], acc_S_frag[j + 1, g]), (Float32(1.0), Float32(1.0)), (-bias, -bias))
            else:
                for j in cutlass.range(0, self.sf_vec_size_pv, 2, unroll_full=True):
                    acc_S_frag[j, g], acc_S_frag[j + 1, g] = utils.fma_packed_f32x2(
                        (acc_S_frag[j, g], acc_S_frag[j + 1, g]),
                        (Float32(1.0), Float32(1.0)),
                        (-bias, -bias),
                    )
            for j in cutlass.range_constexpr(0, self.sf_vec_size_pv, 2):
                _pair = g * (self.sf_vec_size_pv // 2) + j // 2
                if const_expr("schg" in self.xf and not scores_raw and self.fp4_log2_e2e_freq > 0
                              and _pair % self.fp4_log2_e2e_freq == 0):
                    acc_S_frag[j, g], acc_S_frag[j + 1, g] = sch_bf16_pair(acc_S_frag[j, g], acc_S_frag[j + 1, g])
                elif const_expr(self.fp4_log2_e2e_freq > 0 and _pair % self.fp4_log2_e2e_freq == 0):
                    # FMA-pipe exp2 (packed polynomial) for this pair
                    acc_S_frag[j, g], acc_S_frag[j + 1, g] = self._ex2_emu_pair(
                        acc_S_frag[j, g], acc_S_frag[j + 1, g], clamp=clamp_neg_inf
                    )
                else:
                    acc_S_frag[j, g] = cute.arch.exp2(acc_S_frag[j, g])
                    acc_S_frag[j + 1, g] = cute.arch.exp2(acc_S_frag[j + 1, g])

            sf = cute.arch.exp2(bias) if const_expr(sf_int is None) else sf_int
            tSrPSF_f32[g] = sf
            # Reconstruct this group's contribution to row_sum from the
            # scaled values weighted back by SF.
            row_sum_acc += sf * softmax._compute_row_sum(acc_S_frag[None, g].load())

            # Quantize this group to the P dtype
            tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, g], cute.Int32)
            if const_expr(self.v_dtype.width == 4):
                for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                    tSrP_u32_view[k] = packed_float_to_e2m1(
                        acc_S_frag[k * 8, g],
                        acc_S_frag[k * 8 + 1, g],
                        acc_S_frag[k * 8 + 2, g],
                        acc_S_frag[k * 8 + 3, g],
                        acc_S_frag[k * 8 + 4, g],
                        acc_S_frag[k * 8 + 5, g],
                        acc_S_frag[k * 8 + 6, g],
                        acc_S_frag[k * 8 + 7, g],
                    )
            else:
                # MXFP8: E4M3 elements, 4 per u32
                for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                    tSrP_u32_view[k] = packed_float_to_ue4m3(
                        acc_S_frag[k * 4, g],
                        acc_S_frag[k * 4 + 1, g],
                        acc_S_frag[k * 4 + 2, g],
                        acc_S_frag[k * 4 + 3, g],
                    )

        # row_sum update with the same rescale semantics as update_row_sum
        if const_expr(is_first):
            softmax.row_sum[0] = row_sum_acc
        else:
            softmax.row_sum[0] = softmax.row_sum[0] * acc_scale + row_sum_acc

        # Pack SF values (UE4M3 for NVFP4, UE8M0 for MXFP8)
        tSrPSF_u32_view = cute.recast_tensor(tSrPSF, cute.Int32)
        for i in cutlass.range_constexpr(0, max(1, cute.size(tSrPSF_f32) // 4)):
            if const_expr(is_e8m0_sf):
                tSrPSF_u32_view[i] = sm100_utils.packed_float_to_ue8m0(
                    tSrPSF_f32[i * 4],
                    tSrPSF_f32[i * 4 + 1],
                    tSrPSF_f32[i * 4 + 2],
                    tSrPSF_f32[i * 4 + 3],
                )
            else:
                tSrPSF_u32_view[i] = packed_float_to_ue4m3(
                    tSrPSF_f32[i * 4],
                    tSrPSF_f32[i * 4 + 1],
                    tSrPSF_f32[i * 4 + 2],
                    tSrPSF_f32[i * 4 + 3],
                )

        return tSrPSF_f32, tSrPSF

    def _schm(self):
        """fwdgap "schmK" (K = 0..8): K of every 8 pairs on MUFU, the rest Schraudolph. None when absent."""
        for x in self.xf:
            if x.startswith("schm") and x[4:].isdigit():
                return int(x[4:])
        return None

    def _sch_pair_sel(self, p: int, emu: bool) -> bool:
        k = self._schm()
        if k is not None:
            return p % 8 >= k
        return ("scha" in self.xf) or ("sche" in self.xf and emu)

    @cute.jit
    def _exp_sch_pass(self, softmax, tSrS: cute.Tensor, row_max: Float32, gscale,
                      e2e: cutlass.Constexpr[bool], e2e_freq: cutlass.Constexpr[int],
                      e2e_start_frg: cutlass.Constexpr[int], e2e_res: cutlass.Constexpr[int] = 4):
        """fwdgap "sche" / "scha", four-pass step: per pair the scale-subtract FFMA2 and its exp2 -- MUFU, the FMA-pipe
        emulation (apply_exp2_convert's pairs) or the Schraudolph read-out (sch_bf16_pair) on the merged fixed-point
        FFMA2."""
        n = cute.size(tSrS)
        frg_tile = 32
        frg_cnt = n // frg_tile
        scale = softmax.scale_log2
        bias = Float32(softmax.max_offset) - row_max * scale
        sbias = bias * Float32(128.0) + Float32(SCH_C)
        for e in cutlass.range_constexpr(0, n, 2):
            j, k = e // frg_tile, e % frg_tile
            emu = const_expr(e2e and e2e_freq > 0 and k % e2e_freq >= e2e_freq - e2e_res and j < frg_cnt - 1
                             and j >= e2e_start_frg)
            sc = scale if const_expr(gscale is None) else gscale[e // 32]
            if const_expr(self._schm() is not None):
                emu = False
            if const_expr(self._sch_pair_sel(e // 2, emu)):
                ssc = sc * Float32(128.0)
                tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (ssc, ssc), (sbias, sbias))
                tSrS[e], tSrS[e + 1] = sch_bf16_pair(tSrS[e], tSrS[e + 1])
            elif const_expr(emu):
                tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (sc, sc), (bias, bias))
                tSrS[e], tSrS[e + 1] = utils.ex2_emulation_2(tSrS[e], tSrS[e + 1])
            else:
                tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (sc, sc), (bias, bias))
                tSrS[e] = cute.math.exp2(tSrS[e], fastmath=True)
                tSrS[e + 1] = cute.math.exp2(tSrS[e + 1], fastmath=True)

    @cute.jit
    def _gmx_update_row_max(self, softmax, row_max_new: Float32, is_first: cutlass.Constexpr[bool]):
        """SoftmaxSm100.update_row_max with the new row max given (fwdgap "gmx"): the same expressions."""
        if const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = softmax.row_max[0]
            row_max_safe = row_max_new if row_max_new != -Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * softmax.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if const_expr(softmax.rescale_threshold > 0.0):
                if acc_scale_ >= -softmax.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        softmax.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def _softmax_fused_e4m3(
        self,
        softmax,
        tSrS: cute.Tensor,   # raw S (this thread's row slice, fp32); exp2'd in place
        tSrP: cute.Tensor,   # E4M3 view of the P r2t backing storage (tSrS's layout)
        row_max: Float32,
        acc_scale: Float32,
        is_first: cutlass.Constexpr[bool],
        e2e: cutlass.Constexpr[bool],
        e2e_freq: cutlass.Constexpr[int],
        e2e_start_frg: cutlass.Constexpr[int],
        e2e_res: cutlass.Constexpr[int] = 4,
        gscale=None,   # token-block QK fold: per 32-key group multiplier a_g * scale_log2 (bias unchanged)
    ):
        """softmax_fused: scale-subtract, exp2 (MUFU or the FMA-pipe emulation on the
        same pairs apply_exp2_convert picks), the E4M3 pack and the row sum in ONE loop over
        pairs, in FlashInfer's shape (fmha.py apply_exp_and_cvt): the FFMA2 of pair p+4 is
        issued between the two exp2 of pair p, the cvt of a 4-element group and its row-sum FADD2
        trail 4 pairs behind. The row sum keeps fadd_reduce's order exactly (4 packed lane
        accumulators over consecutive element pairs, the running sum seeded into lane 0 of the
        first, the same final tree), so O and LSE are bitwise those of the separate passes."""
        n = cute.size(tSrS)
        assert n % 8 == 0
        npairs = n // 2
        frg_tile = 32
        frg_cnt = n // frg_tile
        LOOK = 4  # pairs of scale-subtract lookahead (8 elements)
        LAG = 4   # pairs the row sum / cvt trail behind (8 elements)
        scale = softmax.scale_log2
        # the same bias scale_subtract_rowmax computes
        bias = Float32(softmax.max_offset) - row_max * scale
        # the FFMA2 multiplier of element e: scale, or the token-block group's a_g * scale (gscale[e // 32])
        sc_of = (lambda e_: scale) if const_expr(gscale is None) else (lambda e_: gscale[e_ // 32])
        tSrP_u32 = cute.recast_tensor(tSrP, cute.Int32)
        acc = [None] * 4
        if const_expr(not is_first):
            init_val = softmax.row_sum[0] * acc_scale

        def is_emu(p_):
            e_ = 2 * p_
            return (e2e and e2e_freq > 0 and (e_ % frg_tile) % e2e_freq >= e2e_freq - e2e_res
                    and e_ // frg_tile < frg_cnt - 1 and e_ // frg_tile >= e2e_start_frg)

        # fwdgap "sche" / "scha" / "schmK": Schraudolph exp2 on the emulated pairs / on every pair / on 8 - K of every 8
        # pairs (the other K on MUFU, no polynomial pairs) -- opt-in, approximate
        sch_of = lambda p_: self._sch_pair_sel(p_, is_emu(p_))  # noqa: E731
        if const_expr(self._schm() is not None):
            is_emu = lambda p_: False  # noqa: E731
        any_sch = any(sch_of(p_) for p_ in range(npairs))
        if const_expr(any_sch):
            sbias = bias * Float32(128.0) + Float32(SCH_C)
            if const_expr(gscale is None):
                ssc = scale * Float32(128.0)
                ssc_of = lambda e_: ssc  # noqa: E731
            else:
                ssc_g = [gscale[g_] * Float32(128.0) for g_ in range(n // 32)]
                ssc_of = lambda e_: ssc_g[e_ // 32]  # noqa: E731

        def ffma_pair(e_):
            if const_expr(sch_of(e_ // 2)):
                tSrS[e_], tSrS[e_ + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS[e_], tSrS[e_ + 1]), (ssc_of(e_), ssc_of(e_)), (sbias, sbias))
            else:
                tSrS[e_], tSrS[e_ + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS[e_], tSrS[e_ + 1]), (sc_of(e_), sc_of(e_)), (bias, bias))
        for p in cutlass.range_constexpr(LOOK):
            ffma_pair(2 * p)
        for p in cutlass.range_constexpr(npairs + LAG):
            if const_expr(p >= LAG):
                # consume pair q = p - LAG: row sum (fadd_reduce's order) and the E4M3 pack
                q = p - LAG
                lane = q % 4
                if const_expr(q < 4):
                    if const_expr(lane == 0 and not is_first):
                        acc[0] = cute.arch.add_packed_f32x2((init_val, 0.0), (tSrS[2 * q], tSrS[2 * q + 1]))
                    else:
                        acc[lane] = (tSrS[2 * q], tSrS[2 * q + 1])
                else:
                    acc[lane] = cute.arch.add_packed_f32x2(acc[lane], (tSrS[2 * q], tSrS[2 * q + 1]))
                if const_expr(q % 2 == 1):
                    g = q // 2
                    tSrP_u32[g] = packed_float_to_ue4m3(
                        tSrS[4 * g], tSrS[4 * g + 1], tSrS[4 * g + 2], tSrS[4 * g + 3])
            if const_expr(p < npairs):
                e = 2 * p
                emu = const_expr(is_emu(p))
                if const_expr(sch_of(p)):
                    tSrS[e], tSrS[e + 1] = sch_bf16_pair(tSrS[e], tSrS[e + 1])
                    if const_expr(p + LOOK < npairs):
                        ffma_pair(e + 2 * LOOK)
                elif const_expr(emu):
                    tSrS[e], tSrS[e + 1] = utils.ex2_emulation_2(tSrS[e], tSrS[e + 1])
                    if const_expr(p + LOOK < npairs):
                        ffma_pair(e + 2 * LOOK)
                else:
                    tSrS[e] = cute.math.exp2(tSrS[e], fastmath=True)
                    if const_expr(p + LOOK < npairs):
                        ffma_pair(e + 2 * LOOK)
                    tSrS[e + 1] = cute.math.exp2(tSrS[e + 1], fastmath=True)
        acc[0] = cute.arch.add_packed_f32x2(acc[0], acc[1])
        acc[2] = cute.arch.add_packed_f32x2(acc[2], acc[3])
        acc[0] = cute.arch.add_packed_f32x2(acc[0], acc[2])
        softmax.row_sum[0] = acc[0][0] + acc[0][1]

    @cute.jit
    def _softmax_fused_e4m3_tail(
        self,
        softmax,
        tSrS: cute.Tensor,
        tSrP: cute.Tensor,
        row_max: Float32,
        e2e: cutlass.Constexpr[bool],
        e2e_freq: cutlass.Constexpr[int],
        e2e_start_frg: cutlass.Constexpr[int],
        e2e_res: cutlass.Constexpr[int] = 4,
        gscale=None,   # token-block QK fold: per 32-key group multiplier a_g * scale_log2 (bias unchanged)
    ):
        """softmax_fused=2: scale-subtract + exp2 + pack in FlashInfer's issue order -- the
        MUFU pairs first in element order (the FFMA2 of the 4th-next MUFU pair between the two
        exp2), the emulated pairs (the same 20) as a tail, each cvt as soon as its 4-element group
        is complete. No row sum here: the caller keeps update_row_sum after the P store (fadd_reduce's
        order), so the values and their accumulation are those of the separate passes."""
        n = cute.size(tSrS)
        npairs = n // 2
        frg_tile = 32
        frg_cnt = n // frg_tile
        LOOK = 4
        scale = softmax.scale_log2
        bias = Float32(softmax.max_offset) - row_max * scale
        sc_of = (lambda e_: scale) if const_expr(gscale is None) else (lambda e_: gscale[e_ // 32])
        tSrP_u32 = cute.recast_tensor(tSrP, cute.Int32)
        # trace-time schedule: the MUFU pairs, the emulated pairs, and after which position of the
        # combined order each 4-element group's cvt can go
        emu = [
            bool(e2e) and e2e_freq > 0
            and ((2 * p) % frg_tile) % e2e_freq >= e2e_freq - e2e_res
            and (2 * p) // frg_tile < frg_cnt - 1
            and (2 * p) // frg_tile >= e2e_start_frg
            for p in range(npairs)
        ]
        order = [p for p in range(npairs) if not emu[p]] + [p for p in range(npairs) if emu[p]]
        n_mufu = npairs - sum(emu)
        pos_of = {p: i for i, p in enumerate(order)}
        cvt_after = [[] for _ in range(len(order))]
        for g in cutlass.range_constexpr(npairs // 2):   # trace-time (a plain range would be a dynamic loop)
            cvt_after[max(pos_of[2 * g], pos_of[2 * g + 1])].append(g)
        for i in cutlass.range_constexpr(min(LOOK, n_mufu)):
            e = 2 * order[i]
            tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2(
                (tSrS[e], tSrS[e + 1]), (sc_of(e), sc_of(e)), (bias, bias))
        for i in cutlass.range_constexpr(len(order)):
            e = 2 * order[i]
            if const_expr(i < n_mufu):
                tSrS[e] = cute.math.exp2(tSrS[e], fastmath=True)
                if const_expr(i + LOOK < n_mufu):
                    e2 = 2 * order[i + LOOK]
                    tSrS[e2], tSrS[e2 + 1] = cute.arch.fma_packed_f32x2(
                        (tSrS[e2], tSrS[e2 + 1]), (sc_of(e2), sc_of(e2)), (bias, bias))
                tSrS[e + 1] = cute.math.exp2(tSrS[e + 1], fastmath=True)
            else:
                tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS[e], tSrS[e + 1]), (sc_of(e), sc_of(e)), (bias, bias))
                tSrS[e], tSrS[e + 1] = utils.ex2_emulation_2(tSrS[e], tSrS[e + 1])
            for gi in cutlass.range_constexpr(len(cvt_after[i])):
                g = cvt_after[i][gi]
                tSrP_u32[g] = packed_float_to_ue4m3(
                    tSrS[4 * g], tSrS[4 * g + 1], tSrS[4 * g + 2], tSrS[4 * g + 3])

    @cute.jit
    def _pack_fp8(self, tSrP_f32: cute.Tensor, tSrP: cute.Tensor):
        # Pack FP32 probabilities directly into FP8 lanes so the pure-FP8 PV path
        # avoids the generic elementwise convert-and-pack sequence.
        if const_expr(self.v_dtype == Float8E4M3FN):
            tSrP_f32_frag = cute.logical_divide(tSrP_f32, cute.make_layout(4))
            tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(4))
            for i in cutlass.range_constexpr(0, cute.size(tSrP_frag, mode=[1])):
                tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, i], cute.Int32)
                for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                    tSrP_u32_view[k] = packed_float_to_ue4m3(
                        tSrP_f32_frag[k * 4, i],
                        tSrP_f32_frag[k * 4 + 1, i],
                        tSrP_f32_frag[k * 4 + 2, i],
                        tSrP_f32_frag[k * 4 + 3, i],
                    )
        else:
            # Keep the generic path for E5M2 until we add a dedicated pack helper.
            softmax_like = cute.logical_divide(tSrP, cute.make_layout(min(32, cute.size(tSrP))))
            src_like = cute.logical_divide(tSrP_f32, cute.make_layout(min(32, cute.size(tSrP_f32))))
            for j in cutlass.range_constexpr(cute.size(src_like, mode=[1])):
                softmax_like[None, j].store(src_like[None, j].load().to(tSrP.element_type))

    @cute.jit
    def _split_mask_half(self, tSrS_h: cute.Tensor, tScSh_t2r: cute.Tensor, half: cutlass.Constexpr[int], col_limit: Int32):
        """Causal / seqlen_k mask of one 64-column half of a split-pipeline S tile: columns >= col_limit (tile
        coordinates) become -inf. tScSh_t2r holds the (row, col) coordinates of the half's fragment (col < 64)."""
        lim = col_limit - 64 * half
        for i in cutlass.range_constexpr(cute.size(tSrS_h)):
            tSrS_h[i] = -Float32.inf if tScSh_t2r[i][1] >= lim else tSrS_h[i]

    @cute.jit
    def _tbqk_load(self, tScS_r2: cute.Tensor, thr_tmem_load, batch_idx: Int32, head_idx: Int32, m_block: Int32,
                   n_block: Int32, pref: cute.Tensor, is_first: cutlass.Constexpr[bool]):
        """token-block-qk-fold factors of this step, software-pipelined one step ahead: the first step of a work tile
        loads f_K(n_block) and f_Q(row) itself; every step then issues the loads of f_K(n_block - 1) into `pref` (a
        register buffer that lives across the tile's steps), which the next step consumes -- the global-load latency
        hides behind a whole step instead of sitting between the S wait and the row max."""
        ngr = self.n_block_size // 32
        kvh = head_idx // self.qhead_per_kvhead
        fac = self._tb_fac
        a = cute.make_rmem_tensor(cute.make_layout(ngr + 1), Float32)
        if const_expr(is_first):
            row = thr_tmem_load.partition_D(tScS_r2)[0][0] + m_block * self.m_block_size
            pref[ngr] = fac[0, batch_idx, head_idx, row // 32]
            for g in cutlass.range_constexpr(ngr):
                a[g] = fac[1, batch_idx, kvh, n_block * ngr + g]
        else:
            for g in cutlass.range_constexpr(ngr):
                a[g] = pref[g]
        a[ngr] = pref[ngr]
        nb_next = cutlass.max(n_block - 1, 0)
        for g in cutlass.range_constexpr(ngr):
            pref[g] = fac[1, batch_idx, kvh, nb_next * ngr + g]
        return a

    @cute.jit
    def _tbqk_prep(self, softmax, a: cute.Tensor):
        """a_g = f_Q f_K(g) (powers of two, so a_g S is exact), floored at the smallest normal (an all-zero Q or K
        block has factor 0 and S = 0: any positive a_g gives the same scores, and a masked -inf stays -inf); and the
        FFMA2 multiplier m_g = a_g * scale_log2 (exact)."""
        ngr = self.n_block_size // 32
        gs = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        fq = a[ngr]
        for g in cutlass.range_constexpr(ngr):
            a[g] = cute.arch.fmax(fq * a[g], Float32(1.1754943508222875e-38))
            gs[g] = a[g] * softmax.scale_log2
        return gs

    @cute.jit
    def _tbqk_row_max(self, softmax, tSrS: cute.Tensor, a: cute.Tensor, init_val, gm=None) -> Float32:
        """max_j a_g(j) S_j = max_g a_g max_{j in g} S_j (exact for a_g > 0): the max of the folded scores, which is the
        max the block-scaled S GEMM's output would give."""
        # per group two interleaved chains of 3-input maxes (16 FMNMX per 32 keys, the per-tensor row tree's count);
        # max is exact, so the result is the block-scaled GEMM's row max bit for bit
        ngr = cute.size(tSrS) // 32
        tm = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        for g in cutlass.range_constexpr(ngr):
            if const_expr(gm is not None):   # SM103: ld.red's x32 tile max of this group (the same exact max)
                tm[g] = a[g] * gm[g]
            else:
                o = 32 * g
                m0 = utils.fmax(tSrS[o], tSrS[o + 1], tSrS[o + 2])
                m1 = utils.fmax(tSrS[o + 3], tSrS[o + 4], tSrS[o + 5])
                for i in cutlass.range_constexpr(6, 30, 4):
                    m0 = utils.fmax(m0, tSrS[o + i], tSrS[o + i + 1])
                    m1 = utils.fmax(m1, tSrS[o + i + 2], tSrS[o + i + 3])
                m0 = utils.fmax(m0, tSrS[o + 30], tSrS[o + 31])
                tm[g] = a[g] * cute.arch.fmax(m0, m1)
        if const_expr(init_val is None):
            return utils.fmax(utils.fmax(tm[0], tm[1], tm[2]), tm[3]) if const_expr(ngr == 4) \
                else softmax._compute_row_max(tm.load())
        return utils.fmax(utils.fmax(tm[0], tm[1], tm[2]), tm[3], init_val) if const_expr(ngr == 4) \
            else softmax._compute_row_max(tm.load(), init_val=init_val)

    @cute.jit
    def _tbqk_scale_subtract(self, softmax, tSrS: cute.Tensor, gs: cute.Tensor, row_max: Float32):
        """scale_subtract_rowmax with the per-group multiplier: S (a_g sl2) + (off - M sl2), the bias as there."""
        bias = Float32(softmax.max_offset) - row_max * softmax.scale_log2
        for e in cutlass.range_constexpr(0, cute.size(tSrS), 2):
            tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2(
                (tSrS[e], tSrS[e + 1]), (gs[e // 32], gs[e // 32]), (bias, bias))

    @cute.jit
    def _tb_load(self, tScS_r2: cute.Tensor, thr_tmem_load, softmax, batch_idx: Int32, head_idx: Int32, m_block: Int32,
                 n_block: Int32, q_off: Int32 = None, k_blk_off: Int32 = None):
        """tb_fwd, issued BEFORE the S wait so the global loads overlap it: per 32-key group g of this tile the
        multiplier a_g = f_Q(row) f_K(g) (FP8 QK; unused for NVFP4 QK, whose scale factors are per token), the bias
        c_g = log2 f_V(g) / scale_log2 and the row-sum weight u_g = 1 / f_V(g) (0 for an all-zero V block)."""
        ngr = self.n_block_size // 32
        kvh = head_idx // self.qhead_per_kvhead
        fac = self._tb_fac
        a = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        c = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        u = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        fq = Float32(1.0)
        if const_expr(self.tb_tok_qk):
            row = thr_tmem_load.partition_D(tScS_r2)[0][0] + m_block * self.m_block_size
            if const_expr(self.tb_varlen):
                row = row + q_off
            fq = self._tb_tok_v[0, batch_idx, head_idx, row]  # token-local operands: this query's own factor
        elif const_expr(self.tb_qk):
            row = thr_tmem_load.partition_D(tScS_r2)[0][0] + m_block * self.m_block_size
            if const_expr(self.tb_varlen):
                row = row + q_off
            fq = fac[0, batch_idx, head_idx, row // 32]
        # raw loads only here (a / c / u hold f_K, f_V until _tb_prep): their first use is after the S wait, so the
        # load latency is hidden behind it rather than stalling in front of it
        for g in cutlass.range_constexpr(ngr):
            kb = n_block * ngr + g
            if const_expr(self.tb_varlen and k_blk_off is not None):
                kb = kb + k_blk_off
            if const_expr(self.tb_tok_v):
                u[g] = Float32(1.0)  # token-local operands: V's factor is per key (_tb_tok_v_bias / _tb_tok_v_row_sum)
            else:
                u[g] = fac[2, batch_idx, kvh, kb]
            if const_expr(self.tb_tok_qk):
                a[g] = Float32(1.0)  # token-local operands: f_K is per key (_tb_tok_v_bias)
            elif const_expr(self.tb_qk):
                a[g] = fac[1, batch_idx, kvh, kb]
        c[0] = fq
        return a, c, u

    @cute.jit
    def _tb_prep(self, softmax, a: cute.Tensor, c: cute.Tensor, u: cute.Tensor):
        """After the S wait: a_g = f_Q f_K(g), c_g = log2 f_V(g) / scale_log2, u_g = 1 / f_V(g) (0 for f_V = 0)."""
        ngr = cute.size(u)
        inv_scale = cute.arch.rcp_approx(softmax.scale_log2)
        fq = c[0]
        for g in cutlass.range_constexpr(ngr):
            fv = u[g]
            if const_expr(self.tb_qk):
                # floored at the smallest normal: an all-zero Q or K block (factor 0) has S = 0 exactly, so any
                # positive a_g gives the same S' = c_g, and a masked -inf stays -inf (0 * -inf would be NaN)
                a[g] = cute.arch.fmax(fq * a[g], Float32(1.1754943508222875e-38))
            # fv = 2^(e - 127) or 0: log2 and 1 / fv from the bits with integer ops. The MUFU pipe is the softmax's
            # exp2 pipe: lg2.approx + an IEEE division per group measured +4..8 % of the forward (A2, fp4 / fp8)
            l2, rc = utils.pow2_log2_rcp(fv)
            c[g] = l2 * inv_scale
            u[g] = rc

    @cute.jit
    def _tb_row_max(self, softmax, tSrS: cute.Tensor, a: cute.Tensor, c: cute.Tensor, init_val, gm=None) -> Float32:
        """Row max of the transformed scores S' = a_g S + c_g without transforming S: per 32-key group the raw max
        m_g (the same fmax count as one row max), then max_g(a_g m_g + c_g), exact since a_g > 0. The element order
        of the thread's row slice is the key order (update_row_sum_sage groups it the same way)."""
        frag = cute.logical_divide(tSrS, cute.make_layout(32))
        ngr = cute.size(frag, mode=[1])
        tm = cute.make_rmem_tensor(cute.make_layout(ngr), Float32)
        for g in cutlass.range_constexpr(ngr):
            if const_expr(gm is not None):   # SM103: ld.red's x32 tile max of this group (the same exact max)
                m = gm[g]
            else:
                m = softmax._compute_row_max(frag[None, g].load())
            if const_expr(self.tb_qk):
                tm[g] = a[g] * m + c[g]
            else:
                tm[g] = m + c[g]
        return softmax._compute_row_max(tm.load(), init_val=init_val)

    @cute.jit
    def _tb_update_row_max(self, softmax, row_max_new: Float32, is_first: cutlass.Constexpr[bool]):
        """softmax.update_row_max's tail (fastmath exp2, rescale threshold) on a max already reduced with the old one."""
        row_max = row_max_new if row_max_new != -cutlass.Float32.inf else Float32(0.0)
        acc_scale = Float32(0.0)
        if const_expr(not is_first):
            row_max_old = softmax.row_max[0]
            acc_scale_ = (row_max_old - row_max) * softmax.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if const_expr(softmax.rescale_threshold > 0.0):
                if acc_scale_ >= -softmax.rescale_threshold:
                    row_max_new = row_max_old
                    row_max = row_max_old
                    acc_scale = Float32(1.0)
        softmax.row_max[0] = row_max_new
        return row_max, acc_scale

    @cute.jit
    def _tb_scale_subtract(self, softmax, tSrS: cute.Tensor, a: cute.Tensor, c: cute.Tensor, row_max: Float32):
        """softmax.scale_subtract_rowmax of S' = a_g S + c_g folded into its one FFMA2 per pair:
        S' sl2 + off - M sl2 = S (a_g sl2) + ((c_g - M) sl2 + off). No per-element cost over the plain step."""
        ngr = cute.size(tSrS) // 32
        sl2 = softmax.scale_log2
        off = Float32(softmax.max_offset)
        for g in cutlass.range_constexpr(ngr):
            mul = sl2
            if const_expr(self.tb_qk):
                mul = a[g] * sl2
            bias = (c[g] - row_max) * sl2 + off
            for i in cutlass.range_constexpr(0, 32, 2):
                e = 32 * g + i
                tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (mul, mul), (bias, bias))

    @cute.jit
    def _tb_tok_prefetch(self, batch_idx: Int32, head_idx: Int32, n_block: Int32, k_off: Int32 = None):
        """token-local operands, issued BEFORE the S wait (the latency hides behind it): lane j of each softmax
        warp loads keys 4 j .. 4 j + 3 of this key tile from the token table rows log2 f_V, 1 / f_V (and f_K on the
        8-bit route) -- one coalesced 16-byte load per row per lane."""
        kvh = head_idx // self.qhead_per_kvhead
        lane = cute.arch.lane_idx()
        key0 = n_block * self.n_block_size + 4 * lane
        if const_expr(self.tb_varlen):
            key0 = key0 + k_off
        r = cute.make_rmem_tensor(cute.make_layout(4 * self.tok_rows), Float32)
        rows = (4, 5, 1)
        for i in cutlass.range_constexpr(self.tok_rows):
            if const_expr(self.codes_packed):   # (cov3: a packed key start need not be 4-aligned: scalar loads)
                for e in cutlass.range_constexpr(4):
                    r[4 * i + e] = self._tb_tok_v[rows[i], batch_idx, kvh, key0 + e]
            else:
                cute.autovec_copy(cute.make_tensor(self._tb_tok_v[rows[i], batch_idx, kvh, None].iterator + key0,
                                                   cute.make_layout(4)),
                                  cute.make_tensor(r.iterator + 4 * i, cute.make_layout(4)))
        return r

    @cute.jit
    def _tb_tok_stage(self, r: cute.Tensor):
        """After the S wait: this warp's prefetched per-key values into its SMEM slot (the warp then reads all 128
        keys back as broadcast loads)."""
        lane = cute.arch.lane_idx()
        w = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 8
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(self.tok_rows):
            for e in cutlass.range_constexpr(4):
                self._tok_slot[4 * lane + e, i, w] = r[4 * i + e]
        cute.arch.sync_warp()

    @cute.jit
    def _tok_row32(self, row: cutlass.Constexpr[int], g: cutlass.Constexpr[int]):
        """32 consecutive per-key values (keys 32 g .. 32 g + 31 of the tile) of a staged row, from this warp's slot
        (16-byte broadcast loads)."""
        w = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 8
        f = cute.make_rmem_tensor(32, Float32)
        cute.autovec_copy(cute.make_tensor(self._tok_slot[None, row, w].iterator + 32 * g, cute.make_layout(32)), f)
        return f

    @cute.jit
    def _tb_tok_v_bias(self, softmax, tSrS: cute.Tensor, batch_idx: Int32, head_idx: Int32, n_block: Int32, a_row):
        """token-local operands (fold forward): S_j += log2 f_V(key j) / scale_log2 (masked -inf stays -inf),
        so the scale-subtract's exp2 gives P' = P f_V(key) -- the per-key form of the group bias c_g -- and the online
        row max runs over the biased scores (only visible keys enter it). E4M3 Q / K (8-bit route): S_j f_K(key j)
        first, the bias divided by the row multiplier a_row = f_Q(row) (the scale-subtract multiplies by it). One packed
        FMA per pair; the per-key rows come from this warp's staged slot (SCALING-CONSISTENCY RULE: the backward reads
        the same token-local codes and factors)."""
        inv_scale = cute.arch.rcp_approx(softmax.scale_log2)
        if const_expr(self.tb_tok_qk):
            inv_scale = inv_scale / a_row
        for g in cutlass.range_constexpr(cute.size(tSrS) // 32):
            lv = self._tok_row32(0, g)
            if const_expr(self.tb_tok_qk):
                fk = self._tok_row32(2, g)
                for i in cutlass.range_constexpr(0, 32, 2):
                    e = 32 * g + i
                    b0, b1 = cute.arch.mul_packed_f32x2((lv[i], lv[i + 1]), (inv_scale, inv_scale))
                    tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (fk[i], fk[i + 1]), (b0, b1))
            else:
                for i in cutlass.range_constexpr(0, 32, 2):
                    e = 32 * g + i
                    tSrS[e], tSrS[e + 1] = cute.arch.fma_packed_f32x2((lv[i], lv[i + 1]), (inv_scale, inv_scale),
                                                                      (tSrS[e], tSrS[e + 1]))

    @cute.jit
    def _tb_tok_v_row_sum(self, softmax, tSrS: cute.Tensor, batch_idx: Int32, head_idx: Int32, n_block: Int32,
                          row_scale: Float32, is_first: cutlass.Constexpr[bool]):
        """token-local V: the row sum of P = P' / f_V(key), one packed FMA per pair with 1 / f_V from the staged
        slot (update_row_sum_sage's per-key form)."""
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        for g in cutlass.range_constexpr(cute.size(tSrS) // 32):
            rc = self._tok_row32(1, g)
            for i in cutlass.range_constexpr(0, 32, 2):
                e = 32 * g + i
                acc0, acc1 = cute.arch.fma_packed_f32x2((tSrS[e], tSrS[e + 1]), (rc[i], rc[i + 1]), (acc0, acc1))
        tot = acc0 + acc1
        if const_expr(not is_first):
            tot = tot + softmax.row_sum[0] * row_scale
        softmax.row_sum[0] = tot

    def _vtok_kv(self, head_idx):
        return head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead

    @cute.jit
    def vtok_stager(self, mbar_ptr: cute.Pointer, block_info: BlockInfo, num_splits: Int32, SeqlenInfoCls: Callable,
                    TileSchedulerCls: Callable):
        """vtok_w on the (otherwise empty) warp: walk the load warp's key tiles in its order; per tile wait for the K(t)
        ring slot to be empty (the load warp's producer index / phase: slot t - ring is then no longer read), copy the tile's
        256 floats [A_k, u_k] into SMEM ring slot (t mod ring), and arrive the vtok barrier, which the load warp waits before
        loading K(t). Host traffic leaves the load / MMA / softmax chains. The overwrite guard: before staging tile g >= ring - 1
        the stager consumes one step-done arrival of each softmax stage."""
        count = Int32(0)
        gtot = Int32(0)   # key tiles staged so far
        dph0 = Int32(0)   # softmax(0) / softmax(1) step-done barrier phases
        dph1 = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            kvh = self._vtok_kv(head_idx)
            for i in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - 1 - i
                if gtot >= self.vtok_ring - 1:
                    # slot (gtot mod ring) was last written for tile gtot - ring: wait until both softmax stages finished one more
                    # step, so each has completed tile gtot - ring + 1 or later
                    self._mbar_wait_h(mbar_ptr + self.mbar_vtok_offset + 1, dph0)
                    dph0 ^= 1
                    self._mbar_wait_h(mbar_ptr + self.mbar_vtok_offset + 2, dph1)
                    dph1 ^= 1
                slot = (count + i) % self.vtok_ring
                with cute.arch.elect_one():
                    for c in cutlass.range_constexpr(4):
                        fr = cute.make_rmem_tensor(64, Float32)
                        fr.store(cute.make_tensor(self._vtok_w[batch_idx, kvh, None].iterator + n_block * 256 + 64 * c,
                                                  cute.make_layout(64)).load())
                        cute.make_tensor(self._vtok_smem_t.iterator + slot * 256 + 64 * c, cute.make_layout(64)).store(fr.load())
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_vtok_offset)
                gtot += 1
            count = (count + n_block_max - n_block_min) % self.vtok_ring
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def _vtok_ring_stage(self, batch_idx: Int32, kvh: Int32, n_block: Int32, slot: Int32):
        """vtok_w, load warp: key tile n_block's 256 floats [A_k, u_k] into SMEM ring slot `slot` (its tile count mod ring).
        Called after the K(t) slot-empty wait: slot t was last used by tile t - ring, whose softmax finished long before."""
        with cute.arch.elect_one():
            fr = cute.make_rmem_tensor(256, Float32)
            fr.store(cute.make_tensor(self._vtok_w[batch_idx, kvh, None].iterator + n_block * 256, cute.make_layout(256)).load())
            cute.make_tensor(self._vtok_smem_t.iterator + slot * 256, cute.make_layout(256)).store(fr.load())
            cute.arch.fence_view_async_shared()

    def _vtok_add_bias(self, tSrS_h: cute.Tensor, batch_idx: Int32, head_idx: Int32, n_block: Int32, half: cutlass.Constexpr[int],
                       b_fr=None, kinv=None, koff=None):
        """vtok: S_qk += beta_k over one 64-key half of a split-pipeline S row (masked entries stay -inf). b_fr: the step's 128
        biases from the vtok SMEM slot (vtok_tsf / vtok_w)."""
        if const_expr(self.vtok_tma):
            # vtok_tsf: the slot holds beta'_k = A_k / scale_log2 (host float); S' = S + beta' as packed adds -- the backward
            # adds the same float, so the P^ reproduction is bitwise
            bf = cute.make_rmem_tensor(64, Float32)
            bf.store(cute.make_tensor(b_fr.iterator + 64 * half, cute.make_layout(64)).load())
            for i in range(0, 64, 2):
                tSrS_h[i], tSrS_h[i + 1] = cute.arch.add_packed_f32x2((tSrS_h[i], tSrS_h[i + 1]), (bf[i], bf[i + 1]))
            return
        if const_expr(self.vtok_w):
            # A_k = e_k + 8 from the SMEM slot; S' = S + A / scale_log2 as one packed FMA per pair (the backward's arithmetic)
            bf = cute.make_rmem_tensor(64, Float32)
            bf.store(cute.make_tensor(b_fr.iterator + 64 * half, cute.make_layout(64)).load())
            for i in range(0, 64, 2):
                tSrS_h[i], tSrS_h[i + 1] = utils.fma_packed_f32x2((bf[i], bf[i + 1]), (kinv, kinv), (tSrS_h[i], tSrS_h[i + 1]))
            return
        if const_expr(b_fr is not None):
            for i in range(64):
                tSrS_h[i] = tSrS_h[i] + b_fr[64 * half + i]
            return
        kvh = self._vtok_kv(head_idx)
        base = n_block * self.n_block_size + 64 * half
        if const_expr(koff is not None):   # (cov5 packed rows: (1, h_kv, T') tables at the packed key)
            base = base + koff
            batch_idx = Int32(0)
        if const_expr(self.vtok_bulk):
            # one bulk load of the half's 64 biases into a register fragment (vector loads instead of 64 scalar loads)
            bview = cute.make_tensor(self._vtok_bias[batch_idx, kvh, None].iterator + base, cute.make_layout(64))
            bfrag = cute.make_rmem_tensor(64, Float32)
            bfrag.store(bview.load())
            for i in range(64):
                tSrS_h[i] = tSrS_h[i] + bfrag[i]
            return
        for i in range(64):  # trace-time unrolled (plain helper: range_constexpr is jit-only)
            tSrS_h[i] = tSrS_h[i] + self._vtok_bias[batch_idx, kvh, base + i]

    @cute.jit
    def _ex2_emu_pair(self, x: Float32, y: Float32, clamp: cutlass.Constexpr[bool] = False,
                      poly: cutlass.Constexpr[int] = 0) -> Tuple[Float32, Float32]:
        """FMA-pipe exp2 of a pair (E2E_POLY_*). Inputs are log2-domain P arguments
        (<= log2 6); the degree-2 form has no clamp (rel err 1.7e-3, fine for E2M1 codes
        and the row sum) -- except in masked steps (`clamp`), where -inf entries would
        turn into NaN through the magic-number floor (causal / seqlen-boundary tiles)."""
        # `e2e-poly2-unclamped-range` (found by adversarial input gates): the degree-2 form's magic-number floor
        # wraps for x < -127 -- a score more than ~88 nats below its group max, which trained models reach (byte-GPT
        # capture, step 1250, layer 0: max |s| scale ~150 nats) -- and returns garbage instead of 0: the MXFP4 route
        # under the exact guard (clamp 2 -> exp2 split step) gave cos(O) -0.01, |O| x 739 and NaN LSE there. The clamp
        # (2 FMNMX per emulated pair) is a no-op for x >= -126 (bitwise on every in-range input); E2E_CLAMP=0
        # restores the unclamped form.
        x = cute.arch.fmax(x, Float32(-126.0))
        y = cute.arch.fmax(y, Float32(-126.0))
        if const_expr((poly if poly else self.fp4_e2e_poly) == 2):
            MAGIC = 12582912.0  # 1.5 * 2^23: round-to-nearest integer in the low mantissa bits
            C0, C1, C2 = 1.0004426738244065, 0.7034358772399556, 0.23842222937876753
            t0, t1 = cute.arch.add_packed_f32x2((x, y), (Float32(MAGIC), Float32(MAGIC)))
            u0, u1 = cute.arch.add_packed_f32x2((t0, t1), (Float32(-MAGIC), Float32(-MAGIC)))
            f0, f1 = cute.arch.fma_packed_f32x2((u0, u1), (Float32(-1.0), Float32(-1.0)), (x, y))
            p0, p1 = cute.arch.fma_packed_f32x2((f0, f1), (Float32(C2), Float32(C2)), (Float32(C1), Float32(C1)))
            p0, p1 = cute.arch.fma_packed_f32x2((p0, p1), (f0, f1), (Float32(C0), Float32(C0)))
            return utils.combine_int_frac_ex2(t0, p0), utils.combine_int_frac_ex2(t1, p1)
        else:
            return utils.ex2_emulation_2(x, y)

    def _directp_group(
        self,
        frag,        # (sf_vec, ng) raw scores of one half (register fragment view)
        g,           # trace-time group index within the half
        scale2,
        c2,
        tSrP_frag,   # (sf_vec, 2*ng) fp4 view of the P store fragment
        pg,          # trace-time group index within the tile
        masked=False,  # fp4all masked step: the classifier clamp is forced on (-inf scores)
        vu=None,     # vtok: (batch_idx, kv head, first key of this group) -> Mitchell values weighted by u_k in the row sum
        fn=None,     # p_mix: this group position's value function ('M' / 'E' / '2' / '3'); None = the configuration's
    ):
        """Direct-P quantization of one SF group (direct_p): packed magic FFMA,
        LEA + F2FP.RELU pack, row-sum part from the codes (2x the quantized values) or from
        the Mitchell values (diagnostic). Returns this group's (unscaled) row-sum part."""
        sf_vec = self.sf_vec_size_pv
        clamp = self.directp_clamp   # always 2: the clamped classifier (exact for any score, incl. masked -inf)
        mit = (fn == "M") if fn is not None else not self.fp4all_exp2   # the Mitchell classifier for this group
        for j in range(0, sf_vec, 2):
            frag[j, g], frag[j + 1, g] = utils.fma_packed_f32x2(
                (frag[j, g], frag[j + 1, g]), (scale2, scale2), (c2, c2),
            )
        pv_u32 = cute.recast_tensor(tSrP_frag[None, pg], cute.Int32)
        if const_expr(self.directp_rowsum == "codes" and mit and vu is None):
            # LEA fused into the pack; row-sum part from the packed codes (PRMT LUT + DP4A)
            isum = Int32(0)
            for k in range(cute.size(pv_u32, mode=[0])):
                if const_expr(clamp or self.directp_window != 64):
                    pv_u32[k] = packed_mitchell_to_e2m1_w(
                        frag[k * 8, g], frag[k * 8 + 1, g], frag[k * 8 + 2, g], frag[k * 8 + 3, g],
                        frag[k * 8 + 4, g], frag[k * 8 + 5, g], frag[k * 8 + 6, g], frag[k * 8 + 7, g],
                        self.directp_lea, clamp,
                    )
                else:
                    pv_u32[k] = sm100_utils.packed_mitchell_to_e2m1(
                        frag[k * 8, g], frag[k * 8 + 1, g], frag[k * 8 + 2, g], frag[k * 8 + 3, g],
                        frag[k * 8 + 4, g], frag[k * 8 + 5, g], frag[k * 8 + 6, g], frag[k * 8 + 7, g],
                    )
                isum = sm100_utils.e2m1x8_value_sum2(pv_u32[k], isum)
            return sm100_utils.i32_to_f32(isum)
        else:
            # Mitchell values materialized once (one LEA each): F2FP.RELU packs them, the FADD2
            # tree sums them. With delta = log2(1.25) - 0.25 the values average 0.99-1.01 x 2^x
            # (uniform vs peaked fraction distribution), so no bias correction is applied.
            if const_expr(fn is not None and not mit):
                # p_mix: one value function for the whole group -- 'E' ex2.approx (MUFU), '2' / '3' the clamped FMA-pipe
                # polynomial (always clamped: x >= -126, so -inf and far-below-max scores give 0 exactly as ex2 does)
                if const_expr(fn == "E"):
                    for j in range(sf_vec):
                        frag[j, g] = cute.arch.exp2(frag[j, g])
                else:
                    for j in range(0, sf_vec, 2):
                        frag[j, g], frag[j + 1, g] = self._ex2_emu_pair(frag[j, g], frag[j + 1, g], clamp=True,
                                                                        poly=int(fn))
            elif const_expr(self.fp4all_exp2):
                # fp4all_exp2: FA4's exp2 path. ex2.approx on the MUFU pipe, one pair in every
                # fp4_log2_e2e_freq emulated on the FMA pipe so the two pipes can be balanced (the
                # classifier FFMA above already produced s*scale + log2 6 - gm*scale).
                e2e = const_expr(self.fp4_log2_e2e_freq)
                for j in range(0, sf_vec, 2):
                    _pair = g * (sf_vec // 2) + j // 2
                    if const_expr(e2e > 0 and _pair % e2e == 0):
                        # masked steps carry -inf entries: the emulation's magic floor needs the clamp
                        frag[j, g], frag[j + 1, g] = self._ex2_emu_pair(
                            frag[j, g], frag[j + 1, g], clamp=const_expr(bool(masked)))
                    else:
                        frag[j, g] = cute.arch.exp2(frag[j, g])
                        frag[j + 1, g] = cute.arch.exp2(frag[j + 1, g])
            else:
                for j in range(sf_vec):
                    frag[j, g] = mitchell_value_w(frag[j, g], self.directp_lea, clamp)
            pack_fn = sm100_utils.packed_float_to_e2m1_relu
            for k in range(cute.size(pv_u32, mode=[0])):
                pv_u32[k] = pack_fn(
                    frag[k * 8, g], frag[k * 8 + 1, g], frag[k * 8 + 2, g], frag[k * 8 + 3, g],
                    frag[k * 8 + 4, g], frag[k * 8 + 5, g], frag[k * 8 + 6, g], frag[k * 8 + 7, g],
                )
            if const_expr(self.directp_rowsum == "codes"):
                # the codes row sum on the exp2 value function and / or the key-local weights (`reproduce-forward-p`):
                # 2 x the represented values (the caller halves it), each times u_k on a key-local step
                if const_expr(vu is None):
                    isum = Int32(0)
                    for k in range(cute.size(pv_u32, mode=[0])):
                        isum = sm100_utils.e2m1x8_value_sum2(pv_u32[k], isum)
                    return sm100_utils.i32_to_f32(isum)
                v_b, v_kvh, v_base, v_fr, v_rel = vu[:5]
                assert v_fr is None, "repro: the plain vtok_u gathers"
                csum = Float32(0.0)
                for k in range(cute.size(pv_u32, mode=[0])):
                    vals = e2m1x8_values2(pv_u32[k])
                    for i in range(8):
                        csum = csum + vals[i] * self._vtok_u[v_b, v_kvh, v_base + 8 * k + i]
                return csum
            if const_expr(vu is not None):
                v_b, v_kvh, v_base, v_fr, v_rel = vu[:5]
                v_pad = vu[5] if const_expr(len(vu) > 5) else None
                if const_expr(v_pad is not None):
                    # (cov5 packed rows) keys past the document's end: the aligned layout's padding weight
                    for j in range(sf_vec):
                        frag[j, g] = frag[j, g] * cutlass.select_(Int32(v_rel + j) < v_pad[0],
                                                                  self._vtok_u[v_b, v_kvh, v_base + j], v_pad[1])
                elif const_expr(v_fr is not None and self.vtok_tma):
                    for j in range(sf_vec):  # v_fr: this half's 64 weights, already in registers
                        frag[j, g] = frag[j, g] * v_fr[v_rel + j]
                elif const_expr(v_fr is not None and self.vtok_w):
                    uf = cute.make_rmem_tensor(sf_vec, Float32)
                    uf.store(cute.make_tensor(v_fr.iterator + v_rel, cute.make_layout(sf_vec)).load())
                    for j in range(sf_vec):
                        frag[j, g] = frag[j, g] * uf[j]
                else:
                    for j in range(sf_vec):
                        frag[j, g] = frag[j, g] * self._vtok_u[v_b, v_kvh, v_base + j]
            return utils.fadd_reduce(frag[None, g].load(), None, arch=100)

    # split step, masked tiles: group max used for all-masked 16-key groups (finite; -1e9 * scale ~ -1.3e8 log2 units)
    # (the all-4-bit backward's gm_floor is the same constant)
    MASKED_GROUP_MAX_PLACEHOLDER = SPLIT_MASKED_GM
    # `reproduce-forward-p`: frame changes recorded per query row (mRepro words [3, 3 + R)); a row with more changes is
    # counted in word 2 and the backward cannot reproduce its earliest tiles (none seen: <= 4 on trained captures)
    REPRO_RECORDS = 5
    REPRO_WORDS = 3 + REPRO_RECORDS

    def _repro_max_offset(self) -> float:
        """The NVFP4 two-level frame offset as softmax_step's caller builds it (Python float; `reproduce-forward-p`)."""
        mo = math.log2(6) + (math.log2(448) if self.fp4_pv_two_level else 0.0)
        thr = self._fp4_pv_rescale_threshold()
        if thr > 0.0 and self.fp4_pv_two_level:
            mo -= thr
        return mo

    def _rp_bm(self, batch_idx, mt, seqlen):
        """`reproduce-forward-p` export coordinates (batch, query tile) of query tile mt: under aligned varlen documents
        the b = 1 buffer's tile (the document's first tile + mt; documents start on 128-row tiles)."""
        if const_expr(self.codes_packed):
            # (cov5 packed rows: (h, W, T') per packed query row -- the tile's first packed row and its real rows)
            return self._cu_q[batch_idx] + mt * self.m_block_size, seqlen.seqlen_q - mt * self.m_block_size
        if const_expr(seqlen.has_cu_seqlens_q):
            return Int32(0), mt + seqlen.offset_q // self.m_block_size
        return batch_idx, mt

    @cute.jit
    def _rp_put(self, rp_b, rp_m, head_idx, w: cutlass.Constexpr[int], row, val):
        """One `reproduce-forward-p` export word (cov5: the packed rows' (h, W, T') layout, real rows only)."""
        rp_t = self._repro
        if const_expr(self.codes_packed):
            if row < rp_m:
                rp_t[head_idx, w, rp_b + row] = val
        else:
            rp_t[rp_b, head_idx, rp_m, w, row] = val

    @cute.jit
    def _repro_finish(self, batch_idx, mt, head_idx, row, frame_rm, row_sum):
        """`reproduce-forward-p`: the row's final frame F, row sum Z (bits), change count, and the unused record slots
        (n_block 32767: never before any tile in processing order)."""
        cnt = Int32(frame_rm[1])
        self._rp_put(batch_idx, mt, head_idx, 0, row, Int32(frame_rm[0]))
        from flash_attn.cute.pv4_repro import div_rn
        self._rp_put(batch_idx, mt, head_idx, 1, row, f32_bits(div_rn(Float32(1.0), row_sum)))   # 1 / Z, IEEE
        self._rp_put(batch_idx, mt, head_idx, 2, row, cnt)
        for j in cutlass.range_constexpr(self.REPRO_RECORDS):
            if Int32(j) >= cnt:
                self._rp_put(batch_idx, mt, head_idx, 3 + j, row, Int32(32767 << 16))

    @cute.jit
    def _mx_floor(self, x: Float32) -> Float32:
        """floor(x) for the MXFP4 group constants (UE8M0 SFs need integer exponents): -ceil(-x)."""
        return -sm100_utils.ceil_f32(-x)

    def _lse_log2_offset(self) -> float:
        """log2 bias carried by the softmax row sum on the NVFP4 PV path (trace-time constant): the group SFs
        are exp2(max_offset - row_max*scale - c_g), so row_sum = 2^max_offset * sum 2^((s - row_max)*scale)
        and LSE must subtract max_offset (= log2 6 + log2 448 - rescale threshold = 3.39 by default; measured
        LSE bias +2.351 nats on both the split and non-split pipes before this correction). MXFP4 PV (E2M1 + UE8M0) runs
        the same log2 group quantisation and carries the same bias (measured +2.3506 nats, std 0.012). Other PV modes: 0
        (MXFP8 PV not verified here). Non-block-scaled E4M3 PV: row_sum carries 2^fp8_pv_log2_offset
        (P^ = E4M3(P * 2^offset)), so the LSE subtracts that offset -- the same attribute the softmax
        object was created with, never a second read of the knob."""
        if self.p_rowframe8:
            return self.fp8_pv_log2_offset
        if self.quant_pv and self.v_dtype.width == 8 and not self.mx_pv_split:
            # MXFP8 PV (token-block scales): the step folds _max_offset = log2 448 into the P^ range (row max at 448, the
            # UE8M0 group SFs absorb the rest), so row_sum carries 2^log2(448) and the LSE must subtract it. Measured
            # before this: every gradient of a backward fed this LSE came back x 1/448 (norm ratio 0.0022, cosine intact).
            return math.log2(448.0)
        if not (self.quant_pv and self.v_dtype.width == 4):
            return 0.0
        if self.mx_pv_split:
            # integer frame: row_sum is 2^(max_offset - F) sum 2^(s scale - ...), LSE = (F + log2 Z - max_offset) ln 2,
            # with max_offset = MX_INT_FRAME_OFFSET, the same constant the group SFs use (see its comment).
            return float(self.MX_INT_FRAME_OFFSET)
        off = math.log2(6) + (math.log2(448) if self.fp4_pv_two_level else 0.0)
        thr = self._fp4_pv_rescale_threshold()
        if thr > 0.0 and self.fp4_pv_two_level:
            off -= thr
        return off

    def _s2t_or_skip(self, tc, tCs, idx, dst):
        """One SF s2t copy (nothing when tc is None).
        Used through functools.partial from mma_split, where a closure is not allowed."""
        if const_expr(tc is not None):
            cute.copy(tc, tCs[None, None, None, None, idx], dst)

    def _softmax_step_fn(self):
        """The softmax step for this configuration (plain Python, resolved at trace time): the
        split-QK step for block-scaled P^ (E2M1) or the generic step."""
        if not self.qk_split_pipe:
            return self.softmax_step
        assert self.quant_pv, "the split-QK pipeline hosts block-scaled P^ only"
        return self.softmax_step_split

    def _gmax16(self, x):
        """max of 16 values with 8 three-input FMNMX3 (depth 3)."""
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
    def softmax_step_split(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.atom.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tCtSFP: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
        mask_is_noop: cutlass.Constexpr[bool] = False,
        thr_tmem_load_h=None,
        tStS_t2r_a: Optional[cute.Tensor] = None,
        tStS_t2r_b: Optional[cute.Tensor] = None,
        tScSh: Optional[cute.Tensor] = None,
        tSrS_a_persist: Optional[cute.Tensor] = None,
        split_masked: cutlass.Constexpr[bool] = False,
        split_mask_seqlen: cutlass.Constexpr[bool] = False,
        pmax_rm: Optional[cute.Tensor] = None,
        frame_rm: Optional[cute.Tensor] = None,
        vt_c0: Optional[Int32] = None,  # vtok_w ring: this step's slot = (vt_c0 - n_block) mod ring
        tf_drop_raw: Optional[Float32] = None,  # `per-tile-row-frame`: fp4_pv_tile_frame_drop / scale_log2
    ):
        """Split-QK softmax step (qk_split_pipe, NVFP4/MXFP4 PV, group-max-first log2
        quant, no masking). a-half: [load (prefetched by the previous step), group maxes,
        (b-half load issued), exps, packs]; b-half: [group maxes, row max + correction
        signal, SFs, SFP + P k-block 0 hand-off, exps, packs]; row sum; P k-block 1;
        prefetch of the next tile's a-half."""
        assert self.quant_pv and self.v_dtype.width == 4 and self.fp4_pv_log2_quant
        assert self.corr_empty_early
        # UE8M0 group SFs (MXFP4 PV) need an integer exponent: mx_pv_split floors the per-group constant and folds the
        # integer frame instead of the raw row max (see __init__); UE4M3 SFs keep the exact fold.
        assert self.sf_dtype_pv != Float8E8M0FNU or self.mx_pv_split
        sf_vec = self.sf_vec_size_pv
        scale = softmax.scale_log2
        LOG2_PMAX = math.log2(6.0)
        # masked variant (causal diagonal tiles, ragged first tile): the mask is applied to each half after its
        # TMEM load, the E2M1 codes come from exp2 (Direct-P's magic floor and the FMA-pipe exp2 both need finite
        # arguments), all-masked groups get a finite group max; everything else is the unmasked step
        masked = const_expr(split_masked and (self.is_causal or self.is_local or (split_mask_seqlen and self.qk_split_mask_seqlen)))
        if const_expr(masked):
            assert not self.is_local, "split pipeline: local masks not supported"
        e2e_freq = self.fp4_log2_e2e_freq if const_expr(not masked) else 0
        thread_idx = thr_tmem_load.thr_idx
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        full_layout = cute.make_layout(thr_tmem_load.partition_D(tScS).shape)
        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP = cute.make_tensor(cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype), full_layout)
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(sf_vec))
        ng_h = 64 // sf_vec
        cg = cute.make_rmem_tensor(cute.make_layout(2 * ng_h), Float32)
        gsum_a = cute.make_rmem_tensor(cute.make_layout(ng_h), Float32)
        sf_b = cute.make_rmem_tensor(cute.make_layout(ng_h), Float32)
        sfw = cute.make_rmem_tensor(cute.make_layout(2 * ng_h), self.sf_dtype_pv)
        # fp4all: Direct-P on masked steps too (the backward reproduces one classifier); the group maxes entering the
        # classifier constants and the SFs are clamped to a finite placeholder (all-masked group: SF exp2(-1e9 scale) = 0)
        fp4all = const_expr(self.fp4all)
        mx = const_expr(self.mx_pv_split)  # UE8M0 group SFs: integer exponents (floored cg + integer frame)
        intf = const_expr(self.int_frame)
        tile_frame = const_expr(self.fp4_pv_tile_frame)  # `per-tile-row-frame` (see __init__)
        # MXFP4 keeps Direct-P (mxfp4_directp=0 falls back to exp2). Under the same guard it is the faster path --
        # 3.55 vs 4.53 ms at s32768 c with the window guard; the earlier "slower" reading compared Direct-P under the exact
        # clamp (4.91) against exp2, which needs no guard at all.
        # fp4all_exp2: the same lean structure, value = ex2.approx instead of Mitchell (see _directp_group).
        # It is an fp4all variant and nothing else: `self.fp4all and`. Without that conjunct the flag leaked into
        # every non-fp4all split step whenever Direct-P was off (its default was `use_exp2 and not use_direct_p`), which
        # took the lean structure with a DP_K = 1 value function. NVFP4 survived it; MXFP4 did not, because the lean
        # structure's floored group constants and integer row frame are derived from the 2^23-framed Mitchell classifier
        # (mx_pv_split, int_frame) and are meaningless at DP_K = 1: measured cos(O) 0.776 / 0.0 / 0.685 / 0.0 at
        # A1 / A2 / A3 / A4 against 0.989 / 0.987 / 0.989 / 0.987. With the conjunct, Direct-P off falls through to the
        # generic exp2 path, which is the FP4 tree's `nodirectp` behaviour and is correct.
        dpx = const_expr(self.fp4all and self.fp4all_exp2)
        mx_dp = self.mx_directp
        directp = const_expr((self.qk_split_directp or dpx) and (not masked or fp4all) and (not mx or mx_dp))
        GM_FLOOR = Float32(self.MASKED_GROUP_MAX_PLACEHOLDER)
        if const_expr(masked):
            tScSh_t2r = thr_tmem_load_h.partition_D(tScSh)  # (row, col) coordinates of the half's fragment
            row_idx = tScSh_t2r[0][0] + m_block * self.m_block_size
            causal_row_offset = seqlen.seqlen_k - n_block * self.n_block_size - seqlen.seqlen_q
            if const_expr(self.is_causal):
                col_limit = row_idx + causal_row_offset + 1
            else:
                col_limit = Int32(self.n_block_size)
            if const_expr(split_mask_seqlen):
                col_limit = cutlass.min(col_limit, seqlen.seqlen_k - n_block * self.n_block_size)
        if const_expr(directp):
            # Direct-P: t = s*scale2 + cg2[g] with the group's magic constant; see
            # sm100_utils.packed_mitchell_to_e2m1.
            # exp2 variant: DP_K = 1 and no magic constant, so c2 = log2 6 - gm*scale is exactly the exp2 offset
            DP_K = 1.0 if const_expr(dpx) else float(1 << sm100_utils.DIRECTP_FRAC_BITS)
            scale2 = scale * DP_K
            c_dp = (Float32(LOG2_PMAX) if const_expr(dpx) else
                    Float32(LOG2_PMAX * DP_K + self.directp_mag - self.directp_delta * DP_K))  # window MAG (DIRECTP_WINDOW)
            c_dp_mx = Float32(self.directp_mag - self.directp_delta * DP_K)  # MXFP4: cg (floored) carries the log2 6 - gm term
            cg2 = cute.make_rmem_tensor(cute.make_layout(2 * ng_h), Float32)
            dp_codes_sum = const_expr(self.directp_rowsum == "codes")
            # p_mix: per group position, the Mitchell classifier's constants (scale 2^16, c_dp with the window MAG) or the
            # exp2 argument's (scale, log2 6); the group constant c2 and the per-element FFMA take the group's pair
            pmx = self.p_mix
            scale2_e = scale
            c_dp_e = Float32(LOG2_PMAX)
        lean = const_expr(directp and self.split_lean and not mx)  # lean keeps only the group maxes; MXFP4 needs the floored cg
        if const_expr(lean):
            assert sf_vec == 16
            gmv = cute.make_rmem_tensor(cute.make_layout(2 * ng_h), Float32)

        # diagonal-only-key-local-v-scale: on an unmasked step the V payload is the fast key-grouped one, whose group
        # scales are already causally sound (every key of the tile is visible to every query of the tile), so the
        # per-key bias / row-sum weight is not just unnecessary but wrong there -- it is compiled out.
        vtok = const_expr(self.vtok and (masked or not self.vtok_diag))
        vb_fr = None
        vu_fr = None
        # (aligned varlen documents: the per-key tensors are document-major, (n_docs, h_kv, L) -- quant.fp4_diagonal --
        # so [batch_idx = the document, kv head, local key] is the batched indexing exactly)
        vt_b = batch_idx
        vt_koff = None
        vt_pad = None
        if const_expr(vtok):
            vt_kvh = self._vtok_kv(head_idx)
            vt_base = n_block * self.n_block_size
            if const_expr(self.codes_packed):
                # (cov5 packed rows: the per-key tables are (1, h_kv, T') at the packed key -- the document's start)
                vt_koff = self._cu_kc[batch_idx]
                vt_base = vt_base + vt_koff
                vt_b = Int32(0)
                if const_expr(self.directp_rowsum != "codes"):
                    # the Mitchell / exp2 row sum weights every key's value (a masked key's is not 0) by u_k: the keys
                    # past the document's end are the aligned layout's zero rows there, u = 2^(120 + c_n) (e_k = -120,
                    # the tile's constant c_n) -- here the next document's keys, so their weight is substituted
                    from flash_attn.cute.quant.fp4 import e4m3_byte_value
                    _wd = self._v4_ctile[vt_kvh, seqlen.offset_k // self.n_block_size + n_block]
                    vt_pad = (seqlen.seqlen_k - n_block * self.n_block_size,
                              e4m3_byte_value(_wd, 0) * Float32(1.329227995784916e+36))   # 2^c_n 2^120
            if const_expr(self.vtok_tma):
                vt_slot = (1 + 2 * ((vt_c0 - n_block) % (self.kv_stage // 2))) % self.kv_stage  # V(n_block)'s kv ring slot
                vt_u8 = self._vtok_sf_base + vt_slot * self.vtok_sf_slot_bytes
                # Every slot is `vtok_sf_slot_bytes` (1024) from a 1024-byte-aligned base, so each
                # one is 16-byte aligned and the 64-float reads below vectorize. On masked steps the
                # DSL cannot prove that through the dynamic slot index and falls back to align<1>,
                # which makes the vectorized load derive a zero-width shape; state it explicitly.
                _vb = cute.recast_ptr(vt_u8, dtype=Float32).align(16)
                _vu = cute.recast_ptr(vt_u8 + 512, dtype=Float32).align(16)
                vb_fr = cute.make_tensor(_vb, cute.make_layout(128))
                vu_fr = cute.make_tensor(_vu, cute.make_layout(128))
            elif const_expr(self.vtok_w and not self.vtok_tma):
                vt_off = ((vt_c0 - n_block) % self.vtok_ring) * 256  # staged by the load warp with K(n_block)
                vb_fr = cute.make_tensor(self._vtok_smem_t.iterator + vt_off, cute.make_layout(128))
                vu_fr = cute.make_tensor(self._vtok_smem_t.iterator + vt_off + 128, cute.make_layout(128))

        # ---------------- a-half: S cols [0, 64)
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sa_full_offset + stage, mma_si_consumer_phase)
        tSrS_a = cute.make_rmem_tensor(thr_tmem_load_h.partition_D(tScSh).shape, Float32)
        # SM103: on an unmasked step without a per-key bias the raw group maxes are the fused loads' (ld.red x<sf_vec>:
        # each group's values + their max; max is exact, so the same value as the FMNMX tree's)
        hw_split = const_expr(self.use_ldred_rowmax and not masked and not vtok and sf_vec in (16, 32)
                              and self.score_mod is None)
        gm_hw_a = None
        gm_hw_b = None
        if const_expr(hw_split):
            gm_hw_a = cute.make_rmem_tensor(cute.make_layout(ng_h), Float32)
            gm_hw_b = cute.make_rmem_tensor(cute.make_layout(ng_h), Float32)
            sm100_utils.tmem_ld_red_max_n(tStS_t2r_a, tSrS_a, gm_hw_a, sf_vec)
        else:
            cute.copy(thr_tmem_load_h, tStS_t2r_a, tSrS_a)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sa_loaded_offset + stage)
        if const_expr(masked):
            self._split_mask_half(tSrS_a, tScSh_t2r, 0, col_limit)
        vu_h = vu_fr
        if const_expr(vtok):
            self._vtok_add_bias(tSrS_a, batch_idx, head_idx, n_block, 0, vb_fr,
                                kinv=Float32(1.0) / scale if const_expr(self.vtok_w) else None, koff=vt_koff)
            if const_expr(self.vtok_tma):
                vu_h = cute.make_rmem_tensor(64, Float32)
                vu_h.store(cute.make_tensor(vu_fr.iterator, cute.make_layout(64)).load())
        frag_a = cute.logical_divide(tSrS_a, cute.make_layout(sf_vec))
        rmax_a = Float32(-cutlass.Float32.inf)
        for g in cutlass.range_constexpr(ng_h):
            if const_expr(lean):
                gm_g = gm_hw_a[g] if const_expr(hw_split) else self._gmax16(frag_a[None, g].load())
                if const_expr(fp4all and masked):
                    gm_g = cute.arch.fmax(gm_g, GM_FLOOR)
                gmv[g] = gm_g
            else:
                gm_g = gm_hw_a[g] if const_expr(hw_split) else softmax._compute_row_max(frag_a[None, g].load())
                if const_expr(masked):
                    # all-masked group: finite c far above every score, so its SF exp2(k_row - c) is exactly 0. The old
                    # placeholder 0 gave SF = exp2(max_offset - log2 6 - row_max * scale), which overflows to inf once the
                    # step's row max is < ~-127 log2 units (a masked diagonal tile whose few visible keys score < -88 nats):
                    # inf * 0 (the group's zero row-sum part) = NaN row sum -> LSE -inf and O left unscaled (|O| 3e4)
                    gm_c = gm_g if gm_g != -cutlass.Float32.inf else Float32(self.MASKED_GROUP_MAX_PLACEHOLDER)
                    _cg = fma_rn_f32(-gm_c, scale, Float32(LOG2_PMAX))   # LOG2_PMAX - gm_c * scale, one rounding (pinned)
                    cg[g] = self._mx_floor(_cg) if const_expr(mx) else _cg
                else:
                    # unmasked step: gm_g is finite. MXFP4: floor so k_row - cg is an integer (codes peak in (3, 6])
                    _cg = fma_rn_f32(-gm_g, scale, Float32(LOG2_PMAX))
                    cg[g] = self._mx_floor(_cg) if const_expr(mx) else _cg
                rmax_a = cute.arch.fmax(rmax_a, gm_g)
            if const_expr(directp):
                # MXFP4: the classifier constant must use the same floored group constant as the UE8M0 SF exponent
                if const_expr(pmx is not None and pmx[g] != "M"):
                    cg2[g] = fma_rn_f32(-gm_g, scale2_e, c_dp_e)
                else:
                    cg2[g] = cg[g] * DP_K + c_dp_mx if const_expr(mx) else (fma_rn_f32(-gm_g, scale2, c_dp) if const_expr(self.repro or self.fp4all) else c_dp - gm_g * scale2)
        tSrS_b = cute.make_rmem_tensor(thr_tmem_load_h.partition_D(tScSh).shape, Float32)
        bpos = 0 if self.qk_split_bprefetch else ng_h
        if const_expr(bpos == 0):
            # b-half load in flight during the a-half exps
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
            if const_expr(hw_split):
                sm100_utils.tmem_ld_red_max_n(tStS_t2r_b, tSrS_b, gm_hw_b, sf_vec)
            else:
                cute.copy(thr_tmem_load_h, tStS_t2r_b, tSrS_b)
        for g in cutlass.range_constexpr(ng_h):
            if const_expr(directp):
                gsum_a[g] = self._directp_group(frag_a, g,
                                                scale2_e if const_expr(pmx is not None and pmx[g] != "M") else scale2,
                                                cg2[g], tSrP_frag, g, masked=masked,
                                                vu=(vt_b, vt_kvh, vt_base + g * sf_vec, vu_h, g * sf_vec, vt_pad) if const_expr(vtok) else None,
                                                fn=pmx[g] if const_expr(pmx is not None) else None)
            else:
                c = cg[g]
                for j in cutlass.range_constexpr(0, sf_vec, 2):
                    frag_a[j, g], frag_a[j + 1, g] = utils.fma_packed_f32x2(
                        (frag_a[j, g], frag_a[j + 1, g]), (scale, scale), (c, c),
                    )
                for j in cutlass.range_constexpr(0, sf_vec, 2):
                    _pair = g * (sf_vec // 2) + j // 2
                    if const_expr(e2e_freq > 0 and _pair % e2e_freq == 0):
                        frag_a[j, g], frag_a[j + 1, g] = self._ex2_emu_pair(frag_a[j, g], frag_a[j + 1, g])
                    else:
                        frag_a[j, g] = cute.arch.exp2(frag_a[j, g])
                        frag_a[j + 1, g] = cute.arch.exp2(frag_a[j + 1, g])
                gsum_a[g] = softmax._compute_row_sum(frag_a[None, g].load())
                pv_u32 = cute.recast_tensor(tSrP_frag[None, g], cute.Int32)
                for k in cutlass.range_constexpr(0, cute.size(pv_u32, mode=[0])):
                    pv_u32[k] = packed_float_to_e2m1(
                        frag_a[k * 8, g], frag_a[k * 8 + 1, g], frag_a[k * 8 + 2, g], frag_a[k * 8 + 3, g],
                        frag_a[k * 8 + 4, g], frag_a[k * 8 + 5, g], frag_a[k * 8 + 6, g], frag_a[k * 8 + 7, g],
                    )

        # ---------------- b-half: S cols [64, 128)
        if const_expr(bpos >= ng_h):
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
            if const_expr(hw_split):
                sm100_utils.tmem_ld_red_max_n(tStS_t2r_b, tSrS_b, gm_hw_b, sf_vec)
            else:
                cute.copy(thr_tmem_load_h, tStS_t2r_b, tSrS_b)
        cute.arch.fence_view_async_tmem_load()
        # the whole S(t) is in registers: the MMA may issue QK_a(t+1) (own bank, cols [0,64))
        self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset + stage)
        if const_expr(masked):
            self._split_mask_half(tSrS_b, tScSh_t2r, 1, col_limit)
        if const_expr(vtok):
            self._vtok_add_bias(tSrS_b, batch_idx, head_idx, n_block, 1, vb_fr,
                                kinv=Float32(1.0) / scale if const_expr(self.vtok_w) else None, koff=vt_koff)
            if const_expr(self.vtok_tma):
                vu_h = cute.make_rmem_tensor(64, Float32)
                vu_h.store(cute.make_tensor(vu_fr.iterator + 64, cute.make_layout(64)).load())
        frag_b = cute.logical_divide(tSrS_b, cute.make_layout(sf_vec))
        rmax_b = Float32(-cutlass.Float32.inf)
        for g in cutlass.range_constexpr(ng_h):
            if const_expr(lean):
                gm_g = gm_hw_b[g] if const_expr(hw_split) else self._gmax16(frag_b[None, g].load())
                if const_expr(fp4all and masked):
                    gm_g = cute.arch.fmax(gm_g, GM_FLOOR)
                gmv[ng_h + g] = gm_g
            else:
                gm_g = gm_hw_b[g] if const_expr(hw_split) else softmax._compute_row_max(frag_b[None, g].load())
                if const_expr(masked):
                    gm_c = gm_g if gm_g != -cutlass.Float32.inf else Float32(self.MASKED_GROUP_MAX_PLACEHOLDER)
                    _cg = fma_rn_f32(-gm_c, scale, Float32(LOG2_PMAX))
                    cg[ng_h + g] = self._mx_floor(_cg) if const_expr(mx) else _cg
                else:
                    _cg = fma_rn_f32(-gm_g, scale, Float32(LOG2_PMAX))
                    cg[ng_h + g] = self._mx_floor(_cg) if const_expr(mx) else _cg
                rmax_b = cute.arch.fmax(rmax_b, gm_g)
            if const_expr(directp):
                if const_expr(pmx is not None and pmx[ng_h + g] != "M"):
                    cg2[ng_h + g] = fma_rn_f32(-gm_g, scale2_e, c_dp_e)
                else:
                    cg2[ng_h + g] = cg[ng_h + g] * DP_K + c_dp_mx if const_expr(mx) else (fma_rn_f32(-gm_g, scale2, c_dp) if const_expr(self.repro or self.fp4all) else c_dp - gm_g * scale2)
        if const_expr(lean):
            # row max over the 8 group maxes (+ the running max): 4 three-input ops
            _t0 = utils.fmax(gmv[0], gmv[1], gmv[2])
            _t1 = utils.fmax(gmv[3], gmv[4], gmv[5])
            if const_expr(not is_first and not tile_frame):
                _t2 = utils.fmax(gmv[6], gmv[7], softmax.row_max[0])
            else:
                _t2 = utils.fmax(gmv[6], gmv[7])
            row_max_new = utils.fmax(_t0, _t1, _t2)
        else:
            row_max_new = cute.arch.fmax(rmax_a, rmax_b)
            if const_expr(not is_first and not tile_frame):
                row_max_new = cute.arch.fmax(row_max_new, softmax.row_max[0])
        if const_expr(pmax_rm is not None and not fp4all):
            pmax_rm[0] = cute.arch.fmax(pmax_rm[0], row_max_new)
        if const_expr(intf):
            # integer frame: R = round(row_max * scale + 1/2) (1.5 * 2^23 magic: FADD rounds to an integer), kept until the
            # snapped max grows past the rescale threshold; acc_scale = 2^(F_old - F) (1 when kept). softmax.row_max holds the
            # true running raw max (no stale value), frame_rm[0] the frame, k_row = max_offset - F.
            softmax.row_max[0] = row_max_new
            _MG = Float32(12582912.0)
            # a fully masked row has row_max -inf: the frame would be -inf and k_row +inf (SF inf * sum 0 = NaN row sum).
            # Its output is discarded (LSE -inf through acc_O_mn_row_is_zero_or_nan), so any finite frame will do.
            _rm_f = row_max_new if const_expr(not mx) else (row_max_new if row_max_new != -cutlass.Float32.inf else Float32(0.0))
            _rc = _rm_f * scale + Float32(0.5) + _MG
            r_cand = _rc - _MG
            if const_expr(is_first):
                frame = r_cand
                acc_scale = Float32(1.0)
            else:
                f_old = frame_rm[0]
                frame = r_cand if r_cand > f_old + Float32(softmax.rescale_threshold) else f_old
                acc_scale = cute.arch.exp2(f_old - frame)
                if const_expr(self.repro):
                    # `reproduce-forward-p`: every frame change of this row, in processing order, so the backward knows the
                    # frame each key tile's group SFs were computed in. The frame before the change is an integer; an
                    # all-masked prefix's placeholder frame (~ -1e8) saturates to int16 -- its tiles hold only code 0.
                    rp_t = self._repro
                    n_rec = Int32(self.REPRO_RECORDS)
                    rp_cnt = Int32(frame_rm[1])
                    rp_chg = r_cand > f_old + Float32(softmax.rescale_threshold)
                    if rp_chg and rp_cnt < n_rec and m_block * self.m_block_size < seqlen.seqlen_q:
                        f_i = Int32(cute.arch.fmax(-cute.arch.fmax(-f_old, Float32(-32767.0)), Float32(-32768.0)))
                        rp_b, rp_m = self._rp_bm(batch_idx, m_block, seqlen)
                        if const_expr(self.codes_packed):
                            if thread_idx < rp_m:
                                rp_t[head_idx, 3 + rp_cnt, rp_b + thread_idx] = (n_block << 16) | (f_i & Int32(0xFFFF))
                        else:
                            rp_t[rp_b, head_idx, rp_m, 3 + rp_cnt, thread_idx] = (n_block << 16) | (f_i & Int32(0xFFFF))
                    frame_rm[1] = frame_rm[1] + (Float32(1.0) if rp_chg else Float32(0.0))
            frame_rm[0] = frame
            row_max = row_max_new
        elif const_expr(tile_frame):
            # `per-tile-row-frame` (SageAttention3 Alg. 1 line 10): row_max_new is THIS tile's row max; the tile is
            # scaled against it (floored at the running max - drop), and O / row_sum move to that frame (acc_scale > 1
            # when the reference falls). frame_rm = true running max, softmax.row_max = the reference.
            if const_expr(is_first):
                run = row_max_new
                ref = row_max_new
                acc_scale = Float32(1.0)
            else:
                run = cute.arch.fmax(frame_rm[0], row_max_new)
                ref = cute.arch.fmax(row_max_new, run - tf_drop_raw)
            row_max = ref if ref != -cutlass.Float32.inf else Float32(0.0)
            if const_expr(not is_first):
                acc_scale = cute.arch.exp2((softmax.row_max[0] - row_max) * scale)
            frame_rm[0] = run
            softmax.row_max[0] = ref
        else:
            row_max, acc_scale = softmax.update_row_max_from_local(row_max_new, is_first)
        if const_expr(not is_first):
            if const_expr(self.max_tmem):
                # max_tmem on the split step: the rescale factor as a one-f32 fragment
                tSrScale_r2t = cute.make_rmem_tensor(
                    thr_tmem_store_scale.partition_S(cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))).shape,
                    Float32,
                )
                tSrScale_r2t[0] = acc_scale
                if const_expr(self.corr_empty_early):
                    # double-buffered by tile parity (column n_block & 1), like sScale; branch-free
                    cute.copy(thr_tmem_store_scale, tSrScale_r2t,
                              cute.make_tensor(tStScale_r2t.iterator + (n_block & 1), tStScale_r2t.layout))
                else:
                    cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
                cute.arch.fence_view_async_tmem_store()
            else:
                sScale[thread_idx + stage * self.m_block_size + (n_block & 1) * (4 * self.m_block_size)] = acc_scale
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
        # group SFs: SF_g = exp2(max_offset - row_max*scale - c_g); a-half row-sum part now
        if const_expr(intf and self.repro):
            # `reproduce-forward-p`: the two frame constants as exactly representable fp32 values (pv4_repro.f32c), so the
            # backward's kb = (max_offset - F_n) - log2 6 is this one bit for bit whatever the constant conversion does
            from flash_attn.cute.pv4_repro import f32c
            k_row = Float32(f32c(self._repro_max_offset())) - frame
        elif const_expr(intf):
            k_row = Float32(softmax.max_offset) - frame
        else:
            k_row = Float32(softmax.max_offset) - row_max * scale
        if const_expr(lean and self.repro):
            kb = k_row - Float32(f32c(LOG2_PMAX))
        elif const_expr(lean):
            kb = k_row - LOG2_PMAX
        sfw_u32 = cute.recast_tensor(sfw, cute.Int32)
        xdbg = cute.make_rmem_tensor(cute.make_layout(2 * ng_h), Float32) if const_expr(self.repro_dbg) else None
        row_sum_acc = Float32(0.0)
        if const_expr(mx):
            # 2 groups per half (32-element SF groups): all four exponents pack into one UE8M0 word
            sa0 = cute.arch.exp2(k_row - cg[0])
            sa1 = cute.arch.exp2(k_row - cg[1])
            sb0 = cute.arch.exp2(k_row - cg[ng_h])
            sb1 = cute.arch.exp2(k_row - cg[ng_h + 1])
            sfw_u32[0] = sm100_utils.packed_float_to_ue8m0(sa0, sa1, sb0, sb1)
            row_sum_acc += sa0 * gsum_a[0] + sa1 * gsum_a[1]
            sf_b[0], sf_b[1] = sb0, sb1
        for i in cutlass.range_constexpr(0 if const_expr(mx) else ng_h // 4):
            if const_expr(lean):
                # SF_g = exp2(gm_g*scale + (k_row - log2 6)): one FFMA per group. Explicit (packed) FMAs:
                # left to ptxas, some sites become FADD on a product reused from the classifier constants
                # (1-ulp SF / row-sum / LSE differences between otherwise unrelated builds)
                _a0, _a1 = utils.fma_packed_f32x2((gmv[4 * i], gmv[4 * i + 1]), (scale, scale), (kb, kb))
                _a2, _a3 = utils.fma_packed_f32x2((gmv[4 * i + 2], gmv[4 * i + 3]), (scale, scale), (kb, kb))
                if const_expr(self.repro_dbg):
                    xdbg[4 * i], xdbg[4 * i + 1], xdbg[4 * i + 2], xdbg[4 * i + 3] = _a0, _a1, _a2, _a3
                s0 = cute.arch.exp2(_a0)
                s1 = cute.arch.exp2(_a1)
                s2 = cute.arch.exp2(_a2)
                s3 = cute.arch.exp2(_a3)
            else:
                s0 = cute.arch.exp2(k_row - cg[4 * i])
                s1 = cute.arch.exp2(k_row - cg[4 * i + 1])
                s2 = cute.arch.exp2(k_row - cg[4 * i + 2])
                s3 = cute.arch.exp2(k_row - cg[4 * i + 3])
            sfw_u32[i] = packed_float_to_ue4m3(s0, s1, s2, s3)
            if const_expr(fp4all and self.fp4all_sf_decode):
                # the E4M3 values the PV GEMM multiplies (sum_k P^ / row sum = 1 exactly)
                s0, s1, s2, s3 = sm100_utils.packed_ue4m3_to_float(sfw_u32[i])
            row_sum_acc += s0 * gsum_a[4 * i] + s1 * gsum_a[4 * i + 1] + s2 * gsum_a[4 * i + 2] + s3 * gsum_a[4 * i + 3]
        if const_expr(mx):
            pass
        elif const_expr(lean):
            for g in cutlass.range_constexpr(0, ng_h, 2):
                _b0, _b1 = utils.fma_packed_f32x2((gmv[ng_h + g], gmv[ng_h + g + 1]), (scale, scale), (kb, kb))
                if const_expr(self.repro_dbg):
                    xdbg[ng_h + g], xdbg[ng_h + g + 1] = _b0, _b1
                sf_b[g] = cute.arch.exp2(_b0)
                sf_b[g + 1] = cute.arch.exp2(_b1)
        else:
            for g in cutlass.range_constexpr(ng_h):
                sf_b[g] = cute.arch.exp2(k_row - cg[ng_h + g])
        for i in cutlass.range_constexpr(0 if const_expr(mx) else ng_h // 4):
            sfw_u32[ng_h // 4 + i] = packed_float_to_ue4m3(sf_b[4 * i], sf_b[4 * i + 1], sf_b[4 * i + 2], sf_b[4 * i + 3])
            if const_expr(fp4all and self.fp4all_sf_decode):
                sf_b[4 * i], sf_b[4 * i + 1], sf_b[4 * i + 2], sf_b[4 * i + 3] = sm100_utils.packed_ue4m3_to_float(sfw_u32[ng_h // 4 + i])
        # SFP -> smem, then P k-block 0 -> TMEM: PV(t) may start
        lane_id = thread_idx % 32
        warp_id = thread_idx // 32
        base_offset = lane_id * 16 + (warp_id % 4) * 4
        sfp_thread_layout = cute.make_layout((4, (2 * ng_h) // 4), stride=(1, 512))
        sSFP_thread = cute.make_tensor(sSFP[None, None, None, stage].iterator + base_offset, sfp_thread_layout)
        cute.autovec_copy(cute.logical_divide(sfw, cute.make_layout(4)), sSFP_thread)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, 0], tStP_r2t[None, None, 0])
        if const_expr(not self.qk_split_late_p0):
            cute.arch.fence_view_async_tmem_store()
            self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        # b-half exps / packs, row sum accumulated per group
        for g in cutlass.range_constexpr(ng_h):
            if const_expr(directp):
                row_sum_acc += sf_b[g] * self._directp_group(
                    frag_b, g, scale2_e if const_expr(pmx is not None and pmx[ng_h + g] != "M") else scale2,
                    cg2[ng_h + g], tSrP_frag, ng_h + g, masked=masked,
                    vu=(vt_b, vt_kvh, vt_base + (ng_h + g) * sf_vec, vu_h,
                        (g if const_expr(self.vtok_tma) else ng_h + g) * sf_vec, vt_pad)
                    if const_expr(vtok) else None,
                    fn=pmx[ng_h + g] if const_expr(pmx is not None) else None)
            else:
                c = cg[ng_h + g]
                for j in cutlass.range_constexpr(0, sf_vec, 2):
                    frag_b[j, g], frag_b[j + 1, g] = utils.fma_packed_f32x2(
                        (frag_b[j, g], frag_b[j + 1, g]), (scale, scale), (c, c),
                    )
                for j in cutlass.range_constexpr(0, sf_vec, 2):
                    _pair = (ng_h + g) * (sf_vec // 2) + j // 2
                    if const_expr(e2e_freq > 0 and _pair % e2e_freq == 0):
                        frag_b[j, g], frag_b[j + 1, g] = self._ex2_emu_pair(frag_b[j, g], frag_b[j + 1, g])
                    else:
                        frag_b[j, g] = cute.arch.exp2(frag_b[j, g])
                        frag_b[j + 1, g] = cute.arch.exp2(frag_b[j + 1, g])
                row_sum_acc += sf_b[g] * softmax._compute_row_sum(frag_b[None, g].load())
                pv_u32 = cute.recast_tensor(tSrP_frag[None, ng_h + g], cute.Int32)
                for k in cutlass.range_constexpr(0, cute.size(pv_u32, mode=[0])):
                    pv_u32[k] = packed_float_to_e2m1(
                        frag_b[k * 8, g], frag_b[k * 8 + 1, g], frag_b[k * 8 + 2, g], frag_b[k * 8 + 3, g],
                        frag_b[k * 8 + 4, g], frag_b[k * 8 + 5, g], frag_b[k * 8 + 6, g], frag_b[k * 8 + 7, g],
                    )
        if const_expr(self.repro_dbg):
            # debug build (`reproduce-forward-p` gate): every code and group SF byte of this step, (query, key) indexed
            tScSh_dbg = thr_tmem_load_h.partition_D(tScSh)
            dbg_row = m_block * self.m_block_size + tScSh_dbg[0][0]
            sfw_dbg = cute.recast_tensor(sfw, cute.Int32)
            for pg in cutlass.range_constexpr(2 * ng_h):
                words_dbg = cute.recast_tensor(tSrP_frag[None, pg], cute.Int32)
                for kw in cutlass.range_constexpr(cute.size(words_dbg)):
                    for i in cutlass.range_constexpr(8):
                        e_dbg = (pg % ng_h) * sf_vec + 8 * kw + i
                        key_dbg = n_block * self.n_block_size + 64 * (pg // ng_h) + tScSh_dbg[e_dbg][1]
                        self._repro_dbg[batch_idx, head_idx, dbg_row, key_dbg] = (
                            (words_dbg[kw] >> (4 * i)) & Int32(0xF)).to(cutlass.Uint8)
                self._repro_dbg[batch_idx, head_idx, dbg_row, seqlen.seqlen_k + n_block * (2 * ng_h) + pg] = (
                    (sfw_dbg[pg // 4] >> (8 * (pg % 4))) & Int32(0xFF)).to(cutlass.Uint8)
                gb_dbg = f32_bits(gmv[pg])
                xb_dbg = f32_bits(xdbg[pg])
                for bb in cutlass.range_constexpr(4):
                    self._repro_dbg[batch_idx, head_idx, dbg_row,
                                    seqlen.seqlen_k + seqlen.seqlen_k // 16 + 4 * (n_block * (2 * ng_h) + pg) + bb] = (
                        (gb_dbg >> (8 * bb)) & Int32(0xFF)).to(cutlass.Uint8)
                    self._repro_dbg[batch_idx, head_idx, dbg_row,
                                    seqlen.seqlen_k + 5 * (seqlen.seqlen_k // 16) + 4 * (n_block * (2 * ng_h) + pg) + bb] = (
                        (xb_dbg >> (8 * bb)) & Int32(0xFF)).to(cutlass.Uint8)
        if const_expr(directp and dp_codes_sum):
            # code sums are 2x the quantized values (integer LUT)
            row_sum_acc = row_sum_acc * Float32(0.5)
        if const_expr(is_first):
            softmax.row_sum[0] = row_sum_acc
        else:
            softmax.row_sum[0] = softmax.row_sum[0] * acc_scale + row_sum_acc
        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, 1], tStP_r2t[None, None, 1])
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.qk_split_late_p0):
            self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        self._arrive_mma(mbar_ptr + self.mbar_P_full_2_offset + stage)
        if const_expr(self.vtok_w and not self.vtok_tma):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_vtok_offset + 1 + stage)  # this step's ring slot is no longer read
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def softmax_zero_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        stage: int | Int32,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        thr_mma_qk: cute.atom.ThrMma,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        sSFP: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """two-cta-forward: a KV tile of the pair's range that lies wholly above the causal diagonal of this warpgroup's
        rows (the pair's range is the union of its four 128-row groups'; this warpgroup's own range is the 1-CTA kernel's
        for these rows). The step keeps every barrier of softmax_step in order and computes nothing: no S load, P^ = 0
        (so the PV adds exactly zero to O), rescale factor 1.0, the row statistics untouched."""
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        # S(stage) is not read: its bank may take the other stage's scale-factor copies right away (only the block-scaled
        # QK waits for it)
        if const_expr(self.quant_qk):
            self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset + (self.q_stage - 1 - stage))
        if const_expr(self.max_tmem):
            tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            tSrScale_r2t[0] = Float32(1.0)
            if const_expr(self.corr_empty_early):
                cute.copy(thr_tmem_store_scale, tSrScale_r2t,
                          cute.make_tensor(tStScale_r2t.iterator + (n_block & 1), tStScale_r2t.layout))
            else:
                cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            cute.arch.fence_view_async_tmem_store()
        elif const_expr(self.corr_empty_early):
            sScale[tidx + stage * self.m_block_size + (n_block & 1) * (4 * self.m_block_size)] = Float32(1.0)
        else:
            sScale[tidx + stage * self.m_block_size] = Float32(1.0)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
        if const_expr(self.p_sf_inkernel):
            # the group-SF P^ route copies this stage's SMEM SF tile before every PV: give it finite factors (1.0,
            # UE8M0 127) -- an unwritten tile may hold the UE8M0 NaN code, and NaN x 0 poisons O
            assert self.sf_vec_size_pv == 32 and self.v_dtype.width == 8
            _base = (tidx % 32) * 16 + ((tidx // 32) % 4) * 4
            _sfp_w = cute.make_tensor(
                cute.recast_ptr(sSFP[None, None, None, stage].iterator + _base, dtype=cutlass.Uint32),
                cute.make_layout(1),
            )
            _sfp_w[0] = cutlass.Uint32(0x7F7F7F7F)
            cute.arch.fence_proxy("async.shared", space="cta")
        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t_f32.fill(0.0)
        _n_p = cute.size(tStP_r2t.shape[2])
        for i in cutlass.range_constexpr(self.mbar_p_split(_n_p)):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(self.mbar_p_split(_n_p), _n_p):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        self._arrive_mma(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.atom.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tCtSFP: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
        # True when mask_fn is a compile-time no-op for this call site
        # (non-causal/non-local, no mask_mod, mask_seqlen=False) -- gates the
        # use of the tcgen05.ld.red hardware max.
        mask_is_noop: cutlass.Constexpr[bool] = False,
        thr_tmem_load_h=None,
        tStS_t2r_a: Optional[cute.Tensor] = None,
        tStS_t2r_b: Optional[cute.Tensor] = None,
        tScSh: Optional[cute.Tensor] = None,
        tSrS_a_persist: Optional[cute.Tensor] = None,
        split_masked: cutlass.Constexpr[bool] = False,  # split-pipeline flags (ignored here)
        split_mask_seqlen: cutlass.Constexpr[bool] = False,
        pmax_rm: Optional[cute.Tensor] = None,
        tbqk_pref: Optional[cute.Tensor] = None,  # token-block-qk-fold: factors prefetched one step ahead
        # two-cta-forward: this first step of the warpgroup's own KV range may follow bookkept (fully masked) tiles of the
        # pair's range, so the correction reads a rescale factor for it: 1.0 (O is exactly zero before it)
        unit_scale_first: cutlass.Constexpr[bool] = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        pure_fp8_pv = const_expr(self.p_rowframe8)
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        # P size when in FP32
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))

        tb_a = tb_c = tb_uu = None
        tq_a = tq_gs = None
        if const_expr(self.tbqk_fold):
            tq_a = self._tbqk_load(tScS[(None, None), 0, 0], thr_tmem_load, batch_idx, head_idx, m_block, n_block,
                                   tbqk_pref, is_first)
        tok_r = None
        tb_b = batch_idx
        tb_vl = {}
        if const_expr(self.tb_varlen):
            # documents: the tables' batch 0, the document's offsets (multiples of 128: whole 32-token blocks)
            tb_b = Int32(0)
            if const_expr(self.codes_packed):   # (cov3: the tables per packed row; no per-32-block factor read)
                tb_vl = dict(q_off=self._tb_pk[0][batch_idx], k_blk_off=None)
            else:
                tb_vl = dict(q_off=seqlen.offset_q, k_blk_off=seqlen.offset_k // 32)
        if const_expr(self.tb_tok_v):
            tok_r = self._tb_tok_prefetch(tb_b, head_idx, n_block,   # token-local operands: before the S wait
                                          **({"k_off": (self._tb_pk[1][batch_idx] if const_expr(self.codes_packed)
                                                        else seqlen.offset_k)} if const_expr(self.tb_varlen) else {}))
        if const_expr(self.tb_fwd):
            # the token-block factors of this tile: loads issued before the S wait (they overlap it)
            tb_a, tb_c, tb_uu = self._tb_load(tScS[(None, None), 0, 0], thr_tmem_load, softmax, tb_b, head_idx,
                                              m_block, n_block, **tb_vl)
        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        # Load S from TMEM. SM103: tcgen05.ld.red fuses the load with a
        # per-x32-tile max reduction in the TMEM controller -- row max comes
        # nearly free, and for sf_vec_size=32 (MXFP8 P) each tile max IS the
        # scale-factor group max.
        use_ldred = const_expr(self.use_ldred_rowmax and self.score_mod is None)
        # rank-2 accumulator coordinates, to match thr_tmem_load's rank-2 tiler (see the note where
        # it is built). The P store keeps the rank-3 tScP: its tiler comes from tStP, not tStSi.
        tScS_r2 = tScS[(None, None), 0, 0]
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS_r2).shape, self.qk_acc_dtype)
        hw_tile_maxes = None
        if const_expr(use_ldred):
            if const_expr((self.p_sf_inkernel and self.sf_vec_size_pv == 32 and self.fp4_pv_log2_quant)
                          or self.tbqk_fold or self.tb_fwd):
                # One x32 tile per ld.red instruction (tiles = total / atom).
                _n_ldred_tiles = cute.size(tStS_t2r.shape) // cute.size(tStS_t2r.shape[0])
                hw_tile_maxes = cute.make_rmem_tensor(
                    cute.make_layout(_n_ldred_tiles), Float32
                )
            hw_max = sm100_utils.tmem_ld_red_max(tStS_t2r, tSrS_t2r, hw_tile_maxes)
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # unblock sfqk load
        cute.arch.fence_view_async_tmem_load()
        sfqk_stage = self.q_stage - 1 - stage
        if const_expr(self.quant_qk or not self.use_2cta_forward):
            # (two-cta-forward: a remote arrive, skipped where nothing waits for it -- only the block-scaled QK does)
            self._arrive_mma(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)

        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                aux_tensors,
                fastdiv_mods,
                seqlen_info=seqlen,
            )

        tb_u = None
        if const_expr(self.tb_fwd):
            # S itself is not transformed: the row max below takes the per-group maxes of the (masked) raw S, and the
            # scale-subtract FFMA2 carries a_g and c_g (_tb_row_max / _tb_scale_subtract)
            self._tb_prep(softmax, tb_a, tb_c, tb_uu)
            tb_u = tb_uu
        if const_expr(self.tbqk_fold):
            tq_gs = self._tbqk_prep(softmax, tq_a)
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        if const_expr(self.tb_tok_v):
            self._tb_tok_stage(tok_r)
            self._tb_tok_v_bias(softmax, tSrS_t2r, batch_idx, head_idx, n_block, tb_a[0])

        # The hardware max is only valid when masking left the values
        # untouched -- masked iterations (seqlen boundary, causal, mask_mod)
        # fall back to the software reduce over the post-mask values.
        # Group-max-first (fp4_group_max_first) applies on the log2 quant
        # path when the software row max is used: the per-group raw maxes are
        # computed here, the row max is their max, and the scale/row-max
        # subtraction is folded into the quant loop's per-group FMA.
        gmf_active = const_expr(
            self.fp4_group_max_first
            and self.p_sf_inkernel
            and self.fp4_pv_log2_quant
            and (not self.force_e2e or self.v_dtype.width == 8)
            and not (use_ldred and mask_is_noop)
        )
        raw_group_maxes = None
        # fwdgap "gmx" (opt-in, bitwise): the in-kernel group-SF P^ step takes its per-32-key group maxes from the RAW
        # scores (one FMNMX3 tree per group, the row max from the four group maxes) and maps them through the step's
        # own scale-subtract FFMA (monotone, one rounding: max_i fma(s_i, c, b) = fma(max_i s_i, c, b)), instead of a
        # 128-wide row-max pass plus a second 128-wide group-max pass over the scaled scores.
        gmx_active = const_expr(
            "gmx" in self.xf and self.p_sf_inkernel and self.fp4_pv_log2_quant and self.v_dtype.width == 8
            and self.sf_vec_size_pv == 32 and not gmf_active and not (use_ldred and mask_is_noop)
            and not self.tbqk_fold and not self.tb_fwd
        )
        gmx_raw = None
        if const_expr(gmx_active):
            ngr_x = cute.size(tSrS_t2r) // 32
            gmx_raw = cute.make_rmem_tensor(cute.make_layout(ngr_x), Float32)
            for g in cutlass.range_constexpr(ngr_x):
                o = 32 * g
                m0 = utils.fmax(tSrS_t2r[o], tSrS_t2r[o + 1], tSrS_t2r[o + 2])
                m1 = utils.fmax(tSrS_t2r[o + 3], tSrS_t2r[o + 4], tSrS_t2r[o + 5])
                for i in cutlass.range_constexpr(6, 30, 4):
                    m0 = utils.fmax(m0, tSrS_t2r[o + i], tSrS_t2r[o + i + 1])
                    m1 = utils.fmax(m1, tSrS_t2r[o + i + 2], tSrS_t2r[o + i + 3])
                m0 = utils.fmax(m0, tSrS_t2r[o + 30], tSrS_t2r[o + 31])
                gmx_raw[g] = cute.arch.fmax(m0, m1)
        if const_expr(gmx_active):
            if const_expr(is_first):
                row_max_new = softmax._compute_row_max(gmx_raw.load())
            else:
                row_max_new = softmax._compute_row_max(gmx_raw.load(), init_val=softmax.row_max[0])
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], row_max_new)
            row_max, acc_scale = self._gmx_update_row_max(softmax, row_max_new, is_first)
        elif const_expr(self.tbqk_fold):
            _gm = hw_tile_maxes if const_expr(use_ldred and mask_is_noop) else None
            if const_expr(is_first):
                row_max_new = self._tbqk_row_max(softmax, tSrS_t2r, tq_a, None, gm=_gm)
            else:
                row_max_new = self._tbqk_row_max(softmax, tSrS_t2r, tq_a, softmax.row_max[0], gm=_gm)
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], row_max_new)
            row_max, acc_scale = self._tb_update_row_max(softmax, row_max_new, is_first)
        elif const_expr(self.tb_fwd):
            # the max of the transformed scores (gmf is off for tb_fwd), from ld.red's raw group maxes where unmasked
            # (not with Q.K factors: a_g m_g + c_g from the ld.red maxes was not bitwise the software tree's build,
            # as fma.rn and as mul.rn + add.rn -- B300, fwdreg/tbfma_*; that form keeps the software max)
            _gm = hw_tile_maxes if const_expr(use_ldred and mask_is_noop and not self.tb_qk) else None
            if const_expr(is_first):
                row_max_new = self._tb_row_max(softmax, tSrS_t2r, tb_a, tb_c, None, gm=_gm)
            else:
                row_max_new = self._tb_row_max(softmax, tSrS_t2r, tb_a, tb_c, softmax.row_max[0], gm=_gm)
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], row_max_new)
            row_max, acc_scale = self._tb_update_row_max(softmax, row_max_new, is_first)
        elif const_expr(use_ldred and mask_is_noop):
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], hw_max)
            row_max, acc_scale = softmax.update_row_max_precomputed(hw_max, is_first)
        elif const_expr(gmf_active):
            _acc_S_frag = cute.logical_divide(tSrS_t2r, cute.make_layout(self.sf_vec_size_pv))
            _ng = cute.size(_acc_S_frag, mode=[1])
            raw_group_maxes = cute.make_rmem_tensor(cute.make_layout(_ng), Float32)
            for _g in cutlass.range_constexpr(_ng):
                raw_group_maxes[_g] = softmax._compute_row_max(_acc_S_frag[None, _g].load())
            if const_expr(is_first):
                row_max_new = softmax._compute_row_max(raw_group_maxes.load())
            else:
                row_max_new = softmax._compute_row_max(raw_group_maxes.load(), init_val=softmax.row_max[0])
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], row_max_new)
            row_max, acc_scale = softmax.update_row_max_from_local(row_max_new, is_first)
        else:
            if const_expr(pmax_rm is not None):
                pmax_rm[0] = cute.arch.fmax(pmax_rm[0], softmax._compute_row_max(tSrS_t2r.load()))
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        tSrPSF_f32 = None
        tSrPSF = None

        if const_expr(not is_first or unit_scale_first):
            # tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            _scale_w = Float32(1.0) if const_expr(is_first) else acc_scale
            if const_expr(self.max_tmem):
                # max_tmem: tcgen05.st into the stage's vec slot (S column tmem_vec_rel; P sits at +64)
                tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
                tSrScale_r2t[0] = _scale_w
                if const_expr(self.corr_empty_early):
                    # the correction releases the slot at its first read (one tile behind), so the
                    # slot is double-buffered by tile parity exactly as sScale is: column (n_block & 1).
                    # Branch-free (a dynamic branch around the fragment splits the unrolled step)
                    cute.copy(thr_tmem_store_scale, tSrScale_r2t,
                              cute.make_tensor(tStScale_r2t.iterator + (n_block & 1), tStScale_r2t.layout))
                else:
                    cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
                cute.arch.fence_view_async_tmem_store()
            elif const_expr(self.corr_empty_early):
                sScale[thread_idx + stage * self.m_block_size + (n_block & 1) * (4 * self.m_block_size)] = _scale_w
            else:
                sScale[thread_idx + stage * self.m_block_size] = _scale_w
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        # softmax_fused: the pure-E4M3 explicit-pack route only (no chunk pipeline / range
        # unroll / fused-pack experiments, no group-max-first): scale-subtract, exp2,
        # pack and row sum in one loop (_softmax_fused_e4m3)
        fused_active = const_expr(
            self.softmax_fused
            and pure_fp8_pv
            and self.v_dtype == Float8E4M3FN
            and self.fp8_pv_use_explicit_pack
            and not gmf_active
        )
        # mode 1 sums the row inside the loop; mode 2 keeps update_row_sum after the P store
        fused_rowsum = const_expr(fused_active and self.softmax_fused_mode == 1)
        # mask_loops: the masked steps run all-MUFU exp2 (their body already carries the
        # mask's compare/select stream); the unmasked steps keep the emulated pairs
        e2e_step = const_expr(self.force_e2e and not (self.mask_loops >= 2 and mask_fn is not None))
        # fwdgap "sche" / "scha" on the four-pass (unfused) pure-E4M3 step: scale-subtract and exp2 in one pass
        sch_pass = const_expr(
            ("sche" in self.xf or "scha" in self.xf or self._schm() is not None) and pure_fp8_pv and self.v_dtype == Float8E4M3FN
            and self.fp8_pv_use_explicit_pack and not fused_active and not gmf_active and not self.tb_fwd
            and not self.p_sf_inkernel
        )
        if const_expr(sch_pass):
            pass
        elif const_expr(self.tb_fwd):
            self._tb_scale_subtract(softmax, tSrS_t2r, tb_a, tb_c, row_max)
        elif const_expr(self.tbqk_fold and not fused_active):
            self._tbqk_scale_subtract(softmax, tSrS_t2r, tq_gs, row_max)
        elif const_expr(gmx_active):
            gmx_bias = Float32(softmax.max_offset) - row_max * softmax.scale_log2
            for e in cutlass.range_constexpr(0, cute.size(tSrS_t2r), 2):
                tSrS_t2r[e], tSrS_t2r[e + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS_t2r[e], tSrS_t2r[e + 1]), (softmax.scale_log2, softmax.scale_log2), (gmx_bias, gmx_bias))
            for g in cutlass.range_constexpr(0, cute.size(gmx_raw), 2):
                gmx_raw[g], gmx_raw[g + 1] = cute.arch.fma_packed_f32x2(
                    (gmx_raw[g], gmx_raw[g + 1]), (softmax.scale_log2, softmax.scale_log2), (gmx_bias, gmx_bias))
        elif const_expr(not gmf_active and not fused_active):
            softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        # The FP8 P path recasts this Float32 backing storage into packed FP8 lanes.
        # Zero-initialize first so any padding / partially covered bytes cannot leak
        # stale register contents into the TMEM store on later configs.
        if const_expr(self.fp8_pv_zero_fill_regs):
            tSrP_r2t_f32.fill(0.0)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype),
            tSrS_t2r.layout, # shape of S owned by this thread
        )


        if const_expr(self.p_sf_inkernel):
            # The non-log2 path packs P as E2M1 only; MXFP8 PV (E4M3 P) must
            # take the log2 path even with exp2 emulation on (B200 default) --
            # before this it silently produced garbage on B200 (cos 0.11).
            fp4_log2_active = const_expr(
                self.fp4_pv_log2_quant
                and (not self.force_e2e or self.v_dtype.width == 8)
            )
            if const_expr(not fp4_log2_active):
                # Exp2 with softmax scale and sp1 scaling
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    ex2_emu_freq=self.e2e_freq if self.force_e2e else 0,
                    ex2_emu_start_frg=self.e2e_start_frg,
                )
                # update_row_sum BEFORE scale_groupwise so it uses original P values
                softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
            # else: exp2, row_sum and quant are all fused in
            # _fused_log2_group_quant below.
            if const_expr(fp4_log2_active):
                tSrPSF_f32, tSrPSF = self._fused_log2_group_quant(
                    softmax, tSrS_t2r, tSrP_r2t, acc_scale, is_first,
                    hw_group_maxes=hw_tile_maxes if cutlass.const_expr(
                        mask_is_noop and not self.tb_fwd and not self.tbqk_fold) else None,
                    row_max=row_max,
                    raw_group_maxes=raw_group_maxes,
                    clamp_neg_inf=not mask_is_noop,
                    scaled_group_maxes=gmx_raw,
                )
            else:
                tSrPSF_f32, tSrPSF = self._fused_group_max_scale_quant(
                    softmax, tSrS_t2r, tSrP_r2t,
                )
            # R2S: Copy tSrPSF (registers) to sSFP (shared memory).
            # The SFP smem layout is BlockScaledBasicChunk(sf_vec_size).layout
            # tile_to_shape((M=128, K=128)). The atom is ((32,4),(sf_vec,4))
            # with stride ((16,4),(0,1)) -- those strides are independent of
            # sf_vec_size, so `base_offset = lane_id*16 + (warp_id%4)*4` is
            # correct for both NVFP4 (sf_vec=16) and MXFP8 (sf_vec=32).
            # What changes is the K-group count per row:
            #   sf_vec=16, K=128 -> 8 SFs per row -> 4 inner + 2 outer (rest_k stride 512)
            #   sf_vec=32, K=128 -> 4 SFs per row -> 4 inner + 1 outer (no rest_k)
            if const_expr(sSFP is not None):
                thread_idx = thr_tmem_load.thr_idx
                lane_id = thread_idx % 32
                warp_id = thread_idx // 32
                base_offset = lane_id * 16 + (warp_id % 4) * 4
                # PV is (M=128, N=head_dim_v, K=n_block_size). SFP scales the
                # P (=A) operand along the K dim, so per-row SF count is
                # n_block_size // sf_vec_size.
                k_groups_per_row = const_expr(self.mma_tiler_pv[2] // self.sf_vec_size_pv)
                k_inner = const_expr(min(k_groups_per_row, 4))
                k_outer = const_expr(k_groups_per_row // k_inner)
                # cute.cosize of one atom = 512 bytes for both sf_vec sizes.
                sfp_thread_layout = cute.make_layout((k_inner, k_outer), stride=(1, 512))
                sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
                sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
                tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(k_inner))
                cute.autovec_copy(tSrPSF_2d, sSFP_thread)
                # The MMA warp reads sSFP through the async proxy (tcgen05.cp);
                # generic-proxy st.shared must be fenced before the P_full
                # arrive or the S2T copy can see stale SF bytes (measured as
                # launch-to-launch nondeterminism of the unfenced kernel).
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
        else:
            # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
            if const_expr(pure_fp8_pv and self.v_dtype == Float8E4M3FN and self.fp8_pv_use_explicit_pack):
                if const_expr(fused_active and self.softmax_fused_mode == 1):
                    # softmax_fused=1: scale-subtract + exp2 + pack + row sum, one loop
                    self._softmax_fused_e4m3(
                        softmax, tSrS_t2r, tSrP_r2t, row_max, acc_scale, is_first,
                        e2e=e2e_step, e2e_freq=self.e2e_freq, e2e_start_frg=self.e2e_start_frg,
                        gscale=tq_gs,
                    )
                elif const_expr(fused_active):
                    # softmax_fused=2: MUFU pairs first, emulated tail, cvt as groups
                    # complete; the row sum stays after the P store
                    self._softmax_fused_e4m3_tail(
                        softmax, tSrS_t2r, tSrP_r2t, row_max,
                        e2e=e2e_step, e2e_freq=self.e2e_freq, e2e_start_frg=self.e2e_start_frg,
                        gscale=tq_gs,
                    )
                elif const_expr(sch_pass):
                    self._exp_sch_pass(softmax, tSrS_t2r, row_max, tq_gs, e2e=e2e_step, e2e_freq=self.e2e_freq,
                                       e2e_start_frg=self.e2e_start_frg)
                else:
                    softmax.apply_exp2_convert(
                        tSrS_t2r,
                        ex2_emu_freq=self.e2e_freq if e2e_step else 0,
                        ex2_emu_start_frg=self.e2e_start_frg,
                    )
                if const_expr(not fused_active):
                    self._pack_fp8(tSrS_t2r, tSrP_r2t)
            else:
                # BF16 PV: exp2 and the bf16x2 pack are fused inside
                # apply_exp2_convert -- EXP covers both, QUANT is zero-width.
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    tSrP_r2t,
                    ex2_emu_freq=self.e2e_freq if self.force_e2e else 0,
                    ex2_emu_start_frg=self.e2e_start_frg,
                )
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2]))):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready (and SFP is in SMEM)
        self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2])), cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        self._arrive_mma(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )

        if const_expr(not self.p_sf_inkernel and not fused_rowsum):
            if const_expr(self.tb_tok_v):
                self._tb_tok_v_row_sum(softmax, tSrS_t2r, batch_idx, head_idx, n_block, acc_scale, is_first)
            elif const_expr(self.tb_fwd):
                # tb_fwd: the exp values are P' = P f_V; the row sum of P weights each 32-key group by 1 / f_V
                softmax.update_row_sum_sage(tSrS_t2r, tb_u, acc_scale, is_first)
            else:
                softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.atom.ThrMma,
        thr_mma_pv: cute.atom.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        v_descale: Optional[cute.Tensor] = None,
        mPmax: Optional[cute.Tensor] = None,
        mma_tile_coord_v: Optional[Int32] = None,   # two-cta-forward: this CTA's index in the pair
        block_info_wg: Optional[BlockInfo] = None,  # two-cta-forward: the 256-row groups' KV ranges
        mCuSeqlensO: Optional[cute.Tensor] = None,  # o_rows_packed: the packed O row starts
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        _cta_v = mma_tile_coord_v if const_expr(self.use_2cta_forward) else 0
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
            self.qk_acc_dtype,
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        for stage in cutlass.range_constexpr(self.q_stage):
            self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

        softmax_corr_consumer_phase = Int32(0)
        corr_ph0 = Int32(0) if const_expr(self.stage_skip) else None  # stage-0 skip: stage 0's own phase
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            # Per-(batch, kv-head) V dequant scale for externally-quantized FP8
            # V (FA3 semantics): folds into the output normalization. Identity
            # when not provided.
            v_descale_val = Float32(1.0)
            if const_expr(v_descale is not None):
                kv_head_idx = (
                    head_idx
                    if const_expr(self.pack_gqa)
                    else head_idx // self.qhead_per_kvhead
                )
                v_descale_val = Float32(v_descale[batch_idx, kv_head_idx])
            if const_expr(self.score_scale_doc and self.score_mod is None):
                softmax_scale_log2 = self._doc_score_log2(batch_idx, head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            elif const_expr(self.o_rows_packed):
                mO_cur = cute.domain_offset((mCuSeqlensO[batch_idx],) + (None,) * (cute.rank(mO) - 1),
                                            mO)[None, None, head_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage
            pm_stats = [-Float32.inf] * self.q_stage  # return_pmax: true row max per stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            corr_z0 = None
            if const_expr(self.stage_skip):
                corr_z0 = self._pair_stage_skip(block_info_wg, seqlen, m_block, 0, split_idx, num_splits, n_block_max)
            if has_work:
                # Ignore first signal from softmax as no correction is required
                if const_expr(self.stage_skip):
                    # stage 0's first step is step z0 of the pair: ignored here only when z0 = 0, else inside the
                    # loop right before stage 1's step z0 -- ignoring it up front would hold stage 1's corrections
                    # (and so the PVs, and with SF copies the QK(0) that stage 0's first step waits for) behind it
                    if corr_z0 == 0:
                        self._mbar_wait_h(mbar_ptr + self.mbar_softmax_corr_full_offset + 0, corr_ph0)
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                        corr_ph0 ^= 1
                else:
                    self._mbar_wait_h(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase,
                    )
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                if const_expr(self.q_stage == 2):
                    self._mbar_wait_h(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 1, softmax_corr_consumer_phase
                    )
                    if const_expr(self.corr_empty_early):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)
                softmax_corr_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    if const_expr(self.stage_skip):
                        # two-cta-forward, stage-0 skip: stage 0's steps start z0 tiles late (its own phase)
                        if i + 1 == corr_z0:
                            # stage 0's first step: no correction (its first PV zero-initialises O)
                            self._mbar_wait_h(mbar_ptr + self.mbar_softmax_corr_full_offset + 0, corr_ph0)
                            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                            corr_ph0 ^= 1
                        if i >= corr_z0:
                            self._corr_pair_step(0, i, corr_ph0, n_block_max, mbar_ptr, tidx, sScale, thr_mma_pv, tOtOs,
                                                 thr_tmem_load_vec, tStScales_t2r, tSrScale_t2r_shape)
                            corr_ph0 ^= 1
                        self._corr_pair_step(1, i, softmax_corr_consumer_phase, n_block_max, mbar_ptr, tidx, sScale,
                                             thr_mma_pv, tOtOs, thr_tmem_load_vec, tStScales_t2r, tSrScale_t2r_shape)
                    for stage in cutlass.range_constexpr(self.q_stage if const_expr(not self.stage_skip) else 0):
                        # wait for S0 / S1
                        self._mbar_wait_h(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        if const_expr(self.max_tmem):
                            # max_tmem: tcgen05.ld of the stage's vec slot; under
                            # corr_empty_early the slot of tile i+1 is column ((n_block_max + i) & 1)
                            # (both loaded unconditionally, the value selected -- a load inside a
                            # dynamic branch would re-bind the fragment)
                            if const_expr(self.corr_empty_early):
                                cute.copy(thr_tmem_load_vec,
                                          cute.make_tensor(tStScales_t2r[stage].iterator + ((n_block_max + i) & 1),
                                                           tStScales_t2r[stage].layout),
                                          tSrScale_t2r)
                            else:
                                cute.copy(thr_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                            cute.arch.fence_view_async_tmem_load()
                            scale = tSrScale_t2r[0]
                        elif const_expr(self.corr_empty_early):
                            # tile i+1 of this Q tile has n_block = n_block_max - 2 - i
                            scale = sScale[tidx + stage * self.m_block_size + ((n_block_max + i) & 1) * (4 * self.m_block_size)]
                        else:
                            scale = sScale[tidx + stage * self.m_block_size]
                        if const_expr(self.corr_empty_early):
                            # slot read: the softmax may write the next tile's acc_scale
                            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                        should_rescale = cute.arch.vote_ballot_sync((scale != 1.0) if const_expr(self.fp4_pv_tile_frame) else (scale < 1.0)) != 0
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # self._mbar_wait_h(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                        # kept: removing it changes CuTe DSL codegen (measured: different PTX, ~1 % slower):
                        # the former P-cross O_full wait; with the phase flip below it keeps o_corr_consumer_phase loop-carried.
                        if const_expr(False):
                            self._mbar_wait_h(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                            sm100_utils.tcgen05_after_thread_sync()
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtOs[stage if self.q_stage == 2 else 0], tidx, scale
                            )
                        self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
                        if const_expr(not self.corr_empty_early):
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_softmax_corr_empty_offset + (self.q_stage - 1 - stage)
                            )
                    softmax_corr_consumer_phase ^= 1
                    if const_expr(False):  # kept: removing it changes CuTe DSL codegen (see the note above)
                        o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2 and not self.corr_empty_early):
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    self._mbar_wait_h(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        corr_ph0 if const_expr(self.stage_skip and stage == 0) else softmax_corr_consumer_phase,
                    )
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size + ((2 * self.m_block_size) if const_expr(self.corr_empty_early) else 0)]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + ((6 * self.m_block_size) if const_expr(self.corr_empty_early) else (2 * self.m_block_size))]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                    if const_expr(learnable_sink is not None):
                        # `learnable-sink-offset-omitted`. This used to be an inline fold that omitted BOTH of
                        # the shared helper's extra terms: the `+ max_offset` inside the exp2, and `2**max_offset`
                        # as a fully masked row's row_sum. Everything else in this kernel keeps row_sum in the
                        # offset domain, because \hat{P} = quant(P * 2^offset) -- so the sink alone entered weighted
                        # 2^-offset. On the shipped pure-FP8 PV that offset is 6, i.e. the sink counted 1/64 of what
                        # it should (O too large by the sink's share, LSE too small), and a fully masked row got
                        # LSE wrong by exactly 6*ln2. The NVFP4 routes are off by 2^-3.39, so this reached the 4-bit
                        # production forward too; only the NVFP4-QK + FP8-PV hybrid, whose offset is 0, was right.
                        # Upstream (flash_fwd_sm100.py) passes both terms; so do we now, through the one helper, so
                        # the offset cannot drift out of step with the row sum again. The offset is exactly the one
                        # the LSE epilogue subtracts, `_lse_log2_offset()`.
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            _sink_offset = Float32(self._lse_log2_offset())
                            row_max, row_sum = apply_learnable_sink(
                                row_max, row_sum, learnable_sink_val[stage], softmax_scale_log2,
                                max_offset=_sink_offset,
                                empty_row_sum=Float32(2.0 ** self._lse_log2_offset()),
                            )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    if const_expr(mPmax is not None):
                        pm_stats[stage] = sScale[tidx + stage * self.m_block_size + self.sscale_pmax_off]
                    if const_expr(self.fp4all):
                        # exact reciprocal (once per row): the backward normalises by 1 / Z from the exported LSE
                        scale = Float32(1.0) / (row_sum if not acc_O_mn_row_is_zero_or_nan else Float32(1.0))
                    else:
                        scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    # Externally-quantized FP8 V dequant (identity if v_descale is None).
                    scale = scale * v_descale_val
                    seqlen_q_store = seqlen.seqlen_q
                    if const_expr(self.zero_pad_rows):
                        # rows past seqlen_q of the document's last tile: written as zeros (scale 0)
                        if const_expr(self.use_2cta_forward):
                            # two-cta-forward on aligned documents: this CTA's 128-row tile of the pair's group
                            scale = cutlass.select_(
                                ((self.q_stage * m_block + stage) * self.cta_group_size + mma_tile_coord_v)
                                * self.m_block_size + tidx < seqlen.seqlen_q, scale, Float32(0.0))
                        else:
                            scale = cutlass.select_(
                                (self.q_stage * m_block + stage) * self.m_block_size + tidx < seqlen.seqlen_q,
                                scale, Float32(0.0))
                        if const_expr(not self.o_rows_packed):   # (packed O: the real rows only)
                            seqlen_q_store = (seqlen.seqlen_q + (self.m_block_size - 1)) // self.m_block_size \
                                * self.m_block_size
                    self._mbar_wait_h(
                        mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                    )
                    _es = stage % self.epi_stage   # E42: sO slot / corr<->epi barrier index
                    if const_expr(not (self.use_correction_warps_for_epi or self.o_regstore)):
                        self._mbar_wait_h(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + _es, corr_epi_producer_phase
                        )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtOs[stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen_q_store,
                        scale,
                        sO[None, None, _es],
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O,
                    )
                    if const_expr(not (self.use_correction_warps_for_epi or self.o_regstore)):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + _es)
                    if const_expr(self.epi_stage == 1):
                        corr_epi_producer_phase ^= 1   # the single pair cycles once per Q tile
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                if const_expr(self.stage_skip):
                    corr_ph0 ^= 1
                if const_expr(self.epi_stage == 2):
                    corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.codes_packed):   # (the packed LSE rows)
                        offset = mCuSeqlensO[batch_idx]
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    # this CTA's 128-row tile (two-cta-forward: half _cta_v of the pair's 256-row group)
                    m_tile_idx = (self.q_stage * m_block + stage) * self.cta_group_size + _cta_v
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            (row_max if const_expr(self.int_frame) else row_max * softmax_scale_log2)
                            + utils.log2f(row_sum)
                            - Float32(self._lse_log2_offset())  # FP8 PV: the P^ range offset; NVFP4 PV: the SF frame
                        ) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                        # This actually just works with PackGQA too
                        gLSE[tidx] = lse
                    elif const_expr(self.fp4all and self.zero_pad_rows and not self.codes_packed):
                        # aligned documents, all-4-bit: the padding rows of a document's last tile get LSE -inf (the
                        # backward's uinv 0 there) and, below, frame 0 -- defined values for the rows its masked tiles
                        # read. Only a tile that holds rows of this document: the CTA's second tile may be the next
                        # document's first.
                        if m_tile_idx * self.m_block_size < seqlen_q:
                            gLSE[tidx] = -Float32.inf
                    if const_expr(mPmax is not None):
                        # log P_max = max_k S_qk * scale - LSE (nats); -inf for empty rows
                        pm_row = pm_stats[stage]
                        if const_expr(self.fp4all):
                            log_pmax = pm_row  # fp4all: the final frame F (log2 units)
                        else:
                            log_pmax = (pm_row * softmax_scale_log2 * LN2 - lse) if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
                        if const_expr(not seqlen.has_cu_seqlens_q):
                            gPmax = cute.local_tile(mPmax[None, head_idx, batch_idx], (self.m_block_size,), (m_tile_idx,))
                        else:
                            # aligned varlen documents: (h, total_q), at the document's offset (as the LSE)
                            gPmax = cute.local_tile(cute.domain_offset((seqlen.offset_q if const_expr(not self.codes_packed)
                                                                        else mCuSeqlensO[batch_idx],),
                                                                       mPmax[None, head_idx]),
                                                    (self.m_block_size,), (m_tile_idx,))
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            gPmax[tidx] = log_pmax
                        elif const_expr(self.fp4all and self.zero_pad_rows and not self.codes_packed):
                            if m_tile_idx * self.m_block_size < seqlen_q:
                                gPmax[tidx] = Float32(0.0)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def _corr_pair_step(self, stage: cutlass.Constexpr[int], i: Int32, phase: Int32, n_block_max: Int32,
                        mbar_ptr: cute.Pointer, tidx: Int32, sScale: cute.Tensor, thr_mma_pv, tOtOs,
                        thr_tmem_load_vec, tStScales_t2r, tSrScale_t2r_shape):
        """two-cta-forward stage-0 skip: one rescale step of the correction loop for one Q stage (the generic loop's body
        on the default path: corr_empty_early)."""
        self._mbar_wait_h(mbar_ptr + self.mbar_softmax_corr_full_offset + stage, phase)
        if const_expr(self.max_tmem):
            tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
            cute.copy(thr_tmem_load_vec,
                      cute.make_tensor(tStScales_t2r[stage].iterator + ((n_block_max + i) & 1), tStScales_t2r[stage].layout),
                      tSrScale_t2r)
            cute.arch.fence_view_async_tmem_load()
            scale = tSrScale_t2r[0]
        else:
            scale = sScale[tidx + stage * self.m_block_size + ((n_block_max + i) & 1) * (4 * self.m_block_size)]
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
        if should_rescale:
            self.correction_rescale(thr_mma_pv, tOtOs[stage], tidx, scale)
        self._arrive_mma(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.atom.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)
        frg_count = self.head_dim_v_padded // corr_tile_size
        # corr_rescale_batch fragments in flight: issue the batch's tcgen05.ld
        # back to back, then multiply, then store, so the TMEM load latency is
        # exposed once per batch (batch=1 reproduces the original serialized loop).
        batch = max(1, min(self.corr_rescale_batch, frg_count))
        assert frg_count % batch == 0, "corr_rescale_batch must divide head_dim_v/16"
        for b in cutlass.range_constexpr(frg_count // batch):
            frags = []
            for k in cutlass.range_constexpr(batch):
                i = b * batch + k
                frag = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
                tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
                cute.copy(thr_tmem_load, tOtO_t2r_i, frag)
                frags.append(frag)
            for k in cutlass.range_constexpr(batch):
                frag = frags[k]
                for j in cutlass.range(0, cute.size(frag), 2, unroll_full=True):
                    frag[j], frag[j + 1] = utils.mul_packed_f32x2(
                        (frag[j], frag[j + 1]),
                        (scale, scale),
                    )
            for k in cutlass.range_constexpr(batch):
                i = b * batch + k
                tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
                cute.copy(thr_tmem_store, frags[k], tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.atom.ThrMma,
        tOtO: cute.Tensor, # tmem
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.atom.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor(softmax denominator) to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        # two-cta-forward: sO holds only this CTA's rows, so it is partitioned with the leader's (slice 0) mapping
        tOsO = (thr_mma.get_slice(0) if const_expr(self.use_2cta_forward) else thr_mma).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_forward,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0]).get_slice(
            tidx
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        if const_expr(self.o_regstore):
            # o_regstore: the gmem O tile partitioned like the tmem load (one row of
            # corr_tile_size contiguous columns per thread per chunk), stored with autovec_copy
            # (FlashInfer correction_epilog, st.global path); rows >= seqlen_q predicated
            assert not self.pack_gqa
            _gO_tile = gO[None, None, self.q_tiles * m_block + stage]
            tOgO = thr_mma.partition_C(_gO_tile)
            tOgO_i = cute.logical_divide(tOgO, cute.make_layout((self.m_block_size, corr_tile_size)))
            tOgO_r2g = thr_tmem_load.partition_D(tOgO_i[(None, None), None])
            _grow = (self.q_tiles * m_block + stage) * self.m_block_size + tOcO_t2r[0, 0, 0, 0][0]
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_rmem_tensor(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            if const_expr(self.o_regstore):
                if _grow < seqlen_q:
                    cute.autovec_copy(tOrO_frg_cvt, tOgO_r2g[None, 0, 0, i])
            else:
                cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        if const_expr(self.use_correction_warps_for_epi):
            assert(not self.use_tma_O)
            assert(gmem_tiled_copy_O is not None)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE)
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
            assert not self.pack_gqa
            pack_gqa = PackGQA(
                self.m_block_size,
                self.head_dim_v_padded,
                self.check_hdim_v_oob,
                self.qhead_per_kvhead,
            )
        
            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            # copy acc O from rmem to gmem
            if const_expr(not self.pack_gqa):
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen_q
                        - (self.q_tiles * m_block + stage) * self.m_block_size
                        - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None, self.q_tiles * m_block + stage],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_tiles * m_block + stage,
                    seqlen_q,
                )

    @cute.jit
    def add_delta_s(self, acc: cute.Tensor, sDeltaS: cute.Tensor, stage: int):
        """Add delta_s smoothing factors (computed from avg pooled qkv attn) to attention accumulator.
        
        This function implements the delta_s addition exactly like SageAttention:
        1. Recast delta_s to float4 for efficient processing
        2. Use quad-based indexing with thread coordination
        3. Apply delta_s values using complex coordinate indexing
        
        :param acc: Attention accumulator tensor to modify
        :type acc: cute.Tensor
        :param sDeltaS: Shared memory tensor containing delta_s values
        :type sDeltaS: cute.Tensor
        :param stage: Processing stage (0 or 1)
        :type stage: int
        """
        if const_expr(sDeltaS is None):
            return
            
        # Get thread index for quad-based processing (matches SageAttention)
        tidx, _, _ = cute.arch.thread_idx()
        quad_id = (tidx % 4) * 2
        
        # Recast delta_s to float4 for efficient processing (matches SageAttention)
        sDeltaS_stage = sDeltaS[None, None, stage]
        tSsDS_stage = cute.recast(sDeltaS_stage, Float32)
        
        # Recast accumulator to float4 for efficient processing
        acc_float4 = cute.recast(acc, Float32)
        
        # Process in groups of 4 float4 values (matches SageAttention pattern)
        for i in cutlass.range(0, 4, unroll=True):
            num = quad_id + i * 8
            
            # Load delta_s values for current quad using coordinate indexing
            # This matches the SageAttention pattern exactly
            delta_s_0 = tSsDS_stage[0, num]
            delta_s_1 = tSsDS_stage[0, num + 1]
            
            # Apply delta_s to accumulator using quad-based indexing
            # This follows the exact SageAttention coordinate pattern
            acc_float4[0, 0, i] += delta_s_0
            acc_float4[0, 1, i] += delta_s_0
            acc_float4[1, 0, i] += delta_s_1
            acc_float4[1, 1, i] += delta_s_1

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mma_tile_coord_v: Optional[Int32] = None,   # two-cta-forward: this CTA's index in the pair
        mCuSeqlensO: Optional[cute.Tensor] = None,  # o_rows_packed: the packed O row starts
    ):
        epi_consumer_phase = Int32(0)
        _cta_v = mma_tile_coord_v if const_expr(self.use_2cta_forward) else 0
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            seqlen_q_store = seqlen.seqlen_q
            if const_expr(self.zero_pad_rows and not self.o_rows_packed):
                seqlen_q_store = (seqlen.seqlen_q + (self.m_block_size - 1)) // self.m_block_size * self.m_block_size

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                elif const_expr(self.o_rows_packed):
                    mO_cur = cute.domain_offset((mCuSeqlensO[batch_idx],) + (None,) * (cute.rank(mO) - 1),
                                                mO)[None, None, head_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.o_regstore):
                    pass   # E41 / o_regstore: the correction warps stored O directly
                elif const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    if const_expr(self.epi_stage == 1):
                        # E42: one sO stage -- store tile 0, wait until the TMA engine has read
                        # smem, release, then tile 1 (the barrier pair cycles once per Q tile)
                        for stage in cutlass.range_constexpr(self.q_tiles):
                            self._mbar_wait_h(
                                mbar_ptr + self.mbar_corr_epi_full_offset + 0, epi_consumer_phase
                            )
                            if const_expr(self.varlen_tma_o):
                                if (self.q_tiles * m_block + stage) * self.m_block_size < seqlen.seqlen_q:
                                    store_O(src_idx=0, dst_idx=self.q_tiles * m_block + stage)
                            else:
                                store_O(src_idx=0, dst_idx=self.q_tiles * m_block + stage)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + 0)
                            epi_consumer_phase ^= 1
                    else:
                        for stage in cutlass.range_constexpr(self.q_tiles):
                            # wait from corr, issue tma store on smem
                            # 1. wait for O0 / O1 final
                            self._mbar_wait_h(
                                mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                            )
                            # 2. copy O0 / O1 to gmem (two-cta-forward: this CTA's half of the pair's 256-row group)
                            if const_expr(self.varlen_tma_o and self.use_2cta_forward):
                                # aligned documents on the pair: this CTA's 128-row tile, stored if it starts inside
                                # the document (a tile past seqlen_q may lie in the next slot)
                                _t2 = (self.q_tiles * m_block + stage) * self.cta_group_size + _cta_v
                                if const_expr(self.o_rows_tma):
                                    # packed rows: a whole tile by TMA, a document's partial last tile by row copies
                                    if (_t2 + 1) * self.m_block_size <= seqlen.seqlen_q:
                                        store_O(src_idx=stage, dst_idx=_t2)
                                    elif _t2 * self.m_block_size < seqlen.seqlen_q:
                                        self._o_rows_partial(sO[None, None, stage], mCuSeqlensO[batch_idx], head_idx,
                                                             _t2, seqlen.seqlen_q - _t2 * self.m_block_size,
                                                             gmem_tiled_copy_O)
                                elif _t2 * self.m_block_size < seqlen.seqlen_q:
                                    store_O(src_idx=stage, dst_idx=_t2)
                            elif const_expr(self.o_rows_tma1):
                                # packed rows (1-CTA, spd3): a whole tile by TMA, a document's partial last tile by row
                                # copies (a TMA tile would overwrite the next document's rows)
                                _t1 = self.q_tiles * m_block + stage
                                if (_t1 + 1) * self.m_block_size <= seqlen.seqlen_q:
                                    store_O(src_idx=stage, dst_idx=_t1)
                                elif _t1 * self.m_block_size < seqlen.seqlen_q:
                                    self._o_rows_partial_loop(sO[None, None, stage], mCuSeqlensO[batch_idx], head_idx,
                                                              _t1, seqlen.seqlen_q - _t1 * self.m_block_size,
                                                              gmem_tiled_copy_O)
                            elif const_expr(self.varlen_tma_o):
                                # aligned documents: a Q tile that starts past seqlen_q may lie in the next slot
                                if (self.q_tiles * m_block + stage) * self.m_block_size < seqlen.seqlen_q:
                                    store_O(src_idx=stage, dst_idx=self.q_tiles * m_block + stage)
                            else:
                                store_O(src_idx=stage,
                                        dst_idx=(self.q_tiles * m_block + stage) * self.cta_group_size + _cta_v)
                            cute.arch.cp_async_bulk_commit_group()
                        for stage in cutlass.range_constexpr(self.q_tiles):
                            # Ensure O0 / O1 buffer is ready to be released
                            cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
                    assert not self.pack_gqa
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    for stage in cutlass.range_constexpr(self.q_tiles):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        self._mbar_wait_h(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen_q_store
                                    - (self.q_tiles * m_block + stage) * self.m_block_size
                                    - tOcO[0][0]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[None, rest_m, None, self.q_tiles * m_block + stage],
                                        pred=tOpO[None, rest_m, None]
                                        if const_expr(self.check_hdim_v_oob)
                                        else None,
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur,
                                tOrO,
                                gmem_tiled_copy_O,
                                tidx,
                                self.q_tiles * m_block + stage,
                                seqlen.seqlen_q,
                            )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

                if const_expr(self.epi_stage == 2):
                    epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _leader_expect_tx(self, is_leader_cta: Boolean, mbar: cute.Pointer, nbytes: cutlass.Constexpr[int]):
        if is_leader_cta:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, nbytes)

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        load_SFQ_fn: Optional[Callable] = None,
        is_leader_cta=True,
    ):
        self._mbar_wait_h(mbar_empty_ptr + stage, phase)
        # two-cta-forward: only the leader arms its full barrier (with both CTAs' bytes); the peer's 2SM TMA signals it
        if const_expr(self.use_2cta_forward):
            self._leader_expect_tx(is_leader_cta, mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        else:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)
        
        # Load scale factor for Q if provided
        if const_expr(load_SFQ_fn is not None):
            load_SFQ_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def _o_rows_partial_loop(self, sO_st, ps, head_idx, tile, nr, gmem_tiled_copy_O):
        """o_rows_tma1 (spd3): _o_rows_partial as a rolled loop over the copy's row groups that hold a real row (one
        group's SMEM load + predicated global store per iteration). The unrolled copy (128 rows) is ~1.8 K SASS
        instructions of the epilogue warp in every packed causal kernel: measured +25 % on the d 64 forward, +2 % on
        whole-tile shapes (eq64x1024: no partial tile at all) through instruction fetch."""
        mO_cur = cute.domain_offset((ps,) + (None,) * (cute.rank(self._o_rows) - 1), self._o_rows)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        thr = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = thr.partition_S(sO_st)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOgO = thr.partition_D(gO)
        tOcO = thr.partition_S(cO)
        rows_m = self.m_block_size // cute.size(tOsO.shape[1])   # (rows per row group: the copy's thread rows)
        tOrO = cute.make_fragment_like(tOsO[None, 0, None], self.o_dtype)
        for rest_m in cutlass.range((nr + rows_m - 1) // rows_m, unroll=1):
            cute.autovec_copy(tOsO[None, rest_m, None], tOrO)
            if rest_m * rows_m + tOcO[0][0] < nr:
                cute.copy(gmem_tiled_copy_O, tOrO, tOgO[None, rest_m, None, tile])

    @cute.jit
    def _o_rows_partial(self, sO_st, ps, head_idx, tile, nr, gmem_tiled_copy_O):
        """o_rows_tma: the first nr rows of O tile `tile` (document-local 128-row tile) from its sO stage to the packed
        rows at ps (the document's packed start) -- the epilogue warp's predicated row copies (the rows past the
        document are the next document's and are not written)."""
        mO_cur = cute.domain_offset((ps,) + (None,) * (cute.rank(self._o_rows) - 1), self._o_rows)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        thr = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = thr.partition_S(sO_st)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOgO = thr.partition_D(gO)
        tOcO = thr.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < nr - tOcO[0][0]:
                cute.copy(gmem_tiled_copy_O, tOrO[None, rest_m, None], tOgO[None, rest_m, None, tile])

    @cute.jit
    def _sfd_publish(self, full, tx_bytes):
        """sfd_atoms.publish; the pair (o_rows_tma): the leader's elected thread arms its full barrier with both CTAs'
        bytes, the peer's arrives on the leader's barrier (cluster scope, after its own atom stores and proxy fence) --
        the leader's cta_group::2 scale-factor copies read both CTAs' atoms."""
        from flash_attn.cute import sfd_atoms as SA
        if const_expr(self.o_rows_tma):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                if self._pair_v == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full, tx_bytes)
                else:
                    cute.arch.mbarrier_arrive(full, peer_cta_rank_in_cluster=Int32(0),
                                              scope=_nvvm_scope_cluster())
        else:
            SA.publish(full, tx_bytes)

    @cute.jit
    def _sfd_kv_issue(self, kvs, t, n):
        """Staging copies of key tile t of the work tile (n_block = n_block_max - 1 - t) into ring slot n % slots: the
        K exponent bytes (E4M3 S) or K's NVFP4 SF words (NVFP4 S; from the 4-aligned row, at +288), the V exponent
        bytes (at +144), the bytes from the 16-aligned row."""
        mtk, mEvV, head_kv, psk0, n_block_max, sKvStg, mbar = kvs[:7]
        nk = self.head_dim_padded // 64
        # (n, ps >= 0, SFD_KV_SLOTS a power of 2: masks, not the signed % / // sequences)
        slot = n & (SFD_KV_SLOTS - 1)
        ps = psk0 + (n_block_max - 1 - t) * self.n_block_size
        p16 = ps & ~15
        base = sKvStg.iterator + slot * self.sfd_kv_slot
        with cute.arch.elect_one():
            if const_expr(self.fp8_sfd):
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + slot, 288)
                copy_utils.cpasync_bulk_g2s(mtk + p16, base, mbar + slot, 144)
            elif const_expr(self.v4_packed):
                # (cov5) K's NVFP4 SF words, V's dense SF rows (from the even row), the diagonal payload's tile word
                v4s = kvs[7]
                vt = v4s[8] + (n_block_max - 1 - t)
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + slot, 528 * nk + V4_SF_BYTES
                                                        + (16 if const_expr(self.vtok_diag) else 0))
                copy_utils.cpasync_bulk_g2s(mtk + (ps & ~3) * nk, base + 288, mbar + slot, 528 * nk)
                copy_utils.cpasync_bulk_g2s(v4s[6] + 2 * (ps & ~1), base + 288 + 528 * nk, mbar + slot, V4_SF_BYTES)
                if const_expr(self.vtok_diag):
                    copy_utils.cpasync_bulk_g2s(v4s[7] + (vt & ~3), base + 288 + 528 * nk + V4_SF_BYTES, mbar + slot,
                                                16)
            else:
                # (cov1: no V exponent bytes when V is a per-tensor E4M3 operand)
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + slot, (144 if self.quant_pv else 0) + 528 * nk)
                copy_utils.cpasync_bulk_g2s(mtk + (ps & ~3) * nk, base + 288, mbar + slot,
                                            528 * nk)
            if const_expr(self.quant_pv and not self.v4_packed):
                copy_utils.cpasync_bulk_g2s(mEvV + p16, base + 144, mbar + slot, 144)

    @cute.jit
    def _sfd_kv_word(self, sKvStg, slot, which: cutlass.Constexpr[int], ps, nr):
        """Key-tile staging slot `slot`'s tile word (K: which 0, V: 1), blocks past nr real rows 0; the slot already
        waited for. (sfd_atoms.combine_bytes: byte permutes -- only each load's low byte reaches the word, so a
        sign-extended exponent >= 128 cannot fill the higher bytes -- and one mask.)"""
        from flash_attn.cute import sfd_atoms as SA
        base = sKvStg.iterator + slot * self.sfd_kv_slot + 144 * which + (ps & 15)
        raw = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        for i in cutlass.range_constexpr(4):
            raw[i] = Int32(cute.make_tensor(base + 32 * i, cute.make_layout(1))[0])
        return SA.combine_bytes(raw, nr)

    @cute.jit
    def _sfd_load_K(self, tma_atom, tXgX, tXsX, mbar_ptr, block, producer_state, sSFK, kvs, seqlen_k, n,
                    part: cutlass.Constexpr[int] = 0, do_tma=None):
        """sf_dense: K tile `block` into KV stage producer_state.index, its SF atom from staging slot n. The staged
        word is read BEFORE the empty wait, so after it only the TMA, one 16-byte store, the fence and the arrive
        remain. part: 0 = everything, 1 = the TMA, 2 = the atom + arrive."""
        from flash_attn.cute import sfd_atoms as SA
        stage, phase = producer_state.index, producer_state.phase
        full = mbar_ptr + self.mbar_load_kv_full_offset + stage
        mtk, mEvV, head_kv, psk0, n_block_max, sKvStg, mbar = kvs[:7]
        ps = psk0 + block * self.n_block_size
        nr = seqlen_k - block * self.n_block_size
        w = Int32(0)
        wv = Int32(0)
        nk4 = self.head_dim_padded // 64
        v4 = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
        if const_expr(part != 1):
            # the slot's K word / NVFP4 SF words and V's word, read before the empty wait (V's is returned for
            # _sfd_load_V: one slot wait a key tile)
            slot = n & (SFD_KV_SLOTS - 1)
            cute.arch.mbarrier_wait(mbar + slot, (n >> SFD_KV_SLOTS.bit_length() - 1) & 1)
            if const_expr(self.fp8_sfd):
                w = self._sfd_kv_word(sKvStg, slot, 0, ps, nr)
            else:
                SA.read_sf4_from_smem(sKvStg.iterator + slot * self.sfd_kv_slot + 288, ps, nr, nk4, v4)
            if const_expr(self.quant_pv and not self.v4_packed):
                wv = self._sfd_kv_word(sKvStg, slot, 1, ps, nr)
        if const_expr(part != 2):
            # (do_tma: a dynamic condition -- False for a work tile's first key tile, whose TMA was issued already)
            if const_expr(do_tma is None):
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + stage, phase)
                cute.copy(tma_atom, tXgX[None, block], tXsX[None, stage], tma_bar_ptr=full)
            elif do_tma:
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + stage, phase)
                cute.copy(tma_atom, tXgX[None, block], tXsX[None, stage], tma_bar_ptr=full)
        if const_expr(part != 1):
            dst = sSFK.iterator.toint() + stage * self.sfd_bytes[1]
            if const_expr(self.fp8_sfd):
                SA.store_tok_atom(w, dst)
            else:
                SA.store_sf4_regs(v4, nk4, dst)
            self._sfd_publish(full, self.tma_copy_bytes["K"])
        return wv

    @cute.jit
    def _sfd_load_V(self, tma_atom, tXgX, tXsX, mbar_ptr, block, producer_state, sSFV, w,
                    part: cutlass.Constexpr[int] = 0, do_tma=None):
        """sf_dense: V tile `block` (the row codes, MN-major) into KV stage producer_state.index, its d-rows atom of
        word w (_sfd_load_K's return: the same staging slot; part as _sfd_load_K's)."""
        from flash_attn.cute import sfd_atoms as SA
        stage, phase = producer_state.index, producer_state.phase
        full = mbar_ptr + self.mbar_load_kv_full_offset + stage
        if const_expr(part != 2):
            if const_expr(do_tma is None):
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + stage, phase)
                cute.copy(tma_atom, tXgX[None, block], tXsX[None, stage], tma_bar_ptr=full)
            elif do_tma:
                self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + stage, phase)
                cute.copy(tma_atom, tXgX[None, block], tXsX[None, stage], tma_bar_ptr=full)
        if const_expr(part != 1):
            if const_expr(self.quant_pv):   # (cov1: a per-tensor V has no atom -- the arrive with its tx bytes only)
                SA.store_dim_atom(w, sSFV.iterator.toint() + stage * self.sfd_bytes[2])
            self._sfd_publish(full, self.tma_copy_bytes["V"])

    @cute.jit
    def _v4_issue(self, v4s, block, cnt):
        """(cov5 v4_packed) Key tile `block`'s 128 V^T row windows (80 bytes from the 16-byte-aligned column below the
        tile's first byte) into V staging slot cnt & 1 by one tensor TMA (the plain or the diagonal payload's copies)."""
        from flash_attn.cute import v4_packed as V4
        stg, smb = self._v4_stg
        b0d, n_diag_min, row0 = v4s[3], v4s[4], v4s[9]
        b0 = b0d + block * (self.n_block_size // 2)
        slot = cnt % V4.STAGES
        sstg = cute.make_tensor(cute.make_ptr(cutlass.Uint8, stg + slot * V4.STAGE_BYTES, cute.AddressSpace.smem,
                                              assumed_align=128),
                                cute.make_ordered_layout((128, V4.ROW_WINDOW), order=(1, 0)))
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(smb + slot, V4.STAGE_BYTES)
        if block >= n_diag_min:
            V4.tma_stage(v4s[12], v4s[13], row0, b0 & ~15, sstg, smb + slot)
        else:
            V4.tma_stage(v4s[10], v4s[11], row0, b0 & ~15, sstg, smb + slot)

    @cute.jit
    def _v4_load_V(self, mbar_ptr, block, producer_state, sSFV, sV, kvs, n, cnt, wait_empty):
        """(cov5 v4_packed) V tile `block` into KV stage producer_state.index: wait its staging slot (cnt), wait the
        stage empty (wait_empty: False for a work tile's first tile, waited already), realign the codes, the SF atoms
        from the key-tile staging slot n (K's, waited for by _sfd_load_K), fence and arrive."""
        from flash_attn.cute import v4_packed as V4
        from flash_attn.cute import sfd_atoms as SA
        stage, phase = producer_state.index, producer_state.phase
        full = mbar_ptr + self.mbar_load_kv_full_offset + stage
        sKvStg = kvs[5]
        v4s = kvs[7]
        stg, smb = self._v4_stg
        b0 = v4s[3] + block * (self.n_block_size // 2)
        nr = v4s[5] - block * self.n_block_size
        slot = cnt % V4.STAGES
        cute.arch.mbarrier_wait(smb + slot, (cnt // V4.STAGES) & 1)
        if wait_empty:
            self._mbar_wait_h(mbar_ptr + self.mbar_load_kv_empty_offset + stage, phase)
        dst = self._v4_sv + stage * self._v4_stage_bytes
        V4.realign(stg + slot * V4.STAGE_BYTES, b0 & 15, nr, dst)
        nk = self.head_dim_padded // 64
        sbase = sKvStg.iterator.toint() + (n & (SFD_KV_SLOTS - 1)) * self.sfd_kv_slot + 288 + 528 * nk
        ps = kvs[3] + block * self.n_block_size
        dsf = sSFV.iterator.toint() + stage * self.sfd_bytes[2]
        if block >= v4s[4]:
            vt = v4s[8] + block
            wd = cute.make_tensor(cute.make_ptr(Int32, sbase + V4_SF_BYTES + 4 * (vt & 3), cute.AddressSpace.smem,
                                                assumed_align=4), cute.make_layout(1))[0]
            V4.store_const_sf_atoms(wd, dsf)
        else:
            V4.store_sf_atoms(sbase + 8 * (ps & 1), nr, dsf)
        SA.publish(full, 0)

    @cute.jit
    def _sfd_load_Q(self, load_Q_fn, mbar_ptr, block, stage, phase, sSFQ, sfd_q, qt: cutlass.Constexpr[int],
                    part: cutlass.Constexpr[int] = 0):
        """sf_dense: the 128-row Q tile `block` (document-local) into Q stage `stage`, its SF atom built here from the
        work tile's prefetched bytes / words (sfd_q = (raw, packed start of the work tile, seqlen_q), tile qt).
        part: 0 = everything, 1 = the empty wait + TMA, 2 = the atom + arrive."""
        from flash_attn.cute import sfd_atoms as SA
        rq, psq0, seqlen_q = sfd_q
        nk4 = self.head_dim_padded // 64
        full = mbar_ptr + self.mbar_load_q_full_offset + stage
        if const_expr(part != 2):
            self._mbar_wait_h(mbar_ptr + self.mbar_load_q_empty_offset + stage, phase)
            load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=full)
        if const_expr(part != 1):
            if const_expr(self.o_rows_tma):
                nr = seqlen_q - (block * self.cta_group_size + self._pair_v) * self.m_block_size
            else:
                nr = seqlen_q - block * self.m_block_size
            dst = sSFQ.iterator.toint() + stage * self.sfd_bytes[0]
            if const_expr(self.fp8_sfd):
                SA.store_tok_atom(SA.gather_word(rq[0], 4 * qt, nr), dst)
            else:
                SA.store_sf4_atoms(cute.make_tensor(rq.iterator + 4 * nk4 * qt, cute.make_layout(4 * nk4)), nr, nk4,
                                   dst)
            self._sfd_publish(full, self.tma_copy_bytes["Q"])

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        tma_atom_sf: Optional[cute.CopyAtom] = None,
        tXgSF: Optional[cute.Tensor] = None,
        tXsSF: Optional[cute.Tensor] = None,
        vtok_ring: Optional[tuple] = None,  # (batch_idx, kv head, count base): stage the vtok ring slot of K(block)
        tma_atom_sf2: Optional[cute.CopyAtom] = None,  # vtok_tsf: the vtok float bytes, same barrier as V / SFV
        tXgSF2: Optional[cute.Tensor] = None,
        tXsSF2: Optional[cute.Tensor] = None,
        is_leader_cta=True,           # two-cta-forward: only the leader arms the full barrier
        sf_mcast_mask=None,           # two-cta-forward: the multicast mask of the B-side scale-factor tile
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        self._mbar_wait_h(mbar_empty_ptr + stage, phase)
        if const_expr(vtok_ring is not None):
            vr_b, vr_kvh, vr_base = vtok_ring
            self._vtok_ring_stage(vr_b, vr_kvh, block, (vr_base - block) % self.vtok_ring)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                self._mbar_wait_h(mbar_empty_ptr + 1, phase)

        if const_expr(self.use_tma_KV):
            assert (
                tXgX is not None and
                tXsX is not None and
                tma_atom is not None
            )
            if is_leader_cta:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V],
                    )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)
        
        # Load scale factor for K or V if provided (uses same barrier as K/V)
        if const_expr(tma_atom_sf is not None and tXgSF is not None and tXsSF is not None):
            tXsSF_cur = tXsSF[None, stage]
            # After tma_partition with rank-1 grouping, tXgSF has structure: ((atom_v, rest_v), RestL)
            if const_expr(page_idx is None):
                tXgSF_cur = tXgSF[None, block]
            else:
                tXgSF_cur = tXgSF[None, page_idx]
            if const_expr(sf_mcast_mask is not None):
                cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage, mcast_mask=sf_mcast_mask)
            else:
                cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage)
        if const_expr(tma_atom_sf2 is not None and tXgSF2 is not None and tXsSF2 is not None):
            cute.copy(tma_atom_sf2, tXgSF2[None, block], tXsSF2[None, stage], tma_bar_ptr=mbar_full_ptr + stage)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tCtSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr, use_k_bytes: bool = True, cta_layout_vmnk=None):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids) * (2 if self.o_rows_tma else 1)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"] if use_k_bytes else self.tma_copy_bytes["V"],
                **({"cta_layout_vmnk": cta_layout_vmnk} if cta_layout_vmnk is not None else {}),
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids) * cute.arch.WARP_SIZE
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        seqlen_info=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        # rank-2 coordinates, as softmax_step partitions S (thr_tmem_load has a rank-2 tiler): partitioning the rank-3
        # tScS handed the score_mod wrong q / kv indices (any index-dependent score_mod was wrong)
        tScS_t2r = thr_tmem_load.partition_D(tScS[(None, None), 0, 0])

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical = q_physical // self.qhead_per_kvhead
            head_offset = q_physical - q_idx_logical * self.qhead_per_kvhead
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(self.codes_packed and self.fp8_sfd and getattr(self.score_mod, "_lp_softcap", False)):
            # packed rows (cov4): a query row past its document holds the NEXT document's codes under a zero exponent
            # byte (scale 2^-127) -- a tiny S where the aligned rows' zero codes give S = 0; such values send the
            # score_mod's IEEE division (softcap: S / c) down its slow path (fp8 softcap forward 4186 -> 5718 us, h16/4
            # d128 lognorm900 T 131072). The row is never stored: S = 0 there, as on the aligned rows (4517 us). (The
            # keys past the document in the last key tile stay: a per-element select costs more than it saves.) Only
            # the softcap on E4M3 token-block S: NVFP4 S rows past the document are exact zeros, and the fill costs
            # +5 % (fp4 softcap) / +23 % (fp8 index-bias score_mod) forward time where no division slow path is hit.
            if q_idx_logical >= seqlen_info.seqlen_q:
                tSrS_t2r.fill(0.0)

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = seqlen_q_divmod.divmod(q_idx_logical)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            _as_aux_data(aux_tensors),
            fastdiv_mods,
            seqlen_info,  # masks / varlen: apply_score_mod_inner took seqlen_info upstream (score_mod was dead here)
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )

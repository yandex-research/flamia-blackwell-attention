# quantize_fp4_causal_diagonal: every operand of the causal diagonal routes (the all-FP4 step's
# `diagonal-only-key-local-v-scale` with the K channel, and option A's diagonal V), in ONE launch.
#
# Work items: one per (key tile t, batch, kv head) -- K and V of those 128 keys in the same CTA -- plus, optionally, one per
# (query tile, batch, q head) for Q's NVFP4 row copy (the S operand). Items are taken from an atomic counter in the order
# (key tile, then batch x kv head), so an item depends only on items handed out before it: deadlock-free.
#
# The cross-tile dependencies are scans over key tiles, done in-kernel with a decoupled look-back (CUB's single-pass
# pattern: every CTA publishes its tile aggregate with a status flag as soon as it has it, then walks back over its
# predecessors, adding aggregates until it meets a published inclusive prefix, and publishes its own inclusive prefix):
#   sums   the per-tile mean of e (K and V): c_n = rint(mean over key tiles [0, n - 2]), clamped to +-7, 0 for n < 2 --
#          the key-local payload's tile constant (exact: every partial sum is a multiple of 2^-7 below 2^15);
#   maxima the per-key ||V_k|| maximum (max(||V||, ||V|| u_k), times r_k under the K fold) and r_k: their causal prefix
#          maxima, written per query row (the backward's dS bound), replacing torch's cummax chain.
# Status flags carry the launch's epoch (workspace[2], advanced by the last CTA to finish), so the workspace never needs
# clearing; the tile counter and the finished-CTA counter are reset by that last CTA (CUDA-graph safe).
#
# Per key (e_k = ceil(log2(max(amax_k * f32(1/6), 2^-120))) in [-120, 120], from the row's plain amax):
#   V plain payload   NVFP4 16 x 16 blocks (nearest): row copy (v4 / sfv) + transposed copy into the first half of the
#                     forward's concatenated (b, hk, d, s) tile (vt / sfvt);
#   V key-local       codes of V / 2^e_k at unit scale: row copy (v4b) + transposed copy into the second half of vt; their
#                     scale bytes are all E4M3(2^c_n) (sfvb, sfvt's second half);
#   V per key         bias = (e - c) f32(1 / (scale log2 e)), u = 2^-e 2^c, s = 2^e (b, s, hk); c (b, tiles, hk); option A's
#                     r = 2^(e - c) as (b, hk, s);
#   K key-local       K^T codes of K / 2^e_k at unit scale in the kernels' key order (xt4b) with sftb = E4M3(2^c_n);
#   K plain (cut)     K^T NVFP4 1 x 16 along the keys, key order (xt4 / sft, and sft1 = the second d half's atom copy);
#   K per key         kr = [1 | r] (cut) or [r | r]; vb2 = [0 | bias]; vu2 = [1 | u] or, folded, [r | u r]; fold: dva =
#                     448 2^-A (A = max(0, max_k e_k - c)), fdk = 2^(c - e), fdv = 2^(A + c - e), and the dP operands' row
#                     scale bytes times r_k (sfv2, sfvb2).
# Bitwise the old builders (fp4q_diag_triton v_diag_fused / k_diag_fused_v, the prep chain of interface_bwd_fp4q) except the
# plain V payload's codes inside zero-scale groups (the encoding's zero-scale rule; dequantised values identical).
import math
from dataclasses import dataclass
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync

try:
    from . import fp4 as F
except ImportError:  # loaded by file path
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fp4.py")
    _spec = _ilu.spec_from_file_location("_quant_fp4_leaf", _p)
    F = _ilu.module_from_spec(_spec)
    _sys.modules["_quant_fp4_leaf"] = F
    _spec.loader.exec_module(F)
rot = F.rot

LOG2_E = 1.4426950408889634
SLOT0 = 16
SLOT = 16
K_MODES = ("off", "cut", "all", "fold")


# ---------------------------------------------------------------------------------------------------------------- helpers
def _asm_i32(op_text: str, args, cons: str, effects=False):
    return llvm.inline_asm(T.i32(), [a.ir_value() for a in args], op_text, cons, has_side_effects=effects,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_acquire(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value()], "ld.acquire.gpu.global.b32 $0, [$1];",
                                 "=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def st_release(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(val).ir_value()],
                    "st.release.gpu.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_relaxed(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value()], "ld.relaxed.gpu.global.b32 $0, [$1];",
                                 "=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def st_relaxed(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(val).ir_value()],
                    "st.relaxed.gpu.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


def div_rn(a, b) -> Float32:
    return F._asm_f32("div.rn.f32", a, b)


def rint(a) -> Float32:
    return F._asm_f32("cvt.rni.f32.f32", a)


def sqrt_rn(a) -> Float32:
    return F._asm_f32("sqrt.rn.f32", a)


def pow2i(e_float: Float32) -> Float32:
    """2^e from the exponent bits, for integer-valued e: ((int(e) + 127) << 23) -- the old builders' _pow2i, including its
    behaviour outside the normal range."""
    return F.bits_f32((Int32(e_float) + 127) << 23)


@cute.jit
def ceil_log2_amax6(amax: Float32):
    """(e as float, e as Int32): ceil(log2(max(amax * f32(1/6), 2^-120))), clamped to [-120, 120]; -0.0 when the log is in
    (-1, 0) (the torch reference's ceil gives -0.0 there and the sign survives into the raw-score bias)."""
    y = cute.arch.fmax(amax * Float32(0.16666667), Float32(7.52316384526264e-37))
    bits = F.f32_bits(y)
    ex = ((bits >> 23) & 0xFF) - 127
    e = ex
    if (bits & 0x7FFFFF) != 0:
        e = ex + 1
    if e < -120:
        e = Int32(-120)
    if e > 120:
        e = Int32(120)
    ef = Float32(e)
    if e == 0:
        if y < Float32(1.0):
            ef = Float32(-0.0)
    return ef, e


@cute.jit
def torch_row_norm(x: cute.Tensor) -> Float32:
    """||x||_2 of a 128-value row exactly as torch's x.float().norm(dim=-1) reduces it (recovered by probing
    torch: 32 lanes of 4 consecutive values, fma-accumulated in order, then an xor butterfly over the lanes 16, 8, 4, 2, 1),
    sqrt.rn."""
    p = cute.make_rmem_tensor(cute.make_layout(32), Float32)
    for g in cutlass.range_constexpr(32):
        a = F.mul_rn(x[4 * g], x[4 * g])
        for j in cutlass.range_constexpr(1, 4):
            a = F.fma_rn(x[4 * g + j], x[4 * g + j], a)
        p[g] = a
    s = F.xor_tree_sum(p, 32, (16, 8, 4, 2, 1))
    return sqrt_rn(s)


@cute.jit
def transpose_codes_to_smem(words: cute.Tensor, sB: cute.Tensor, r):
    """Row r's 128 E2M1 codes (16 words) -> sB[c, r // 2] = the byte (row 2k low nibble, row 2k + 1 high nibble) of d column c.
    The lane pair (2k, 2k + 1) exchanges half a row by shuffles; the even lane writes columns 0..63, the odd one 64..127."""
    odd = r % 2
    k = r // 2
    for w in cutlass.range_constexpr(8):
        send = words[w]
        if odd == 0:
            send = words[w + 8]
        recv = cute.arch.shuffle_sync_bfly(send, offset=1)
        mine = words[w + 8]
        if odd == 0:
            mine = words[w]
        lo_w = mine
        hi_w = recv
        if odd == 1:
            lo_w = recv
            hi_w = mine
        for e in cutlass.range_constexpr(8):
            byte = ((lo_w >> (4 * e)) & 0xF) | (((hi_w >> (4 * e)) & 0xF) << 4)
            sB[8 * w + e + odd * 64, k] = cutlass.Uint8(byte)


@cute.jit
def read_transposed_words(sB: cute.Tensor, dd, key_order: cutlass.Constexpr, out: cute.Tensor):
    """d row dd of the transposed tile: 64 bytes (128 tokens), its 32-token blocks in the kernels' order when key_order."""
    for blk in cutlass.range_constexpr(4):
        src = F._perm_block(blk) if const_expr(key_order) else blk
        cute.autovec_copy(F._aligned16(Int32, sB[dd, None].iterator + 16 * src, 4, cute.AddressSpace.smem), out[None, blk])


@cute.jit
def encode_unit_words(x: cute.Tensor, mult: Float32, words: cute.Tensor):
    """codes of x * mult at unit scale (RTN, saturating at 6): the key-local payload (mult = 2^-e_k, exact)."""
    for w in cutlass.range_constexpr(16):
        b = 8 * w
        words[w] = F.packed_float_to_e2m1(x[b] * mult, x[b + 1] * mult, x[b + 2] * mult, x[b + 3] * mult,
                                          x[b + 4] * mult, x[b + 5] * mult, x[b + 6] * mult, x[b + 7] * mult)


@cute.jit
def encode_nvfp4_row(x: cute.Tensor, words: cute.Tensor, sfw: cute.Tensor, blocks_2d: cutlass.Constexpr):
    """NVFP4 (nearest) of a 128-value row: 1 x 16 groups, or 16 x 16 blocks (amax shared by the 16 lanes of a row block)."""
    amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
    for g in cutlass.range_constexpr(8):
        m = Float32(0.0)
        for j in cutlass.range_constexpr(16):
            m = cute.arch.fmax(m, F.abs_ftz(x[16 * g + j]))
        amax[g] = m
    if const_expr(blocks_2d):
        for g in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(4):
                amax[g] = cute.arch.fmax(amax[g], cute.arch.shuffle_sync_bfly(amax[g], offset=1 << k))
    sfw[0] = Int32(0)
    sfw[1] = Int32(0)
    for g in cutlass.range_constexpr(8):
        byte, sfv, mult = F.nvfp4_scale_from_amax(amax[g], "nearest")
        sfw[g // 4] = sfw[g // 4] | ((byte & 0xFF) << (8 * (g % 4)))
        for w in cutlass.range_constexpr(2):
            b = 16 * g + 8 * w
            words[2 * g + w] = F.packed_float_to_e2m1(
                F.mul_ftz(x[b], mult), F.mul_ftz(x[b + 1], mult), F.mul_ftz(x[b + 2], mult), F.mul_ftz(x[b + 3], mult),
                F.mul_ftz(x[b + 4], mult), F.mul_ftz(x[b + 5], mult), F.mul_ftz(x[b + 6], mult),
                F.mul_ftz(x[b + 7], mult))


@cute.jit
def warp_lookback(mWork: cute.Tensor, bh, tiles, t, e1, flag_a: cutlass.Constexpr[int], flag_p: cutlass.Constexpr[int],
                  val_a: cutlass.Constexpr[int], val_p: cutlass.Constexpr[int], use_max: cutlass.Constexpr[bool],
                  jstart=None):
    """Exclusive prefix (two values) over key tiles [0, t) of a decoupled look-back chain, by one full warp: lane i looks at
    tile j0 - i; the nearest tile with a published inclusive prefix (or the chain's start) ends the walk, every closer tile
    must have published its aggregate (else the warp re-reads). Sums are exact here, so the combination order is free.
    jstart (documents): the chain starts at key tile jstart (the document's first tile), not 0."""
    lane = cute.arch.lane_idx()
    acc0 = Float32(0.0)
    acc1 = Float32(0.0)
    j0 = t - 1
    active = Int32(1)
    if const_expr(jstart is None):
        if t <= 0:
            active = Int32(0)
    else:
        if t <= jstart:
            active = Int32(0)
    while active != 0:
        j = j0 - lane
        st = Int32(3)                                   # 3: before the chain's start (an empty inclusive prefix)
        jin = j >= 0
        if const_expr(jstart is not None):
            jin = j >= jstart
        if jin:
            sj = SLOT0 + (bh * tiles + j) * SLOT
            st = Int32(0)
            if ld_acquire(mWork.iterator + sj + flag_p) == e1:
                st = Int32(2)
            elif ld_acquire(mWork.iterator + sj + flag_a) == e1:
                st = Int32(1)
        pmask = cute.arch.vote_ballot_sync(st >= 2)
        amask = cute.arch.vote_ballot_sync(st == 1)
        first = Int32(32)
        if pmask != 0:
            first = cute.arch.popc((pmask & (-pmask)) - 1)
        need = Int32(-1)
        if first < 32:
            need = (Int32(1) << first) - 1
        if (amask & need) == need:
            c0 = Float32(0.0)
            c1 = Float32(0.0)
            if lane < first:
                sj = SLOT0 + (bh * tiles + j) * SLOT
                c0 = F.bits_f32(ld_relaxed(mWork.iterator + sj + val_a))
                c1 = F.bits_f32(ld_relaxed(mWork.iterator + sj + val_a + 1))
            elif lane == first:
                if st == 2:
                    sj = SLOT0 + (bh * tiles + j) * SLOT
                    c0 = F.bits_f32(ld_relaxed(mWork.iterator + sj + val_p))
                    c1 = F.bits_f32(ld_relaxed(mWork.iterator + sj + val_p + 1))
            if const_expr(use_max):
                acc0 = cute.arch.fmax(acc0, cute.arch.warp_reduction_max(c0))
                acc1 = cute.arch.fmax(acc1, cute.arch.warp_reduction_max(c1))
            else:
                acc0 = acc0 + cute.arch.warp_reduction_sum(c0)
                acc1 = acc1 + cute.arch.warp_reduction_sum(c1)
            if first < 32:
                active = Int32(0)
            else:
                j0 = j0 - 32
    return acc0, acc1


# ---------------------------------------------------------------------------------------------------------------- kernel
@dataclass(frozen=True)
class _DiagSpec:
    input_dtype: object
    k_mode: str = "fold"                   # off | cut | all | fold
    with_query: bool = False               # Q's NVFP4 row copy (S operand) from the same launch
    with_key_rows: bool = False            # K's NVFP4 row copy (S operand)
    option_a_rk: bool = False              # r_k = 2^(e - c) as (b, hk, s) (option A's dP factor)
    prefix_maxima: bool = False            # per query row: causal prefix max of the dP-operand key norms (and of r_k)
    r_gqa: int = 1
    hadamard_signs: Optional[tuple] = None  # Q / K rotation (V is never rotated on this route)
    rope: str = "none"                     # Q / K RoPE: none | transposed_tables | row_tables
    query_rides_with_kv: bool = False      # s_q == s_k: the KV item of key tile t also quantises Q tile t of its q heads
    query_row_amax: bool = False           # per query row max |Q| (b, h, s_q) and two zeroed max slots (the backward's)
    # aligned varlen documents (b = 1, every document at a 128-aligned start; lp_varlen.LPDocs): the look-back scans
    # restart at each document's first key tile, the prefix maxima are written per document (its own diagonal offset
    # len_k - len_q), and the backward's max slots are per document ((n_docs + 1) x 2)
    docs: bool = False
    # cov5 (lp_docmaps compact maps, with docs): the caller's PACKED rows -- the key / query tiles are the maps' aligned
    # 128-row tiles (tile t = packed rows vt[0, t] + r, r < vt[1, t], zero rows after), persistent over the DEVICE tile
    # counts; every per-row output at the packed row, the SF bytes as dense per-row words, the transposed codes as the
    # key-parity copies (quant.fp4 transposed_row_packed), the per-tile values per aligned tile
    packed: bool = False


# workspace layout (int32): [0] item counter, [1] finished CTAs, [2] epoch; slots from 16, 16 words per key tile:
# [flag sums A, flag sums P, flag max A, flag max P, sumK A, sumV A, sumK P, sumV P, vmax A, rmax A, vmax P, rmax P, maxeK]
MIN_CTAS_PER_SM = 0     # occupancy hint (tuning)


class _DiagKernel:
    def __init__(self, spec: _DiagSpec):
        self.s = spec
        self.threads = 128

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
                 mQ4: cute.Tensor, mSFQ: cute.Tensor, mK4: cute.Tensor, mSFK: cute.Tensor,
                 mV4: cute.Tensor, mSFV: cute.Tensor, mVT: cute.Tensor, mSFVT: cute.Tensor, mV4B: cute.Tensor,
                 mSFVB: cute.Tensor, mBias: cute.Tensor, mU: cute.Tensor, mSK: cute.Tensor, mC: cute.Tensor,
                 mRK: cute.Tensor, mKT: cute.Tensor, mSFKT: cute.Tensor, mSFKT1: cute.Tensor, mKTB: cute.Tensor,
                 mSFKTB: cute.Tensor, mKR: cute.Tensor, mVB2: cute.Tensor, mVU2: cute.Tensor, mDVA: cute.Tensor,
                 mFDK: cute.Tensor, mFDV: cute.Tensor, mSFV2: cute.Tensor, mSFVB2: cute.Tensor, mVmax: cute.Tensor,
                 mRmax: cute.Tensor, mQAmax: cute.Tensor, mAmaxSlots: cute.Tensor, mWork: cute.Tensor,
                 inv_scale_log2e: Float32, offset: Int32, n_kv: Int32, n_items: Int32,
                 mDocTab: Optional[cute.Tensor], mTileDoc: Optional[cute.Tensor], mPk: Optional[list], stream):
        self.kernel(mQ, mK, mV, mCos, mSin, mQ4, mSFQ, mK4, mSFK, mV4, mSFV, mVT, mSFVT, mV4B, mSFVB, mBias, mU, mSK,
                    mC, mRK, mKT, mSFKT, mSFKT1, mKTB, mSFKTB, mKR, mVB2, mVU2, mDVA, mFDK, mFDV, mSFV2, mSFVB2, mVmax,
                    mRmax, mQAmax, mAmaxSlots, mWork, inv_scale_log2e, offset, n_kv, n_items, mDocTab, mTileDoc,
                    mPk).launch(
            grid=[n_items, 1, 1], block=[self.threads, 1, 1], smem=self.smem_bytes(), stream=stream,
            min_blocks_per_mp=MIN_CTAS_PER_SM)

    def smem_bytes(self):
        row = 128 + 8                                   # bf16 / fp16 staged rows (16 bytes of padding)
        n = 2 * 128 * row * 2 + 64                      # K and V tiles
        n += 2 * (128 * 80 + 64)                        # transposed code bytes (two tiles)
        n += 128 * 2 * 4 + 64                           # SF words of the tile's rows
        n += 64 * 4 + 64                                # reductions / broadcasts
        if self.s.k_mode == "cut":
            esz = 4 if self.s.hadamard_signs is not None else 2
            n += 128 * (128 + 16 // esz) * esz + 64     # K values transposed (the plain K^T copy)
        return n

    @cute.jit
    def stage_tile(self, mX: cute.Tensor, sX: cute.Tensor, bb, q0, hh):
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.s.input_dtype
        vec = 128 // dt.width
        cpr = 128 // vec
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dt, num_bits_per_copy=128)
        for i in cutlass.range_constexpr(cpr):
            k = i * 128 + tidx
            rr = k // cpr
            ch = k % cpr
            cute.copy(atom, F._aligned16(dt, mX[bb, q0 + rr, hh, None].iterator + ch * vec, vec, cute.AddressSpace.gmem),
                      F._aligned16(dt, sX[rr, None].iterator + ch * vec, vec, cute.AddressSpace.smem))

    @cute.jit
    def stage_packed(self, mX: cute.Tensor, sX: cute.Tensor, pb, nvr, hh):
        """(packed) the tile's rows pb + rr, rr < nvr, zero rows after (address clamped)."""
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.s.input_dtype
        vec = 128 // dt.width
        cpr = 128 // vec
        for i in cutlass.range_constexpr(cpr):
            k = i * 128 + tidx
            rr = k // cpr
            ch = k % cpr
            F._cp_async_16_zfill(Int32(sX[rr, None].iterator.toint()) + ch * 16,
                                 mX[0, cutlass.min(pb + rr, mX.shape[1] - 1), hh, None].iterator + ch * vec,
                                 cutlass.select_(rr < nvr, Int32(16), Int32(0)))

    @cute.jit
    def store_parity(self, mVTw: cute.Tensor, sB: cute.Tensor, hh, pb, nvr):
        """(packed) d row tidx of the tile's transposed codes into the key-parity copies (quant.fp4
        transposed_row_packed): copy pb & 1, bytes (pb + c) / 2 + j, j < ceil(nvr / 2)."""
        tidx, _, _ = cute.arch.thread_idx()
        c = pb & 1
        F.store_bytes_unaligned(mVTw[c, hh, tidx, None].iterator, (pb + c) >> 1, (nvr + 1) >> 1, sB[tidx, None].iterator)

    @cute.jit
    def store_row_packed(self, mC4: cute.Tensor, mSF: cute.Tensor, words: cute.Tensor, sfw: cute.Tensor, prow, hh,
                         nheads):
        """(packed) a row copy at packed row prow and its SF bytes as the dense per-row words (h, R, 2) (flat)."""
        cute.autovec_copy(words, cute.make_tensor(mC4[0, prow, hh, None].iterator, cute.make_layout(16)))
        R = mSF.shape[0] // (2 * nheads)
        mSF[(hh * R + prow) * 2] = sfw[0]
        mSF[(hh * R + prow) * 2 + 1] = sfw[1]

    @cute.jit
    def row_f32(self, sX: cute.Tensor, r, x: cute.Tensor):
        rowb = cute.make_rmem_tensor(cute.make_layout(128), self.s.input_dtype)
        cute.autovec_copy(cute.make_tensor(sX[r, None].iterator, cute.make_layout(128)), rowb)
        x.store(rowb.load().to(Float32))

    @cute.jit
    def qk_transform(self, x: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, pos):
        sp = self.s
        if const_expr(sp.rope != "none"):
            rot.rope_rotate_row(x, mCos, mSin, pos, sp.rope == "transposed_tables", 128)
            rb = cute.make_rmem_tensor(cute.make_layout(128), sp.input_dtype)
            rb.store(x.load().to(sp.input_dtype))
            x.store(rb.load().to(Float32))
        if const_expr(sp.hadamard_signs is not None):
            rot.hadamard_rotate_row(x, sp.hadamard_signs)

    @cute.jit
    def store_row_copy(self, mC4: cute.Tensor, mSF: cute.Tensor, words: cute.Tensor, sfw: cute.Tensor, bb, q, hh, sf_base, r):
        cute.autovec_copy(words, cute.make_tensor(mC4[bb, q, hh, None].iterator, cute.make_layout(16)))
        i = r % 32
        j = r // 32
        mSF[sf_base + i * 4 + j] = sfw[0]
        mSF[sf_base + 128 + i * 4 + j] = sfw[1]

    @cute.jit
    def store_row_amax(self, x: cute.Tensor, mQAmax: cute.Tensor, idx):
        """max |x| of the raw (untransformed) row: the input of the backward's per-tensor E4M3 Q amax."""
        m = Float32(0.0)
        for j in cutlass.range_constexpr(128):
            m = cute.arch.fmax(m, cute.math.absf(x[j]))
        mQAmax[idx] = m

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
               mQ4: cute.Tensor, mSFQ: cute.Tensor, mK4: cute.Tensor, mSFK: cute.Tensor,
               mV4: cute.Tensor, mSFV: cute.Tensor, mVT: cute.Tensor, mSFVT: cute.Tensor, mV4B: cute.Tensor,
               mSFVB: cute.Tensor, mBias: cute.Tensor, mU: cute.Tensor, mSK: cute.Tensor, mC: cute.Tensor,
               mRK: cute.Tensor, mKT: cute.Tensor, mSFKT: cute.Tensor, mSFKT1: cute.Tensor, mKTB: cute.Tensor,
               mSFKTB: cute.Tensor, mKR: cute.Tensor, mVB2: cute.Tensor, mVU2: cute.Tensor, mDVA: cute.Tensor,
               mFDK: cute.Tensor, mFDV: cute.Tensor, mSFV2: cute.Tensor, mSFVB2: cute.Tensor, mVmax: cute.Tensor,
               mRmax: cute.Tensor, mQAmax: cute.Tensor, mAmaxSlots: cute.Tensor, mWork: cute.Tensor,
               inv_scale_log2e: Float32, offset: Int32, n_kv: Int32, n_items: Int32,
               mDocTab: Optional[cute.Tensor] = None, mTileDoc: Optional[cute.Tensor] = None,
               mPk: Optional[list] = None):
        sp = self.s
        if const_expr(sp.packed):
            # (cov5) [key tile table (2, tiles), query tile table (2, q tiles), device tile counts (2,) [q, k]]
            self._pk = mPk
        if const_expr(sp.docs):
            # documents: (6, n_docs + 1) [Q start, K start, Q length, K length, ..] and the document of each key tile
            self._doc_tab = mDocTab
            self._tile_doc = mTileDoc
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        RS = 136
        sK = smem.allocate_tensor(sp.input_dtype, cute.make_layout((128, RS), stride=(RS, 1)), 16)
        sV = smem.allocate_tensor(sp.input_dtype, cute.make_layout((128, RS), stride=(RS, 1)), 16)
        sB = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((128, 80), stride=(80, 1)), 16)
        sB2 = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((128, 80), stride=(80, 1)), 16)
        sS = smem.allocate_tensor(Int32, cute.make_layout((128, 2), stride=(2, 1)), 16)
        sRed = smem.allocate_tensor(Float32, cute.make_layout(64), 16)
        sI = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        sKT = sK
        if const_expr(sp.k_mode == "cut"):
            kdt = Float32 if const_expr(sp.hadamard_signs is not None) else sp.input_dtype
            kpad = 16 // (kdt.width // 8)
            sKT = smem.allocate_tensor(kdt, cute.make_layout((128, 128 + kpad), stride=(128 + kpad, 1)), 16)
        epoch = ld_relaxed(mWork.iterator + 2)
        e1 = epoch + 1
        if tidx == 0:
            sI[0] = cute.arch.atomic_add(mWork.iterator, Int32(1))
        cute.arch.sync_threads()
        item = sI[0]
        if const_expr(sp.query_row_amax):
            # the backward's two max-reduced slots (max |Q| / F_q and max |dO|, interface_bwd_fp4q.fp4q_prepare): zero
            # here, in the forward, so the backward's dO launch can max-reduce into them with no fill launch
            if const_expr(sp.docs):
                # per document: (n_docs + 1) x [max |Q| / F_q, max |dO|]
                if item == 0:
                    for i in cutlass.range((mAmaxSlots.shape[0] - tidx + 127) // 128, unroll=1):
                        mAmaxSlots[tidx + 128 * i] = Float32(0.0)
            else:
                if item == 0 and tidx < 2:
                    mAmaxSlots[tidx] = Float32(0.0)
        nkv_heads = mK.shape[2]
        batch = mK.shape[0]
        s_k = mK.shape[1]
        tiles = s_k // 128
        if const_expr(sp.packed):
            # (cov5) persistent: items (key tile, kv head) over the device key tile count, then (query tile, q head);
            # each CTA takes the next item from the counter (in hand-out order: the look-back stays deadlock-free)
            tiles = self._pk[0].shape[1]
            ntk = self._pk[2][1]
            nkv_dev = ntk * nkv_heads
            tot = nkv_dev
            if const_expr(sp.with_query):
                tot = nkv_dev + self._pk[2][0] * mQ.shape[2]
            while item < tot:
                if item < nkv_dev:
                    hh = item % nkv_heads
                    t = item // nkv_heads
                    self.kv_item(mQ, mQ4, mSFQ, mK, mV, mCos, mSin, mK4, mSFK, mV4, mSFV, mVT, mSFVT, mV4B, mSFVB, mBias,
                                 mU, mSK, mC, mRK, mKT, mSFKT, mSFKT1, mKTB, mSFKTB, mKR, mVB2, mVU2, mDVA, mFDK, mFDV,
                                 mSFV2, mSFVB2, mVmax, mRmax, mQAmax, mWork, inv_scale_log2e, offset, sK, sV, sB, sB2,
                                 sS, sRed, sI, sKT, Int32(0), hh, t, tiles, nkv_heads, s_k, e1)
                else:
                    self.q_item_packed(mQ, mQ4, mSFQ, mQAmax, mCos, mSin, sK, item - nkv_dev)
                cute.arch.sync_threads()
                if tidx == 0:
                    sI[0] = cute.arch.atomic_add(mWork.iterator, Int32(1))
                cute.arch.sync_threads()
                item = sI[0]
        elif item < n_kv:
            bh = item % (batch * nkv_heads)
            t = item // (batch * nkv_heads)
            self.kv_item(mQ, mQ4, mSFQ, mK, mV, mCos, mSin, mK4, mSFK, mV4, mSFV, mVT, mSFVT, mV4B, mSFVB, mBias, mU, mSK, mC, mRK,
                         mKT, mSFKT, mSFKT1, mKTB, mSFKTB, mKR, mVB2, mVU2, mDVA, mFDK, mFDV, mSFV2, mSFVB2, mVmax, mRmax,
                         mQAmax, mWork, inv_scale_log2e, offset, sK, sV, sB, sB2, sS, sRed, sI, sKT, bh // nkv_heads,
                         bh % nkv_heads, t, tiles, nkv_heads, s_k, e1)
        else:
            if const_expr(sp.with_query and not sp.query_rides_with_kv):
                qi = item - n_kv
                qtiles = mQ.shape[1] // 128
                t = qi % qtiles
                hh = (qi // qtiles) % mQ.shape[2]
                bb = qi // (qtiles * mQ.shape[2])
                self.stage_tile(mQ, sK, bb, t * 128, hh)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
                self.row_f32(sK, tidx, x)
                if const_expr(sp.query_row_amax):
                    self.store_row_amax(x, mQAmax, ((bb * mQ.shape[2] + hh) * mQ.shape[1]) + t * 128 + tidx)
                self.qk_transform(x, mCos, mSin, t * 128 + tidx)
                words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
                encode_nvfp4_row(x, words, sfw, False)
                self.store_row_copy(mQ4, mSFQ, words, sfw, bb, t * 128 + tidx, hh,
                                    ((bb * mQ.shape[2] + hh) * qtiles + t) * 256, tidx)
        # the last CTA resets the counters and advances the epoch (every flag written in this launch becomes stale)
        # (packed: n_items is the grid, every CTA finishes once)
        cute.arch.sync_threads()
        if tidx == 0:
            done = cute.arch.atomic_add(mWork.iterator + 1, Int32(1), sem="acq_rel", scope="gpu")
            if done == n_items - 1:
                mWork[0] = Int32(0)
                mWork[1] = Int32(0)
                st_release(mWork.iterator + 2, e1)

    @cute.jit
    def q_item_packed(self, mQ: cute.Tensor, mQ4: cute.Tensor, mSFQ: cute.Tensor, mQAmax: cute.Tensor, mCos: cute.Tensor,
                      mSin: cute.Tensor, sK: cute.Tensor, qi):
        """(cov5 packed) Q's NVFP4 row copy of query tile t, head hq (qi = t * h + hq): codes at the packed rows, the SF
        bytes as dense per-row words; the raw rows' max |Q| per packed row."""
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        nh = mQ.shape[2]
        hq = qi % nh
        t = qi // nh
        pb = self._pk[1][0, t]
        nvr = self._pk[1][1, t] & Int32(0xFFFF)
        self.stage_packed(mQ, sK, pb, nvr, hq)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        self.row_f32(sK, tidx, x)
        if tidx < nvr:
            if const_expr(sp.query_row_amax):
                self.store_row_amax(x, mQAmax, hq * (mQAmax.shape[0] // nh) + pb + tidx)
        self.qk_transform(x, mCos, mSin, t * 128 + tidx)
        words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        encode_nvfp4_row(x, words, sfw, False)
        if tidx < nvr:
            self.store_row_packed(mQ4, mSFQ, words, sfw, pb + tidx, hq, nh)

    @cute.jit
    def kv_item(self, mQ: cute.Tensor, mQ4: cute.Tensor, mSFQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, mK4: cute.Tensor,
                mSFK: cute.Tensor, mV4: cute.Tensor, mSFV: cute.Tensor, mVT: cute.Tensor, mSFVT: cute.Tensor,
                mV4B: cute.Tensor, mSFVB: cute.Tensor, mBias: cute.Tensor, mU: cute.Tensor, mSK: cute.Tensor,
                mC: cute.Tensor, mRK: cute.Tensor, mKT: cute.Tensor, mSFKT: cute.Tensor, mSFKT1: cute.Tensor,
                mKTB: cute.Tensor, mSFKTB: cute.Tensor, mKR: cute.Tensor, mVB2: cute.Tensor, mVU2: cute.Tensor,
                mDVA: cute.Tensor, mFDK: cute.Tensor, mFDV: cute.Tensor, mSFV2: cute.Tensor, mSFVB2: cute.Tensor,
                mVmax: cute.Tensor, mRmax: cute.Tensor, mQAmax: cute.Tensor, mWork: cute.Tensor,
                inv_scale_log2e: Float32, offset: Int32, sK: cute.Tensor, sV: cute.Tensor, sB: cute.Tensor, sB2: cute.Tensor, sS: cute.Tensor, sRed: cute.Tensor,
                sI: cute.Tensor, sKT: cute.Tensor, bb, hh, t, tiles, nkv, s_k, e1):
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        r = tidx
        lane = tidx % 32
        warp = tidx // 32
        q0 = t * 128
        key = q0 + r
        bh = bb * nkv + hh
        kmode = sp.k_mode
        # documents: this key tile's position inside its document (t_loc), the chain start (jst) and the document's
        # [valid, Q start, Q length, K length]; batched: t itself, chains from tile 0
        t_loc = t
        jst = None
        dinfo = None
        if const_expr(sp.docs):
            dd = self._tile_doc[t]
            t_loc = t - self._doc_tab[1, dd] // 128
            dvalid = Int32(1)
            if dd == self._doc_tab.shape[1] - 1:
                dvalid = Int32(0)          # a whole padding tile (the sentinel document): a chain of its own
            if t_loc < 0:
                dvalid = Int32(0)
            if dvalid == 0:
                t_loc = Int32(0)
            jst = t - t_loc
            dinfo = (dvalid, self._doc_tab[4 if const_expr(sp.packed) else 0, dd], self._doc_tab[2, dd],
                     self._doc_tab[3, dd])   # (packed: the document's PACKED query start)
        # (cov5 packed: the tile's packed rows pb + r, r < nvr; pkey = this thread's packed key, pok = a real row)
        pb = nvr = pkey = pok = None
        if const_expr(sp.packed):
            pb = self._pk[0][0, t]
            nvr = self._pk[0][1, t] & Int32(0xFFFF)
            pkey = pb + r
            pok = r < nvr
            self.stage_packed(mV, sV, pb, nvr, hh)
        else:
            self.stage_tile(mV, sV, bb, q0, hh)
        cute.arch.cp_async_commit_group()
        if const_expr(kmode != "off" or sp.with_key_rows):
            if const_expr(sp.packed):
                self.stage_packed(mK, sK, pb, nvr, hh)
            else:
                self.stage_tile(mK, sK, bb, q0, hh)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)                            # V landed; K still in flight
        cute.arch.sync_threads()
        slot = SLOT0 + (bh * tiles + t) * SLOT

        # ---------------------------------------------------------------- V: payloads, e_V
        v = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        self.row_f32(sV, r, v)
        if const_expr(sp.query_rides_with_kv):
            cute.arch.sync_threads()                                # every row of V is in registers: sV is free
            self.stage_tile(mQ, sV, bb, q0, hh * sp.r_gqa)
            cute.arch.cp_async_commit_group()
        vamax = Float32(0.0)
        for j in cutlass.range_constexpr(128):
            vamax = cute.arch.fmax(vamax, cute.math.absf(v[j]))
        evf, evi = ceil_log2_amax6(vamax)
        vnorm = Float32(0.0)
        if const_expr(sp.prefix_maxima):
            vnorm = torch_row_norm(v)
        words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        encode_nvfp4_row(v, words, sfw, True)                       # plain: 16 x 16 blocks, nearest
        if const_expr(sp.packed):
            if pok:
                self.store_row_packed(mV4, mSFV, words, sfw, pkey, hh, nkv)
        else:
            self.store_row_copy(mV4, mSFV, words, sfw, bb, key, hh, (bh * tiles + t) * 256, r)
        sS[r, 0] = sfw[0]
        sS[r, 1] = sfw[1]
        transpose_codes_to_smem(words, sB, r)
        # key-local V codes (unit scale): row copy, and the transposed tile in the second byte tile (one barrier for both)
        vmul = pow2i(-evf)
        wl = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        encode_unit_words(v, vmul, wl)
        if const_expr(sp.packed):
            if pok:
                cute.autovec_copy(wl, cute.make_tensor(mV4B[0, pkey, hh, None].iterator, cute.make_layout(16)))
        else:
            cute.autovec_copy(wl, cute.make_tensor(mV4B[bb, key, hh, None].iterator, cute.make_layout(16)))
        transpose_codes_to_smem(wl, sB2, r)
        cute.arch.sync_threads()
        wt = cute.make_rmem_tensor(cute.make_layout((4, 4)), Int32)
        dd = tidx
        if const_expr(sp.packed):
            # (cov5) both payloads' transposed codes into their key-parity copies (mVT: plain, self._pk[3]: key-local);
            # the plain payload's SF: the row copy's dense words; the key-local one's: the per-tile word (below)
            self.store_parity(mVT, sB, hh, pb, nvr)
            self.store_parity(self._pk[3], sB2, hh, pb, nvr)
        else:
            read_transposed_words(sB, dd, False, wt)
            vt_it = mVT[bb, hh, dd, None].iterator
            cute.autovec_copy(wt, cute.make_tensor(vt_it + t * 16, cute.make_layout((4, 4))))
            # plain transposed SF (block scales of (row block, d group dd // 16)); atoms per (b, hk): 2 * s / 64
            g = dd // 16
            st0 = Int32(0)
            st1 = Int32(0)
            for gt in cutlass.range_constexpr(8):
                byte = (sS[16 * gt, g // 4] >> (8 * (g % 4))) & 0xFF
                if const_expr(gt < 4):
                    st0 = st0 | (byte << (8 * gt))
                else:
                    st1 = st1 | (byte << (8 * (gt - 4)))
            satoms = 2 * (s_k // 64)
            vt_base = (bh * satoms + 2 * t) * 128 + (dd % 32) * 4 + dd // 32
            mSFVT[vt_base] = st0
            mSFVT[vt_base + 128] = st1
            read_transposed_words(sB2, dd, False, wt)
            cute.autovec_copy(wt, cute.make_tensor(vt_it + (s_k // 8) + t * 16, cute.make_layout((4, 4))))
        if const_expr(sp.query_rides_with_kv):
            cute.arch.cp_async_wait_group(1)                        # K landed (Q may still be in flight)
        else:
            cute.arch.cp_async_wait_group(0)                        # K landed
        cute.arch.sync_threads()

        # ---------------------------------------------------------------- K: payloads, e_K
        ekf = Float32(0.0)
        if const_expr(kmode != "off" or sp.with_key_rows):
            kx = cute.make_rmem_tensor(cute.make_layout(128), Float32)
            self.row_f32(sK, r, kx)
            self.qk_transform(kx, mCos, mSin, key)
            if const_expr(sp.with_key_rows):
                kw = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                ksf = cute.make_rmem_tensor(cute.make_layout(2), Int32)
                encode_nvfp4_row(kx, kw, ksf, False)
                if const_expr(sp.packed):
                    if pok:
                        self.store_row_packed(mK4, mSFK, kw, ksf, pkey, hh, nkv)
                else:
                    self.store_row_copy(mK4, mSFK, kw, ksf, bb, key, hh, (bh * tiles + t) * 256, r)
            if const_expr(kmode != "off"):
                kamax = Float32(0.0)
                for j in cutlass.range_constexpr(128):
                    kamax = cute.arch.fmax(kamax, cute.math.absf(kx[j]))
                ekf, eki = ceil_log2_amax6(kamax)
                if const_expr(kmode == "cut"):
                    self.plain_kt(kx, sKT, mKT, mSFKT, mSFKT1, bb, hh, t, tiles, nkv, r)
                encode_unit_words(kx, pow2i(-ekf), words)
                transpose_codes_to_smem(words, sB, r)
                cute.arch.sync_threads()
                if const_expr(sp.packed):
                    # (cov5) the key-local K^T in natural key order into its key-parity copies (the backward realigns
                    # it into its key order)
                    self.store_parity(mKTB, sB, hh, pb, nvr)
                else:
                    read_transposed_words(sB, dd, True, wt)
                    cute.autovec_copy(wt, cute.make_tensor(mKTB[bb, dd, hh, None].iterator + t * 16,
                                                           cute.make_layout((4, 4))))

        # ---------------------------------------------------------------- sums look-back -> c_K, c_V
        se = cute.arch.warp_reduction_sum(ekf)
        sv = cute.arch.warp_reduction_sum(evf)
        me = cute.arch.warp_reduction_max(ekf)
        if lane == 0:
            sRed[warp] = se
            sRed[4 + warp] = sv
            sRed[8 + warp] = me
        cute.arch.sync_threads()
        mk = ((sRed[0] + sRed[1]) + (sRed[2] + sRed[3])) * Float32(0.0078125)     # per-tile means (exact)
        mv = ((sRed[4] + sRed[5]) + (sRed[6] + sRed[7])) * Float32(0.0078125)
        pk = Float32(0.0)
        pv = Float32(0.0)
        if tidx == 0:
            st_relaxed(mWork.iterator + slot + 4, F.f32_bits(mk))
            st_relaxed(mWork.iterator + slot + 5, F.f32_bits(mv))
            st_release(mWork.iterator + slot + 0, e1)
        if const_expr(sp.query_rides_with_kv):
            # the Q tiles of this key tile's q heads, between publishing the aggregate and walking back
            for j in cutlass.range_constexpr(sp.r_gqa):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                qx = cute.make_rmem_tensor(cute.make_layout(128), Float32)
                self.row_f32(sV, r, qx)
                if const_expr(sp.query_row_amax):
                    self.store_row_amax(qx, mQAmax, (bb * nkv * sp.r_gqa + hh * sp.r_gqa + j) * s_k + key)
                if const_expr(j + 1 < sp.r_gqa):
                    cute.arch.sync_threads()
                    self.stage_tile(mQ, sV, bb, q0, hh * sp.r_gqa + j + 1)
                    cute.arch.cp_async_commit_group()
                self.qk_transform(qx, mCos, mSin, key)
                qw = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                qsf = cute.make_rmem_tensor(cute.make_layout(2), Int32)
                encode_nvfp4_row(qx, qw, qsf, False)
                hq = hh * sp.r_gqa + j
                self.store_row_copy(mQ4, mSFQ, qw, qsf, bb, key, hq, ((bb * nkv * sp.r_gqa + hq) * tiles + t) * 256, r)
        if warp == 0:
            # exclusive prefix through t - 1, and the aggregate of t - 1 (sums through t - 2 = the difference, exact)
            pk, pv = warp_lookback(mWork, bh, tiles, t, e1, 0, 1, 4, 6, False, jst)
        if tidx == 0:
            st_relaxed(mWork.iterator + slot + 6, F.f32_bits(pk + mk))
            st_relaxed(mWork.iterator + slot + 7, F.f32_bits(pv + mv))
            st_release(mWork.iterator + slot + 1, e1)
            ck = Float32(0.0)
            cv = Float32(0.0)
            if t_loc >= 2:
                sp_ = SLOT0 + (bh * tiles + t - 1) * SLOT
                fa1 = ld_acquire(mWork.iterator + sp_ + 0)
                while fa1 != e1:
                    fa1 = ld_acquire(mWork.iterator + sp_ + 0)
                ak = F.bits_f32(ld_relaxed(mWork.iterator + sp_ + 4))
                av = F.bits_f32(ld_relaxed(mWork.iterator + sp_ + 5))
                cnt = Float32(t_loc - 1)
                ck = cute.arch.fmin(cute.arch.fmax(rint(div_rn(pk - ak, cnt)), Float32(-7.0)), Float32(7.0))
                cv = cute.arch.fmin(cute.arch.fmax(rint(div_rn(pv - av, cnt)), Float32(-7.0)), Float32(7.0))
            sRed[16] = ck
            sRed[17] = cv
        cute.arch.sync_threads()
        ck = sRed[16]
        cv = sRed[17]
        meK = cute.arch.fmax(cute.arch.fmax(sRed[8], sRed[9]), cute.arch.fmax(sRed[10], sRed[11]))

        # ---------------------------------------------------------------- per key: V
        cbv = F.e4m3_rn(pow2i(cv))
        fillv = cbv | (cbv << 8) | (cbv << 16) | (cbv << 24)
        if const_expr(sp.packed):
            # (cov5) the key-local payload's SF: one word per (kv head, aligned tile) (h_kv, tiles + 4)
            if tidx == 0:
                mSFVB[hh * (mSFVB.shape[0] // nkv) + t] = fillv
        else:
            mSFVT[(bh * satoms + (s_k // 64) + 2 * t) * 128 + r] = fillv
            mSFVT[(bh * satoms + (s_k // 64) + 2 * t) * 128 + 128 + r] = fillv
            mSFVB[(bh * tiles + t) * 256 + r] = fillv
            mSFVB[(bh * tiles + t) * 256 + 128 + r] = fillv
        rl = evf - cv
        bias = rl * inv_scale_log2e
        u = pow2i(-evf) * pow2i(cv)
        o = (bb * s_k + key) * nkv + hh
        # bias / u in the kernels' (b, h_k, s) layout (the host returns (b, s, h_k) views): no layout copy downstream
        if const_expr(sp.packed):
            # (cov5) (h_kv, T') per packed key: the forward reads [kv head, packed start + local key]
            tp_ = mBias.shape[0] // nkv
            if pok:
                mBias[hh * tp_ + pkey] = bias
                mU[hh * tp_ + pkey] = u
            elif pkey >= s_k:
                # the last key tile's rows past T: zeros (the forward reads a tile's 128 keys; past T no document
                # wrote them, and a NaN there would reach the masked scores' -inf through the bias)
                mBias[hh * tp_ + pkey] = Float32(0.0)
                mU[hh * tp_ + pkey] = Float32(0.0)
        elif const_expr(sp.docs):
            # documents: the forward's per-key bias / u are DOCUMENT-major, (n_docs, h_kv, L) with L >= every document's
            # key count, so the forward reads them exactly as batched -- [document, kv head, local key] -- with no key
            # offset live in its softmax step (one more live value spilled it: +27 % / +78 % on the all-4-bit / option A
            # forward). Padding tiles (the sentinel) and keys past L have no slot.
            dvalid_, _qs, _lq, _lk = dinfo
            ndoc_ = self._doc_tab.shape[1] - 1
            kl_ = mBias.shape[0] // (ndoc_ * nkv)
            klocal = t_loc * 128 + r
            if dvalid_ != 0:
                if klocal < kl_:
                    di_ = self._tile_doc[t]
                    mBias[(di_ * nkv + hh) * kl_ + klocal] = bias
                    mU[(di_ * nkv + hh) * kl_ + klocal] = u
        else:
            mBias[bh * s_k + key] = bias
            mU[bh * s_k + key] = u
        if const_expr(sp.packed):
            if pok:
                mSK[pkey * nkv + hh] = pow2i(evf)
        else:
            mSK[o] = pow2i(evf)
        if tidx == 0:
            mC[(bb * tiles + t) * nkv + hh] = cv
        if const_expr(sp.option_a_rk):
            if const_expr(sp.packed):
                if pok:
                    mRK[hh * (mRK.shape[0] // nkv) + pkey] = pow2i(evf) * pow2i(-cv)
            else:
                mRK[bh * s_k + key] = pow2i(evf) * pow2i(-cv)
        vn2 = cute.arch.fmax(vnorm, vnorm * u)

        # ---------------------------------------------------------------- per key: K
        rk = Float32(1.0)
        if const_expr(kmode != "off" and sp.packed):
            # (cov5 packed) per (kv head, aligned tile): the K^T payload's SF word (mSFKTB), dva; per packed key (h_kv,
            # T'): fdk, fdv; doubled (h_kv, 2 T') [first | second]: kr, vb2, vu2; the dP operands' row SF bytes x r_k as
            # dense per-row words (h_kv, R, 2): sfv2, sfvb2. Rows past T: neutral values (the backward reads whole tiles)
            cbk = F.e4m3_rn(pow2i(ck))
            fillk = cbk | (cbk << 8) | (cbk << 16) | (cbk << 24)
            if tidx == 0:
                mSFKTB[hh * (mSFKTB.shape[0] // nkv) + t] = fillk
            rk = pow2i(ekf) * pow2i(-ck)
            tp1 = mFDK.shape[0] // nkv if const_expr(kmode == "fold") else mKR.shape[0] // (2 * nkv)
            b2 = hh * 2 * tp1 + pkey
            if pok:
                mKR[b2] = rk
                mKR[b2 + tp1] = rk
                mVB2[b2] = Float32(0.0)
                mVB2[b2 + tp1] = bias
                if const_expr(kmode == "fold"):
                    mVU2[b2] = rk
                    mVU2[b2 + tp1] = u * rk
                else:
                    mVU2[b2] = Float32(1.0)
                    mVU2[b2 + tp1] = u
            elif pkey >= s_k:
                mKR[b2] = Float32(0.0)
                mKR[b2 + tp1] = Float32(0.0)
                mVB2[b2] = Float32(0.0)
                mVB2[b2 + tp1] = Float32(0.0)
                mVU2[b2] = Float32(0.0)
                mVU2[b2 + tp1] = Float32(0.0)
            pad = self._pk[4]
            if r == nvr:
                # the tile's padding-key values (every row past nvr is a zero row: the same values): the aligned layout's
                # [vu2 first, vu2 second, kr] of a padding key, which the backward substitutes for the keys past the
                # document's end (here the next document's) -- (1, h_kv, tiles, 4)
                pt = (hh * tiles + t) * 4
                if const_expr(kmode == "fold"):
                    pad[pt] = rk
                    pad[pt + 1] = u * rk
                else:
                    pad[pt] = Float32(1.0)
                    pad[pt + 1] = u
                pad[pt + 2] = rk
            if const_expr(kmode == "fold"):
                a = cute.arch.fmax(meK - ck, Float32(0.0))
                if tidx == 0:
                    mDVA[hh * tiles + t] = Float32(448.0) * pow2i(-a)
                if pok:
                    mFDK[hh * tp1 + pkey] = pow2i(ck - ekf)
                    mFDV[hh * tp1 + pkey] = pow2i(a + ck - ekf)
                    R2 = mSFV2.shape[0] // (2 * nkv)
                    for k in cutlass.range_constexpr(2):
                        w2 = Int32(0)
                        wb = Int32(0)
                        for q in cutlass.range_constexpr(4):
                            sv_ = F.e4m3_byte_value(sfw[k], q)
                            w2 = w2 | (F.e4m3_rn(cute.arch.fmin(sv_ * rk, Float32(448.0))) << (8 * q))
                            wb = wb | (F.e4m3_rn(cute.arch.fmin(pow2i(cv) * rk, Float32(448.0))) << (8 * q))
                        mSFV2[(hh * R2 + pkey) * 2 + k] = w2
                        mSFVB2[(hh * R2 + pkey) * 2 + k] = wb
                elif pkey >= s_k:
                    mFDK[hh * tp1 + pkey] = Float32(0.0)
                    mFDV[hh * tp1 + pkey] = Float32(0.0)
                vn2 = vn2 * rk
        elif const_expr(kmode == "off" and sp.packed):
            # (cov6, key channel 'off' = k_channel 'open') the tile's padding-key [1, u, 1] (the aligned layout's
            # doubled weight / r_k of a padding key), substituted by the backward past the document's end
            if r == nvr:
                pad = self._pk[4]
                pt = (hh * tiles + t) * 4
                pad[pt] = Float32(1.0)
                pad[pt + 1] = u
                pad[pt + 2] = Float32(1.0)
        elif const_expr(kmode != "off"):
            cbk = F.e4m3_rn(pow2i(ck))
            fillk = cbk | (cbk << 8) | (cbk << 16) | (cbk << 24)
            mSFKTB[(bh * (s_k // 64) + 2 * t) * 128 + r] = fillk
            mSFKTB[(bh * (s_k // 64) + 2 * t) * 128 + 128 + r] = fillk
            rk = pow2i(ekf) * pow2i(-ck)
            mKR[bh * 2 * s_k + s_k + key] = rk
            if const_expr(kmode == "cut"):
                mKR[bh * 2 * s_k + key] = Float32(1.0)
            else:
                mKR[bh * 2 * s_k + key] = rk
            mVB2[bh * 2 * s_k + key] = Float32(0.0)
            mVB2[bh * 2 * s_k + s_k + key] = bias
            if const_expr(kmode == "fold"):
                mVU2[bh * 2 * s_k + key] = rk
                mVU2[bh * 2 * s_k + s_k + key] = u * rk
                a = cute.arch.fmax(meK - ck, Float32(0.0))
                if tidx == 0:
                    mDVA[bh * tiles + t] = Float32(448.0) * pow2i(-a)
                mFDK[bh * s_k + key] = pow2i(ck - ekf)
                mFDV[bh * s_k + key] = pow2i(a + ck - ekf)
                # the dP operands' row scale bytes times r_k (plain V payload: this row's block scales; diagonal: 2^c_V)
                i = r % 32
                jj = r // 32
                for k in cutlass.range_constexpr(2):
                    w2 = Int32(0)
                    wb = Int32(0)
                    for q in cutlass.range_constexpr(4):
                        sv_ = F.e4m3_byte_value(sfw[k], q)
                        w2 = w2 | (F.e4m3_rn(cute.arch.fmin(sv_ * rk, Float32(448.0))) << (8 * q))
                        wb = wb | (F.e4m3_rn(cute.arch.fmin(pow2i(cv) * rk, Float32(448.0))) << (8 * q))
                    mSFV2[(bh * tiles + t) * 256 + 128 * k + i * 4 + jj] = w2
                    mSFVB2[(bh * tiles + t) * 256 + 128 * k + i * 4 + jj] = wb
                vn2 = vn2 * rk
            else:
                mVU2[bh * 2 * s_k + key] = Float32(1.0)
                mVU2[bh * 2 * s_k + s_k + key] = u

        # ---------------------------------------------------------------- maxima look-back -> per-query prefix maxima
        if const_expr(sp.prefix_maxima):
            self.prefix_maxima(vn2, rk, mVmax, mRmax, mWork, sRed, bb, hh, t, tiles, nkv, s_k, slot, e1, offset,
                               t_loc, jst, dinfo)

    @cute.jit
    def prefix_maxima(self, vn2: Float32, rk: Float32, mVmax: cute.Tensor, mRmax: cute.Tensor, mWork: cute.Tensor,
                      sRed: cute.Tensor, bb, hh, t, tiles, nkv, s_k, slot, e1, offset: Int32, t_loc=None, jst=None,
                      dinfo=None):
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp = tidx // 32
        bh = bb * nkv + hh
        # in-tile inclusive prefix max over the 128 keys (warp scan, then the warps' totals)
        pv = vn2
        pr = rk
        for k in cutlass.range_constexpr(5):
            ov = cute.arch.shuffle_sync_up(pv, offset=1 << k)
            orr = cute.arch.shuffle_sync_up(pr, offset=1 << k)
            if lane >= (1 << k):
                pv = cute.arch.fmax(pv, ov)
                pr = cute.arch.fmax(pr, orr)
        if lane == 31:
            sRed[24 + warp] = pv
            sRed[28 + warp] = pr
        cute.arch.sync_threads()
        av = cute.arch.fmax(cute.arch.fmax(sRed[24], sRed[25]), cute.arch.fmax(sRed[26], sRed[27]))
        ar = cute.arch.fmax(cute.arch.fmax(sRed[28], sRed[29]), cute.arch.fmax(sRed[30], sRed[31]))
        xv = Float32(0.0)
        xr = Float32(0.0)
        if warp == 0:
            if lane == 0:
                st_relaxed(mWork.iterator + slot + 8, F.f32_bits(av))
                st_relaxed(mWork.iterator + slot + 9, F.f32_bits(ar))
                st_release(mWork.iterator + slot + 2, e1)
            xv, xr = warp_lookback(mWork, bh, tiles, t, e1, 2, 3, 8, 10, True, jst)
        if tidx == 0:
            st_relaxed(mWork.iterator + slot + 10, F.f32_bits(cute.arch.fmax(xv, av)))
            st_relaxed(mWork.iterator + slot + 11, F.f32_bits(cute.arch.fmax(xr, ar)))
            st_release(mWork.iterator + slot + 3, e1)
            sRed[32] = xv
            sRed[33] = xr
        cute.arch.sync_threads()
        # the warps before this one, then the tiles before this one
        for w in cutlass.range_constexpr(3):
            if warp > w:
                pv = cute.arch.fmax(pv, sRed[24 + w])
                pr = cute.arch.fmax(pr, sRed[28 + w])
        pv = cute.arch.fmax(pv, sRed[32])
        pr = cute.arch.fmax(pr, sRed[33])
        if const_expr(sp.docs):
            # this document's query row of key k: q = k - (len_k - len_q) (bottom-right), written while 0 <= q < len_q;
            # with len_q > len_k the rows q < len_q - len_k see no key: 0 (torch's prefix max of an empty set here), from
            # the document's first key tile. The row stride is the Q allocation (s_k - offset = total_q).
            dvalid, qs0, lq, lk = dinfo
            ql = t_loc * 128 + tidx - (lk - lq)
            # (packed: qs0 is the packed query start, the row stride the packed query count -- s_k - offset there)
            if dvalid != 0:
                if ql >= 0:
                    if ql < lq:
                        for j in cutlass.range_constexpr(sp.r_gqa):
                            hq = hh * sp.r_gqa + j
                            mVmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + qs0 + ql] = pv
                            if const_expr(sp.k_mode == "fold"):
                                mRmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + qs0 + ql] = pr
                if t_loc == 0:
                    nz = lq - lk
                    if nz > 0:
                        for i in cutlass.range((nz - tidx + 127) // 128, unroll=1):
                            for j in cutlass.range_constexpr(sp.r_gqa):
                                hq = hh * sp.r_gqa + j
                                mVmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + qs0 + tidx + 128 * i] = Float32(0.0)
                                if const_expr(sp.k_mode == "fold"):
                                    mRmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + qs0 + tidx + 128 * i] = \
                                        Float32(0.0)
        else:
            q = t * 128 + tidx - offset
            if q >= 0:
                for j in cutlass.range_constexpr(sp.r_gqa):
                    hq = hh * sp.r_gqa + j
                    mVmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + q] = pv
                    if const_expr(sp.k_mode == "fold"):
                        mRmax[(bb * nkv * sp.r_gqa + hq) * (s_k - offset) + q] = pr

    @cute.jit
    def plain_kt(self, kx: cute.Tensor, sKT: cute.Tensor, mKT: cute.Tensor, mSFKT: cute.Tensor, mSFKT1: cute.Tensor,
                 bb, hh, t, tiles, nkv, r):
        """The plain K^T copy (cut mode): NVFP4 1 x 16 along the keys (nearest), the kernels' key order; sft1 holds the same
        bytes with the d row blocks [2, 3, 0, 1] (the second d half's own SF tile)."""
        sp = self.s
        tidx, _, _ = cute.arch.thread_idx()
        kdt = sKT.element_type
        kb = cute.make_rmem_tensor(cute.make_layout(128), kdt)
        kb.store(kx.load().to(kdt))
        for c in cutlass.range_constexpr(128):
            sKT[c, r] = kb[c]
        cute.arch.sync_threads()
        dd = tidx
        colb = cute.make_rmem_tensor(cute.make_layout(128), kdt)
        row_s8 = cute.logical_divide(sKT[dd, None], cute.make_layout(8))
        colb8 = cute.logical_divide(colb, cute.make_layout(8))
        for blk in cutlass.range_constexpr(4):
            src_blk = F._perm_block(blk)
            for v in cutlass.range_constexpr(4):
                cute.autovec_copy(row_s8[None, src_blk * 4 + v], colb8[None, blk * 4 + v])
        xt = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        xt.store(colb.load().to(Float32))
        words = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        encode_nvfp4_row(xt, words, sfw, False)
        cute.autovec_copy(words, cute.make_tensor(mKT[bb, dd, hh, None].iterator + t * 16, cute.make_layout(16)))
        bh = bb * nkv + hh
        base = (bh * (tiles * 2) + t * 2) * 128
        mSFKT[base + (dd % 32) * 4 + dd // 32] = sfw[0]
        mSFKT[base + 128 + (dd % 32) * 4 + dd // 32] = sfw[1]
        mSFKT1[base + (dd % 32) * 4 + (dd // 32 + 2) % 4] = sfw[0]
        mSFKT1[base + 128 + (dd % 32) * 4 + (dd // 32 + 2) % 4] = sfw[1]
        cute.arch.sync_threads()


# ---------------------------------------------------------------------------------------------------------------- host
_WORK = {}


def _diag_workspace(dev, n_slots: int):
    """(16 + 16 n_slots) int32, zero at allocation; never cleared afterwards (epoch-tagged flags, self-reset counters).
    One per (device, stream); it grows by reallocating (a fresh zero buffer: epoch 0, flags 0)."""
    if F.COMPILE_ONLY:
        return torch.empty(SLOT0 + SLOT * n_slots, dtype=torch.int32, device=dev)
    key = (str(dev), torch.cuda.current_stream(dev).cuda_stream)
    need = SLOT0 + SLOT * n_slots
    if key not in _WORK or _WORK[key].numel() < need:
        _WORK[key] = torch.zeros(max(need, 1 << 16), dtype=torch.int32, device=dev)
    return _WORK[key]


def quantize_fp4_causal_diagonal(key: torch.Tensor, value: torch.Tensor, softmax_scale: float, *,
                                 key_channel: str = "fold", query: Optional[torch.Tensor] = None,
                                 make_key_row_copy: bool = False, make_option_a_key_factors: bool = False,
                                 make_key_norm_prefix_maxima: bool = False, query_seqlen: Optional[int] = None,
                                 query_heads: Optional[int] = None, apply_hadamard_rotation: bool = False,
                                 hadamard_sign_seed: int = rot.DEFAULT_HADAMARD_SIGN_SEED,
                                 rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None,
                                 make_query_row_amax: bool = False,
                                 hadamard_rotation_method: Optional[str] = None,
                                 documents: Optional[dict] = None) -> dict:
    """Every operand of the causal diagonal routes in ONE launch (see the module header).

    key, value: (b, s_k, h_kv, 128) bf16 / fp16; query (optional): (b, s_q, h, 128), its NVFP4 row copy ("q4", "sfq").
    key_channel: "off" | "cut" | "all" | "fold" -- how dQ's K operand is kept from reading future keys' scales (the earlier
        key-channel modes 0 / 1 / all / fold); "off" builds only the V side (option A).
    make_key_row_copy: K's NVFP4 row copy ("k4", "sfk", the S operand). make_option_a_key_factors: "rk" (b, h_kv, s_k).
    make_key_norm_prefix_maxima: "vmax_q" (b, h, s_q): per query row, the causal prefix max over the keys it sees of the
        dP-operand norm max(||V_k||, ||V_k|| u_k) (x r_k under the fold) -- and, with the fold, "rmax_q", the prefix max of
        r_k -- the dS bound's key norms (query_seqlen / query_heads default to s_k / h_kv; s_k >= s_q).
    apply_hadamard_rotation / rope_*: applied to Q and K (never to V). softmax_scale: the raw-score bias's scale.
    hadamard_rotation_method: accepted for the quantisers' common signature; this kernel rotates with the butterfly
        (quant._rotations: its thread-per-row rows would need a shared fp32 tile next to its K / V / transposed tiles, and
        the tile kernel's tensor-core form measured slower than the butterfly; logged).
    make_query_row_amax (needs query): "query_row_amax" (b, h, s_q) float32, max |Q| of each raw query row (the backward's
        per-tensor E4M3 Q amax, saved by the forward), and "backward_amax_slots" (2,) float32 zeroed by this launch (the
        backward's dO launch max-reduces max |Q| / F_q and max |dO| into them: no fill launch).
    vtok_bias / vtok_u are (b, s_k, h_kv) views of (b, h_kv, s_k) storage: the forward / backward kernels' layout.
    documents (aligned varlen, b = 1): dict(doc_tab=(6, n_docs + 1) int32 (lp_varlen.LPDocs.doc_tab), tile_doc_k=(s_k / 128,)
        int32, doc_key_len=L): the c_n and prefix-maximum scans restart at every document's first key tile, the prefix
        maxima use each document's own diagonal offset, "backward_amax_slots" is ((n_docs + 1) * 2,): one pair per
        document, and vtok_bias / vtok_u are document-major (n_docs, L, h_kv) views (L >= every document's key count,
        a multiple of 128: the forward indexes them [document, kv head, local key]).
    documents["packed"] (cov5; lp_docmaps compact maps, b = 1): (key tile table (2, tiles), query tile table (2, q tiles),
        device tile counts (2,) int32 [q, k]) -- key / value / query are the caller's PACKED (1, T, h, 128) rows (any T):
        the tiles are the maps' aligned tiles over the device counts (a persistent grid); "v4" / "v4b" / "q4" / "k4" at
        the packed rows with the SF bytes as dense per-row words ("sfv_rows", "sfq_rows", "sfk_rows": (h, R * 2)
        int32, R = quant.fp8.dense_row_stride(T)); "vt4" / "vt4b_packed" the two payloads' transposed codes as key-parity
        copies (2, h_kv, d, quant.fp4.packed_vt_bytes(T)); "sfvb_tiles" (h_kv, tiles + 4) int32 the key-local payload's
        SF word per aligned tile; vtok_bias / vtok_u (1, T', h_kv) views of (1, h_kv, T') per packed key (T' = T +
        128); vtok_s (1, T, h_kv); vtok_c (1, tiles, h_kv); rk (1, h_kv, T'). Built: the V side, Q / K row copies
        (key_channel "off", no prefix maxima / RoPE / rotation).
    Returns a dict with the old builders' names (v4, sfv, vt4, sfvt, v4b, sfvb, vtok_bias, vtok_u, vtok_s, vtok_c; xt4, sft,
    sft1, xt4b, sftb, kr, vb2, vu2, dva, fdk, fdv, sfv2, sfvb2; q4, sfq, k4, sfk; rk; vmax_q, rmax_q)."""
    import numpy as np
    if key_channel not in K_MODES:
        raise ValueError(f"key_channel must be one of {K_MODES}, got {key_channel!r}")
    b, s, hk, d = value.shape
    pkd = documents.get("packed") if documents is not None else None
    if key.shape != value.shape:
        raise ValueError(f"key {tuple(key.shape)} and value {tuple(value.shape)} must have the same shape")
    if pkd is not None:
        return _diag_packed(key, value, softmax_scale, key_channel=key_channel, query=query,
                            make_key_row_copy=make_key_row_copy, make_option_a_key_factors=make_option_a_key_factors,
                            make_key_norm_prefix_maxima=make_key_norm_prefix_maxima, query_heads=query_heads,
                            apply_hadamard_rotation=apply_hadamard_rotation, rope_cos=rope_cos,
                            make_query_row_amax=make_query_row_amax, documents=documents)
    if d != 128 or s % 128:
        raise NotImplementedError(f"head_dim 128 and s_k % 128 == 0 (got {tuple(value.shape)})")
    if value.dtype not in (torch.bfloat16, torch.float16) or key.dtype != value.dtype:
        raise TypeError("key / value: bf16 or fp16, the same dtype")
    for nm, x_ in (("key", key), ("value", value)):
        if x_.stride(3) != 1:
            raise ValueError(f"{nm}: the head dim must be contiguous")
    s_q = query_seqlen if query_seqlen is not None else (query.shape[1] if query is not None else s)
    h_q = query_heads if query_heads is not None else (query.shape[2] if query is not None else hk)
    if h_q % hk:
        raise ValueError(f"query heads {h_q} not a multiple of kv heads {hk}")
    r_gqa = h_q // hk
    off = s - s_q
    docs = documents is not None
    if docs:
        if b != 1:
            raise ValueError("documents: the aligned varlen layout (b = 1)")
        dtab, tdoc = documents["doc_tab"], documents["tile_doc_k"]
        assert dtab.dtype == torch.int32 and dtab.dim() == 2 and dtab.shape[0] >= 4 and dtab.is_contiguous()
        assert tdoc.dtype == torch.int32 and tdoc.numel() == s // 128 and tdoc.is_contiguous()
    if make_key_norm_prefix_maxima and off < 0 and not docs:
        raise NotImplementedError("key-norm prefix maxima with s_q > s_k (queries that see no key) are not built")
    rope = "none"
    if rope_cos is not None:
        rope = "transposed_tables" if tuple(rope_cos.shape) == (64, s) else "row_tables"
    signs = rot.hadamard_signs(hadamard_sign_seed) if apply_hadamard_rotation else None
    if apply_hadamard_rotation or hadamard_rotation_method is not None:
        rot.resolve_rotation_method(hadamard_rotation_method, "quantize_fp4_causal_diagonal",
                                    "not built for the causal-diagonal kernel (thread-per-row, no spare shared memory for "
                                    "the fp32 rotated tile; the thread-per-row tile kernel measured slower on the tensor "
                                    "cores)" if apply_hadamard_rotation else None,
                                    tcgen05_unsupported=(
                                        "is not built for the causal-diagonal kernel (its K / V / transposed tiles leave "
                                        "no shared memory for the 64 KB UMMA operand images)"
                                        if apply_hadamard_rotation else None))
    spec = _DiagSpec(input_dtype=F._cute_dtype(value.dtype), k_mode=key_channel, with_query=query is not None,
                     with_key_rows=make_key_row_copy, option_a_rk=make_option_a_key_factors,
                     prefix_maxima=make_key_norm_prefix_maxima, r_gqa=r_gqa, hadamard_signs=signs, rope=rope,
                     query_rides_with_kv=query is not None and query.shape[1] == s,
                     query_row_amax=bool(make_query_row_amax), docs=docs)
    dev = value.device
    tiles = s // 128
    u8 = dict(dtype=torch.uint8, device=dev)
    f32 = dict(dtype=torch.float32, device=dev)
    di, df, dd_, dcos = F._dummies(dev)
    out = {}
    v4 = torch.empty(b, s, hk, 64, **u8)
    sfv = torch.empty(b, hk, tiles, 2, 32, 4, 4, **u8)
    vt = torch.empty(b, hk, d, s, **u8)
    sfvt = torch.empty(b, hk, 1, 2 * (s // 64), 32, 4, 4, **u8)
    v4b = torch.empty(b, s, hk, 64, **u8)
    sfvb = torch.empty(b, hk, tiles, 2, 32, 4, 4, **u8)
    if docs:
        # document-major (n_docs, h_kv, L), L = documents' key slot bound (doc_key_len, a multiple of 128)
        n_docs = documents["doc_tab"].shape[1] - 1
        L = int(documents["doc_key_len"])
        bias, u = (torch.empty(n_docs, hk, L, **f32) for _ in range(2))
    else:
        bias, u = (torch.empty(b, hk, s, **f32) for _ in range(2))
    sk = torch.empty(b, s, hk, **f32)
    c = torch.empty(b, tiles, hk, **f32)
    out.update(v4=v4, sfv=F._sf_view(sfv), vt4=vt, sfvt=F._sf_view(sfvt), v4b=v4b, sfvb=F._sf_view(sfvb),
               vtok_bias=bias.permute(0, 2, 1), vtok_u=u.permute(0, 2, 1), vtok_s=sk, vtok_c=c)
    W = lambda z: z.view(torch.int32).view(-1)                       # noqa: E731
    Wr = lambda z: z.view(torch.int32)                               # noqa: E731
    dummy_c = di.view(1, 1, 1, 16)
    rk = torch.empty(b, hk, s, **f32) if make_option_a_key_factors else None
    if rk is not None:
        out["rk"] = rk
    q4 = sfq = k4 = sfk = None
    if query is not None:
        if query.shape[0] != b or query.shape[3] != d or query.shape[1] % 128 or query.stride(3) != 1:
            raise ValueError(f"query {tuple(query.shape)}: (b, s_q, h, 128), s_q % 128 == 0")
        q4 = torch.empty(query.shape[0], query.shape[1], query.shape[2], 64, **u8)
        sfq = torch.empty(b, query.shape[2], query.shape[1] // 128, 2, 32, 4, 4, **u8)
        out.update(q4=q4, sfq=F._sf_view(sfq))
    if make_key_row_copy:
        k4 = torch.empty(b, s, hk, 64, **u8)
        sfk = torch.empty(b, hk, tiles, 2, 32, 4, 4, **u8)
        out.update(k4=k4, sfk=F._sf_view(sfk))
    km = key_channel
    kt = sft = sft1 = ktb = sftb = kr = vb2 = vu2 = dva = fdk = fdv = sfv2 = sfvb2 = None
    if km != "off":
        ktb = torch.empty(b, d, hk, s // 2, **u8)
        sftb = torch.empty(b, hk, 1, s // 64, 32, 4, 4, **u8)
        kr, vb2, vu2 = (torch.empty(b, hk, 2 * s, **f32) for _ in range(3))
        out.update(xt4b=ktb, sftb=F._sf_view(sftb), kr=kr, vb2=vb2, vu2=vu2)
        if km == "cut":
            kt = torch.empty(b, d, hk, s // 2, **u8)
            sft = torch.empty(b, hk, 1, s // 64, 32, 4, 4, **u8)
            sft1 = torch.empty(b, hk, 1, s // 64, 32, 4, 4, **u8)
            out.update(xt4=kt, sft=F._sf_view(sft), sft1=F._sf_view(sft1))
        if km == "fold":
            dva = torch.empty(b, hk, tiles, **f32)
            fdk, fdv = torch.empty(b, hk, s, **f32), torch.empty(b, hk, s, **f32)
            sfv2 = torch.empty_like(sfv)
            sfvb2 = torch.empty_like(sfvb)
            out.update(dva=dva, fdk=fdk, fdv=fdv, sfv2=F._sf_view(sfv2), sfvb2=F._sf_view(sfvb2))
    qamax = slots = None
    if make_query_row_amax:
        if query is None:
            raise ValueError("make_query_row_amax needs the query")
        qamax = torch.empty(b, query.shape[2], query.shape[1], **f32)
        slots = torch.empty(2 * documents["doc_tab"].shape[1] if docs else 2, **f32)
        out.update(query_row_amax=qamax, backward_amax_slots=slots)
    vmax = rmax = None
    if make_key_norm_prefix_maxima:
        vmax = torch.empty(b, h_q, s_q, **f32)
        out["vmax_q"] = vmax
        if km == "fold":
            rmax = torch.empty(b, h_q, s_q, **f32)
            out["rmax_q"] = rmax
    # PyTorch's fp32 tensor / Python scalar: the reciprocal in double, rounded to fp32, then an fp32 multiply
    inv = float(np.float32(1.0 / (softmax_scale * LOG2_E)))
    n_kv = b * hk * tiles
    n_items = n_kv + ((query.shape[0] * query.shape[2] * (query.shape[1] // 128))
                      if (query is not None and not spec.query_rides_with_kv) else 0)
    work = _diag_workspace(dev, n_kv)
    cos = rope_cos if rope_cos is not None else dcos
    sin = rope_sin if rope_sin is not None else dcos
    qx = query if query is not None else key
    args = [qx, key, value, cos, sin,
            Wr(q4) if q4 is not None else dummy_c, W(sfq) if sfq is not None else di,
            Wr(k4) if k4 is not None else dummy_c, W(sfk) if sfk is not None else di,
            Wr(v4), W(sfv), Wr(vt), W(sfvt), Wr(v4b), W(sfvb), bias.view(-1), u.view(-1), sk.view(-1), c.view(-1),
            rk.view(-1) if rk is not None else df,
            Wr(kt) if kt is not None else dummy_c, W(sft) if sft is not None else di, W(sft1) if sft1 is not None else di,
            Wr(ktb) if ktb is not None else dummy_c, W(sftb) if sftb is not None else di,
            kr.view(-1) if kr is not None else df, vb2.view(-1) if vb2 is not None else df,
            vu2.view(-1) if vu2 is not None else df, dva.view(-1) if dva is not None else df,
            fdk.view(-1) if fdk is not None else df, fdv.view(-1) if fdv is not None else df,
            W(sfv2) if sfv2 is not None else di, W(sfvb2) if sfvb2 is not None else di,
            vmax.view(-1) if vmax is not None else df, rmax.view(-1) if rmax is not None else df,
            qamax.view(-1) if qamax is not None else df, slots if slots is not None else df, work]
    _launch_diag(spec, args, inv, off, n_kv, n_items, (documents["doc_tab"], documents["tile_doc_k"]) if docs else (None, None))
    return out


_DIAG_CACHE = {}


def _diag_packed(key, value, softmax_scale, *, key_channel, query, make_key_row_copy, make_option_a_key_factors,
                 make_key_norm_prefix_maxima, query_heads, apply_hadamard_rotation, rope_cos, make_query_row_amax,
                 documents):
    """(cov5) quantize_fp4_causal_diagonal on the packed rows of compact maps (documents["packed"])."""
    import numpy as np
    from flash_attn.cute.quant.fp8 import dense_row_stride
    b, s, hk, d = value.shape
    if (b != 1 or d != 128 or key_channel not in ("off", "all", "fold") or apply_hadamard_rotation
            or rope_cos is not None or value.dtype not in (torch.bfloat16, torch.float16) or key.dtype != value.dtype
            or value.stride(3) != 1 or key.stride(3) != 1):
        raise NotImplementedError("quantize_fp4_causal_diagonal(documents=packed): (1, T, h, 128) 16-bit rows, "
                                  "key_channel 'off' / 'all' / 'fold', no rotation / RoPE")
    vt_k, vt_q, ntiles = documents["packed"]
    tiles = vt_k.shape[1]
    dtab, tdoc = documents["doc_tab"], documents["tile_doc_k"]
    assert tdoc.numel() == tiles and dtab.dtype == torch.int32
    h_q = query_heads if query_heads is not None else (query.shape[2] if query is not None else hk)
    spec = _DiagSpec(input_dtype=F._cute_dtype(value.dtype), k_mode=key_channel, with_query=query is not None,
                     with_key_rows=make_key_row_copy, option_a_rk=make_option_a_key_factors,
                     prefix_maxima=bool(make_key_norm_prefix_maxima),
                     r_gqa=h_q // hk, query_rides_with_kv=False, query_row_amax=bool(make_query_row_amax), docs=True,
                     packed=True)
    dev = value.device
    u8 = dict(dtype=torch.uint8, device=dev)
    f32 = dict(dtype=torch.float32, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    di, df, dd_, dcos = F._dummies(dev)
    R = dense_row_stride(s)
    TB = F.packed_vt_bytes(s)
    Tp = (s + 128 + 3) // 4 * 4
    out = {}
    v4 = torch.empty(1, s, hk, 64, **u8)
    sfv_rows = torch.empty(hk, R * 2, **i32)
    vt = torch.empty(2, hk, d, TB, **u8)
    vtb = torch.empty(2, hk, d, TB, **u8)
    v4b = torch.empty(1, s, hk, 64, **u8)
    sfvb_tiles = torch.empty(hk, (tiles + 4 + 3) // 4 * 4, **i32)   # (16-byte rows: the forward bulk-copies 4 words)
    bias, u = (torch.empty(1, hk, Tp, **f32) for _ in range(2))
    sk = torch.empty(1, s, hk, **f32)
    c = torch.empty(1, tiles, hk, **f32)
    ph = torch.empty(1, hk, 1, 2, 32, 4, 4, **u8)   # (the atom-view placeholders: the kernels read the dense words)
    out.update(v4=v4, sfv=F._sf_view(ph), sfv_rows=sfv_rows, vt4=vt, sfvt=F._sf_view(ph), vt4b_packed=vtb,
               sfvb_tiles=sfvb_tiles, v4b=v4b, sfvb=F._sf_view(ph), vtok_bias=bias.permute(0, 2, 1),
               vtok_u=u.permute(0, 2, 1), vtok_s=sk, vtok_c=c)
    W = lambda z: z.view(torch.int32).view(-1)                       # noqa: E731
    Wr = lambda z: z.view(torch.int32)                               # noqa: E731
    dummy_c = di.view(1, 1, 1, 16)
    rk = None
    if make_option_a_key_factors:
        rk = torch.empty(1, hk, Tp, **f32)
        out["rk"] = rk
    q4 = sfq_rows = k4 = sfk_rows = None
    if query is not None:
        if query.shape[0] != 1 or query.shape[3] != d or query.stride(3) != 1:
            raise ValueError(f"query {tuple(query.shape)}: packed (1, T_q, h, 128)")
        Rq = dense_row_stride(query.shape[1])
        q4 = torch.empty(1, query.shape[1], query.shape[2], 64, **u8)
        sfq_rows = torch.empty(query.shape[2], Rq * 2, **i32)
        out.update(q4=q4, sfq=F._sf_view(torch.empty(1, query.shape[2], 1, 2, 32, 4, 4, **u8)), sfq_rows=sfq_rows)
    if make_key_row_copy:
        k4 = torch.empty(1, s, hk, 64, **u8)
        sfk_rows = torch.empty(hk, R * 2, **i32)
        out.update(k4=k4, sfk=F._sf_view(ph), sfk_rows=sfk_rows)
    qamax = slots = None
    if make_query_row_amax:
        if query is None:
            raise ValueError("make_query_row_amax needs the query")
        qamax = torch.empty(1, query.shape[2], query.shape[1], **f32)
        slots = torch.empty(2 * dtab.shape[1], **f32)
        out.update(query_row_amax=qamax, backward_amax_slots=slots)
    km = key_channel
    ktb = sftb = kr = vb2 = vu2 = dva = fdk = fdv = sfv2 = sfvb2 = None
    if km != "off":
        # (the backward's K-channel operands: K^T key-local as key-parity copies, its SF per tile, the doubled per-key
        # tables (h_kv, 2 T') [first | second], fold's per-tile / per-key factors and the dP operands' SF words x r_k)
        ktb = torch.empty(2, hk, d, TB, **u8)
        sftb = torch.empty(hk, (tiles + 4 + 3) // 4 * 4, **i32)
        kr, vb2, vu2 = (torch.empty(hk, 2 * Tp, **f32) for _ in range(3))
        out.update(xt4b=ktb, sftb_tiles=sftb, kr=kr, vb2=vb2, vu2=vu2)
        if km == "fold":
            dva = torch.empty(hk, tiles, **f32)
            fdk, fdv = torch.empty(hk, Tp, **f32), torch.empty(hk, Tp, **f32)
            sfv2 = torch.empty(hk, R * 2, **i32)
            sfvb2 = torch.empty(hk, R * 2, **i32)
            out.update(dva=dva, fdk=fdk, fdv=fdv, sfv2_rows=sfv2, sfvb2_rows=sfvb2)
    vmax = rmax = None
    tq = query.shape[1] if query is not None else s
    if make_key_norm_prefix_maxima:
        vmax = torch.empty(1, h_q, tq, **f32)
        out["vmax_q"] = vmax
        if km == "fold":
            rmax = torch.empty(1, h_q, tq, **f32)
            out["rmax_q"] = rmax
    inv = float(np.float32(1.0 / (softmax_scale * LOG2_E)))
    n_slots = hk * tiles
    work = _diag_workspace(dev, n_slots)
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = max(1, min(hk * tiles + (query.shape[2] * vt_q.shape[1] if query is not None else 0), 2 * n_sm))
    qx = query if query is not None else key
    args = [qx, key, value, dcos, dcos,
            Wr(q4) if q4 is not None else dummy_c, sfq_rows.view(-1) if sfq_rows is not None else di,
            Wr(k4) if k4 is not None else dummy_c, sfk_rows.view(-1) if sfk_rows is not None else di,
            Wr(v4), sfv_rows.view(-1), Wr(vt), di, Wr(v4b), sfvb_tiles.view(-1), bias.view(-1), u.view(-1),
            sk.view(-1), c.view(-1), rk.view(-1) if rk is not None else df,
            dummy_c, di, di, Wr(ktb) if ktb is not None else dummy_c, sftb.view(-1) if sftb is not None else di,
            kr.view(-1) if kr is not None else df, vb2.view(-1) if vb2 is not None else df,
            vu2.view(-1) if vu2 is not None else df, dva.view(-1) if dva is not None else df,
            fdk.view(-1) if fdk is not None else df, fdv.view(-1) if fdv is not None else df,
            sfv2.view(-1) if sfv2 is not None else di, sfvb2.view(-1) if sfvb2 is not None else di,
            vmax.view(-1) if vmax is not None else df, rmax.view(-1) if rmax is not None else df,
            qamax.view(-1) if qamax is not None else df, slots if slots is not None else df, work]
    pad = torch.empty(1, hk, tiles, 4, **f32)   # (cov6: also key channel 'off')
    out["pad_keys"] = pad
    _launch_diag(spec, args, inv, s - tq, 0, grid, (dtab, tdoc), pk=[vt_k, vt_q, ntiles, Wr(vtb), pad.view(-1)])
    return out


def _launch_diag(spec, args, inv, off, n_kv, n_items, doc_args=(None, None), pk=None):
    T_ = F.to_cute_tensor
    key = ("fp4_diag", repr(spec), MIN_CTAS_PER_SM, "pk2")
    cache = F._jit_cache()
    if key not in cache:
        cargs = []
        for i, a in enumerate(args):
            if i < 3:
                cargs.append(T_(a, divisibility=8))
            elif a.dim() == 1:
                cargs.append(T_(a, assumed_align=4, leading_dim=0))
            elif a.dim() == 2:
                cargs.append(T_(a, assumed_align=4))
            else:
                cargs.append(T_(a, divisibility=4))
        dcargs = [T_(doc_args[0], assumed_align=4) if doc_args[0] is not None else None,
                  T_(doc_args[1], assumed_align=4, leading_dim=0) if doc_args[1] is not None else None]
        pkargs = () if pk is None else ([T_(pk[0], assumed_align=4), T_(pk[1], assumed_align=4),
                                         T_(pk[2], assumed_align=4, leading_dim=0), T_(pk[3], divisibility=4),
                                         T_(pk[4], assumed_align=4, leading_dim=0)],)
        cache[key] = cute.compile(_DiagKernel(spec), *cargs, Float32(1.0), Int32(0), Int32(1), Int32(1), *dcargs,
                                  pkargs[0] if pkargs else None,
                                  cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                                  options="--enable-tvm-ffi")
    if F.COMPILE_ONLY:
        return
    cache[key](*args, Float32(inv), Int32(off), Int32(n_kv), Int32(n_items), *doc_args,
               None if pk is None else list(pk))

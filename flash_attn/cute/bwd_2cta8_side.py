# Side kernels of the 2-CTA low-precision backward (flash_bwd_sm100_2cta8, driven by interface_bwd_2cta8).
#
# `Bwd2Cta8Prologue` is ONE launch in place of four: the dQ-accumulator fill (torch.zeros), upstream's BF16 preprocess
# (D = rowsum(dO * O), LSE * log2 e), and the two torch elementwise ops that followed it (LSE * log2 e - P_SHIFT and
# D * c, c = -ds_b / (s_v s_dO)). The values it writes are the same bits the four launches wrote:
#   * D uses upstream's thread layout, per-thread reduce and 16-lane butterfly verbatim, so the summation order and
#     therefore the fp32 result are identical; the one extra op is a multiply of the finished sum by Float32(c),
#     which is what torch's `mul_` by a Python scalar computes (the scalar is cast to fp32 first).
#   * LSE * log2 e is upstream's expression, and the shift is subtracted with an explicit `sub.rn.f32`: a plain
#     subtract after the multiply is contracted into one FFMA by ptxas, which rounds once where torch rounded twice.
#   * the accumulator is zeroed over the same (b, h, m-block) extent the 2-CTA kernel reduce-adds into.
# The kernel serves the 2-CTA layout only: seqlen_q % 128 == 0, head_dim 128, no varlen, no pack-GQA.
import math
import operator
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from quack import copy_utils

from flash_attn.cute import utils


@dsl_user_op
def sub_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """a - b rounded once, never contracted with the producer of `a` (an explicit rounding mode forbids it)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class Bwd2Cta8Prologue:
    """D * c, LSE * log2 e - shift, and a zeroed dQ accumulator, for the 2-CTA backward, in one launch.

    Grid (seqlen_q / 128, heads, batch), 256 threads: upstream's preprocess geometry (16 threads per 128-wide row,
    8 rows per pass), which is what fixes D's summation order. Each CTA also zeroes its m-block's 128 x 128
    accumulator elements (32 KB in f16, 64 KB in fp32) with coalesced 16-byte stores.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], acc_dtype, head_dim: int = 128, tile_m: int = 128,
                 num_threads: int = 256, zero_late: bool = False):
        assert head_dim == 128 and tile_m == 128 and num_threads == 256
        self.zero_late = zero_late  # accumulator stores after the O / dO loads are issued, not before
        self.dtype = dtype
        self.acc_dtype = acc_dtype  # None: no accumulator to zero
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (b, s, h, d) bf16 / f16
        mdO: cute.Tensor,  # same
        mLSE: cute.Tensor,  # (b, h, s) fp32, natural log
        mPdPsum: cute.Tensor,  # (b, h, s) fp32 out: D * d_scale
        mLSElog2: cute.Tensor,  # (b, h, s) fp32 out: LSE * log2 e - lse_shift (0 - lse_shift where LSE = -inf)
        mdQaccum: cute.Tensor,  # (b, h, s * d) f16 / fp32, zeroed; may be None
        d_scale: Float32,
        lse_shift: Float32,
        stream: cuda.CUstream,
    ):
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        # (b, s, h, d) -> (s, d, h, b), exactly upstream's view
        mO, mdO = [cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=[1, 3, 2, 0])) for mX in (mO, mdO)]
        num_copy_elems = 128 // self.dtype.width
        gmem_tiled_copy_O = copy_utils.tiled_copy_2d(self.dtype, 128 // num_copy_elems, self.num_threads, num_copy_elems)
        grid = [cute.ceil_div(mO.shape[0], self.tile_m), mO.shape[2], mO.shape[3]]
        self.kernel(mO, mdO, mLSE, mPdPsum, mLSElog2, mdQaccum, d_scale, lse_shift, gmem_tiled_copy_O).launch(
            grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.jit
    def zero_acc(self, mdQaccum: cute.Tensor, tidx, m_block, head_idx, batch_idx):
        """This m-block's 128 x 128 accumulator elements, coalesced 16-byte stores."""
        if const_expr(self.acc_dtype is not None):
            elems16 = 128 // self.acc_dtype.width  # elements per 16-byte store
            n_tile = self.tile_m * self.head_dim
            base = mdQaccum[batch_idx, head_idx, None].iterator + m_block * n_tile
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
            zero = cute.make_rmem_tensor(cute.make_layout(elems16), self.acc_dtype)
            zero.fill(0)
            for i in cutlass.range_constexpr(n_tile // (elems16 * self.num_threads)):
                ptr = cute.make_ptr(self.acc_dtype, (base + (i * self.num_threads + tidx) * elems16).toint(),
                                    cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(elems16)))

    @cute.kernel
    def kernel(self, mO: cute.Tensor, mdO: cute.Tensor, mLSE: cute.Tensor, mPdPsum: cute.Tensor,
               mLSElog2: cute.Tensor, mdQaccum: cute.Tensor, d_scale: Float32, lse_shift: Float32,
               gmem_tiled_copy_O: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        if const_expr(not self.zero_late):
            self.zero_acc(mdQaccum, tidx, m_block, head_idx, batch_idx)

        # ---- D: upstream FlashAttentionBackwardPreprocess.kernel, non-varlen, head_dim 128, verbatim arithmetic
        seqlen_q = mO.shape[0]
        seqlen_limit = seqlen_q - m_block * self.tile_m
        mO_cur, mdO_cur = [mX[None, None, head_idx, batch_idx] for mX in (mO, mdO)]
        gLSE = cute.local_tile(mLSE[batch_idx, head_idx, None], (self.tile_m,), (m_block,))
        lse = Float32.inf
        if tidx < seqlen_limit:
            lse = gLSE[tidx]
        blk_shape = (self.tile_m, self.head_dim)
        gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
        gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOgO = gmem_thr_copy_O.partition_S(gO)
        tOgdO = gmem_thr_copy_O.partition_S(gdO)
        cO = cute.make_identity_tensor(blk_shape)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOrO = cute.make_rmem_tensor_like(tOgO)
        tOrdO = cute.make_rmem_tensor_like(tOgdO)
        for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
            if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                copy_utils.copy(tOgO[None, m, None], tOrO[None, m, None])
                copy_utils.copy(tOgdO[None, m, None], tOrdO[None, m, None])
        if const_expr(self.zero_late):
            self.zero_acc(mdQaccum, tidx, m_block, head_idx, batch_idx)
        pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
        )
        threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
        pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
        PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
        PdP_sum.store(pdpsum)
        gPdPsum = cute.local_tile(mPdPsum[batch_idx, head_idx, None], (self.tile_m,), (m_block,))
        if tOcO[0, 0, 0][1] == 0:
            for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                row = tOcO[0, m, 0][0]
                PdPsum_val = 0.0
                if row < seqlen_limit:
                    PdPsum_val = PdP_sum[m]
                gPdPsum[row] = PdPsum_val * d_scale

        LOG2_E = math.log2(math.e)
        lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0
        gLSElog2 = cute.local_tile(mLSElog2[batch_idx, head_idx, None], (self.tile_m,), (m_block,))
        if tidx < self.tile_m:
            gLSElog2[tidx] = sub_rn_f32(lse_log2, lse_shift)


class Bwd2Cta8Postprocess:
    """dQ = f16 accumulator * scale -> (b, s, h, 128) bf16 / f16, for the 2-CTA accumulator order (see
    dq_postprocess_cute.DQPostprocess2CTA for the layout). The same arithmetic per element as that kernel (f16 -> f32
    exact, one FMUL2 by `scale`, one cvt.rn to the output type), so the same accumulator gives the same bits; what
    changes is the thread mapping. DQPostprocess2CTA gives each thread one query row, so every 16-byte store of a warp
    lands in 32 different rows (4 KB apart), and each thread holds 128 values. Here a warp covers two rows x 16
    column chunks per step: its stores are two contiguous 256-byte row segments, its loads are 16 full 32-byte
    sectors (rows r and r + 1 of a column chunk are adjacent in the accumulator), and a thread holds 8 chunks.

    Grid (seqlen_q / 128, heads, batch), 256 threads; 8 steps of 16 rows. zero_after writes zeros back over the
    accumulator tile (the `pool` policy of HostSpec.dq_zero).

    had_signs (qk_hadamard; the 128 signs, compile-time): the accumulator holds dQ R, and the SAME launch rotates it back
    (kernel_had): a lane pair per query row with interleaved 8-column chunks (qk_hadamard_cute.unrotate_pair64; one
    shuffle level), bitwise DQPostprocess2CTA(had_signs) -- which it replaces -- at about the unrotated kernel's speed
    (A1: 26.4 vs 23.7 us unrotated, 38.2 the thread-per-row form)."""

    def __init__(self, out_dtype, zero_after: bool = False, tiles: int = 1, had_signs=None, tile_scale: bool = False,
                 row_scale: bool = False):
        # dqfix: mScale is the per-(b, 128-row tile) table, index b tq + tile (device_scales.dq_tile_planes)
        self.tile_scale = tile_scale
        self.row_scale = row_scale   # dq-causal: mScale is the per-(b, row) plane (b * s_q floats)
        assert not row_scale or tile_scale
        self.acc_dtype = cutlass.Float16
        self.out_dtype = out_dtype
        self.zero_after = zero_after
        self.tiles = tiles  # 128-query tiles per CTA, all loads issued before any store
        self.threads = 256
        self.had_signs = had_signs
        assert had_signs is None or tiles == 1

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream,
                 mScale: cute.Tensor = None):
        grid = [cute.ceil_div(mdQ.shape[1], 128 * self.tiles), mdQ.shape[2], mdQ.shape[0]]
        if const_expr(self.had_signs is not None):
            self.kernel_had(mAcc, mdQ, scale, mScale).launch(grid=grid, block=[256, 1, 1], stream=stream)
        else:
            self.kernel(mAcc, mdQ, scale, mScale).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel_had(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None):
        if const_expr(mScale is not None and not self.tile_scale):
            scale = mScale[0]  # a device scale (as kernel)
        from flash_attn.cute.qk_hadamard_cute import unrotate_pair64
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        if const_expr(mScale is not None and self.tile_scale and not self.row_scale):
            scale = mScale[bb * (mAcc.shape[2] // 16384) + t]
        h = tidx % 2
        q = tidx // 2  # the query row within the tile; chunk cc = 2 k + h holds columns 8 cc .. 8 cc + 7
        if const_expr(mScale is not None and self.row_scale):   # dq-causal: this row's own scale
            scale = mScale[bb * (mAcc.shape[2] // 128) + t * 128 + q]
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        acc = cute.make_rmem_tensor(cute.make_layout(64), self.acc_dtype)
        row_acc = mAcc[bb, hh, None].iterator + (t * 16384 + 8192 * (q // 64) + 8 * (q % 64))
        for k in cutlass.range_constexpr(8):
            cc = 2 * k + h
            ptr = cute.make_ptr(self.acc_dtype, (row_acc + (1024 * (cc % 8) + 512 * (cc // 8))).toint(),
                                cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(8)),
                      cute.make_tensor(acc.iterator + 8 * k, cute.make_layout(8)))
        if const_expr(self.zero_after):
            zero = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
            zero.fill(0)
            for k in cutlass.range_constexpr(8):
                cc = 2 * k + h
                ptr = cute.make_ptr(self.acc_dtype, (row_acc + (1024 * (cc % 8) + 512 * (cc // 8))).toint(),
                                    cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(8)))
        x = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        x.store(acc.load().to(Float32))
        unrotate_pair64(x, h, scale, self.had_signs)
        o = cute.make_rmem_tensor(cute.make_layout(64), self.out_dtype)
        o.store(x.load().to(self.out_dtype))
        row_it = mdQ[bb, t * 128 + q, hh, None].iterator
        for k in cutlass.range_constexpr(8):
            optr = cute.make_ptr(self.out_dtype, (row_it + (16 * k + 8 * h)).toint(), cute.AddressSpace.gmem,
                                 assumed_align=16)
            cute.copy(atom_o, cute.make_tensor(o.iterator + 8 * k, cute.make_layout(8)),
                      cute.make_tensor(optr, cute.make_layout(8)))

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None):
        if const_expr(mScale is not None and not self.tile_scale):
            scale = mScale[0]  # a device scale (the data-sized dQ pre-scale's inverse, device_scales._dq_prescale)
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        j = lane % 16  # column chunk: d = 64 (j // 8) + 8 (j % 8) + (0..7)
        n = 8 * self.tiles
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        acc = cute.make_rmem_tensor(cute.make_layout((8, n)), self.acc_dtype)
        for i in cutlass.range_constexpr(n):
            q = 16 * (i % 8) + 2 * warp + lane // 16
            c = 1024 * (q // 64) + 128 * (j % 8) + 64 * (j // 8) + q % 64
            tile_it = mAcc[bb, hh, None].iterator + (t * self.tiles + i // 8) * 16384
            ptr = cute.make_ptr(self.acc_dtype, (tile_it + c * 8).toint(), cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(8)), acc[None, i])
        if const_expr(self.zero_after):
            zero = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
            zero.fill(0)
            for i in cutlass.range_constexpr(n):
                q = 16 * (i % 8) + 2 * warp + lane // 16
                c = 1024 * (q // 64) + 128 * (j % 8) + 64 * (j // 8) + q % 64
                tile_it = mAcc[bb, hh, None].iterator + (t * self.tiles + i // 8) * 16384
                ptr = cute.make_ptr(self.acc_dtype, (tile_it + c * 8).toint(), cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(8)))
        for i in cutlass.range_constexpr(n):
            q = 16 * (i % 8) + 2 * warp + lane // 16
            x = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            x.store(acc[None, i].load().to(Float32))
            if const_expr(mScale is not None and self.tile_scale and not self.row_scale and i % 8 == 0):
                scale = mScale[bb * (mAcc.shape[2] // 16384) + t * self.tiles + i // 8]
            if const_expr(mScale is not None and self.row_scale):
                scale = mScale[bb * (mAcc.shape[2] // 128) + (t * self.tiles + i // 8) * 128 + q]
            for e in cutlass.range_constexpr(0, 8, 2):
                x[e], x[e + 1] = cute.arch.mul_packed_f32x2((x[e], x[e + 1]), (scale, scale))
            o = cute.make_rmem_tensor(cute.make_layout(8), self.out_dtype)
            o.store(x.load().to(self.out_dtype))
            row_it = mdQ[bb, (t * self.tiles + i // 8) * 128 + q, hh, None].iterator + (64 * (j // 8) + 8 * (j % 8))
            optr = cute.make_ptr(self.out_dtype, row_it.toint(), cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom_o, o, cute.make_tensor(optr, cute.make_layout(8)))


_post_cache = {}


def dq_postprocess_2cta8(dq_accum, s_q: int, scale, out_dtype, zero_after: bool = False, had: bool = False,
                         tile_scale: bool = False, row_scale: bool = False):
    """Bwd2Cta8Postprocess on a (b, h, s_q * 128) f16 accumulator -> dq (b, s_q, h, 128). scale: a Python float, or a
    one-element float32 CUDA tensor the kernel reads (no host read). zero_after hands the re-zeroed accumulator back to
    dq_postprocess_cute's pool, exactly as dq_postprocess_2cta_cute does.
    had (qk_hadamard): the accumulator is rotated; the same launch rotates it back (dq carries `_had_unrotated`)."""
    import torch
    from flash_attn.cute.interface import torch2cute_dtype_map
    from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor
    b, h, n = dq_accum.shape
    assert dq_accum.dtype == torch.float16 and dq_accum.is_contiguous() and s_q % 128 == 0 and n == s_q * 128
    dq = torch.empty(b, s_q, h, 128, dtype=out_dtype, device=dq_accum.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dev = isinstance(scale, torch.Tensor)
    if tile_scale:   # dqfix: the per-(b, 128-row tile) postprocess plane
        assert dev and scale.dtype == torch.float32 and scale.numel() == (b * s_q if row_scale else b * (s_q // 128))
        scale_t = scale.reshape(-1)
    else:
        scale_t = scale.reshape(1).float() if dev else None
    key = (out_dtype, zero_after, dev) + (("had",) if had else ()) + (("tile_scale",) if tile_scale else ()) \
        + (("row_scale",) if row_scale else ())
    if key not in _post_cache:
        sg = None
        if had:
            from flash_attn.cute.qk_hadamard import signs_tuple
            sg = signs_tuple(128)
        obj = Bwd2Cta8Postprocess(torch2cute_dtype_map[out_dtype], zero_after, had_signs=sg,
                                  **({"tile_scale": True} if tile_scale else {}),
                                  **({"row_scale": True} if row_scale else {}))
        _post_cache[key] = cute.compile(obj, to_cute_tensor(dq_accum, divisibility=8), to_cute_tensor(dq, divisibility=8),
                                        Float32(1.0 if dev else scale), stream,
                                        to_cute_tensor(scale_t, assumed_align=4, leading_dim=0) if dev else None,
                                        options="--enable-tvm-ffi")
    _post_cache[key](dq_accum, dq, 1.0 if dev else scale, stream, scale_t)
    if zero_after:
        from flash_attn.cute.dq_postprocess_cute import _DQ_ACCUM_POOL
        _DQ_ACCUM_POOL[(b, h, n, dq_accum.dtype, str(dq_accum.device))] = dq_accum
    if had:
        dq._had_unrotated = True
    return dq

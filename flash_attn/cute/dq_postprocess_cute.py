# dQ postprocess for the FP4 backward (CuTe DSL): one kernel in place of the torch chain
# accum.float() * scale -> to(bf16) -> permute -> contiguous (four launches, ~6 us of fixed cost per call).
# accum (b, h, s_rounded, d) f16 / fp32  ->  dq (b, s, h, d) bf16 / fp16, dq = accum * scale.
# One CTA per (batch, head, 128-query tile), 4 threads per query row (a quarter row each; 1 for the Hadamard path).
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
import cuda.bindings.driver as cuda

from flash_attn.cute.interface import torch2cute_dtype_map
from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor
from flash_attn.cute.qk_hadamard_cute import (chunk8_factors, neg_masks16, unrotate_chunk8, unrotate_pair64,
                                              unrotate_row128)
from flash_attn.cute.quant.fp4 import rcp_rn

_cache = {}


@cute.jit
def _advance_seed(mSeedCtr, tid, t, hh, bb):
    """The dk8 step's SR counter (interface_bwd_fp4.ds_sr_counter, int64 (1,)): advanced by one in this launch, after the
    backward kernel read it (stream order), so the next call -- or the next replay of a captured step -- draws a new
    dither. No-op (no code) without a counter."""
    if const_expr(mSeedCtr is not None):
        if tid == 0 and t == 0 and hh == 0 and bb == 0:
            mSeedCtr[0] = mSeedCtr[0] + cutlass.Int64(1)


@cute.jit
def _row_scale(mScale: cute.Tensor, mRowTile: cute.Tensor, q):
    """lp_docmaps packed rows (b = 1): the accumulator's row q is packed row q, whose aligned row mRowTile[q]
    (DocMaps.dst_q) gives its 128-row tile, hence its document's postprocess scale (the dqfix plane, per aligned tile)."""
    return mScale[mRowTile[q] // 128]


@cute.jit
def _tile_scale(mScale: cute.Tensor, mAcc: cute.Tensor, h, t, hh, bb):
    """dqfix: the per-(batch row, 128-row tile) postprocess scale (device_scales.dq_tile_planes), index b tq + t with
    tq the accumulator's 128-row tiles (the scale is its row's: one document or batch row; every head the same)."""
    return mScale[bb * (mAcc.shape[2] // 128) + t]


class DQPostprocess:
    def __init__(self, acc_dtype, out_dtype, head_dim: int, had_signs=None, tile_scale: bool = False):
        self.tile_scale = tile_scale   # dqfix: mScale is the per-tile table (_tile_scale)
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.head_dim = head_dim
        # qk_hadamard: the accumulator holds dQ R (rotated operands); dQ = (acc H / sqrt(128)) * signs, in registers
        self.had_signs = had_signs
        assert had_signs is None or head_dim == 128
        # threads per query row: 4 (a quarter row each) unless the Hadamard back-rotation needs the whole row in one
        # thread: four times the loads / stores in flight per row (the (b, s, h, d) output rows of a warp are h * d
        # apart, so the old one-thread-per-row form was latency-bound): 22.1 -> 10.4 us at D64c, same bytes out
        self.tpr = 1 if had_signs is not None else 4
        self.cols = head_dim // self.tpr
        self.threads = 128 * self.tpr

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream,
                 mScale: cute.Tensor = None, mSeedCtr: cute.Tensor = None, mRows: cute.Tensor = None,
                 mRowTile: cute.Tensor = None):
        grid = [cute.ceil_div(mdQ.shape[1], 128), mdQ.shape[2], mdQ.shape[0]]
        if const_expr(mRows is not None):   # lp_docmaps: rows q of the accumulator go to dQ rows mRows[q]
            grid = [cute.ceil_div(mRows.shape[0], 128), mdQ.shape[2], mdQ.shape[0]]
        self.kernel(mAcc, mdQ, scale, mScale, mSeedCtr, mRows, mRowTile).launch(
            grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None,
               mSeedCtr: cute.Tensor = None, mRows: cute.Tensor = None, mRowTile: cute.Tensor = None):
        if const_expr(mScale is not None and not self.tile_scale):
            scale = mScale[0]  # a device scale (device_scales): the float32 the host would have passed
        tid, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        if const_expr(mScale is not None and self.tile_scale and mRowTile is None):
            scale = _tile_scale(mScale, mAcc, mdQ.shape[2], t, hh, bb)
        _advance_seed(mSeedCtr, tid, t, hh, bb)
        tidx = tid // self.tpr
        c0 = cute.assume((tid % self.tpr) * self.cols, divby=self.cols)  # keeps the row's 16 B vector alignment
        q = t * 128 + tidx
        if const_expr(mRows is not None):
            if q < mRows.shape[0]:
                qo = mRows[q]
                if qo >= 0:
                    self._row(mAcc, mdQ, scale, bb, hh, q, qo, c0)
        elif q < mdQ.shape[1]:
            if const_expr(mRowTile is not None):   # (packed rows: the row's document's scale)
                scale = _row_scale(mScale, mRowTile, q)
            acc = cute.make_rmem_tensor(cute.make_layout(self.cols), self.acc_dtype)
            cute.autovec_copy(cute.make_tensor(mAcc[bb, hh, q, None].iterator + c0, cute.make_layout(self.cols)), acc)
            x = cute.make_rmem_tensor(cute.make_layout(self.cols), Float32)
            x.store(acc.load().to(Float32))
            if const_expr(self.had_signs is not None):
                unrotate_row128(x, scale, self.had_signs)
            else:
                for j in cutlass.range_constexpr(0, self.cols, 2):
                    x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (scale, scale))
            out = cute.make_rmem_tensor(cute.make_layout(self.cols), self.out_dtype)
            out.store(x.load().to(self.out_dtype))
            cute.autovec_copy(out, cute.make_tensor(mdQ[bb, q, hh, None].iterator + c0, cute.make_layout(self.cols)))

    @cute.jit
    def _row(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, bb, hh, q, qo, c0):
        """One accumulator row q to dQ row qo (the lp_docmaps scatter; the same arithmetic as kernel's row)."""
        acc = cute.make_rmem_tensor(cute.make_layout(self.cols), self.acc_dtype)
        cute.autovec_copy(cute.make_tensor(mAcc[bb, hh, q, None].iterator + c0, cute.make_layout(self.cols)), acc)
        x = cute.make_rmem_tensor(cute.make_layout(self.cols), Float32)
        x.store(acc.load().to(Float32))
        if const_expr(self.had_signs is not None):
            unrotate_row128(x, scale, self.had_signs)
        else:
            for j in cutlass.range_constexpr(0, self.cols, 2):
                x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (scale, scale))
        out = cute.make_rmem_tensor(cute.make_layout(self.cols), self.out_dtype)
        out.store(x.load().to(self.out_dtype))
        cute.autovec_copy(out, cute.make_tensor(mdQ[bb, qo, hh, None].iterator + c0, cute.make_layout(self.cols)))


class DQPostprocessRows:
    """DQPostprocess for d = 128 without the Hadamard back-rotation, with a coalesced thread mapping: the same
    arithmetic per element (acc -> f32 exact, one FMUL2 by `scale`, one cvt.rn to the output type), so the same
    accumulator gives the same bits. DQPostprocess gives each thread one query row, so a warp's 16-byte access touches
    32 rows 256 bytes apart and a thread holds 128 values; it took 42 us for b4 s4096 h16 (~3.2 TB/s). Here a warp
    covers two rows x 16 column chunks of 8 elements per step: its loads are 512 contiguous accumulator bytes (f16;
    two 16-byte halves per chunk for f32), its stores two contiguous 256-byte dQ row segments. Grid
    (ceil(s_q / 128), heads, batch), 256 threads, 8 steps of 16 rows; all loads are issued before any store.

    had_signs (qk_hadamard): the accumulator holds dQ R and the SAME launch rotates it back, bitwise DQPostprocess(
    had_signs), which it replaces: f16 accumulators keep this mapping (a half warp per row, levels 8..64 as shuffles:
    qk_hadamard_cute.unrotate_chunk8); f32 ones take a lane pair per row with interleaved 8-column chunks (one shuffle
    level: unrotate_pair64, kernel_pair). A1: 25.9 (f16) / 36.8 (f32) us vs 44.4 / 54.9 thread-per-row, 19.0 / 31.3
    unrotated."""

    def __init__(self, acc_dtype, out_dtype, had_signs=None, tile_scale: bool = False, row_scale: bool = False):
        self.tile_scale = tile_scale   # dqfix: mScale is the per-tile table (_tile_scale)
        # dq-causal: mScale is the per-(b, row) plane (b * s_rounded floats); the kernel reads each row's own scale
        self.row_scale = row_scale
        assert not row_scale or (tile_scale and (had_signs is None or acc_dtype.width == 16))
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.threads = 256
        self.had_signs = had_signs
        self.negm = None if had_signs is None else neg_masks16(had_signs, lambda j, e: 8 * j + e)

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream,
                 mScale: cute.Tensor = None, mSeedCtr: cute.Tensor = None, mRows: cute.Tensor = None,
                 mRowTile: cute.Tensor = None):
        grid = [cute.ceil_div(mdQ.shape[1], 128), mdQ.shape[2], mdQ.shape[0]]
        if const_expr(mRows is not None):   # lp_docmaps: rows q of the accumulator go to dQ rows mRows[q]
            grid = [cute.ceil_div(mRows.shape[0], 128), mdQ.shape[2], mdQ.shape[0]]
        if const_expr(self.had_signs is not None and self.acc_dtype.width == 32):
            self.kernel_pair(mAcc, mdQ, scale, mScale, mSeedCtr, mRows, mRowTile).launch(
                grid=grid, block=[256, 1, 1], stream=stream)
        else:
            self.kernel(mAcc, mdQ, scale, mScale, mSeedCtr, mRows, mRowTile).launch(
                grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel_pair(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None,
                    mSeedCtr: cute.Tensor = None, mRows: cute.Tensor = None, mRowTile: cute.Tensor = None):
        if const_expr(mScale is not None and not self.tile_scale):
            scale = mScale[0]
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        if const_expr(mScale is not None and self.tile_scale and mRowTile is None):
            scale = _tile_scale(mScale, mAcc, mdQ.shape[2], t, hh, bb)
        _advance_seed(mSeedCtr, tidx, t, hh, bb)
        h = tidx % 2
        r = tidx // 2  # the row within the tile; chunk cc = 2 k + h holds columns 8 cc .. 8 cc + 7
        if const_expr(mRowTile is not None):   # (packed rows: the row's document's scale)
            scale = _row_scale(mScale, mRowTile, t * 128 + r)
        per16 = 128 // self.acc_dtype.width
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        acc = cute.make_rmem_tensor(cute.make_layout(64), self.acc_dtype)
        row_acc = mAcc[bb, hh, t * 128, None].iterator + r * 128
        for k in cutlass.range_constexpr(8):
            for hv in cutlass.range_constexpr(8 // per16):
                ptr = cute.make_ptr(self.acc_dtype, (row_acc + (8 * (2 * k + h) + hv * per16)).toint(),
                                    cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(per16)),
                          cute.make_tensor(acc.iterator + 8 * k + hv * per16, cute.make_layout(per16)))
        x = cute.make_rmem_tensor(cute.make_layout(64), Float32)
        x.store(acc.load().to(Float32))
        unrotate_pair64(x, h, scale, self.had_signs)
        o = cute.make_rmem_tensor(cute.make_layout(64), self.out_dtype)
        o.store(x.load().to(self.out_dtype))
        if const_expr(mRows is not None):
            qo = mRows[t * 128 + r]
            if qo >= 0:
                row_it = mdQ[bb, qo, hh, None].iterator
                for k in cutlass.range_constexpr(8):
                    optr = cute.make_ptr(self.out_dtype, (row_it + (16 * k + 8 * h)).toint(), cute.AddressSpace.gmem,
                                         assumed_align=16)
                    cute.copy(atom_o, cute.make_tensor(o.iterator + 8 * k, cute.make_layout(8)),
                              cute.make_tensor(optr, cute.make_layout(8)))
        else:
            row_it = mdQ[bb, t * 128 + r, hh, None].iterator
            for k in cutlass.range_constexpr(8):
                optr = cute.make_ptr(self.out_dtype, (row_it + (16 * k + 8 * h)).toint(), cute.AddressSpace.gmem,
                                     assumed_align=16)
                cute.copy(atom_o, cute.make_tensor(o.iterator + 8 * k, cute.make_layout(8)),
                          cute.make_tensor(optr, cute.make_layout(8)))

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None,
               mSeedCtr: cute.Tensor = None, mRows: cute.Tensor = None, mRowTile: cute.Tensor = None):
        if const_expr(mScale is not None and not self.tile_scale):
            scale = mScale[0]
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        if const_expr(mScale is not None and self.tile_scale and mRowTile is None and not self.row_scale):
            scale = _tile_scale(mScale, mAcc, mdQ.shape[2], t, hh, bb)
        _advance_seed(mSeedCtr, tidx, t, hh, bb)
        warp = tidx // 32
        lane = tidx % 32
        j = lane % 16  # column chunk: d = 8 j + (0..7)
        n = 8
        per16 = 128 // self.acc_dtype.width  # accumulator elements per 16-byte copy (8 f16 / 4 f32)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        acc = cute.make_rmem_tensor(cute.make_layout((8, n)), self.acc_dtype)
        # s_q % 128 == 0 (the caller's condition): no row predicate, so the 8 (16 for f32) loads issue back to back
        tile_it = mAcc[bb, hh, t * 128, None].iterator  # the tile's 128 rows are contiguous (row-major, d = 128)
        for i in cutlass.range_constexpr(n):
            r = 16 * i + 2 * warp + lane // 16
            for hv in cutlass.range_constexpr(8 // per16):
                ptr = cute.make_ptr(self.acc_dtype, (tile_it + (r * 128 + 8 * j + hv * per16)).toint(),
                                    cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(per16)),
                          cute.make_tensor(acc[None, i].iterator + hv * per16, cute.make_layout(per16)))
        if const_expr(self.had_signs is not None):
            mult = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            pm = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            if const_expr(mRowTile is None and not self.row_scale):
                chunk8_factors(j, scale, self.negm, mult, pm)
        rsc = cute.make_rmem_tensor(cute.make_layout(n), Float32)
        if const_expr(mRowTile is not None):   # (packed rows: each row's document's scale, loads issued up front)
            for i in cutlass.range_constexpr(n):
                rsc[i] = _row_scale(mScale, mRowTile, t * 128 + 16 * i + 2 * warp + lane // 16)
        if const_expr(self.row_scale):   # dq-causal: each row's own scale (the plane (b, s_rounded))
            for i in cutlass.range_constexpr(n):
                rsc[i] = mScale[bb * mAcc.shape[2] + t * 128 + 16 * i + 2 * warp + lane // 16]
        for i in cutlass.range_constexpr(n):
            r = 16 * i + 2 * warp + lane // 16
            if const_expr(mRowTile is not None or self.row_scale):
                scale = rsc[i]
                if const_expr(self.had_signs is not None):
                    chunk8_factors(j, scale, self.negm, mult, pm)
            x = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            x.store(acc[None, i].load().to(Float32))
            if const_expr(self.had_signs is not None):
                unrotate_chunk8(x, mult, pm)
            else:
                for e in cutlass.range_constexpr(0, 8, 2):
                    x[e], x[e + 1] = cute.arch.mul_packed_f32x2((x[e], x[e + 1]), (scale, scale))
            o = cute.make_rmem_tensor(cute.make_layout(8), self.out_dtype)
            o.store(x.load().to(self.out_dtype))
            if const_expr(mRows is not None):
                qo = mRows[t * 128 + r]
                if qo >= 0:
                    optr = cute.make_ptr(self.out_dtype, (mdQ[bb, qo, hh, None].iterator + 8 * j).toint(),
                                         cute.AddressSpace.gmem, assumed_align=16)
                    cute.copy(atom_o, o, cute.make_tensor(optr, cute.make_layout(8)))
            else:
                optr = cute.make_ptr(self.out_dtype, (mdQ[bb, t * 128 + r, hh, None].iterator + 8 * j).toint(),
                                     cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom_o, o, cute.make_tensor(optr, cute.make_layout(8)))


def dev_ok(x) -> bool:
    return isinstance(x, torch.Tensor) and x.dtype == torch.float32


def dq_postprocess_cute(dq_accum: torch.Tensor, s_q: int, scale, out_dtype=torch.bfloat16,
                        had: bool = False, seed_counter: torch.Tensor = None, tile_scale: bool = False,
                        row_tile: Optional[torch.Tensor] = None, row_scale: bool = False) -> torch.Tensor:
    """dq_accum (b, h, s_rounded, d) contiguous f16 / fp32 -> dq (b, s_q, h, d) out_dtype, scaled by `scale`: a Python float,
    or a one-element CUDA tensor (float32; a float64 one is converted to float32 by torch) that the
    kernel reads from device memory (no host read).
    had (qk_hadamard): the accumulator is rotated; the rotation is undone in registers (dq carries `_had_unrotated`).
    row_tile (lp_docmaps packed rows, b = 1, with tile_scale): (s_q,) int32, accumulator row q's aligned row
    (DocMaps.dst_q); the scale plane is per aligned 128-row tile, read per row."""
    b, h, s_r, d = dq_accum.shape
    assert dq_accum.is_contiguous() and s_r >= s_q and d % 8 == 0
    from flash_attn.cute import lp_docmaps   # (a doc_maps backward: dQ rows scattered to the caller's packed rows)
    rows = lp_docmaps.dq_scatter_rows(b, s_q) if lp_docmaps._DQ_SCATTER else None
    dq = torch.empty(b, s_q if rows is None else rows[1], h, d, dtype=out_dtype, device=dq_accum.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dev = isinstance(scale, torch.Tensor)
    ctr = seed_counter is not None   # seed_counter: advanced by one in this launch (_advance_seed)
    key = (dq_accum.dtype, out_dtype, d, dev, bool(had), s_q % 128 == 0) + (("seed_counter",) if ctr else ()) \
        + (("rows",) if rows is not None else ()) + (("tile_scale",) if tile_scale else ()) \
        + (("row_tile",) if row_tile is not None else ()) + (("row_scale",) if row_scale else ())
    rw = () if rows is None else (rows[0],)
    tsk = {"tile_scale": True} if tile_scale else {}
    if row_scale:   # dq-causal: scale is the per-(b, row) plane; the coalesced d = 128 kernel only
        assert tile_scale and dev_ok(scale) and d == 128 and s_q % 128 == 0 and (not had or dq_accum.dtype == torch.float16) \
            and rows is None \
            and row_tile is None and scale.numel() == b * s_r
        tsk["row_scale"] = True
    assert row_tile is None or (tile_scale and b == 1 and rows is None and row_tile.dtype == torch.int32
                                and row_tile.shape == (s_q,))
    if tile_scale and row_tile is None and not row_scale:   # dqfix: scale is the per-(b, 128-row tile) float32 plane
        assert dev and scale.dtype == torch.float32 and scale.numel() == b * (s_r // 128) and s_r % 128 == 0
    rw_t = () if row_tile is None else (row_tile,)
    if dev:
        scale_t = scale.reshape(-1) if tile_scale else scale.reshape(1).float()
    if key not in _cache:
        from flash_attn.cute.qk_hadamard import signs_tuple
        if d == 128 and s_q % 128 == 0:  # coalesced (same bits as DQPostprocess, with or without the back-rotation)
            obj = DQPostprocessRows(torch2cute_dtype_map[dq_accum.dtype], torch2cute_dtype_map[out_dtype],
                                    had_signs=signs_tuple(d) if had else None, **tsk)
        else:
            obj = DQPostprocess(torch2cute_dtype_map[dq_accum.dtype], torch2cute_dtype_map[out_dtype], d,
                                had_signs=signs_tuple(d) if had else None, **tsk)
        _cache[key] = cute.compile(
            obj, to_cute_tensor(dq_accum, divisibility=8), to_cute_tensor(dq, divisibility=8), Float32(1.0 if dev else scale),
            stream, to_cute_tensor(scale_t, assumed_align=4, leading_dim=0) if dev else None,
            *((to_cute_tensor(seed_counter, assumed_align=8, leading_dim=0),) if ctr else
              ((None,) if (rw or rw_t) else ())),
            *(to_cute_tensor(t_, assumed_align=4) for t_ in rw), *((None,) if (rw_t and not rw) else ()),
            *(to_cute_tensor(t_, assumed_align=4, leading_dim=0) for t_ in rw_t),
            options="--enable-tvm-ffi",
        )
    sc = (seed_counter,) if ctr else ((None,) if (rw or rw_t) else ())
    rr = rw + ((None,) if (rw_t and not rw) else ()) + rw_t
    if dev:
        _cache[key](dq_accum, dq, 1.0, stream, scale_t, *sc, *rr)
    else:
        _cache[key](dq_accum, dq, scale, stream, None, *sc, *rr)
    if had:
        dq._had_unrotated = True
    return dq


class FP4QPostprocess:
    """The all-FP4 backward's one post launch: dQ = bf16(accum * scale / F_q) (the causal dS scale's per-row power of two,
    formerly a torch copy + div + copy after the 1-CTA postprocess) and, under the K fold, dK / dV rows times their
    per-key powers of two in place (formerly the Triton _rows_scale2 launch). Each power-of-two factor commutes with the
    bf16 rounding outside the subnormal range, so the outputs are bitwise the former two-step results.
    Grid: [dQ tiles (s_q / 128 x h x b)] + [dK tiles, dV tiles (s_k / 128 x h_k x b each)]; one thread per row."""

    def __init__(self, acc_dtype, out_dtype, row_divisor: bool, fold_rows: bool, device_scale: bool = False,
                 seed_counter: bool = False, doc_scale: bool = False):
        self.device_scale = device_scale
        # aligned varlen documents (b = 1): the device scale is a per-document table read by the row tile's document
        self.doc_scale = bool(doc_scale)
        # the all-4-bit step's SR counter (int64 (1,)): advanced by one here, after the dO quantiser and the backward
        # kernel read it (stream order), so the next step -- or the next replay of a captured one -- gets a new seed
        self.seed_counter = seed_counter
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.row_divisor = row_divisor
        self.fold_rows = fold_rows
        self.threads = 128

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mRowF: cute.Tensor, mdK: cute.Tensor,
                 mdV: cute.Tensor, mFdK: cute.Tensor, mFdV: cute.Tensor, mScale: cute.Tensor, n_q_ctas: cutlass.Int32,
                 n_kv_ctas: cutlass.Int32, mSeedCtr: Optional[cute.Tensor], mTileDoc: Optional[cute.Tensor],
                 stream: cuda.CUstream):
        self.kernel(mAcc, mdQ, scale, mRowF, mdK, mdV, mFdK, mFdV, mScale, n_q_ctas, mSeedCtr, mTileDoc).launch(
            grid=[n_q_ctas + n_kv_ctas, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mRowF: cute.Tensor, mdK: cute.Tensor,
               mdV: cute.Tensor, mFdK: cute.Tensor, mFdV: cute.Tensor, mScale: cute.Tensor, n_q_ctas: cutlass.Int32,
               mSeedCtr: Optional[cute.Tensor] = None, mTileDoc: Optional[cute.Tensor] = None):
        if const_expr(self.device_scale and not self.doc_scale):
            scale = mScale[0]   # (1,) float32 on the device (the dense route's dq_scale): no host read
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if const_expr(self.seed_counter):
            if bid == 0 and tidx == 0:
                mSeedCtr[0] = mSeedCtr[0] + cutlass.Int64(1)
        if bid < n_q_ctas:
            s_q = mdQ.shape[1]
            h = mdQ.shape[2]
            tiles = cute.ceil_div(s_q, 128)
            t = bid % tiles
            hh = (bid // tiles) % h
            bb = bid // (tiles * h)
            q = t * 128 + tidx
            if const_expr(self.doc_scale):
                scale = mScale[mTileDoc[t]]   # the row tile's document's dq_scale
            if q < s_q:
                acc = cute.make_rmem_tensor(cute.make_layout(128), self.acc_dtype)
                cute.autovec_copy(cute.make_tensor(mAcc[bb, hh, q, None].iterator, cute.make_layout(128)), acc)
                x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
                x.store(acc.load().to(Float32))
                for j in cutlass.range_constexpr(0, 128, 2):
                    x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (scale, scale))
                if const_expr(self.row_divisor):
                    rinv = rcp_rn(mRowF[(bb * h + hh) * s_q + q])   # exact: F_q is a power of two
                    for j in cutlass.range_constexpr(0, 128, 2):
                        x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (rinv, rinv))
                out = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
                out.store(x.load().to(self.out_dtype))
                cute.autovec_copy(out, cute.make_tensor(mdQ[bb, q, hh, None].iterator, cute.make_layout(128)))
        else:
            if const_expr(self.fold_rows):
                k_bid = bid - n_q_ctas
                s_k = mdK.shape[1]
                hk = mdK.shape[2]
                ktiles = s_k // 128
                which = k_bid % 2
                p = k_bid // 2
                t = p % ktiles
                hh = (p // ktiles) % hk
                bb = p // (ktiles * hk)
                key = t * 128 + tidx
                fi = (bb * hk + hh) * s_k + key
                if which == 0:
                    self.scale_row(mdK, mFdK[fi], bb, key, hh)
                else:
                    self.scale_row(mdV, mFdV[fi], bb, key, hh)

    @cute.jit
    def scale_row(self, mX: cute.Tensor, f: Float32, bb, key, hh):
        v = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
        cute.autovec_copy(cute.make_tensor(mX[bb, key, hh, None].iterator, cute.make_layout(128)), v)
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        x.store(v.load().to(Float32))
        for j in cutlass.range_constexpr(0, 128, 2):
            x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (f, f))
        v.store(x.load().to(self.out_dtype))
        cute.autovec_copy(v, cute.make_tensor(mX[bb, key, hh, None].iterator, cute.make_layout(128)))


def fp4q_postprocess(dq_accum: torch.Tensor, s_q: int, scale: float, out_dtype=torch.bfloat16,
                     row_factor: torch.Tensor = None, dk: torch.Tensor = None, dv: torch.Tensor = None,
                     dk_row_factor: torch.Tensor = None, dv_row_factor: torch.Tensor = None,
                     seed_counter: torch.Tensor = None, rows: bool = False, tile_doc: torch.Tensor = None) -> torch.Tensor:
    """ONE launch: dq (b, s_q, h, 128) = accum (b, h, s, 128) * scale / row_factor (b, h, s_q) (None: no divisor), and, when
    given, dk / dv (b, s_k, h_k, 128) rows times dk_row_factor / dv_row_factor (b, h_k, s_k) in place. seed_counter: the
    step's SR counter (int64 (1,)), advanced by one in the same launch."""
    b, h, s_r, d = dq_accum.shape
    assert dq_accum.is_contiguous() and s_r >= s_q and d == 128
    dq = torch.empty(b, s_q, h, d, dtype=out_dtype, device=dq_accum.device)
    fold = dk is not None
    if fold:
        assert dv is not None and dk_row_factor is not None and dv_row_factor is not None and dk.shape == dv.shape
        assert dk.shape[1] % 128 == 0 and dk.stride() == dv.stride() and dk.stride(3) == 1 and dk.dtype == out_dtype
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy = _pp_dummies(dq_accum.device)
    rf = row_factor.reshape(-1) if row_factor is not None else dummy[0]
    dk_, dv_ = (dk, dv) if fold else (dummy[1], dummy[1])
    fk, fv = (dk_row_factor.reshape(-1), dv_row_factor.reshape(-1)) if fold else (dummy[0], dummy[0])
    dev_scale = isinstance(scale, torch.Tensor)
    docs = tile_doc is not None
    if dev_scale:
        assert scale.dtype == torch.float32 and (scale.numel() == 1 or docs)
    assert not docs or (dev_scale and b == 1), "tile_doc: a per-document device scale table, b = 1"
    scale_t = scale.reshape(-1) if dev_scale else dummy[0]
    scale_f = 1.0 if dev_scale else float(scale)
    ctr = seed_counter is not None
    # rows (KernelTuning.bwd_fp4q_post = 'rows'): fp4q_post_rows.FP4QPostprocessRows, the same bits, coalesced
    rows = bool(rows) and s_q % 128 == 0 and s_r == s_q and not docs
    key = ("fp4q_post_rows" if rows else "fp4q_post", dq_accum.dtype, out_dtype, row_factor is not None, fold, dev_scale) + (("seed_counter",) if ctr
                                                                                              else ()) \
        + (("docs",) if docs else ())
    n_q = ((s_q + 127) // 128) * h * b
    n_kv = 2 * (dk.shape[1] // 128) * dk.shape[2] * dk.shape[0] if fold else 0
    if key not in _cache:
        cls = FP4QPostprocess
        if rows:
            from flash_attn.cute.fp4q_post_rows import FP4QPostprocessRows as cls
        obj = cls(torch2cute_dtype_map[dq_accum.dtype], torch2cute_dtype_map[out_dtype],
                  row_factor is not None, fold, dev_scale, seed_counter=ctr, **({"doc_scale": True} if docs else {}))
        _cache[key] = cute.compile(
            obj, to_cute_tensor(dq_accum, divisibility=8), to_cute_tensor(dq, divisibility=8), Float32(scale_f),
            to_cute_tensor(rf, assumed_align=4, leading_dim=0), to_cute_tensor(dk_, divisibility=8),
            to_cute_tensor(dv_, divisibility=8), to_cute_tensor(fk, assumed_align=4, leading_dim=0),
            to_cute_tensor(fv, assumed_align=4, leading_dim=0), to_cute_tensor(scale_t, assumed_align=4, leading_dim=0),
            cutlass.Int32(1), cutlass.Int32(0),
            to_cute_tensor(seed_counter, assumed_align=8, leading_dim=0) if ctr else None,
            to_cute_tensor(tile_doc, assumed_align=4, leading_dim=0) if docs else None, stream,
            options="--enable-tvm-ffi",
        )
    _cache[key](dq_accum, dq, scale_f, rf, dk_, dv_, fk, fv, scale_t, n_q, n_kv, seed_counter if ctr else None,
                tile_doc if docs else None, stream)
    return dq


_PP_DUMMIES = {}


def _pp_dummies(dev):
    k = str(dev)
    if k not in _PP_DUMMIES:
        _PP_DUMMIES[k] = (torch.ones(1, dtype=torch.float32, device=dev),
                          torch.empty(1, 128, 1, 128, dtype=torch.bfloat16, device=dev))
    return _PP_DUMMIES[k]


class DQPostprocess2CTA:
    """2-CTA backward accumulator (flash_bwd_sm100_2cta8, flat (b, h, s * 128)). The reduce warps write per 128-query tile, per
    CTA half (queries 64 h'), per 8-column stage sigma, thread t = 64 (d // 64) + q % 64 the values v = d % 8 of its rows. f32 (two
    4-element copies per thread): 8192 (q // 64) + 1024 sigma + 512 (v // 4) + 4 t + v % 4; f16 (one 8-element copy): 8192 (q // 64)
    + 1024 sigma + 8 t + v, sigma = (d % 64) // 8. One thread per query row; dq = accum * scale."""

    def __init__(self, acc_dtype, out_dtype, zero_after: bool = False, had_signs=None):
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.zero_after = zero_after  # write zeros back over what was read: the accumulator is clean for the next call
        self.threads = 128
        self.had_signs = had_signs  # qk_hadamard: undo the rotation in registers (see DQPostprocess)

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream,
                 mScale: cute.Tensor = None):
        grid = [cute.ceil_div(mdQ.shape[1], 128), mdQ.shape[2], mdQ.shape[0]]
        self.kernel(mAcc, mdQ, scale, mScale).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mScale: cute.Tensor = None):
        if const_expr(mScale is not None):
            scale = mScale[0]  # a device scale (the data-sized dQ pre-scale's inverse, device_scales._dq_prescale)
        tidx, _, _ = cute.arch.thread_idx()
        t, hh, bb = cute.arch.block_idx()
        q = t * 128 + tidx
        if q < mdQ.shape[1]:
            x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
            # Every per-thread chunk is 16 B and 16 B-aligned (the offsets are multiples of 8 f16 / 4 f32 on a 16 B-aligned
            # base), which the DSL cannot prove through the dynamic offset arithmetic: the chunk pointers are rebuilt
            # with assumed_align=16 (utils.domain_offset_aligned) so that the copies are 128-bit. All 16 loads are
            # issued before any store: a store between them would make the compiler order the later loads after it
            # (possible alias) and serialise the row's loads.
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
            if const_expr(self.acc_dtype.width == 16):
                row_it = mAcc[bb, hh, None].iterator + (t * 16384 + 8192 * (tidx // 64) + 8 * (tidx % 64))
                for hgrp in cutlass.range_constexpr(2):
                    for sg in cutlass.range_constexpr(8):
                        acc = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
                        ptr = cute.make_ptr(self.acc_dtype, (row_it + (1024 * sg + 512 * hgrp)).toint(),
                                            cute.AddressSpace.gmem, assumed_align=16)
                        cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(8)), acc)
                        for j in cutlass.range_constexpr(8):
                            x[64 * hgrp + 8 * sg + j] = acc[j].to(Float32)
                if const_expr(self.zero_after):
                    zero = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
                    zero.fill(0)
                    for hgrp in cutlass.range_constexpr(2):
                        for sg in cutlass.range_constexpr(8):
                            ptr = cute.make_ptr(self.acc_dtype, (row_it + (1024 * sg + 512 * hgrp)).toint(),
                                                cute.AddressSpace.gmem, assumed_align=16)
                            cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(8)))
            else:
                row_it = mAcc[bb, hh, None].iterator + (t * 16384 + 8192 * (tidx // 64) + 4 * (tidx % 64))
                for hgrp in cutlass.range_constexpr(2):
                    for g in cutlass.range_constexpr(16):
                        acc = cute.make_rmem_tensor(cute.make_layout(4), self.acc_dtype)
                        ptr = cute.make_ptr(self.acc_dtype, (row_it + (512 * g + 256 * hgrp)).toint(),
                                            cute.AddressSpace.gmem, assumed_align=16)
                        cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(4)), acc)
                        for j in cutlass.range_constexpr(4):
                            x[64 * hgrp + 4 * g + j] = acc[j].to(Float32)
                if const_expr(self.zero_after):
                    zero = cute.make_rmem_tensor(cute.make_layout(4), self.acc_dtype)
                    zero.fill(0)
                    for hgrp in cutlass.range_constexpr(2):
                        for g in cutlass.range_constexpr(16):
                            ptr = cute.make_ptr(self.acc_dtype, (row_it + (512 * g + 256 * hgrp)).toint(),
                                                cute.AddressSpace.gmem, assumed_align=16)
                            cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(4)))
            if const_expr(self.had_signs is not None):
                unrotate_row128(x, scale, self.had_signs)
            else:
                for j in cutlass.range_constexpr(0, 128, 2):
                    x[j], x[j + 1] = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (scale, scale))
            out = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
            out.store(x.load().to(self.out_dtype))
            cute.autovec_copy(out, cute.make_tensor(mdQ[bb, q, hh, None].iterator, cute.make_layout(128)))


# ---- the 2-CTA accumulator's zeros. HostSpec.dq_zero (KernelTuning.two_cta8_dq_zero) selects where the f16 accumulator the
# cluster kernel reduce-adds into gets zeroed (b4 s4096 h16, 67 MB: torch fill 0.019 ms + a launch):
#   "fused" (default) the caller zeroes it inside its dO-quantiser launch (fp4_bwd_autograd, the Triton tile's linear
#           store pattern: +0.007 ms on that kernel, no launch) and passes it as fp8_scales["dq_accum"]. Callers that
#           do not pass one get the fill.
#   "pool"  the previous call's postprocess wrote zeros back over what it read (+0.038 ms on the postprocess: its
#           strided 16 B-chunk stores at 4 warps / CTA are slower than the fill, so this does not pay; kept as the
#           stateful alternative), and the buffer is kept here per shape, so the next backward of the same shape starts
#           on a clean accumulator with no fill launch. One buffer per (b, h, s, dtype, device), taken out of the pool
#           for the duration of a call and returned after its postprocess; a call that keeps the accumulator
#           (return_accum) or fails before the postprocess never returns it and the next call allocates a fresh one.
#           Correct for sequential use on one stream (the usual training loop, eager or graph-captured); two streams
#           driving the same shape concurrently must not use it.
#   "fill"  torch.zeros in the interface (the original).
DQ_ZERO_DEFAULT = "fused"
_DQ_ACCUM_POOL = {}


def dq_zero_mode(host) -> str:
    """HostSpec.dq_zero (KernelTuning.two_cta8_dq_zero). host: the call's lowprec.specs.HostSpec (from its plan)."""
    return host.dq_zero


def take_zeroed_dq_accum_2cta(b: int, h: int, n: int, dtype, device) -> torch.Tensor:
    """A zero (b, h, n) accumulator: the pooled one for this shape if a postprocess has re-zeroed it, else fresh."""
    t = _DQ_ACCUM_POOL.pop((b, h, n, dtype, str(device)), None)
    if t is None:
        t = torch.zeros(b, h, n, dtype=dtype, device=device)
    return t


def reset_dq_accum_pool():
    """Drop the pooled accumulators (between unrelated workloads, or to free the memory)."""
    _DQ_ACCUM_POOL.clear()


def dq_postprocess_2cta_cute(dq_accum: torch.Tensor, s_q: int, scale, out_dtype=torch.bfloat16,
                             zero_after: bool = False, had: bool = False) -> torch.Tensor:
    """dq_accum (b, h, s_rounded * 128) f16 / fp32 in the 2-CTA accumulator order -> dq (b, s_q, h, 128) out_dtype.
    scale: a Python float, or a one-element float32 CUDA tensor the kernel reads (no host read).
    zero_after: the kernel also writes zeros over the accumulator (every element when n == s_q * 128) and, in that case,
    hands it to the pool for the next take_zeroed_dq_accum_2cta of the same shape."""
    b, h, n = dq_accum.shape
    assert dq_accum.is_contiguous() and n % (128 * 128) == 0 and n // 128 >= s_q
    dq = torch.empty(b, s_q, h, 128, dtype=out_dtype, device=dq_accum.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dev = isinstance(scale, torch.Tensor)
    scale_t = scale.reshape(1).float() if dev else None
    key = ("2cta", dq_accum.dtype, out_dtype, zero_after, bool(had), dev)
    if key not in _cache:
        from flash_attn.cute.qk_hadamard import signs_tuple
        obj = DQPostprocess2CTA(torch2cute_dtype_map[dq_accum.dtype], torch2cute_dtype_map[out_dtype], zero_after,
                                had_signs=signs_tuple(128) if had else None)
        _cache[key] = cute.compile(
            obj, to_cute_tensor(dq_accum, divisibility=4), to_cute_tensor(dq, divisibility=8),
            Float32(1.0 if dev else scale), stream,
            to_cute_tensor(scale_t, assumed_align=4, leading_dim=0) if dev else None, options="--enable-tvm-ffi",
        )
    _cache[key](dq_accum, dq, 1.0 if dev else scale, stream, scale_t)
    if zero_after and n == s_q * 128:
        _DQ_ACCUM_POOL[(b, h, n, dq_accum.dtype, str(dq_accum.device))] = dq_accum
    if had:
        dq._had_unrotated = True
    return dq

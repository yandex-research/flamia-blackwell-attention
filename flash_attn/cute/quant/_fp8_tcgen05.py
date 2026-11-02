# quantize_fp8_blocks' token-block kernel with hadamard_rotation_method="tcgen05" (opt-in, never a
# default): kernel_tc's outputs and arithmetic, the rotation on tcgen05 UMMA (quant._tcgen05: NOT bitwise the
# tensor_core method -- the 128-term sum is the tensor core's single accumulation).
#
# Persistent, one CTA per SM (grid = min(#SM, tiles)), 320 threads:
#   warp 8   TMA producer: 128 x 128 16-bit tiles of x into a STAGES-deep ring (SW128 K-major, the UMMA A image);
#   warp 9   MMA issuer + TMEM owner (256 columns: two 128-column fp32 accumulators);
#   warps 0-7 two epilogue warpgroups taking alternate tiles; thread = token row = TMEM lane, warp = 32-token block (the
#            block maxima are warp shuffles). B = diag(signs) H_128 is built once per CTA (quant._tcgen05.build_b_image).
# Options (all compile-time, kernel_tc's set): block_columns 32 / 64 / 128, E4M3 / E5M2, the exponent table, the
# token-block factors, output_amax, flush_subnormal_inputs, RoPE ((64, s) / (s, 64) tables: the epilogue warpgroup
# rewrites its stage with the RoPE'd 16-bit row before the MMA reads it), the NVFP4 copy (token_block / raw / per_tensor
# scaling, dp_operand / qk_operand encoding, nearest / stochastic), hadamard_rotation_scope "nvfp4_copy" (the E4M3 side of
# the raw rows read back from the TMA'd stage, the NVFP4 copy of the rotated rows). Codes / NVFP4 codes leave through
# swizzled SMEM tiles and one TMA tensor store each per tile; the SF tables, exponent table, factors and amax are
# kernel_tc's words / bytes at kernel_tc's indices.
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cuda.bindings.driver as cuda

from flash_attn.cute.quant import _tcgen05 as T5
from flash_attn.cute.quant._rotations import INV_SQRT_128, rope_rotate_row
from flash_attn.cute.quant._fp8_nvfp4 import nvfp4_group16

EPI_WARPS = 8
TMA_WARP = 8
MMA_WARP = 9
NTHREADS = 320
FLT_MIN = 1.1754943508222875e-38


class TokenBlockTcgen05Kernel:
    def __init__(self, in_dtype, fmt, fmax, cols, exponent_table, factors, flush, signs, rope, rope_transposed,
                 nvfp4, nvfp4_scaling, nvfp4_stochastic, nvfp4_encoding, stat_amax, nvfp4_only, stages=3):
        self.dt = in_dtype
        self.half = "f16" if in_dtype is cutlass.Float16 else "bf16"
        self.fmt = fmt
        self.fmax = fmax
        self.cols = cols
        self.exponent_table = exponent_table
        self.factors = factors
        self.flush = flush
        self.signs = signs
        self.rope = rope
        self.rope_t = rope_transposed
        self.nvfp4 = nvfp4
        self.nvfp4_scaling = nvfp4_scaling
        self.nvfp4_stochastic = nvfp4_stochastic
        self.nvfp4_encoding = nvfp4_encoding
        self.stat_amax = stat_amax
        self.nvfp4_only = nvfp4_only
        self.stages = stages
        self.sign_words = T5.b_sign_words(signs)
        self.smem_bytes = 32768 * (stages + 1) + 65536 + (32768 if nvfp4 else 0) + 2048
        # cov3 (lp_docmaps compact maps): x / the codes / the NVFP4 codes are the caller's PACKED (1, T, h, d) rows; the
        # tiles are the maps' aligned 128-row tiles (tile t = packed rows mVt[0, t] + r, r < mVt[1, t] & 0xFFFF: the TMA
        # box starts at the packed start, rows past the document are zeroed after the rotation -- every row is rotated
        # on its own, so the real rows' values are the aligned layout's), persistent over the device tile count mNT;
        # codes / NVFP4 codes stored per real row from the registers (no TMA store: the next document's rows), no
        # per-tile table but the amax; the dense per-row exponent bytes (mEvR) and NVFP4 SF words (mSF4R)
        self.compact = False

    # ------------------------------------------------------------------------------------------------------------- host
    @cute.jit
    def __call__(self, mX, mQb, mX4b, mSFTok, mSFDim, mExp, mFac, mSF4, mNvS, mCos, mSin, mStats, seed: Int64,
                 grid_x: Int32, stream: cuda.CUstream, mVt=None, mNT=None, mEvR=None, mSF4R=None):
        tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
            self.dt, self.dt, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K, Float32,
            tcgen05.CtaGroup.ONE, (128, 128))
        sA_l = sm100_utils_basic.make_smem_layout_a(tiled_mma, (128, 128, 128), self.dt, self.stages)
        sB_l = sm100_utils_basic.make_smem_layout_b(tiled_mma, (128, 128, 128), self.dt, 1)
        sC_l = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0,
                                         cute.make_layout((128, 128, 4), stride=(128, 1, 16384)))
        sC4_l = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0,
                                          cute.make_layout((128, 64, 4), stride=(64, 1, 8192)))
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        mXv = cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=[1, 3, 2, 0]))   # (s, d, h, b)
        tma_ld, mXt = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE), mXv, cute.select(sA_l, mode=[0, 1, 2]),
            (128, 128, 128), tiled_mma, cta_layout_vmnk.shape)
        mQv = cute.make_tensor(mQb.iterator, cute.select(mQb.layout, mode=[1, 3, 2, 0]))
        tma_st, mQt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mQv,
                                                  cute.select(sC_l, mode=[0, 1]), (128, 128))
        mX4v = cute.make_tensor(mX4b.iterator, cute.select(mX4b.layout, mode=[1, 3, 2, 0]))
        tma_st4, mX4t = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mX4v,
                                                    cute.select(sC4_l, mode=[0, 1]), (128, 64))
        self.kernel(mXt, mX, mQt, mX4t, mSFTok, mSFDim, mExp, mFac, mSF4, mNvS, mCos, mSin, mStats, seed, tma_ld,
                    tma_st, tma_st4, tiled_mma, sA_l, sB_l, sC_l, sC4_l,
                    *((mVt, mNT, mEvR, mSF4R, mQb, mX4b) if const_expr(self.compact) else ())).launch(
            grid=[grid_x, 1, 1], block=[NTHREADS, 1, 1], smem=self.smem_bytes, stream=stream)

    # ----------------------------------------------------------------------------------------------------------- device
    @cute.jit
    def _decode(self, idx, tiles, nheads):
        """Tile idx -> (t, h, b, b h + h): heads fastest (the concurrent tiles share rows: DRAM locality). (compact:
        b = 1, t indexes the tile table)"""
        if const_expr(self.compact):
            hh = idx % nheads
            t = idx // nheads
            return t, hh, Int32(0), hh
        hh = idx % nheads
        rest = idx // nheads
        t = rest % tiles
        bb = rest // tiles
        return t, hh, bb, bb * nheads + hh

    @cute.kernel
    def kernel(self, mXt, mX, mQt, mX4t, mSFTok, mSFDim, mExp, mFac, mSF4, mNvS, mCos, mSin, mStats, seed: Int64,
               tma_ld: cute.CopyAtom, tma_st: cute.CopyAtom, tma_st4: cute.CopyAtom, tiled_mma: cute.TiledMma,
               sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, sC_l: cute.ComposedLayout,
               sC4_l: cute.ComposedLayout, mVt=None, mNT=None, mEvR=None, mSF4R=None, mQg=None, mX4g=None):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        s = mX.shape[1]
        nheads = mX.shape[2]
        tiles = s // 128
        ntiles = mX.shape[0] * nheads * tiles
        if const_expr(self.compact):   # (the tile table's bound; the device tile count)
            tiles = mVt.shape[1]
            s = tiles * 128
            ntiles = mNT[0] * nheads
        S = self.stages
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(self.dt, sA_l.outer, 1024, sA_l.inner)
        sB = smem.allocate_tensor(self.dt, sB_l.outer, 1024, sB_l.inner)
        sC = smem.allocate_tensor(Uint8, sC_l.outer, 1024, sC_l.inner)
        sC4 = sC
        if const_expr(self.nvfp4):
            sC4 = smem.allocate_tensor(Uint8, sC4_l.outer, 1024, sC4_l.inner)
        # full[S] (TMA), empty[S] (MMA commit, + the 128 raw readers with nvfp4_only), ready[S] (RoPE'd stage),
        # accf[2] (MMA commit), acce[2] (the 128 epilogue threads)
        bars = smem.allocate_tensor(Int64, cute.make_layout(3 * S + 4), 8)
        tmem_hold = smem.allocate_tensor(Int32, cute.make_layout(1), 8)
        sfx = smem.allocate_tensor(Int32, cute.make_layout(16), 16)          # [eg][parity][block] group-0 bytes
        sfm = smem.allocate_tensor(Float32, cute.make_layout(16), 16)        # [eg][parity][block] block maxima
        full = bars.iterator
        empty = bars.iterator + S
        ready = bars.iterator + 2 * S
        accf = bars.iterator + 3 * S
        acce = bars.iterator + 3 * S + 2
        thr_mma = tiled_mma.get_slice(0)
        if warp == TMA_WARP:
            if lane == 0:
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 129 if const_expr(self.nvfp4_only) else 1)
                    cute.arch.mbarrier_init(ready + i, 128)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(accf + i, 1)
                    cute.arch.mbarrier_init(acce + i, 128)
            cpasync.prefetch_descriptor(tma_ld)
            cpasync.prefetch_descriptor(tma_st)
            if const_expr(self.nvfp4):
                cpasync.prefetch_descriptor(tma_st4)
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(256, tmem_hold.iterator)
        if warp < 4:
            T5.build_b_image(sB.iterator.toint(), tidx, self.sign_words, self.dt is cutlass.Float16)
        cute.arch.mbarrier_init_fence()
        cute.arch.fence_proxy("async.shared", space="cta")
        if warp == TMA_WARP:      # the producer starts loading while B is being built
            cute.arch.barrier_arrive(barrier_id=15, number_of_threads=NTHREADS)
        else:
            cute.arch.barrier(barrier_id=15, number_of_threads=NTHREADS)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_hold.iterator)
        acc_layout = thr_mma.make_fragment_C(thr_mma.partition_shape_C((128, 128))).layout
        nmine = (ntiles - cta + ncta - 1) // ncta
        if warp == TMA_WARP:
            for i in cutlass.range(nmine, unroll=1):
                t, hh, bb, bh = self._decode(cta + i * ncta, tiles, nheads)
                st = i % S
                cute.arch.mbarrier_wait(empty + st, ((i // S) & 1) ^ 1)
                gX = None
                if const_expr(self.compact):   # (the box at the tile's packed start; rows past T zero-filled)
                    gX = cute.local_tile(cute.domain_offset((mVt[0, t], 0), mXt[None, None, hh, bb]), (128, 128), (0, 0))
                else:
                    gX = cute.local_tile(mXt[None, None, hh, bb], (128, 128), (t, 0))
                tCgX = thr_mma.partition_A(gX)
                tAsA, tAgA = cpasync.tma_partition(tma_ld, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                                   cute.group_modes(tCgX, 0, 3))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + st, 32768)
                cute.copy(tma_ld, tAgA, tAsA[None, st], tma_bar_ptr=full + st)
        elif warp == MMA_WARP:
            cute.arch.relinquish_tmem_alloc_permit()
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            for i in cutlass.range(nmine, unroll=1):
                st = i % S
                b = i % 2
                if const_expr(self.rope):
                    cute.arch.mbarrier_wait(ready + st, (i // S) & 1)
                else:
                    cute.arch.mbarrier_wait(full + st, (i // S) & 1)
                cute.arch.mbarrier_wait(acce + b, ((i // 2) & 1) ^ 1)
                T5.tcgen05_after_sync()
                tAcc = cute.make_tensor(tmem_ptr + 128 * b, acc_layout)
                for k in cutlass.range_constexpr(8):
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k != 0)
                    cute.gemm(mma_atom, tAcc, tCrA[None, None, k, st], tCrB[None, None, k, 0], tAcc)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + st)
                    tcgen05.commit(accf + b)
        else:
            self._epilogue(warp, tidx, lane, cta, ncta, nmine, tiles, nheads, s, sA, sC, sC4, sfx, sfm, full, empty,
                           ready, accf, acce, tmem_ptr, acc_layout, thr_mma, mQt, mX4t, mSFTok, mSFDim, mExp, mFac,
                           mSF4, mNvS, mCos, mSin, mStats, seed, tma_st, tma_st4,
                           *((mVt, mEvR, mSF4R, mQg, mX4g) if const_expr(self.compact) else ()))
        cute.arch.sync_threads()
        if warp == MMA_WARP:
            cute.arch.dealloc_tmem(tmem_ptr, 256)

    @cute.jit
    def _rope_stage(self, i, cta, ncta, tiles, nheads, sA, full, ready, r, mCos, mSin):
        """RoPE on tile i's stage, in place: thread r reads its raw row, rotates the (j, j + 64) pairs in fp32 (the
        quantisers' fma forms), rounds to the storage dtype and writes the row back for the MMA."""
        S = self.stages
        st = i % S
        t, _hh, _bb, _bh = self._decode(cta + i * ncta, tiles, nheads)
        cute.arch.mbarrier_wait(full + st, (i // S) & 1)
        addr = sA.iterator.toint() + Int32(32768) * st
        w = cute.make_rmem_tensor(cute.make_layout(64), Int32)
        T5.read_a_row(addr, r, w)
        av = cute.make_tensor(cute.recast_ptr(w.iterator, dtype=self.dt), cute.make_layout(128))
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        x.store(av.load().to(Float32))
        rope_rotate_row(x, mCos, mSin, t * 128 + r, self.rope_t)
        av.store(x.load().to(self.dt))
        T5.write_a_row(addr, r, w)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(ready + st)

    @cute.jit
    def _group_bytes(self, am, byte4, inv4):
        """kernel_tc's selector on the warp's (32-token block's) group maxima am[0..3] (already merged per block_columns
        and reduced over the warp): byte4 / inv4 per 32-column group."""
        for g in cutlass.range_constexpr(4):
            if const_expr(g == 0 or self.cols < 128):
                if const_expr(self.cols == 64 and g % 2 == 1):
                    byte4[g] = byte4[g - 1]
                    inv4[g] = inv4[g - 1]
                else:
                    byte_g, _sfr, inv_g = _ue8m0_select(am[g], self.fmax, self.flush)
                    byte4[g] = byte_g
                    inv4[g] = inv_g
            else:
                byte4[g] = byte4[0]
                inv4[g] = inv4[0]

    @cute.jit
    def _block_maxima(self, am):
        """Merge the row's four 32-column maxima per block_columns, then the warp's 32 rows (5 shuffle levels)."""
        if const_expr(self.cols == 128):
            m = cute.arch.fmax(cute.arch.fmax(am[0], am[1]), cute.arch.fmax(am[2], am[3]))
            for lv in cutlass.range_constexpr(5):
                m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << lv))
            for g in cutlass.range_constexpr(4):
                am[g] = m
        else:
            if const_expr(self.cols == 64):
                am[0] = cute.arch.fmax(am[0], am[1])
                am[1] = am[0]
                am[2] = cute.arch.fmax(am[2], am[3])
                am[3] = am[2]
            for g in cutlass.range_constexpr(4):
                m = am[g]
                for lv in cutlass.range_constexpr(5):
                    m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << lv))
                am[g] = m

    @cute.jit
    def _codes(self, src, inv4, sCw, buf, r, packed: cutlass.Constexpr, grow=None, gok=None):
        """FP8 codes of the row (packed: 64 words of 16-bit values, exact in fp32; else 128 fp32) times the group's
        1 / scale -> the swizzled SMEM code tile (16-byte chunk q of row r at q ^ (r & 7)). kernel_tc's per-element
        arithmetic: mul (ftz under the flush rule), clamp to +-FMAX, cvt.rn.satfinite."""
        fmax = self.fmax
        half = self.half
        for blk in cutlass.range_constexpr(8):
            w4 = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            inv = inv4[blk // 2]
            for wq in cutlass.range_constexpr(4):
                y = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                for e in cutlass.range_constexpr(0, 4, 2):
                    if const_expr(packed):
                        wd = 8 * blk + 2 * wq + e // 2
                        x0 = half_lo_f32(src[wd], half)
                        x1 = half_hi_f32(src[wd], half)
                    else:
                        x0 = src[16 * blk + 4 * wq + e]
                        x1 = src[16 * blk + 4 * wq + e + 1]
                    y[e], y[e + 1] = cute.arch.mul_packed_f32x2((x0, x1), (inv, inv),
                                                                ftz=True if const_expr(self.flush) else None)
                    y[e] = _clamp(y[e], fmax)
                    y[e + 1] = _clamp(y[e + 1], fmax)
                w4[wq] = _pack4(self.fmt, y[0], y[1], y[2], y[3])
            if const_expr(grow is not None):   # (compact: the row's codes straight to its packed row, real rows only)
                if gok:
                    cute.autovec_copy(w4, cute.make_tensor(cute.make_ptr(Int32, (grow + 16 * blk).toint(),
                                                                         cute.AddressSpace.gmem, assumed_align=16),
                                                           cute.make_layout(4)))
            else:
                off = buf * 4096 + r * 32 + 4 * (blk ^ (r & 7))
                cute.autovec_copy(w4, cute.make_tensor(sCw.iterator + off, cute.make_layout(4)))

    @cute.jit
    def _nvfp4(self, v, inv4, sinv, sC4w, buf, r, rl, rw, bh, tiles, t, idx_row: Int64, seed: Int64, mSF4,
               grow4=None, gok=None, prow=None, mSF4R=None):
        """kernel_tc's NVFP4 1x16 copy (_tc_nvfp4) of the rotated row, one thread per row: 8 groups of 16
        (nvfp4_group16 = nvfp4_half_group8 on both halves: the same SF, codes and stochastic indices)."""
        sfw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        sfw[0] = Int32(0)
        sfw[1] = Int32(0)
        for g in cutlass.range_constexpr(8):
            v16 = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            for e in cutlass.range_constexpr(16):
                xv = v[16 * g + e]
                if const_expr(self.flush and not (self.nvfp4_encoding == "qk_operand" and self.nvfp4_scaling == "raw")):
                    if cute.math.absf(xv) < Float32(FLT_MIN):
                        xv = mul_rn(xv, Float32(0.0))
                if const_expr(self.nvfp4_scaling == "token_block"):
                    xv = xv * inv4[g // 2]
                elif const_expr(self.nvfp4_scaling == "per_tensor"):
                    xv = xv * sinv
                v16[e] = xv
            gb, _sr, w0, w1 = nvfp4_group16(v16, self.nvfp4_stochastic, idx_row + Int64(g), seed, self.nvfp4_encoding)
            sfw[g // 4] = sfw[g // 4] | (gb << (8 * (g % 4)))
            w2 = cute.make_rmem_tensor(cute.make_layout(2), Int32)
            w2[0] = w0
            w2[1] = w1
            if const_expr(grow4 is not None):   # (compact: words 2 g, 2 g + 1 of the row's packed row)
                if gok:
                    cute.autovec_copy(w2, cute.make_tensor(cute.make_ptr(Int32, (grow4 + 8 * g).toint(),
                                                                         cute.AddressSpace.gmem, assumed_align=8),
                                                           cute.make_layout(2)))
            else:
                off4 = buf * 2048 + r * 16 + 4 * ((g // 2) ^ ((r >> 1) & 3)) + 2 * (g % 2)
                cute.autovec_copy(w2, cute.make_tensor(sC4w.iterator + off4, cute.make_layout(2)))
        if const_expr(grow4 is not None):   # (compact: the dense per-row SF words)
            if gok:
                for hw in cutlass.range_constexpr(2):
                    mSF4R[bh, prow * 2 + hw] = sfw[hw]
        else:
            for hw in cutlass.range_constexpr(2):
                mSF4[(bh * tiles + t) * 256 + 128 * hw + rl * 4 + rw] = sfw[hw]

    @cute.jit
    def _epilogue(self, warp, tidx, lane, cta, ncta, nmine, tiles, nheads, s, sA, sC, sC4, sfx, sfm, full, empty,
                  ready, accf, acce, tmem_ptr, acc_layout, thr_mma, mQt, mX4t, mSFTok, mSFDim, mExp, mFac, mSF4, mNvS,
                  mCos, mSin, mStats, seed: Int64, tma_st: cute.CopyAtom, tma_st4: cute.CopyAtom, mVt=None, mEvR=None,
                  mSF4R=None, mQg=None, mX4g=None):
        S = self.stages
        eg = warp // 4
        r = tidx % 128
        rw = r // 32
        rl = r % 32
        atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        tiled_t2r = tcgen05.make_tmem_copy(atom, cute.make_tensor(tmem_ptr, acc_layout))
        thr_t2r = tiled_t2r.get_slice(r)
        tTR_t = thr_t2r.partition_S(cute.make_tensor(tmem_ptr + 128 * eg, acc_layout))
        tTR_c = thr_t2r.partition_D(thr_mma.partition_C(cute.make_identity_tensor((128, 128))))
        half = self.half
        sCw = cute.make_tensor(cute.recast_ptr(sC.iterator, dtype=Int32), cute.make_layout(4 * 4096))
        sC4w = cute.make_tensor(cute.recast_ptr(sC4.iterator, dtype=Int32), cute.make_layout(4 * 2048))
        sinv = Float32(1.0)
        if const_expr(self.nvfp4 and self.nvfp4_scaling == "per_tensor"):
            sinv = div_full_f32(Float32(1.0), mNvS[0])
        if const_expr(self.rope):
            if eg < nmine:
                self._rope_stage(eg, cta, ncta, tiles, nheads, sA, full, ready, r, mCos, mSin)
        for j in cutlass.range((nmine - eg + 1) // 2, unroll=1):
            i = 2 * j + eg
            if const_expr(self.rope):
                if i + 2 < nmine:
                    self._rope_stage(i + 2, cta, ncta, tiles, nheads, sA, full, ready, r, mCos, mSin)
            t, hh, bb, bh = self._decode(cta + i * ncta, tiles, nheads)
            q0 = t * 128
            st = i % S
            prow = None
            gok = None
            grow = None
            grow4 = None
            if const_expr(self.compact):   # (this thread's packed row; real rows only)
                gok = r < (mVt[1, t] & Int32(0xFFFF))
                prow = mVt[0, t] + r
                pr0 = cutlass.select_(gok, prow, Int32(0))
                grow = mQg[0, pr0, hh, None].iterator
                if const_expr(self.nvfp4):
                    grow4 = mX4g[0, pr0, hh, None].iterator
            buf = 2 * eg + (j & 1)
            par = j & 1
            tb = (bh * tiles + t) * 128
            am = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            byte4 = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            inv4 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            v = cute.make_rmem_tensor(cute.make_layout(128), Float32)
            idx_row = (Int64(bh) * Int64(s) + Int64(q0)) * Int64(4096) + Int64(r) * Int64(8)
            if const_expr(self.nvfp4_only):
                # ---- the E4M3 side of the RAW (RoPE'd) row: the stage, read back before it is released; it does not
                # need the MMA, so it runs while the rotation is in flight
                if const_expr(not self.rope):
                    cute.arch.mbarrier_wait(full + st, (i // S) & 1)
                a = cute.make_rmem_tensor(cute.make_layout(64), Int32)
                T5.read_a_row(sA.iterator.toint() + Int32(32768) * st, r, a)
                cute.arch.mbarrier_arrive(empty + st)
                if const_expr(self.compact):   # (a row past the document: the aligned layout's zero row)
                    for wd in cutlass.range_constexpr(64):
                        a[wd] = cutlass.select_(gok, a[wd], Int32(0))
                for g in cutlass.range_constexpr(4):
                    acc = Int32(0)
                    for wd in cutlass.range_constexpr(16 * g, 16 * g + 16):
                        acc = absmax_x2(acc, a[wd], half)
                    am[g] = cute.arch.fmax(half_lo_f32(acc, half), half_hi_f32(acc, half))
                self._block_maxima(am)
                self._group_bytes(am, byte4, inv4)
                self._codes(a, inv4, sCw, buf, r, True, grow, gok)
            # ---- the rotated row from TMEM
            cute.arch.mbarrier_wait(accf + eg, j & 1)
            T5.tcgen05_after_sync()
            rR = cute.make_rmem_tensor(tTR_c.shape, Float32)
            cute.copy(tiled_t2r, tTR_t, rR)
            cute.arch.fence_view_async_tmem_load()
            T5.tcgen05_before_sync()
            cute.arch.mbarrier_arrive(acce + eg)
            for c in cutlass.range_constexpr(0, 128, 2):
                v[c], v[c + 1] = cute.arch.mul_packed_f32x2((rR[c], rR[c + 1]),
                                                            (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))
            if const_expr(self.compact):   # (a row past the document: the aligned layout's zero row, rotated: +0)
                for c in cutlass.range_constexpr(128):
                    v[c] = cutlass.select_(gok, v[c], Float32(0.0))
            if const_expr(not self.nvfp4_only):
                for g in cutlass.range_constexpr(4):
                    acc8 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                    for e in cutlass.range_constexpr(4):
                        acc8[e] = cute.math.absf(v[32 * g + e])
                    for e in cutlass.range_constexpr(4, 32):
                        acc8[e % 4] = cute.arch.fmax(acc8[e % 4], cute.math.absf(v[32 * g + e]))
                    am[g] = cute.arch.fmax(cute.arch.fmax(cute.arch.fmax(acc8[0], acc8[1]),
                                                          cute.arch.fmax(acc8[2], acc8[3])), Float32(0.0))
                self._block_maxima(am)
                self._group_bytes(am, byte4, inv4)
                self._codes(v, inv4, sCw, buf, r, False, grow, gok)
            if const_expr(self.nvfp4):
                self._nvfp4(v, inv4, sinv, sC4w, buf, r, rl, rw, bh, tiles, t, idx_row, seed, mSF4, grow4, gok, prow,
                            mSF4R)
            cute.arch.fence_proxy("async.shared", space="cta")
            if r == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)   # the previous tile's stores have left SMEM
            # ---- the tables (kernel_tc's words and bytes; compact: none)
            if const_expr(not self.compact):
                mSFTok[tb + rl * 4 + rw] = byte4[0] | (byte4[1] << 8) | (byte4[2] << 16) | (byte4[3] << 24)
                for gc in cutlass.range_constexpr(4):
                    bptr = cute.make_ptr(Uint8, mSFDim.iterator.toint() + Int64(tb + lane * 4 + gc) * Int64(4)
                                         + Int64(rw), cute.AddressSpace.gmem, assumed_align=1)
                    cute.make_tensor(bptr, cute.make_layout(1))[0] = Uint8(byte4[gc])
            if const_expr(self.exponent_table and not self.compact):
                if lane < 4:
                    bv = byte4[0]
                    for gq in cutlass.range_constexpr(1, 4):
                        if lane == gq:
                            bv = byte4[gq]
                    mExp[(bh * (tiles * 4) + 4 * t + rw) * 4 + lane] = Uint8(bv)
            if lane == 0:
                sfx[8 * eg + 4 * par + rw] = byte4[0]
                sfm[8 * eg + 4 * par + rw] = cute.arch.fmax(cute.arch.fmax(am[0], am[1]), cute.arch.fmax(am[2], am[3]))
            cute.arch.barrier(barrier_id=1 + eg, number_of_threads=128)
            if const_expr(self.compact):
                # sfd: each real row's block's effective exponent (the factor rule: an all-zero block takes its tile's
                # largest), at its packed row; tile 0 zeroes the tail rows [T, T + 128)
                evd = sfx[8 * eg + 4 * par + rw]
                if evd == 0:
                    evd = cutlass.max(cutlass.max(sfx[8 * eg + 4 * par], sfx[8 * eg + 4 * par + 1]),
                                      cutlass.max(sfx[8 * eg + 4 * par + 2], sfx[8 * eg + 4 * par + 3]))
                if gok:
                    mEvR[bh, prow] = Uint8(evd)
                if t == 0:
                    mEvR[bh, mQg.shape[1] + r] = Uint8(0)
            if const_expr(not self.compact):
                if r == 0:
                    gC = cute.local_tile(mQt[None, None, hh, bb], (128, 128), (t, 0))
                    tCsC, tCgC = cpasync.tma_partition(tma_st, 0, cute.make_layout(1),
                                                       cute.group_modes(sC[None, None, buf], 0, 2),
                                                       cute.group_modes(gC, 0, 2))
                    cute.copy(tma_st, tCsC, tCgC)
                    if const_expr(self.nvfp4):
                        gC4 = cute.local_tile(mX4t[None, None, hh, bb], (128, 64), (t, 0))
                        tCsC4, tCgC4 = cpasync.tma_partition(tma_st4, 0, cute.make_layout(1),
                                                             cute.group_modes(sC4[None, None, buf], 0, 2),
                                                             cute.group_modes(gC4, 0, 2))
                        cute.copy(tma_st4, tCsC4, tCgC4)
                    cute.arch.cp_async_bulk_commit_group()
            if const_expr(self.factors and not self.compact):
                if lane == 0:   # 2^(e - 127) of column group 0; an all-zero block: its tile's largest, 1 for a zero tile
                    ev = byte4[0]
                    if ev == 0:
                        ev = cutlass.max(cutlass.max(sfx[8 * eg + 4 * par], sfx[8 * eg + 4 * par + 1]),
                                         cutlass.max(sfx[8 * eg + 4 * par + 2], sfx[8 * eg + 4 * par + 3]))
                    f = Float32(1.0)
                    if ev > 0:
                        f = ex2_ftz(Float32(ev - 127))
                    mFac[bb, hh, 4 * t + rw] = f
            if const_expr(self.stat_amax):
                if r == 32:
                    tm = cute.arch.fmax(cute.arch.fmax(sfm[8 * eg + 4 * par], sfm[8 * eg + 4 * par + 1]),
                                        cute.arch.fmax(sfm[8 * eg + 4 * par + 2], sfm[8 * eg + 4 * par + 3]))
                    mStats[bh * tiles + t] = tm
        if r == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)


# kernel_tc's arithmetic helpers (quant.fp8), imported late (fp8 imports this module)
def _bind():
    global _ue8m0_select, _clamp, _pack4, absmax_x2, half_lo_f32, half_hi_f32, mul_rn, ex2_ftz, div_full_f32
    from flash_attn.cute.quant import fp8 as F
    _ue8m0_select, _clamp, _pack4 = F._ue8m0_select, F._clamp, F._pack4
    absmax_x2, half_lo_f32, half_hi_f32, mul_rn, ex2_ftz = F.absmax_x2, F.half_lo_f32, F.half_hi_f32, F.mul_rn, F.ex2_ftz
    div_full_f32 = F.div_full_f32

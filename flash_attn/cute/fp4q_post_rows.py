"""FP4QPostprocessRows (bwdk): the all-FP4 backward's post launch with a coalesced thread mapping. Opt-in
(KernelTuning.bwd_fp4q_post = 'rows'); FP4QPostprocess stays the default.

The same arithmetic per element as FP4QPostprocess (acc -> f32, FMUL2 by scale, FMUL2 by rcp_rn(F_q); dK / dV rows:
FMUL2 by the row factor; one cvt.rn), so the same bits. FP4QPostprocess gives each thread one 256-byte row (a warp's
16-byte access spans 32 rows 256 B * h apart): 104 us at A1 for 402 MB moved. Here (DQPostprocessRows' mapping) a warp
covers 2 rows x 16 column chunks of 8 elements per step -- 512 contiguous accumulator bytes per load, two contiguous
256-byte output row segments per store -- 256 threads x 8 steps of 16 rows per 128-row tile, all loads before any store.
dQ needs s_q % 128 == 0 (the caller checks); dK / dV tiles as FP4QPostprocess (s_k % 128 == 0)."""
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from flash_attn.cute.dq_postprocess_cute import FP4QPostprocess
from flash_attn.cute.quant.fp4 import bits_f32, f32_bits, rcp_rn


class FP4QPostprocessRows(FP4QPostprocess):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.threads = 256
        # 1 / F_q from the exponent bits (F_q is a power of two): rcp.rn.f32 is a multi-instruction IEEE sequence and
        # cost the dQ part 20.6 -> 31.2 us at b4 s4096 h16 (micro_post.py); bitwise rcp_rn for every normal power of two
        self.pow2_rcp = True

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mdQ: cute.Tensor, scale: Float32, mRowF: cute.Tensor, mdK: cute.Tensor,
               mdV: cute.Tensor, mFdK: cute.Tensor, mFdV: cute.Tensor, mScale: cute.Tensor, n_q_ctas: cutlass.Int32,
               mSeedCtr: Optional[cute.Tensor] = None, mTileDoc: Optional[cute.Tensor] = None):
        if const_expr(self.device_scale):
            scale = mScale[0]   # (documents never take this form: fp4q_postprocess keeps them on the per-row kernel)
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if const_expr(self.seed_counter):
            if bid == 0 and tidx == 0:
                mSeedCtr[0] = mSeedCtr[0] + cutlass.Int64(1)
        warp = tidx // 32
        lane = tidx % 32
        j = lane % 16           # column chunk: d = 8 j + (0..7)
        n = 8
        if bid < n_q_ctas:
            s_q = mdQ.shape[1]
            h = mdQ.shape[2]
            tiles = s_q // 128
            t = bid % tiles
            hh = (bid // tiles) % h
            bb = bid // (tiles * h)
            per16 = 128 // self.acc_dtype.width
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
            atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
            acc = cute.make_rmem_tensor(cute.make_layout((8, n)), self.acc_dtype)
            tile_it = mAcc[bb, hh, t * 128, None].iterator
            fq = cute.make_rmem_tensor(cute.make_layout(n), Float32)
            if const_expr(self.row_divisor):   # the row factors first: their latency under the accumulator loads
                for i in cutlass.range_constexpr(n):
                    r = 16 * i + 2 * warp + lane // 16
                    fq[i] = mRowF[(bb * h + hh) * s_q + t * 128 + r]
            for i in cutlass.range_constexpr(n):
                r = 16 * i + 2 * warp + lane // 16
                for hv in cutlass.range_constexpr(8 // per16):
                    ptr = cute.make_ptr(self.acc_dtype, (tile_it + (r * 128 + 8 * j + hv * per16)).toint(),
                                        cute.AddressSpace.gmem, assumed_align=16)
                    cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(per16)),
                              cute.make_tensor(acc[None, i].iterator + hv * per16, cute.make_layout(per16)))
            rinv = cute.make_rmem_tensor(cute.make_layout(n), Float32)
            if const_expr(self.row_divisor):
                for i in cutlass.range_constexpr(n):
                    if const_expr(self.pow2_rcp):
                        # F_q = 2^e (normal): 1 / F_q = 2^-e exactly, = rcp_rn(F_q), from the exponent bits
                        rinv[i] = bits_f32(cutlass.Int32(0x7F000000) - f32_bits(fq[i]))
                    else:
                        rinv[i] = rcp_rn(fq[i])
            for i in cutlass.range_constexpr(n):
                r = 16 * i + 2 * warp + lane // 16
                x = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                x.store(acc[None, i].load().to(Float32))
                for e in cutlass.range_constexpr(0, 8, 2):
                    x[e], x[e + 1] = cute.arch.mul_packed_f32x2((x[e], x[e + 1]), (scale, scale))
                if const_expr(self.row_divisor):
                    for e in cutlass.range_constexpr(0, 8, 2):
                        x[e], x[e + 1] = cute.arch.mul_packed_f32x2((x[e], x[e + 1]), (rinv[i], rinv[i]))
                o = cute.make_rmem_tensor(cute.make_layout(8), self.out_dtype)
                o.store(x.load().to(self.out_dtype))
                optr = cute.make_ptr(self.out_dtype, (mdQ[bb, t * 128 + r, hh, None].iterator + 8 * j).toint(),
                                     cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom_o, o, cute.make_tensor(optr, cute.make_layout(8)))
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
                if which == 0:
                    self.scale_tile_rows(mdK, mFdK, bb, t, hh, warp, lane)
                else:
                    self.scale_tile_rows(mdV, mFdV, bb, t, hh, warp, lane)

    @cute.jit
    def scale_tile_rows(self, mX: cute.Tensor, mF: cute.Tensor, bb, t, hh, warp, lane):
        n = 8
        j = lane % 16
        s_k = mX.shape[1]
        hk = mX.shape[2]
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        v = cute.make_rmem_tensor(cute.make_layout((8, n)), self.out_dtype)
        f = cute.make_rmem_tensor(cute.make_layout(n), Float32)
        for i in cutlass.range_constexpr(n):
            r = 16 * i + 2 * warp + lane // 16
            ptr = cute.make_ptr(self.out_dtype, (mX[bb, t * 128 + r, hh, None].iterator + 8 * j).toint(),
                                cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom_o, cute.make_tensor(ptr, cute.make_layout(8)),
                      cute.make_tensor(v[None, i].iterator, cute.make_layout(8)))
            f[i] = mF[(bb * hk + hh) * s_k + t * 128 + r]
        for i in cutlass.range_constexpr(n):
            r = 16 * i + 2 * warp + lane // 16
            x = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            x.store(v[None, i].load().to(Float32))
            for e in cutlass.range_constexpr(0, 8, 2):
                x[e], x[e + 1] = cute.arch.mul_packed_f32x2((x[e], x[e + 1]), (f[i], f[i]))
            o = cute.make_rmem_tensor(cute.make_layout(8), self.out_dtype)
            o.store(x.load().to(self.out_dtype))
            ptr = cute.make_ptr(self.out_dtype, (mX[bb, t * 128 + r, hh, None].iterator + 8 * j).toint(),
                                cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom_o, o, cute.make_tensor(ptr, cute.make_layout(8)))

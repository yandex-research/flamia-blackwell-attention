# CuTe DSL pieces of the fused Q / K Hadamard back-rotation (the qk_hadamard option, see qk_hadamard.py).
#
# The backward's dQ / dK come out of GEMMs against rotated operands, i.e. rotated: dQ~ = dQ R. The true gradient is
# dQ = dQ~ R^T = (dQ~ H / sqrt(d)) * signs (H symmetric). Both helpers below do that in registers, inside kernels that
# already hold the values: the dQ postprocess (one thread per query row, all 128 columns in registers) and the dK
# epilogue of the dk8 / FP8 backward kernels (two warpgroups per 128-column row, 64 columns each).
#
# All index work is trace-time: `range_constexpr` loops over compile-time indices, and the signs are compile-time
# constants (a tuple of +-1), so a sign costs nothing -- it selects which of two precomputed multipliers scales a column.
import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

INV_SQRT_128 = 1.0 / math.sqrt(128.0)


@cute.jit
def wht_inplace(x: cute.Tensor, n: cutlass.Constexpr[int], base: cutlass.Constexpr[int] = 0):
    """Unnormalised Walsh-Hadamard butterflies over x[base : base + n] (register fragment, flat index), in place."""
    for lvl in cutlass.range_constexpr(int(math.log2(n))):
        hh = 1 << lvl
        for blk in cutlass.range_constexpr(0, n, 2 * hh):
            for j in cutlass.range_constexpr(hh):
                a = x[base + blk + j]
                b = x[base + blk + j + hh]
                x[base + blk + j] = a + b
                x[base + blk + j + hh] = a - b


@cute.jit
def unrotate_row128(x: cute.Tensor, scale: Float32, sg: cutlass.Constexpr):
    """x[0:128] (one row, column d at index d) <- scale * (x H / sqrt(128)) * signs, in place (packed: bitwise the
    scalar butterflies)."""
    wht_packed(x, 128)
    sp = scale * Float32(INV_SQRT_128)
    sn = -sp
    for d in cutlass.range_constexpr(0, 128, 2):
        x[d], x[d + 1] = cute.arch.mul_packed_f32x2((x[d], x[d + 1]), (sp if const_expr(sg[d] > 0) else sn,
                                                                       sp if const_expr(sg[d + 1] > 0) else sn))


@cute.jit
def dk_unrotated_half(thr_copy_t2r: cute.TiledCopy, src0: cute.Tensor, src1: cute.Tensor, frag_like: cute.Tensor,
                      u: cute.Tensor, wg_idx, scale, n_st: cutlass.Constexpr[int], rep: cutlass.Constexpr[int],
                      sg: cutlass.Constexpr):
    """The dK epilogue's back-rotation. The 128 columns of a dK row live in TMEM; warpgroup w normally loads and stores
    columns [64 w, 64 w + 64). Here each thread loads BOTH halves of its row, stage by stage (src0 / src1: the t2r source
    partitions of half 0 / half 1, indexed by stage when n_st > 1), and folds the top Hadamard level (column c with
    c + 64) straight away -- u_c = A_c + A_{c+64} for w = 0, A_c - A_{c+64} for w = 1 -- so only this half's 64 values
    stay live. The six lower levels act within the half. u (64 fp32, flat) receives scale * (row H / sqrt(128)) * signs
    for this half's columns, stage-major: u[st * rep + i] is column 64 w + st * rep + i."""
    sgnw = Float32(1.0) - Float32(2.0) * Float32(wg_idx)
    for st in cutlass.range_constexpr(n_st):
        a = cute.make_rmem_tensor(frag_like.shape, Float32)
        b = cute.make_rmem_tensor(frag_like.shape, Float32)
        if const_expr(n_st > 1):
            cute.copy(thr_copy_t2r, src0[None, st], a)
            cute.copy(thr_copy_t2r, src1[None, st], b)
        else:
            cute.copy(thr_copy_t2r, src0, a)
            cute.copy(thr_copy_t2r, src1, b)
        cute.arch.fence_view_async_tmem_load()
        # a + sgnw * b: an exact product and one rounding (FFMA2), bitwise the scalar a +- b
        for i in cutlass.range_constexpr(0, rep, 2):
            u[st * rep + i], u[st * rep + i + 1] = cute.arch.fma_packed_f32x2((b[i], b[i + 1]), (sgnw, sgnw),
                                                                              (a[i], a[i + 1]))
    wht_packed(u, n_st * rep)
    sc = Float32(INV_SQRT_128)
    if const_expr(scale is not None):
        sc = scale * Float32(INV_SQRT_128)
    m_same = sc
    m_flip = sc * sgnw
    half = n_st * rep
    for c in cutlass.range_constexpr(0, half, 2):
        f = []
        for e in cutlass.range_constexpr(2):
            if const_expr(sg[c + e] == sg[half + c + e]):
                f.append(m_same if const_expr(sg[c + e] > 0) else -m_same)
            else:
                f.append(m_flip if const_expr(sg[c + e] > 0) else -m_flip)
        u[c], u[c + 1] = cute.arch.mul_packed_f32x2((u[c], u[c + 1]), (f[0], f[1]))


@cute.jit
def wht_packed(x: cute.Tensor, n: cutlass.Constexpr[int]):
    """wht_inplace(x, n, 0) with packed f32x2 adds / subtracts for the levels H >= 2 (two IEEE-rn results per instruction:
    bitwise the scalar form); level 1 stays scalar (its operands are one register pair)."""
    for k in cutlass.range_constexpr(0, n, 2):
        a = x[k]
        b = x[k + 1]
        x[k] = a + b
        x[k + 1] = a - b
    for lvl in cutlass.range_constexpr(1, int(math.log2(n))):
        hh = 1 << lvl
        for blk in cutlass.range_constexpr(0, n, 2 * hh):
            for j in cutlass.range_constexpr(0, hh, 2):
                i0 = blk + j
                a0, a1 = x[i0], x[i0 + 1]
                b0, b1 = x[i0 + hh], x[i0 + hh + 1]
                x[i0], x[i0 + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                x[i0 + hh], x[i0 + hh + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))


# ---- the dQ postprocess's back-rotation in COALESCED thread mappings
# The thread-per-row DQPostprocess / DQPostprocess2CTA scatter 16-byte stores over 32 rows per warp instruction. The two
# mappings below keep a row in a lane pair or a half warp and are bitwise unrotate_row128: the same Hadamard levels in
# the same order (H = 1, 2, 4, ..., 64), a cross-lane level as fma(own, +-1, partner) (lower lane own + partner = a + b,
# upper partner - own = a - b; exact products, one rounding), the same final scale * (+-1 / sqrt(128)) multiply.


@cute.jit
def unrotate_pair64(x: cute.Tensor, h, scale: Float32, sg: cutlass.Constexpr):
    """Lane pair, interleaved 8-column chunks: lane parity h holds x[8 k + e] = column 16 k + 8 h + e (k, e = 0..7) of a
    row whose other chunks the partner lane (h ^ 1) holds. Levels 1, 2, 4 (e) and 16, 32, 64 (k) in registers, level 8
    one shfl.xor(1) per value. x <- scale * (row H / sqrt(128)) * signs, in place."""
    pm = Float32(1.0) - Float32(2.0) * Float32(h)
    for k in cutlass.range_constexpr(8):  # level 1 (scalar: its pair is one register pair)
        for e in cutlass.range_constexpr(0, 8, 2):
            a = x[8 * k + e]
            b = x[8 * k + e + 1]
            x[8 * k + e] = a + b
            x[8 * k + e + 1] = a - b
    for lv in cutlass.range_constexpr(1, 3):  # levels 2, 4
        hs = 1 << lv
        for k in cutlass.range_constexpr(8):
            for blk in cutlass.range_constexpr(0, 8, 2 * hs):
                for j in cutlass.range_constexpr(0, hs, 2):
                    i0 = 8 * k + blk + j
                    a0, a1 = x[i0], x[i0 + 1]
                    b0, b1 = x[i0 + hs], x[i0 + hs + 1]
                    x[i0], x[i0 + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                    x[i0 + hs], x[i0 + hs + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
    for c in cutlass.range_constexpr(0, 64, 2):  # level 8 across the lane pair
        o0 = cute.arch.shuffle_sync_bfly(x[c], offset=1)
        o1 = cute.arch.shuffle_sync_bfly(x[c + 1], offset=1)
        x[c], x[c + 1] = cute.arch.fma_packed_f32x2((x[c], x[c + 1]), (pm, pm), (o0, o1))
    for lv in cutlass.range_constexpr(3):  # levels 16, 32, 64 over the chunk index k
        m = 1 << lv
        for k in cutlass.range_constexpr(8):
            if const_expr((k & m) == 0):
                for e in cutlass.range_constexpr(0, 8, 2):
                    i0 = 8 * k + e
                    i1 = 8 * (k + m) + e
                    a0, a1 = x[i0], x[i0 + 1]
                    b0, b1 = x[i1], x[i1 + 1]
                    x[i0], x[i0 + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                    x[i1], x[i1 + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
    sp = scale * Float32(INV_SQRT_128)
    sph = sp * pm  # where the column's sign differs between the two lanes: +-sp chosen by h
    for k in cutlass.range_constexpr(8):
        for e in cutlass.range_constexpr(0, 8, 2):
            f = []
            for u in cutlass.range_constexpr(2):
                lo = sg[16 * k + e + u]
                hi = sg[16 * k + 8 + e + u]
                if const_expr(lo == hi):
                    f.append(sp if const_expr(lo > 0) else -sp)
                else:
                    f.append(sph if const_expr(lo > 0) else -sph)
            i0 = 8 * k + e
            x[i0], x[i0 + 1] = cute.arch.mul_packed_f32x2((x[i0], x[i0 + 1]), (f[0], f[1]))


def neg_masks16(sg, colf) -> tuple:
    """For unrotate_chunk8: per register index e, a 16-bit mask over the half-warp lane j of the columns colf(j, e) whose
    sign is -1."""
    return tuple(sum(1 << j for j in range(16) if sg[colf(j, e)] < 0) for e in range(8))


@cute.jit
def chunk8_factors(j, scale: Float32, negm: cutlass.Constexpr, mult: cute.Tensor, pm: cute.Tensor):
    """Per-lane constants of unrotate_chunk8 (once per thread): mult[e] = +-scale / sqrt(128) for this lane's column e,
    pm[s] = +1 / -1 for the lower / upper member of cross-lane level 8 * 2^s."""
    sp = scale * Float32(INV_SQRT_128)
    for e in cutlass.range_constexpr(8):
        bit = (cutlass.Int32(negm[e]) >> j) & cutlass.Int32(1)
        mult[e] = sp - Float32(2.0) * sp * Float32(bit)  # sp or -sp, exactly
    for s in cutlass.range_constexpr(4):
        pm[s] = Float32(1.0) - Float32(2.0) * Float32((j >> s) & cutlass.Int32(1))


@cute.jit
def unrotate_chunk8(x: cute.Tensor, mult: cute.Tensor, pm: cute.Tensor):
    """Half warp, one row: lane j (0..15) holds 8 columns whose bits 0-2 are the register index e and bits 3-6 are j's
    bits 0-3 (its signs come in `mult`). Levels 1, 2, 4 in registers, 8 .. 64 as shfl.xor(1, 2, 4, 8)."""
    for k in cutlass.range_constexpr(0, 8, 2):  # level 1
        a = x[k]
        b = x[k + 1]
        x[k] = a + b
        x[k + 1] = a - b
    for lv in cutlass.range_constexpr(1, 3):  # levels 2, 4 (packed)
        hs = 1 << lv
        for blk in cutlass.range_constexpr(0, 8, 2 * hs):
            for jj in cutlass.range_constexpr(0, hs, 2):
                a0, a1 = x[blk + jj], x[blk + jj + 1]
                b0, b1 = x[blk + jj + hs], x[blk + jj + hs + 1]
                x[blk + jj], x[blk + jj + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                x[blk + jj + hs], x[blk + jj + hs + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
    for s in cutlass.range_constexpr(4):  # levels 8, 16, 32, 64 across the half warp
        for c in cutlass.range_constexpr(0, 8, 2):
            o0 = cute.arch.shuffle_sync_bfly(x[c], offset=1 << s)
            o1 = cute.arch.shuffle_sync_bfly(x[c + 1], offset=1 << s)
            x[c], x[c + 1] = cute.arch.fma_packed_f32x2((x[c], x[c + 1]), (pm[s], pm[s]), (o0, o1))
    for c in cutlass.range_constexpr(0, 8, 2):
        x[c], x[c + 1] = cute.arch.mul_packed_f32x2((x[c], x[c + 1]), (mult[c], mult[c + 1]))

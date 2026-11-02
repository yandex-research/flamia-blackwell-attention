# cov5: the NVFP4 P.V operand V^T (key-contiguous, K-major) on the lp_docmaps packed rows.
#
# kind::mxf4nvf4 takes both operands K-major, so P.V needs V's codes key-contiguous in SMEM with each document's key 0
# on a 16-key scale-group boundary of the tile (the V scale groups count from the document's start). A tensor TMA box
# must start on a 16-byte (32-key) address (PTX ISA 5.5.3.1; an 8-byte start traps with an illegal instruction), and a
# packed document starts at any key. So the quantiser writes the transposed codes into two KEY-PARITY copies (quant.fp4
# transposed_row_packed: a document starting at key ps lives in copy c = ps & 1 at key column k + c, so it starts on a
# byte; (2, h_kv, d, TB) bytes, O(T)), and the forward's load warp
#   1. loads the 128 d rows' 80-byte windows (the tile's 64 bytes from the 16-byte-aligned column below its first
#      byte) into a staging slot by ONE tensor TMA (tma_stage; issued STAGES key tiles ahead -- 128 single-row bulk
#      copies measured 3.4x the forward kernel), then
#   2. realigns it into the V stage by the window's byte shift (realign: 5 16-byte shared loads, 16 funnel shifts and
#      4 16-byte stores per row, the word offset a warp-uniform branch; on a document's last tile the keys past its real
#      ones zeroed -- the aligned layout's zero rows) in the MMA's K-major SW64 layout (byte b of row n at
#      64 n + (16 (b / 16) ^ 16 ((n / 2) % 4)) + b % 16), and
#   3. builds the two SF atoms from the dense per-row SF words (quant.fp4 sf4_rows: every key row its 16 x 16 block's
#      8 d-block bytes; rows ps + 16 g, groups past the real keys 0 -- an all-zero block's SF), or the diagonal
#      payload's per-tile constant word.
# The stores are generic-proxy writes read by the MMA (async proxy): the caller fences and arrives (sfd_atoms.publish).
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass.cute.nvgpu import cpasync
from flash_attn.cute.quant.fp8 import prmt

ROW_WINDOW = 80          # staged bytes per d row (64 + the 16-byte alignment slack)
STAGE_BYTES = 128 * ROW_WINDOW
STAGES = 2               # V staging slots (issued STAGES key tiles ahead; 3 measured +1 % on the forward)


@dsl_user_op
def shf_r(lo, hi, sh, *, loc=None, ip=None) -> Int32:
    """(hi:lo) >> sh (0 <= sh < 32): funnel shift right, clamp."""
    return Int32(llvm.inline_asm(T.i32(), [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip),
                                           Int32(sh).ir_value(loc=loc, ip=ip)],
                                 "shf.r.clamp.b32 $0, $1, $2, $3;", "=r,r,r,r", has_side_effects=False,
                                 is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def tma_stage(atom, mT, row0, col0, sstg, mbar):
    """The 128 d rows' 80-byte windows (rows row0 .., byte column col0, 16-byte aligned) of the 2D (rows, TB) view mT
    into the staging tensor sstg (128, 80) by one tensor TMA on mbar (armed by the caller)."""
    g = cute.local_tile(cute.domain_offset((row0, col0), mT), (128, ROW_WINDOW), (0, 0))
    ts, tg = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(sstg, 0, 2),
                                   cute.group_modes(g, 0, 2))
    cute.copy(atom, tg, ts, tma_bar_ptr=mbar)   # (warp-collective: the TMA copy elects its issuing lane itself)


@cute.jit
def _realign_rows(stg_addr, q: cutlass.Constexpr[int], bs, nvr, dst_addr, masked: cutlass.Constexpr[bool]):
    """realign's body for a window word offset q (compile-time: warp-uniform branch, static register indexing)."""
    lane = cute.arch.lane_idx()
    for j in cutlass.range_constexpr(4):
        n = lane + 32 * j
        w = cute.make_rmem_tensor(cute.make_layout(20), Int32)
        cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, stg_addr + ROW_WINDOW * n, cute.AddressSpace.smem,
                                                         assumed_align=16), cute.make_layout(20)), w)
        o = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        for i in cutlass.range_constexpr(16):
            o[i] = shf_r(w[q + i], w[q + i + 1], bs)
            if const_expr(masked):
                lim = nvr - 8 * i                     # real keys of this word
                o[i] = o[i] & cutlass.select_(lim >= 8, Int32(-1),
                                              cutlass.select_(lim <= 0, Int32(0), (Int32(1) << (4 * lim)) - 1))
        sw = (n >> 1) & 3
        for c in cutlass.range_constexpr(4):
            dst = cute.make_tensor(cute.make_ptr(Int32, dst_addr + 64 * n + 16 * (c ^ sw), cute.AddressSpace.smem,
                                                 assumed_align=16), cute.make_layout(4))
            cute.autovec_copy(cute.make_tensor(o.iterator + 4 * c, cute.make_layout(4)), dst)


@cute.jit
def realign(stg_addr, sh, nvr, dst_addr):
    """Staging slot (waited for) -> the V stage at dst_addr: d row n's 64 bytes from window byte sh, keys >= nvr 0 (only
    a document's last tile masks)."""
    q = sh >> 2
    bs = (sh & 3) * 8
    if nvr >= 128:
        if q == 0:
            _realign_rows(stg_addr, 0, bs, nvr, dst_addr, False)
        elif q == 1:
            _realign_rows(stg_addr, 1, bs, nvr, dst_addr, False)
        elif q == 2:
            _realign_rows(stg_addr, 2, bs, nvr, dst_addr, False)
        else:
            _realign_rows(stg_addr, 3, bs, nvr, dst_addr, False)
    else:
        if q == 0:
            _realign_rows(stg_addr, 0, bs, nvr, dst_addr, True)
        elif q == 1:
            _realign_rows(stg_addr, 1, bs, nvr, dst_addr, True)
        elif q == 2:
            _realign_rows(stg_addr, 2, bs, nvr, dst_addr, True)
        else:
            _realign_rows(stg_addr, 3, bs, nvr, dst_addr, True)


@cute.jit
def store_sf_atoms(src_addr, nvr, dst_addr):
    """The tile's two V SF atoms (B operand: rows = d, K groups = 16 keys) from the dense per-row words staged at smem
    src_addr (key row ps + r at 8 r bytes: word 0 = d blocks 0..3, word 1 = 4..7): atom a, row n, byte kg = the SF of
    key group 4 a + kg, d block n / 16; groups past nvr real keys 0. Lane l stores rows l + 32 j as words 4 l + j."""
    lane = cute.arch.lane_idx()
    src = cute.make_tensor(cute.make_ptr(Int32, src_addr, cute.AddressSpace.smem, assumed_align=8),
                           cute.make_layout(256))
    db = lane >> 4                       # d block of row l (0 / 1); rows l + 32 j: blocks db + 2 j
    for a in cutlass.range_constexpr(2):
        # words of the 4 groups of this atom: g0 / g1 (low / high d-block words), groups past nvr 0
        lo = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        hi = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        for kg in cutlass.range_constexpr(4):
            g = 4 * a + kg
            ok = Int32(16 * g) < nvr
            lo[kg] = cutlass.select_(ok, src[32 * g], Int32(0))
            hi[kg] = cutlass.select_(ok, src[32 * g + 1], Int32(0))
        v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        for j in cutlass.range_constexpr(4):
            # d block b = db + 2 j: word j // 2 (lo for j < 2), byte db + 2 (j % 2)
            bsel = (db + 2 * (j % 2))
            ws = lo if j < 2 else hi
            # byte bsel of each of the 4 group words -> bytes kg of the atom word
            p01 = prmt(ws[0], ws[1], bsel | ((bsel + 4) << 4))
            p23 = prmt(ws[2], ws[3], bsel | ((bsel + 4) << 4))
            v[j] = prmt(p01, p23, Int32(0x5410))
        cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_addr + 512 * a + 16 * lane,
                                                            cute.AddressSpace.smem, assumed_align=16),
                                              cute.make_layout(4)))


@cute.jit
def store_const_sf_atoms(word, dst_addr):
    """Both SF atoms of a tile with every byte of `word` (the diagonal payload's per-tile constant 2^c_n, replicated)."""
    lane = cute.arch.lane_idx()
    v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
    for j in cutlass.range_constexpr(4):
        v[j] = word
    for a in cutlass.range_constexpr(2):
        cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_addr + 512 * a + 16 * lane,
                                                            cute.AddressSpace.smem, assumed_align=16),
                                              cute.make_layout(4)))


@cute.jit
def realign_kt_gmem(src_row0, row_stride, b0, nvr, dst_addr):
    """(the all-4-bit backward's dQ operand) K^T's 128 d rows of one key tile from a key-parity copy in global memory
    (row n at src_row0 + n row_stride, the tile's first byte b0) straight into the K^T stages: d rows 0-63 the first
    stage, 64-127 the second (4096 bytes apart), each row's four 32-key blocks in the kernels' order [0, 2, 1, 3], the
    SW64 swizzle; keys >= nvr zero. Lane l: rows l + 32 j (16-byte-aligned 80-byte windows, five 16-byte loads)."""
    lane = cute.arch.lane_idx()
    a0 = b0 & ~15
    sh = b0 & 15
    q = sh >> 2
    bs = (sh & 3) * 8
    for j in cutlass.range_constexpr(4):
        n = lane + 32 * j
        src = cute.make_tensor(cute.recast_ptr(src_row0 + Int64(n) * Int64(row_stride) + Int64(a0), dtype=Int32)
                               .align(16), cute.make_layout(20))
        w = cute.make_rmem_tensor(cute.make_layout(20), Int32)
        cute.autovec_copy(src, w)
        o = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        for i in cutlass.range_constexpr(16):
            # (q runtime: the window word offset by a select chain over its 4 values)
            lo = cutlass.select_(q == 0, w[i], cutlass.select_(q == 1, w[i + 1], cutlass.select_(q == 2, w[i + 2],
                                                                                                    w[i + 3])))
            hi = cutlass.select_(q == 0, w[i + 1], cutlass.select_(q == 1, w[i + 2], cutlass.select_(q == 2, w[i + 3],
                                                                                                        w[i + 4])))
            lim = nvr - 8 * i
            msk = cutlass.select_(lim >= 8, Int32(-1), cutlass.select_(lim <= 0, Int32(0), (Int32(1) << (4 * lim)) - 1))
            o[i] = shf_r(lo, hi, bs) & msk
        hf = n >> 6
        nn = n & 63
        sw = (nn >> 1) & 3
        for c in cutlass.range_constexpr(4):
            pc = (0, 2, 1, 3)[c]          # natural 32-key block c -> its position in the kernels' order
            dst = cute.make_tensor(cute.make_ptr(Int32, dst_addr + 4096 * hf + 64 * nn + 16 * (pc ^ sw),
                                                 cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4))
            cute.autovec_copy(cute.make_tensor(o.iterator + 4 * c, cute.make_layout(4)), dst)


@cute.jit
def fill_words(word, dst_addr, nbytes: cutlass.Constexpr[int]):
    """nbytes (a multiple of 512) of shared memory at dst_addr filled with `word` (a constant SF word), by the warp."""
    lane = cute.arch.lane_idx()
    v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
    for i in cutlass.range_constexpr(4):
        v[i] = word
    for k in cutlass.range_constexpr(nbytes // 512):
        cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_addr + 512 * k + 16 * lane,
                                                            cute.AddressSpace.smem, assumed_align=16),
                                              cute.make_layout(4)))


@cute.jit
def store_kt_group_sf(tab_head, slot0, nvr, dst_addr):
    """(cov6) K^T's SF atoms of one key tile (the all-4-bit backward's dQ operand, plain NVFP4 1 x 16 along keys) from
    the packed group table (quant.fp4 transposed_sf_groups: (h, G_b, 128) bytes, natural group g of the tile at slot
    slot0 + g, one byte per d row) into both K^T SF stages: stage 0 the atoms as quant.fp4's transposed copy (kernel key
    order [0, 2, 1, 3]: atom 0 the natural groups 0, 1, 4, 5, atom 1 2, 3, 6, 7; row n's word at 16 (n % 32) + 4 (n / 32)),
    stage 1 (1024 bytes on) the d rows 64-127 moved to columns 0, 1 (halves swapped). Groups without a real key (16 g >=
    nvr) are an all-zero group's SF (0). Lane l: d rows 4 l .. 4 l + 3 (one word per group)."""
    lane = cute.arch.lane_idx()
    tab = cute.make_tensor(cute.recast_ptr(tab_head, dtype=Int32), cute.make_layout(1 << 30))
    w = cute.make_rmem_tensor(cute.make_layout(8), Int32)
    for g in cutlass.range_constexpr(8):
        w[g] = cutlass.select_(Int32(16 * g) < nvr, tab[(slot0 + g) * 32 + lane], Int32(0))
    dst = cute.make_tensor(cute.make_ptr(Int32, dst_addr, cute.AddressSpace.smem, assumed_align=16),
                           cute.make_layout(512))
    for a in cutlass.range_constexpr(2):
        gs = (0, 1, 4, 5) if a == 0 else (2, 3, 6, 7)
        # bytes r of the four group words -> the word of d row 4 lane + r (byte kg = group gs[kg])
        p01lo = prmt(w[gs[0]], w[gs[1]], Int32(0x5140))   # [g0 r0, g1 r0, g0 r1, g1 r1]
        p01hi = prmt(w[gs[0]], w[gs[1]], Int32(0x7362))   # [g0 r2, g1 r2, g0 r3, g1 r3]
        p23lo = prmt(w[gs[2]], w[gs[3]], Int32(0x5140))
        p23hi = prmt(w[gs[2]], w[gs[3]], Int32(0x7362))
        rows = (prmt(p01lo, p23lo, Int32(0x5410)), prmt(p01lo, p23lo, Int32(0x7632)),
                prmt(p01hi, p23hi, Int32(0x5410)), prmt(p01hi, p23hi, Int32(0x7632)))
        for r in cutlass.range_constexpr(4):
            n = 4 * lane + r
            dst[128 * a + 4 * (n % 32) + n // 32] = rows[r]
            dst[256 + 128 * a + 4 * (n % 32) + ((n // 32) + 2) % 4] = rows[r]

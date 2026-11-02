# sfd (the SF redesign of the lp_docmaps packed rows): scale-factor atoms built in SMEM by a load warp from the DENSE
# per-row tables the quantisers write (quant.fp8 dense_out) -- no per-aligned-tile table exists on that path.
#
# Token-block E4M3 operands (32 tokens x d, one UE8M0 exponent per block) have per 128-row tile only 4 bytes of
# scale information, the exponents e0..e3 of its 32-row blocks (the dense exponent table: every packed row holds its
# block's byte; a tile at packed row ps with nr real rows reads rows ps + 32 i, i < 4, a block past the document is 0
# -- the aligned layout's all-zero block). The 512 B UMMA scale-factor atom (32 x 4 x 4: row r's 4 K-group bytes at
# word 4 (r % 32) + r / 32) is then
#   token rows (the S operands' A / B):  row r -> rep4(e_{r / 32}): lane l stores [rep4(e0), rep4(e1), rep4(e2), rep4(e3)]
#   d rows (MN-major / P.V B operands):  every row -> e0 | e1 << 8 | e2 << 16 | e3 << 24 (all four words equal)
# one 16-byte shared store per lane. The NVFP4 S operand's SF words are genuinely per row: lane l gathers rows l, l + 32,
# l + 64, l + 96 of the dense words (coalesced) and stores them as atom words 4 l .. 4 l + 3.
# The stores are generic-proxy writes read by tcgen05.cp (async proxy): the writer fences (fence.proxy.async) and the
# warp syncs before the stage's (release) arrive.
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from flash_attn.cute.quant.fp8 import prmt


@cute.jit
def load_bytes(mEv, bh, ps, raw):
    """The 4 block-start bytes of the tile at packed row ps into raw[0..3] (Int32), unpredicated (rows ps .. ps + 96
    are inside the (b h, R) table for any ps < T: R >= T + 144). No arithmetic on them here: their latency is paid at
    combine_bytes."""
    for i in cutlass.range_constexpr(4):
        raw[i] = Int32(mEv[bh, ps + 32 * i])


@cute.jit
def combine_bytes(raw, nr):
    """raw (load_bytes, or a staging slot's bytes) -> the tile word, blocks past nr real rows 0."""
    # w = b0 | b1 << 8 | b2 << 16 | b3 << 24 (two byte permutes), then the blocks past nr cleared by one mask
    w = prmt(prmt(raw[0], raw[1], Int32(0x0040)), prmt(raw[2], raw[3], Int32(0x0040)), Int32(0x5410))
    nb = cutlass.min(cutlass.max(nr + 31, Int32(0)) // 32, Int32(4))   # real blocks
    return w & cutlass.select_(nb >= 4, Int32(-1), (Int32(1) << (8 * nb)) - 1)


@cute.jit
def gather_word(b, lane0: cutlass.Constexpr[int], nr):
    """The tile word from per-lane block bytes (lane lane0 + i holds block i's byte, as loaded): b0 | b1 << 8 |
    b2 << 16 | b3 << 24 (& 0xFF: no sign extension of exponents >= 128), blocks past nr real rows 0. Warp-collective
    (every lane gets the word)."""
    w = Int32(0)
    for i in cutlass.range_constexpr(4):
        x = cute.arch.shuffle_sync(b, lane0 + i) & 0xFF
        w = w | (cutlass.select_(Int32(32 * i) < nr, x, Int32(0)) << (8 * i))
    return w


@cute.jit
def load_sf4(mSF4, bh, ps, nk: cutlass.Constexpr[int], raw):
    """The tile's NVFP4 SF words (lane l: rows l + 32 j, atom k at raw[4 k + j]), unpredicated (rows < ps + 128 <=
    T + 127 < R), masked at store_sf4_atoms."""
    lane = cute.arch.lane_idx()
    for k in cutlass.range_constexpr(nk):
        for j in cutlass.range_constexpr(4):
            raw[4 * k + j] = mSF4[bh, (ps + lane + 32 * j) * nk + k]


@cute.jit
def store_sf4_atoms(raw, nr, nk: cutlass.Constexpr[int], dst_byte_addr):
    """The NVFP4 SF atom(s) (nk = d / 64 of them, 512 B apart) from load_sf4's words, 0 past nr real rows."""
    lane = cute.arch.lane_idx()
    for k in cutlass.range_constexpr(nk):
        v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        for j in cutlass.range_constexpr(4):
            v[j] = cutlass.select_(lane + 32 * j < nr, raw[4 * k + j], Int32(0))
        cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_byte_addr + 512 * k + 16 * lane,
                                                            cute.AddressSpace.smem, assumed_align=16),
                                              cute.make_layout(4)))


@cute.jit
def store_tok_atom(w, dst_byte_addr):
    """The token-rows atom of word w at dst_byte_addr (16-byte aligned smem address, an Int32 / int)."""
    lane = cute.arch.lane_idx()
    v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
    for i in cutlass.range_constexpr(4):
        v[i] = prmt(w, Int32(0), Int32(0x1111 * i))   # byte i replicated
    cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_byte_addr + 16 * lane, cute.AddressSpace.smem,
                                                        assumed_align=16), cute.make_layout(4)))


@cute.jit
def store_dim_atom(w, dst_byte_addr):
    """The d-rows atom of word w (every word w)."""
    lane = cute.arch.lane_idx()
    v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
    for i in cutlass.range_constexpr(4):
        v[i] = w
    cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_byte_addr + 16 * lane, cute.AddressSpace.smem,
                                                        assumed_align=16), cute.make_layout(4)))


@cute.jit
def publish(mbar, tx_bytes):
    """After a stage's generic SF writes (and its TMA issues): proxy fence, warp sync, the elected arrive with the
    stage's tx bytes (the TMA may have landed before the expect-tx: the phase still needs this arrival)."""
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, tx_bytes)


@cute.jit
def read_sf4_from_smem(src_ptr, ps, nr, nk: cutlass.Constexpr[int], v):
    """The NVFP4 SF atom words of a tile from its staged dense words (src_ptr: smem bytes of rows (ps & ~3) .., nk
    words a row) into v (4 nk Int32 registers; lane l: rows l + 32 j), 0 past nr real rows -- issued before the stage's
    empty wait."""
    lane = cute.arch.lane_idx()
    src = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=Int32), cute.make_layout(132 * nk))
    off = ps & 3   # (ps >= 0)
    for k in cutlass.range_constexpr(nk):
        for j in cutlass.range_constexpr(4):
            r = lane + 32 * j
            v[4 * k + j] = cutlass.select_(r < nr, src[(off + r) * nk + k], Int32(0))


@cute.jit
def store_sf4_regs(v, nk: cutlass.Constexpr[int], dst_byte_addr):
    """read_sf4_from_smem's words as the atom(s) at dst_byte_addr (one 16-byte store per lane and atom)."""
    lane = cute.arch.lane_idx()
    for k in cutlass.range_constexpr(nk):
        cute.autovec_copy(cute.make_tensor(v.iterator + 4 * k, cute.make_layout(4)),
                          cute.make_tensor(cute.make_ptr(Int32, dst_byte_addr + 512 * k + 16 * lane,
                                                         cute.AddressSpace.smem, assumed_align=16),
                                           cute.make_layout(4)))

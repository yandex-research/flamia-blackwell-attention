# Key-tile geometry of the low-precision backwards (masks / varlen): which query tiles a key tile meets, and in which
# coordinates its mask, score_mod and mask_mod run.
#
# Two layouts:
#   batched   (b, s, h, d): the upstream BlockInfo / SeqlenInfoQK rules, unchanged (causal / window / seqused_k).
#   aligned-packed varlen ("docs"): b = 1, every document starts on a multiple of `align` tokens (128, or 256 for the
#             2-CTA key pair), so every token-spanning scale block of the quantisers (<= 128 tokens, 128-aligned) and every
#             key tile of the kernel lies inside one document. Two int32 tables carry the documents (built on the GPU by
#             lp_varlen.py, no host sync):
#               mDocTab (6, n_docs + 1): padded start of Q, padded start of K, true length of Q, true length of K,
#                   the caller's start of Q, of K; the
#                   last column is an empty sentinel document (length 0) for the key tiles past the last document.
#               mNDoc (n_key_tiles,): the document of each key tile.
#             The document is the "batch" of SeqlenInfoQK (offsets = padded starts, seqused = true lengths), so BlockInfo
#             and AttentionMask run in document-local coordinates exactly as upstream varlen does; the kernel addresses
#             tensors with the global tile index (local + the document's first tile).
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo


@dataclass(frozen=True)
class LPTileInfo:
    """get(batch_idx, n_block) -> (seqlen, m_block_min, m_block_max, m_block_offset, n_block_local).
    m_block_min / m_block_max are global query-tile indices; seqlen and n_block_local are in the coordinates of the
    mask (the document's under docs). A frozen dataclass (as BlockInfo), so the DSL can carry it through loops."""

    block_info: BlockInfo
    seqlen_q_static: Int32
    seqlen_k_static: Int32
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    mSeqUsedK: Optional[cute.Tensor] = None
    mDocTab: Optional[cute.Tensor] = None
    mNDoc: Optional[cute.Tensor] = None
    docs: cutlass.Constexpr[bool] = False

    @cute.jit
    def seqlen(self, batch_idx: Int32, n_block: Int32):
        if const_expr(self.docs):
            # the caller's offsets and the true lengths: what the mask needs (lengths) and what the mods see
            doc = self.mNDoc[n_block]
            return SeqlenInfoQK.create(
                doc, self.seqlen_q_static, self.seqlen_k_static,
                mCuSeqlensQ=self.mDocTab[4, None], mCuSeqlensK=self.mDocTab[5, None],
                mSeqUsedQ=self.mDocTab[2, None], mSeqUsedK=self.mDocTab[3, None],
                tile_m=self.tile_m, tile_n=self.tile_n,
            )
        else:
            return SeqlenInfoQK.create(
                batch_idx, self.seqlen_q_static, self.seqlen_k_static,
                mCuSeqlensQ=None, mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=self.mSeqUsedK,
                tile_m=self.tile_m, tile_n=self.tile_n,
            )

    @cute.jit
    def get(self, batch_idx: Int32, n_block: Int32):
        seqlen = self.seqlen(batch_idx, n_block)
        m_off = Int32(0)
        n_loc = n_block
        if const_expr(self.docs):
            doc = self.mNDoc[n_block]
            m_off = self.mDocTab[0, doc] // self.tile_m  # the aligned (repacked) starts
            n_loc = n_block - self.mDocTab[1, doc] // self.tile_n
        m_lo, m_hi = self.block_info.get_m_block_min_max(seqlen, n_loc)
        return seqlen, m_lo + m_off, m_hi + m_off, m_off, n_loc

    @cute.jit
    def n_block_min_local(self, seqlen, m_block_local: Int32):
        """First key tile (mask coordinates) that meets query tile m_block_local: the deterministic dQ reduce waits for
        (n_block_local - this) earlier contributions instead of n_block (the window / document start)."""
        n_lo, _ = self.block_info.get_n_block_min_max(seqlen, m_block_local)
        return n_lo

    @cute.jit
    def batch_for_mods(self, batch_idx: Int32, n_block: Int32):
        """The `batch` a mask_mod / score_mod sees: the document under docs (upstream varlen semantics)."""
        if const_expr(self.docs):
            return self.mNDoc[n_block]
        else:
            return batch_idx


@dataclass(frozen=True)
class LPPairTileInfo:
    """The 2-CTA key pair over documents packed at 128 (flash_bwd_sm100_2cta8 doc_t128): the pair is the global key tiles
    (2 p, 2 p + 1) of 128 keys, and the two halves may lie in different documents (a document starts on any multiple of
    128). Both CTAs iterate the HULL of the two halves' query ranges (the 2-CTA MMAs share the Q tile); each CTA masks
    in the coordinates of ITS tile's document, and a query tile of the other document is masked whole (an m tile never
    straddles a document: query starts are 128-aligned too). block_info is the 128-key-tile BlockInfo.

    get(batch_idx, n_pair) -> (seqlen of the own document, hull m_min, hull m_max, own m offset, own local key tile)."""

    block_info: BlockInfo
    seqlen_q_static: Int32
    seqlen_k_static: Int32
    tile_m: cutlass.Constexpr[int]
    mDocTab: cute.Tensor
    mNDoc: cute.Tensor

    @cute.jit
    def _doc(self, t: Int32):
        # a tile past the 128-tile map (the odd last pair's upper half) is the sentinel's (the empty last column)
        n_t = cute.size(self.mNDoc.shape[0])
        sent = cute.size(self.mDocTab.shape[1]) - 1
        doc = Int32(sent)
        if t < n_t:
            doc = self.mNDoc[t]
        return doc

    @cute.jit
    def _seqlen(self, doc: Int32):
        return SeqlenInfoQK.create(
            doc, self.seqlen_q_static, self.seqlen_k_static,
            mCuSeqlensQ=self.mDocTab[4, None], mCuSeqlensK=self.mDocTab[5, None],
            mSeqUsedQ=self.mDocTab[2, None], mSeqUsedK=self.mDocTab[3, None],
            tile_m=self.tile_m, tile_n=128,
        )

    @cute.jit
    def _range(self, t: Int32):
        """(doc, global m_min, global m_max (m_min when empty), m offset, local key tile) of key tile t."""
        doc = self._doc(t)
        seqlen = self._seqlen(doc)
        m_off = self.mDocTab[0, doc] // self.tile_m
        n_loc = t - self.mDocTab[1, doc] // 128
        lo, hi = self.block_info.get_m_block_min_max(seqlen, n_loc)
        lo = cutlass.max(lo, 0)
        if seqlen.seqlen_k <= 0:
            hi = lo
        if hi < lo:
            hi = lo
        return doc, lo + m_off, hi + m_off, m_off, n_loc

    @cute.jit
    def get(self, batch_idx: Int32, n_pair: Int32):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        d0, lo0, hi0, off0, nl0 = self._range(2 * n_pair)
        d1, lo1, hi1, off1, nl1 = self._range(2 * n_pair + 1)
        e0 = hi0 > lo0
        e1 = hi1 > lo1
        lo = Int32(0)
        hi = Int32(0)
        if e0 and e1:
            lo = cutlass.min(lo0, lo1)
            hi = cutlass.max(hi0, hi1)
        elif e0:
            lo = lo0
            hi = hi0
        elif e1:
            lo = lo1
            hi = hi1
        doc = d0
        m_off = off0
        n_loc = nl0
        if rank == 1:
            doc = d1
            m_off = off1
            n_loc = nl1
        return self._seqlen(doc), lo, hi, m_off, n_loc

    @cute.jit
    def batch_for_mods(self, batch_idx: Int32, n_pair: Int32):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        return self._doc(2 * n_pair + rank)

    @cute.jit
    def dq_lock_prep(self, n_pair: Int32):
        """The per-work-tile facts of dq_lock (two document lookups, hoisted out of the m loop: the lock is on the dQ
        reduce's critical chain)."""
        d0, lo0, hi0, off0, nl0 = self._range(2 * n_pair)
        d1, lo1, hi1, off1, nl1 = self._range(2 * n_pair + 1)
        nq1 = cute.ceil_div(self.mDocTab[2, d1], self.tile_m)
        return (self._seqlen(d0), self._seqlen(d1), d0 == d1, lo0, hi0, lo1, hi1, off0, off1, nq1,
                self.mDocTab[1, d0] // 128, self.mDocTab[1, d1] // 128)

    @cute.jit
    def dq_lock(self, prep, n_pair: Int32, m_block: Int32, spt: cutlass.Constexpr[bool]):
        """(active, lock value) of this pair at query tile m_block for the deterministic dQ reduce. The contributors of
        m_block are the pairs holding a key tile of m_block's document that meets it: a contiguous run of pairs. A pair
        is inactive at a hull tile none of whose halves (in m_block's document) meets it: its dQ part is exactly zero
        and it neither waits nor counts. Lock: the pair's rank in the run (ascending, descending under spt).
        prep = dq_lock_prep(n_pair): arithmetic only here."""
        sq0, sq1, same, lo0, hi0, lo1, hi1, off0, off1, nq1, t00, t01 = prep
        in1 = m_block >= off1 and m_block < off1 + nq1
        active = m_block >= lo0 and m_block < hi0
        t0 = t00
        n_lo, n_hi = self.block_info.get_n_block_min_max(sq0, m_block - off0)
        if in1:
            active = m_block >= lo1 and m_block < hi1
            if same:
                active = active or (m_block >= lo0 and m_block < hi0)
            t0 = t01
            n_lo, n_hi = self.block_info.get_n_block_min_max(sq1, m_block - off1)
        lock = Int32(0)
        if const_expr(spt):
            lock = (t0 + n_hi - 1) // 2 - n_pair
        else:
            lock = n_pair - (t0 + n_lo) // 2
        return active, lock

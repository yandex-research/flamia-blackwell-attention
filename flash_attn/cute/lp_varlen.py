# Aligned-packed varlen for the low-precision kernels.
#
# The user passes upstream's packed layout: q (total_q, h, d), k / v (total_k, h_k, d), cu_seqlens_q / _k (b + 1)
# int32 on the GPU, arbitrary (unaligned) document starts. The low-precision kernels need every token-spanning scale
# block (NVFP4 SF atoms of 128 rows, 32-token E4M3 blocks, 32 x 32 dS blocks, 16-key V groups) and every key tile to
# lie inside one document, so the documents are repacked on the GPU with each start rounded up to `align` tokens
# (128; 256 when the 2-CTA key pair must not straddle a document), zero rows in between, and unpacked on the way out.
# No host sync: the padded length is a host-side upper bound (total + b (align - 1), rounded up to align), and every
# index map is built with GPU ops. The repacked tensors are one batch (b = 1, s = the padded length).
#
# The kernels see the documents through two int32 tables (lp_bwd_tiles.LPTileInfo):
#   doc_tab (6, b + 1): padded start of Q, padded start of K, true length of Q, true length of K, the caller's start
#       of Q, of K (the mods' seqlen_info, upstream varlen semantics); column b is an
#       empty sentinel document (start = end of the last document, length 0) for the tail tiles of the allocation.
#   tile_doc(tile) (n_tiles,): the document of each key tile of `tile` tokens (aligned path: a tile wholly past its
#       document's true length is the sentinel's, so it is never scheduled).
#
# The aligned fast path (LPDocs.aligned_from, FA3's convention): the caller's packed tensors already hold every document at
# a start that is a multiple of 128 (256 for the 2-CTA backward), cu_seqlens = the starts, seqused = the true lengths.
# Nothing is copied: the tables are built straight from the caller's cu_seqlens / seqused (padded start = caller's start
# = cu, true length = seqused, the allocation s = total) by ONE prep kernel (_aligned_prep), which also writes the
# quantisers' padding mask row_end_q / row_end_k ((1, total / 128): the end of the real rows of the document that holds
# each 128-token tile) and a validity flag that torch._assert_async checks on the GPU (starts % align == 0,
# 0 <= seqused <= slot length, the last slot inside the allocation): no host read anywhere.
#
# The padding rows are a CONTRACT (owner decision): rows past seqused inside a document's slot must be zero, or at least
# finite and no larger than the real values; nothing validates them. ALIGNED_ROW_PREDICATE (off by default, an A/B
# option) makes the quantisers read them as zero rows instead (quant.fp8.quantize_fp8_blocks / quant.fp4.
# quantize_fp4_rows row_end: an in-kernel row predicate, no extra pass), which also tolerates NaN / inf padding.
ALIGNED_ROW_PREDICATE = False
from dataclasses import dataclass, field
from typing import Optional

import torch


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def doc_key_len_of(max_seqlen_k: Optional[int], total_k: int) -> int:
    """The key slot bound of document-major per-key tables (the 4-bit P.V routes' forward bias / weight): max_seqlen_k
    rounded up to 128, at most the allocation; the allocation when max_seqlen_k is not given."""
    if max_seqlen_k is None:
        return int(total_k)
    return min(_round_up(int(max_seqlen_k), 128), int(total_k))


@dataclass
class LPDocs:
    cu_q: torch.Tensor          # (b + 1,) int32, the caller's
    cu_k: torch.Tensor
    cu_pad_q: torch.Tensor      # (b + 1,) int32, aligned starts (+ the end)
    cu_pad_k: torch.Tensor
    len_q: torch.Tensor         # (b,) int32
    len_k: torch.Tensor
    dst_q: torch.Tensor         # (total_q,) int64: packed row -> aligned row
    dst_k: torch.Tensor
    s_q: int                    # padded (allocated) lengths, multiples of align
    s_k: int
    align: int
    b: int
    max_seqlen_q: Optional[int] = None
    max_seqlen_k: Optional[int] = None
    doc_tab: torch.Tensor = field(default=None)
    keep_k: Optional[torch.Tensor] = None  # (total_k,) bool under seqused_k: keys that exist
    _tile_doc: dict = field(default_factory=dict)
    # the aligned fast path (LPDocs.aligned): the caller's tensors are the kernels' tensors (no dst maps, no repacking);
    # row_end_q / row_end_k (1, s / 128) int32: the quantisers' padding mask (quant.fp8 / quant.fp4 row_end)
    aligned: bool = False
    row_end_q: Optional[torch.Tensor] = None
    row_end_k: Optional[torch.Tensor] = None

    @staticmethod
    def aligned_from(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, seqused_q: torch.Tensor,
                     seqused_k: torch.Tensor, total_q: int, total_k: int, align: int = 128,
                     max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
                     row_predicate: Optional[bool] = None, require_keys: int = 0) -> "LPDocs":
        """The aligned fast path's documents, built from the caller's own tables (FA3 convention: document b starts at
        cu_seqlens[b], a multiple of `align`, and holds seqused[b] <= cu_seqlens[b + 1] - cu_seqlens[b] real rows; the
        rows after them up to the next start are padding). One prep launch + one device-side assert; no host read.
        require_keys (the 4-bit P.V routes, whose forward needs a key tile under every query tile): 1 = a document with
        a query has a key (seqused_k > 0 where seqused_q > 0), 2 = seqused_q <= seqused_k (causal, bottom-right: no query
        without a key); both also seqused_k <= doc_key_len(self) (their document-major per-key tables) -- checked by the
        same device-side assert."""
        for nm, t, n in (("cu_seqlens_q", cu_seqlens_q, None), ("cu_seqlens_k", cu_seqlens_k, None),
                         ("seqused_q", seqused_q, -1), ("seqused_k", seqused_k, -1)):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.dim() != 1 or not t.is_cuda:
                raise ValueError(f"{nm} must be a 1-D int32 CUDA tensor")
        b = cu_seqlens_q.shape[0] - 1
        if cu_seqlens_k.shape[0] != b + 1 or seqused_q.shape[0] != b or seqused_k.shape[0] != b or b < 1:
            raise ValueError(f"aligned varlen: cu_seqlens_q / cu_seqlens_k must be (b + 1,) and seqused_q / seqused_k "
                             f"(b,), b >= 1; got {tuple(cu_seqlens_q.shape)}, {tuple(cu_seqlens_k.shape)}, "
                             f"{tuple(seqused_q.shape)}, {tuple(seqused_k.shape)}")
        if total_q % align or total_k % align:
            raise ValueError(f"aligned varlen: total_q {total_q} and total_k {total_k} must be multiples of {align}")
        dev = cu_seqlens_q.device
        cu_q, cu_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
        used_q, used_k = seqused_q.contiguous(), seqused_k.contiguous()
        tiles_q, tiles_k = total_q // 128, total_k // 128
        tab = torch.empty(6, b + 1, dtype=torch.int32, device=dev)
        ndoc_k = torch.empty(tiles_k, dtype=torch.int32, device=dev)
        ndoc_q = torch.empty(tiles_q, dtype=torch.int32, device=dev)
        rend = torch.empty(tiles_q + tiles_k, dtype=torch.int32, device=dev)
        q_valid = torch.empty(total_q, dtype=torch.bool, device=dev)
        ok = torch.empty(1, dtype=torch.int32, device=dev)
        n_prog = max(triton.cdiv(total_q, _PREP_ROWS), triton.cdiv(tiles_k, _PREP_COLS), triton.cdiv(b + 1, _PREP_COLS))
        steps = max(1, int(b).bit_length() + 1)
        _aligned_prep[(n_prog,)](cu_q, cu_k, used_q, used_k, tab, ndoc_k, ndoc_q, rend, q_valid, ok, b, tiles_q, tiles_k,
                                 total_q, total_k, align, steps, ROWS=_PREP_ROWS, COLS=_PREP_COLS,
                                 **({"REQ_KEYS": int(require_keys), "max_k": doc_key_len_of(max_seqlen_k, total_k)}
                                    if require_keys else {}))
        # starts % align == 0, 0 <= seqused <= slot length, the slots inside the allocation: checked on the GPU (a
        # device-side assert; no host read). The kernels run after it on the stream.
        torch._assert_async(ok, "aligned varlen: cu_seqlens starts must be multiples of the alignment (128; 256 with "
                                "the 2-CTA backward), 0 <= seqused <= cu_seqlens[i + 1] - cu_seqlens[i], "
                                "cu_seqlens[0] >= 0 and cu_seqlens[b] <= total" if not require_keys else
                                "aligned varlen: starts % align, 0 <= seqused <= slot, the 4-bit P.V routes: a key "
                                "for every document with a query" + (", seqused_q <= seqused_k (causal)"
                                                                     if require_keys == 2 else ""))
        r = LPDocs(cu_q, cu_k, cu_q, cu_k, used_q, used_k, None, None, total_q, total_k, align, b, max_seqlen_q,
                   max_seqlen_k)
        r.doc_tab = tab
        r._tile_doc[128] = ndoc_k
        r.tile_doc_q = ndoc_q   # the query-side twin: the document of each 128-row Q tile (the sentinel past its rows)
        r._q_valid = q_valid
        r.aligned = True
        # the dO quantiser's +inf LSE on padding rows (fp4_bwd_autograd.tb_bwd_glue lse_row_end): always
        r.row_end_tab_q = rend[:tiles_q].view(1, tiles_q)
        r.row_end_tab_k = rend[tiles_q:].view(1, tiles_k)
        if ALIGNED_ROW_PREDICATE if row_predicate is None else row_predicate:
            # the quantisers' in-kernel row predicate (the padding contract does not need it: off by default)
            r.row_end_q, r.row_end_k = r.row_end_tab_q, r.row_end_tab_k
        return r

    @staticmethod
    def build(cu_seqlens_q: torch.Tensor, cu_seqlens_k: Optional[torch.Tensor], total_q: int, total_k: int,
              align: int = 128, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
              seqused_k: Optional[torch.Tensor] = None) -> "LPDocs":
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        for nm, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
            if cu.dtype != torch.int32 or cu.dim() != 1 or not cu.is_cuda:
                raise ValueError(f"{nm} must be a 1-D int32 CUDA tensor (b + 1), as upstream")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same number of documents")
        b = cu_seqlens_q.shape[0] - 1
        dev = cu_seqlens_q.device

        def one(cu, total):
            ln = (cu[1:] - cu[:-1]).to(torch.int32)
            pad = (ln + (align - 1)) // align * align
            cu_pad = torch.zeros(b + 1, dtype=torch.int32, device=dev)
            cu_pad[1:] = torch.cumsum(pad, 0, dtype=torch.int32)
            s_alloc = _round_up(total + b * (align - 1), align) if b > 0 else align
            s_alloc = max(s_alloc, align)
            tok = torch.arange(total, device=dev, dtype=torch.int32)
            doc = torch.searchsorted(cu[1:].contiguous(), tok, right=True)
            dst = (cu_pad[doc] + (tok - cu[doc])).long()
            return ln, cu_pad, dst, s_alloc, tok, doc

        len_q, cu_pad_q, dst_q, s_q, _, _ = one(cu_seqlens_q, total_q)
        len_k, cu_pad_k, dst_k, s_k, tok_k, doc_k = one(cu_seqlens_k, total_k)
        keep_k = None
        if seqused_k is not None:
            # upstream seqused_k: only the first seqused_k[b] keys of document b exist. The others are zeroed in the
            # repacked K / V (so no scale block ever sees their values) and get zero dK / dV.
            if seqused_k.dtype != torch.int32 or seqused_k.shape != (b,):
                raise ValueError("seqused_k must be (b,) int32, as upstream")
            len_k = torch.minimum(len_k, seqused_k)
            keep_k = (tok_k - cu_seqlens_k[doc_k]) < len_k[doc_k]
        tab = torch.zeros(6, b + 1, dtype=torch.int32, device=dev)
        tab[0] = cu_pad_q
        tab[1] = cu_pad_k
        tab[2, :b] = len_q
        tab[3, :b] = len_k
        tab[4] = cu_seqlens_q  # the caller's offsets: what a score_mod / mask_mod sees as seqlen_info.offset_q / _k
        tab[5] = cu_seqlens_k
        r = LPDocs(cu_seqlens_q, cu_seqlens_k, cu_pad_q, cu_pad_k, len_q, len_k, dst_q, dst_k, s_q, s_k, align, b,
                   max_seqlen_q, max_seqlen_k)
        r.keep_k = keep_k
        r.doc_tab = tab
        return r

    def key_len(self) -> int:
        """doc_key_len_of for these documents (the 4-bit P.V routes' document-major per-key tables)."""
        return doc_key_len_of(self.max_seqlen_k, self.s_k)

    def tile_doc(self, tile: int = 128) -> torch.Tensor:
        """(s_k / tile,) int32: the document of each key tile; b (the sentinel) past the last document."""
        if tile not in self._tile_doc:  # (aligned: the prep kernel built the 128-token map; others from the starts)
            assert self.align % tile == 0 or tile % self.align == 0
            starts = torch.arange(0, self.s_k, tile, device=self.cu_pad_k.device, dtype=torch.int32)
            d = torch.searchsorted(self.cu_pad_k[1:].contiguous(), starts, right=True).to(torch.int32)
            self._tile_doc[tile] = d.contiguous()
        return self._tile_doc[tile]

    # ---- repacking (all GPU)
    def pack_q(self, x: torch.Tensor) -> torch.Tensor:
        """(total_q, ...) -> (1, s_q, ...) with zero rows between documents."""
        out = x.new_zeros((self.s_q,) + tuple(x.shape[1:]))
        out.index_copy_(0, self.dst_q, x)
        return out.unsqueeze(0)

    def pack_k(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((self.s_k,) + tuple(x.shape[1:]))
        out.index_copy_(0, self.dst_k, x)
        return out.unsqueeze(0)

    def unpack_q(self, x: torch.Tensor) -> torch.Tensor:
        """(1, s_q, ...) -> (total_q, ...)."""
        return x[0].index_select(0, self.dst_q)

    def unpack_k(self, x: torch.Tensor) -> torch.Tensor:
        return x[0].index_select(0, self.dst_k)

    def q_row_valid(self) -> torch.Tensor:
        """(s_q,) bool: rows that hold a real query (not padding). The aligned path's comes from its prep kernel."""
        if getattr(self, "_q_valid", None) is None:
            m = torch.zeros(self.s_q, dtype=torch.bool, device=self.dst_q.device)
            self._q_valid = m.index_fill_(0, self.dst_q, True)  # a device fill: no host scalar copy (graph-safe)
        return self._q_valid

    def lse_pad_rows_inf(self, lse: torch.Tensor) -> torch.Tensor:
        """(1, h, s_q) LSE with +inf on the padding rows: P = exp2(S - inf) = 0 there exactly (upstream's preprocess
        does the same for rows past seqlen_q)."""
        valid = self.q_row_valid()
        return lse.masked_fill(~valid.view(1, 1, -1), float("inf"))

    def seqused(self):
        return self.len_q, self.len_k

    def doc_quant_args(self, side, h_kv: int) -> dict:
        """quant.fp8.quantize_fp8_per_tensor(documents=...): the per-(document, kv head) scale tables of the aligned
        path (one scale per document and kv head of every operand: the FA3 / Meta granularity). side: per tensor 'q'
        (Q, dO: the query tiles) or 'k' (K, V). Rows n_rows = b + 1: the documents and the sentinel (whole padding
        tiles; its scale is never read by a computed tile)."""
        assert self.aligned, "per-(document, kv head) scales: the aligned varlen path"
        return dict(side=tuple(side), row_end_q=self.row_end_tab_q, row_end_k=self.row_end_tab_k,
                    tile_doc_q=self.tile_doc_q, tile_doc_k=self._tile_doc[128], n_rows=self.b + 1, h_kv=int(h_kv))


# ---------------------------------------------------------------- repacking kernels (one read, one write per row)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_PREP_ROWS = 1024   # query rows per program (q_valid; their 8 tiles' row_end_q)
_PREP_COLS = 1024   # key tiles / table columns per program


@triton.jit
def _upper_bound(cu_ptr, x, b, steps):
    """Per element of x: #{i in 1..b : cu[i] <= x} (torch.searchsorted(cu[1:], x, right=True)): the document of
    position x (b: past the last document)."""
    lo = tl.zeros_like(x)
    hi = tl.zeros_like(x) + b
    for _ in range(steps):
        go = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(cu_ptr + mid + 1, mask=go, other=0)
        right = go & (v <= x)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(go & (v > x), mid, hi)
    return lo


@triton.jit
def _upper_bound_wide(cu_ptr, x, b, steps, B: tl.constexpr):
    """_upper_bound by B-ary steps (cu nondecreasing; steps: B^(steps - 1) > b): log_B(b) dependent loads, not
    log_2(b). The answer stays in [lo, hi]; a step tests B evenly spaced starts of it."""
    lo = tl.zeros_like(x)
    hi = tl.zeros_like(x) + b
    jj = 1 + tl.arange(0, B)
    for _ in range(steps):
        s = tl.maximum((hi - lo + B - 1) // B, 1)
        idx = lo[:, None] + jj[None, :] * s[:, None]
        m = idx <= hi[:, None]
        v = tl.load(cu_ptr + idx, mask=m, other=0)
        cnt = tl.sum((m & (v <= x[:, None])).to(tl.int32), axis=1)
        lo = lo + cnt * s
        hi = tl.minimum(lo + s - 1, hi)
    return lo


@triton.jit
def _aligned_prep(cu_q, cu_k, used_q, used_k, tab, ndoc_k, ndoc_q, rend, q_valid, ok, b, tiles_q, tiles_k, total_q,
                  total_k, align, steps, ROWS: tl.constexpr, COLS: tl.constexpr, REQ_KEYS: tl.constexpr = 0, max_k=0):
    """The aligned varlen tables in one launch (LPDocs.aligned_from): doc_tab (6, b + 1), the key-tile -> document map,
    the quantisers' row_end of every 128-token Q / K tile (rend: tiles_q then tiles_k), q_valid (total_q,) and the
    validity flag ok (program 0)."""
    pid = tl.program_id(0)
    # ---- doc_tab columns: starts (padded = the caller's), true lengths (0 for the sentinel column b), caller starts
    c = pid * COLS + tl.arange(0, COLS)
    mc = c <= b
    sq = tl.load(cu_q + c, mask=mc, other=0)
    sk = tl.load(cu_k + c, mask=mc, other=0)
    lq = tl.load(used_q + c, mask=mc & (c < b), other=0)
    lk = tl.load(used_k + c, mask=mc & (c < b), other=0)
    n = b + 1
    tl.store(tab + c, sq, mask=mc)
    tl.store(tab + n + c, sk, mask=mc)
    tl.store(tab + 2 * n + c, lq, mask=mc)
    tl.store(tab + 3 * n + c, lk, mask=mc)
    tl.store(tab + 4 * n + c, sq, mask=mc)
    tl.store(tab + 5 * n + c, sk, mask=mc)
    # ---- key tiles: document and row end
    kt = pid * COLS + tl.arange(0, COLS)
    mk = kt < tiles_k
    dk = _upper_bound(cu_k, kt * 128, b, steps)
    ek = tl.load(cu_k + dk, mask=mk, other=0) + tl.load(used_k + dk, mask=mk & (dk < b), other=0)
    ek = tl.where(dk < b, ek, 0)
    # a key tile wholly past its document's true length goes to the sentinel (an empty document): the backward
    # never schedules it (its query range is empty). Left in its document, a causal tile past seqlen_k still met a
    # query range (the range is not capped by seqlen_k): wasted work, and the deterministic dQ order deadlocked on it.
    dk = tl.where(kt * 128 < ek, dk, b)
    tl.store(ndoc_k + kt, dk, mask=mk)
    tl.store(rend + tiles_q + kt, ek, mask=mk)
    # ---- query rows: q_valid, and the row end of each of their tiles
    rows = pid * ROWS + tl.arange(0, ROWS)
    mr = rows < total_q
    dq = _upper_bound(cu_q, (rows // 128) * 128, b, steps)
    eq = tl.load(cu_q + dq, mask=mr, other=0) + tl.load(used_q + dq, mask=mr & (dq < b), other=0)
    eq = tl.where(dq < b, eq, 0)
    tl.store(q_valid + rows, rows < eq, mask=mr)
    tl.store(rend + rows // 128, eq, mask=mr & (rows % 128 == 0))
    # the document of each query tile (the sentinel b for a tile wholly past its document's true length, as ndoc_k)
    tl.store(ndoc_q + rows // 128, tl.where(rows < eq, dq, b), mask=mr & (rows % 128 == 0))
    # ---- validity (program 0): every start % align == 0, 0 <= seqused <= slot, cu[0] >= 0, cu[b] <= total
    if pid == 0:
        good = pid * 0 + 1
        for c0 in range(0, b + 1, COLS):
            i = c0 + tl.arange(0, COLS)
            m = i <= b
            aq = tl.load(cu_q + i, mask=m, other=0)
            ak = tl.load(cu_k + i, mask=m, other=0)
            m1 = i < b
            nq = tl.load(cu_q + i + 1, mask=m1, other=0)
            nk = tl.load(cu_k + i + 1, mask=m1, other=0)
            uq = tl.load(used_q + i, mask=m1, other=0)
            uk = tl.load(used_k + i, mask=m1, other=0)
            bad = m & ((aq % align != 0) | (ak % align != 0) | (aq < 0) | (ak < 0) | (aq > total_q) | (ak > total_k))
            bad = bad | (m1 & ((uq < 0) | (uk < 0) | (uq > nq - aq) | (uk > nk - ak)))
            if REQ_KEYS == 1:
                bad = bad | (m1 & (uq > 0) & (uk == 0))
            if REQ_KEYS == 2:
                bad = bad | (m1 & (uq > uk))
            if REQ_KEYS > 0:
                bad = bad | (m1 & (uk > max_k))
            good = good & (tl.max(bad.to(tl.int32), axis=0) == 0).to(tl.int32)
        tl.store(ok, good)


@triton.jit
def _row_gather(x_ptr, idx_ptr, out_ptr, C, BLOCK: tl.constexpr):
    """out[r, :] = x[idx[r], :] if idx[r] >= 0 else 0; one program per (row, BLOCK columns)."""
    r = tl.program_id(0)
    cb = tl.program_id(1)
    src = tl.load(idx_ptr + r)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    m = cols < C
    v = tl.load(x_ptr + src.to(tl.int64) * C + cols, mask=m & (src >= 0), other=0.0)
    tl.store(out_ptr + r.to(tl.int64) * C + cols, v, mask=m)


def row_gather(x: torch.Tensor, idx: torch.Tensor, rows: int) -> torch.Tensor:
    """(N, ...) -> (rows, ...): row r = x[idx[r]], or zeros where idx[r] < 0."""
    x = x.contiguous()
    C = 1
    for n in x.shape[1:]:
        C *= int(n)
    out = torch.empty((rows,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    BLOCK = 2048 if C >= 2048 else triton.next_power_of_2(C)
    _row_gather[(rows, triton.cdiv(C, BLOCK))](x, idx, out, C, BLOCK=BLOCK)
    return out


def _src_index(dst: torch.Tensor, rows: int) -> torch.Tensor:
    """The inverse row map: src[dst[i]] = i, -1 elsewhere (int32)."""
    src = torch.full((rows,), -1, dtype=torch.int32, device=dst.device)
    src.index_copy_(0, dst, torch.arange(dst.numel(), dtype=torch.int32, device=dst.device))
    return src


class _Pack(torch.autograd.Function):
    """packed (total, ...) -> aligned (rows, ...) with zero padding rows; backward = the row gather back."""

    @staticmethod
    def forward(ctx, x, src, dst, rows):
        ctx.save_for_backward(dst)
        return row_gather(x, src, rows)

    @staticmethod
    def backward(ctx, g):
        (dst,) = ctx.saved_tensors
        return g.contiguous().index_select(0, dst), None, None, None


class _Unpack(torch.autograd.Function):
    """aligned (rows, ...) -> packed (total, ...); backward = the row gather into zero-padded rows."""

    @staticmethod
    def forward(ctx, x, src, dst, rows):
        ctx.save_for_backward(src)
        ctx.rows = rows
        return x.contiguous().index_select(0, dst)

    @staticmethod
    def backward(ctx, g):
        (src,) = ctx.saved_tensors
        return row_gather(g, src, ctx.rows), None, None, None


def _src_of(docs: "LPDocs", side: str):
    dst, rows = (docs.dst_q, docs.s_q) if side == "q" else (docs.dst_k, docs.s_k)
    key = "_src_" + side
    if getattr(docs, key, None) is None:
        setattr(docs, key, _src_index(dst, rows))
    return getattr(docs, key), dst, rows


def pack_rows(docs: "LPDocs", x: torch.Tensor, side: str) -> torch.Tensor:
    """Differentiable repack of a packed (total, h, d) tensor to the aligned (1, s, h, d) layout (side 'q' / 'k')."""
    src, dst, rows = _src_of(docs, side)
    return _Pack.apply(x, src, dst, rows).unsqueeze(0)


def unpack_rows(docs: "LPDocs", x: torch.Tensor, side: str) -> torch.Tensor:
    """Differentiable inverse of pack_rows: (1, s, h, d) -> (total, h, d)."""
    src, dst, rows = _src_of(docs, side)
    return _Unpack.apply(x.squeeze(0), src, dst, rows)  # a view (x[0] would back-propagate a zero fill + copy)


# ---------------------------------------------------------------- the aligned path's scheduling work lists (step 3)
# The switches are lowprec.specs.HostSpec fields (varlen_fwd_list, varlen_fwd_tma_o, varlen_fwd_stage_skip,
# varlen_bwd_list, varlen_l2_sections; KernelTuning sets them); the drivers pass the resolved values in.
_L2_BYTES = 50 * 1024 * 1024  # FA4 SingleTileLPTScheduler's K / V budget (the forward's lists)
# the backward's lists (Q, dO and the f16 dQ accumulator under reduce-add atomics): measured on the 1-CTA backward,
# causal, h16 d128, paired, vs the plain grid: 16 MB -1.5 / -4 / -9 % (eq16x1024 / eq4x4096 / skew8192+32x256),
# +1 % eq64x1024; 50 MB loses 4 % on eq64x1024, 4-8 MB loses the skewed cell's gain
_L2_BYTES_BWD = 16 * 1024 * 1024


def l2_heads(num_head: int, bytes_per_head: int, sections: bool = True, budget: int = _L2_BYTES) -> int:
    """Heads per L2 section (FA4 SingleTileLPTScheduler / SingleTileLPTBwdScheduler's swizzle: the largest power of two
    of heads whose per-head footprint fits the budget), clamped to [1, num_head]; num_head = one section."""
    if not sections or bytes_per_head <= 0:
        return num_head
    n = budget // bytes_per_head
    s = 1 if n < 1 else 1 << (int(n).bit_length() - 1)
    return max(1, min(s, num_head))


def l2_span(bytes_per_token_head: int, sec_heads: int, total_tokens: int, num_head: int, sections: bool = True,
            budget: int = _L2_BYTES) -> int:
    """Document-section span in tokens (varlen_worklist span): 0 when every head's whole allocation fits the L2
    budget, else the tokens whose footprint for sec_heads heads fits it."""
    if not sections or total_tokens * num_head * bytes_per_token_head <= budget:
        return 0
    return max(budget // (bytes_per_token_head * sec_heads), 128)


def varlen_worklist(docs: "LPDocs", kind: str, causal: bool, sec_heads: int, pos_heads: int = 0,
                    span: int = 0, ctr: bool = False) -> torch.Tensor:
    """The aligned path's work list for tile_scheduler.VarlenWorkListScheduler, built on the GPU by one launch (no host
    sync; cached on `docs` per (kind, causal, sec_heads)). int32 (2 + 2 n_max,): [n_entries, sec_heads, a_0, b_0, ...].

    kind 'fwd': one entry per (document, 256-row Q block) with a real row, (a, b) = (document, block); n_max =
    s_q / 128 (every entry holds >= 128 rows of its 128-aligned slot). kind 'bwd': one entry per 128-key tile with a
    real key, (a, b) = (0, tile); n_max = s_k / 128. Order: longest first by the tile's KV / Q block count (causal:
    bottom-right diagonal, as BlockInfo), then the document's work (len_q x len_k) descending, the document index, and
    inside a document the longest causal tile first (fwd: the last Q block; bwd: the first key tile). Equal-length
    documents give the batched PersistentZigzagLPTScheduler order (level-major, then document, then head).

    pos_heads = H > 0 (the one-CTA-per-position backward): the list also carries the decoded position table after the
    entries, n_max x H int32 words: position p -> (b << 10) | head, or -1 past n_entries x H, so a CTA reads its tile
    with ONE load at blockIdx.x (no header read, no division) -- the dependent loads at CTA start were measured +8-10 %
    on the backward.

    ctr (fwd only): 4 extra int32 words after the entries, zeroed by this launch -- the dynamic claim
    counters of tile_scheduler.VarlenWorkListDynScheduler (which resets them at the end of every launch itself).

    span > 0: document sections -- the entries are grouped by allocation range (document start // span tokens: a
    range whose K / V (fwd) or Q / dO / dQ accumulator (bwd) for sec_heads heads fits L2) ahead of the longest-first
    order, so the work in flight stays inside one range (FA4's (batch, head) L2 sections); 0 = one range.
    """
    key = (kind, bool(causal), int(sec_heads), int(pos_heads), int(span)) + (("ctr",) if ctr else ())
    if getattr(docs, "doc_maps", None) is not None:   # (lp_docmaps: once per microbatch, not per call)
        from flash_attn.cute.lp_docmaps import worklist_cache
        cache = worklist_cache(docs.doc_maps)
    else:
        cache = docs.__dict__.setdefault("_worklists", {})
    if key in cache:
        return cache[key]
    assert docs.aligned and kind in ("fwd", "bwd")
    n_max = (docs.s_q if kind == "fwd" else docs.s_k) // 128
    assert 0 <= pos_heads < 1024
    assert not ctr or (kind == "fwd" and pos_heads == 0)
    out = torch.empty(2 + 2 * n_max + n_max * pos_heads + (4 if ctr else 0), dtype=torch.int32, device=docs.cu_q.device)
    _worklist_build(docs, out, kind == "fwd", bool(causal), n_max, int(sec_heads), int(pos_heads),
                    int(span) if span > 0 else 1 << 30, ctr)
    cache[key] = out
    return out


# The work list sorts the candidate tiles by the total order of varlen_worklist, as the distinct key triple
#   hi = (section ascending, tile work descending; invalid last), x = -(document work), t = (document, index order)
# (t: bwd the index; fwd the index mirrored inside its document's tile range, so the last Q block comes first).
# O(n log n) over the valid range n_lim = ceil(cu[b] / 128) (device-side) at any document count (the former rank
# sort was O(n^2): 14 ms for 65536 one-token documents, 50 ms for 131072). A fixed number of ordinary launches, one
# per phase (stream order between them; no grid barrier, no spin wait, no shared counter):
#   _wl_keys:  the keys of every candidate below n_lim (half 1 of the key buffers) and each program's valid count;
#   _wl_runs:  each run of _WL_BLK keys sorted by in-run rank (pairwise; -> half 0), and the order-free words
#              (header, counters, the position table's -1 past the entries);
#   _wl_merge: one launch per _WL_K-way merge round (a key's rank in its group's other runs by binary search),
#              ping-pong between the halves; the round count is static (n_max), a round whose runs already cover
#              n_lim exits at once.
# The phase that produces the final order (the run sort or the last merge round that runs) writes the entries and
# their position-table words at their final slots. Launches: 2 + the merge rounds (ceil(log_8(n_max / 256))).
# Per-build buffers (static sizes), no host read: CUDA-graph capturable, safe on concurrent streams.
_WL_FB = tl.constexpr(32)       # candidates per key step
_WL_SEARCH = tl.constexpr(16)   # the document search's fan-out
_WL_BLK = tl.constexpr(256)     # the run sort's run length
_WL_BI = tl.constexpr(32)       # rows per run-sort / emit step
_WL_BJ = tl.constexpr(32)       # columns per run-sort step
_WL_K = tl.constexpr(8)         # merge fan-in
_WL_BM = tl.constexpr(128)      # positions per merge step
_WL_LAUNCH: dict = {}           # (device, n_max, b) -> launch geometry


def _wl_geometry(dev, n_max: int, b: int):
    lk = (dev.index, n_max, b)
    if lk not in _WL_LAUNCH:
        steps = 2
        while _WL_SEARCH.value ** (steps - 1) <= b:
            steps += 1
        rounds, run = [], _WL_BLK.value
        while run < n_max:   # (run, binary-search steps: run + 1 <= 2^steps, source half)
            rounds.append((run, run.bit_length(), len(rounds) % 2))
            run *= _WL_K.value
        cap = 4 * torch.cuda.get_device_properties(dev).multi_processor_count   # programs per phase (strided loops)
        grid = lambda tile: (max(1, min(cap, triton.cdiv(n_max, tile))),)
        _WL_LAUNCH[lk] = (steps, tuple(rounds), grid(_WL_FB.value), grid(_WL_BI.value), grid(_WL_BM.value))
    return _WL_LAUNCH[lk]


def _worklist_build(docs: "LPDocs", out: torch.Tensor, fwd: bool, causal: bool, n_max: int, sec_heads: int,
                    pos_heads: int, span: int, ctr: bool):
    dev = docs.cu_q.device
    steps, rounds, g_keys, g_rows, g_merge = _wl_geometry(dev, n_max, docs.b)
    cu_side = docs.cu_q if fwd else docs.cu_k
    # [hi, x, payload] (int64), t (int32), two halves each (the merge rounds' ping-pong); then the valid count of each
    # _wl_keys program (int32; every program writes its own)
    keys = torch.empty(7 * n_max + (g_keys[0] + 1) // 2, dtype=torch.int64, device=dev)
    _wl_keys[g_keys](docs.cu_q, docs.cu_k, docs.len_q, docs.len_k, keys, docs.b, n_max, span, steps, FWD=fwd,
                     CAUSAL=causal)
    _wl_runs[g_rows](cu_side, keys, out, docs.b, n_max, sec_heads, pos_heads, g_keys[0], CTR=ctr)
    for run, msteps, src in rounds:
        _wl_merge[g_merge](cu_side, keys, out, docs.b, n_max, sec_heads, pos_heads, g_keys[0], run, msteps, src)


@triton.jit
def _wl_feat(c, cu_q, cu_k, len_q, len_k, b, n_max, steps, FWD: tl.constexpr, CAUSAL: tl.constexpr,
             SEARCH: tl.constexpr):
    """Candidate c (a 128-row Q tile (FWD) / 128-key tile of the allocation) -> (valid, tile work, document work,
    document)."""
    inb = c < n_max
    cu_side = cu_q if FWD else cu_k
    d = _upper_bound_wide(cu_side, c * 128, b, steps, SEARCH)
    ind = inb & (d < b)
    st = tl.load(cu_side + d, mask=ind, other=0)
    lq = tl.load(len_q + d, mask=ind, other=0)
    lk = tl.load(len_k + d, mask=ind, other=0)
    r = c * 128 - st
    if FWD:
        valid = ind & (r % 256 == 0) & (r < lq)
        nk = (lk + 127) // 128
        if CAUSAL:
            n_idx = r + 256 + lk - lq          # (m + 1) * 256 + seqlen_k - seqlen_q, m = r / 256
            w = tl.minimum(nk, tl.where(n_idx > 0, (n_idx + 127) // 128, 0))
        else:
            w = nk
    else:
        valid = ind & (r < lk)
        nq = (lq + 127) // 128
        if CAUSAL:
            m_idx = r + lq - lk                # n_loc * 128 + seqlen_q - seqlen_k
            m_min = tl.where(m_idx > 0, m_idx // 128, 0)
            w = tl.maximum(nq - m_min, 0)
        else:
            w = nq
    dw = lq.to(tl.int64) * lk.to(tl.int64)
    return valid, w, dw, d, st


@triton.jit
def _wl_lt(ha, xa, ta, hb, xb, tb):
    """Key a before key b (the work list's total order on (hi, x, t))."""
    return (ha < hb) | ((ha == hb) & ((xa < xb) | ((xa == xb) & (ta < tb))))


@triton.jit
def _wl_nlim(cu_side, b, n_max):
    # every valid candidate lies below the side's aligned total (cu[b] / 128 tiles)
    return tl.minimum((tl.load(cu_side + b) + 127) // 128, n_max)


@triton.jit
def _wl_keys(cu_q, cu_k, len_q, len_k, keys, b, n_max, span, steps, FWD: tl.constexpr, CAUSAL: tl.constexpr):
    """The keys of every candidate below n_lim (half 1) and this program's valid count."""
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    cu_side = cu_q if FWD else cu_k
    n_lim = _wl_nlim(cu_side, b, n_max)
    kh, kx, kp = keys + n_max, keys + 3 * n_max, keys + 5 * n_max
    kt = (keys + 6 * n_max).to(tl.pointer_type(tl.int32)) + n_max
    nv = tl.zeros([_WL_FB], dtype=tl.int32)
    for c0 in range(pid * _WL_FB, n_lim, P * _WL_FB):
        c = c0 + tl.arange(0, _WL_FB)
        v, w, dw, d, st = _wl_feat(c, cu_q, cu_k, len_q, len_k, b, n_max, steps, FWD, CAUSAL, _WL_SEARCH)
        g = st // span   # document section (span = 2^30 from the host: one section)
        # hi: the section ascending, then the tile work descending; invalid last
        h = tl.where(v, (g.to(tl.int64) << 32) | (2147483647 - w).to(tl.int64), tl.full([_WL_FB], 1 << 62, tl.int64))
        x = tl.where(v, -dw, 0)   # the document work descending
        if FWD:
            # the document, then the last Q block first: the index mirrored inside the document's candidate range
            # [ceil(cu[d] / 128) (0 for d = 0), ceil(cu[d + 1] / 128))
            lo = tl.where(d == 0, 0, (st + 127) // 128)
            hi = tl.minimum((tl.load(cu_side + d + 1, mask=v, other=0) + 127) // 128, n_max)
            t = tl.where(v, lo + hi - 1 - c, c)
            a = d
            bb = (c * 128 - tl.load(cu_q + d, mask=v, other=0)) // 256
        else:
            t = c   # the document, then the first key tile first: the index order
            a = tl.zeros_like(d)
            bb = c
        pl = (a.to(tl.int64) << 32) | (bb.to(tl.int64) & 0xFFFFFFFF)   # payload: the entry
        m = c < n_lim
        tl.store(kh + c, h, mask=m)
        tl.store(kx + c, x, mask=m)
        tl.store(kt + c, t, mask=m)
        tl.store(kp + c, pl, mask=m)
        nv += v.to(tl.int32)
    vcnt = (keys + 7 * n_max).to(tl.pointer_type(tl.int32))
    tl.store(vcnt + pid, tl.sum(nv, axis=0))


@triton.jit
def _wl_runs(cu_side, keys, out, b, n_max, sec_heads, pos_heads, n_vc, CTR: tl.constexpr):
    """Each run of _WL_BLK keys below n_lim sorted by in-run rank (half 1 -> half 0); the order-free work-list words;
    the entries when the runs already cover n_lim (no merge round runs)."""
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    n_lim = _wl_nlim(cu_side, b, n_max)
    n_ent = _wl_nent(keys, n_max, n_vc)
    _wl_fill(out, pid, P, n_ent, n_max, sec_heads, pos_heads, CTR)
    kt = (keys + 6 * n_max).to(tl.pointer_type(tl.int32))
    for i0 in range(pid * _WL_BI, n_lim, P * _WL_BI):
        i = i0 + tl.arange(0, _WL_BI)
        mi = i < n_lim
        hi = tl.load(keys + n_max + i, mask=mi, other=0)
        xi = tl.load(keys + 3 * n_max + i, mask=mi, other=0)
        pi = tl.load(keys + 5 * n_max + i, mask=mi, other=0)
        ti = tl.load(kt + n_max + i, mask=mi, other=0)
        r0 = (i0 // _WL_BLK) * _WL_BLK
        r1 = tl.minimum(r0 + _WL_BLK, n_lim)
        rank = tl.zeros([_WL_BI], dtype=tl.int32)
        for j0 in range(r0, r1, _WL_BJ):
            j = j0 + tl.arange(0, _WL_BJ)
            mj = j < r1
            hj = tl.load(keys + n_max + j, mask=mj, other=0)
            xj = tl.load(keys + 3 * n_max + j, mask=mj, other=0)
            tj = tl.load(kt + n_max + j, mask=mj, other=0)
            lt = _wl_lt(hj[None, :], xj[None, :], tj[None, :], hi[:, None], xi[:, None], ti[:, None]) & mj[None, :]
            rank += tl.sum(lt.to(tl.int32), axis=1)
        o = r0 + rank
        if n_lim <= _WL_BLK:   # the final order
            _wl_put(out, o, pi, mi, n_ent, n_max, sec_heads, pos_heads)
        else:
            tl.store(keys + o, hi, mask=mi)
            tl.store(keys + 2 * n_max + o, xi, mask=mi)
            tl.store(keys + 4 * n_max + o, pi, mask=mi)
            tl.store(kt + o, ti, mask=mi)


@triton.jit
def _wl_merge(cu_side, keys, out, b, n_max, sec_heads, pos_heads, n_vc, run, msteps, src):
    """One _WL_K-way merge round of the sorted runs of `run` keys below n_lim (half src -> the other): a key's slot =
    its group's start + its index in its run + #{keys of the group's other runs before it} (binary searches). A round
    whose runs already cover n_lim does nothing; the last round that runs writes the entries."""
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    n_lim = _wl_nlim(cu_side, b, n_max)
    kt = (keys + 6 * n_max).to(tl.pointer_type(tl.int32))
    s0 = src * n_max
    d0 = n_max - s0
    final = run * _WL_K >= n_lim
    n_ent = _wl_nent(keys, n_max, n_vc)
    for p0 in range(pid * _WL_BM, tl.where(run < n_lim, n_lim, 0), P * _WL_BM):
        p = p0 + tl.arange(0, _WL_BM)
        mp = p < n_lim
        hp = tl.load(keys + s0 + p, mask=mp, other=0)
        xp = tl.load(keys + 2 * n_max + s0 + p, mask=mp, other=0)
        pp = tl.load(keys + 4 * n_max + s0 + p, mask=mp, other=0)
        tp = tl.load(kt + s0 + p, mask=mp, other=0)
        own = p // run
        g0 = (own // _WL_K) * _WL_K
        rr = g0[:, None] + tl.arange(0, _WL_K)[None, :]                  # [BM, K]: the group's runs
        rs = rr * run
        lo = rs
        hi = tl.where((rr == own[:, None]) | ~mp[:, None], rs, tl.maximum(tl.minimum(rs + run, n_lim), rs))
        for _s in range(msteps):
            go = lo < hi
            mid = (lo + hi) // 2
            hm = tl.load(keys + s0 + mid, mask=go, other=0)
            xm = tl.load(keys + 2 * n_max + s0 + mid, mask=go, other=0)
            tm = tl.load(kt + s0 + mid, mask=go, other=0)
            lt = _wl_lt(hm, xm, tm, hp[:, None], xp[:, None], tp[:, None])
            lo = tl.where(go & lt, mid + 1, lo)
            hi = tl.where(go & ~lt, mid, hi)
        o = g0 * run + (p - own * run) + tl.sum(lo - rs, axis=1)
        if final:
            _wl_put(out, o, pp, mp, n_ent, n_max, sec_heads, pos_heads)
        else:
            tl.store(keys + d0 + o, hp, mask=mp)
            tl.store(keys + 2 * n_max + d0 + o, xp, mask=mp)
            tl.store(keys + 4 * n_max + d0 + o, pp, mask=mp)
            tl.store(kt + d0 + o, tp, mask=mp)


@triton.jit
def _wl_nent(keys, n_max, n_vc):
    """The valid count: the sum of the _wl_keys programs' counts."""
    vcnt = (keys + 7 * n_max).to(tl.pointer_type(tl.int32))
    acc = tl.zeros([1024], dtype=tl.int32)
    for v0 in range(0, n_vc, 1024):
        vv = v0 + tl.arange(0, 1024)
        acc += tl.load(vcnt + vv, mask=vv < n_vc, other=0)
    return tl.sum(acc, axis=0)


@triton.jit
def _wl_put(out, o, pl, m, n_ent, n_max, sec_heads, pos_heads):
    """The work-list words of the keys m at their final slots o (valid keys sort first: o < n_ent): entry o = the
    payload's (a, b), and its position-table words (VarlenWorkListScheduler pos_off: head sections of sec_heads heads,
    section-major, then the entry, then the head inside the section)."""
    vk = m & (o < n_ent)
    a = (pl >> 32).to(tl.int32)
    bb = pl.to(tl.int32)
    tl.store(out + 2 + 2 * o, a, mask=vk)
    tl.store(out + 3 + 2 * o, bb, mask=vk)
    if pos_heads > 0:
        base = out + 2 + 2 * n_max
        for h0 in range(0, pos_heads, 32):
            hh = h0 + tl.arange(0, 32)
            mh = hh < pos_heads
            sec = hh // sec_heads
            n_full = pos_heads // sec_heads
            width = tl.where(sec < n_full, sec_heads, pos_heads - n_full * sec_heads)
            p = sec[None, :] * (n_ent * sec_heads) + o[:, None] * width[None, :] + (hh - sec * sec_heads)[None, :]
            val = (bb[:, None] << 10) | hh[None, :]
            tl.store(base + p, val, mask=vk[:, None] & mh[None, :])


@triton.jit
def _wl_fill(out, pid, P, n_ent, n_max, sec_heads, pos_heads, CTR: tl.constexpr):
    """The order-free words: the header (n_entries, sec_heads), the counters, and the position table's -1 past
    n_entries x H."""
    if pid == 0:
        tl.store(out, n_ent)
        tl.store(out + 1, sec_heads)
        if CTR:
            # the dynamic claim counters (VarlenWorkListDynScheduler): zero at the first launch that uses the list
            tl.store(out + 2 + 2 * n_max + tl.arange(0, 4), tl.zeros([4], dtype=tl.int32))
    if pos_heads > 0:
        base = out + 2 + 2 * n_max
        for k0 in range(pid * _WL_BI, n_max, P * _WL_BI):
            k = k0 + tl.arange(0, _WL_BI)
            for h0 in range(0, pos_heads, 32):
                hh = h0 + tl.arange(0, 32)
                q = k[:, None] * pos_heads + hh[None, :]
                tl.store(base + q, tl.full([_WL_BI, 32], -1, tl.int32),
                         mask=(k < n_max)[:, None] & (hh < pos_heads)[None, :] & (q >= n_ent * pos_heads))

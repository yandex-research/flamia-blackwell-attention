# Precomputed document maps for packed varlen (the repacking path without per-call index building).
#
# A training loop that packs documents back to back (token buffer (T, h, d), unaligned starts) keeps the buffer packed
# and still runs the aligned zero-copy kernels on an aligned layout of its documents (each document at a 128-aligned
# start of an S-row buffer). Every index map that needs -- aligned starts, true lengths, the packed -> aligned and
# aligned -> packed row maps, the kernels' document tables -- is built ONCE per microbatch, on the CPU by the data
# loader worker (numpy, no CUDA) or on the GPU, and reused by every layer: no LPDocs.build / aligned-prep launches, no
# host sync, the scheduling work lists built once per microbatch.
#
# FUSED_GATHER (the token-block E4M3 P.V routes, the defaults): the aligned buffer is never materialised. The
# quantisers read Q / K / V / dO through the row map (quant.fp8.quantize_fp8_blocks: kernel_stream / kernel_tc), the
# forward writes O to the packed rows (flash_fwd_sm100_fp4 o_rows_packed), the backward's dQ post-process scatters to
# them (dq_postprocess_cute) and its dK / dV epilogue writes them (flash_bwd_sm100_fp4 dkv_rows_packed). Other routes
# gather the rows into the aligned buffer and back (pack_rows / unpack_rows). Bitwise the repacking path either way.
# Either way the call is the registered op lowbit_attn::fp4 / ::fp8 (lowprec._op): the maps travel as its document
# tensors (DocMaps is one of the objects it flattens), the fused node takes the caller's packed (1, T, h, d) rows.
#
# Shapes are static: a fixed document capacity N (zero-length trailing documents fill it) and a fixed padded size S (a
# host-side bound, by default round_up(T + N * 127, 256)), so the maps of every microbatch have the same shapes and a
# CUDA graph can take them as static inputs (DocMaps.copy_ refreshes them in place).
#
#   maps = DocMaps.from_cu_seqlens(cu_seqlens, max_num_documents=N, max_seqlen=2048)   # worker, CPU
#   maps = maps.to("cuda", non_blocking=True)                                          # main process
#   out = flash_attn_fp8(q, k, v, causal=True, doc_maps=maps)                          # every layer
#
# All tables live in ONE flat int32 buffer (one H2D copy); the named fields are views of it.
import weakref
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

_TILE = 128


def _round_up(x: int, a: int) -> int:
    return (int(x) + a - 1) // a * a


def default_padded_size(total: int, max_num_documents: int, align: int = 128) -> int:
    """The static padded allocation: every document start rounded up to `align` costs at most align - 1 rows, so
    total + N (align - 1) rows always suffice; rounded up to 256 (and at least 256)."""
    return max(_round_up(int(total) + int(max_num_documents) * (align - 1), 256), 256)


def large_capacity(maps) -> bool:
    """(compact maps) A capacity N far above the documents' tile count: the tile bound above 4 ceil(T / 128) (N > 3 T /
    128; the fork's N = T). The kernels then run persistently over their work lists (the backward) / claim atomically
    (the forward): a grid sized by the static bound would launch mostly empty CTAs (N = T, T = 131072, 8 kv heads: 1 M
    in the backward, +13.7 ms). Below it the per-position grid and CLC stay (the persistent backward is ~12 % slower
    where its work tiles are not mostly empty: its next tile's Q / dO wait for the dK / dV epilogue)."""
    if not getattr(maps, "compact", False):
        return False
    T = int(maps.total_q)
    return tile_bound(T, maps.max_num_documents) > 4 * (-(-T // _TILE))


def tile_bound(total: int, max_num_documents: int) -> int:
    """(compact maps) The static bound of a side's 128-row document tiles: every tile holds >= 1 real row, and a
    document of length L has ceil(L / 128) <= L / 128 + 1 of them -- so tiles <= min(T, ceil(T / 128) + N), never more
    than T (at least 1)."""
    return max(1, min(int(total), -(-int(total) // _TILE) + int(max_num_documents)))


def _layout(n: int, tq: int, tk: int, sq: int, sk: int, compact: bool = False):
    """Offsets (in int32 words) of the views in the flat buffer. q_valid: sq bytes = sq / 4 words. compact: no S-row
    view (src, q_valid, cu_vt); sq / sk are 128 x the tile bounds; vt_tile_q / vt_tile hold each tile's PACKED start
    (row 0) and real rows (row 1); ntiles = the two sides' device tile counts."""
    if compact:
        parts = (("cu_q", (n + 1,)), ("cu_k", (n + 1,)), ("cu_pad_q", (n + 1,)), ("cu_pad_k", (n + 1,)),
                 ("len_q", (n,)), ("len_k", (n,)), ("doc_tab", (6, n + 1)), ("dst_q", (tq,)), ("dst_k", (tk,)),
                 ("tile_doc_q", (sq // _TILE,)), ("tile_doc_k", (sk // _TILE,)), ("row_end_q", (sq // _TILE,)),
                 ("row_end_k", (sk // _TILE,)), ("flags", (1,)), ("vt_tile", (2, sk // _TILE)),
                 ("vt_tile_q", (2, sq // _TILE)), ("ntiles", (2,)))
        off, out = 0, {}
        for nm, shp in parts:
            cnt = int(np.prod(shp))
            out[nm] = (off, shp)
            off += cnt
        return out, off
    parts = (("cu_q", (n + 1,)), ("cu_k", (n + 1,)), ("cu_pad_q", (n + 1,)), ("cu_pad_k", (n + 1,)),
             ("len_q", (n,)), ("len_k", (n,)), ("doc_tab", (6, n + 1)), ("dst_q", (tq,)), ("dst_k", (tk,)),
             ("src_q", (sq,)), ("src_k", (sk,)), ("tile_doc_q", (sq // _TILE,)), ("tile_doc_k", (sk // _TILE,)),
             ("row_end_q", (sq // _TILE,)), ("row_end_k", (sk // _TILE,)), ("flags", (1,)), ("q_valid", (sq // 4,)),
             ("cu_vt", (n + 1,)), ("vt_tile", (2, sk // _TILE)), ("cu_vt_q", (n + 1,)), ("vt_tile_q", (2, sq // _TILE)))
    off, out = 0, {}
    for nm, shp in parts:
        cnt = int(np.prod(shp))
        out[nm] = (off, shp)
        off += cnt
    return out, off


VT_LAST = 1 << 16   # vt_tile[1] flag: the last real key tile (zeroes its columns up to 128 past its document)

# flags bits (one int32 word): the 4-bit P.V routes' per-document key checks (LPDocs.aligned_from require_keys)
FLAG_KEYS_ANY = 1      # every document with a query has a key
FLAG_KEYS_CAUSAL = 2   # len_q <= len_k for every document


_CACHE = weakref.WeakKeyDictionary()   # DocMaps -> its derived objects (lpdocs, the lean copy)
_WL_CACHE: dict = {}   # id(maps.buf) -> (weakref(buf), {(capture key, buf version): {work-list key: tensor}})


@dataclass(eq=False)   # (identity-hashed: the derived objects are cached per maps object)
class DocMaps:
    """The document maps of one microbatch of packed varlen documents (see the module comment). Build with
    from_cu_seqlens (CPU numpy or GPU torch), move with to(device, non_blocking=True), pass as doc_maps= to
    flash_attn_fp4 / flash_attn_fp8. Host facts are plain ints; every tensor is a view of `buf`.

    cu_q / cu_k (N + 1,): the caller's packed starts; cu_pad_q / cu_pad_k (N + 1,): aligned starts (+ the end);
    len_q / len_k (N,): true lengths; dst_q / dst_k (T,): packed row -> aligned row; src_q / src_k (S,): aligned row ->
    packed row (-1 on padding rows); doc_tab (6, N + 1), tile_doc_q / tile_doc_k (S / 128,), row_end_q / row_end_k
    (S / 128,), q_valid (S,) bool: the aligned path's kernel tables (lp_varlen.LPDocs.aligned_from's prep, rows 4 / 5
    of doc_tab = the caller's packed starts, as the repack path); flags: FLAG_KEYS_* bits. cu_vt (N + 1,) / vt_tile
    (2, S_k / 128): the packed rows' channel-major V codes (PACKED_ROWS), whose documents start at multiples of 16 keys
    (cu_vt: the starts, cumsum of round_up(len_k, 16)): per aligned key tile its first key's column there and its real
    rows (-1 / 0 past the documents); cu_vt_q / vt_tile_q (S_q / 128): the same for the queries (the packed quantisers
    read a query tile's real rows from row 1; the backward's statistics -- shifted LSE, D -- are per aligned row)."""
    total_q: int
    total_k: int
    max_num_documents: int
    padded_size_q: int
    padded_size_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    align: int
    buf: torch.Tensor
    num_documents: Optional[int] = None   # host-known on the CPU build (documents before the zero-length tail)
    host_flags: Optional[int] = None      # host-known on the CPU build
    # compact (the default without a padded size; sfd): padded_size_q = padded_size_k = 0 (no padded size), so the
    # constructor fields / shape_key keep their form (a caller's pytree round trip rebuilds compact maps). No table is
    # sized by a padded size -- every per-tile table is sized by the static tile bound min(T, ceil(T / 128) + N)
    # (rows_q / rows_k = 128 x it: a virtual aligned coordinate space, never allocated), no row map of the aligned
    # rows; the packed path (lp_docmaps.packed_*) reads the tile tables vt_tile / vt_tile_q (each tile's packed start
    # and real rows) and ntiles. Other plans take the full maps of the same documents (aligned_maps(): the default
    # padded size, built on first use)
    compact: bool = False
    # host-known length skew of the key sequences, max length / mean length over the non-empty documents (CPU builds;
    # None on the GPU build: no host sync). Read by fp4_bwd_autograd._aligned_fwd_sched (non-causal scheduler rule).
    sched_skew: Optional[float] = None
    # sched_skew >= SKEW_CLC_MIN, host-known (the coarse bit survives into the stripped copy the torch ops key on)
    sched_skewed: Optional[bool] = None

    def __post_init__(self):
        if self.padded_size_q == 0 and self.padded_size_k == 0:
            self.compact = True

    @property
    def rows_q(self) -> int:
        """The aligned row space of the queries: the padded size, or (compact) 128 x the tile bound."""
        return _TILE * tile_bound(self.total_q, self.max_num_documents) if self.compact else self.padded_size_q

    @property
    def rows_k(self) -> int:
        return _TILE * tile_bound(self.total_k, self.max_num_documents) if self.compact else self.padded_size_k

    # ---- views
    def _view(self, name):
        off, shp = _layout(self.max_num_documents, self.total_q, self.total_k, self.rows_q, self.rows_k,
                           self.compact)[0][name]
        return self.buf[off:off + int(np.prod(shp))].view(shp)

    def __getattr__(self, name):
        if name in ("cu_q", "cu_k", "cu_pad_q", "cu_pad_k", "len_q", "len_k", "doc_tab", "dst_q", "dst_k", "src_q",
                    "src_k", "tile_doc_q", "tile_doc_k", "row_end_q", "row_end_k", "flags", "cu_vt", "vt_tile",
                    "cu_vt_q", "vt_tile_q", "ntiles"):
            return self._view(name)
        if name == "q_valid":
            return self._view("q_valid").view(torch.uint8).view(torch.bool)
        raise AttributeError(name)

    @property
    def device(self):
        return self.buf.device

    # ---- construction
    @staticmethod
    def from_cu_seqlens(cu_seqlens_q, cu_seqlens_k=None, *, max_num_documents: Optional[int] = None,
                        total_q: Optional[int] = None, total_k: Optional[int] = None,
                        padded_size_q: Optional[int] = None, padded_size_k: Optional[int] = None,
                        max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
                        max_seqlen: Optional[int] = None, padded_size: Optional[int] = None,
                        align: int = 128) -> "DocMaps":
        """Maps from packed cu_seqlens ((n + 1,), starts of n back-to-back documents, cu[0] = 0, cu[n] = total).

        CPU input (numpy array, list, or a CPU tensor): built with numpy on the host (no CUDA; for a dataloader
        worker); the result's tensors are CPU (pin_memory() / to(device, non_blocking=True)). CUDA tensor input: built
        with GPU ops and one prep launch, no host sync -- then total_q / total_k (the packed lengths) and max_seqlen_*
        must be given (host ints).

        max_num_documents N >= n: the fixed capacity (zero-length trailing documents fill it; default n).
        padded_size(_*): the aligned allocation (a multiple of 128 holding every aligned document; default
        default_padded_size(T, N)). max_seqlen(_*): the kernels' grid bound, >= every document's length; give a fixed
        bound for CUDA graphs (default on the CPU build: the longest document). align: 128 (the aligned path's alignment;
        256 also works)."""
        if align % _TILE or align <= 0:
            raise ValueError(f"align must be a positive multiple of {_TILE}, got {align}")
        if padded_size is None and padded_size_q is None and padded_size_k is None and align == _TILE:
            # (sfd) the compact maps: no padded size anywhere (the packed path); an explicit padded size keeps the
            # full maps (the fused aligned path, 57cab39 / 66d620f)
            if max_seqlen is not None:
                max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
                max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k
            if isinstance(cu_seqlens_q, torch.Tensor) and cu_seqlens_q.is_cuda:
                return _build_compact_gpu(cu_seqlens_q, cu_seqlens_k, max_num_documents, total_q, total_k,
                                          max_seqlen_q, max_seqlen_k)
            return _build_compact_cpu(cu_seqlens_q, cu_seqlens_k, max_num_documents, total_q, total_k, max_seqlen_q,
                                      max_seqlen_k)
        if padded_size is not None:
            padded_size_q = padded_size if padded_size_q is None else padded_size_q
            padded_size_k = padded_size if padded_size_k is None else padded_size_k
        if max_seqlen is not None:
            max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
            max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k
        if isinstance(cu_seqlens_q, torch.Tensor) and cu_seqlens_q.is_cuda:
            return _build_gpu(cu_seqlens_q, cu_seqlens_k, max_num_documents, total_q, total_k, padded_size_q,
                              padded_size_k, max_seqlen_q, max_seqlen_k, align)
        return _build_cpu(cu_seqlens_q, cu_seqlens_k, max_num_documents, total_q, total_k, padded_size_q,
                          padded_size_k, max_seqlen_q, max_seqlen_k, align)

    def to(self, device, non_blocking: bool = False) -> "DocMaps":
        """The same maps on `device` (one copy of the flat buffer). non_blocking: asynchronous from pinned memory."""
        return replace(self, buf=self.buf.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "DocMaps":
        """Pinned host copy (torch.utils.data.DataLoader(pin_memory=True) calls this on a returned DocMaps)."""
        return replace(self, buf=self.buf.pin_memory())

    def copy_(self, other: "DocMaps", non_blocking: bool = False) -> "DocMaps":
        """Refresh these (static, e.g. CUDA-graph input) maps in place from maps of the same shapes."""
        if self.shape_key() != other.shape_key():
            raise ValueError(f"DocMaps.copy_: shapes differ: {self.shape_key()} vs {other.shape_key()}")
        self.buf.copy_(other.buf, non_blocking=non_blocking)
        self.num_documents, self.host_flags = other.num_documents, other.host_flags
        self.sched_skew, self.sched_skewed = other.sched_skew, other.sched_skewed
        _CACHE.pop(self, None)
        return self

    @property
    def vt_cols_q(self) -> int:
        """The queries' statistics buffers' static row count (as vt_cols)."""
        return _round_up(self.total_q + 15 * self.max_num_documents, 16) + _TILE

    @property
    def vt_cols(self) -> int:
        """The packed channel-major V's static column count: holds cu_vt[-1] <= T_k + 15 N plus the last tile's 128
        zero columns, a multiple of 16."""
        return _round_up(self.total_k + 15 * self.max_num_documents, 16) + _TILE

    def shape_key(self) -> tuple:
        """Everything host-side a call specialises on (equal keys: interchangeable maps, one CUDA graph)."""
        return (self.total_q, self.total_k, self.max_num_documents, self.padded_size_q, self.padded_size_k,
                self.max_seqlen_q, self.max_seqlen_k, self.align)   # (compact: padded sizes 0)

    def aligned_maps(self) -> "DocMaps":
        """(compact maps) The full maps of the same documents (default padded size) for the plans the packed path does
        not serve: built on the GPU from these tables on first use (no host read) and cached on the maps object. Full
        maps: themselves."""
        if not self.compact:
            return self
        key = _capture_key(self.buf)
        c = _CACHE.setdefault(self, {})
        if key is not None and ("aligned", key) in c:
            return c[("aligned", key)]
        n = self.max_num_documents
        if self.buf.is_cuda:
            m = _build_gpu(self.cu_q, self.cu_k, n, self.total_q, self.total_k, None, None, self.max_seqlen_q,
                           self.max_seqlen_k, _TILE)
        else:
            m = _build_cpu(self.cu_q.numpy(), self.cu_k.numpy(), n, self.total_q, self.total_k, None, None,
                           self.max_seqlen_q, self.max_seqlen_k, _TILE)
        m = replace(m, num_documents=self.num_documents, host_flags=self.host_flags, sched_skew=self.sched_skew,
                    sched_skewed=self.sched_skewed)
        if key is not None:
            c[("aligned", key)] = m
        return m

    # ---- the library side
    def lpdocs(self, fused: bool = False, packed: bool = False):
        """The aligned path's lp_varlen.LPDocs of the padded buffer, from these tables (no launch); fused: the doc_maps
        call with the fused gathers (doc_maps_fused, lowprec._routes.forward). Cached on the maps so every layer
        reuses it (a stable object: one structure for lowprec._op); under CUDA-graph capture one per capture sequence.
        The work lists derived from it (lp_varlen.varlen_worklist) are cached per maps buffer and its version
        (worklist_cache): built once per microbatch, inside the graph under capture. copy_() refreshes both; do not
        edit buf in place otherwise."""
        key = _capture_key(self.buf)
        c = _CACHE.setdefault(self, {})
        if key is not None and (key, fused, packed) in c:
            return c[(key, fused, packed)]
        from flash_attn.cute import lp_varlen as LV
        n = self.max_num_documents
        r = LV.LPDocs(self.cu_pad_q, self.cu_pad_k, self.cu_pad_q, self.cu_pad_k, self.len_q, self.len_k, None, None,
                      self.rows_q, self.rows_k, _TILE, n, self.max_seqlen_q, self.max_seqlen_k)
        r.doc_tab = self.doc_tab
        r._tile_doc[_TILE] = self.tile_doc_k
        r.tile_doc_q = self.tile_doc_q
        r._q_valid = None if self.compact else self.q_valid
        r.aligned = True
        r.row_end_tab_q = self.row_end_q.view(1, -1)
        r.row_end_tab_k = self.row_end_k.view(1, -1)
        if LV.ALIGNED_ROW_PREDICATE:
            r.row_end_q, r.row_end_k = r.row_end_tab_q, r.row_end_tab_k
        # the maps without their per-microbatch host facts: what the kernels' hooks and lowprec._op's structure see
        if "lean" not in c:
            c["lean"] = replace(self, num_documents=None, host_flags=None, sched_skew=None)
        r.doc_maps = c["lean"]
        r.doc_maps_fused = bool(fused)
        if packed:   # (only a packed call's documents carry the attribute: the fused call's structure is unchanged)
            r.doc_maps_packed = True
        if key is not None:
            c[(key, fused, packed)] = r
        return r

    def check_call(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, entry: str):
        """Host-side checks of a call against these maps (shapes and device only: no device read)."""
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(f"flash_attn_{entry}(doc_maps=...): q, k, v are packed (total, h, d)")
        if q.shape[0] != self.total_q or k.shape[0] != self.total_k or v.shape[0] != self.total_k:
            raise ValueError(f"flash_attn_{entry}(doc_maps=...): the maps were built for total_q {self.total_q}, "
                             f"total_k {self.total_k}; got q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
        if self.buf.device != q.device:
            raise ValueError(f"flash_attn_{entry}(doc_maps=...): the maps are on {self.buf.device}, q on {q.device} "
                             "(move them once per microbatch: maps.to(q.device, non_blocking=True))")

    def stand_in(self, x: torch.Tensor, side: str) -> torch.Tensor:
        """A zero-memory (S, h, d) stand-in of x for plan resolution (shape, dtype, device, requires_grad)."""
        rows = self.rows_q if side == "q" else self.rows_k
        base = torch.empty((), dtype=x.dtype, device=x.device)
        if x.requires_grad:
            base.requires_grad_(True)
        return base.expand((rows,) + tuple(x.shape[1:]))

    def check_keys(self, require_keys: int, entry: str):
        """The 4-bit P.V routes' per-document key checks (LPDocs.aligned_from require_keys): host fail when the CPU
        build knows, else a device-side assert (no host read)."""
        if not require_keys:
            return
        bit = FLAG_KEYS_CAUSAL if require_keys == 2 else FLAG_KEYS_ANY
        msg = ("the 4-bit P.V routes need a key for every document with a query" +
               (" and len_q <= len_k per document (causal)" if require_keys == 2 else ""))
        if self.host_flags is not None:
            if not self.host_flags & bit:
                raise ValueError(f"flash_attn_{entry}(doc_maps=...): {msg}")
            return
        torch._assert_async((self.flags & bit) != 0, msg)


def _capture_key(buf: torch.Tensor):
    """The cache key of lpdocs(): "eager" outside stream capture; the capture sequence id inside one (an entry made
    during a capture holds work lists built by that graph's own launches, so it is reused only within that capture);
    None (no caching) when the id cannot be read."""
    if not buf.is_cuda or not torch.cuda.is_current_stream_capturing():
        return "eager"
    try:
        from cuda.bindings import runtime as rt
        r = rt.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream)
        if r[0] != rt.cudaError_t.cudaSuccess or r[1] != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            return None
        return ("capture", int(r[2]))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------------------- fused gather
# FUSED_GATHER (doc_maps calls on the token-block E4M3 P.V routes, fused_ok): the gathered (1, S, h, d) Q / K / V / dO are
# never written. The route gets SHAPE CARRIERS (uninitialised buffers) registered here with their packed source and row
# map; quant.fp8.quantize_fp8_blocks reads a registered carrier through the row map (kernel_stream; kernel_tc for the
# rotated Q / K), and every other quantiser entry (the other rotation kernels, quant.fp4, the per-tensor quantiser) fills
# the carrier first (materialise: the plain gather). O_PACKED / DQ_SCATTER / DKV_PACKED: the outputs straight to the
# packed rows (a kernel that does not take the request leaves the aligned buffer, gathered back here). Off: the
# doc_maps call gathers every input and output (pack_rows / unpack_rows) as the repacking path does.
FUSED_GATHER = True
DQ_SCATTER = True   # (with FUSED_GATHER) dQ's post-process writes the caller's packed rows (no dQ gather)
_LAZY: dict = {}   # carrier data_ptr -> (carrier shape, packed source (1, T, h, d), row map (S,) int32)


O_PACKED = True     # (with FUSED_GATHER) the forward writes O to the caller's packed rows (no O gather)
_O_PACKED: list = []     # [maps, used] of the fused doc_maps forward running now (lowprec._routes._o_buf)


def _same_maps(a, b) -> bool:
    """The same maps (lowprec._op rebuilds the documents' objects from their tensors: compared by buffer)."""
    return a is b or (a is not None and b is not None and a.buf.data_ptr() == b.buf.data_ptr()
                      and a.shape_key() == b.shape_key())


def packed_o_rows(docs):
    """(packed starts (N + 1,) int32, T) when the fused doc_maps forward running now wants O in the caller's packed
    rows for these documents, else None. Marks the request taken."""
    if not _O_PACKED or not _same_maps(getattr(docs, "doc_maps", None), _O_PACKED[-1][0]):
        return None
    _O_PACKED[-1][1] = True
    m = _O_PACKED[-1][0]
    return m.cu_q, m.total_q


DKV_PACKED = True   # (with FUSED_GATHER) the 1-CTA backward writes dK / dV to the caller's packed key rows
_DKV_PACKED: list = []   # [maps, used] of the fused doc_maps backward running now (interface_bwd_fp4)


def dkv_packed_rows(docs):
    """T_k when the fused doc_maps backward running now wants dK / dV in the caller's packed key rows for these
    documents (the 1-CTA backward's epilogue), else None. Marks the request taken."""
    if not _DKV_PACKED or not _same_maps(getattr(docs, "doc_maps", None), _DKV_PACKED[-1][0]) or _DKV_PACKED[-1][1]:
        return None
    _DKV_PACKED[-1][1] = True
    return _DKV_PACKED[-1][0].total_k


_DQ_SCATTER: list = []   # [row map (S,) int32, T, used] of the doc_maps backward running now (dq_postprocess_cute)


def dq_scatter_rows(b: int, s_q: int):
    """dq_postprocess_cute: (row map, packed rows) when a fused doc_maps backward is running and the dQ being
    post-processed is its aligned (1, S) buffer, else None."""
    if not _DQ_SCATTER or b != 1 or _DQ_SCATTER[-1][0].shape[0] != s_q or _DQ_SCATTER[-1][2]:
        return None
    _DQ_SCATTER[-1][2] = True
    return _DQ_SCATTER[-1][:2]


def lazy_source(x: torch.Tensor):
    """(packed source, row map) of a registered carrier x ((1, S, h, d), contiguous), else None."""
    e = _LAZY.get(x.data_ptr())
    if e is None or tuple(x.shape) != e[0] or not x.is_contiguous():
        return None
    return e[1], e[2]


def materialise(x: torch.Tensor, gather=None) -> torch.Tensor:
    """Fill a registered carrier with its gathered rows (one row gather) and drop it from the registry; x unchanged
    otherwise. Returns x."""
    g = gather if gather is not None else lazy_source(x)
    if g is None:
        return x
    from flash_attn.cute.lp_varlen import _row_gather, triton
    xs, src = g
    rows = src.shape[0]
    C = x.numel() // rows
    BLOCK = 2048 if C >= 2048 else triton.next_power_of_2(C)
    _row_gather[(rows, triton.cdiv(C, BLOCK))](xs, src, x, C, BLOCK=BLOCK)
    _LAZY.pop(x.data_ptr(), None)
    return x


UNPACK_TRITON = True   # the output gathers (O, dK, dV, and dQ unscattered) by lp_varlen's row gather, not index_select


def _rows_of(x: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """x[dst] (rows), a plain copy either way."""
    if not UNPACK_TRITON:
        return x.index_select(0, dst)
    from flash_attn.cute.lp_varlen import row_gather
    return row_gather(x, dst, dst.shape[0])


def _carrier(maps: DocMaps, x: torch.Tensor, side: str) -> torch.Tensor:
    src, rows = (maps.src_q, maps.padded_size_q) if side == "q" else (maps.src_k, maps.padded_size_k)
    x = x.contiguous()
    c = torch.empty((1, rows) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    _LAZY[c.data_ptr()] = (tuple(c.shape), x.unsqueeze(0), src)
    return c


def _release(*cs):
    for c in cs:
        _LAZY.pop(c.data_ptr(), None)


def fused_ok(plan) -> bool:
    """The plans whose every read of Q / K / V / dO goes through the quantisers (quant.fp8.quantize_fp8_blocks /
    quant.fp4): the token-block E4M3 P.V routes at head_dim 64 / 128, no pre-quantised inputs. Anything else takes the
    plain gathers."""
    from flash_attn.cute.interface_bwd_fp4 import e4_blk32_fp4, op_blk32_fp8
    if plan.prequant or plan.shape.d not in (64, 128) or plan.shape.d_v != plan.shape.d:
        return False
    qs = plan.specs.quant
    if plan.route == "fp8":
        return bool(op_blk32_fp8(qs))
    if plan.route == "fp4.e4m3_pv":
        return bool(e4_blk32_fp4(qs))
    return False


def packed_only_ok(plan, maps) -> bool:
    """(compact maps) The plans the packed rows serve without a fused aligned form (fused_ok is False for them, so full
    maps keep the gathers): the 8-bit token-block route at a padded head dim (96 / 80 / 32 ...: the quantisers stage the
    missing code columns as zeros in the packed rows, the kernels run at 64 / 128); the explicit per-tensor operands
    (operand_scaling="per_tensor": one E4M3 scale per (document, kv head), quant.fp8.quantize_fp8_per_tensor's packed
    documents mode) on the 8-bit route at head_dim 64 / 128 and on the 4-bit E4M3-P.V route at head_dim 128 (its NVFP4
    S operand from quant.fp4's packed rows: the SF bytes as dense per-row words)."""
    if maps is None or not maps.compact or plan.prequant:
        return False
    if plan.route in PACKED_FP4PV_ROUTES:   # (cov5: the 4-bit P.V routes -- fp4pv_packed_ok)
        return plan.shape.d == 128 and plan.shape.d_v == 128
    from flash_attn.cute.interface_bwd_fp4 import op_blk32_fp8
    from flash_attn.cute.fp4_bwd_autograd import padded_head_dim
    d, qs = plan.shape.d, plan.specs.quant
    if plan.shape.d_v != d:
        return False
    if plan.route == "fp8" and not op_blk32_fp8(qs):   # (per-tensor operands)
        return (d in (64, 128) and qs.operand_scaling_fp8 == "per_tensor"
                and plan.specs.host.fp8_forward_kernel in ("split", "fork", "auto"))
    if plan.route == "fp4.e4m3_pv":   # (per-tensor E4M3 operands, the NVFP4 1 x 16 S operand)
        from flash_attn.cute.interface_bwd_fp4 import e4_blk32_fp4
        return (d == 128 and not e4_blk32_fp4(qs) and qs.operand_scaling_fp4 == "per_tensor"
                and int(qs.s_sf_vec) == 16)
    return (plan.route == "fp8" and d not in (64, 128) and d % 16 == 0 and 0 < d < 128
            and padded_head_dim(d) in (64, 128) and bool(op_blk32_fp8(qs)))


def per_tensor_packed(docs):
    """quant.fp8.quantize_fp8_per_tensor(documents=dict(packed=...)) of a packed doc_maps call's per-tensor operands:
    (vt_tile_q, vt_tile, ntiles) of its compact maps, else None."""
    if docs is None or not getattr(docs, "doc_maps_packed", False):
        return None
    m = docs.doc_maps
    return (m.vt_tile_q, m.vt_tile, m.ntiles) if m is not None and m.compact else None


def worklist_cache(maps: DocMaps) -> dict:
    """lp_varlen.varlen_worklist's cache for documents from these maps: per maps buffer, its version (copy_ bumps
    it) and the capture sequence; a fresh dict where none can be kept."""
    key = _capture_key(maps.buf)
    if key is None:
        return {}
    buf = maps.buf
    e = _WL_CACHE.get(id(buf))
    if e is None or e[0]() is not buf:
        e = (weakref.ref(buf, lambda _r, i=id(buf): _WL_CACHE.pop(i, None)), {})
        _WL_CACHE[id(buf)] = e
    per = e[1]
    k2 = (key, maps.buf._version)
    if k2 not in per:
        per.clear()
        per[k2] = {}
    return per[k2]


def fused_forward(forward, plan, st, q, k, v, masks, prequant=None):
    """lowprec._routes.forward of a fused doc_maps call (masks.docs.doc_maps_fused): q / k / v are the caller's
    packed (1, T, h, d) rows; the route's forward (`forward`) runs on registered carriers of the aligned buffer.
    Returns (out (1, T, h, dv), lse (1, h, S)) -- O written to the packed rows by the forward kernel, or gathered."""
    maps = masks.docs.doc_maps
    if getattr(masks.docs, "doc_maps_packed", False):
        return packed_forward(forward, plan, st, q, k, v, masks, prequant)
    st.doc_maps_fused = maps
    cq, ck, cv = _carrier(maps, q[0], "q"), _carrier(maps, k[0], "k"), _carrier(maps, v[0], "k")
    po = [maps, False]
    o_pk = O_PACKED and not plan.fwd.two_cta   # (the 2-CTA forward keeps its TMA O epilogue)
    if o_pk:
        _O_PACKED.append(po)
    try:
        out, lse = forward(plan, st, cq, ck, cv, None, masks)
    finally:
        _release(cq, ck, cv)
        if o_pk:
            _O_PACKED.pop()
    return (out if po[1] else _rows_of(out[0], maps.dst_q).unsqueeze(0)), lse


def packed_backward(backward, plan, st, dout):
    """lowprec._routes.backward of a packed doc_maps call (packed_forward): dout (1, T, h, dv) packed, quantised in the
    packed layout (its D / shifted LSE per aligned row, padding rows +2^120 / +inf); the 1-CTA backward reads the packed codes
    (flash_bwd_sm100_fp4 codes_packed) and writes dK / dV to the packed key rows and dQ through a packed (T) accumulator.
    Returns dq, dk, dv as the caller's packed (1, T, ., .) rows."""
    maps = st.doc_maps_fused
    PACKED_CALLS[1] += 1
    if plan.route in PACKED_FP4PV_ROUTES:   # (cov5: the all-4-bit backward takes the packed rows directly -- fp4q_prepare)
        dq, dk, dv = backward(plan, st, dout)
        assert tuple(dk.shape[:2]) == (1, maps.total_k) and tuple(dq.shape[:2]) == (1, maps.total_q)
        return dq, dk, dv
    key = dout.data_ptr()
    _PACKED[key] = (tuple(dout.shape),) + _pk_entry(maps, "q")
    pkv = [maps, False]
    _DKV_PACKED.append(pkv)
    try:
        dq, dk, dv = backward(plan, st, dout)
    finally:
        _PACKED.pop(key, None)
        _DKV_PACKED.pop()
    assert pkv[1] and tuple(dk.shape[:2]) == (1, maps.total_k) and tuple(dq.shape[:2]) == (1, maps.total_q), \
        "packed rows: the backward must write the packed gradients"
    return dq, dk, dv


def fused_backward(backward, plan, st, dout):
    """lowprec._routes.backward of a fused doc_maps call: dout (1, T, h, dv) packed; returns dq, dk, dv as the
    caller's packed (1, T, ., .) rows (written by the post-process / epilogues, or gathered)."""
    if getattr(st, "doc_maps_packed", False):
        return packed_backward(backward, plan, st, dout)
    maps = st.doc_maps_fused
    cd = _carrier(maps, dout[0], "q")
    scat = [maps.src_q, maps.total_q, False]
    pkv = [maps, False]
    if DQ_SCATTER:
        _DQ_SCATTER.append(scat)
    if DKV_PACKED:
        _DKV_PACKED.append(pkv)
    try:
        dq, dk, dv = backward(plan, st, cd)
    finally:
        _release(cd)
        if DQ_SCATTER:
            _DQ_SCATTER.pop()
        if DKV_PACKED:
            _DKV_PACKED.pop()
    dq = dq if scat[2] else _rows_of(dq[0], maps.dst_q).unsqueeze(0)
    if not pkv[1]:
        dk, dv = _rows_of(dk[0], maps.dst_k).unsqueeze(0), _rows_of(dv[0], maps.dst_k).unsqueeze(0)
    return dq, dk, dv


# --------------------------------------------------------------------------------------------------------- packed rows
# PACKED_ROWS (the default for doc_maps calls; with FUSED_GATHER, on the plans packed_ok accepts -- every other plan keeps
# the fused aligned path): no aligned (S-row) buffer of codes / activations. Q / K / V / dO are quantised IN the packed
# layout (quant.fp8.quantize_fp8_blocks packed_rows=: the codes, the channel-major V codes, the NVFP4 S copies and the
# zeroed dQ accumulator are T rows, row-for-row with the input; D and the shifted LSE (h, S) per aligned row;
# the 32-token blocks and 128-row SF tiles counted from each document's start, the scale factors / tables per aligned
# 128-row tile -- S / 128 tiles, at most T / 128 + N), the forward loads the code tiles at the packed starts
# (flash_fwd_sm100_fp4 codes_packed) and writes O and the LSE (h, T) at the packed rows. A tile's rows past its
# document's end are the next document's (finite codes, masked by the seqlen masks: P = 0, not stored). No carriers.
# PACKED_BWD: the backward too (flash_bwd_sm100_fp4 codes_packed); without it a grad-carrying call keeps the fused path.
# Both on by default (pk3: at or below the fused path's step time at every TABLE A cell, -24 % memory per call); False
# restores the fused aligned path (57cab39), PTX-identical.
PACKED_ROWS = True   # (as FUSED_GATHER: a module switch)
PACKED_BWD = True
# sfd (SF redesign): the packed path's quantisers also write the DENSE per-row tables (quant.fp8 packed_dense: exponent
# rows, NVFP4 SF words, D'' / shifted LSE per packed row) and the 1-CTA backward reads them (flash_bwd_sm100_fp4
# sf_dense) instead of the per-aligned-tile atoms / tables and the (h, S) statistics
SF_DENSE = True
# sfd: the packed forward reads V's ROW codes as its MN-major P.V operand (no channel-major copy, no 16-aligned
# document columns): the forward's V offset is the packed key start
V_ROWS = False
# cov2: the 2-CTA forward (forward_two_cta='on') on the packed rows of compact maps (flash_fwd_sm100_fp4 o_rows_tma:
# each CTA of the pair builds its own SF atoms from the dense tables; O by TMA tiles + the partial tiles' row copies)
PACKED_FWD_2CTA = True
_PACKED: dict = {}   # data_ptr of a packed (1, T, h, d) operand -> (shape, row map src (S,), S)
_KV_PACKED: list = []   # [maps] of the packed doc_maps forward running now (the forward's mask kwargs)
PACKED_CALLS = [0, 0]   # (forward, backward) calls that took the packed rows (tests / probes)


def _pk_entry(maps: DocMaps, side: str) -> tuple:
    """A packed operand's registry entry (quant.fp8 packed_rows): (row map, S, tile table, columns) of full maps, or
    (sfd) (None, S, tile table (2, tiles), T, device tile count (1,)) of compact maps."""
    if maps.compact:
        if side == "q":
            return (None, maps.rows_q, maps.vt_tile_q, maps.total_q, maps.ntiles[0:1])
        return (None, maps.rows_k, maps.vt_tile, maps.total_k, maps.ntiles[1:2])
    if side == "q":
        return (maps.src_q, maps.padded_size_q, maps.vt_tile_q, maps.vt_cols_q)
    return (maps.src_k, maps.padded_size_k, maps.vt_tile, maps.vt_cols)


def compact_operand(x: torch.Tensor) -> bool:
    """x is a registered packed operand of compact maps (sfd: no per-tile factor / exponent tables)."""
    e = packed_rows_of(x) if _PACKED else None
    return e is not None and e[0] is None


def packed_rows_of(x: torch.Tensor):
    """(src, S) of a registered packed operand x ((1, T, h, d), contiguous), else None."""
    e = _PACKED.get(x.data_ptr())
    if e is None or tuple(x.shape) != e[0] or not x.is_contiguous():
        return None
    return e[1:]


def aligned_rows(x: torch.Tensor) -> int:
    """The aligned row count S of the per-tile tables of operand x: S for a registered packed operand, else x.shape[1]."""
    if _PACKED:
        e = packed_rows_of(x)
        if e is not None:
            return e[1]
    return x.shape[1]


def packed_kv_rows(docs):
    """(packed key starts (N + 1,), the channel-major V's starts cu_vt (N + 1,)) when the packed doc_maps forward
    running now is these documents', else None."""
    if not _KV_PACKED or not _same_maps(getattr(docs, "doc_maps", None), _KV_PACKED[-1]):
        return None
    if V_ROWS or _KV_PACKED[-1].compact:
        return _KV_PACKED[-1].cu_k, _KV_PACKED[-1].cu_k
    return _KV_PACKED[-1].cu_k, _KV_PACKED[-1].cu_vt


def packed_ok(plan, masks, requires_grad: bool) -> bool:
    """The calls the packed rows serve: the fused plans (fused_ok) on the default token-block forward / backward --
    no tcgen05 rotation, token-local operands, fold forward, MXFP4 S operand, 16 x 16 Q blocks, 2-CTA forward; a
    grad-carrying call needs PACKED_BWD and the 1-CTA backward (on compact maps also with ds_dq_rowlocal and without the
    dO launch's D fold). Everything else keeps the aligned layout (the fused carriers or the gathers).
    A window / softcap / score_mod / mask_mod runs on the packed rows too (cov4): the masks are evaluated in each
    document's coordinates, as on the aligned rows; a key past its document only occurs in the document's last key
    tile, which is a masked tile (the key-boundary mask, after any score_mod), and a padding query row gets LSE = +inf
    / D'' = +2^120 from its statistics row, so P = dS = +0 there whatever the mods make of its S. A window / mask_mod
    can leave a key tile with no query: its packed dK / dV rows start zero (interface_bwd_fp4, as the aligned rows)."""
    _dm = getattr(getattr(masks, "docs", None), "doc_maps", None)
    if not (PACKED_ROWS and FUSED_GATHER and (fused_ok(plan) or packed_only_ok(plan, _dm))):
        return False
    if plan.route in PACKED_FP4PV_ROUTES:
        return fp4pv_packed_ok(plan, masks, requires_grad)
    if _dm is not None and _dm.compact and not SF_DENSE:
        return False   # (compact maps run only the sfd packed path; otherwise their full maps' fused path)
    qs, f = plan.specs.quant, plan.fwd
    from flash_attn.cute.fp4_bwd_autograd import tb_fold_fwd, q_blk2d
    from flash_attn.cute.interface_bwd_fp4 import mx_cols
    from flash_attn.cute import qk_hadamard as QH
    compact = _dm is not None and _dm.compact
    if qs.qk_hadamard and QH.rotation_method(qs) not in (None, "tensor_core") and not compact:
        return False   # (the rotated quantiser: the tensor-core token-block kernel; cov3: tcgen05 on compact maps)
    if f.operand_scaling == "token_local" and not (compact and packed_d_fold(qs, plan.route == "fp8")
                                                   and not plan.fwd.two_cta):
        # (cov3: token-local operands on compact maps -- the fold forward and the tables per packed row. The
        # deterministic backward too (hangfix): its intermittent deadlock was the dQ issue warp's lagging-lane hang,
        # closed by the sync_warp in flash_bwd_sm100_fp4.dQ_issue)
        return False
    if int(qs.s_sf_vec) == 32 and plan.route != "fp8":
        return False   # (the MXFP4 S operand: the 4-bit routes refuse documents; the 8-bit route never reads s_sf_vec)
    if q_blk2d(qs) and not compact:
        return False   # (cov3: the 16 x 16 Q blocks from quant.fp4's packed rows on compact maps; full maps keep theirs)
    if (tb_fold_fwd(mx_cols(masks, quant=qs, host=plan.specs.host), qs) and f.operand_scaling != "token_local"):
        return False
    if plan.fwd.two_cta and not (PACKED_FWD_2CTA and _dm is not None and _dm.compact and SF_DENSE):
        return False   # (cov2: the pair on packed rows reads the sf_dense tables of compact maps only)
    if requires_grad and not PACKED_BWD:
        return False
    if requires_grad and plan.bwd.ds_dq_rowlocal != "off" and not compact:
        return False   # (cov3: the row-local dQ-side dS exponents on compact maps' sf_dense backward; full maps keep theirs)
    if requires_grad and not packed_d_fold(qs, plan.route == "fp8"):
        # the packed backward's padding statistics rows: with the dO launch's D fold D'' = +2^120 (dS = +0 x (b dP +
        # 2^120)); cov3: unfolded (ds_fold=False) D = -2^100 on compact maps' sf_dense backward (dS = +0 x (dP - s D))
        if not (compact and _blk_ds(qs, plan.route == "fp8")):
            return False
    return True


# cov5: the 4-bit P.V routes on the packed rows of compact maps (no full-map fallback): option A (pv_precision='fp4',
# inference) -- the all-FP4 forward with V^T realigned from the key-parity copies by the load warp
# (flash_attn.cute.v4_packed), Q / K / V quantised in the packed rows, the per-key / per-tile tables per packed key /
# per tile of the maps
PACKED_FP4PV_ROUTES = ("fp4.fp4_pv", "fp4.all_fp4")


def fp4pv_packed_ok(plan, masks, requires_grad: bool) -> bool:
    """(cov5) The 4-bit P.V calls the packed rows serve: compact maps (sf_dense), head_dim 128, documents only (the plan
    refuses every other mask on these routes), the 1-CTA forward."""
    _dm = getattr(getattr(masks, "docs", None), "doc_maps", None)
    if _dm is None or not _dm.compact or not SF_DENSE or plan.fwd.two_cta:
        return False
    if masks is not None and (masks.local or masks.any_mod or getattr(masks, "softcap", None)):
        return False
    if plan.route == "fp4.fp4_pv":
        return not requires_grad
    if plan.route == "fp4.all_fp4":
        from flash_attn.cute import interface_bwd_fp4q as iq
        qs, bq = plan.specs.quant, plan.specs.bwdq
        if iq.fp4all_stab_mode(qs) not in ("sr", "rtn"):
            return False
        if not plan.fwd.causal:   # (cov6: the dense route -- the forward's packed K^T 1 x 16 copy, the dense backward)
            if not (PACKED_ALL4_DENSE and iq.fp4all_vtok_mode(causal=False, qspec=qs) == "off"):
                return False
            return not requires_grad or iq.packed_bwd_dense_ok(qs, bq)
        if not (PACKED_ALL4_FWD and iq.fp4all_vtok_mode(causal=True, qspec=qs) == "diag"):
            return False
        return not requires_grad or (PACKED_ALL4_BWD and iq.packed_bwd_ok(qs, bq))
    return False


PACKED_ALL4_FWD = True    # (cov5: the causal all-4-bit forward on the packed rows)
PACKED_ALL4_BWD = True    # (cov5: its backward -- flash_bwd_sm100_fp4q codes_packed, interface_bwd_fp4q.fp4q_prepare_packed)
# cov6: the dense (non-causal) all-4-bit forward and backward on the packed rows (fp4all_quantize_qkv packed backward
# operands, interface_bwd_fp4q._fp4q_prepare_packed_dense)
PACKED_ALL4_DENSE = True


def packed_tiles(x: torch.Tensor):
    """(cov5) (tile table (2, tiles), device tile count (1,)) of a registered packed operand of compact maps (the
    quantisers' packed_rows=), else None."""
    e = packed_rows_of(x) if _PACKED else None
    if e is None or e[0] is not None:
        return None
    return e[2], e[4]


def _blk_ds(qs, fp8: bool) -> bool:
    """The block-scaled dS (the packed backward's only dS: no per-tensor dS bound)."""
    from flash_attn.cute.interface_bwd_fp4 import ds_blk32_fp4
    return bool((qs.ds_scaling_fp8 != "per_tensor") if fp8 else ds_blk32_fp4(quant=qs))


def packed_d_fold(qs, fp8: bool) -> bool:
    """The dO launch folds D (D'' = m D, block-scaled dS with ds_fold): what the packed backward's padding statistics
    rows rely on (a padding query's D'' = +2^120, so dS = +0 x (b dP + D'') = +0 with no per-element emulation)."""
    from flash_attn.cute.interface_bwd_fp4 import ds_blk32_fp4
    blk = (qs.ds_scaling_fp8 != "per_tensor") if fp8 else ds_blk32_fp4(quant=qs)
    return bool(blk and qs.ds_fold)


def pair_as_one_cta(plan):
    """(cov2) The plan with the 1-CTA forward in place of the pair (forward_two_cta='on'): at a large capacity
    (large_capacity: the fork's N = T) the pair's one-cluster-per-work-tile grid is sized by the static bound
    (T + 511 N rows) and every cluster decodes its tile by a prefix scan over the N documents -- 10.8 s per step at
    T = N = 65536 (lognorm documents, h16/hkv4 d128) -- while the 1-CTA packed forward claims from its device work list.
    O / LSE are bitwise the pair's (the pair is the same step on two CTAs; t2cta / test_pair_capacity_t)."""
    import dataclasses
    sp = plan.specs
    return dataclasses.replace(plan, fwd=dataclasses.replace(plan.fwd, two_cta=False),
                               specs=dataclasses.replace(sp, host=dataclasses.replace(sp.host, fwd_two_cta=False)))


def packed_forward(forward, plan, st, q, k, v, masks, prequant=None):
    """lowprec._routes.forward of a packed doc_maps call: q / k / v are the caller's packed (1, T, h, d) rows and the
    route runs on them directly (registered: the quantisers take packed_rows=, the per-tile tables S rows). Returns
    (out (1, T, h, dv), lse (1, h, T))."""
    maps = masks.docs.doc_maps
    st.doc_maps_fused = maps
    st.doc_maps_packed = True
    PACKED_CALLS[0] += 1
    if plan.fwd.two_cta and large_capacity(maps):
        plan = pair_as_one_cta(plan)
    regs = [(q, _pk_entry(maps, "q")), (k, _pk_entry(maps, "k")), (v, _pk_entry(maps, "k"))]
    for x, e in regs:
        _PACKED[x.data_ptr()] = (tuple(x.shape),) + e
    po = [maps, False]
    _O_PACKED.append(po)
    _KV_PACKED.append(maps)
    try:
        out, lse = forward(plan, st, q, k, v, prequant, masks)
    finally:
        for x, _ in regs:
            _PACKED.pop(x.data_ptr(), None)
        _O_PACKED.pop()
        _KV_PACKED.pop()
    assert po[1] and tuple(out.shape[:2]) == (1, maps.total_q), "packed rows: the forward must write the packed O"
    return out, lse


def pack_rows(maps: DocMaps, x: torch.Tensor, side: str) -> torch.Tensor:
    """Differentiable gather of a packed (T, h, d) tensor into the aligned (S, h, d) layout (zero padding rows)."""
    from flash_attn.cute.lp_varlen import _Pack
    src, dst, rows = (maps.src_q, maps.dst_q, maps.padded_size_q) if side == "q" else \
        (maps.src_k, maps.dst_k, maps.padded_size_k)
    return _Pack.apply(x, src, dst, rows)


def unpack_rows(maps: DocMaps, x: torch.Tensor, side: str) -> torch.Tensor:
    """Differentiable inverse of pack_rows: (S, h, d) -> (T, h, d)."""
    from flash_attn.cute.lp_varlen import _Unpack
    src, dst, rows = (maps.src_q, maps.dst_q, maps.padded_size_q) if side == "q" else \
        (maps.src_k, maps.dst_k, maps.padded_size_k)
    return _Unpack.apply(x, src, dst, rows)


# ---------------------------------------------------------------------------------------------------------------- CPU
def _cu_np(cu, name):
    if isinstance(cu, torch.Tensor):
        cu = cu.detach().cpu().numpy()
    cu = np.asarray(cu, dtype=np.int64).reshape(-1)
    if cu.shape[0] < 2 or cu[0] != 0 or np.any(np.diff(cu) < 0):
        raise ValueError(f"{name} must be (n + 1,) non-decreasing packed starts with {name}[0] = 0 and n >= 1")
    return cu


def _side_np(cu, n_cap, total, s_pad, align, name):
    n = cu.shape[0] - 1
    if n > n_cap:
        raise ValueError(f"{n} documents exceed max_num_documents = {n_cap}")
    if total is None:
        total = int(cu[-1])
    if int(cu[-1]) != total:
        raise ValueError(f"{name}[-1] = {int(cu[-1])} must equal the packed length {total}")
    cu = np.concatenate([cu, np.full(n_cap - n, cu[-1], dtype=np.int64)])
    ln = np.diff(cu)
    cu_pad = np.zeros(n_cap + 1, dtype=np.int64)
    cu_pad[1:] = np.cumsum((ln + align - 1) // align * align)
    if s_pad is None:
        s_pad = default_padded_size(total, n_cap, align)
    if s_pad % 128 or s_pad < cu_pad[-1]:
        raise ValueError(f"padded size {s_pad} must be a multiple of 128 holding the aligned documents "
                         f"({int(cu_pad[-1])} rows)")
    doc = np.repeat(np.arange(n_cap, dtype=np.int64), ln)
    tok = np.arange(total, dtype=np.int64)
    dst = cu_pad[doc] + (tok - cu[doc])
    src = np.full(s_pad, -1, dtype=np.int64)
    src[dst] = tok
    # the aligned prep's tables (lp_varlen._aligned_prep): the document of each 128-row tile (the sentinel n_cap past
    # its document's true length), the end of that document's real rows, the valid rows
    t0 = np.arange(0, s_pad, _TILE, dtype=np.int64)
    d = np.searchsorted(cu_pad[1:], t0, side="right")
    dc = np.minimum(d, n_cap - 1)
    e = np.where(d < n_cap, cu_pad[dc] + ln[dc], 0)
    tile_doc = np.where(t0 < e, d, n_cap)
    valid = np.arange(s_pad, dtype=np.int64) < np.repeat(e, _TILE)
    cu_vt = np.zeros(n_cap + 1, dtype=np.int64)
    cu_vt[1:] = np.cumsum((ln + 15) // 16 * 16)
    real = tile_doc < n_cap
    vt_tile = np.stack([np.where(real, cu_vt[dc] + (t0 - cu_pad[dc]), -1),
                        np.where(real, np.clip(e - t0, 0, _TILE), 0)])
    nz = np.nonzero(vt_tile[1] > 0)[0]
    if nz.size:   # the last real tile also zeroes its 128 columns past cu_vt[-1] (never left unwritten: TMA reads them)
        vt_tile[1, nz[-1]] += VT_LAST
    return dict(cu=cu, ln=ln, cu_pad=cu_pad, dst=dst, src=src, tile_doc=tile_doc, row_end=e, valid=valid,
                total=total, s_pad=s_pad, n=n, cu_vt=cu_vt, vt_tile=vt_tile)


SKEW_CLC_MIN = 4.0   # CLC vs static, non-causal forward, 12 shapes (REPORT): lost at skew 1.0 (-0.1..-1.5 %) and 1.88 (-0.5..-1.0 %); won at 2.88 (+0.7..+1.5 %) and 3.25 (+1.1 %) but not at 3.38 (64k log-uniform: -0.2..-1.0 % in-process, within noise in the kernel-alone row), won at 16.5 and 28 (+4..+10 %). 4.0 keeps every measured row at or above static


def length_skew(lens) -> Optional[float]:
    """Host statistic of a CPU build: longest / mean length over the non-empty documents (1.0 = equal lengths)."""
    nz = np.asarray(lens)
    nz = nz[nz > 0]
    return round(float(nz.max()) * nz.size / float(nz.sum()), 4) if nz.size else None


def _assemble(n_cap, qd, kd, flags, max_q, max_k, align, num_docs, host_flags) -> DocMaps:
    lay, words = _layout(n_cap, qd["total"], kd["total"], qd["s_pad"], kd["s_pad"])
    buf = np.zeros(words, dtype=np.int32)

    def put(nm, val):
        off, shp = lay[nm]
        buf[off:off + int(np.prod(shp))] = np.asarray(val).reshape(-1)
    tab = np.zeros((6, n_cap + 1), dtype=np.int64)
    tab[0], tab[1] = qd["cu_pad"], kd["cu_pad"]
    tab[2, :n_cap], tab[3, :n_cap] = qd["ln"], kd["ln"]
    tab[4], tab[5] = qd["cu"], kd["cu"]
    for nm, val in (("cu_q", qd["cu"]), ("cu_k", kd["cu"]), ("cu_pad_q", qd["cu_pad"]), ("cu_pad_k", kd["cu_pad"]),
                    ("len_q", qd["ln"]), ("len_k", kd["ln"]), ("doc_tab", tab), ("dst_q", qd["dst"]),
                    ("dst_k", kd["dst"]), ("src_q", qd["src"]), ("src_k", kd["src"]), ("tile_doc_q", qd["tile_doc"]),
                    ("tile_doc_k", kd["tile_doc"]), ("row_end_q", qd["row_end"]), ("row_end_k", kd["row_end"]),
                    ("flags", [flags]), ("cu_vt", kd["cu_vt"]), ("vt_tile", kd["vt_tile"]),
                    ("cu_vt_q", qd["cu_vt"]), ("vt_tile_q", qd["vt_tile"])):
        put(nm, val)
    off, shp = lay["q_valid"]
    buf[off:off + shp[0]] = qd["valid"].astype(np.uint8).view(np.int32)
    return DocMaps(qd["total"], kd["total"], n_cap, qd["s_pad"], kd["s_pad"], int(max_q), int(max_k), align,
                   torch.from_numpy(buf), num_documents=num_docs, host_flags=host_flags)


def _build_cpu(cu_q, cu_k, n_cap, total_q, total_k, s_q, s_k, max_q, max_k, align) -> DocMaps:
    cq = _cu_np(cu_q, "cu_seqlens_q")
    ck = cq if cu_k is None else _cu_np(cu_k, "cu_seqlens_k")
    if cq.shape != ck.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same number of documents")
    n = cq.shape[0] - 1
    n_cap = n if n_cap is None else int(n_cap)
    qd = _side_np(cq, n_cap, total_q, s_q, align, "cu_seqlens_q")
    kd = qd if (cu_k is None and total_k in (None, total_q) and s_k in (None, s_q)) else \
        _side_np(ck, n_cap, total_k, s_k, align, "cu_seqlens_k")
    lq, lk = qd["ln"], kd["ln"]
    mq, mk = int(lq.max(initial=0)), int(lk.max(initial=0))
    max_q = mq if max_q is None else int(max_q)
    max_k = mk if max_k is None else int(max_k)
    if mq > max_q or mk > max_k:
        raise ValueError(f"max_seqlen_q / _k = {max_q} / {max_k} below the longest document ({mq} / {mk})")
    flags = (FLAG_KEYS_ANY if not np.any((lq > 0) & (lk == 0)) else 0) | \
        (FLAG_KEYS_CAUSAL if not np.any(lq > lk) else 0)
    m = _assemble(n_cap, qd, kd, flags, max_q, max_k, align, n, flags)
    m.sched_skew = length_skew(lk)
    m.sched_skewed = None if m.sched_skew is None else bool(m.sched_skew >= SKEW_CLC_MIN)
    return m


# ---------------------------------------------------------------------------------------------------------------- GPU
def _build_gpu(cu_q, cu_k, n_cap, total_q, total_k, s_q, s_k, max_q, max_k, align) -> DocMaps:
    """GPU build from CUDA cu_seqlens: torch ops plus the aligned prep launch, no host read."""
    from flash_attn.cute import lp_varlen as LV
    if total_q is None or max_q is None or (cu_k is not None and (total_k is None or max_k is None)):
        raise ValueError("DocMaps from CUDA cu_seqlens: total_q / total_k and max_seqlen_q / max_seqlen_k (host ints) "
                         "must be given (no host read)")
    cu_k = cu_q if cu_k is None else cu_k
    total_k = total_q if total_k is None else total_k
    max_k = max_q if max_k is None else max_k
    for nm, cu in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
        if cu.dtype != torch.int32 or cu.dim() != 1:
            raise ValueError(f"{nm} must be (n + 1,) int32")
    n = cu_q.shape[0] - 1
    n_cap = n if n_cap is None else int(n_cap)
    if n > n_cap or cu_k.shape[0] != cu_q.shape[0]:
        raise ValueError("cu_seqlens_q / _k: equal document counts, at most max_num_documents")
    s_q = default_padded_size(total_q, n_cap, align) if s_q is None else int(s_q)
    s_k = default_padded_size(total_k, n_cap, align) if s_k is None else int(s_k)
    if s_q % 128 or s_k % 128:
        raise ValueError("padded sizes must be multiples of 128")
    lay, words = _layout(n_cap, total_q, total_k, s_q, s_k)
    dev = cu_q.device
    buf = torch.zeros(words, dtype=torch.int32, device=dev)
    m = DocMaps(total_q, total_k, n_cap, s_q, s_k, int(max_q), int(max_k), align, buf)

    def side(cu, total, s_pad, nm_cu, nm_pad, nm_len, nm_dst, nm_src):
        c = getattr(m, nm_cu)
        c[:n + 1].copy_(cu)
        c[n + 1:].copy_(cu[n:].expand(n_cap - n))
        ln = c[1:] - c[:-1]
        getattr(m, nm_len).copy_(ln)
        pad = getattr(m, nm_pad)
        pad[1:].copy_(torch.cumsum((ln + (align - 1)) // align * align, 0, dtype=torch.int32))
        tok = torch.arange(total, dtype=torch.int32, device=dev)
        doc = torch.searchsorted(c[1:].contiguous(), tok, right=True).clamp_(max=n_cap - 1)
        dst = pad[doc] + (tok - c[doc])
        getattr(m, nm_dst).copy_(dst)
        src = getattr(m, nm_src)
        src.fill_(-1)
        src.index_copy_(0, dst.long(), tok)
    side(cu_q, total_q, s_q, "cu_q", "cu_pad_q", "len_q", "dst_q", "src_q")
    side(cu_k, total_k, s_k, "cu_k", "cu_pad_k", "len_k", "dst_k", "src_k")
    # the kernel tables: the aligned prep on the padded starts / true lengths (lp_varlen._aligned_prep), one launch
    tab = m.doc_tab
    tiles_q, tiles_k = s_q // _TILE, s_k // _TILE
    rend = torch.empty(tiles_q + tiles_k, dtype=torch.int32, device=dev)
    ok = torch.empty(1, dtype=torch.int32, device=dev)
    q_valid = m.q_valid
    n_prog = max(LV.triton.cdiv(s_q, LV._PREP_ROWS), LV.triton.cdiv(tiles_k, LV._PREP_COLS),
                 LV.triton.cdiv(n_cap + 1, LV._PREP_COLS))
    steps = max(1, int(n_cap).bit_length() + 1)
    LV._aligned_prep[(n_prog,)](m.cu_pad_q, m.cu_pad_k, m.len_q, m.len_k, tab, m.tile_doc_k, m.tile_doc_q, rend,
                                q_valid, ok, n_cap, tiles_q, tiles_k, s_q, s_k, align, steps,
                                ROWS=LV._PREP_ROWS, COLS=LV._PREP_COLS)
    m.row_end_q.copy_(rend[:tiles_q])
    m.row_end_k.copy_(rend[tiles_q:])
    # the packed channel-major V's / the queries' statistics tables (documents at multiples of 16 rows)
    for ln_, cvt, tdoc, rend_, cpad, vt, s_x, tiles_x in (
            (m.len_k, m.cu_vt, m.tile_doc_k, m.row_end_k, m.cu_pad_k, m.vt_tile, s_k, tiles_k),
            (m.len_q, m.cu_vt_q, m.tile_doc_q, m.row_end_q, m.cu_pad_q, m.vt_tile_q, s_q, tiles_q)):
        cvt[1:].copy_(torch.cumsum((ln_ + 15) // 16 * 16, 0, dtype=torch.int32))
        tdk = tdoc.long()
        real = tdk < n_cap
        dc_ = tdk.clamp(max=n_cap - 1)
        t0_ = torch.arange(0, s_x, _TILE, dtype=torch.int32, device=dev)
        vt[0].copy_(torch.where(real, cvt[dc_] + (t0_ - cpad[dc_]), -1))
        vt[1].copy_(torch.where(real, (rend_ - t0_).clamp(0, _TILE), 0))
        ar_ = torch.arange(tiles_x, dtype=torch.int32, device=dev)
        last_ = torch.where(vt[1] > 0, ar_, -1).max()
        vt[1].add_(torch.where(ar_ == last_, VT_LAST, 0).to(torch.int32))
    tab[4].copy_(m.cu_q)
    tab[5].copy_(m.cu_k)
    lq, lk = m.len_q, m.len_k
    flags = torch.where(((lq > 0) & (lk == 0)).any(), 0, FLAG_KEYS_ANY) + \
        torch.where((lq > lk).any(), 0, FLAG_KEYS_CAUSAL)
    m.flags.copy_(flags.to(torch.int32).view(1))
    # the allocation must hold the aligned documents, every length within max_seqlen, cu non-decreasing from 0
    good = (m.cu_pad_q[-1] <= s_q) & (m.cu_pad_k[-1] <= s_k) & (lq.max() <= max_q) & (lk.max() <= max_k) & \
        (lq.min() >= 0) & (lk.min() >= 0) & (m.cu_q[0] == 0) & (m.cu_k[0] == 0) & (m.cu_q[-1] == total_q) & \
        (m.cu_k[-1] == total_k)
    torch._assert_async(good, "DocMaps: cu_seqlens must be non-decreasing from 0 to the packed length, every document "
                              "within max_seqlen, the aligned documents within the padded size")
    return m



# ------------------------------------------------------------------------------------------------------- compact (sfd)
def _compact_side_np(cu, n_cap, total, name):
    n = cu.shape[0] - 1
    if n > n_cap:
        raise ValueError(f"{n} documents exceed max_num_documents = {n_cap}")
    if total is None:
        total = int(cu[-1])
    if int(cu[-1]) != total:
        raise ValueError(f"{name}[-1] = {int(cu[-1])} must equal the packed length {total}")
    cu = np.concatenate([cu, np.full(n_cap - n, cu[-1], dtype=np.int64)])
    ln = np.diff(cu)
    cu_pad = np.zeros(n_cap + 1, dtype=np.int64)
    cu_pad[1:] = np.cumsum((ln + _TILE - 1) // _TILE * _TILE)
    tb = tile_bound(total, n_cap)
    doc = np.repeat(np.arange(n_cap, dtype=np.int64), ln)
    tok = np.arange(total, dtype=np.int64)
    dst = cu_pad[doc] + (tok - cu[doc])
    t0 = np.arange(0, tb * _TILE, _TILE, dtype=np.int64)
    d = np.searchsorted(cu_pad[1:], t0, side="right")
    dc = np.minimum(d, n_cap - 1)
    e = np.where(d < n_cap, cu_pad[dc] + ln[dc], 0)
    real = (d < n_cap) & (t0 < e)
    tile_doc = np.where(real, d, n_cap)
    vt = np.stack([np.where(real, cu[dc] + (t0 - cu_pad[dc]), 0), np.where(real, np.clip(e - t0, 0, _TILE), 0)])
    return dict(cu=cu, ln=ln, cu_pad=cu_pad, dst=dst, tile_doc=tile_doc, row_end=e, vt=vt, total=total, tb=tb,
                ntiles=int(cu_pad[-1]) // _TILE)


def _build_compact_cpu(cu_q, cu_k, n_cap, total_q, total_k, max_q, max_k) -> DocMaps:
    cq = _cu_np(cu_q, "cu_seqlens_q")
    ck = cq if cu_k is None else _cu_np(cu_k, "cu_seqlens_k")
    if cq.shape != ck.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same number of documents")
    n = cq.shape[0] - 1
    n_cap = n if n_cap is None else int(n_cap)
    qd = _compact_side_np(cq, n_cap, total_q, "cu_seqlens_q")
    kd = qd if (cu_k is None and total_k in (None, total_q)) else _compact_side_np(ck, n_cap, total_k, "cu_seqlens_k")
    lq, lk = qd["ln"], kd["ln"]
    mq, mk = int(lq.max(initial=0)), int(lk.max(initial=0))
    max_q = mq if max_q is None else int(max_q)
    max_k = mk if max_k is None else int(max_k)
    if mq > max_q or mk > max_k:
        raise ValueError(f"max_seqlen_q / _k = {max_q} / {max_k} below the longest document ({mq} / {mk})")
    flags = (FLAG_KEYS_ANY if not np.any((lq > 0) & (lk == 0)) else 0) | \
        (FLAG_KEYS_CAUSAL if not np.any(lq > lk) else 0)
    sq, sk = qd["tb"] * _TILE, kd["tb"] * _TILE
    lay, words = _layout(n_cap, qd["total"], kd["total"], sq, sk, True)
    buf = np.zeros(words, dtype=np.int32)

    def put(nm, val):
        off, shp = lay[nm]
        buf[off:off + int(np.prod(shp))] = np.asarray(val).reshape(-1)
    tab = np.zeros((6, n_cap + 1), dtype=np.int64)
    tab[0], tab[1] = qd["cu_pad"], kd["cu_pad"]
    tab[2, :n_cap], tab[3, :n_cap] = qd["ln"], kd["ln"]
    tab[4], tab[5] = qd["cu"], kd["cu"]
    for nm, val in (("cu_q", qd["cu"]), ("cu_k", kd["cu"]), ("cu_pad_q", qd["cu_pad"]), ("cu_pad_k", kd["cu_pad"]),
                    ("len_q", qd["ln"]), ("len_k", kd["ln"]), ("doc_tab", tab), ("dst_q", qd["dst"]),
                    ("dst_k", kd["dst"]), ("tile_doc_q", qd["tile_doc"]), ("tile_doc_k", kd["tile_doc"]),
                    ("row_end_q", qd["row_end"]), ("row_end_k", kd["row_end"]), ("flags", [flags]),
                    ("vt_tile", kd["vt"]), ("vt_tile_q", qd["vt"]), ("ntiles", [qd["ntiles"], kd["ntiles"]])):
        put(nm, val)
    return DocMaps(qd["total"], kd["total"], n_cap, 0, 0, int(max_q), int(max_k), _TILE, torch.from_numpy(buf),
                   num_documents=n, host_flags=flags, sched_skew=length_skew(lk),
                   sched_skewed=(None if length_skew(lk) is None else bool(length_skew(lk) >= SKEW_CLC_MIN)))   # (padded sizes 0: compact)


def _build_compact_gpu(cu_q, cu_k, n_cap, total_q, total_k, max_q, max_k) -> DocMaps:
    """The compact maps from CUDA cu_seqlens: torch ops only, no host read (static shapes from T, N)."""
    if total_q is None or max_q is None or (cu_k is not None and (total_k is None or max_k is None)):
        raise ValueError("DocMaps from CUDA cu_seqlens: total_q / total_k and max_seqlen_q / max_seqlen_k (host ints) "
                         "must be given (no host read)")
    cu_k = cu_q if cu_k is None else cu_k
    total_k = total_q if total_k is None else total_k
    max_k = max_q if max_k is None else max_k
    for nm, cu in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
        if cu.dtype != torch.int32 or cu.dim() != 1:
            raise ValueError(f"{nm} must be (n + 1,) int32")
    n = cu_q.shape[0] - 1
    n_cap = n if n_cap is None else int(n_cap)
    if n > n_cap or cu_k.shape[0] != cu_q.shape[0]:
        raise ValueError("cu_seqlens_q / _k: equal document counts, at most max_num_documents")
    tbq, tbk = tile_bound(total_q, n_cap), tile_bound(total_k, n_cap)
    sq, sk = tbq * _TILE, tbk * _TILE
    lay, words = _layout(n_cap, total_q, total_k, sq, sk, True)
    dev = cu_q.device
    buf = torch.zeros(words, dtype=torch.int32, device=dev)
    m = DocMaps(total_q, total_k, n_cap, 0, 0, int(max_q), int(max_k), _TILE, buf)   # (padded sizes 0: compact)

    def side(cu, total, tb, nm_cu, nm_pad, nm_len, nm_dst, nm_tdoc, nm_rend, nm_vt, ti):
        c = getattr(m, nm_cu)
        c[:n + 1].copy_(cu)
        c[n + 1:].copy_(cu[n:].expand(n_cap - n))
        ln = c[1:] - c[:-1]
        getattr(m, nm_len).copy_(ln)
        pad = getattr(m, nm_pad)
        pad[1:].copy_(torch.cumsum((ln + (_TILE - 1)) // _TILE * _TILE, 0, dtype=torch.int32))
        tok = torch.arange(total, dtype=torch.int32, device=dev)
        doc = torch.searchsorted(c[1:].contiguous(), tok, right=True).clamp_(max=n_cap - 1)
        getattr(m, nm_dst).copy_(pad[doc] + (tok - c[doc]))
        t0 = torch.arange(0, tb * _TILE, _TILE, dtype=torch.int32, device=dev)
        d = torch.searchsorted(pad[1:].contiguous(), t0, right=True)
        dc = d.clamp(max=n_cap - 1)
        e = torch.where(d < n_cap, pad[dc] + ln[dc], 0)
        real = (d < n_cap) & (t0 < e)
        getattr(m, nm_tdoc).copy_(torch.where(real, d, n_cap))
        getattr(m, nm_rend).copy_(e)
        vt = getattr(m, nm_vt)
        vt[0].copy_(torch.where(real, c[dc] + (t0 - pad[dc]), 0))
        vt[1].copy_(torch.where(real, (e - t0).clamp(0, _TILE), 0))
        m.ntiles[ti].copy_(pad[-1] // _TILE)
    side(cu_q, total_q, tbq, "cu_q", "cu_pad_q", "len_q", "dst_q", "tile_doc_q", "row_end_q", "vt_tile_q", 0)
    side(cu_k, total_k, tbk, "cu_k", "cu_pad_k", "len_k", "dst_k", "tile_doc_k", "row_end_k", "vt_tile", 1)
    tab = m.doc_tab
    tab[0].copy_(m.cu_pad_q)
    tab[1].copy_(m.cu_pad_k)
    tab[2, :n_cap].copy_(m.len_q)
    tab[3, :n_cap].copy_(m.len_k)
    tab[4].copy_(m.cu_q)
    tab[5].copy_(m.cu_k)
    lq, lk = m.len_q, m.len_k
    flags = torch.where(((lq > 0) & (lk == 0)).any(), 0, FLAG_KEYS_ANY) + \
        torch.where((lq > lk).any(), 0, FLAG_KEYS_CAUSAL)
    m.flags.copy_(flags.to(torch.int32).view(1))
    good = (lq.max() <= max_q) & (lk.max() <= max_k) & (lq.min() >= 0) & (lk.min() >= 0) & (m.cu_q[0] == 0) & \
        (m.cu_k[0] == 0) & (m.cu_q[-1] == total_q) & (m.cu_k[-1] == total_k)
    torch._assert_async(good, "DocMaps: cu_seqlens must be non-decreasing from 0 to the packed length, every document "
                              "within max_seqlen")
    return m

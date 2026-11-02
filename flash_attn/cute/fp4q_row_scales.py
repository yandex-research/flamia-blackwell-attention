# The all-FP4 backward's dense (non-causal) per-query-row boost, one small CuTe launch over the (b, h, s) row vectors
# (interface_bwd_fp4q.fp4q_prepare, global dS multiplier with the row boost; it replaced ~20 torch launches):
#   r_q   = 2^min(floor(log2(max(bound / max(bound_q, 1e-30), 1))), cap)   (bound = max_q bound_q, from the dO launch)
#   uinv' = uinv r_q
#   and, reduced over all rows by the launch's last tile, the per-tensor E4M3 amaxes of the dK / dV B operands
#   max |bf16(Q / r_q)| and max |bf16(dO / r_q)| from the forward's max |Q| per row and the dO launch's max |dO| per row.
# Every op is torch's fp32 op in the same order (see fp4q_prepare's torch reference path); r_q is a power of two.
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cuda.bindings.driver as cuda
from typing import Optional

from flash_attn.cute.quant import fp4 as F


class _DenseRowBoostKernel:
    def __init__(self, cap: float, docs: bool = False):
        self.cap = float(cap)
        self.threads = 128
        # aligned varlen documents (b = 1): the bound and the amaxes per document of the row's 128-row tile
        self.docs = bool(docs)

    @cute.jit
    def __call__(self, mBoundRows: cute.Tensor, mBound: cute.Tensor, mUinv: cute.Tensor, mQRowAmax: cute.Tensor,
                 mDoRowAmax: cute.Tensor, mRowBoost: cute.Tensor, mUinvOut: cute.Tensor, mPartials: cute.Tensor,
                 mCounter: cute.Tensor, mAmaxOut: cute.Tensor, n_rows: Int32, mTileDoc: Optional[cute.Tensor],
                 stream: cuda.CUstream):
        self.kernel(mBoundRows, mBound, mUinv, mQRowAmax, mDoRowAmax, mRowBoost, mUinvOut, mPartials, mCounter,
                    mAmaxOut, n_rows, mTileDoc).launch(grid=[(n_rows + 127) // 128, 1, 1], block=[self.threads, 1, 1],
                                             stream=stream)

    @cute.kernel
    def kernel(self, mBoundRows: cute.Tensor, mBound: cute.Tensor, mUinv: cute.Tensor, mQRowAmax: cute.Tensor,
               mDoRowAmax: cute.Tensor, mRowBoost: cute.Tensor, mUinvOut: cute.Tensor, mPartials: cute.Tensor,
               mCounter: cute.Tensor, mAmaxOut: cute.Tensor, n_rows: Int32, mTileDoc: Optional[cute.Tensor] = None):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bnd = None
        tiles = None
        if const_expr(self.docs):
            tiles = mTileDoc.shape[0]
            bnd = mBound[mTileDoc[bid % tiles]]   # this CTA's 128 rows: one tile of one head, one document
        smem = cutlass.utils.SmemAllocator()
        sRed = smem.allocate_tensor(Float32, cute.make_layout(32), 16)
        i = bid * 128 + tidx
        qa = Float32(0.0)
        da = Float32(0.0)
        if i < n_rows:
            # (bound / bound_q.clamp(min=1e-30)).clamp(min=1.0), float32 (div.rn)
            ratio = cute.arch.fmax(F.div_rn_f32(bnd if const_expr(self.docs) else mBound[0], cute.arch.fmax(mBoundRows[i], Float32(1e-30))), Float32(1.0))
            # floor(log2 ratio) exactly: the exponent field (ratio >= 1 normal, or +inf -> 128 -> the cap);
            # interface_bwd_fp4q.pow2_floor_bits is the same on the torch side
            e = cute.arch.fmin(Float32((F.f32_bits(ratio) >> 23) - 127), Float32(self.cap))
            r = F.bits_f32((Int32(e) + 127) << 23)                                 # 2^e of an integer: exact
            mRowBoost[i] = r
            mUinvOut[i] = F.mul_rn(mUinv[i], r)
            qa = F.round_to_bf16(F.div_rn_f32(mQRowAmax[i], r))                    # (q / r).to(bf16), its row max
            da = F.round_to_bf16(F.div_rn_f32(mDoRowAmax[i], r))
        qa = cute.arch.warp_reduction_max(qa)
        da = cute.arch.warp_reduction_max(da)
        if tidx % 32 == 0:
            sRed[tidx // 32] = qa
            sRed[4 + tidx // 32] = da
        cute.arch.sync_threads()
        total = (n_rows + 127) // 128
        if tidx == 0:
            F.st_relaxed_f32(mPartials.iterator + 2 * bid,
                             cute.arch.fmax(cute.arch.fmax(sRed[0], sRed[1]), cute.arch.fmax(sRed[2], sRed[3])))
            F.st_relaxed_f32(mPartials.iterator + 2 * bid + 1,
                             cute.arch.fmax(cute.arch.fmax(sRed[4], sRed[5]), cute.arch.fmax(sRed[6], sRed[7])))
            F.fence_acq_rel_gpu()
            done = cute.arch.atomic_add(mCounter.iterator, Int32(1), sem="acq_rel", scope="gpu")
            sRed[8] = Float32(0.0)
            if done == total - 1:
                sRed[8] = Float32(1.0)
        cute.arch.sync_threads()
        if const_expr(self.docs):
            if sRed[8] == Float32(1.0):
                # per document: zero the (n_docs + 1) x 2 outputs, max-reduce every CTA's pair into its document's
                F.fence_acq_rel_gpu()
                n2 = mAmaxOut.shape[0]
                for j in cutlass.range((n2 - tidx + 127) // 128, unroll=1):
                    F.st_relaxed_f32(mAmaxOut.iterator + (tidx + j * 128), Float32(0.0))
                F.fence_acq_rel_gpu()
                cute.arch.sync_threads()
                for j in cutlass.range((total - tidx + 127) // 128, unroll=1):
                    k = tidx + j * 128
                    dd = mTileDoc[k % tiles]
                    F.red_max_nonneg_f32(F.ld_relaxed_f32(mPartials.iterator + 2 * k), mAmaxOut.iterator + 2 * dd)
                    F.red_max_nonneg_f32(F.ld_relaxed_f32(mPartials.iterator + 2 * k + 1), mAmaxOut.iterator + 2 * dd + 1)
                if tidx == 0:
                    mCounter[0] = Int32(0)
        elif sRed[8] == Float32(1.0):
            F.fence_acq_rel_gpu()
            mq = Float32(0.0)
            md = Float32(0.0)
            for j in cutlass.range((total - tidx + 127) // 128, unroll=1):
                k = tidx + j * 128
                mq = cute.arch.fmax(mq, F.ld_relaxed_f32(mPartials.iterator + 2 * k))
                md = cute.arch.fmax(md, F.ld_relaxed_f32(mPartials.iterator + 2 * k + 1))
            mq = cute.arch.warp_reduction_max(mq)
            md = cute.arch.warp_reduction_max(md)
            if tidx % 32 == 0:
                sRed[12 + tidx // 32] = mq
                sRed[16 + tidx // 32] = md
            cute.arch.sync_threads()
            if tidx == 0:
                mAmaxOut[0] = cute.arch.fmax(cute.arch.fmax(sRed[12], sRed[13]), cute.arch.fmax(sRed[14], sRed[15]))
                mAmaxOut[1] = cute.arch.fmax(cute.arch.fmax(sRed[16], sRed[17]), cute.arch.fmax(sRed[18], sRed[19]))
                mCounter[0] = Int32(0)


_CACHE = {}


def dense_row_boost(bound_rows: torch.Tensor, bound: torch.Tensor, uinv: torch.Tensor, query_row_amax: torch.Tensor,
                    grad_row_amax: torch.Tensor, cap: float, tile_doc: torch.Tensor = None):
    """(row_boost r_q, uinv * r_q, amax (2,) float32 [max |bf16(Q / r)|, max |bf16(dO / r)|]) of the dense all-FP4
    backward. bound_rows / uinv / query_row_amax / grad_row_amax: (b, h, s) float32 contiguous; bound: (1,) float32 (the
    dO launch's max of bound_rows). One launch, no host read.
    tile_doc (aligned varlen documents, b = 1): the document of every 128-row tile; bound is then (n_docs + 1,) per
    document and amax ((n_docs + 1) * 2,) [max |bf16(Q / r)|, max |bf16(dO / r)|] per document."""
    for t in (bound_rows, uinv, query_row_amax, grad_row_amax):
        assert t.dtype == torch.float32 and t.is_contiguous() and t.shape == bound_rows.shape
    dev = bound_rows.device
    n = bound_rows.numel()
    r = torch.empty_like(bound_rows)
    u2 = torch.empty_like(uinv)
    docs = tile_doc is not None
    amax = torch.empty(2 * bound.numel() if docs else 2, dtype=torch.float32, device=dev)
    parts = torch.empty(2 * ((n + 127) // 128), dtype=torch.float32, device=dev)
    counter = F._max_counter(dev)
    T_ = lambda t: F.to_cute_tensor(t.view(-1), assumed_align=4, leading_dim=0)  # noqa: E731
    key = ("dense_row_boost", float(cap)) + (("docs",) if docs else ())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (bound_rows, bound, uinv, query_row_amax, grad_row_amax, r, u2, parts, counter, amax)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(_DenseRowBoostKernel(cap, docs), *[T_(a) for a in args], Int32(n),
                                   T_(tile_doc) if docs else None, stream, options="--enable-tvm-ffi")
    _CACHE[key](*[a.view(-1) for a in args], Int32(n), tile_doc if docs else None, stream)
    return r, u2, amax

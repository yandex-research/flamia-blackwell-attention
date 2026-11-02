"""FA3-style pre-quantised inputs for flash_attn_fp8: E4M3 q / k / v plus float32 (b, h_kv) descales.

The convention is FA3's (hopper/flash_attn_interface.py q_descale / k_descale / v_descale; applied in
mainloop_fwd_sm90_tma_gmma_ws.hpp): the caller hands over E4M3 codes with one dequantisation scale per (batch row,
kv head) -- per (document, kv head) under varlen, b being the document index -- so that x ~= code * descale. The
q heads of a GQA group share their kv head's Q descale, as in FA3. No Q / K / V quantiser runs.

What runs (lowprec._routes._fp8_forward dispatches here):
  forward    the fork forward on the caller's codes. It reads, per work tile, the score scale
             float32((softmax_scale * q_descale) * k_descale) and the V descale float32(v_descale) of the tile's (batch
             row or document, kv head) -- the per-(document, kv head) tables of varlen step 2 (score_scale_doc), here
             built from the descales by ONE small launch instead of by the quantisers. Batched: the batched kernel with
             a (b, h_kv) table. Aligned varlen: the documents in place, as the internal per-tensor varlen path.
  backward   (training, grad_proxies) the per-(document, kv head) 1-CTA backward of step 2 on the caller's codes:
             dO is quantised here, per (document, kv head) (the dO launch also folds D and writes the shifted LSE /
             the zeroed f16 dQ accumulator), the kernel's scalar table comes from the caller's descales. The f16 dQ
             pre-scale is data-sized from max ||V_k||^2 of the dequantised V codes (one read of the V codes) and the dO
             launch's norms. A batched call with grad keeps the batched forward and runs the same 1-CTA kernel with
             every batch row as a document (doc_scaled without doc_varlen: the table read at the tile's batch index)
             after a batched dO launch (quant.fp8 documents, batched=True: per-(row, kv head) dO scale, D (b, h, s)
             folded per row, the shifted LSE, the zeroed accumulator).

Gradients: PyTorch casts every gradient to its input's dtype, so an E4M3 leaf could only receive E4M3 codes on no
scale. The backward therefore delivers dQ / dK / dV to `grad_proxies` -- bf16 tensors of q / k / v's shapes (usually
the pre-quantisation activations): a straight-through gradient, d loss / d proxy = the gradient at the dequantised
codes. E4M3 inputs that require grad themselves are refused, naming grad_proxies.

Bitwise equivalence (tests/cute/test_prequant_fa3.py): codes + scales from our own per-tensor quantiser (float32 scales,
FA3's dtype; float64 is still accepted and converted to float32 once) give outputs bitwise equal to the internal
per-tensor path -- batched (one scale per tensor, broadcast) and aligned varlen (one per (document, kv head)).
"""
from __future__ import annotations

import threading
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_attn.cute.device_scales import _f32

E4M3 = torch.float8_e4m3fn

__all__ = ["flash_attn_fp8_prequant", "flash_attn_fp8_prequant_doc_maps", "refuse_fp4", "fa3_tables", "fp8_forward"]


def _fail(exc, msg: str):
    raise exc(f"[lowprec rule 32] {msg}")


def refuse_fp4():
    """flash_attn_fp4 with E4M3 inputs / descales: its forward's Q.K^T is NVFP4 -- E2M1 codes with an E4M3 scale per
    16 elements, and the backward recomputes S from those same 4-bit codes. E4M3 Q / K would have to be re-quantised to
    NVFP4 (a quantiser pass reading Q and K -- the very pass pre-quantised input removes -- and a second rounding of the
    caller's values) or run as an E4M3 Q.K^T, which is flash_attn_fp8's forward, not the 4-bit route."""
    _fail(NotImplementedError,
          "flash_attn_fp4 takes no pre-quantised E4M3 q / k / v (q_descale / k_descale / v_descale): its Q.K^T is NVFP4 "
          "(E2M1 codes with an E4M3 scale per 16 elements; the backward recomputes S from the same 4-bit codes), so E4M3 "
          "codes would need a re-quantisation to NVFP4 -- a quantiser pass over Q / K and a second rounding of the "
          "caller's values -- or an E4M3 Q.K^T, which is flash_attn_fp8's forward. Use flash_attn_fp8(q, k, v, "
          "q_descale=..., k_descale=..., v_descale=...).")


# ---------------------------------------------------------------------------------------------------------------------
# host-side validation (shapes / dtypes / devices only: nothing is read from the GPU)
# ---------------------------------------------------------------------------------------------------------------------
def _validate(q, k, v, descs, proxies, cu_seqlens_q, n_docs=None):
    for nm, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            _fail(TypeError, f"flash_attn_fp8: {nm} must be a torch.Tensor; got {type(t).__name__}.")
    if not all(t.dtype == E4M3 for t in (q, k, v)):
        given = [nm for nm, x in zip(("q_descale", "k_descale", "v_descale", "grad_proxies"), list(descs) + [proxies])
                 if x is not None]
        _fail(ValueError, f"flash_attn_fp8: {' / '.join(given) or 'pre-quantised input'} given, so q, k, v must all be "
                          f"torch.float8_e4m3fn codes (FA3's convention); got {q.dtype}, {k.dtype}, {v.dtype}.")
    if any(d is None for d in descs):
        miss = [nm for nm, d in zip(("q_descale", "k_descale", "v_descale"), descs) if d is None]
        _fail(ValueError, f"flash_attn_fp8: E4M3 q / k / v need q_descale, k_descale and v_descale (float32 (b, h_kv), "
                          f"x ~= code * descale); missing {', '.join(miss)}. A unit scale is torch.ones(b, h_kv).")
    for nm, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_cuda or t.stride(-1) != 1:
            _fail(ValueError, f"flash_attn_fp8: {nm} must be a CUDA tensor contiguous in its last dim.")
    varlen = cu_seqlens_q is not None
    n = (cu_seqlens_q.shape[0] - 1) if varlen and isinstance(cu_seqlens_q, torch.Tensor) else q.shape[0]
    if n_docs is not None:   # (doc_maps: one row per document slot of the maps' capacity)
        varlen, n = True, n_docs
    hk = k.shape[-2]
    for nm, d in zip(("q_descale", "k_descale", "v_descale"), descs):
        if not isinstance(d, torch.Tensor) or d.dtype not in (torch.float32, torch.float64) or not d.is_cuda \
                or d.device != q.device or tuple(d.shape) != (n, hk):
            _fail(ValueError, f"flash_attn_fp8: {nm} must be a float32 (or float64) CUDA tensor of shape "
                              f"({'n_docs' if varlen else 'b'}, h_kv) = ({n}, {hk}) on q's device -- one dequantisation "
                              f"scale per ({'document' if varlen else 'batch row'}, kv head), FA3's convention; got "
                              f"{getattr(d, 'dtype', type(d).__name__)} {tuple(getattr(d, 'shape', ()))}"
                              f"{'' if not isinstance(d, torch.Tensor) else ' on ' + str(d.device)}.")
        if d.requires_grad:
            _fail(ValueError, f"flash_attn_fp8: {nm} carries no gradient (pass a detached tensor).")
    if any(t.requires_grad for t in (q, k, v)) and torch.is_grad_enabled():
        _fail(ValueError, "flash_attn_fp8: E4M3 q / k / v that require grad: PyTorch casts a gradient to its input's "
                          "dtype, so dQ / dK / dV would arrive as E4M3 codes on no scale. Pass detached codes and "
                          "grad_proxies=(q_bf16, k_bf16, v_bf16) (bf16 tensors of the same shapes, usually the "
                          "pre-quantisation activations): their .grad receives the gradients (straight-through).")
    if proxies is not None:
        if not isinstance(proxies, (tuple, list)) or len(proxies) != 3:
            _fail(ValueError, "flash_attn_fp8: grad_proxies must be a (q, k, v) tuple of bf16 tensors.")
        for nm, p, t in zip("qkv", proxies, (q, k, v)):
            if not isinstance(p, torch.Tensor) or p.dtype != torch.bfloat16 or tuple(p.shape) != tuple(t.shape) \
                    or p.device != t.device:
                _fail(ValueError, f"flash_attn_fp8: grad_proxies[{'qkv'.index(nm)}] must be a bf16 tensor of {nm}'s "
                                  f"shape {tuple(t.shape)} on its device; got {getattr(p, 'dtype', type(p).__name__)} "
                                  f"{tuple(getattr(p, 'shape', ()))}.")
    return n


# A/B knob (tests): run a batched call with grad as equal-length aligned documents (the first form of this path) instead
# of the batched kernels with per-(batch row, kv head) tables. Same math (per-(row, kv head) dO, D fold): bitwise equal.
BATCHED_AS_DOCUMENTS = False

# equal-length documents of a batched call with grad: (device, b, s_q, s_k) -> (cu_q, cu_k, used_q, used_k), built once
_BATCH_DOCS = {}
_BATCH_DOCS_LOCK = threading.Lock()


def _batch_docs(dev, b: int, s_q: int, s_k: int):
    key = (str(dev), b, s_q, s_k)
    with _BATCH_DOCS_LOCK:
        r = _BATCH_DOCS.get(key)
        if r is None:
            ar = torch.arange(b + 1, dtype=torch.int32, device=dev)
            r = ((ar * s_q).contiguous(), (ar * s_k).contiguous(), torch.full((b,), s_q, dtype=torch.int32, device=dev),
                 torch.full((b,), s_k, dtype=torch.int32, device=dev))
            _BATCH_DOCS[key] = r
    return r


# the batched training path's document tables: (device, b, s_q, s_k) -> dict, built once per shape (graph-safe after)
_BATCH_TABS = {}


def _batch_tabs(dev, b: int, s_q: int, s_k: int):
    """Every batch row a document: the dO launch's tables (quant.fp8 documents, batched=True: per 128-row tile of the
    flattened (b s) rows -- the row's index -- with row ends LOCAL to the row, s = no padding) and fa3_vn2's (the absolute
    end of each tile's row in the flattened V)."""
    key = (str(dev), b, s_q, s_k)
    with _BATCH_DOCS_LOCK:
        r = _BATCH_TABS.get(key)
        if r is None:
            tq, tk = s_q // 128, s_k // 128
            doc_q = torch.arange(b * tq, dtype=torch.int32, device=dev) // tq
            doc_k = torch.arange(b * tk, dtype=torch.int32, device=dev) // tk
            r = dict(tile_doc_q=doc_q, tile_doc_k=doc_k,
                     row_end_q=torch.full((b * tq,), s_q, dtype=torch.int32, device=dev),
                     row_end_k=torch.full((b * tk,), s_k, dtype=torch.int32, device=dev),
                     vn_end_k=((doc_k + 1) * s_k).to(torch.int32))
            _BATCH_TABS[key] = r
    return r


class _RowDocs:
    """fa3_vn2's view of the batch rows as documents."""

    def __init__(self, b, tabs):
        self.b, self._t = b, tabs
        self.row_end_tab_k = tabs["vn_end_k"]

    def tile_doc(self, tile):
        return self._t["tile_doc_k"]


def _batched_backward(plan, st, dout):
    """The batched training path's backward (lowprec._routes._fp8_backward hands over): the 1-CTA per-(batch row, kv
    head) kernel (flash_bwd_sm100_fp4 doc_scaled without doc_varlen) on the caller's codes. One dO launch in the batched
    layout (quant.fp8 documents, batched=True): dO's E4M3 scale per (batch row, kv head), D (b, h, s) folded per row,
    the shifted LSE and the zeroed f16 dQ accumulator -- the aligned varlen backward's prologue, row for document."""
    from flash_attn.cute.interface_bwd_2cta8 import P_SHIFT_LOG2
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4, e4m3_p_shift_log2, shape_carrier
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    sp = plan.specs
    qsp = sp.quant
    out, lse, q8, k8, v8 = st.saved_tensors
    qs, ks, vs, dt, dev = st.qkv_meta
    dout = dout.contiguous()
    b_, s_, h_, d_ = dout.shape
    assert qsp.ds_scaling_fp8 != "per_tensor", "pre-quantised E4M3: the block-scaled dS"
    acc = torch.empty(b_, h_, s_ * d_, dtype=torch.float16, device=dout.device)
    fold = (st.doc_scales[2], 1.0 / float(2 ** e4m3_p_shift_log2())) if qsp.ds_fold else None
    t = st.batch_tabs
    r = quantize_fp8_per_tensor(dout, compute_d_with_output=out.contiguous(), shifted_log2_lse_from=lse.contiguous(),
                                log2_lse_shift=float(P_SHIFT_LOG2), zero_accumulator=acc, output_row_norm2_max=True,
                                power_of_two_scale=qsp.per_tensor_pow2, documents=dict(
                                    side=("q",), row_end_q=t["row_end_q"], row_end_k=t["row_end_k"],
                                    tile_doc_q=t["tile_doc_q"], tile_doc_k=t["tile_doc_k"], n_rows=b_ + 1,
                                    h_kv=ks[2], fold=fold, batched=True))
    sc = dict(dpsum=r["D"], lse_log2=r["lse_log2"], dpsum_owned=True, dq_accum=acc, vn2=st.vn2,
              dn2=r["row_norm2_max"], glue_lse_log2=r["lse_log2"], glue_dq_accum=acc,
              glue_lse_log2_shift=float(P_SHIFT_LOG2), doc_d_folded=bool(qsp.ds_fold), batch_scaled=True,
              doc_scales=dict(q=st.doc_scales[0], k=st.doc_scales[1], v=st.doc_scales[2], do=r["doc_scales"][0],
                              k_global=st.k_global),
              q8=q8, sq=st.sq, k8=k8, sk=st.sk, v8=v8, sv=st.sv, do8=r["codes"], sdo=r["scales"])
    return _flash_attn_bwd_fp4(
        shape_carrier(qs, dt, dev), shape_carrier(ks, dt, dev), shape_carrier(vs, dt, dev),
        out, dout, lse, softmax_scale=st.softmax_scale, causal=st.causal, deterministic=st.deterministic,
        fp8_scales=sc, masks=st.masks, ds_dq_rowlocal=plan.bwd.ds_dq_rowlocal, specs=sp)


def flash_attn_fp8_prequant(cached_plan, run, resolve, q, k, v, q_descale, k_descale, v_descale, grad_proxies, kw,
                            aux_tensors=None):
    """lowprec.api.flash_attn_fp8 with E4M3 q / k / v (see the module docstring)."""
    descs = (q_descale, k_descale, v_descale)
    _validate(q, k, v, descs, grad_proxies, kw.get("cu_seqlens_q"))
    if kw.get("q_quantized") is not None or kw.get("k_quantized") is not None or kw.get("q_scale_fused"):
        _fail(ValueError, "flash_attn_fp8: q_descale / k_descale / v_descale (E4M3 q / k / v) and q_quantized / "
                          "k_quantized / q_scale_fused are two forms of pre-quantised input: pass one.")
    grad = grad_proxies is not None and torch.is_grad_enabled() and any(p.requires_grad for p in grad_proxies)
    proxies = tuple(grad_proxies) if grad else None
    varlen = kw.get("cu_seqlens_q") is not None or kw.get("cu_seqlens_k") is not None
    if varlen and kw.get("seqused_q") is None and kw.get("seqused_k") is None and q.dim() == 3:
        return _repacked(cached_plan, run, resolve, q, k, v, descs, proxies, kw, aux_tensors)
    if not varlen and kw.get("seqused_k") is not None and q.dim() == 4 and q.shape[1] % 128 == 0 \
            and k.shape[1] % 128 == 0:
        # batched seqused_k: the batch rows as aligned documents (zero-copy views), the keys past seqused_k[b] padding
        b, s_q, h, d = q.shape
        s_k = k.shape[1]
        cu_q, cu_k, uq, _ = _batch_docs(q.device, b, s_q, s_k)
        kw2 = dict(kw, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seqused_q=uq, seqused_k=kw["seqused_k"], max_seqlen_q=s_q,
                   max_seqlen_k=s_k)
        flat = lambda t, s: t.reshape(b * s, t.shape[2], t.shape[3])  # noqa: E731
        pp = None
        if proxies is not None:
            # keys past seqused_k get zero dK / dV (the batched seqused_k contract; the aligned form leaves its padding
            # rows' gradients unspecified): the K / V proxies enter masked (their values are never read)
            keep = (torch.arange(s_k, device=q.device).view(1, s_k) < kw["seqused_k"].view(b, 1)).view(b, s_k, 1, 1)
            z = torch.zeros((), dtype=proxies[1].dtype, device=q.device)
            pp = (flat(proxies[0], s_q), flat(torch.where(keep, proxies[1], z), s_k),
                  flat(torch.where(keep, proxies[2], z), s_k))
        r = _call(cached_plan, run, resolve, flat(q, s_q), flat(k, s_k), flat(v, s_k), descs, pp, kw2, aux_tensors)
        if isinstance(r, tuple):
            out, lse = r
            return out.view(b, s_q, h, out.shape[-1]), lse.view(h, b, s_q).permute(1, 0, 2).contiguous()
        return r.view(b, s_q, h, r.shape[-1])
    if BATCHED_AS_DOCUMENTS and grad and not varlen and q.dim() == 4 and k.dim() == 4:
        # a batched call with grad runs as equal-length aligned documents (zero-copy views): the per-(document, kv
        # head) backward reads the forward's LSE / D in the documents' (1, h, total) layout
        b, s_q, h, d = q.shape
        s_k, hk = k.shape[1], k.shape[2]
        if s_q % 128 or s_k % 128 or kw.get("seqused_k") is not None:
            pass   # resolve refuses it below with the batched message (rule 32)
        else:
            cu_q, cu_k, uq, uk = _batch_docs(q.device, b, s_q, s_k)
            kw2 = dict(kw, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seqused_q=uq, seqused_k=uk, max_seqlen_q=s_q,
                       max_seqlen_k=s_k)
            flat = lambda t, s: t.reshape(b * s, t.shape[2], t.shape[3])  # noqa: E731
            r = _call(cached_plan, run, resolve, flat(q, s_q), flat(k, s_k), flat(v, s_k), descs,
                      (flat(proxies[0], s_q), flat(proxies[1], s_k), flat(proxies[2], s_k)), kw2, aux_tensors)
            if isinstance(r, tuple):
                out, lse = r
                return out.view(b, s_q, h, out.shape[-1]), lse.view(h, b, s_q).permute(1, 0, 2).contiguous()
            return r.view(b, s_q, h, r.shape[-1])
    return _call(cached_plan, run, resolve, q, k, v, descs, proxies, kw, aux_tensors)


def flash_attn_fp8_prequant_doc_maps(cached_plan, run, resolve, q, k, v, q_descale, k_descale, v_descale,
                                     grad_proxies, kw, doc_maps, aux_tensors=None):
    """lowprec.api.flash_attn_fp8 with E4M3 q / k / v and doc_maps (cov2): the caller's codes in its PACKED rows --
    q (T_q, h, d), k / v (T_k, h_kv, d) torch.float8_e4m3fn, row for row the documents of the maps -- and descales of
    shape (max_num_documents, h_kv) (row i: document i of the maps; the rows past the call's document count are not
    read). No layout constraint on the codes: no aligned copy, no channel-major V (the forward reads V's row codes as
    its MN-major P.V operand, lp_docmaps packed rows). The kernels read every tile at its document's packed start, a
    tile's rows past the document (the next document's codes, finite E4M3) are masked. The codes must be finite
    (E4M3 NaN codes in one document would reach the P.V of the previous document's last key tile through P = 0)."""
    descs = (q_descale, k_descale, v_descale)
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        _fail(ValueError, "flash_attn_fp8(doc_maps=...): q, k, v are packed (total, h, d) codes")
    doc_maps.check_call(q, k, v, "fp8")
    _validate(q, k, v, descs, grad_proxies, None, n_docs=doc_maps.max_num_documents)
    w = kw.get("window_size") or (None, None)
    if any(x is not None for x in w) or kw.get("softcap") or kw.get("score_mod") is not None \
            or kw.get("mask_mod") is not None or aux_tensors is not None:
        _fail(NotImplementedError, "flash_attn_fp8(doc_maps=...) with E4M3 q / k / v: window / softcap / score_mod / "
                                   "mask_mod are not built on the packed rows (plain causal / dense); use the aligned "
                                   "varlen form (cu_seqlens + seqused_*).")
    if kw.get("q_scale_fused"):
        _fail(ValueError, "flash_attn_fp8: q_descale / k_descale / v_descale and q_scale_fused are two forms of "
                          "pre-quantised input: pass one.")
    grad = grad_proxies is not None and torch.is_grad_enabled() and any(p.requires_grad for p in grad_proxies)
    if grad and not (PACKED_BACKWARD and doc_maps.compact):
        _fail(NotImplementedError, "flash_attn_fp8(doc_maps=...) with E4M3 q / k / v and grad_proxies: the packed "
                                   "backward runs on compact maps (DocMaps.from_cu_seqlens without padded_size); "
                                   "use those, or the aligned varlen form (cu_seqlens + seqused_*).")
    proxies = tuple(grad_proxies) if grad else None
    ins = proxies if proxies is not None else (q, k, v)
    stand = (doc_maps.stand_in(ins[0], "q"), doc_maps.stand_in(ins[1], "k"), doc_maps.stand_in(ins[2], "k"))
    vk = dict(cu_seqlens_q=doc_maps.cu_pad_q, cu_seqlens_k=doc_maps.cu_pad_k, seqused_q=doc_maps.len_q,
              seqused_k=doc_maps.len_k, max_seqlen_q=doc_maps.max_seqlen_q, max_seqlen_k=doc_maps.max_seqlen_k)
    kw2 = {n: x for n, x in kw.items() if n not in ("q_scale_fused",)}
    plan = cached_plan(resolve, *stand, dict(kw2, prequant_fa3=True, **vk))
    prequant = {"fa3": (q, k, v) + tuple(descs)}
    return run(plan, *ins, score_mod=None, score_mod_bwd=None, mask_mod=None, aux_tensors=None,
               cu_seqlens_q=vk["cu_seqlens_q"], cu_seqlens_k=vk["cu_seqlens_k"], seqused_k=vk["seqused_k"],
               prequant=prequant, seqused_q=vk["seqused_q"], doc_maps=doc_maps)


# cov2: the packed backward of pre-quantised doc_maps calls (compact maps): the per-(document, kv head) 1-CTA kernel on
# packed rows -- the per-tensor packed path (quant.fp8.quantize_fp8_per_tensor documents packed=, the dense statistics
# rows), max ||V||^2 by fa3_vn2_packed
PACKED_BACKWARD = True


def _repacked(cached_plan, run, resolve, q, k, v, descs, proxies, kw, aux_tensors):
    """Repacked varlen (cu_seqlens without seqused_*): the codes (one byte per element) and the grad proxies are copied
    into a zero-padded buffer with every document start rounded up to 128 (lp_varlen.LPDocs.build, the repack
    fallback's maps), the aligned form runs on it (cu_seqlens = the padded starts, seqused = the true lengths), and the
    output / LSE rows are gathered back (the proxies' gradients through the autograd of the row gathers). The descales
    are per document either way."""
    from flash_attn.cute.lp_varlen import LPDocs, _src_of, pack_rows, row_gather, unpack_rows
    cu_q, cu_k = kw["cu_seqlens_q"], kw["cu_seqlens_k"]
    rp = LPDocs.build(cu_q, cu_k, q.shape[0], k.shape[0], align=128, max_seqlen_q=kw.get("max_seqlen_q"),
                      max_seqlen_k=kw.get("max_seqlen_k"))

    def codes(t, side):
        src, _, rows = _src_of(rp, side)
        return row_gather(t.view(torch.uint8), src, rows).view(E4M3)

    kw2 = dict(kw, cu_seqlens_q=rp.cu_pad_q, cu_seqlens_k=rp.cu_pad_k, seqused_q=rp.len_q, seqused_k=rp.len_k)
    pp = None if proxies is None else tuple(pack_rows(rp, p, sd)[0] for p, sd in zip(proxies, "qkk"))
    r = _call(cached_plan, run, resolve, codes(q, "q"), codes(k, "k"), codes(v, "k"), descs, pp, kw2, aux_tensors)
    out, lse = r if isinstance(r, tuple) else (r, None)
    out = unpack_rows(rp, out.unsqueeze(0), "q") if pp is not None else out.index_select(0, rp.dst_q)
    return out if lse is None else (out, lse.index_select(1, rp.dst_q))


def _call(cached_plan, run, resolve, q, k, v, descs, proxies, kw, aux_tensors):
    ins = proxies if proxies is not None else (q, k, v)
    plan = cached_plan(resolve, *ins, dict(kw, prequant_fa3=True))
    prequant = {"fa3": (q, k, v) + tuple(descs)}
    return run(plan, *ins, score_mod=kw.get("score_mod"), score_mod_bwd=kw.get("score_mod_bwd"),
               mask_mod=kw.get("mask_mod"), aux_tensors=aux_tensors, cu_seqlens_q=kw.get("cu_seqlens_q"),
               cu_seqlens_k=kw.get("cu_seqlens_k"), seqused_k=kw.get("seqused_k"), prequant=prequant,
               seqused_q=kw.get("seqused_q"))


# ---------------------------------------------------------------------------------------------------------------------
# the descale tables (one launch) and max ||V_k||^2 of the dequantised codes (one read of V, backward only)
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def _fa3_tables_kernel(qd_p, kd_p, vd_p, qs0, qs1, ks0, ks1, vs0, vs1, score_p, vdesc_p, dsc_p, kg_p, vn_p, n, hk,
                       softmax, DOCS: tl.constexpr, BLOCK: tl.constexpr):
    # score[r, h] = (q_descale float32(softmax_scale)) k_descale in float32 -- the order of the internal path's
    # quantiser-derived table (quant.fp8 documents derived (0, 1, softmax_scale)) and of score_scale_dev; vdesc =
    # v_descale. DOCS: the backward's (3, n + 1, h_kv) float32 scale tables (row n = the sentinel, 1.0), the global K
    # scale max_r,h k_descale (the f16 dQ range's max|K| <= 448 s_k) and the zeroed max ||V||^2 slot.
    softmax = softmax.to(tl.float32)
    kmax = tl.zeros([BLOCK], tl.float32)
    nt = n * hk
    for i0 in range(0, nt, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        m = idx < nt
        r = idx // hk
        c = idx % hk
        qd = tl.load(qd_p + r * qs0 + c * qs1, mask=m, other=1.0).to(tl.float32)
        kd = tl.load(kd_p + r * ks0 + c * ks1, mask=m, other=1.0).to(tl.float32)
        vd = tl.load(vd_p + r * vs0 + c * vs1, mask=m, other=1.0).to(tl.float32)
        tl.store(score_p + idx, (qd * softmax) * kd, mask=m)
        tl.store(vdesc_p + idx, vd, mask=m)
        if DOCS:
            rs = (n + 1) * hk
            tl.store(dsc_p + idx, qd, mask=m)
            tl.store(dsc_p + rs + idx, kd, mask=m)
            tl.store(dsc_p + 2 * rs + idx, vd, mask=m)
            kmax = tl.maximum(kmax, tl.where(m, kd, 0.0))
    if DOCS:
        rs = (n + 1) * hk
        for c0 in range(0, hk, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            mc = c < hk
            one = tl.full([BLOCK], 1.0, tl.float32)
            tl.store(dsc_p + nt + c, one, mask=mc)
            tl.store(dsc_p + rs + nt + c, one, mask=mc)
            tl.store(dsc_p + 2 * rs + nt + c, one, mask=mc)
        tl.store(kg_p, tl.max(kmax, axis=0))
        tl.store(vn_p, 0.0)


@triton.jit
def _fa3_vn2_kernel(v_p, sv_r, sv_h, vdsc_p, tile_doc_p, rend_p, vn_p, n_docs, hk, D: tl.constexpr):
    # max over the real rows of ||code * v_descale||^2 for one (128-row tile, kv head): the padding rows (at or past
    # their document's true length) and sentinel tiles read nothing
    t = tl.program_id(0)
    h = tl.program_id(1)
    doc = tl.load(tile_doc_p + t)
    rend = tl.load(rend_p + t)
    rows = t * 128 + tl.arange(0, 128)
    ok = (rows < rend) & (doc < n_docs)
    cols = tl.arange(0, D)
    x = tl.load(v_p + rows[:, None].to(tl.int64) * sv_r + h * sv_h + cols[None, :], mask=ok[:, None], other=0.0)
    x = x.to(tl.float32)
    s = tl.load(vdsc_p + tl.minimum(doc, n_docs) * hk + h).to(tl.float32)
    xs = x * s
    nrm = tl.sum(xs * xs, axis=1)
    tl.atomic_max(vn_p, tl.max(nrm, axis=0))


@triton.jit
def _fa3_vn2_packed_kernel(v_p, sv_r, sv_h, vdsc_p, tile_doc_p, vt_p, n_tb, vn_p, n_docs, hk, D: tl.constexpr):
    # _fa3_vn2_kernel on lp_docmaps packed rows (cov2): tile t of the compact maps' tile table holds the real rows
    # [ps, ps + nr) of document tile_doc[t] (vt_tile = (packed starts, real rows)); the same per-row norms and max
    t = tl.program_id(0)
    h = tl.program_id(1)
    doc = tl.load(tile_doc_p + t)
    ps = tl.load(vt_p + t)
    nr = tl.load(vt_p + n_tb + t) & 65535   # (the full maps' VT_LAST flag bit: not in compact maps)
    r = tl.arange(0, 128)
    rows = ps + r
    ok = (r < nr) & (doc < n_docs)
    cols = tl.arange(0, D)
    x = tl.load(v_p + rows[:, None].to(tl.int64) * sv_r + h * sv_h + cols[None, :], mask=ok[:, None], other=0.0)
    x = x.to(tl.float32)
    s = tl.load(vdsc_p + tl.minimum(doc, n_docs) * hk + h).to(tl.float32)
    xs = x * s
    nrm = tl.sum(xs * xs, axis=1)
    tl.atomic_max(vn_p, tl.max(nrm, axis=0))


def fa3_vn2_packed(v8, maps, v_doc_scales, vn):
    """fa3_vn2 of the caller's packed V codes (1, T, h_kv, d) with compact lp_docmaps maps (cov2). One launch over the
    maps' static tile bound (tiles past the device tile count read nothing)."""
    v2 = v8[0] if v8.dim() == 4 else v8
    total, hk, d = v2.shape
    vt = maps.vt_tile.reshape(2, -1)
    tb = vt.shape[1]
    _fa3_vn2_packed_kernel[(tb, hk)](v2, v2.stride(0), v2.stride(1), v_doc_scales, maps.tile_doc_k, vt, tb, vn,
                                     maps.max_num_documents, hk, D=d, num_warps=4)
    return vn


def fa3_tables(q_descale, k_descale, v_descale, softmax_scale: float, docs: bool):
    """(score (n, h_kv) float32, vdesc (n, h_kv) float32, doc_scales (3, n + 1, h_kv) float32 | None, k_global (1,)
    float32 | None, vn2 (1,) float32 | None (zeroed; filled by fa3_vn2)). One launch. Float64 descales are converted
    to float32 first (once, by torch: the kernel does float32 arithmetic only)."""
    q_descale, k_descale, v_descale = (d.float() if d.dtype == torch.float64 else d
                                       for d in (q_descale, k_descale, v_descale))
    n, hk = q_descale.shape
    dev = q_descale.device
    score = torch.empty(n, hk, dtype=torch.float32, device=dev)
    vdesc = torch.empty(n, hk, dtype=torch.float32, device=dev)
    if docs:
        dsc = torch.empty(3, n + 1, hk, dtype=torch.float32, device=dev)
        kg = torch.empty(1, dtype=torch.float32, device=dev)
        vn = torch.empty(1, dtype=torch.float32, device=dev)
    else:
        dsc, kg, vn = score, score, score   # unread placeholders
    _fa3_tables_kernel[(1,)](q_descale, k_descale, v_descale, q_descale.stride(0), q_descale.stride(1),
                             k_descale.stride(0), k_descale.stride(1), v_descale.stride(0), v_descale.stride(1),
                             score, vdesc, dsc, kg, vn, n, hk, _f32(softmax_scale), DOCS=bool(docs),
                             BLOCK=1024, num_warps=4)
    return score, vdesc, (dsc if docs else None), (kg if docs else None), (vn if docs else None)


def fa3_vn2(v8, docs, v_doc_scales, vn):
    """max_k ||V_k||^2 of the dequantised V codes (aligned documents) into vn (zeroed by fa3_tables). One launch."""
    v2 = v8[0] if v8.dim() == 4 else v8   # (total_k, h_kv, d)
    total, hk, d = v2.shape
    tiles = total // 128
    _fa3_vn2_kernel[(tiles, hk)](v2, v2.stride(0), v2.stride(1), v_doc_scales, docs.tile_doc(128),
                                 docs.row_end_tab_k.reshape(-1), vn, docs.b, hk, D=d, num_warps=4)
    return vn


# ---------------------------------------------------------------------------------------------------------------------
# the route: lowprec._routes._fp8_forward(prequant={"fa3": ...}) -> here
# ---------------------------------------------------------------------------------------------------------------------
def fp8_forward(plan, st, q, k, v, fa3, masks):
    """The forward on the caller's E4M3 codes. q / k / v are the autograd node's inputs (the codes, or the bf16
    grad_proxies; read for metadata only); fa3 = (q8, k8, v8, q_descale, k_descale, v_descale)."""
    from flash_attn.cute.fp4_bwd_autograd import _fwd_masks, _unbatch
    from flash_attn.cute.interface_fwd_fp4 import _flash_attn_fwd
    from flash_attn.cute.lp_masks import lp_masks_or_none
    sp = plan.specs
    hs = sp.host
    softmax_scale, causal, deterministic = plan.softmax_scale, plan.fwd.causal, plan.bwd.deterministic
    grad = plan.bwd.kernel != "none"
    q8, k8, v8, qd, kd, vd = fa3
    masks = lp_masks_or_none(masks, causal)
    st.masks = masks
    if masks is not None:
        causal = masks.causal
    docs = masks.docs if masks is not None else None
    if docs is not None:
        q8, k8, v8 = (t.unsqueeze(0) if t.dim() == 3 else t for t in (q8, k8, v8))   # (1, total, h, d) views
    score, vdesc, dsc, kg, vn = fa3_tables(qd, kd, vd, softmax_scale, docs=grad)
    from flash_attn.cute.lowprec._routes import _fwd_kw
    _mk = _fwd_masks(masks, causal, q8, k8, hs)
    if "kv_cu_seqlens_packed" in _mk:   # (cov2, lp_docmaps packed rows: V's row codes, MN-major, at the key starts)
        _mk["vt_cu_seqlens_packed"] = _mk["kv_cu_seqlens_packed"]
    out = torch.empty(q8.shape[:-1] + (v8.shape[-1],), dtype=torch.bfloat16, device=q8.device)
    if docs is not None:
        _q, _k, _v, _o = _unbatch(docs, q8, k8, v8, out)
        res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale, softmax_scale_t=score, v_descale=vdesc,
                              return_lse=True, out=_o, force_fp4_impl=True, use_2cta_forward=hs.fwd_two_cta, **_mk,
                              **_fwd_kw(sp))
        lse = res[1].unsqueeze(0)
    else:
        res = _flash_attn_fwd(q8, k8, v8, softmax_scale=softmax_scale, softmax_scale_t=score, v_descale=vdesc,
                              return_lse=True, out=out, force_fp4_impl=True, use_2cta_forward=hs.fwd_two_cta, **_mk,
                              **_fwd_kw(sp))
        lse = res[1]
    if grad:
        # the backward of the internal per-tensor varlen path (lowprec._routes._fp8_backward, docs): the caller's codes,
        # the scale tables from its descales, max ||V||^2 of the dequantised V codes for the f16 dQ range
        if docs is None:   # batched: every batch row a document (the 1-CTA per-(row, kv head) backward)
            b_, s_q, s_k = q8.shape[0], q8.shape[1], k8.shape[1]
            st.batch_tabs = _batch_tabs(q8.device, b_, s_q, s_k)
            st.fa3_backward = _batched_backward
            st.vn2 = fa3_vn2(v8.reshape(b_ * s_k, v8.shape[2], v8.shape[3]), _RowDocs(b_, st.batch_tabs), dsc[2], vn)
        elif getattr(docs, "doc_maps_packed", False):   # (cov2: the caller's packed rows, compact maps)
            st.vn2 = fa3_vn2_packed(v8, docs.doc_maps, dsc[2], vn)
        else:
            st.vn2 = fa3_vn2(v8, docs, dsc[2], vn)
        st.doc_scales = dsc
        st.k_global = kg
        st.sq = st.sk = st.sv = kg   # (the per-tensor scalars slot: unread by the per-(document, kv head) kernel)
        st.save_for_backward(out, lse, q8, k8, v8)
        st.qkv_meta = (q.shape, k.shape, v.shape, q.dtype, q.device)
        st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    st.mx = False
    st.had = False
    return out, lse

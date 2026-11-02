"""The route bodies: what one resolved plan runs, forward and backward, as plain functions of the plan.

``forward(plan, st, q, k, v, prequant, masks) -> (out, lse)`` and ``backward(plan, st, dout) -> (dq, dk, dv)``. ``st`` is
the route's saved state (``Saved``): the tensors it saves for the backward and the values it hands over. The plan is
the only configuration: every spec a driver, quantiser or kernel reads comes from ``plan.specs`` (lowprec.specs,
built once by lowprec.plan.build_specs) and is passed down explicitly. ``lowprec.api._LowPrecAttnFunc`` is the one
autograd node around them.

Routes (lowprec.plan):
  fp8           E4M3 Q / K / V / dO; the 8-bit forward (the fork kernel, or upstream's for per-tensor operands it
                cannot serve) and the E4M3 backward (1-CTA or 2-CTA)
  fp4.e4m3_pv   NVFP4 Q / K (S in both passes) with an E4M3 V / P.V; the E4M3-operand ("dk8") backward
  fp4.fp4_pv    option A: the all-FP4 forward with the reproduced-P^ export; inference only
  fp4.all_fp4   the all-FP4 forward and the query-row all-4-bit backward
"""
from __future__ import annotations

import math

import torch

from flash_attn.cute import fp4_bwd_autograd as ag
from flash_attn.cute.fp4_bwd_autograd import (FAC_DO, FAC_K, FAC_Q, FAC_V, _docs_of, _fwd_masks, _lse_for_bwd,
                                              _mx_do_prologue, _nv4_rows, _q4_blk2d, _unbatch, dk8_backward_prologue,
                                              dk8_prologue_fusable, fac_row, fused_ds_bound_ok, padded_head_dim,
                                              q_blk2d, score_scale_dev, tb_bwd_glue, tb_fold_fwd, tb_glue_tag,
                                              tok_row, token_block_factor_table, token_table, v_descale_dev,
                                              tl_fold_fac)
from flash_attn.cute.interface_fwd_fp4 import _flash_attn_fwd
from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4
from flash_attn.cute.lp_masks import lp_masks_or_none
from flash_attn.cute.quant.fp4 import quantize_fp4_rows

from .plan import AttnPlan

__all__ = ["Saved", "forward", "backward", "masks_of", "run_documents"]


class Saved:
    """A route forward's state for its backward: ``save_for_backward`` records the tensors autograd keeps (the autograd
    node saves them on its context and hands them back as ``saved_tensors``); every other value is an attribute."""

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *ts):
        self.saved_tensors = ts


def _o_buf(q, docs, shape=None):
    """The forward's O buffer: (b, s, h, d) like q (repacked varlen: zeroed, its padding rows stay 0); a doc_maps call
    with packed O (lp_docmaps.packed_o_rows) gets the caller's packed (1, T, h, d) rows instead."""
    shape = tuple(q.shape) if shape is None else tuple(shape)
    if docs is not None and getattr(docs, "doc_maps", None) is not None:
        from flash_attn.cute.lp_docmaps import packed_o_rows
        po = packed_o_rows(docs)
        if po is not None:
            return torch.empty((1, po[1]) + shape[2:], dtype=q.dtype, device=q.device)
    if docs is None or docs.aligned:
        return torch.empty(shape, dtype=q.dtype, device=q.device)
    return torch.zeros(shape, dtype=q.dtype, device=q.device)


def _packed_rows(x) -> bool:
    """x is a registered lp_docmaps packed operand (a packed doc_maps call, lp_docmaps.PACKED_ROWS)."""
    from flash_attn.cute import lp_docmaps
    return bool(lp_docmaps._PACKED) and lp_docmaps.packed_rows_of(x) is not None


def _sfd_fwd(mq, mk, mv, sf4=False) -> dict:
    """The forward's sfd_rows (sfd: the dense per-row tables of a packed lp_docmaps.SF_DENSE call), else {}."""
    if mq.get("ev_rows") is None or mk.get("ev_rows") is None or mv.get("ev_rows") is None:
        return {}
    rows = [mq["ev_rows"], mk["ev_rows"], mv["ev_rows"]]
    if sf4:
        if mq.get("sf4_rows") is None or mk.get("sf4_rows") is None:
            return {}
        rows += [mq["sf4_rows"], mk["sf4_rows"]]
    return {"sfd_rows": rows}


def _v_operand(mv):
    """The forward's P.V operand of V's token-block launch: the channel-major copy (b, s, h, d) view, or (sfd,
    lp_docmaps.V_ROWS on a packed call) the row codes themselves (an MN-major operand; no channel-major copy made)."""
    if mv.get("codes_t") is None:
        return mv["codes"]
    return mv["codes_t"].permute(0, 3, 1, 2)


def _v_cm(v) -> bool:
    """V's launch makes the channel-major copy (not on a packed lp_docmaps.V_ROWS call)."""
    from flash_attn.cute import lp_docmaps
    return not (_packed_rows(v) and (lp_docmaps.V_ROWS or lp_docmaps.compact_operand(v)))


def _fwd_kw(sp) -> dict:
    """The forward driver's spec arguments of a plan."""
    return dict(spec=sp.fwd, host_spec=sp.host, quant_spec=sp.quant)


def _rows(st, side: str) -> dict:
    """The aligned varlen fast path (lp_varlen.LPDocs.aligned_from): the quantisers' padding mask of a Q-side ('q': Q,
    dO) or K-side ('k': K, V) operand -- the caller's padding rows are read as zero rows, so they never reach a scale
    block, a code or a statistic -- only with lp_varlen.ALIGNED_ROW_PREDICATE (an A/B option; the default trusts the
    padding contract: zero, or finite and no larger than the real rows). {} elsewhere."""
    d = _docs_of(st)
    if d is None or not getattr(d, "aligned", False) or d.row_end_q is None:
        return {}   # (the aligned default trusts the padding contract: no predicate)
    return dict(row_end=d.row_end_q if side == "q" else d.row_end_k)


# =====================================================================================================================
# fp4.e4m3_pv: NVFP4 Q / K + E4M3 V (per-tensor or token-block), the dk8 backward
# =====================================================================================================================

def _mx4_rows(x):
    """(codes, SF view) of x's MXFP4 1 x 32 row copy (KernelTuning.fp4_qk_sf_vec = 32)."""
    r = quantize_fp4_rows(x, scale_group_size=32)
    return r["codes"], r["scale_factors"]


def _qk_vec(qs):
    """The forward's qk_sf_vec_size: None on the NVFP4 default (the kernel's own default, unchanged), 32 for MXFP4."""
    return 32 if int(qs.s_sf_vec) == 32 else None


def _e4pv_forward(plan: AttnPlan, st, q, k, v, masks):
    """NVFP4 Q / K (S in both passes) + E4M3 V shared by the forward's PV and the backward's dP, E4M3 dO. Every scale is
    computed fresh from this call's tensors. SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the plan's
    one operand scaling, QuantSpec.operand_scaling_fp4)."""
    from flash_attn.cute.interface_bwd_fp4 import max_row_norm, row_norms_f32, e4_blk32_fp4
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    from flash_attn.cute import qk_hadamard
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    softmax_scale, causal, deterministic = plan.softmax_scale, plan.fwd.causal, plan.bwd.deterministic
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # masks / varlen: window / softcap / score_mod / mask_mod as one explicit value, carried to the backward
    masks = lp_masks_or_none(masks, causal)
    st.masks = masks
    if masks is not None:
        causal = masks.causal
    st.had_back = False
    # varlen: token-block operands on both passes, or (explicit per-tensor operands, aligned documents) one scale per
    # (document, kv head) of every operand -- never one scale across documents
    docs = masks.docs if masks is not None else None
    st.e4blk = e4_blk32_fp4(qs)
    if docs is not None and not st.e4blk and not docs.aligned:
        raise NotImplementedError("per-tensor operands on repacked varlen documents (the plan serves per-(document, kv "
                                  "head) scales on the aligned varlen path only)")
    if st.e4blk:  # token-block scales: block-scaled E4M3 Q / K / V, no per-tensor scale at all
        return _e4pv_forward_blk32(plan, st, q, k, v, softmax_scale, causal, deterministic)
    if docs is not None:
        return _e4pv_forward_docs(plan, st, q, k, v, docs, softmax_scale, causal, deterministic)
    # sq / sk / sv on the GPU (no host read, one launch that also writes V's codes); vn2: max_k ||V_k||^2 from V's
    # quantiser read, for the backward's dS bound (fused_ds_bound)
    # (token-block scales: under the default block-scaled dS the bound is not read; P_max is still requested -- the forward
    # variant without it measured 4-8 % slower on dense cells -- and the fused bound costs nothing extra)
    want_pmax = True
    fuse = fused_ds_bound_ok(q, qs)
    # V's row norm comes from its quantiser on the token-tile layout (a ragged V takes the torch norm, as before)
    v_norm_in_quantiser = fuse and v.shape[1] % 128 == 0 and v.shape[-1] == 128
    if qk_hadamard.enabled(qs):
        # Q R / K R in the same quantiser launches (and the rotated amax in the scale launch); V plain, its dS-bound slot
        # a fresh zero (fp4_qk_operands refuses the 16 x 16 Q blocks)
        ops = qk_hadamard.fp4_qk_operands(q, k, v, qspec=qs)
        s3 = ops["s"]
        q4, sfq, q8, k4, sfk, k8 = (ops[n] for n in ("q4", "sfq", "q8", "k4", "sfk", "k8"))
        st.had_back = ops["back"]
        rv = quantize_fp8_per_tensor(v, scales=s3[2:3], output_row_norm2_max=v_norm_in_quantiser)
        v8 = rv["codes"]
    else:
        # Q / K: scales only (their E4M3 copies come from the NVFP4 quantiser's read); V: codes too
        rv = quantize_fp8_per_tensor([q, k, v], output_codes=(False, False, True),
                                     output_row_norm2_max=(False, False, v_norm_in_quantiser),
                                     power_of_two_scale=qs.per_tensor_pow2)
        s3, v8 = rv["scales"], rv["codes"][2]
        # one read per Q / K: NVFP4 codes (S in both passes) + the E4M3 copy of the backward's dK / dQ GEMMs
        rq = quantize_fp4_rows(q, also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=s3[0:1])
        q4, sfq, q8 = rq["codes"], rq["scale_factors"], rq["e4m3_copy"]
        if q_blk2d(qs):
            q4, sfq = _q4_blk2d(q)
        rk = quantize_fp4_rows(k, also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=s3[1:2])
        k4, sfk, k8 = rk["codes"], rk["scale_factors"], rk["e4m3_copy"]
        if int(qs.s_sf_vec) == 32:  # KernelTuning.fp4_qk_sf_vec = 32 (opt-in): MXFP4 S operand in both passes
            # (the per-tensor quantiser's NVFP4 codes above are discarded: E4M3 copies only from that read)
            q4, sfq, k4, sfk = _mx4_rows(q) + _mx4_rows(k)
    st.sq, st.sk = s3[0], s3[1]
    sv = s3[2]
    vn2 = None
    if v_norm_in_quantiser:
        vn2 = rv["row_norm2_max"][-1:]
    elif fuse:
        vn2 = row_norms_f32(v).amax().square().reshape(1)
    e2 = torch.float4_e2m1fn_x2
    desc = v_descale_dev(sv, v.shape[0], q.shape[2])
    # P_max feeds the dS bound of the E4M3 dQ operand
    out = torch.empty_like(q)
    _mk = _fwd_masks(st.masks, causal, q, k, hs)
    res = _flash_attn_fwd(q4.view(e2), k4.view(e2), v8, softmax_scale=softmax_scale, mSFQ=sfq, mSFK=sfk,
                          v_descale=desc, return_lse=True, return_pmax=want_pmax, out=out, qk_sf_vec_size=_qk_vec(qs),
                          use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp))
    out, lse = res[0], res[1]
    pmax = res[2] if want_pmax else None
    st.has_pmax = want_pmax
    # save_bf16_inputs=False (HostSpec.save_lean): do not keep the bf16 Q / K / V. The default dk8 backward reads them for
    # metadata only, plus one scalar max_k ||V_k|| (the dS scale bound) -- computed here, from the same data, bitwise the
    # value the backward would compute. Saved bytes per (token, head) at d=128: 1560 -> 792 (BF16 keeps 1028).
    # Requires the default operand modes (dk8, D from dO8, P_max present); otherwise the bf16 tensors are kept.
    lean = (hs.save_lean
            and qs.d_from_do8 and q.shape[1] % 128 == 0
            and hs.bwd_op_mode == "fp4")
    st.lean = lean
    st.vmax = None
    if lean:
        st.vmax = max_row_norm(v) if vn2 is None else None  # the fused bound reads vn2 instead (no pass)
        st.qkv_meta = (q.shape, k.shape, v.shape, q.dtype, q.device)
        st.save_for_backward(out, lse, q4, sfq, k4, sfk, v8, q8, k8, pmax, vn2)
    else:
        st.save_for_backward(q, k, v, out, lse, q4, sfq, k4, sfk, v8, q8, k8, pmax, vn2)
    st.sv = sv
    st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    return out, lse


def _e4pv_forward_docs(plan: AttnPlan, st, q, k, v, docs, softmax_scale, causal, deterministic):
    """Per-tensor E4M3 operands on aligned varlen documents: one scale per (document, kv head) of Q / K / V (and dO in
    the backward), reduced over each document's real rows only. One quant.fp8 launch pair for the three scales (and V's
    codes), one quant.fp4 launch per Q / K (the NVFP4 S codes and, from the same read, the E4M3 copy with its document's
    scale); the forward reads V's descale per (document, kv head). Mirrors _e4pv_forward's per-tensor branch."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    from flash_attn.cute import qk_hadamard
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    hk = k.shape[2]
    had = qk_hadamard.enabled(qs)
    plain = had and qk_hadamard.e4m3_mode(qspec=qs) == "plain"
    if had:
        qk_hadamard.check_shape(q, "q")
        qk_hadamard.refuse_q_blk2d(qs)
    v_norm = v.shape[-1] == 128   # max ||V_k||^2 for the f16 dQ range, from V's read
    # the E4M3 scales of the copies (of the rotated Q / K unless qk_hadamard_e4m3 = plain; V plain) and V's codes: one
    # launch pair, per (document, kv head); the forward's V descale table comes out of the same launch
    rot = (had and not plain, had and not plain, False)
    from flash_attn.cute.lp_docmaps import per_tensor_packed
    _pkd = per_tensor_packed(docs)   # (cov1, compact maps: the quantisers read / write the caller's packed rows)
    rv = quantize_fp8_per_tensor([q, k, v], apply_hadamard_rotation=rot, output_codes=(False, False, True),
                                 hadamard_rotation_method=qk_hadamard.rotation_method(qs) if any(rot) else None,
                                 output_row_norm2_max=(False, False, v_norm), power_of_two_scale=qs.per_tensor_pow2,
                                 documents=dict(docs.doc_quant_args(("q", "k", "k"), hk), derived=((2, -1, 1.0),),
                                                **({"packed": _pkd} if _pkd is not None else {})))
    s3, v8, dsc = rv["scales"], rv["codes"][2], rv["doc_scales"]
    vn2 = rv["row_norm2_max"][2:3] if v_norm else None
    # one quant.fp4 read per Q / K: the NVFP4 S codes (rotated with qk_hadamard) and the E4M3 copy coded with its
    # document's scale (the padding rows: the padding contract, as the token-block path's codes)
    hkw = dict(apply_hadamard_rotation=True, hadamard_sign_seed=qk_hadamard.SEED,
               hadamard_rotation_method=qk_hadamard.rotation_method(qs), e4m3_copy_of_unrotated_values=plain) \
        if had else {}
    # (packed: the NVFP4 S codes and E4M3 copies at the packed rows, the SF bytes dense per row -- sf4_rows)
    pkq = dict(packed_rows=(_pkd[0], _pkd[2][0:1])) if _pkd is not None else {}
    pkk = dict(packed_rows=(_pkd[1], _pkd[2][1:2])) if _pkd is not None else {}
    rq = quantize_fp4_rows(q, also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=dsc[0],
                           e4m3_copy_documents=docs.tile_doc_q, **hkw, **pkq)
    rk = quantize_fp4_rows(k, also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=dsc[1],
                           e4m3_copy_documents=docs.tile_doc(128), **hkw, **pkk)
    q4, sfq, q8 = rq["codes"], rq["scale_factors"], rq["e4m3_copy"]
    if q_blk2d(qs):   # (as the batched per-tensor branch: the S codes of 16 queries x 16 d blocks, inside a document)
        q4, sfq = _q4_blk2d(q, rq if _pkd is not None else None)
    k4, sfk, k8 = rk["codes"], rk["scale_factors"], rk["e4m3_copy"]
    st.had_back = had and not plain
    st.sq, st.sk, st.sv = s3[0], s3[1], s3[2]
    st.doc_scales, st.k_global = dsc, s3[1:2]
    e2 = torch.float4_e2m1fn_x2
    out = torch.empty_like(q)   # aligned documents: padding rows unspecified (the dO launch zeroes them out of D)
    _mk = _fwd_masks(st.masks, causal, q, k, hs)
    _q, _k, _v, _o = _unbatch(docs, q4.view(e2), k4.view(e2), v8, out)
    sfd_kw = {}
    st.sfd_sf4 = None
    if _pkd is not None:   # (packed: the forward builds the NVFP4 S atoms from the dense words; V has no SF)
        from flash_attn.cute.interface_bwd_fp4 import sfd_placeholder
        st.sfd_sf4 = (rq["sf4_rows"], rk["sf4_rows"])
        _ph = sfd_placeholder(q.device)
        sfd_kw = dict(sfd_rows=[_ph, _ph, _ph, rq["sf4_rows"], rk["sf4_rows"]])
    res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale, mSFQ=sfq, mSFK=sfk,
                          v_descale=rv["doc_derived"][0][:docs.b], return_lse=True, out=_o,
                          use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp), **sfd_kw)
    lse = res[1].unsqueeze(0)
    st.has_pmax = False
    st.lean = True   # the backward reads the codes only (shape carriers for q / k / v)
    st.vmax = None
    st.qkv_meta = (q.shape, k.shape, v.shape, q.dtype, q.device)
    st.save_for_backward(out, lse, q4, sfq, k4, sfk, v8, q8, k8, None, vn2)
    st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    return out, lse


def _fold_masks_ok(st):
    """The fold forward (token-local operands / KernelTuning.token_block_forward='fold') takes documents (varlen /
    seqused_k: the tables read at the document's offsets, flash_fwd_sm100_fp4 tb_varlen) but no window / softcap /
    score_mod / mask_mod (its score transform runs without the mod / mask paths)."""
    m = getattr(st, "masks", None)
    if m is not None and (m.local or m.any_mod):
        raise NotImplementedError("the fold forward (operand_scaling='token_local' / KernelTuning.token_block_forward="
                                  "'fold') does not take window / softcap / score_mod / mask_mod; use the default "
                                  "token-block forward ('mx')")


def _e4pv_forward_blk32(plan: AttnPlan, st, q, k, v, softmax_scale, causal, deterministic):
    """Token-block / 32 x 32 E4M3 operands (operand_scaling 'token_block' / 'block_32x32' / 'token_local'): no
    per-tensor scale anywhere and no host read. Q / K: NVFP4 1x16 for S (both passes) + block-scaled MXFP8 copies (the dK /
    dQ B operands, d-rows SFs). V: ONE block-scaled quantisation -- the transposed codes (keys contiguous) with the
    keys-grouped SF replication feed the forward's MXFP8 PV (NVFP4 QK + MXFP8 PV), the row copy with the d-grouped
    replication feeds the dP GEMM."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.interface_bwd_fp4 import mx_cols
    from flash_attn.cute import qk_hadamard
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    had = qk_hadamard.enabled(qs)
    # qk_hadamard_e4m3 = plain: the E4M3 copies (the dQ / dK GEMMs' operands) of the unrotated Q / K, only the NVFP4 S
    # codes rotated -- the gradients come out in the true basis (no back-rotation anywhere)
    e4m3_plain = had and qk_hadamard.e4m3_mode("blocks", qspec=qs) == "plain"
    st.had_back = had and not e4m3_plain
    # SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling
    cols = mx_cols(getattr(st, "masks", None), quant=qs, host=hs)
    st.op_cols = cols  # the backward hands these codes back with their width: checked against its resolved scaling
    b_, s_, h_, _ = q.shape
    hk_ = k.shape[2]
    # the (4, b, h, s / 32) factor table: rows Q / K / V / dO, each written whole by its quantiser launch when
    # h_k == h (no fill launch then); under GQA rows K / V cover h_k heads, the rest stay zero
    # (the kernels read rows K / V at kv heads only: the uncovered GQA rows are never read, so no zero fill either)
    fac = token_block_factor_table(q, k) if cols == 128 else None
    if had:
        # qk_hadamard (SCALING-CONSISTENCY RULE): ONE rotated read per Q / K gives the NVFP4 S codes of Q R / K R AND their
        # E4M3 token-block copies, the block exponents taken on the rotated rows (or, e4m3_plain, the E4M3 copies of
        # Q / K themselves). The forward multiplies these codes, the backward re-reads exactly them (saved below) and,
        # with rotated E4M3 copies, rotates dQ / dK back in its epilogues (had_signs).
        mq, mk = qk_hadamard.tb_qk_operands(q, k, cols=cols, fac=fac, fp4=True, k_statistics=True,
                                            e4m3_of_unrotated=e4m3_plain, qspec=qs,
                                            row_end_q=_rows(st, "q").get("row_end"),
                                            row_end_k=_rows(st, "k").get("row_end"))
        q4, sfq, k4, sfk = mq["x4"], mq["sf"], mk["x4"], mk["sf"]
    else:
        # with cols = d the token-block launch of Q / K also writes S's NVFP4 1x16 codes and E4M3 SFs from its
        # own read (the S operand's encoding, nvfp4_copy_encoding="qk_operand": bitwise quantize_fp4_rows' RTN codes and
        # SF bytes, same layouts), so Q and K are read once instead of twice (a separate-launch variant is not
        # shipped).
        mx4 = int(qs.s_sf_vec) == 32   # KernelTuning.fp4_qk_sf_vec = 32: MXFP4 S operand (below), no fused NVFP4 copy
        # (lp_docmaps packed rows at d = 64 too: the fused copy is bitwise quantize_fp4_rows' there, and the packed
        # rows have no quant.fp4 path)
        nv4 = (cols == q.shape[-1] or _packed_rows(q)) and not mx4
        q_nv4 = nv4 and not q_blk2d(qs)
        kw_nv4 = dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand")
        mq = quantize_fp8_blocks(q, block_columns=cols, output_token_block_factors=fac_row(fac, FAC_Q, q),
                                 **(kw_nv4 if q_nv4 else {}), **_rows(st, "q"))
        tl = plan.fwd.operand_scaling == "token_local"
        tok = token_table(q, k) if tl else None
        st.tok = tok
        # token-local K: per-token codes; its dQ-side scale is unit (f_K folds into the dQ side of dS)
        tlk = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 1, k), token_axis_scale="unit") \
            if tl else {}
        mk = quantize_fp8_blocks(k, block_columns=cols, output_amax=True,
                                 output_token_block_factors=fac_row(fac, FAC_K, k),
                                 **tlk, **(kw_nv4 if nv4 else {}), **_rows(st, "k"))
        if mx4:
            # the S operand is MXFP4 (UE8M0 per 32) in both passes; the E4M3 copies above are unchanged. One extra read
            # of Q and K (the quantiser's fused 4-bit copy is NVFP4 only)
            q4, sfq, k4, sfk = _mx4_rows(q) + _mx4_rows(k)
        else:
            if q_nv4:
                q4, sfq = mq["x4"], mq["sf4"]
            else:
                if q_blk2d(qs) and _rows(st, "q"):
                    raise NotImplementedError("q_block_2d with lp_varlen.ALIGNED_ROW_PREDICATE: the 2D-block quantiser "
                                              "has no padding-row predicate (the padding contract serves it)")
                q4, sfq = _q4_blk2d(q, mq if _packed_rows(q) else None) if q_blk2d(qs) else _nv4_rows(q, **_rows(st, "q"))
            k4, sfk = (mk["x4"], mk["sf4"]) if nv4 else _nv4_rows(k, **_rows(st, "k"))
    fold = tb_fold_fwd(cols, qs)
    tok = getattr(st, "tok", None)
    tlv = {}
    if tok is not None:
        # token-local V: the fold forward (P' = P f_V(key), the row sum weighted 1 / f_V)
        fold = True
        tlv = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 2, v),
                   output_token_log2_factors=tok_row(tok, 4, v), output_token_inverse_factors=tok_row(tok, 5, v))
    mv = quantize_fp8_blocks(v, make_channel_major_copy=not fold and _v_cm(v), block_columns=cols,
                             output_row_norm2_max=True,
                             output_token_block_factors=fac_row(fac, FAC_V, v), **tlv, **_rows(st, "k"))
    st.fac = fac
    if mq.get("ev_rows") is not None:   # sfd: the dense exponent rows of Q / K / V (the backward's SF atoms, factors)
        st.sfd_ev = (mq["ev_rows"], mk["ev_rows"], mv["ev_rows"])
        if mq.get("sf4_rows") is not None and mk.get("sf4_rows") is not None:   # the NVFP4 S operand's dense SF words
            st.sfd_sf4 = (mq["sf4_rows"], mk["sf4_rows"])
    # the backward's f16 dQ range (interface_bwd_fp4.dq_range_of): max|K| and max ||V_k||^2 from these launches
    st.dq_range_kv = (mk["amax"], mv["row_norm2_max"])
    st.dq_kv_rows = _dq_kv_rows(st, k, v, causal)   # dq-causal (knob): per-row prefix K / V statistics, else None
    e2 = torch.float4_e2m1fn_x2
    if fold:  # token blocks: plain E4M3 PV, f_V folded into the scores (no block-scaled PV GEMM)
        _fold_masks_ok(st)
        docs = _docs_of(st)
        out = _o_buf(q, docs)
        _mk = _fwd_masks(getattr(st, "masks", None), causal, q, k, hs)
        _q, _k, _v, _o = _unbatch(docs, q4.view(e2), k4.view(e2), mv["codes"], out)
        res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale,
                              mSFQ=sfq, mSFK=sfk, v_descale=None, qk_sf_vec_size=_qk_vec(qs),  # unit: V's factors are in the scores
                              return_lse=True, out=_o, tb_fac=tl_fold_fac(q, fac), tb_tok_v=tok, **_mk, **_fwd_kw(sp),
                              **(_sfd_fwd(mq, mk, mv, sf4=True) if "kv_cu_seqlens_packed" in _mk else {}))
        lse = res[1] if docs is None else res[1].unsqueeze(0)
        out = res[0] if docs is None else out
    else:
        docs = _docs_of(st)
        # repacked varlen: padding rows stay 0 (D = 0). Aligned varlen: O is not zero-filled (not free): its padding
        # rows are unspecified, and the dO quantiser masks O's padding rows out of D (row_end)
        out = _o_buf(q, docs)
        _mk = _fwd_masks(getattr(st, "masks", None), causal, q, k, hs)
        _q, _k, _v, _o = _unbatch(docs, q4.view(e2), k4.view(e2), _v_operand(mv), out)
        res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale,
                              mSFQ=sfq, mSFK=sfk, mSFV=mv["sf_dim"], return_lse=True, out=_o, qk_sf_vec_size=_qk_vec(qs),
                              mxfp8_pv_row_p=hs.fwd4_row_p, use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp),
                              **(_sfd_fwd(mq, mk, mv, sf4=True) if "kv_cu_seqlens_packed" in _mk else {}))
        lse = res[1] if docs is None else res[1].unsqueeze(0)  # (h, s) -> (1, h, s)
        out = res[0] if docs is None else out
    st.qkv_meta = (q.shape, k.shape, v.shape, q.dtype, q.device)
    st.save_for_backward(out, lse, q4, sfq, k4, sfk, mq["codes"], mq["sf_dim_raw"], mk["codes"], mk["sf_dim_raw"],
                         mv["codes"], mv["sf_tok_raw"], mq["e"], mk["e"], mv["e"])
    st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    return out, lse


def _dq_kv_rows(st, k, v, causal):
    """dq-causal (module switch interface_bwd_fp4.DQ_CAUSAL_PRESCALE): the per-row prefix K / V statistics the backward
    sizes its f16 dQ pre-scale from; None when off, not causal, or no input needs a gradient (inference)."""
    from flash_attn.cute.interface_bwd_fp4 import DQ_CAUSAL_PRESCALE, dq_causal_kv_rows
    if not (DQ_CAUSAL_PRESCALE and causal) or not any(getattr(st, "needs_input_grad", (True,))[:3]):
        return None
    return dq_causal_kv_rows(k, v)


def _e4pv_backward(plan: AttnPlan, st, dout):
    if getattr(st, "e4blk", False):
        return _e4pv_backward_blk32(plan, st, dout)
    from flash_attn.cute.interface_bwd_fp4 import shape_carrier
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    if st.lean:
        out, lse, q4, sfq, k4, sfk, v8, q8, k8, pmax, vn2 = st.saved_tensors
        qsh, ksh, vsh, dt, dev = st.qkv_meta
        q, k, v = shape_carrier(qsh, dt, dev), shape_carrier(ksh, dt, dev), shape_carrier(vsh, dt, dev)
    else:
        q, k, v, out, lse, q4, sfq, k4, sfk, v8, q8, k8, pmax, vn2 = st.saved_tensors
    dout = dout.contiguous()
    sc = {"vmax": st.vmax} if st.vmax is not None else {}
    docs = _docs_of(st)
    # D = sum(dO O) from exactly the E4M3 dO the dP GEMM uses (softmax-consistent; bf16 dO in D drifted late in training)
    if docs is not None:
        # per-(document, kv head) scales: the dO launch also folds D per document and writes +inf as the shifted LSE of
        # every padding row (no preprocess, no torch chain)
        from flash_attn.cute.interface_bwd_fp4 import e4m3_p_shift_log2
        if not (qs.d_from_do8 and dk8_prologue_fusable(hs) and dout.shape[-1] == 128):
            raise NotImplementedError("per-tensor varlen on the 4-bit route: the fused dk8 prologue (d = 128) only")
        fold = (st.doc_scales[2], float(2 ** e4m3_p_shift_log2()), "dk8") if qs.ds_fold else None
        from flash_attn.cute.lp_docmaps import per_tensor_packed
        _pkd = per_tensor_packed(docs)
        do8, sdo, extra = dk8_backward_prologue(dout, out.contiguous(), lse, qspec=qs, hspec=hs, documents=dict(
            docs.doc_quant_args(("q",), k.shape[2]), fold=fold, **({"packed": _pkd} if _pkd is not None else {})))
        if getattr(st, "sfd_sf4", None) is not None:   # (cov1, packed: the NVFP4 S operand's dense SF words)
            sc["sfd_sf4"] = st.sfd_sf4
        sc.update(extra)
        sc.update(doc_d_folded=bool(qs.ds_fold),
                  doc_scales=dict(q=st.doc_scales[0], k=st.doc_scales[1], v=st.doc_scales[2],
                                  do=extra.pop("doc_scales_do"), k_global=st.k_global))
    elif qs.d_from_do8 and dout.shape[1] % 128 == 0:
        bound = None
        if vn2 is not None and pmax is not None and dk8_prologue_fusable(hs):
            bound = (pmax.float().contiguous(), vn2)
        do8, sdo, extra = dk8_backward_prologue(dout, out.contiguous(), lse, bound_inputs=bound, qspec=qs, hspec=hs)
        sc.update(extra)
    else:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        r = quantize_fp8_per_tensor(dout, power_of_two_scale=qs.per_tensor_pow2)
        do8, sdo = r["codes"], r["scales"][0]
    sc.update(v8=v8, sv=st.sv, do8=do8, sdo=sdo, q8=q8, sq=st.sq, k8=k8, sk=st.sk)
    if vn2 is not None:  # the f16 dQ range: V's max row norm from its quantiser (dn2 from the dO prologue)
        sc["vn2"] = vn2
    if pmax is not None:
        sc["pmax"] = pmax
    if st.had_back:
        sc["had_signs"] = True
    dq, dk, dv = _flash_attn_bwd_fp4(
        q, k, v, out, dout, lse, softmax_scale=st.softmax_scale, causal=st.causal, deterministic=st.deterministic,
        q4=q4, sfq=sfq, k4=k4, sfk=sfk, fp8_scales=sc, masks=getattr(st, "masks", None),
        ds_dq_rowlocal=plan.bwd.ds_dq_rowlocal, specs=sp,
    )
    if st.had_back:
        from flash_attn.cute import qk_hadamard
        dq, dk, _where = qk_hadamard.finish_grads(dq, dk)
    return dq, dk, dv


def _e4pv_backward_blk32(plan: AttnPlan, st, dout):
    from flash_attn.cute.interface_bwd_fp4 import shape_carrier
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.interface_bwd_fp4 import mx_cols
    sp = plan.specs
    qsp, hs = sp.quant, sp.host
    out, lse, q4, sfq, k4, sfk, q8, sfqd, k8, sfkd, v8, sfvt, eq, ek, ev = st.saved_tensors
    qs, ks, vs, dt, dev = st.qkv_meta
    dout = dout.contiguous()
    fac = getattr(st, "fac", None)
    glue = tb_bwd_glue(dout, lse, getattr(st, "masks", None), fp8=False, quant=qsp, host=hs) \
        or _mx_do_prologue(st, dout, lse, host=hs)
    if "lse_row_end" not in glue:  # (aligned documents: the dO launch writes the padding rows' +inf itself)
        lse = _lse_for_bwd(st, lse)
    tok = getattr(st, "tok", None)
    # token-local dO: unit dV-side scale (f_dO folds into the block-scaled P^ of dV)
    tld = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 3, dout), token_axis_scale="unit") \
        if tok is not None else {}
    mdo = quantize_fp8_blocks(dout, compute_d_with_output=out.contiguous(),
                              block_columns=mx_cols(getattr(st, "masks", None), quant=qsp, host=hs),
                              output_token_block_factors=fac_row(fac, FAC_DO, dout), output_row_norm2_max=True,
                              **tld, **glue, **_rows(st, "q"))
    tb_glue_tag(mdo, glue)
    c = st.op_cols
    _ev = getattr(st, "sfd_ev", None) or (None, None, None)   # sfd: the dense exponent rows
    mx = dict(q=dict(codes=q8, sf_dim_raw=sfqd, e=eq, cols=c, ev_rows=_ev[0]),
              k=dict(codes=k8, sf_dim_raw=sfkd, e=ek, cols=c, ev_rows=_ev[1]),
              v=dict(codes=v8, sf_tok_raw=sfvt, e=ev, cols=c, ev_rows=_ev[2]), do=mdo, fac=fac,
              owned=("do",))  # mdo is this backward's own: its D may be folded in place
    if tok is not None:
        mx["tok"] = tok
    sc = {"mx": mx}
    if getattr(st, "sfd_sf4", None) is not None:
        sc["sfd_sf4"] = st.sfd_sf4
    if getattr(st, "dq_range_kv", None) is not None:  # the f16 dQ range: dO's norm from this launch, K / V's saved
        sc.update(dn2=mdo["row_norm2_max"], kmax=st.dq_range_kv[0], vn2=st.dq_range_kv[1])
        if getattr(st, "dq_kv_rows", None) is not None:
            sc["dq_kv_rows"] = st.dq_kv_rows
    had = getattr(st, "had_back", False)
    if had:  # the saved codes are Q R / K R: the dK epilogue and the dQ postprocess rotate the gradients back
        sc["had_signs"] = True
    dq, dk, dv = _flash_attn_bwd_fp4(
        shape_carrier(qs, dt, dev), shape_carrier(ks, dt, dev), shape_carrier(vs, dt, dev), out, dout, lse,
        softmax_scale=st.softmax_scale, causal=st.causal, deterministic=st.deterministic,
        q4=q4, sfq=sfq, k4=k4, sfk=sfk, fp8_scales=sc, masks=getattr(st, "masks", None),
        ds_dq_rowlocal=plan.bwd.ds_dq_rowlocal, specs=sp,
    )
    if had:
        from flash_attn.cute import qk_hadamard
        dq, dk, _where = qk_hadamard.finish_grads(dq, dk)
    return dq, dk, dv


# =====================================================================================================================
# fp8: E4M3 Q / K / V / dO, the 8-bit forward and the E4M3 backward
# =====================================================================================================================

def _fp8_forward(plan: AttnPlan, st, q, k, v, prequant=None, masks=None):
    """All-FP8 attention: E4M3 Q / K / V in the forward, and a backward whose five GEMMs are all FP8. The forward and
    the backward consume the *same* v8 codes, so D = rowsum(dO O) is consistent by construction. O is kept in bf16
    deliberately: the pure-FP8 route's default E4M3 output rounds O, which breaks the backward's D."""
    if prequant is not None and "fa3" in prequant:   # FA3-style E4M3 q / k / v + (b, h_kv) descales: no quantiser
        from .prequant_fa3 import fp8_forward as _fa3_forward
        return _fa3_forward(plan, st, q, k, v, prequant["fa3"], masks)
    from flash_attn.cute.interface_bwd_fp4 import op_blk32_fp8
    from flash_attn.cute.device_scales import dev32
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    softmax_scale, causal, deterministic = plan.softmax_scale, plan.fwd.causal, plan.bwd.deterministic
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the plan's one operand scaling of Q, K, V)
    # masks / varlen: window / softcap / score_mod / mask_mod as one explicit value, carried to the backward
    masks = lp_masks_or_none(masks, causal)
    st.masks = masks
    if masks is not None:
        causal = masks.causal
    # varlen: token-block operands on both passes, or (explicit per-tensor operands, aligned documents) one scale per
    # (document, kv head) of every operand -- never one scale across documents
    docs = masks.docs if masks is not None else None
    st.mx = op_blk32_fp8(qs)
    if docs is not None and not st.mx and not docs.aligned:
        raise NotImplementedError("per-tensor operands on repacked varlen documents (the plan serves per-(document, kv "
                                  "head) scales on the aligned varlen path only)")
    if st.mx:  # token-block scales: MXFP8 Q / K / V, no per-tensor scale at all
        from flash_attn.cute import qk_hadamard
        if qk_hadamard.enabled(qs) and prequant is not None:
            raise ValueError("qk_hadamard with caller-supplied Q / K codes: rotate before quantising (quant.fp8 / "
                             "quant.fp4 take apply_hadamard_rotation=) and pass qk_hadamard=False")
        if prequant is not None:
            raise ValueError("block-scaled Q / K / V carry no per-tensor scales (q_quantized= is a per-tensor E4M3 "
                             "feature)")
        return _fp8_forward_mx(plan, st, q, k, v, softmax_scale, causal, deterministic)
    # ---- prequantised-qk-fused-scale: caller-supplied Q / K codes skip the forward's quantiser entirely. That is the whole
    # measurable saving of the feature -- one HBM round trip per tensor per step plus its launch (the backward
    # still consumes the same codes, so nothing downstream changes).
    pq = dict(prequant or {})
    pq_q, pq_k = pq.get("q"), pq.get("k")
    vn2 = None
    q_scale_fused = bool(pq.get("q_scale_fused", False))
    from flash_attn.cute import qk_hadamard
    had = qk_hadamard.enabled(qs)
    st.had = had
    if had and pq_q is not None:
        raise ValueError("qk_hadamard with caller-supplied Q / K codes: rotate before quantising (quant.fp8 / quant.fp4 "
                         "take apply_hadamard_rotation=) and pass qk_hadamard=False")
    if (pq_q is None) != (pq_k is None):
        raise ValueError("flash_attn_fp8: q_quantized and k_quantized must be supplied together")
    if pq_q is not None:
        from flash_attn.cute import prequant as _pq
        pq_q = _pq.QuantizedOperand(*pq_q) if isinstance(pq_q, (tuple, list)) else pq_q
        pq_k = _pq.QuantizedOperand(*pq_k) if isinstance(pq_k, (tuple, list)) else pq_k
        for nm, op, t in (("q_quantized", pq_q, q), ("k_quantized", pq_k, k)):
            if op.fmt != "e4m3":
                raise ValueError(f"flash_attn_fp8: {nm} must be E4M3 per-tensor, got {op.fmt!r}")
            op.validate(nm, shape=tuple(t.shape), head_dim=t.shape[-1])
        q8, k8 = pq_q.codes, pq_k.codes
        # the backward re-derives S from these same codes and needs the plain descales, so it gets the
        # unfolded numbers verbatim; only the forward's score scalar carries the fold.
        sq, sk = dev32(pq_q.unfused_scale(), q.device), dev32(pq_k.unfused_scale(), q.device)
        r = quantize_fp8_per_tensor(v, power_of_two_scale=qs.per_tensor_pow2)
        v8, sv = r["codes"], r["scales"][0]
    elif q_scale_fused:
        raise ValueError("q_scale_fused=True needs q_quantized (the fold lives in the supplied Q's descale)")
    else:
        # ONE launch for the three E4M3 operands with their scales computed on the GPU; qk_hadamard: Q R / K R rotated in
        # registers, scales from the rotated amax, V plain. vn2: max_k ||V_k||^2 from V's read, for the backward's f16 dQ
        # range (fp8-dq-f16-overflow), where the operands are token tiles
        if had:
            qk_hadamard.check_shape(q, "q")
        # (lp_docmaps packed rows: the aligned rows the operands stand for -- the fused path's shapes)
        from flash_attn.cute.lp_docmaps import aligned_rows, per_tensor_packed
        norm_ok = (all(aligned_rows(t) % 128 == 0 for t in (q, k, v))
                   and len({t.shape[-1] for t in (q, k, v)}) == 1 and q.shape[-1] in (64, 128, 256))
        # aligned documents: one scale per (document, kv head) of Q / K / V from the document's real rows (the same
        # launches), plus the forward kernel's per-document score scale float32((s_q softmax_scale) s_k) and V descale
        _pkd = per_tensor_packed(docs)   # (packed rows: the quantiser reads / writes the caller's packed rows)
        dkw = {} if docs is None else dict(documents=dict(
            docs.doc_quant_args(("q", "k", "k"), k.shape[2]), derived=((0, 1, softmax_scale), (2, -1, 1.0)),
            **({"packed": _pkd} if _pkd is not None else {})))
        r = quantize_fp8_per_tensor([q, k, v], apply_hadamard_rotation=(had, had, False),
                                    hadamard_rotation_method=qk_hadamard.rotation_method(qs) if had else None,
                                    output_row_norm2_max=(False, False, norm_ok), power_of_two_scale=qs.per_tensor_pow2,
                                    **dkw)
        q8, k8, v8 = r["codes"]
        sq, sk, sv = r["scales"][0], r["scales"][1], r["scales"][2]
        if norm_ok:
            vn2 = r["row_norm2_max"][2:3]
    b, hk = q.shape[0], k.shape[2]
    if docs is not None:
        # (n_docs, h_kv) views of the per-document tables (row n_docs is the sentinel)
        st_score, st_vdesc = r["doc_derived"][0][:docs.b], r["doc_derived"][1][:docs.b]
        st.doc_scales = r["doc_scales"]   # (3, n_docs + 1, h_kv) float32: Q / K / V, for the backward
        st.k_global = r["scales"][1:2]    # the K launch's global scale (max over the documents): the f16 dQ range
    # Upstream's own forward, not the block-scaled driver, where the plan picked it (forward_kernel 'upstream', or the
    # per-tensor operands the fork kernel cannot serve): it supports per-tensor FP8 natively (q/k/v descales, folded into
    # softmax_scale and the epilogue respectively).
    from flash_attn.cute.interface import _flash_attn_fwd as _upstream_fwd
    full = lambda x: v_descale_dev(x, b, hk)  # noqa: E731  (b, h_k) float32 from a device scale: no host read
    out = torch.empty_like(q)  # bf16, see the docstring
    # The fork route has no q/k descale slots: sq*sk folds into softmax_scale (the scores are the code products);
    # v_descale is applied in its epilogue. HostSpec.fp8_forward_kernel: "auto" / "fork" = the fork kernels (the generic
    # step with the mask-free unmasked loop, the fused softmax loop and the TMEM rescale factor), "upstream" = never.
    _fwd_route = hs.fp8_forward_kernel
    _use_fork = _fwd_route in ("split", "fork", "auto")
    # a score_mod (softcap) needs the plain softmax_scale, which the fork route's device score scale replaces
    # (interface_fwd_fp4: softmax_scale_t is refused with score_mod): upstream's forward serves it
    # (aligned documents: the per-(document, kv head) table is read by the fork forward's score_mod too, which applies
    # it before the mod -- upstream's forward has no per-document scale)
    _use_fork = _use_fork and (masks is None or masks.score_mod is None or docs is not None)
    _mk = _fwd_masks(masks, causal, q, k, hs)
    if _use_fork and q_scale_fused and q.shape[1] % 128 == 0 and q.shape[-1] in (64, 128):
        # the fused route goes through the driver's q_quantized / k_quantized API, which folds the descales
        # itself and derives the kernel's score scalar (and its unit-ness) from the operands
        res = _flash_attn_fwd(None, None, v8, softmax_scale=softmax_scale,
                              v_descale=full(sv), return_lse=True, out=out, force_fp4_impl=True,
                              q_quantized=pq_q, k_quantized=pq_k, q_scale_fused=True, **_mk, **_fwd_kw(sp))
    elif docs is not None:
        # aligned documents (per-(document, kv head) scales): the fork forward on the packed (total, h, d) codes; the
        # kernel reads the score scale softmax_scale s_q s_k and the V descale of each work tile's (document, kv head)
        assert _use_fork and q.shape[-1] in (64, 128), "per-tensor varlen: the fork forward at head_dim 64 / 128"
        _q, _k, _v, _o = _unbatch(docs, q8, k8, v8, out)
        res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale, softmax_scale_t=st_score,
                              v_descale=st_vdesc, return_lse=True, out=_o, force_fp4_impl=True,
                              use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp))
        res = (out, res[1].unsqueeze(0))
    elif _use_fork and q.shape[1] % 128 == 0 and q.shape[-1] in (64, 128):
        # the score scale softmax_scale s_q s_k, formed in float32 on the GPU (score_scale_dev) and read by the
        # kernel from device memory (softmax_scale_t)
        res = _flash_attn_fwd(q8, k8, v8, softmax_scale=softmax_scale,
                              softmax_scale_t=score_scale_dev(softmax_scale, sq, sk),
                              v_descale=full(sv), return_lse=True, out=out, force_fp4_impl=True,
                              use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp))
    else:
        # `upstream-fp8-v-descale-wrong`: upstream's kernel mis-applies v_descale (O cosine 0.62), so O is scaled by
        # s_v here instead: bf16(float(O) float32(s_v))
        res = _upstream_fwd(q8, k8, v8, softmax_scale=softmax_scale,
                            q_descale=full(sq), k_descale=full(sk),
                            return_lse=True, out=out, **_mk)
        res[0].copy_(res[0].float().mul_(sv.float()))
    out, lse = res[0], res[1]
    st.save_for_backward(out, lse, q8, k8, v8)
    st.vn2 = vn2
    st.sq, st.sk, st.sv = sq, sk, sv
    st.qkv_meta = (q.shape, k.shape, v.shape, q.dtype, q.device)
    st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    return out, lse


def _fp8_forward_mx(plan: AttnPlan, st, q, k, v, softmax_scale, causal, deterministic):
    """Token-block / 32 x 32 MXFP8 Q, K, V (operand_scaling 'token_block' / 'block_32x32' / 'token_local'), one
    quantisation each, consumed by the forward (Q / K 1 x 32-along-d SF replication, V's keys-grouped replication on its
    transposed codes) and by the backward (the same codes; d-rows replications for the dK / dQ B operands, V's d-grouped
    one for dP). No per-tensor scale, no host read."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.interface_bwd_fp4 import mx_cols
    sp = plan.specs
    qs, hs = sp.quant, sp.host
    # SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling
    cols = mx_cols(getattr(st, "masks", None), quant=qs, host=hs)
    st.op_cols = cols  # the backward hands these codes back with their width: checked against its resolved scaling
    b_, s_, h_, d_ = q.shape
    hk_ = k.shape[2]
    # token-block fallback shapes: a head dim other than 64 / 128 runs padded to the next of them, the padding staged as zeros by the
    # quantisers themselves (no padded copy of Q / K / V / dO); O and the gradients are the first d_ columns
    dp_ = padded_head_dim(d_)
    st.head_dim, st.head_dim_padded = d_, dp_
    pad = dict(pad_head_dim_to=dp_) if dp_ != d_ else {}
    # the (4, b, h, s / 32) factor table: rows Q / K / V / dO, each written whole by its quantiser launch when
    # h_k == h (no fill launch then); under GQA rows K / V cover h_k heads, the rest stay zero
    # (the kernels read rows K / V at kv heads only: the uncovered GQA rows are never read, so no zero fill either)
    fac = token_block_factor_table(q, k) if cols == 128 else None
    from flash_attn.cute import qk_hadamard
    had = qk_hadamard.enabled(qs)
    st.had = had
    tl = plan.fwd.operand_scaling == "token_local"
    if pad and (had or tl or tb_fold_fwd(cols, qs)):
        raise NotImplementedError(f"head_dim {d_} (padded to {dp_}): the rotation, token-local operands and the fold "
                                  "forward are built for head_dim 64 / 128")
    if had:
        # qk_hadamard (SCALING-CONSISTENCY RULE): the MXFP8 / token-block codes of Q R / K R from ONE rotated read each,
        # exponents from the rotated rows; the forward's S and the backward (S recompute, dK / dQ operands) read exactly
        # these codes
        mq, mk = qk_hadamard.tb_qk_operands(q, k, cols=cols, fac=fac, fp4=False, k_statistics=True, qspec=qs,
                                            row_end_q=_rows(st, "q").get("row_end"),
                                            row_end_k=_rows(st, "k").get("row_end"))
    else:
        tok = token_table(q, k) if tl else None
        st.tok = tok
        # token-local Q / K: per-token codes; their dK / dQ-side scales are unit (folded into dS)
        tlq = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 0, q), token_axis_scale="unit") if tl else {}
        tlk = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 1, k), token_axis_scale="unit") \
            if tl else {}
        mq = quantize_fp8_blocks(q, block_columns=cols, output_token_block_factors=fac_row(fac, FAC_Q, q),
                                 **tlq, **pad, **_rows(st, "q"))
        mk = quantize_fp8_blocks(k, block_columns=cols, output_amax=True,
                                 output_token_block_factors=fac_row(fac, FAC_K, k), **tlk, **pad, **_rows(st, "k"))
    fold = tb_fold_fwd(cols, qs)
    tok = getattr(st, "tok", None)
    tlv = {}
    if tok is not None:
        fold = True   # token-local Q / K / V run on the fold forward
        tlv = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 2, v),
                   output_token_log2_factors=tok_row(tok, 4, v), output_token_inverse_factors=tok_row(tok, 5, v))
    mv = quantize_fp8_blocks(v, make_channel_major_copy=not fold and _v_cm(v), block_columns=cols,
                             output_row_norm2_max=True,
                             output_token_block_factors=fac_row(fac, FAC_V, v), **tlv, **pad, **_rows(st, "k"))
    st.fac = fac
    if mq.get("ev_rows") is not None:   # sfd: the dense exponent rows of Q / K / V (the backward's SF atoms, factors)
        st.sfd_ev = (mq["ev_rows"], mk["ev_rows"], mv["ev_rows"])
        if mq.get("sf4_rows") is not None and mk.get("sf4_rows") is not None:   # the NVFP4 S operand's dense SF words
            st.sfd_sf4 = (mq["sf4_rows"], mk["sf4_rows"])
    # the backward's f16 dQ range (interface_bwd_fp4.dq_range_of): max|K| and max ||V_k||^2 from these launches
    st.dq_range_kv = (mk["amax"], mv["row_norm2_max"])
    st.dq_kv_rows = _dq_kv_rows(st, k, v, causal)   # dq-causal (knob): per-row prefix K / V statistics, else None
    if fold:  # token blocks: plain E4M3 QK and PV, f_Q f_K and log2 f_V folded into the scores
        _fold_masks_ok(st)
        docs = _docs_of(st)
        out = _o_buf(q, docs)
        _mk = _fwd_masks(getattr(st, "masks", None), causal, q, k, hs)
        _q, _k, _v, _o = _unbatch(docs, mq["codes"], mk["codes"], mv["codes"], out)
        res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale,
                              v_descale=None, return_lse=True,  # unit: V's factors are in the scores
                              out=_o, force_fp4_impl=True, tb_fac=tl_fold_fac(q, fac), tb_tok_v=tok, **_mk,
                              **_fwd_kw(sp))
        lse = res[1] if docs is None else res[1].unsqueeze(0)
        out = res[0] if docs is None else out
    else:
        docs = _docs_of(st)
        o_shape = (b_, s_, h_, dp_)  # the padded O (q's shape at d = 64 / 128)
        # repacked varlen: padding rows stay 0 (D = 0); aligned varlen: unspecified there (the dO quantiser masks them)
        out = _o_buf(q, docs, o_shape)
        _mk = _fwd_masks(getattr(st, "masks", None), causal, q, k, hs)
        _q, _k, _v, _o = _unbatch(docs, mq["codes"], mk["codes"], _v_operand(mv), out)
        row_p = hs.fwd8_row_p
        if row_p and fac is not None and docs is None and getattr(st, "masks", None) is None and not FWD8_FORCE_MX_QK:
            # token-block-qk-fold: plain E4M3 S GEMM with f_Q f_K folded into the softmax (bitwise the block-scaled
            # S GEMM's scores), V's factors in the block-scaled P.V GEMM
            res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale, mSFV=mv["sf_dim"], tb_fac=fac,
                                  mxfp8_pv_row_p=True, token_block_qk_fold=True, force_fp4_impl=True,
                                  return_lse=True, out=_o, use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp))
        else:
            res = _flash_attn_fwd(_q, _k, _v, softmax_scale=softmax_scale,
                                  mSFQ=mq["sf_tok"], mSFK=mk["sf_tok"], mSFV=mv["sf_dim"], return_lse=True,
                                  out=_o, mxfp8_pv_row_p=row_p, use_2cta_forward=hs.fwd_two_cta, **_mk, **_fwd_kw(sp),
                                  **(_sfd_fwd(mq, mk, mv) if "kv_cu_seqlens_packed" in _mk else {}))
        lse = res[1] if docs is None else res[1].unsqueeze(0)
        out = res[0] if docs is None else out
    st.save_for_backward(out, lse, mq["codes"], mq["sf_tok_raw"], mq["sf_dim_raw"], mk["codes"], mk["sf_tok_raw"],
                         mk["sf_dim_raw"], mv["codes"], mv["sf_tok_raw"], mq["e"], mk["e"], mv["e"])
    # the kernels' shapes: the padded head dim (the backward runs at dp_ and drops the padded gradient columns)
    st.qkv_meta = (tuple(q.shape[:3]) + (dp_,), tuple(k.shape[:3]) + (dp_,), tuple(v.shape[:3]) + (dp_,),
                   q.dtype, q.device) if pad else (q.shape, k.shape, v.shape, q.dtype, q.device)
    st.softmax_scale, st.causal, st.deterministic = softmax_scale, causal, deterministic
    # padded head dim: O as the caller's d-wide contiguous tensor (one copy; callers .view() it, and the document
    # unpacking gathers whole rows)
    return (out if not pad else out[..., :d_].contiguous()), lse


# test-only (timing A/B): force the block-scaled MXFP8 S GEMM (mSFQ / mSFK) instead of the token-block QK fold on the
# 8-bit forward; the two give bitwise-identical O / LSE. Host-side switch, no kernel change when False.
FWD8_FORCE_MX_QK = False


def _fp8_backward(plan: AttnPlan, st, dout):
    if getattr(st, "fa3_backward", None) is not None:   # batched pre-quantised E4M3 (lowprec.prequant_fa3)
        return st.fa3_backward(plan, st, dout)
    if getattr(st, "mx", False):
        return _fp8_backward_mx(plan, st, dout)
    from flash_attn.cute.interface_bwd_fp4 import shape_carrier
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    sp = plan.specs
    qsp, hs = sp.quant, sp.host
    out, lse, q8, k8, v8 = st.saved_tensors
    qs, ks, vs, dt, dev = st.qkv_meta
    dout = dout.contiguous()
    sc = {}
    pow2 = qsp.per_tensor_pow2
    # D from exactly the E4M3 dO the dP GEMM consumes, in the same pass as the quantisation
    from flash_attn.cute.interface_bwd_fp4 import grad_fp8_dtype, grad_fp8_format
    _gdt = grad_fp8_dtype(qsp)
    from flash_attn.cute.lp_docmaps import aligned_rows, per_tensor_packed
    _s_al = aligned_rows(dout)   # (lp_docmaps packed rows: the aligned rows dout stands for)
    if _s_al % 128 == 0 and _gdt is torch.float8_e4m3fn:
        # the whole prologue of the 2-CTA backward in the dO-quantiser launch: E4M3 dO, D,
        # LSE * log2 e already shifted by the kernel's P shift, and (HostSpec.dq_zero 'fused') the zeroed dQ accumulator
        from flash_attn.cute.dq_postprocess_cute import dq_zero_mode
        from flash_attn.cute.interface_bwd_2cta8 import P_SHIFT_LOG2
        from flash_attn.cute.interface_bwd_fp4 import _use_2cta_dk8
        b_, s_, h_, d_ = dout.shape
        acc = None
        from flash_attn.cute.interface_bwd_fp4 import native_d64, bwd_dq_f16
        if dq_zero_mode(hs) == "fused" and (
                (_use_2cta_dk8(shape_carrier(qs, dt, dev), shape_carrier(ks, dt, dev), st.causal, st.deterministic,
                               128, False, None, "all", None, None, None, None, quant=qsp, host=hs))
                or (native_d64(d_) and bwd_dq_f16(hs))):  # d = 64: the 1-CTA kernel's (b, h, s, d) f16 accumulator
            acc = torch.empty(b_, h_, s_ * d_, dtype=torch.float16, device=dout.device)  # the 2-CTA kernel's flat f16 accumulator
        docs = _docs_of(st)
        dkw = {}
        if docs is not None:
            # per-(document, kv head) scales on aligned documents: the 1-CTA kernel takes its whole prologue from this
            # launch (no preprocess): D (folded per document: D'' = -(1 / 2^shift) D / (s_V s_dO), the block-scaled
            # dS^'s ds_boost = 1), the shifted LSE (+inf on every padding row) and the zeroed f16 accumulator
            from flash_attn.cute.interface_bwd_fp4 import e4m3_p_shift_log2
            assert qsp.ds_scaling_fp8 != "per_tensor", "per-tensor varlen: the block-scaled dS"
            acc = torch.empty(b_, h_, s_ * d_, dtype=torch.float16, device=dout.device)
            fold = (st.doc_scales[2], 1.0 / float(2 ** e4m3_p_shift_log2())) if qsp.ds_fold else None
            _pkd = per_tensor_packed(docs)
            dkw = dict(documents=dict(docs.doc_quant_args(("q",), ks[2]), fold=fold,
                                      **({"packed": _pkd} if _pkd is not None else {})))
        lse = lse.contiguous()
        # one launch: the scale (float32 on the GPU), E4M3 dO, D, the shifted LSE, the zeroed accumulator and
        # max_q ||dO_q||^2
        r = quantize_fp8_per_tensor(dout, compute_d_with_output=out.contiguous(), shifted_log2_lse_from=lse,
                                    log2_lse_shift=float(P_SHIFT_LOG2), zero_accumulator=acc, output_row_norm2_max=True,
                                    power_of_two_scale=pow2, **dkw)
        do8, sdo, sc["dpsum"], sc["lse_log2"], dn2 = r["codes"], r["scales"], r["D"], r["lse_log2"], r["row_norm2_max"]
        if getattr(st, "vn2", None) is not None:  # the f16 dQ range (fp8-dq-f16-overflow): V's and dO's row norms
            sc.update(vn2=st.vn2, dn2=dn2)
        sc["dpsum_owned"] = True
        if acc is not None:
            sc["dq_accum"] = acc
        if docs is not None:
            sc.update(glue_lse_log2=sc["lse_log2"], glue_dq_accum=acc, glue_lse_log2_shift=float(P_SHIFT_LOG2),
                      doc_d_folded=bool(qsp.ds_fold),
                      doc_scales=dict(q=st.doc_scales[0], k=st.doc_scales[1], v=st.doc_scales[2],
                                      do=r["doc_scales"][0], k_global=st.k_global))
    elif _s_al % 128 == 0 and _gdt is torch.float8_e4m3fn:
        # the fused kernel emits E4M3 and the D reduction in one read; no E5M2 twin yet
        r = quantize_fp8_per_tensor(dout, compute_d_with_output=out.contiguous(), power_of_two_scale=pow2)
        do8, sdo, sc["dpsum"] = r["codes"], r["scales"], r["D"]   # sdo: (1,) float32, on the GPU
    else:
        r = quantize_fp8_per_tensor(dout, fp8_format=grad_fp8_format(qsp), power_of_two_scale=pow2)
        do8, sdo = r["codes"], r["scales"][0]
    sc.update(q8=q8, sq=st.sq, k8=k8, sk=st.sk, v8=v8, sv=st.sv, do8=do8, sdo=sdo)
    # q/k/v are read for shape and dtype metadata only on this path -- the operands are the
    # saved codes -- so carriers stand in for them and the bf16 activations are never kept.
    if getattr(st, "had", False):
        sc["had_signs"] = True
    dq, dk, dv = _flash_attn_bwd_fp4(
        shape_carrier(qs, dt, dev), shape_carrier(ks, dt, dev), shape_carrier(vs, dt, dev),
        out, dout, lse, softmax_scale=st.softmax_scale, causal=st.causal,
        deterministic=st.deterministic, fp8_scales=sc, masks=getattr(st, "masks", None),
        ds_dq_rowlocal=plan.bwd.ds_dq_rowlocal, specs=sp,
    )
    if getattr(st, "had", False):
        from flash_attn.cute import qk_hadamard
        dq, dk, _where = qk_hadamard.finish_grads(dq, dk)
    return dq, dk, dv


def _fp8_backward_mx(plan: AttnPlan, st, dout):
    from flash_attn.cute.interface_bwd_fp4 import shape_carrier
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.interface_bwd_fp4 import mx_cols
    sp = plan.specs
    qsp, hs = sp.quant, sp.host
    out, lse, q8, sfqt, sfqd, k8, sfkt, sfkd, v8, sfvt, eq, ek, ev = st.saved_tensors
    qs, ks, vs, dt, dev = st.qkv_meta
    dout = dout.contiguous()
    c = st.op_cols
    d_, dp_ = getattr(st, "head_dim", qs[-1]), getattr(st, "head_dim_padded", qs[-1])
    pad = dict(pad_head_dim_to=dp_) if dp_ != d_ else {}   # token-block fallback shapes: dO staged zero-padded, like Q / K / V
    glue = tb_bwd_glue(dout, lse, getattr(st, "masks", None), fp8=True, qs=qs, ks=ks, dt=dt, dev=dev,
                       causal=st.causal, deterministic=st.deterministic, head_dim=dp_, quant=qsp, host=hs) \
        or _mx_do_prologue(st, dout, lse, head_dim=dp_, host=hs)
    if "lse_row_end" not in glue:  # (aligned documents: the dO launch writes the padding rows' +inf itself)
        lse = _lse_for_bwd(st, lse)
    tok = getattr(st, "tok", None)
    tld = dict(token_local_exponents=True, output_token_factors=tok_row(tok, 3, dout), token_axis_scale="unit") \
        if tok is not None else {}
    mdo = quantize_fp8_blocks(dout, compute_d_with_output=out.contiguous(),
                              block_columns=mx_cols(getattr(st, "masks", None), quant=qsp, host=hs),
                              output_token_block_factors=fac_row(getattr(st, "fac", None), FAC_DO, dout),
                              output_row_norm2_max=True, **tld, **glue, **pad, **_rows(st, "q"))
    if pad and glue:  # the driver reads dO for its metadata only (the codes, D, the LSE prologue are in mdo)
        dout = shape_carrier(tuple(dout.shape[:3]) + (dp_,), dout.dtype, dout.device)
    elif pad:  # masks / documents: the driver's own preprocess reads the bf16 dO (D = sum O dO): the padded rows
        dout = torch.nn.functional.pad(dout, (0, dp_ - d_))
    tb_glue_tag(mdo, glue)
    _ev = getattr(st, "sfd_ev", None) or (None, None, None)   # sfd: the dense exponent rows
    mx = dict(q=dict(codes=q8, sf_tok_raw=sfqt, sf_dim_raw=sfqd, e=eq, cols=c, ev_rows=_ev[0]),
              k=dict(codes=k8, sf_tok_raw=sfkt, sf_dim_raw=sfkd, e=ek, cols=c, ev_rows=_ev[1]),
              v=dict(codes=v8, sf_tok_raw=sfvt, e=ev, cols=c, ev_rows=_ev[2]),
              do=mdo,
              fac=getattr(st, "fac", None), owned=("do",))  # the fresh dO dict: D may be folded in place
    if tok is not None:
        mx["tok"] = tok
    sc = {"mx": mx}
    if getattr(st, "dq_range_kv", None) is not None:  # the f16 dQ range: dO's norm from this launch, K / V's saved
        sc.update(dn2=mdo["row_norm2_max"], kmax=st.dq_range_kv[0], vn2=st.dq_range_kv[1])
        if getattr(st, "dq_kv_rows", None) is not None:
            sc["dq_kv_rows"] = st.dq_kv_rows
    had = getattr(st, "had", False)
    if had:  # the saved codes are Q R / K R: the dK epilogue and the dQ postprocess rotate the gradients back
        sc["had_signs"] = True
    dq, dk, dv = _flash_attn_bwd_fp4(
        shape_carrier(qs, dt, dev), shape_carrier(ks, dt, dev), shape_carrier(vs, dt, dev),
        out, dout, lse, softmax_scale=st.softmax_scale, causal=st.causal,
        deterministic=st.deterministic, fp8_scales=sc, masks=getattr(st, "masks", None),
        ds_dq_rowlocal=plan.bwd.ds_dq_rowlocal, specs=sp,
    )
    if had:
        from flash_attn.cute import qk_hadamard
        dq, dk, _where = qk_hadamard.finish_grads(dq, dk)
    if pad:  # the padded gradient columns are dropped (their operand columns were zeros)
        dq, dk, dv = (t[..., :d_].contiguous() for t in (dq, dk, dv))
    return dq, dk, dv


# =====================================================================================================================
# fp4.fp4_pv: option A (inference)
# =====================================================================================================================

def _option_a_forward(plan: AttnPlan, st, q, k, v, masks=None):  # (masks: aligned varlen documents only, vall4)
    """Option A, the 4-bit P.V forward (pv4_route): the all-FP4 forward (Q / K NVFP4 1x16, V NVFP4 16x16; causal: the
    diagonal-only key-local V) with the reproduce-forward-p export, as the inference contract fixed it. INFERENCE only
    (plan rule 36): its research backward is not shipped. The settings of the forward (the codes row
    sum, the E4M3-decoded group SFs, the value function, the all-FP4 step) are its FwdKernelSpec; the E4M3 token-block
    scaling of its Q / K copies its QuantSpec."""
    from flash_attn.cute.pv4_route import pv4_check_specs, pv4_forward
    from flash_attn.cute import qk_hadamard
    if lp_masks_or_none(masks, plan.fwd.causal) is not None and not _docs_only(masks):
        raise NotImplementedError("window / softcap / score_mod / mask_mod on option A (pv_precision='fp4', "
                                  "backward='e4m3'): the masked 4-bit P.V needs the key-local V scale, not built for "
                                  "this route; use pv_precision='fp8'")
    sp = plan.specs
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    had = qk_hadamard.enabled(sp.quant)
    pv4_check_specs(q, k, sp)
    # (the forward builds the inference state only; reproduce-forward-p is its value function's contract)
    mix = plan.bwd.exp_mix if plan.bwd.exp_mix not in ("n/a", "none") else None
    docs_m = masks if _docs_only(masks) else None
    res = pv4_forward(q, k, v, plan.softmax_scale, plan.fwd.causal, p_consistency="reproduce_forward_p",
                      softmax_exp=plan.fwd.softmax_exp, p_mix=mix, qk_had=had, masks=docs_m, specs=sp)
    st.deterministic = plan.bwd.deterministic
    return res["out"], res["lse"]


# =====================================================================================================================
# fp4.all_fp4: the all-FP4 forward and the query-row all-4-bit backward
# =====================================================================================================================

# debug builds of the all-4-bit route (the P^ reproduction gate of the tests): None, or a dict whose "fwd"
# (b, h, s_q, s_k + 9 s_k / 16) uint8 receives the forward's codes / SF bytes and "bwd" (b, h, s_q, s_k) uint8 the
# backward's reproduced codes. Private: a test hook, not configuration.
_ALL4_DBG = None


def _fwd_dbg() -> dict:
    d = _ALL4_DBG
    return {"repro_dbg": d["fwd"]} if d is not None and d.get("fwd") is not None else {}


def _bwd_dbg() -> dict:
    d = _ALL4_DBG
    return {"code_dbg": d["bwd"]} if d is not None and d.get("bwd") is not None else {}


def _docs_only(masks) -> bool:
    """masks carrying aligned varlen documents and nothing else (the 4-bit P.V routes' only masking feature)."""
    return (masks is not None and masks.docs is not None and getattr(masks.docs, "aligned", False) and not masks.local
            and not masks.any_mod and masks.mask_mod is None)


def _all4_forward(plan: AttnPlan, st, q, k, v, masks=None):
    """Every attention matmul NVFP4: the all-FP4 forward (Q^K^T, P^V^) + the query-row backward (Q^K^T, dO^V^T, dS^K^T,
    dS^T Q^T, P^v^T dO^T), stabilised per options.all_fp4_stabiliser (QuantSpec.fp4all_stab)."""
    from flash_attn.cute import interface_bwd_fp4q as iq
    sp = plan.specs
    qs = sp.quant
    softmax_scale, causal = plan.softmax_scale, plan.fwd.causal
    # p_mix (exact classifier): the per-16-key-group-position value functions of the forward's P^; the backward follows them
    p_mix = plan.bwd.exp_mix if plan.bwd.exp_mix not in ("n/a", "none") else None
    st.p_mix = p_mix
    mode = iq.fp4all_stab_mode(qs)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    docs = masks.docs if masks is not None else None
    st.docs = docs
    vmode = iq.fp4all_vtok_mode(causal=causal, qspec=qs)
    if docs is not None and vmode != ("diag" if causal else "off"):
        raise NotImplementedError(
            "the all-4-bit route on documents is built for its default routes (causal: causal_safety 'diagonal'; "
            f"dense: no key-local V); the {vmode!r} V variant is not")
    st.vtok = vmode == "full"
    st.vtok_diag = vmode == "diag"
    fkw = dict(specs=sp)
    if st.vtok_diag:
        # diagonal-only-key-local-v-scale: the key-local V scale on the causal diagonal tiles only. Both kernels carry
        # it -- the backward reproduces the bias on exactly the pairs the forward biased and reads the diagonal payload
        # as its dP operand there -- so dL/dQ_t, the only gradient with a causal restriction, leaks exactly 0 (leaky
        # 2.54e-02). The operand set differs from the full fix's in three ways, each a silent wrongness avoided: the
        # diagonal payload's row-wise V / SF go to the backward as its dP operand on the diagonal pair; the dS bound
        # needs that payload's row norms too (vtok_vmax_b); and dV needs NO 1 / s_k fold, because the hybrid kernel's dV
        # multiplies the unbiased m and is dL/dV already.
        assert mode != "eden_rht", ("vtok_diag: stabiliser sr or rtn (eden_rht rotates V, which the two payloads "
                                    "do not share)")
        assert iq.fp4all_dkv_mode(sp.bwdq) == "e4m3", "vtok_diag: the hybrid (e4m3 dK / dV) backward only"
        assert (not qs.vtok_hadv
                and not qs.vtok_hadqk), (
            "vtok_diag: the Hadamard variants are not wired for the two-payload operand set")
        sm = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        ops = iq.fp4all_quantize_qkv_diag(q, k, v, sm, docs=docs, qspec=qs, bspec=sp.bwdq)
        out_k, lse, frame, ops = iq.flash_attn_fp4all_fwd(q, k, v, softmax_scale=sm, causal=causal, ops=ops, vtok=True,
                                                          p_mix=p_mix, masks=masks if docs is not None else None,
                                                          **_fwd_dbg(), **fkw)
        names = ("q4", "sfq", "k4", "sfk", "v4", "sfv", "vtok_bias", "vtok_u", "vtok_s", "v4b", "sfvb") + tuple(
            n for n in ops if n.startswith("kdiag_") or n in ("vmax_q", "rmax_q", "query_row_amax",
                                                             "backward_amax_slots")) + (
            # (cov5 packed rows: the dense SF words; the backward builds its atoms from them)
            ("sfq_rows", "sfk_rows", "sfv_rows") if "sfv_rows" in ops else ()) + (
            # (cov6 packed, key channel 'all' / 'open': the diagonal payload's per-tile SF words, the dP operand's)
            ("sfvb_tiles",) if "sfvb_tiles" in ops and iq._diag_k_knob(qs) in ("all", "0") else ())
        st.save_for_backward(q, k, v, out_k, lse, frame, *(ops[n] for n in names))
        st.op_names = names
        st.vtok_vmax_b = ops.get("vtok_vmax_b")  # a float (torch builder) or None (fused: computed if ever read)
        st.softmax_scale, st.causal, st.mode = sm, causal, mode
        return out_k, lse
    if st.vtok:
        # causal-safe V^: no V group scale spans sequence positions (v_hadamard: Hadamard on V first)
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        assert mode != "eden_rht", "vtok: stabiliser sr or rtn (eden_rht rotates Q / K / V / dO jointly)"
        sm = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        sv = iq._signs(q.device) if qs.vtok_hadv else None
        # qk_hadamard on this route: random Hadamard on Q / K, fused into the quantiser's reads (Quartet II: a separate
        # transform kernel costs a full BF16 read + write). R = H diag(signs) / sqrt(d) is orthogonal, so
        # S = (Q R)(K R)^T = Q K^T is unchanged and O needs no un-rotation; dQ / dK come back rotated and are
        # un-rotated in the backward.
        sqk = iq._signs(q.device) if qs.vtok_hadqk else None
        ops = iq.fp4all_quantize_qkv_vtok(q, k, v, sm, signs=sqk, signs_v=sv)
        out_k, lse, frame, ops = iq.flash_attn_fp4all_fwd(q, k, v, softmax_scale=sm, causal=causal, ops=ops, vtok=True,
                                                          p_mix=p_mix, **_fwd_dbg(), **fkw)
        out = fwht_rows(out_k, sv, inverse=True) if sv is not None else out_k
        names = ("q4", "sfq", "k4", "sfk", "v4", "sfv", "vtok_bias", "vtok_u", "vtok_s", "vtok_vmax")
        st.save_for_backward(q, k, v, out_k, lse, frame, *(ops[n] for n in names))
        st.op_names, st.had_v, st.had_qk = names, sv is not None, sqk is not None
        st.softmax_scale, st.causal, st.mode = sm, causal, mode
        return out, lse
    signs = iq._signs(q.device) if mode == "eden_rht" else None
    if docs is not None and signs is not None:
        raise NotImplementedError("the all-4-bit route on documents: all_fp4_stabiliser 'sr' or 'rtn' (eden_rht rotates "
                                  "Q / K / V / dO with host-side transforms, not wired for documents)")
    from flash_attn.cute.lp_docmaps import packed_tiles
    pk_v = packed_tiles(v) if docs is not None else None   # (cov6: the packed rows of compact maps, dense)
    ops = iq.fp4all_quantize_qkv(q, k, v, signs=signs, backward_operands=True, docs=docs, packed=pk_v)
    out_k, lse, frame, ops = iq.flash_attn_fp4all_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, ops=ops,
                                                      vtok=False, p_mix=p_mix, masks=masks if docs is not None else None,
                                                      **_fwd_dbg(), **fkw)
    if signs is not None:
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        out = fwht_rows(out_k, signs, inverse=True)
    else:
        out = out_k
    ops = {n: ops[n] for n in ("q4", "sfq", "k4", "sfk", "v4", "sfv", "kt4", "sfkt", "sfkt1", "query_row_amax",
                               "v_norm2_max", "sfq_rows", "sfk_rows", "sfv_rows", "sfkt_groups")
           if n in ops}   # the dense backward's, from the forward's reads (cov6 packed: the dense SF words / K^T groups)
    st.save_for_backward(q, k, v, out_k, lse, frame, *ops.values())
    st.op_names = tuple(ops.keys())
    st.softmax_scale, st.causal, st.mode = softmax_scale, causal, mode
    return out, lse


def _all4_seed(plan: AttnPlan, mode: str, device):
    """The SR dither seed of one all-4-bit backward call: the device's SR counter (interface_bwd_fp4q.sr_counter), read
    by the dO quantiser and the kernel (which derive the seed from it) and advanced by the backward's post launch -- on
    the GPU, so every replay of a captured step draws a new dither. 0 where nothing is stochastically rounded (stabiliser
    'rtn' without the SR dK / dV operands)."""
    from flash_attn.cute import interface_bwd_fp4q as iq
    bq = plan.specs.bwdq
    if mode == "rtn" and not bq.dk_sr and not bq.dv_sr:
        return 0
    return iq.sr_counter(device)


def _all4_backward(plan: AttnPlan, st, dout):
    from flash_attn.cute import interface_bwd_fp4q as iq
    sp = plan.specs
    bkw = dict(spec=sp.bwdq, qspec=sp.quant, fspec=sp.fwd)
    q, k, v, out_k, lse, frame, *opv = st.saved_tensors
    ops = dict(zip(st.op_names, opv))
    mode = st.mode
    if st.vtok_diag:
        ops["vtok_diag"] = True
        if st.vtok_vmax_b is not None:
            ops["vtok_vmax_b"] = st.vtok_vmax_b
        seed = _all4_seed(plan, mode, q.device)
        dq, dk, dv = iq._flash_attn_bwd_fp4q(q, k, v, out_k, dout.contiguous(), lse, frame, ops,
                                             softmax_scale=st.softmax_scale, causal=st.causal,
                                             do_rnd={"sr": "sr", "rtn": "rtn"}[mode], ds_sr=mode != "rtn",
                                             seed=seed, dkv="e4m3", p_mix=st.p_mix, docs=getattr(st, "docs", None),
                                             **_bwd_dbg(), **bkw)
        # no dV fold: the hybrid kernel's dV multiplies the unbiased m, so it is dL/dV directly
        return dq, dk, dv
    if st.vtok:
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        sv = iq._signs(q.device) if st.had_v else None
        sqk = iq._signs(q.device) if getattr(st, "had_qk", False) else None
        dout = dout.contiguous()
        if sv is not None:
            dout = fwht_rows(dout, sv)
        seed = _all4_seed(plan, mode, q.device)
        dq, dk, dv = iq._flash_attn_bwd_fp4q(q, k, v, out_k, dout, lse, frame, ops, softmax_scale=st.softmax_scale,
                                             causal=st.causal, do_rnd={"sr": "sr", "rtn": "rtn"}[mode],
                                             ds_sr=mode != "rtn", seed=seed, signs=sqk, p_mix=st.p_mix, **_bwd_dbg(),
                                             **bkw)
        if sqk is not None:
            # the kernel's dQ / dK are dQ_true R and dK_true R (R orthogonal)
            dq = fwht_rows(dq.contiguous(), sqk, inverse=True)
            dk = fwht_rows(dk.contiguous(), sqk, inverse=True)
        # no per-key dV fold here or anywhere on the host: both backward routes' dV A operand is the unbiased m, so the
        # kernel returns dL/dV -- see `query-row-key-local-v-scale-dv-halved` in fp4q_prepare.
        if sv is not None:
            dv = fwht_rows(dv.contiguous(), sv, inverse=True)
        return dq, dk, dv
    signs = iq._signs(q.device) if mode == "eden_rht" else None
    dout = dout.contiguous()
    if signs is not None:
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        dout = fwht_rows(dout, signs)
    do_rnd = {"sr": "sr", "eden_rht": "eden", "rtn": "rtn"}[mode]
    seed = _all4_seed(plan, mode, q.device)
    dq, dk, dv = iq._flash_attn_bwd_fp4q(q, k, v, out_k, dout, lse, frame, ops, softmax_scale=st.softmax_scale,
                                         causal=st.causal, do_rnd=do_rnd, ds_sr=mode != "rtn", seed=seed, signs=signs,
                                         p_mix=st.p_mix, docs=getattr(st, "docs", None), **_bwd_dbg(), **bkw)
    if signs is not None:
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        dq, dk, dv = (fwht_rows(g.contiguous(), signs, inverse=True) for g in (dq, dk, dv))
    return dq, dk, dv


# =====================================================================================================================
# dispatch
# =====================================================================================================================

def forward(plan: AttnPlan, st, q, k, v, prequant=None, masks=None):
    """The plan's route forward. Returns (out, lse). masks: the call's lp_masks.LPMasks (the E4M3 P.V routes only -- the
    plan refuses masking features elsewhere). A fused doc_maps call (masks.docs.doc_maps_fused): q / k / v are the
    caller's packed rows (lp_docmaps.fused_forward)."""
    if masks is not None and getattr(masks.docs, "doc_maps_fused", False):
        from flash_attn.cute.lp_docmaps import fused_forward
        return fused_forward(_forward_route, plan, st, q, k, v, masks, prequant)
    return _forward_route(plan, st, q, k, v, prequant, masks)


def _forward_route(plan: AttnPlan, st, q, k, v, prequant=None, masks=None):
    r = plan.route
    if r == "fp8":
        out, lse = _fp8_forward(plan, st, q, k, v, prequant, masks)
    elif r == "fp4.e4m3_pv":
        out, lse = _e4pv_forward(plan, st, q, k, v, masks)
    elif r == "fp4.all_fp4":
        assert masks is None or _docs_only(masks), "the plan refuses masking features on the all-4-bit route but documents"
        out, lse = _all4_forward(plan, st, q, k, v, masks)
    elif r == "fp4.fp4_pv":
        # option A is inference only (plan rule 36); refused here too for a plan built elsewhere
        assert masks is None or _docs_only(masks), "the plan refuses masking features on option A but documents"
        if plan.shape.requires_grad:
            from flash_attn.cute.lowprec.plan import OPTION_A_INFERENCE_ONLY
            raise ValueError(OPTION_A_INFERENCE_ONLY)
        out, lse = _option_a_forward(plan, st, q, k, v, masks)
    else:
        raise ValueError(f"unknown route {r!r}")
    if masks is None or masks.docs is None:  # documents: the repacked b = 1 batch's LSE (not returned)
        b, h, s_q = plan.shape.b, plan.shape.h, plan.shape.s_q
        assert lse.dtype == torch.float32 and tuple(lse.shape) == (b, h, s_q), (r, lse.dtype, tuple(lse.shape))
    return out, lse


def backward(plan: AttnPlan, st, dout):
    """The plan's route backward from the forward's saved state. Returns (dq, dk, dv)."""
    if getattr(st, "doc_maps_fused", None) is not None:   # (lp_docmaps.fused_backward: packed dout and gradients)
        from flash_attn.cute.lp_docmaps import fused_backward
        return fused_backward(_backward_route, plan, st, dout)
    return _backward_route(plan, st, dout)


def _backward_route(plan: AttnPlan, st, dout):
    r = plan.route
    if r == "fp8":
        return _fp8_backward(plan, st, dout)
    if r == "fp4.e4m3_pv":
        return _e4pv_backward(plan, st, dout)
    if r == "fp4.all_fp4":
        return _all4_backward(plan, st, dout)
    # option A's research backward (the P-hat rebuild in the dk8 kernels) is not shipped: option A is inference-only and
    # the plan never builds a grad-carrying option A call
    from flash_attn.cute.interface_bwd_fp4 import OPTION_A_BWD_ARCHIVED
    raise NotImplementedError(OPTION_A_BWD_ARCHIVED)


# ---------------------------------------------------------------------------------------------------------------
# masks / varlen: window / softcap / score_mod / mask_mod (one explicit LPMasks value) and the varlen / seqused_k
# documents (the library's aligned repacking around the autograd node)
# ---------------------------------------------------------------------------------------------------------------

def masks_of(plan: AttnPlan, mask_kw: dict):
    """The call's lp_masks.LPMasks, or None for the plain causal / dense call (the kernels exactly as before)."""
    from flash_attn.cute.lp_masks import LPMasks
    return lp_masks_or_none(LPMasks.build(causal=plan.fwd.causal, **mask_kw), plan.fwd.causal)


def run_aligned(plan: AttnPlan, q, k, v, apply_fn, *, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, mask_kw):
    """The aligned varlen fast path (FA3's convention): the caller's packed (total, h, d) tensors already hold every
    document at a start that is a multiple of 128 (the 2-CTA backward too). Nothing is copied: the document
    tables come straight from cu_seqlens / seqused (lp_varlen.LPDocs.aligned_from: one prep launch, the starts and
    lengths asserted on the GPU), and the plan's autograd node runs on the (1, total, h, d) VIEWS of q, k, v. The
    padding rows (after each document's true length) are masked by the quantisers' row_end, so their contents never
    reach a code, a scale or a statistic. Returns out (total_q, h, d) (and lse (h, total_q) with return_lse)."""
    from flash_attn.cute.lp_varlen import LPDocs
    from flash_attn.cute.lp_masks import LPMasks
    s = plan.shape
    # 128 on every backward: the 2-CTA key pair takes documents packed at 128 (flash_bwd_sm100_2cta8 doc_t128)
    align = 128
    docs = LPDocs.aligned_from(cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, q.shape[0], k.shape[0], align=align,
                               max_seqlen_q=s.s_q, max_seqlen_k=s.s_k, require_keys=_require_keys(plan))
    masks = LPMasks.build(causal=plan.fwd.causal, docs=docs, **mask_kw)
    # unsqueeze / squeeze: views both ways, and their backward is a view too (indexing [0] would add a zero-fill and a
    # copy of dO in the backward)
    op = apply_fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), masks)
    if isinstance(op, tuple):  # return_lse: the forward's LSE of the (1, total) batch is (1, h, total_q)
        out, lse = op
        return out.squeeze(0), lse.squeeze(0)
    return op.squeeze(0)


def run_doc_maps(plan: AttnPlan, q, k, v, apply_fn, *, doc_maps, mask_kw, prequant_fa3: bool = False):
    """Packed (T, h, d) documents with precomputed maps (lp_docmaps.DocMaps): the aligned path on the maps' aligned
    buffer, its tables from the maps (no prep launch, no device-side check but the 4-bit P.V routes' key check). On
    the token-block E4M3 P.V routes (lp_docmaps.fused_ok) the node takes the caller's packed rows and the kernels read
    and write them through the maps (lp_docmaps.fused_forward / fused_backward); elsewhere the rows are gathered into
    the aligned buffer and back (lp_docmaps.pack_rows / unpack_rows). Plan: the aligned buffer's (lowprec.api).
    Returns out (T, h, d) (and lse (h, T) with return_lse)."""
    from flash_attn.cute import lp_docmaps as DM
    from flash_attn.cute.lp_masks import LPMasks
    doc_maps.check_keys(_require_keys(plan), plan.entry)
    if prequant_fa3:
        # (cov2) FA3-style E4M3 codes in the caller's packed rows: the packed path, compact or full maps alike (no
        # aligned buffer; lowprec.prequant_fa3.flash_attn_fp8_prequant_doc_maps)
        masks = LPMasks.build(causal=plan.fwd.causal, docs=doc_maps.lpdocs(fused=True, packed=True), **mask_kw)
        op = apply_fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), masks)
        if isinstance(op, tuple):
            return op[0].squeeze(0), op[1][0]
        return op.squeeze(0)
    fused = DM.FUSED_GATHER and DM.fused_ok(plan)
    # (compact maps: plans the packed rows serve with no fused aligned form -- lp_docmaps.packed_only_ok)
    pk_only = not fused and DM.FUSED_GATHER and DM.packed_only_ok(plan, doc_maps)
    masks = LPMasks.build(causal=plan.fwd.causal, docs=doc_maps.lpdocs(fused=fused or pk_only), **mask_kw)
    packed = (fused or pk_only) and DM.PACKED_ROWS and DM.packed_ok(plan, masks, bool(plan.shape.requires_grad))
    fused = fused or packed
    if doc_maps.compact and not packed:
        # (sfd) compact maps serve the packed path only: any other plan takes the full maps of the same documents
        doc_maps = doc_maps.aligned_maps()
        masks = LPMasks.build(causal=plan.fwd.causal, docs=doc_maps.lpdocs(fused=fused), **mask_kw)
    if packed:   # (lp_docmaps packed rows: no aligned buffer; the LSE is (1, h, T))
        masks = LPMasks.build(causal=plan.fwd.causal, docs=doc_maps.lpdocs(fused=True, packed=True), **mask_kw)
    if fused:
        op = apply_fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), masks)
        if isinstance(op, tuple):   # return_lse: (1, h, S) -> (h, T)
            return op[0].squeeze(0), (op[1][0] if packed else op[1][0].index_select(1, doc_maps.dst_q))
        return op.squeeze(0)
    op = apply_fn(DM.pack_rows(doc_maps, q, "q").unsqueeze(0), DM.pack_rows(doc_maps, k, "k").unsqueeze(0),
                  DM.pack_rows(doc_maps, v, "k").unsqueeze(0), masks)
    if isinstance(op, tuple):   # return_lse: (1, h, S) -> (h, T)
        return DM.unpack_rows(doc_maps, op[0].squeeze(0), "q"), op[1][0].index_select(1, doc_maps.dst_q)
    return DM.unpack_rows(doc_maps, op.squeeze(0), "q")


def _require_keys(plan: AttnPlan) -> int:
    """The 4-bit P.V routes' document check (lp_varlen.LPDocs.aligned_from require_keys): their forward needs a key tile
    under every query tile -- causal: seqused_q <= seqused_k per document (as s_q <= s_k batched), dense: a key."""
    if plan.route not in ("fp4.all_fp4", "fp4.fp4_pv"):
        return 0
    return 2 if plan.fwd.causal else 1


def repacked_as_aligned(cu_seqlens_q, cu_seqlens_k, total_q, total_k, max_seqlen_q, max_seqlen_k, seqused_k=None,
                        require_keys: int = 0):
    """The repack fallback's documents in the aligned form: (rp, docs) with rp the repacking (lp_varlen.LPDocs.build:
    the row maps, every start rounded up to 128, zero rows between documents) and docs the aligned path's tables of that
    repacked buffer (lp_varlen.LPDocs.aligned_from on the padded starts / true lengths; the mods' offsets stay the
    caller's, as the repack path's). No host read."""
    from flash_attn.cute.lp_varlen import LPDocs
    rp = LPDocs.build(cu_seqlens_q, cu_seqlens_k, total_q, total_k, align=128, max_seqlen_q=max_seqlen_q,
                      max_seqlen_k=max_seqlen_k, seqused_k=seqused_k)
    docs = LPDocs.aligned_from(rp.cu_pad_q, rp.cu_pad_k, rp.len_q, rp.len_k, rp.s_q, rp.s_k, align=128,
                               max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, require_keys=require_keys)
    docs.doc_tab[4].copy_(rp.doc_tab[4])   # the caller's offsets: what a score_mod / mask_mod sees (LPDocs.build)
    docs.doc_tab[5].copy_(rp.doc_tab[5])
    return rp, docs


def run_repacked_aligned(plan: AttnPlan, q, k, v, apply_fn, *, cu_seqlens_q, cu_seqlens_k, seqused_k, mask_kw):
    """Repacked varlen (cu_seqlens without seqused_*) or batched seqused_k on the ALIGNED path: the documents are copied
    into a zero-padded buffer at 128-aligned starts (lp_varlen.pack_rows), the plan's autograd node runs on that buffer
    with the aligned tables (repacked_as_aligned), and the output (and LSE) rows are gathered back. Used where a feature
    needs the aligned path's per-(document, kv head) tables: per-tensor operands. Same result as calling the aligned
    path on the zero-padded buffer by hand."""
    from flash_attn.cute.lp_masks import LPMasks
    from flash_attn.cute.lp_varlen import pack_rows, unpack_rows
    s = plan.shape
    batched = not s.varlen   # batched seqused_k: the batch rows as documents
    if batched:
        b, sq, h, d = q.shape
        sk = k.shape[1]
        ar = torch.arange(b + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_q, cu_seqlens_k = ar * sq, ar * sk
        q, k, v = q.reshape(b * sq, h, d), k.reshape(b * sk, k.shape[2], d), v.reshape(b * sk, v.shape[2], v.shape[3])
    rp, docs = repacked_as_aligned(cu_seqlens_q, cu_seqlens_k, q.shape[0], k.shape[0], s.s_q, s.s_k, seqused_k,
                                   require_keys=_require_keys(plan))
    masks = LPMasks.build(causal=plan.fwd.causal, docs=docs, **mask_kw)
    if rp.keep_k is not None:   # seqused_k: keys past the used length are zero (and get zero dK / dV)
        # (torch.where, not a product: the aligned path leaves those rows' gradients unspecified, and 0 * NaN is NaN)
        km = rp.keep_k.view(-1, 1, 1)
        k = torch.where(km, k, torch.zeros((), dtype=k.dtype, device=k.device))
        v = torch.where(km, v, torch.zeros((), dtype=v.dtype, device=v.device))
    op = apply_fn(pack_rows(rp, q, "q"), pack_rows(rp, k, "k"), pack_rows(rp, v, "k"), masks)
    lse = None
    if isinstance(op, tuple):   # (out, lse (1, h, s_pad)) -> the caller's rows
        op, lse = op
        lse = lse[0].index_select(1, rp.dst_q)
    out = unpack_rows(rp, op, "q")
    if batched:
        out = out.view(b, sq, h, out.shape[-1])
        if lse is not None:
            lse = lse.view(h, b, sq).permute(1, 0, 2).contiguous()
    return out if lse is None else (out, lse)


def run_documents(plan: AttnPlan, q, k, v, apply_fn, *, cu_seqlens_q, cu_seqlens_k, seqused_k, mask_kw,
                  seqused_q=None):
    """varlen (packed (total, h, d) with cu_seqlens), seqused_k or ragged lengths (s % 128 != 0; the batch
    rows as documents): the documents are repacked at aligned starts on the GPU (fp4_bwd_autograd._varlen_apply, no host
    sync), apply_fn runs the plan's autograd node on the repacked b = 1 batch with the document tables in its masks, and
    the output is unpacked (autograd of the row gathers). The repacking reads the plan's 2-CTA decision (its specs).
    With plan.fwd.return_lse (ragged batch rows only; the plan refuses it under varlen / seqused_k), returns
    (out, lse (b, h, s)), the LSE gathered from the repacked rows."""
    s = plan.shape
    sp = plan.specs
    skw = dict(quant=sp.quant, host=sp.host)
    if s.aligned_varlen:
        return run_aligned(plan, q, k, v, apply_fn, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                           seqused_q=seqused_q, seqused_k=seqused_k, mask_kw=mask_kw)
    if (plan.fwd.operand_scaling == "per_tensor" or plan.route in ("fp4.all_fp4", "fp4.fp4_pv")) \
            and (s.varlen or s.seqused_k):
        # (the 4-bit P.V routes: their per-document statistics and tables are the aligned path's too)
        # per-(document, kv head) per-tensor operands on repacked varlen / batched seqused_k: the aligned path on the
        # repacked buffer (the per-document scale tables are the aligned path's); the token-block repack is unchanged
        return run_repacked_aligned(plan, q, k, v, apply_fn, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                    seqused_k=seqused_k, mask_kw=mask_kw)
    if s.varlen:
        return ag._varlen_apply(apply_fn, q, k, v, cu_seqlens_q, cu_seqlens_k, s.s_q, s.s_k, plan.fwd.causal, 128,
                                mask_kw, plan.bwd.deterministic, seqused_k, **skw)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    ar = torch.arange(b + 1, dtype=torch.int32, device=q.device)
    o = ag._varlen_apply(apply_fn, q.reshape(b * sq, h, d), k.reshape(b * sk, k.shape[2], d),
                         v.reshape(b * sk, v.shape[2], v.shape[3]), ar * sq, ar * sk, sq, sk, plan.fwd.causal, 128,
                         mask_kw, plan.bwd.deterministic, seqused_k, **skw)
    if isinstance(o, tuple):  # return_lse: (h, b * s) packed rows -> (b, h, s)
        o, lse = o
        return o.view(b, sq, h, v.shape[3]), lse.view(h, b, sq).permute(1, 0, 2).contiguous()
    return o.view(b, sq, h, v.shape[3])

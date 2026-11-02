# The host glue of the low-precision routes (lowprec/_routes.py: the 4-bit E4M3-P.V route, option A's inference
# forward, the 8-bit route): device scales, the token-block factor / token tables, the backward prologues fused into the
# dO quantiser launch, and the varlen document repacking. Every helper reads the host specs it is handed
# (lowprec.specs QuantSpec / HostSpec, from the call's plan), never a kernel spec and never any other configuration.
# (The public entry points and their single autograd node are lowprec.api / lowprec._routes; this module holds the glue
# they share.)
from typing import Optional

import torch

from flash_attn.cute.lp_masks import LPMasks, fwd_mask_kwargs
from flash_attn.cute.quant.fp4 import quantize_fp4_rows, quantize_fp4_blocks_2d


def v_descale_dev(s, b: int, h: int) -> torch.Tensor:
    """(b, h) float32 filled with a float32 device scale (a float64 one is rounded to float32, as torch.full rounds)."""
    return torch.empty(b, h, dtype=torch.float32, device=s.device).copy_(s.reshape(1, 1).expand(b, h))


def score_scale_dev(softmax_scale: float, sq: torch.Tensor, sk: torch.Tensor) -> torch.Tensor:
    """(s_q float32(softmax_scale)) s_k in float32 (torch's correctly rounded multiplies; the order of quant.fp8's
    per-document derived score scale and of prequant_fa3's tables), as a (1,) device tensor for
    _flash_attn_fwd(softmax_scale_t=...)."""
    return (sq.reshape(1).float() * softmax_scale) * sk.reshape(1).float()








def _fwd_masks(masks, causal, q, k, host=None) -> dict:
    """The forward driver's mask arguments (masks / varlen); {causal} alone for the plain call. Aligned-packed varlen:
    the forward's own varlen path over the repacked (1, s, h, d) tensors, documents at the aligned starts with their
    true lengths (seqused), no host read of the offsets."""
    if masks is None:
        return dict(causal=causal)
    kw = fwd_mask_kwargs(masks, q.shape[0], q.shape[2], q.shape[1], k.shape[1], q.device)
    if masks.docs is not None:
        d = masks.docs
        kw.update(cu_seqlens_q=d.cu_pad_q, cu_seqlens_k=d.cu_pad_k, seqused_q=d.len_q, seqused_k=d.len_k,
                  cu_seqlens_aligned=True)
        if getattr(d, "aligned", False):
            kw.update(_aligned_fwd_sched(d, masks, q, k, host))
        if getattr(d, "doc_maps", None) is not None:   # lp_docmaps packed O: the caller's packed row starts
            from flash_attn.cute.lp_docmaps import packed_o_rows
            po = packed_o_rows(d)
            if po is not None:
                kw["o_cu_seqlens"] = po[0]
                from flash_attn.cute.lp_docmaps import packed_kv_rows
                pk = packed_kv_rows(d)
                if pk is not None:   # packed rows: the codes at the packed starts too
                    kw["kv_cu_seqlens_packed"], kw["vt_cu_seqlens_packed"] = pk
    return kw


def _aligned_fwd_sched(d, masks, q, k, host=None) -> dict:
    """Aligned documents (step 3): the forward's scheduling arguments -- the persistent longest-first work list, the
    batched TMA O epilogue and the causal stage skip (HostSpec.varlen_fwd_list / varlen_fwd_tma_o /
    varlen_fwd_stage_skip / varlen_l2_sections). Scheduling only: O and LSE bitwise."""
    from flash_attn.cute import lp_varlen as LV
    from flash_attn.cute.lowprec.specs import HostSpec
    hs = host if host is not None else HostSpec()
    # the TMA O epilogue on causal calls only (it also unlocks the stage skip there; paired A/B, step 3: causal +1..+16 %
    # with the skip, dense -0.1..-7.5 % vs the correction-warp copies)
    kw = dict(varlen_tma_o=hs.varlen_fwd_tma_o and bool(masks.causal), varlen_stage_skip=hs.varlen_fwd_stage_skip)
    if hs.fwd_two_cta:
        # the forward on CTA pairs (forward_two_cta='on'): one cluster per work tile with the batched TMA O (the
        # driver sets it); the persistent work list is the 1-CTA kernel's
        return kw
    if hs.varlen_fwd_list and masks.mask_mod is None and not masks.local:
        h, dqk = q.shape[2], q.shape[3]
        s_k = d.max_seqlen_k if d.max_seqlen_k is not None else d.s_k
        per_tok = dqk + k.shape[3]   # K + V, one byte per element (FA4's clamp for sub-byte)
        sec = LV.l2_heads(h, int(s_k) * per_tok, hs.varlen_l2_sections)
        # head sections only: document sections (span) measured -7..-9 % on the forward (eq16x1024 causal), the
        # 148-CTA wave then straddles the section boundary; the backward keeps them
        sched = getattr(hs, "varlen_fwd_sched", "static")
        jit = bool(getattr(hs, "varlen_fwd_sched_jit", False))
        if sched == "auto":
            # owner decision: CLC + just-in-time claim on causal calls (-1.7 .. -5.7 % on 13 of 16 causal cells,
            # -0.6 .. +0.5 % on the other 3; O / LSE bitwise), the static deal on dense ones (CLC there +0.6 .. +2.3 % on
            # uniform lengths)
            sched, jit = ("clc", True) if (masks.causal or _length_skewed(d)) else ("static", jit)
        if sched == "clc":
            # (sfd compact maps of a large capacity, lp_docmaps.large_capacity: the CLC grid is the list's static bound,
            # ~ the tile bound x heads, every entry launched or cancelled -- the claims are atomic there (grid = the SMs),
            # just in time as CLC's. Otherwise CLC as the default: atomic measured -4 % fwd fp8 d64, +4 % fp4 d64)
            from flash_attn.cute.lp_docmaps import large_capacity
            if large_capacity(getattr(d, "doc_maps", None)):
                sched = "atomic"
        if sched == "static":
            kw["varlen_work"] = LV.varlen_worklist(d, "fwd", bool(masks.causal), sec)
        else:
            # dynamic claiming (opt-in): the list carries the claim counters (zeroed by the list build,
            # reset by the last CTA of every launch)
            kw["varlen_work"] = LV.varlen_worklist(d, "fwd", bool(masks.causal), sec, ctr=True)
            kw["varlen_sched"] = sched
            kw["varlen_sched_stages"] = int(getattr(hs, "varlen_fwd_sched_stages", 2))
            kw["varlen_sched_jit"] = jit
            kw["varlen_sched_early"] = bool(getattr(hs, "varlen_fwd_sched_early", False))
    return kw


def _length_skewed(d) -> bool:
    """Non-causal 'auto' scheduler rule: dynamic claiming (CLC, just-in-time claim) when the document lengths are skewed,
    the static deal when they are near-equal. Host facts only, no sync: DocMaps' CPU-build bit (longest / mean length >=
    lp_docmaps.SKEW_CLC_MIN), or for aligned varlen max_seqlen_k * documents / padded total >= SKEW_CLC_MIN; unknown
    (GPU-built maps, no max_seqlen) = static."""
    from flash_attn.cute import lp_docmaps as DM
    m = getattr(d, "doc_maps", None)
    if m is not None:
        return bool(m.sched_skewed)
    mx, tot = getattr(d, "max_seqlen_k", None), getattr(d, "s_k", None)
    if not mx or not tot:
        return False
    return float(mx) * int(d.b) / float(tot) >= DM.SKEW_CLC_MIN


def _docs_of(ctx):
    m = getattr(ctx, "masks", None)
    return None if m is None else m.docs


def _unbatch(docs, *ts):
    """varlen: the forward driver takes packed (total, h, d) tensors -- the repacked b = 1 batch without its batch mode."""
    return ts if docs is None else tuple(t[0] for t in ts)


def _lse_for_bwd(ctx, lse):
    """varlen: +inf on the padding rows (P = 0 there exactly), as upstream's preprocess does past seqlen_q."""
    d = _docs_of(ctx)
    if d is not None and getattr(d, "doc_maps_packed", False):
        return lse   # (lp_docmaps packed rows: no padding rows; the backward masks the rows past a document itself)
    return lse if d is None else d.lse_pad_rows_inf(lse)






# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def tb_fold_fwd(cols: int, qspec) -> bool:
    """QuantSpec.token_block_forward (KernelTuning.token_block_forward; token-block operands, cols = 128; default
    lowprec.specs.TB_FWD_DEFAULT; the fold is not the default because it is untrained and leaks in-block, see docs/lowprec.md):
    "mx" (default) -- the MXFP8 (block-scaled) QK / PV forward; "fold" -- plain E4M3 / NVFP4 GEMMs with the per-32-token
    factors folded into the scores (flash_fwd_sm100_fp4 `tb_fwd`). Both read exactly the codes (and the factors) the
    blk32x128 backward reads: one quantize_fp8_blocks per operand, shared by both passes."""
    v = qspec.token_block_forward
    if v not in ("fold", "mx"):
        raise ValueError(f"token_block_forward must be 'fold' or 'mx', got {v!r}")
    return cols == 128 and v == "fold"


# token-local operands (operand_scaling="token_local"): the (6, b, h, s) fp32 token table, rows f_Q, f_K, f_V, f_dO
# (f = 2^(e_t - 127), each token's own exponent of its token-local E4M3 codes), log2 f_V, 1 / f_V; K / V rows cover the
# h_kv heads. Written in place by the quantisers (no fill); read by both passes (SCALING-CONSISTENCY RULE).
TOKEN_TABLE_ROWS = 6


def token_table(q, k):
    """(6, b, h, max(s_q, s_k)): rows f_Q / f_dO cover the first s_q columns, f_K / f_V (and their log2 / inverse) the
    first s_k (token-block fallback shapes: s_q != s_k); slice a row for an operand with tok_row. (cov3, compact
    lp_docmaps packed rows: per PACKED row, plus a 128-column tail the quantisers fill -- the packed kernels read a
    document's last tile past T.)"""
    from flash_attn.cute.lp_docmaps import compact_operand
    b_, s_, h_, _ = q.shape
    tail = 128 if compact_operand(q) else 0
    return torch.empty(TOKEN_TABLE_ROWS, b_, h_, max(s_, k.shape[1]) + tail, dtype=torch.float32, device=q.device)


def tok_row(tok, row: int, x):
    """Row `row` of the token table for operand x (b, s, h, d): its heads, its s tokens (compact packed rows: and the
    128-column tail)."""
    from flash_attn.cute.lp_docmaps import compact_operand
    if compact_operand(x):
        return tok[row, :, :x.shape[2], :x.shape[1] + 128]
    return tok[row, :, :x.shape[2], :x.shape[1]]


_TL_FAC = {}


def tl_fold_fac(q, fac):
    """The fold forward's factor table argument: fac, or (cov3, compact packed rows, token-local operands: the factors
    are the token table's, the block table is never read) a cached (4, 1, h, 1) placeholder that selects the fold."""
    if fac is not None:
        return fac
    key = (q.device, q.shape[2])
    if key not in _TL_FAC:
        _TL_FAC[key] = torch.ones(4, 1, q.shape[2], 1, dtype=torch.float32, device=q.device)
    return _TL_FAC[key]


def padded_head_dim(d: int) -> int:
    """token-block fallback shapes: the width the token-block kernels run a head dim at -- d itself for 64 / 128, else the next of 64 / 128
    (d a multiple of 16 below 128). The quantisers read the d-wide rows once and stage the missing columns as zeros
    (quantize_fp8_blocks pad_head_dim_to): zero columns add nothing to S / dP and move no block exponent, so the padded
    operands are exactly the d-wide token blocks; O / dQ / dK / dV are the first d columns of the padded results."""
    return d if d in (64, 128) else (64 if d < 64 else 128)


# the token-block factor table (4, b, h, max(s_q, s_k) / 32), rows Q / K / V / dO (2^(e - 127) per 32-token block)
FAC_Q, FAC_K, FAC_V, FAC_DO = 0, 1, 2, 3


def token_block_factor_table(q, k):
    """The (4, b, h, max(s_q, s_k) / 32) fp32 token-block factors: rows Q / dO cover the first s_q / 32 columns, rows
    K / V the first s_k / 32 and the h_kv heads (token-block fallback shapes: the kernels index rows Q / dO by query block and K / V by
    key block, so s_q != s_k takes token blocks too). Each row is written whole by its quantiser launch (fac_row); with
    s_q == s_k every slice is the whole row, the layout of the s_q-wide table before."""
    from flash_attn.cute.lp_docmaps import aligned_rows, compact_operand   # (packed rows: the tables' aligned S)
    if compact_operand(q):
        return None   # (sfd compact maps: the factors come from the dense exponent rows)
    b_, _, h_, _ = q.shape
    return torch.empty(4, b_, h_, max(aligned_rows(q), aligned_rows(k)) // 32, dtype=torch.float32, device=q.device)


def fac_row(fac, row: int, x):
    """Row `row` of the token-block factor table for operand x (b, s, h, d): its heads, its s / 32 token blocks; None
    without a table."""
    if fac is None:
        return None
    from flash_attn.cute.lp_docmaps import aligned_rows   # (packed rows: the tables' aligned S)
    return fac[row, :, :x.shape[2], :aligned_rows(x) // 32]


def q_blk2d(qspec) -> bool:
    """QuantSpec.q_block_2d (options.q_block_2d; default False): Q's NVFP4 codes with one E4M3 scale per 16 queries x 16 head-dim
    elements (quant.fp4.quantize_fp4_blocks_2d, RTN) instead of 1 x 16 along d. The S GEMM reads them unchanged (the SF is replicated along
    the 16 query rows); a transposed copy of the SAME codes can serve a 4-bit dK GEMM (fp4q dkv=fp4). Causal caveat: a
    16-query block shares its scale, so a future query's magnitude moves a past query's codes (E2M1 grid route)."""
    return qspec.q_block_2d


def _nv4_rows(x, row_end=None):
    """(codes, SF view) of x's NVFP4 1 x 16 row copy (quant.fp4; the S operand of the token-block routes). row_end: the
    aligned varlen padding mask (quant.fp4.quantize_fp4_rows)."""
    r = quantize_fp4_rows(x) if row_end is None else quantize_fp4_rows(x, row_end=row_end)
    return r["codes"], r["scale_factors"]


def _q4_blk2d(q, dense_into: Optional[dict] = None):
    """(codes, SF view) of Q's NVFP4 16 x 16 block copy. dense_into (cov3, a packed lp_docmaps call on compact maps):
    the quantiser reads / writes the caller's packed rows (quantize_fp4_blocks_2d packed_rows=) and its dense per-row SF
    words replace dense_into["sf4_rows"] (the sf_dense kernels' S-operand words of Q)."""
    if dense_into is not None:
        from flash_attn.cute.lp_docmaps import packed_rows_of
        e = packed_rows_of(q)
        assert e is not None and e[0] is None, "q_block_2d packed rows: compact maps only"
        r = quantize_fp4_blocks_2d(q, packed_rows=(e[2], e[4]))
        dense_into["sf4_rows"] = r["sf4_rows"]
        return r["codes"], r["scale_factors"]
    r = quantize_fp4_blocks_2d(q)
    return r["codes"], r["scale_factors"]


def fused_ds_bound_ok(q, qspec) -> bool:
    """QuantSpec.fused_ds_bound (KernelTuning.fp4_fused_ds_bound; default on): the dS scale bound's inputs come out of the quantisers that read V and dO
    anyway (max ||V_k||^2 by V's, the bound itself by dO's prologue launch) instead of two separate row-norm passes and
    a torch chain. Fresh values, reduced in this step. Needs s % 128 == 0; the bound's row sums of squares are in the
    quantiser's order (the old Triton one; the torch chain's vector_norm used its own), so s_ds can move by an ulp."""
    qs = qspec
    return (qs.fused_ds_bound
            and q.shape[1] % 128 == 0 and q.shape[-1] == 128 and qs.d_from_do8)


def fresh_forward_scales(q, k, v, softmax_scale, qspec):
    """s = the (3,) float32 device tensor of sq / sk / sv = max(amax / 448, 1e-12) (quant.fp8, no host read; bitwise
    the values the host read gave)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    return quantize_fp8_per_tensor([q, k, v], output_codes=False,
                                   power_of_two_scale=qspec.per_tensor_pow2)["scales"]


def dk8_prologue_fusable(hspec) -> bool:
    """The dk8 backward (1-CTA or 2-CTA) takes D, the shifted LSE and the zeroed accumulator from the dO quantiser (the
    fused 4-bit step glue; there is no unfused fallback)."""
    return hspec.bwd_op_mode == "fp4"


def dk8_backward_prologue(dout, out, lse, bound_inputs=None, *, qspec, hspec, documents=None):
    """E4M3 dO with its scale computed on the GPU and D = sum(O dO8) from exactly those codes (softmax-consistent), and --
    where the dk8 driver takes them (dk8_prologue_fusable) -- LSE * log2 e - e4m3_p_shift_log2() and the zeroed f16 dQ
    accumulator, all in ONE launch (quant.fp8.quantize_fp8_per_tensor). bound_inputs =
    (pmax, vnorm2): also this step's dS scale bound, returned as fp8_scales["s_ds_r"]; only on the fused route.
    documents (aligned varlen, per-tensor operands): quant.fp8.quantize_fp8_per_tensor(documents=...) -- dO's scale per
    (document, kv head), D folded per document, +inf shifted LSE on padding rows; extra["doc_scales_do"] the (n_rows,
    h_kv) float32 dO table. Needs the fused route at d = 128, no bound.
    Returns (do8, sdo (a float32 device scalar), the fp8_scales entries)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    from flash_attn.cute.interface_bwd_fp4 import bwd_dq_f16, e4m3_p_shift_log2
    from flash_attn.cute.dq_postprocess_cute import dq_zero_mode
    b, s, h, d = dout.shape
    pow2 = qspec.per_tensor_pow2
    if dk8_prologue_fusable(hspec) and d == 128:
        acc = None
        if bwd_dq_f16(hspec) and dq_zero_mode(hspec) == "fused":
            acc = torch.empty(b, h, s * d, dtype=torch.float16, device=dout.device)  # both routes' f16 accumulator bytes
        shift = e4m3_p_shift_log2()
        if documents is not None:
            assert bound_inputs is None and acc is not None, "per-tensor varlen: no dS bound; the fused accumulator"
        r = quantize_fp8_per_tensor(dout, compute_d_with_output=out, shifted_log2_lse_from=lse.contiguous(),
                                    log2_lse_shift=float(shift), zero_accumulator=acc, ds_bound_from=bound_inputs,
                                    output_row_norm2_max=True, power_of_two_scale=pow2,
                                    **({"documents": documents} if documents is not None else {}))
        # dn2: max ||dO_q||^2 from the same read, the f16 dQ range (interface_bwd_fp4.dq_range_of)
        extra = dict(dpsum=r["D"], lse_log2=r["lse_log2"], lse_log2_shift=shift, dpsum_owned=True,
                     dn2=r["row_norm2_max"][-1:])
        if bound_inputs is not None:
            extra["s_ds_r"] = r["ds_bound"]
        if acc is not None:
            extra["dq_accum"] = acc
        if documents is not None:
            extra["doc_scales_do"] = r["doc_scales"][0]
        return r["codes"], r["scales"][0], extra
    if documents is not None:
        raise NotImplementedError("per-tensor varlen: the fused dk8 prologue (d = 128) only")
    r = quantize_fp8_per_tensor(dout, compute_d_with_output=out, output_row_norm2_max=True,
                                power_of_two_scale=pow2)
    return r["codes"], r["scales"][0], dict(dpsum=r["D"], dn2=r["row_norm2_max"][-1:])


def tb_bwd_glue(dout, lse, masks, fp8: bool, qs=None, ks=None, dt=None, dev=None, causal=False,
                deterministic=False, head_dim: Optional[int] = None, *, quant, host) -> dict:
    """The token-block backward's prologue in its dO quantiser launch (there is no unfused fallback)
    (quantize_fp8_blocks shifted_log2_lse_from= / zero_accumulator=), as the per-tensor steps' quantize_e4m3_with_d_lse does: LSE * log2 e (-inf rows
    -> 0) minus the E4M3 P^ shift, and the zeroed f16 dQ accumulator. The drivers' torch chain (zeros, isneginf / where /
    mul, sub_) and the accumulator fill go; bitwise the same values. Only where a driver takes them: no masks, d = 128,
    the dk8 operand modes (FP4: dk8_prologue_fusable), the 2-CTA cluster route (FP8; its 1-CTA kernel runs its own
    preprocess). Returns quantize_fp8_blocks keyword arguments ({} = off)."""
    from flash_attn.cute.interface_bwd_fp4 import e4m3_p_shift_log2, bwd_dq_f16, shape_carrier, _use_2cta_dk8
    from flash_attn.cute.interface_bwd_fp4 import grad_fp8_dtype
    from flash_attn.cute.dq_postprocess_cute import dq_zero_mode
    b, s, h, d = dout.shape
    d = head_dim or d  # token-block fallback shapes: the padded head dim the kernels run (padded_head_dim)
    if masks is not None and getattr(getattr(masks, "docs", None), "doc_maps_packed", False):
        # lp_docmaps packed rows (d = 64 / 128): the backward's whole prologue from the dO launch (its folded D and
        # shifted LSE per aligned row -- padding rows +2^120 / +inf, so the kernel needs no emulation -- and the zeroed
        # packed accumulator)
        from flash_attn.cute.interface_bwd_fp4 import ds_blk32_fp4
        kw = dict(shifted_log2_lse_from=lse, log2_lse_shift=float(e4m3_p_shift_log2()))
        blk_ds = (quant.ds_scaling_fp8 != "per_tensor") if fp8 else ds_blk32_fp4(quant=quant)
        # (lp_docmaps.packed_ok: the folded form, the padding rows' D'' = +2^120 is the backward's dS = +0; cov3: on
        # compact maps also unfolded -- the sf_dense backward writes D = -2^100 into the padding rows itself)
        assert blk_ds and (quant.ds_fold or masks.docs.doc_maps.compact), \
            "packed rows: the dO launch folds D (lp_docmaps.packed_d_fold)"
        if quant.ds_fold:
            kw["d_multiplier"] = -(2.0 ** -int(e4m3_p_shift_log2()))
        kw["zero_accumulator"] = torch.empty(b, h, s * d, dtype=torch.float16, device=dout.device)
        return kw
    # aligned varlen documents take the glue too: the dO launch writes +inf as the shifted LSE of the padding rows
    # (lse_row_end), so neither the bf16 preprocess nor the torch LSE prologue runs (DIAGNOSIS F1)
    aligned = masks is not None and getattr(getattr(masks, "docs", None), "aligned", False)
    if ((masks is not None and not aligned) or d != 128 or s % 128 != 0
            or lse.shape != (b, h, s) or lse.dtype != torch.float32 or not lse.is_contiguous()):
        return {}
    if fp8:
        # the 2-CTA route and the 1-CTA FP8 control both take the glue (the control skips its preprocess)
        if not grad_fp8_dtype(quant) is torch.float8_e4m3fn:
            return {}
    elif not dk8_prologue_fusable(host):
        return {}
    kw = dict(shifted_log2_lse_from=lse, log2_lse_shift=float(e4m3_p_shift_log2()))
    if aligned:
        kw["lse_row_end"] = masks.docs.row_end_tab_q
    # the D fold D'' = c D: with unit token-block descales and a block-scaled dS^ every driver's c is the host constant
    # -2^-shift (device_scales' constant scalars); the drivers check it against their own scalars before skipping
    # their dpsum.mul_; the zeroed accumulator comes with it
    from flash_attn.cute.interface_bwd_fp4 import ds_blk32_fp4
    qs = quant
    blk_ds = (qs.ds_scaling_fp8 != "per_tensor") if fp8 else ds_blk32_fp4(quant=quant)
    if blk_ds and qs.ds_fold:
        kw["d_multiplier"] = -(2.0 ** -int(e4m3_p_shift_log2()))
    if "d_multiplier" in kw or (bwd_dq_f16(host) and dq_zero_mode(host) == "fused"):
        kw["zero_accumulator"] = torch.empty(b, h, s * d, dtype=torch.float16, device=dout.device)
    if aligned and "zero_accumulator" not in kw:
        return {}  # the drivers take the glue only with its accumulator; else the masked LSE path runs as before
    return kw


def tb_glue_tag(mdo: dict, glue: dict) -> None:
    """Record what the dO launch produced for the drivers: mdo["lse_log2_shift"], mdo["dq_accum"]."""
    if "shifted_log2_lse_from" in glue:
        mdo["lse_log2_shift"] = glue["log2_lse_shift"]
    if "zero_accumulator" in glue:
        mdo["dq_accum"] = glue["zero_accumulator"]
    if "d_multiplier" in glue:
        mdo["d_folded"] = glue["d_multiplier"]


def _mx_do_prologue(ctx, dout, lse, head_dim: Optional[int] = None, *, host) -> dict:
    """d = 64 (native head_dim 64): the token-block dO quantisation also writes LSE * log2 e - shift and zeroes the f16 dQ accumulator
    (the backward's prologue in the one launch; quantize_fp8_blocks keyword arguments, as tb_bwd_glue's); {} elsewhere
    (d = 128 takes tb_bwd_glue's, bitwise)."""
    from flash_attn.cute.interface_bwd_fp4 import native_d64, bwd_dq_f16, e4m3_p_shift_log2
    b, s, h, d = dout.shape
    d = head_dim or d  # token-block fallback shapes: the padded head dim the kernels run (padded_head_dim)
    if not (native_d64(d) and getattr(ctx, "masks", None) is None and bwd_dq_f16(host) and s % 128 == 0):
        return {}
    return dict(shifted_log2_lse_from=lse.contiguous(), log2_lse_shift=float(e4m3_p_shift_log2()),
                zero_accumulator=torch.empty(b, h, s * d, dtype=torch.float16, device=dout.device))



# ---------------------------------------------------------------- masks / varlen: varlen training entries
def _varlen_apply(apply_fn, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, align, mask_kw,
                  deterministic=False, seqused_k=None, *, quant, host):
    """Upstream's packed layout in, upstream's packed layout out. The documents are repacked on the GPU at aligned
    starts (lp_varlen.LPDocs: no host sync), the low-precision forward and backward run on the repacked b = 1 batch
    with the document tables, and the outputs / gradients are unpacked (autograd of index_copy / index_select)."""
    from flash_attn.cute.lp_varlen import LPDocs
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("varlen: q, k, v are packed (total, heads, head_dim), as upstream's flash_attn_varlen_func")
    # the 2-CTA backward's key pair (256 keys) must not straddle a document: pack at 256 where it will run -- the
    # plan's 2-CTA decision (HostSpec.bwd_two_cta), so the tables are built once (DIAGNOSIS F3)
    two = str(host.bwd_two_cta) in ("on", "1", "true")
    docs = LPDocs.build(cu_seqlens_q, cu_seqlens_k, q.shape[0], k.shape[0], align=256 if two else align,
                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, seqused_k=seqused_k)
    masks = LPMasks.build(causal=causal, docs=docs, **mask_kw)
    from flash_attn.cute.lp_varlen import pack_rows, unpack_rows
    qp = pack_rows(docs, q, "q")  # one Triton row gather each way (the index_copy chain cost 2x the forward kernel)
    if docs.keep_k is not None:  # seqused_k: keys past the used length are zero (and get zero dK / dV)
        k = k * docs.keep_k.view(-1, 1, 1).to(k.dtype)
        v = v * docs.keep_k.view(-1, 1, 1).to(v.dtype)
    kp = pack_rows(docs, k, "k")
    vp = pack_rows(docs, v, "k")
    op = apply_fn(qp, kp, vp, masks)
    if isinstance(op, tuple):  # (out, lse): the LSE (1, h, s_pad) of the repacked batch -> (h, total_q), packed rows
        out, lse = op
        return unpack_rows(docs, out, "q"), lse[0].index_select(1, docs.dst_q)
    return unpack_rows(docs, op, "q")

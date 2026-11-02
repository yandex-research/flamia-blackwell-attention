# Option A of the 4-bit P.V route (pv_precision="fp4"; inference only, docs/lowprec.md): the host side.
#
#   forward : the all-FP4 forward (FwdKernelSpec.fp4all): Q / K NVFP4 1x16 along d, V NVFP4 16x16 blocks (transposed copy -> P.V);
#             causal: the diagonal-only key-local V payload on the forward's masked tiles (quant.fp4_diagonal.quantize_fp4_causal_diagonal).
#   backward: not shipped. A research backward (the dk8 kernels with dP^T = V dO^T as an NVFP4 GEMM on the forward's
#             own V codes, reproduce-forward-p rebuilding the forward's P^ bit for bit) was built and measured but is
#             not part of this library: it was slower and less accurate than the default training route (see
#             docs/lowprec.md, option A). Option A is inference.
#             p_consistency="reproduce_forward_p" (the default) still selects the forward variant that exports its P^
#             frames (the inference contract's O / LSE); "none" the plain one.
import dataclasses
from typing import Optional

import torch

E2 = torch.float4_e2m1fn_x2
_SR_CALL = [0]   # (the dO4 SR seed of the archived backward; kept: the test harnesses reset it)


def pv4_vdiag(causal: bool, qspec) -> bool:
    """The causal-safe V of option A: the diagonal-only key-local payload whenever the kernel is causal, unless the caller
    opted out with causal_safe_v="never" (the leaky 16x16 blocks everywhere, kept reachable for the historical numbers).
    Dense attention has no diagonal tile."""
    if not causal:
        return False
    mode = str(qspec.causal_safe_v)   # QuantSpec.causal_safe_v (the plan's causal_safety: 'off' -> "never")
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"causal_safe_v must be 'auto', 'always' or 'never', got {mode!r}")
    return mode != "never"


def pv4_check_specs(q, k, specs) -> None:
    """Option A is defined on the dk8 blk32x128 route: E4M3 token-block Q / K / dO, blk32 dS, row P^, NVFP4 S (SF 16).
    specs: the call's plan.specs (its QuantSpec states the token-block operand scaling, lowprec.plan.build_specs)."""
    from flash_attn.cute.interface_bwd_fp4 import check_scaling_consistency, ds_blk32_fp4
    from flash_attn.cute.lp_docmaps import packed_tiles
    _pk = packed_tiles(q) is not None   # (cov5: the packed rows of compact maps, any T)
    if q.shape[-1] != 128 or ((q.shape[1] % 128 or k.shape[1] % 128) and not _pk):
        raise NotImplementedError(f"4-bit P.V (option A): head_dim 128 and s % 128 == 0 only, got q {tuple(q.shape)}")
    qs = specs.quant
    check_scaling_consistency("blk32x128", qs.operand_scaling_fp4 or "blk32x128", "pv4 (option A)")
    why = []
    if not ds_blk32_fp4(quant=qs):
        why.append("the block-scaled dS (blk32)")
    if int(qs.s_sf_vec) != 16:
        why.append("NVFP4 S (scale group 16)")
    if qs.p_scaling_fp4 != "row":
        why.append("row P^ (p_scaling='row')")
    if why:
        raise NotImplementedError("4-bit P.V (option A) runs on the dk8 blk32x128 backward only; this configuration "
                                  "changes: " + ", ".join(why))


def diag_tile_ranges(s_q: int, s_k: int, diag_fwd_tiles: int):
    """Per 128-query tile m: [lo, hi) of the key tiles the forward's CTA holding m treats as masked (the diagonal
    pairs). Pure Python on shapes."""
    off = s_k - s_q
    nk = s_k // 128
    out = []
    for m in range(s_q // 128):
        mf = m - m % diag_fwd_tiles
        lo = (128 * mf + off) // 128
        hi = -(-(128 * (mf + diag_fwd_tiles) + off) // 128)
        out.append((max(lo, 0), min(hi, nk)))
    return out


def forward_value_fn(fspec) -> str:
    """The all-FP4 forward's value function as its constructor resolves it: FwdKernelSpec.fp4all_exp2: "exp2" |
    "mitchell"."""
    return "exp2" if fspec.fp4all_exp2 else "mitchell"


def pv4_repro_fwd_spec(fspec, softmax_exp: str):
    """The forward settings `reproduce-forward-p` needs (the plan sets them on option A's FwdKernelSpec; applied here too,
    so a caller's spec cannot drop them): the codes row sum (Z = sum of the dequantised P^, so the backward's
    P~ = P^ / Z sums to 1 and D = dO . O is exact) with the E4M3-decoded group SFs, and the explicit value function
    (Mitchell / exp2)."""
    return fspec.replace(fp4all_rowsum="codes", fp4all_sf_decode=True, fp4all_exp2=softmax_exp == "exp2")


def pv4_forward(q, k, v, softmax_scale: float, causal: bool, vdiag=None,
                p_consistency: Optional[str] = None, softmax_exp: Optional[str] = None, repro_dbg: bool = False,
                p_mix: Optional[str] = None, qk_had: bool = False, masks=None, *, specs) -> dict:
    """The all-FP4 forward and every shared operand the backward needs. Returns a state dict (the autograd ctx content):
    out, lse and the operands. No host read.
    p_consistency="reproduce_forward_p" (pv4_repro.py): the forward also exports, per query, its final frame, its row sum
    and every frame change, so the backward rebuilds this forward's P^ bit for bit; softmax_exp ("mitchell" | "exp2") is
    the forward's value function (the backward follows it). repro_dbg (debug build): st["repro_dbg"], every code / SF.
    p_mix (exact classifier; reproduce-forward-p, softmax_exp "mitchell"): 8 characters over M / E / 2 / 3, the value function of
    each 16-key group position (flash_fwd_sm100_fp4 `p_mix`); the backward follows it bit for bit.
    qk_had (qk_hadamard): Q R / K R -- the NVFP4 S codes AND the E4M3 token-block copies from ONE rotated read per tensor
    (qk_hadamard.tb_qk_operands), the token-block exponents taken on the rotated rows. SCALING-CONSISTENCY RULE: fwd and
    bwd share operand scaling (by design) -- these are the only Q / K codes of the call (the forward's
    S, the backward's S recompute and dK / dQ operands); V is not rotated. The backward rotates dQ / dK back (had_signs).
    Inference only: the quantisers the forward reads and the forward kernel (the backward-only E4M3 token-block Q / K
    copies and the factor table are not built: option A has no backward). The reproduce-forward-p forward
    keeps its export buffer (an empty allocation, no launch): the kernel variant that writes it is the one whose O / LSE
    the inference contract fixed. specs: the call's plan.specs (lowprec.specs.PlanSpecs).
    masks: lp_masks.LPMasks with aligned varlen documents only (b = 1 tensors, vall4): the diagonal V quantiser's tile
    constants restart per document, the forward runs on the documents (its export at each document's query tiles)."""
    from flash_attn.cute.pv4_repro import p_consistency_mode, softmax_exp_mode, REPRO_WORDS
    p_consistency = p_consistency_mode(p_consistency)
    repro = p_consistency == "reproduce_forward_p"
    if softmax_exp is not None and not repro:
        raise ValueError("softmax_exp is the reproduce-forward-p pair's explicit value function (p_consistency="
                         "'reproduce_forward_p'); without it the forward follows the configuration")
    softmax_exp = softmax_exp_mode(softmax_exp) if softmax_exp is not None else forward_value_fn(specs.fwd)
    if p_mix is not None:
        from flash_attn.cute.flash_fwd_sm100_fp4 import parse_p_mix
        p_mix = "".join(parse_p_mix(p_mix))
        if not repro or softmax_exp != "mitchell":
            raise ValueError("p_mix: the reproduce-forward-p forward on the Mitchell configuration (softmax_exp='mitchell'); "
                             f"got p_consistency={p_consistency!r}, softmax_exp={softmax_exp!r}")
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, fp4all_quantize_qkv
    from flash_attn.cute.quant.fp4_diagonal import quantize_fp4_causal_diagonal
    b, s_q, h, d = q.shape
    s_k = k.shape[1]
    docs = masks.docs if masks is not None else None
    from flash_attn.cute.lp_docmaps import packed_tiles
    pk_v = packed_tiles(v)   # (cov5) the packed rows of compact maps: (tile table, device tile count) of the keys
    if docs is not None and repro_dbg:
        raise NotImplementedError("repro_dbg (the debug dump) is batched only")
    if vdiag is None:
        vdiag = pv4_vdiag(causal, specs.quant)
    assert not vdiag or causal
    qk4 = None
    if qk_had:
        from flash_attn.cute import qk_hadamard
        if s_q != s_k and docs is None:
            raise NotImplementedError("4-bit P.V (option A): s_q == s_k (the token-block factor tensor is (4, b, h, s / 32))")
        # (documents: no factor table -- it is the archived backward's; inference reads the NVFP4 S codes only)
        fac = torch.zeros(4, b, h, s_q // 32, dtype=torch.float32, device=q.device) if docs is None else None
        mq, mk = qk_hadamard.tb_qk_operands(q, k, cols=128, fac=fac, fp4=True, qspec=specs.quant)
        qk4 = (mq["x4"], mq["sf"], mk["x4"], mk["sf"])
        if pk_v is not None:   # (cov5: the NVFP4 S codes' dense SF words, quant.fp8 packed_dense)
            qk4 = qk4 + (mq["sf4_rows"], mk["sf4_rows"])
    # the forward's Mitchell classifier is the clamped one (exact for any scores); the backward reproduces it
    if vdiag:
        # ONE launch: Q / K row copies, both V payloads (plain 16 x 16 and the diagonal key-local one), the per-key bias /
        # u for the forward and r_k for the backward. SCALING-CONSISTENCY RULE: forward and backward read these same codes.
        # With qk4 (the rotated Q / K) the diagonal launch builds the V side only.
        split = qk4 is not None
        if pk_v is not None:
            # (cov5) the packed rows of compact maps: the tile tables / device counts of the maps
            dm = docs.doc_maps
            dkw = dict(documents=dict(doc_tab=docs.doc_tab, tile_doc_k=docs.tile_doc(128), doc_key_len=0,
                                      packed=(dm.vt_tile, dm.vt_tile_q, dm.ntiles)))
        else:
            dkw = {} if docs is None else dict(documents=dict(doc_tab=docs.doc_tab, tile_doc_k=docs.tile_doc(128),
                                                             doc_key_len=docs.key_len()))
        ops = quantize_fp4_causal_diagonal(k, v, softmax_scale, key_channel="off", query=None if split else q,
                                           make_key_row_copy=not split, make_option_a_key_factors=True, **dkw)
        if split:
            ops.update(q4=qk4[0], sfq=qk4[1], k4=qk4[2], sfk=qk4[3])
            if pk_v is not None:
                ops.update(sfq_rows=qk4[4], sfk_rows=qk4[5])
        ops["vtok_diag"] = True
    else:
        ops = fp4all_quantize_qkv(q, k, v, qk4=qk4, packed=pk_v)
    rp = rdbg = vb = vu = None
    if repro:
        if pk_v is not None:   # (cov5 packed rows: one record per packed query row, (h, W, T + 256))
            rp = torch.empty(h, REPRO_WORDS, s_q + 256, dtype=torch.int32, device=q.device)
        else:
            rp = torch.empty(b, h, s_q // 128, REPRO_WORDS, 128, dtype=torch.int32, device=q.device)
        if repro_dbg:
            rdbg = torch.zeros(b, h, s_q, s_k + 9 * (s_k // 16), dtype=torch.uint8, device=q.device)   # codes, SF, gm bits
        if vdiag:
            # the per-key tensors in the kernels' (b, h_k, s_k) layout, ONE copy shared by the forward (whose driver view
            # of this permuted tensor is then free) and the backward's P^ reproduction
            vb = ops["vtok_bias"].permute(0, 2, 1).float().contiguous()
            vu = ops["vtok_u"].permute(0, 2, 1).float().contiguous()
            ops = dict(ops, vtok_bias=vb.permute(0, 2, 1), vtok_u=vu.permute(0, 2, 1))
    fsp = dataclasses.replace(specs, fwd=pv4_repro_fwd_spec(specs.fwd, softmax_exp)) if repro else specs
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, ops=ops,
                                                 vtok=vdiag, repro_out=rp, repro_dbg=rdbg, p_mix=p_mix,
                                                 masks=masks if docs is not None else None, specs=fsp)
    # the forward's outputs only (the state dict keeps the forward operands for the LSE lookup)
    return dict(out=out, lse=lse, q4=ops["q4"], sfq=ops["sfq"], k4=ops["k4"], sfk=ops["sfk"], v4=ops["v4"],
                sfv=ops["sfv"], meta=dict(vdiag=vdiag, causal=causal, softmax_scale=softmax_scale,
                                          shape_q=tuple(q.shape), shape_k=tuple(k.shape), dtype=q.dtype,
                                          inference=True))

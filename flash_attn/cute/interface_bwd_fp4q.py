# Host driver of the all-FP4 attention step: the all-FP4 forward (FwdKernelSpec.fp4all, flash_fwd_sm100_fp4.py) and the query-row
# all-FP4 backward (flash_bwd_sm100_fp4q.py).
#   forward : Q4 / K4 row-wise 1x16 NVFP4, V 16x16 (the transposed natural copy multiplies P^, the row-wise copy is kept for
#             the backward's dP); returns O, LSE and the per-query final integer log2 frame F.
#   backward: Q^T (along queries), K^T (along keys, C3 32-block order), dO row-wise + dO^T (1x16, RTN / SR / MS-EDEN, optional
#             head-dim Hadamard) and D = sum_d O dO^ from one Triton read; the kernel reproduces the forward's P^.
import math
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
import cuda.bindings.driver as cuda

from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor, _flash_attn_fwd
from flash_attn.cute.quant.fp4 import (quantize_fp4_rows, quantize_fp4_blocks_2d, SR_SCALE_DIVISOR as SR_SMAX,
                                       EDEN_SCALE_DIVISOR as EDEN_SMAX, CausalRowScale)
from flash_attn.cute.quant.fp4_diagonal import quantize_fp4_causal_diagonal
from flash_attn.cute.lowprec.specs import resolved_key

E2 = torch.float4_e2m1fn_x2
LOG2_E = math.log2(math.e)
# the forward's frame bias for the backward's LSE (FP4_PV_RESCALE_THRESHOLD: the all-4-bit forward's lazy threshold, shared
# with the kernel's self.max_offset: one module constant)
from flash_attn.cute.flash_bwd_sm100_fp4q import FP4_PV_RESCALE_THRESHOLD  # noqa: E402
LSE_LOG2_OFFSET = math.log2(6) + math.log2(448) - FP4_PV_RESCALE_THRESHOLD


# ---- the host reads go through the call's specs (lowprec.specs, from its plan: lowprec.plan.build_specs), handed to
# every helper explicitly; nothing is built here.
def _quant_spec(qspec):
    if qspec is None:
        raise TypeError("a lowprec.specs.QuantSpec is required (the call's plan.specs.quant)")
    return qspec


def _bwdq_spec(bspec):
    if bspec is None:
        raise TypeError("a lowprec.specs.BwdQSpec is required (the call's plan.specs.bwdq)")
    return bspec


def _make_fp4_ptr(t: Optional[torch.Tensor]):
    from cutlass.cute.runtime import make_ptr

    return make_ptr(cutlass.Float4E2M1FN, 0 if t is None else t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def fp4all_quantize_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, signs: Optional[torch.Tensor] = None,
                        qk4: Optional[tuple] = None, backward_operands: bool = False, docs=None,
                        packed=None) -> dict:
    """Forward operands (outer scale 1, as the NVFP4-PV forward): Q4 / K4 row-wise, V 16x16 -> transposed (forward PV) and
    row-wise (backward dP) copies of the same codes. signs: head-dim Hadamard applied inside the quantiser reads.
    qk4: (q4, sfq, k4, sfk) already made by the caller (option A's rotated token-block quantiser, qk_hadamard): Q / K are
    then not read here, V only.
    backward_operands (signs None, no qk4): the backward's K-derived and statistics operands from the SAME three reads --
    K^T (the dQ operand, "kt4" / "sfkt"), max |Q| per row ("query_row_amax") and max_k ||V_k||^2 ("v_norm2_max", torch's
    norm order) -- so the dense backward reads neither K nor V nor Q's statistics again (fp4q_prepare).
    docs (aligned varlen documents, b = 1): max_k ||V_k||^2 per document ((n_docs + 1,): the backward's per-document dS
    bound); everything else here is per row / per 16-key block inside a document already.
    packed (cov5; lp_docmaps compact maps): (tile table, device tile count) of the keys -- q / k / v are the caller's
    packed (1, T, ., 128) rows; the codes at the packed rows, the SF bytes as dense per-row words ("sfq_rows" /
    "sfk_rows" / "sfv_rows"), V's transposed codes as the key-parity copies ("vt4": (2, h_kv, d, TB), quant.fp4
    transposed_row_packed)."""
    # SCALING-CONSISTENCY RULE: these codes are the forward's AND the backward's operands (S, P^ V^ / dP)
    rot = signs is not None
    bw = backward_operands and not rot and qk4 is None
    rows = {}
    if qk4 is not None:
        q4, sfq, k4, sfk = qk4[:4]
        if packed is not None:
            rows = dict(sfq_rows=qk4[4], sfk_rows=qk4[5])
    elif packed is not None:
        from flash_attn.cute.lp_docmaps import packed_tiles
        assert not rot, "cov5: the packed rows' plain Q / K row copies"
        # (cov6 bw: max |Q| per packed row; K^T as the plain 1 x 16 copy -- key-parity codes, the packed group table)
        qq = quantize_fp4_rows(q, packed_rows=packed_tiles(q), output_row_amax=bw)
        kq = quantize_fp4_rows(k, packed_rows=packed, make_transposed_copy=bw, transposed_copy_in_kernel_key_order=False,
                               make_transposed_sf_halves_swapped=bw)
        q4, sfq, k4, sfk = qq["codes"], qq["scale_factors"], kq["codes"], kq["scale_factors"]
        rows = dict(sfq_rows=qq["sf4_rows"], sfk_rows=kq["sf4_rows"])
    else:
        qq = quantize_fp4_rows(q, apply_hadamard_rotation=rot, output_row_amax=bw)
        kq = quantize_fp4_rows(k, apply_hadamard_rotation=rot, make_transposed_copy=bw, make_transposed_sf_halves_swapped=bw)
        q4, sfq, k4, sfk = qq["codes"], qq["scale_factors"], kq["codes"], kq["scale_factors"]
    vq = quantize_fp4_blocks_2d(v, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False,
                                transposed_copy_layout="batch_head_dim_seq", apply_hadamard_rotation=rot,
                                output_tensor_max="row_norm2" if bw else None,
                                **(dict(row_documents=docs.tile_doc(128), n_documents=docs.b + 1)
                                   if (bw and docs is not None) else {}),
                                **({} if packed is None else dict(packed_rows=packed)))
    ops = dict(q4=q4, sfq=sfq, k4=k4, sfk=sfk, v4=vq["codes"], sfv=vq["scale_factors"], vt4=vq["transposed_codes"],
               sfvt=vq["transposed_scale_factors"], **rows)
    if packed is not None:
        ops["sfv_rows"] = vq["sf4_rows"]
    if bw:
        ops.update(kt4=kq["transposed_codes"], sfkt=kq["transposed_scale_factors"],
                   sfkt1=kq["transposed_scale_factors_halves_swapped"], query_row_amax=qq["row_amax"],
                   v_norm2_max=vq["tensor_max"])
        if packed is not None:   # (cov6) the packed K^T's SF group table (the backward's load warp builds the atoms)
            ops["sfkt_groups"] = kq["transposed_sf_groups"]
    return ops




def fp4all_vtok(causal: bool = False, mask_mod=None, window_size=None, *, qspec) -> bool:
    """Whether to build the causal-leak-free V^ (per-key pow2 scale folded into P^ as a bias).

    A 4-bit PV groups V's scale factors along keys, so one group's scale can be set by a key the
    output must not see: without this, a causal output depends on up to 15 future V rows (measured
    leak 3.13e-01; with it, exactly 0). A dense forward has no such constraint and should not pay
    for the per-key bias and weight, so the decision is made from the kernel's own specialization
    and is compile-time: `causal_safe_v="auto"` turns it on exactly when the kernel is built for a
    mask -- causal, a sliding window, or a mask_mod, any of which can hide a future key inside a
    scale group. "always" / "never" force it.
    """
    qs = _quant_spec(qspec)
    if _vtok_diag_knob(qs):
        # a boolean cannot say "key-local on the diagonal tiles only"; a caller that only knows the boolean would
        # silently run another route under this knob's name, so it is refused here rather than ignored
        raise NotImplementedError(
            "QuantSpec.vtok_diag (diagonal-only-key-local-v-scale, causal_safety='diagonal') is set, but this caller "
            "resolves the causal-safe V route as a boolean and has no diagonal-only branch. Use flash_attn_fp4all_fwd "
            "(fp4all_vtok_mode), or another causal_safety.")
    forced = qs.fp4all_vtok
    if forced is not None:
        return forced
    # `causal-safe-v-opt-out-not-resolvable-in-fp4all`: resolved exactly as the forward driver's guard resolves it --
    # QuantSpec.causal_safe_v -- so "never" reaches both readers of the one setting. The spec is in both compile keys, so
    # the leaky and the safe kernels never share a cache slot.
    mode = str(qs.causal_safe_v)
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError(f"causal_safe_v must be 'auto', 'always' or 'never', got {mode!r}")
    return bool(causal) or mask_mod is not None or window_size is not None


def _vtok_diag_knob(qspec) -> bool:
    """QuantSpec.vtok_diag (causal_safety="diagonal"), read in exactly one place, default off. It reaches both compile
    keys through the spec and as the explicit `vtok_diag` item of each key."""
    return _quant_spec(qspec).vtok_diag


def _diag_k_knob(qspec) -> str:
    """QuantSpec.vtok_diag_k (options k_channel; `k-operand-of-dq-leaks`): "fold" (default) closes the K channel into dQ with the key-local K^T
    on every pair and r_k folded into host-built operands -- no kernel multiply, any head ratio or offset, +2.7 % / +4.0 %
    step against the open channel (A3 / A1); "1" closes it on the cut pair only (aliasing a key-local K^T tile into the V
    tile; MHA, tile-aligned offsets; +17.9 % / +11.3 %); "all" is "1"'s arithmetic on every pair; "0" leaves the channel
    open (k_channel "fold" / "cut" / "all" / "open" map to "fold" / "1" / "all" / "0" here). Keyed through the spec and
    as the backward's resolved diag_k / diag_kr_all / diag_fold."""
    # (QuantSpec.vtok_diag_k, returned in the internal spelling the callers test)
    return {"open": "0", "cut": "1", "all": "all", "fold": "fold"}[_quant_spec(qspec).vtok_diag_k]


def _diag_k_check(r_gqa: int, s_q: int, s_k: int, *, qspec, bspec) -> str:
    """The diagonal route's K-channel mode, "off" / "cut" / "all"; refuses the shapes a mode is not built for rather
    than leaking silently. "cut" aliases the cut pair's key-local K^T tile into the V tile, which holds only when that
    pair is visited once per key tile: MHA and a diagonal offset that is a whole number of tiles."""
    v = _diag_k_knob(qspec)
    if v == "0":
        return "off"
    dq8 = _bwdq_spec(bspec).dq8
    if dq8 or (v == "1" and (r_gqa != 1 or (s_k - s_q) % 128 != 0)):
        raise NotImplementedError(
            f"`k-operand-of-dq-leaks` is not fixed on the diagonal-only route for this shape (k_channel mode {v}, "
            f"q heads per kv head {r_gqa}, s_k - s_q = {s_k - s_q}, dq_operand e4m3 {int(dq8)}): dQ of a query would read "
            f"the E4M3 scale of K^T groups holding keys after it. The cut-pair fix needs MHA, a tile-aligned diagonal "
            f"and the E2M1 dQ; k_channel='all' covers any head ratio and offset (E2M1 dQ); "
            f"k_channel='open' runs with that channel open, knowingly.")
    return {"1": "cut", "all": "all", "fold": "fold"}[v]


def fp4all_quantize_k_diag(k: torch.Tensor, centre: str = "prefix") -> dict:
    """`k-operand-of-dq-leaks`: the key-local K^T payload of the diagonal route's cut pair, the dQ GEMM's B operand there.
    K^T's normal quantisation has E4M3 scales per 16 KEYS, so on the pair the causal boundary cuts a query's dQ depends
    on keys after it. Here every key carries its own power of two: codes of K / 2^e_k (e_k = ceil(log2(amax_k / 6)),
    RTN, unit scale) with the key tile's constant 2^c_n as every scale byte, and r_k = 2^(e_k - c_n) for the kernel to
    multiply into the dQ copy of dS' on that pair only, so dS'^ K^T is unchanged in exact arithmetic. c_n is the causal
    prefix constant of the V payload (_vtok_diag_frame_const), so nothing depends on a key a query cannot see.
    -> dict(xt4, sft: the transposed codes / SF in quant.fp4's transposed-copy layout, C3 key order; kr: (b, s, h_k) f32)."""
    b, s, hk, d = k.shape
    assert d == 128 and s % 128 == 0, (k.shape,)
    kf = k.float()
    e = torch.ceil(torch.log2((kf.abs().amax(-1) / 6.0).clamp(min=2.0 ** -120))).clamp(-120.0, 120.0)   # (b, s, hk)
    c_tile = _vtok_diag_frame_const(e, centre)                                                           # (b, s // 128, hk)
    c = c_tile.repeat_interleave(128, dim=1)
    ktil = (kf / torch.pow(2.0, e)[..., None]).contiguous()
    kq = quantize_fp4_rows(ktil, make_row_copy=False, make_transposed_copy=True, rounding="unit")
    sft = _sft_raw(kq["transposed_scale_factors"]).clone()                                               # (b, hk, 1, s // 64, 32, 4, 4)
    cbyte = torch.pow(2.0, c_tile).to(torch.float8_e4m3fn).view(torch.uint8)                             # (b, s // 128, hk)
    sft.reshape(b, hk, 1, s // 128, 2, 32, 4, 4)[:] = cbyte.permute(0, 2, 1)[:, :, None, :, None, None, None, None]
    return dict(xt4=kq["transposed_codes"], sft=sft.permute(4, 5, 2, 6, 3, 1, 0), kr=torch.pow(2.0, e - c).contiguous(),
                c=c_tile)


def fp4all_vtok_mode(causal: bool = False, mask_mod=None, window_size=None, *, qspec) -> str:
    """The causal-safe V route of the all-FP4 step: "off" (key-grouped V scales, leaky under a mask), "full"
    (`key-local-v-scale` on every tile) or "diag" (`diagonal-only-key-local-v-scale`: key-local on the causal
    diagonal tiles only, the only tiles a V scale group can straddle the causal boundary in; the fast key-grouped
    payload everywhere else).

    "diag" is chosen by QuantSpec.vtok_diag (causal_safety="diagonal") and is causal-only: dense attention has no
    diagonal tile, and a sliding window or mask_mod has masked tiles the diagonal classification does not describe, so
    both are refused rather than silently served another route. QuantSpec.fp4all_vtok=True asks for the full fix
    explicitly, so the two settings together are refused as contradictory. Otherwise the route is `fp4all_vtok`'s decision.
    """
    qs = _quant_spec(qspec)
    if _vtok_diag_knob(qs):
        if not causal:
            raise NotImplementedError(
                "causal_safety='diagonal' is causal-only: dense attention has no diagonal tile")
        if mask_mod is not None or window_size is not None:
            raise NotImplementedError(
                "causal_safety='diagonal' classifies only the causal diagonal; a mask_mod or window is not supported")
        if qs.fp4all_vtok is True:
            raise ValueError("QuantSpec.fp4all_vtok=True (the full key-local fix) and vtok_diag (diagonal-only) "
                             "are mutually exclusive")
        return "diag"
    return "full" if fp4all_vtok(causal=causal, mask_mod=mask_mod, window_size=window_size, qspec=qs) else "off"


def _sft_raw(sft: torch.Tensor) -> torch.Tensor:
    """A transposed copy's SF view (quant.fp4), which is sft_c.permute(4, 5, 2, 6, 3, 1, 0) of a contiguous
    (b, h, 1, s // 64, 32, 4, 4) buffer, back to that buffer."""
    raw = sft.permute(6, 5, 2, 4, 0, 1, 3)
    assert raw.is_contiguous(), raw.shape
    return raw


def _vtok_diag_frame_const(e: torch.Tensor, mode: str = "prefix", tile: int = 128) -> torch.Tensor:
    """The key-local payload's per-key-tile group scale exponent c_n, (b, s // tile, h_k).

    c_n exists only to keep the diagonal tile's P^ frame in the same binades as the unmasked tiles'. It rides in
    the V group scale, NOT on P^, so it must not be a function of any key a query cannot see, or it becomes the
    very leak this route removes (`frame-constant-leaks-through-a-tensor-statistic`: a whole-tensor constant
    moved O for early queries by 1.95e-03 under a region perturbation; the causal prefix leaves it bitwise). Modes:
      "prefix"  (default) c_n = round(mean of e over key tiles [0, n - 1)), c_0 = c_1 = 0. Query tile m's masked
                tiles are n = m + floor(d / tile) and, when the diagonal offset d = s_k - s_q is not a multiple of
                the tile, also n + 1; the earliest query of tile m sees every key below (n - 1) * tile in both
                cases, so dropping the two most recent tiles makes this causally sound at every shape.
      "none"    c = 0. Also sound; leaves the diagonal P^ shifted by log2(V magnitude / 6).
      "tensor"  c = round(mean over ALL keys). LEAKS BY CONSTRUCTION -- a probe control only.
    """
    b, s, hk = e.shape
    nt = s // tile
    per_tile = e.reshape(b, nt, tile, hk).mean(2)                     # (b, nt, hk)
    if mode == "none":
        c = torch.zeros(b, nt, hk, device=e.device, dtype=torch.float32)
    elif mode == "tensor":
        c = e.mean(1, keepdim=True).expand(b, nt, hk).clone()
    elif mode == "prefix":
        csum = torch.cumsum(per_tile, dim=1)
        c = torch.zeros_like(csum)
        if nt > 2:
            cnt = torch.arange(1, nt - 1, device=e.device, dtype=torch.float32).view(1, nt - 2, 1)
            c[:, 2:] = csum[:, :-2] / cnt
    else:
        raise ValueError(f"unknown centring mode {mode!r}")
    return torch.round(c).clamp(-7.0, 7.0)


def fp4all_quantize_qkv_diag(q, k, v, softmax_scale: float, signs: Optional[torch.Tensor] = None,
                             centre: str = "prefix", docs=None, *, qspec, bspec) -> dict:
    """Operands for `diagonal-only-key-local-v-scale`. V is quantised TWICE and the two payloads are concatenated
    along the key axis of the forward's transposed copy:
      keys [0, s)   -- `fp4all_quantize_qkv`'s key-grouped 16x16 block scales, read by the fully unmasked tiles,
                       where every key of the tile is visible to every query of the tile, so a group scale is a
                       function of visible keys only;
      keys [s, 2s)  -- the key-local payload v~ = V / s_k, s_k = 2^e_k per (batch, key, kv head), read by the
                       causal masked (diagonal) tiles, the only tiles a group can straddle the boundary in.
    The key-local payload's group scale is the per-key-tile constant 2^c_n (_vtok_diag_frame_const); only the
    residual r_k = 2^(e_k - c_n) rides on P^, as the raw-score bias log2(r_k) / (softmax_scale log2 e) with the
    row-sum weight u_k = 1 / r_k. Both are powers of two, so the bias and the weight are exact.

    The backward needs the diagonal payload too (`key-grouped-v-scale-leaks-through-dp-too`): its dP = dO V^T is
    V's second consumer, so the row-wise copy of the key-local payload (v4b / sfvb) is its dP operand on the
    diagonal pair, and the dS bound has to cover that payload's row norms (vtok_vmax_b).
    """
    b, s, hk, d = v.shape
    from flash_attn.cute.lp_docmaps import packed_tiles
    pk_ = packed_tiles(v) if docs is not None else None   # (cov5: the packed rows of compact maps)
    assert d == 128 and (s % 128 == 0 or pk_ is not None), (v.shape,)
    # `k-operand-of-dq-leaks`: refused here, at the forward, for the shapes the backward cannot close it for
    # (documents: the per-document offsets are not host-known; "fold" / "all" serve any offset, "cut" is refused there)
    kmode = _diag_k_check(q.shape[2] // hk, q.shape[1], s if docs is None else q.shape[1], qspec=qspec, bspec=bspec)
    if docs is not None and (signs is not None or centre != "prefix" or kmode == "cut"):
        raise NotImplementedError("the diagonal route on documents: the fused builder (no Hadamard, prefix centring, "
                                  "k_channel fold / all / open)")
    # (the diagonal route's operands come from the fused builder; the torch reference chain below, bitwise the same
    # bytes, remains only for the rotated / non-prefix calls)
    if signs is None and centre == "prefix":
        # ONE launch (quant.fp4_diagonal): Q / K row copies, both V payloads with their scales and per-key tensors, the
        # backward's K-channel operands (formerly k_diag_fused_v in the backward) and the per-query-row key-norm prefix
        # maxima of the causal dS scale. SCALING-CONSISTENCY RULE: the forward and the backward read these same codes.
        maxima = q.shape[1] <= s and fp4all_ds_scale_mode(True, True, qspec=qspec) == "causal"
        if docs is not None:
            maxima = fp4all_ds_scale_mode(True, True, qspec=qspec) == "causal"
        if pk_ is not None:   # (cov5) the packed rows: the maps' tile tables / device counts
            dm_ = docs.doc_maps
            dkw_ = dict(documents=dict(doc_tab=docs.doc_tab, tile_doc_k=docs.tile_doc(128), doc_key_len=0,
                                       packed=(dm_.vt_tile, dm_.vt_tile_q, dm_.ntiles)))
        else:
            dkw_ = {} if docs is None else dict(documents=dict(doc_tab=docs.doc_tab, tile_doc_k=docs.tile_doc(128),
                                                               doc_key_len=docs.key_len()))
        r = quantize_fp4_causal_diagonal(k, v, softmax_scale, key_channel=kmode, query=q, make_key_row_copy=True,
                                         make_key_norm_prefix_maxima=maxima, query_seqlen=q.shape[1],
                                         query_heads=q.shape[2], make_query_row_amax=maxima, **dkw_)
        ops = {n: r[n] for n in ("q4", "sfq", "k4", "sfk", "v4", "sfv", "vt4", "sfvt", "v4b", "sfvb", "vtok_bias",
                                 "vtok_u", "vtok_s", "vtok_c")}
        for n in ("sfq_rows", "sfk_rows", "sfv_rows", "vt4b_packed", "sfvb_tiles"):   # (cov5 packed)
            if n in r:
                ops[n] = r[n]
        for n in ("xt4", "sft", "sft1", "xt4b", "sftb", "kr", "vb2", "vu2", "dva", "fdk", "fdv", "sfv2", "sfvb2",
                  "sftb_tiles", "sfv2_rows", "sfvb2_rows", "pad_keys"):
            if n in r:
                ops["kdiag_" + n] = r[n]
        for n in ("vmax_q", "rmax_q", "query_row_amax", "backward_amax_slots"):
            if n in r:
                ops[n] = r[n]   # saved for the backward's fused dO launch (fp4q_prepare)
        ops["vtok_diag"] = True
        return ops
    ops = fp4all_quantize_qkv(q, k, v, signs=signs)   # payload A (fast, key-grouped)
    vf = v.float()
    amax = vf.abs().amax(-1)                                   # (b, s, hk)
    e = torch.ceil(torch.log2((amax / 6.0).clamp(min=2.0 ** -120))).clamp(-120.0, 120.0)
    c_tile = _vtok_diag_frame_const(e, centre)                 # (b, s // 128, hk)
    c = c_tile.repeat_interleave(128, dim=1)                   # (b, s, hk)
    s_k = torch.pow(2.0, e)
    vtil = (vf / s_k[..., None]).contiguous()
    vq2 = quantize_fp4_blocks_2d(vtil, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False,
                                 rounding="unit")  # SF == 1
    vq2 = dict(x4=vq2["codes"], sf=vq2["scale_factors"], xt4=vq2["transposed_codes"], sft=vq2["transposed_scale_factors"])
    # the unit group scale becomes the constant 2^c_n of the key tile (exact in E4M3 for |c| <= 7). One key tile of
    # 128 keys is two 64-key SF atoms, so the atom axis reshapes to (tiles, 2).
    sf2 = _sft_raw(vq2["sft"]).clone()                          # (b, hk, 1, s // 64, 32, 4, 4) uint8 E4M3
    cbyte = torch.pow(2.0, c_tile).to(torch.float8_e4m3fn).view(torch.uint8)   # (b, s // 128, hk)
    sf2v = sf2.reshape(b, hk, 1, s // 128, 2, 32, 4, 4)
    sf2v[:] = cbyte.permute(0, 2, 1)[:, :, None, :, None, None, None, None]
    ops["vt4"] = torch.cat([ops["vt4"], vq2["xt4"].permute(0, 2, 1, 3).contiguous()], dim=3).contiguous()
    ops["sfvt"] = torch.cat([_sft_raw(ops["sfvt"]), sf2], dim=3).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    # the diagonal payload's ROW-WISE copy and SF: the backward's dP operand on the diagonal pair
    sfb_row = vq2["sf"].permute(6, 5, 2, 4, 0, 1, 3).clone()      # (b, hk, s // 128, d // 64, 32, 4, 4)
    sfb_row[:] = cbyte.permute(0, 2, 1)[:, :, :, None, None, None, None]
    ops["v4b"] = vq2["x4"]
    ops["sfvb"] = sfb_row.permute(4, 5, 2, 6, 3, 1, 0)
    ops["vtok_c"] = c_tile
    ops["vtok_diag"] = True
    r_log2 = (e - c).contiguous()                              # (b, s, hk)
    ops["vtok_bias"] = (r_log2 / (softmax_scale * LOG2_E)).contiguous()
    ops["vtok_u"] = torch.pow(2.0, -r_log2).contiguous()
    # the backward's dP operand on the diagonal pair dequantises to V / r_k, so the dS bound must cover ITS row norms
    # as well as plain V's -- otherwise dS' saturates E4M3 on the diagonal pair and dQ / dK come back several per
    # cent too large, which a cosine does not show and a norm ratio does
    ops["vtok_vmax_b"] = (vf / torch.pow(2.0, r_log2)[..., None]).norm(dim=-1).amax()  # a device scalar: no host read
    ops["vtok_s"] = s_k
    return ops


def fp4all_quantize_qkv_vtok(q, k, v, softmax_scale: float, signs: Optional[torch.Tensor] = None,
                             signs_v: Optional[torch.Tensor] = None) -> dict:
    """Causal-safe operands. Q^ / K^ as fp4all_quantize_qkv (signs: shared Q / K head-dim Hadamard). V: s_k = E4M3(amax_d |V_k| / 6)
    per (b, key, kv head) (floored at 2^-9), v~ = RTN_sat6(V / s_k) with every SF = 1 (Triton rnd="unit"), in the forward's
    transposed layout and the backward's row-wise layout (same codes). beta = log2(s_k) / (softmax_scale log2 e) (raw-score bias),
    u = 1 / s_k, both (b, s, h_k). signs_v: head-dim Hadamard of V before the per-token scale (spreads outlier channels over the
    row the scale covers; the caller un-rotates O and dV and rotates dO)."""
    qq = quantize_fp4_rows(q, apply_hadamard_rotation=signs is not None)
    kq = quantize_fp4_rows(k, apply_hadamard_rotation=signs is not None)
    q4, sfq, k4, sfk = qq["codes"], qq["scale_factors"], kq["codes"], kq["scale_factors"]
    vf = v.float()
    if signs_v is not None:
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        vf = fwht_rows(v.contiguous(), signs_v, dtype=torch.float32)
    # power-of-two s_k = 2^e_k, e_k = clamp(ceil(log2(amax / 6)), -8, 7) (exact bias A_k = e_k + 8 and exact weight
    # u_k = 2^-e_k; the uniform +8 shifts every row's frame / LSE by 8 log2 units and cancels in P^ and Z). The per-key
    # weight tensor and its TMA staging are always on (an E4M3 s_k and non-TMA stagings are not shipped).
    e = torch.ceil(torch.log2((vf.abs().amax(-1) / 6.0).clamp(min=1e-30))).clamp(-8, 7)
    s_k = torch.pow(2.0, e)
    vtil = (vf / s_k[..., None]).contiguous()
    vq = quantize_fp4_blocks_2d(vtil, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False,
                                transposed_copy_layout="batch_head_dim_seq", rounding="unit")
    lg = torch.log2(s_k)
    ops = dict(q4=q4, sfq=sfq, k4=k4, sfk=sfk, v4=vq["codes"], sfv=vq["scale_factors"],
               vt4=vq["transposed_codes"], sfvt=vq["transposed_scale_factors"],
               vtok_s=s_k, vtok_bias=(lg / (softmax_scale * LOG2_E)).contiguous(), vtok_u=(1.0 / s_k).contiguous(),
               vtok_vmax=vtil.norm(dim=-1).amax())  # max ||v~_k||: the backward's dP operand norm (dS bound)
    b_, s_, hk_ = s_k.shape
    A = (e + 8.0).float()
    ops["vtok_A"] = A.contiguous()
    ops["vtok_bias"] = (A / (softmax_scale * LOG2_E)).contiguous()  # the forward's effective raw bias (reference)
    ops["vtok_w"] = torch.cat([A.permute(0, 2, 1).reshape(b_, hk_, s_ // 128, 128),
                               ops["vtok_u"].permute(0, 2, 1).reshape(b_, hk_, s_ // 128, 128)], -1).contiguous()
    # the 256 floats of each key tile as the 1024 bytes of its two SFV atoms (32, 4, 4), in the SFV view's layout
    # per key tile [beta'_k = A_k / (scale log2 e), u_k]: the kernel adds beta' (the backward adds the same float)
    tsf = torch.cat([ops["vtok_bias"].permute(0, 2, 1).reshape(b_, hk_, s_ // 128, 128),
                     ops["vtok_u"].permute(0, 2, 1).reshape(b_, hk_, s_ // 128, 128)], -1).contiguous().float()
    by = tsf.view(torch.uint8).reshape(b_, hk_, s_ // 128, 2, 32, 4, 4).reshape(b_, hk_, 1, s_ // 64, 32, 4, 4)
    ops["vtok_tsf"] = by.contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return ops


def flash_attn_fp4all_fwd(q, k, v, softmax_scale=None, causal=False, ops: Optional[dict] = None,
                          vtok: Optional[bool] = None,
                          repro_out: Optional[torch.Tensor] = None, repro_dbg: Optional[torch.Tensor] = None,
                          p_mix: Optional[str] = None, masks=None, *, specs):
    """-> (out, lse (b, h, s) nats, frame (b, h, s) float32 log2 frame, ops). The Mitchell classifier is the clamped one
    (exact for any scores). ONE forward launch.

    `vtok` (`key-local-v-scale`) selects the causal-safe V quantisation when this function builds the
    operands itself: None = `fp4all_vtok(causal=causal)`, i.e. `causal_safe_v="auto"` turns it on for
    every masked kernel, so **a causal caller gets a correct kernel by default**. Before this the
    function always called `fp4all_quantize_qkv`, so `flash_attn_fp4all_fwd(q, k, v, causal=True)`
    could only run by taking the `causal_safe_v="never"` opt-out and being knowingly leaky --
    `causal-4bit-pv-unguarded`, path 2. Passing `ops` keeps full control, and an `ops` without the
    per-key tensors under a mask is refused here rather than deep inside the forward driver.
    specs: the call's plan.specs (lowprec.specs.PlanSpecs); its FwdKernelSpec is the all-FP4 forward's (fp4all).
    masks: lp_masks.LPMasks carrying aligned varlen documents only (b = 1 tensors; the operands must be built with the
    same documents): the forward's varlen path over the packed keys / queries, out / lse / frame of the (1, total) batch.
    """
    if not specs.fwd.fp4all:
        raise ValueError("flash_attn_fp4all_fwd runs the all-FP4 forward: specs.fwd.fp4all must be True (the plan sets it "
                         "on the all-4-bit route and option A)")
    b, s, h, d = q.shape
    _v4p = ops is not None and "sfv_rows" in ops   # (cov5: the packed rows of compact maps)
    assert d == 128 and (s % 128 == 0 or _v4p), (q.shape,)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if ops is None:
        # the one place this entry resolves the causal-safe route when it builds the operands itself
        mode = fp4all_vtok_mode(causal=causal, qspec=specs.quant) if vtok is None else ("full" if vtok else "off")
        vtok = mode != "off"
        if mode == "diag":
            ops = fp4all_quantize_qkv_diag(q, k, v, softmax_scale, qspec=specs.quant, bspec=specs.bwdq)
        elif mode == "full":
            ops = fp4all_quantize_qkv_vtok(q, k, v, softmax_scale)
        else:
            ops = fp4all_quantize_qkv(q, k, v)
    elif vtok is None:
        vtok = fp4all_vtok_mode(causal=causal, qspec=specs.quant) != "off"
    diag = bool(ops.get("vtok_diag", False))
    assert not diag or causal, "vtok_diag operands are causal-only: dense attention has no diagonal tile"
    if vtok and "vtok_bias" not in ops:
        raise ValueError(
            "key-local-v-scale is required here (causal / masked 4-bit PV) but `ops` carries no per-key V scale. "
            "Build the operands with fp4all_quantize_qkv_vtok(q, k, v, softmax_scale), or pass causal_safety='off' "
            "(QuantSpec.causal_safe_v='never') to reproduce the leaky key-grouped V."
        )
    kw = dict(softmax_scale=softmax_scale, mSFQ=ops["sfq"], mSFK=ops["sfk"], mSFV=ops["sfvt"], return_lse=True,
              return_pmax=True, out=torch.empty(b, s, h, d, dtype=q.dtype, device=q.device),
              vtok_bias=None if "vtok_w" in ops else ops.get("vtok_bias"),
              vtok_u=None if "vtok_w" in ops else ops.get("vtok_u"),
              vtok_w=None if "vtok_tsf" in ops else ops.get("vtok_w"), vtok_tsf=ops.get("vtok_tsf"),
              **_vtok_diag_fwd_kw(diag))
    # This entry point *is* the all-FP4 forward: specs.fwd carries fp4all (and option A's reproduce-forward-p settings)
    if repro_out is not None:
        kw.update(repro_out=repro_out, repro_dbg=repro_dbg)   # `reproduce-forward-p` export (pv4_repro.py)
    elif repro_dbg is not None:
        kw.update(repro_dbg=repro_dbg)   # debug build: every code / SF byte (the all-FP4 backward's P^ gate)
    if p_mix is not None:
        kw.update(p_mix=p_mix)
    if masks is not None and masks.docs is not None:
        # aligned varlen documents: the packed (total, h, d) operands (the b = 1 batch without its batch mode), the
        # K-major V keeps its (1, h_k, d, keys / 2) buffer; lse / frame come back (h, total_q)
        from flash_attn.cute.fp4_bwd_autograd import _fwd_masks
        assert b == 1 and masks.docs.aligned and not masks.local and not masks.any_mod and masks.mask_mod is None
        mk = _fwd_masks(masks, causal, q, k, specs.host)
        mk.pop("causal", None)
        if not causal and specs.host.varlen_fwd_tma_o and "varlen_tma_o" in mk:
            # the batched TMA O epilogue on dense documents too: the all-4-bit forward measured 112.8 -> 108.8 us with it
            # (eq16x1024 h16, profiler; the 8-bit route keeps it causal-only, step 3)
            mk["varlen_tma_o"] = True
        kw["out"] = kw["out"][0]
        vt = ops["vt4"]
        if _v4p:
            # (cov5) the packed rows: the NVFP4 S atoms from the dense SF words, V^T from the key-parity copies (the
            # load warp realigns them, flash_attn.cute.v4_packed); v only shapes the kernel's views
            assert "kv_cu_seqlens_packed" in mk, "cov5: the packed operands need the packed doc_maps forward"
            from flash_attn.cute.interface_bwd_fp4 import sfd_placeholder
            _ph = sfd_placeholder(q.device)
            vtb = ops.get("vt4b_packed", vt)
            sfvb = ops.get("sfvb_tiles", ops["sfv_rows"])
            kw["sfd_rows"] = [_ph, _ph, _ph, ops["sfq_rows"], ops["sfk_rows"], vt, ops["sfv_rows"], vtb, sfvb]
            vt = vt[0:1]
        out, lse, frame = _flash_attn_fwd(ops["q4"].view(E2)[0], ops["k4"].view(E2)[0], vt.view(E2),
                                          causal=causal, spec=specs.fwd, host_spec=specs.host, quant_spec=specs.quant,
                                          **kw, **mk)
        return out.unsqueeze(0), lse.unsqueeze(0), frame.unsqueeze(0), ops
    out, lse, frame = _flash_attn_fwd(ops["q4"].view(E2), ops["k4"].view(E2), ops["vt4"].view(E2), causal=causal,
                                      spec=specs.fwd, host_spec=specs.host, quant_spec=specs.quant, **kw)
    return out, lse, frame, ops


def _vtok_diag_fwd_kw(diag: bool) -> dict:
    """The forward driver's `vtok_diag` argument, passed only when on, so the default call is byte-for-byte the old one."""
    return {"vtok_diag": True} if diag else {}


def _flash_attn_bwd_fp4q(
    q, k, v, out, dout, lse, frame, ops: dict, softmax_scale: Optional[float] = None, causal: bool = False,
    do_rnd: Optional[str] = None, ds_sr: Optional[bool] = None, seed: int = 0, signs: Optional[torch.Tensor] = None,
    return_accum: bool = False, dkv: Optional[str] = None, p_mix: Optional[str] = None,
    code_dbg: Optional[torch.Tensor] = None, docs=None, *, spec, qspec, fspec,
):
    """All-FP4 backward. q, k, v (b, s, h, 128) unrotated; out, dout in the frame the forward kernel ran in (rotated when
    signs are given: out = the kernel's O, dout = rotated dO); lse (b, h, s) nats; frame (b, h, s) from
    flash_attn_fp4all_fwd; ops its operands. signs: the forward's head-dim Hadamard, applied to Q^T / K^T inside their
    quantiser reads. Returns (dq, dk, dv) in the kernel frame (rotated when signs are given)."""
    prep = fp4q_prepare(q, k, v, out, dout, lse, frame, ops, softmax_scale, causal, do_rnd, ds_sr, seed, signs, dkv,
                        p_mix=p_mix, code_dbg=code_dbg, docs=docs, spec=spec, qspec=qspec, fspec=fspec)
    dq, dk, dv = fp4q_run(prep)
    if return_accum:
        return dq, dk, dv, prep["info"]
    return dq, dk, dv


def fp4all_dkv_mode(bspec) -> str:
    """BwdQSpec.dkv (options fp4_backward_dkdv): "e4m3" (default: S, dP, dQ NVFP4; dK, dV per-tensor E4M3) or "fp4" (every
    backward matmul NVFP4)."""
    return _bwdq_spec(bspec).dkv


def _trans_placeholder(b, s, h, d, dev):
    """E2M1 transposed operand + SF view of the quantiser's shapes, unused by the dkv8 kernel (it keeps the signature):
    torch.empty -- the kernel never reads them, so no zero-fill launch."""
    x = torch.empty(b, d, h, s // 2, dtype=torch.uint8, device=dev)
    sf = torch.empty(b, h, 1, s // 64, 32, 4, 4, dtype=torch.uint8, device=dev).permute(4, 5, 2, 6, 3, 1, 0)
    return dict(xt4=x, sft=sf)


def fp4all_ds_scale_mode(causal: bool, dkv8: bool, *, qspec) -> str:
    """QuantSpec.fp4q_ds_scale, read once: how the backward sizes its score-gradient (dS) scale.
      "global"  one multiplier for the whole tensor from max over rows of a bound built from max over ALL keys of
                ||V|| and from max |dO| over all queries -- whole-tensor statistics, so a FUTURE row moves every query's
                dS^ quantisation (`per-tensor-ds-scale-is-a-global-channel`: 2.0e-01 to 2.6e-01 into dQ).
      "causal"  no data-dependent global multiplier at all: each query row q gets its own power of two F_q from a bound
                that sees only what row q may see (its own dO, D and P_max, and the PREFIX max of the dP operand's key
                norms up to its last visible key), applied by scaling dO row q before its quantiser (so dP' and D'
                carry it) and undone exactly on dQ row q and on Q row q in dK's B operand.
      "auto"    (default) "causal" for a causal dK/dV-in-E4M3 backward, "global" otherwise (dense has no causal
                restriction and keeps its numbers bitwise; the all-FP4 dK/dV route quantises Q^T along queries, where a
                per-row factor would sit inside a scale group -- not built)."""
    m = _quant_spec(qspec).fp4q_ds_scale
    if m not in ("auto", "global", "causal"):
        raise ValueError(f"QuantSpec.fp4q_ds_scale must be 'auto', 'global' or 'causal', got {m!r}")
    if m == "auto":
        return "causal" if (causal and dkv8) else "global"
    if m == "causal" and not (causal and dkv8):
        raise NotImplementedError("the causal dS scale (QuantSpec.fp4q_ds_scale='causal') needs a causal backward with E4M3 dK / dV")
    return m


def _key_norm_prefix_max(vn: torch.Tensor, s_q: int, r_gqa: int) -> torch.Tensor:
    """vn (b, s_k, h_k) per-key norms of the dP operand -> (b, h, s_q): for each query row, the max over the keys it can
    see under the bottom-right causal alignment (k <= q + s_k - s_q). Rows that see no key get 0."""
    b, s_k, h_k = vn.shape
    cm = torch.cummax(vn.float(), dim=1).values                                   # (b, s_k, h_k)
    idx = torch.arange(s_q, device=vn.device) + (s_k - s_q)
    vis = (idx >= 0)
    g = cm[:, idx.clamp(min=0, max=s_k - 1)] * vis[None, :, None]                 # (b, s_q, h_k)
    return g.repeat_interleave(r_gqa, dim=2).permute(0, 2, 1).contiguous()       # (b, h, s_q)


def pow2_floor_bits(x: torch.Tensor, cap: Optional[float] = None) -> torch.Tensor:
    """2^min(floor(log2 x), cap) for float32 x > 0 (normal, or +inf), exactly: the exponent field of x (quant.fp4
    pow2_floor_as_torch / fp4q_row_scales take the same bits). Integer ops only (no rounded log2)."""
    e = ((x.float().view(torch.int32) >> 23) & 0xFF) - 127
    if cap is not None:
        e = e.clamp(max=int(math.floor(cap)))
    return ((e + 127) << 23).view(torch.float32)


def _pow2_floor_t(x: torch.Tensor) -> torch.Tensor:
    return pow2_floor_bits(x)


def _diag_fwd_tiles(fspec) -> int:
    """`diagonal-pairs-are-the-forwards-cta`: the forward's diagonal granularity in the backward's 128-query tiles.

    The diagonal forward classifies a step as masked per CTA (flash_fwd_sm100_fp4: n_diag_min from the CTA's m_block),
    and one CTA spans q_tiles x 128 rows -- q_tiles = FwdKernelSpec.q_stage (2 at head_dim 128
    without SplitKV, which the diagonal route refuses). For the default that is 256 queries: the forward biases key tile
    2j, and reads its diagonal payload, for query tiles 2j AND 2j + 1, where the backward used to test its own 128-query
    tile and saw key tile 2j as unmasked for query tile 2j + 1. The two passes then computed different P^ and dP
    operands on that pair; the D the backward takes from the forward's O stopped matching its own sum_k P^ dP (4.0e-03
    of |O||dO| on mixed rows vs 2.4e-04), sum_k dS_qk != 0 on every odd query tile, and dQ picked up a component along
    the mean key: dQ norm 4.17x exact at training step 1250 (the full fix 1.64x).

    `gqa-diag-tiles-assume-packing`: the rows are whole 128-query tiles of ONE head for any GQA ratio. The block-scaled
    forward never packs GQA (interface_fwd_fp4: pack_gqa = False on use_blockscaled_impl), so dividing the CTA's rows by
    the ratio (as this function used to) gave 1 instead of 2 under GQA, and the backward reproduced the odd query tiles'
    P^ without the forward's key-local bias on key tile 2j (dQ of GQA != dQ of the same call on K / V repeated to MHA,
    on exactly the odd tiles; tests/cute/test_fp4_gqa.py::test_fp4all_diag_gqa_equals_repeated_mha).

    Mirrors the forward's resolution (the same FwdKernelSpec value)."""
    # FwdKernelSpec.q_stage, None = 2 (head_dim 128, no SplitKV: FwdKernelSpec.rule_q_stage(False, 128))
    if fspec is None:
        raise TypeError("_diag_fwd_tiles: the forward's FwdKernelSpec is required (the call's plan.specs.fwd)")
    return int(fspec.q_stage if fspec.q_stage is not None else 2)


def _div_f32(t: torch.Tensor, p):
    """t / p for a float32 tensor t, as torch's CUDA division computes it for a Python-float p -- t * (1 / p) -- also
    when p is a float32 device scalar (no host read): t * reciprocal(p), the reciprocal correctly rounded in float32
    (quant.fp4 dense_bound_row's rcp.rn)."""
    return t * torch.reciprocal(p.float()) if isinstance(p, torch.Tensor) else t / p


_ROUNDING = {"rtn": "nearest", "sr": "stochastic", "eden": "ms_eden"}


def _quantize_do(dout, global_scale, dkv8: bool, do_rnd: str, seed: int, out):
    """dO's NVFP4 1 x 16 row copy (and, for the fp4 dK / dV, its transposed copy in natural query order) and D = sum_d O dO^
    from the same read (quant.fp4). Returns the old builder's names: x4, sf[, xt4, sft], D."""
    r = quantize_fp4_rows(dout, make_transposed_copy=not dkv8, transposed_copy_in_kernel_key_order=False,
                          global_scale=global_scale, rounding=_ROUNDING[do_rnd], stochastic_rounding_seed=seed,
                          compute_d_with_output=out)
    res = dict(x4=r["codes"], sf=r["scale_factors"], D=r["d"])
    if not dkv8:
        res.update(xt4=r["transposed_codes"], sft=r["transposed_scale_factors"])
    return res


def _doc_major_to_aligned(t: torch.Tensor, docs, s_k: int, neutral: float) -> torch.Tensor:
    """(cov6) A document-major per-key table (n_docs, h_k, L) (quant.fp4_diagonal documents: the forward's vtok bias /
    weight) at the aligned keys: (1, h_k, s_k), key j of document i at slot j - start_i; keys outside every document's
    slot (the sentinel's tiles, slots past L) the neutral value. GPU gathers, no host read."""
    n_docs, hk, L = t.shape
    j = torch.arange(s_k, device=t.device, dtype=torch.int64)
    doc = docs.tile_doc(128).long()[j // 128]
    real = doc < n_docs
    docc = doc.clamp(max=n_docs - 1)
    loc = j - docs.doc_tab[1].long()[docc]
    ok = real & (loc >= 0) & (loc < L)
    idx = docc * L + loc.clamp(0, L - 1)
    g = t.permute(1, 0, 2).reshape(hk, n_docs * L)[:, idx]
    return torch.where(ok[None], g, torch.full_like(g, neutral)).unsqueeze(0).contiguous()


def fp4q_prepare(q, k, v, out, dout, lse, frame, ops: dict, softmax_scale: Optional[float] = None, causal: bool = False,
                 do_rnd: Optional[str] = None, ds_sr: Optional[bool] = None, seed: int = 0, signs: Optional[torch.Tensor] = None,
                 dkv: Optional[str] = None, deterministic: bool = False, p_mix: Optional[str] = None,
                 code_dbg: Optional[torch.Tensor] = None, docs=None, *, spec, qspec, fspec):
    """Everything of the all-FP4 backward outside the kernel: the Triton quantiser reads (Q^T, K^T, dO^, dO^T, D), the host
    scalars (two host reads: max |dO| and the dS bound), the output buffers and the compiled kernel. fp4q_run(prep) is the
    kernel call (graph-capturable). spec / qspec / fspec: the call's plan.specs.bwdq (the kernel's), .quant (the host's)
    and .fwd (the forward's: its Q-stage count is the diagonal granularity the backward reproduces), required.
    docs (lp_varlen.LPDocs, aligned; b = 1): the documents of the (1, total, h, d) tensors -- every statistic of the
    backward (the dO launch's E4M3 amax slots, the E4M3 Q / dO scales, the kernel's [dk, dv, ds_mul]) per document, the
    kernel's tile ranges / masks / diagonal classification in each document's coordinates."""
    from flash_attn.cute.flash_bwd_sm100_fp4q import FlashAttentionBackwardSm100FP4Q
    if docs is not None and getattr(docs, "doc_maps_packed", False):
        return fp4q_prepare_packed(q, k, v, out, dout, lse, frame, ops, softmax_scale=softmax_scale, causal=causal,
                                   do_rnd=do_rnd, ds_sr=ds_sr, seed=seed, signs=signs, dkv=dkv,
                                   deterministic=deterministic, p_mix=p_mix, code_dbg=code_dbg, docs=docs, spec=spec,
                                   qspec=qspec, fspec=fspec)
    spec = _bwdq_spec(spec)
    qs = _quant_spec(qspec)
    # seed: an int, or the step's SR counter (an int64 (1,) device tensor, sr_counter): the quantisers and the kernel then
    # derive the seed on the GPU, so a CUDA-graph replay of the step draws a new dither
    seed_t = seed if isinstance(seed, torch.Tensor) else None

    b, s_q, h, d = q.shape
    _, s_k, h_k, _ = k.shape
    assert d == 128 and s_q % 128 == 0 and s_k % 128 == 0 and h % h_k == 0
    r_gqa = h // h_k
    if docs is not None:
        assert b == 1 and docs.aligned, "documents: the aligned (1, total, h, d) layout"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    # the caller's explicit values (the all-4-bit route passes both)
    do_rnd = do_rnd if do_rnd is not None else "sr"
    ds_sr = ds_sr if ds_sr is not None else True
    dkv = dkv if dkv is not None else fp4all_dkv_mode(spec)
    dkv8 = dkv == "e4m3"
    dev = q.device
    dout = dout.contiguous()
    # backward-only operands (one read each): K^T along keys (C3 order), dO row-wise + D; fp4 dK / dV: Q^T, dO^T;
    # dkv8: per-tensor E4M3 Q and dO (in the kernel frame: rotated when signs are given)
    smax = {"rtn": 6.0, "sr": SR_SMAX, "eden": EDEN_SMAX}[do_rnd]
    ds_scale_mode = fp4all_ds_scale_mode(causal, dkv8, qspec=qs)
    vtok = "vtok_bias" in ops
    # diagonal-only: the per-key bias / weight apply on the diagonal pairs only, and the dP operand is plain V's
    # key-grouped payload on every other pair and the diagonal payload (V / r_k) on the diagonal pair, so the dS bound
    # takes the larger of the two row-norm maxima.
    vtok_diag = bool(ops.get("vtok_diag", False)) and vtok
    kmode = _diag_k_check(r_gqa, s_q, s_k, qspec=qs, bspec=spec) if vtok_diag else "off"
    diag_k, diag_kr_all, diag_fold = kmode == "cut", kmode == "all", kmode == "fold"
    # no CPU read of any scale on the diagonal route (fused path): the per-tensor E4M3 scales of dK / dV's operands and
    # the kernel's dk_scale / dv_scale stay on the device; the global dS-scale mode, whose max|dO| and dS bound are host
    # reads, is refused there (the torch builder keeps it as the reference)
    # device E4M3 scales on every dkv8 route with the causal dS scale (bitwise the host path, quant.fp8); the torch
    # builder of the diagonal route keeps the host path as its reference
    devscl = (dkv8 and ds_scale_mode == "causal" and signs is None and not spec.dq8
              )
    diag_fused = vtok_diag and dkv8 and signs is None
    if diag_fused and spec.dq8:
        raise NotImplementedError("the diagonal route reads no scale on the host, and the E4M3 dQ operand (dq_operand='e4m3') quantises K per "
                                  "tensor with a host scale; use the E2M1 dQ (the default)")
    if diag_fused and ds_scale_mode != "causal":
        raise NotImplementedError(
            f"the diagonal route reads no scale on the host, and the dS scale mode {ds_scale_mode!r} needs max|dO| and "
            f"the global dS bound there; use the causal dS scale (the default)")
    scl = None
    kfused = kmode != "off" and signs is None
    kf_ = None
    if kfused:
        # the K-channel operands came from the forward's quantize_fp4_causal_diagonal launch (no backward launch): the
        # plain K^T and its second-half SF copy, the key-local K^T, r_k and the doubled bias / weight in the kernel's
        # layout (fold: r_k in the weight, the dP operands' SFs)
        kf_ = {n[len("kdiag_"):]: t for n, t in ops.items() if n.startswith("kdiag_")}
        if "xt4b" not in kf_:
            raise ValueError("the diagonal route's backward needs the forward's K-channel operands (ops['kdiag_*'] from "
                             "fp4all_quantize_qkv_diag)")
        kd = dict(xt4=kf_["xt4b"], sft=kf_["sftb"], kr=kf_["kr"][..., s_k:].permute(0, 2, 1), krv=kf_["kr"])
        if diag_k:
            kt = dict(xt4=kf_["xt4"], sft=kf_["sft"], sft1=kf_["sft1"])
        else:   # "all" / "fold": the key-local K^T is the dQ operand on every pair
            kt = dict(xt4=kf_["xt4b"], sft=kf_["sftb"], sft1=kf_["sftb"])
    elif "kt4" in ops and signs is None:
        kt = dict(xt4=ops["kt4"], sft=ops["sfkt"], sft1=ops["sfkt1"])   # the forward's K read made them
        kd = fp4all_quantize_k_diag(k) if kmode != "off" else None
    else:
        ktq = quantize_fp4_rows(k, make_row_copy=False, make_transposed_copy=True, apply_hadamard_rotation=signs is not None)
        kt = dict(xt4=ktq["transposed_codes"], sft=ktq["transposed_scale_factors"])
        kd = fp4all_quantize_k_diag(k) if kmode != "off" else None
    F_row = None
    # the fused dO launch (quant.fp4 causal_row_scale): the forward saved every per-key / per-query input of the causal dS
    # scale (vmax_q / rmax_q, max |Q| per row, the zeroed max slots), so ONE launch reads dO and O and writes the NVFP4
    # dO^ of dO F_q, D', F_q, uinv, the E4M3 amaxes and the zeroed dQ accumulator -- bitwise the torch chain below
    fused_do = (ds_scale_mode == "causal" and devscl and "query_row_amax" in ops and "vmax_q" in ops
                and (not diag_fold or "rmax_q" in ops) and lse.dtype == torch.float32 and lse.is_contiguous()
                and frame.dtype == torch.float32 and frame.is_contiguous() and lse.shape == (b, h, s_q))
    qrow_ = qs.fp4q_qrow if qs.fp4q_qrow is not None else qs.rule_fp4q_qrow(causal)
    # the dense fused path (global dS multiplier + row boost): dO amax launch -> dO launch (NVFP4 dO / s_dO, D, D', the
    # per-row bound, uinv, max |dO| per row, the global bound by its last tile, the dQ accumulator zeroed) -> the row-boost
    # launch (r_q, uinv r_q, the E4M3 amaxes) -> the E4M3 Q / r, dO / r launch with the device scales: 4 launches,
    # bitwise the torch path below
    dense_fused = (ds_scale_mode == "global" and dkv8 and qrow_ and signs is None and not vtok
                   and not spec.dq8 and "v_norm2_max" in ops
                   and "query_row_amax" in ops and "kt4" in ops and lse.dtype == torch.float32 and lse.is_contiguous()
                   and frame.dtype == torch.float32 and frame.is_contiguous() and lse.shape == (b, h, s_q))
    dq_accum = None
    uinv = None
    dense_ = None
    if docs is not None and not ((fused_do and not qs.per_tensor_pow2) or dense_fused):
        raise NotImplementedError(
            "all-4-bit backward on documents: built for the causal dS scale's fused dO launch (the default causal "
            "route); this configuration (dense / global dS scale, power_of_two_scale, Hadamard, torch builders) is not")
    if dense_fused:
        from flash_attn.cute.quant.fp4 import DenseRowBound
        from flash_attn.cute.fp4q_row_scales import dense_row_boost
        from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_dense
        dq_accum = torch.empty(b, h, s_q, d, dtype=torch.float16, device=dev)
        # documents: every statistic per document of the row's tile (s_dO, max ||V||^2 from the forward, the dS bound,
        # the E4M3 amaxes), the padding rows zero rows
        dkw = (dict(row_documents=docs.tile_doc_q, n_documents=docs.b + 1, row_end=docs.row_end_tab_q)
               if docs is not None else {})
        r0 = quantize_fp4_rows(dout, make_row_copy=False, output_tensor_max="abs",
                               fp4_global_scale_divisor=smax * (256.0 if do_rnd == "eden" else 448.0), **dkw)
        so_do = r0["fp4_global_scale"]                      # (1,) float32: div32(max |dO|, smax 448).clamp_min(1e-30)
        r1 = quantize_fp4_rows(dout, global_scale=so_do, rounding=_ROUNDING[do_rnd], stochastic_rounding_seed=seed,
                               compute_d_with_output=out,
                               dense_row_bound=DenseRowBound(lse=lse, frame=frame, value_norm2_max=ops["v_norm2_max"],
                                                             lse_log2_offset=LSE_LOG2_OFFSET, zero_accumulator=dq_accum),
                               **dkw)
        dq_ = dict(x4=r1["codes"], sf=r1["scale_factors"], D=r1["d"])
        cap_ = float(qs.fp4q_qrow_cap)
        r_q_, uinv, amax_ = dense_row_boost(r1["bound_rows"], r1["tensor_max"], r1["uinv"], ops["query_row_amax"],
                                            r1["row_amax"], cap_,
                                            **({"tile_doc": docs.tile_doc_q} if docs is not None else {}))
        if docs is not None:
            from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_dense_docs
            q8_, do8_, scl4_, dqs_ = e4m3_q_do_dense_docs(q.contiguous(), dout, r_q_, softmax_scale, amax_, so_do,
                                                          r1["tensor_max"], docs.tile_doc_q)
            dense_ = dict(r_q=r_q_, q8=q8_, do8=do8_, scl=scl4_, dq_scale=dqs_, Dp=r1["d_scaled"],
                          bound_rows=r1["bound_rows"], bound=r1["tensor_max"])
        else:
            q8_, do8_, scl4_ = e4m3_q_do_dense(q.contiguous(), dout, r_q_, softmax_scale, amax_, so_do, r1["tensor_max"])
            dense_ = dict(r_q=r_q_, q8=q8_, do8=do8_, scl=scl4_[:3], dq_scale=scl4_[3:4], Dp=r1["d_scaled"],
                          bound_rows=r1["bound_rows"], bound=r1["tensor_max"])
    elif fused_do:
        dq_accum = torch.empty(b, h, s_q, d, dtype=torch.float16, device=dev)
        crs = CausalRowScale(lse=lse, frame=frame, key_norm_prefix_max=ops["vmax_q"],
                             query_row_amax=ops["query_row_amax"], max_slots=ops["backward_amax_slots"],
                             top=smax * (256.0 if do_rnd == "eden" else 448.0), lse_log2_offset=LSE_LOG2_OFFSET,
                             key_factor_prefix_max=ops["rmax_q"] if diag_fold else None, zero_accumulator=dq_accum)
        r_ = quantize_fp4_rows(dout, make_transposed_copy=False, rounding=_ROUNDING[do_rnd], stochastic_rounding_seed=seed,
                               compute_d_with_output=out, causal_row_scale=crs,
                               **(dict(row_documents=docs.tile_doc_q, row_end=docs.row_end_tab_q) if docs is not None
                                  else {}))
        dq_ = dict(x4=r_["codes"], sf=r_["scale_factors"], D=r_["d"])
        F_row, uinv = r_["row_factor"], r_["uinv"]
        F_bshd = F_row.permute(0, 2, 1)[..., None]    # a view (dQ / dK row factors; the postprocess reads F_row)
        so_do = 1.0
    elif ds_scale_mode == "causal":
        lse2 = lse.float() * LOG2_E
        # P_max bound per query row (<= 2^(F + thr - lse2)), row-local
        pmax_b = torch.pow(2.0, (frame.float() + 8.0 - lse2).clamp(max=0.0))
        do_n = dout.float().norm(dim=-1).permute(0, 2, 1)
        # per-key norms of the dP operand: V (leaky), v~ = V / s_k (full key-local fix), max(V, V / r_k) (diagonal-only)
        if "vmax_q" in ops and (not diag_fold or "rmax_q" in ops):
            # the diagonal route: the per-query-row prefix maxima came from the forward's quantize_fp4_causal_diagonal
            # launch (its look-back scan), bitwise the torch chain below
            vmax_q = ops["vmax_q"]
            rmax_q = ops.get("rmax_q")
        else:
            vn = v.float().norm(dim=-1)                                              # (b, s_k, h_k)
            if vtok_diag:
                vn = torch.maximum(vn, vn * ops["vtok_u"].float())
            elif vtok:
                vn = vn / ops["vtok_s"].float()
            if diag_fold:   # dS' = r_k dS: the dP operand rows are V r_k, and the D term carries r_k too
                r_bsh = kd["kr"].float()                                             # (b, s_k, h_k)
                vn = vn * r_bsh
            vmax_q = _key_norm_prefix_max(vn, s_q, r_gqa)                            # (b, h, s_q), causal
            rmax_q = _key_norm_prefix_max(r_bsh, s_q, r_gqa) if diag_fold else None  # causal prefix max of r_k
        d_est = (out.float() * dout.float()).sum(-1).permute(0, 2, 1)                # row-local D estimate for the bound
        if diag_fold:
            d_est = d_est * rmax_q
        bound_rows = pmax_b * (do_n * vmax_q + d_est.abs())                          # |dS_q| bound, natural units
        # F_q: the dS' multiplier of row q, a power of two with |dS'_q| <= 448 (the E4M3 dS^ rows of dK), and small
        # enough that dO row q x F_q stays inside its quantiser's E4M3 block-scale range. Every input is row q's own.
        dmax_q = dout.float().abs().amax(-1).permute(0, 2, 1)
        top = smax * (256.0 if do_rnd == "eden" else 448.0)
        f = torch.minimum(448.0 / bound_rows.clamp(min=1e-30), top / dmax_q.clamp(min=1e-30))
        f = torch.where((bound_rows > 0) & (dmax_q > 0), f, torch.ones_like(f))
        F_row = _pow2_floor_t(f.clamp(min=2.0 ** -100, max=2.0 ** 100))               # (b, h, s_q) powers of two
        F_bshd = F_row.permute(0, 2, 1)[..., None].float()                           # (b, s_q, h, 1)
        so_do = 1.0
        dout_q = (dout.float() * F_bshd).to(dout.dtype)                              # exact: a power of two per row
        dq_ = _quantize_do(dout_q, None, dkv8, do_rnd, seed, out)
    if dense_fused or F_row is not None:
        pass
    else:
        lse2 = lse.float() * LOG2_E
        pmax_b = torch.pow(2.0, (frame.float() + 8.0 - lse2).clamp(max=0.0))
        do_n = dout.float().norm(dim=-1).permute(0, 2, 1)
        # max(amax / (smax 256|448), 1e-30) in float32 on the GPU (div.rn), no read
        from flash_attn.cute.device_scales import div32
        so_do = div32(dout.abs().amax(), smax * (256.0 if do_rnd == "eden" else 448.0)).clamp_min(1e-30)
        dq_ = _quantize_do(dout, so_do.reshape(1), dkv8, do_rnd, seed, out)
    q8 = do8 = None
    if dkv8:
        qt = _trans_placeholder(b, s_q, h, d, dev)
        dq_.update(_trans_placeholder(b, s_q, h, d, dev))
    else:
        qtq = quantize_fp4_rows(q, make_row_copy=False, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False,
                                apply_hadamard_rotation=signs is not None)
        qt = dict(xt4=qtq["transposed_codes"], sft=qtq["transposed_scale_factors"])
    so_v = 1.0
    D = dq_["D"]  # natural units (of dO x F_q in the causal mode: exactly the D' the kernel pairs with dP')
    if uinv is None:
        uinv = torch.where(torch.isfinite(lse2), torch.pow(2.0, (frame.float() - lse2 - LSE_LOG2_OFFSET).clamp(max=1000)),
                           torch.zeros_like(lse2))
    # dS' = dS / (s_dO s_V) is multiplied by ds_mul before the 16x16 E4M3 quantiser: bound max |dS| by
    # P_max (<= 2^(F + thr - lse2)) x (||dO_q|| max ||V|| + |D_q|)
    if vtok_diag and F_row is None:
        vmb = ops.get("vtok_vmax_b")
        if vmb is None:   # the fused builder skips it: ||V_k / r_k|| = ||V_k|| u_k
            vmb = (v.float().norm(dim=-1) * ops["vtok_u"].float()).amax()
        v_max = torch.maximum(v.float().norm(dim=-1).amax(), vmb)  # device scalars: no host read
    elif vtok_diag:
        v_max = None       # the causal dS scale bounds per row (vn above); no global bound, no host read
    elif vtok:
        v_max = ops["vtok_vmax"]                 # vtok: dP multiplies v~ = V / s_k on every pair
    elif dense_fused:
        v_max = None                             # the dO launch read sqrt(max ||V||^2) from the forward's V quantiser
    else:
        v_max = v.float().norm(dim=-1).amax()
    if dense_fused:
        bound_q, bound = dense_["bound_rows"], dense_["bound"]
    elif F_row is None:
        bound_q = _div_f32(pmax_b * (do_n * v_max + D.abs()), so_do * so_v)   # (b, h, s_q): the per-row bound, before its maximum
        bound = bound_q.amax()  # a float32 device scalar (no host read)
    else:
        bound_q = bound = None   # the causal dS scale: ds_mul = 1, the global bound is never read (no host sync)
    # NOTE: this is the P_max BOUND, and in FP8 the equivalent measured 4.7x loose and cost
    # +0.029 of val loss against an exact amax (measured on the 8-bit route). So the
    # obvious question is whether part of FP4's +0.0172 is the same penalty rather than a
    # property of four bits.
    #
    # Substituting an exact amax here does NOT answer it: this path runs
    # Direct-P with the `frame` representation, so its dS' is not the plain P (dP - D) that
    # function measures, and swapping the two NaNs at step 500 on a unit mismatch. Answering it
    # needs the estimate expressed in this path's units.
    #
    # There may be little to find. Unlike FP8's per-tensor dS, this path's dS carries per-16 E4M3
    # block scales, which rescale locally and absorb most of a globally loose value -- and
    # consistent with that, forcing the tight scale moved the gradient cosines by 0.0002
    # (dQ 0.9514 -> 0.9516, dK 0.9670 -> 0.9678) where FP8's equivalent moved them by 0.02.
    r_bshd = None
    k8, sk8 = None, 1.0
    dq8 = dkv8 and spec.dq8
    # QuantSpec.fp4q_qrow (default on for non-causal attention): the per-query-row pow2
    # boost under the global dS multiplier -- dense A4 dQ 0.701 -> 0.946. Causal keeps "0" (its default dS scale is the
    # row-local causal mode, which subsumes it).
    qrow = qs.fp4q_qrow if qs.fp4q_qrow is not None else qs.rule_fp4q_qrow(causal)
    if F_row is not None:
        assert not qrow, (
            "QuantSpec.fp4q_qrow is subsumed by the causal dS scale (QuantSpec.fp4q_ds_scale): both set a per-row factor")
        # dQ row q and Q row q in dK's B operand carry 1 / F_q; dV's operands are untouched (P and dO unscaled)
        r_bshd = F_bshd
    if dense_fused:
        r_q = dense_["r_q"]
        r_bshd = r_q.permute(0, 2, 1)[..., None]    # a view; the postprocess reads r_q
        q8, do8, scl = dense_["q8"], dense_["do8"], dense_["scl"]
        sq8 = sdo8 = None
    elif dkv8 and qrow:
        # r_q = 2^floor(log2(bound / bound_q)), capped so that the Q^t / dO^ rows stay inside E4M3's normal range
        cap = float(qs.fp4q_qrow_cap)
        # (float32 division, then the exact floor of log2: fp4q_row_scales' arithmetic)
        r_q = pow2_floor_bits((bound / bound_q.clamp(min=1e-30)).clamp(min=1.0), cap)
        r_bshd = r_q.permute(0, 2, 1)[..., None].float()             # (b, s_q, h, 1)
        uinv = uinv * r_q
    if devscl:
        # E4M3(Q / F_q / s_Q), E4M3(dO / s_dO) and scl = [dk_scale = softmax_scale s_Q, dv_scale = s_dO / 448, ds_mul = 1]
        # on the device: bitwise the host path below. fused_do (fuse-all): the amax slots come from the dO launch, so
        # only the quantiser launch remains (fp4q_diag_triton's Triton _e4_quant; to move into quant.fp8's per-tensor
        # quantiser as a follow-up); otherwise quant.fp8 in one launch.
        if docs is not None:   # per document: the dO launch's per-document slots, an (n_docs + 1, 3) scl table
            from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_device_docs
            q8, do8, scl = e4m3_q_do_device_docs(q.contiguous(), dout, F_row, softmax_scale, ops["backward_amax_slots"],
                                                 docs.tile_doc_q)
        elif fused_do and not qs.per_tensor_pow2:   # (power_of_two_scale: the quant.fp8 launch)
            from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_device
            q8, do8, scl = e4m3_q_do_device(q.contiguous(), dout, F_row, softmax_scale, amax=ops["backward_amax_slots"])
        else:
            from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
            r = quantize_fp8_per_tensor([q.contiguous(), dout.contiguous()],
                                        row_divisors=[None if F_row is None else F_row.contiguous(), None],
                                        output_scale_scalars=((0, softmax_scale, 1.0), (1, 1.0, 448.0), (None, 1.0, 1.0)),
                                        power_of_two_scale=qs.per_tensor_pow2)
            (q8, do8), scl = r["codes"], r["scale_scalars"]
        sq8 = sdo8 = None
    if dkv8 and not devscl and not dense_fused:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor

        def e4m3(t):  # per-tensor E4M3, the scale computed on the GPU (a float32 device scalar)
            r = quantize_fp8_per_tensor(t, power_of_two_scale=qs.per_tensor_pow2)
            return r["codes"], r["scales"][0]
        from flash_attn.cute.nvfp4_2d_triton import fwht_rows
        q_e4 = fwht_rows(q, signs) if signs is not None else q.contiguous()
        if F_row is not None:     # causal dS scale: only dK's Q operand carries 1 / F_q (dV multiplies the unscaled P)
            q_e4 = (q_e4.float() / r_bshd).to(q.dtype).contiguous()
            dout_e4 = dout
        elif r_bshd is not None:  # the dK / dV B operands carry the inverse boost of the query they multiply
            q_e4 = (q_e4.float() / r_bshd).to(q.dtype).contiguous()
            dout_e4 = (dout.float() / r_bshd).to(dout.dtype).contiguous()
        else:
            dout_e4 = dout
        q8, sq8 = e4m3(q_e4)
        do8, sdo8 = e4m3(dout_e4)
        if dq8:
            k_e4 = fwht_rows(k, signs) if signs is not None else k.contiguous()
            # the dS^ SMEM tile the dQ GEMM reads (A from SMEM) has its keys in natural order (the C3-permuted TMEM-A
            # variant and a constant-K isolation probe are not shipped)
            kp = k_e4.contiguous()
            # quantise in the (b, s, h, d) layout the fused quantiser is written for -- handing it a transposed tensor makes
            # it read the key count as the head dimension -- and transpose the bytes afterwards
            k8, sk8 = e4m3(kp.contiguous())
            k8 = k8.permute(0, 3, 2, 1).contiguous()   # (b, d, h_kv, s_k): keys contiguous, the K-major B of dQ
    ds_smax = SR_SMAX if ds_sr else 6.0
    from flash_attn.cute.device_scales import div32, rdiv32  # float32 divisions rounded once (div.rn)
    if F_row is not None:
        # causal dS scale: no data-dependent global multiplier; row q's F_q already sits in dP' and D' (via dO row q)
        ds_mul = 1.0
        if devscl:   # the kernel reads them from `scl`
            dk_scale = dv_scale = 1.0
        else:
            dk_scale = softmax_scale * so_do * so_v * sq8 / ds_mul
            dv_scale = div32(sdo8, 448.0)
    elif dense_fused:
        ds_mul = dk_scale = dv_scale = 1.0   # the kernel reads [dk, dv, ds_mul] from `scl`
    elif dkv8:
        # |dS'| <= 448: the E4M3 dS^ rows of dK take dS' as is; the 1x16 E4M3 scales of dQ's E2M1 dS^ absorb the range
        # (float32 device scalars: max(bound, 1e-30) is clamp_min)
        ds_mul = rdiv32(448.0, bound.clamp_min(1e-30))
        dk_scale = softmax_scale * so_do * so_v * sq8 / ds_mul
        dv_scale = div32(sdo8, 448.0)
    else:
        ds_mul = rdiv32(ds_smax * 448.0, bound.clamp_min(1e-30))
        dk_scale = softmax_scale * so_do * so_v / ds_mul
        dv_scale = so_do / float(2 ** 11)
    # dq8: the dQ partials are f16 and reduce-added over every key tile, while both operands now span E4M3's full range,
    # so they are pre-scaled in the kernel by a pow2 sized from the worst case (448 x 448 per key, s_k keys, f16 headroom)
    dq_shift = 1.0
    if dq8:
        dq_shift = 2.0 ** -max(0, math.ceil(math.log2(448.0 * 448.0 * s_k / 16384.0)))
    dq_scale = None if dense_fused else softmax_scale * so_do * so_v * (sk8 / dq_shift if dq8 else 1.0) / ds_mul
    info_scales = {}
    if scl is None and any(isinstance(x_, torch.Tensor) for x_ in (dk_scale, dv_scale, ds_mul)):
        # device scalars (the global dS mode; the per-tensor E4M3 scales of the non-devscl routes): every one is a float32
        # device value (torch float32 ops in the expression's order); the kernel reads [dk, dv, ds_mul] from `scl` and the
        # postprocess reads dq_scale -- no host read
        _d32 = lambda x_: (x_.float() if isinstance(x_, torch.Tensor)  # noqa: E731
                           else torch.full((), float(x_), dtype=torch.float32, device=dev)).reshape(())
        scl = torch.stack([_d32(dk_scale), _d32(dv_scale), _d32(ds_mul)])
        info_scales = dict(dk_scale=dk_scale, dv_scale=dv_scale, ds_mul=ds_mul)  # the values themselves, for `info`
        dk_scale = dv_scale = ds_mul = 1.0
    if dense_fused:
        dq_scale = dense_["dq_scale"]        # (1,) float32: the scale kernel's float32 expression
    Dp = dense_["Dp"] if dense_fused else (D if (not isinstance(so_do, torch.Tensor) and so_do * so_v == 1.0) else _div_f32(D, so_do * so_v)).float().contiguous()
    frame_c = frame.float().contiguous()
    uinv_c = uinv.float().contiguous()
    dq_accum_zeroed = dq_accum is not None      # the fused dO launch zeroed it
    if dq_accum is None:
        dq_accum = torch.zeros(b, h, s_q, d, dtype=torch.float16, device=dev)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # ---- compile key
    # (kernel class, spec.key(): every compile-time setting the constructor reads), the driver's resolved constructor
    # arguments, and the constexprs it settled on (`resolved_key`), which is how the five items below arrive without
    # another list to keep:
    #   directp_clamp        the forward's classifier is always the clamped one (2); the kernel's default.
    #   FP4_PV_RESCALE_THRESHOLD   `lse-offset-frozen-at-import`: once read at import into
    #                        LSE_LOG2_OFFSET and again in the kernel. `self.max_offset` in
    #                        `resolved_key` is the RESOLVED value, so the forward/backward P^
    #                        agreement is auditable from the key.
    #   head_dim             pinned to 128, latent; explicit now.
    #   deterministic        was hardcoded False at the construction site (`fp4q-backward-no-
    #                        deterministic`); a parameter and keyed, still defaulting to False.
    #   dq8                  the resolved `dkv8 and BwdQSpec.dq8` composite.
    # The kernel is constructed before the lookup (pure Python) so both of those can be in the key. The spec is keyed
    # by value (spec.key(), a tuple of plain values), never by identity: equal specs share one compiled kernel.
    head_dim = 128
    fold_epi = diag_fold and spec.post == "epi"
    kern = FlashAttentionBackwardSm100FP4Q(head_dim, is_causal=causal, deterministic=deterministic,
                                           qhead_per_kvhead=r_gqa, p_mix=p_mix,
                                           ds_sr=ds_sr, dkv=dkv, vtok_diag=vtok_diag,
                                           # one diagonal pair per key tile: MHA, offset a whole number of tiles
                                           diag_single=bool(vtok_diag and r_gqa == 1 and (s_k - s_q) % 128 == 0
                                                            and docs is None),
                                           diag_fwd_tiles=_diag_fwd_tiles(fspec) if vtok_diag else 1,
                                           diag_k=diag_k, diag_kr_all=diag_kr_all, diag_fold=diag_fold, spec=spec,
                                           **({"doc_varlen": True} if docs is not None else {}))
    # (spec.key() covers every setting the kernel and this driver read)
    key = (FlashAttentionBackwardSm100FP4Q.__qualname__, spec.key(),
           r_gqa, causal, ds_sr, q.dtype, dkv, vtok, head_dim, deterministic,
           vtok_diag,   # diagonal-only: the P^ reproduction's per-pair bias, a second V tile and the dP operand select
           scl is not None,   # dK / dV epilogue scales (and ds_mul) read from the device
           code_dbg is not None,   # debug build: the reproduced codes dump
           fold_epi,   # bwdk: the K fold's dK / dV row factors in the epilogue (BwdQSpec.post == "epi")
           seed_t is not None,     # the SR seed from the step's device counter
           *((("docs", True),) if docs is not None else ()),   # aligned varlen documents (doc_varlen)
           cutlass.__version__,
           resolved_key(kern, ("max_offset", "dkv8", "dq8", "fp4q_exp2", "e2e_freq",
                               "e2e_poly", "directp_clamp", "directp_window", "directp_delta",
                               "gm_floor", "ds_sr", "deterministic",
                               "qhead_per_kvhead", "mxds", "mxp", "vtok_diag_arg", "diag_single",
                               "diag_fwd_tiles", "diag_k", "diag_kr_all", "diag_fold", "p_mix_key")))
    from flash_attn.cute.interface_fwd_fp4 import _vtok_kernel_view
    vb = _vtok_kernel_view(ops["vtok_bias"]) if (vtok and not (vtok_diag and kfused)) else None
    vu = _vtok_kernel_view(ops["vtok_u"]) if (vtok and not (vtok_diag and kfused)) else None
    if vtok_diag and kfused:
        vb, vu = kf_["vb2"], kf_["vu2"]   # written by the fused pass B (fold: [r | u r])
    elif vtok_diag:
        # diagonal-only: [zero bias | bias] and [unit weight | u] along keys; the kernel's diagonal pairs index the second
        # half, every other pair the neutral first half (flash_bwd_sm100_fp4q, `diagonal-backward-spills-under-dsl48`)
        if docs is not None:
            # (cov6) documents (k_channel 'open'): the forward's tables are document-major (n_docs, h_k, L); the backward
            # indexes them at the aligned key (the fused builder's kr / vb2 / vu2 layout) -- gathered here
            vb, vu = _doc_major_to_aligned(vb, docs, s_k, 0.0), _doc_major_to_aligned(vu, docs, s_k, 1.0)
        vb = torch.cat([torch.zeros_like(vb), vb], dim=-1).contiguous()
        vu = torch.cat([torch.ones_like(vu), vu], dim=-1).contiguous()
    shp = lambda t: tuple(t.shape[:3]) + (d,)
    shpT = lambda t: (b, d, t.shape[2], t.shape[3] * 2)
    args_t = [ops["q4"], ops["k4"], ops["v4"], dq_["x4"], qt["xt4"], kt["xt4"], dq_["xt4"]]
    # K^T SF atom of the d rows 64-127 moved to columns 0, 1 (the second dQ half's own SF tile)
    if "sft1" in kt:
        kt_sf1 = kt["sft1"]   # written by the fused builder
    else:
        kt_sfc = kt["sft"].permute(6, 5, 2, 4, 0, 1, 3)
        # [2, 3, 0, 1] as two slices: a Python-list index is a pageable host->device copy and a stream sync
        kt_sf1 = torch.cat([kt_sfc[..., 2:4, :], kt_sfc[..., 0:2, :]], dim=-2).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    sfs = [ops["sfq"], ops["sfk"], kf_["sfv2"] if diag_fold else ops["sfv"], dq_["sf"], qt["sft"], kt["sft"], dq_["sft"], kt_sf1]
    if diag_kr_all and not kfused:
        # the key-local K^T in K^T's place on every pair; its scale is the key tile's constant, so the second d half's
        # SF tile (rows 64-127 moved to columns 0, 1) is the same bytes
        args_t[5] = kd["xt4"]
        sfs[5] = kd["sft"]
        sfs[7] = kd["sft"]
    # diagonal-only: the diagonal payload's row-wise V and SF, the dP operand on the diagonal pair
    v4b = ops["v4b"] if vtok_diag else None
    sfvb = (kf_["sfvb2"] if diag_fold else ops["sfvb"]) if vtok_diag else None
    # diag_k: the cut pair's key-local K^T payload / SF and the doubled [1 | r_k] (the gather's neutral first half)
    kt4b = kd["xt4"] if diag_k else None
    sfktb = kd["sft"] if diag_k else None
    if kfused:
        krv = kd["krv"] if (diag_k or diag_kr_all) else None
    else:
        krv = (torch.cat([torch.ones_like(_vtok_kernel_view(kd["kr"])) if diag_k else _vtok_kernel_view(kd["kr"]),
                          _vtok_kernel_view(kd["kr"])], dim=-1).contiguous() if kmode != "off" else None)   # "all": [r | r]
    if key not in _flash_attn_bwd_fp4q.cache:
        sf_ts = [to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in sfs]
        st_ts = [to_cute_tensor(t, divisibility=4) for t in (frame_c, uinv_c, Dp)]
        _flash_attn_bwd_fp4q.cache[key] = cute.compile(
            kern, *[_make_fp4_ptr(None)] * 7, *sf_ts,
            *st_ts, to_cute_tensor(dq_accum, divisibility=4), to_cute_tensor(dk, divisibility=8), to_cute_tensor(dv, divisibility=8),
            Float32(softmax_scale), Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0), Int32(0),
            tuple(Int32(0) for _ in range(4)), tuple(Int32(0) for _ in range(4)), tuple(Int32(0) for _ in range(4)),
            tuple(Int32(0) for _ in range(4)), tuple(Int32(0) for _ in range(4)), tuple(Int32(0) for _ in range(4)),
            tuple(Int32(0) for _ in range(4)),
            to_cute_tensor(q8, divisibility=8) if dkv8 else None, to_cute_tensor(do8, divisibility=8) if dkv8 else None,
            to_cute_tensor(k8, divisibility=8) if dq8 else None,
            to_cute_tensor(vb, assumed_align=4) if vtok else None, to_cute_tensor(vu, assumed_align=4) if vtok else None,
            _make_fp4_ptr(None) if vtok_diag else None,
            to_cute_tensor(sfvb, leading_dim=3, assumed_align=16) if vtok_diag else None,
            _make_fp4_ptr(None) if diag_k else None,
            to_cute_tensor(sfktb, leading_dim=3, assumed_align=16) if diag_k else None,
            to_cute_tensor(krv, assumed_align=4) if (diag_k or diag_kr_all) else None,
            to_cute_tensor(scl, assumed_align=4) if scl is not None else None,
            to_cute_tensor(kf_["dva"], assumed_align=4) if diag_fold else None,
            to_cute_tensor(code_dbg, assumed_align=1) if code_dbg is not None else None,
            to_cute_tensor(seed_t, assumed_align=8, leading_dim=0) if seed_t is not None else None,
            *((to_cute_tensor(kf_["fdk"], assumed_align=4), to_cute_tensor(kf_["fdv"], assumed_align=4)) if fold_epi
              else (None, None)),
            *((to_cute_tensor(docs.doc_tab, assumed_align=4), to_cute_tensor(docs.tile_doc(128), assumed_align=4,
                                                                             leading_dim=0))
              if docs is not None else (None, None)),
            current_stream, options="--enable-tvm-ffi",
        )
    call_args = (*[_make_fp4_ptr(t) for t in args_t], *sfs, frame_c, uinv_c, Dp, dq_accum, dk, dv,
                 softmax_scale, ds_mul, dk_scale, dv_scale, dq_shift, 0 if seed_t is not None else int(seed) & 0x7FFFFFFF,
                 shp(ops["q4"]), shp(ops["k4"]), shp(ops["v4"]), shp(dq_["x4"]), shpT(qt["xt4"]), shpT(kt["xt4"]), shpT(dq_["xt4"]),
                 q8, do8, k8, vb, vu,
                 _make_fp4_ptr(v4b) if vtok_diag else None, sfvb,
                 _make_fp4_ptr(kt4b) if diag_k else None, sfktb, krv, scl,
                 kf_["dva"] if diag_fold else None, code_dbg, seed_t,
                 *((kf_["fdk"], kf_["fdv"]) if fold_epi else (None, None)),
                 *((docs.doc_tab, docs.tile_doc(128)) if docs is not None else (None, None)))
    # `query-row-key-local-v-scale-dv-halved`. With `key-local-v-scale` the V operand is v~_k = V_k / s_k, and the
    # all-FP4 dV GEMM used to take the BIASED w' = P^' / Z as its A operand, which carries the per-key 2^A_k, so its
    # dV was dL/dv~_k = s_k dL/dV_k and this function divided it by s_k once. The fold now lives in the kernel:
    # the all-FP4 dV's A operand is the UNBIASED m = Mitchell u SF / Z, as on
    # the dkv8 route, so both routes return dL/dV and no entry point divides anything. The proof is unchanged: dL/dV
    # is invariant under V -> 2^3 V (v~ and its codes are bitwise unchanged), so n = log2(||dV(8V)|| / ||dV(V)||) / 3
    # must be 0 on both routes, and dV's norm ratio must be 1 (tests/cute/test_lowprec_causal_safe_v.py).
    row_divisor = None
    if r_bshd is not None:
        row_divisor = F_row if F_row is not None else r_q      # (b, h, s_q) contiguous float32 powers of two
        row_divisor = row_divisor.float().contiguous()
    info = dict(ds_mul=ds_mul, so_do=so_do, bound=bound, dq_=dq_, qt=qt, kt=kt, D=D, dq_accum=dq_accum, dq_scale=dq_scale, kd=kd,
                r_bshd=r_bshd, row_divisor=row_divisor, ds_scale_mode=ds_scale_mode, ds_row=F_row,
                dk_scale=dk_scale, dv_scale=dv_scale, dkv=dkv, q8=q8, do8=do8,
                sq8=sq8 if dkv8 else None, sdo8=sdo8 if dkv8 else None,
                # diag_fold: the dK / dV row factors 1 / r_k, 2^A / r_k that fp4q_run applies after the kernel
                fold_rows=(kf_["fdk"], kf_["fdv"]) if (diag_fold and not fold_epi) else None)
    info.update(info_scales)
    return dict(kern=_flash_attn_bwd_fp4q.cache[key], args=call_args, dq_accum_zeroed=dq_accum_zeroed, seed_t=seed_t, keep=(args_t, sfs, q8, do8, k8, vb, vu, v4b, sfvb, kt4b, sfktb, krv, scl, kf_ if kfused else None),
                dq_accum=dq_accum, dk=dk, dv=dv,
                s_q=s_q, dq_scale=dq_scale, dtype=q.dtype, info=info, post_rows=spec.post in ("rows", "epi"),
                tile_doc_q=docs.tile_doc_q if (docs is not None and isinstance(dq_scale, torch.Tensor)) else None)


PACKED_K_MODES = ("fold", "all", "off")   # (cov6: the key channels the packed backward serves; _diag_k_check's spelling)


def packed_bwd_ok(qspec, bspec) -> bool:
    """(cov5) The all-4-bit backward options the packed rows serve (fp4q_prepare_packed): the causal diagonal route with
    key channel 'fold' (cov6: and PACKED_K_MODES), the E4M3 dK / dV, the E2M1 dQ, the causal dS scale, any post
    (cov6: 'epi' -- the fold's row factors in the packed dK / dV epilogue)."""
    qs, sp = _quant_spec(qspec), _bwdq_spec(bspec)
    kmode = {"fold": "fold", "all": "all", "0": "off"}.get(_diag_k_knob(qs))
    return (kmode in PACKED_K_MODES and sp.dkv == "e4m3" and not sp.dq8
            and not qs.per_tensor_pow2 and fp4all_ds_scale_mode(True, True, qspec=qs) == "causal")


def packed_bwd_dense_ok(qspec, bspec) -> bool:
    """(cov6) The dense all-4-bit backward options the packed rows serve (_fp4q_prepare_packed_dense): the global dS
    bound with the row boost, the E4M3 dK / dV, the E2M1 dQ (any post: no K fold on the dense route)."""
    qs, sp = _quant_spec(qspec), _bwdq_spec(bspec)
    qrow_ = qs.fp4q_qrow if qs.fp4q_qrow is not None else qs.rule_fp4q_qrow(False)
    return (sp.dkv == "e4m3" and not sp.dq8 and bool(qrow_) and not qs.per_tensor_pow2
            and fp4all_ds_scale_mode(False, True, qspec=qs) == "global")


# (spd4) the packed backward's capped grid: with the static key tile bound min(T, ceil(T / 128) + N) above the cap
# 5 / 4 ceil(T / 128) (the fork's N = T: the bound is T), two launches -- key tiles [0, cap) one per CTA, the device
# count's rest persistent -- instead of one CTA per bound tile. OFF by default (owner's call): at N = T it takes the
# realistic batches from 1.8-4.6x the N = docs step to within ~1 %, but when nearly every tile is past the cap (all
# 1-token documents, T = 131072) the persistent overflow costs +23 % (causal) / +7 % (dense) over one CTA per tile.
# PACKED_BWD_CAP_TILES (tests): force the cap (and the split) to that many key tiles.
PACKED_BWD_CAP = False
PACKED_BWD_CAP_TILES = None


def _packed_cap(maps):
    """(spd4) The main launch's key tiles of the packed backward, or None (one launch over the static tile bound)."""
    if not PACKED_BWD_CAP and PACKED_BWD_CAP_TILES is None:
        return None
    tb = maps.rows_k // 128
    tq = -(-int(maps.total_k) // 128)
    cap = tq + (tq + 3) // 4 if PACKED_BWD_CAP_TILES is None else max(1, int(PACKED_BWD_CAP_TILES))
    return cap if tb > cap else None


def _cap_args(maps, cap):
    """(spd4) the kernel's trailing (pk_nt, pk_cap) arguments of a capped launch (call form)."""
    return (maps.ntiles[1:2], Int32(cap)) if cap is not None else ()


def _cap_cargs(maps, cap):
    """(spd4) the same at compile time."""
    return (to_cute_tensor(maps.ntiles[1:2], assumed_align=4, leading_dim=0), Int32(cap)) if cap is not None else ()


def fp4q_prepare_packed(q, k, v, out, dout, lse, frame, ops: dict, softmax_scale=None, causal=False, do_rnd=None,
                        ds_sr=None, seed=0, signs=None, dkv=None, deterministic=False, p_mix=None, code_dbg=None,
                        docs=None, *, spec, qspec, fspec):
    """(cov5) fp4q_prepare on the packed rows of compact DocMaps (lp_docmaps packed doc_maps calls): q / k / v / out /
    dout the caller's packed (1, T, ., 128) rows, lse / frame (1, h, T), ops the packed forward's (quant.fp4_diagonal
    documents["packed"]: dense SF words, key-parity copies, per-packed-key tables, the padding-key table). The
    causal diagonal route (key channel 'fold', the default): the dO launch per packed row (quant.fp4 causal_row_scale,
    the stochastic rounding hashed with the full maps' padded size -- bitwise the fused path's dither), the per-document
    E4M3 Q / dO (fp4q_diag_triton e4m3_q_do_device_packed), the kernel's codes_packed mode, the packed post launch."""
    from flash_attn.cute.flash_bwd_sm100_fp4q import FlashAttentionBackwardSm100FP4Q
    from flash_attn.cute.lp_docmaps import default_padded_size
    from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_device_packed
    spec = _bwdq_spec(spec)
    qs = _quant_spec(qspec)
    seed_t = seed if isinstance(seed, torch.Tensor) else None
    b, T, h, d = q.shape
    _, Tk, h_k, _ = k.shape
    r_gqa = h // h_k
    maps = docs.doc_maps
    assert b == 1 and d == 128 and maps is not None and maps.compact
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    do_rnd = do_rnd if do_rnd is not None else "sr"
    ds_sr = ds_sr if ds_sr is not None else True
    dkv = dkv if dkv is not None else fp4all_dkv_mode(spec)
    vtok_diag = bool(ops.get("vtok_diag", False)) and "vtok_bias" in ops
    if not vtok_diag and "vtok_bias" not in ops and not causal:
        return _fp4q_prepare_packed_dense(q, k, v, out, dout, lse, frame, ops, softmax_scale, do_rnd, ds_sr, seed, signs,
                                          dkv, deterministic, p_mix, code_dbg, docs, spec, qs, fspec)
    kmode = _diag_k_check(r_gqa, T, Tk, qspec=qs, bspec=spec) if vtok_diag else "off"
    if not (vtok_diag and dkv == "e4m3" and signs is None and kmode in PACKED_K_MODES and not spec.dq8
            and code_dbg is None and fp4all_ds_scale_mode(causal, True, qspec=qs) == "causal" and not qs.per_tensor_pow2):
        raise NotImplementedError("the all-4-bit backward on the packed rows of compact DocMaps: the causal diagonal "
                                  "route with key channel 'fold' (the default) / 'all' / 'open'")
    fold, kr_all = kmode == "fold", kmode == "all"
    fold_epi = fold and spec.post == "epi"   # (cov6: the K fold's dK / dV row factors in the packed epilogue)
    dev = q.device
    smax = {"rtn": 6.0, "sr": SR_SMAX, "eden": EDEN_SMAX}[do_rnd]
    kf_ = {n[len("kdiag_"):]: t for n, t in ops.items() if n.startswith("kdiag_")}
    dout = dout.contiguous()
    # the dO launch: codes / SF words / D / F_q / uinv per packed row, the per-document amax slots, the zeroed dQ rows
    dq_accum = torch.empty(1, h, T + 256, d, dtype=torch.float16, device=dev)
    crs = CausalRowScale(lse=lse.contiguous(), frame=frame.contiguous(), key_norm_prefix_max=ops["vmax_q"],
                         query_row_amax=ops["query_row_amax"], max_slots=ops["backward_amax_slots"],
                         top=smax * (256.0 if do_rnd == "eden" else 448.0), lse_log2_offset=LSE_LOG2_OFFSET,
                         key_factor_prefix_max=ops["rmax_q"] if fold else None, zero_accumulator=dq_accum)
    S_hash = default_padded_size(maps.total_q, maps.max_num_documents)   # (the fused path's aligned rows)
    r_ = quantize_fp4_rows(dout, make_transposed_copy=False, rounding=_ROUNDING[do_rnd], stochastic_rounding_seed=seed,
                           compute_d_with_output=out, causal_row_scale=crs, row_documents=docs.tile_doc_q,
                           packed_rows=(maps.vt_tile_q, maps.ntiles[0:1], S_hash))
    F_row, uinv, D = r_["row_factor"], r_["uinv"], r_["d"]
    q8, do8, scl = e4m3_q_do_device_packed(q.contiguous(), dout, F_row, softmax_scale, ops["backward_amax_slots"],
                                           docs.tile_doc_q, maps.vt_tile_q, maps.ntiles[0:1])
    # the statistics windows: the frame in a (1, h, T + 256) buffer (the kernel's 132-row windows of the last tiles)
    T_st = (T + 256 + 3) // 4 * 4   # (16-byte row strides for the statistics' TMA)
    frame_c, uinv_c, D_c = (torch.zeros(1, h, T_st, dtype=torch.float32, device=dev) for _ in range(3))
    frame_c[:, :, :T].copy_(frame.reshape(1, h, T))
    uinv_c[:, :, :T].copy_(uinv.reshape(1, h, T))
    D_c[:, :, :T].copy_(D.reshape(1, h, T))
    uinv, D = uinv_c, D_c
    dq_scale = softmax_scale
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    fl = lambda t: t.reshape(1, t.shape[0], -1) if t.dim() == 2 else t   # noqa: E731  (h_k, X) -> (1, h_k, X)
    # the doubled per-key tables (h_kv, 2 T'): fold [r | u r], 'all' [1 | u] (and [r | r] for the dQ copy's r_k);
    # 'open' [0 | bias], [1 | u] from the forward's per-packed-key tables
    if kmode == "off":
        b_, u_ = ops["vtok_bias"].permute(0, 2, 1), ops["vtok_u"].permute(0, 2, 1)   # (1, h_kv, T') views
        vb = torch.cat([torch.zeros_like(b_), b_], dim=-1).contiguous()
        vu = torch.cat([torch.ones_like(u_), u_], dim=-1).contiguous()
        # the dQ operand: the plain NVFP4 1 x 16 K^T in the key-parity copies, its SF in the packed group table
        ktq = quantize_fp4_rows(k, make_row_copy=False, make_transposed_copy=True,
                                transposed_copy_in_kernel_key_order=False, packed_rows=(maps.vt_tile, maps.ntiles[1:2]))
        kt_codes, kt_sf = ktq["transposed_codes"], ktq["transposed_sf_groups"]
    else:
        vb, vu = fl(kf_["vb2"]), fl(kf_["vu2"])
        kt_codes, kt_sf = kf_["xt4b"], kf_["sftb_tiles"]
    Tp = vb.shape[-1] // 2
    dva = fl(kf_["dva"]) if fold else None
    kr = fl(kf_["kr"]) if kr_all else None
    fdk_, fdv_ = (fl(kf_["fdk"]), fl(kf_["fdv"])) if fold_epi else (None, None)
    cap = _packed_cap(maps)
    kern = FlashAttentionBackwardSm100FP4Q(128, is_causal=causal, deterministic=deterministic, qhead_per_kvhead=r_gqa,
                                           p_mix=p_mix, ds_sr=ds_sr, dkv=dkv, vtok_diag=True, diag_single=False,
                                           diag_fwd_tiles=_diag_fwd_tiles(fspec), diag_k=False, diag_kr_all=kr_all,
                                           diag_fold=fold, spec=spec, doc_varlen=True, codes_packed=True)
    key = (FlashAttentionBackwardSm100FP4Q.__qualname__, spec.key(), r_gqa, causal, ds_sr, q.dtype, dkv, True, 128,
           deterministic, True, True, False, fold_epi, seed_t is not None, ("docs", True), ("packed", True), kmode,
           *((("cap", True),) if cap is not None else ()), cutlass.__version__,
           resolved_key(kern, ("max_offset", "dkv8", "dq8", "fp4q_exp2", "e2e_freq", "e2e_poly", "directp_clamp",
                               "directp_window", "directp_delta", "gm_floor", "ds_sr", "deterministic",
                               "qhead_per_kvhead", "mxds", "mxp", "vtok_diag_arg", "diag_single", "diag_fwd_tiles",
                               "diag_k", "diag_kr_all", "diag_fold", "p_mix_key")))
    u8 = dict(dtype=torch.uint8, device=dev)
    # shape carriers of the operands the codes_packed kernel does not TMA (K^T, the transposed placeholders) and of
    # the SF views (the load warp builds every atom from the dense words): never read
    kt_dummy = torch.zeros(1, 128, h_k, 128, **u8)
    qt = _trans_placeholder(1, T, h, d, dev)
    dq_t = _trans_placeholder(1, T, h, d, dev)
    sf_ph = ops["sfq"]
    sfs = [ops["sfq"], ops["sfk"], ops["sfv"], ops["sfq"], qt["sft"], ops["sfk"], dq_t["sft"], ops["sfk"]]
    shp = lambda t: tuple(t.shape[:3]) + (d,)   # noqa: E731
    shpT = lambda t: (1, d, t.shape[2], t.shape[3] * 2)   # noqa: E731
    args_t = [ops["q4"], ops["k4"], ops["v4"], r_["codes"], qt["xt4"], kt_dummy, dq_t["xt4"]]
    # the dP operands' SF words: fold the plain / diagonal payloads' x r_k (dense per-row), 'all' the plain payload's
    # dense per-row words and the diagonal payload's per-tile constant words
    pk = [maps.cu_q, maps.cu_k, ops["sfq_rows"], ops["sfk_rows"], r_["sf4_rows"],
          kf_["sfv2_rows"] if fold else ops["sfv_rows"], kf_["sfvb2_rows"] if fold else ops["sfvb_tiles"],
          kt_codes, kt_sf, kf_["pad_keys"]]
    if key not in _flash_attn_bwd_fp4q.cache:
        sf_ts = [to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in sfs]
        st_ts = [to_cute_tensor(t, divisibility=4) for t in (frame_c, uinv, D)]
        pk_ts = [to_cute_tensor(t, assumed_align=4, leading_dim=0) for t in pk[:2]] + \
            [to_cute_tensor(t, assumed_align=16) for t in pk[2:7]] + [to_cute_tensor(pk[7], assumed_align=16),
                                                                      to_cute_tensor(pk[8], assumed_align=16),
                                                                      to_cute_tensor(pk[9], assumed_align=16)]
        _flash_attn_bwd_fp4q.cache[key] = cute.compile(
            kern, *[_make_fp4_ptr(None)] * 7, *sf_ts,
            *st_ts, to_cute_tensor(dq_accum, divisibility=4), to_cute_tensor(dk, divisibility=8),
            to_cute_tensor(dv, divisibility=8),
            Float32(softmax_scale), Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0), Int32(0),
            *[tuple(Int32(0) for _ in range(4))] * 7,
            to_cute_tensor(q8, divisibility=8), to_cute_tensor(do8, divisibility=8), None,
            to_cute_tensor(vb, assumed_align=4), to_cute_tensor(vu, assumed_align=4),
            _make_fp4_ptr(None), to_cute_tensor(ops["sfvb"], leading_dim=3, assumed_align=16),
            None, None, to_cute_tensor(kr, assumed_align=4) if kr is not None else None,
            to_cute_tensor(scl, assumed_align=4), to_cute_tensor(dva, assumed_align=4) if dva is not None else None,
            None,
            to_cute_tensor(seed_t, assumed_align=8, leading_dim=0) if seed_t is not None else None,
            *((to_cute_tensor(fdk_, assumed_align=4), to_cute_tensor(fdv_, assumed_align=4)) if fold_epi
              else (None, None)),
            to_cute_tensor(docs.doc_tab, assumed_align=4), to_cute_tensor(docs.tile_doc(128), assumed_align=4,
                                                                           leading_dim=0),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream), pk_ts, Int32(maps.rows_q), Int32(maps.rows_k),
            Int32(Tp), *_cap_cargs(maps, cap), options="--enable-tvm-ffi",
        )
    call_args = (*[_make_fp4_ptr(t) for t in args_t], *sfs, frame_c, uinv, D, dq_accum, dk, dv,
                 softmax_scale, 1.0, 1.0, 1.0, 1.0, 0 if seed_t is not None else int(seed) & 0x7FFFFFFF,
                 shp(ops["q4"]), shp(ops["k4"]), shp(ops["v4"]), shp(r_["codes"]), shpT(qt["xt4"]),
                 shpT(kt_dummy), shpT(dq_t["xt4"]),
                 q8, do8, None, vb, vu, _make_fp4_ptr(ops["v4b"]), ops["sfvb"], None, None, kr, scl, dva, None, seed_t,
                 fdk_, fdv_, docs.doc_tab, docs.tile_doc(128))
    pk_args = (pk, Int32(maps.rows_q), Int32(maps.rows_k), Int32(Tp), *_cap_args(maps, cap))
    info = dict(ds_mul=1.0, so_do=1.0, bound=None, dq_scale=dq_scale, ds_scale_mode="causal", ds_row=F_row,
                dkv=dkv, q8=q8, do8=do8, fold_rows=(kf_["fdk"], kf_["fdv"]) if (fold and not fold_epi) else None)
    return dict(kern=_flash_attn_bwd_fp4q.cache[key], args=call_args, pk_args=pk_args, dq_accum_zeroed=True, seed_t=seed_t,
                keep=(args_t, sfs, q8, do8, vb, vu, kr, scl, pk, frame_c, uinv, D), dq_accum=dq_accum, dk=dk, dv=dv, s_q=T,
                dq_scale=dq_scale, dtype=q.dtype, info=info, packed=True, row_factor=F_row)


def _fp4q_prepare_packed_dense(q, k, v, out, dout, lse, frame, ops, softmax_scale, do_rnd, ds_sr, seed, signs, dkv,
                               deterministic, p_mix, code_dbg, docs, spec, qs, fspec):
    """(cov6) fp4q_prepare_packed of the DENSE (non-causal) all-4-bit route: the aligned dense fused chain per packed row
    -- the per-document max |dO| pass and the dO launch (global dS bound, D', uinv, max |dO| per row; quant.fp4
    packed_rows=, the stochastic rounding hashed on the full maps' padded size), the row boost
    (fp4q_diag_triton.dense_row_boost_packed), the per-document E4M3 Q / r, dO / r (e4m3_q_do_dense_packed: also the dQ
    scale per packed row), the kernel's codes_packed mode (K^T: the forward's plain 1 x 16 copy, its SF from the packed
    group table) and the packed post launch (dQ x the row's document's scale / r_q)."""
    from flash_attn.cute.flash_bwd_sm100_fp4q import FlashAttentionBackwardSm100FP4Q
    from flash_attn.cute.lp_docmaps import default_padded_size
    from flash_attn.cute.quant.fp4 import DenseRowBound
    from flash_attn.cute.fp4q_diag_triton import dense_row_boost_packed, e4m3_q_do_dense_packed
    seed_t = seed if isinstance(seed, torch.Tensor) else None
    b, T, h, d = q.shape
    _, Tk, h_k, _ = k.shape
    r_gqa = h // h_k
    maps = docs.doc_maps
    qrow_ = qs.fp4q_qrow if qs.fp4q_qrow is not None else qs.rule_fp4q_qrow(False)
    if not (dkv == "e4m3" and signs is None and not spec.dq8 and code_dbg is None and qrow_ and not deterministic
            and fp4all_ds_scale_mode(False, True, qspec=qs) == "global" and "sfkt_groups" in ops
            and "v_norm2_max" in ops and "query_row_amax" in ops and do_rnd in ("sr", "rtn")):
        raise NotImplementedError("the dense all-4-bit backward on the packed rows of compact DocMaps: the default "
                                  "dense route (global dS bound with the row boost, E4M3 dK / dV, E2M1 dQ)")
    dev = q.device
    smax = {"rtn": 6.0, "sr": SR_SMAX}[do_rnd]
    dout = dout.contiguous()
    vt_q, nt_q = maps.vt_tile_q, maps.ntiles[0:1]
    tdoc = docs.tile_doc_q
    n_docs1 = docs.b + 1
    S_hash = default_padded_size(maps.total_q, maps.max_num_documents)   # (the fused path's aligned rows)
    dq_accum = torch.empty(1, h, T + 256, d, dtype=torch.float16, device=dev)
    dkw = dict(row_documents=tdoc, n_documents=n_docs1, packed_rows=(vt_q, nt_q, S_hash))
    r0 = quantize_fp4_rows(dout, make_row_copy=False, output_tensor_max="abs", fp4_global_scale_divisor=smax * 448.0,
                           **dkw)
    so_do = r0["fp4_global_scale"]                      # (n_docs + 1,) float32 per document
    r1 = quantize_fp4_rows(dout, global_scale=so_do, rounding=_ROUNDING[do_rnd], stochastic_rounding_seed=seed,
                           compute_d_with_output=out,
                           dense_row_bound=DenseRowBound(lse=lse.contiguous(), frame=frame.contiguous(),
                                                         value_norm2_max=ops["v_norm2_max"],
                                                         lse_log2_offset=LSE_LOG2_OFFSET, zero_accumulator=dq_accum),
                           **dkw)
    r_q, uinv, amax_ = dense_row_boost_packed(r1["bound_rows"], r1["tensor_max"], r1["uinv"], ops["query_row_amax"],
                                              r1["row_amax"], float(qs.fp4q_qrow_cap), tdoc, vt_q, nt_q)
    q8, do8, scl, dqs_rows = e4m3_q_do_dense_packed(q.contiguous(), dout, r_q, softmax_scale, amax_, so_do,
                                                    r1["tensor_max"], tdoc, vt_q, nt_q)
    T_st = (T + 256 + 3) // 4 * 4   # (the statistics windows of the last tiles; 16-byte row strides)
    frame_c, uinv_c, D_c = (torch.zeros(1, h, T_st, dtype=torch.float32, device=dev) for _ in range(3))
    frame_c[:, :, :T].copy_(frame.reshape(1, h, T))
    uinv_c[:, :, :T].copy_(uinv.reshape(1, h, T))
    D_c[:, :, :T].copy_(r1["d_scaled"].reshape(1, h, T))
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    cap = _packed_cap(maps)
    kern = FlashAttentionBackwardSm100FP4Q(128, is_causal=False, deterministic=False, qhead_per_kvhead=r_gqa,
                                           p_mix=p_mix, ds_sr=ds_sr, dkv=dkv, vtok_diag=False, diag_single=False,
                                           diag_fwd_tiles=1, diag_k=False, diag_kr_all=False, diag_fold=False,
                                           spec=spec, doc_varlen=True, codes_packed=True)
    key = (FlashAttentionBackwardSm100FP4Q.__qualname__, spec.key(), r_gqa, False, ds_sr, q.dtype, dkv, False, 128,
           False, False, True, False, False, seed_t is not None, ("docs", True), ("packed", True), "dense",
           *((("cap", True),) if cap is not None else ()), cutlass.__version__,
           resolved_key(kern, ("max_offset", "dkv8", "dq8", "fp4q_exp2", "e2e_freq", "e2e_poly", "directp_clamp",
                               "directp_window", "directp_delta", "gm_floor", "ds_sr", "deterministic",
                               "qhead_per_kvhead", "mxds", "mxp", "vtok_diag_arg", "diag_single", "diag_fwd_tiles",
                               "diag_k", "diag_kr_all", "diag_fold", "p_mix_key")))
    u8 = dict(dtype=torch.uint8, device=dev)
    kt_dummy = torch.zeros(1, 128, h_k, 128, **u8)
    qt = _trans_placeholder(1, T, h, d, dev)
    dq_t = _trans_placeholder(1, T, h, d, dev)
    sfs = [ops["sfq"], ops["sfk"], ops["sfv"], ops["sfq"], qt["sft"], ops["sfk"], dq_t["sft"], ops["sfk"]]
    shp = lambda t: tuple(t.shape[:3]) + (d,)   # noqa: E731
    shpT = lambda t: (1, d, t.shape[2], t.shape[3] * 2)   # noqa: E731
    args_t = [ops["q4"], ops["k4"], ops["v4"], r1["codes"], qt["xt4"], kt_dummy, dq_t["xt4"]]
    pad_dummy = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=dev)
    pk = [maps.cu_q, maps.cu_k, ops["sfq_rows"], ops["sfk_rows"], r1["sf4_rows"], ops["sfv_rows"], ops["sfv_rows"],
          ops["kt4"], ops["sfkt_groups"], pad_dummy]
    if key not in _flash_attn_bwd_fp4q.cache:
        sf_ts = [to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in sfs]
        st_ts = [to_cute_tensor(t, divisibility=4) for t in (frame_c, uinv_c, D_c)]
        pk_ts = [to_cute_tensor(t, assumed_align=4, leading_dim=0) for t in pk[:2]] + \
            [to_cute_tensor(t, assumed_align=16) for t in pk[2:]]
        _flash_attn_bwd_fp4q.cache[key] = cute.compile(
            kern, *[_make_fp4_ptr(None)] * 7, *sf_ts,
            *st_ts, to_cute_tensor(dq_accum, divisibility=4), to_cute_tensor(dk, divisibility=8),
            to_cute_tensor(dv, divisibility=8),
            Float32(softmax_scale), Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0), Int32(0),
            *[tuple(Int32(0) for _ in range(4))] * 7,
            to_cute_tensor(q8, divisibility=8), to_cute_tensor(do8, divisibility=8), None,
            None, None, None, None, None, None, None,
            to_cute_tensor(scl, assumed_align=4), None, None,
            to_cute_tensor(seed_t, assumed_align=8, leading_dim=0) if seed_t is not None else None,
            None, None,
            to_cute_tensor(docs.doc_tab, assumed_align=4), to_cute_tensor(docs.tile_doc(128), assumed_align=4,
                                                                           leading_dim=0),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream), pk_ts, Int32(maps.rows_q), Int32(maps.rows_k),
            Int32(T + 256), *_cap_cargs(maps, cap), options="--enable-tvm-ffi",
        )
    call_args = (*[_make_fp4_ptr(t) for t in args_t], *sfs, frame_c, uinv_c, D_c, dq_accum, dk, dv,
                 softmax_scale, 1.0, 1.0, 1.0, 1.0, 0 if seed_t is not None else int(seed) & 0x7FFFFFFF,
                 shp(ops["q4"]), shp(ops["k4"]), shp(ops["v4"]), shp(r1["codes"]), shpT(qt["xt4"]),
                 shpT(kt_dummy), shpT(dq_t["xt4"]),
                 q8, do8, None, None, None, None, None, None, None, None, scl, None, None, seed_t,
                 None, None, docs.doc_tab, docs.tile_doc(128))
    pk_args = (pk, Int32(maps.rows_q), Int32(maps.rows_k), Int32(T + 256), *_cap_args(maps, cap))
    info = dict(ds_mul=None, so_do=so_do, bound=r1["tensor_max"], dq_scale=dqs_rows, ds_scale_mode="global",
                ds_row=r_q, dkv=dkv, q8=q8, do8=do8, fold_rows=None)
    return dict(kern=_flash_attn_bwd_fp4q.cache[key], args=call_args, pk_args=pk_args, dq_accum_zeroed=True,
                seed_t=seed_t, keep=(args_t, sfs, q8, do8, scl, pk, frame_c, uinv_c, D_c, dqs_rows, r_q), dq_accum=dq_accum,
                dk=dk, dv=dv, s_q=T, dq_scale=dqs_rows, dtype=q.dtype, info=info, packed=True, row_factor=r_q)


def fp4q_run(prep: dict):
    from flash_attn.cute.dq_postprocess_cute import dq_postprocess_cute, fp4q_postprocess
    if not prep.get("dq_accum_zeroed", False):
        prep["dq_accum"].zero_()
    prep["kern"](*prep["args"], cuda.CUstream(torch.cuda.current_stream().cuda_stream), *prep.get("pk_args", ()))
    if prep.get("packed", False):   # (cov5) the packed rows' post launch
        from flash_attn.cute.fp4q_diag_triton import fp4q_postprocess_packed
        fr = prep["info"]["fold_rows"]   # (fold: the K fold's dK / dV row factors; 'all': none)
        fk, fv = fr if fr is not None else (None, None)
        dq = fp4q_postprocess_packed(prep["dq_accum"], prep["s_q"], prep["dq_scale"], prep["dtype"],
                                     row_factor=prep["row_factor"], dk=prep["dk"] if fr is not None else None,
                                     dv=prep["dv"] if fr is not None else None, dk_row_factor=fk,
                                     dv_row_factor=fv, seed_counter=prep.get("seed_t"))
        return dq, prep["dk"], prep["dv"]
    info = prep["info"]
    fold = info.get("fold_rows")
    row_div = info.get("row_divisor")   # (b, h, s_q) powers of two the dQ rows carry (F_q / r_q), or None
    dsc = prep["dq_scale"]
    tdoc = prep.get("tile_doc_q")   # documents: dq_scale is a per-document table
    dev_ok = not isinstance(dsc, torch.Tensor) or (dsc.dtype == torch.float32 and (dsc.numel() == 1 or tdoc is not None))
    if dev_ok and (row_div is not None or info.get("r_bshd") is None):
        # ONE post launch: dQ x scale / row factor, and the K fold's dK / dV row factors in place
        dq = fp4q_postprocess(prep["dq_accum"], prep["s_q"], prep["dq_scale"], prep["dtype"], row_factor=row_div,
                              dk=prep["dk"] if fold is not None else None, dv=prep["dv"] if fold is not None else None,
                              dk_row_factor=fold[0] if fold is not None else None,
                              dv_row_factor=fold[1] if fold is not None else None,
                              seed_counter=prep.get("seed_t"),   # the SR counter advances in the same launch
                              rows=prep.get("post_rows", False), **({"tile_doc": tdoc} if tdoc is not None else {}))
        return dq, prep["dk"], prep["dv"]
    if prep.get("seed_t") is not None:   # (the reference paths without the fused post launch)
        prep["seed_t"].add_(1)
    if fold is not None:
        from flash_attn.cute.fp4q_diag_triton import rows_scale_dk_dv
        rows_scale_dk_dv(prep["dk"], prep["dv"], *fold)
    dq = dq_postprocess_cute(prep["dq_accum"], prep["s_q"], prep["dq_scale"], prep["dtype"])
    if info.get("r_bshd") is not None:  # dQ rows carry the boost of their own query
        dq = (dq.float() / info["r_bshd"]).to(dq.dtype)
    return dq, prep["dk"], prep["dv"]


_flash_attn_bwd_fp4q.cache = {}


# ------------------------------------------------------------------ autograd: the all-FP4 attention training step
_HAD_SIGNS = {}
_SR_COUNTER = {}


def sr_counter(device) -> torch.Tensor:
    """The all-4-bit backward's stochastic-rounding counter on `device`: an int64 (1,) tensor, starting at 1. The dO
    quantiser and the backward kernel derive the call's seed from it, (c * 0x9E3779B1) mod 2^31 -- the formula of the
    former host counter, so a fresh process gives the same seed sequence -- and the backward's post launch advances it
    by one on the GPU, so every step, and every replay of a captured step, gets a new seed."""
    key = str(torch.device(device))
    c = _SR_COUNTER.get(key)
    if c is None:
        c = _SR_COUNTER[key] = torch.ones(1, dtype=torch.int64, device=device)
    return c


def reset_sr_counter() -> None:
    """Restart every device's SR seed sequence at 1 (test harnesses: one fixed sequence per run, the former host
    counter's: the first step's seed is the formula's value at 1)."""
    for c in _SR_COUNTER.values():
        c.fill_(1)


def fp4all_stab_mode(qspec) -> str:
    """QuantSpec.fp4all_stab (options all_fp4_stabiliser): "sr" (SR dO^ + SR dS^), "eden_rht" (head-dim Hadamard on Q, K, V, dO;
    MS-EDEN dO^; SR dS^), "rtn"."""
    return _quant_spec(qspec).fp4all_stab


def _signs(dev):
    if dev not in _HAD_SIGNS:
        from flash_attn.cute.nvfp4_2d_triton import hadamard_signs
        _HAD_SIGNS[dev] = hadamard_signs(128, 4321, dev)
    return _HAD_SIGNS[dev]

# Host-side driver of the FP4 backward (flash_bwd_sm100_fp4.py): NVFP4 quantisers for the four
# S/dP operands, the (reused) BF16 preprocess, the main kernel, and the dQ^T postprocess.
import functools
import math
from typing import Optional, Tuple

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
import cuda.bindings.driver as cuda

from flash_attn.cute.interface import torch2cute_dtype_map, _get_device_capability, _get_device_arch
from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.device_scales import div32
from flash_attn.cute.flash_bwd_sm100_fp4 import FlashAttentionBackwardSm100FP4
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.lp_masks import lp_masks_or_none, to_cute_aux
from flash_attn.cute.lowprec.specs import (Bwd1Spec, HostSpec, PlanSpecs, QuantSpec, DS_SCALING_DEFAULT,
                                           OP_SCALING_DEFAULT, resolved_key)

SF_VEC_SIZE = 16
SF_TILE_M = 128  # rows per scale-factor atom (32 x 4)


def _quant(quant: Optional[QuantSpec]) -> QuantSpec:
    """The call's QuantSpec (from its plan, lowprec.plan.build_specs): required, never built here."""
    if quant is None:
        raise TypeError("a lowprec.specs.QuantSpec is required (the call's plan.specs.quant)")
    return quant


def _host(host: Optional[HostSpec]) -> HostSpec:
    """The call's HostSpec (from its plan): required, never built here."""
    if host is None:
        raise TypeError("a lowprec.specs.HostSpec is required (the call's plan.specs.host)")
    return host


def _pow2(quant: QuantSpec, scales=None) -> bool:
    """power_of_two_scale of a per-tensor quantiser call: QuantSpec.per_tensor_pow2 for computed scales only (given
    scales -- the forward's, handed to the backward -- are used as given)."""
    return scales is None and quant.per_tensor_pow2


@functools.lru_cache(maxsize=None)
def _arch_key() -> tuple:
    """The compile key's arch items: (device arch, CuTe DSL version)."""
    return (_get_device_arch(), cutlass.__version__)


def permute_queries_c3(x: torch.Tensor) -> torch.Tensor:
    """(b, s, h, d) -> the same with the 32-query blocks of every 128-query tile reordered [0, 2, 1, 3]: the
    K order of the C3 dV / dK GEMMs (each compute WG owns two interleaved 32-query blocks, see
    FlashAttentionBackwardSm100FP4.tmem_ld_rep). Involution: applying it twice restores the order."""
    b, s, h, d = x.shape
    assert s % 128 == 0, s
    return x.reshape(b, s // 128, 4, 32, h, d)[:, :, [0, 2, 1, 3]].reshape(b, s, h, d)


def _rows(x: torch.Tensor, scale_group_size: int = 16):
    """(codes, SF view) of x's NVFP4 (16) / MXFP4 (32) row copy (quant.fp4)."""
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    r = quantize_fp4_rows(x, scale_group_size=scale_group_size)
    return r["codes"], r["scale_factors"]


def _make_fp4_ptr(t: Optional[torch.Tensor]):
    from cutlass.cute.runtime import make_ptr

    return make_ptr(cutlass.Float4E2M1FN, 0 if t is None else t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def row_norms_f32(x: torch.Tensor) -> torch.Tensor:
    """||x||_2 along the last dim in fp32, without the fp32 copy of x that x.float().norm(dim=-1) materialises (a 128 /
    256 MiB temporary at b4-s4096 / b1-s32768). The same reduction as `max_row_norm`, whose value the lean-save path
    already relies on being the backward's (bitwise equal)."""
    return torch.linalg.vector_norm(x, dim=-1, dtype=torch.float32)


def grad_fp8_format(quant: Optional[QuantSpec] = None) -> str:
    """grad_fp8_dtype() as quant.fp8's fp8_format name."""
    return "e5m2" if grad_fp8_dtype(quant) is torch.float8_e5m2 else "e4m3"


def grad_fp8_dtype(quant: Optional[QuantSpec] = None) -> torch.dtype:
    """Element type of the backward's gradient operands (dO, and dS where it is host-scaled): QuantSpec.grad_format."""
    name = _quant(quant).grad_format
    if name == "e5m2":
        return torch.float8_e5m2
    if name == "e4m3":
        return torch.float8_e4m3fn
    raise ValueError(f"fp8_grad_dtype must be 'e4m3' or 'e5m2', got {name!r}")


def e4m3_p_shift_log2() -> int:
    r"""log2 of the power of two by which an E4M3 \hat{P} is lifted through the LSE (P' = 2^shift P), undone in host scalars.

    `p-shift-two-names`: ONE quantity used by three routes -- the 1-CTA FP8 control, the 1-CTA dk8 route's dv8 path
    (was a hardcoded `dv8_shift = 8`) and the 2-CTA kernel (was `interface_bwd_2cta8.P_SHIFT_LOG2 = 8`) -- now one
    name, one default, one parser. It is host-side in all three (LSE shift, dv / ds scales); no kernel specialises on
    it. 8: P <= 1 -> P' <= 256 < 448 (a larger lift would saturate E4M3 at P = 1). A constant, not a tuning field.
    """
    return E4M3_P_SHIFT_LOG2


def fp4_p_shift_log2() -> int:
    """The f16-range shift of P of the former bf16 dV path: 0. That path is not shipped; the E4M3 P^ of the 4-bit route
    takes its own lift (`e4m3_p_shift_log2`).
    The kernel's p_shift is the same constant (`_check_fp4_p_shift`)."""
    return 0


def _check_fp4_p_shift(kern) -> None:
    want = float(2 ** fp4_p_shift_log2())
    if getattr(kern, "p_shift", want) != want:
        raise AssertionError(f"fp4 P shift: kernel p_shift {kern.p_shift} != driver {want}")


# The E4M3 P^ lift (log2) of every low-precision backward route (a constant: 8).
E4M3_P_SHIFT_LOG2 = 8

# The only E4M3 P^ shift the 2-CTA route runs: `fp4_bwd_autograd`'s fused dO prologue shifts LSE by it before the
# driver is reached. `_use_2cta_dk8` declines any other `e4m3_p_shift_log2()` value, so the 1-CTA kernel runs.
P8_SHIFT_2CTA = 8


# The f16 dQ partials' fixed pre-scale (log2) of an 8-bit call without the operand norms (a constant: 6; with the norms
# the pre-scale is data-sized, dq_range_of).
FP8_DQ_BOOST_LOG2 = 6


def bwd_dq_f16(host: Optional[HostSpec] = None) -> bool:
    """f16 dQ partials (half the reduce-add bytes) instead of fp32, for every low-precision backward.

    `dq-f16-two-names`: the 1-CTA and the 2-CTA kernels once read this choice under two different names. It is one
    setting now: KernelTuning.bwd_dq_f16 -> HostSpec.bwd_dq_f16, default True, for every route.
    """
    return bool(_host(host).bwd_dq_f16)


def _pmax_bound(q, k, dout, v, lse, dpsum, softmax_scale, causal, chunk=1024):
    """max_q P_max_q * (||dO_q|| max_k ||V_k|| + |D_q|) >= max |dS| (exact row max of S, chunked over queries), as a float32
    device scalar (0-dim): max(best, 1e-30), where the host version took the same maxima through .item()."""
    b, s_q, h, d = q.shape
    s_k, h_k = k.shape[1], k.shape[2]
    r = h // h_k
    vmax = v.float().norm(dim=-1).amax()  # a float32 device scalar (the host read gave the same float32 value)
    best = torch.zeros((), dtype=torch.float32, device=q.device)
    kk = k.float().permute(0, 2, 1, 3)  # (b, h_k, s_k, d)
    for i in range(0, s_q, chunk):
        qi = q[:, i:i + chunk].float().permute(0, 2, 1, 3)  # (b, h, c, d)
        S = torch.einsum("bhcd,bhkd->bhck", qi.reshape(b, h_k, r, -1, d).reshape(b, h, -1, d), kk.repeat_interleave(r, dim=1)) * softmax_scale
        if causal:
            qpos = torch.arange(i, i + qi.shape[2], device=q.device)[:, None] + (s_k - s_q)
            S = S.masked_fill(torch.arange(s_k, device=q.device)[None, :] > qpos, float("-inf"))
        pmax = torch.exp(S.amax(-1) - lse[:, :, i:i + chunk].float())  # (b, h, c)
        y = dout[:, i:i + chunk].float().norm(dim=-1).permute(0, 2, 1) * vmax + dpsum[:, :, i:i + chunk].abs().float()
        best = torch.maximum(best, (pmax.clamp(max=1.0) * y).amax())
    return best.clamp_min(1e-30)


def ds_bound_cheap(qd, kd, dout, vd, lse, dpsum, softmax_scale, causal, rows=512):
    """Estimate max|dS| for the dS^ scale, by evaluating dS exactly on sampled query rows.

    `qd`, `kd`, `vd` must be the DEQUANTISED operands (codes x descale) -- what the kernel
    actually multiplies. On the FP8 path the caller's q/k/v are shape carriers holding no data
    (the step saves codes, not activations); reading those gave max_k S_qk = 350 against a true
    4, and a scale two orders out.

    **Tight beats safe, and not for the obvious reason.** The natural move is an upper bound,
    max|dS| <= max_q P_max_q (||dO_q|| max_k ||V_k|| + |D_q|), which can never clip. Measured, a
    bound is the wrong object. E4M3 is floating point, so a loose scale costs nothing in
    *relative* precision on the large values -- but it pushes the small dS entries under the
    subnormal floor and annihilates them. At logit scale 1.0 the fraction of nonzero dS flushed
    to zero runs 0.38 % at the true max, 0.43 % at 1.1x, 1.8 % at 4.7x and 29 % at 95x, while the
    norm-error metric reads 2.65e-02 for every one of them and sees nothing. Six-thousand-step
    convergence tracks the flush rate, not the norm error: val loss 0.8590 / 0.8809 / 0.8988 at
    1.1x / 4.7x / 95x.

    So estimate the actual maximum instead of bounding it: take `rows` query rows spread evenly
    over the sequence, form their exact dS = P (dP - D), and take its max. Two skinny matmuls,
    O(b h rows s d) -- 1/64 of the O(b h s^2) exact version at s = 32768. Sampling can land a
    little under the true max, clipping a sliver of the extreme tail; measured, that is far
    cheaper than being loose (the scale that trained best sat at 0.75-1.13x of the true max).

    Better still, have the forward return P_max as the FP4-all path already does.
    """
    s_q, h = qd.shape[1], qd.shape[2]
    s_k, h_k = kd.shape[1], kd.shape[2]
    r = h // h_k
    idx = torch.linspace(0, s_q - 1, min(rows, s_q), device=qd.device).long().unique()
    qi = qd[:, idx].float().permute(0, 2, 1, 3)                           # (b, h, n, d)
    doi = dout[:, idx].float().permute(0, 2, 1, 3)                        # (b, h, n, d)
    kk = kd.float().permute(0, 2, 1, 3)                                   # (b, h_k, s_k, d)
    vv = vd.float().permute(0, 2, 1, 3)
    if r > 1:
        kk = kk.repeat_interleave(r, dim=1)
        vv = vv.repeat_interleave(r, dim=1)
    S = torch.einsum("bhnd,bhkd->bhnk", qi, kk) * softmax_scale
    if causal:
        qpos = idx[:, None] + (s_k - s_q)
        S = S.masked_fill(torch.arange(s_k, device=qd.device)[None, :] > qpos, float("-inf"))
    P = torch.exp(S - lse[:, :, idx].float()[..., None])                  # (b, h, n, s_k)
    dP = torch.einsum("bhnd,bhkd->bhnk", doi, vv)
    dS = P * (dP - dpsum[:, :, idx].float()[..., None])
    return dS.abs().amax().float().clamp_min(1e-30)  # a float32 device scalar: no host read


def _flash_attn_bwd_fp8_control(
    q, k, v, out, dout, lse, softmax_scale, causal, deterministic, tile_m, return_accum, fp8_scales=None, seqused_k=None, out_pass="all",
    masks=None, ds_dq_rowlocal: str = "off", spec: Optional[Bwd1Spec] = None, quant: Optional[QuantSpec] = None,
    host: Optional[HostSpec] = None,
):
    """FP8 control (HostSpec.bwd_op_mode = "fp8", the 8-bit route): the FP4 backward kernel compiled with E4M3 operands + per-tensor
    descales (kind::f8f6f4). fp8_scales: optional dict of pre-quantised operands {q8, k8, v8, do8} and the scales they
    were quantised with {sq, sk, sv, sdo} (this call's own amax / 448), plus optionally s_ds (a bound on max|dS| / 448
    computed from this call's tensors); the descales are folded into softmax_scale (s_q s_k), the dPsum scale
    (1 / (s_v s_dO)) and the dV / dK / dQ output scales. P is shifted by 2^E4M3_P_SHIFT_LOG2 (8: P <= 1 ->
    <= 256 < 448) through LSE; with per-tensor dS the boost is (s_v s_dO) / s_ds."""
    b, s_q, h, d = q.shape
    r_gqa = h // k.shape[2]
    device = q.device
    from flash_attn.cute.device_scales import dev32, bwd1_fp8_scalars
    # the specs of this call (its plan's)
    if spec is None:
        raise TypeError("_flash_attn_bwd_fp8_control: a lowprec.specs.Bwd1Spec is required (the plan's specs.bwd1)")
    quant, host = _quant(quant), _host(host)
    fp8_scales = dict(fp8_scales or {})
    mxq = None
    do_owned = False
    if _docs(masks) and quant.ds_scaling_fp8 not in ("blk32", "mx32"):
        raise NotImplementedError("varlen: a per-tensor dS scale is a channel between documents; use the block-scaled dS")
    if op_blk32_fp8(quant) or (_docs(masks) and not doc_scaled(masks, quant, fp8=True)):
        # QuantSpec.operand_scaling_fp8 = blk32 / blk32x128: 32-token-block MXFP8 Q, K, V, dO (unit descales); fp8_scales["mx"] =
        # the caller's quantize_fp8_blocks dicts (the forward's codes); anything missing is quantised here from the bf16 tensors
        if quant.ds_scaling_fp8 not in ("blk32", "mx32"):
            raise ValueError("block-scaled 8-bit operands need a block-scaled dS (QuantSpec.ds_scaling_fp8 blk32 / mx32)")
        mxq = dict(fp8_scales.get("mx") or {})
        need = {nm: t for nm, t in (("q", q), ("k", k), ("v", v)) if nm not in mxq}
        # the dO dict's D may be folded in place only when it is this call's own (quantised here, or handed over)
        do_owned = "do" not in mxq or "do" in (mxq.get("owned", ()) or ())
        if "do" not in mxq:
            mxq.pop("fac", None)  # a caller's factor table has no row for the dO quantised here: rebuilt from the "e"s
            mxq.update(mx_operands(dout=dout, out=out, masks=masks, quant=quant, host=host))
        if need:
            mxq.update(mx_operands(**need, masks=masks, quant=quant, host=host))
        _had_in = fp8_scales.get("had_signs")  # qk_hadamard: the caller's codes are Q R / K R (kept for the back-rotation)
        from flash_attn.cute.device_scales import unit_descales
        _u = unit_descales(q.device)   # the unit descales as views of a per-device constant: no fill launches
        # the f16 dQ range (dq_range_of): the caller's norms, else the statistics of the operands quantised here
        _rng = {key: fp8_scales[key] if fp8_scales.get(key) is not None else mxq[nm].get(stat)
                for key, nm, stat in (("dn2", "do", "row_norm2_max"), ("vn2", "v", "row_norm2_max"),
                                      ("kmax", "k", "amax"))}
        _rng["dq_kv_rows"] = dq_kv_rows_of(fp8_scales, k, v, causal)
        fp8_scales = dict(q8=mxq["q"]["codes"], k8=mxq["k"]["codes"], v8=mxq["v"]["codes"], do8=mxq["do"]["codes"],
                          sq=_u[0], sk=_u[1], sv=_u[2], sdo=_u[3], **_rng)
        # the dO launch's backward glue (fp4_bwd_autograd.tb_bwd_glue): LSE * log2 e - shift and the zeroed f16 dQ
        # accumulator, when this call owns that launch's outputs
        if "do" in mxq.get("owned", ()):
            for n_ in ("lse_log2", "lse_log2_shift", "dq_accum"):
                if mxq["do"].get(n_) is not None:
                    fp8_scales["glue_" + n_] = mxq["do"][n_]
        if _had_in:
            fp8_scales["had_signs"] = _had_in
    if "q8" in fp8_scales:  # pre-quantised E4M3 operands + their scales (timing: quantisers outside); scales on the GPU
        q8, k8, v8, do8 = (fp8_scales[n] for n in ("q8", "k8", "v8", "do8"))
        sq, sk, sv, sdo = (dev32(fp8_scales[n], q.device) for n in ("sq", "sk", "sv", "sdo"))
    else:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        # V's and dO's max row norms from their quantisers' reads: the f16 dQ range (dq_range_of)
        r = {nm: quantize_fp8_per_tensor(t, scales=fp8_scales.get("s" + nm), output_row_norm2_max=nm == "v",
                                         power_of_two_scale=_pow2(quant, fp8_scales.get("s" + nm)))
             for nm, t in (("q", q), ("k", k), ("v", v))}
        (q8, sq), (k8, sk), (v8, sv) = ((r[nm]["codes"], dev32(r[nm]["scales"][0], q.device)) for nm in "qkv")
        fp8_scales.setdefault("vn2", r["v"]["row_norm2_max"][-1:])
        # dO is a gradient operand: E5M2 when configured (128x the range of E4M3 for two
        # fewer mantissa bits), which is the usual trade for tensors whose dynamic range
        # arrives from upstream and cannot be calibrated as tightly as an activation.
        r = quantize_fp8_per_tensor(dout, scales=fp8_scales.get("sdo"), fp8_format=grad_fp8_format(quant),
                                    output_row_norm2_max=True,
                                    power_of_two_scale=_pow2(quant, fp8_scales.get("sdo")))
        do8, sdo = r["codes"], dev32(r["scales"][0], q.device)
        fp8_scales.setdefault("dn2", r["row_norm2_max"][-1:])
    # Resolved after the preprocess, once dpsum exists (see below): the default of 1.0 assumed
    # dS already sits inside E4M3, which it does not. A caller-computed s_ds (this call's bound) gives it directly.
    s_ds_dev = dev32(fp8_scales["s_ds"], q.device) if "s_ds" in fp8_scales else None
    masks = lp_masks_or_none(masks, causal)  # masks / varlen: None = the plain causal / dense kernel, bitwise
    _pk = packed_docs(masks)   # lp_docmaps packed rows: T-row codes / accumulator, (h, S) statistics (DocMaps.vt_tile_q)
    if masks is not None:
        causal = masks.causal
        _check_masks_ds(masks, quant.ds_scaling_fp8 == "per_tensor")
    p_shift_log2 = e4m3_p_shift_log2()
    p_shift = float(2 ** p_shift_log2)
    # the f16 dQ partials' pre-scale of a call without the operand norms (dq_range_of); with them it is data-sized
    dq_boost = float(2 ** FP8_DQ_BOOST_LOG2)
    dtype = torch2cute_dtype_map[q.dtype]
    m_block_size = tile_m
    n_block_size = 128
    seqlen_q_rounded = (s_q + m_block_size - 1) // m_block_size * m_block_size
    if _pk:   # (packed rows: the accumulator is the T packed rows; the kernel masks a tile's rows past its document)
        seqlen_q_rounded = s_q
    dq_f16 = bwd_dq_f16(host)
    # d = 64 (native head_dim 64): no preprocess launch. D comes from the dO quantiser that produced the dP GEMM's own codes (the
    # token-block dO dict's D, or the per-tensor step glue's dpsum: softmax-consistent, as the 2-CTA kernel takes it),
    # LSE * log2 e - shift from the glue (or one torch expression), the f16 accumulator zeroed by the glue when it
    # offers one. d = 128: the preprocess (it zeroes both accumulator types), or the
    # dO launch's glue when it carries LSE / the zeroed accumulator / D.
    _ext_ll = fp8_scales.get("lse_log2")
    if native_d64(d) and mxq is not None and mxq["do"].get("lse_log2") is not None \
            and mxq["do"].get("lse_log2_shift") == float(e4m3_p_shift_log2()):
        _ext_ll = mxq["do"]["lse_log2"]  # the token-block dO quantiser's (d = 64 prologue, _mx_do_prologue)
        fp8_scales["dq_accum"] = mxq["do"].get("dq_accum")
    ext64 = (native_d64(d) and dq_f16 and (seqlen_q_rounded == s_q or _pk)
             and (mxq is not None or ("dpsum" in fp8_scales and _ext_ll is not None
                                      and fp8_scales.get("lse_log2_shift", e4m3_p_shift_log2()) == e4m3_p_shift_log2())))
    _ext_acc = fp8_scales.get("dq_accum") if ext64 else None
    p_shift_glue = fp8_scales.get("glue_lse_log2_shift")
    # (per-(document, kv head) per-tensor operands: the per-tensor dO launch carries the same glue, doc_scaled)
    # (batch_scaled: a batched call with per-(batch row, kv head) scales, lowprec.prequant_fa3 -- the same kernel and glue)
    pt_docs = doc_scaled(masks, quant, fp8=True) or bool(fp8_scales.get("batch_scaled"))
    # (batch_scaled with window / mask_mod / score_mod: batched masks without documents, the batched dO launch's glue)
    _bs_masks = bool(fp8_scales.get("batch_scaled")) and masks is not None and masks.docs is None
    glue = (not ext64 and (mxq is not None or pt_docs) and (masks is None or aligned_docs(masks) or _bs_masks) and dq_f16
            and (seqlen_q_rounded == s_q or _pk)
            and fp8_scales.get("glue_lse_log2") is not None and fp8_scales.get("glue_dq_accum") is not None
            and p_shift_glue is not None and float(p_shift_glue) == float(p_shift_log2))
    if glue:
        # the dO launch already wrote lse_log2 (shifted), the zeroed accumulator and D (owned): no preprocess launch,
        # no D copy, no LSE subtraction -- bitwise the preprocess + copy_ + sub_ chain below (the 2-CTA glue)
        dq_accum = fp8_scales["glue_dq_accum"].view(b, h, seqlen_q_rounded, d)
        dq_accum_flat = dq_accum.view(b, h, seqlen_q_rounded * d)
    elif dq_f16 and _ext_acc is not None and _ext_acc.dtype == torch.float16 and _ext_acc.is_contiguous() \
            and _ext_acc.numel() == b * h * seqlen_q_rounded * d:
        dq_accum = _ext_acc.view(b, h, seqlen_q_rounded, d)  # zeroed by the dO quantiser's launch
        dq_accum_flat = None
    elif ext64:  # no preprocess launch to zero it
        dq_accum = torch.zeros(b, h, seqlen_q_rounded, d, dtype=torch.float16, device=device)
        dq_accum_flat = None
    else:
        dq_accum = torch.empty(b, h, seqlen_q_rounded, d, dtype=torch.float16 if dq_f16 else torch.float32, device=device)
        dq_accum_flat = dq_accum.view(b, h, seqlen_q_rounded * d)
    if pt_docs and not (glue or ext64):
        raise NotImplementedError("per-tensor varlen: the 1-CTA backward takes D, the shifted LSE and the zeroed f16 dQ "
                                  "accumulator from the per-tensor dO launch (fp8_scales glue_*)")
    if glue:
        dpsum = mxq["do"]["D"] if mxq is not None else fp8_scales["dpsum"]
        lse_log2 = fp8_scales["glue_lse_log2"]
    else:
        dpsum = torch.empty(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
    # lp_docmaps (a fused doc_maps backward): dK / dV straight to the caller's packed key rows (the kernel's epilogue);
    # packed rows with a window / mask_mod: they start zero, as the aligned rows would
    _dkv_rows = None
    if masks is not None and getattr(masks.docs, "doc_maps", None) is not None and out_pass == "all" \
            and (_pk or not dkv_zero_fill(masks)):
        from flash_attn.cute.lp_docmaps import dkv_packed_rows
        _dkv_rows = dkv_packed_rows(masks.docs)
    if _dkv_rows is not None:
        _new = torch.zeros if dkv_zero_fill(masks) else torch.empty
        dk = _new((1, _dkv_rows) + tuple(k.shape[2:]), dtype=k.dtype, device=k.device)
        dv = _new((1, _dkv_rows) + tuple(v.shape[2:]), dtype=v.dtype, device=v.device)
    else:
        # a key tile with an empty query range (window / document tail) writes no dK / dV: zeros, as upstream's epilogue
        dk = torch.zeros_like(k) if dkv_zero_fill(masks) else torch.empty_like(k)
        dv = torch.zeros_like(v) if dkv_zero_fill(masks) else torch.empty_like(v)
    dQ_semaphore = (
        torch.zeros(b, h, (masks.docs.s_q if _pk else seqlen_q_rounded) // m_block_size, 1, dtype=torch.int32,
                    device=device) if deterministic else None
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if ext64:
        if mxq is not None:
            _d = mxq["do"]["D"]
            # the D fold below scales D in place: only a dO dict this call owns (quantised here, or handed over)
            dpsum = _d if do_owned else _d.clone()
            if _ext_ll is not None:
                lse_log2 = _ext_ll
            else:
                lse_log2 = (torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse * math.log2(math.e))
                            - float(p_shift_log2)).contiguous()
        else:
            dpsum = fp8_scales["dpsum"] if fp8_scales.get("dpsum_owned") else fp8_scales["dpsum"].float().clone()
            lse_log2 = _ext_ll  # already LSE * log2 e - shift (the per-tensor dO launch's)
    elif not glue:
        key_pre = ("fp8_control_zeroes_acc", dtype, d, m_block_size, dq_f16)  # the FP4 path's entry passes no f16 accumulator
        if key_pre not in _flash_attn_bwd_fp4.cache_pre:
            o_t, do_t = [to_cute_tensor(t, divisibility=8) for t in (out, dout)]
            dps_t, lsel_t = [to_cute_tensor(t, divisibility=4) for t in (dpsum, lse_log2)]
            dqa_t = to_cute_tensor(dq_accum_flat, divisibility=8 if dq_f16 else 4) if dq_accum_flat is not None else None
            lse_t = to_cute_tensor(lse, assumed_align=4)
            pre = FlashAttentionBackwardPreprocess(dtype, d, d, m_block_size, num_threads=256,
                                                   dqaccum_dtype=cutlass.Float16 if dq_f16 else Float32)
            _flash_attn_bwd_fp4.cache_pre[key_pre] = cute.compile(
                pre, o_t, do_t, dps_t, lse_t, lsel_t, dqa_t, None, None, None, None, None, Float32(1.0), None, current_stream, options="--enable-tvm-ffi",
            )
        _flash_attn_bwd_fp4.cache_pre[key_pre](out, dout, dpsum, lse, lse_log2, dq_accum_flat, None, None, None, None, None, Float32(1.0), None, current_stream)
        if mxq is not None:  # D from exactly the dequantised dO the dP GEMM multiplies (softmax-consistent)
            dpsum[:, :, :s_q].copy_(mxq["do"]["D"])
        lse_log2.sub_(float(p_shift_log2))  # P' = P * 2^shift
    if s_ds_dev is None and quant.ds_scaling_fp8 == "per_tensor":
        # d\hat{S} = ds_boost dS / (s_v s_dO) has to land inside E4M3, so a per-tensor dS needs its scale s_ds = max|dS| / 448
        # (measured without one: dQ / dK cosines 0.45 / 0.50). There is no in-driver fallback (an exact amax would be
        # O(s^2) on the host): the caller passes s_ds.
        raise ValueError("_flash_attn_bwd_fp8_control: a per-tensor dS needs fp8_scales['s_ds'] "
                         "(the exact max|dS| fallback was removed)")
    if quant.ds_scaling_fp8 != "per_tensor":
        # block-scaled d\hat{S} (in-kernel UE8M0 scale factors): no global max|dS|, no boost -- the SFs carry
        # the range and the MMA applies them, so the epilogue scales see ds_boost = 1
        s_ds_dev = None
    # folded descales (see the kernel's FP8 notes), on the GPU (device_scales._bwd1_fp8_scalars_kernel): S = s_q s_k \hat{S},
    # dP = s_v s_dO d\hat{P}; softmax_scale_k = softmax_scale s_q s_k, dpsum_scale = 1 / (s_v s_dO), dv = s_dO / p_shift,
    # dk = softmax_scale s_v s_dO s_q / ds_boost, dq = dq_boost s_v s_dO s_k / ds_boost; the kernel multiplies
    # P' (d\hat{P} - D'), P' = 2^shift P, so its boost is ds_boost / 2^shift (d\hat{S} = ds_boost dS / (s_v s_dO))
    # the f16 dQ range from the quantisers' norms (device_scales._dq_prescale)
    dq_range = dq_range_of(fp8_scales, block_scaled=mxq is not None) if dq_f16 else None
    dqb_dyn = dq_range is not None
    # dqfix: the pre-scale per (row, q head, query tile) where the statistics are per tile (token-block operands)
    dq_tiles = (dq_tiles_of(dq_range, masks, b, h, k.shape[2], *_pk_aligned(masks, s_q, k.shape[1]), tile_m, causal=causal,
                          rows_src=None if d != 128 else ((fp8_scales or {}).get("dq_kv_rows"), dout, _fuse_backrot(fp8_scales)))
                if (mxq is not None and not pt_docs and not return_accum and seqlen_q_rounded == s_q) else None)
    dq_docs = None
    post_scale = None
    if pt_docs:
        # per-(document, kv head) scales: the (n_docs + 1, h_kv, 8) scalar table the kernel reads per work tile, one
        # launch (as the per-tensor scalar launch); the D fold D'' = m D was applied by the dO launch per document
        from flash_attn.cute.device_scales import bwd1_fp8_doc_scalars
        dsc = fp8_scales["doc_scales"]
        if s_ds_dev is not None or not dq_f16:
            raise NotImplementedError("per-tensor varlen: the block-scaled dS and f16 dQ partials only")
        if quant.ds_fold and not fp8_scales.get("doc_d_folded"):
            raise RuntimeError("per-tensor varlen with ds_fold: the dO launch must fold D per document (documents fold)")
        # (packed rows: the plane per ALIGNED query tile of the documents, as the fused path's)
        dq_docs = dq_docs_of(dq_range, masks, b, h, _pk_aligned(masks, s_q, k.shape[1])[0], d) \
            if seqlen_q_rounded == s_q else None
        scal, post_scale = bwd1_fp8_doc_scalars(softmax_scale, p_shift, dq_boost, dsc["q"], dsc["k"], dsc["v"],
                                                dsc["do"], dsc["k_global"], dq_range=dq_range, dq_docs=dq_docs)
    else:
        scal = bwd1_fp8_scalars(q.device, softmax_scale, p_shift, dq_boost, sq, sk, sv, sdo, s_ds=s_ds_dev,
                                dq_range=dq_range, dq_tiles=dq_tiles)
        if dq_tiles is not None:
            from flash_attn.cute.device_scales import dq_tile_planes
            post_scale = dq_tile_planes(scal, dq_tiles)
    # sfd: the dense per-packed-row statistics and exponent rows (the kernel's sf_dense)
    _sfd = _sfd_args(_pk, mxq)
    if _sfd is None and pt_docs and (glue or ext64):   # cov1: per-tensor operands on packed rows (the statistics rows)
        _sfd = _sfd_args_pt(_pk, lse_log2, dpsum)
    if _sfd is not None:
        SFD_CALLS[0] += 1
        lse_log2, dpsum = _sfd[0].view(b, h, -1), _sfd[1].view(b, h, -1)
    if quant.ds_fold and not pt_docs:  # kernel: dS' = P' FMA(d\hat{P}, b, D''), D'' = -b D / (s_v s_dO)
        from flash_attn.cute.device_scales import apply_d_fold
        apply_d_fold(dpsum, scal, 6, mxq["do"].get("d_folded")
                     if (mxq is not None and "do" in mxq.get("owned", ())) else None)
    # Bwd1Spec.ds_sr (sr_ds): a fresh dither seed per call, so the dS^ rounding error is unbiased across steps rather than a
    # fixed pattern; 0 keeps round-to-nearest
    ds_sr_seed = _ds_sr_seed(spec)
    scalars = [Float32(1.0)] * 3 + [Int32(ds_sr_seed)] + [Float32(1.0)] * 3  # ignored: the kernel reads `scal`
    ds_fold = bool(quant.ds_fold)

    # ---- compile key: (kernel class, spec.key(), the call's dtypes / shape-class flags, the resolved constructor
    # constexprs (`resolved_key`), arch, DSL version). Every knob the constructor reads is a Bwd1Spec field, so
    # spec.key() covers it by construction (tests/cute/test_lowprec_spec_keys.py) -- the hand-kept opt list that once
    # missed DO_STAGE, DS_SR, ... cannot come back. do8.dtype: the gradient operand may be E4M3 or E5M2 and the
    # kernel specializes its dP / dV MMAs on it. (Every setting reaches the kernel through the spec.)
    kern = FlashAttentionBackwardSm100FP4(d, d, is_causal=causal, tile_m=m_block_size, tile_n=n_block_size, deterministic=deterministic, qhead_per_kvhead=r_gqa,
                                          dq_reduce_dtype=cutlass.Float16 if dq_f16 else Float32, seqused_k=seqused_k is not None,
                                          out_pass=out_pass, ds_dq_rowlocal=ds_dq_rowlocal, **_mask_ctor(masks),
                                          token_local_operands=bool(mxq is not None and mxq.get("tok") is not None),
                                          det_issue_warp=spec.det_issue_warp, det_issue_bufs=spec.det_issue_bufs,
                                          spec=spec, **({"doc_scaled": True} if pt_docs else {}),
                                          **({"dq_tile_scaled": True} if dq_tiles is not None else {}),
                                          **({"dq_tile_causal": 2 if dq_tiles.get("rowwise") else 1} if (dq_tiles is not None and dq_tiles.get("causal")) else {}),
                                          **({"dkv_rows_packed": True} if _dkv_rows is not None else {}),
                                          **({"codes_packed": True} if _pk else {}),
                                          **({"sf_dense": True} if _sfd is not None else {}),
                                          **({"sf_persist": True} if _sfd is not None and _sfd_persist(masks) else {}))
    if kern.det_gqa_rot:
        _check_det_gqa_rot(q, k)
    _check_fp4_p_shift(kern)
    # qk_hadamard (fp8_scales["had_signs"]): q8 / k8 are rotated; the dK epilogue and the dQ postprocess undo it
    from flash_attn.cute.qk_hadamard import fuse_backrot
    _had = fuse_backrot(fp8_scales)
    if _had:
        from flash_attn.cute.qk_hadamard import signs_tuple
        assert d == 128 and kern.fp8, "qk_hadamard: d = 128"
        kern.had_dk = signs_tuple(128)
    tok = mxq.get("tok") if mxq is not None else None  # token-local operands: Q / K / V / dO
    wl = _bwd_worklist(masks, deterministic, h, r_gqa, d, host)
    key = (type(kern).__qualname__, spec.key(), "fp8", _had, tok is not None, _mask_key(masks), wl is not None, do8.dtype,
           *(("doc_scaled",) if pt_docs else ()), *(("dq_tiles",) if dq_tiles is not None else ()),
           *(("dq_causal", bool(dq_tiles.get("rowwise"))) if (dq_tiles is not None and dq_tiles.get("causal")) else ()),
           out_pass, seqused_k is not None, r_gqa, dtype, d, d, causal, deterministic,
           m_block_size, n_block_size, dq_f16, ds_fold, p_shift, kern.sf_vec_size, _arch_key(),
           resolved_key(kern, ("dk8", "dq8", "dv8", "dp8", "mxds", "mxds_row", "mxop", "mxop_s", "mxdv", "tmem_ld_rep",
                                     "mxds_dq_sf", "mxp", "fp8", "ld_rep", "warp_sync", "ds_sr",
                                     "sf_vec_size", "tile_m", "tile_n", "out_pass", "pp",
                                     "deterministic", "qhead_per_kvhead", "dq_reduce_dtype", "lpt", "spt", "det_issue_warp_arg", "det_issue_bufs_req", "ds_dq_rowlocal", "sf_asm", "sf_asm_q", "dkv_rows_packed", "codes_packed", "sf_dense", "bwd_persist")))
    mx_args = [None] * 9
    if mxq is not None:
        tb = mx_cols(masks, quant=quant, host=host) == 128
        # token-local Q / K: the quantisers wrote unit dK / dQ-side scales (f_Q / f_K fold into dS in the kernel)
        _sfqd, _sfkd = mx_sf_arg(mxq["q"]["sf_dim_raw"]), mx_sf_arg(mxq["k"]["sf_dim_raw"])
        mx_args = [None if tb else mx_sf_arg(mxq["q"]["sf_tok_raw"]), None if tb else mx_sf_arg(mxq["k"]["sf_tok_raw"]),
                   None if tb else mx_sf_arg(mxq["v"]["sf_tok_raw"]), None if tb else mx_sf_arg(mxq["do"]["sf_tok_raw"]),
                   _sfqd, _sfkd, mx_sf_arg(mxq["do"]["sf_dim_raw"]),
                   mx_factors(mxq) if tb else None, tok]
    if key not in _flash_attn_bwd_fp4.cache:
        assert host.bwd_op_mode == "fp8", "the 8-bit backward needs HostSpec.bwd_op_mode='fp8' (kernel constexpr)"
        q8_t, k8_t, v8_t, do8_t = [to_cute_tensor(t, divisibility=16) for t in (q8, k8, v8, do8)]
        dk_t, dv_t = [to_cute_tensor(t, divisibility=8) for t in (dk, dv)]
        lsel_t, dps_t, dqa_t = [to_cute_tensor(t, divisibility=4) for t in (lse_log2, dpsum, dq_accum)]
        sem_t = to_cute_tensor(dQ_semaphore, assumed_align=4) if deterministic else None
        assert kern.fp8
        _flash_attn_bwd_fp4.cache[key] = cute.compile(
            kern, q8_t, k8_t, do8_t, _make_fp4_ptr(None), _make_fp4_ptr(None), v8_t, _make_fp4_ptr(None),
            None, None, None, None, lsel_t, dps_t, dqa_t, dk_t, dv_t, *scalars, *_mask_window(masks, True), sem_t,
            (), (), (), (), None, None, None, None, (), (),
            to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0) if seqused_k is not None else None,
            *[to_cute_tensor(t, assumed_align=16) if t is not None else None for t in mx_args],
            _DOC_SCALARS_T(q.device) if pt_docs else _SCALARS_T(q.device), *_mask_tables(masks, True, wl),
            current_stream, *_pk_tail(masks, True), *_sfd_tail(_sfd, True), options="--enable-tvm-ffi",
        )
    _flash_attn_bwd_fp4.cache[key](
        q8, k8, do8, _make_fp4_ptr(None), _make_fp4_ptr(None), v8, _make_fp4_ptr(None),
        None, None, None, None, lse_log2, dpsum, dq_accum, dk, dv, *scalars, *_mask_window(masks), dQ_semaphore,
        (), (), (), (), None, None, None, None, (), (),
        seqused_k, *mx_args, scal, *_mask_tables(masks, False, wl), current_stream, *_pk_tail(masks, False),
        *_sfd_tail(_sfd, False),
    )
    if not dq_f16:
        # `dq-fp32-partials-unscaled`: the kernel folds `dq_scale` into the partials only before an f16 reduce, so fp32
        # partials arrive in quantised units -- dQ came back dq_scale^-1 too large (x14418 measured, cosine unchanged).
        # Bring them to the f16 path's units here, so the postprocess and a `return_accum` caller see one convention.
        dq_accum.mul_(scal[5])
    if host.dq_post == "cute":
        from flash_attn.cute.dq_postprocess_cute import dq_postprocess_cute
        dq = dq_postprocess_cute(dq_accum, s_q, post_scale if post_scale is not None else
                                 (scal[7:8] if dqb_dyn else softmax_scale / dq_boost), q.dtype, had=_had,
                                 **({"tile_scale": True} if (dq_tiles is not None or dq_docs is not None) else {}),
                                 **({"row_scale": True} if (dq_tiles is not None and dq_tiles.get("rowwise")) else {}),
                                 **_pk_row_tile(masks, dq_tiles if dq_tiles is not None else dq_docs))
    else:
        dq = dq_post_torch(dq_accum, s_q, **_pk_row_tile(masks, dq_tiles if dq_tiles is not None else dq_docs), post=post_scale if post_scale is not None else
                           (scal[7:8] if dqb_dyn else softmax_scale / dq_boost)).to(
            q.dtype).permute(0, 2, 1, 3).contiguous()
    if _had:
        dk._had_unrotated = True
    if return_accum:
        return dq, dk, dv, dq_accum
    return dq, dk, dv


_CARRIER_BASE = {}


def packed_docs(masks) -> bool:
    """An lp_docmaps packed-rows call (lp_docmaps.PACKED_ROWS): codes / accumulator in the caller's packed rows."""
    return masks is not None and bool(getattr(getattr(masks, "docs", None), "doc_maps_packed", False))


def _pk_aligned(masks, s_q: int, s_k: int) -> tuple:
    """(s_q, s_k) of the per-tile statistics / tables: the aligned layout's S_q / S_k on a packed-rows call (its
    accumulator and codes are the T packed rows, the per-128-row-tile maxima and tables per aligned tile)."""
    if packed_docs(masks):
        return int(masks.docs.s_q), int(masks.docs.s_k)
    return s_q, s_k


def _pk_row_tile(masks, dq_tiles) -> dict:
    """The dQ postprocess's per-row tile map on a packed-rows call with the per-row pre-scale (dqfix): accumulator row
    q = packed row q, aligned row DocMaps.dst_q[q]."""
    if dq_tiles is None or not packed_docs(masks):
        return {}
    return {"row_tile": masks.docs.doc_maps.dst_q}


def _pk_tail(masks, compile_time: bool, seed: bool = True) -> tuple:
    """The 1-CTA kernel's trailing arguments of a packed-rows call: (mSeed slot when `seed`, mVtQ); () otherwise."""
    if not packed_docs(masks):
        return ()
    vt = masks.docs.doc_maps.vt_tile_q
    vt = to_cute_tensor(vt, assumed_align=4) if compile_time else vt
    return ((None,) if seed else ()) + (vt,)


SFD_CALLS = [0]   # (tests / probes) backward calls that took the sf_dense kernel


def _sfd_args(pk: bool, mxq, sf4=None):
    """sfd (lp_docmaps.SF_DENSE): (lse rows, D rows, [EvQ, EvK, EvV, EvdO, SF4Q, SF4K]) of a packed-rows call whose
    quantisers wrote the dense per-row tables, else None."""
    if not pk or mxq is None:
        return None
    do = mxq.get("do") or {}
    ev = [(mxq.get(n) or {}).get("ev_rows") for n in ("q", "k", "v", "do")]
    if do.get("lse_rows") is None or do.get("D_rows") is None or any(e is None for e in ev):
        return None
    return do["lse_rows"], do["D_rows"], ev + (list(sf4) if sf4 is not None else [])


_SFD_PT_DUMMY = {}


def _sfd_args_pt(pk: bool, lse_rows, d_rows, sf4=None):
    """cov1: the sf_dense arguments of a packed-rows call with per-(document, kv head) per-tensor operands (doc_scaled):
    the per-tensor dO launch's dense statistics rows (lse rows, D'' rows; quant.fp8.quantize_fp8_per_tensor's packed
    documents mode) and no exponent rows (one-word placeholders the kernel never reads), [+ the NVFP4 S words]."""
    if not pk or lse_rows is None or d_rows is None:
        return None
    ph = sfd_placeholder(lse_rows.device)
    return lse_rows, d_rows, [ph] * 4 + (list(sf4) if sf4 is not None else [])


def sfd_placeholder(dev) -> torch.Tensor:
    """A one-word (1, 16) uint8 stand-in for an sf_dense exponent-row table the kernel never reads (per-tensor
    operands: no E4M3 exponent rows). Cached per device."""
    if dev not in _SFD_PT_DUMMY:
        _SFD_PT_DUMMY[dev] = torch.zeros(1, 16, dtype=torch.uint8, device=dev)
    return _SFD_PT_DUMMY[dev]


def _sfd_tail(sfd, compile_time: bool) -> tuple:
    """The kernel's trailing mSfd argument (a list) of an sf_dense call; () otherwise."""
    if sfd is None:
        return ()
    ts = sfd[2]
    if compile_time:
        ts = [to_cute_tensor(t, assumed_align=4) for t in ts]
    return (ts,)


def shape_carrier(shape, dtype, device) -> torch.Tensor:
    """A zero-memory stand-in for a bf16 operand the default dk8 backward never reads: one element expanded to `shape`
    (right shape / dtype / device / numel, every stride 0). With q8 / k8 / v8 / do8, q4 / k4, D and P_max supplied, the
    backends use bf16 q, k, v only for metadata -- so the training path need not keep them alive.
    The 0-dim base is cached per (dtype, device): building a carrier is then a single `expand`, no allocation."""
    key = (dtype, device)
    base = _CARRIER_BASE.get(key)
    if base is None:
        base = _CARRIER_BASE[key] = torch.empty((), dtype=dtype, device=device)
    return base.expand(shape)


def max_row_norm(v: torch.Tensor) -> torch.Tensor:
    """max over (b, s, h) of ||v||_2 along head_dim, as a 0-dim float32 tensor (no host sync)."""
    return torch.linalg.vector_norm(v, dim=-1, dtype=torch.float32).amax()


def _is_carrier(t: torch.Tensor) -> bool:
    return t.dim() > 0 and t.numel() > 1 and all(st == 0 for st in t.stride())


def contig(t: torch.Tensor) -> torch.Tensor:
    """.contiguous() that leaves a shape_carrier alone (it would otherwise materialise the full tensor)."""
    return t if _is_carrier(t) else t.contiguous()


def _check_det_gqa_rot(q, k):
    """Bwd1Spec.det_gqa_rot (bwdk opt-in): the rotated walk's ranks are a permutation only when every m block has
    exactly nm = s_k / 128 writers, and its dependencies point at the next key tile of the head, so all of them must be
    co-resident (one CTA per SM, head-contiguous grid)."""
    s_q, s_k = q.shape[1], k.shape[1]
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    if s_q != s_k or s_k % 128 != 0 or s_k // 128 > sms:
        raise ValueError(f"det_gqa_rot: needs s_q == s_k, s_k % 128 == 0 and s_k / 128 <= {sms} SMs (got s_q {s_q}, "
                         f"s_k {s_k})")


def _use_2cta_dk8(q, k, causal, deterministic, tile_m, return_accum, seqused_k, out_pass, v4, do4, qt4, dot4, masks=None,
                  quant: Optional[QuantSpec] = None, host: Optional[HostSpec] = None) -> bool:
    """Route the dk8 backward to the 2-CTA kernel (flash_bwd_sm100_2cta8): MHA, head_dim 128, dense, the default dk8 operand
    configuration. two_cta="auto" (default): where it measured faster on the 52-cell matrix (s >= 16384: -3.6..-6.9 %;
    non-causal s = 8192: -1.2..-2.5 %; shorter / causal s <= 8192 keep the 1-CTA kernel, 2-9 % faster there); "on" = always
    when eligible; "off" = never."""
    # HostSpec.bwd_two_cta is "auto" | "on" | "off"; "0" / "1" are accepted as the same values.
    # Both spellings are honoured -- "off" used to fall through to the auto heuristic.
    quant, host = _quant(quant), _host(host)
    mode = host.bwd_two_cta  # "auto" | "on" | "off" (HostSpec: both spellings parsed)
    if mode in ("0", "off", "false"):
        return False
    if lp_masks_or_none(masks, causal) is not None and not _2cta_takes_masks(masks):
        return False
    _m = lp_masks_or_none(masks, causal)
    # documents packed at 256 (no window / mods): the 2-CTA kernel takes any total_q / total_k and, deterministic, its
    # document-local dQ lock (flash_bwd_sm100_2cta8 _dq_semaphore_lock_value); the plan reaches it with two_cta='on'
    _docs_only = _m is not None and _m.docs is not None and not _m.local and not _m.any_mod
    if deterministic and _m is not None and not _docs_only:
        return False  # the 2-CTA kernel's dQ lock order is built for batched causal / dense tiles and documents only
    b, s_q, h, d = q.shape
    s_k = k.shape[1]
    # The cluster kernel hardcodes P_SHIFT_LOG2 = 8 (interface_bwd_2cta8.py), which is the FP8
    # route's default and the FP4 route's "no in-kernel shift". Test the name each route uses:
    # the FP8 route's shift used to be judged on the FP4 route's name, so a user tuning the FP4 shift
    # silently lost 2-CTA and an FP8 user could never lose it (both shifts are constants now).
    fp8_mode = host.bwd_op_mode == "fp8"
    # Both routes: the cluster kernel runs E4M3 P^ at exactly 2^8 (`e4m3_p_shift_log2`, now a constant)
    p_shift_ok = e4m3_p_shift_log2() == P8_SHIFT_2CTA
    # block-scaled d\hat{S} (QuantSpec.ds_scaling_fp8): the 2-CTA kernel has the blk32 rung only
    mxds_1cta_only = fp8_mode and quant.ds_scaling_fp8 not in ("per_tensor", "blk32", "blk32x64", "tile128")  # mx32 / row
    # `2cta-e5m2-grad-mistyped`: the cluster kernel builds every FP8 GEMM with dO's element type, so an E5M2 gradient
    # reads the E4M3 V / P^ / dS^ as E5M2 (dQ, dK NaN). Only the FP8 route quantises dO to `grad_fp8_dtype()`; the FP4
    # routes always use E4M3. An E5M2 gradient takes the 1-CTA FP8 kernel, which types dO on its own.
    grad_e4m3 = (not fp8_mode) or grad_fp8_dtype(quant) is torch.float8_e4m3fn
    # token-block scales: the 32 x 32 block-scaled E4M3 operands and the dk8 route's mx32 P^ exist on the 1-CTA kernel only;
    # the token-block variant (blk32x128) has a 2-CTA port
    if (op_blk32_fp8(quant) if fp8_mode else (e4_blk32_fp4(quant) or quant.p_scaling_fp4 != "row")) \
            and mx_cols(quant=quant, host=host) != 128:
        return False
    if (not fp8_mode) and quant.p_scaling_fp4 != "row":
        return False
    # GQA (KernelTuning.bwd_2cta_gqa): fp32 dK / dV reduce-adds, ordered by q head when deterministic (dK / dV
    # semaphores); documents packed at 128 or 256
    # batched s_q != s_k and s_k % 256 == 128 (the last pair's upper half masked: flash_bwd_sm100_2cta8 key_tail)
    _k_ok = s_k % 128 == 0 and s_q % 128 == 0
    eligible = (d == 128 and (k.shape[2] == h or host.bwd_2cta_gqa)
                and tile_m == 128 and not return_accum
                and seqused_k is None and out_pass == "all" and _k_ok
                and p_shift_ok and not mxds_1cta_only and grad_e4m3)
    if not eligible:
        return False
    if mode in ("1", "on", "true"):
        return True
    _op8 = quant.operand_scaling_fp8 if fp8_mode else None
    if deterministic and fp8_mode and (_op8 == "per_tensor" or (
            _op8 is None and quant.ds_scaling_fp8 not in ("blk32", "mx32"))):
        # by design: 8-bit per-tensor operands (explicit, or the per-tensor-dS fallback) keep the 2-CTA ordered dQ
        # reduce when deterministic: the 1-CTA kernel differs there in dK and is ~1.1e-4 less accurate in dQ / dK cosine
        # on causal shapes (lowprec.plan._resolve_two_cta)
        return True
    if deterministic:
        # the 1-CTA deterministic kernel (dQ issue / release warps) beats the 2-CTA ordered dQ reduce except at
        # causal s_k >= 32768, where the two tie (lowprec.plan._resolve_two_cta; O / dK / dV bitwise either way on the
        # token-block / block-scaled and 4-bit routes)
        return bool(causal) and s_k >= 32768
    if fp8_mode:
        # The s >= 16384 rule was measured on the dk8 (NVFP4 S) kernels. On the all-E4M3 pair the
        # cluster kernel wins everywhere it is eligible: +12 % causal / +9 % non-causal at s4096
        # (1326 vs 1186, 1709 vs 1562 TF), +24 % / +17 % at s32768.
        return True
    # the 4-bit step with token-block E4M3 operands (QuantSpec.operand_scaling_fp4 = blk32x128, the default) and the
    # block-scaled dS takes the cluster kernel wherever eligible. Measured (published benchmark, FP4
    # step, 1700 MHz, paired, 2 rounds): 2-CTA vs 1-CTA -3.6 / -3.3 % at b8 s2048 causal / dense, -5.2 / -3.6 % at
    # b4 s4096, -6.3 % at b2 s8192 causal (dense s8192 and s16384 already took it); dK / dV / O / LSE bitwise the 1-CTA
    # kernel's (deterministic calls stay on the 1-CTA kernel). The threshold below was measured on per-tensor operands.
    if e4_blk32_fp4(quant) and mx_cols(quant=quant, host=host) == 128:
        return True
    return s_k >= 16384 or (not causal and s_k >= 8192)


def ds_blk32_fp4(out_pass: str = "all", quant: Optional[QuantSpec] = None) -> bool:
    """QuantSpec.ds_scaling_fp4 (dk8 route): "blk32" (default: in-kernel UE8M0 per 32 x 32 dS block, the FP8 route's rung)
    or "per_tensor" (the dS bound s_ds). Read by the 1-CTA kernel (`mxds_fp4`), the 2-CTA kernel and this driver. True
    only where it takes effect: the E4M3 dK / dQ route (dk8) on the single-pass d = 128 kernel; head_dim > 128 (out_pass
    dk / dq / dv) keeps the per-tensor dS bound (the kernel's `mxds_fp4` resolves the same way)."""
    v = _quant(quant).ds_scaling_fp4
    if v not in ("per_tensor", "blk32"):
        raise ValueError(f"QuantSpec.ds_scaling_fp4 must be 'per_tensor' or 'blk32', got {v!r}")
    return v == "blk32" and out_pass == "all"


# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def e4_blk32_fp4(quant: Optional[QuantSpec] = None) -> bool:
    """QuantSpec.operand_scaling_fp4 (dk8 route): the E4M3 copies of Q, K, V, dO (the dK / dQ / dP / dV GEMM operands).
      "blk32x128" (default, specs.OP_SCALING_DEFAULT): one UE8M0 exponent per 32 tokens x the whole head dim
                  (quantize_fp8_blocks block_columns=128); the S / dP GEMMs need no scale factor (the factors fold into the softmax /
                  dS arithmetic, flash_bwd_sm100_fp4.mxop_tb), the others take MXFP8 SF tiles.
      "blk32":    one exponent per 32 tokens x 32 d (MXFP8 SF on every GEMM).
      "per_tensor": one amax-derived scale per tensor (the pre-2026-09-30 default; set explicitly it restores those
                  results bitwise).
    Needs the blk32 dS (QuantSpec.ds_scaling_fp4 = blk32). The default resolves per call (resolve_op_scaling): d != 128,
    s % 128 != 0, seqused_k, the out_pass kernels, tile_m != 128 -> per_tensor. Measured (B200, public graph step,
    vs per_tensor): O / dQ / dK / dV cosines within 5e-5 (no per-tensor amax: the amax-leak probe is flat),
    step +7..12 % (fwd +21..37 % from the MXFP8 forward it needs, bwd +0..2 %)."""
    v = _quant(quant).operand_scaling_fp4
    v = OP_SCALING_DEFAULT if v is None else v
    if v not in ("per_tensor", "blk32", "blk32x128"):
        raise ValueError(f"QuantSpec.operand_scaling_fp4 must be 'per_tensor', 'blk32' or 'blk32x128', got {v!r}")
    return v != "per_tensor"


# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def op_blk32_fp8(quant: Optional[QuantSpec] = None) -> bool:
    """QuantSpec.operand_scaling_fp8 (FP8 route): Q, K, V, dO scaling of the backward's E4M3 operands.
      "blk32x128" (default, specs.OP_SCALING_DEFAULT): 32 tokens x the whole head dim per UE8M0 exponent;
      "blk32": 32 x 32 MXFP8; "per_tensor": the pre-2026-09-30 default (bitwise when set explicitly).
    Needs the block-scaled dS (QuantSpec.ds_scaling_fp8 blk32 / mx32) and E4M3 gradients; otherwise, and for the shapes
    listed in resolve_op_scaling, the default resolves to per_tensor. Measured (B200, public graph step, vs
    per_tensor): step +8..12 % (mostly the MXFP8 forward), cosines within 5e-5, no amax leak."""
    v = _quant(quant).operand_scaling_fp8
    v = OP_SCALING_DEFAULT if v is None else v
    if v not in ("per_tensor", "blk32", "blk32x128"):
        raise ValueError(f"QuantSpec.operand_scaling_fp8 must be 'per_tensor', 'blk32' or 'blk32x128', got {v!r}")
    return v != "per_tensor"


# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def mx_cols(masks=None, quant: Optional[QuantSpec] = None, host: Optional[HostSpec] = None) -> int:
    """Columns per exponent of the block-scaled E4M3 operands: 32 (blk32, 32 x 32 blocks) or 128 (blk32x128: 32 tokens x
    the whole head dim -- the S / dP GEMMs then need no SF, see flash_bwd_sm100_fp4.mxop_tb)."""
    if _docs(masks):
        return 128  # varlen: token blocks (a per-tensor scale over all documents would be a cross-document channel)
    fp8 = _host(host).bwd_op_mode == "fp8"
    quant = _quant(quant)
    v = quant.operand_scaling_fp8 if fp8 else quant.operand_scaling_fp4
    v = OP_SCALING_DEFAULT if v is None else v
    return 128 if v == "blk32x128" else 32


SCALING_RULE = ("SCALING-CONSISTENCY RULE: forward and backward must use the same scaling (format, granularity and scale "
                "values) of every operand they share (Q, K, V); the backward reuses the forward's codes and never "
                "re-quantizes a shared operand in another scaling. Mixed designs (e.g. a per-tensor forward with a "
                "block-scaled backward) are not supported.")


def operand_scaling_of(fp8_scales: Optional[dict]) -> Optional[str]:
    """The scaling of the shared-operand codes a caller (normally the autograd forward) hands the backward in fp8_scales:
    "per_tensor" (q8 / k8 / v8 with their scalars), "blk32" / "blk32x128" (fp8_scales["mx"] quantize_fp8_blocks dicts,
    by their "cols"; "mx" when a dict does not say), or None when no Q / K / V codes are given (the driver quantises
    them itself from this call's tensors, in the scaling it resolves). Both kinds at once is itself a mismatch."""
    sc = fp8_scales or {}
    pt = [n for n in ("q8", "k8", "v8") if n in sc]
    mx = {n: d for n, d in (sc.get("mx") or {}).items() if n in ("q", "k", "v")}
    if pt and mx:
        raise ValueError(f"{SCALING_RULE} Got per-tensor codes {pt} and block-scaled codes {sorted(mx)} in one call.")
    if pt:
        return "per_tensor"
    if mx:
        cols = {d.get("cols") for d in mx.values()} - {None}
        if len(cols) > 1:
            raise ValueError(f"{SCALING_RULE} The block-scaled codes {sorted(mx)} mix column widths {sorted(cols)}.")
        return {32: "blk32", 128: "blk32x128"}[cols.pop()] if cols else "mx"
    return None


def check_scaling_consistency(fwd: Optional[str], bwd: Optional[str], where: str) -> None:
    """THE check of the scaling-consistency rule (one place; called where the operand scaling is resolved for a call:
    op_scaling_config, i.e. every backward driver and every autograd forward). fwd: the scaling of the shared codes
    the forward produced / a caller passed (operand_scaling_of; None = none passed); bwd: the backward's resolved
    operand scaling. Raises ValueError naming the rule on any mismatch."""
    if fwd is None or bwd is None:
        return
    if fwd == bwd or (fwd == "mx" and bwd != "per_tensor"):
        return
    raise ValueError(f"{SCALING_RULE} {where}: the shared Q / K / V codes are {fwd}-scaled but the backward's operand "
                     f"scaling is {bwd!r}. Pass codes quantised in {bwd!r} (the plan's operand_scaling).")


# ---- option A (pv4_route.py): its research backward (the NVFP4 dP on the forward's V codes, the P-hat rebuild) is not
# shipped; option A is inference-only (docs/lowprec.md) and this driver refuses a training call.
OPTION_A_BWD_ARCHIVED = (
    "option A's backward (the 4-bit P.V training route: NVFP4 dP on the forward's V codes, reproduce-forward-p) is "
    "not part of this build. Option A runs as inference; train with pv_precision='fp8' or the all-4-bit backward "
    "(backward='fp4').")


# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def check_codes_scaling(fp8: bool, fp8_scales: Optional[dict], masks, quant: QuantSpec, where: str) -> None:
    """The scaling of the shared Q / K / V codes a caller hands this backward (operand_scaling_of) against the call's
    operand scaling, which the plan states in QuantSpec (lowprec.plan rule 16; varlen documents always take token
    blocks). Raises ValueError naming the scaling-consistency rule on a mismatch."""
    v = quant.operand_scaling_fp8 if fp8 else quant.operand_scaling_fp4
    v = "blk32x128" if (_docs(masks) and not doc_scaled(masks, quant, fp8)) else (OP_SCALING_DEFAULT if v is None else v)
    check_scaling_consistency(operand_scaling_of(fp8_scales), v, where)


def mx_factors(mxq: dict) -> torch.Tensor:
    """(4, b, h, max(s_q, s_k) / 32) fp32 = 2^(e - 127) per token block of Q, K, V, dO (K / V padded to the q-head
    count; rows Q / dO cover s_q / 32 columns, K / V s_k / 32: fp4_bwd_autograd.token_block_factor_table's layout)."""
    if mxq.get("fac") is not None:  # written by the quantisers themselves (fac_out): no extra launches
        return mxq["fac"]
    if any(mxq[n].get("e") is None for n in ("q", "k", "v", "do")):
        # (sfd compact maps: no per-block table; the sf_dense kernel takes its factors from the dense exponent rows --
        # a placeholder for the unused argument)
        cd = mxq["q"]["codes"]
        return torch.ones(4, cd.shape[0], cd.shape[2], 4, dtype=torch.float32, device=cd.device)
    es = [mxq[n]["e"][..., 0] for n in ("q", "k", "v", "do")]           # (b, h_x, s / 32) uint8
    b, h, _ = es[0].shape
    n = max(e.shape[2] for e in es)
    out = torch.zeros(4, b, h, n, dtype=torch.float32, device=es[0].device)
    for i, e in enumerate(es):
        out[i, :, :e.shape[1], :e.shape[2]] = torch.exp2(e.float() - 127.0)
    return out


def mx_sf_arg(raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """(b, h, R/128, K/128, 32, 4, 4) raw SF atoms (quantize_fp8_blocks `*_raw`, d = 128 so one of R/128, K/128 is 1) ->
    (b, h, n_tiles, 512) uint8, the kernel's per-128-token-tile atom view."""
    if raw is None:
        return None
    b, h = raw.shape[0], raw.shape[1]
    return raw.reshape(b, h, -1, 512)


def mx_operands(q=None, k=None, v=None, dout=None, out=None, want_tok_qk=False, masks=None,
                quant: Optional[QuantSpec] = None, host: Optional[HostSpec] = None):
    """32 x 32 MXFP8 copies (codes + both SF replications) of the operands given; dO with D = sum(dequant(dO) O). K, V
    and dO also carry their statistics (amax, row_norm2_max: the f16 dQ range, dq_range_of) from the same launch."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    res = {}
    cols = mx_cols(masks, quant=quant, host=host)
    for nm, t in (("q", q), ("k", k), ("v", v)):
        if t is not None:
            res[nm] = dict(quantize_fp8_blocks(t, block_columns=cols, output_amax=nm == "k",
                                               output_row_norm2_max=nm == "v"), cols=cols)
    if dout is not None:
        res["do"] = dict(quantize_fp8_blocks(dout, compute_d_with_output=out, block_columns=cols,
                                             output_row_norm2_max=True), cols=cols)
    return res


def dq_range_of(sc, block_scaled: bool):
    """`dq-f16-range`: (dn2, vn2, kmax) for device_scales._dq_prescale, or None (the route's fixed pre-scale stays) --
    float32 device slots from the quantisers' own reads: max ||dO_q||^2 ("dn2"), max ||V_k||^2 ("vn2"), and max|K|
    ("kmax"; None for per-tensor E4M3 K, whose 448 s_k bounds it). Block-scaled K has no per-tensor scale, so it needs
    "kmax"."""
    if not sc:
        return None
    dn2, vn2, kmax = sc.get("dn2"), sc.get("vn2"), sc.get("kmax")
    if dn2 is None or vn2 is None or (block_scaled and kmax is None):
        return None
    return dn2, vn2, (kmax if block_scaled else None)


# dqfix (`dq-f16-range-per-row`): the f16 dQ pre-scale per row (a document, or a batch row) instead of
# ONE per call. With one pre-scale sized from max ||dO|| max ||V|| max|K| over the whole call, a document (or batch row)
# with large activations pushed every other document's f16 dQ partials into the subnormal / zero range (x1000 on one
# document: 76-98 % of the others' dQ flushed to 0). Off restores the one-per-call pre-scale (57cab39) bitwise.
DQ_TILE_PRESCALE = True
# dq-causal: with the per-row / per-tile pre-scale table, the f16 dQ pre-scale of query row i (tile t) is sized from that
# row's own dO and the keys <= i (the prefix of key tiles 0..t) only (causal square batched calls without window / mods /
# documents). The per-call (per-row of the batch) pre-scale reads the whole sequence's max ||V|| and max|K|, so a future
# key that raises either moved the f16 dQ rounding of EVERY earlier query (relative 5e-4 default, 1.5e-5 exact options at a
# K / V row above the tensor amax). False (default) restores the per-batch-row pre-scale bitwise; set True to enable
# (a module switch like DQ_TILE_PRESCALE; promoting it to an options field is the follow-up).
DQ_CAUSAL_PRESCALE = False


def _fuse_backrot(fp8_scales):
    from flash_attn.cute.qk_hadamard import fuse_backrot
    return fuse_backrot(fp8_scales)


def is_carrier(t) -> bool:
    """A shape_carrier (zero-memory stand-in for an operand the backward never reads): every stride 0."""
    return t is None or (t.numel() > 1 and all(st == 0 for st in t.stride()))


def dq_causal_kv_rows(k, v):
    """dq-causal: (vn2c, kmc, kmc2), float32 (b * s,) each -- the running max over keys j <= i (and the kv heads) of ||V_j||^2,
    of max|K_j| (unrotated K) and of ||K_j||_2 (a rotation-invariant bound of max|K_j R|: the Hadamard routes), from the bf16
    K / V (the same values the quantisers' statistics read). One read of each (Triton row statistics), one prefix-max launch
    over the three arrays; no host sync. None where K / V are not real tensors or d != 128."""
    if is_carrier(k) or is_carrier(v) or k.shape[-1] != 128 or v.shape[-1] != 128:
        return None
    from flash_attn.cute.device_scales import row_stats, prefix_max_
    b, s = k.shape[0], k.shape[1]
    kn2, kam = row_stats(k, True)
    vn2 = row_stats(v)
    st = torch.stack([vn2, kam, kn2.sqrt_()])   # (3, b * s)
    prefix_max_(st.view(3 * b, s), s)
    return st[0], st[1], st[2]


def dq_kv_rows_of(fp8_scales, k, v, causal) -> Optional[tuple]:
    """The caller's (forward-saved) dq_kv_rows, else computed here from real bf16 K / V; None when off / not causal."""
    if not DQ_CAUSAL_PRESCALE or not causal:
        return None
    kv = (fp8_scales or {}).get("dq_kv_rows")
    return kv if kv is not None else dq_causal_kv_rows(k, v)


def dq_causal_rows(kv_rows, dout, rotated: bool = False):
    """dq-causal, per-row statistics of the kernels: (dn2, vn2c, kmc), float32 (b * s,) each -- max ||dO_i||^2 over
    the q heads of row i and kv_rows' running maxima over keys j <= i (||K||_2 where K is Hadamard-rotated). Row i's f16
    dQ pre-scale then depends on keys the query may see only (device_scales._dq_tile_table, CAUSAL 2)."""
    from flash_attn.cute.device_scales import row_stats
    return row_stats(dout), kv_rows[0], kv_rows[2 if rotated else 1]


def dq_tiles_of(dq_range, masks, b: int, h: int, h_k: int, s_q: int, s_k: int, tile_m: int = 128,
                causal: bool = False, rows_src=None):
    """The per-row pre-scale's shape facts (device_scales._dq_tile_table), or None (the one-per-call pre-scale stays):
    block-scaled operands whose statistics are the quantisers' per-128-row-tile maxima (dn2 (b h s_q / 128,), vn2 /
    kmax (b h_k s_k / 128,), contiguous), 128-row query tiles. Documents (b = 1): one row per document + the sentinel,
    from the aligned tables doc_tab; batched: one row per batch row."""
    if not DQ_TILE_PRESCALE or dq_range is None or dq_range[2] is None or tile_m != 128:
        return None
    if s_q % 128 or s_k % 128:
        return None
    tq, tk = s_q // 128, s_k // 128
    dn2, vn2, kmax = dq_range
    for x, n in ((dn2, b * h * tq), (vn2, b * h_k * tk), (kmax, b * h_k * tk)):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.numel() != n or x.dim() != 1 \
                or (n > 1 and x.stride(0) != 1):
            return None
    tab = None
    rows = b
    if _docs(masks):
        tab = masks.docs.doc_tab
        if b != 1 or tab is None or not tab.is_contiguous() or tab.dtype != torch.int32 or tab.shape[0] < 4:
            return None
        rows = int(tab.shape[1])
    # dq-causal: causal square batched rows, no window / mods: per-(row, query tile) causal-prefix pre-scale
    causal_tiles = bool(DQ_CAUSAL_PRESCALE and causal and tab is None and s_q == s_k
                        and (masks is None or (not masks.local and not masks.any_mod and masks.docs is None)))
    out = dict(tab=tab, rows=rows, hq=h, hk=h_k, tq=tq, tk=tk, n=b * tq, causal=causal_tiles)
    if causal_tiles and rows_src is not None and rows_src[0] is not None and not is_carrier(rows_src[1]):
        # a scale per query ROW (exactly causal) from the forward's / this call's per-row prefix K / V statistics
        out.update(rowwise=True, n=b * s_q, rng=dq_causal_rows(*rows_src))
    return out


def dq_docs_of(dq_range, masks, b: int, h: int, s_q: int, d: int):
    """dqfix, per-(document / batch row, kv head) per-tensor operands (doc_scaled): the per-document pre-scale's facts
    (device_scales._doc_prescale, in the per-document scalar launch), or None. The postprocess then reads a (b tq,)
    plane: each 128-row tile's document (tile_doc_q) or batch row."""
    if not DQ_TILE_PRESCALE or dq_range is None or s_q % 128:
        return None
    tdq = None
    if _docs(masks):
        tdq = getattr(masks.docs, "tile_doc_q", None)
        if b != 1 or tdq is None or tdq.numel() != s_q // 128:
            return None
    return dict(tile_doc_q=tdq, hq=h, tq=s_q // 128, n=b * (s_q // 128), d=d)


def dq_post_torch(dq_accum: torch.Tensor, s_q: int, post, row_tile=None) -> torch.Tensor:
    """HostSpec.dq_post = "torch": accum (b, h, s, d) times the scale ((1,) / float, or the (b tq,) per-tile plane;
    row_tile: lp_docmaps packed rows, each row's aligned row -> its tile's scale)."""
    a = dq_accum[:, :, :s_q].float()
    if isinstance(post, torch.Tensor) and post.numel() == a.shape[0] * s_q and row_tile is None:   # dq-causal rows: one scale per (b, row)
        return a * post.view(a.shape[0], 1, s_q, 1)
    if row_tile is not None:
        return a * post[row_tile.long() // 128].view(1, 1, -1, 1)
    if isinstance(post, torch.Tensor) and post.numel() > 1:
        b_, h_, s_, d_ = a.shape
        return (a.view(b_, h_, s_ // 128, 128, d_) * post.view(b_, 1, -1, 1, 1)).view(b_, h_, s_, d_)
    return a * post

# ---------------------------------------------------------------- masks / varlen: explicit mask arguments (lp_masks.LPMasks)
def _mask_ctor(masks) -> dict:
    """Kernel constructor arguments; {} for the plain call (the constructor defaults, i.e. the existing kernel)."""
    if masks is None:
        return {}
    kw = masks.ctor_kwargs()
    kw.pop("is_causal")  # the call's own flag (masks.causal after resolution)
    return kw


def _mask_key(masks) -> tuple:
    return () if masks is None else masks.key()


def _mask_window(masks, placeholder: bool = False):
    """The kernel's window_size_left / _right slots (runtime Int32; compile-time presence)."""
    if masks is None:
        return None, None
    return masks.window_args(placeholder)


def _mask_tables(masks, compile_time: bool = False, work=None):
    """(mDocTab, mNDoc, mWork, aux_tensors) of the kernel call."""
    if masks is None:
        return None, None, None, None
    tab, ndoc = masks.table_args()
    aux = masks.aux_list()
    if compile_time:
        tab = None if tab is None else to_cute_tensor(tab, assumed_align=4, leading_dim=1)
        ndoc = None if ndoc is None else to_cute_tensor(ndoc, assumed_align=4, leading_dim=0)
        work = None if work is None else to_cute_tensor(work, assumed_align=4, leading_dim=0)
        aux = to_cute_aux(aux)
    return tab, ndoc, work, aux


def _bwd_worklist(masks, deterministic: bool, h: int, r_gqa: int, d: int, host):
    """Aligned documents, non-deterministic, no window / mask_mod (step 3, F6): the 1-CTA backward's document-aware
    longest-first key-tile list (lp_varlen.varlen_worklist 'bwd'; HostSpec.varlen_bwd_list), else None (the plain grid).
    Deterministic runs keep their scheduler: the dQ lock order is tied to it. dK / dV bitwise; only the arrival order
    of the (unordered) dQ partials changes."""
    from flash_attn.cute import lp_varlen as LV
    # causal only: dense key tiles are equal, and the list measured -0.3..-1.3 % there (paired A/B, step 3)
    if not (aligned_docs(masks) and not deterministic and host.varlen_bwd_list and masks.causal and not masks.local
            and masks.mask_mod is None):
        return None
    docs = masks.docs
    nh = h // r_gqa if r_gqa > 1 else h
    s_q = docs.max_seqlen_q if docs.max_seqlen_q is not None else docs.s_q
    # footprint per (query token, kv head): Q / dO (one byte each) + the f16 dQ accumulator, per streamed q head
    per_tok = 4 * d * r_gqa
    sec = LV.l2_heads(nh, int(s_q) * per_tok, host.varlen_l2_sections, LV._L2_BYTES_BWD)
    span = LV.l2_span(per_tok, sec, docs.s_q, nh, host.varlen_l2_sections, LV._L2_BYTES_BWD)
    return LV.varlen_worklist(docs, "bwd", bool(masks.causal), sec, pos_heads=nh, span=span)


def _sfd_persist(masks) -> bool:
    """sfd compact maps whose static per-position grid (the work list's n_max = the key tile bound min(T, ceil(T / 128)
    + N), x heads) exceeds the capped grid 5 / 4 ceil(T / 128) x heads: the backward launches the capped grid and runs
    the per-position code when the list's real positions fit it (read on the device), a persistent loop otherwise.
    (The per-position grid launched an empty CTA per position past the documents' tiles: N = 2048 at T = 131072, d 64
    h 32 / 8: +170 us, N = T: 1 M CTAs.)"""
    docs = getattr(masks, "docs", None) if masks is not None else None
    m = getattr(docs, "doc_maps", None)
    if m is None or not getattr(m, "compact", False):
        return False
    from flash_attn.cute.lp_docmaps import tile_bound
    tk = int(m.total_k)
    tq = -(-tk // 128)
    return tile_bound(tk, m.max_num_documents) > tq + (tq + 3) // 4


def _docs(masks) -> bool:
    return masks is not None and getattr(masks, "docs", None) is not None


def doc_scaled(masks, quant: Optional[QuantSpec], fp8: bool) -> bool:
    """Per-tensor operands on aligned varlen documents: one scale per (document, kv head) of every operand
    (lp_varlen.DocScales; the plan's explicit operand_scaling='per_tensor'). Every other document call runs on token
    blocks."""
    q = _quant(quant)
    v = q.operand_scaling_fp8 if fp8 else q.operand_scaling_fp4
    return aligned_docs(masks) and v == "per_tensor"


def aligned_docs(masks) -> bool:
    """The aligned varlen fast path's documents (lp_varlen.LPDocs.aligned_from)."""
    return _docs(masks) and bool(getattr(masks.docs, "aligned", False))


def dkv_zero_fill(masks) -> bool:
    """Whether dK / dV must start zero: a key tile with an empty query range (window / mask_mod / document tail) writes
    nothing. Aligned documents without a window or mask_mod skip the fill: every real key tile of a document with
    seqused_q > 0 has a query range (causal is bottom-right), and the padding rows are undefined under the padding
    contract (a document with seqused_q = 0 gets undefined dK / dV, documented)."""
    if masks is None:
        return False
    return not (aligned_docs(masks) and not masks.local and masks.mask_mod is None)


def _2cta_takes_masks(masks) -> bool:
    """The 2-CTA kernel's mask support: window, softcap, score_mod, mask_mod and documents packed at
    256 (a key pair never straddles a document); documents packed at 128 without score_mod / mask_mod (the pair
    straddles documents: flash_bwd_sm100_2cta8 doc_t128)."""
    if masks.docs is not None and masks.docs.align % 256 != 0:
        return masks.docs.align % 128 == 0 and not masks.any_mod
    return True


def _check_masks_ds(masks, per_tensor_ds: bool) -> None:
    """A per-tensor dS scale is a bound on max|dS| computed without the score_mod: a softcap (derivative <= 1), a
    window, a mask_mod or documents keep it an upper bound; an arbitrary score_mod_bwd does not."""
    if per_tensor_ds and masks.score_mod is not None and masks.softcap == 0.0:
        raise NotImplementedError("score_mod with a per-tensor dS scale: the dS bound does not cover score_mod_bwd; "
                                  "use the block-scaled dS (the default)")

_DS_SR_CALL = [0]
_DS_SR_COUNTER = {}


def ds_sr_counter(device) -> torch.Tensor:
    """The dk8 backward's stochastic-rounding counter (Bwd1Spec.ds_sr, options.sr_ds) on `device`: an int64 (1,) tensor
    starting at 1. Where the SR is live (the per-tensor dS^), the kernel derives the call's seed (c * 0x9E3779B1) mod 2^31
    from it -- the former host counter's formula, so a fresh process draws the same seed sequence -- and the dQ
    postprocess advances it by one on the GPU, so every call, and every replay of a captured step, dithers anew. Same
    scheme as the all-4-bit backward's interface_bwd_fp4q.sr_counter."""
    key = str(torch.device(device))
    c = _DS_SR_COUNTER.get(key)
    if c is None:
        c = _DS_SR_COUNTER[key] = torch.ones(1, dtype=torch.int64, device=device)
    return c


def reset_ds_sr_counter() -> None:
    """Restart every device's dk8 SR seed sequence at 1 (and the host counter of the routes that keep it)."""
    for c in _DS_SR_COUNTER.values():
        c.fill_(1)
    _DS_SR_CALL[0] = 0


def _SCALARS_T(device):
    """The compile-time stand-in of a kernel's mScalars (device_scales): a float32 vector, divisibility 1."""
    return to_cute_tensor(torch.empty(8, dtype=torch.float32, device=device), assumed_align=4, leading_dim=0)


def _DOC_SCALARS_T(device):
    """The compile-time stand-in of the per-(document, kv head) scalar table (doc_scaled kernels): (rows, h_kv, 8)."""
    return to_cute_tensor(torch.empty(2, 1, 8, dtype=torch.float32, device=device), assumed_align=4, leading_dim=2)


def _ds_sr_seed(spec: Bwd1Spec) -> int:
    """Bwd1Spec.ds_sr (options.sr_ds): a fresh dither seed per call (0 = round to nearest). A fixed dither would leave the
    dS^ error systematic across steps, which is the bias being removed."""
    if not spec.ds_sr:
        return 0
    _DS_SR_CALL[0] += 1
    return (_DS_SR_CALL[0] * 0x9E3779B1) & 0x7FFFFFFF


def native_d64(d: int, out_pass: str = "all") -> bool:
    """head_dim 64 runs the single-pass kernels natively (N = 64 dK / dV / dQ accumulators, K = 64 S / dP GEMMs) instead
    of zero-padding every operand to 128 and slicing the gradients back."""
    return d == 64 and out_pass == "all"


def _flash_attn_bwd_fp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    q4=None, sfq=None, k4=None, sfk=None, v4=None, sfv=None, do4=None, sfdo=None,
    qt4=None, sfqt=None, dot4=None, sfdot=None,
    tile_m: int = 128,
    return_accum: bool = False,
    fp8_scales: Optional[dict] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out_pass: str = "all",
    masks=None,
    ds_dq_rowlocal: str = "off",
    *,
    specs: PlanSpecs,
):
    """The 1-CTA backward of the 4-bit route ("dk8": S from NVFP4 Q / K, every other GEMM on E4M3 operands) and of the
    8-bit route (HostSpec.bwd_op_mode "fp8"), or the 2-CTA backward where the call is eligible (flash_attn_bwd_2cta8).
    specs: the call's plan.specs (lowprec.specs.PlanSpecs): Bwd1Spec / Bwd2Spec for the kernels, QuantSpec / HostSpec for
    this driver. q4 / k4 (+ SFs) default to a row-wise NVFP4 quantisation of q, k.
    v4 / sfv / do4 / sfdo / qt4 / sfqt / dot4 / sfdot are accepted and ignored: the research operand modes that read them
    (NVFP4 dP, E2M1 dV / dK) are not shipped."""
    assert _get_device_capability() == 10
    if not isinstance(specs, PlanSpecs):
        raise TypeError("_flash_attn_bwd_fp4: specs must be the call's lowprec.specs.PlanSpecs (plan.specs)")
    quant, host = specs.quant, specs.host
    # SCALING-CONSISTENCY RULE: the codes handed over were quantised in the plan's one operand scaling
    check_codes_scaling(host.bwd_op_mode == "fp8", fp8_scales, masks, quant, "_flash_attn_bwd_fp4")
    b, s_q, h, d = q.shape
    # masks / varlen: window / softcap / score_mod / mask_mod / aligned-packed varlen, as one explicit value (lp_masks)
    masks = lp_masks_or_none(masks, causal)
    if masks is not None:
        causal = masks.causal
    if d < 128 and not native_d64(d, out_pass):
        # head_dim < 128: zero-pad to 128 (zero rows add nothing to S / dP; the padded gradient columns are dropped).
        # Caller-provided quantised operands must already be padded (pass them for the padded tensors).
        assert 128 % d == 0 or d % 16 == 0, d
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        pad = lambda t: torch.nn.functional.pad(t, (0, 128 - d))
        sc = None
        if fp8_scales is not None:
            sc = dict(fp8_scales)
            for nm in ("q8", "k8", "v8", "do8"):
                if nm in sc:
                    sc[nm] = pad(sc[nm].view(torch.uint8)).view(torch.float8_e4m3fn)  # E4M3 byte 0 = +0.0
        res = _flash_attn_bwd_fp4(pad(q), pad(k), pad(v), pad(out), pad(dout), lse, softmax_scale=softmax_scale, causal=causal,
                                  deterministic=deterministic, q4=q4, sfq=sfq, k4=k4, sfk=sfk, v4=v4, sfv=sfv, do4=do4, sfdo=sfdo,
                                  qt4=qt4, sfqt=sfqt, dot4=dot4, sfdot=sfdot, tile_m=tile_m, return_accum=False, fp8_scales=sc,
                                  seqused_k=seqused_k, masks=masks, ds_dq_rowlocal=ds_dq_rowlocal, specs=specs)
        return tuple(t[..., :d].contiguous() for t in res)
    if 128 < d < 256:
        # head_dim 129..255 (e.g. 192): zero-pad to 256 and run the 256 passes
        assert d % 16 == 0, d
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        pad = lambda t: torch.nn.functional.pad(t, (0, 256 - d))
        sc = None
        if fp8_scales is not None:
            sc = dict(fp8_scales)
            for nm in ("q8", "k8", "v8", "do8"):
                if nm in sc:
                    sc[nm] = pad(sc[nm].view(torch.uint8)).view(torch.float8_e4m3fn)
        res = _flash_attn_bwd_fp4(pad(q), pad(k), pad(v), pad(out), pad(dout), lse, softmax_scale=softmax_scale, causal=causal,
                                  deterministic=deterministic, tile_m=tile_m, fp8_scales=sc, seqused_k=seqused_k, masks=masks,
                                  specs=specs)
        return tuple(t[..., :d].contiguous() for t in res)
    if out_pass == "all" and d == 256:
        if ds_dq_rowlocal != "off":
            raise ValueError(f"ds_dq_rowlocal={ds_dq_rowlocal!r} refines the block-scaled dS of the single-pass d = 128 "
                             f"kernel; head_dim {d} runs the three-pass kernels on a per-tensor dS")
        return _flash_attn_bwd_fp4_passes(q, k, v, out, dout, lse, softmax_scale, causal, deterministic, tile_m, fp8_scales, seqused_k,
                                          q4=q4, sfq=sfq, k4=k4, sfk=sfk, masks=masks, specs=specs)
    _, s_k, h_k, _ = k.shape
    assert h % h_k == 0, (h, h_k)
    r_gqa = h // h_k  # GQA: kv-stationary tiles loop over the r q heads of each kv head
    assert d in (64, 128, 256) and v.shape[-1] == d and (d != 256 or out_pass != "all") and (d != 64 or out_pass == "all"), \
        (d, out_pass)
    assert q.dtype in (torch.bfloat16, torch.float16) and q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert (lse.shape == (b, h, s_q) or packed_docs(masks)) and lse.dtype == torch.float32
    q, k, v, out, dout, lse = [contig(t) for t in (q, k, v, out, dout, lse)]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    # quant / host are what this driver reads, specs.bwd1 / specs.bwd2 what the kernel constructors read
    if host.bwd_op_mode == "fp8":
        # The 2-CTA cluster kernel has an all-E4M3 configuration (nvfp4_s=False) that is a full
        # FP8 backward in its own right. It used to be unreachable: this branch returned before
        # the 2-CTA eligibility test below, so the fp8 mode always took the 1-CTA kernel.
        # `2cta-e5m2-grad-mistyped`: the cluster kernel types every FP8 GEMM with dO's element type, so it takes E4M3
        # operands only. `_use_2cta_dk8` declines a configured E5M2 gradient; operands a caller supplied pre-quantised
        # are checked here, where they are visible. Either way the 1-CTA FP8 kernel runs, which handles E5M2.
        _sc_e4m3 = all(n not in (fp8_scales or {}) or fp8_scales[n].dtype == torch.float8_e4m3fn
                       for n in ("q8", "k8", "v8", "do8"))
        if _sc_e4m3 and not packed_docs(masks) and _use_2cta_dk8(q, k, causal, deterministic, tile_m, return_accum, seqused_k, out_pass,
                                      None, None, None, None, masks=masks, quant=quant, host=host):
            from flash_attn.cute.interface_bwd_2cta8 import flash_attn_bwd_2cta8
            return flash_attn_bwd_2cta8(q, k, v, out, dout, lse, softmax_scale=softmax_scale, causal=causal,
                                        fp8_scales=fp8_scales, nvfp4_s=False, masks=masks, ds_dq_rowlocal=ds_dq_rowlocal, deterministic=deterministic,
                                        spec=specs.bwd2, quant=quant, host=host)
        return _flash_attn_bwd_fp8_control(q, k, v, out, dout, lse, softmax_scale, causal, deterministic, tile_m, return_accum, fp8_scales,
                                           seqused_k=seqused_k, out_pass=out_pass, masks=masks,
                                           ds_dq_rowlocal=ds_dq_rowlocal, spec=specs.bwd1, quant=quant, host=host)
    # QuantSpec.s_sf_vec = 32: MXFP4 S (UE8M0 SFs per 32) instead of NVFP4 (E4M3 per 16); the rest of dk8 stays E4M3
    s_sf_vec = int(quant.s_sf_vec)
    if q4 is None or k4 is None:
        if q4 is None:
            q4, sfq = _rows(q, s_sf_vec)
        if k4 is None:
            k4, sfk = _rows(k, s_sf_vec)
    # caller-supplied operands must match the selected SF format (NVFP4 SFs have twice the MXFP4 count for the same tensor)
    assert packed_docs(masks) or (sfq.numel() * s_sf_vec == q.numel() and sfk.numel() * s_sf_vec == k.numel()), (
        f"scale-factor count does not match QuantSpec.s_sf_vec={s_sf_vec}: sfq {sfq.numel()} for q {q.numel()}"
    )
    # option A's research backward (the NVFP4 dP on the forward's V codes, fp8_scales["pv4"]) is not shipped: refused
    if (fp8_scales or {}).get("pv4") is not None:
        raise NotImplementedError(OPTION_A_BWD_ARCHIVED)
    _pk = packed_docs(masks)   # lp_docmaps packed rows (the 1-CTA kernel's codes_packed)
    if not _pk and _use_2cta_dk8(q, k, causal, deterministic, tile_m, return_accum, seqused_k, out_pass, v4, do4, qt4, dot4, masks=masks,
                     quant=quant, host=host):
        from flash_attn.cute.interface_bwd_2cta8 import flash_attn_bwd_2cta8
        return flash_attn_bwd_2cta8(q, k, v, out, dout, lse, softmax_scale=softmax_scale, causal=causal, fp8_scales=fp8_scales,
                                    nvfp4_s=True, q4=q4, sfq=sfq, k4=k4, sfk=sfk, s_sf_vec_size=s_sf_vec, masks=masks,
                                    ds_dq_rowlocal=ds_dq_rowlocal, deterministic=deterministic,
                                    spec=specs.bwd2, quant=quant, host=host)
    spec = specs.bwd1  # the 1-CTA kernel's compile-time settings
    if spec is None:
        raise TypeError("_flash_attn_bwd_fp4: the plan has no Bwd1Spec (not an E4M3-operand backward route)")
    # E4M3 V / dO for dP (fp8_scales may carry pre-quantised v8 / do8)
    # varlen: token-block operands (see mx_cols), or per-(document, kv head) per-tensor operands (doc_scaled)
    pt_docs = doc_scaled(masks, quant, fp8=False)
    e4blk = e4_blk32_fp4(quant) or (_docs(masks) and not pt_docs)
    mxq = None
    if e4blk:
        # QuantSpec.operand_scaling_fp4 = blk32: the four E4M3 operands as 32 x 32 MXFP8, unit host scales. Caller-supplied copies
        # arrive as fp8_scales["mx"] = {q, k, v, do: quantize_fp8_blocks dicts} (the autograd forward's own V / Q / K codes)
        if not ds_blk32_fp4(out_pass, quant):
            raise ValueError("32 x 32 block-scaled E4M3 operands need the dk8 operand modes and the blk32 dS (ds_scaling 'block_32x32')")
        mxq = dict((fp8_scales or {}).get("mx") or {})
        need = {nm: t for nm, t in (("q", q), ("k", k), ("v", v)) if nm not in mxq}
        # D (mx["do"]["D"]) is folded IN PLACE by the dS scale: only when this call quantised dO itself, or the caller
        # handed that buffer over (mx["owned"] names it: the autograd backward's fresh dO dict). A caller's reused
        # operands are never written (`blk-backward-mutates-caller-D`: a second call on the same operands read the
        # folded D and gave dQ cos 0.910, norm 1.134 at A1).
        d_owned = ("do" not in mxq or "do" in mxq.get("owned", ()))
        if "do" not in mxq:
            mxq.pop("fac", None)  # a caller's factor table has no row for the dO quantised here: rebuilt from the "e"s
            mxq.update(mx_operands(dout=dout, out=out, masks=masks, quant=quant, host=host))
        if need:
            mxq.update(mx_operands(**need, masks=masks, quant=quant, host=host))
        fp8_scales = {kk: vv for kk, vv in (fp8_scales or {}).items()
                      if kk not in ("v8", "do8", "q8", "k8", "sv", "sdo", "sq", "sk", "s_ds", "pmax", "vmax", "mx",
                                    "dpsum", "lse_log2", "lse_log2_shift", "dpsum_owned")}
        # the four unit descales as views of ONE float32 device tensor (one fill launch instead of four dev32 fills)
        from flash_attn.cute.device_scales import unit_descales
        _u = unit_descales(q.device)   # a per-device constant: no fill launch
        fp8_scales.update(v8=mxq["v"]["codes"], sv=_u[0], do8=mxq["do"]["codes"], sdo=_u[1],
                          q8=mxq["q"]["codes"], sq=_u[2], k8=mxq["k"]["codes"], sk=_u[3],
                          dpsum=mxq["do"]["D"], dpsum_owned=d_owned,
                          d_folded=mxq["do"].get("d_folded") if d_owned else None)
        # the fused step glue (fp4_bwd_autograd.tb_bwd_glue): the shifted LSE * log2 e and the zeroed f16 dQ accumulator
        # from the autograd backward's own dO launch (its fresh, owned dict)
        if "do" in mxq.get("owned", ()):
            if mxq["do"].get("lse_log2") is not None:
                fp8_scales.update(lse_log2=mxq["do"]["lse_log2"], lse_log2_shift=mxq["do"]["lse_log2_shift"])
            if mxq["do"].get("dq_accum") is not None:
                fp8_scales["dq_accum"] = mxq["do"]["dq_accum"]
        # the f16 dQ range (dq_range_of): the caller's norms, else the statistics of the operands quantised here
        for _key, _nm, _stat in (("dn2", "do", "row_norm2_max"), ("vn2", "v", "row_norm2_max"), ("kmax", "k", "amax")):
            if fp8_scales.get(_key) is None and isinstance(mxq.get(_nm), dict) and mxq[_nm].get(_stat) is not None:
                fp8_scales[_key] = mxq[_nm][_stat]
        if fp8_scales.get("dq_kv_rows") is None:
            fp8_scales["dq_kv_rows"] = dq_kv_rows_of(fp8_scales, k, v, causal)
    from flash_attn.cute.device_scales import dev32, bwd1_fp4_scalars
    sc = dict(fp8_scales or {})
    if "v8" in sc:  # pre-quantised (timing: quantisers outside the kernel); scales stay on the GPU
        v8, sv8, do8, sdo8 = sc["v8"], dev32(sc["sv"], q.device), sc["do8"], dev32(sc["sdo"], q.device)
    else:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        rv, rd = (quantize_fp8_per_tensor(t, scales=sc.get(n), power_of_two_scale=_pow2(quant, sc.get(n)))
                  for t, n in ((v, "sv"), (dout, "sdo")))
        v8, sv8, do8, sdo8 = (rv["codes"], dev32(rv["scales"][0], q.device), rd["codes"],
                              dev32(rd["scales"][0], q.device))
    device = q.device
    dtype = torch2cute_dtype_map[q.dtype]
    m_block_size = tile_m
    n_block_size = 128
    seqlen_q_rounded = (s_q + m_block_size - 1) // m_block_size * m_block_size
    if _pk:   # (packed rows: T-row accumulator / codes; the SF atoms are the aligned tiles')
        seqlen_q_rounded = s_q
    assert _pk or s_k % n_block_size == 0, "C1: seqlen_k multiple of 128 (SF atoms)"
    dq_f16 = bwd_dq_f16(host)  # f16 dQ partials (half the reduce-add bytes); 0 = fp32
    p_shift = fp4_p_shift_log2()  # P * 2^shift inside the kernel (f16 range)
    # row-major dQ accumulator: fp32 zeroed by the (reused) BF16 preprocess, or f16 (half the reduce-add bytes)
    # zeroed by torch (the preprocess only zeroes fp32 accumulators, by its own extent) -- or, fp8_scales["dq_accum"],
    # zeroed by the caller's dO-quantiser launch (quant.fp8.quantize_fp8_per_tensor(zero_accumulator=...), the
    # fused backward prologue). Its (b, h, s * d) or (b, h, s, d) f16 bytes are this
    # accumulator's layout exactly when seqlen_q_rounded == s_q.
    _ext_acc = (fp8_scales or {}).get("dq_accum")
    if (dq_f16 and _ext_acc is not None and (seqlen_q_rounded == s_q or _pk) and _ext_acc.dtype == torch.float16
            and _ext_acc.is_contiguous() and _ext_acc.numel() == b * h * seqlen_q_rounded * d and _ext_acc.device == q.device):
        dq_accum = _ext_acc.view(b, h, seqlen_q_rounded, d)
        dq_accum_flat = None
    elif dq_f16:  # zeroed by the preprocess (or explicitly when it is skipped), as the fp32 accumulator
        dq_accum = torch.empty(b, h, seqlen_q_rounded, d, dtype=torch.float16, device=device)
        dq_accum_flat = dq_accum.view(b, h, seqlen_q_rounded * d)
    else:
        dq_accum = torch.empty(b, h, seqlen_q_rounded, d, dtype=torch.float32, device=device)
        dq_accum_flat = dq_accum.view(b, h, seqlen_q_rounded * d)  # preprocess zeroes it (layout-agnostic)
    dpsum = torch.empty(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
    # lp_docmaps (a fused doc_maps backward): dK / dV straight to the caller's packed key rows (the kernel's epilogue);
    # packed rows with a window / mask_mod: they start zero, as the aligned rows would
    _dkv_rows = None
    if masks is not None and getattr(masks.docs, "doc_maps", None) is not None and out_pass == "all" \
            and (_pk or not dkv_zero_fill(masks)):
        from flash_attn.cute.lp_docmaps import dkv_packed_rows
        _dkv_rows = dkv_packed_rows(masks.docs)
    if _dkv_rows is not None:
        _new = torch.zeros if dkv_zero_fill(masks) else torch.empty
        dk = _new((1, _dkv_rows) + tuple(k.shape[2:]), dtype=k.dtype, device=k.device)
        dv = _new((1, _dkv_rows) + tuple(v.shape[2:]), dtype=v.dtype, device=v.device)
    else:
        # a key tile with an empty query range (window / document tail) writes no dK / dV: zeros, as upstream's epilogue
        dk = torch.zeros_like(k) if dkv_zero_fill(masks) else torch.empty_like(k)
        dv = torch.zeros_like(v) if dkv_zero_fill(masks) else torch.empty_like(v)
    dQ_semaphore = (
        torch.zeros(b, h, (masks.docs.s_q if _pk else seqlen_q_rounded) // m_block_size, 1, dtype=torch.int32,
                    device=device) if deterministic else None
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # ---- preprocess (dPsum, LSE*log2e, zero dq_accum). fp8_scales["dpsum"] (b, h, s) supplies D (e.g. from exactly the E4M3
    # dO the dP GEMM uses, quant.fp8.quantize_fp8_per_tensor): then LSE*log2e comes from torch and the kernel is skipped
    # `softmax-consistent-d-dropped`. The condition used to be
    #     ext_d is not None and dq_f16 and seqlen_q_rounded == s_q
    # and the `else` branch runs the preprocess, which OVERWRITES dpsum with the bf16-dO value of D. Neither extra
    # conjunct was ever about D: `dq_f16` was there because the fp32 branch relies on the preprocess to zero
    # dq_accum, and the rounding check because the preprocess also fills the padding rows. So with the default
    # KernelTuning.fp4_d_do8 on and bwd_dq_f16 off -- or on any shape where seqlen_q_rounded != s_q -- the caller's
    # softmax-consistent D was silently discarded, sum_k dS_qk stopped being 0, and that is exactly the documented
    # divergence (attention collapses to one-hot; 3/3 seeds at step 500). The two concerns are separated here the
    # way `cluster-2cta-backward`'s driver already separates them: keep ext_d whenever it is supplied, and zero
    # dq_accum explicitly. The padding rows get the zeros the preprocess would have left.
    ext_d = (fp8_scales or {}).get("dpsum")
    # fp8_scales["lse_log2"]: LSE * log2 e with -inf rows -> 0, ALREADY minus the dv8 P shift (e4m3_p_shift_log2()), from
    # the same launch as D (quant.fp8.quantize_fp8_per_tensor); the torch chain and the subtraction below are then skipped.
    # fp8_scales["dpsum_owned"]: the D tensor is this call's to fold in place (no clone). Both as in the 2-CTA driver.
    ext_ll = (fp8_scales or {}).get("lse_log2") if ext_d is not None else None
    if ext_ll is not None:
        if seqlen_q_rounded != s_q and not _pk:
            raise ValueError("fp8_scales['lse_log2'] (pre-shifted LSE) is valid for s_q % "
                             f"{m_block_size} == 0 only (s_q={s_q})")
        _sh = (fp8_scales or {}).get("lse_log2_shift", e4m3_p_shift_log2())
        if float(_sh) != float(e4m3_p_shift_log2()):
            raise ValueError(f"fp8_scales['lse_log2'] was shifted by {_sh}, the dv8 route subtracts {e4m3_p_shift_log2()}")
        assert (_pk or ext_ll.shape == (b, h, s_q)) and ext_ll.dtype == torch.float32 and ext_ll.is_contiguous(), \
            ext_ll.shape   # (packed rows: the (h, S) aligned statistics)
    if ext_d is not None:
        assert _pk or ext_d.shape == (b, h, s_q), (tuple(ext_d.shape), (b, h, s_q))
        if (fp8_scales or {}).get("dpsum_owned") and ext_d.dtype == torch.float32 and ext_d.is_contiguous():
            _dps = ext_d
        else:
            _dps = ext_d.float().contiguous().clone()  # the dS fold scales D in place; never touch the caller's tensor
        if ext_ll is not None:
            _lsel = ext_ll
        else:
            _lsel = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse * math.log2(math.e)).float().contiguous()
        if seqlen_q_rounded != s_q and not _pk:
            dpsum = torch.zeros(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
            lse_log2 = torch.zeros(b, h, seqlen_q_rounded, dtype=torch.float32, device=device)
            dpsum[:, :, :s_q], lse_log2[:, :, :s_q] = _dps, _lsel
        else:
            dpsum, lse_log2 = _dps, _lsel
        if dq_accum_flat is not None:
            dq_accum_flat.zero_()  # the preprocess is skipped, so its other job is done here
    else:
        key_pre = (dtype, d, m_block_size, dq_f16, dq_accum_flat is not None)
        if key_pre not in _flash_attn_bwd_fp4.cache_pre:
            o_t, do_t = [to_cute_tensor(t, divisibility=8) for t in (out, dout)]
            dps_t, lsel_t = [to_cute_tensor(t, divisibility=4) for t in (dpsum, lse_log2)]
            dqa_t = to_cute_tensor(dq_accum_flat, divisibility=8 if dq_f16 else 4) if dq_accum_flat is not None else None
            lse_t = to_cute_tensor(lse, assumed_align=4)
            pre = FlashAttentionBackwardPreprocess(dtype, d, d, m_block_size, num_threads=256,
                                                   dqaccum_dtype=cutlass.Float16 if dq_f16 else Float32)
            _flash_attn_bwd_fp4.cache_pre[key_pre] = cute.compile(
                pre, o_t, do_t, dps_t, lse_t, lsel_t, dqa_t, None, None, None, None, None, Float32(1.0), None, current_stream, options="--enable-tvm-ffi",
            )
        _flash_attn_bwd_fp4.cache_pre[key_pre](out, dout, dpsum, lse, lse_log2, dq_accum_flat, None, None, None, None, None, Float32(1.0), None, current_stream)
    # E4M3 \hat{P} of P x 2^8 (through LSE) x the E4M3 dO tile; the 2^8 is undone in the runtime dS boost (dS math) and the
    # dV epilogue scale. dQ / dK from the E4M3 dS^ x the E4M3 K / Q copies.
    sc8 = dict(fp8_scales or {})
    if "q8" in sc8:
        q8, sq8 = sc8["q8"], dev32(sc8["sq"], q.device)
    else:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        r = quantize_fp8_per_tensor(q, scales=sc8.get("sq"), power_of_two_scale=_pow2(quant, sc8.get("sq")))
        q8, sq8 = r["codes"], dev32(r["scales"][0], q.device)
    if "k8" in sc8:
        k8, sk8 = sc8["k8"], dev32(sc8["sk"], q.device)
    else:
        from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
        r = quantize_fp8_per_tensor(k, scales=sc8.get("sk"), power_of_two_scale=_pow2(quant, sc8.get("sk")))
        k8, sk8 = r["codes"], dev32(r["scales"][0], q.device)
    dv8_shift = e4m3_p_shift_log2()  # was a literal 8: the same quantity as the FP8 control's and the 2-CTA's
    assert p_shift == 0
    if ext_ll is None:  # a supplied lse_log2 carries the shift already
        lse_log2.sub_(float(dv8_shift))

    # ---- main kernel
    # ---- compile key: (class, spec.key(), ...) as on the FP8 control above. The hand-picked entries of an older key
    # are the spec's fields, and the constexprs the constructor settled on
    # are `resolved_key` -- `hand-rolled-compile-keys` is the root cause of that whole class. Beyond the spec:
    #   dk8 / dq8 / dks       the resolved composite operand modes (functions of keyed knobs, so the
    #                         old key was sound, but the specification wants them checkable);
    #   mxds / mxds_row / mxds_dq_sf / mxp   resolved from Bwd1Spec.ds_scaling_fp8 (keyed through the
    #                         spec), stated explicitly;
    #   do8.dtype             `dp8` is a bool, not a dtype, so an E5M2 dP was unkeyed on this path
    #                         while its FP8 sibling key had it right.
    # The kernel is constructed before the lookup (pure Python, microseconds) so both can be keyed.
    # dqfix: the f16 dQ pre-scale per (row, q head, query tile) where the statistics are per tile (token-block
    # operands); decided before the kernel is built (a compile-time choice of its dQ reduce)
    dq_tiles = (dq_tiles_of(dq_range_of(fp8_scales, block_scaled=True), masks, b, h, h_k, *_pk_aligned(masks, s_q, s_k),
                            tile_m, causal=causal, rows_src=None if d != 128 else ((fp8_scales or {}).get("dq_kv_rows"), dout, _fuse_backrot(fp8_scales)))
                if (mxq is not None and dq_f16 and not return_accum and not pt_docs and seqlen_q_rounded == s_q)
                else None)
    _sfd = _sfd_args(_pk, mxq, sf4=(fp8_scales or {}).get("sfd_sf4")) \
        if (fp8_scales or {}).get("sfd_sf4") is not None else None
    if _sfd is None and pt_docs and ext_ll is not None and (fp8_scales or {}).get("sfd_sf4") is not None:
        # cov1: per-tensor operands on packed rows -- the dO launch's dense statistics rows and the NVFP4 S words
        _sfd = _sfd_args_pt(_pk, lse_log2, dpsum, sf4=fp8_scales["sfd_sf4"])
    if _sfd is not None:   # sfd: the dense per-packed-row statistics
        SFD_CALLS[0] += 1
        lse_log2, dpsum = _sfd[0].view(b, h, -1), _sfd[1].view(b, h, -1)
    kern = FlashAttentionBackwardSm100FP4(d, d, is_causal=causal, tile_m=m_block_size, tile_n=n_block_size, deterministic=deterministic, qhead_per_kvhead=r_gqa,
                                          dq_reduce_dtype=cutlass.Float16 if dq_f16 else Float32, seqused_k=seqused_k is not None,
                                          out_pass=out_pass, sf_vec_size=s_sf_vec, ds_dq_rowlocal=ds_dq_rowlocal, **_mask_ctor(masks),
                                          token_local_operands=bool(mxq is not None and mxq.get("tok") is not None),
                                          det_issue_warp=spec.det_issue_warp, det_issue_bufs=spec.det_issue_bufs,
                                          spec=spec, **({"doc_scaled": True} if pt_docs else {}),
                                          **({"dq_tile_scaled": True} if dq_tiles is not None else {}),
                                          **({"dq_tile_causal": 2 if dq_tiles.get("rowwise") else 1} if (dq_tiles is not None and dq_tiles.get("causal")) else {}),
                                          **({"dkv_rows_packed": True} if _dkv_rows is not None else {}),
                                          **({"codes_packed": True} if _pk else {}),
                                          **({"sf_dense": True} if _sfd is not None else {}),
                                          **({"sf_persist": True} if _sfd is not None and _sfd_persist(masks) else {}))
    if kern.det_gqa_rot:
        _check_det_gqa_rot(q, k)
    _check_fp4_p_shift(kern)
    # qk_hadamard (fp8_scales["had_signs"]): the dK operand Q8 and the dQ operand K8 are rotated; the dK epilogue and
    # the dQ postprocess undo it (fused), the tensors are tagged for qk_hadamard.finish_grads
    from flash_attn.cute.qk_hadamard import fuse_backrot
    _had = fuse_backrot(fp8_scales)
    if _had:
        from flash_attn.cute.qk_hadamard import signs_tuple
        if not (d == 128 and out_pass == "all"):
            raise NotImplementedError("qk_hadamard on the 1-CTA backward needs the dk8 operand modes (E4M3 Q8 / K8), "
                                      "d = 128, out_pass all")
        kern.had_dk = signs_tuple(128)
    # token-local operands: K / V / dO copies (mx["tok"]: (4, b, h, s) per-token factors)
    tok = mxq.get("tok") if mxq is not None else None
    wl = _bwd_worklist(masks, deterministic, h, r_gqa, d, host)
    key = (type(kern).__qualname__, spec.key(), _had, _mask_key(masks), wl is not None, out_pass, seqused_k is not None,
           *(("doc_scaled",) if pt_docs else ()), *(("dq_tiles",) if dq_tiles is not None else ()),
           *(("dq_causal", bool(dq_tiles.get("rowwise"))) if (dq_tiles is not None and dq_tiles.get("causal")) else ()),
           r_gqa, dtype, d, d, causal, deterministic,
           m_block_size, n_block_size, dq_f16, p_shift, s_sf_vec, tok is not None,
           do8.dtype, _arch_key(),
           resolved_key(kern, ("dk8", "dq8", "dv8", "dp8", "mxds", "mxds_row", "mxop", "mxop_s", "mxdv", "tmem_ld_rep",
                                     "mxds_dq_sf", "mxp", "fp8", "ld_rep", "warp_sync", "ds_sr",
                                     "sf_vec_size", "tile_m", "tile_n", "out_pass", "pp",
                                     "deterministic", "qhead_per_kvhead", "dq_reduce_dtype", "lpt", "spt", "det_issue_warp_arg", "det_issue_bufs_req",
                                     "ds_dq_rowlocal", "sf_asm", "sf_asm_q", "dkv_rows_packed", "codes_packed", "sf_dense", "bwd_persist")))
    # mxop SF atoms: (Q tok, K tok, V tok, dO tok, Q dim, K dim, dO dim); the dk8 route's S is NVFP4, so no Q / K tok
    mx_args = [None] * 9
    if e4blk:
        tb = mx_cols(masks, quant=quant, host=host) == 128
        _sfkd = mx_sf_arg(mxq["k"]["sf_dim_raw"])   # token-local K: the quantiser wrote the unit dQ-side scale
        mx_args = [None, None, None if tb else mx_sf_arg(mxq["v"]["sf_tok_raw"]), None if tb else mx_sf_arg(mxq["do"]["sf_tok_raw"]),
                   mx_sf_arg(mxq["q"]["sf_dim_raw"]), _sfkd, mx_sf_arg(mxq["do"]["sf_dim_raw"]),
                   mx_factors(mxq) if tb else None, tok]
    if key not in _flash_attn_bwd_fp4.cache:
        q_t, k_t, do_t, dk_t, dv_t = [to_cute_tensor(t, divisibility=8) for t in (q8, k8, do8, dk, dv)]
        sf_ts = [to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in (sfq, sfk)]
        sf_ts += [None, None]
        lsel_t, dps_t, dqa_t = [to_cute_tensor(t, divisibility=4) for t in (lse_log2, dpsum, dq_accum)]
        sem_t = to_cute_tensor(dQ_semaphore, assumed_align=4) if deterministic else None
        sym = lambda t: tuple(cutlass.Int32(0) for _ in range(4))
        sfqt_t = None
        sfdot_t = None
        _flash_attn_bwd_fp4.cache[key] = cute.compile(
            kern, q_t, k_t, do_t, _make_fp4_ptr(None), _make_fp4_ptr(None),
            to_cute_tensor(v8, divisibility=8), _make_fp4_ptr(None),
            *sf_ts, lsel_t, dps_t, dqa_t, dk_t, dv_t, Float32(softmax_scale), Float32(1.0), Float32(1.0), Int32(0),
            *[Float32(1.0)] * 3, *_mask_window(masks, True), sem_t,
            sym(q4), sym(k4), sym(q4), sym(q4),
            None, None, sfqt_t, sfdot_t,
            (), (),
            to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0) if seqused_k is not None else None,
            *[to_cute_tensor(t, assumed_align=16) if t is not None else None for t in mx_args],
            _DOC_SCALARS_T(q.device) if pt_docs else _SCALARS_T(q.device), *_mask_tables(masks, True, wl),
            current_stream,
            *((to_cute_tensor(ds_sr_counter(q.device), assumed_align=8, leading_dim=0),) if kern.ds_sr_live
              else ((None,) if _pk else ())),
            *_pk_tail(masks, True, seed=False), *_sfd_tail(_sfd, True), options="--enable-tvm-ffi",
        )
    # Bwd1Spec.ds_sr where it is live (per-tensor or block-scaled dS^): the seed comes from the device counter, read by
    # the kernel and advanced by the dQ postprocess below, so a captured step replays with a new dither (the host counter froze it)
    _seed_rt = (ds_sr_counter(q.device),) if kern.ds_sr_live else ()
    shp = lambda t: (b, t.shape[1], t.shape[2], d)  # per tensor: q / dO have h heads, k / v have h_kv
    # ---- the kernel scalars, on the GPU (device_scales._bwd1_fp4_scalars_kernel transcribes the host expressions):
    #   dps_scale = 1 / (s_v s_dO), ds_b = s_v s_dO [/ 2^dv8_shift], dv_s = s_dO / 2^dv8_shift (dv8);
    #   dq8: kernel dS = dS / s_ds with s_ds = B / 448, B = max_q ||dO_q|| max_k ||V_k|| + max |D| >= max |dS| (P <= 1): the
    #   E4M3 d\hat{S} words never saturate; ds_b = (s_v s_dO / 2^dv8_shift) / s_ds, dK SFs get s_ds back (dk_s = s_ds, or
    #   softmax_scale s_ds s_q with dk8), dQ partials dq_s = s_ds s_k. The D fold multiplier -ds_b dps_scale comes out
    #   with them.
    sds_kw = {}
    if masks is not None:
        _check_masks_ds(masks, not ds_blk32_fp4(out_pass, quant))
    if ds_blk32_fp4(out_pass, quant):
        # QuantSpec.ds_scaling_fp4 = blk32: in-kernel UE8M0 exponents per 32 x 32 dS block carry the range, so the kernel
        # sees dS in true units (s_ds = 1): no P_max bound, no supplied scale
        sds_kw = dict(sds_mode=3)
    elif "s_ds" in sc8:  # computed by the caller from this call's tensors (timing: nothing inside the call)
        sds_kw = dict(s_ds=dev32(sc8["s_ds"], q.device), sds_mode=0)
    elif "s_ds_r" in sc8:  # the P_max bound r, max-reduced by the dO / V quantisers of this step (no norm passes)
        sds_kw = dict(r=sc8["s_ds_r"], sds_mode=1)
    elif "pmax" in sc8:  # per-query P_max (b, h, s_q) from the forward's row max: max_q P_max_q y_q / 448
        # max_k ||V_k||: the ONLY data this path reads from bf16 v; "vmax" lets the caller save that scalar instead of v
        vmax = sc8["vmax"] if "vmax" in sc8 else row_norms_f32(v).amax()
        y = row_norms_f32(dout).permute(0, 2, 1) * vmax + dpsum[:, :, :s_q].abs().float()
        sds_kw = dict(r=(sc8["pmax"].float().clamp(max=1.0) * y).amax().reshape(1), sds_mode=1)
    else:
        # exact per-query P_max = exp(max_k S_qk - LSE_q) (O(s^2), chunked, HOST-SIDE: an accuracy tool, never a
        # training default -- the steps pass the forward's P_max) times y_q = ||dO_q|| max_k ||V_k|| + |D_q|
        # (other bounds were measured looser or costlier and are not shipped)
        sds_kw = dict(s_ds=dev32(div32(_pmax_bound(q, k, dout, v, lse, dpsum[:, :, :s_q], softmax_scale, causal),
                                       448.0), q.device), sds_mode=0)
    # the f16 dQ partials' data-sized pre-scale (device_scales._dq_prescale): scal[7] is then the postprocess scale
    dq_range = (dq_range_of(fp8_scales, block_scaled=mxq is not None)
                if (dq_f16 and not return_accum) else None)
    # dp8 / dv8 / dq8 / dk8: the E4M3 operand chain, the only one of the shipped 4-bit route
    post_scale = None
    dq_docs = None
    if pt_docs:
        # per-(document, kv head) scales: the (n_docs + 1, h_kv, 8) scalar table (one launch, as the per-tensor scalar
        # launch); the D fold D'' = m D was applied per document by the dO launch
        from flash_attn.cute.device_scales import bwd1_fp4_doc_scalars
        dsc = sc8["doc_scales"]
        if sds_kw.get("sds_mode") != 3 or not dq_f16 or return_accum:
            raise NotImplementedError("per-tensor varlen: the block-scaled dS and f16 dQ partials only")
        if quant.ds_fold and not sc8.get("doc_d_folded"):
            raise RuntimeError("per-tensor varlen with ds_fold: the dO launch must fold D per document (documents fold)")
        # (packed rows: the plane per ALIGNED query tile of the documents, as the fused path's)
        dq_docs = dq_docs_of(dq_range, masks, b, h, _pk_aligned(masks, s_q, s_k)[0], d) \
            if seqlen_q_rounded == s_q else None
        scal, post_scale = bwd1_fp4_doc_scalars(softmax_scale, dv8_shift, p_shift, dsc["q"], dsc["k"], dsc["v"],
                                                dsc["do"], dsc["k_global"], dq_range=dq_range, dq_docs=dq_docs)
    else:
        scal = bwd1_fp4_scalars(q.device, softmax_scale, dv8_shift, True, True, True, True,
                                sv=sv8, sdo=sdo8, sq=sq8, sk=sk8, dq_range=dq_range,
                                p_shift=p_shift, dq_tiles=dq_tiles if dq_range is not None else None, **sds_kw)
        if dq_tiles is not None:
            assert dq_range is not None and dq_range[2] is not None
            from flash_attn.cute.device_scales import dq_tile_planes
            post_scale = dq_tile_planes(scal, dq_tiles)
    if quant.ds_fold and not pt_docs:  # kernel: dS = P' FMA(d\hat{P}, b, D''), D'' = -b D / (s_v s_dO)
        from flash_attn.cute.device_scales import apply_d_fold
        apply_d_fold(dpsum, scal, 6, (fp8_scales or {}).get("d_folded"))
    _flash_attn_bwd_fp4.cache[key](
        q8, k8, do8, _make_fp4_ptr(q4), _make_fp4_ptr(k4),
        v8, _make_fp4_ptr(None),
        sfq, sfk, None, None,
        lse_log2, dpsum, dq_accum, dk, dv, 1.0, 1.0, 1.0,
        0 if kern.ds_sr_live else _ds_sr_seed(spec), 1.0, 1.0, 1.0,
        *_mask_window(masks), dQ_semaphore,
        shp(q4), shp(k4), shp(k4), shp(q4),
        None, None,
        None, None,
        (), (),
        seqused_k, *mx_args, scal, *_mask_tables(masks, False, wl), current_stream,
        *(_seed_rt if (_seed_rt or not _pk) else (None,)), *_pk_tail(masks, False, seed=False),
        *_sfd_tail(_sfd, False),
    )
    if not dq_f16:
        # `dq-fp32-partials-unscaled`: as on the FP8 control above -- dq_s is applied in-kernel only before an f16 reduce
        # (fp32 partials came back 1/dq_s too large: x2857 measured at s1024, 1/dq_s = 2856.81)
        dq_accum.mul_(scal[5])
    # ---- postprocess: dQ = scale * accum, (b, h, s, d) -> (b, s, h, d)
    if host.dq_post == "cute":
        from flash_attn.cute.dq_postprocess_cute import dq_postprocess_cute
        dq = dq_postprocess_cute(dq_accum, s_q, post_scale if post_scale is not None else
                                 (scal[7:8] if dq_range is not None else softmax_scale / 2 ** p_shift),
                                 q.dtype, had=_had, seed_counter=_seed_rt[0] if _seed_rt else None,  # one kernel
                                 **({"tile_scale": True} if (dq_tiles is not None or dq_docs is not None) else {}),
                                 **({"row_scale": True} if (dq_tiles is not None and dq_tiles.get("rowwise")) else {}),
                                 **_pk_row_tile(masks, dq_tiles if dq_tiles is not None else dq_docs))
    else:
        dq = dq_post_torch(dq_accum, s_q, **_pk_row_tile(masks, dq_tiles if dq_tiles is not None else dq_docs),
                           post=post_scale if post_scale is not None else
                           (scal[7:8] if dq_range is not None else softmax_scale / 2 ** p_shift)
                           ).to(q.dtype).permute(0, 2, 1, 3).contiguous()
        if _seed_rt:
            _seed_rt[0].add_(1)   # HostSpec.dq_post = "torch" (not the default): one more small launch
    if _had:
        dk._had_unrotated = True
    if return_accum:
        return dq, dk, dv, dq_accum
    return dq, dk, dv


_flash_attn_bwd_fp4.cache = {}
_flash_attn_bwd_fp4.cache_pre = {}


def _flash_attn_bwd_fp4_passes(q, k, v, out, dout, lse, softmax_scale, causal, deterministic, tile_m, fp8_scales, seqused_k,
                               q4=None, sfq=None, k4=None, sfk=None, masks=None, *, specs: PlanSpecs):
    r"""head_dim 256: three kernel calls, one per output (out_pass dk / dq / dv), each with the
    256-column output accumulator in TMEM [128,384). Quantisation and the d\hat{S} scale are computed once and shared."""
    b, s_q, h, d = q.shape
    q, k, v, out, dout, lse = [t.contiguous() for t in (q, k, v, out, dout, lse)]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    quant, host = specs.quant, specs.host
    fp8 = host.bwd_op_mode == "fp8"
    sc = dict(fp8_scales or {})
    if fp8:
        for nm, t in (("q8", q), ("k8", k), ("v8", v), ("do8", dout)):
            if nm not in sc:
                from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
                from flash_attn.cute.device_scales import dev32
                r = quantize_fp8_per_tensor(t, scales=dev32(sc["s" + nm[:-1]], t.device) if "s" + nm[:-1] in sc else None,
                                            power_of_two_scale=_pow2(quant, sc.get("s" + nm[:-1])))
                sc[nm], sc["s" + nm[:-1]] = r["codes"], r["scales"][0]
    else:
        if q4 is None:
            q4, sfq = _rows(q)
        if k4 is None:
            k4, sfk = _rows(k)
        for nm, t in (("v8", v), ("do8", dout), ("k8", k), ("q8", q)):
            if nm not in sc:
                from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
                r = quantize_fp8_per_tensor(t, power_of_two_scale=_pow2(quant))
                sc[nm], sc["s" + nm[:-1]] = r["codes"], r["scales"][0]
        if "s_ds" not in sc:
            y_src = sc.pop("pmax", None)
            D = (out.float() * dout.float()).sum(-1).permute(0, 2, 1)
            if y_src is not None:
                y = dout.float().norm(dim=-1).permute(0, 2, 1) * v.float().norm(dim=-1).amax() + D.abs()
                # max(r, 1e-30) / 448 in float32 on the GPU (div.rn) on the exact float32 max
                sc["s_ds"] = div32((y_src.float().clamp(max=1.0) * y).amax().float().clamp_min(1e-30), 448.0)
            else:
                sc["s_ds"] = div32(_pmax_bound(q, k, dout, v, lse, D, softmax_scale, causal), 448.0)
    res = {}
    for p in ("dk", "dq", "dv"):
        kw = {} if fp8 else dict(q4=q4, sfq=sfq, k4=k4, sfk=sfk)
        res[p] = _flash_attn_bwd_fp4(q, k, v, out, dout, lse, softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
                                     tile_m=tile_m, fp8_scales=sc, seqused_k=seqused_k, out_pass=p, masks=masks, specs=specs,
                                     **kw)
    return res["dq"][0], res["dk"][1], res["dv"][2]

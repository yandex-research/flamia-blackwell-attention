# Copyright (c) 2025, Tri Dao.
"""Host driver for the block-scaled (NVFP4 / MXFP4 / MXFP8) SM100 forward kernel.

Kept as a separate entry point rather than woven into ``interface._flash_attn_fwd``:

* the block-scaled path takes operands upstream's signature has no place for (per-operand scale
  factor tensors, sub-byte packed Q/K/V passed as raw pointers, a K-major packed V, the exported
  log2 frame the low-precision backward consumes), and needs its own compile-cache key;
* keeping upstream's ``_flash_attn_fwd`` untouched means upstream's fixes to the BF16/FP8 path keep
  applying to this tree unchanged, and the BF16 baseline we measure against is genuinely upstream's.

The backward drivers (``interface_bwd_fp4``, ``interface_bwd_fp4q``, ``interface_bwd_2cta8``) follow
the same pattern.
"""

import math
import os
from functools import lru_cache
from typing import Optional, Tuple, Callable, Union, Type
from dataclasses import dataclass

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import utils
from flash_attn.cute.lowprec.specs import FwdKernelSpec, HostSpec, QuantSpec
from flash_attn.cute.flash_fwd_sm100_fp4 import (
    FlashAttentionForwardSm100 as FlashAttentionForwardSm100FP4,
)
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80, FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_tensors,
    normalize_block_sparse_config,
    get_block_sparse_expected_shapes,
)

# Upstream helpers reused verbatim so this module tracks their behaviour rather than copying it.
from flash_attn.cute.interface import (
    _get_device_capability,
    _get_device_arch,
    _tile_size_fwd_sm90,
    _validate_head_dims,
    torch2cute_dtype_map,
)

# upstream renamed this helper with a leading underscore
tile_size_fwd_sm90 = _tile_size_fwd_sm90


def maybe_contiguous(x):
    if x is None or x.stride(-1) == 1:
        return x
    # FP4 dtypes have no copy_ kernel -- .contiguous() would fail. Callers (FP4 V)
    # are responsible for handing in already-properly-laid-out FP4 tensors.
    if hasattr(torch, "float4_e2m1fn_x2") and x.dtype == torch.float4_e2m1fn_x2:
        return x
    return x.contiguous()


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, divisibility=1):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1.
    `divisibility` (elements) is propagated to the dynamic strides so that dynamically offset
    slices keep a provable alignment (the DSL >= 4.5 verifier otherwise treats them as
    element-aligned and rejects 128-bit universal copies, e.g. in the backward preprocess)."""
    if isinstance(t, cute.Tensor):
        return t
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility > 1:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=t.dim_order(), divisibility=divisibility
        )
    return tensor


# ---------------------------------------------------------------------------------------------------------------------
# prequantised-operands: binding caller-supplied quantised Q / K operands (prequant.py)
# ---------------------------------------------------------------------------------------------------------------------
def _as_quantized_operand(obj, name: str):
    """Accept a prequant.QuantizedOperand, or a bare (codes, scale) pair whose format is inferred from the dtypes."""
    from flash_attn.cute.prequant import QuantizedOperand

    if obj is None:
        return None
    if isinstance(obj, QuantizedOperand):
        return obj
    if not (isinstance(obj, (tuple, list)) and len(obj) == 2):
        raise TypeError(f"{name} must be a prequant.QuantizedOperand or a (codes, scale) pair, got {type(obj)}")
    codes, scale = obj
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"{name}[0] (codes) must be a torch.Tensor, got {type(codes)}")
    if codes.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        from flash_attn.cute.prequant import is_scalar_scale
        if isinstance(scale, torch.Tensor) and not is_scalar_scale(scale):  # uint8 scale factors: block-scaled
            fmt = "mxfp8"
        else:
            fmt = "e4m3" if codes.dtype == torch.float8_e4m3fn else "e5m2"
    elif codes.dtype in (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8)):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name}: E2M1 codes need a scale-factor tensor, got {type(scale)}")
        fmt = "nvfp4"  # per-16 E4M3 is the default; pass a QuantizedOperand to declare mxfp4
    else:
        raise TypeError(f"{name}: cannot infer a quantised format from codes dtype {codes.dtype}")
    return QuantizedOperand(codes, scale, fmt)


def _bind_quantized_operands(q, k, q_op, k_op, mSFQ, mSFK, qk_sf_vec_size, softmax_scale, q_scale_fused,
                             head_dim_hint=None):
    """Replace (q, k, mSFQ, mSFK, softmax_scale, qk_sf_vec_size) by what the supplied operands imply.

    Returns the same tuple plus `score_scale_unit`: True when the kernel's single score scalar came out bitwise
    1.0, which is the only condition under which the kernel can specialise the multiplication away.
    """
    from flash_attn.cute import prequant

    if q_op is None and k_op is None:
        return q, k, mSFQ, mSFK, qk_sf_vec_size, softmax_scale, False
    if q_op is None or k_op is None:
        raise ValueError("q_quantized and k_quantized must be supplied together (the QK MMA takes one format)")
    if q_op.fmt != k_op.fmt:
        raise ValueError(f"q_quantized/k_quantized format mismatch: {q_op.fmt} vs {k_op.fmt}")
    for nm, t in (("q", q), ("k", k)):
        if t is not None and not isinstance(t, (tuple, list)):
            raise ValueError(f"{nm} must be None when {nm}_quantized is supplied (the point is to read no bf16 {nm})")
    if mSFQ is not None or mSFK is not None:
        raise ValueError("mSFQ / mSFK must not be passed alongside q_quantized / k_quantized")
    if k_op.fused != 1.0:
        raise ValueError(f"k_quantized.fused must be 1.0; the softmax factor belongs to Q (got {k_op.fused})")

    code_dtype, gran, _ = prequant.format_info(q_op.fmt)
    packed = gran and code_dtype is torch.uint8
    b, s_q, h_q = q_op.codes.shape[:3]
    head_dim = q_op.codes.shape[3] * (2 if packed else 1)
    if head_dim_hint is not None and head_dim != head_dim_hint:
        raise ValueError(f"q_quantized implies head_dim {head_dim}, caller said {head_dim_hint}")
    q_op.validate("q_quantized", shape=(b, s_q, h_q, head_dim), head_dim=head_dim)
    k_op.validate("k_quantized", shape=(*k_op.codes.shape[:3], head_dim), head_dim=head_dim)

    # per-tensor descales: a Python float, or a device scalar (then every product below stays on the GPU and reaches
    # the kernel as a device score scale -- see _flash_attn_fwd's softmax_scale_t); block-scaled: in the SF tensor
    dq = 1.0 if gran else (q_op.scale.float() if isinstance(q_op.scale, torch.Tensor) else float(q_op.scale))
    dk = 1.0 if gran else (k_op.scale.float() if isinstance(k_op.scale, torch.Tensor) else float(k_op.scale))
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if not q_scale_fused:
        # exactly the product the hand-rolled callers already form, in the same order, so this route is bitwise
        # the q/k route it replaces. A block-scaled operand's descales live in its SF tensor, so dq = dk = 1.
        softmax_scale = softmax_scale * dq * dk
        score_scale_unit = False
    else:
        c = prequant.softmax_fold_factor(head_dim, softmax_scale)
        if abs(q_op.fused / c - 1.0) > 1e-6:
            raise ValueError(
                f"q_scale_fused=True but q_quantized.fused={q_op.fused!r} is not log2(e) * softmax_scale = {c!r}. "
                f"Quantise Q with prequant.quantize_fused(q, fmt, c=prequant.softmax_fold_factor(d, softmax_scale))."
            )
        # the single fp32 scalar the kernel is left with; raises when the fold has nowhere to live
        scale_log2 = prequant.score_scale_log2(q_op, k_op, head_dim, softmax_scale, True)
        if isinstance(scale_log2, torch.Tensor):
            # device descales: unit only by construction (quantize_fused_unit_score), never by reading the value
            score_scale_unit = bool(q_op.unit_score and k_op.unit_score)
        else:
            score_scale_unit = bool(scale_log2 == 1.0)
        # the kernel multiplies by log2(e) itself, so hand it the scalar in natural units. When the scalar is
        # exactly 1.0 the kernel ignores this value (it rebinds the constant inside the kernel), so this
        # division's rounding is unobservable there; otherwise the fold is a re-association of one product.
        softmax_scale = scale_log2 / prequant.LOG2E

    qc, kc = q_op.codes, k_op.codes
    if packed:
        if not hasattr(torch, "float4_e2m1fn_x2"):
            raise RuntimeError("this torch build has no float4_e2m1fn_x2; E2M1 operands cannot be bound")
        qc = qc.view(torch.float4_e2m1fn_x2) if qc.dtype == torch.uint8 else qc
        kc = kc.view(torch.float4_e2m1fn_x2) if kc.dtype == torch.uint8 else kc
    if gran:
        mSFQ, mSFK = q_op.scale, k_op.scale
        if qk_sf_vec_size is None and gran == 32:
            qk_sf_vec_size = 32
        if qk_sf_vec_size is not None and qk_sf_vec_size != gran:
            raise ValueError(f"qk_sf_vec_size={qk_sf_vec_size} contradicts {q_op.fmt}'s {gran}-element scale groups")
    return qc, kc, mSFQ, mSFK, qk_sf_vec_size, softmax_scale, score_scale_unit


# Check if dtype is nvfp4
def is_nvfp4_dtype(dtype):
    """Check if a torch dtype is nvfp4 (FP4)"""
    # Check by dtype name or string representation
    dtype_str = str(dtype).lower()
    # NOTE: cutlass uses int8 for nvfp4 for now, change later.
    return 'nvfp4' in dtype_str or 'fp4' in dtype_str or 'e2m1' in dtype_str or hasattr(dtype, 'nvfp4') or dtype == torch.int8


def is_valid_dtypes_and_scale_factor_vec_size(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
) -> bool:
    """
    Check if the dtypes and sf_vec_size are valid combinations for block-scaled quantization.

    :param ab_dtype: The data type of the Q/K/V operands (typically Float4E2M1FN for FP4)
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: The data type of the scale factor
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: The vector size of the scale factor
    :type sf_vec_size: int

    :return: True if the dtypes and sf_vec_size are valid, False otherwise
    :rtype: bool
    """
    is_valid = True

    # Check valid ab_dtype (for FP4 flash attention, this should be Float4E2M1FN)
    if ab_dtype not in {
        cutlass.Float4E2M1FN,
        cutlass.Float8E5M2,
        cutlass.Float8E4M3FN,
    }:
        is_valid = False

    # Check valid sf_vec_size
    if sf_vec_size not in {16, 32}:
        is_valid = False

    # Check valid sf_dtype
    if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
        is_valid = False

    # Check valid sf_dtype and sf_vec_size combinations
    if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
        is_valid = False
    if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
        is_valid = False

    return is_valid


# The block-scaled kernel's __call__ takes, after (..., mPmax, q_ptr_shape, k_ptr_shape), the optional
# (v_ptr_shape, mVBias, mVU, mVTokW, mVTokSF, mSoftmaxScale). _N_FIXED_ARGS counts the call arguments up to and including
# k_ptr_shape (q ... aux_tensors: 17, the five SF / descale / pmax slots, the two shapes).
_N_FIXED_ARGS = 17 + 5 + 2


def _score_scale_tail(n_optional_given: int, t):
    """The argument tail that reaches mSoftmaxScale: defaults for the optional V-side slots not given, then the
    score-scale tensor (or None)."""
    defaults = [(), None, None, None, None]  # v_ptr_shape, mVBias, mVU, mVTokW, mVTokSF
    assert 0 <= n_optional_given <= len(defaults), n_optional_given
    return defaults[n_optional_given:] + [t]


def _repro_tail(n_optional_given: int, rp, dbg):
    """The argument tail that reaches mRepro / mReproDbg: defaults for the optional slots before them (v_ptr_shape ..
    mVTokSF, mSoftmaxScale, mTbFac) not given, then the two tensors."""
    defaults = [(), None, None, None, None, None, None]
    assert 0 <= n_optional_given <= len(defaults), n_optional_given
    return defaults[n_optional_given:] + [rp, dbg]


def _varlen_work_tail(n_optional_given: int, t):
    """The argument tail that reaches mVarlenWork: defaults for the optional slots before it (v_ptr_shape .. mVTokSF,
    mSoftmaxScale, mTbFac, mRepro, mReproDbg, mTbTokV) not given, then the work list."""
    defaults = [(), None, None, None, None, None, None, None, None, None]
    assert 0 <= n_optional_given <= len(defaults), n_optional_given
    return defaults[n_optional_given:] + [t]


O_PACKED_EPI = "correction"   # lp_docmaps packed O: "correction" (the correction warps store) | "epilogue"
# spd3: the 1-CTA forward on packed rows takes the aligned form's batched epilogue wherever that form does (varlen_tma_o:
# causal calls) -- the sO epilogue warp's TMA store of every 128-row tile inside its document, row copies of a
# document's partial last tile (flash_fwd_sm100_fp4 o_rows_tma1) -- and with it the 1-CTA causal stage skip
# (d 128). False: the correction warps' predicated row copies (the cov2 / cov5 kernels). O / LSE bitwise either way.
O_PACKED_TILE = True


SFD_FWD_CALLS = [0]   # (tests / probes) forward calls that took the sf_dense kernel


def _o_rows_tail(n_optional_given: int, t):
    """The argument tail that reaches mCuSeqlensO: defaults for the optional slots before it (v_ptr_shape .. mTbTokV,
    mVarlenWork) not given, then the packed O row starts."""
    defaults = [(), None, None, None, None, None, None, None, None, None, None]
    assert 0 <= n_optional_given <= len(defaults), n_optional_given
    return defaults[n_optional_given:] + [t]


def _flash_attn_fwd(
    q: Union[torch.Tensor, cute.Tensor],
    k: Union[torch.Tensor, cute.Tensor],
    v: Union[torch.Tensor, cute.Tensor],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    mSFQ: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for Q
    mSFK: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for K
    mSFV: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for V
    v_descale: Optional[torch.Tensor] = None,  # Per-(batch, kv-head) FP8 V dequant scale
    force_fp4_impl: bool = False, # Test fp4 attn impl under bf16 precision w/o sf
    pv_sf_vec_size: Optional[int] = None,  # 32 with FP4 V -> MXFP4 PV (E8M0 SFs per 32)
    qk_sf_vec_size: Optional[int] = None,  # 32 with FP4 Q / K -> MXFP4 QK (UE8M0 SFs per 32, kind::mxf4nvf4 scale_vec::2X)
    return_pmax: bool = False,  # block-scaled SM100 kernel: also return P_max_q = max_k softmax(S)_qk (b, h, s_q) float32
    # Upstream spelling of (m_block_size, n_block_size); the upstream tests (test_mask_mod.py and friends)
    # pass it. When given it overrides the two separate arguments.
    tile_mn: Optional[Tuple[int, int]] = None,
    # Upstream's arch override, in SM units (major*10 + minor, e.g. 100 for sm_100). The fork keys kernels on the
    # compute-capability MAJOR only, so it maps to _compute_capability = _arch // 10.
    _arch: Optional[int] = None,
    # all-4-bit forward (FwdKernelSpec.fp4all) causal-safe V (flash_fwd_sm100_fp4 vtok): per-key bias log2(s_k) / scale_log2 and weight 1 / s_k,
    # (b, s_k, h_k) float32 each; the V operand is then v~ = V / s_k with unit group scales
    vtok_bias: Optional[torch.Tensor] = None,
    vtok_u: Optional[torch.Tensor] = None,
    # per-key-tile form: (b, h_k, s_k / 128, 256) float32 per key tile [A_k = e_k + 8, u_k = 2^-e_k] (staged by the MMA warp)
    vtok_w: Optional[torch.Tensor] = None,
    # TMA form: the same floats as bytes in an SFV-shaped uint8 view (staged by the load warp's TMA copy)
    vtok_tsf: Optional[torch.Tensor] = None,
    # diagonal-only-key-local-v-scale (flash_fwd_sm100_fp4 `vtok_diag`): v / mSFV carry TWO payloads concatenated
    # along the key axis -- keys [0, s_k) the fast key-grouped quantisation for the fully unmasked tiles, keys
    # [s_k, 2 s_k) the key-local one (with vtok_bias / vtok_u) for the causal masked (diagonal) tiles. Causal only.
    vtok_diag: bool = False,
    # token-block operands (flash_fwd_sm100_fp4 `tb_fwd`): (4, b, h, s / 32) float32 factors
    # 2^(e - 127) of Q, K, V (, dO) per 32-token block (quantize_fp8_blocks block_columns=128). Q / K are then the block codes (E4M3) or
    # NVFP4 (whose per-token scales need no factor), V the E4M3 block codes with v_descale 1; the kernel folds f_Q f_K
    # and log2 f_V into the scores and weights the row sum by 1 / f_V. No block-scaled PV / QK GEMM.
    tb_fac: Optional[torch.Tensor] = None,
    # token-local operands: per-key V factors (b, h_kv, s_k) fp32, with tb_fac (the fold forward)
    tb_tok_v: Optional[torch.Tensor] = None,
    # ---- prequantised-qk-fused-scale: pre-quantised Q / K with a fused softmax factor (prequant.py) ------------------------
    # Caller-supplied quantised operands: a prequant.QuantizedOperand, or a bare (codes, scale) pair. When given,
    # the matching q / k argument must be None -- no quantiser runs and no bf16 Q or K is read. `softmax_scale` is
    # then the PLAIN attention scale (this driver folds the operands' descales in itself), not the descale product
    # the hand-rolled FP8 callers pass in the `q`/`k` route.
    q_quantized=None,
    k_quantized=None,
    # The supplied Q already carries c = log2(e) * softmax_scale, so the kernel must not apply it again. Reaches
    # the kernel as the compile-time constexpr `score_scale_unit` (and is in the compile key).
    q_scale_fused: bool = False,
    # GPU-only score scale (device_scales): a 1-element float32 CUDA tensor holding the whole score scale (e.g.
    # softmax_scale * s_q * s_k for per-tensor FP8 Q / K), read by the block-scaled kernel from device memory. Replaces
    # `softmax_scale` (which then only labels the call) so a data-dependent scale is never read back to the host.
    softmax_scale_t: Optional[torch.Tensor] = None,
    # varlen: the caller built cu_seqlens_q / _k with 128-aligned document starts (lp_varlen.LPDocs), so the
    # alignment check -- a host read of the offsets -- is skipped (no host sync on the varlen training step)
    cu_seqlens_aligned: bool = False,
    # aligned varlen (cu_seqlens_aligned): the precomputed work list (lp_varlen.varlen_worklist, int32 (2 + 2 n_max,))
    # -> the persistent longest-first work-list scheduler (flash_fwd_sm100_fp4 mVarlenWork); None = the one-CTA-per-tile
    # varlen scheduler. varlen_tma_o: the batched sO + TMA O epilogue (flash_fwd_sm100_fp4 varlen_tma_o);
    # varlen_stage_skip: the 1-CTA causal stage skip on aligned documents (needs varlen_tma_o). All bitwise.
    varlen_work: Optional[torch.Tensor] = None,
    varlen_tma_o: bool = False,
    varlen_stage_skip: bool = False,
    # clc (opt-in): the work list's claiming -- 'static' (the boustrophedon deal, default), 'atomic' / 'clc'
    # (dynamic, flash_fwd_sm100_fp4 varlen_sched; varlen_work must then carry the counter words: varlen_worklist ctr=True)
    varlen_sched: str = "static",
    varlen_sched_stages: int = 2,
    varlen_sched_jit: bool = False,
    varlen_sched_early: bool = False,
    # lp_docmaps (aligned documents): O written to the caller's packed rows, document b's rows from o_cu_seqlens[b]
    # (int32 (b + 1,)); out is then the packed (T, h, dv) tensor (flash_fwd_sm100_fp4 o_rows_packed). LSE unchanged.
    o_cu_seqlens: Optional[torch.Tensor] = None,
    # lp_docmaps packed rows (with o_cu_seqlens): q / k / v are the PACKED codes too -- document b's Q rows (and its
    # LSE rows, (h, T)) from o_cu_seqlens[b], its K / V rows from kv_cu_seqlens_packed[b]; the scale factors stay per
    # aligned tile (cu_seqlens_q / _k: the aligned starts). flash_fwd_sm100_fp4 codes_packed.
    kv_cu_seqlens_packed: Optional[torch.Tensor] = None,
    # ... and the channel-major V's document starts (multiples of 16 keys; v then has at least cu_vt[-1] keys)
    vt_cu_seqlens_packed: Optional[torch.Tensor] = None,
    # sfd (with kv_cu_seqlens_packed): the dense per-row tables [EvQ, EvK, EvV (b h, R) uint8 (+ SF4Q, SF4K int32 on the
    # NVFP4 S route)] -- the kernel builds the SF atoms from them (flash_fwd_sm100_fp4 sf_dense; mSFQ / mSFK / mSFV
    # then only shape the SMEM / TMA views), and v is the row codes (MN-major; vt_cu_seqlens_packed = the key starts)
    sfd_rows: Optional[list] = None,
    # `mxfp8-pv-row-frame-p` (flash_fwd_sm100_fp4 pv_row_p): with MXFP8 P.V (E4M3 V + UE8M0 SFV per 32 keys),
    # P^ is the per-tensor 8-bit step's row-frame E4M3 P^ (offset / lazy max (4, 4), fused loop, FMA-pipe exp2 share)
    # and the P scale factors are a constant 1.0 tile: the block-scaled GEMM carries only V's token-block factors.
    # Changes P^ numerics against the in-kernel per-32-key P^ group scale (False); keyed.
    mxfp8_pv_row_p: bool = False,
    # `token-block-qk-fold` (flash_fwd_sm100_fp4 tbqk_fold): q / k are the token-block E4M3 codes WITHOUT their
    # SF tiles and tb_fac carries f_Q / f_K (rows 0 / 1): the S GEMM is plain E4M3 and f_Q f_K is folded into the
    # softmax's scale-subtract FFMA2 and row max (exact: powers of two). Needs mSFV and mxfp8_pv_row_p. Keyed.
    token_block_qk_fold: bool = False,
    # `reproduce-forward-p` (option A, all-4-bit forward only): (b, h, s_q / 128, REPRO_WORDS, 128) int32 the kernel fills with
    # the per-query frame / row sum / frame-change records the backward needs to rebuild this forward's P^ bit for bit;
    # repro_dbg (debug build): (b, h, s_q, s_k + s_k / 16) uint8, every E2M1 code and group SF byte
    repro_out: Optional[torch.Tensor] = None,
    repro_dbg: Optional[torch.Tensor] = None,
    # exact classifier: the per-16-key-group-position value functions of the exact all-FP4 step (flash_fwd_sm100_fp4 `p_mix`:
    # 8 characters over M / E / 2 / 3; reproduce-forward-p only). None = the configuration's single value function.
    p_mix: Optional[str] = None,
    # `two-cta-forward` (flash_fwd_sm100_fp4 use_2cta_forward): the block-scaled / FP8 fork forward on CTA pairs
    # (cta_group::2 MMAs of M = 256, each CTA loading half of every K / V tile). O and LSE are bitwise the 1-CTA
    # forward's. Generic softmax step, fixed-length MHA / GQA (no pack_gqa), head_dim 128 only; keyed. The plan
    # (lowprec.plan) decides it per call; False here keeps every direct caller on the 1-CTA kernel.
    use_2cta_forward: bool = False,
    # the forward kernel's compile-time settings and the driver's routing / scheduling settings (lowprec/specs.py): the
    # call's plan.specs.fwd / .host, required. quant_spec (plan.specs.quant) carries the causal-safe V opt-out the 4-bit
    # P.V guard below honours (None: no opt-out).
    spec: Optional[FwdKernelSpec] = None,
    host_spec: Optional[HostSpec] = None,
    quant_spec: Optional[QuantSpec] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity. 
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    if tile_mn is not None:
        m_block_size, n_block_size = tile_mn
    if _arch is not None and _compute_capability is None:
        _compute_capability = _arch // 10
    # prequantised-operands: bind caller-supplied quantised operands BEFORE anything reads q / k, so no bf16 Q or K is
    # ever touched. score_scale_unit is the derived compile-time specialisation (see the kwarg comments).
    score_scale_unit = False
    if q_quantized is not None or k_quantized is not None:
        (q, k, mSFQ, mSFK, qk_sf_vec_size, softmax_scale, score_scale_unit) = _bind_quantized_operands(
            q, k, _as_quantized_operand(q_quantized, "q_quantized"), _as_quantized_operand(k_quantized, "k_quantized"),
            mSFQ, mSFK, qk_sf_vec_size, softmax_scale, q_scale_fused)
        if isinstance(softmax_scale, torch.Tensor):
            # device descales: the score scale travels as a device scalar (float32, as Float32(host float) rounds)
            if not score_scale_unit:
                assert softmax_scale_t is None
                softmax_scale_t = softmax_scale.float().reshape(1)
            softmax_scale = 1.0  # a label only: the kernel reads softmax_scale_t (or the unit constant)
    elif q_scale_fused:
        raise ValueError("q_scale_fused=True needs q_quantized (the fold lives in the supplied Q's descale)")
    # Handle CUTE tensors - use them directly, no conversion needed
    # Only make contiguous if they are torch tensors
    if not isinstance(q, cute.Tensor):
        q, k = [maybe_contiguous(t) for t in (q, k)]
        # Block-scaled MXFP8 V is intentionally K-major (seqlen contiguous);
        # don't flatten it back to headdim-contiguous.
        v_is_kmajor_mxfp8 = (
            mSFV is not None
            and hasattr(v, "dtype")
            and v.dtype == getattr(torch, "float8_e4m3fn", None)
            and v.stride(v.dim() - 3) == 1  # the key mode: 1 in (b, s, h, d), 0 in packed varlen
        )
        if not v_is_kmajor_mxfp8:
            v = maybe_contiguous(v)
    num_head, head_dim = q.shape[-2:]
    # For FP4 packed dtypes (float4_e2m1fn_x2), the last dim is headdim/2.
    # For int8 FP4 buffers (from cute_tensor_like), shape already has full headdim -- no correction needed.
    if not isinstance(q, cute.Tensor) and hasattr(torch, 'float4_e2m1fn_x2') and q.dtype == torch.float4_e2m1fn_x2:
        head_dim = head_dim * 2
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    # For FP4 V (float4_e2m1fn_x2), V is passed as the K-major (= S-major) FP4
    # tensor with shape (b, h, d, s/2): each int8 byte holds two seqlen-adjacent
    # FP4 values (high/low nibble), so the FP4 stride for S is 0.5 byte = 1
    # element. nvfp4_quantize on `v.permute(0,2,3,1).reshape(b*h*d, s)` produces
    # this layout naturally.
    is_fp4_v = (
        not isinstance(v, cute.Tensor)
        and hasattr(torch, "float4_e2m1fn_x2")
        and v.dtype == torch.float4_e2m1fn_x2
    )
    if is_fp4_v:
        # v.shape = (b, h, d, s/2): num_head_kv = h, head_dim_v = d
        # diagonal-only-key-local-v-scale doubles the key axis (two payloads back to back)
        _v_keys_mult = 2 if vtok_diag else 1
        # (aligned varlen documents: the b = 1 buffer (1, h, d, total_k / 2) of the packed keys)
        # (cov5 v4_packed: v is a shape carrier of the key-parity copies, the kernel reads them from sfd_rows)
        _v4p = kv_cu_seqlens_packed is not None and sfd_rows is not None and len(sfd_rows) == 9
        assert v.shape[0] == (batch_size if cu_seqlens_k is None else 1) and (
            v.shape[3] * 2 == seqlen_k * _v_keys_mult or _v4p), (
            f"FP4 V shape {v.shape} must be (b={batch_size if cu_seqlens_k is None else 1}, h, d, "
            f"s/2={seqlen_k * _v_keys_mult // 2})"
        )
        assert not vtok_diag or (causal and vtok_bias is not None
                                 and ((cu_seqlens_k is None and seqused_k is None) or cu_seqlens_aligned or _v4p)
                                 and page_table is None and not (num_splits > 1)), (
            "vtok_diag needs causal, the vtok_bias / vtok_u form, and no paged / split KV (varlen: aligned documents)")
        num_head_kv = v.shape[1]
        head_dim_v = v.shape[2]
    else:
        head_dim_v = v.shape[-1]
    # For shape assertions, use the packed dim (what's actually in the tensor)
    head_dim_packed = q.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim_packed), f"k shape {k.shape} != expected {(batch_size, seqlen_k, num_head_kv, head_dim_packed)}"
            if not is_fp4_v:
                assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim_packed)
            if not is_fp4_v:
                assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim_packed)
        if not is_fp4_v:   # (packed codes: the channel-major V has its 16-aligned document columns)
            assert v.shape == (seqlen_k, num_head_kv, head_dim_v) or (
                vt_cu_seqlens_packed is not None and v.shape[0] >= seqlen_k
                and tuple(v.shape[1:]) == (num_head_kv, head_dim_v))
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    if spec is None or host_spec is None:
        raise TypeError("_flash_attn_fwd: spec (FwdKernelSpec) and host_spec (HostSpec) are required: the call's "
                        "plan.specs.fwd / .host (lowprec.plan.build_specs)")
    # Check if using FP4 (nvfp4)
    # Handle both torch and CUTE tensors
    is_cute_q = isinstance(q, cute.Tensor)
    if is_cute_q:
        q_dtype = q.element_type
        use_fp4 = q.element_type == cutlass.Float4E2M1FN 
        if not use_fp4:
            k_dtype = k.element_type if isinstance(k, cute.Tensor) else k.dtype
            v_dtype = v.element_type if isinstance(v, cute.Tensor) else v.dtype
            # Allow pure FP8 Q/K/V (no block scale) -- routes through
            # FlashAttentionForwardSm100 with kind::f8f6f4. Gated on HostSpec.fp8_pure_qk
            # (set by the plan on the 8-bit route); without it pure FP8 is rejected
            # and the block-scaled path is required.
            allowed_q_dtypes = [cutlass.Float16, cutlass.BFloat16]
            if mSFQ is not None or host_spec.fp8_pure_qk:
                allowed_q_dtypes.extend([cutlass.Float8E4M3FN, cutlass.Float8E5M2])
            assert q_dtype in allowed_q_dtypes, (
                "inputs must be float16/bfloat16, or FP8 when block-scaled QK scale factors are provided"
            )
            assert q_dtype == k_dtype, "Q and K must have the same dtype"
            _mixed_ok = bool(host_spec.fp8_pure_qk)
            if mSFQ is None and not _mixed_ok:
                assert q_dtype == v_dtype, "inputs must have the same dtype"
            else:
                _allowed_v = [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2,
                              torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
                assert v_dtype in _allowed_v, "V must be BF16/FP16 or FP8 when block-scaled QK is enabled"
    else:
        use_fp4 = is_nvfp4_dtype(q.dtype)
        if not use_fp4:
            allowed_q_dtypes = [torch.float16, torch.bfloat16]
            if mSFQ is not None or host_spec.fp8_pure_qk:
                allowed_q_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
            assert q.dtype in allowed_q_dtypes, (
                "inputs must be float16/bfloat16, or FP8 when block-scaled QK scale factors are provided"
            )
            assert q.dtype == k.dtype, "Q and K must have the same dtype"
            _mixed_ok = bool(host_spec.fp8_pure_qk)
            if mSFQ is None and not _mixed_ok:
                assert q.dtype == v.dtype, "inputs must have the same dtype"
            else:
                assert v.dtype in [
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ], "V must be BF16/FP16 or FP8 when block-scaled QK is enabled"
    
    # Store is_cute_q for later use
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    # Check CUDA device - CUTE tensors are always on device, torch tensors need .is_cuda check
    assert all(
        t is None or (isinstance(t, cute.Tensor) or t.is_cuda)
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    # Handle both torch and CUTE tensors for element_size
    if is_cute_q:
        # For CUTE tensors, FP4 is 4 bits but packed as 1 byte (2 values per byte)
        # For alignment purposes, use 1 byte for FP4
        if q.element_type == cutlass.Float4E2M1FN:
            q_element_size = 1
        else:
            q_element_size = q.element_type.width // 8
    else:
        q_element_size = q.element_size()
    alignment = 16 // q_element_size
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    use_blockscaled_impl = use_fp4 or mSFQ is not None or force_fp4_impl
    if softmax_scale_t is not None:
        # (a score_mod reads the device score scale too: score_mod(S softmax_scale_t), the kernel's log2 scale the change
        # of base -- per-tensor E4M3 operands with softcap / score_mod; softcap arrives here as its score_mod)
        assert use_blockscaled_impl and not score_scale_unit and softcap is None, \
            "softmax_scale_t: the block-scaled SM100 forward without softcap= / fused Q scale (pass softcap as score_mod)"
        assert softmax_scale_t.is_cuda and softmax_scale_t.dtype == torch.float32
        if softmax_scale_t.dim() == 2:
            # per-(document, kv head) score scales (per-tensor operands on aligned varlen documents): (n_docs, h_kv);
            # batched (no cu_seqlens): per-(batch, kv head), (b, h_kv) -- the kernel indexes it by the work tile's batch
            assert softmax_scale_t.is_contiguous() and (cu_seqlens_aligned or (
                cu_seqlens_q is None and cu_seqlens_k is None
                and tuple(softmax_scale_t.shape) == (batch_size, num_head_kv))), \
                "softmax_scale_t (n_docs, h_kv): aligned varlen documents, or (b, h_kv) on a batched call"
        else:
            assert softmax_scale_t.numel() == 1
            softmax_scale_t = softmax_scale_t.reshape(1)
    score_doc = softmax_scale_t is not None and softmax_scale_t.dim() == 2
    if score_scale_unit and not use_blockscaled_impl:
        # fail loudly rather than silently re-applying the factor
        raise ValueError(
            "q_scale_fused=True with a unit score scale is only honoured by the SM100 block-scaled forward "
            "(flash_fwd_sm100_fp4). Pass force_fp4_impl=True for per-tensor FP8 operands, or run with "
            "q_scale_fused=False (the factor then stays in the kernel's runtime scalar, numerically identical)."
        )
    if score_scale_unit and (score_mod is not None or softcap is not None or learnable_sink is not None):
        # with a score_mod the kernel hijacks softmax_scale_log2 as the pure change of base and applies
        # softmax_scale separately, so a unit scalar would silently mean something else; a learnable sink is
        # stated in unscaled score units, which the fold redefines.
        raise ValueError(
            "q_scale_fused=True cannot be honoured together with score_mod / softcap / learnable_sink: those "
            "paths use softmax_scale_log2 as a change of base, not as the whole score scale."
        )

    if not use_blockscaled_impl and not is_cute_q:
        # Non-block-scaled operands (BF16/FP16, or pure E4M3 Q/K/V with HostSpec.fp8_pure_qk) run
        # the upstream FlashAttentionForwardSm100 kernel. Its call signature moved under this route
        # (AuxData, the persistent-scheduler tensors, stream last) and the argument list kept below
        # rotted with it: every such call died with ARG_ANNOTATION_MISMATCH on mCuSeqlensQ. The
        # upstream interface owns that kernel's argument construction, so hand over to it; v_descale
        # is the same (b, h_k) V dequant scale in both APIs. Returns (out, lse) like the fork.
        from flash_attn.cute import interface as _upstream
        _tile_mn = tile_mn if tile_mn is not None else (
            (m_block_size, n_block_size) if (m_block_size, n_block_size) != (128, 128) else None)
        _arch_up = _arch if _arch is not None else (
            _compute_capability * 10 if _compute_capability is not None else None)
        _res = _upstream._flash_attn_fwd(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, seqused_q=seqused_q,
            seqused_k=seqused_k, page_table=page_table, softmax_scale=softmax_scale, causal=causal,
            softcap=softcap, window_size_left=window_size_left, window_size_right=window_size_right,
            learnable_sink=learnable_sink, tile_mn=_tile_mn, num_threads=num_threads,
            num_splits=num_splits, pack_gqa=pack_gqa, _arch=_arch_up, score_mod=score_mod,
            mask_mod=mask_mod, block_sparse_tensors=block_sparse_tensors, return_lse=return_lse,
            out=out, lse=lse, aux_tensors=aux_tensors, v_descale=None)
        _o = _res[0]
        if v_descale is not None:
            # `upstream-fp8-v-descale-wrong`: upstream's FlashAttentionForwardSm100 applies a (b, h_kv) v_descale wrongly
            # (FP8 Q / K / V, any shape: O cosine 0.62 and norm x50 against the dequantised reference; without it and
            # O * s_v the cosine is 0.9997). So it is applied here,
            # after the kernel, per (batch, kv head) broadcast over the query heads it serves: O = bf16(float(O) * v_d).
            _b, _h = _o.shape[0], _o.shape[2]
            _vd = v_descale.float().repeat_interleave(_h // v_descale.shape[1], dim=1)
            _o.copy_((_o.float() * _vd[:, None, :, None]).to(_o.dtype))
        return _o, _res[1]

    # Handle both torch and CUTE tensors for dtype and device
    if is_cute_q:
        # For CUTE tensors, we need to get the torch dtype for output allocation
        if q_dtype == cutlass.Float4E2M1FN or use_blockscaled_impl:
            out_torch_dtype = torch.bfloat16
        else:
            out_torch_dtype = cute2torch_dtype_map[q_dtype]
        device = torch.device('cuda')  # CUTE tensors are always on CUDA
    else:
        if use_blockscaled_impl:
            out_torch_dtype = torch.bfloat16
        elif q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and v.dtype in (torch.float16, torch.bfloat16):
            # Mixed FP8 QK + BF16/FP16 PV: output follows v_dtype (the widest operand).
            out_torch_dtype = v.dtype
        else:
            out_torch_dtype = q.dtype
        device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    if isinstance(q, torch.Tensor):
        requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    else:
        requires_grad = False # NOTE(Wenxuan): cute tensor has no grad attr, currently hardcode for inference

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        expected_out_shape = (*q_batch_seqlen_shape, num_head, head_dim_v)
        if o_cu_seqlens is not None:   # (packed O rows: any packed length)
            expected_out_shape = (out.shape[0], num_head, head_dim_v)
        assert out.shape == expected_out_shape, (
            f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        )
        if out.dtype != out_torch_dtype and out_torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) \
                and out.dtype in (torch.bfloat16, torch.float16):
            out_torch_dtype = out.dtype  # FP8 inputs with a 16-bit output (training keeps O in bf16 for D = rowsum(dO * O))
        assert out.dtype == out_torch_dtype, (
            f"out tensor dtype {out.dtype} does not match expected dtype {out_torch_dtype}"
        )
        assert out.device == device, (
            f"out tensor device {out.device} does not match input device {device}"
        )
        assert out.is_cuda, "out tensor must be on CUDA device"

    if return_pmax:
        # per-query P_max from the kernel's true row max (log domain in the kernel, exp'ed below); needs LSE
        assert use_blockscaled_impl and ((cu_seqlens_q is None and seqused_q is None) or cu_seqlens_aligned) \
            and num_splits == 1, "return_pmax: the block-scaled SM100 forward, batched or aligned varlen documents"
        return_lse = True
    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert lse.shape == lse_shape, (
            f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        )
        assert lse.dtype == torch.float32, (
            f"lse tensor dtype {lse.dtype} does not match expected dtype torch.float32"
        )
        assert lse.device == device, (
            f"lse tensor device {lse.device} does not match input device {device}"
        )
        assert lse.is_cuda, "lse tensor must be on CUDA device"

    # Get CUTE dtype - use directly if CUTE tensor, otherwise convert from torch dtype
    if is_cute_q:
        dtype = q.element_type
    else:
        dtype = torch2cute_dtype_map.get(q.dtype, cutlass.Float16) if not use_fp4 else cutlass.Float16


    compute_capability = (
        _get_device_capability()
        if _compute_capability is None
        else _compute_capability
    )

    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    use_block_sparsity = block_sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if head_dim == head_dim_v == 128 and not causal and not local and not use_block_sparsity:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False
        if tb_fac is not None:
            pack_gqa = False  # tb_fwd: one q head per row block (f_Q is indexed by the head)

    if num_splits < 1:
        max_seqlen_k = seqlen_k if cu_seqlens_k is None else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )
    if score_mod is not None:
        # varlen: without aux tensors the fork kernel's score_mod sees batch-local indices (SeqlenInfoQK), as
        # upstream (which no longer refuses this); with aux tensors the per-batch fastdiv wrap is not rebuilt here
        if is_varlen and aux_tensors is not None:
            raise NotImplementedError(
                "score_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if mask_mod is not None:
        if not use_block_sparsity and not use_blockscaled_impl:  # the fork kernel evaluates it per tile
            raise NotImplementedError(
                "mask_mod requires the use of block sparsity. This will be fixed in a future PR."
            )
        if is_varlen and aux_tensors is not None:  # as score_mod above
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    # Block-scaled QK ab/sf dtype must be in the compile key -- otherwise NVFP4 and
    # MXFP8 share the same key and the second mode silently reuses the kernel
    # compiled for the first (was producing NaN/inf for whichever was second).
    # We can derive the QK ab dtype from the Q tensor's element_type here (cute.Tensor
    # path) and from the FP4 hint (torch path); for the SF dtype, derive from mSFQ's
    # element_type if it's already a cute.Tensor, else infer from sf_vec_size which we
    # also derive from the SF tensor shape.
    if isinstance(q, cute.Tensor):
        _key_qk_ab_dtype = q.element_type
    elif use_fp4:
        _key_qk_ab_dtype = cutlass.Float4E2M1FN
    else:
        _key_qk_ab_dtype = torch2cute_dtype_map.get(q.dtype, cutlass.BFloat16)

    if mSFQ is None:
        _key_sf_dtype = None
    elif isinstance(mSFQ, cute.Tensor):
        _key_sf_dtype = mSFQ.element_type
    else:
        # The torch underlying tensor for an FP8/E4M3/E8M0 SF is plain int8 with no
        # dtype hint -- we have to infer from the AB dtype which is already known.
        # MXFP8 (Float8E4M3FN/E5M2 ab) uses E8M0 SF; NVFP4 (Float4E2M1FN ab) uses E4M3.
        if _key_qk_ab_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2) or qk_sf_vec_size == 32:
            _key_sf_dtype = cutlass.Float8E8M0FNU
        else:
            _key_sf_dtype = cutlass.Float8E4M3FN

    # v_dtype key: distinguish mixed FP8 QK + BF16 PV vs pure FP8 QK + FP8 PV.
    if isinstance(v, cute.Tensor):
        _key_v_dtype = v.element_type
    else:
        _key_v_dtype = torch2cute_dtype_map.get(v.dtype, cutlass.BFloat16)
    # Split-QK cross-tile pipeline (FP4 kernel): E2M1 P/V with block scales, no masking of
    # any kind (its softmax step applies none), dense non-paged KV whose length is a
    # multiple of the KV tile.
    _v_is_e4m3 = bool((isinstance(v, cute.Tensor) and v.element_type == cutlass.Float8E4M3FN)
                      or (hasattr(v, "dtype") and v.dtype == getattr(torch, "float8_e4m3fn", None)))
    _q_is_e4m3 = bool((isinstance(q, cute.Tensor) and q.element_type == cutlass.Float8E4M3FN)
                      or (hasattr(q, "dtype") and q.dtype == getattr(torch, "float8_e4m3fn", None)))
    _qk_split_ok = bool(
        use_fp4 and mSFV is not None
        and (not causal or host_spec.qk_split_causal)
        and not local
        and mask_mod is None
        and score_mod is None
        and not use_block_sparsity
        and not is_split_kv
        # (aligned varlen documents on the all-4-bit forward, which needs this pipeline: its per-document lengths
        # are seqused_q / _k -- the K rows past a document's length are the seqlen mask's, as seqused_k's)
        and ((cu_seqlens_q is None and cu_seqlens_k is None and seqused_q is None
              and (seqused_k is None or host_spec.qk_split_seqused))  # padded K rows + seqused_k: seqlen mask on the first tile
             or (cu_seqlens_aligned and spec.fp4all))
        and page_table is None
        and window_size_left is None
        and (window_size_right is None or (causal and window_size_right == 0))  # causal is encoded as right window 0
        and learnable_sink is None
        and m_block_size == 128
        and n_block_size == 128
        # K rows padded to the tile; a shorter used length goes through seqused_k (cov5: the packed rows, any T)
        and (seqlen_k % n_block_size == 0 or (kv_cu_seqlens_packed is not None and cu_seqlens_aligned))
        and (pv_sf_vec_size != 32 or spec.mxfp4_split)  # MXFP4 PV: integer frame + floored group constants
        and not _v_is_e4m3  # E4M3 V with block-scaled QK stays on the generic step
    )
    # `qk-split-pipe-overrides-keyed-flag`: FwdKernelSpec.qk_split_pipe is resolved HERE, by the kernel's single reader, so the
    # key item below is the pipeline that runs (the constructor's own call is idempotent).
    if tb_fac is not None:
        _qk_split_ok = False  # tb_fwd runs on the generic softmax step
    _qk_split_ok = FlashAttentionForwardSm100FP4.resolve_qk_split_pipe(_qk_split_ok, spec)
    if _qk_split_ok and (mSFV is None or _v_is_e4m3):
        raise NotImplementedError(
            "qk_split_pipe=True: the split-QK pipeline hosts the 4-bit (E2M1) P^ / V only; an E4M3 P^ (per-tensor or "
            "MXFP8) runs on the generic softmax step (the 8-bit split step was removed). Leave KernelTuning.qk_split_pipe "
            "unset or set it to False."
        )
    if tb_fac is not None:
        assert not _qk_split_ok, "tb_fac: qk_split_pipe=True is not supported (the generic softmax step)"
        assert tb_fac.dtype == torch.float32 and tb_fac.dim() == 4 and tb_fac.shape[0] >= 3, tb_fac.shape
        # (documents at 128-aligned starts in a b = 1 buffer, cu_seqlens_aligned: the fold forward reads its tables at
        # the document's offsets -- flash_fwd_sm100_fp4 tb_varlen; token_block_qk_fold stays batched)
        assert (mSFV is None or token_block_qk_fold) and page_table is None and (
            (cu_seqlens_aligned and not token_block_qk_fold)
            or (cu_seqlens_q is None and cu_seqlens_k is None and seqused_k is None))
    if token_block_qk_fold:
        assert tb_fac is not None and mSFV is not None and mSFQ is None and mSFK is None and mxfp8_pv_row_p, \
            "token_block_qk_fold: E4M3 Q / K codes + tb_fac (f_Q, f_K), MXFP8 V + mSFV, mxfp8_pv_row_p=True"
    _qk_split_mask_seqlen = bool(_qk_split_ok and seqused_k is not None)

    # ------------------------------------------------------------------ invariant 6: fail loudly
    # `causal-4bit-pv-unguarded` and `seqused-k-straddles-v-scale-group`. This is the one choke point
    # every 4-bit forward passes through -- the public routes, flash_attn_fp4all_fwd, the plain NVFP4/MXFP4
    # forward and the benchmark all call this function -- so the invariant is stated here rather than three times.
    #
    # The mechanism. A 4-bit PV needs V K-major (the block-scaled MMA raises otherwise), so V's scale group runs
    # along KEYS: 16 of them for NVFP4, 32 for MXFP4. One scale therefore reconstructs 16 or 32 consecutive keys
    # TOGETHER, and its amax is taken over all of them. Masking removes future keys from the row sum, but the
    # quantiser has already let them set the shared scale, so a masked output still depends on them. P cannot leak
    # (the mask is applied to S before the group amax) and dV / dK contract over queries and are clean; the carrier
    # is V alone, and on the backward side K's groups do the same to dQ.
    #
    # `key-local-v-scale` (vtok) is the fix: a per-key power-of-two V scale with every group SF = 1, so no scale
    # group spans sequence positions. It arrives as vtok_bias / vtok_w / vtok_tsf.
    # token-block scales: an E4M3 V with UE8M0 SFs (NVFP4 QK + MXFP8 PV) is an 8-bit PV -- its round-up key-grouped scale
    # leaks only through subnormals (the in-32-token-block channel, docs/lowprec.md), so it is not refused as a 4-bit PV
    _pv_is_4bit = bool(use_fp4 and mSFV is not None and not _v_is_e4m3)
    # pv_sf_vec_size is None on the NVFP4 default (the kernel resolves it to 16 further down); resolve it here so
    # the messages below name a number rather than None
    _pv_grp = int(pv_sf_vec_size) if pv_sf_vec_size is not None else 16
    _masked_any = bool(causal or local or (mask_mod is not None)
                       or window_size_left is not None or window_size_right is not None)
    _has_key_local_v = any(t is not None for t in (vtok_bias, vtok_w, vtok_tsf))
    if _pv_is_4bit and _masked_any and not _has_key_local_v:
        # causal_safe_v="never" is the explicit, recorded opt-out for a caller that wants the old behaviour
        # (an ablation, or reproducing a published leaky number). It is in the compile key.
        # QuantSpec.causal_safe_v="never" lets a benchmark or ablation reproduce the leaky configuration -- which is
        # what every published causal 4-bit number actually measures. The spec is in the compile key, so the two
        # kernels never share a cache slot.
        if quant_spec is None or quant_spec.causal_safe_v != "never":
            raise NotImplementedError(
                "causal (or local / mask_mod) attention with a 4-bit PV and key-grouped V scales lets a masked "
                "output depend on future V rows: V must be K-major for the block-scaled MMA, so its scale group "
                f"spans {_pv_grp} consecutive KEYS and their amax is shared. Pass a per-key V scale "
                "(vtok_bias / vtok_w / vtok_tsf, e.g. from fp4all_quantize_qkv_vtok), or use an 8-bit PV, or set "
                "QuantSpec(causal_safe_v=\"never\") to accept the leak deliberately."
            )
    # `seqused-k-straddles-v-scale-group`, the same mechanism without a mask. seqused_k is a per-batch
    # device tensor, so its values cannot be checked here without a host sync -- and a sync per call is not a price
    # this path can pay. So the requirement is stated and the combination is refused unless the caller acknowledges
    # it, which is the only honest option that stays free: whoever sets the acknowledgement has read this and has
    # either padded seqused_k up to the group or zeroed V's padding rows.
    # (aligned varlen documents, cu_seqlens_aligned: the padding rows are a documented contract -- zero, or finite and no
    # larger than the real rows -- so a partial group's amax is a real key's)
    if _pv_is_4bit and seqused_k is not None and not host_spec.seqused_padding_ok and not cu_seqlens_aligned:
        raise NotImplementedError(
            f"seqused_k with a 4-bit PV requires every used length to be a multiple of V's scale group "
            f"({_pv_grp} keys), and that cannot be checked here without a host sync. The last partial group "
            "mixes real keys with padding rows -- arbitrary content by the FlashAttention convention -- and the "
            "padding sets that group's amax, so it changes the reconstruction of the real keys sharing it; the "
            f"mask removes the padding from the sum only after the quantiser has used it. Round seqused_k up to "
            f"{_pv_grp}, or zero V's padding rows, then pass "
            "HostSpec(seqused_padding_ok=True). (Varlen document starts are "
            "separately asserted 128-aligned, which incidentally covers 16 and 32 -- that assert is load-bearing "
            "for this invariant too, not only for the d-axis grouping its docstring describes.)"
        )

    # Block sparsity with a SPARSE block size that differs from the kernel tile (upstream's q_subtile_factor): the
    # metadata's block_size=(sparse_q, tile_n) may cover several q tiles. The BF16 SM100 kernel supports it
    # (FlashAttentionForwardSm100.q_subtile_factor), but this interface normalised against the tile size and
    # rejected every such mask ("mask_block_cnt ... cannot be expanded"). Normalise once, here, with the same
    # q_stage the kernel is built with below, and key the kernel on the result.
    bs_q_subtile_factor = None
    bs_broadcast_pattern = None
    bs_normalized_early = None
    if block_sparse_tensors is not None and not use_blockscaled_impl and compute_capability == 10 and seqlen_q is not None:
        _bs_hd_pad = int(math.ceil(head_dim / 16) * 16)
        _bs_hdv_pad = int(math.ceil(head_dim_v / 16) * 16)
        # verbatim the kernel-construction expression below, so q_stage agrees with the compiled kernel
        _bs_2cta = (
            False  # block sparsity forces 1-CTA (see the kernel construction below); kept for the q_stage expression
            and compute_capability // 10 in [10, 11]
            and not causal and not local and not is_split_kv
            and cu_seqlens_q is None and seqused_q is None
            and page_size in [None, 128]
            and _bs_hd_pad in [128, 192] and _bs_hdv_pad == 128
            and seqlen_q_packgqa > 2 * m_block_size
            and (m_block_size % qhead_per_kvhead == 0 or not pack_gqa)
        )
        _bs_q_stage = 1 if (_bs_hdv_pad > 128 and not _bs_2cta) else 2
        bs_normalized_early, bs_broadcast_pattern, bs_q_subtile_factor = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_stage=_bs_q_stage,
        )
    # ---- constructor-argument resolution, BEFORE the key
    # These four booleans/counts specialise the kernel and used to be derived INSIDE the cache-miss
    # branch, i.e. after the key had been built without them. `is_persistent`
    # additionally mixes two keyed knobs with unkeyed varlen/split-KV state, and `pack_gqa` is
    # overridden to False on the block-scaled path -- the same shape of defect as
    # `qk-split-pipe-overrides-keyed-flag`. Resolved once here, keyed, and passed to whichever
    # constructor runs below, so the key and the kernel read one variable.
    _head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    _head_dim_v_padded = int(math.ceil(head_dim_v / 16) * 16)
    if use_blockscaled_impl:
        # PackGQA is off on the block-scaled kernel: in packed mode it derives the SF-Q layout from
        # the packed Q shape, which is not the SF memory layout (GQA cosine 0.92 instead of 0.98).
        pack_gqa = False
    # seqlen_q * qhead_per_kvhead: `_use_2cta_instrs` needs it only where cu_seqlens_q is None, so no
    # varlen host reduction is added here.
    _sq_packgqa_static = (seqlen_q or 0) * qhead_per_kvhead
    _use_2cta_instrs = bool(
        compute_capability // 10 in [10, 11]
        and not use_block_sparsity  # as upstream: the block-sparse forward path is 1-CTA
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and page_size in [None, 128]
        and _head_dim_padded in [128, 192]
        and _head_dim_v_padded == 128
        and _sq_packgqa_static > 2 * m_block_size
        and (m_block_size % qhead_per_kvhead == 0 or not pack_gqa)
    )
    _use_clc = bool(
        not (cu_seqlens_q is None and not causal and not local)
        and not (cu_seqlens_q is not None and qhead_per_kvhead == 1)
    )
    # TMEM budget: S 2x128 + O q_stage x hdim_v <= 512 columns. Without 2-CTA, hdim_v > 128 only fits
    # with a single Q stage per CTA.
    _q_stage = 1 if (_head_dim_v_padded > 128 and not _use_2cta_instrs) else 2
    # KernelTuning.fwd_causal_persistent=True: causal / local on the persistent LPT scheduler.
    # KernelTuning.fwd_nc_persistent=False: one CTA per tile for dense attention too (FlashInfer's cute-dsl kernel
    # launches that way); default persistent.
    # fwd_causal_persistent default None ("auto" (`persistent-causal-scheduler`, short causal cells only): the persistent
    # LPT grid on the pure-E4M3 fork route when causal (not local), fixed-length, seqlen_k <= 2048
    # (KernelTuning.fwd_causal_persistent_max_sk). Paired locked-clock medians of 3, b * s = 16384 tokens, rescale window (4, 4):
    # s_k 1024 +22 / +24 %, 2048 +9 / +11 % (h16 / h32); 4096 -1.1 / +0.8 %; 8192 -9 / -7 %; 16384 -12 / -9 %;
    # 32768 -8.6 / -5.2 %. O and LSE bitwise (a permutation of work tiles). "1" / "0" force it as before.
    _pure_e4m3_fork = bool(force_fp4_impl and not use_fp4 and mSFQ is None and mSFV is None
                           and _q_is_e4m3 and _v_is_e4m3)
    # mxfp8-pv-row-frame-p runs the pure-E4M3 generic step (its scheduler / epilogue / emulation-share defaults too)
    _pure_e4m3_fork = _pure_e4m3_fork or bool(mxfp8_pv_row_p and _q_is_e4m3 and _v_is_e4m3)
    _causal_pers = ("auto" if host_spec.fwd_causal_persistent is None
                    else ("1" if host_spec.fwd_causal_persistent else "0"))
    # `causal-balanced-persistent`, the "auto" default where it applies:
    # causal, fixed-length s_q == s_k <= 16384, 8-bit P.V (E4M3 V: the 8-bit entry and the 4-bit entry's E4M3-P.V
    # route), head_dim 128, generic softmax step -> the persistent grid with the balanced zigzag tile order
    # (PersistentZigzagLPTScheduler) AND the 1-CTA stage skip (causal_stage_skip). O and LSE bitwise (a permutation of
    # work tiles; the skipped steps are exact zeros). Paired medians of 3 vs the one-CTA-per-tile LPT grid, 8-bit
    # default: b4 s4096 -12.3 %, GQA h16/hk4 -12.8 %, b2 s8192 -4.3 %, b1 s16384 -0.5 %; vs the stripe-persistent
    # grid at b16 s1024 -7.2 %, b8 s2048 -12.3 %; 4-bit default b4 s4096 -8.7 %, b2 s8192 -3.7 %. b1 s32768: +1.4 %
    # inside the power-limit scatter (+-3 %), so not there.
    _causal_balanced = bool(
        _causal_pers == "auto" and causal and not local and cu_seqlens_q is None and cu_seqlens_k is None
        and seqused_q is None and seqused_k is None and page_table is None and not is_split_kv
        and not use_2cta_forward and not pack_gqa and seqlen_q is not None and seqlen_q == seqlen_k
        and seqlen_k <= 16384 and _v_is_e4m3 and not _qk_split_ok
        and _head_dim_padded == 128 and _head_dim_v_padded == 128
        and m_block_size == 128 and n_block_size == 128
        and score_mod is None and mask_mod is None and not use_block_sparsity and learnable_sink is None)
    _causal_pers_on = _causal_pers == "1" or _causal_balanced or (
        _causal_pers == "auto" and _pure_e4m3_fork and causal and not local and cu_seqlens_k is None
        and seqused_k is None and page_table is None
        and seqlen_k <= int(host_spec.fwd_causal_persistent_max_sk))
    _is_persistent = bool(
        ((not (causal or local) and host_spec.fwd_nc_persistent)
         or _causal_pers_on)
        and cu_seqlens_q is None and seqused_q is None and not is_split_kv
        and not (use_2cta_forward and (causal or local))   # the 2-CTA causal grid is one cluster per tile
    )
    # `register-direct-o-store` default (KernelTuning.fwd_o_regstore unset): the correction warps store O straight from registers
    # on the pure-E4M3 fork route, dense, fixed-length, once the key tile loop is long enough that the store's cost at
    # the tile end is amortised. Paired locked-clock medians against the sO + TMA store, rescale window (4, 4) on:
    # s_k 1024 -5.0 %, 2048 -4.0 %, 4096 +0.7 / -1.3 %, 8192 +2.6 %, 16384 +3.1 %, 32768 +2.1 %; causal -5.8 / -1.9 %.
    # O and LSE bitwise on every gate cell. Resolved here so the compile key carries it (the explicit setting is
    # FwdKernelSpec.o_regstore, in spec.key()).
    _o_regstore_default = bool(
        _pure_e4m3_fork and not causal and not local and cu_seqlens_q is None and cu_seqlens_k is None
        and seqused_q is None and seqused_k is None and page_table is None and not is_split_kv
        and seqlen_k >= int(host_spec.fwd_o_regstore_min_sk)
    )
    # `emulation-share-short-dense`: KernelTuning.fp8_generic_e2e's default. The FMA-pipe exp2 emulation share of the pure-E4M3
    # generic step: 16 of 128 elements' pairs (every 8th) on dense, fixed-length 2048 <= s_k < 8192, else 20 (every
    # 6th). Paired locked-clock medians of 3, every-8th vs every-6th: s_k 2048 +2.0 %, b4-s4096 +2.0 / +2.6 %, GQA
    # h16/hk4 s4096 +2.7 %, ragged 3001 +0.3 %; outside the window it loses (s_k 1024 -3.4 %, 8192 -2.0 %, 32768
    # -0.3 %, causal -6.7 / -8.0 %). FlashInfer's own 10 pairs measure -6 to -8 % here (MUFU-bound). Emulated and MUFU
    # exp2 agree to ulps: O cosine, norm ratio and LSE error unchanged to 6 digits. Keyed below.
    # The token-block row-P forward (mxfp8_pv_row_p) keeps every 6th pair on every cell -- its dense s4096 optimum
    # (b4 s4096 dense: 0.3176 ms at every 8th vs 0.3060 at every 6th; 7th 0.3159, 10th 0.3242), and the emulation
    # schedule its causal training arm ran
    # SM103 (B300): no emulation on any cell (FwdKernelSpec.rule_fp8_generic_e2e) -- MUFU.EX2 is no longer the wall.
    # The target architecture, once per call: FLASH_ATTENTION_ARCH (CPU-only compiles), else the device. It is in the
    # compile key and the kernel's constructor argument, so every SM103 rule is resolved from the same value.
    _sm103 = _get_device_arch() in (103,)
    _fp8_generic_e2e_default = FwdKernelSpec.rule_fp8_generic_e2e(_sm103, bool(
        _pure_e4m3_fork and not mxfp8_pv_row_p and not causal and not local and cu_seqlens_q is None and cu_seqlens_k is None
        and seqused_q is None and seqused_k is None and page_table is None and not is_split_kv
        and int(host_spec.fp8_e2e8_min_sk) <= seqlen_k < int(host_spec.fp8_e2e8_max_sk)
    ))
    _is_static_persistent = bool(not causal and not local and cu_seqlens_q is None
                                 and seqused_q is None and not is_split_kv)
    # driver-rule constructor arguments (FwdKernelSpec ctor fields): the spec's value, else the driver's rule
    _zigzag = (spec.causal_zigzag_order if spec.causal_zigzag_order is not None
               else (_causal_balanced and _is_persistent))
    _stage_skip = spec.causal_stage_skip if spec.causal_stage_skip is not None else _causal_balanced
    _varlen_tma_o = bool(varlen_tma_o and cu_seqlens_aligned and use_blockscaled_impl and not is_split_kv
                         and not pack_gqa and not use_2cta_forward)
    if use_2cta_forward and cu_seqlens_aligned:
        # two-cta-forward on aligned documents: the pair's epilogue is the sO + TMA store, so the batched TMA O
        # (whole 128-row tiles inside a document's slot, a tile past seqlen_q not stored) is required
        assert use_blockscaled_impl and not is_split_kv and not pack_gqa, \
            "use_2cta_forward on aligned documents: the block-scaled forward, no SplitKV / pack_gqa"
        _varlen_tma_o = True
    if _varlen_tma_o and varlen_stage_skip and spec.causal_stage_skip is None and not use_2cta_forward:
        # (_causal_balanced's kernel conditions, minus fixed length)
        _stage_skip = bool(causal and not local and score_mod is None and mask_mod is None
                           and not use_block_sparsity and learnable_sink is None and _v_is_e4m3 and not _qk_split_ok
                           and _head_dim_padded == 128 and _head_dim_v_padded == 128
                           and m_block_size == 128 and n_block_size == 128)
    if varlen_work is not None:
        assert cu_seqlens_aligned and use_blockscaled_impl and not is_split_kv and not pack_gqa \
            and not use_2cta_forward, "varlen_work: aligned varlen on the block-scaled 1-CTA forward only"
    _o_packed = o_cu_seqlens is not None
    if _o_packed and O_PACKED_EPI == "correction" and not use_2cta_forward and not (O_PACKED_TILE and _varlen_tma_o):
        # (packed O rows: the correction warps' predicated copies; the epilogue warp's single-warp copies measured
        # +36..40 % on the causal forward kernel)
        _varlen_tma_o = False
        _stage_skip = spec.causal_stage_skip if spec.causal_stage_skip is not None else _causal_balanced
    if _o_packed:
        # (cov2: the pair on packed rows with the sf_dense tables -- its sO epilogue warp's TMA tiles + row copies)
        assert cu_seqlens_aligned and use_blockscaled_impl and not is_split_kv and not pack_gqa \
            and (not use_2cta_forward or (sfd_rows is not None and kv_cu_seqlens_packed is not None)) \
            and o_cu_seqlens.dtype == torch.int32 \
            and o_cu_seqlens.shape == cu_seqlens_q.shape, \
            "o_cu_seqlens: aligned varlen on the block-scaled 1-CTA forward (no SplitKV / pack_gqa)"
    assert kv_cu_seqlens_packed is None or (_o_packed and kv_cu_seqlens_packed.dtype == torch.int32
                                            and kv_cu_seqlens_packed.shape == cu_seqlens_k.shape
                                            and vt_cu_seqlens_packed is not None
                                            and vt_cu_seqlens_packed.shape == cu_seqlens_k.shape), \
        "kv_cu_seqlens_packed: with o_cu_seqlens (packed codes)"
    compile_key = (
        # every compile-time setting of the forward kernel is a FwdKernelSpec field (value-keyed, never by
        # identity); the driver's routing settings reach the key through the resolved constructor arguments below.
        spec.key(),
        dtype,
        _key_v_dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        bs_q_subtile_factor,
        bs_broadcast_pattern,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        ("zero_pad_rows", bool(cu_seqlens_aligned)) if cu_seqlens_aligned else None,
        ("o_rows_packed",) if _o_packed else None,
        ("codes_packed",) if kv_cu_seqlens_packed is not None else None,
        ("sf_dense", len(sfd_rows), v.stride(-1) == 1) if sfd_rows is not None else None,
        (("varlen_work", varlen_work is not None, "tma_o", _varlen_tma_o)
         + ((("sched", varlen_sched, int(varlen_sched_stages), bool(varlen_sched_jit), bool(varlen_sched_early)),)
            if varlen_work is not None and varlen_sched != "static"
            else ())) if cu_seqlens_aligned else None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        ("sm103", _sm103),   # the SM103 rules (ld.red, emulation shares, tuning table) are traced in
        page_size not in [None, 128],  # paged KV non-TMA
        use_fp4,  # Include FP4 flag in compile key
        out_torch_dtype,  # output dtype (FP8 inputs may write a 16-bit O)
        _qk_split_ok,  # split-QK pipeline applicability (E2M1 PV, unmasked, seqlen_k % n_block == 0)
        _qk_split_mask_seqlen,
        mSFQ is not None,  # Include block-scaled path activation
        mSFQ is not None,  # Include scale factor flags
        mSFK is not None,
        mSFV is not None,
        v_descale is not None,  # FP8 V dequant scale present
        force_fp4_impl,
        pv_sf_vec_size,
        qk_sf_vec_size,  # was missing: silent kernel reuse across QK SF granularities
        _key_qk_ab_dtype,  # NVFP4 (Float4E2M1FN) vs MXFP8 (Float8E4M3FN/E5M2)
        _key_sf_dtype,     # E4M3 (NVFP4) vs E8M0 (MXFP8) -- was previously not in the
                           # key; both modes shared a slot and the second silently
                           # reused the kernel compiled for the first.
        local,
        return_pmax,
        spec.fp4all,  # all-FP4 training forward (integer frames, frame export in the pmax slot)
        vtok_bias is not None,
        vtok_w is not None,
        vtok_tsf is not None,
        tb_fac is not None,   # tb_fwd specialises the softmax step
        repro_out is not None, repro_dbg is not None,   # `reproduce-forward-p` export / debug dump
        None if p_mix is None else str(p_mix).upper(),  # exact classifier: per-group-position value functions
        tb_tok_v is not None,  # token-local operands: per-key V factors
        # diagonal-only-key-local-v-scale: specialises every softmax step (the masked ones keep the per-key bias /
        # row-sum weight, the unmasked ones compile it out) and the load warp's V tile index
        vtok_diag,
        softmax_scale_t is not None,  # GPU-only score scale: the kernel reads it from device memory
        score_doc,  # ... per (document, kv head), per work tile
        # prequantised-operands: q_scale_fused is host-side (it only changes which scalars are multiplied together on the
        # host), score_scale_unit specialises the kernel body. Both are keyed so that two
        # values in one process cannot silently share a cubin.
        q_scale_fused,
        score_scale_unit,
        # the constructor arguments resolved above, keyed on the value the
        # constructor actually receives.
        _use_2cta_instrs,
        _q_stage,
        _use_clc,
        _is_persistent,
        _causal_balanced,
        _is_static_persistent,
        _o_regstore_default,
        _fp8_generic_e2e_default,
        host_spec.fp8_pure_qk,
        _zigzag,
        _stage_skip,
        # `tmem-store-rep-presence-test`: KernelTuning.fp8_pv_tmem_store_rep is spec.fp8_pv_tmem_store_rep,
        # presence (None) and value alike, so spec.key() above keys it; no raw environment item is needed.
        # the settings `max_offset` and `_lse_log2_offset()` resolve from, named
        # explicitly rather than left inside the tuning tuple, so the softmax/LSE agreement is
        # readable off the key. On the MXFP4 split path the two resolved values are one kernel constant
        # (MX_INT_FRAME_OFFSET), so they agree by construction and the threshold is not an input to either
        # (`mxfp4-threshold-invisible-to-lse` was a misdiagnosis).
        spec.fp4_pv_rescale_threshold,
        spec.fp8_pv_log2_offset,
        spec.fp8_pv_rescale_threshold,
        ("mxfp8_pv_row_p", bool(mxfp8_pv_row_p)),
        ("token_block_qk_fold", bool(token_block_qk_fold)),
        ("use_2cta_forward", bool(use_2cta_forward)),
    )
    pmax_log = torch.empty(lse_shape, dtype=torch.float32, device=device) if return_pmax else None
    fp4_qk = use_fp4 and not is_cute_q
    if fp4_qk and (cu_seqlens_q is not None or cu_seqlens_k is not None) and not cu_seqlens_aligned:
        # Varlen FP4: documents must start on a 128-row boundary in the packed tensor. A block-scaled
        # scale-factor tensor's sequence mode is ((32, 4), n_atoms) -- one atom per 128 rows, and the
        # within-atom map r -> 16*(r%32) + 4*(r//32) is not linear in r -- so a document offset that is
        # not a whole number of atoms cannot be expressed as a tensor offset at all. Pad each document
        # up to a multiple of 128 when packing; everything still runs as ONE kernel call.
        # (The scale factors themselves are varlen-safe: each is a max within one token's row along d,
        # so a 128-row atom spanning a document boundary is identical to quantising each alone.)
        for _nm, _cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
            if _cu is not None:
                assert bool((_cu[:-1] % 128 == 0).all()), (
                    f"FP4 varlen needs 128-row-aligned document starts, got {_nm}={_cu.tolist()}. "
                    f"Pad each document to a multiple of 128 when packing."
                )
    # FP4 V also needs the make_ptr path: dlpack reports half-headdim shape for
    # float4_e2m1fn_x2, but the kernel needs to know the full headdim (and the
    # K-major stride) to build SFV's TMA descriptor correctly.
    fp4_v = (
        use_fp4 and not isinstance(v, cute.Tensor)
        and hasattr(torch, "float4_e2m1fn_x2")
        and v.dtype == torch.float4_e2m1fn_x2
    )
    # Compute q_shape, k_shape and qk_ab_dtype for pointer-based Q/K path (used by FP4 kernel)
    if fp4_qk:
        q_ptr_shape = tuple(int(s) for s in (*q.shape[:-1], q.shape[-1] * 2))
        k_ptr_shape = tuple(int(s) for s in (*k.shape[:-1], k.shape[-1] * 2))
        qk_ab_dtype = cutlass.Float4E2M1FN
    elif is_cute_q:
        q_ptr_shape = tuple(int(s) for s in q.shape)
        k_ptr_shape = tuple(int(s) for s in k.shape)
        qk_ab_dtype = q.element_type
    else:
        q_ptr_shape = tuple(int(s) for s in q.shape)
        k_ptr_shape = tuple(int(s) for s in k.shape)
        qk_ab_dtype = torch2cute_dtype_map.get(q.dtype, cutlass.BFloat16)
    if fp4_v:
        # K-major V FP4 tensor: torch shape (b, h, d, s/2) packs 2 seqlen-adjacent
        # FP4 per byte. Convert to logical (b, s, h, d) for the kernel.
        b_v, h_v, d_v, s_half = (int(x) for x in v.shape)
        # aligned varlen documents: the packed keys' K-major V, (total_k, h, d) without a batch mode
        v_ptr_shape = (b_v, s_half * 2, h_v, d_v) if cu_seqlens_k is None else (s_half * 2, h_v, d_v)
    else:
        v_ptr_shape = ()
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        # Extract scale factor dtype and vec_size before converting tensors
        sf_dtype = None
        sf_vec_size = 16  # Default for FP4
        if use_blockscaled_impl and mSFQ is not None:
            if isinstance(mSFQ, cute.Tensor):
                sf_dtype = mSFQ.element_type
            else:
                # Convert torch dtype to CUTLASS dtype
                sf_dtype = torch2cute_dtype_map.get(mSFQ.dtype, cutlass.Float8E4M3FN)
                if (
                    sf_dtype == cutlass.Float8E4M3FN
                    and qk_ab_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}
                    and mSFQ.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}
                ):
                    # Float8E8M0FNU scale tensors currently round-trip through the benchmark
                    # as byte-backed torch tensors, so infer MXFP8 scale metadata from Q/K dtype.
                    sf_dtype = cutlass.Float8E8M0FNU
                if qk_sf_vec_size == 32:  # MXFP4 QK: byte-backed UE8M0 SFs per 32
                    sf_dtype = cutlass.Float8E8M0FNU
            # Set default sf_vec_size based on sf_dtype
            if sf_dtype == cutlass.Float8E4M3FN:
                sf_vec_size = 16
            elif sf_dtype == cutlass.Float8E8M0FNU:
                sf_vec_size = 32
            else:
                raise ValueError(f"Invalid scale factor dtype: {sf_dtype}")
        
        # For torch FP4 tensors: use make_ptr path (kernel builds tensor from pointer + shape)
        # For cute tensors or non-FP4 torch tensors: use to_cute_tensor as before
        if fp4_qk:
            from cutlass.cute.runtime import make_ptr
            q_tensor = make_ptr(qk_ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
            k_tensor = make_ptr(qk_ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        else:
            q_tensor = to_cute_tensor(q)
            k_tensor = to_cute_tensor(k)
        if fp4_v:
            from cutlass.cute.runtime import make_ptr
            v_tensor = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        else:
            # MXFP8 PV passes V K-major (seqlen contiguous, dim 1); everything
            # else is headdim-contiguous.
            _v_kmajor = (
                mSFV is not None
                and hasattr(v, "dtype")
                and v.dtype == getattr(torch, "float8_e4m3fn", None)
                and v.stride(-1) != 1   # (sfd: the row codes, MN-major -- lp_docmaps.V_ROWS)
            )
            # (varlen: packed (total, h, d), the key mode is 0)
            v_tensor = to_cute_tensor(v, leading_dim=(v.dim() - 3) if _v_kmajor else -1)
        o_tensor = to_cute_tensor(out if not is_split_kv else out_partial)
        # Pass through scale factor tensors when using the SM100 block-scaled kernel.
        mSFQ_tensor = mSFK_tensor = mSFV_tensor = None
        v_descale_tensor = None
        if use_blockscaled_impl:
            mSFQ_tensor = to_cute_tensor(mSFQ, leading_dim=3, assumed_align=16) if mSFQ is not None else None
            mSFK_tensor = to_cute_tensor(mSFK, leading_dim=3, assumed_align=16) if mSFK is not None else None
            mSFV_tensor = to_cute_tensor(mSFV, leading_dim=3, assumed_align=16) if mSFV is not None else None
            # Per-(batch, kv-head) FP8 V dequant scale (Float32, shape [b, h_kv]).
            v_descale_tensor = to_cute_tensor(v_descale, leading_dim=1, assumed_align=4) if v_descale is not None else None
        elif v_descale is not None:
            # Non-block-scaled (pure-FP8) route: this lands on upstream's forward, which takes the
            # descales as a DescaleTensors in its own argument slot. Previously a bare None was
            # inserted there and v_descale was silently dropped, so pure-FP8 ran unit-descale.
            v_descale_tensor = to_cute_tensor(v_descale, leading_dim=1, assumed_align=4)
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if block_sparse_tensors is not None:
            if seqlen_q is None:
                raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
            if bs_normalized_early is None:
                expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
                    batch_size, num_head, seqlen_q, seqlen_k,
                    m_block_size, n_block_size, _q_stage,  # was compute_capability (the q_stage slot)
                )
            if bs_normalized_early is not None:
                compile_time_normalized = bs_normalized_early
            else:
                compile_time_normalized = normalize_block_sparse_tensors(
                    block_sparse_tensors,
                    expected_count_shape=expected_count_shape,
                    expected_index_shape=expected_index_shape,
                )
            sparse_tensors = to_cute_block_sparse_tensors(compile_time_normalized)

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif compute_capability == 10:
            if use_blockscaled_impl:
                # Use the SM100 block-scaled attention kernel for NVFP4 / MXFP8 QK.
                if sf_dtype is None:
                    sf_dtype = cutlass.Float8E4M3FN  # Default scale factor dtype for FP4
                # Validate dtype and scale factor combinations
                ab_dtype = q_tensor.value_type if fp4_qk else q_tensor.element_type
                if not force_fp4_impl and not is_valid_dtypes_and_scale_factor_vec_size(ab_dtype, sf_dtype, sf_vec_size):
                    raise ValueError(
                        f"Invalid dtype combination: ab_dtype={ab_dtype}, "
                        f"sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}"
                    )
                # MXFP8 PV: NVFP4 QK with E4M3 V + SFV present -> P/V use
                # sf_vec 32 with E8M0 scale factors.
                # token-block scales: also with MXFP8 Q / K (sf_vec_size 32, UE8M0), i.e. pure MXFP8 Q, K, V
                _mxfp8_pv = (
                    mSFV is not None
                    and sf_vec_size in (16, 32)
                    and (
                        (isinstance(v, cute.Tensor) and v.element_type == cutlass.Float8E4M3FN)
                        or (hasattr(v, "dtype") and v.dtype == getattr(torch, "float8_e4m3fn", None))
                    )
                )
                # MXFP4 PV: FP4 V + SFV with pv_sf_vec_size=32 -> E2M1 P/V with
                # E8M0 SFs per 32 (tcgen05 kind::mxf4, scale_vec::2X).
                _mxfp4_pv = mSFV is not None and pv_sf_vec_size == 32 and not _mxfp8_pv
                _e8m0_pv = _mxfp8_pv or _mxfp4_pv
                fa_fwd = FlashAttentionForwardSm100FP4(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    # resolved above, before the compile key, so the key sees the same value
                    is_persistent=_is_persistent,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, 128],
                    is_varlen_q=cu_seqlens_q is not None
                        or seqused_q is not None,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    sf_dtype_pv=cutlass.Float8E8M0FNU if _e8m0_pv else None,
                    sf_vec_size_pv=32 if _e8m0_pv else None,
                    pv_elem_width=4 if _mxfp4_pv else (8 if _mxfp8_pv else None),
                    qk_elem_width=4 if ab_dtype == cutlass.Float4E2M1FN else None,
                    qk_split_pipe=_qk_split_ok,
                    qk_split_mask_seqlen=_qk_split_mask_seqlen,
                    return_pmax=return_pmax,
                    # prequantised-operands: the score scalar came out bitwise 1.0, so the kernel drops every
                    # multiplication by it (row-max scaling, correction factor, LSE, masked-tile constants).
                    score_scale_unit=score_scale_unit,
                    vtok_diag=vtok_diag,
                    p_mix=p_mix,
                    o_regstore_default=_o_regstore_default,
                    fp8_generic_e2e_default=_fp8_generic_e2e_default,
                    is_sm103=_sm103,
                    mxfp8_pv_row_p=bool(mxfp8_pv_row_p),
                    token_block_qk_fold=bool(token_block_qk_fold),
                    use_2cta_forward=bool(use_2cta_forward),
                    causal_zigzag_order=_zigzag,
                    causal_stage_skip=_stage_skip,
                    zero_pad_rows=bool(cu_seqlens_aligned and use_blockscaled_impl),
                    varlen_tma_o=_varlen_tma_o,
                    **({"o_rows_packed": True} if _o_packed else {}),
                    **({"codes_packed": True} if kv_cu_seqlens_packed is not None else {}),
                    **({"sf_dense": True} if sfd_rows is not None else {}),
                    **({"score_scale_doc": True} if score_doc else {}),
                    **({"varlen_sched": varlen_sched, "varlen_sched_stages": int(varlen_sched_stages),
                        "varlen_sched_jit": bool(varlen_sched_jit), "varlen_sched_early": bool(varlen_sched_early)}
                       if varlen_work is not None and varlen_sched != "static" else {}),
                    spec=spec,
                )
            else:
                # _use_2cta_instrs / _use_clc / _q_stage / _is_static_persistent are resolved before
                # the compile key, so the key and the kernel read one variable.
                fa_fwd = FlashAttentionForwardSm100(
                    head_dim,
                    head_dim_v,
                    q_stage=_q_stage,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    is_static_persistent=_is_static_persistent,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, 128],
                    is_varlen_q=cu_seqlens_q is not None
                        or seqused_q is not None,
                    use_2cta_instrs=_use_2cta_instrs,
                    use_clc_scheduler=_use_clc,
                    q_subtile_factor=bs_q_subtile_factor,
                )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        fake_stream = cute.runtime.make_fake_stream()
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            # current_stream,
            fake_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            sparse_tensors,
            cute_aux_tensors,
        ]
        if not use_blockscaled_impl:
            # FlashAttentionForwardSm100 (the BF16 kernel) takes descale_tensors BETWEEN learnable_sink and
            # blocksparse_tensors; the FP4 kernel has no such slot. Passing the FP4 order to both made every
            # BF16 block-sparse forward fail with "argument #17 (descale_tensors) ... got BlockSparseTensors" --
            # invisible until the upstream mask_mod tests could run.
            # [0]=kernel ... [15]=learnable_sink, [16]=descale_tensors
            from flash_attn.cute.flash_fwd_sm100 import DescaleTensors as _DescaleTensors
            compile_args.insert(16, _DescaleTensors(v_descale=v_descale_tensor)
                                if v_descale_tensor is not None else None)
        # Add scale factor tensors if using the block-scaled SM100 kernel
        if use_blockscaled_impl:
            compile_args.extend([mSFQ_tensor, mSFK_tensor, mSFV_tensor, v_descale_tensor,
                                 to_cute_tensor(pmax_log, assumed_align=4) if return_pmax else None])
        # Add q/k shapes for the block-scaled kernel (it always builds tensors from pointer + shape)
        if use_blockscaled_impl:
            if fp4_qk:
                sym_q_shape = tuple(cutlass.Int32(0) for _ in q_ptr_shape)
                sym_k_shape = tuple(cutlass.Int32(0) for _ in k_ptr_shape)
            else:
                sym_q_shape = tuple(cutlass.Int32(0) for _ in range(len(q_tensor.shape)))
                sym_k_shape = tuple(cutlass.Int32(0) for _ in range(len(k_tensor.shape)))
            compile_args.extend([sym_q_shape, sym_k_shape])
            if fp4_v:
                sym_v_shape = tuple(cutlass.Int32(0) for _ in v_ptr_shape)
                compile_args.append(sym_v_shape)
                if vtok_bias is not None:
                    compile_args.extend([to_cute_tensor(_vtok_kernel_view(vtok_bias), assumed_align=4),
                                         to_cute_tensor(_vtok_kernel_view(vtok_u), assumed_align=4)])
                elif vtok_w is not None:
                    compile_args.extend([None, None, to_cute_tensor(_vtok_w_view(vtok_w), assumed_align=4)])
                elif vtok_tsf is not None:
                    compile_args.extend([None, None, None, to_cute_tensor(vtok_tsf, leading_dim=3, assumed_align=16)])
        if softmax_scale_t is not None or tb_fac is not None:
            # pad the optional V-side slots up to mVTokSF, then the score-scale tensor (see _score_scale_tail)
            _ct = lambda t: None if t is None else to_cute_tensor(t, assumed_align=4, leading_dim=t.dim() - 1)  # noqa: E731
            compile_args.extend(_score_scale_tail(len(compile_args) - _N_FIXED_ARGS - 1, _ct(softmax_scale_t)))
        if tb_fac is not None:
            compile_args.append(to_cute_tensor(tb_fac, assumed_align=4))   # mTbFac, after mSoftmaxScale
        if repro_out is not None or repro_dbg is not None or tb_tok_v is not None:
            compile_args.extend(_repro_tail(len(compile_args) - _N_FIXED_ARGS - 1,
                                            None if repro_out is None else to_cute_tensor(repro_out, assumed_align=16),
                                            None if repro_dbg is None else to_cute_tensor(repro_dbg, assumed_align=1)))
        if tb_tok_v is not None:
            compile_args.append(to_cute_tensor(tb_tok_v, assumed_align=4))   # mTbTokV, after mReproDbg (token-local operands)
        if varlen_work is not None:
            compile_args.extend(_varlen_work_tail(len(compile_args) - _N_FIXED_ARGS - 1,
                                                  to_cute_tensor(varlen_work, assumed_align=4, leading_dim=0)))
        if _o_packed:
            compile_args.extend(_o_rows_tail(len(compile_args) - _N_FIXED_ARGS - 1,
                                             to_cute_tensor(o_cu_seqlens, assumed_align=4, leading_dim=0)))
            if kv_cu_seqlens_packed is not None:   # mCuSeqlensKc, mCuSeqlensVt after mCuSeqlensO
                compile_args.append(to_cute_tensor(kv_cu_seqlens_packed, assumed_align=4, leading_dim=0))
                compile_args.append(to_cute_tensor(vt_cu_seqlens_packed, assumed_align=4, leading_dim=0))
                if sfd_rows is not None:
                    compile_args.append([to_cute_tensor(t, assumed_align=4) for t in sfd_rows])
        _copts = "--enable-tvm-ffi"
        if spec.ptxas_options:   # FwdKernelSpec.ptxas_options (opt-in, in spec.key())
            _copts += f" --ptxas-options '{spec.ptxas_options}'"
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            *compile_args,
            options=_copts,
        )

    # Expand block sparse tensors to match actual head count (may be broadcast from 1)
    normalized_block_sparse_tensors = bs_normalized_early
    if block_sparse_tensors is not None and bs_normalized_early is None:
        expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
            batch_size, num_head, seqlen_q, seqlen_k,
            m_block_size, n_block_size, _q_stage,  # was compute_capability (the q_stage slot)
        )
        normalized_block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
        )
    if fp4_qk:
        from cutlass.cute.runtime import make_ptr as _make_ptr
        q_data_ptr = q.data_ptr() if hasattr(q, 'data_ptr') else q.iterator.data_ptr
        k_data_ptr = k.data_ptr() if hasattr(k, 'data_ptr') else k.iterator.data_ptr
        q_call = _make_ptr(qk_ab_dtype, q_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
        k_call = _make_ptr(qk_ab_dtype, k_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
    else:
        q_call = q
        k_call = k
    if fp4_v:
        from cutlass.cute.runtime import make_ptr as _make_ptr
        v_data_ptr = v.data_ptr()
        v_call = _make_ptr(cutlass.Float4E2M1FN, v_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
    else:
        v_call = v
    call_args = [
        q_call,
        k_call,
        v_call,
        out if not is_split_kv else out_partial,
        lse_partial if is_split_kv else lse,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        window_size_left,
        window_size_right,
        learnable_sink,
        # The kernel's BlockSparseTensors struct has exactly these 8 fields. Passing the BlockSparseTensorsTorch
        # NamedTuple itself (which also carries block_size and spt) failed the runtime tuple-arity check -- the
        # same explicit form upstream uses and the backward already uses.
        (
            normalized_block_sparse_tensors.mask_block_cnt,
            normalized_block_sparse_tensors.mask_block_idx,
            normalized_block_sparse_tensors.full_block_cnt,
            normalized_block_sparse_tensors.full_block_idx,
            normalized_block_sparse_tensors.cu_total_m_blocks,
            normalized_block_sparse_tensors.cu_block_idx_offsets,
            normalized_block_sparse_tensors.dq_write_order,
            normalized_block_sparse_tensors.dq_write_order_full,
        )
        if normalized_block_sparse_tensors is not None
        else None,
        aux_tensors,
    ]
    if not use_blockscaled_impl:
        # descale_tensors slot of the BF16 kernel (see the compile-time comment)
        from flash_attn.cute.flash_fwd_sm100 import DescaleTensors as _DescaleTensors
        call_args.insert(15, _DescaleTensors(v_descale=v_descale)
                         if v_descale is not None else None)

    # Add scale factor tensors if using the block-scaled SM100 kernel
    if use_blockscaled_impl:
        call_args.extend([mSFQ, mSFK, mSFV, v_descale, pmax_log])
    # Add q/k shapes for the block-scaled kernel
    if use_blockscaled_impl:
        call_args.extend([q_ptr_shape, k_ptr_shape])
        if fp4_v:
            call_args.append(v_ptr_shape)
            if vtok_bias is not None:
                call_args.extend([_vtok_kernel_view(vtok_bias), _vtok_kernel_view(vtok_u)])
            elif vtok_w is not None:
                call_args.extend([None, None, _vtok_w_view(vtok_w)])
            elif vtok_tsf is not None:
                call_args.extend([None, None, None, vtok_tsf])
    if softmax_scale_t is not None or tb_fac is not None:
        call_args.extend(_score_scale_tail(len(call_args) - _N_FIXED_ARGS, softmax_scale_t))
    if tb_fac is not None:
        call_args.append(tb_fac)
    if repro_out is not None or repro_dbg is not None or tb_tok_v is not None:
        call_args.extend(_repro_tail(len(call_args) - _N_FIXED_ARGS, repro_out, repro_dbg))
    if tb_tok_v is not None:
        call_args.append(tb_tok_v)
    if varlen_work is not None:
        call_args.extend(_varlen_work_tail(len(call_args) - _N_FIXED_ARGS, varlen_work))
    if _o_packed:
        call_args.extend(_o_rows_tail(len(call_args) - _N_FIXED_ARGS, o_cu_seqlens))
        if kv_cu_seqlens_packed is not None:
            call_args.extend([kv_cu_seqlens_packed, vt_cu_seqlens_packed])
            if sfd_rows is not None:
                call_args.append(list(sfd_rows))
                SFD_FWD_CALLS[0] += 1

    _flash_attn_fwd.compile_cache[compile_key](*call_args)

    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    if return_pmax:
        if spec.fp4all:
            return out, lse, pmax_log  # all-4-bit forward: the final integer log2 frame per query (the backward's P^ frame)
        return out, lse, pmax_log.exp_()  # P_max = exp(max_k S * scale - LSE); 0 for empty rows
    return out, lse


def _vtok_kernel_view(t: torch.Tensor) -> torch.Tensor:
    """(b, s_k, h_k) per-key vector -> contiguous (b, h_k, s_k) float32, the kernel's [batch, kv head, key] indexing."""
    return t.permute(0, 2, 1).to(torch.float32).contiguous()


def _vtok_w_view(t: torch.Tensor) -> torch.Tensor:
    """(b, h_k, tiles, 256) -> contiguous (b, h_k, tiles * 256) float32: kernel [batch, kv head, None] + n_block * 256."""
    b, h, n, w = t.shape
    return t.reshape(b, h, n * w).to(torch.float32).contiguous()


# Compile cache for the block-scaled forward (keyed on the FP4-aware key built above).
_flash_attn_fwd.compile_cache = {}

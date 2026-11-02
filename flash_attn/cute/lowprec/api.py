"""Public entry points: flash_attn_fp4 and flash_attn_fp8.

Each call resolves one immutable plan (``lowprec.plan``) before anything is quantised or compiled, then runs it
(``lowprec._routes``). With an input that requires grad the call is a ``torch.autograd.Function`` whose context
carries the plan to the backward; nothing is re-resolved there and no other configuration exists. Plans are cached per
call signature (the arguments and the tensors' metadata), so a training loop resolves each distinct call once.

The node runs as the registered op ``lowbit_attn::fp4`` / ``::fp8`` (lowprec._op: SAC can save it, torch.compile
traces it) for every call form expressible as an op; user score_mod / mask_mod / aux_tensors and pre-quantised
operands keep the ``autograd.Function`` (``_LowPrecAttnFunc``) directly.
"""
from __future__ import annotations

import collections
import threading
import warnings
from typing import Callable, Literal, Optional, Tuple, Union

import torch

from . import _op  # noqa: F401  (registers the ops lowbit_attn::fp4 / ::fp8 on import)
from .options import FP4AttnOptions, FP8AttnOptions
from .plan import AttnPlan, resolve_fp4, resolve_fp8

__all__ = ["flash_attn_fp4", "flash_attn_fp8"]


class _Routes:
    """lowprec._routes, imported on first use: it imports the drivers, which import lowprec.specs (this package), so an
    eager import here would make `import flash_attn.cute.interface_fwd_fp4` (or any driver) first a circular import."""

    def __getattr__(self, name):
        from . import _routes
        return getattr(_routes, name)


_routes = _Routes()


class _LowPrecAttnFunc(torch.autograd.Function):
    """One autograd node per call (per repacked document batch under varlen / seqused_k). ctx.plan is the only
    configuration the backward sees; ctx.st the route's saved state (lowprec._routes.Saved); masks: the call's
    lp_masks.LPMasks (or None), an explicit argument as well."""

    @staticmethod
    def forward(ctx, q, k, v, plan: AttnPlan, prequant, masks):
        ctx.plan = plan
        st = _routes.Saved()
        out, lse = _routes.forward(plan, st, q, k, v, prequant, masks)
        ctx.save_for_backward(*st.saved_tensors)
        st.saved_tensors = ()
        ctx.st = st
        if plan.fwd.return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, dout, *unused):
        st = ctx.st
        st.saved_tensors = ctx.saved_tensors
        try:
            dq, dk, dv = _routes.backward(ctx.plan, st, dout.contiguous())
        finally:
            st.saved_tensors = ()
        return dq, dk, dv, None, None, None


_REPACK_WARNED = False
REPACK_WARNING = (
    "flash_attn_{entry} with cu_seqlens_q / cu_seqlens_k but without seqused_q / seqused_k: the documents are COPIED "
    "into a repacked buffer with every start rounded up to 128 (or 256) tokens -- one extra read and write of q, k, v "
    "and o per pass (and of dO, dQ, dK, dV in the backward), up to 255 padding rows per document -- and unpacked "
    "afterwards. The zero-copy fast path takes the documents in place: lay each document out at a start that is a "
    "multiple of 128 (cu_seqlens = those starts, cu_seqlens[-1] = total, total % 128 == 0) and pass its true length as "
    "seqused_q / seqused_k (int32, (b,)): flash_attn_{entry}(q, k, v, cu_seqlens_q=starts, cu_seqlens_k=starts, "
    "seqused_q=lens, seqused_k=lens). See docs/lowprec_usage.md, 'Aligned varlen'. (Warned once per process.)")


def _warn_repack(entry: str):
    global _REPACK_WARNED
    if not _REPACK_WARNED:
        _REPACK_WARNED = True
        warnings.warn(REPACK_WARNING.format(entry=entry), UserWarning, stacklevel=3)


def _run(plan: AttnPlan, q, k, v, *, score_mod, score_mod_bwd, mask_mod, aux_tensors, cu_seqlens_q, cu_seqlens_k,
         seqused_k, prequant, seqused_q=None, doc_maps=None):
    # masks: window / softcap / score_mod / mask_mod as ONE explicit value, forward and backward
    mask_kw = dict(window_size=plan.fwd.window or (None, None), softcap=plan.fwd.softcap, score_mod=score_mod,
                   score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, aux_tensors=aux_tensors)
    if plan.fwd.varlen and not plan.fwd.aligned_varlen and doc_maps is None:
        _warn_repack(plan.entry)
    if prequant is None and score_mod is None and score_mod_bwd is None and mask_mod is None and aux_tensors is None:
        # the registered op lowbit_attn::fp4 / ::fp8 (lowprec._op): SAC can save it, torch.compile traces it
        def apply_fn(qp, kp, vp, m):
            return _op.op_apply(plan, qp, kp, vp, m)
    else:   # user callables / pre-quantised operands: the autograd.Function (not expressible as op arguments)
        def apply_fn(qp, kp, vp, m):
            return _LowPrecAttnFunc.apply(qp, kp, vp, plan, prequant, m)
    if doc_maps is not None:   # packed q, k, v; the plan is the maps' aligned buffer's
        if prequant is not None and "fa3" in prequant:   # (cov2: caller codes in the packed rows)
            return _routes.run_doc_maps(plan, q, k, v, apply_fn, doc_maps=doc_maps, mask_kw=mask_kw, prequant_fa3=True)
        return _routes.run_doc_maps(plan, q, k, v, apply_fn, doc_maps=doc_maps, mask_kw=mask_kw)
    if plan.fwd.varlen or plan.fwd.seqused_k or plan.fwd.ragged_documents:
        return _routes.run_documents(plan, q, k, v, apply_fn, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                     seqused_k=seqused_k, mask_kw=mask_kw, seqused_q=seqused_q)
    return apply_fn(q, k, v, _routes.masks_of(plan, mask_kw))


def _doc_maps_in(entry, doc_maps, q, k, v, given: dict):
    """doc_maps (lp_docmaps.DocMaps): the plan is resolved for the maps' aligned buffer (zero-memory stand-ins of its
    shape) with the aligned varlen arguments of that buffer; the call then runs the aligned path with the maps'
    precomputed tables (no per-call index building, no host sync)."""
    bad = [n for n, x in given.items() if x is not None]
    if bad:
        raise ValueError(f"flash_attn_{entry}(doc_maps=...): the maps carry the documents; do not pass "
                         f"{', '.join(bad)}")
    doc_maps.check_call(q, k, v, entry)
    return (doc_maps.stand_in(q, "q"), doc_maps.stand_in(k, "k"), doc_maps.stand_in(v, "k"),
            dict(cu_seqlens_q=doc_maps.cu_pad_q, cu_seqlens_k=doc_maps.cu_pad_k, seqused_q=doc_maps.len_q,
                 seqused_k=doc_maps.len_k, max_seqlen_q=doc_maps.max_seqlen_q, max_seqlen_k=doc_maps.max_seqlen_k))


_VARLEN_ARGS = ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "max_seqlen_q", "max_seqlen_k")


# ---------------------------------------------------------------------------------------------------------------------
# the plan cache: resolve_fp4 / resolve_fp8 are pure functions of the arguments and of tensor METADATA, so a plan is
# reused for every call with the same arguments and metadata (shape, dtype, requires_grad, grad mode, device type).
# Refusals are not cached (they raise every time). Bounded LRU; the key holds the callables and options by value.
# ---------------------------------------------------------------------------------------------------------------------
_PLAN_CACHE_SIZE = 1024
_PLAN_CACHE: "collections.OrderedDict" = collections.OrderedDict()
_PLAN_LOCK = threading.Lock()


def _meta(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return (tuple(t.shape), t.dtype, bool(t.requires_grad))
    return ("obj", type(t).__name__)


def _cached_plan(resolve, q, k, v, kw: dict,
                 tensor_kw=("cu_seqlens_q", "cu_seqlens_k", "seqused_k", "seqused_q")) -> AttnPlan:
    """resolve(q, k, v, **kw) through the plan cache. Any unhashable argument bypasses the cache."""
    try:
        key = (resolve.__name__, torch.is_grad_enabled(), _meta(q), _meta(k), _meta(v),
               tuple(sorted((n, _meta(x) if n in tensor_kw or n in ("q_quantized", "k_quantized") else x)
                            for n, x in kw.items())))
        hash(key)
    except TypeError:
        return resolve(q, k, v, **kw)
    with _PLAN_LOCK:
        plan = _PLAN_CACHE.get(key)
        if plan is not None:
            _PLAN_CACHE.move_to_end(key)
            return plan
    plan = resolve(q, k, v, **kw)
    with _PLAN_LOCK:
        _PLAN_CACHE[key] = plan
        if len(_PLAN_CACHE) > _PLAN_CACHE_SIZE:
            _PLAN_CACHE.popitem(last=False)
    return plan


def flash_attn_fp4(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[tuple] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
    pv_precision: Literal["fp8", "fp4"] = "fp8",
    backward: Literal["e4m3", "fp4"] = "e4m3",
    causal_safety: Literal["auto", "key_local", "diagonal", "off"] = "auto",
    qk_hadamard: Optional[bool] = None,
    options: Optional[FP4AttnOptions] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    doc_maps=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """4-bit (NVFP4) attention, forward and backward, on Blackwell.

    q: (b, s, h, d); k, v: (b, s, h_kv, d); bf16 or fp16; h % h_kv == 0. Packed (total, h, d) with cu_seqlens_*.
    Returns out (same dtype as q, (b, s, h, d)); with return_lse=True, (out, lse) with lse (b, h, s) float32 in nats,
    non-differentiable.

    softmax_scale: default 1/sqrt(d). causal: bottom-right aligned. window_size: (left, right) sliding window.
    softcap: tanh soft-capping of the scores (0 = off). score_mod / mask_mod: CuTe DSL callables (upstream's
    signatures); score_mod_bwd: the score_mod's backward on dS (required with a score_mod when an input requires grad);
    aux_tensors: the mods' extra tensors.
    cu_seqlens_q / cu_seqlens_k / max_seqlen_q / max_seqlen_k: variable length, token-block operands. Two forms:
        aligned (zero-copy, FA3's convention): cu_seqlens_* (b + 1,) int32 = document START offsets, each a multiple
            of 128 (the 2-CTA backward too), total % 128 == 0; seqused_q / seqused_k (b,) int32 = each
            document's true length (<= its slot cu[i + 1] - cu[i]). The kernels run on views of q, k, v. The rows
            after a document's true length are padding and a CONTRACT: zero, or finite and no larger than the real
            rows (unchecked). O / dQ / dK / dV are written zero on the padding rows of a document's last tile; LSE
            there is unspecified. Starts / lengths are checked by a device-side assert (no host sync).
        repacked (cu_seqlens without seqused_*): any document boundaries; the documents are copied into an aligned
            buffer on the GPU and back (warned once per process).
    seqused_k without cu_seqlens: per-batch used key length (served as documents of the batch rows). seqused_q: the
    aligned varlen form only (with cu_seqlens and seqused_k). deterministic: bitwise-reproducible dQ.
    window_size, softcap, score_mod, mask_mod, varlen and seqused_k have the low-precision backward on
    pv_precision="fp8"; the 4-bit P.V routes take causal only. return_lse: (h, total_q) on the aligned varlen form
    (padding rows unspecified); not returned under repacked varlen / batched seqused_k.

    pv_precision: "fp8" (default; NVFP4 Q.K, E4M3 P-hat . E4M3 V) | "fp4" (NVFP4 P-hat . NVFP4 V, the fastest forward;
        needs causal_safety under a mask). Causality of the 8-bit P.V: exact only under options.operand_scaling=
        "token_local" (or per-tensor V with a pinned amax). The default token-block E4M3 V shares one exponent per 32
        keys, so a later key in the same 32-token block can re-round earlier keys' subnormal V codes: the documented
        in-block channel, sub-ULP to ~1e-3 in O, reaching < 32 rows (flash_attn_fp8 has the same channel).
    backward: "e4m3" (default; S recomputed from the forward's 4-bit Q/K codes, dP / dV / dK / dQ on E4M3 operands,
        1-CTA or 2-CTA) | "fp4" (the query-row all-4-bit backward; needs pv_precision="fp4", d = 128).
    causal_safety: "auto" | "key_local" | "diagonal" | "off": how a 4-bit P.V avoids reading future keys through V's
        shared scale. "auto": nothing on an 8-bit P.V or dense; diagonal-only key-local V with the K-channel fold on
        the all-4-bit route under causal; key-local otherwise under a mask.
    qk_hadamard: random-sign Hadamard rotation of Q and K along d (d = 128): S = (Q R)(K R)^T = Q K^T, the rotation fused
        into the Q / K quantisers (one read per tensor, the scale statistics -- per 16 for NVFP4, per token block / per
        32 x 32 block / per tensor for the E4M3 copies -- taken on the rotated rows), the same rotated codes read by the
        forward and the backward, dQ / dK rotated back in the backward's epilogues. Works with every operand_scaling but
        "token_local" (no rotated token-local quantiser: None falls back to off with a resolution-log line, True
        raises).
        None (default): ON wherever it runs; where it cannot (head_dim != 128, q_block_2d, forward-only features, the
        all-4-bit route) the call runs without it and the resolution log says why. True: required (those cases
        refuse). False: off. One fixed sign vector (seed 4321) for every call; forward and backward share it.
    options: FP4AttnOptions (advanced; see its docstring).
    q_descale / k_descale / v_descale: refused (FA3-style pre-quantised E4M3 inputs are a flash_attn_fp8 feature; see
        lowprec.prequant_fa3.refuse_fp4).
    doc_maps: flash_attn.cute.lp_docmaps.DocMaps of packed (total, h, d) documents (instead of cu_seqlens_* /
        seqused_* / max_seqlen_*), built once per microbatch (on the CPU by a dataloader worker, or on the GPU) and
        reused by every layer: the rows are gathered into the maps' aligned buffer and the aligned varlen path runs
        with the precomputed tables (no per-call index building, no host sync; CUDA-graph safe with static maps).
        Outputs are bitwise the repacking path's. return_lse: (h, total_q).
    """
    if q_descale is not None or k_descale is not None or v_descale is not None or \
            any(isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn for t in (q, k, v)):
        from .prequant_fa3 import refuse_fp4
        refuse_fp4()
    pq, pk, pv = q, k, v
    if doc_maps is not None:
        pq, pk, pv, vk = _doc_maps_in("fp4", doc_maps, q, k, v, dict(zip(_VARLEN_ARGS, (
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k))))
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k = (vk[n] for n in _VARLEN_ARGS)
    plan = _cached_plan(resolve_fp4, pq, pk, pv, dict(
        softmax_scale=softmax_scale, causal=causal, window_size=window_size, softcap=softcap, score_mod=score_mod,
        score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, seqused_k=seqused_k, deterministic=deterministic,
        return_lse=return_lse, pv_precision=pv_precision, backward=backward, causal_safety=causal_safety,
        qk_hadamard=qk_hadamard, options=options, seqused_q=seqused_q))
    return _run(plan, q, k, v, score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod,
                aux_tensors=aux_tensors, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, seqused_k=seqused_k,
                prequant=None, seqused_q=seqused_q, doc_maps=doc_maps)


def flash_attn_fp8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[tuple] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_lse: bool = False,
    qk_hadamard: Optional[bool] = None,
    q_quantized=None,
    k_quantized=None,
    q_scale_fused: bool = False,
    options: Optional[FP8AttnOptions] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    grad_proxies: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    doc_maps=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """8-bit (E4M3) attention, forward and backward, on Blackwell.

    Shapes, returns and the common arguments as flash_attn_fp4. E4M3 Q, K, V and dO on token blocks (32 tokens x d
    UE8M0) for every served shape: head_dim <= 128 in steps of 16 (64 / 128 natively, the others padded
    to 64 / 128 by the quantisers), s_q != s_k, and any length -- s % 128 != 0 runs the batch rows as documents
    (return_lse works there). The fork forward; the E4M3 backward, 2-CTA wherever eligible.

    q_quantized / k_quantized: prequant.QuantizedOperand (or (codes, descale)), E4M3 per-tensor, supplied together;
        they skip the forward's quantiser and force per-tensor operands. The scaling is the caller's: a per-tensor scale
        is a global causal channel (a future token's max moves every earlier token's rounding), the caller's
        responsibility on this path. q_scale_fused: the supplied Q's descale
        already carries log2(e) * softmax_scale (requires q_quantized).
    qk_hadamard: as flash_attn_fp4 (default None = ON where it runs); here it also falls back for pre-quantised
        Q / K (and for s % 128 != 0 on explicit per-tensor operands).
    options: FP8AttnOptions (advanced; see its docstring).

    Pre-quantised inputs, FA3's convention (lowprec.prequant_fa3; docs/lowprec_usage.md "Pre-quantised E4M3 inputs"):
    q, k, v as torch.float8_e4m3fn codes with q_descale / k_descale / v_descale float32 (b, h_kv) dequantisation scales
    (x ~= code * descale[b, kv head]; b = the batch row, or the document under varlen; float64 is accepted and converted
    to float32 once; our own quantiser's scales are float32). No Q / K / V quantiser runs; the output is bf16. Batched (s % 128 == 0) or
    the aligned varlen form; head_dim 64 / 128; causal or dense; MHA / GQA. grad_proxies: optional bf16 (q, k, v) whose
    .grad receives dQ / dK / dV (a straight-through gradient: PyTorch casts a gradient to its input's dtype, so E4M3
    leaves cannot carry one); the backward reuses the caller's codes and quantises dO itself.
    doc_maps: as flash_attn_fp4 (not with pre-quantised inputs).
    """
    if doc_maps is not None:
        if q_quantized is not None or k_quantized is not None:
            raise ValueError("flash_attn_fp8(doc_maps=...): q_quantized / k_quantized take the aligned varlen form "
                             "(cu_seqlens + seqused_*), not doc_maps; FA3-style E4M3 q / k / v with q_descale / "
                             "k_descale / v_descale take doc_maps")
        if q.dtype == torch.float8_e4m3fn or any(x is not None for x in (q_descale, k_descale, v_descale, grad_proxies)):
            # cov2: FA3-style E4M3 codes in the caller's packed rows (lowprec.prequant_fa3.flash_attn_fp8_prequant_doc_maps)
            from .prequant_fa3 import flash_attn_fp8_prequant_doc_maps
            bad = [n for n, x in zip(_VARLEN_ARGS, (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q,
                                                   max_seqlen_k)) if x is not None]
            if bad:
                raise ValueError(f"flash_attn_fp8(doc_maps=...): the maps carry the documents; do not pass "
                                 f"{', '.join(bad)}")
            return flash_attn_fp8_prequant_doc_maps(
                _cached_plan, _run, resolve_fp8, q, k, v, q_descale, k_descale, v_descale, grad_proxies, dict(
                    softmax_scale=softmax_scale, causal=causal, window_size=window_size, softcap=softcap,
                    score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, deterministic=deterministic,
                    return_lse=return_lse, qk_hadamard=qk_hadamard, q_scale_fused=q_scale_fused, options=options),
                doc_maps, aux_tensors=aux_tensors)
        pq, pk, pv, vk = _doc_maps_in("fp8", doc_maps, q, k, v, dict(zip(_VARLEN_ARGS, (
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k))))
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k = (vk[n] for n in _VARLEN_ARGS)
    else:
        pq, pk, pv = q, k, v
    if q.dtype == torch.float8_e4m3fn or any(x is not None for x in (q_descale, k_descale, v_descale, grad_proxies)):
        from .prequant_fa3 import flash_attn_fp8_prequant
        return flash_attn_fp8_prequant(
            _cached_plan, _run, resolve_fp8, q, k, v, q_descale, k_descale, v_descale, grad_proxies, dict(
                softmax_scale=softmax_scale, causal=causal, window_size=window_size, softcap=softcap,
                score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, seqused_k=seqused_k,
                deterministic=deterministic, return_lse=return_lse, qk_hadamard=qk_hadamard, q_quantized=q_quantized,
                k_quantized=k_quantized, q_scale_fused=q_scale_fused, options=options, seqused_q=seqused_q),
            aux_tensors=aux_tensors)
    plan = _cached_plan(resolve_fp8, pq, pk, pv, dict(
        softmax_scale=softmax_scale, causal=causal, window_size=window_size, softcap=softcap, score_mod=score_mod,
        score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, seqused_k=seqused_k, deterministic=deterministic,
        return_lse=return_lse, qk_hadamard=qk_hadamard, q_quantized=q_quantized, k_quantized=k_quantized,
        q_scale_fused=q_scale_fused, options=options, seqused_q=seqused_q))
    prequant = None
    if plan.prequant or q_scale_fused:
        prequant = {"q": q_quantized, "k": k_quantized, "q_scale_fused": q_scale_fused}
    return _run(plan, q, k, v, score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod,
                aux_tensors=aux_tensors, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, seqused_k=seqused_k,
                prequant=prequant, seqused_q=seqused_q, doc_maps=doc_maps)

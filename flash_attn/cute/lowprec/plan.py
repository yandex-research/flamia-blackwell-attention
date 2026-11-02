"""Plan resolution: the public arguments -> one immutable, fully resolved plan per call.

``resolve_fp4`` / ``resolve_fp8`` are pure functions of the call's arguments and of tensor METADATA (shape, dtype,
requires_grad): they never read a tensor's values, the environment, or any global configuration, and they launch
nothing. They

  1. validate every combination (numbered rules; the exception carries ``.rule``),
  2. resolve every "auto" against the call's shapes (the resolution log records each decision and its reason),
  3. apply the scaling-consistency rule (forward and backward share the scaling of every operand they share),
  4. refuse, with ``NotImplementedError`` naming the missing piece, every combination whose implementation is still
     being built (the ``PENDING`` table -- one row per missing piece, deleted when its branch lands),

and return a frozen ``AttnPlan``. The autograd context carries the plan to the backward; nothing is re-resolved there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .options import FP4AttnOptions, FP8AttnOptions, KernelTuning

__all__ = ["AttnPlan", "FwdSpec", "BwdSpec", "CallShape", "PENDING", "resolve_fp4", "resolve_fp8", "pending_error"]


# ================================================================================================================
# Features whose implementation is still being built. ONE table: a row is deleted when its branch lands, and the
# combination it guards then resolves like any other. Each row names the missing piece and where it is being built.
# ================================================================================================================
PENDING = {
    # (empty: every earlier row -- the masked / varlen / seqused_k backward, option A's shared V and its causal route,
    # the E4M3-route Q/K Hadamard -- is built; the MXFP4 Q/K and P.V formats were dropped from the API instead)
}


# Option A -- the 4-bit P-hat.V forward with the E4M3-operand (dk8) backward -- stays an option, off by default, for
# INFERENCE only. Its training backward (the backward rebuilding the forward's 4-bit P-hat bit for bit) is not part of
# this library (docs/lowprec.md, "Option A").
OPTION_A_INFERENCE_ONLY = (
    "pv_precision='fp4' with backward='e4m3' (option A: the 4-bit P-hat.V forward) is an inference-only option: no input "
    "may require grad. Its training route needs either an extra consistency pass over the scores (prohibited: one main "
    "kernel per pass) or the backward's rebuild of the forward's 4-bit P-hat, which measured slower than the default step "
    "(1.60-1.64 vs 1.00 ms at b4 s4096 h16 d128 causal) and trained ~0.008 worse (a byte-level GPT, 6000 steps, two "
    "seeds). Train with backward='e4m3' and pv_precision='fp8' (the default) or with the all-4-bit route backward='fp4', "
    "or call this one with requires_grad=False.")


def pending_error(key: str, rule=None, **fmt) -> NotImplementedError:
    row = PENDING[key]
    feat = row["feature"].format(**fmt)
    missing = row["missing"].format(**fmt)
    e = NotImplementedError(f"{feat}: {missing} is not built yet (pending on branch {row['branch']}).")
    e.rule = rule
    e.pending = key
    return e


def _exp_mix(o, route, sm_exp, log) -> str:
    """The per-group-position value functions of the exact 4-bit P-hat. auto: option A's
    inference forward takes EMMEMMEM (MUFU ex2 on group positions 0, 3, 6: forward -8..-13.5 % at A1..A4, O cosine equal
    or better, 1.6 % -> 1.0 % of codes off the exact-exp2 code); the all-4-bit training route stays pure Mitchell (the
    mix moves its step by -0.4..+1.5 %, its backward rebuild costs what the forward saves)."""
    if route not in ("fp4.fp4_pv", "fp4.all_fp4"):
        return "n/a"
    if o.exp_mix == "none":
        return "none"
    if o.exp_mix != "auto":
        return log("exp_mix", o.exp_mix.upper(), "explicit")
    if route == "fp4.fp4_pv" and sm_exp == "mitchell" and o.p_consistency == "reproduce_forward_p":
        return log("exp_mix", "EMMEMMEM", "auto: option A's inference forward, MUFU on group positions 0, 3, 6")
    return log("exp_mix", "none", "auto: pure Mitchell (the all-4-bit step gains nothing from the mix)")


def _fail(rule, exc_cls, msg):
    e = exc_cls(msg)
    e.rule = rule
    raise e


# ================================================================================================================
# The plan
# ================================================================================================================

@dataclass(frozen=True)
class CallShape:
    b: int
    s_q: int
    s_k: int
    h: int
    h_kv: int
    d: int
    d_v: int
    dtype: str
    varlen: bool
    seqused_k: bool
    requires_grad: bool
    tot_q: int                     # packed totals under varlen (= s_q / s_k otherwise)
    tot_k: int
    # the aligned varlen fast path (cu_seqlens with seqused_q AND seqused_k, FA3's convention): the documents sit in the
    # caller's tensors at aligned starts and run in place (no copy); seqused_k is then a document length (seqused_k
    # above stays False: it names the batched seqused_k)
    aligned_varlen: bool = False

    @property
    def mha(self) -> bool:
        return self.h == self.h_kv


@dataclass(frozen=True)
class FwdSpec:
    """What the forward runs. kernel: fork_fp4 | fork_fp4_all | fork_fp8 | upstream_fp8."""
    kernel: str
    qk_format: str                 # nvfp4 | mxfp4 | e4m3
    pv_operand: str                # e4m3 | nvfp4 | mxfp4
    operand_scaling: str           # of the E4M3 V (4-bit, pv fp8) / Q, K, V (8-bit); "n/a" on the 4-bit P.V routes
    softmax_exp: str               # exp2 | mitchell
    causal_safety: str             # none | key_local | diagonal | off
    p_frame: str                   # two_level | per_tile | single_level | n/a
    q_block_2d: bool
    qk_hadamard: bool
    v_hadamard: bool
    causal: bool
    window: Optional[Tuple[Optional[int], Optional[int]]]
    softcap: float
    score_mod: bool
    mask_mod: bool
    varlen: bool
    seqused_k: bool
    return_lse: bool
    per_tensor_scale: str = "exact"  # exact | power_of_two: per-tensor operand scales
    two_cta: bool = False          # the fork forward on CTA pairs (O / LSE bitwise the 1-CTA forward's)
    ragged_documents: bool = False # s % 128 != 0 without varlen: the batch rows run as documents
    aligned_varlen: bool = False   # cu_seqlens + seqused_q + seqused_k: the documents in place (no repacking)


@dataclass(frozen=True)
class BwdSpec:
    """What the backward runs. kernel: 1cta | 2cta | query_row | none (no gradient required)."""
    kernel: str
    head_dim_path: str             # single_pass | zero_pad_64 | zero_pad_128 | three_pass | n/a
    operand_scaling: str
    ds_scaling: str
    ds_dq_rowlocal: str            # off | diagonal | all: row-local dQ-side exponents of the block_32x32 dS
    p_scaling: str
    grad_format: str
    deterministic: bool
    sr_ds: bool
    sr_dk_operand: bool
    sr_dv_operand: bool
    fp4_backward_dkdv: str
    dq_operand: str
    all_fp4_stabiliser: str
    fp4_ds_multiplier: str
    k_channel: str
    per_tensor_ds_source: str      # "recompute" (the 4-bit three-pass per-tensor dS) | "n/a" (8-bit: removed, rule 33)
    save_bf16_inputs: bool
    softmax_exp: str               # the backward's P-hat recompute: the forward's value function where they must agree
    p_consistency: str = "n/a"     # option A: reproduce_forward_p | none (the backward rebuilds the forward's P-hat); n/a elsewhere
    do_rounding: str = "n/a"       # option A: rtn | sr, the 4-bit dO of the dP GEMM; n/a elsewhere
    exp_mix: str = "n/a"           # option A (reproduce_forward_p, Mitchell): the per-group-position value functions; n/a else


@dataclass(frozen=True)
class AttnPlan:
    entry: str                     # fp4 | fp8
    route: str                     # fp4.e4m3_pv | fp4.fp4_pv | fp4.all_fp4 | fp8
    softmax_scale: float
    shape: CallShape
    fwd: FwdSpec
    bwd: BwdSpec
    tuning: KernelTuning
    prequant: bool
    q_scale_fused: bool
    consistency: Tuple[Tuple[str, str, str], ...]     # (operand, forward scaling, backward scaling)
    pending: Tuple[str, ...]                          # PENDING rows this plan runs under (exemptions), if any
    resolution_log: Tuple[Tuple[str, str, str], ...]  # (field, resolved value, reason)
    options: object = field(compare=True)             # the options object as given (FP4AttnOptions / FP8AttnOptions)
    # every spec the route runs with (lowprec.specs.PlanSpecs), built once from the fields above (build_specs)
    specs: object = field(default=None, compare=False, repr=False)

    def log(self) -> str:
        return "\n".join(f"{f} = {v}: {r}" for f, v, r in self.resolution_log)


# ================================================================================================================
# the specs of a plan (lowprec.specs): built once per plan from its resolved values, KernelTuning copied in last
# ================================================================================================================

# the plan's operand scaling -> the spec value of the E4M3 copies (token_local runs the token-block kernels with the
# token-local operand arguments)
_OP_SPEC = {"token_block": "blk32x128", "block_32x32": "blk32", "per_tensor": "per_tensor", "token_local": "blk32x128"}
# the 8-bit plan's dS scaling -> the spec value (no "per_tensor": the 8-bit per-tensor dS was removed, rule 33)
_FP8_DS_SPEC = {"block_32x32": "blk32", "row_local": "row", "row_1x32": "mx32", "block_32x64": "blk32x64",
                "tile_128": "tile128"}


def build_specs(route: str, f: "FwdSpec", b: "BwdSpec", tuning: KernelTuning, shape: "CallShape",
                fwd8_row_p: bool = True, fwd4_row_p: bool = False):
    """Every spec the plan's route runs with (lowprec.specs.PlanSpecs), fully resolved from the plan's values: the
    public arguments and options set their fields, then every non-None KernelTuning field is copied into the spec(s) it
    belongs to (specs.apply_kernel_tuning, KT_TO_SPEC). Only legal spec values are produced (a spec's own validation
    raises otherwise). Fields no public argument reaches keep their spec default ("the measured rule")."""
    from .specs import (Bwd1Spec, Bwd2Spec, BwdQSpec, FwdKernelSpec, HostSpec, PlanSpecs, QuantSpec,
                        apply_kernel_tuning)
    fp8 = route == "fp8"
    e4pv = route == "fp4.e4m3_pv"
    opta = route == "fp4.fp4_pv"
    all4 = route == "fp4.all_fp4"
    e4m3_bwd = fp8 or e4pv                       # the routes with the E4M3-operand (1-CTA / 2-CTA) backward
    fp4_p = e4pv or opta                         # the 4-bit routes whose dk8 backward reads the P^ / dS options
    exp2_4bit = (opta or all4) and f.softmax_exp == "exp2"   # the 4-bit P^ formed with exp2 instead of Direct-P
    op = _OP_SPEC.get(f.operand_scaling)
    ds8 = _FP8_DS_SPEC[b.ds_scaling] if fp8 and b.ds_scaling != "block_32x32" else "blk32"
    # the 4-bit routes' dS: per_tensor only where the plan resolves it itself (head_dim > 128, the three-pass kernels,
    # whose out_pass != "all" never builds the blk32 rung); the specs state it rather than leaving it to the driver
    ds4 = "per_tensor" if fp4_p and b.ds_scaling == "per_tensor" else "blk32"
    p8 = "mx32" if fp8 and b.p_scaling == "block_1x32" else "row"
    p4 = "mx32" if fp4_p and b.p_scaling == "block_1x32" else "row"
    masked = f.causal or f.window is not None or f.mask_mod

    # ---- the forward kernel: the all-FP4 forward on the 4-bit P.V routes; option A's inference forward with the
    # reproduce-forward-p settings (the codes row sum, the E4M3-decoded group SFs, the plan's value function)
    fwd = FwdKernelSpec(
        fp4_pv_two_level=not (opta and f.p_frame == "single_level"),
        fp4_pv_tile_frame=opta and f.p_frame == "per_tile",
        fp4_pv_rescale_threshold=0.0 if (opta and f.p_frame == "per_tile") else None,
        direct_p=not exp2_4bit,
        fp4all=opta or all4,
        fp4all_exp2=exp2_4bit,
        fp4all_rowsum="codes" if opta else "mitchell",
        fp4all_sf_decode=opta)

    # ---- the host's quantisation choices
    if all4:
        if f.causal_safety == "key_local" and not masked:
            csv = "always"
        elif f.causal_safety in ("off", "none") and masked:
            csv = "never"
        else:
            csv = "auto"
    else:
        csv = "never" if (opta and f.causal_safety == "off") else "auto"
    mult_default = "row_local" if f.causal else "global_row_boost"
    mult_set = all4 and b.fp4_ds_multiplier != mult_default
    quant = QuantSpec(
        q_block_2d=e4pv and f.q_block_2d,
        qk_hadamard=bool(f.qk_hadamard) and route in ("fp8", "fp4.e4m3_pv", "fp4.fp4_pv"),
        per_tensor_pow2=f.per_tensor_scale == "power_of_two",
        grad_format="e5m2" if fp8 and b.grad_format == "e5m2" else "e4m3",
        # option A's forward resolves its E4M3 copies explicitly to token blocks (its dk8 blk32x128 contract)
        operand_scaling_fp4=op if e4pv else ("blk32x128" if opta else None),
        operand_scaling_fp8=op if fp8 else None,
        ds_scaling_fp8=ds8,
        ds_scaling_fp4=ds4,
        p_scaling_fp4=p4,
        causal_safe_v=csv,
        vtok_diag=all4 and f.causal_safety == "diagonal",
        vtok_diag_k=b.k_channel if (all4 and f.causal_safety == "diagonal" and b.k_channel not in ("fold", "n/a"))
        else "fold",
        vtok_hadqk=all4 and bool(f.qk_hadamard),
        vtok_hadv=all4 and bool(f.v_hadamard),
        fp4all_stab=b.all_fp4_stabiliser if all4 else "sr",
        fp4q_ds_scale={"row_local": "causal", "global": "global", "global_row_boost": "global"}[b.fp4_ds_multiplier]
        if mult_set else "auto",
        fp4q_qrow=(b.fp4_ds_multiplier == "global_row_boost") if mult_set else None)

    # ---- routing / scheduling of the drivers and the glue
    fk = "auto"   # "auto" = the fork forward; stated only where the plan's kernel is not the shape's default one
    if fp8:
        fork_default = shape.tot_q % 128 == 0 and shape.tot_k % 128 == 0 and shape.d in (64, 128)
        if (f.kernel == "fork_fp8") != fork_default:
            fk = "fork" if f.kernel == "fork_fp8" else "upstream"
    host = HostSpec(
        fp8_forward_kernel=fk,
        fwd8_row_p=bool(fwd8_row_p),
        fwd4_row_p=bool(fwd4_row_p),
        fwd_two_cta=bool(f.two_cta),
        bwd_op_mode="fp8" if fp8 else "fp4",
        bwd_two_cta={"1cta": "off", "2cta": "on"}.get(b.kernel, "auto"),
        # GQA on the 2-CTA backward (KernelTuning.bwd_2cta_gqa, two_cta='on' or the long-sequence default)
        bwd_2cta_gqa=b.kernel == "2cta" and shape.h != shape.h_kv,
        save_lean=e4pv and not b.save_bf16_inputs and f.operand_scaling == "per_tensor")

    bwd1 = bwd2 = bwdq = None
    if e4m3_bwd:
        bwd1 = Bwd1Spec(fp8=fp8, ds_sr=fp4_p and bool(b.sr_ds), p_scaling_fp4=p4, op_scaling=op, ds_scaling_fp8=ds8,
                        ds_scaling_fp4=ds4, p_scaling_fp8=p8)
        if op in ("per_tensor", "blk32x128"):   # blk32 (32 x 32 operands) is 1-CTA only: no 2-CTA spec
            bwd2 = Bwd2Spec(ds_scaling_fp8=ds8, ds_scaling_fp4=ds4, p_scaling_fp8=p8, op_scaling=op)
    if all4:
        bwdq = BwdQSpec(exp2=exp2_4bit, dq8=b.dq_operand == "e4m3", dk_sr=bool(b.sr_dk_operand),
                        dv_sr=bool(b.sr_dv_operand), dkv=b.fp4_backward_dkdv)
    # KernelTuning last: a tuning field is the more specific statement
    kt = lambda s: None if s is None else apply_kernel_tuning(s, tuning)  # noqa: E731
    return PlanSpecs(fwd=kt(fwd), quant=kt(quant), host=kt(host), bwd1=kt(bwd1), bwd2=kt(bwd2), bwdq=kt(bwdq))


# ================================================================================================================
# helpers
# ================================================================================================================

_GRAD_FORMATS = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def _requires_grad(*ts) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in ts)


def _shape(entry, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k,
           seqused_q=None, prequant_fa3: bool = False) -> CallShape:
    for nm, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"flash_attn_{entry}: {nm} must be a torch.Tensor; got {type(t).__name__}.")
    # FA3-style pre-quantised inputs (lowprec.prequant_fa3): E4M3 codes, a bf16 output
    e4 = prequant_fa3 and all(t.dtype == torch.float8_e4m3fn for t in (q, k, v))
    if not e4 and (q.dtype not in _GRAD_FORMATS or k.dtype != q.dtype or v.dtype != q.dtype):
        raise ValueError(f"flash_attn_{entry}: q, k, v must all be bfloat16 or all float16; got "
                         f"{q.dtype}, {k.dtype}, {v.dtype}.")
    varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    aligned = False
    if seqused_q is not None and not varlen:
        raise ValueError(f"flash_attn_{entry}: seqused_q is the true document length of the aligned varlen fast path: "
                         "it needs cu_seqlens_q / cu_seqlens_k (and seqused_k). A batched (b, s, h, d) call takes "
                         "seqused_k only.")
    if varlen:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(f"flash_attn_{entry}: cu_seqlens_q and cu_seqlens_k must be given together.")
        if (seqused_q is None) != (seqused_k is None):
            given, missing = ("seqused_q", "seqused_k") if seqused_q is not None else ("seqused_k", "seqused_q")
            raise ValueError(
                f"flash_attn_{entry}: with cu_seqlens_q / cu_seqlens_k, pass seqused_q AND seqused_k (the aligned "
                f"zero-copy path: cu_seqlens = document starts, multiples of 128; seqused = true lengths) or neither "
                f"(the repacking path: cu_seqlens = packed document boundaries); got {given} without {missing}.")
        aligned = seqused_q is not None
        if aligned:
            nb = cu_seqlens_q.shape[0] - 1 if cu_seqlens_q.dim() == 1 else -1
            for nm, t, n in (("cu_seqlens_q", cu_seqlens_q, nb + 1), ("cu_seqlens_k", cu_seqlens_k, nb + 1),
                             ("seqused_q", seqused_q, nb), ("seqused_k", seqused_k, nb)):
                if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.dim() != 1 or t.shape[0] != n \
                        or n < 1:
                    raise ValueError(f"flash_attn_{entry}: aligned varlen: {nm} must be a 1-D int32 tensor of "
                                     f"{'b + 1' if nm.startswith('cu') else 'b'} entries (b = len(cu_seqlens_q) - 1 "
                                     f">= 1); got {getattr(t, 'dtype', type(t).__name__)} "
                                     f"{tuple(getattr(t, 'shape', ()))}.")
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(f"flash_attn_{entry}: varlen inputs are packed (total, h, d); got {tuple(q.shape)}.")
        b = cu_seqlens_q.shape[0] - 1
        s_q = int(max_seqlen_q) if max_seqlen_q is not None else q.shape[0]
        s_k = int(max_seqlen_k) if max_seqlen_k is not None else k.shape[0]
        _, h, d = q.shape
        _, h_kv, d_k = k.shape
        d_v = v.shape[-1]
        tot_q, tot_k = q.shape[0], k.shape[0]
    else:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"flash_attn_{entry}: q, k, v must be (b, s, h, d); got {tuple(q.shape)}, "
                             f"{tuple(k.shape)}, {tuple(v.shape)}.")
        b, s_q, h, d = q.shape
        _, s_k, h_kv, d_k = k.shape
        d_v = v.shape[-1]
        tot_q, tot_k = s_q, s_k
    if d_k != d or h % h_kv != 0 or v.shape[-2] != h_kv:
        raise ValueError(f"flash_attn_{entry}: shapes q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} are "
                         "inconsistent (need equal head_dim for q / k and h % h_kv == 0).")
    if d_v != d:
        raise ValueError(f"flash_attn_{entry}: head_dim_v != head_dim ({d_v} vs {d}) is not supported.")
    shp = CallShape(b=b, s_q=s_q, s_k=s_k, h=h, h_kv=h_kv, d=d, d_v=d_v, dtype="bf16" if e4 else _GRAD_FORMATS[q.dtype],
                    varlen=varlen, seqused_k=seqused_k is not None and not aligned,
                    requires_grad=_requires_grad(q, k, v), tot_q=tot_q, tot_k=tot_k, aligned_varlen=aligned)
    return shp


def _check_aligned(entry: str, shp: CallShape, two: str, tuning: KernelTuning, had_on: bool, log: "_Log",
                   causal: bool = False):
    """The aligned varlen fast path's host-side facts (shapes only; the values are checked on the GPU by
    lp_varlen.LPDocs.aligned_from's device-side assert). A = 128 (the 2-CTA backward's key pair takes documents packed at
    128 too): the totals must be multiples of A, so the last slot ends aligned too."""
    if not shp.aligned_varlen:
        return
    a = 128
    if shp.tot_q % a or shp.tot_k % a:
        _fail(37, ValueError, f"flash_attn_{entry}: aligned varlen needs total_q and total_k to be multiples of {a} "
                              f"({'the 2-CTA backward pairs 256 keys' if a == 256 else 'the 128-row scale-factor atom'}): "
                              f"got total_q = {shp.tot_q}, total_k = {shp.tot_k}. Pad the packed tensors (and the last "
                              f"document's slot) up to a multiple of {a} (padding rows: zero, or finite and no larger than the real rows).")
    log("documents", "aligned", f"cu_seqlens + seqused_q + seqused_k: the documents run in place in the caller's "
                                f"tensors (no copy), starts asserted % {a} == 0 on the GPU; padding rows: the caller's "
                                "contract (zero, or finite and no larger than the real rows)")
    if tuning.varlen_fwd_sched in (None, "auto"):
        log("varlen_fwd_sched", "clc (jit claim)" if causal else "static or clc (by length skew)",
            "auto: Cluster Launch Control with the just-in-time claim on causal calls and on dense calls whose document "
            "lengths are skewed (longest / mean length >= lp_docmaps.SKEW_CLC_MIN, host-known), the static deal on "
            "near-equal dense ones (where the persistent work list runs: no mask_mod / window, 1-CTA forward; "
            "fp4_bwd_autograd._aligned_fwd_sched)")


def _window(window_size):
    if window_size is None:
        return None
    if not (isinstance(window_size, (tuple, list)) and len(window_size) == 2):
        raise ValueError(f"window_size must be a (left, right) pair; got {window_size!r}.")
    left, right = (None if (w is None or int(w) < 0) else int(w) for w in window_size)
    return None if (left is None and right is None) else (left, right)


class _Log:
    def __init__(self):
        self.rows = []

    def __call__(self, name, value, reason):
        self.rows.append((name, str(value), reason))
        return value


class _HadamardRequest:
    """qk_hadamard's resolution (ON by default: 0.1-0.6 % in the quantiser, training within seed noise, and it helps
    one-step accuracy with K outlier channels).
    None (the default) = on wherever the rotation runs; a condition that would refuse an explicit True turns it off with
    the reason in the resolution log instead (a logged fallback, never a refusal). True = required: those conditions
    refuse as before. False = off.
    Seed policy: one fixed seed (qk_hadamard.SEED = 4321); the 128 signs are compile-time constants of every kernel that
    reads or undoes the rotation, so the forward and the backward of a call -- and every call -- use the same R; the
    forward saves the rotated codes and the backward reads only those (SCALING-CONSISTENCY RULE)."""

    def __init__(self, requested, entry):
        if requested is not None and not isinstance(requested, bool):
            raise ValueError(f"flash_attn_{entry}: qk_hadamard must be None (default: on where it runs), True or False; "
                             f"got {requested!r}.")
        self.auto = requested is None
        self.on = requested is None or requested
        self.why_off = []

    def unavailable(self, reason, rule=None, exc_cls=NotImplementedError, msg=None, pending=None):
        """The rotation cannot run on this call: off with the reason under the default, refused when required."""
        if not self.on:
            return
        if self.auto:
            self.on = False
            self.why_off.append(reason)
            return
        if pending is not None:
            raise pending_error(pending)
        _fail(rule, exc_cls, msg or reason)

    def log(self, log, on_reason):
        if self.auto and self.on:
            log("qk_hadamard", "on", "default (on where it runs): " + on_reason)
        elif self.auto:
            log("qk_hadamard", "off", "default on; falls back here: " + "; ".join(self.why_off))
        elif self.on:
            log("qk_hadamard", "on", "explicit: " + on_reason)
        else:
            log("qk_hadamard", "off", "explicit")
        return self.on


def _mods(feats):
    """The masking / score-modification features of _features, without the documents (varlen / seqused_k)."""
    return [f for f in feats if f not in ("cu_seqlens_q / cu_seqlens_k", "seqused_k")]


def _features(window, softcap, score_mod, mask_mod, shp: CallShape):
    """The masking / score-modification / varlen features of the call, by argument name."""
    out = []
    if shp.varlen:
        out.append("cu_seqlens_q / cu_seqlens_k")
    if window is not None:
        out.append("window_size")
    if softcap:
        out.append("softcap")
    if mask_mod is not None:
        out.append("mask_mod")
    if score_mod is not None:
        out.append("score_mod")
    if shp.seqused_k:
        out.append("seqused_k")
    return out


# (route, causal) of the 4-bit P.V routes that take varlen / seqused_k documents (vall4)
_FOUR_BIT_DOCS = {("fp4.all_fp4", True), ("fp4.all_fp4", False), ("fp4.fp4_pv", True), ("fp4.fp4_pv", False)}


def _check_features(entry: str, route: str, feats, causal, window, softcap, score_mod, score_mod_bwd, mask_mod,
                    shp: CallShape, return_lse: bool, grad: bool, prequant: bool, ds: Optional[str], log: _Log):
    """masks / varlen: window / softcap / score_mod / mask_mod / varlen / seqused_k run forward AND backward on the E4M3
    P.V routes (8-bit, pv_precision='fp8'); the 4-bit P.V routes' V scale is key-local for causal only (rule 28).
    The argument rules are lp_masks.LPMasks.build's, raised here before anything runs."""
    if score_mod_bwd is not None and score_mod is None:
        _fail(28, ValueError, "score_mod_bwd is the backward of a score_mod; no score_mod was given.")
    if not feats:
        return
    if route not in ("fp8", "fp4.e4m3_pv"):
        # the 4-bit P.V routes take varlen / seqused_k documents (vall4: every statistic per document, the kernels in
        # each document's coordinates) where built (_FOUR_BIT_DOCS); no window / mask / score modification
        other = [f for f in feats if f not in ("cu_seqlens_q / cu_seqlens_k", "seqused_k")]
        if causal and shp.seqused_k and not shp.varlen and not other:
            _fail(28, NotImplementedError,
                  "causal batched seqused_k on the 4-bit P.V routes: bottom-right causal over seqused_k keys leaves the "
                  "first s_q - seqused_k queries of a row without a key, which the 4-bit forward does not serve (as "
                  "s_q > s_k batched). Use pv_precision='fp8', or varlen documents with seqused_q <= seqused_k.")
        if other or (route, bool(causal)) not in _FOUR_BIT_DOCS:
            _fail(28, NotImplementedError,
                  f"The 4-bit P.V routes take no {' / '.join(other or feats)}"
                  + (": their V scale is key-local for causal attention only (no window / mask aware V scale is built)"
                     if other else f" on this route ({route}, {'causal' if causal else 'dense'}): not built")
                  + ". Use pv_precision='fp8'.")
    if softcap and score_mod is not None:
        _fail(28, ValueError, "softcap is a score_mod: pass one or the other.")
    if mask_mod is not None and (causal or window is not None):
        _fail(28, ValueError, "mask_mod with causal / window_size: express the causal / window condition inside the "
                              "mask_mod (the low-precision kernels, like upstream's backward, apply one or the other).")
    if score_mod is not None and score_mod_bwd is None and grad:
        _fail(28, ValueError, "score_mod needs score_mod_bwd for the backward (it is applied to dS), as upstream; pass "
                              "score_mod_bwd, or call with no input requiring grad.")
    if score_mod is not None and ds == "per_tensor":
        _fail(28, NotImplementedError, "score_mod with a per-tensor dS scale: the dS bound does not cover "
                                       "score_mod_bwd; use the block-scaled dS (ds_scaling='auto').")
    if prequant:
        _fail(32, NotImplementedError, "pre-quantised Q / K with window / softcap / score_mod / mask_mod / varlen / "
                                       "seqused_k: not wired (the masked entries quantise Q / K themselves).")
    docs = shp.varlen or shp.seqused_k
    # return_lse on repacked varlen / batched seqused_k: the repacked batch's LSE rows are gathered back to the caller's
    # rows on the GPU ((h, total_q) varlen, (b, h, s) seqused_k), as for ragged batch rows
    log("features", ", ".join(feats), "masks / varlen: forward and backward" + (
        " (aligned documents in place)" if shp.aligned_varlen else
        " (documents repacked at aligned starts)" if docs else ""))


def _tuning_pending(tuning: KernelTuning, route: str, op_res: str, qk_hadamard: bool, masked: bool):
    """Tuning fields this call would not read (no silent no-ops): raise before anything runs."""
    # token_block_forward (fp4_bwd_autograd.tb_fold_fwd): the token-block forward's form, token-block operands only,
    # no masks
    if tuning.token_block_forward is not None and (route not in ("fp8", "fp4.e4m3_pv") or op_res != "token_block"
                                                   or (masked and tuning.token_block_forward == "fold")):
        _fail("E12", ValueError, "KernelTuning.token_block_forward selects the forward of token-block E4M3 operands "
                                 "(operand_scaling 'token_block' on the E4M3 P.V routes); 'fold' takes no window / "
                                 "softcap / score_mod / mask_mod / varlen.")
    # qk_hadamard_e4m3 (qk_hadamard.e4m3_mode): the E4M3 copies of Q / K on the rotated 4-bit E4M3-P.V route (the
    # dQ / dK GEMMs' operands; per-tensor, token-block and 32 x 32 scaling)
    # qk_hadamard_rotation_method (qk_hadamard.rotation_method): "tcgen05" only where every rotated Q / K
    # quantiser of the route has the tcgen05 form -- the token-block / 32 x 32 operands (quant.fp8 token blocks,
    # quant.fp4 rows); the per-tensor quantiser and the all-4-bit route's causal-diagonal quantiser have none
    if tuning.qk_hadamard_rotation_method == "tcgen05" and not (qk_hadamard and route != "fp4.all_fp4"
                                                                 and op_res in ("token_block", "block_32x32")):
        _fail("E12", ValueError, "KernelTuning.qk_hadamard_rotation_method='tcgen05' needs qk_hadamard on and the "
                                 "token_block / block_32x32 Q / K operands of the 8-bit, 4-bit E4M3-P.V or option A "
                                 "routes (the per-tensor and all-4-bit quantisers have no tcgen05 form); use "
                                 "qk_hadamard_rotation_method='tensor_core' (the default) here.")
    if tuning.qk_hadamard_e4m3 is not None and not (qk_hadamard and route == "fp4.e4m3_pv"
                                                    and op_res in ("per_tensor", "token_block", "block_32x32")):
        _fail("E12", ValueError, "KernelTuning.qk_hadamard_e4m3 configures the E4M3 copies of Q / K on the rotated "
                                 "4-bit route (qk_hadamard=True, pv_precision='fp8').")


def _consistency(route: str, fwd_pairs, requires_grad: bool, log: _Log):
    """THE scaling-consistency check (a design invariant): forward and backward use the same scaling of every
    operand they share (format, granularity and scale values) -- the backward consumes the forward's saved codes and
    never re-quantises a shared operand in another scaling. Mixed designs are refused here, at plan resolution (no
    exemption: option A shares its NVFP4 V too)."""
    pend = []
    if not requires_grad:
        return tuple(fwd_pairs), tuple(pend)  # no backward: nothing is shared
    for op, f, b in fwd_pairs:
        if f == b:
            continue
        _fail("consistency", ValueError,
              f"scaling-consistency rule: the forward uses {op} as {f} but the backward would use {b}. Forward and "
              "backward must share the scaling of every operand they share (the backward consumes the forward's "
              "saved codes); mixed designs are not supported.")
    return tuple(fwd_pairs), tuple(pend)


_PER_TENSOR_OPT_IN = (
    "Per-tensor operand scales are a global causal channel (one amax over all tokens: a future token's max moves every "
    "earlier token's rounding, 0.23-1.2 in O measured), so operand_scaling='auto' does not fall "
    "back to them. Pass operand_scaling='per_tensor' to run this call on "
    "per-tensor scales, accepting that channel")


def _token_block_shape_note(shp: CallShape) -> str:
    """The resolution log's note on the shapes that token blocks now serve (they used to fall back to per-tensor
    operands)."""
    notes = []
    if shp.s_q != shp.s_k:
        notes.append(f"s_q != s_k ({shp.s_q} vs {shp.s_k}): the factors of Q / dO by query block, of K / V by key block")
    if shp.d not in (64, 128):
        notes.append(f"head_dim {shp.d} padded to {64 if shp.d < 64 else 128} by the quantisers")
    return "".join(f"; {n}" for n in notes)


def _refuse_auto_per_tensor(entry: str, why) -> None:
    """operand_scaling='auto' on a call token blocks cannot serve: refused, naming the explicit per-tensor opt-in."""
    _fail(16, ValueError, f"flash_attn_{entry}: operand_scaling='auto' resolves to token blocks, which cannot serve this "
                          f"call: {'; '.join(why)}. {_PER_TENSOR_OPT_IN}.")


def _check_token_local(op_res: str, rowq: str, backward: str, qk_hadamard: bool, window, softcap, score_mod,
                       mask_mod, shp, log: _Log, causal: bool = False) -> None:
    """operand_scaling="token_local": per-token exponents of the E4M3 copies. Its dQ / dK folds write the dQ
    side's own dS codes (ds_dq_rowlocal="all"); its forward is the fold forward (no masks / varlen) and its quantiser
    has no rotation yet. No silent no-ops: anything it cannot serve raises."""
    if op_res != "token_local":
        return
    if backward != "e4m3":
        _fail(16, ValueError, "operand_scaling='token_local' is built on the E4M3-operand backward (backward='e4m3').")
    if rowq != "all":
        _fail(16, ValueError, "operand_scaling='token_local' needs ds_dq_rowlocal='all': the per-key factors of K fold "
                              "into the dQ side's own dS codes.")
    if qk_hadamard:
        _fail(16, ValueError, "operand_scaling='token_local' with qk_hadamard: the rotated token-local quantiser is not "
                              "built.")
    if window is not None or softcap or score_mod is not None or mask_mod is not None:
        _fail(16, ValueError, "operand_scaling='token_local' runs the fold forward, which takes no window / softcap / "
                              "score_mod / mask_mod (varlen / seqused_k documents run on it).")
    if shp.d not in (64, 128):
        _fail(16, ValueError, f"operand_scaling='token_local' is built for head_dim 64 and 128; got {shp.d} (use "
                              "operand_scaling='auto': token blocks, padded).")
    if causal and shp.s_q > shp.s_k:
        # s_q != s_k: bottom-right causal with s_q > s_k leaves the first s_q - s_k queries without a key;
        # the fold forward writes NaN there (the token-block forward writes O = 0, LSE = -inf)
        _fail(16, ValueError, f"operand_scaling='token_local' with causal s_q > s_k ({shp.s_q} vs {shp.s_k}): the first "
                              "s_q - s_k queries see no key, which the fold forward does not serve (it writes NaN rows); "
                              "use operand_scaling='auto' (token blocks).")
    log("token_local", "on", "per-token E4M3 exponents: codes and scales of no token depend on another token; fold "
                             "forward, f_K / f_Q folded into the dQ / dK sides of dS, f_dO into the block-scaled dV P-hat")


def _resolve_per_tensor_scale(value: str, op_res: str, log: _Log) -> str:
    """per_tensor_scale: "power_of_two" rounds the GPU-computed per-tensor scales up to a power of two."""
    if value == "exact":
        return "exact"
    if op_res != "per_tensor":
        _fail(16, ValueError, f"per_tensor_scale={value!r} shapes per-tensor operand scales; this call's operands are "
                              f"{op_res!r}.")
    return log("per_tensor_scale", value, "explicit (amax / 448 rounded up to a power of two)")


def _resolve_ds_dq_rowlocal(entry: str, value: str, ds: str, causal: bool, window, mask_mod, log: _Log) -> str:
    """ds_dq_rowlocal: row-local dQ-side exponents of the block_32x32 dS. dS is backward-only, so the
    scaling-consistency rule (operands the two passes share) is not involved. No silent no-ops: a value the call cannot
    use raises."""
    if value == "off":
        return "off"
    if ds != "block_32x32":
        _fail("E10", ValueError, f"ds_dq_rowlocal={value!r} refines the block_32x32 dS (one exponent per 32 x 32 block); "
                                 f"this call's dS scaling is {ds!r}.")
    if value == "diagonal" and (not causal or window is not None or mask_mod is not None):
        _fail("E10", ValueError, "ds_dq_rowlocal='diagonal' acts on the 32 x 32 dS blocks the causal diagonal crosses: it "
                                 "needs causal=True without a window / mask_mod (use 'all' for every block).")
    return log("ds_dq_rowlocal", value, "explicit (the dQ side of the dS blocks gets one exponent per query; the dK "
                                        "side keeps the block exponent)")


def _check_kernel_tuning(tuning: KernelTuning, bkern: str, ds: str, p_scaling: str) -> None:
    """Tuning fields the resolved backward kernel cannot compile, refused here with the reason instead of the kernel
    constructor's assert. A block-scaled dS^ (any ds_scaling but per_tensor) or a block-scaled P^ (p_scaling='block_1x32')
    indexes its scale-factor bytes 2 * stage + warpgroup -- the x32 TMEM-load mapping with two compute warpgroups -- on
    both E4M3-dS kernels (flash_bwd_sm100_fp4 / flash_bwd_sm100_2cta8 `ld-rep-64-sf-index-swap`)."""
    if bkern not in ("1cta", "2cta") or (ds == "per_tensor" and p_scaling == "row"):
        return
    what = " and ".join(([f"ds_scaling='{ds}'"] if ds != "per_tensor" else [])
                        + ([f"p_scaling='{p_scaling}'"] if p_scaling != "row" else []))
    bad = []
    if bkern == "1cta" and tuning.bwd_fp4_ld_rep == 64:
        bad.append("bwd_fp4_ld_rep=64 (the 1-CTA kernel)")
    if bkern == "2cta" and tuning.two_cta8_ld_rep == 64:
        bad.append("two_cta8_ld_rep=64 (the 2-CTA kernel)")
    if bad:
        _fail("E11", ValueError, f"KernelTuning {', '.join(bad)} cannot serve the block-scaled scale factors of {what}: "
                                 "their byte index is the x32 TMEM-load mapping with two compute warpgroups. Leave the "
                                 "field at its default (only the per-tensor dS with p_scaling='row' takes it: "
                                 "flash_attn_fp4's head_dim 256, where ds_scaling='auto' resolves to it; neither entry "
                                 "has an explicit per-tensor dS, rule 33).")


# ================================================================================================================
# 2-CTA eligibility (the rule of interface_bwd_fp4._use_2cta_dk8, restated on the plan's resolved values)
# ================================================================================================================

def _two_cta_ineligible(shp: CallShape, deterministic: bool, operand_scaling: str, extra: Tuple[Tuple[bool, str], ...],
                        gqa_ok: bool = False):
    reasons = []
    if not shp.mha and gqa_ok:
        # KernelTuning.bwd_2cta_gqa: the fp32 dK / dV reduce-add form, ordered by q head when deterministic (upstream's
        # dK / dV semaphores); batched and documents
        pass
    elif not shp.mha:
        reasons.append(f"GQA / MQA (h {shp.h}, h_kv {shp.h_kv}): the 2-CTA backward is MHA-only")
    if shp.d != 128:
        reasons.append(f"head_dim {shp.d} (the 2-CTA backward is d = 128)")
    if shp.aligned_varlen or shp.varlen or shp.seqused_k:
        # documents: aligned varlen runs packed at 128 (the key pair straddles documents, flash_bwd_sm100_2cta8
        # doc_t128); repacked varlen / batched seqused_k repack at 256 when the plan picks the 2-CTA kernel; any
        # total_q / total_k, deterministic with the document-local dQ order
        if shp.aligned_varlen and operand_scaling == "token_local":
            reasons.append("token-local operands on documents packed at 128 run on the 1-CTA kernel")
    else:
        # batched: any s_q / s_k (multiples of 128); s_k % 256 == 128 masks the last pair's upper half (key_tail)
        if shp.s_k % 256 != 0 and operand_scaling == "token_local":
            reasons.append(f"token-local operands with s_k % 256 != 0 (s_k = {shp.s_k}) run on the 1-CTA kernel")
    if operand_scaling == "block_32x32":
        reasons.append("block_32x32 operands are 1-CTA only")
    for bad, why in extra:
        if bad:
            reasons.append(why)
    return reasons


# The long-sequence default (owner decision 2026-10-07): with two_cta="auto", a key length of at least this many keys
# takes the 2-CTA backward wherever it is supported -- GQA (the q-head pairs, ordered when deterministic), deterministic,
# documents packed at 128 (aligned varlen) or repacked at 256, batched s_q != s_k and s_k % 256 == 128 tails. The key
# length is s_k batched, or the host-known max_seqlen_k on varlen (total_k when it is not given): no device read.
# Below it the rules are exactly the earlier ones.
TWO_CTA_LONG_KEYS = 32768


def _two_cta_gqa_ok(tuning_gqa, two_cta: str, shp: CallShape) -> bool:
    """GQA on the 2-CTA backward: KernelTuning.bwd_2cta_gqa=True; or, unset, two_cta='on' or the long-sequence default."""
    if tuning_gqa is not None:
        return bool(tuning_gqa)
    return two_cta == "on" or (two_cta == "auto" and shp.s_k >= TWO_CTA_LONG_KEYS)


def _resolve_two_cta(entry, two_cta, shp, deterministic, operand_scaling, extra, causal, log, gqa_ok: bool = False):
    reasons = _two_cta_ineligible(shp, deterministic, operand_scaling, extra, gqa_ok)
    if two_cta == "on":
        if reasons:
            _fail(15, ValueError, f"two_cta='on' but the 2-CTA backward cannot serve this call: {'; '.join(reasons)}.")
        return log("two_cta", "on", "explicit")
    if two_cta == "off":
        return log("two_cta", "off", "explicit")
    if shp.s_k >= TWO_CTA_LONG_KEYS:
        if reasons:
            return log("two_cta", "off", "auto: ineligible (" + "; ".join(reasons) + ")")
        if deterministic and not (entry == "fp8" and operand_scaling == "per_tensor"):
            # measured (cta2 NOTES, paired step, s_k 32768-131072): the 2-CTA ordered dQ reduce is not faster than the
            # 1-CTA kernel's dQ issue / release warps -- batched -1.8..+4.9 %, documents +3..+6 % -- so a
            # deterministic call keeps the 1-CTA backward (two_cta='on' takes the pair, GQA and documents included)
            return log("two_cta", "off", f"auto: deterministic at key length {shp.s_k} >= {TWO_CTA_LONG_KEYS} keeps the "
                                         "1-CTA backward (the 2-CTA ordered dQ reduce measured no faster; "
                                         "two_cta='on' takes it)")
        return log("two_cta", "on", f"auto: key length {shp.s_k} >= {TWO_CTA_LONG_KEYS} takes the 2-CTA backward "
                                    "wherever it is supported (long-sequence rule)")
    # ---- below the long-sequence threshold: the earlier rules, unchanged
    if shp.aligned_varlen:
        return log("two_cta", "off", "auto: aligned varlen below the long-sequence threshold takes the 1-CTA backward "
                                     "(two_cta='on' takes the 2-CTA kernel, its key pair straddling documents packed "
                                     "at 128)")
    if shp.varlen or shp.seqused_k:
        return log("two_cta", "off", "auto: repacked varlen / seqused_k documents below the long-sequence threshold take "
                                     "the 1-CTA backward (two_cta='on' takes the 2-CTA kernel, the documents repacked at "
                                     "256-aligned starts)")
    if shp.s_q != shp.s_k or shp.s_k % 256 != 0:
        return log("two_cta", "off", f"auto: batched s_q {shp.s_q} / s_k {shp.s_k} (s_q != s_k or s_k % 256 != 0) "
                                     "below the long-sequence threshold takes the 1-CTA backward")
    if not shp.mha and deterministic:
        return log("two_cta", "off", "auto: deterministic GQA below the long-sequence threshold takes the 1-CTA backward")
    if reasons:
        return log("two_cta", "off", "auto: ineligible (" + "; ".join(reasons) + ")")
    if deterministic and entry == "fp8" and operand_scaling == "per_tensor":
        # 8-bit explicit per-tensor operands keep the 2-CTA kernel when deterministic: there the 1-CTA kernel is not
        # bitwise the 2-CTA one in dK (max |diff| 0.17-0.20 causal) and ~1.1e-4 lower in dQ / dK cosine vs fp32 on causal
        # shapes.
        return log("two_cta", "on", "auto: deterministic 8-bit per-tensor operands keep the 2-CTA ordered dQ reduce "
                                    "(its dQ / dK measured the more accurate there)")
    if deterministic:
        # measured: the 1-CTA deterministic kernel with its dQ issue / release warps against the 2-CTA ordered reduce,
        # bwd (fp8 / fp4): s4096 causal -7 / -10 %, dense -5 / -9 %; s8192 causal -5 / -8 %; s16384 causal -0.6 / -0.9 %;
        # s32768 dense -2.7 / -2.7 %, causal a tie (-1.2..+0.7 %, two paired runs) -> causal s_k >= 32768 keeps the
        # 2-CTA kernel. O / dK / dV bitwise the 2-CTA kernel's on the token-block / block-scaled and 4-bit routes, dQ a
        # different fixed order (8-bit per-tensor: above).
        if causal and shp.s_k >= 32768:
            return log("two_cta", "on", "auto: deterministic causal s_k >= 32768 keeps the 2-CTA ordered dQ reduce "
                                        "(a tie with the 1-CTA kernel there)")
        return log("two_cta", "off", "auto: deterministic takes the 1-CTA kernel (its dQ issue / release warps beat "
                                     "the 2-CTA ordered dQ reduce below causal s_k 32768)")
    if entry == "fp8":
        return log("two_cta", "on", "auto: 8-bit takes the 2-CTA kernel wherever eligible")
    if operand_scaling in ("token_block", "token_local"):
        return log("two_cta", "on", "auto: 4-bit token-block operands take the 2-CTA kernel wherever eligible "
                                    "(3-6 % faster at s2048-8192, bitwise dK / dV)")
    if shp.s_k >= 16384 or (not causal and shp.s_k >= 8192):
        return log("two_cta", "on", f"auto: 4-bit s_k {shp.s_k} >= 16384, or dense >= 8192")
    return log("two_cta", "off", f"auto: 4-bit s_k {shp.s_k} below the 2-CTA threshold (1-CTA 2-9 % faster there)")


# ================================================================================================================
# the forward on CTA pairs (flash_fwd_sm100_fp4 use_2cta_forward)
# ================================================================================================================

def _mxfp8_pv_forward(route: str, op_res: str, tuning: KernelTuning, fork: bool = True) -> bool:
    """Whether the call's forward is the token-block MXFP8 P.V forward (E4M3 P-hat x block-scaled E4M3 V): the E4M3 P.V
    routes on token-block / 32 x 32 operands, unless KernelTuning.token_block_forward='fold' (plain E4M3 GEMMs with the
    factors folded into the scores) or the token-local operands (the fold forward) run instead."""
    return (fork and route in ("fp8", "fp4.e4m3_pv") and op_res in ("token_block", "block_32x32")
            and not (op_res == "token_block" and tuning.token_block_forward == "fold"))


def _resolve_p_hat_frame(entry: str, value: str, route: str, op_res: str, tuning: KernelTuning, log: _Log,
                         fork: bool = True) -> bool:
    """options.p_hat_frame -> the row-frame E4M3 P-hat (True) or the group-SF one (False) of the token-block MXFP8 P.V
    forward (HostSpec.fwd8_row_p / fwd4_row_p). "auto": row on the 8-bit entry, group on the 4-bit one (opt-in there:
    a small consistent training cost; lowprec.specs FWD8_ROW_P_DEFAULT / FWD4_ROW_P_DEFAULT). No silent no-ops:
    an explicit value on a forward without that choice is refused (rule E14)."""
    from .specs import FWD4_ROW_P_DEFAULT, FWD8_ROW_P_DEFAULT
    default = bool(FWD8_ROW_P_DEFAULT if entry == "fp8" else FWD4_ROW_P_DEFAULT)
    if not _mxfp8_pv_forward(route, op_res, tuning, fork):
        if value != "auto":
            _fail("E14", ValueError, f"p_hat_frame={value!r} frames the E4M3 P-hat of the token-block forward (MXFP8 "
                                     "P.V: operand_scaling 'token_block' / 'block_32x32' on an E4M3 P.V); this call's "
                                     f"forward ({route}, operand_scaling {op_res!r}) has no such choice.")
        return default
    if value == "auto":
        return log("p_hat_frame", "row" if default else "group",
                   "auto: the row-frame E4M3 P-hat (8-bit default)" if default else
                   "auto: one UE8M0 scale per 32 keys of P-hat (the row frame is opt-in on the 4-bit entry)") == "row"
    return log("p_hat_frame", value, "explicit") == "row"


# the asserts of the kernels and drivers that a public call could reach, as plan refusals. Each message names the
# rule.
_SPLIT_STEP_KT = (("qk_split_pipe", False), ("split_lean", False), ("corr_empty_early", False), ("fp4_q_stage", 1),
                  ("fp4_pv_log2_quant", False))


def _check_kernel_asserts(tuning: KernelTuning, route: str, op_res: str, causal: bool, bkern: str, row_p: bool,
                          fork: bool = True) -> None:
    """KernelTuning values the resolved kernels cannot run, refused before anything compiles (they used to reach a
    bare assert or a PTX assembly failure)."""
    # E18: the 4-bit P.V routes run the all-FP4 forward, which is the split-QK lean pipeline on two Q stages only
    if route in ("fp4.fp4_pv", "fp4.all_fp4"):
        bad = [f"{n}={v!r}" for n, v in _SPLIT_STEP_KT if getattr(tuning, n) == v]
        if causal and tuning.qk_split_causal is False:
            bad.append("qk_split_causal=False (causal)")
        if bad:
            _fail("E18", ValueError, f"KernelTuning {', '.join(bad)} cannot serve the all-FP4 forward of the 4-bit P.V "
                                     "routes: it is the split-QK lean pipeline with the log-domain P quantisation, the "
                                     "early acc-scale release and two Q stages only. Leave these fields at their "
                                     "defaults here.")
    mx = _mxfp8_pv_forward(route, op_res, tuning, fork)
    # E19: the MXFP8 P.V forward quantises P-hat in the log domain only
    if mx and tuning.fp4_pv_log2_quant is False:
        _fail("E19", ValueError, "KernelTuning fp4_pv_log2_quant=False cannot serve the token-block MXFP8 P.V forward "
                                 "(it requires the log-domain P quantisation); it applies to per-tensor operands only.")
    # E20: the E4M3 P-hat's offset + lazy-rescale threshold must stay within E4M3's range, with the route's defaults for
    # the field left unset (the forward's rule: (4, 4) on the pure E4M3 P-hat, else (0, 8))
    off, thr = tuning.fp8_pv_p_log2_offset, tuning.fp8_pv_rescale_threshold
    if off is not None or thr is not None:
        if route == "fp8":
            pure = (not mx) or row_p
        elif route == "fp4.e4m3_pv":
            pure = mx and row_p
        else:
            pure = False
        off_e = off if off is not None else (4.0 if pure else 0.0)
        thr_e = thr if thr is not None else (4.0 if pure else 8.0)
        if off_e + thr_e > math.log2(448) + 1e-9:
            _fail("E20", ValueError, f"KernelTuning fp8_pv_p_log2_offset + fp8_pv_rescale_threshold = {off_e} + {thr_e} "
                                     f"(the unset one at this forward's default) exceeds log2(448) = "
                                     f"{math.log2(448):.3f}: P-hat = E4M3(P * 2^offset) would saturate.")
    # E21: the fused tcgen05.ld.red row max does not assemble on the NVFP4-Q.K / E4M3-P.V forward
    if route == "fp4.e4m3_pv" and tuning.ldred_rowmax is True:
        _fail("E21", ValueError, "KernelTuning ldred_rowmax=True cannot serve the NVFP4-Q.K / E4M3-P.V forward (PTX "
                                 "assembly fails); it runs on the 8-bit and the 4-bit P.V forwards.")
    # E11: the E4M3-operand backward kernels' block-scaled paths
    bad = []
    if bkern == "1cta" and op_res != "per_tensor" and tuning.bwd_fp4_do_stage == 2:
        bad.append("bwd_fp4_do_stage=2 (the 1-CTA kernel's block-scaled operands take one dO stage)")
    if tuning.bwd_fp4_q_stage == 3 and (route != "fp8" or op_res == "block_32x32"):
        bad.append("bwd_fp4_q_stage=3 (the third Q stage is built on the 8-bit 1-CTA kernel with per-tensor or "
                   "token-block operands only)")
    if tuning.bwd_fp4_q_dyn_issue is True and (route != "fp8" or op_res == "block_32x32"):
        bad.append("bwd_fp4_q_dyn_issue=True (built on the 8-bit 1-CTA kernel with per-tensor or token-block operands "
                   "only)")
    if bkern == "2cta" and tuning.two_cta8_q_stage == 2:
        bad.append("two_cta8_q_stage=2 (the 2-CTA kernel's block-scaled dS takes one Q stage)")
    if bkern == "2cta" and tuning.two_cta8_warp_sync is False:
        bad.append("two_cta8_warp_sync=False (the 2-CTA kernel's dS scale exchange needs the per-warp hand-offs)")
    if tuning.two_cta8_warp_sync is False and tuning.two_cta8_mx_xw == "":
        bad.append("two_cta8_warp_sync=False with two_cta8_mx_xw='' (never run: suspected hang)")
    if bad:
        _fail("E11", ValueError, f"KernelTuning {'; '.join(bad)}. Leave the field at its default.")


def _fwd_two_cta_ineligible(shp: CallShape, fork: bool, features: bool, prequant: bool, split_step: bool,
                            docs: bool = False):
    reasons = []
    if not fork:
        reasons.append("not the fork forward's generic step on an E4M3 P.V")
    if features:
        reasons.append("window / softcap / score_mod / mask_mod (the pair runs the plain causal / dense step)")
    # (varlen / seqused_k documents: the pair on the documents in place -- one cluster per 512-row work tile of a
    # document, the batched TMA O storing only the pair's 128-row tiles that start inside it)
    if shp.d != 128 or shp.d_v != 128:
        reasons.append(f"head_dim {shp.d} / {shp.d_v} (the 2-CTA forward is d = d_v = 128)")
    if shp.s_q != shp.s_k and not docs:
        reasons.append(f"s_q != s_k ({shp.s_q} vs {shp.s_k})")
    if prequant:
        reasons.append("caller-pre-quantised Q / K")
    if split_step:
        reasons.append("the split-QK step (softmax_exp='mitchell')")
    return reasons


def _resolve_fwd_two_cta(value: str, shp: CallShape, causal: bool, fork: bool, features: bool, prequant: bool,
                         split_step: bool, log, nvfp4_qk_row_p_token_block: bool = False, docs: bool = False) -> bool:
    reasons = _fwd_two_cta_ineligible(shp, fork, features, prequant, split_step, docs)
    if value == "on":
        if reasons:
            _fail(15, ValueError, f"forward_two_cta='on' but the 2-CTA forward cannot serve this call: "
                                  f"{'; '.join(reasons)}.")
        log("forward_two_cta", "on", "explicit")
        return True
    if value == "off":
        log("forward_two_cta", "off", "explicit")
        return False
    if reasons:
        log("forward_two_cta", "off", "auto: ineligible (" + "; ".join(reasons) + ")")
        return False
    if docs:
        log("forward_two_cta", "off", "auto: varlen / seqused_k documents take the 1-CTA forward (its persistent "
                                      "longest-first work list); the pair is opt-in there (forward_two_cta='on')")
        return False
    on, why = _fwd_two_cta_auto(shp, causal, nvfp4_qk_row_p_token_block)
    log("forward_two_cta", "on" if on else "off", "auto: " + why)
    return on


# the shortest dense key length at which the pair was measured faster (b4 s4096 h16)
_FWD2CTA_MIN_SK = 4096


def _fwd_two_cta_auto(shp: CallShape, causal: bool, nvfp4_qk_row_p_token_block: bool):
    """The measured rule (paired medians, 1-CTA vs 2-CTA forward, O / LSE bitwise):
    the pair is faster only on the 4-bit-QK token-block forward with the row-frame P^ (NVFP4 S GEMM + MXFP8 P.V,
    the code default of fp4_bwd_autograd.fwd4_row_p) on dense calls: -2.9 / -3.7 / -3.9 % at b4 s4096 / b1 s16384 / b1 s32768. Every
    causal cell (stage-0 skip included: +1..+10 %), every 8-bit route (+0..+11 %) and the group-SF P^ forwards (+2..+10 %)
    are slower on the pair."""
    if not nvfp4_qk_row_p_token_block:
        return False, ("1-CTA: the pair measured slower on this forward (the 8-bit / per-tensor / group-SF P^ routes, "
                       "+0..+11 %)")
    if causal:
        return False, "1-CTA: causal (the pair measured +1..+6 % on the row-P NVFP4-QK forward)"
    if shp.s_k < _FWD2CTA_MIN_SK:
        return False, f"1-CTA: s_k {shp.s_k} < {_FWD2CTA_MIN_SK} (not measured faster there)"
    return True, (f"2-CTA: dense row-P NVFP4-QK token-block forward, s_k {shp.s_k} >= {_FWD2CTA_MIN_SK} "
                  "(-2.9..-3.9 % measured)")


# ================================================================================================================
# flash_attn_fp4
# ================================================================================================================

_ROUTES = {("fp8", "e4m3"): "fp4.e4m3_pv", ("fp4", "e4m3"): "fp4.fp4_pv", ("fp4", "fp4"): "fp4.all_fp4"}


def _refuse_fp4_per_tensor_ds(o) -> None:
    """Rule 33 on the 4-bit entry (as on the 8-bit one): the per-tensor dS
    is no explicit option of flash_attn_fp4 any more, on any route -- the block-scaled dS is the only explicit 4-bit dS.
    The per-tensor dS stays only where the plan resolves it itself: ds_scaling="auto" at head_dim 256 (the three-pass
    kernels have no block-scaled dS), and inside the all-4-bit kernel on the diagonal route's r_k pairs."""
    if o.ds_scaling == "per_tensor":
        _fail(33, ValueError, "ds_scaling='per_tensor' was removed from flash_attn_fp4: one dS scale per tensor, sized from "
                              "the forward's P_max bound, cost the default 4-bit route 0.024 / 0.031 val loss against the "
                              "block-scaled dS in training (2 seeds). Use the block-scaled dS (ds_scaling='auto' / "
                              "'block_32x32'); head_dim 256 (the three-pass kernels) resolves to a per-tensor dS by "
                              "itself under ds_scaling='auto'.")


def resolve_fp4(q, k, v, *, softmax_scale=None, causal=False, window_size=(None, None), softcap=0.0,
                score_mod=None, score_mod_bwd=None, mask_mod=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                max_seqlen_k=None, seqused_k=None, deterministic=False, return_lse=False, pv_precision="fp8",
                backward="e4m3", causal_safety="auto", qk_hadamard=None,
                options: Optional[FP4AttnOptions] = None, seqused_q=None) -> AttnPlan:
    o = options if options is not None else FP4AttnOptions()
    if not isinstance(o, FP4AttnOptions):
        raise TypeError(f"flash_attn_fp4: options must be FP4AttnOptions or None; got {type(o).__name__}.")
    for nm, val, allowed in (("pv_precision", pv_precision, ("fp8", "fp4")), ("backward", backward, ("e4m3", "fp4")),
                             ("causal_safety", causal_safety, ("auto", "key_local", "diagonal", "off"))):
        if val not in allowed:
            raise ValueError(f"flash_attn_fp4: {nm} must be one of {', '.join(repr(a) for a in allowed)}; got {val!r}.")
    for nm, val in (("causal", causal), ("deterministic", deterministic), ("return_lse", return_lse)):
        if not isinstance(val, bool):
            raise ValueError(f"flash_attn_fp4: {nm} must be a bool; got {val!r}.")
    had = _HadamardRequest(qk_hadamard, "fp4")
    softcap = float(softcap or 0.0)
    log = _Log()
    shp = _shape("fp4", q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k, seqused_q)
    tot_q, tot_k = shp.tot_q, shp.tot_k
    window = _window(window_size)
    grad = shp.requires_grad
    tuning = o.tuning if o.tuning is not None else KernelTuning()
    masked = bool(causal or window is not None or mask_mod is not None)

    # ---- rule 33: no explicit per-tensor dS on the 4-bit entry either ----------------------------------------------------
    _refuse_fp4_per_tensor_ds(o)

    # ---- rule 1: the route ------------------------------------------------------------------------------------------
    if backward == "fp4" and pv_precision == "fp8":
        _fail(1, ValueError, "backward='fp4' is the query-row backward of the all-4-bit forward: it reproduces a 4-bit "
                             "P-hat. Use pv_precision='fp4', or backward='e4m3'.")
    route = _ROUTES[(pv_precision, backward)]
    log("route", route, f"pv_precision={pv_precision!r}, backward={backward!r}")

    # ---- rule E13: the head dims the 4-bit kernels serve (measured: d = 96 / 192 fail in the quantisers / backward) ----
    if shp.d not in (64, 128, 256):
        _fail("E13", NotImplementedError, f"flash_attn_fp4 serves head_dim 64, 128 and 256; got {shp.d} (the "
                                          "block-scaled operands are 64 or 128 wide, or two 128-wide halves).")

    # ---- rule 27: the scale-factor atom and the query-row kernel's head dim -------------------------------------------
    # (varlen documents are repacked at 128-aligned starts by the step itself: any lengths)
    if not shp.varlen and (tot_q % 128 != 0 or tot_k % 128 != 0):
        _fail(27, ValueError, f"flash_attn_fp4 needs seqlen % 128 == 0 (the scale-factor atom is 128 rows); got "
                              f"s_q = {tot_q}, s_k = {tot_k}. flash_attn_fp8 serves other lengths.")
    if backward == "fp4" and shp.d != 128:
        _fail(27, ValueError, f"backward='fp4' needs head_dim 128 (the query-row kernel); got {shp.d}.")
    if pv_precision == "fp4" and shp.d != 128:
        # option A (pv4_route: the NVFP4 16x16 V of the 4-bit P.V and its backward dP) is built at d = 128 only; it used
        # to pass the plan and raise inside the driver's quantiser
        _fail(27, NotImplementedError, f"pv_precision='fp4' (the 4-bit P.V, option A) is built for head_dim 128; got "
                                       f"{shp.d}. Use pv_precision='fp8' (the default).")
    if pv_precision == "fp4" and not shp.varlen and (shp.s_k == 0 or (causal and shp.s_q > shp.s_k)):
        # (m4) a query tile with no visible key HANGS the 4-bit P.V forward (measured: all-FP4 b1 s_q 512 / s_k 256 and
        # b2 1024 / 512 causal, published 58ae0d8 included): bottom-right causal with s_q > s_k leaves the first
        # s_q - s_k queries -- whole 128-row tiles, as s % 128 == 0 -- without a key. Same contract as varlen documents
        # (lp_varlen require_keys: causal seqused_q <= seqused_k, dense a key) and causal batched seqused_k (rule 28).
        _fail(28, NotImplementedError,
              f"The 4-bit P.V routes need a key under every query: causal s_q > s_k ({shp.s_q} vs {shp.s_k}) leaves "
              "the first s_q - s_k queries without a key (and s_k = 0 every query), which the 4-bit forward does not "
              "serve. Use pv_precision='fp8'.")

    # ---- rules 4, 2, 3: Hadamard and the 2D Q blocks ------------------------------------------------------------------
    if shp.d != 128:
        had.unavailable(f"head_dim {shp.d} (the rotation is 128-point)", 4,
                        msg=f"qk_hadamard is built for head_dim 128 (the fused back-rotation kernels are d = 128); "
                            f"got {shp.d}.")
    if o.q_block_2d and masked:
        _fail(2, ValueError, "q_block_2d shares one scale across 16 queries, so a later query moves an earlier query's "
                             "codes (Q->O leak 0.27). It is refused under causal / window / mask_mod; use the default "
                             "1x16 Q scaling. (Dense varlen / seqused_k take it: documents start at multiples of 128, "
                             "so no 16-query block spans two documents, and a dense document has no query order.)")
    if o.q_block_2d:
        had.unavailable("q_block_2d (re-quantises the unrotated Q)", 3, ValueError,
                        "q_block_2d re-quantises the unrotated Q while K is rotated (wrong S); no rotated-Q 2D path "
                        "exists.")

    # (rules 24, 25, 6 are retired: the API offers no MXFP4 format; every 4-bit operand is NVFP4)
    # ---- rule E40: KernelTuning.fp4_qk_sf_vec = 32, the MXFP4 S operand (opt-in, not API) --------------------------------
    if tuning.fp4_qk_sf_vec == 32:
        why = []
        if (pv_precision, backward) != ("fp8", "e4m3"):
            why.append("it is wired on pv_precision='fp8', backward='e4m3' only")
        if o.q_block_2d:
            why.append("q_block_2d")
        feats = _features(window, softcap, score_mod, mask_mod, shp)
        if shp.aligned_varlen:   # (merge: varlen everywhere) the MXFP4 row quantiser has no padding-row predicate
            feats.append("aligned varlen documents")
        if feats:
            why.append(", ".join(feats))
        if why:
            _fail("E40", ValueError, "KernelTuning.fp4_qk_sf_vec=32 (MXFP4 Q / K) cannot serve this call: "
                                     + "; ".join(why) + ".")
        had.unavailable("KernelTuning.fp4_qk_sf_vec=32 (no rotated MXFP4 Q / K quantiser)", "E40", ValueError,
                        "qk_hadamard=True with KernelTuning.fp4_qk_sf_vec=32: no rotated MXFP4 Q / K quantiser exists.")

    # ---- causal safety: rules 7, 8, 9 then resolution -------------------------------------------------------------------
    if causal_safety in ("key_local", "diagonal") and pv_precision == "fp8":
        _fail(7, ValueError, f"causal_safety='{causal_safety}' applies to a 4-bit P.V (its key-local V scale); with "
                             "pv_precision='fp8' there is no 4-bit V group along keys for it to fix. Exact causality "
                             "on the 8-bit P.V needs operand_scaling='token_local' (or per-tensor V with a pinned "
                             "amax): the default token-block E4M3 V keeps the documented in-32-token-block channel "
                             "(sub-ULP to ~1e-3, < 32 rows).")
    if causal_safety == "key_local" and route == "fp4.fp4_pv":
        _fail(8, NotImplementedError, "Option A (pv_precision='fp4', backward='e4m3') has the diagonal-only key-local "
                                      "V under causal (causal_safety='diagonal'); the key-local V on every tile is "
                                      "built on the all-4-bit route only.")
    if causal_safety == "diagonal" and (not causal or window is not None or mask_mod is not None):
        _fail(9, ValueError, "causal_safety='diagonal' classifies the causal diagonal: it needs causal=True and no "
                             "window.")
    if pv_precision == "fp8":
        cs = log("causal_safety", "none", "8-bit P.V: no 4-bit V group along keys; exact causality only under "
                                          "operand_scaling='token_local' (or per-tensor V with a pinned amax): token-block "
                                          "/ 32 x 32 E4M3 V keep the documented in-32-token-block channel (sub-ULP to "
                                          "~1e-3, < 32 rows), a moving per-tensor amax the global one")
    elif causal_safety != "auto":
        cs = log("causal_safety", causal_safety, "explicit")
    elif not masked:
        cs = log("causal_safety", "none", "auto: dense, no masked tile")
    elif backward == "fp4" and causal and window is None and mask_mod is None:
        cs = log("causal_safety", "diagonal", "auto on the all-4-bit route under causal: diagonal-only key-local V with "
                                              "the K-channel fold")
    elif causal and window is None and mask_mod is None:
        cs = log("causal_safety", "diagonal", "auto on option A under causal: the diagonal-only key-local V "
                                              "(pv4_route.pv4_vdiag)")
    else:
        cs = log("causal_safety", "key_local", "auto: 4-bit P.V under a mask the diagonal route does not describe")
    if cs == "off" and not masked and pv_precision == "fp4":
        cs = log("causal_safety", "none", "'off' on a dense call: there is no masked tile to leak through")

    # ---- rule 10: what the diagonal route is built for -------------------------------------------------------------------
    if cs == "diagonal" and (o.fp4_backward_dkdv == "fp4" or o.v_hadamard or o.all_fp4_stabiliser == "eden_rht"):
        _fail(10, NotImplementedError, "causal_safety='diagonal' is built for the hybrid backward "
                                       "(fp4_backward_dkdv='e4m3') without Hadamard variants.")
    # ---- rules 11, 12: the K channel of the diagonal route ----------------------------------------------------------------
    diag_q = cs == "diagonal" and backward == "fp4"   # the query-row backward's diagonal route (K channel)
    if o.k_channel != "fold" and not diag_q:
        _fail(11, ValueError, "k_channel selects how the diagonal route closes dQ's K operand; it has no meaning "
                              "without causal_safety='diagonal'.")
    if diag_q and o.k_channel != "open":
        bad = []
        if o.dq_operand == "e4m3":
            bad.append("dq_operand='e4m3' (every closed K channel needs the E2M1 dQ operand)")
        if o.k_channel == "cut" and not shp.mha:
            bad.append(f"GQA (h {shp.h}, h_kv {shp.h_kv}): the cut-pair alias needs MHA")
        if o.k_channel == "cut" and (shp.s_k - shp.s_q) % 128 != 0:
            bad.append(f"s_k - s_q = {shp.s_k - shp.s_q} is not a whole number of 128-key tiles")
        if bad:
            _fail(12, NotImplementedError,
                  f"k_channel='{o.k_channel}' cannot close dQ's K channel on this call: {'; '.join(bad)}. dQ of a "
                  "query would read the E4M3 scale of K^T groups holding keys after it. k_channel='all' covers any "
                  "head ratio and offset (E2M1 dQ); k_channel='open' runs with that channel open, knowingly.")
    k_channel = o.k_channel if diag_q else "n/a"

    # ---- rule 5: the all-4-bit Q/K Hadamard is the key-local route's --------------------------------------------------------
    if backward == "fp4" and had.auto:
        # the all-4-bit route's Q/K Hadamard is its own implementation (QuantSpec.vtok_hadqk, key-local route only),
        # not the zero-cost path measured for the default routes: it stays an explicit choice
        had.unavailable("the all-4-bit route (its own Q/K Hadamard is explicit: qk_hadamard=True on the key-local "
                        "route)")
    if backward == "fp4" and cs != "key_local":
        had.unavailable("the all-4-bit route off the key-local route", 5,
                        msg="On the all-4-bit route the Q/K Hadamard is wired for the key-local route only; for dense "
                            "use options.all_fp4_stabiliser='eden_rht' (rotates Q, K, V, dO jointly).")
    # ---- rule 31 ------------------------------------------------------------------------------------------------------------
    if backward == "fp4" and o.all_fp4_stabiliser == "eden_rht" and cs == "key_local":
        _fail(31, ValueError, "eden_rht rotates V jointly with Q/K/dO; the key-local V payload does not share it.")
    # ---- rule 23: the per-tile frame is not defined on the integer-frame all-FP4 step (the only 4-bit P.V forward) -------
    if o.p_frame == "per_tile":
        _fail(23, ValueError, "p_frame='per_tile' is not defined on the integer-frame all-FP4 step that the 4-bit P.V "
                              "forward runs (pv_precision='fp4'); use p_frame='two_level' (the default) or "
                              "'single_level'.")
    # ---- rules 13, 29, 30: options of the other backward --------------------------------------------------------------------
    if deterministic and backward == "fp4":
        _fail(13, NotImplementedError, "The all-4-bit backward reduces dQ with atomics across key tiles; a deterministic "
                                       "dQ is not built. Use backward='e4m3'.")
    if backward == "e4m3":
        for fname, default in (("dq_operand", "fp4"), ("sr_dk_operand", True), ("sr_dv_operand", False),
                               ("fp4_backward_dkdv", "e4m3"), ("all_fp4_stabiliser", "sr"),
                               ("fp4_ds_multiplier", "auto")):
            if getattr(o, fname) != default:
                _fail(29, ValueError, f"{fname} configures the all-4-bit backward (backward='fp4').")
    if o.sr_ds and backward == "fp4":
        _fail(30, ValueError, "On backward='fp4' dS rounding follows all_fp4_stabiliser.")

    # ---- softmax value function: rule 21 (22 retired with direct_p_guard) -------------------------------------------------------------------------------
    if o.softmax_exp == "auto":
        sm_exp = log("softmax_exp", "mitchell" if pv_precision == "fp4" else "exp2",
                     "auto: Mitchell (Direct-P) where P-hat is 4-bit" if pv_precision == "fp4"
                     else "auto: exp2 where P-hat is E4M3 (Mitchell there drifted late in training)")
    else:
        sm_exp = log("softmax_exp", o.softmax_exp, "explicit")
    if sm_exp == "mitchell" and pv_precision == "fp8":
        _fail(21, ValueError, "Mitchell P-hat on an E4M3 P.V was removed (the 8-bit P-hat is"
                              " formed with exp2 only); use softmax_exp='exp2'.")

    # ---- rule 26: seqused_k_group_aligned (seqused_k with a 4-bit P.V: no such route, rule 28) ----------------------------
    if o.seqused_k_group_aligned:
        _fail(26, ValueError, "seqused_k_group_aligned applies to seqused_k with a 4-bit P.V, which no route takes "
                              "(the 4-bit P.V routes' V scale is key-local for causal only); seqused_k runs on "
                              "pv_precision='fp8'.")

    # ---- rule 28: the masking / varlen features --------------------------------------------------------------------------
    feats = _features(window, softcap, score_mod, mask_mod, shp)
    _check_features("fp4", route, feats, causal, window, softcap, score_mod, score_mod_bwd, mask_mod, shp, return_lse,
                    grad, False, None if backward == "fp4" else
                    ("per_tensor" if o.ds_scaling == "auto" and shp.d > 128 else "block_32x32"), log)
    docs = shp.varlen or shp.seqused_k

    # ---- options that only one route reads (no silent no-ops) -------------------------------------------------------------
    if o.p_consistency != "reproduce_forward_p" and not (pv_precision == "fp4" and backward == "e4m3"):
        _fail("E1", ValueError, "p_consistency selects option A's reproduced forward P-hat (pv_precision='fp4', "
              "backward='e4m3'); this route does not read it.")
    if o.do_rounding != "rtn" and not (pv_precision == "fp4" and backward == "e4m3" and grad):
        _fail("E1", ValueError, "do_rounding rounds option A's 4-bit dO (pv_precision='fp4', backward='e4m3') in its "
              "backward; this call does not read it (no other route has it, and option A's inference forward has no dO).")
    if o.exp_mix not in ("auto", "none"):
        from flash_attn.cute.flash_fwd_sm100_fp4 import parse_p_mix
        parse_p_mix(o.exp_mix)   # raises on a malformed mix
        if not (pv_precision == "fp4" and o.softmax_exp in ("auto", "mitchell")
                and (backward == "fp4" or o.p_consistency == "reproduce_forward_p")):
            _fail("E1", ValueError, "exp_mix mixes the value functions of the exact 4-bit P-hat (pv_precision='fp4', "
                  "softmax_exp Mitchell: the all-4-bit route backward='fp4', or option A with p_consistency="
                  "'reproduce_forward_p'); this call does not read it.")
    if o.p_frame != "two_level" and route != "fp4.fp4_pv":
        _fail("E2", ValueError, "p_frame frames the 4-bit P-hat of pv_precision='fp4', backward='e4m3'.")
    if o.q_block_2d and route != "fp4.e4m3_pv":
        _fail("E3", ValueError, "q_block_2d is wired on pv_precision='fp8' (the E4M3 P.V route) only.")
    if o.v_hadamard and not (backward == "fp4" and cs == "key_local"):
        _fail("E4", ValueError, "v_hadamard applies to the all-4-bit key-local route (backward='fp4', "
                                "causal_safety resolving to 'key_local').")
    if o.p_scaling != "row" and backward == "fp4":
        _fail("E5", ValueError, "p_scaling configures the E4M3 P-hat of the E4M3-operand backward (backward='e4m3').")

    # ---- operand scaling: rule 16 -------------------------------------------------------------------------------------------
    # scaling-consistency: the operand scaling picked here is the ONE scaling of V (and, on the backward, of the E4M3
    # Q / K / dO copies) that both passes use; the bridge quantises each operand once.
    op = o.operand_scaling
    if route == "fp4.fp4_pv":
        # option A (pv4_route.pv4_check_config): the dk8 blk32x128 backward -- token-block E4M3 copies of Q / K / dO for
        # the dK / dQ / dV GEMMs; V has no E4M3 copy (the dP GEMM reads the forward's own NVFP4 16x16 V codes)
        if op in ("per_tensor", "block_32x32", "token_local"):
            _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: option A's E4M3 copies of Q / K / dO "
                                  "are token blocks (the dk8 blk32x128 backward). Use operand_scaling='auto'.")
        op_res = log("operand_scaling", "token_block", "option A: token-block E4M3 copies of Q / K / dO; V is the "
                                                       "forward's NVFP4 16x16 codes in both passes")
    elif route != "fp4.e4m3_pv":
        # the all-4-bit route's E4M3 copies (the hybrid dK / dV) are per-tensor
        if op in ("token_block", "block_32x32", "token_local"):
            _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: the all-4-bit route's E4M3 operand "
                                  "copies are per-tensor only. Use operand_scaling='auto' to fall back to per_tensor.")
        op_res = log("operand_scaling", "per_tensor", "the all-4-bit route's E4M3 copies (hybrid dK / dV) are "
                                                      "per-tensor")
    elif docs and op == "per_tensor" and shp.d == 128:
        # varlen documents with explicit per-tensor operands: one scale per (document, kv head) of every E4M3 operand
        # (FA3 / Meta granularity), reduced on the GPU over each document's real rows only -- never one scale across
        # documents (the NVFP4 S codes carry their own 1 x 16 scales). Repacked varlen / batched seqused_k: the
        # aligned path on the repacked buffer (lowprec._routes.run_documents)
        op_res = log("operand_scaling", "per_tensor", "explicit: varlen documents, one scale per (document, kv head) "
                                                      "of the E4M3 Q / K / V / dO copies from the document's real rows"
                                                      + ("" if shp.aligned_varlen else
                                                         " (the aligned path on the repacked documents)"))
    elif docs and op == "token_local" and shp.d in (64, 128):
        # token-local operands on documents: per-token exponents (no scale spans two tokens, let alone two documents);
        # the fold forward reads its factor tables at each document's offsets (flash_fwd_sm100_fp4 tb_varlen)
        op_res = log("operand_scaling", "token_local", "explicit: varlen / seqused_k documents, per-token exponents "
                                                       "(the fold forward on the documents)")
    elif docs:
        # varlen documents (seqused_k is served as documents of the batch rows) take token-block operands
        # on both passes; a per-tensor scale would be a channel between documents
        bad = [f"head_dim {shp.d} (token blocks are built for 64 and 128)"] if shp.d not in (64, 128) else []
        if op in ("per_tensor", "block_32x32", "token_local") or bad:
            _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: varlen / seqused_k documents take "
                                  "token-block operands (a per-tensor scale over all documents would be a channel "
                                  "between documents; token-local operands need head_dim 64 / 128)"
                                  + ("; per-(document, kv head) per-tensor scales run at head_dim 128 only"
                                     if op == "per_tensor" else "")
                                  + (f"; {bad[0]}" if bad else "") + ".")
        op_res = log("operand_scaling", "token_block", "varlen / seqused_k documents: token blocks on both passes")
    else:
        why = []
        if shp.d not in (64, 128):
            why.append(f"head_dim {shp.d} (token blocks are built for 64 and 128)")
        # (no explicit per-tensor dS any more, rule 33: the only per-tensor dS is head_dim 256's, named above)
        if o.p_scaling == "block_1x32":
            why.append("p_scaling='block_1x32' on the 4-bit route")
        # (s_q != s_k takes token blocks too: the factor table's Q / dO rows are indexed by query block and its
        # K / V rows by key block, fp4_bwd_autograd.token_block_factor_table)
        if op == "auto":
            if why:
                _refuse_auto_per_tensor("fp4", why)
            op_res = log("operand_scaling", "token_block", "auto: 32 tokens x d UE8M0 (the certified blk32x128)"
                         + _token_block_shape_note(shp))
        else:
            bad = why
            if op != "per_tensor" and bad:
                _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: {'; '.join(bad)}. Pass "
                                      "operand_scaling='per_tensor' explicitly to run it on per-tensor scales.")
            op_res = log("operand_scaling", op, "explicit")

    if op_res == "token_local":
        had.unavailable("operand_scaling='token_local' (the rotated token-local quantiser is not built)", 16, ValueError,
                        "operand_scaling='token_local' with qk_hadamard: the rotated token-local quantiser is not built.")
    _tuning_pending(tuning, route, op_res, had.on, bool(feats))

    # ---- head-dim path and dS scaling: rule 18 --------------------------------------------------------------------------
    hd_path = "single_pass" if shp.d in (64, 128) else ("zero_pad_128" if shp.d < 128 else "three_pass")
    if shp.d > 256:
        _fail("E6", ValueError, f"flash_attn_fp4 serves head_dim <= 256; got {shp.d}.")
    if backward == "e4m3":
        if o.ds_scaling == "block_32x32" and shp.d > 128:
            _fail(18, ValueError, f"block_32x32 dS is built into the single-pass d = 128 kernel; the head_dim {shp.d} "
                                  "three-pass kernels use a per-tensor dS. Use ds_scaling='auto'.")
        if o.ds_scaling == "auto":
            ds = log("ds_scaling", "per_tensor" if shp.d > 128 else "block_32x32",
                     f"auto: head_dim {shp.d} > 128 runs the three-pass kernels (per-tensor dS)" if shp.d > 128
                     else "auto: one UE8M0 per 32 x 32 dS block")
        else:
            ds = log("ds_scaling", o.ds_scaling, "explicit")
    else:
        # (ds_scaling is "auto" or "block_32x32" here: an explicit per-tensor dS is refused, rule 33)
        ds = log("ds_scaling", "block_32x32", "the dK-side E4M3 dS (the kernel keeps per-tensor on the diagonal route's "
                                              "r_k pairs)")
    # ---- rules E15, E16: what the query-row backward's drivers / kernel build (they used to fail inside the call) -------
    if backward == "fp4":
        if o.fp4_ds_multiplier == "row_local" and not (causal and o.fp4_backward_dkdv == "e4m3"):
            _fail("E15", ValueError, "fp4_ds_multiplier='row_local' sizes each query row's dS from what that row may see: "
                                     "it needs causal=True and fp4_backward_dkdv='e4m3' (the E4M3 dK / dV).")
        if o.fp4_ds_multiplier in ("global", "global_row_boost") and cs == "diagonal":
            _fail("E15", ValueError, f"fp4_ds_multiplier={o.fp4_ds_multiplier!r} needs max|dO| and the global dS bound "
                                     "as host scalars, and the diagonal route (causal_safety='diagonal') reads no scale "
                                     "on the host; use fp4_ds_multiplier='auto' (row_local there).")
        if o.dq_operand == "e4m3":
            _fail("E16", NotImplementedError,
                  "dq_operand='e4m3' (the E4M3 dS of the dQ GEMM) is not built: with fp4_backward_dkdv='e4m3' the "
                  "query-row kernel does not compile it off the diagonal route and the diagonal route reads no host "
                  "scale for it; fp4_backward_dkdv='fp4' does not read it. Use dq_operand='fp4' (the default).")
    if backward == "fp4":
        mult = (log("fp4_ds_multiplier", "row_local" if causal else "global_row_boost",
                    "auto: row-local when causal (closes the global dS channel), per-row boost when dense")
                if o.fp4_ds_multiplier == "auto" else log("fp4_ds_multiplier", o.fp4_ds_multiplier, "explicit"))
    else:
        mult = "n/a"
    if backward == "fp4" and o.ds_dq_rowlocal != "off":
        _fail(29, ValueError, "ds_dq_rowlocal configures the E4M3-dS backward (backward='e4m3'); the query-row backward "
                              "sizes its 4-bit dS with fp4_ds_multiplier.")
    rowq = _resolve_ds_dq_rowlocal("fp4", o.ds_dq_rowlocal, ds, causal, window, mask_mod, log)
    _check_token_local(op_res, rowq, backward, had.on, window, softcap, score_mod, mask_mod, shp, log, causal)
    pts = _resolve_per_tensor_scale(o.per_tensor_scale, op_res, log)

    # ---- 2-CTA: rules 17, 15 (rule 14 retired: the 2-CTA kernel orders dQ) ---------------------------------------
    if op_res == "block_32x32" and o.two_cta == "on":
        _fail(17, ValueError, "block_32x32 operands are 1-CTA only; use operand_scaling='token_block'.")
    if backward == "fp4":
        if o.two_cta == "on":
            _fail(15, ValueError, "two_cta='on' but the 2-CTA backward cannot serve this call: the query-row backward "
                                  "(backward='fp4') has no 2-CTA kernel.")
        two = log("two_cta", "off", "the query-row backward has no 2-CTA kernel")
    else:
        two = _resolve_two_cta("fp4", o.two_cta, shp, deterministic, op_res,
                               ((o.p_scaling != "row", "p_scaling='block_1x32' is 1-CTA only on the 4-bit route"),
                                (docs and op_res == "per_tensor",
                                 "per-(document, kv head) per-tensor scales run on the 1-CTA kernel"),
                                (o.sr_ds, "sr_ds: the stochastically rounded dS-hat is built in the 1-CTA kernel only"),
                                (shp.aligned_varlen and (mask_mod is not None or score_mod is not None or softcap),
                                 "score_mod / mask_mod / softcap on documents packed at 128 run on the 1-CTA kernel "
                                 "(the 2-CTA key pair straddles documents there)"),
                                (deterministic and (window is not None or mask_mod is not None or softcap
                                                    or score_mod is not None),
                                 "deterministic with a window / mask_mod / softcap / score_mod orders dQ on the 1-CTA "
                                 "kernel only")),
                               causal, log, gqa_ok=_two_cta_gqa_ok(tuning.bwd_2cta_gqa, o.two_cta, shp))
    _check_aligned("fp4", shp, two if grad else "off", tuning, had.on, log, causal=causal)

    # ---- rule 34: the lean save --------------------------------------------------------------------------------------------
    if not o.save_bf16_inputs:
        why = []
        if route != "fp4.e4m3_pv":
            why.append(f"the {route} route's backward reads the BF16 Q / K / V")
        if shp.d != 128:
            why.append(f"head_dim {shp.d} (the padded / three-pass backward reads the BF16 tensors)")
        if why:
            _fail(34, ValueError, "save_bf16_inputs=False cannot serve this call: " + "; ".join(why) + ".")
        if op_res != "per_tensor":
            log("save_bf16_inputs", False, f"{op_res} operands never keep the BF16 tensors (already lean)")

    # ---- the scaling-consistency rule ---------------------------------------------------------------------------------------
    qk_fmt = "nvfp4" + ("_2d_q" if o.q_block_2d else "")
    if route == "fp4.e4m3_pv":
        pairs = [("Q/K (S GEMM)", f"{qk_fmt} 1x16", f"{qk_fmt} 1x16 (the forward's saved codes)"),
                 ("V", f"e4m3 {op_res}", f"e4m3 {op_res} (the forward's saved codes)")]
    elif route == "fp4.fp4_pv":
        # option A: one NVFP4 16x16 V quantisation read by the forward's P.V and the backward's dP
        v_fwd = "nvfp4 16x16 + key-local diagonal" if cs == "diagonal" else "nvfp4 16x16"
        pairs = [("Q/K (S GEMM)", "nvfp4 1x16", "nvfp4 1x16 (the forward's saved codes)"),
                 ("V", v_fwd, v_fwd + " (the forward's saved codes)")]
    else:
        v_fwd = {"none": "nvfp4 16x16", "off": "nvfp4 16x16", "key_local": "nvfp4 key-local",
                 "diagonal": "nvfp4 16x16 + key-local diagonal"}[cs]
        pairs = [("Q/K (S GEMM)", "nvfp4 1x16", "nvfp4 1x16 (the forward's saved codes)"),
                 ("V", v_fwd, v_fwd + " (the forward's saved codes)")]
    pairs = [(n, f, b.split(" (")[0]) for n, f, b in pairs]
    # ---- rule 36: option A (pv_precision="fp4", backward="e4m3") is inference only ------------------------------------
    if route == "fp4.fp4_pv" and grad:
        _fail(36, ValueError, OPTION_A_INFERENCE_ONLY)
    consistency, pend = _consistency(route, pairs, grad, log)
    # SCALING-CONSISTENCY RULE: the rotated codes are the ONE quantisation of Q / K both passes read
    qk_hadamard = had.log(log, "Q R / K R (R = diag(signs) H_128 / sqrt(128), seed 4321) quantised in one rotated read "
                          "per tensor; " + (
                              f"the {'E4M3 ' + op_res if route == 'fp4.e4m3_pv' else 'E4M3 token_block'} copies take "
                              "their scale statistics on the rotated rows; forward and backward read the same rotated "
                              "codes; dQ / dK rotated back in the backward's epilogues" if route != "fp4.all_fp4" else
                              "the all-4-bit key-local route's own rotation"))

    if softmax_scale is None:
        softmax_scale = log("softmax_scale", 1.0 / math.sqrt(shp.d), "default 1/sqrt(head_dim)")

    row_p = _resolve_p_hat_frame("fp4", o.p_hat_frame, route, op_res, tuning, log)
    fwd = FwdSpec(kernel="fork_fp4_all" if route == "fp4.all_fp4" else "fork_fp4", qk_format="nvfp4",
                  pv_operand="e4m3" if pv_precision == "fp8" else "nvfp4", operand_scaling=op_res,
                  softmax_exp=sm_exp, causal_safety=cs,
                  p_frame=o.p_frame if route == "fp4.fp4_pv" else "n/a", q_block_2d=o.q_block_2d,
                  qk_hadamard=qk_hadamard, v_hadamard=o.v_hadamard, causal=causal, window=window, softcap=softcap,
                  score_mod=score_mod is not None, mask_mod=mask_mod is not None, varlen=shp.varlen,
                  seqused_k=shp.seqused_k, return_lse=return_lse, per_tensor_scale=pts,
                  aligned_varlen=shp.aligned_varlen,
                  two_cta=_resolve_fwd_two_cta(o.forward_two_cta, shp, causal, route == "fp4.e4m3_pv",
                                               bool(_mods(feats)), False, False, log,
                                               nvfp4_qk_row_p_token_block=bool(
                                                   route == "fp4.e4m3_pv" and op_res == "token_block"
                                                   and not qk_hadamard and row_p), docs=bool(docs)))
    if not grad:
        bkern = "none"
    elif backward == "fp4":
        bkern = "query_row"
    else:
        bkern = "2cta" if two == "on" else "1cta"
    _check_kernel_tuning(tuning, bkern, ds, o.p_scaling)
    _check_kernel_asserts(tuning, route, op_res, causal, bkern, row_p)
    bwd = BwdSpec(kernel=bkern, head_dim_path=hd_path, operand_scaling=op_res, ds_scaling=ds, ds_dq_rowlocal=rowq,
                  p_scaling=o.p_scaling,
                  grad_format="e4m3", deterministic=deterministic, sr_ds=o.sr_ds,
                  sr_dk_operand=o.sr_dk_operand if backward == "fp4" else False,
                  sr_dv_operand=o.sr_dv_operand if backward == "fp4" else False,
                  fp4_backward_dkdv=o.fp4_backward_dkdv if backward == "fp4" else "n/a",
                  dq_operand=o.dq_operand if backward == "fp4" else "e4m3",
                  all_fp4_stabiliser=o.all_fp4_stabiliser if backward == "fp4" else "n/a",
                  fp4_ds_multiplier=mult, k_channel=k_channel, per_tensor_ds_source="recompute",
                  save_bf16_inputs=o.save_bf16_inputs,
                  # option A with reproduce_forward_p rebuilds the forward's P-hat: the forward's value function
                  softmax_exp=sm_exp if (backward == "fp4" or (route == "fp4.fp4_pv"
                                                               and o.p_consistency == "reproduce_forward_p")) else "exp2",
                  p_consistency=o.p_consistency if route == "fp4.fp4_pv" and grad else "n/a",
                  do_rounding=o.do_rounding if route == "fp4.fp4_pv" and grad else "n/a",
                  exp_mix=_exp_mix(o, route, sm_exp, log))
    # (the E4M3-operand backward recomputes P with exp2 from the forward's LSE on every 4-bit route; the query-row
    # backward reproduces the forward's P-hat, so its value function IS the forward's -- one field, both halves)
    return AttnPlan(entry="fp4", route=route, softmax_scale=float(softmax_scale), shape=shp, fwd=fwd, bwd=bwd,
                    tuning=tuning, prequant=False, q_scale_fused=False, consistency=consistency, pending=pend,
                    resolution_log=tuple(log.rows), options=o,
                    specs=build_specs(route, fwd, bwd, tuning, shp, fwd4_row_p=row_p))


# ================================================================================================================
# flash_attn_fp8
# ================================================================================================================

def _refuse_fp8_per_tensor_ds(o) -> None:
    """Rule 33: the per-tensor dS is no explicit option of the 8-bit entry any more -- the
    block-scaled dS is the only explicit 8-bit dS (and "auto" never resolved to per-tensor here). Both ways of sizing it
    are refused by name: the exact amax ("recompute") is an extra full pass (403-633 % of the backward kernel), the
    forward's P_max bound ("forward_bound") is 4.7x loose (+0.029 val); either is a global causal channel."""
    if o.ds_scaling == "per_tensor":
        _fail(33, ValueError, "ds_scaling='per_tensor' was removed from flash_attn_fp8: sizing a per-tensor dS needs "
                              "either an extra full pass for the exact max|dS| (403-633 % of the backward kernel) or "
                              "the forward's P_max bound, 4.7x loose (+0.029 val), and either is a global causal channel. "
                              "Use the block-scaled dS (ds_scaling='auto' / 'block_32x32').")
    if o.per_tensor_ds_source == "recompute":
        _fail(33, ValueError, "per_tensor_ds_source='recompute' was removed with the per-tensor dS: the exact max|dS| "
                              "costs an extra full pass (403-633 % of the backward kernel). Use the block-scaled dS "
                              "(ds_scaling='auto').")
    if o.per_tensor_ds_source == "forward_bound":
        _fail(33, ValueError, "per_tensor_ds_source='forward_bound' was removed with the per-tensor dS: the forward's "
                              "P_max bound is 4.7x loose (+0.029 val). Use the block-scaled dS (ds_scaling='auto').")


_FP8_DS_1CTA_ONLY = ("row_local", "row_1x32")
_FP8_DS_2CTA_ONLY = ("block_32x64", "tile_128")


def _check_prequant_fa3(shp: CallShape, o, window, softcap, score_mod, mask_mod, q_quantized, q_scale_fused,
                        log: _Log):
    """Rule 32, FA3-style pre-quantised inputs (E4M3 Q / K / V + (b, h_kv) q_descale / k_descale / v_descale): what the
    kernels serve. The forward is the fork kernel on per-tensor operands (head_dim 64 / 128; s % 128 == 0 batched, or
    the aligned varlen documents in place); a backward runs on the per-(document, kv head) 1-CTA kernel."""
    if q_quantized is not None or q_scale_fused:
        _fail(32, ValueError, "flash_attn_fp8: q_descale / k_descale / v_descale (E4M3 q / k / v) and q_quantized / "
                              "k_quantized / q_scale_fused are two forms of pre-quantised input: pass one.")
    # (window / softcap / score_mod / mask_mod run on this path: the forward's score_mod reads the per-(row or
    # document, kv head) score scale before the mod, the backward's doc_scaled kernel applies the masks and mods)
    if shp.seqused_k:
        _fail(32, NotImplementedError, "pre-quantised E4M3 inputs with seqused_k (batched) reach the plan only from "
                                       "flash_attn_fp8, which runs them as aligned documents of the batch rows "
                                       "(lowprec.prequant_fa3); resolve that form instead.")
    if shp.varlen and not shp.aligned_varlen:
        _fail(32, NotImplementedError, "pre-quantised E4M3 inputs on repacked varlen (cu_seqlens without seqused_q / "
                                       "seqused_k) reach the plan only from flash_attn_fp8, which repacks the codes into "
                                       "an aligned buffer and runs the aligned form (lowprec.prequant_fa3); resolve "
                                       "that form instead.")
    if shp.d not in (64, 128):
        _fail(32, NotImplementedError, f"pre-quantised E4M3 inputs: head_dim 64 or 128 (the fork forward on per-tensor "
                                       f"operands); got {shp.d}.")
    if not shp.varlen and (shp.tot_q % 128 or shp.tot_k % 128):
        _fail(32, NotImplementedError, f"pre-quantised E4M3 inputs on a batched call need s_q % 128 == 0 and s_k % 128 "
                                       f"== 0 (got {shp.tot_q}, {shp.tot_k}); ragged lengths take the aligned varlen "
                                       "form (cu_seqlens at multiples of 128 + seqused_q / seqused_k).")
    if o.operand_scaling not in ("auto", "per_tensor"):
        _fail(32, ValueError, f"operand_scaling='{o.operand_scaling}' with pre-quantised E4M3 inputs: the caller's "
                              "codes carry one scale per (batch row or document, kv head); leave operand_scaling at "
                              "'auto' (or 'per_tensor').")
    if o.forward_kernel == "upstream":
        _fail(32, ValueError, "forward_kernel='upstream' with pre-quantised E4M3 inputs: the per-(row, kv head) score "
                              "scale is read by the fork forward only.")
    if o.grad_format != "e4m3":
        _fail(32, ValueError, "grad_format='e5m2' with pre-quantised E4M3 inputs: the per-(document, kv head) backward "
                              "quantises dO to E4M3.")
    log("prequant", "fa3", "E4M3 q / k / v with (b, h_kv) float32 descales (FA3's q_descale / k_descale / v_descale): "
                           "no Q / K / V quantiser; the kernels read the caller's codes and a per-(batch row or "
                           "document, kv head) score scale / V descale built from the descales on the GPU")


def resolve_fp8(q, k, v, *, softmax_scale=None, causal=False, window_size=(None, None), softcap=0.0,
                score_mod=None, score_mod_bwd=None, mask_mod=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                max_seqlen_k=None, seqused_k=None, deterministic=False, return_lse=False, qk_hadamard=None,
                q_quantized=None, k_quantized=None, q_scale_fused=False,
                options: Optional[FP8AttnOptions] = None, seqused_q=None, prequant_fa3: bool = False) -> AttnPlan:
    o = options if options is not None else FP8AttnOptions()
    if not isinstance(o, FP8AttnOptions):
        raise TypeError(f"flash_attn_fp8: options must be FP8AttnOptions or None; got {type(o).__name__}.")
    for nm, val in (("causal", causal), ("deterministic", deterministic), ("return_lse", return_lse),
                    ("q_scale_fused", q_scale_fused)):
        if not isinstance(val, bool):
            raise ValueError(f"flash_attn_fp8: {nm} must be a bool; got {val!r}.")
    had = _HadamardRequest(qk_hadamard, "fp8")
    softcap = float(softcap or 0.0)
    log = _Log()
    shp = _shape("fp8", q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k, seqused_q,
                 prequant_fa3=prequant_fa3)
    tot_q, tot_k = shp.tot_q, shp.tot_k
    window = _window(window_size)
    grad = shp.requires_grad
    tuning = o.tuning if o.tuning is not None else KernelTuning()
    log("route", "fp8", "E4M3 Q, K, V, dO; the E4M3 backward")
    if prequant_fa3:
        # rule 32 (FA3-style pre-quantised inputs): the call's facts the kernels need, refused here before anything runs
        _check_prequant_fa3(shp, o, window, softcap, score_mod, mask_mod, q_quantized, q_scale_fused, log)
    # ---- rule E13: the head dims the 8-bit kernels serve: d <= 128 in steps of 16. 64 / 128 natively; the others
    # on the token-block kernels padded to the next of 64 / 128 -- the quantisers stage the missing columns
    # as zeros (no padded copy), which add nothing to S / dP and move no block exponent -- forward AND backward (the
    # backward used to serve 64 / 128 only, and the forward ran upstream's per-tensor kernel). 192 / 256 fail in the
    # forward.
    if shp.d > 128 or shp.d % 16 != 0:
        _fail("E13", NotImplementedError, f"flash_attn_fp8 serves head_dim <= 128 (a multiple of 16); got {shp.d}. "
                                          "flash_attn_fp4 serves 256.")
    padded_d = shp.d not in (64, 128)
    if padded_d:
        log("head_dim_padding", 64 if shp.d < 64 else 128,
            f"head_dim {shp.d}: the token-block kernels run it padded to {64 if shp.d < 64 else 128} (zero columns staged "
            "by the quantisers; O / dQ / dK / dV are the first head_dim columns)")

    # ---- rule 4 / rule 32 ----------------------------------------------------------------------------------------------
    if shp.d != 128:
        had.unavailable(f"head_dim {shp.d} (the rotation is 128-point)", 4,
                        msg=f"qk_hadamard is built for head_dim 128 (the fused back-rotation kernels are d = 128); "
                            f"got {shp.d}.")
    if (q_quantized is None) != (k_quantized is None):
        _fail(32, ValueError, "flash_attn_fp8: q_quantized and k_quantized must be supplied together")
    if q_scale_fused and q_quantized is None:
        _fail(32, ValueError, "q_scale_fused=True needs q_quantized (the fold lives in the supplied Q's descale)")
    prequant = q_quantized is not None or prequant_fa3
    if prequant:
        had.unavailable("caller-pre-quantised Q / K (unrotated codes)", 32, ValueError,
                        "pre-quantised Q/K are unrotated codes; qk_hadamard cannot rotate them.")

    # ---- rule 33: no explicit per-tensor dS on the 8-bit entry ---------------------------------------------------------
    _refuse_fp8_per_tensor_ds(o)

    # ---- rule 28: the masking / varlen features --------------------------------------------------------------------------
    feats = _features(window, softcap, score_mod, mask_mod, shp)
    docs = shp.varlen or shp.seqused_k
    _check_features("fp8", "fp8", feats, causal, window, softcap, score_mod, score_mod_bwd, mask_mod, shp, return_lse,
                    grad, prequant and not prequant_fa3, "block_32x32" if o.ds_scaling == "auto" else o.ds_scaling, log)
    # ---- ragged lengths: s % 128 != 0 without varlen runs the batch rows as documents -- repacked at aligned
    # starts on the GPU, token-block operands on both passes, the fork forward -- exactly the seqused_k path. It used to
    # fall back to upstream's forward with per-tensor operands (a global causal channel). An explicit per-tensor /
    # 32 x 32 / token-local operand scaling, forward_kernel='upstream', caller-pre-quantised Q / K and the explicit
    # per-tensor-only options (E5M2 gradients, a per-tensor dS) keep their own path.
    ragged = not docs and (tot_q % 128 != 0 or tot_k % 128 != 0)
    ragged_docs = (ragged and o.operand_scaling in ("auto", "token_block") and o.forward_kernel != "upstream"
                   and not prequant and o.grad_format == "e4m3" and o.ds_scaling in ("auto", "block_32x32", "row_1x32"))
    if ragged_docs:
        docs = True
        log("documents", "ragged", f"s % 128 != 0 (s_q {tot_q}, s_k {tot_k}): the batch rows run as documents repacked "
                                   "at aligned starts (token-block operands, the fork forward; as seqused_k)")
    if not docs and (tot_q % 128 != 0 or tot_k % 128 != 0):
        had.unavailable(f"s % 128 != 0 (s_q {tot_q}, s_k {tot_k}: the rotation runs in the 128-token-tile quantisers)",
                        4, msg=f"qk_hadamard rotates inside the 128-token-tile quantisers: it needs s % 128 == 0; got "
                               f"s_q = {tot_q}, s_k = {tot_k}.")

    # ---- forward kernel ----------------------------------------------------------------------------------------------------
    # (varlen documents are repacked at aligned starts: the fork kernel serves any lengths there; other head dims run
    # it padded on token-block operands, while the per-tensor operands keep it at 64 / 128)
    tb_requested = o.operand_scaling in ("auto", "token_block", "token_local") and not prequant and o.grad_format == "e4m3" \
        and o.ds_scaling in ("auto", "block_32x32", "row_1x32")
    fork_ok = (docs or (tot_q % 128 == 0 and tot_k % 128 == 0)) and (shp.d in (64, 128) or tb_requested)
    if o.forward_kernel == "fork" and not fork_ok:
        _fail("E7", ValueError, f"forward_kernel='fork' needs s % 128 == 0 and head_dim 64 or 128; got s_q = {tot_q}, "
                                f"s_k = {tot_k}, d = {shp.d}. Use forward_kernel='auto'.")
    if o.forward_kernel == "auto":
        fk = log("forward_kernel", "fork" if fork_ok else "upstream",
                 "auto: the fork kernel (s % 128 == 0 or documents; d = 64 / 128, or padded on token-block operands)"
                 if fork_ok else
                 "auto: upstream's forward (the fork kernel needs s % 128 == 0 and token-block operands or d = 64 / 128)")
    else:
        fk = log("forward_kernel", o.forward_kernel, "explicit")

    # ---- softmax value function: rule 21 -----------------------------------------------------------------------------------
    if o.softmax_exp == "mitchell":
        _fail(21, ValueError, "Mitchell P-hat on an E4M3 P.V was removed (the 8-bit P-hat is"
                              " formed with exp2 only); use softmax_exp='exp2'.")
    sm_exp = log("softmax_exp", "exp2", "default")

    # ---- dS scaling and E5M2 --------------------------------------------------------------------------------------------------
    ds = log("ds_scaling", "block_32x32" if o.ds_scaling == "auto" else o.ds_scaling,
             "auto: one UE8M0 per 32 x 32 dS block" if o.ds_scaling == "auto" else "explicit")
    rowq = _resolve_ds_dq_rowlocal("fp8", o.ds_dq_rowlocal, ds, causal, window, mask_mod, log)

    # ---- operand scaling: rule 16 --------------------------------------------------------------------------------------------
    # scaling-consistency: this is the ONE scaling of Q, K and V; the forward's codes are what the backward consumes.
    why = []
    if not docs and (tot_q % 128 != 0 or tot_k % 128 != 0):
        why.append(f"s % 128 != 0 (s_q {tot_q}, s_k {tot_k})")
    if prequant:
        # the caller's own scaling. Pre-quantised Q / K arrive as per-tensor E4M3 codes the caller made; their
        # scale (and so the global causal channel a per-tensor scale opens) is the caller's responsibility, and the
        # kernels pair them with per-tensor V / dO
        why.append("caller-pre-quantised per-tensor Q/K (the caller's own scaling: its per-tensor scale is a global "
                   "causal channel the caller owns; V / dO follow per-tensor)")
    if ds not in ("block_32x32", "row_1x32"):
        why.append(f"ds_scaling='{ds}' (block-scaled operands need a block-scaled dS)")
    if o.grad_format == "e5m2":
        why.append("E5M2 gradients")
    if fk == "upstream":
        why.append("forward_kernel='upstream' (block-scaled operands exist on the fork kernel only)")
    op = o.operand_scaling
    if prequant_fa3:
        # the caller's E4M3 codes and (b, h_kv) descales ARE the operands' scaling: one scale per (batch row or document,
        # kv head) of Q / K / V (FA3's granularity); dO, quantised here, follows per (document, kv head)
        op_res = log("operand_scaling", "per_tensor", "caller-pre-quantised E4M3 Q / K / V with (b, h_kv) descales "
                                                      "(FA3): one scale per (batch row or document, kv head); the "
                                                      "caller owns that scaling (a per-tensor scale is a causal channel "
                                                      "inside its row / document)")
    elif docs and op == "per_tensor" and not why and shp.d in (64, 128):
        # varlen documents with explicit per-tensor operands: one scale per (document, kv head) of every operand
        # (FA3 / Meta granularity), reduced on the GPU over each document's real rows only -- never one scale across
        # documents (lp_varlen.DocScales). Repacked varlen / batched seqused_k: the aligned path on the repacked buffer
        op_res = log("operand_scaling", "per_tensor", "explicit: varlen documents, one scale per (document, kv head) "
                                                      "of Q / K / V / dO from the document's real rows"
                                                      + ("" if shp.aligned_varlen else
                                                         " (the aligned path on the repacked documents)"))
    elif docs and op == "token_local" and not why and shp.d in (64, 128) and not ragged_docs:
        # token-local operands on documents: per-token exponents (no scale spans two tokens, let alone two documents);
        # the fold forward reads its factor tables at each document's offsets (flash_fwd_sm100_fp4 tb_varlen)
        op_res = log("operand_scaling", "token_local", "explicit: varlen / seqused_k documents, per-token exponents "
                                                       "(the fold forward on the documents)")
    elif docs:
        # varlen documents (seqused_k: documents of the batch rows) take token-block operands on both
        # passes; a per-tensor scale would be a channel between documents
        if op in ("per_tensor", "block_32x32", "token_local") or why:
            _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: varlen / seqused_k documents take "
                                  "token-block operands (as do ragged batch rows; a per-tensor scale over all "
                                  "documents would be a channel between documents; token-local operands need head_dim "
                                  "64 / 128 on the fork forward with the block-scaled dS and E4M3 gradients)"
                                  + ("; per-(document, kv head) per-tensor scales run at head_dim 64 / 128 on the "
                                     "fork forward with the block-scaled dS and E4M3 gradients"
                                     if op == "per_tensor" else "")
                                  + (f"; {'; '.join(why)}" if why else "") + ".")
        op_res = log("operand_scaling", "token_block", "varlen / seqused_k / ragged documents: token blocks on both "
                                                       "passes")
    elif op == "auto" and o.grad_format == "e5m2":
        # E5M2 gradients are a per-tensor option. Their two extra exponent bits buy range for ONE scale over
        # all of dO; a token-block dO carries its own exponent per 32 tokens, where E4M3's extra mantissa bit is strictly
        # better. Resolving auto to per-tensor operands opened a global causal channel without the caller asking for it.
        _fail(16, ValueError, "grad_format='e5m2' is a per-tensor option (the extra exponent range serves one scale over "
                              "all of dO; a token-block dO has an exponent per 32 tokens and E4M3's mantissa is more "
                              "precise there). Its per-tensor Q / K / V / dO scales are a global causal channel (a "
                              "future token's max moves every earlier token's rounding): pass "
                              "operand_scaling='per_tensor' to accept that explicitly, or keep grad_format='e4m3' (the "
                              "default) for token-block operands.")
    elif op == "auto":
        # (s_q != s_k takes token blocks too: see resolve_fp4)
        if prequant:
            op_res = log("operand_scaling", "per_tensor", "auto: " + "; ".join(why))
        elif why:
            _refuse_auto_per_tensor("fp8", why)
        else:
            op_res = log("operand_scaling", "token_block", "auto: 32 tokens x d UE8M0 (the certified blk32x128)"
                         + _token_block_shape_note(shp))
    else:
        bad = why
        if op != "per_tensor" and bad:
            _fail(16, ValueError, f"operand_scaling='{op}' cannot serve this call: {'; '.join(bad)}. Pass "
                                  "operand_scaling='per_tensor' explicitly to run it on per-tensor scales.")
        op_res = log("operand_scaling", op, "explicit")
    if op_res == "token_local":
        had.unavailable("operand_scaling='token_local' (the rotated token-local quantiser is not built)", 16, ValueError,
                        "operand_scaling='token_local' with qk_hadamard: the rotated token-local quantiser is not built.")
    if padded_d and op_res == "per_tensor" and shp.d not in (32, 96):
        # measured: upstream's per-tensor forward fails to build at 48 / 80 / 112
        # ("failed to partition (128, d) with tiled_mma") and gives O cosine 0.97 at 16
        _fail("E13", NotImplementedError, f"the 8-bit per-tensor forward (upstream's kernel) serves head_dim 32, 64, 96 "
                                          f"and 128; got {shp.d} (token-block operands serve it padded: "
                                          "operand_scaling='auto').")
    if padded_d and op_res == "per_tensor" and grad:
        _fail("E13", NotImplementedError, f"the 8-bit backward on per-tensor operands serves head_dim 64 and 128; got "
                                          f"{shp.d} (token-block operands serve it padded: operand_scaling='auto').")
    _tuning_pending(tuning, "fp8", op_res, had.on, bool(feats))
    _check_token_local(op_res, rowq, "e4m3", had.on, window, softcap, score_mod, mask_mod, shp, log, causal)
    pts = _resolve_per_tensor_scale(o.per_tensor_scale, op_res, log)

    # ---- 2-CTA: rules 17, 19, 20, 15 (rule 14 retired) -------------------------------------------------------------
    if op_res == "block_32x32" and o.two_cta == "on":
        _fail(17, ValueError, "block_32x32 operands are 1-CTA only; use operand_scaling='token_block'.")
    if ds in _FP8_DS_1CTA_ONLY and o.two_cta == "on":
        _fail(19, ValueError, f"ds_scaling='{ds}' exists on the 1-CTA kernel only.")
    if o.grad_format == "e5m2" and o.two_cta == "on":
        _fail(20, ValueError, "The 2-CTA backward types every FP8 GEMM with dO's element type; E5M2 gradients run on "
                              "the 1-CTA kernel.")
    two = _resolve_two_cta("fp8", o.two_cta, shp, deterministic, op_res,
                           ((ds in _FP8_DS_1CTA_ONLY, f"ds_scaling='{ds}' is 1-CTA only"),
                            (o.grad_format == "e5m2", "E5M2 gradients run on the 1-CTA kernel"),
                            (docs and op_res == "per_tensor",
                             "per-(document, kv head) per-tensor scales run on the 1-CTA kernel"),
                            (prequant_fa3, "pre-quantised E4M3: the per-(batch row / document, kv head) scales run on "
                                           "the 1-CTA kernel"),
                            (shp.aligned_varlen and (mask_mod is not None or score_mod is not None or softcap),
                             "score_mod / mask_mod / softcap on documents packed at 128 run on the 1-CTA kernel (the "
                             "2-CTA key pair straddles documents there)"),
                            (deterministic and (window is not None or mask_mod is not None or softcap
                                                or score_mod is not None),
                             "deterministic with a window / mask_mod / softcap / score_mod orders dQ on the 1-CTA kernel "
                             "only")), causal, log, gqa_ok=_two_cta_gqa_ok(tuning.bwd_2cta_gqa, o.two_cta, shp))
    if ds in _FP8_DS_2CTA_ONLY and two != "on":
        _fail(19, ValueError, f"ds_scaling='{ds}' exists on the 2-CTA kernel only.")
    _check_aligned("fp8", shp, two if grad else "off", tuning, had.on, log, causal=causal)
    consistency, pend = _consistency("fp8", [("Q/K (S GEMM)", f"e4m3 {op_res}", f"e4m3 {op_res}"),
                                             ("V", f"e4m3 {op_res}", f"e4m3 {op_res}")], grad, log)
    # SCALING-CONSISTENCY RULE: the rotated codes are the ONE quantisation of Q / K both passes read
    qk_hadamard = had.log(log, f"Q R / K R (R = diag(signs) H_128 / sqrt(128), seed 4321) quantised in one rotated read "
                          f"per tensor; the E4M3 {op_res} scales come from the rotated rows; forward and backward read "
                          "the same rotated codes; dQ / dK rotated back in the backward's epilogues")
    if softmax_scale is None:
        softmax_scale = log("softmax_scale", 1.0 / math.sqrt(shp.d), "default 1/sqrt(head_dim)")
    hd_path = "single_pass" if shp.d in (64, 128) else ("zero_pad_64" if shp.d < 64 else "zero_pad_128")
    row_p = _resolve_p_hat_frame("fp8", o.p_hat_frame, "fp8", op_res, tuning, log, fork=fk == "fork")
    fwd = FwdSpec(kernel="fork_fp8" if fk == "fork" else "upstream_fp8", qk_format="e4m3", pv_operand="e4m3",
                  operand_scaling=op_res, softmax_exp=sm_exp, causal_safety="none",
                  p_frame="n/a", q_block_2d=False, qk_hadamard=qk_hadamard, v_hadamard=False, causal=causal,
                  window=window, softcap=softcap, score_mod=score_mod is not None, mask_mod=mask_mod is not None,
                  varlen=shp.varlen, seqused_k=shp.seqused_k, return_lse=return_lse, per_tensor_scale=pts,
                  aligned_varlen=shp.aligned_varlen,
                  two_cta=_resolve_fwd_two_cta(o.forward_two_cta, shp, causal, fk == "fork", bool(_mods(feats)),
                                               prequant and not prequant_fa3, sm_exp == "mitchell", log,
                                               docs=bool(docs)),
                  ragged_documents=ragged_docs)
    _check_kernel_tuning(tuning, "none" if not grad else ("2cta" if two == "on" else "1cta"), ds, o.p_scaling)
    _check_kernel_asserts(tuning, "fp8", op_res, causal, "none" if not grad else ("2cta" if two == "on" else "1cta"),
                          row_p, fork=fk == "fork")
    bwd = BwdSpec(kernel=("none" if not grad else ("2cta" if two == "on" else "1cta")), head_dim_path=hd_path,
                  operand_scaling=op_res, ds_scaling=ds, ds_dq_rowlocal=rowq, p_scaling=o.p_scaling,
                  grad_format=o.grad_format,
                  deterministic=deterministic, sr_ds=False, sr_dk_operand=False, sr_dv_operand=False,
                  fp4_backward_dkdv="n/a", dq_operand="e4m3", all_fp4_stabiliser="n/a", fp4_ds_multiplier="n/a",
                  k_channel="n/a", per_tensor_ds_source="n/a", save_bf16_inputs=False,
                  softmax_exp=sm_exp)
    return AttnPlan(entry="fp8", route="fp8", softmax_scale=float(softmax_scale), shape=shp, fwd=fwd, bwd=bwd,
                    tuning=tuning, prequant=prequant, q_scale_fused=q_scale_fused, consistency=consistency,
                    pending=pend, resolution_log=tuple(log.rows), options=o,
                    specs=build_specs("fp8", fwd, bwd, tuning, shp, fwd8_row_p=row_p))

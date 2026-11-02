"""Option objects of the low-precision attention entry points.

``FP4AttnOptions`` and ``FP8AttnOptions`` carry the advanced arguments of ``flash_attn_fp4`` / ``flash_attn_fp8``;
``KernelTuning`` carries the measured performance constants (importable, not documented as API, no stability
promise). All three are frozen dataclasses: an options object is a value, it is hashed into the resolved plan, and
it is validated when it is built, so an illegal value never reaches plan resolution.

Every field defaults to today's certified behaviour, so ``options=None`` means "the certified configuration".
``"auto"`` fields are resolved once per call, from the call's shapes, by ``lowprec.plan``; the plan records each
decision and its reason.

Nothing in this module reads the environment or any global configuration.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

__all__ = ["FP4AttnOptions", "FP8AttnOptions", "KernelTuning", "TUNING_FIELDS"]


def _check_choice(owner: str, name: str, value: Any, allowed: Tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{owner}.{name} must be one of {', '.join(repr(a) for a in allowed)}; got {value!r}.")


def _check_bool(owner: str, name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{owner}.{name} must be a bool; got {value!r}.")


# ----------------------------------------------------------------------------------------------------------------
# FP4AttnOptions
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FP4AttnOptions:
    """Advanced options of ``flash_attn_fp4``. Every default reproduces the certified 4-bit configuration.

    The 4-bit operands are NVFP4 (E2M1 with an E4M3 scale per 16 along d); the MXFP4 formats are not part of this API.
    operand_scaling: "auto" | "token_block" | "block_32x32" | "per_tensor". Default "auto". The scaling of the E4M3
        operand copies: V on pv_precision="fp8", and the backward's E4M3 Q/K/V/dO. "token_block" = one UE8M0
        exponent per 32 tokens x the whole head dim (closes the last global causal channels, 2-CTA capable);
        "block_32x32" = 32 tokens x 32 d (1-CTA only); "per_tensor" = one GPU-computed scalar per tensor (fastest,
        keeps a global causal channel). "auto" resolves to "token_block" -- head dims 64 / 128, s_q == s_k or not,
        varlen / seqused_k documents -- and never falls back to "per_tensor" silently (a per-tensor scale is a global
        causal channel): a call token blocks cannot serve (head_dim 256, p_scaling="block_1x32") raises and names the
        explicit opt-in operand_scaling="per_tensor". The all-4-bit route's hybrid E4M3 copies are
        per-tensor; varlen / seqused_k documents and option A (pv_precision="fp4", backward="e4m3": Q / K / dO copies)
        are token blocks only. An explicit value the call cannot serve raises.
        Forward and backward always share one operand scaling (scaling-consistency rule).
        "token_local" = every token of the E4M3 copies carries its OWN power-of-two exponent (1 x d) and no token's
        codes or scale depend on any other token: closes the 32-token-block causal channel exactly (measured 0 on
        every probe; docs/lowprec.md, "Causal safety"). Needs ds_dq_rowlocal="all", the token-block eligibility, no qk_hadamard and
        no window / softcap / score_mod / mask_mod / varlen; runs the fold forward and the block-scaled dV P-hat.
        No rotated token-local quantiser exists: qk_hadamard=None turns the rotation off (a resolution-log line says
        so), qk_hadamard=True raises. The other operand scalings are not causally exact: "token_block" /
        "block_32x32" keep the in-32-token-block channel of the shared exponent (sub-ULP to ~1e-3, < 32 rows),
        "per_tensor" the global one unless its amax is pinned.
    per_tensor_scale: "exact" | "power_of_two". Default "exact". Per-tensor operand scales (operand_scaling
        resolving to "per_tensor"): amax / 448, or that rounded UP to a power of two. A power of two makes a
        later token's change of the tensor amax invisible to every earlier token's codes unless it crosses a
        power-of-two boundary: a 20-60x smaller causal channel at no cost, but not zero (crossings leak O 1e-3..1.4e-2).
        Refused when the operands are not per-tensor.
    ds_scaling: "auto" | "block_32x32". Default "auto". The scaling of the E4M3 score gradient dS-hat. "auto" =
        "block_32x32" (the largest training effect measured: -0.024 / -0.031 val against the per-tensor dS), except
        at head_dim 256, where the three-pass kernels resolve it to a per-tensor dS; the all-4-bit kernel keeps a
        per-tensor dS on the diagonal route's r_k pairs by itself. An explicit "per_tensor" is not offered (as on
        flash_attn_fp8: a per-tensor dS is a global causal channel and trains worse) and is refused (rule 33).
    ds_dq_rowlocal: "off" | "diagonal" | "all". Default "off". backward="e4m3" with the block_32x32 dS only. The block
        dS-hat shares one UE8M0 exponent per 32 keys x 32 queries between the dQ and the dK GEMM, so a later query's dS
        sets the exponent that rounds an earlier query's dS: a causal channel from later keys / dO / Q into earlier
        queries' dQ. "diagonal" gives the dQ side one exponent per query row (over the block's 32 keys) and its own
        E4M3 codes on the 32 x 32 blocks the causal diagonal crosses (needs causal, no window / mask_mod); "all" on
        every block; the dK side keeps the block exponent (exact dK legitimately depends on later queries). dS is
        backward-only, so the scaling-consistency rule is unaffected. Measured: "diagonal" does NOT close the channel
        (it runs along the block's query axis in every block); "all" closes it exactly at +21-38 % backward kernel
        time, training-neutral (docs/lowprec.md, "Causal safety").
    fp4_ds_multiplier: "auto" | "global" | "global_row_boost" | "row_local". Default "auto". backward="fp4" only.
        How the query-row backward sizes the multiplier under its 4-bit dS: "auto" = "row_local" when causal (closes
        a 2.0e-01..2.6e-01 dQ leak), "global_row_boost" when dense (A4 dQ 0.701 -> 0.946).
    p_scaling: "row" | "block_1x32". Default "row". The E4M3 P-hat of the backward's dV GEMM: P * 2^8 with no scale
        factor, or one UE8M0 per key row and 32 queries (dV cosine 0.9977 -> 0.9985 at A4, +1-4 % backward, 1-CTA
        only on the 4-bit route).
    p_frame: "two_level" | "per_tile" | "single_level". Default "two_level". pv_precision="fp4", backward="e4m3"
        only: how the forward's 4-bit P-hat is framed (SageAttention3 two-level; the pre-SageAttention single level).
        "per_tile" (SageAttention3's per-tile form) is not defined on the integer-frame all-FP4 step option A runs and
        is refused (rule 23).
    p_hat_frame: "auto" | "row" | "group". Default "auto" = "group" (flash_attn_fp4: the row frame is opt-in on the
        4-bit entry, where it carried a small consistent training cost, +0.0015 val loss, 4 / 4 seeds). The E4M3 P-hat of the token-block forward (operand_scaling "token_block" /
        "block_32x32" on pv_precision="fp8": NVFP4 Q.K, MXFP8 P.V): "row" = the row-frame E4M3 P-hat of the per-tensor
        step with a constant P scale tile (only V's token-block factors in the P.V GEMM); "group" = one UE8M0 scale per
        32 keys of P-hat, computed in the softmax. Both are exact-softmax forwards; "row" changes P-hat's rounding. An
        explicit value where the forward has no such choice (per-tensor or token-local operands, the 4-bit P.V routes,
        KernelTuning.token_block_forward='fold') is refused (rule E14).
    softmax_exp: "auto" | "exp2" | "mitchell". Default "auto". How P-hat = 2^x is formed. "auto" = "mitchell"
        (Direct-P) where P-hat is 4-bit, "exp2" where it is E4M3. "mitchell" with an E4M3 P.V is refused.
    k_channel: "fold" | "cut" | "all" | "open". Default "fold". causal_safety="diagonal" only: how dQ's K operand is
        kept from reading future keys' scales ("fold" certified at 2 seeds; "cut" MHA only; "open" leaves it open).
    q_block_2d: bool. Default False. NVFP4 Q with one scale per 16 queries x 16 d. Worse at every cell and leaks
        Q->O 0.27 under a mask; refused with any mask and with qk_hadamard.
    v_hadamard: bool. Default False. Hadamard on V, all-4-bit key-local route only. No recorded win.
    all_fp4_stabiliser: "sr" | "rtn" | "eden_rht". Default "sr". backward="fp4": SR of dO-hat and the 4-bit dS;
        round-to-nearest; or a joint head-dim Hadamard of Q, K, V, dO with an MS-EDEN dO (dense only).
    sr_ds: bool. Default False. backward="e4m3": stochastic rounding of the E4M3 dS-hat (off until re-measured
        under the block-32x32 dS). Built in the 1-CTA kernel (per-tensor and block-scaled dS-hat; under
        ds_dq_rowlocal="all" the dQ side's own codes stay round-to-nearest): it takes the call to the 1-CTA backward.
    sr_dk_operand: bool. Default True. backward="fp4": SR of the dK operand's E4M3 dS-hat.
    sr_dv_operand: bool. Default False. backward="fp4": SR of the dV operand's P-hat (measured neutral).
    fp4_backward_dkdv: "e4m3" | "fp4". Default "e4m3". backward="fp4": the dK/dV operands of the query-row
        backward; "fp4" makes every backward GEMM 4-bit (gradients fall below the gate).
    dq_operand: "fp4" | "e4m3". Default "fp4". backward="fp4": the dQ GEMM's dS operand.
    two_cta: "auto" | "on" | "off". Default "auto". Whether the backward may use the 2-CTA cluster kernel. "auto":
        from key length 32768 (s_k batched, max_seqlen_k on varlen) wherever supported (d = 128; MHA and GQA; batched
        s_q != s_k and s_k % 256 == 128; documents), deterministic calls excepted; below it (4-bit) token-block
        operands, or s_k >= 16384 / dense s_k >= 8192, when MHA, d = 128, s_q == s_k, s % 256 == 0, not
        deterministic. "on" on an ineligible call raises.
    forward_two_cta: "auto" | "on" | "off". Default "auto". Whether the forward runs on CTA pairs (cta_group::2 MMAs,
        each CTA loading half of every K / V tile; O and LSE bitwise the 1-CTA forward's). "auto" follows the measured
        rule in lowprec.plan (_resolve_fwd_two_cta); "on" on an ineligible call raises.
    seqused_k_group_aligned: bool. Default False. Was the caller's statement that every seqused_k value is a
        multiple of V's 4-bit scale group, for seqused_k with a 4-bit P.V; no 4-bit P.V route takes seqused_k (their
        V scale is key-local for causal only), so True is refused.
    save_bf16_inputs: bool. Default True. False keeps only the quantised codes for the backward (1560 -> 792 bytes
        per (token, head) at d = 128, bitwise); pv_precision="fp8", backward="e4m3" at per-tensor operands.
    p_consistency: "reproduce_forward_p" | "none". Default "reproduce_forward_p". pv_precision="fp4", backward="e4m3"
        (option A) only: the backward's P in dS = P (dP - D) and in dV is the forward's dequantised 4-bit P-hat / Z bit for
        bit (the forward exports its frames and row sum; the row sum is the codes' sum), so D = dO . O from the forward's
        O is exact for it and no extra pass runs. "none" = the exact-P backward with D from the forward's O (inconsistent:
        trains to NaN; the experiment arm). softmax_exp is the forward's value function and the backward follows it.
    do_rounding: "rtn" | "sr". Default "rtn". pv_precision="fp4", backward="e4m3" (option A) only: the rounding of the
        4-bit dO of the backward's dP GEMM. RTN: training does not distinguish the two, RTN has the better one-step dQ
        (0.980 vs 0.973) and the cheaper quantiser. Option A is inference only (plan rule 36),
        so no call reads it: "sr" raises (rule E1) rather than being ignored.
    exp_mix: "auto" | "none" | 8 characters over M / E / 2 / 3. Default "auto" (option A inference: "EMMEMMEM"; the
        all-4-bit route: "none"). pv_precision="fp4", backward="e4m3", p_consistency=
        "reproduce_forward_p", softmax_exp Mitchell only: the value function of each 16-key group position of the 4-bit
        P-hat (M the clamped Mitchell classifier, E ex2.approx on the MUFU pipe, 2 / 3 the clamped degree-2 / -3 FMA-pipe
        exp2); a compile-time function of the position, which the backward reproduces bit for bit. "none" = Mitchell
        everywhere.
    tuning: KernelTuning | None. Measured performance constants; not API.
    """

    operand_scaling: Literal["auto", "token_block", "block_32x32", "per_tensor", "token_local"] = "auto"
    per_tensor_scale: Literal["exact", "power_of_two"] = "exact"
    ds_scaling: Literal["auto", "block_32x32", "per_tensor"] = "auto"  # "per_tensor": refused (rule 33)
    ds_dq_rowlocal: Literal["off", "diagonal", "all"] = "off"
    fp4_ds_multiplier: Literal["auto", "global", "global_row_boost", "row_local"] = "auto"
    p_scaling: Literal["row", "block_1x32"] = "row"
    p_frame: Literal["two_level", "per_tile", "single_level"] = "two_level"   # "per_tile": refused (rule 23)
    p_hat_frame: Literal["auto", "row", "group"] = "auto"
    softmax_exp: Literal["auto", "exp2", "mitchell"] = "auto"
    k_channel: Literal["fold", "cut", "all", "open"] = "fold"
    q_block_2d: bool = False
    v_hadamard: bool = False
    all_fp4_stabiliser: Literal["sr", "rtn", "eden_rht"] = "sr"
    sr_ds: bool = False
    sr_dk_operand: bool = True
    sr_dv_operand: bool = False
    fp4_backward_dkdv: Literal["e4m3", "fp4"] = "e4m3"
    dq_operand: Literal["fp4", "e4m3"] = "fp4"
    two_cta: Literal["auto", "on", "off"] = "auto"
    forward_two_cta: Literal["auto", "on", "off"] = "auto"
    seqused_k_group_aligned: bool = False
    save_bf16_inputs: bool = True
    p_consistency: Literal["reproduce_forward_p", "none"] = "reproduce_forward_p"
    do_rounding: Literal["rtn", "sr"] = "rtn"
    exp_mix: str = "auto"
    tuning: Optional["KernelTuning"] = None

    _CHOICES = {
        "operand_scaling": ("auto", "token_block", "block_32x32", "per_tensor", "token_local"),
        "per_tensor_scale": ("exact", "power_of_two"),
        "ds_scaling": ("auto", "block_32x32", "per_tensor"),
        "ds_dq_rowlocal": ("off", "diagonal", "all"),
        "fp4_ds_multiplier": ("auto", "global", "global_row_boost", "row_local"),
        "p_scaling": ("row", "block_1x32"),
        "p_frame": ("two_level", "per_tile", "single_level"),
        "p_hat_frame": ("auto", "row", "group"),
        "softmax_exp": ("auto", "exp2", "mitchell"),
        "k_channel": ("fold", "cut", "all", "open"),
        "all_fp4_stabiliser": ("sr", "rtn", "eden_rht"),
        "fp4_backward_dkdv": ("e4m3", "fp4"),
        "dq_operand": ("fp4", "e4m3"),
        "two_cta": ("auto", "on", "off"),
        "p_consistency": ("reproduce_forward_p", "none"),
        "do_rounding": ("rtn", "sr"),
        "forward_two_cta": ("auto", "on", "off"),
    }
    _BOOLS = ("q_block_2d", "v_hadamard", "sr_ds", "sr_dk_operand", "sr_dv_operand", "seqused_k_group_aligned",
              "save_bf16_inputs")

    def __post_init__(self):
        for n, allowed in self._CHOICES.items():
            _check_choice("FP4AttnOptions", n, getattr(self, n), allowed)
        for n in self._BOOLS:
            _check_bool("FP4AttnOptions", n, getattr(self, n))
        if self.tuning is not None and not isinstance(self.tuning, KernelTuning):
            raise ValueError(f"FP4AttnOptions.tuning must be a KernelTuning or None; got {type(self.tuning).__name__}.")



# ----------------------------------------------------------------------------------------------------------------
# FP8AttnOptions
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FP8AttnOptions:
    """Advanced options of ``flash_attn_fp8``. Every default reproduces the certified 8-bit configuration.

    operand_scaling: "auto" | "token_block" | "block_32x32" | "per_tensor". Default "auto". The scaling of the E4M3
        Q/K/V (shared by forward and backward) and of dO. "auto" = "token_block" on every head dim the entry serves
        (64 / 128 natively, 16..112 padded to 64 / 128 by the quantisers), s_q != s_k, varlen / seqused_k documents and
        ragged lengths (s % 128 != 0: the batch rows run as documents). It never falls back to "per_tensor" silently
        (a per-tensor scale is a global causal channel): a call token blocks cannot serve (a per-tensor / row-local / tile dS,
        E5M2 gradients, forward_kernel="upstream") raises and names the explicit opt-in operand_scaling="per_tensor".
        Pre-quantised Q / K (q_quantized / k_quantized) are the caller's own per-tensor scaling -- the global causal
        channel of their scale is the caller's responsibility -- and run per-tensor V / dO with them. An explicit
        value the call cannot serve raises.
        "token_local" = every token of the E4M3 copies carries its OWN power-of-two exponent (1 x d) and no token's
        codes or scale depend on any other token: closes the 32-token-block causal channel exactly (measured 0 on
        every probe; docs/lowprec.md, "Causal safety"). Needs ds_dq_rowlocal="all", the token-block eligibility, no qk_hadamard and
        no window / softcap / score_mod / mask_mod / varlen; runs the fold forward and the block-scaled dV P-hat.
        No rotated token-local quantiser exists: qk_hadamard=None turns the rotation off (a resolution-log line says
        so), qk_hadamard=True raises. The other operand scalings are not causally exact: "token_block" /
        "block_32x32" keep the in-32-token-block channel of the shared exponent (sub-ULP to ~1e-3, < 32 rows),
        "per_tensor" the global one unless its amax is pinned.
    per_tensor_scale: "exact" | "power_of_two". Default "exact". As FP4AttnOptions.per_tensor_scale.
    ds_scaling: "auto" | "block_32x32" | "row_local" | "row_1x32" | "block_32x64" | "tile_128". Default "auto"
        (= "block_32x32"). "row_local" / "row_1x32" are exactly leak-free (1-CTA only, 1.4-1.5x backward);
        "block_32x64" / "tile_128" are 2-CTA only, slower, not convergence-tested. "per_tensor" is not offered and is
        refused (rule 33): its exact amax costs an extra full pass (403-633 % of the
        backward kernel), the forward's P_max bound is 4.7x loose (+0.029 val), and either way it is a global causal
        channel (0.11-0.16). Block-scaled dS is the only explicit 8-bit dS.
    ds_dq_rowlocal: "off" | "diagonal" | "all". Default "off". ds_scaling="block_32x32" only (1-CTA and 2-CTA). The block
        dS-hat shares one UE8M0 exponent per 32 keys x 32 queries between the dQ and the dK GEMM, so a later query's dS
        sets the exponent that rounds an earlier query's dS: a causal channel from later keys / dO / Q into earlier
        queries' dQ. "diagonal" gives the dQ side one exponent per query row (over the block's 32 keys) and its own
        E4M3 codes on the 32 x 32 blocks the causal diagonal crosses (needs causal, no window / mask_mod); "all" on
        every block; the dK side keeps the block exponent (exact dK legitimately depends on later queries). dS is
        backward-only, so the scaling-consistency rule is unaffected. Measured: "diagonal" does NOT close the channel
        (it runs along the block's query axis in every block); "all" closes it exactly at +21-38 % backward kernel
        time, training-neutral (docs/lowprec.md, "Causal safety").
    per_tensor_ds_source: None. Removed with the per-tensor dS: "recompute" (the exact
        amax: an extra full pass, 403-633 % of the backward kernel) and "forward_bound" (the forward's P_max bound:
        4.7x loose, +0.029 val) are refused by name (rule 33).
    p_scaling: "row" | "block_1x32". Default "row". The E4M3 P-hat of the dV GEMM.
    grad_format: "e4m3" | "e5m2". Default "e4m3". Element type of dO and dS (E4M3 +0.0021 val vs E5M2 +0.0196).
        E5M2 runs on the 1-CTA kernel only, on per-tensor operands: its extra exponent range serves one scale over all
        of dO (a token-block dO has an exponent per 32 tokens, where E4M3 is more precise), so it needs the explicit
        operand_scaling="per_tensor".
    softmax_exp: "exp2". Default "exp2". The E4M3 P-hat is formed with exp2 (MUFU + FMA-pipe emulation);
        "mitchell" was removed and is refused (rule 21).
    forward_kernel: "auto" | "fork" | "upstream". Default "auto": the fork kernel on token-block operands (any served
        head dim and length); upstream's only for per-tensor operands it cannot serve (s % 128 != 0 or
        d not 64 / 128). "upstream" is a per-tensor kernel: it needs the explicit operand_scaling="per_tensor".
    two_cta: "auto" | "on" | "off". Default "auto" (= always when eligible on 8-bit; from key length 32768 also GQA,
        documents and the batched tails, deterministic calls excepted).
    forward_two_cta: "auto" | "on" | "off". Default "auto". The forward on CTA pairs, as FP4AttnOptions.forward_two_cta
        (the fork forward only).
    p_hat_frame: "auto" | "row" | "group". Default "auto" = "row" (the row frame is the 8-bit default: -17 / -21 % of
        the token-block forward kernel at A1 / A2, training within seed noise). The E4M3 P-hat of the token-block forward (operand_scaling "token_block" / "block_32x32": MXFP8 P.V):
        "row" = the per-tensor step's row-frame E4M3 P-hat with a constant P scale tile (and, without masks / documents,
        the plain E4M3 S GEMM with the token-block factors folded into the softmax); "group" = one UE8M0 scale per 32
        keys of P-hat, computed in the softmax. An explicit value where the forward has no such choice (per-tensor or
        token-local operands, KernelTuning.token_block_forward='fold') is refused (rule E14).
    tuning: KernelTuning | None. Measured performance constants; not API.
    """

    operand_scaling: Literal["auto", "token_block", "block_32x32", "per_tensor", "token_local"] = "auto"
    per_tensor_scale: Literal["exact", "power_of_two"] = "exact"
    ds_scaling: Literal["auto", "block_32x32", "per_tensor", "row_local", "row_1x32", "block_32x64", "tile_128"] = "auto"
    ds_dq_rowlocal: Literal["off", "diagonal", "all"] = "off"
    per_tensor_ds_source: Optional[Literal["recompute", "forward_bound"]] = None  # removed: any value is refused
    p_scaling: Literal["row", "block_1x32"] = "row"
    grad_format: Literal["e4m3", "e5m2"] = "e4m3"
    softmax_exp: Literal["exp2", "mitchell"] = "exp2"
    forward_kernel: Literal["auto", "fork", "upstream"] = "auto"
    two_cta: Literal["auto", "on", "off"] = "auto"
    forward_two_cta: Literal["auto", "on", "off"] = "auto"
    p_hat_frame: Literal["auto", "row", "group"] = "auto"
    tuning: Optional["KernelTuning"] = None

    _CHOICES = {
        "operand_scaling": ("auto", "token_block", "block_32x32", "per_tensor", "token_local"),
        "per_tensor_scale": ("exact", "power_of_two"),
        "ds_scaling": ("auto", "block_32x32", "per_tensor", "row_local", "row_1x32", "block_32x64", "tile_128"),
        "ds_dq_rowlocal": ("off", "diagonal", "all"),
        "per_tensor_ds_source": (None, "recompute", "forward_bound"),
        "p_scaling": ("row", "block_1x32"),
        "grad_format": ("e4m3", "e5m2"),
        "softmax_exp": ("exp2", "mitchell"),
        "forward_kernel": ("auto", "fork", "upstream"),
        "two_cta": ("auto", "on", "off"),
        "forward_two_cta": ("auto", "on", "off"),
        "p_hat_frame": ("auto", "row", "group"),
    }

    def __post_init__(self):
        for n, allowed in self._CHOICES.items():
            _check_choice("FP8AttnOptions", n, getattr(self, n), allowed)
        if self.tuning is not None and not isinstance(self.tuning, KernelTuning):
            raise ValueError(f"FP8AttnOptions.tuning must be a KernelTuning or None; got {type(self.tuning).__name__}.")


# ----------------------------------------------------------------------------------------------------------------
# KernelTuning
# ----------------------------------------------------------------------------------------------------------------
# One row per measured performance constant: field -> value kind. None = "the measured rule". Each field is copied
# into the kernel / host spec fields that read it (lowprec.specs.KT_TO_SPEC, apply_kernel_tuning); the spec is the
# compile key, so every field is in the compile key by construction.
#
# Value kinds:
#   "bool"            True / False
#   ("int", lo, hi)   an integer in [lo, hi] (hi None = unbounded)
#   ("ints", values)  an integer from a fixed set
#   ("float", lo, hi) a float in [lo, hi] (either bound None = open)
#   ("enum", values)  a string from a fixed set
#   ("re", pattern)   a structured string (register budgets)

_POW2_64 = (1, 2, 4, 8, 16, 32, 64)

TUNING_FIELDS = {
    # ---- forward -------------------------------------------------------------------------------------------------
    "fp4_q_stage": ("ints", (1, 2)),
    "qk_split_pipe": "bool",
    "qk_split_causal": "bool",
    "qk_split_seqused": "bool",
    "fp8_generic_e2e": ("int", 0, None),
    "fp8_e2e8_min_sk": ("int", 0, None),
    "fp8_e2e8_max_sk": ("int", 0, None),
    "fwd_mask_loops": ("ints", (0, 1, 2)),
    "fwd_softmax_fused": "bool",
    "fwd_max_tmem": "bool",
    "fwd_o_regstore": "bool",
    "fwd_o_regstore_min_sk": ("int", 0, None),
    "fwd_causal_persistent": ("enum", ("auto", "0", "1")),
    "fwd_causal_persistent_max_sk": ("int", 0, None),
    "fwd_nc_persistent": "bool",
    "q_dbuf": "bool",
    "corr_empty_early": "bool",
    "corr_rescale_batch": "bool",
    "qk_split_bprefetch": "bool",
    "qk_split_late_p0": "bool",
    "split_lean": "bool",
    "mxfp4_split": "bool",
    "mxfp4_directp": ("enum", ("auto", "0", "1")),
    "fp8_pv_use_explicit_pack": "bool",
    "fp8_pv_zero_fill_regs": "bool",
    "fp8_pv_kv_stage_cap": ("int", 1, None),
    "fp8_pv_p_split_num": ("int", 1, None),
    "fp8_pv_p_split_den": ("int", 1, None),
    "fp4_pv_p_split_num": ("int", 1, None),
    "fp4_pv_p_split_den": ("int", 1, None),
    "fp4_pv_tmem_store_rep": ("ints", _POW2_64),
    "fp8_pv_tmem_store_rep": ("ints", _POW2_64),
    "e2e_freq": ("int", 0, None),
    "e2e_start_frg": ("int", 0, None),
    "force_e2e": "bool",
    "mxfp8_use_inline_ptx": "bool",
    "ldred_rowmax": "bool",
    "fp4_group_max_first": "bool",
    "fp4_pv_log2_quant": "bool",
    "fp4_exp2_emulation_every": ("int", 0, None),
    "fp8_pv_p_log2_offset": ("float", 0.0, math.log2(448)),
    "fp8_pv_rescale_threshold": ("float", 0.0, math.log2(448)),
    "fp4_pv_tile_frame_drop": ("float", 1e-9, 80.0),
    "token_block_forward": ("enum", ("fold", "mx")),
    "varlen_fwd_list": "bool",
    "varlen_fwd_tma_o": "bool",
    "varlen_fwd_stage_skip": "bool",
    "varlen_l2_sections": "bool",
    "varlen_fwd_sched": ("enum", ("auto", "static", "atomic", "clc")),
    "varlen_fwd_sched_stages": ("int", 1, 4),
    "varlen_fwd_sched_jit": "bool",
    "varlen_fwd_sched_early": "bool",
    # fwdgap: opt-in forward experiments. fwd_ptxas: extra ptxas options of the forward kernel's compile;
    # fwd_exp: comma-separated experimental forward switches (FwdKernelSpec.exp). Both default to the certified build.
    "fwd_ptxas": ("re", r"[-A-Za-z0-9_=., ]{0,160}"),
    "fwd_exp": ("re", r"[A-Za-z0-9_=.,]{0,160}"),
    # ---- 1-CTA backward -------------------------------------------------------------------------------------------
    "bwd_fp4_ds_fold": "bool",
    "bwd_dq_f16": "bool",
    "bwd_fp4_ld_rep": ("ints", (32, 64)),
    "bwd_fp4_sload_early": "bool",
    "bwd_fp4_warp_sync": "bool",
    "bwd_fp4_causal_diag": "bool",
    "bwd_fp4_dp_ld2": "bool",
    "bwd_fp4_do_stage": ("ints", (1, 2)),
    "bwd_fp4_q_stage": ("ints", (2, 3)),
    "bwd_fp4_q_dyn_issue": "bool",
    "bwd_fp4_stat_shfl": ("enum", ("off", "lse", "dpsum", "both")),
    "bwd_fp4_reduce_chunked": "bool",
    "bwd_fp4_dq_ncol": ("ints", (8, 16, 32, 64, 128)),
    "bwd_fp4_dq_stages": ("int", 1, 8),
    "bwd_fp4_dq_post": ("enum", ("cute", "torch")),
    "varlen_bwd_list": "bool",
    "bwd_fp4_det_gqa_rot": "bool",
    # ---- 2-CTA backward -------------------------------------------------------------------------------------------
    "two_cta8_warp_sync": "bool",
    "two_cta8_mx_xw": ("enum", ("", "dq", "all")),
    "two_cta8_regs": ("re", r"\d{1,3}x\d{1,3}x\d{1,3}"),
    "two_cta8_causal_diag": "bool",
    "two_cta8_ld_rep": ("ints", (32, 64)),
    "two_cta8_q_stage": ("ints", (1, 2)),
    "two_cta8_dq_ncol": ("ints", (8, 16, 32, 64)),
    "two_cta8_dq_stages": ("int", 1, 8),
    "two_cta8_lpt_l2": "bool",
    "two_cta8_dq_zero": ("enum", ("fused", "pool", "fill")),
    "bwd_2cta_gqa": "bool",
    # ---- query-row (all-4-bit) backward -----------------------------------------------------------------------------
    "bwd_fp4q_qrow_cap": ("float", 0.0, 64.0),
    "bwd_fp4q_sr_dither": ("enum", ("hw", "weyl", "hash")),
    "bwd_fp4q_post": ("enum", ("legacy", "rows", "epi")),
    # opt-in: stochastic rounding of the dK side's E4M3 dS-hat under the default block-scaled dS (sr_dk_operand reaches
    # only the per-tensor dS); None = off
    "bwd_fp4q_dk_sr_blk32": "bool",
    # ---- quantisers and host glue -----------------------------------------------------------------------------------
    "fp4_d_do8": "bool",
    "fp4_fused_ds_bound": "bool",
    "qk_hadamard_e4m3": ("enum", ("rotated", "plain")),
    # the rotated Q / K quantisers' hadamard_rotation_method (quant._rotations): None = "tensor_core" (the default,
    # bitwise every earlier release); "tcgen05" = the opt-in UMMA rotation (NOT bitwise: the tensor core's single
    # 128-term accumulation; token-block / 32 x 32 Q / K operands only, refused elsewhere by the plan)
    "qk_hadamard_rotation_method": ("enum", ("tensor_core", "tcgen05")),
    # the 4-bit S operand's scale group (QuantSpec.s_sf_vec): None = 16 (NVFP4, E4M3 scale per 16); 32 = MXFP4 (UE8M0
    # per 32), OPT-IN and not API: measured less accurate and slower than NVFP4; pv_precision="fp8", backward="e4m3",
    # causal / dense only, no qk_hadamard / q_block_2d (plan rule E40)
    "fp4_qk_sf_vec": ("ints", (16, 32)),
}


def _tuning_check(name: str, value: Any, kind) -> None:
    where = f"KernelTuning.{name}"
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be a bool (None = the measured rule); got {value!r}.")
        return
    tag = kind[0]
    if tag in ("int", "ints"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an int (None = the measured rule); got {value!r}.")
        if tag == "ints" and value not in kind[1]:
            raise ValueError(f"{where} must be one of {kind[1]}; got {value!r}.")
        if tag == "int" and (value < kind[1] or (kind[2] is not None and value > kind[2])):
            raise ValueError(f"{where} must be in [{kind[1]}, {kind[2] if kind[2] is not None else 'inf'}]; got {value!r}.")
        return
    if tag == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{where} must be a finite float (None = the measured rule); got {value!r}.")
        lo, hi = kind[1], kind[2]
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"{where} must be in [{lo}, {hi}]; got {value!r}.")
        return
    if tag == "enum":
        if value not in kind[1]:
            raise ValueError(f"{where} must be one of {', '.join(repr(v) for v in kind[1])}; got {value!r}.")
        return
    if tag == "re":
        if not isinstance(value, str) or re.fullmatch(kind[1], value) is None:
            raise ValueError(f"{where} must match {kind[1]!r} (a structured register budget); got {value!r}.")
        return
    raise AssertionError(kind)


def _tuning_class():
    ns = {"__doc__": (
        "Measured performance constants of the low-precision kernels.\n\n"
        "Every field defaults to None, meaning 'the measured rule'. A non-None field is validated against its value "
        "set when the object is built. Not documented as API and without any stability promise; importable for "
        "benchmark campaigns from flash_attn.cute.lowprec.tuning. TUNING_FIELDS lists every field and its value "
        "kind; lowprec.specs.KT_TO_SPEC the spec fields each one sets."),
        "__annotations__": {n: Optional[Any] for n in TUNING_FIELDS}}
    for n in TUNING_FIELDS:
        ns[n] = None

    def __post_init__(self):
        for n, kind in TUNING_FIELDS.items():
            v = getattr(self, n)
            if v is not None:
                _tuning_check(n, v, kind)
        off, thr = self.fp8_pv_p_log2_offset, self.fp8_pv_rescale_threshold
        if off is not None and thr is not None and off + thr > math.log2(448) + 1e-9:
            raise ValueError("KernelTuning.fp8_pv_p_log2_offset + fp8_pv_rescale_threshold must be <= log2(448) "
                             f"(the E4M3 P-hat saturation bound); got {off} + {thr}.")

    def non_default(self) -> dict:
        """{field: value} of every field that is not None."""
        return {n: getattr(self, n) for n in TUNING_FIELDS if getattr(self, n) is not None}

    ns["__post_init__"] = __post_init__
    ns["non_default"] = non_default
    ns["__module__"] = __name__
    ns["__qualname__"] = "KernelTuning"
    cls = type("KernelTuning", (), ns)
    return dataclass(frozen=True)(cls)


KernelTuning = _tuning_class()

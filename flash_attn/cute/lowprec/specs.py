"""Kernel specs: the compile-time settings of the low-precision kernels as typed, frozen, hashable values.

One spec per kernel family plus two host-side specs:

  FwdKernelSpec  constexprs of FlashAttentionForwardSm100FP4 (flash_fwd_sm100_fp4.py) and the driver rules that become
                 its constructor arguments (interface_fwd_fp4.py)
  Bwd1Spec       constexprs of FlashAttentionBackwardSm100FP4, the 1-CTA backward (flash_bwd_sm100_fp4.py)
  Bwd2Spec       constexprs of FlashAttentionBackwardSm100TwoCta8, the 2-CTA backward (flash_bwd_sm100_2cta8.py)
  BwdQSpec       constexprs of FlashAttentionBackwardSm100FP4Q, the query-row (all-4-bit) backward (flash_bwd_sm100_fp4q.py)
  QuantSpec      how the host quantises and scales the operands: quantiser choices, rotation, causal-safe V layout, the
                 operand / dS / P^ scaling the glue, the route decision and the quantisers read (the kernel specs carry
                 their own copies of the values their constructors read; the plan sets both from one value)
  HostSpec       routing, scheduling and launch decisions of the drivers and the autograd glue (which kernel, persistent
                 grids, 2-CTA, dQ accumulator handling, saved tensors)

Rules of every spec:
  * a field is exactly one compile-time constexpr or driver rule; its metadata names the KernelTuning field (``kt``)
    that may override it, if any;
  * ``Optional[...] = None`` means "the constructor's (or driver's) measured rule"; the rule is documented on the field
    and mirrored by a ``rule_*`` static method;
  * ``key()`` is a stable tuple (class name, key version, every (field, value)) for the compile cache, so every field
    is in the compile key by construction;
  * nothing here reads the environment or any global configuration: the plan builds every spec once
    (lowprec.plan.build_specs, KernelTuning copied in through ``apply_kernel_tuning``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

__all__ = [
    "FwdKernelSpec", "Bwd1Spec", "Bwd2Spec", "BwdQSpec", "QuantSpec", "HostSpec", "SPECS", "SPEC_KEY_VERSION",
    "KT_TO_SPEC", "apply_kernel_tuning", "PlanSpecs", "resolved_key",
]

SPEC_KEY_VERSION = 1

# defaults shared with the tree (tests/cute/test_lowprec_spec_keys.py asserts they equal the tree's constants)
DS_SCALING_DEFAULT = "blk32"        # the 32 x 32 block-scaled dS^ (both routes)
FP4Q_MXDS_DEFAULT = "blk32"         # the all-4-bit backward's dK-side dS^
OP_SCALING_DEFAULT = "blk32x128"    # token-block E4M3 operands (one exponent per 32 tokens x 128 d)
TB_FWD_DEFAULT = "mx"               # the token-block forward runs the MXFP8 GEMMs (not the fold)
FWD8_ROW_P_DEFAULT = True           # 8-bit token-block forward: row-frame P^ on
FWD4_ROW_P_DEFAULT = False          # 4-bit token-block forward: row-frame P^ opt-in
DQ_ZERO_DEFAULT = "fused"           # dq_postprocess_cute.DQ_ZERO_DEFAULT
FP4_PV_RESCALE_THRESHOLD = 8.0      # flash_fwd_sm100_fp4.FP4_PV_RESCALE_THRESHOLD
QK_HADAMARD_E4M3_DEFAULT = {"blocks": "plain", "per_tensor": "rotated"}  # qk_hadamard.E4M3_DEFAULT


# ================================================================================================================
# KernelTuning value parsers: a KernelTuning value, spelled as a string ("1" / "0" for a bool), -> the spec field's value
# ================================================================================================================

def _p_bool(v: str) -> bool:
    return v == "1"


def _p_strict01(name: str) -> Callable[[str], bool]:
    """A bool that refuses anything but "0" / "1"."""
    def p(v: str) -> bool:
        if v not in ("0", "1"):
            raise ValueError(f"{name} must be '0' or '1', got {v!r}")
        return v == "1"
    return p


def _p_int(v: str) -> int:
    return int(v)


def _p_float(v: str) -> float:
    return float(v)


def _p_enum(name: str, allowed: Tuple[str, ...], mapping: Optional[Mapping[str, Any]] = None) -> Callable[[str], Any]:
    def p(v: str):
        if v not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {v!r}")
        return mapping[v] if mapping is not None else v
    return p


def _p_tri(name: str) -> Callable[[str], Optional[bool]]:
    """"auto" / "0" / "1" -> None / False / True, anything else refused."""
    return _p_enum(name, ("auto", "0", "1"), {"auto": None, "0": False, "1": True})


def _p_mxdp(v: str) -> Optional[bool]:
    """KernelTuning.mxfp4_directp: "1" -> True, "auto" -> None (the rule: exp2), anything else -> False."""
    if v == "auto":
        return None
    return v == "1"


def _p_regs(sep: str) -> Callable[[str], Optional[Tuple[int, int, int]]]:
    """A register budget "AsepBsepC" -> (A, B, C); "" (unset) -> None."""
    def p(v: str):
        if v == "":
            return None
        a, b, c = (int(x) for x in v.split(sep))
        return (a, b, c)
    return p


def _p_e2e(v: str) -> Optional[int]:
    """``int(v) if v else None`` (an empty string is 'unset')."""
    return int(v) if v else None


def _p_lpt(v: str) -> Optional[bool]:
    """"" = the driver's L2 rule, "0" / "1" force, anything else refused."""
    return _p_enum("KernelTuning.two_cta8_lpt_l2", ("", "0", "1"), {"": None, "0": False, "1": True})(v)


# ================================================================================================================
# field helper
# ================================================================================================================

_UNSET = object()


def F(default, *, doc: str, typ: tuple, parse: Optional[Callable[[str], Any]] = None, kt: Optional[str] = None,
      alt: Any = _UNSET):
    """One spec field.

    doc       what it controls and its default rule (the field's docstring)
    typ       ("bool",) | ("int", allowed-or-None) | ("float",) | ("enum", allowed) | ("regs",); Optional when the
              default is None
    kt        the KernelTuning field copied into it (KT_TO_SPEC is built from these); parse = how a KernelTuning value,
              spelled as a string ("1" / "0" for a bool), becomes the field's value (derived from typ when omitted)
    alt       a legal value different from the default (the two-value compile-key test); derived for bool fields
    """
    if alt is _UNSET:
        if isinstance(default, bool):
            alt = not default
        elif default is None and typ[0] == "bool":
            alt = True
        else:
            raise TypeError(f"field with default {default!r} needs an explicit alt value")
    if kt is not None and parse is None:
        parse = {"bool": _p_bool, "int": _p_int, "float": _p_float}.get(typ[0])
        if parse is None:
            raise TypeError(f"KernelTuning.{kt}: a parser is required for typ {typ}")
    return field(default=default, metadata=dict(doc=doc, typ=typ, parse=parse, kt=kt, alt=alt))


def _check(owner: str, name: str, value, typ: tuple, optional: bool) -> None:
    where = f"{owner}.{name}"
    if value is None:
        if not optional:
            raise ValueError(f"{where} must not be None")
        return
    kind = typ[0]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be a bool; got {value!r}")
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an int; got {value!r}")
        if len(typ) > 1 and typ[1] is not None and value not in typ[1]:
            raise ValueError(f"{where} must be one of {typ[1]}; got {value!r}")
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a float; got {value!r}")
    elif kind == "enum":
        if value not in typ[1]:
            raise ValueError(f"{where} must be one of {typ[1]}; got {value!r}")
    elif kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a str; got {value!r}")
    elif kind == "regs":
        if not (isinstance(value, tuple) and len(value) == 3 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                     for x in value)):
            raise ValueError(f"{where} must be a tuple of three ints; got {value!r}")
    else:
        raise AssertionError(typ)


class _SpecBase:
    """Shared machinery of the frozen spec dataclasses."""

    def __post_init__(self):
        for f in fields(self):
            _check(type(self).__name__, f.name, getattr(self, f.name), f.metadata["typ"], f.default is None)

    def key(self) -> tuple:
        """The stable compile-cache item: (class name, key version, every (field, value)). Two specs that differ in any
        field give different keys; equal specs give equal keys (tests/cute/test_lowprec_spec_keys.py)."""
        return (type(self).__name__, SPEC_KEY_VERSION) + tuple((f.name, getattr(self, f.name)) for f in fields(self))

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def non_default(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != f.default}

    @classmethod
    def field_docs(cls) -> str:
        out = []
        for f in fields(cls):
            m = f.metadata
            src = f"  [KernelTuning.{m['kt']}]" if m["kt"] else ""
            out.append(f"{f.name} = {f.default!r}{src}\n    {m['doc']}")
        return "\n".join(out)


def _spec(cls):
    """frozen dataclass + the generated field documentation appended to the class docstring."""
    cls = dataclass(frozen=True)(cls)
    cls.__doc__ = (cls.__doc__ or "") + "\n\nFields:\n" + cls.field_docs()
    return cls


_FP8_DS = ("per_tensor", "mx32", "blk32", "row", "blk32x64", "tile128")
_OP = ("per_tensor", "blk32", "blk32x128")


# ================================================================================================================
# FwdKernelSpec
# ================================================================================================================

@_spec
class FwdKernelSpec(_SpecBase):
    """Compile-time settings of FlashAttentionForwardSm100FP4 (flash_fwd_sm100_fp4.py), plus the driver rules that become
    its constructor arguments (interface_fwd_fp4.py). Option A's inference forward (pv4_route.pv4_repro_cfg) sets
    fp4all_rowsum / fp4all_sf_decode / fp4all_exp2; the all-4-bit route sets fp4all."""

    q_stage: Optional[int] = F(None, typ=("int", (1, 2)), kt="fp4_q_stage", alt=1,
        doc="Q tiles (128-row stages) per CTA. None = rule_q_stage: 1 on SplitKV or head_dim_v > 128, else 2.")
    corr_rescale_batch: int = F(1, typ=("int", None), kt="corr_rescale_batch", alt=0,
        doc="Correction warps: 16-column O fragments loaded back to back before the rescale multiplies. Default 1.")
    fp4_group_max_first: Optional[bool] = F(None, typ=("bool",), kt="fp4_group_max_first",
        doc="Log2 P quantisation: per-group maxes on the raw scores, row max from them, one packed FMA per pair. "
            "None = rule_group_max_first: on for an E2M1 P.V, off for E4M3.")
    exp2_emulation_every: Optional[int] = F(None, typ=("int", None), kt="fp4_exp2_emulation_every", alt=2,
        doc="One exp2 pair in every N on the FMA pipe (0 = all MUFU) on the log2-quantised step. None = "
            "rule_exp2_emulation_every: 3 on an E2M1 P.V, else 0. Must equal BwdQSpec.exp2_emulation_every on the "
            "all-4-bit route (the backward rebuilds P^).")
    mxfp8_use_inline_ptx: bool = F(False, typ=("bool",), kt="mxfp8_use_inline_ptx",
        doc="MXFP8 QK through the inline-PTX MMA helper instead of the generic one.")
    fp8_pv_use_explicit_pack: bool = F(True, typ=("bool",), kt="fp8_pv_use_explicit_pack",
        doc="E4M3 P^: explicit cvt pack (on) instead of the generic conversion.")
    e2e_freq: Optional[int] = F(None, typ=("int", None), parse=_p_e2e, kt="e2e_freq", alt=8,
        doc="FA4's exp2 emulation interleave on the generic step (one pair in N). None = the per-shape tuning table "
            "and the __call__ rules (an explicit value disables those rules: presence is significant).")
    e2e_start_frg: Optional[int] = F(None, typ=("int", None), kt="e2e_start_frg", alt=1,
        doc="First emulated fragment of the generic step. None = the per-shape table's e2e_start_frg, else 0.")
    force_e2e: Optional[bool] = F(None, typ=("bool",), kt="force_e2e",
        doc="Force the exp2 emulation on / off. None = the table's enable_e2e on NVFP4 QK, and on for the pure-E4M3 "
            "generic step (presence is significant: an explicit value disables the __call__ rule).")
    softmax_fused: Optional[int] = F(None, typ=("int", (0, 1, 2)), kt="fwd_softmax_fused", alt=2,
        doc="Generic E4M3 step's fused softmax loop: 0 four passes, 1 one fused loop, 2 FlashInfer's issue order. "
            "None = rule_softmax_fused: 1 causal, 0 otherwise. (KernelTuning's bool maps to 1 / 0.)")
    mask_loops: int = F(1, typ=("int", (0, 1, 2)), kt="fwd_mask_loops", alt=2,
        doc="1: the unmasked loop gets mask_fn=None; 2: also no emulation on masked steps; 0: neither. Default 1.")
    max_tmem: Optional[bool] = F(None, typ=("bool",), parse=_p_strict01("FwdKernelSpec.max_tmem"),
        kt="fwd_max_tmem", doc="Rescale factor softmax -> correction through TMEM instead of SMEM. None = _resolve_max_tmem's rule "
            "(split step: 4-bit P.V, not MXFP4 split, Direct-P, not all-FP4 unless causal key-local; generic: causal).")
    fp8_pv_zero_fill_regs: bool = F(True, typ=("bool",), kt="fp8_pv_zero_fill_regs",
        doc="Zero-fill the E4M3 P^ registers. Default on.")
    fp4_pv_log2_quant: bool = F(True, typ=("bool",), kt="fp4_pv_log2_quant",
        doc="Log-domain P quantisation (group max on the pre-exp scores, 1/max folded into exp2). Default on.")
    fp4_pv_two_level: bool = F(True, typ=("bool",),
        doc="NVFP4 P.V two-level P scaling (row max at 6*448). Default on; FP4AttnOptions.p_frame='single_level' "
            "turns it off (set by the plan).")
    fp4_pv_tile_frame: bool = F(False, typ=("bool",),
        doc="Per-tile row frame (SageAttention3 paper form). Default off; p_frame='per_tile' sets it together "
            "with fp4_pv_rescale_threshold = 0.")
    fp4_pv_tile_frame_drop: float = F(64.0, typ=("float",), kt="fp4_pv_tile_frame_drop", alt=32.0,
        doc="Per-tile frame: the reference floor below the running max (log2 units). Read only with fp4_pv_tile_frame.")
    fp4_pv_rescale_threshold: Optional[float] = F(None, typ=("float",), alt=0.0,
        doc="Lazy P-frame rescale threshold of the E2M1 P.V (log2 units). None = rule_fp4_pv_rescale_threshold: "
            "8.0 (FP4_PV_RESCALE_THRESHOLD) on an E2M1 P.V, else 0.0. p_frame='per_tile' sets 0.")
    ldred_rowmax: Optional[bool] = F(None, typ=("bool",), kt="ldred_rowmax",
        doc="tcgen05.ld.red fused row max. None = rule_ldred_rowmax: SM103 only.")
    corr_empty_early: bool = F(True, typ=("bool",), kt="corr_empty_early",
        doc="Correction releases a stage's acc_scale slot right after reading it.")
    qk_split_pipe: Optional[bool] = F(None, typ=("bool",), parse=_p_strict01("FwdKernelSpec.qk_split_pipe"),
        kt="qk_split_pipe",
        doc="Split-QK cross-tile pipeline override. None = the driver's applicability request (_qk_split_ok).")
    qk_split_bprefetch: bool = F(True, typ=("bool",), kt="qk_split_bprefetch",
        doc="Split step: b-half load issued right after the a-half maxes.")
    qk_split_late_p0: bool = F(True, typ=("bool",), kt="qk_split_late_p0",
        doc="Split step: the P k-block-0 store's wait deferred to the k-block-1 store.")
    split_lean: bool = F(True, typ=("bool",), kt="split_lean",
        doc="Split step: FMNMX3 group-max trees and one-FFMA SF arguments (E46). Default on.")
    direct_p: bool = F(True, typ=("bool",),
        doc="Direct-P (Mitchell) value function on the split step instead of exp2. Default True (the shipped 4-bit "
            "forward); FP4AttnOptions.softmax_exp sets it.")
    fp4all: bool = F(False, typ=("bool",),
        doc="The all-FP4 training forward (integer frames, frame export in the pmax slot). Set by the all-4-bit route "
            "(interface_bwd_fp4q.flash_attn_fp4all_fwd) and option A.")
    fp4all_exp2: bool = F(False, typ=("bool",),
        doc="All-FP4 step: exp2 value function inside the lean Direct-P structure (False: Direct-P). Option A's "
            "reproduce-forward-p sets it from softmax_exp.")
    fp4all_rowsum: str = F("mitchell", typ=("enum", ("mitchell", "codes")), alt="codes",
        doc="All-FP4 row sum source: 'mitchell' (FADD2 tree) or 'codes' (exact sum of the quantised values). Option "
            "A's inference forward sets 'codes' (pv4_route.pv4_repro_cfg). Read only on the all-FP4 step.")
    fp4all_sf_decode: bool = F(False, typ=("bool",),
        doc="All-FP4: row sum from the E4M3-rounded group SFs. Option A's inference forward sets it.")
    mxfp4_split: bool = F(True, typ=("bool",), kt="mxfp4_split",
        doc="MXFP4 P.V on the split pipeline (integer frame). Read by the kernel and by the driver's applicability "
            "rule.")
    mxfp4_directp: Optional[bool] = F(None, typ=("bool",), parse=_p_mxdp, kt="mxfp4_directp",
        doc="MXFP4 split: Direct-P (clamped). None = 'auto' = exp2 (False).")
    fp8_pv_log2_offset: Optional[float] = F(None, typ=("float",), kt="fp8_pv_p_log2_offset", alt=2.0,
        doc="E4M3 P^ = E4M3(P 2^offset). None = rule_fp8_pv_log2_offset: 4.0 on a pure E4M3 P.V, else 0.0.")
    fp8_pv_rescale_threshold: Optional[float] = F(None, typ=("float",), kt="fp8_pv_rescale_threshold", alt=2.0,
        doc="E4M3 P^ lazy row-max refresh threshold. None = rule_fp8_pv_rescale_threshold: 4.0 pure E4M3, else 8.0.")
    o_regstore: Optional[bool] = F(None, typ=("bool",), kt="fwd_o_regstore",
        doc="Correction warps store O from registers. None = the driver's o_regstore_default (HostSpec "
            "fwd_o_regstore_min_sk rule), still subject to the kernel's exclusions.")
    q_dbuf: Optional[bool] = F(None, typ=("bool",), kt="q_dbuf",
        doc="Split step: Q + SFQ double-buffered across work tiles. None = rule_q_dbuf: on for pure E4M3 Q / K / V.")
    fp8_pv_kv_stage_cap: Optional[int] = F(None, typ=("int", None), kt="fp8_pv_kv_stage_cap", alt=3,
        doc="KV stage cap of the row-frame E4M3 P.V. None = rule_fp8_pv_kv_stage_cap: 4 at head_dim_v >= 128, "
            "else 0 (no cap).")
    fp8_generic_e2e: Optional[int] = F(None, typ=("int", None), kt="fp8_generic_e2e",
        alt=7, doc="Pure-E4M3 generic step emulation share (one pair in N). None = the driver's fp8_generic_e2e_default "
            "(rule_fp8_generic_e2e: 0 on SM103; on SM100 the HostSpec fp8_e2e8_min_sk / max_sk rule, 8 or 6).")
    fp8_pv_p_split_num: int = F(1, typ=("int", None), kt="fp8_pv_p_split_num", alt=3,
        doc="E4M3 P.V: P_full split numerator (1/2 on B200).")
    fp8_pv_p_split_den: int = F(2, typ=("int", None), kt="fp8_pv_p_split_den", alt=4,
        doc="E4M3 P.V: P_full split denominator.")
    fp4_pv_p_split_num: int = F(1, typ=("int", None), kt="fp4_pv_p_split_num", alt=3,
        doc="E2M1 P.V: P store chunk split numerator.")
    fp4_pv_p_split_den: int = F(2, typ=("int", None), kt="fp4_pv_p_split_den", alt=4,
        doc="E2M1 P.V: P store chunk split denominator.")
    fp8_pv_tmem_store_rep: Optional[int] = F(None, typ=("int", (1, 2, 4, 8, 16, 32, 64)),
        kt="fp8_pv_tmem_store_rep", alt=16,
        doc="Row-frame E4M3 P store repetition. None = rule_fp8_pv_tmem_store_rep: 4 at head_dim_v <= 64, else 8.")
    fp4_pv_tmem_store_rep: int = F(8, typ=("int", (1, 2, 4, 8, 16, 32, 64)), kt="fp4_pv_tmem_store_rep", alt=4,
        doc="Block-scaled P store repetition (smaller = finer P hand-off). Default 8.")
    causal_zigzag_order: Optional[bool] = F(None, typ=("bool",),
        doc="Causal persistent grid: balanced zigzag tile order (measured: the causal persistent grid's load balance). "
            "None = the driver rule "
            "(_causal_balanced and _is_persistent). O / LSE bitwise either way.")
    causal_stage_skip: Optional[bool] = F(None, typ=("bool",),
        doc="Causal 1-CTA: Q stage 0 skips the tile's last KV block (measured: the causal persistent grid's load "
            "balance). None = the driver "
            "rule (_causal_balanced). O / LSE bitwise either way.")
    ptxas_options: str = F("", typ=("str",), parse=lambda v: v, kt="fwd_ptxas", alt="-O3",
        doc="Extra ptxas options of this kernel's compile (cute.compile --ptxas-options). '' = none (the certified "
            "build). Opt-in schedule experiments (fwdgap).")
    exp: str = F("", typ=("str",), parse=lambda v: v, kt="fwd_exp", alt="x",
        doc="Comma-separated experimental forward switches (fwdgap, opt-in; '' = the certified kernel), read as "
            "FlashAttentionForwardSm100FP4.xf. 'sfi': the group-SF P^ step's ceil / 2^ceil on the FMA / ALU pipes "
            "(bitwise). 'gmx': its group maxes from the raw scores (bitwise). 'sche' / 'scha' / 'schmK' (K = 0..8): "
            "the pure-E4M3 P^ step's Schraudolph exp2 on the emulated pairs / on every pair / on 8 - K of every 8 "
            "pairs (approximate, -3.9 / +2.0 %). 'schg': the group-SF step's emulated pairs (exp2_emulation_every) "
            "take the Schraudolph exp2 (approximate).")

    # ---- the constructor's rules ----------------------------------------------------------------------------------
    @staticmethod
    def rule_q_stage(is_split_kv: bool, head_dim_v_padded: int) -> int:
        return 1 if (is_split_kv or head_dim_v_padded > 128) else 2

    @staticmethod
    def rule_group_max_first(e2m1_pv: bool) -> bool:
        return bool(e2m1_pv)

    @staticmethod
    def rule_exp2_emulation_every(e2m1_pv: bool) -> int:
        return 3 if e2m1_pv else 0

    @staticmethod
    def rule_softmax_fused(is_causal: bool) -> int:
        return 1 if is_causal else 0

    @staticmethod
    def rule_fp4_pv_rescale_threshold(e2m1_pv: bool) -> float:
        return FP4_PV_RESCALE_THRESHOLD if e2m1_pv else 0.0

    @staticmethod
    def rule_ldred_rowmax(is_sm103: bool) -> bool:
        return bool(is_sm103)

    @staticmethod
    def rule_fp8_generic_e2e(is_sm103: bool, short_dense_window: bool) -> int:
        # SM100: every 8th pair on the short dense window, else every 6th (the MUFU / FMA balance point of B200).
        # SM103: MUFU.EX2 has twice the throughput, so every emulated pair only loads the FMA pipe -- no emulation
        # (B300, b300/fwdreg ab2: 0 vs every 6th / 8th -24 to -28 % on A1-A4, vs every 32nd -1 to -4 %; O unchanged).
        if is_sm103:
            return 0
        return 8 if short_dense_window else 6

    @staticmethod
    def rule_fp8_pv_log2_offset(pure_fp8_pv: bool) -> float:
        return 4.0 if pure_fp8_pv else 0.0

    @staticmethod
    def rule_fp8_pv_rescale_threshold(pure_fp8_pv: bool) -> float:
        return 4.0 if pure_fp8_pv else 8.0

    @staticmethod
    def rule_q_dbuf(quant_qk: bool, quant_pv: bool) -> bool:
        return (not quant_qk) and (not quant_pv)

    @staticmethod
    def rule_fp8_pv_kv_stage_cap(head_dim_v_padded: int) -> int:
        return 4 if head_dim_v_padded >= 128 else 0

    @staticmethod
    def rule_fp8_pv_tmem_store_rep(head_dim_v_padded: int) -> int:
        return 4 if head_dim_v_padded <= 64 else 8


# ================================================================================================================
# Bwd1Spec
# ================================================================================================================

@_spec
class Bwd1Spec(_SpecBase):
    """Compile-time settings of FlashAttentionBackwardSm100FP4, the 1-CTA backward (flash_bwd_sm100_fp4.py; drivers
    interface_bwd_fp4._flash_attn_bwd_fp4 / _flash_attn_bwd_fp8_control). ``fp8`` selects the route and
    ``op_scaling`` is that route's operand scaling."""

    fp8: bool = F(False, typ=("bool",),
        doc="The 8-bit control kernel (all five GEMMs E4M3) instead of the 4-bit dk8 route. Set by the entry point "
            "(flash_attn_fp8); equals HostSpec.bwd_op_mode == 'fp8'.")
    compute_wgs: int = F(2, typ=("int", (1, 2, 4)), alt=4,
        doc="Compute warpgroups: 2 (the block-scaled dS needs two; 4 failed to fit its registers on the per-tensor "
            "dS). "
            "Not a KernelTuning field.")
    ds_fold: bool = F(True, typ=("bool",), kt="bwd_fp4_ds_fold",
        doc="dS = P (dP^ b - D b s) as FMA(dP^, b, D'') with D'' pre-scaled on the host. Must equal QuantSpec.ds_fold.")
    ds_sr: bool = F(False, typ=("bool",), doc="Stochastic rounding of the E4M3 dS^ (FP4AttnOptions.sr_ds).")
    ds_scaling_fp4: str = F(DS_SCALING_DEFAULT, typ=("enum", ("per_tensor", "blk32")), alt="per_tensor",
        doc="dk8 dS^ scaling: 'blk32' (in-kernel UE8M0 per 32 x 32) or 'per_tensor' (head_dim 256's automatic one). "
            "Must equal QuantSpec.ds_scaling_fp4.")
    p_scaling_fp4: str = F("row", typ=("enum", ("row", "mx32")), alt="mx32",
        doc="dk8 P^ scaling: 'row' (fixed 2^8 lift) or 'mx32' (UE8M0 per key row x 32 queries). = "
            "QuantSpec.p_scaling_fp4.")
    op_scaling: str = F(OP_SCALING_DEFAULT, typ=("enum", _OP), alt="per_tensor",
        doc="E4M3 operand scaling of Q, K, V, dO (QuantSpec.operand_scaling_fp8 on the 8-bit route, "
            "QuantSpec.operand_scaling_fp4 on the 4-bit one, resolved by the plan); varlen forces "
            "blk32x128 in the constructor. = the route's QuantSpec.operand_scaling_* once resolved.")
    ld_rep: Optional[int] = F(None, typ=("int", (32, 64)), kt="bwd_fp4_ld_rep", alt=64,
        doc="S / dP TMEM load repetition. None = rule_ld_rep: 32 on fp8 / block-scaled dS / mx32 P / block-scaled "
            "operands, else 64.")
    sload_early: bool = F(True, typ=("bool",), kt="bwd_fp4_sload_early",
        doc="Signal 'S(t) loaded' right after the TMEM load.")
    warp_sync: bool = F(True, typ=("bool",), kt="bwd_fp4_warp_sync",
        doc="Per-warp hand-offs instead of a 256-thread named barrier.")
    causal_diag: bool = F(True, typ=("bool",), kt="bwd_fp4_causal_diag",
        doc="Mask only the causal diagonal tiles.")
    dp_ld2: bool = F(False, typ=("bool",), kt="bwd_fp4_dp_ld2",
        doc="Issue both x32 dP loads before stage 0's dS math (forced on by row dS).")
    ds_scaling_fp8: str = F(DS_SCALING_DEFAULT, typ=("enum", _FP8_DS), alt="mx32",
        doc="8-bit dS^ scaling (per_tensor | mx32 | blk32 | row on this kernel). Read on both routes (the dk8 route "
            "overrides it with blk32 under mxds_fp4). = QuantSpec.ds_scaling_fp8.")
    p_scaling_fp8: str = F("row", typ=("enum", ("row", "mx32")), alt="mx32",
        doc="8-bit P^ scaling: 'row' (fixed 2^8 shift) or 'mx32'. Must equal Bwd2Spec.p_scaling_fp8 for one plan.")
    do_stage: int = F(1, typ=("int", (1, 2)), kt="bwd_fp4_do_stage", alt=2,
        doc="dO ring depth (block-scaled operands assert 1).")
    stat_shfl: str = F("off", typ=("enum", ("off", "lse", "dpsum", "both")), parse=lambda v: v, kt="bwd_fp4_stat_shfl",
        alt="both",
        doc="U4 prototype (opt-in, measured slower): LSE / dPsum of a 32-query stage held one per lane (one LDS.32) and "
            "broadcast by 32 SHFL.IDX instead of the 8 broadcast LDS.128.")
    q_stage: int = F(2, typ=("int", (2, 3)), kt="bwd_fp4_q_stage", alt=3,
        doc="Q ring depth (item C, opt-in): 3 = one more E4M3 Q tile + LSE row (+16 KB SMEM), FP8 kernel only "
            "(not the 32 x 32 block-scaled operands); always with the dynamic Q-stage issue (q_dyn_issue).")
    q_dyn_issue: bool = F(False, typ=("bool",), kt="bwd_fp4_q_dyn_issue",
        doc="MMA warp: S / block-scaled dK GEMMs issued on the dynamically indexed Q stage instead of one static copy "
            "per stage (opt-in; implied by q_stage 3). FP8 kernel, per-tensor or token-block operands.")
    pp: bool = F(True, typ=("bool",),
        doc="d = 64 warpgroup ping-pong (dQ TMEM bank of its own; -9 % at D64c). Not a KernelTuning field.")
    reduce_chunked: bool = F(True, typ=("bool",), kt="bwd_fp4_reduce_chunked",
        doc="Reduce warps load dQ one chunk at a time (+40 compute registers).")
    regs4: Tuple[int, int, int] = F((96, 56, 40), typ=("regs",), alt=(88, 64, 64),
        doc="(compute, reduce, load/mma/empty) registers at compute_wgs == 4 only (unreachable: not a KernelTuning "
            "field).")
    dq_ncol: Optional[int] = F(None, typ=("int", (8, 16, 32, 64, 128)), kt="bwd_fp4_dq_ncol", alt=32,
        doc="dQ reduce chunk columns. None = rule_dq_ncol: 64 at 2 compute WGs, else 32.")
    dq_stages: Optional[int] = F(None, typ=("int", None), kt="bwd_fp4_dq_stages",
        alt=4, doc="dQ SMEM stages. None = rule_dq_stages: 2 at 2 WGs, else 4.")
    det_issue_warp: Optional[bool] = F(None, typ=("bool",),
        doc="Deterministic staged dQ: idle warp 0 issues the reduce-adds (measured faster than a compute warp issuing "
            "them). None = on.")
    det_issue_bufs: Optional[int] = F(None, typ=("int", (1, 2)), alt=1,
        doc="Deterministic issue warp: staged f16 dQ buffers. None = 2 (the constructor's rule).")
    det_gqa_rot: bool = F(False, typ=("bool",), kt="bwd_fp4_det_gqa_rot", alt=True,
        doc="bwdk opt-in: deterministic DENSE GQA with the rotated walk (key tile n starts its m walk at block n, so "
            "the key tiles of a kv head write distinct m blocks each step; lock = the walk step). Natural (head-"
            "contiguous) grid order; needs s_q == s_k and every key tile of a head co-resident (s_k / 128 <= SMs).")

    @staticmethod
    def rule_ld_rep(fp8: bool, mxds_fp4: bool, mxp_fp4: bool, mxop: bool) -> int:
        return 32 if (fp8 or mxds_fp4 or mxp_fp4 or mxop) else 64

    @staticmethod
    def rule_dq_ncol(compute_wgs: int) -> int:
        return 64 if compute_wgs == 2 else 32

    @staticmethod
    def rule_dq_stages(compute_wgs: int) -> int:
        return 2 if compute_wgs == 2 else 4


# ================================================================================================================
# Bwd2Spec
# ================================================================================================================

@_spec
class Bwd2Spec(_SpecBase):
    """Compile-time settings of FlashAttentionBackwardSm100TwoCta8, the 2-CTA backward (flash_bwd_sm100_2cta8.py; driver
    interface_bwd_2cta8.flash_attn_bwd_2cta8). ``op_scaling`` is the route's operand scaling (the 4-bit route's
    under NVFP4 S, else the 8-bit route's)."""

    ds_scaling_fp8: str = F(DS_SCALING_DEFAULT, typ=("enum", _FP8_DS), alt="tile128",
        doc="8-bit dS^ scaling (per_tensor | blk32 | blk32x64 | tile128 on this kernel). = QuantSpec.ds_scaling_fp8.")
    ds_scaling_fp4: str = F(DS_SCALING_DEFAULT, typ=("enum", ("per_tensor", "blk32")), alt="per_tensor",
        doc="dk8 dS^ scaling under nvfp4_s. = QuantSpec.ds_scaling_fp4.")
    p_scaling_fp8: str = F("row", typ=("enum", ("row", "mx32")), alt="mx32",
        doc="8-bit P^ scaling (FP*AttnOptions.p_scaling).")
    op_scaling: str = F(OP_SCALING_DEFAULT, typ=("enum", ("per_tensor", "blk32x128")), alt="per_tensor",
        doc="E4M3 operand scaling (blk32 is 1-CTA only); varlen forces blk32x128 in the constructor.")
    warp_sync: bool = F(True, typ=("bool",), kt="two_cta8_warp_sync",
        doc="Per-warp hand-offs (MX_XW needs it).")
    mx_xw: Optional[str] = F(None, typ=("enum", ("", "dq", "all")), parse=_p_enum(
        "Bwd2Spec.mx_xw", ("", "dq", "all")), kt="two_cta8_mx_xw", alt="dq",
        doc="Block-scaled dS SF exchange warp: '' none, 'dq', 'all'. None = rule_mx_xw: 'all' under blk32, else ''.")
    regs: Optional[Tuple[int, int, int]] = F(None, typ=("regs",), parse=_p_regs("x"),
        kt="two_cta8_regs", alt=(136, 144, 88),
        doc="(reduce, compute, load = mma) register budgets. None = the constructor's table.")
    causal_diag: bool = F(True, typ=("bool",), kt="two_cta8_causal_diag",
        doc="Mask only the causal diagonal tiles.")
    ld_rep: int = F(32, typ=("int", (32, 64)), kt="two_cta8_ld_rep", alt=64,
        doc="S / dP TMEM load repetition (64 needs a per-tensor dS).")
    q_stage: int = F(1, typ=("int", (1, 2)), kt="two_cta8_q_stage", alt=2,
        doc="Q / Q^T / LSE ring depth on the 2-CTA instructions (mxds asserts 1).")
    dq_ncol: Optional[int] = F(None, typ=("int", (8, 16, 32, 64)), kt="two_cta8_dq_ncol",
        alt=32, doc="dQ reduce chunk columns. None = rule_dq_ncol: 16 deterministic, 8 otherwise.")
    dq_stages: Optional[int] = F(None, typ=("int", None), kt="two_cta8_dq_stages",
        alt=3, doc="dQ TMA commit depth. None = rule_dq_stages: 2 deterministic, 4 otherwise.")
    lpt_l2: Optional[bool] = F(None, typ=("bool",), parse=_p_lpt, kt="two_cta8_lpt_l2",
        doc="LPT-L2 swizzle of the persistent grid. None = the driver's rule (_lpt_l2_worth_it: 4+ heads fit L2).")

    @staticmethod
    def rule_mx_xw(effective_ds_scaling: str) -> str:
        """effective_ds_scaling = the kernel's self.ds_scaling after the mxds_fp4 override ('blk32' under nvfp4_s with
        ds_scaling_fp4 == 'blk32', else ds_scaling_fp8)."""
        return "all" if effective_ds_scaling == "blk32" else ""

    @staticmethod
    def rule_dq_ncol(deterministic: bool) -> int:
        return 16 if deterministic else 8

    @staticmethod
    def rule_dq_stages(deterministic: bool) -> int:
        return 2 if deterministic else 4


# ================================================================================================================
# BwdQSpec
# ================================================================================================================

@_spec
class BwdQSpec(_SpecBase):
    """Compile-time settings of FlashAttentionBackwardSm100FP4Q, the query-row all-4-bit backward
    (flash_bwd_sm100_fp4q.py; driver interface_bwd_fp4q._flash_attn_bwd_fp4q)."""

    exp2: bool = F(False, typ=("bool",),
        doc="Rebuild P^ with exp2 (the forward's fp4all_exp2; False by default). Must equal "
            "FwdKernelSpec.fp4all_exp2 for one step.")
    exp2_emulation_every: Optional[int] = F(None, typ=("int", None), kt="fp4_exp2_emulation_every", alt=2,
        doc="The forward's FMA-pipe exp2 share, reproduced. None = rule: 3 (the E2M1 P.V forward's).")
    dq8: bool = F(False, typ=("bool",),
        doc="E4M3 dS for the dQ GEMM (A from the dK operand's SMEM tile). FP4AttnOptions.dq_operand='e4m3'. Effective "
            "only "
            "with dkv == 'e4m3'.")
    sr_dither: str = F("hw", typ=("enum", ("hw", "weyl", "hash")),
        parse=_p_enum("BwdQSpec.sr_dither", ("hw", "weyl", "hash")), kt="bwd_fp4q_sr_dither", alt="weyl",
        doc="SR dither source: 'hw' (cvt.rs), 'weyl', 'hash'.")
    post: str = F("legacy", typ=("enum", ("legacy", "rows", "epi")),
        parse=_p_enum("BwdQSpec.post", ("legacy", "rows", "epi")), kt="bwd_fp4q_post", alt="rows",
        doc="The post launch (dQ convert + the K fold's dK / dV rows): 'legacy' (FP4QPostprocess, a thread per row), "
            "'rows' (fp4q_post_rows.FP4QPostprocessRows, coalesced), 'epi' (rows, and the K fold's dK / dV row factors "
            "in the kernel's dK / dV epilogue). All three give the same bits.")
    mxds: Optional[str] = F(None, typ=("enum", ("per_tensor", "mx32", "blk32")), alt="per_tensor",
        doc="dK-side E4M3 dS^ scaling ('per_tensor' / 'mx32' / 'blk32'). None = rule_mxds: per_tensor on the diagonal "
            "r_k "
            "routes, else blk32 (follows ds_scaling; plan-resolved).")
    dk_sr: bool = F(True, typ=("bool",),
        doc="SR on the dK operand's E4M3 dS^ (FP4AttnOptions.sr_dk_operand). Effective only with dkv == 'e4m3'.")
    dv_sr: bool = F(False, typ=("bool",), doc="SR on the dV operand's E4M3 P^ (FP4AttnOptions.sr_dv_operand).")
    dk_sr_blk32: bool = F(False, typ=("bool",), kt="bwd_fp4q_dk_sr_blk32",
        doc="SR on the dK operand's E4M3 dS^ under the blk32 dS (opt-in; dk_sr reaches only the per-tensor dS).")
    dkv: str = F("e4m3", typ=("enum", ("e4m3", "fp4")), alt="fp4",
        doc="dK / dV operands: 'e4m3' (hybrid) or 'fp4' (every backward GEMM NVFP4). FP4AttnOptions.fp4_backward_dkdv.")

    @staticmethod
    def rule_exp2_emulation_every() -> int:
        return 3

    @staticmethod
    def rule_mxds(diag_kr: bool) -> str:
        return "per_tensor" if diag_kr else FP4Q_MXDS_DEFAULT


# ================================================================================================================
# QuantSpec
# ================================================================================================================

def _cfg_str(v):
    return None if v is None else str(v)


@_spec
class QuantSpec(_SpecBase):
    """How the host quantises and scales the operands (quantisers, rotations, the causal-safe V layout, the operand /
    dS / P^ scaling read by the route decision, the glue and the quantisers). The kernel specs carry the copies their
    constructors read; the plan sets both from one value (scaling-consistency rule)."""

    token_block_forward: str = F(TB_FWD_DEFAULT, typ=("enum", ("mx", "fold")),
        parse=_p_enum("QuantSpec.token_block_forward", ("fold", "mx")), kt="token_block_forward", alt="fold",
        doc="Forward of the token-block operands: 'mx' (MXFP8 GEMMs; the default: the fold leaks 2.3e-02 inside a "
            "32-key block and is untrained) or 'fold'.")
    q_block_2d: bool = F(False, typ=("bool",), doc="Q's NVFP4 scales per 16 x 16 block (FP4AttnOptions.q_block_2d).")
    fused_ds_bound: bool = F(True, typ=("bool",), kt="fp4_fused_ds_bound",
        doc="dS bound inputs from the V / dO quantisers (no separate norm passes).")
    d_from_do8: bool = F(True, typ=("bool",), kt="fp4_d_do8",
        doc="D = sum(dO8 O) from the E4M3 dO the dP GEMM multiplies (softmax-consistent).")
    qk_hadamard: bool = F(False, typ=("bool",),
        doc="Random Hadamard rotation of Q / K on the E4M3-operand routes (the qk_hadamard argument; the API default "
            "is on).")
    qk_hadamard_e4m3: Optional[str] = F(None, typ=("enum", ("rotated", "plain")),
        parse=_p_enum("QuantSpec.qk_hadamard_e4m3", ("rotated", "plain")), kt="qk_hadamard_e4m3", alt="rotated",
        doc="Basis of the E4M3 copies under qk_hadamard. None = QK_HADAMARD_E4M3_DEFAULT per path "
            "(blocks: plain, per_tensor: rotated).")
    qk_hadamard_rotation_method: Optional[str] = F(None, typ=("enum", ("tensor_core", "tcgen05")),
        parse=_p_enum("QuantSpec.qk_hadamard_rotation_method",
                                                              ("tensor_core", "tcgen05")),
        kt="qk_hadamard_rotation_method", alt="tcgen05",
        doc="Rotated quantisers' hadamard_rotation_method. None = the quantisers' default ('tensor_core').")
    per_tensor_pow2: bool = F(False, typ=("bool",),
        doc="Per-tensor scales rounded up to a power of two (per_tensor_scale='power_of_two').")
    s_sf_vec: int = F(16, typ=("int", (16, 32)), alt=32, kt="fp4_qk_sf_vec",
        doc="The 4-bit S operand format: 16 = NVFP4, 32 = MXFP4 (no option field; reachable only through the advanced KernelTuning.fp4_qk_sf_vec=32, measured worse than NVFP4).")
    grad_format: str = F("e4m3", typ=("enum", ("e4m3", "e5m2")), alt="e5m2",
        doc="8-bit gradient operand type (FP8AttnOptions.grad_format).")
    operand_scaling_fp4: Optional[str] = F(None, typ=("enum", _OP), alt="per_tensor",
        doc="4-bit route's E4M3 copies (operand_scaling). None = resolve_op_scaling per call (default blk32x128, "
            "per_tensor where token blocks cannot serve the shape).")
    operand_scaling_fp8: Optional[str] = F(None, typ=("enum", _OP), alt="per_tensor",
        doc="8-bit route's operand scaling (operand_scaling). None = resolve_op_scaling per call.")
    ds_scaling_fp4: str = F(DS_SCALING_DEFAULT, typ=("enum", ("per_tensor", "blk32")), alt="per_tensor",
        doc="4-bit route's dS^ scaling (ds_scaling). Kernel copies: Bwd1Spec / Bwd2Spec.ds_scaling_fp4.")
    ds_scaling_fp8: str = F(DS_SCALING_DEFAULT, typ=("enum", _FP8_DS), alt="row",
        doc="8-bit route's dS^ scaling (ds_scaling). Kernel copies: Bwd1Spec / Bwd2Spec.ds_scaling_fp8.")
    p_scaling_fp4: str = F("row", typ=("enum", ("row", "mx32")), alt="mx32",
        doc="4-bit route's P^ scaling (p_scaling; mx32 takes the 1-CTA kernel). Kernel copy: Bwd1Spec.p_scaling_fp4.")
    ds_fold: bool = F(True, typ=("bool",), kt="bwd_fp4_ds_fold",
        doc="Host side of the D fold (D'' = -s b D pre-scaled; the glue's d_multiplier). Kernel copy: "
            "Bwd1Spec.ds_fold.")
    causal_safe_v: str = F("auto", typ=("enum", ("auto", "always", "never")), alt="never",
        doc="Key-local V scaling on the 4-bit P.V: 'auto' (masked kernels), 'always', 'never' (causal_safety).")
    fp4all_vtok: Optional[bool] = F(None, typ=("bool",),
        doc="All-4-bit: force the full key-local V on / off. None = causal_safe_v's rule (causal_safety).")
    vtok_diag: bool = F(False, typ=("bool",),
        doc="All-4-bit: key-local V on the causal diagonal tiles only (causal_safety='diagonal').")
    vtok_diag_k: str = F("fold", typ=("enum", ("fold", "cut", "all", "open")), alt="cut",
        doc="Diagonal route's K channel into dQ: 'fold', 'cut', 'all', 'open' "
            "(FP4AttnOptions.k_channel).")
    vtok_hadqk: bool = F(False, typ=("bool",),
        doc="All-4-bit: random Hadamard on Q / K fused into the quantisers (the qk_hadamard argument on that route).")
    vtok_hadv: bool = F(False, typ=("bool",),
        doc="All-4-bit: Hadamard on V before the key-local quantiser (FP4AttnOptions.v_hadamard).")
    fp4all_stab: str = F("sr", typ=("enum", ("sr", "eden_rht", "rtn")), alt="rtn",
        doc="All-4-bit stabiliser: 'sr', 'eden_rht', 'rtn' (FP4AttnOptions.all_fp4_stabiliser).")
    fp4q_ds_scale: str = F("auto", typ=("enum", ("auto", "global", "causal")), alt="global",
        doc="All-4-bit dS multiplier: 'auto' (causal on a causal E4M3 dK/dV backward, else global) "
            "(FP4AttnOptions.fp4_ds_multiplier).")
    fp4q_qrow: Optional[bool] = F(None, typ=("bool",),
        doc="All-4-bit per-query-row pow2 boost under the global multiplier. None = rule_fp4q_qrow: on for dense, off "
            "for causal (fp4_ds_multiplier='global_row_boost').")
    fp4q_qrow_cap: float = F(6.0, typ=("float",), kt="bwd_fp4q_qrow_cap", alt=4.0,
        doc="Cap (log2) of the per-row boost.")

    @staticmethod
    def rule_fp4q_qrow(causal: bool) -> bool:
        return not causal

    @staticmethod
    def rule_qk_hadamard_e4m3(path: str) -> str:
        return QK_HADAMARD_E4M3_DEFAULT[path]


# ================================================================================================================
# HostSpec
# ================================================================================================================

@_spec
class HostSpec(_SpecBase):
    """Routing, scheduling and launch decisions of the drivers and the autograd glue. These choose constructor arguments
    (already in each kernel's key as the resolved value), which kernel runs, and how the host prepares its buffers."""

    # ---- forward driver (interface_fwd_fp4) -------------------------------------------------------------------------
    qk_split_causal: bool = F(True, typ=("bool",), kt="qk_split_causal",
        doc="Split-QK pipeline allowed on causal calls.")
    qk_split_seqused: bool = F(True, typ=("bool",), kt="qk_split_seqused",
        doc="Split-QK pipeline allowed with seqused_k.")
    fwd_causal_persistent: Optional[bool] = F(None, typ=("bool",), parse=_p_tri("HostSpec.fwd_causal_persistent"),
        kt="fwd_causal_persistent",
        doc="Causal persistent grid. None = 'auto' (balanced zigzag where it applies, short pure-E4M3 causal cells).")
    fwd_causal_persistent_max_sk: int = F(2048, typ=("int", None), kt="fwd_causal_persistent_max_sk", alt=4096,
        doc="'auto' persistent causal grid on the pure-E4M3 route up to this s_k.")
    fwd_nc_persistent: bool = F(True, typ=("bool",), kt="fwd_nc_persistent",
        doc="Persistent grid for dense attention.")
    fwd_o_regstore_min_sk: int = F(8192, typ=("int", None), kt="fwd_o_regstore_min_sk", alt=4096,
        doc="o_regstore_default: pure-E4M3 dense fixed-length calls from this s_k.")
    fp8_e2e8_min_sk: int = F(2048, typ=("int", None), kt="fp8_e2e8_min_sk", alt=1024,
        doc="fp8_generic_e2e_default = 8 on dense min_sk <= s_k < max_sk (else 6).")
    fp8_e2e8_max_sk: int = F(8192, typ=("int", None), kt="fp8_e2e8_max_sk", alt=16384,
        doc="Upper bound of the every-8th emulation window.")
    seqused_padding_ok: bool = F(False, typ=("bool",),
        doc="Caller acknowledges seqused_k padding for a 4-bit P.V (FP4AttnOptions.seqused_k_group_aligned; refused at "
            "True).")
    fp8_pure_qk: bool = F(True, typ=("bool",), doc="Allow E4M3 Q / K without block scales (implied by flash_attn_fp8).")
    fp8_forward_kernel: str = F("auto", typ=("enum", ("auto", "split", "fork", "upstream")), alt="upstream",
        doc="8-bit forward kernel: 'auto' / 'split' / 'fork' = the fork kernel, 'upstream' "
            "(FP8AttnOptions.forward_kernel).")
    fwd8_row_p: bool = F(FWD8_ROW_P_DEFAULT, typ=("bool",),
        doc="8-bit token-block forward: row-frame E4M3 P^ (on by default; the p_hat_frame option).")
    fwd4_row_p: bool = F(FWD4_ROW_P_DEFAULT, typ=("bool",),
        doc="4-bit token-block forward: row-frame E4M3 P^ (opt-in; the p_hat_frame option).")
    fwd_two_cta: bool = F(False, typ=("bool",),
        doc="Forward on CTA pairs (plan._resolve_fwd_two_cta; explicit already).")
    # aligned varlen (seqused_q / seqused_k documents in place): scheduling only, O / LSE / dK / dV bitwise either way
    varlen_fwd_list: bool = F(True, typ=("bool",), kt="varlen_fwd_list",
        doc="Aligned varlen forward: the persistent longest-first work-list scheduler (lp_varlen.varlen_worklist, "
            "tile_scheduler.VarlenWorkListScheduler); False = one CTA per tile (SingleTileVarlenScheduler).")
    varlen_fwd_tma_o: bool = F(True, typ=("bool",), kt="varlen_fwd_tma_o",
        doc="Aligned varlen causal forward: the batched sO + TMA O epilogue (flash_fwd_sm100_fp4 varlen_tma_o; dense "
            "calls keep the correction warps' predicated O copies, measured faster there); False = those copies.")
    varlen_fwd_stage_skip: bool = F(True, typ=("bool",), kt="varlen_fwd_stage_skip",
        doc="Aligned varlen causal forward: the 1-CTA stage skip (causal_stage_skip; needs varlen_fwd_tma_o).")
    varlen_fwd_sched: str = F("auto", typ=("enum", ("auto", "static", "atomic", "clc")), parse=lambda v: v,
        kt="varlen_fwd_sched", alt="static",
        doc="Aligned varlen forward work list (varlen_fwd_list): how CTAs take its entries. 'auto' (default) = 'clc' "
            "with the just-in-time claim (varlen_fwd_sched_jit forced True) on causal calls and on dense calls with skewed "
            "document lengths (host statistic, lp_docmaps.SKEW_CLC_MIN), 'static' otherwise "
            "(resolved in fp4_bwd_autograd._aligned_fwd_sched; causal forward -1.7 .. -5.7 % on 13 of 16 cells, -0.6 .. +0.5 % on the rest). 'static' = the "
            "boustrophedon deal over a persistent grid; 'atomic' = dynamic claiming in list order by a global counter "
            "(persistent grid); 'clc' = dynamic claiming by Cluster Launch Control (one CTA per position, cancelled "
            "launches). Explicit values take varlen_fwd_sched_jit as given. O / LSE bitwise either way.")
    varlen_fwd_sched_stages: int = F(2, typ=("int", (1, 2, 3, 4)), parse=lambda v: int(v), kt="varlen_fwd_sched_stages",
        alt=1, doc="SMEM ring depth of the dynamic claiming (varlen_fwd_sched != 'static').")
    varlen_fwd_sched_jit: bool = F(False, typ=("bool",), kt="varlen_fwd_sched_jit",
        doc="Dynamic claiming (varlen_fwd_sched != 'static'): the load warp claims the next entry when it has issued "
            "the current tile's last K / V load, instead of a producer warp claiming whenever a ring slot frees.")
    varlen_fwd_sched_early: bool = F(False, typ=("bool",), kt="varlen_fwd_sched_early",
        doc="With varlen_fwd_sched_jit: the claim is made before the tile's last K / V pair's loads (their slot waits "
            "hide its latency) instead of after them.")
    varlen_l2_sections: bool = F(True, typ=("bool",), kt="varlen_l2_sections",
        doc="Aligned varlen work lists: L2 head / document sections (lp_varlen.l2_heads / l2_span); False = one "
            "section (global longest-first order).")
    # ---- backward drivers and glue ----------------------------------------------------------------------------------
    bwd_op_mode: str = F("fp4", typ=("enum", ("fp4", "fp8")), alt="fp8",
        doc="Backward route: 'fp4' (dk8) or 'fp8' (all-E4M3). The entry point (flash_attn_fp8) sets it.")
    bwd_two_cta: str = F("auto", typ=("enum", ("auto", "on", "off")), alt="off",
        doc="2-CTA backward: 'auto' (the measured rule), 'on' (when eligible), 'off' (two_cta).")
    bwd_2cta_gqa: bool = F(False, typ=("bool",), kt="bwd_2cta_gqa", alt=True,
        doc="GQA on the 2-CTA backward (bwdk, opt-in): each pair computes one q head's dK / dV and reduce-adds them in "
            "fp32 (upstream's dKV_postprocess form; dK / dV summation order changes). With two_cta='on' / 'auto'.")
    det_two_cta: Optional[bool] = F(None, typ=("bool",),
        doc="Deterministic calls' 1-CTA / 2-CTA choice (measured per route). None = the rule: "
            "2-CTA for 8-bit per-tensor operands, else only causal s_k >= 32768.")
    bwd_dq_f16: bool = F(True, typ=("bool",), parse=_p_strict01("HostSpec.bwd_dq_f16"),
        kt="bwd_dq_f16", doc="f16 dQ partials (half the reduce-add bytes) on every low-precision backward.")
    varlen_bwd_list: bool = F(True, typ=("bool",), kt="varlen_bwd_list",
        doc="Aligned varlen 1-CTA causal backward, non-deterministic: the document-aware longest-first key-tile list "
            "(lp_varlen.varlen_worklist 'bwd'); False = the plain (key tile, head) grid. Deterministic calls keep "
            "their scheduler.")
    dq_post: str = F("cute", typ=("enum", ("cute", "torch")), parse=lambda v: v,
        kt="bwd_fp4_dq_post", alt="torch",
        doc="1-CTA dQ postprocess: 'cute' kernel or the torch expression ('cute' on any value other than 'cute' "
            "is NOT taken: the reader tests == 'cute').")
    dq_zero: str = F(DQ_ZERO_DEFAULT, typ=("enum", ("fused", "pool", "fill")), parse=lambda v: v, kt="two_cta8_dq_zero",
        alt="pool",
        doc="dQ accumulator zeroing: 'fused' (by the dO prologue / postprocess), 'pool', 'fill'.")
    save_lean: bool = F(False, typ=("bool",),
        doc="Do not keep the bf16 Q / K / V for the backward (save_bf16_inputs=False).")


SPECS = (FwdKernelSpec, Bwd1Spec, Bwd2Spec, BwdQSpec, QuantSpec, HostSpec)


def resolved_key(obj, names) -> tuple:
    """The named constexprs read off a CONSTRUCTED kernel, i.e. after any ``__init__`` override (a compile-key item next
    to ``spec.key()``: the values the constructor settled on, e.g. a rule applied to a None field). An absent attribute
    contributes None rather than raising, so adding a name cannot break a kernel that does not have it yet."""
    out = []
    for n in names:
        v = getattr(obj, n, None)
        out.append((n, v if isinstance(v, (bool, int, float, str, type(None))) else str(v)))
    return tuple(out)


@dataclass(frozen=True)
class PlanSpecs:
    """Every spec one resolved plan runs with, built once by the plan (lowprec.plan) and handed to the route bodies, the
    drivers and the kernels as explicit arguments. fwd / quant / host on every route; bwd1 and bwd2 on the E4M3-operand
    backward routes (bwd2 only where its operand scaling exists on the 2-CTA kernel: per_tensor / blk32x128), bwdq on the
    all-4-bit route. A spec the route does not run is None."""
    fwd: FwdKernelSpec
    quant: QuantSpec
    host: HostSpec
    bwd1: Optional[Bwd1Spec] = None
    bwd2: Optional[Bwd2Spec] = None
    bwdq: Optional[BwdQSpec] = None

    def key(self) -> tuple:
        return tuple(None if s is None else s.key() for s in (self.fwd, self.quant, self.host, self.bwd1, self.bwd2,
                                                              self.bwdq))


# ================================================================================================================
# KernelTuning -> spec fields
# ================================================================================================================

def _build_kt_map() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    out: Dict[str, list] = {}
    for cls in SPECS:
        for f in fields(cls):
            kt = f.metadata["kt"]
            if kt is not None:
                out.setdefault(kt, []).append((cls.__name__, f.name))
    return {k: tuple(v) for k, v in out.items()}


# KernelTuning field -> ((spec class, spec field), ...). Several targets = the same value read by two kernels, or by a
# kernel and the host (the plan writes every target from the one field).
KT_TO_SPEC: Dict[str, Tuple[Tuple[str, str], ...]] = _build_kt_map()

def _value_str(value, kind) -> str:
    """A KernelTuning value as the string spelling the field parsers take ("1" / "0" for a bool)."""
    if kind == "bool":
        return "1" if value else "0"
    return str(value)


def apply_kernel_tuning(spec, tuning) -> Any:
    """``spec`` with every non-None KernelTuning field that maps into its class copied in, converted to the spec field's
    type by the field's parser (e.g. a bool KernelTuning field into an int spec field, "auto" / "0" / "1" into
    None / False / True, a register budget string into a tuple)."""
    from .options import TUNING_FIELDS
    cls = type(spec)
    by_name = {f.name: f for f in fields(cls)}
    kw = {}
    for kt_name, value in tuning.non_default().items():
        for owner, fname in KT_TO_SPEC.get(kt_name, ()):
            if owner != cls.__name__:
                continue
            f = by_name[fname]
            parse = f.metadata["parse"]
            kw[fname] = parse(_value_str(value, TUNING_FIELDS[kt_name])) if parse is not None else value
    return dataclasses.replace(spec, **kw) if kw else spec

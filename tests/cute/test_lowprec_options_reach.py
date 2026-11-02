"""Every public option and every KernelTuning field reaches the kernel that runs.

The chain is: options / KernelTuning -> the resolved plan (resolve_fp4 / resolve_fp8) -> plan.specs (or, for the few
options that are explicit driver arguments, plan.fwd / plan.bwd) -> the kernel constructor, which reads its spec and
nothing else -> the built kernel's attributes. This file checks each link:

  * every options field (both entries) moves the spec field / plan field it documents, or is refused with its rule
    (CPU, meta tensors; the table must name every field);
  * every KernelTuning field, given through options.tuning, lands in the spec field(s) it maps to (CPU);
  * each kernel constructor, given a spec built from a plan plus one KernelTuning / option value, resolves exactly the
    spec's value (with the documented rules for None) -- pure Python, CPU, no compile;
  * the host-side readers of QuantSpec / HostSpec / BwdQSpec agree with the spec they are given (CPU);
  * on an SM100 GPU, a forward option lands in the forward kernel built, a backward option in the backward kernel
    built on autograd's backward thread, and two_cta="off" keeps the 2-CTA kernel out.
"""
import threading
from dataclasses import fields

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions as O4
from flash_attn.cute.lowprec import FP8AttnOptions as O8
from flash_attn.cute.lowprec import KernelTuning, resolve_fp4, resolve_fp8
from flash_attn.cute.lowprec import specs as S
from flash_attn.cute.lowprec.options import TUNING_FIELDS

_SLOT = {"FwdKernelSpec": "fwd", "QuantSpec": "quant", "HostSpec": "host", "Bwd1Spec": "bwd1", "Bwd2Spec": "bwd2",
         "BwdQSpec": "bwdq"}
ALL4 = dict(pv_precision="fp4", backward="fp4")


def _qkv(b=1, s=1024, h=2, d=128, rg=True):
    return [torch.empty(b, s, h, d, dtype=torch.bfloat16, device="meta").requires_grad_(rg) for _ in range(3)]


# named calls: (entry, keyword arguments, options fields, tensor arguments)
CALLS = {
    "fp4": ("fp4", {}, {}, {}),
    "fp4_causal": ("fp4", dict(causal=True), {}, {}),
    "fp4_1cta": ("fp4", dict(causal=True), dict(two_cta="off"), {}),
    "fp4_pt": ("fp4", dict(causal=True), dict(operand_scaling="per_tensor"), {}),
    "fp4_tl": ("fp4", dict(causal=True, qk_hadamard=False), dict(ds_dq_rowlocal="all"), {}),
    "fp4_d256": ("fp4", {}, dict(operand_scaling="per_tensor", two_cta="off"), dict(d=256)),
    "fp8": ("fp8", {}, {}, {}),
    "fp8_1cta": ("fp8", dict(causal=True), dict(two_cta="off"), {}),
    "fp8_pt": ("fp8", dict(causal=True), dict(operand_scaling="per_tensor"), {}),
    "fp8_tl": ("fp8", dict(causal=True, qk_hadamard=False), dict(ds_dq_rowlocal="all"), {}),
    "all4": ("fp4", dict(ALL4), {}, {}),
    "all4_causal": ("fp4", dict(causal=True, **ALL4), {}, {}),
    "all4_key_local": ("fp4", dict(causal=True, causal_safety="key_local", **ALL4), {}, {}),
    "optA": ("fp4", dict(pv_precision="fp4"), {}, dict(rg=False)),
}


def _resolve(call, **opts):
    entry, kw, base, tkw = CALLS[call]
    resolve, O = (resolve_fp4, O4) if entry == "fp4" else (resolve_fp8, O8)
    return resolve(*_qkv(**tkw), options=O(**{**base, **opts}), **kw)


def _at(plan, path):
    obj = plan
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


# ----------------------------------------------------------------------------------------------------------------
# every options field
# ----------------------------------------------------------------------------------------------------------------

CHANGED = object()   # "differs from the same call without the option"
INERT = object()     # accepted and changes nothing in the shipped tree (documented in the row's comment)

# (options class, field) -> [(call, value, where, expected)]; where "refused" -> expected = the rule
REACH = {
    (O4, "operand_scaling"): [("fp4", "per_tensor", "specs.quant.operand_scaling_fp4", "per_tensor"),
                              ("fp4", "block_32x32", "specs.bwd1.op_scaling", "blk32"),
                              ("fp4", "token_block", "specs.bwd2.op_scaling", "blk32x128"),
                              ("fp4_tl", "token_local", "fwd.operand_scaling", "token_local")],
    (O4, "per_tensor_scale"): [("all4", "power_of_two", "specs.quant.per_tensor_pow2", True)],
    (O4, "ds_scaling"): [("fp4", "block_32x32", "bwd.ds_scaling", "block_32x32"), ("fp4", "per_tensor", "refused", 33)],
    (O4, "ds_dq_rowlocal"): [("fp4_causal", "diagonal", "bwd.ds_dq_rowlocal", "diagonal"),   # the drivers' argument
                             ("fp4_causal", "all", "bwd.ds_dq_rowlocal", "all")],
    (O4, "fp4_ds_multiplier"): [("all4", "global", "specs.quant.fp4q_ds_scale", "global")],
    (O4, "p_scaling"): [("fp4_pt", "block_1x32", "specs.bwd1.p_scaling_fp4", "mx32")],
    (O4, "p_frame"): [("optA", "single_level", "specs.fwd.fp4_pv_two_level", False), ("optA", "per_tile", "refused", 23)],
    (O4, "p_hat_frame"): [("fp4", "row", "specs.host.fwd4_row_p", True)],
    (O4, "softmax_exp"): [("all4", "exp2", "specs.fwd.direct_p", False), ("all4", "exp2", "specs.bwdq.exp2", True)],
    (O4, "k_channel"): [("all4_causal", "cut", "specs.quant.vtok_diag_k", "cut")],
    (O4, "q_block_2d"): [("fp4", True, "specs.quant.q_block_2d", True)],
    (O4, "v_hadamard"): [("all4_key_local", True, "specs.quant.vtok_hadv", True)],
    (O4, "all_fp4_stabiliser"): [("all4", "rtn", "specs.quant.fp4all_stab", "rtn"),
                                 ("all4", "eden_rht", "specs.quant.fp4all_stab", "eden_rht")],
    (O4, "sr_ds"): [("fp4", True, "specs.bwd1.ds_sr", True)],
    (O4, "sr_dk_operand"): [("all4", False, "specs.bwdq.dk_sr", False)],
    (O4, "sr_dv_operand"): [("all4", True, "specs.bwdq.dv_sr", True)],
    (O4, "fp4_backward_dkdv"): [("all4", "fp4", "specs.bwdq.dkv", "fp4")],
    (O4, "dq_operand"): [("all4", "e4m3", "refused", "E16")],
    (O4, "two_cta"): [("fp4", "off", "specs.host.bwd_two_cta", "off"), ("fp4", "on", "specs.host.bwd_two_cta", "on")],
    (O4, "forward_two_cta"): [("fp4", "on", "specs.host.fwd_two_cta", True)],
    (O4, "seqused_k_group_aligned"): [("fp4", True, "refused", 26)],
    (O4, "save_bf16_inputs"): [("fp4_pt", False, "specs.host.save_lean", True)],
    (O4, "p_consistency"): [("optA", "none", "bwd.exp_mix", CHANGED)],   # the forward's value function follows it
    # option A is inference only (rule 36): nothing reads its dO rounding, so an explicit "sr" is refused, never inert
    (O4, "do_rounding"): [("optA", "sr", "refused", "E1"), ("fp4", "sr", "refused", "E1")],
    (O4, "exp_mix"): [("optA", "none", "bwd.exp_mix", "none"), ("optA", "MMMMMMMM", "bwd.exp_mix", "MMMMMMMM")],
    (O8, "operand_scaling"): [("fp8", "per_tensor", "specs.quant.operand_scaling_fp8", "per_tensor"),
                              ("fp8", "block_32x32", "specs.bwd1.op_scaling", "blk32"),
                              ("fp8", "token_block", "specs.bwd2.op_scaling", "blk32x128"),
                              ("fp8_tl", "token_local", "fwd.operand_scaling", "token_local")],
    (O8, "per_tensor_scale"): [("fp8_pt", "power_of_two", "specs.quant.per_tensor_pow2", True)],
    (O8, "ds_scaling"): [("fp8", "row_1x32", "specs.bwd1.ds_scaling_fp8", "mx32"),
                         ("fp8_pt", "row_local", "specs.bwd1.ds_scaling_fp8", "row"),
                         ("fp8_pt", "block_32x64", "specs.bwd2.ds_scaling_fp8", "blk32x64"),
                         ("fp8_pt", "tile_128", "specs.bwd2.ds_scaling_fp8", "tile128"),
                         ("fp8", "per_tensor", "refused", 33)],
    (O8, "ds_dq_rowlocal"): [("fp8", "all", "bwd.ds_dq_rowlocal", "all")],
    (O8, "per_tensor_ds_source"): [("fp8", "recompute", "refused", 33), ("fp8", "forward_bound", "refused", 33)],
    (O8, "p_scaling"): [("fp8", "block_1x32", "specs.bwd1.p_scaling_fp8", "mx32"),
                        ("fp8", "block_1x32", "specs.bwd2.p_scaling_fp8", "mx32")],
    (O8, "grad_format"): [("fp8_pt", "e5m2", "specs.quant.grad_format", "e5m2")],
    (O8, "softmax_exp"): [("fp8", "mitchell", "refused", 21)],
    (O8, "forward_kernel"): [("fp8_pt", "upstream", "specs.host.fp8_forward_kernel", "upstream")],
    (O8, "two_cta"): [("fp8", "off", "specs.host.bwd_two_cta", "off")],
    (O8, "forward_two_cta"): [("fp8", "on", "specs.host.fwd_two_cta", True)],
    (O8, "p_hat_frame"): [("fp8", "group", "specs.host.fwd8_row_p", False)],
}
_ROWS = [(cls, name, row) for (cls, name), rows in REACH.items() for row in rows]


def test_every_option_field_is_covered():
    for cls in (O4, O8):
        names = {f.name for f in fields(cls)} - {"tuning"}
        covered = {n for (c, n) in REACH if c is cls}
        assert covered == names, (cls.__name__, sorted(names ^ covered))


@pytest.mark.parametrize("cls,name,row", _ROWS, ids=[f"{c.__name__}.{n}={r[1]!r}@{r[0]}" for c, n, r in _ROWS])
def test_option_reaches_its_spec(cls, name, row):
    call, value, where, expected = row
    assert (CALLS[call][0] == "fp4") == (cls is O4), "the row's call must use the option's entry"
    if where == "refused":
        with pytest.raises((ValueError, NotImplementedError)) as e:
            _resolve(call, **{name: value})
        assert e.value.rule == expected, (e.value.rule, str(e.value)[:200])
        return
    base, p = _resolve(call), _resolve(call, **{name: value})
    if expected is INERT:
        assert p.specs == base.specs and p.fwd == base.fwd
        return
    got = _at(p, where)
    if expected is CHANGED:
        assert got != _at(base, where), (where, got)
    else:
        assert got == expected, (where, got, expected)
        # a value that is not the call's default moves the plan (the specs or the explicit driver arguments)
        if _at(base, where) != expected:
            assert (p.specs.key(), p.fwd, p.bwd) != (base.specs.key(), base.fwd, base.bwd)


# ----------------------------------------------------------------------------------------------------------------
# every KernelTuning field, through options.tuning
# ----------------------------------------------------------------------------------------------------------------

def _kt_values(name):
    kind = TUNING_FIELDS[name]
    if kind == "bool":
        return True, False
    tag = kind[0]
    if tag == "ints":
        return kind[1][0], kind[1][1]
    if tag == "int":
        return kind[1] + 1, kind[1] + 2
    if tag == "float":
        lo = 0.0 if kind[1] is None else kind[1]
        return lo + 1.0, lo + 2.0
    if tag == "enum":   # the last two: an enum's first value may be "auto" (= the measured rule, None in the spec)
        return kind[1][-2], kind[1][-1]
    return {"two_cta8_regs": ("136x144x88", "128x136x96"), "fwd_exp": ("sfi", "gmx"),
            "fwd_ptxas": ("--register-usage-level=3", "--register-usage-level=7")}[name]


# the first call that resolves both values and builds every target spec (the plan refuses a field the resolved kernel
# cannot compile: rule E11 / E15)
_KT_CALLS = ("fp4", "fp4_1cta", "fp8", "fp8_pt", "all4")


@pytest.mark.parametrize("kt", sorted(TUNING_FIELDS))
def test_kernel_tuning_reaches_its_spec(kt):
    v1, v2 = _kt_values(kt)
    targets = S.KT_TO_SPEC[kt]
    tried = []
    for call in _KT_CALLS:
        try:
            p1 = _resolve(call, tuning=KernelTuning(**{kt: v1}))
            p2 = _resolve(call, tuning=KernelTuning(**{kt: v2}))
        except (ValueError, NotImplementedError) as e:
            tried.append((call, getattr(e, "rule", None)))
            continue
        if any(getattr(p1.specs, _SLOT[o]) is None for o, _ in targets):
            tried.append((call, "no target spec"))
            continue
        for owner, fname in targets:
            a, b = (getattr(getattr(p.specs, _SLOT[owner]), fname) for p in (p1, p2))
            assert a != b, (kt, owner, fname, a)
            assert None not in (a, b), (kt, owner, fname)   # a given field is never "the measured rule"
        assert p1.tuning == KernelTuning(**{kt: v1}) and p1.specs.key() != p2.specs.key()
        return
    pytest.fail(f"{kt}: no call resolves both values: {tried}")


def test_unknown_option_is_refused():
    with pytest.raises(TypeError):
        O8(not_an_option=1)
    with pytest.raises(TypeError):
        O4(not_an_option=1)
    with pytest.raises(TypeError):
        KernelTuning(not_a_field=1)
    with pytest.raises(ValueError):
        O8(tuning={"fwd_max_tmem": False})   # tuning is a KernelTuning, never a mapping of names


# ----------------------------------------------------------------------------------------------------------------
# the kernel constructors resolve exactly their spec (CPU: pure Python constructors, nothing compiled)
# ----------------------------------------------------------------------------------------------------------------

def _kernels():
    try:
        import cutlass  # noqa: F401
        from flash_attn.cute.flash_bwd_sm100_2cta8 import FlashAttentionBackwardSm100TwoCta8
        from flash_attn.cute.flash_bwd_sm100_fp4 import FlashAttentionBackwardSm100FP4
        from flash_attn.cute.flash_bwd_sm100_fp4q import FlashAttentionBackwardSm100FP4Q
        from flash_attn.cute.flash_fwd_sm100_fp4 import FlashAttentionForwardSm100
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CuTe DSL not importable: {e}")
    return FlashAttentionForwardSm100, FlashAttentionBackwardSm100FP4, FlashAttentionBackwardSm100TwoCta8, \
        FlashAttentionBackwardSm100FP4Q


def _or(v, rule):
    return rule if v is None else v


def _kt_candidates(kt, alt):
    """KernelTuning values of field ``kt`` that could spell the spec value ``alt``."""
    kind = TUNING_FIELDS[kt]
    if kind == "bool":
        return [True, False]
    if kind[0] in ("ints", "enum"):
        return list(kind[1])
    if kind[0] == "re":
        if isinstance(alt, str):   # a free-form string field (fwd_exp / fwd_ptxas): the value itself
            return [alt]
        return ["x".join(str(x) for x in alt)] if isinstance(alt, tuple) else []
    return [alt]


def _variants(spec):
    """(field, spec) for the plan's spec and for every field moved to its alternative value: the KernelTuning-backed
    fields through KernelTuning (lowprec.specs.apply_kernel_tuning, the plan's own last step), every other field by
    value (a kernel-level probe of what an option / route sets)."""
    out = [(None, spec)]
    for f in fields(spec):
        alt, kt = f.metadata["alt"], f.metadata["kt"]
        if kt is not None:
            seen = {getattr(spec, f.name)}
            for v in _kt_candidates(kt, alt):
                try:
                    s = S.apply_kernel_tuning(spec, KernelTuning(**{kt: v}))
                except ValueError:
                    continue
                if getattr(s, f.name) not in seen:
                    seen.add(getattr(s, f.name))
                    out.append((f.name, s))
            assert len(seen) > 1, (kt, f.name)   # KernelTuning moves every field it maps to
            if alt in seen:
                continue
        out.append((f.name, spec.replace(**{f.name: alt})))   # the spec's own alternative (e.g. not a KT spelling)
    return out


# ---- Bwd2Spec ----------------------------------------------------------------------------------------------------

def _bwd2_check(spec, k, det):
    eff_ds = "blk32" if k.mxds_fp4 else spec.ds_scaling_fp8
    obs = {"ds_scaling_fp4": (k.ds_scaling_fp4, spec.ds_scaling_fp4),
           "p_scaling_fp8": (k.p_scaling, spec.p_scaling_fp8),
           "op_scaling": ("blk32x128" if k.mxop_tb else "per_tensor", spec.op_scaling),
           "warp_sync": (k.warp_sync, spec.warp_sync),
           "mx_xw": (k.mx_xw, _or(spec.mx_xw, S.Bwd2Spec.rule_mx_xw(eff_ds)) if k.mxds else ""),
           "causal_diag": (k.causal_diag_only, spec.causal_diag),
           "ld_rep": (k.ld_rep, spec.ld_rep),
           "q_stage": (k.Q_stage, spec.q_stage),
           "dq_ncol": (k.dQ_reduce_ncol, _or(spec.dq_ncol, S.Bwd2Spec.rule_dq_ncol(det))),
           "dq_stages": (k.sdQaccum_stage, _or(spec.dq_stages, S.Bwd2Spec.rule_dq_stages(det)))}
    if not k.mxds_fp4:
        obs["ds_scaling_fp8"] = (k.ds_scaling, spec.ds_scaling_fp8)
    if spec.regs is not None:
        obs["regs"] = ((k.num_regs_reduce, k.num_regs_compute, k.num_regs_load), spec.regs)
    return obs


# (route: NVFP4 S on the 4-bit route, the spec's base values beyond the plan's)
_BWD2_BASES = [(True, {}), (False, {}), (False, dict(ds_scaling_fp8="per_tensor", op_scaling="per_tensor")),
               (True, dict(ds_scaling_fp4="per_tensor", op_scaling="per_tensor"))]


def test_bwd2_constructor_resolves_the_spec():
    _, _, B2, _ = _kernels()
    plan_spec = {True: _resolve("fp4").specs.bwd2, False: _resolve("fp8").specs.bwd2}
    checked, refused = set(), set()
    for nvfp4_s, base in _BWD2_BASES:
        for fname, spec in _variants(plan_spec[nvfp4_s].replace(**base)):
            for det in (False, True):
                try:
                    k = B2(128, 128, is_causal=True, tile_m=128, tile_n=128, cluster_size=2, use_2cta_instrs=True,
                           deterministic=det, nvfp4_s=nvfp4_s, lpt_l2=False, spec=spec)
                    k._setup_attributes()
                except (AssertionError, ValueError):
                    refused.add(fname)
                    continue
                for name, (got, want) in _bwd2_check(spec, k, det).items():
                    assert got == want, (fname, nvfp4_s, base, det, name, got, want)
                checked.add(fname)
    assert {"ld_rep", "mx_xw", "q_stage", "regs", "warp_sync", "op_scaling", "dq_ncol"} <= checked
    # every alternative value constructs in at least one context
    assert refused <= checked, sorted(refused - checked)


# ---- Bwd1Spec ----------------------------------------------------------------------------------------------------

def _bwd1_check(spec, k):
    obs = {"fp8": (k.fp8, spec.fp8), "compute_wgs": (k.num_compute_wgs, spec.compute_wgs),
           "ds_fold": (k.ds_fold, spec.ds_fold), "ds_sr": (k.ds_sr, spec.ds_sr),
           "ds_scaling_fp4": (k.ds_scaling_fp4, spec.ds_scaling_fp4),
           "p_scaling_fp4": (k.p_scaling_fp4, spec.p_scaling_fp4), "op_scaling": (k.op_scaling, spec.op_scaling),
           "ld_rep": (k.tmem_ld_rep, _or(spec.ld_rep, S.Bwd1Spec.rule_ld_rep(k.fp8, k.mxds_fp4, k.mxp_fp4, k.mxop))),
           "sload_early": (k.s_loaded_early, spec.sload_early), "warp_sync": (k.warp_sync, spec.warp_sync),
           "causal_diag": (k.causal_diag_only, spec.causal_diag),
           "dp_ld2": (k.dp_ld2, spec.dp_ld2 or k.mxds_row),
           "p_scaling_fp8": (k.p_scaling, spec.p_scaling_fp8), "do_stage": (k.dO_stage, spec.do_stage),
           "reduce_chunked": (k.reduce_chunked, spec.reduce_chunked or k.num_compute_wgs == 4),
           "dq_ncol": (k.dQ_reduce_ncol, _or(spec.dq_ncol, S.Bwd1Spec.rule_dq_ncol(k.num_compute_wgs))),
           "dq_stages": (k.sdQaccum_stage, _or(spec.dq_stages, S.Bwd1Spec.rule_dq_stages(k.num_compute_wgs))),
           "pp": (k.pp, bool(spec.pp and k.tile_hdim == 64 and k.ds_scaling in ("per_tensor", "blk32") and not k.mxp
                             and k.op_scaling in ("per_tensor", "blk32x128") and k.num_compute_wgs == 2
                             and k.tmem_ld_rep == 32))}
    if not k.mxds_fp4:
        obs["ds_scaling_fp8"] = (k.ds_scaling, spec.ds_scaling_fp8)
    if k.num_compute_wgs == 4:
        obs["regs4"] = ((k.num_regs_compute, k.num_regs_reduce, k.num_regs_load), spec.regs4)
    return obs


# (the call whose plan gives the base spec, the spec's base values beyond the plan's)
_BWD1_BASES = [("fp4", {}), ("fp4", dict(ds_scaling_fp4="per_tensor", op_scaling="per_tensor")),
               ("fp8", {}), ("fp8", dict(ds_scaling_fp8="per_tensor", op_scaling="per_tensor")),
               ("fp4", dict(ds_scaling_fp4="per_tensor", op_scaling="per_tensor", compute_wgs=4))]


def test_bwd1_constructor_resolves_the_spec():
    _, B1, _, _ = _kernels()
    checked, refused = set(), set()
    for call, base in _BWD1_BASES:
        for fname, spec in _variants(_resolve(call).specs.bwd1.replace(**base)):
            for d in (128, 64):
                try:
                    k = B1(d, d, is_causal=True, tile_m=128, tile_n=128, spec=spec)
                    k._setup_attributes()
                except (AssertionError, ValueError):
                    refused.add(fname)
                    continue
                for name, (got, want) in _bwd1_check(spec, k).items():
                    assert got == want, (fname, call, base, d, name, got, want)
                checked.add(fname)
    assert {"ld_rep", "regs4", "pp", "dq_ncol", "compute_wgs"} <= checked
    assert refused <= checked, sorted(refused - checked)


# ---- BwdQSpec ----------------------------------------------------------------------------------------------------

def _bwdq_check(spec, k):
    dkv8 = spec.dkv == "e4m3"
    mx = "per_tensor" if not k.mxds else ("blk32" if k.mxds_blk32 else "mx32")
    return {"exp2": (k.fp4q_exp2, spec.exp2),
            "exp2_emulation_every": (k.e2e_freq, _or(spec.exp2_emulation_every,
                                                     S.BwdQSpec.rule_exp2_emulation_every())),
            "dq8": (k.dq8, spec.dq8 and dkv8), "sr_dither": (k.sr_dither, spec.sr_dither),
            "mxds": (mx, _or(spec.mxds, S.BwdQSpec.rule_mxds(k.diag_kr)) if dkv8 else "per_tensor"),
            "dk_sr": (k.dk_sr, spec.dk_sr and dkv8), "dv_sr": (k.dv_sr, spec.dv_sr and dkv8),
            "dkv": (k.dkv8, dkv8)}


# the plan's spec, and the options that move its base (softmax_exp="exp2", fp4_backward_dkdv="fp4") plus a KernelTuning
_BWDQ_BASES = [{}, dict(softmax_exp="exp2"), dict(fp4_backward_dkdv="fp4"),
               dict(tuning=KernelTuning(fp4_exp2_emulation_every=5))]


def test_bwdq_constructor_resolves_the_spec():
    _, _, _, BQ = _kernels()
    for opts in _BWDQ_BASES:
        for fname, spec in _variants(_resolve("all4", **opts).specs.bwdq):
            k = BQ(128, is_causal=True, dkv=spec.dkv, spec=spec)
            k._mxds_init()
            for name, (got, want) in _bwdq_check(spec, k).items():
                assert got == want, (opts, fname, name, got, want)


# ---- FwdKernelSpec (the __init__ reads; _setup_attributes needs the __call__ dtypes) ------------------------------

def _fwd_is_sm103():
    from flash_attn.cute.flash_fwd_sm100_fp4 import _is_sm103
    return _is_sm103()


def _fwd_check(spec, k, causal, e2m1_pv):
    return {
        "q_stage": (k.q_stage, _or(spec.q_stage, S.FwdKernelSpec.rule_q_stage(False, 128))),
        "corr_rescale_batch": (k.corr_rescale_batch, spec.corr_rescale_batch),
        "fp4_group_max_first": (k.fp4_group_max_first, _or(spec.fp4_group_max_first,
                                                            S.FwdKernelSpec.rule_group_max_first(e2m1_pv))),
        "exp2_emulation_every": (k.fp4_log2_e2e_freq, _or(spec.exp2_emulation_every,
                                                           S.FwdKernelSpec.rule_exp2_emulation_every(e2m1_pv))),
        "mxfp8_use_inline_ptx": (k.debug_mxfp8_use_inline_ptx, spec.mxfp8_use_inline_ptx),
        "fp8_pv_use_explicit_pack": (k.fp8_pv_use_explicit_pack, spec.fp8_pv_use_explicit_pack),
        "e2e_freq": (k._e2e_freq_override, spec.e2e_freq),
        "e2e_start_frg": (k.e2e_start_frg, _or(spec.e2e_start_frg, k._tune.get("e2e_start_frg", 0))),
        "force_e2e": ((k._force_e2e_from_env, k.force_e2e if k._force_e2e_from_env else None),
                      (spec.force_e2e is not None, spec.force_e2e)),
        "softmax_fused": (k.softmax_fused_mode, _or(spec.softmax_fused, S.FwdKernelSpec.rule_softmax_fused(causal))),
        "mask_loops": (k.mask_loops, spec.mask_loops),
        "max_tmem": (k._max_tmem_knob, None if spec.max_tmem is None else ("1" if spec.max_tmem else "0")),
        "fp8_pv_zero_fill_regs": (k.fp8_pv_zero_fill_regs, spec.fp8_pv_zero_fill_regs),
        "fp4_pv_log2_quant": (k.fp4_pv_log2_quant, spec.fp4_pv_log2_quant),
        "fp4_pv_two_level": (k.fp4_pv_two_level, spec.fp4_pv_two_level),
        "fp4_pv_tile_frame": (k.fp4_pv_tile_frame, spec.fp4_pv_tile_frame),
        "fp4_pv_tile_frame_drop": (k.fp4_pv_tile_frame_drop, spec.fp4_pv_tile_frame_drop if spec.fp4_pv_tile_frame
                                   else 0.0),
        "ldred_rowmax": (k.use_ldred_rowmax, _or(spec.ldred_rowmax, S.FwdKernelSpec.rule_ldred_rowmax(_fwd_is_sm103()))),
        "rescale_threshold_rule": (float(k.fp4_rescale_threshold_default),
                                   S.FwdKernelSpec.rule_fp4_pv_rescale_threshold(e2m1_pv)),
    }


# the plans whose forward spec is the base (the 4-bit P^ forwards with Direct-P / exp2, option A's), plus the per-tile
# frame as a kernel-level base (its public value is refused, rule 23; the kernel still takes it with threshold 0)
_FWD_BASES = [("fp4", {}, {}), ("all4", {}, {}), ("all4", dict(softmax_exp="exp2"), {}), ("optA", {}, {}),
              ("fp4", {}, dict(fp4_pv_tile_frame=True, fp4_pv_rescale_threshold=0.0))]


def test_fwd_constructor_resolves_the_spec():
    FW, _, _, _ = _kernels()
    import cutlass
    pv_variants = [dict(), dict(sf_dtype_pv=cutlass.Float8E8M0FNU, sf_vec_size_pv=32, pv_elem_width=8)]
    n = 0
    for call, opts, base in _FWD_BASES:
        for fname, spec in _variants(_resolve(call, **opts).specs.fwd.replace(**base)):
            for causal in (False, True):
                for pvv in pv_variants:
                    e2m1 = not pvv
                    try:
                        k = FW(128, 128, is_causal=causal, sf_dtype=cutlass.Float8E4M3FN, sf_vec_size=16, spec=spec,
                               **pvv)
                        split = FW.resolve_qk_split_pipe(True, spec)
                    except ValueError:
                        # the per-tile frame without threshold 0 (or on an E4M3 P.V): refused by the constructor
                        assert spec.fp4_pv_tile_frame and (_or(spec.fp4_pv_rescale_threshold,
                            S.FwdKernelSpec.rule_fp4_pv_rescale_threshold(e2m1)) != 0.0 or not spec.fp4_pv_two_level)
                        continue
                    for name, (got, want) in _fwd_check(spec, k, causal, e2m1).items():
                        assert got == want, (call, opts, base, fname, causal, e2m1, name, got, want)
                    assert split == _or(spec.qk_split_pipe, True)
                    n += 1
    assert n > 0


# ----------------------------------------------------------------------------------------------------------------
# the host-side readers agree with the spec they are given
# ----------------------------------------------------------------------------------------------------------------

def _host_observers():
    try:
        from flash_attn.cute import fp4_bwd_autograd as ag
        from flash_attn.cute import interface_bwd_fp4 as ibwd
        from flash_attn.cute import interface_bwd_fp4q as ibq
        from flash_attn.cute import pv4_route as pv4r
        from flash_attn.cute import qk_hadamard as had
        from flash_attn.cute.dq_postprocess_cute import dq_zero_mode
    except Exception as e:  # pragma: no cover
        pytest.skip(f"drivers not importable: {e}")

    class _Q:
        shape = (1, 128, 1, 128)

    return {
        ("QuantSpec", "token_block_forward"): lambda q, h, b: (ag.tb_fold_fwd(128, qspec=q),
                                                               q.token_block_forward == "fold"),
        ("QuantSpec", "q_block_2d"): lambda q, h, b: (ag.q_blk2d(qspec=q), q.q_block_2d),
        ("QuantSpec", "fused_ds_bound"): lambda q, h, b: (ag.fused_ds_bound_ok(_Q(), qspec=q),
                                                          q.fused_ds_bound and q.d_from_do8),
        ("QuantSpec", "qk_hadamard"): lambda q, h, b: (had.enabled(qspec=q), q.qk_hadamard),
        ("QuantSpec", "qk_hadamard_e4m3"): lambda q, h, b: (
            (had.e4m3_mode("blocks", qspec=q), had.e4m3_mode("per_tensor", qspec=q)),
            (_or(q.qk_hadamard_e4m3, S.QuantSpec.rule_qk_hadamard_e4m3("blocks")),
             _or(q.qk_hadamard_e4m3, S.QuantSpec.rule_qk_hadamard_e4m3("per_tensor")))),
        ("QuantSpec", "qk_hadamard_rotation_method"): lambda q, h, b: (had.rotation_method(qspec=q),
                                                                       q.qk_hadamard_rotation_method),
        ("QuantSpec", "grad_format"): lambda q, h, b: (ibwd.grad_fp8_format(quant=q), q.grad_format),
        ("QuantSpec", "ds_scaling_fp4"): lambda q, h, b: (ibwd.ds_blk32_fp4(quant=q), q.ds_scaling_fp4 == "blk32"),
        ("QuantSpec", "operand_scaling_fp4"): lambda q, h, b: (
            ibwd.e4_blk32_fp4(quant=q), _or(q.operand_scaling_fp4, S.OP_SCALING_DEFAULT) != "per_tensor"),
        ("QuantSpec", "operand_scaling_fp8"): lambda q, h, b: (
            ibwd.op_blk32_fp8(quant=q), _or(q.operand_scaling_fp8, S.OP_SCALING_DEFAULT) != "per_tensor"),
        ("QuantSpec", "causal_safe_v"): lambda q, h, b: (pv4r.pv4_vdiag(True, qspec=q), q.causal_safe_v != "never"),
        ("QuantSpec", "vtok_diag"): lambda q, h, b: (ibq._vtok_diag_knob(qspec=q), q.vtok_diag),
        ("QuantSpec", "vtok_diag_k"): lambda q, h, b: (ibq._diag_k_knob(qspec=q),
                                                       {"fold": "fold", "cut": "1", "all": "all",
                                                        "open": "0"}[q.vtok_diag_k]),
        ("QuantSpec", "fp4all_stab"): lambda q, h, b: (ibq.fp4all_stab_mode(qspec=q), q.fp4all_stab),
        ("QuantSpec", "fp4q_ds_scale"): lambda q, h, b: (ibq.fp4all_ds_scale_mode(True, True, qspec=q),
                                                         "causal" if q.fp4q_ds_scale == "auto" else q.fp4q_ds_scale),
        ("HostSpec", "bwd_dq_f16"): lambda q, h, b: (ibwd.bwd_dq_f16(host=h), h.bwd_dq_f16),
        ("HostSpec", "dq_zero"): lambda q, h, b: (dq_zero_mode(h), h.dq_zero),
        ("HostSpec", "bwd_op_mode"): lambda q, h, b: (ag.dk8_prologue_fusable(hspec=h), h.bwd_op_mode == "fp4"),
        ("BwdQSpec", "dkv"): lambda q, h, b: (ibq.fp4all_dkv_mode(bspec=b), b.dkv),
    }


def test_host_readers_resolve_the_spec():
    obs = _host_observers()
    base = _resolve("all4_causal").specs
    q0, h0, b0 = base.quant, base.host, base.bwdq
    for (owner, fname), fn in obs.items():
        cls = getattr(S, owner)
        alt = cls.__dataclass_fields__[fname].metadata["alt"]
        variants = [(q0, h0, b0)]
        if owner == "QuantSpec":
            variants.append((q0.replace(**{fname: alt}), h0, b0))
        elif owner == "HostSpec":
            variants.append((q0, h0.replace(**{fname: alt}), b0))
        else:
            variants.append((q0, h0, b0.replace(**{fname: alt})))
        for q, h, b in variants:
            got, want = fn(q, h, b)
            assert got == want, (owner, fname, got, want)


# ----------------------------------------------------------------------------------------------------------------
# end to end (SM100): the option lands in the kernel that is built, on the thread that builds it
# ----------------------------------------------------------------------------------------------------------------

_gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                          reason="SM100 kernels")


@pytest.fixture
def built(monkeypatch):
    """Every kernel object constructed during the test, with whether it was built on the main thread."""
    FW, B1, B2, _ = _kernels()
    log, main = [], threading.get_ident()
    for K in (FW, B1, B2):
        def wrapped(obj, *a, _orig=K.__init__, **kw):
            _orig(obj, *a, **kw)
            log.append((obj, threading.get_ident() == main))
        monkeypatch.setattr(K, "__init__", wrapped)
    return log


def _step(entry, options):
    from flash_attn.cute.lowprec import flash_attn_fp4, flash_attn_fp8
    f = flash_attn_fp4 if entry == "fp4" else flash_attn_fp8
    gen = torch.Generator().manual_seed(0)
    q, k, v, g = (torch.randn(1, 4096, 4, 128, generator=gen).to(torch.bfloat16).cuda() for _ in range(4))
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    o = f(qg, kg, vg, causal=False, options=options)
    torch.autograd.grad(o, (qg, kg, vg), g)
    torch.cuda.synchronize()
    plan = (resolve_fp4 if entry == "fp4" else resolve_fp8)(qg, kg, vg, causal=False, options=options)
    return plan


@_gpu
@pytest.mark.parametrize("entry,options,kernel,field,value", [
    ("fp4", O4(sr_ds=True, two_cta="off"), "FlashAttentionBackwardSm100FP4", "ds_sr", True),
    ("fp8", O8(tuning=KernelTuning(two_cta8_causal_diag=False)), "FlashAttentionBackwardSm100TwoCta8",
     "causal_diag", False),
])
def test_backward_option_reaches_backward_kernel_on_backward_thread(built, entry, options, kernel, field, value):
    plan = _step(entry, options)
    slot = "bwd1" if kernel.endswith("FP4") else "bwd2"
    assert getattr(getattr(plan.specs, slot), field) == value
    mine = [(k, on_main) for k, on_main in built if type(k).__name__ == kernel]
    assert mine, [type(k).__name__ for k, _ in built]
    for k, on_main in mine:
        assert not on_main, kernel   # built by autograd's backward thread, from the plan the forward saved
        assert getattr(k.spec, field) == value and k.spec == getattr(plan.specs, slot)


@_gpu
def test_forward_option_reaches_forward_kernel(built):
    FW = _kernels()[0]
    plan = _step("fp8", O8(tuning=KernelTuning(fwd_max_tmem=False)))
    fwd = [k for k, _ in built if isinstance(k, FW)]
    assert fwd and all(k.spec.max_tmem is False and k.spec == plan.specs.fwd for k in fwd), \
        [k.spec.max_tmem for k in fwd]
    assert all(getattr(k, "_max_tmem_knob", None) == "0" and not k.max_tmem for k in fwd), \
        [(getattr(k, "_max_tmem_knob", None), getattr(k, "max_tmem", None)) for k in fwd]


@_gpu
def test_two_cta_off_reaches_the_backward_thread(built):
    plan = _step("fp4", O4(two_cta="off"))
    assert plan.specs.host.bwd_two_cta == "off" and plan.bwd.kernel == "1cta"
    names = [(type(k).__name__, on_main) for k, on_main in built]
    assert ("FlashAttentionBackwardSm100TwoCta8", False) not in names and \
        not any(n == "FlashAttentionBackwardSm100TwoCta8" for n, _ in names), names
    one = [on_main for n, on_main in names if n == "FlashAttentionBackwardSm100FP4"]
    assert one and not any(one), names

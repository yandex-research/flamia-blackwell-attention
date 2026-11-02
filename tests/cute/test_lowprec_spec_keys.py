"""G-key for the kernel specs (flash_attn.cute.lowprec.specs), CPU only: no CUDA context, no kernel launch.

The plan builds every spec once (lowprec.plan.build_specs) from the public arguments, the options object and
KernelTuning; the drivers key their compile caches on ``spec.key()`` (by value, never by identity). The property a key
needs: two values of a setting give two keys, and the same value gives one (a dropped field would silently reuse a stale
compiled kernel). This file proves it for

1. every spec field, on the spec itself (two values -> two keys; the key carries the class and the key version);
2. every KernelTuning field, through plan.build_specs on a resolved plan (KernelTuning copied in last);
3. every spec field a public argument or an options field sets, through resolve_fp4 / resolve_fp8 (meta tensors);
4. the classification is complete: every spec field is set by KernelTuning, by an option / argument, by the route, or
   is in the documented list of fields no public input reaches (constants / constructor rules / refused values);
5. recorded goldens (when present): the compile keys recorded at the default all-4-bit cells embed exactly the spec
   keys the plan resolves today (skipped without the golden file).
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions as O4
from flash_attn.cute.lowprec import FP8AttnOptions as O8
from flash_attn.cute.lowprec import KernelTuning, resolve_fp4, resolve_fp8
from flash_attn.cute.lowprec import specs as S
from flash_attn.cute.lowprec.options import TUNING_FIELDS
from flash_attn.cute.lowprec.plan import build_specs

SPEC_FIELDS = [(cls, f.name) for cls in S.SPECS for f in fields(cls)]
_ID = lambda p: f"{p[0].__name__}.{p[1]}"  # noqa: E731
_SLOT = {"FwdKernelSpec": "fwd", "QuantSpec": "quant", "HostSpec": "host", "Bwd1Spec": "bwd1", "Bwd2Spec": "bwd2",
         "BwdQSpec": "bwdq"}


def _alt(cls, name):
    return cls.__dataclass_fields__[name].metadata["alt"]


def _default(cls, name):
    return cls.__dataclass_fields__[name].default


def _qkv(b=1, s=1024, h=2, d=128, rg=True):
    """Meta tensors: shape, dtype and requires_grad only (the plan reads metadata, nothing else)."""
    return [torch.empty(b, s, h, d, dtype=torch.bfloat16, device="meta").requires_grad_(rg) for _ in range(3)]


ALL4 = dict(pv_precision="fp4", backward="fp4")
OPTA = dict(pv_precision="fp4")

# named calls: (entry, keyword arguments, tensor arguments). options= inside the kwargs is a dict of options fields.
CALLS = {
    "fp4": ("fp4", {}, {}),
    "fp4_pt": ("fp4", dict(causal=True, options=dict(operand_scaling="per_tensor")), {}),
    "fp8": ("fp8", {}, {}),
    "fp8_pt": ("fp8", dict(causal=True, options=dict(operand_scaling="per_tensor")), {}),
    "all4": ("fp4", dict(ALL4), {}),
    "all4_causal": ("fp4", dict(causal=True, **ALL4), {}),
    "all4_inference": ("fp4", dict(ALL4), dict(rg=False)),
    "optA": ("fp4", dict(OPTA), dict(rg=False)),
}


def _resolve(call, **over):
    """Resolve a named call, with keyword arguments / options fields overridden by ``over`` (``opt_<field>``)."""
    entry, kw, tkw = CALLS[call]
    kw = dict(kw)
    opts = dict(kw.pop("options", {}))
    for k, v in over.items():
        if k.startswith("opt_"):
            opts[k[4:]] = v
        else:
            kw[k] = v
    resolve, O = (resolve_fp4, O4) if entry == "fp4" else (resolve_fp8, O8)
    return resolve(*_qkv(**tkw), options=O(**opts), **kw)


def _get(plan, owner, name):
    s = getattr(plan.specs, _SLOT[owner])
    assert s is not None, f"{owner} is not built on route {plan.route}"
    return getattr(s, name)


# ----------------------------------------------------------------------------------------------------------------
# 1. every spec field: two values -> two keys, the same value -> one
# ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls,name", SPEC_FIELDS, ids=[_ID(p) for p in SPEC_FIELDS])
def test_two_values_two_keys(cls, name):
    d, a = _default(cls, name), _alt(cls, name)
    assert d != a, "alt must differ from the default"
    s0, s1 = cls(), cls().replace(**{name: a})
    assert s0.key() != s1.key()
    assert s0 != s1 and hash(s0.key()) is not None
    # the same value -> the same key (and one cache entry)
    assert cls().key() == s0.key() and cls().replace(**{name: a}).key() == s1.key()
    cache = {}
    for s in (s0, s1, cls(), cls().replace(**{name: a})):
        cache.setdefault(s.key(), s)
    assert len(cache) == 2
    # the key carries the class: equal field values in two spec classes never share an entry
    assert s0.key()[0] == cls.__name__ and s0.key()[1] == S.SPEC_KEY_VERSION
    # and the plan's composite key (PlanSpecs.key) separates the two as well
    slot = _SLOT[cls.__name__]
    base = _resolve("all4" if slot == "bwdq" else "fp4").specs
    p0 = S.PlanSpecs(**{**{f.name: getattr(base, f.name) for f in fields(S.PlanSpecs)}, slot: s0})
    p1 = S.PlanSpecs(**{**{f.name: getattr(base, f.name) for f in fields(S.PlanSpecs)}, slot: s1})
    assert p0.key() != p1.key()


@pytest.mark.parametrize("cls,name", SPEC_FIELDS, ids=[_ID(p) for p in SPEC_FIELDS])
def test_field_documented_and_validated(cls, name):
    m = cls.__dataclass_fields__[name].metadata
    assert m["doc"] and len(m["doc"]) > 20, "every field documents what it controls and its default rule"
    with pytest.raises(ValueError):
        bad = object()
        cls().replace(**{name: bad})


def test_plan_specs_key_separates_present_and_absent_specs():
    p = _resolve("fp4").specs
    no_bwd2 = S.PlanSpecs(fwd=p.fwd, quant=p.quant, host=p.host, bwd1=p.bwd1, bwd2=None, bwdq=None)
    assert p.bwd2 is not None and p.key() != no_bwd2.key()
    # the same specs -> the same key, whatever object holds them
    assert S.PlanSpecs(**{f.name: getattr(p, f.name) for f in fields(S.PlanSpecs)}).key() == p.key()


# ----------------------------------------------------------------------------------------------------------------
# 2. every KernelTuning field, through plan.build_specs
# ----------------------------------------------------------------------------------------------------------------

def _kt_values(name):
    """Two distinct legal KernelTuning values of a field."""
    kind = TUNING_FIELDS[name]
    if kind == "bool":
        return True, False
    tag = kind[0]
    if tag == "ints":
        return kind[1][0], kind[1][1]
    if tag == "int":
        lo = kind[1]
        return lo + 1, lo + 2
    if tag == "float":
        lo = 0.0 if kind[1] is None else kind[1]
        return lo + 1.0, lo + 2.0
    if tag == "enum":
        return kind[1][0], kind[1][1]
    if tag == "re":
        return {"two_cta8_regs": ("136x144x88", "128x136x96"), "fwd_exp": ("sfi", "gmx"),
                "fwd_ptxas": ("--register-usage-level=3", "--register-usage-level=7")}[name]
    raise AssertionError(kind)


def _base_for(owners):
    """A resolved default plan whose route builds every spec class in ``owners``."""
    if any(o == "BwdQSpec" for o in owners):
        assert all(o in ("BwdQSpec", "FwdKernelSpec", "QuantSpec", "HostSpec") for o in owners), owners
        return _resolve("all4")
    return _resolve("fp4")   # fwd / quant / host / bwd1 / bwd2


def _specs_with(plan, tuning):
    return build_specs(plan.route, plan.fwd, plan.bwd, tuning, plan.shape, fwd8_row_p=plan.specs.host.fwd8_row_p,
                       fwd4_row_p=plan.specs.host.fwd4_row_p)


def test_kernel_tuning_mapping_complete():
    """Every KernelTuning field lands in at least one spec field (a field nothing reads would be a silent no-op), and
    names the spec field's kt."""
    assert set(S.KT_TO_SPEC) == set(TUNING_FIELDS), sorted(set(TUNING_FIELDS) ^ set(S.KT_TO_SPEC))
    for kt, targets in S.KT_TO_SPEC.items():
        assert targets, kt
        for owner, fname in targets:
            assert getattr(S, owner).__dataclass_fields__[fname].metadata["kt"] == kt, (kt, owner, fname)


@pytest.mark.parametrize("call", ["fp4", "fp8", "all4", "optA"])
def test_build_specs_reproduces_the_plan(call):
    """build_specs on a plan's own resolved values gives exactly plan.specs: the KernelTuning test below varies the one
    input the plan would."""
    p = _resolve(call)
    assert _specs_with(p, p.tuning) == p.specs and _specs_with(p, KernelTuning()).key() == p.specs.key()


@pytest.mark.parametrize("kt", sorted(TUNING_FIELDS))
def test_kernel_tuning_two_values_two_keys(kt):
    v1, v2 = _kt_values(kt)
    owners = [o for o, _ in S.KT_TO_SPEC[kt]]
    plan = _base_for(owners)
    s1 = _specs_with(plan, KernelTuning(**{kt: v1}))
    s2 = _specs_with(plan, KernelTuning(**{kt: v2}))
    for owner, fname in S.KT_TO_SPEC[kt]:
        slot = _SLOT[owner]
        assert getattr(getattr(s1, slot), fname) != getattr(getattr(s2, slot), fname), (kt, owner, fname)
        assert getattr(s1, slot).key() != getattr(s2, slot).key()
    assert s1.key() != s2.key()
    # the same value -> the same key; and the default KernelTuning is neither of them unless the value is the rule's
    assert _specs_with(plan, KernelTuning(**{kt: v1})).key() == s1.key()
    cache = {}
    for s in (s1, s2, _specs_with(plan, KernelTuning(**{kt: v1})), _specs_with(plan, KernelTuning(**{kt: v2}))):
        cache.setdefault(s.key(), s)
    assert len(cache) == 2


# ----------------------------------------------------------------------------------------------------------------
# 3. the spec fields public arguments and options set, through resolve_fp4 / resolve_fp8
# ----------------------------------------------------------------------------------------------------------------

# "Spec.field" -> (call, the override that moves it). The default call and the overridden call must give two keys.
OPTION_PATHS = {
    "FwdKernelSpec.fp4_pv_two_level": ("optA", dict(opt_p_frame="single_level")),
    "FwdKernelSpec.direct_p": ("all4", dict(opt_softmax_exp="exp2")),
    "FwdKernelSpec.fp4all_exp2": ("all4", dict(opt_softmax_exp="exp2")),
    "Bwd1Spec.ds_sr": ("fp4", dict(opt_sr_ds=True)),
    "Bwd1Spec.p_scaling_fp4": ("fp4_pt", dict(opt_p_scaling="block_1x32")),
    "Bwd1Spec.op_scaling": ("fp4", dict(opt_operand_scaling="per_tensor")),
    "Bwd1Spec.ds_scaling_fp8": ("fp8", dict(opt_ds_scaling="row_1x32")),
    "Bwd1Spec.p_scaling_fp8": ("fp8", dict(opt_p_scaling="block_1x32")),
    "Bwd2Spec.ds_scaling_fp8": ("fp8_pt", dict(opt_ds_scaling="block_32x64")),
    "Bwd2Spec.p_scaling_fp8": ("fp8", dict(opt_p_scaling="block_1x32")),
    "Bwd2Spec.op_scaling": ("fp4", dict(opt_operand_scaling="per_tensor")),
    "BwdQSpec.exp2": ("all4", dict(opt_softmax_exp="exp2")),
    "BwdQSpec.dk_sr": ("all4", dict(opt_sr_dk_operand=False)),
    "BwdQSpec.dv_sr": ("all4", dict(opt_sr_dv_operand=True)),
    "BwdQSpec.dkv": ("all4", dict(opt_fp4_backward_dkdv="fp4")),
    "QuantSpec.q_block_2d": ("fp4", dict(opt_q_block_2d=True)),
    "QuantSpec.qk_hadamard": ("fp4", dict(qk_hadamard=False)),
    "QuantSpec.per_tensor_pow2": ("all4", dict(opt_per_tensor_scale="power_of_two")),
    "QuantSpec.grad_format": ("fp8_pt", dict(opt_grad_format="e5m2")),
    "QuantSpec.operand_scaling_fp4": ("fp4", dict(opt_operand_scaling="per_tensor")),
    "QuantSpec.operand_scaling_fp8": ("fp8", dict(opt_operand_scaling="per_tensor")),
    "QuantSpec.ds_scaling_fp8": ("fp8", dict(opt_ds_scaling="row_1x32")),
    "QuantSpec.p_scaling_fp4": ("fp4_pt", dict(opt_p_scaling="block_1x32")),
    "QuantSpec.causal_safe_v": ("all4", dict(causal_safety="key_local")),
    "QuantSpec.vtok_diag": ("all4_causal", dict(causal_safety="key_local")),
    "QuantSpec.vtok_diag_k": ("all4_causal", dict(opt_k_channel="cut")),
    "QuantSpec.vtok_hadqk": ("all4_causal", dict(causal_safety="key_local", qk_hadamard=True)),
    "QuantSpec.vtok_hadv": ("all4_causal", dict(causal_safety="key_local", opt_v_hadamard=True)),
    "QuantSpec.fp4all_stab": ("all4", dict(opt_all_fp4_stabiliser="rtn")),
    "QuantSpec.fp4q_ds_scale": ("all4", dict(opt_fp4_ds_multiplier="global")),
    "QuantSpec.fp4q_qrow": ("all4", dict(opt_fp4_ds_multiplier="global")),
    "HostSpec.fp8_forward_kernel": ("fp8_pt", dict(opt_forward_kernel="upstream")),
    "HostSpec.fwd8_row_p": ("fp8", dict(opt_p_hat_frame="group")),
    "HostSpec.fwd4_row_p": ("fp4", dict(opt_p_hat_frame="row")),
    "HostSpec.fwd_two_cta": ("fp4", dict(opt_forward_two_cta="on")),
    "HostSpec.bwd_two_cta": ("fp4", dict(opt_two_cta="off")),
    "HostSpec.save_lean": ("fp4_pt", dict(opt_save_bf16_inputs=False)),
}

# "Spec.field" -> (call A, call B): fields the route (entry / pv_precision / backward) sets
ROUTE_PATHS = {
    "FwdKernelSpec.fp4all": ("fp4", "all4"),
    "FwdKernelSpec.fp4all_rowsum": ("all4_inference", "optA"),
    "FwdKernelSpec.fp4all_sf_decode": ("all4_inference", "optA"),
    "Bwd1Spec.fp8": ("fp4", "fp8"),
    "HostSpec.bwd_op_mode": ("fp4", "fp8"),
}

# fields no public argument, option or KernelTuning field reaches (their two-value keys are proved in section 1)
UNREACHED = {
    "FwdKernelSpec.fp4_pv_tile_frame": "p_frame='per_tile' is refused on option A's integer-frame step (rule 23)",
    "FwdKernelSpec.fp4_pv_rescale_threshold": "set only with the refused p_frame='per_tile' (rule 23)",
    "FwdKernelSpec.causal_zigzag_order": "driver rule (by design: fixed, not a KernelTuning field)",
    "FwdKernelSpec.causal_stage_skip": "driver rule (by design: fixed, not a KernelTuning field)",
    "Bwd1Spec.compute_wgs": "KernelTuning field deleted (4 warpgroups assert / run out of registers)",
    "Bwd1Spec.regs4": "only read at compute_wgs == 4 (unreachable)",
    "Bwd1Spec.pp": "constant (d = 64 ping-pong); no public setter",
    "Bwd1Spec.det_issue_warp": "driver rule (by design)",
    "Bwd1Spec.det_issue_bufs": "driver rule (by design)",
    "Bwd1Spec.ds_scaling_fp4": "the explicit per-tensor dS is refused (rule 33); the plan sets per_tensor from head_dim "
                               "> 128 (a shape, not an option: test_head_dim_256_ds_scaling_in_the_specs)",
    "Bwd2Spec.ds_scaling_fp4": "as Bwd1Spec.ds_scaling_fp4",
    "QuantSpec.ds_scaling_fp4": "as Bwd1Spec.ds_scaling_fp4",
    "BwdQSpec.dq8": "dq_operand='e4m3' is not built (rule E16)",
    "BwdQSpec.mxds": "the kernel's rule (rule_mxds); no public setter",
    "QuantSpec.fp4all_vtok": "follows causal_safe_v's rule; no public setter",
    "HostSpec.seqused_padding_ok": "seqused_k_group_aligned=True is refused",
    "HostSpec.fp8_pure_qk": "implied by flash_attn_fp8; no public setter",
    "HostSpec.det_two_cta": "driver rule (by design)",
}


def _kt_targets():
    return {f"{o}.{f}" for v in S.KT_TO_SPEC.values() for o, f in v}


def test_every_spec_field_is_classified():
    allf = {_ID(p) for p in SPEC_FIELDS}
    kt = _kt_targets()
    groups = [kt, set(OPTION_PATHS), set(ROUTE_PATHS), set(UNREACHED)]
    union = set().union(*groups)
    assert union == allf, (sorted(allf - union), sorted(union - allf))
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b), sorted(a & b)


def test_unreached_fields_are_unreached_by_options():
    """No options value or public argument moves an UNREACHED field on the calls of this file (a field that becomes
    reachable must move to OPTION_PATHS, with its two-key proof)."""
    for call, over in OPTION_PATHS.values():
        a, b = _resolve(call), _resolve(call, **over)
        for name in UNREACHED:
            owner, fname = name.split(".")
            if getattr(a.specs, _SLOT[owner]) is None or getattr(b.specs, _SLOT[owner]) is None:
                continue
            assert _get(a, owner, fname) == _get(b, owner, fname), (name, call, over)


@pytest.mark.parametrize("name", sorted(OPTION_PATHS))
def test_option_two_values_two_keys(name):
    owner, fname = name.split(".")
    call, over = OPTION_PATHS[name]
    a, b = _resolve(call), _resolve(call, **over)
    assert _get(a, owner, fname) != _get(b, owner, fname), (name, _get(a, owner, fname))
    slot = _SLOT[owner]
    assert getattr(a.specs, slot).key() != getattr(b.specs, slot).key()
    assert a.specs.key() != b.specs.key()
    # the same arguments -> the same keys
    assert _resolve(call).specs.key() == a.specs.key() and _resolve(call, **over).specs.key() == b.specs.key()


@pytest.mark.parametrize("name", sorted(ROUTE_PATHS))
def test_route_two_values_two_keys(name):
    owner, fname = name.split(".")
    ca, cb = ROUTE_PATHS[name]
    a, b = _resolve(ca), _resolve(cb)
    assert a.route != b.route
    assert _get(a, owner, fname) != _get(b, owner, fname)
    assert getattr(a.specs, _SLOT[owner]).key() != getattr(b.specs, _SLOT[owner]).key()



def test_head_dim_256_ds_scaling_in_the_specs():
    """head_dim 256 runs the three-pass kernels on a per-tensor dS (plan.bwd.ds_scaling = per_tensor); the specs say
    so (QuantSpec / Bwd1Spec / Bwd2Spec.ds_scaling_fp4), and the keys tell it from head_dim 128's blk32 dS."""
    q, k, v = _qkv(d=256)
    p = resolve_fp4(q, k, v, options=O4(operand_scaling="per_tensor"))
    assert p.bwd.ds_scaling == "per_tensor"
    for slot in ("quant", "bwd1", "bwd2"):
        sp = getattr(p.specs, slot)
        assert sp is None or sp.ds_scaling_fp4 == "per_tensor", slot
    p128 = resolve_fp4(*_qkv(), options=O4(operand_scaling="per_tensor"))
    assert p128.bwd.ds_scaling == "block_32x32" and p128.specs.bwd1.ds_scaling_fp4 == "blk32"
    assert p.specs.bwd1.key() != p.specs.bwd1.replace(ds_scaling_fp4="blk32").key()

# ----------------------------------------------------------------------------------------------------------------
# 4. constants shared by the specs and the tree
# ----------------------------------------------------------------------------------------------------------------

def test_shared_constants_match_the_tree():
    from flash_attn.cute import qk_hadamard
    from flash_attn.cute.dq_postprocess_cute import DQ_ZERO_DEFAULT
    from flash_attn.cute.flash_fwd_sm100_fp4 import FP4_PV_RESCALE_THRESHOLD
    assert S.DQ_ZERO_DEFAULT == DQ_ZERO_DEFAULT
    assert S.FP4_PV_RESCALE_THRESHOLD == FP4_PV_RESCALE_THRESHOLD
    assert S.QK_HADAMARD_E4M3_DEFAULT == qk_hadamard.E4M3_DEFAULT
    # the specs' defaults are the certified ones the plan resolves on the default calls
    p4, p8 = _resolve("fp4"), _resolve("fp8")
    assert p4.specs.quant.operand_scaling_fp4 == S.OP_SCALING_DEFAULT == p8.specs.quant.operand_scaling_fp8
    assert p4.specs.bwd1.ds_scaling_fp4 == S.DS_SCALING_DEFAULT == p8.specs.bwd1.ds_scaling_fp8
    assert _resolve("all4").specs.bwdq.mxds is None and S.BwdQSpec.rule_mxds(False) == S.FP4Q_MXDS_DEFAULT
    assert p8.specs.host.fwd8_row_p is S.FWD8_ROW_P_DEFAULT and p4.specs.host.fwd4_row_p is S.FWD4_ROW_P_DEFAULT
    assert p4.specs.quant.token_block_forward == S.TB_FWD_DEFAULT


# ----------------------------------------------------------------------------------------------------------------
# 5. goldens: the recorded compile keys embed the spec keys the plan resolves
# ----------------------------------------------------------------------------------------------------------------

# a recorded golden capture of the compile keys (a gkey.jsonl named by LOWPREC_GOLDEN_GKEY; skipped without it)
_GOLD = Path(os.environ.get("LOWPREC_GOLDEN_GKEY", "/nonexistent/gkey.jsonl"))
# tests/cute/test_lowprec_api.py CELLS (b, s, h, d, causal)
_CELLS = {"S1": (1, 1024, 4, 128, True), "A1": (4, 4096, 16, 128, True), "A2": (4, 4096, 16, 128, False)}


def _golden_rows():
    if not _GOLD.exists():
        pytest.skip(f"no golden capture (set LOWPREC_GOLDEN_GKEY; got {_GOLD})")
    return [json.loads(line) for line in _GOLD.read_text().splitlines() if line.strip()]


def test_golden_default_all4_keys_embed_the_plan_specs():
    n = 0
    for row in _golden_rows():
        group, route, cell = row["id"].split("/")
        assert (group, route) == ("default", "all4"), row["id"]
        b, s, h, d, causal = _CELLS[cell]
        p = resolve_fp4(*_qkv(b=b, s=s, h=h, d=d), causal=causal, **ALL4)
        by_cache = {}
        for k in row["keys"]:
            by_cache.setdefault(k["cache"].rsplit(".", 2)[-2] if k["cache"].count(".") >= 2 else k["cache"],
                                []).append(k["key"])
        bwdq = [k for k in by_cache.get("_flash_attn_bwd_fp4q", [])]
        fwd = [k for k in by_cache.get("_flash_attn_fwd", [])]
        assert len(bwdq) == 1 and len(fwd) == 1, (row["id"], sorted(by_cache))
        assert repr(p.specs.bwdq.key()) in bwdq[0], row["id"]
        assert repr(p.specs.fwd.key()) in fwd[0], row["id"]
        n += 1
    assert n == 3  # default all-4-bit at S1, A1, A2

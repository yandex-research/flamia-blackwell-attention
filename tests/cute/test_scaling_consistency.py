"""SCALING-CONSISTENCY RULE (a design invariant): forward and backward use the same scaling of every shared
operand (Q, K, V). The plan resolves ONE operand scaling per call (options.operand_scaling) that both passes run with;
the backward driver checks the codes handed to it against the plan's QuantSpec. Every mismatched combination raises
ValueError naming the rule; every supported value passes.

The resolution tests need no GPU (meta tensors). The end-to-end tests run the public entries and the backward driver on
a small shape and are skipped without an SM100 GPU."""
import pytest
import torch

from flash_attn.cute.interface_bwd_fp4 import (
    SCALING_RULE,
    check_codes_scaling,
    check_scaling_consistency,
    operand_scaling_of,
)
from flash_attn.cute.lowprec import FP4AttnOptions as O4
from flash_attn.cute.lowprec import FP8AttnOptions as O8
from flash_attn.cute.lowprec import resolve_fp4, resolve_fp8
from flash_attn.cute.lowprec import specs as S

ENTRY = {False: (resolve_fp4, O4), True: (resolve_fp8, O8)}
# the public operand_scaling -> the scaling of the codes its forward produces (operand_scaling_of's names)
CODES_OF = {"token_block": "blk32x128", "block_32x32": "blk32", "per_tensor": "per_tensor"}


def _e4m3_host_scale(x):
    """Per-tensor E4M3 of x and its scale as a host float (quant.fp8.quantize_fp8_per_tensor; one host read, as the
    harness always took): s = max(amax / 448, 1e-12)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    r = quantize_fp8_per_tensor(x)
    return r["codes"], r["scales"][0].item()


def _qkv(d=128, s=512, rg=True):
    return [torch.empty(1, s, 2, d, dtype=torch.bfloat16, device="meta").requires_grad_(rg) for _ in range(3)]


def _mx(cols):
    return {"q": {"cols": cols}, "k": {"cols": cols}, "v": {"cols": cols}}


def _codes(kind):
    """The fp8_scales dict a forward hands the backward for codes of scaling ``kind`` (operand_scaling_of's input)."""
    t = torch.empty(1)
    return {None: None, "per_tensor": {"q8": t, "k8": t, "v8": t}, "blk32x128": {"mx": _mx(128)},
            "blk32": {"mx": _mx(32)}, "mx": {"mx": {"q": {}, "fac": t}}}[kind]


def _plan(fp8, operand_scaling="auto", **kw):
    resolve, O = ENTRY[fp8]
    return resolve(*_qkv(**kw), options=O(operand_scaling=operand_scaling))


def _consistent(plan):
    assert plan.consistency, plan.log()
    for op, f, b in plan.consistency:
        assert f == b, (op, f, b)


# --- the codes a caller hands the backward ------------------------------------------------------------------------------
def test_operand_scaling_of():
    t = torch.empty(1)
    assert operand_scaling_of(None) is None
    assert operand_scaling_of({"pmax": t}) is None
    assert operand_scaling_of({"q8": t, "k8": t, "v8": t}) == "per_tensor"
    assert operand_scaling_of({"mx": _mx(128)}) == "blk32x128"
    assert operand_scaling_of({"mx": _mx(32)}) == "blk32"
    assert operand_scaling_of({"mx": {"q": {}, "fac": t}}) == "mx"
    with pytest.raises(ValueError, match="SCALING-CONSISTENCY RULE"):
        operand_scaling_of({"v8": t, "mx": _mx(128)})
    with pytest.raises(ValueError, match="SCALING-CONSISTENCY RULE"):
        operand_scaling_of({"mx": {"q": {"cols": 32}, "k": {"cols": 128}}})


@pytest.mark.parametrize("fwd,bwd", [("per_tensor", "blk32x128"), ("per_tensor", "blk32"), ("blk32x128", "per_tensor"),
                                     ("blk32", "per_tensor"), ("blk32", "blk32x128"), ("blk32x128", "blk32"),
                                     ("mx", "per_tensor")])
def test_check_mismatch_raises(fwd, bwd):
    with pytest.raises(ValueError, match="SCALING-CONSISTENCY RULE"):
        check_scaling_consistency(fwd, bwd, "test")


@pytest.mark.parametrize("fwd,bwd", [("per_tensor", "per_tensor"), ("blk32", "blk32"), ("blk32x128", "blk32x128"),
                                     ("mx", "blk32"), ("mx", "blk32x128"), (None, "per_tensor"), (None, "blk32x128")])
def test_check_match_passes(fwd, bwd):
    check_scaling_consistency(fwd, bwd, "test")


# --- the plan's one scaling vs the codes handed to the backward ---------------------------------------------------------
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("option,given", [("per_tensor", "blk32x128"), ("per_tensor", "blk32"),
                                          ("token_block", "per_tensor"), ("block_32x32", "per_tensor"),
                                          ("token_block", "blk32"), ("block_32x32", "blk32x128")])
def test_explicit_scaling_vs_codes_raises(fp8, option, given):
    quant = _plan(fp8, option).specs.quant
    with pytest.raises(ValueError, match="SCALING-CONSISTENCY RULE"):
        check_codes_scaling(fp8, _codes(given), None, quant, "test")


@pytest.mark.parametrize("option", ["token_block", "block_32x32"])
def test_explicit_block_with_prequant_per_tensor_raises(option):
    """flash_attn_fp8(q_quantized=per-tensor codes) under a block-scaled operand_scaling: a per-tensor forward with a
    block backward is refused at plan resolution, naming the explicit per-tensor opt-in. (flash_attn_fp4 takes no
    pre-quantised codes: no 4-bit counterpart.)"""
    q, k, v = _qkv()
    pre = (torch.empty(1, device="meta"), 1.0)
    with pytest.raises(ValueError, match="per_tensor") as e:
        resolve_fp8(q, k, v, q_quantized=pre, k_quantized=pre, qk_hadamard=False,
                    options=O8(operand_scaling=option))
    assert e.value.rule == 16
    # the default follows the caller's per-tensor codes, and both passes share them
    p = resolve_fp8(q, k, v, q_quantized=pre, k_quantized=pre, qk_hadamard=False)
    assert p.fwd.operand_scaling == p.bwd.operand_scaling == "per_tensor"
    _consistent(p)
    check_codes_scaling(True, _codes("per_tensor"), None, p.specs.quant, "test")


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("option", ["auto", "token_block", "block_32x32", "per_tensor"])
def test_the_forwards_codes_pass_the_backward(fp8, option):
    """Every operand scaling resolves forward and backward to the same value, and the codes that forward produces pass
    the backward driver's check (and the block / unannotated codes of the other width do not)."""
    p = _plan(fp8, option)
    _consistent(p)
    assert p.fwd.operand_scaling == p.bwd.operand_scaling
    resolved = p.fwd.operand_scaling
    assert option == "auto" and resolved == "token_block" or resolved == option
    kind = CODES_OF[resolved]
    for given in (None, kind) + (("mx",) if kind != "per_tensor" else ()):
        check_codes_scaling(fp8, _codes(given), None, p.specs.quant, "test")


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("d,s", [(96, 512), (192, 512), (256, 512), (128, 320)])
def test_auto_never_falls_back_to_per_tensor_silently(fp8, d, s):
    """Shapes the token-block kernels once could not serve (by design): "auto" resolves
    to token blocks where they serve now (padded head dims, ragged lengths as documents), and otherwise refuses (naming
    the explicit per-tensor opt-in where that serves the call) -- it never resolves to per-tensor operands by itself."""
    resolve, O = ENTRY[fp8]
    try:
        p = resolve(*_qkv(d=d, s=s), causal=True)
    except (ValueError, NotImplementedError) as e:
        assert getattr(e, "rule", None) is not None, e   # a refusal names the rule it breaks
        if "per_tensor" in str(e):
            p = resolve(*_qkv(d=d, s=s), causal=True, options=O(operand_scaling="per_tensor"))
            assert p.fwd.operand_scaling == p.bwd.operand_scaling == "per_tensor"
            _consistent(p)
        return
    assert p.fwd.operand_scaling == p.bwd.operand_scaling == "token_block", p.log()
    _consistent(p)


@pytest.mark.parametrize("fp8", [False, True])
def test_default_d64_is_token_blocks(fp8):
    """head_dim 64 is a native token-block head dim (no longer an unservable shape)."""
    p = _plan(fp8, d=64)
    assert p.fwd.operand_scaling == p.bwd.operand_scaling == "token_block"
    assert getattr(p.specs.quant, "operand_scaling_fp8" if fp8 else "operand_scaling_fp4") == S.OP_SCALING_DEFAULT


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("option", ["per_tensor", "block_32x32", "token_block"])
def test_explicit_scaling_matching_codes_passes(fp8, option):
    p = _plan(fp8, option)
    assert getattr(p.specs.quant, "operand_scaling_fp8" if fp8 else "operand_scaling_fp4") == CODES_OF[option]
    for name in ("bwd1", "bwd2"):
        s = getattr(p.specs, name)
        if s is not None:   # the 2-CTA spec exists only where its operand scaling does (not block_32x32)
            assert s.op_scaling == CODES_OF[option], (name, s.op_scaling)
    check_codes_scaling(fp8, _codes(CODES_OF[option]), None, p.specs.quant, "test")


def test_rule_text_names_the_rule():
    assert SCALING_RULE.startswith("SCALING-CONSISTENCY RULE") and "Mixed designs" in SCALING_RULE


# --- end to end (SM100) ---------------------------------------------------------------------------------------------------
_gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                          reason="SM100 GPU required")


def _inputs(s=512):
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v, do = (torch.randn(1, s, 2, 128, generator=g).to("cuda", torch.bfloat16) for _ in range(4))
    return q, k, v, do


@_gpu
@pytest.mark.parametrize("route", ["fp4", "fp8"])
@pytest.mark.parametrize("option", ["auto", "per_tensor", "token_block"])
def test_autograd_supported_passes(route, option):
    from flash_attn.cute.lowprec import flash_attn_fp4, flash_attn_fp8
    f, O = (flash_attn_fp4, O4) if route == "fp4" else (flash_attn_fp8, O8)
    q, k, v, do = _inputs()
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    o = f(q, k, v, causal=True, options=O(operand_scaling=option))
    g = torch.autograd.grad(o, (q, k, v), do)
    assert all(torch.isfinite(t).all() for t in g)


@_gpu
@pytest.mark.parametrize("option", ["token_block", "block_32x32"])
def test_driver_per_tensor_codes_block_backward_raises(option):
    """A per-tensor forward's codes handed to a block-scaled backward (before: silently re-quantised). The driver is
    called with the plan's specs (the private driver hook: spec= arguments, no configuration)."""
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4
    q, k, v, do = _inputs()
    out, lse = _flash_attn_fwd(q, k, v, return_lse=True)[:2]
    sc = {}
    for nm, t in (("q", q), ("k", k), ("v", v), ("do", do)):
        sc[nm + "8"], sc["s" + nm] = _e4m3_host_scale(t)
    block = resolve_fp4(*_qkv(), options=O4(operand_scaling=option, two_cta="off"))
    with pytest.raises(ValueError, match="SCALING-CONSISTENCY RULE"):
        _flash_attn_bwd_fp4(q, k, v, out, do, lse, fp8_scales=sc, specs=block.specs)
    # the plan whose forward made per-tensor codes: runs
    pt = resolve_fp4(*_qkv(), options=O4(operand_scaling="per_tensor", two_cta="off"))
    g = _flash_attn_bwd_fp4(q, k, v, out, do, lse, fp8_scales=sc, specs=pt.specs)
    assert all(torch.isfinite(t).all() for t in g)

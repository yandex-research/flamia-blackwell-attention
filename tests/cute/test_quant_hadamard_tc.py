"""hadamard_rotation_method="tensor_core" (quant._rotations.quad_hadamard_rotate: the 128-point rotation on mma.sync)
against the eager references (whose rotation is the butterfly's order) and against the register butterfly.

The tensor-core rotation computes the same R = diag(signs) H / sqrt(128) with the same sign flips and the same final
* fp32(1 / sqrt(128)); only the fp32 summation order differs (an 8-point stage on the tensor cores, then a 4-level
register butterfly). So the rotated values are bitwise the butterfly's wherever every partial sum is exact in fp32, and differ by
an fp32 ulp elsewhere. The stated tolerance for every code / scale-factor output: at most FLIP_RATE of the elements
differ at the VALUE level (an E4M3 / E2M1 code of -0 counts as +0: the tensor cores return +0 where the butterfly may
return -0, and the two decode to the same value); measured rates: 0 on N(0, 1) inputs,
2.5e-6 of the codes on log-normal(3) inputs. Float side outputs (D, the tile amax, row norms) within a relative 1e-6.
Rows that can overflow are outside the contract (an inf after RoPE in fp16, or a partial sum of the rotation above
FLT_MAX: the two summation orders overflow at different places, and +inf / -inf then meet as inf or NaN): the
comparisons take the 128-token tiles whose transformed rows all have an L1 norm below FLT_MAX (no partial sum of the
rotation can overflow there; the token-block statistics never mix tiles), and the row norms only where the reference's
are finite.

    pytest tests/cute/test_quant_hadamard_tc.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the test helpers next to this file
import quant_reference as R  # noqa: E402
from flash_attn.cute.quant import _rotations as ROT  # noqa: E402

FLIP_RATE = 1e-3
DISTS = ("n01", "logit3", "outlier", "zeros", "tiny", "mixed", "subnorm_sf")


def make(shape, dist, dtype, seed=0):
    from test_quant_fp8 import make as make8
    return make8(shape, dist, dtype, seed)


def _value_view(t: torch.Tensor, kind: str) -> torch.Tensor:
    u = t.contiguous().view(torch.uint8)
    if kind == "fp8":
        return torch.where(u == 0x80, torch.zeros_like(u), u)
    if kind == "nib":
        lo, hi = u & 0xF, u >> 4
        lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
        hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
        return lo | (hi << 4)
    return u


def flip_rate(a: torch.Tensor, b: torch.Tensor, kind: str) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    va, vb = _value_view(a, kind), _value_view(b, kind)
    if va.numel() == 0:       # nothing inside the contract (every tile can overflow)
        return 0.0
    return (va != vb).float().mean().item()


def assert_codes_close(a, b, kind, what, max_rate=FLIP_RATE):
    """kind: "fp8" (FP8 codes), "nib" (packed E2M1 codes), "byte" (scale factors, tables: exact bytes)."""
    r = flip_rate(a, b, kind)
    assert r <= max_rate, f"{what}: {100 * r:.4f} % of the elements differ (> {100 * max_rate:.3f} %)"


def safe_tiles(x, rope_cos=None, rope_sin=None, row_divisor=None):
    """(b, s, h) bool, constant over each 128-token tile: the tiles whose transformed rows (row divisor, RoPE) are finite
    with an L1 norm below FLT_MAX (no partial sum of the rotation can overflow)."""
    xf = x.float()
    if row_divisor is not None:
        xf = (xf / row_divisor.permute(0, 2, 1)[..., None]).to(x.dtype).float()
    if rope_cos is not None:
        xf = R.apply_rope(xf.to(x.dtype), rope_cos, rope_sin)
    ok = torch.isfinite(xf).all(-1) & (xf.double().abs().sum(-1) < 3.4e38)          # (b, s, h)
    b, s, h = ok.shape
    tile_ok = ok.reshape(b, s // 128, 128, h).all(2, keepdim=True)
    return tile_ok.expand(b, s // 128, 128, h).reshape(b, s, h)


def assert_float_close(a, b, what, rtol=1e-6):
    a, b = a.double(), b.double().to(a.device)
    err = (a - b).abs().max().item()
    scale = b.abs().max().item() + 1e-300
    assert err <= rtol * scale or err == 0.0, f"{what}: max err {err:.3e} vs scale {scale:.3e}"


# -------------------------------------------------------------------------------------------------- quantize_fp8_blocks
BLOCK_ROWS = [
    ("tb128_fac", dict(block_columns=128), dict(output_token_block_factors=True)),
    ("tb32", dict(block_columns=32), {}),
    ("tb128_s4", dict(block_columns=128, also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw"),
     dict(nvfp4_copy_encoding="qk_operand")),
    ("tb128_a_sr", dict(block_columns=128, also_make_nvfp4_copy=True, nvfp4_copy_scaling="token_block",
                        nvfp4_copy_rounding="stochastic", nvfp4_copy_seed=11), {}),
    ("tb128_rope", dict(block_columns=128, _rope="transposed"), {}),
    ("stream_cm_d", dict(block_columns=128, _d=True), dict(make_channel_major_copy=True)),
    ("rows1x32", dict(block_rows=1, block_columns=32), dict(output_exponent_table=False)),
    ("rows128", dict(block_rows=128, block_columns=128), dict(output_exponent_table=False)),
    ("cols1_explicit", dict(block_rows=32, block_columns=1, _rope="rows"), dict(output_exponent_table=False)),
]


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", [r[0] for r in BLOCK_ROWS])
def test_blocks_tc_vs_reference_and_butterfly(name, dtype, dist):
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    _, ref_kw, kern_kw = next(r for r in BLOCK_ROWS if r[0] == name)
    ref_kw = dict(ref_kw)
    x = make((2, 256, 3, 128), dist, dtype)
    rope = ref_kw.pop("_rope", None)
    if rope is not None:
        g = torch.Generator(device="cuda").manual_seed(5)
        cos = torch.randn(256, 64, device="cuda", generator=g)
        sin = torch.randn(256, 64, device="cuda", generator=g)
        if rope == "transposed":
            cos, sin = cos.t().contiguous(), sin.t().contiguous()
        ref_kw.update(rope_cos=cos, rope_sin=sin)
    if ref_kw.pop("_d", False):
        ref_kw["compute_d_with_output"] = make((2, 256, 3, 128), "n01", dtype, seed=9)
    tc = quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core", **ref_kw,
                             **kern_kw)
    assert ROT.ROTATION_METHOD_LOG["quantize_fp8_blocks"][0] == "tensor_core"
    bf = quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="butterfly", **ref_kw, **kern_kw)
    ref = R.fp8_blocks_reference(x, apply_hadamard_rotation=True, **ref_kw)
    tag = f"{name} {dtype} {dist}"
    fin = safe_tiles(x, ref_kw.get("rope_cos"), ref_kw.get("rope_sin"))
    if not bool(fin.all()):   # tiles that can overflow: the codes of the other tiles only (see the module docstring)
        for other, who in ((ref, "reference"), (bf, "butterfly")):
            assert_codes_close(tc["codes"][fin], other["codes"][fin], "fp8", f"{tag} safe-tile codes vs {who}")
        return
    for other, who in ((ref, "reference"), (bf, "butterfly")):
        assert_codes_close(tc["codes"], other["codes"], "fp8", f"{tag} codes vs {who}")
        for k in ("sf_tok_raw", "sf_dim_raw", "e"):
            if k in tc and k in other:
                assert_codes_close(tc[k], other[k], "byte", f"{tag} {k} vs {who}")
        if "codes_t" in tc and "codes_t" in other:
            assert_codes_close(tc["codes_t"], other["codes_t"], "fp8", f"{tag} codes_t vs {who}")
        if "fac" in tc and "fac" in other:
            assert_codes_close(tc["fac"], other["fac"], "byte", f"{tag} fac vs {who}")
        # (the reference's NVFP4 copy is option A's dp_operand encoding; the qk_operand S codes: vs the butterfly only)
        if "x4" in tc and "x4" in other and (who == "butterfly" or kern_kw.get("nvfp4_copy_encoding") != "qk_operand"):
            assert_codes_close(tc["x4"], other["x4"], "nib", f"{tag} nvfp4 codes vs {who}")
    if "sf4" in tc:
        assert_codes_close(tc["sf4"], bf["sf4"], "byte", f"{tag} nvfp4 sf vs butterfly")
    if "D" in tc:
        assert_float_close(tc["D"], bf["D"], f"{tag} D vs butterfly", rtol=1e-5)


def test_blocks_tc_rotated_amax_is_exact_rotation_max():
    """output_amax of the tensor-core rows: the tile's max |x R| (vs the float64 rotation), as the butterfly's."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks, hadamard_signs_tuple
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(2, 512, 3, 128, device="cuda", generator=g) * 3).bfloat16()
    r = quantize_fp8_blocks(x, apply_hadamard_rotation=True, output_amax=True, hadamard_rotation_method="tensor_core")
    H = torch.ones(1, 1, device="cuda", dtype=torch.float64)
    while H.shape[0] < 128:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    xr = (x.double() * torch.tensor(hadamard_signs_tuple(4321), device="cuda", dtype=torch.float64)) @ H / 128 ** 0.5
    tile = xr.abs().permute(0, 2, 1, 3).reshape(2, 3, 4, -1).amax(-1).reshape(-1)
    assert_float_close(r["amax"], tile, "tile amax", rtol=1e-6)



@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_blocks_rotation_scope_nvfp4_copy(dtype, dist):
    """hadamard_rotation_scope="nvfp4_copy" (the 4-bit routes' Q / K with unrotated E4M3 copies): the FP8 side is bitwise
    the unrotated launch's (codes, SF tables, exponent table, factors, amax) and the NVFP4 copy bitwise the fully
    rotated launch's (codes, SF bytes) -- one read, two bases."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((2, 256, 3, 128), dist, dtype)
    s4 = dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand",
              output_token_block_factors=True, output_amax=True)
    got = quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_scope="nvfp4_copy", **s4)
    plain = quantize_fp8_blocks(x, **s4)
    rot = quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core", **s4)
    for k in ("codes", "sf_tok_raw", "sf_dim_raw", "e", "fac", "amax"):
        assert torch.equal(got[k].view(torch.uint8), plain[k].view(torch.uint8)), f"{dtype} {dist} {k} vs unrotated"
    for k in ("x4", "sf4"):
        assert torch.equal(got[k].view(torch.uint8), rot[k].view(torch.uint8)), f"{dtype} {dist} {k} vs rotated"


def test_blocks_rotation_scope_refusals():
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    s4 = dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand")
    for kw in (dict(apply_hadamard_rotation=True),                                  # no NVFP4 copy
               dict(s4, apply_hadamard_rotation=True, hadamard_rotation_method="butterfly"),
               dict(s4, apply_hadamard_rotation=True, nvfp4_copy_scaling="token_block"),
               dict(s4),                                                             # no rotation
               dict(s4, apply_hadamard_rotation=True, make_channel_major_copy=True)):
        with pytest.raises(ValueError):
            quantize_fp8_blocks(x, hadamard_rotation_scope="nvfp4_copy", **kw)
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x.float(), hadamard_rotation_scope="nvfp4_copy", apply_hadamard_rotation=True, **s4)
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, hadamard_rotation_scope="codes", apply_hadamard_rotation=True, **s4)

# --------------------------------------------------------------------------------------------- quantize_fp8_per_tensor
@pytest.mark.parametrize("dist", ["n01", "logit3", "outlier", "zeros", "mixed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_per_tensor_tc_vs_reference(dist, dtype):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    q = make((1, 384, 2, 128), dist, dtype)
    k = make((1, 384, 2, 128), dist, dtype, seed=3)
    f = torch.exp2(torch.randint(-3, 4, (1, 2, 384), device="cuda").float())
    cos = torch.randn(64, 384, device="cuda")
    sin = torch.randn(64, 384, device="cuda")
    for meth in ("tensor_core", "butterfly"):
        r = quantize_fp8_per_tensor([q, k], apply_hadamard_rotation=(True, True), rope_cos=cos, rope_sin=sin,
                                    row_divisors=[f, None], output_row_norm2_max=True, hadamard_rotation_method=meth)
        assert ROT.ROTATION_METHOD_LOG["quantize_fp8_per_tensor"][0] == meth
        rq = R.fp8_per_tensor_reference(q, apply_hadamard_rotation=True, rope_cos=cos, rope_sin=sin, row_divisor=f)
        rk = R.fp8_per_tensor_reference(k, apply_hadamard_rotation=True, rope_cos=cos, rope_sin=sin)
        tag = f"{dist} {dtype} {meth}"
        if not (bool(safe_tiles(q, cos, sin, f).all()) and bool(safe_tiles(k, cos, sin).all())):
            continue          # a row that can overflow makes the per-tensor scale inf / NaN: outside the contract
        assert_codes_close(r["codes"][0], rq["codes"], "fp8", tag + " q codes")
        assert_codes_close(r["codes"][1], rk["codes"], "fp8", tag + " k codes")
        for got, want in zip(r["scales"].tolist(), (rq["scale"], rk["scale"])):
            assert abs(got - want) <= 1e-6 * abs(want), (tag, got, want)
        want = torch.tensor([rq["row_norm2_max"], rk["row_norm2_max"]])
        if bool(torch.isfinite(want).all()):
            assert_float_close(r["row_norm2_max"], want, tag + " row norms", rtol=1e-5)


# ------------------------------------------------------------------------------------------------- quant.fp4 (explicit)
@pytest.mark.parametrize("dist", ["n01", "logit3", "zeros", "mixed"])
@pytest.mark.parametrize("cfg", ["rows", "rows_e4", "rows_t", "mx", "pad64", "rope", "2d"])
def test_fp4_tc_vs_reference(cfg, dist):
    """quant.fp4's tile kernel with hadamard_rotation_method="tensor_core" given explicitly (its default is the
    butterfly: measured faster there)."""
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows, quantize_fp4_blocks_2d
    d = 64 if cfg == "pad64" else 128
    x = make((1, 256, 2, d), dist, torch.bfloat16)
    sc = torch.tensor([0.05], dtype=torch.float64, device="cuda")
    kw = dict(rows={}, rows_e4=dict(also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=sc),
              rows_t=dict(make_transposed_copy=True), mx=dict(scale_group_size=32), pad64=dict(padded_head_dim=128),
              rope=dict(rope_cos=torch.randn(64, 256, device="cuda"), rope_sin=torch.randn(64, 256, device="cuda")),
              **{"2d": dict(make_transposed_copy=True)})[cfg]
    fn = quantize_fp4_blocks_2d if cfg == "2d" else quantize_fp4_rows
    got = fn(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core", **kw)
    bf = fn(x, apply_hadamard_rotation=True, hadamard_rotation_method="butterfly", **kw)
    ref = R.fp4_rows_reference(x, apply_hadamard_rotation=True, _blocks_2d=cfg == "2d", **kw)
    for other, who in ((ref, "reference"), (bf, "butterfly")):
        for k, kind in (("codes", "nib"), ("transposed_codes", "nib"), ("e4m3_copy", "fp8")):
            if k in got and k in other:
                assert_codes_close(got[k], other[k], kind, f"{cfg} {dist} {k} vs {who}")
    for k in ("scale_factors", "transposed_scale_factors"):
        if k in got:
            assert_codes_close(got[k], bf[k], "byte", f"{cfg} {dist} {k} vs butterfly")


# ------------------------------------------------------------------------------------------------ resolution, fallbacks
def test_rotation_method_resolution_and_fallbacks():
    """None = tensor_core where it measured faster, the butterfly elsewhere (logged); fp32 input and kernels without the
    tensor-core form fall back to the butterfly (logged), never refuse; an unknown name raises."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks, quantize_fp8_per_tensor
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    from flash_attn.cute.quant.fp4_diagonal import quantize_fp4_causal_diagonal
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    log = ROT.ROTATION_METHOD_LOG
    quantize_fp8_blocks(x, apply_hadamard_rotation=True)
    assert log["quantize_fp8_blocks"] == ("tensor_core", "default")
    quantize_fp8_blocks(x, apply_hadamard_rotation=True, block_rows=32, block_columns=1, output_exponent_table=False)
    assert log["quantize_fp8_blocks"][0] == "butterfly" and "thread-per-row" in log["quantize_fp8_blocks"][1]
    quantize_fp8_blocks(x, apply_hadamard_rotation=True, block_rows=32, block_columns=1, output_exponent_table=False,
                        hadamard_rotation_method="tensor_core")
    assert log["quantize_fp8_blocks"] == ("tensor_core", "requested")
    quantize_fp8_blocks(x.float(), apply_hadamard_rotation=True)
    assert log["quantize_fp8_blocks"][0] == "butterfly" and "fp32" in log["quantize_fp8_blocks"][1]
    quantize_fp8_per_tensor([x, x], apply_hadamard_rotation=True)
    assert log["quantize_fp8_per_tensor"] == ("tensor_core", "default")
    quantize_fp8_per_tensor([x, x], apply_hadamard_rotation=True, output_codes=False)
    assert log["quantize_fp8_per_tensor"][0] == "butterfly"
    quantize_fp4_rows(x, apply_hadamard_rotation=True)
    assert log["quantize_fp4_rows"][0] == "butterfly"
    quantize_fp4_rows(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core")
    assert log["quantize_fp4_rows"] == ("tensor_core", "requested")
    quantize_fp4_causal_diagonal(x, x, 0.088, key_channel="off", apply_hadamard_rotation=True,
                                 hadamard_rotation_method="tensor_core")
    assert log["quantize_fp4_causal_diagonal"][0] == "butterfly"
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="mma")


def test_tc_graph_capture_and_no_sync():
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks, quantize_fp8_per_tensor
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    kw = dict(apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core")
    s4 = dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand")
    quantize_fp8_blocks(x, **kw, **s4)
    quantize_fp8_per_tensor(x, **kw)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        r1 = quantize_fp8_blocks(x, **kw, **s4)
        r2 = quantize_fp8_per_tensor(x, **kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        r3 = quantize_fp8_blocks(x, **kw, **s4)
        r4 = quantize_fp8_per_tensor(x, **kw)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    for k in ("codes", "x4", "sf_tok_raw", "sf_dim_raw", "e"):
        assert torch.equal(r3[k].view(torch.uint8), r1[k].view(torch.uint8)), k
    assert torch.equal(r4["codes"].view(torch.uint8), r2["codes"].view(torch.uint8))
    assert torch.equal(r4["scales"], r2["scales"])

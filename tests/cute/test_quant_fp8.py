"""quant.fp8 (quantize_fp8_blocks, quantize_fp8_per_tensor) against the eager references of tests/cute/quant_reference.py, which are
written from the format definitions. Codes and scale-factor bytes must be bitwise equal; float sums (D, row norms, the
dS bound) are compared with a tolerance against float64, and bitwise against the old kernels' summation order
(`old_order_row_dot`, the tree decoded with cancellation probes).

Input families: those of tests/cute/test_own_quantisers.py (N(0,1), logit-3, outliers, zeros, tiny, subnormal,
near-min-normal, huge, mixed magnitudes, SF underflow) plus GQA shapes.

    pytest tests/cute/test_quant_fp8.py
"""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the test helpers next to this file
import quant_reference as R  # noqa: E402

DISTS = ("n01", "logit3", "outlier", "zeros", "tiny", "denorm", "near_min_normal", "huge", "mixed", "subnorm_sf")


def make(shape, dist, dtype, seed=0):
    """The input families of test_own_quantisers.make (same generator and constants)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)
    fmax = torch.finfo(dtype).max if dtype != torch.float32 else 3e38
    small16 = dtype == torch.float16
    if dist == "logit3":
        x = x * 3.0
    elif dist == "outlier":
        m = torch.rand(*shape, generator=g, device="cuda") < 1e-3
        x = torch.where(m, x * 1000.0, x)
    elif dist == "zeros":
        x = x * (torch.rand(*shape[:-1], 1, generator=g, device="cuda") < 0.5)
        x[..., :32] = 0.0
    elif dist == "tiny":
        x = x * (1e-6 if small16 else 1e-30)
    elif dist == "denorm":
        x = x * (1e-7 if small16 else 1e-39)
    elif dist == "near_min_normal":
        x = x * (1e-4 if small16 else 1e-37)
    elif dist == "huge":
        x = (x.clamp(-4, 4) / 4.0) * (min(fmax, 3e38) * 0.99)
    elif dist == "mixed":
        e = torch.randint(-24 if small16 else -60, 15 if small16 else 60, (*shape[:-1], shape[-1] // 16), generator=g,
                          device="cuda").float()
        x = x * torch.exp2(e).repeat_interleave(16, dim=-1)
    elif dist == "subnorm_sf":
        x = x * 1e-3 * (2.0 ** -torch.randint(0, 14, (*shape[:-1], 1), generator=g, device="cuda").float())
    x = x.clamp(-fmax, fmax).to(dtype)
    assert torch.isfinite(x).all()
    return x


def same(a, b, what):
    assert a.shape == b.shape, (what, a.shape, b.shape)
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    n = int((a != b).sum())
    assert n == 0, f"{what}: {n} of {a.numel()} bytes differ"


def close(a, ref, what, rtol=2e-5):
    a, ref = a.double(), ref.double().to(a.device)
    err = (a - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-300
    assert err <= rtol * scale or err == 0.0, f"{what}: max err {err:.3e} vs scale {scale:.3e}"


def fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def old_order_row_dot(a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """sum over the last dim (128) in the old kernels' order: fma(a_c, o_c, fl(a_{c+2} o_{c+2})) for c with bit 1 clear,
    then a butterfly of fp32 adds over the column bits 2, 3, 0, 6, 5, 4 (fma emulated in float64: exact products)."""
    a, o = a.float(), o.float()
    idx = torch.arange(128, device=a.device)
    lo = idx[(idx >> 1) & 1 == 0]
    v = fma(a[..., lo], o[..., lo], a[..., lo + 2] * o[..., lo + 2])
    cols = lo
    for bit in (2, 3, 0, 6, 5, 4):
        keep = ((cols >> bit) & 1) == 0
        partner = cols[keep] | (1 << bit)
        pos = torch.empty(128, dtype=torch.long, device=a.device)
        pos[cols] = torch.arange(cols.numel(), device=a.device)
        v = v[..., keep] + v[..., pos[partner]]
        cols = cols[keep]
    return v[..., 0]


# ------------------------------------------------------------------------------------------------------------- blocks
BLOCKS = [(32, 128), (32, 32), (1, 32), (1, 64), (4, 128), (128, 128), (64, 32), (32, 1), (128, 1)]


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_blocks_vs_reference(dist, dtype):
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((2, 256, 3, 128), dist, dtype)
    for rows, cols in BLOCKS:
        for fmt in (("e4m3", "e5m2") if (rows, cols) in ((32, 128), (1, 32)) else ("e4m3",)):
            tag = f"{dist} {dtype} {rows}x{cols} {fmt}"
            kw = dict(fp8_format=fmt, block_rows=rows, block_columns=cols)
            big = rows >= 32 and cols >= 32
            got = quantize_fp8_blocks(x, make_channel_major_copy=True, output_exponent_table=big,
                                      output_token_block_factors=True if big else None, **kw)
            ref = R.fp8_blocks_reference(x, **kw)
            same(got["codes"], ref["codes"], tag + " codes")
            same(got["codes_t"], ref["codes_t"], tag + " codes_t")
            if cols >= 32:
                same(got["sf_tok_raw"], ref["sf_tok_raw"], tag + " sf_tok")
            if rows >= 32:
                same(got["sf_dim_raw"], ref["sf_dim_raw"], tag + " sf_dim")
            if big:
                same(got["e"], ref["e"], tag + " e")
                same(got["fac"], ref["fac"], tag + " fac")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dist", ["n01", "outlier", "zeros", "mixed"])
def test_blocks_d_and_lse(dist, dtype):
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((1, 384, 2, 128), dist, dtype)
    o = make((1, 384, 2, 128), "n01", dtype, seed=5)
    lse = torch.randn(1, 2, 384, device="cuda")
    lse[0, 1, 3:9] = -math.inf
    for rows, cols in ((32, 128), (32, 32), (1, 32), (32, 1)):
        got = quantize_fp8_blocks(x, block_rows=rows, block_columns=cols, output_exponent_table=False,
                                  compute_d_with_output=o, shifted_log2_lse_from=lse, log2_lse_shift=8.0)
        ref = R.fp8_blocks_reference(x, block_rows=rows, block_columns=cols, compute_d_with_output=o,
                                     shifted_log2_lse_from=lse, log2_lse_shift=8.0)
        same(got["codes"], ref["codes"], "codes")
        close(got["D"], ref["D"], f"D {rows}x{cols}")
        # dequantised values from the reference scales, then the old kernels' summation order: bitwise
        a = (ref["codes"].float().permute(0, 2, 1, 3).double() * _ref_scale(x, rows, cols).double()).float()
        same(got["D"], old_order_row_dot(a, o.float().permute(0, 2, 1, 3)), f"D order {rows}x{cols}")
        same(got["lse_log2"], ref["lse_log2"], "lse_log2")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dist", ["n01", "outlier", "zeros", "mixed", "subnorm_sf"])
def test_head_dim_64(dist, dtype):
    """d = 64: one SF atom along d (K-groups 2, 3 of sf_tok and rows 64..127 of sf_dim repeat 0, 1 / 0..63), token blocks
    over the whole head dim, D in the d = 64 tree, the NVFP4 copy's single SF atom, the zeroed d-wide accumulator."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks, quantize_fp8_per_tensor
    x = make((2, 256, 3, 64), dist, dtype)
    o = make((2, 256, 3, 64), "n01", dtype, seed=5)
    lse = torch.randn(2, 3, 256, device="cuda")
    for rows, cols in ((32, 128), (32, 32), (1, 32), (128, 64)):
        got = quantize_fp8_blocks(x, block_rows=rows, block_columns=cols, compute_d_with_output=o if rows >= 32 else None,
                                  shifted_log2_lse_from=lse, log2_lse_shift=8.0, output_exponent_table=rows >= 32)
        ref = R.fp8_blocks_reference(x, block_rows=rows, block_columns=cols, shifted_log2_lse_from=lse, log2_lse_shift=8.0,
                                     compute_d_with_output=o if rows >= 32 else None)
        for k in ("codes", "sf_tok_raw", "sf_dim_raw", "e", "lse_log2"):
            if k in ref and k in got:
                same(got[k], ref[k], f"d64 {rows}x{cols} {k}")
        if rows >= 32:
            close(got["D"], ref["D"], f"d64 D {rows}x{cols}")
    acc = torch.full((2, 3, 256 * 64), 7.0, dtype=torch.float16, device="cuda")
    got = quantize_fp8_blocks(x, also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", zero_accumulator=acc)
    ref = R.fp8_blocks_reference(x, also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw")
    same(got["x4"], ref["x4"], "d64 x4")
    same(got["sf4"].permute(6, 5, 2, 4, 0, 1, 3).contiguous(), ref["sf4_raw"], "d64 sf4")
    assert int((acc != 0).sum()) == 0
    r = quantize_fp8_per_tensor(x, compute_d_with_output=o)
    rr = R.fp8_per_tensor_reference(x, compute_d_with_output=o)
    same(r["codes"], rr["codes"], "d64 per-tensor codes")
    close(r["D"], rr["D"], "d64 per-tensor D")


def _ref_scale(x, rows, cols, fmt="e4m3"):
    xf = R.flush_subnormals(x.float())
    a = xf.abs().permute(0, 2, 1, 3)
    b, h, s, d = a.shape
    blk = a.reshape(b, h, s // rows, rows, d // cols, cols).amax(dim=(3, 5))
    _, scale = R.ue8m0_exponent(blk, fmt, True)
    return scale.repeat_interleave(rows, 2).repeat_interleave(cols, 3)


@pytest.mark.parametrize("dist", ["n01", "logit3", "zeros", "mixed"])
@pytest.mark.parametrize("method", ["butterfly", "tensor_core"])
def test_blocks_rotations(dist, method):
    """The rotated (and RoPE) blocks vs the reference, whose rotation is the butterfly's order: bitwise for
    hadamard_rotation_method="butterfly"; for "tensor_core" (an fp32 summation order of its own) the codes and SF bytes
    agree up to the stated flip rate (test_quant_hadamard_tc.assert_codes_close: value level, the sign of a zero code
    ignored)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from test_quant_hadamard_tc import assert_codes_close
    x = make((2, 256, 2, 128), dist, torch.bfloat16)
    assert R.hadamard_matrix_check(x[0, :8, 0].float()) < 1e-6
    cos = torch.randn(256, 64, device="cuda")
    sin = torch.randn(256, 64, device="cuda")
    for rope in (None, "rows", "transposed"):
        c, s_ = (None, None) if rope is None else ((cos, sin) if rope == "rows" else (cos.t().contiguous(), sin.t().contiguous()))
        for had in (False, True):
            if rope is None and not had:
                continue
            for rows, cols in ((32, 128), (1, 32)):
                kw = dict(block_rows=rows, block_columns=cols, apply_hadamard_rotation=had, rope_cos=c, rope_sin=s_)
                got = quantize_fp8_blocks(x, output_exponent_table=False, hadamard_rotation_method=method, **kw)
                ref = R.fp8_blocks_reference(x, **kw)
                tag = f"{dist} rope={rope} had={had} {rows}x{cols} {method}"
                if method == "butterfly" or not had:
                    same(got["codes"], ref["codes"], tag + " codes")
                    same(got["sf_tok_raw"], ref["sf_tok_raw"], tag + " sf")
                else:
                    assert_codes_close(got["codes"], ref["codes"], "fp8", tag + " codes")
                    assert_codes_close(got["sf_tok_raw"], ref["sf_tok_raw"], "byte", tag + " sf")


@pytest.mark.parametrize("dist", ["n01", "zeros", "subnorm_sf", "mixed"])
@pytest.mark.parametrize("scaling", ["token_block", "raw", "per_tensor"])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_blocks_option_a_nvfp4_copy(dist, scaling, rounding):
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((1, 256, 2, 128), dist, torch.bfloat16)
    o = make((1, 256, 2, 128), "n01", torch.bfloat16, seed=9)
    sc = torch.tensor([0.25], device="cuda")
    kw = dict(compute_d_with_output=o, also_make_nvfp4_copy=True, nvfp4_copy_scaling=scaling,
              nvfp4_copy_rounding=rounding, nvfp4_copy_seed=77, nvfp4_copy_tensor_scale=sc)
    got = quantize_fp8_blocks(x, flush_subnormal_inputs=False, **kw)
    ref = R.fp8_blocks_reference(x, flush_subnormal_inputs=False, **kw)
    same(got["codes"], ref["codes"], "codes")
    same(got["x4"], ref["x4"], "nvfp4 codes")
    same(got["sf4"].permute(6, 5, 2, 4, 0, 1, 3).contiguous(), ref["sf4_raw"], "nvfp4 sf")
    close(got["D"], ref["D"], "D")


def test_blocks_cudnn_layout_and_defaults():
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    a = quantize_fp8_blocks(x, block_rows=1, block_columns=32, scale_factor_layout="cudnn", output_exponent_table=False)
    b = quantize_fp8_blocks(x, block_rows=1, block_columns=32, output_exponent_table=False)
    same(a["sf_tok"], b["sf_tok_raw"].reshape(-1), "cudnn row SF order")
    a = quantize_fp8_blocks(x, block_rows=32, block_columns=1, scale_factor_layout="cudnn", output_exponent_table=False)
    ref = R.fp8_blocks_reference(x, block_rows=32, block_columns=1)
    same(a["sf_dim"], ref["sf_dim_raw"].permute(2, 0, 1, 3, 4, 5, 6).reshape(-1), "cudnn along-sequence SF order")
    d = quantize_fp8_blocks(x)  # defaults: the default path's token blocks with the exponent table
    assert d["codes"].dtype == torch.float8_e4m3fn and "e" in d and "sf_dim_raw" in d
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, block_columns=1, block_rows=4)


# ------------------------------------------------------------------------------------------------------------- per tensor
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_per_tensor_vs_reference(dist, dtype):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    q = make((2, 256, 4, 128), dist, dtype)
    k = make((2, 256, 2, 128), dist, dtype, seed=1)  # GQA
    v = make((2, 256, 2, 128), "n01", dtype, seed=2)
    for fmt in ("e4m3", "e5m2"):
        for single in (True, False):
            r = quantize_fp8_per_tensor([q, k, v], fp8_format=fmt, single_launch=single,
                                        output_row_norm2_max=(False, False, True))
            for t, got, sc in zip((q, k, v), r["codes"], r["scales"].tolist()):
                ref = R.fp8_per_tensor_reference(t, fp8_format=fmt)
                assert sc == ref["scale"], (sc, ref["scale"])
                same(got, ref["codes"], f"{fmt} single={single} codes")
            close(r["row_norm2_max"][2:3], torch.tensor([R.fp8_per_tensor_reference(v)["row_norm2_max"]]), "vnorm")
    r = quantize_fp8_per_tensor(q, scales=[0.031])
    same(r["codes"], R.fp8_per_tensor_reference(q, scale=0.031)["codes"], "host scale")
    sd = torch.tensor([0.0123], dtype=torch.float64, device="cuda")
    r = quantize_fp8_per_tensor(q, scales=sd)
    same(r["codes"], R.fp8_per_tensor_reference(q, scale=0.0123)["codes"], "device scale")


@pytest.mark.parametrize("dist", ["n01", "outlier", "zeros", "mixed"])
def test_per_tensor_backward_prologue(dist):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    b, s, h = 2, 256, 3
    do = make((b, s, h, 128), dist, torch.bfloat16)
    o = make((b, s, h, 128), "n01", torch.bfloat16, seed=4)
    lse = torch.randn(b, h, s, device="cuda")
    lse[1, 2, :7] = -math.inf
    pmax = torch.rand(b, h, s, device="cuda") * 1.4
    vn2 = torch.tensor([2.5], device="cuda")
    acc = torch.full((b, h, s * 128), 3.0, dtype=torch.float16, device="cuda")
    for single in (True, False):
        r = quantize_fp8_per_tensor(do, compute_d_with_output=o, shifted_log2_lse_from=lse, log2_lse_shift=8.0,
                                    zero_accumulator=acc, ds_bound_from=(pmax, vn2), output_row_norm2_max=True,
                                    single_launch=single)
        ref = R.fp8_per_tensor_reference(do, compute_d_with_output=o, shifted_log2_lse_from=lse, log2_lse_shift=8.0,
                                         ds_bound_from=(pmax, vn2))
        same(r["codes"], ref["codes"], "codes")
        close(r["D"], ref["D"], "D")
        s32 = torch.tensor(ref["scale"], dtype=torch.float64).float()
        a = (ref["codes"].float() * s32).permute(0, 2, 1, 3)
        same(r["D"], old_order_row_dot(a, o.float().permute(0, 2, 1, 3)), "D old order")
        same(r["lse_log2"], ref["lse_log2"], "lse")
        assert int(acc.view(torch.int16).abs().sum()) == 0
        close(r["ds_bound"], torch.tensor([ref["ds_bound"]]), "dS bound", rtol=1e-5)
        close(r["row_norm2_max"], torch.tensor([ref["row_norm2_max"]]), "dO row norm", rtol=1e-5)
        acc.fill_(3.0)


@pytest.mark.parametrize("dist", ["n01", "mixed"])
def test_per_tensor_transforms(dist):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    q = make((1, 256, 2, 128), dist, torch.bfloat16)
    k = make((1, 256, 2, 128), dist, torch.bfloat16, seed=3)
    f = torch.exp2(torch.randint(-3, 4, (1, 2, 256), device="cuda").float())
    cos = torch.randn(64, 256, device="cuda")
    sin = torch.randn(64, 256, device="cuda")
    r = quantize_fp8_per_tensor([q, k], apply_hadamard_rotation=(True, False), rope_cos=cos, rope_sin=sin,
                                rope_applies_to=(True, True), row_divisors=[f, None])
    rq = R.fp8_per_tensor_reference(q, apply_hadamard_rotation=True, rope_cos=cos, rope_sin=sin, row_divisor=f)
    rk = R.fp8_per_tensor_reference(k, rope_cos=cos, rope_sin=sin)
    same(r["codes"][0], rq["codes"], "q codes")
    same(r["codes"][1], rk["codes"], "k codes")
    assert r["scales"].tolist() == [rq["scale"], rk["scale"]]


def test_per_tensor_graph_capture_and_no_sync():
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor, quantize_fp8_blocks
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    quantize_fp8_per_tensor(x)
    quantize_fp8_blocks(x)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        r1 = quantize_fp8_per_tensor(x)
        r2 = quantize_fp8_blocks(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        r3 = quantize_fp8_per_tensor(x)
        r4 = quantize_fp8_blocks(x)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    same(r3["codes"], r1["codes"], "graph per-tensor")
    same(r3["scales"], r1["scales"], "graph scale")
    same(r4["codes"], r2["codes"], "graph blocks")


_LEAF = r"""
import importlib.util, sys, types
root = importlib.util.find_spec("flash_attn").submodule_search_locations[0]
for pkg, path in (("flash_attn.cute", root + "/cute"), ("flash_attn.cute.quant", root + "/cute/quant")):
    m = types.ModuleType(pkg)
    m.__path__ = [path]
    sys.modules[pkg] = m
import flash_attn.cute.quant.fp8
heavy = sorted(k for k in sys.modules if k.startswith("flash_attn.cute.") and not k.startswith("flash_attn.cute.quant"))
assert not heavy, heavy
print("LEAF_OK")
"""


def test_quant_package_is_leaf_importable():
    """quant/ imports only torch / cutlass-dsl and its own files: it loads under stub parent packages (the benchmarks'
    cuDNN environment, where flash_attn.cute's __init__ cannot run)."""
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", _LEAF], env=env, capture_output=True, text=True, timeout=240)
    assert "LEAF_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


@pytest.mark.parametrize("nvfp4", [False, True])
def test_d_multiplier_is_one_fp32_multiply(nvfp4):
    """d_multiplier (the backward's D fold): D' = fp32(D * c) from the same launch, bitwise."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((2, 256, 3, 128), "n01", torch.bfloat16)
    o = make((2, 256, 3, 128), "n01", torch.bfloat16, seed=5)
    kw = dict(compute_d_with_output=o, also_make_nvfp4_copy=nvfp4)
    c = -(2.0 ** -8)
    a = quantize_fp8_blocks(x, **kw)
    b = quantize_fp8_blocks(x, d_multiplier=c, **kw)
    assert b["d_folded"] == c
    same(b["codes"], a["codes"], "codes")
    same(b["D"], a["D"] * torch.tensor(c, dtype=torch.float32, device=x.device), "D folded")


@pytest.mark.parametrize("case", ["plain", "d", "rotated", "b32", "d64", "rows64"])
def test_block_statistics(case):
    """output_amax / output_row_norm2_max: per-tile maxima whose maximum is the tensor's max |x| / max row ||x||^2 of
    the values quantised (rotated rows under apply_hadamard_rotation); every other output bitwise unchanged."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    d = 64 if case == "d64" else 128
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(2, 512, 3, d, device="cuda", generator=g)
         * torch.rand(1, 512, 1, 1, device="cuda", generator=g) * 9).bfloat16()
    kw = {"d": dict(compute_d_with_output=torch.randn_like(x)), "rotated": dict(apply_hadamard_rotation=True),
          "b32": dict(block_columns=32), "rows64": dict(block_rows=64)}.get(case, {})
    norm = case != "rotated"
    off = quantize_fp8_blocks(x, **kw)
    on = quantize_fp8_blocks(x, output_amax=True, output_row_norm2_max=norm, **kw)
    for n, t in off.items():
        if isinstance(t, torch.Tensor):
            assert torch.equal(t.view(torch.uint8) if t.dtype.itemsize == 1 else t,
                               on[n].view(torch.uint8) if t.dtype.itemsize == 1 else on[n]), n
    xr = x.float()
    if case == "rotated":
        from flash_attn.cute.quant.fp8 import hadamard_signs_tuple
        H = torch.ones(1, 1, device="cuda")
        while H.shape[0] < 128:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        xr = (xr * torch.tensor(hadamard_signs_tuple(4321), device="cuda")) @ H / 128 ** 0.5
    assert on["amax"].numel() == 2 * 3 * 4
    torch.testing.assert_close(on["amax"].max(), xr.abs().max(), rtol=1e-6, atol=0)
    if norm:
        torch.testing.assert_close(on["row_norm2_max"].max(), xr.square().sum(-1).max(), rtol=1e-5, atol=0)
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, block_rows=1, block_columns=32, output_amax=True)


def test_token_block_factor_of_an_all_zero_block():
    """An all-zero 32-token block's factor is its 128-token tile's largest factor, 1 for an all-zero tile (never 0): its
    codes are 0, so consumers that multiply by it are unchanged, and the fold forward's log2 f_V / 1 / f_V keep its keys
    in the softmax row sum without lifting the row max above the neighbouring blocks'."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = torch.randn(1, 256, 2, 128, device="cuda").bfloat16()
    x[:, 32:64] = 0                      # one zero block in tile 0
    x[:, 96:128] *= 64                   # tile 0's largest factor
    x[:, 128:256] = 0                    # an all-zero tile
    fac = quantize_fp8_blocks(x, output_token_block_factors=True)["fac"]
    assert torch.equal(fac[:, :, 1], fac[:, :, 3]) and (fac[:, :, 3] > fac[:, :, 0]).all(), fac[:, :, :4]
    assert torch.equal(fac[:, :, 4:], torch.ones_like(fac[:, :, 4:])), fac[:, :, 4:]
    ref = R.fp8_blocks_reference(x)["fac"]
    assert torch.equal(fac, ref.to(fac.dtype))


@pytest.mark.parametrize("route", ["fp8", "fp4"])
@pytest.mark.parametrize("causal", [False, True])
def test_fold_forward_all_zero_v_block(route, causal):
    """The token-block FOLD forward (KernelTuning.token_block_forward='fold') on a V with an all-zero 32-token block: O
    must match the reference as the default forward does. With the block's factor 0 its keys dropped out of the row sum
    (O norm 1.03x dense, 1.53x causal with the block at keys 0..31, O cos 0.85). Routes: flash_attn_fp8 and
    flash_attn_fp4 (NVFP4 Q.K, E4M3 P.V), token-block operands, no Q/K rotation (as the former legacy entries)."""
    from flash_attn.cute.lowprec import FP4AttnOptions, FP8AttnOptions, KernelTuning, flash_attn_fp4, flash_attn_fp8
    g = torch.Generator(device="cuda").manual_seed(0)
    b, s, h, d = 1, 1024, 4, 128
    q, k, v = (torch.randn(b, s, h, d, device="cuda", generator=g).bfloat16() for _ in range(3))
    v[:, :32] = 0
    S = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * d ** -0.5
    if causal:
        S = S.masked_fill(torch.ones(s, s, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    ref = torch.einsum("bhqk,bkhd->bqhd", S.softmax(-1), v.float())
    tuning = KernelTuning(token_block_forward="fold")
    if route == "fp8":
        o = flash_attn_fp8(q, k, v, causal=causal, qk_hadamard=False,
                           options=FP8AttnOptions(operand_scaling="token_block", tuning=tuning))
    else:
        o = flash_attn_fp4(q, k, v, causal=causal, qk_hadamard=False,
                           options=FP4AttnOptions(operand_scaling="token_block", tuning=tuning))
    o = o.float()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    ratio = (o.norm() / ref.norm()).item()
    assert cos > (0.997 if route == "fp8" else 0.989) and abs(ratio - 1) < 0.005, (cos, ratio)


# ------------------------------------------------------------------------------------------- token-local: causal-safe scales
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hd", [128, 64])
def test_blocks_token_local_exponents(dist, dtype, hd):
    """token_local_exponents: codes / sf_tok / D per token, sf_dim / e / fac the block's (bitwise the default's), the
    per-token factors; token_axis_scale="unit"; and no token's codes depend on a later token of its block."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((2, 256, 3, hd), dist, dtype)
    o = make((2, 256, 3, hd), "n01", dtype, seed=3)
    for unit in (False, True):
        tag = f"{dist} {dtype} d{hd} unit={unit}"
        axis = "unit" if unit else "block"
        got = quantize_fp8_blocks(x, output_token_block_factors=True, compute_d_with_output=o,
                                  token_local_exponents=True, token_axis_scale=axis)
        ref = R.fp8_blocks_reference(x, compute_d_with_output=o, token_local_exponents=True, token_axis_scale=axis)
        for k in ("codes", "sf_tok_raw", "sf_dim_raw", "e", "fac"):
            same(got[k], ref[k], f"{tag} {k}")
        same(got["token_factors"], ref["token_factors"].contiguous(), tag + " token_factors")
        if dist != "huge":   # D overflows fp32 there (the reference sums in float64)
            close(got["D"], ref["D"], tag + " D", rtol=1e-5)
        dflt = quantize_fp8_blocks(x, output_token_block_factors=True)
        for k in ("e", "fac") + (() if unit else ("sf_dim_raw",)):
            same(got[k], dflt[k], f"{tag} {k} vs token blocks")
    # the factor table rows written in place (views), with log2 f and 1 / f
    tab = torch.empty(3, 2, 3, 256, dtype=torch.float32, device="cuda")
    quantize_fp8_blocks(x, token_local_exponents=True, output_token_factors=tab[0], output_token_log2_factors=tab[1],
                        output_token_inverse_factors=tab[2])
    f = R.fp8_blocks_reference(x, token_local_exponents=True)["token_factors"]
    same(tab[0], f.contiguous(), "table factors")
    same(tab[1], torch.log2(f).contiguous(), "table log2 factors")
    same(tab[2], (1.0 / f).contiguous(), "table inverse factors")
    # causality: a later token of the same 32-token block moves no earlier token's codes / factor
    x2 = x.clone()
    x2[:, 40] = x2[:, 40] * 64
    a = quantize_fp8_blocks(x, token_local_exponents=True)
    b_ = quantize_fp8_blocks(x2, token_local_exponents=True)
    same(a["codes"][:, :40], b_["codes"][:, :40], "earlier codes")
    same(a["token_factors"][:, :, :40], b_["token_factors"][:, :, :40], "earlier factors")


@pytest.mark.parametrize("dist", ["n01", "outlier", "zeros", "tiny", "huge", "mixed"])
def test_per_tensor_power_of_two_scale(dist):
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    q = make((2, 256, 4, 128), dist, torch.bfloat16)
    k = make((2, 256, 2, 128), dist, torch.bfloat16, seed=1)
    for single in (False, True):
        for rows in (False, True):   # the flat kernel and the rows kernel (row option: the row norm)
            r = quantize_fp8_per_tensor([q, k], power_of_two_scale=True, single_launch=single,
                                        output_row_norm2_max=(rows, False))
            for t, got, sc in zip((q, k), r["codes"], r["scales"].tolist()):
                ref = R.fp8_per_tensor_reference(t, power_of_two_scale=True)
                assert sc == ref["scale"], (dist, single, rows, sc, ref["scale"])
                same(got, ref["codes"], f"{dist} single={single} rows={rows} pow2 codes")
    with pytest.raises(ValueError):
        quantize_fp8_per_tensor(q, scales=[0.5], power_of_two_scale=True)


# ------------------------------------------------------------------------------------------------- head-dim padding
@pytest.mark.parametrize("dist", ["n01", "outlier", "zeros", "mixed", "subnorm_sf"])
@pytest.mark.parametrize("d,dp", [(96, 128), (80, 128), (112, 128), (32, 64), (48, 64), (16, 64)])
@pytest.mark.parametrize("cols", [128, 32])
def test_blocks_pad_head_dim(d, dp, dist, cols):
    """pad_head_dim_to: a d-wide input quantised as if zero-padded to dp, in one read -- every output bitwise that of
    the explicitly zero-padded tensor (codes, both SF replications, the exponent table, the token-block factors, the
    channel-major copy, D against the padded O, the statistics; the zeroed accumulator sized at dp)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    x = make((2, 256, 3, d), dist, torch.bfloat16)
    xp = torch.nn.functional.pad(x, (0, dp - d))
    o = make((2, 256, 3, dp), "n01", torch.bfloat16, seed=5)
    lse = torch.randn(2, 3, 256, device="cuda")
    kw = dict(block_columns=cols, compute_d_with_output=o, shifted_log2_lse_from=lse, log2_lse_shift=8.0,
              output_token_block_factors=True, output_amax=True, output_row_norm2_max=True)
    acc_a = torch.full((2, 3, 256 * dp), 7.0, dtype=torch.float16, device="cuda")
    acc_b = acc_a.clone()
    got = quantize_fp8_blocks(x, pad_head_dim_to=dp, zero_accumulator=acc_a, **kw)
    ref = quantize_fp8_blocks(xp, zero_accumulator=acc_b, **kw)
    for k_ in ("codes", "sf_tok_raw", "sf_dim_raw", "e", "fac", "D", "lse_log2", "amax", "row_norm2_max"):
        same(got[k_], ref[k_], f"pad {d}->{dp} {cols} {k_}")
    assert int((acc_a != 0).sum()) == 0
    got = quantize_fp8_blocks(x, pad_head_dim_to=dp, block_columns=cols, make_channel_major_copy=True)
    ref = quantize_fp8_blocks(xp, block_columns=cols, make_channel_major_copy=True)
    same(got["codes_t"], ref["codes_t"], f"pad {d}->{dp} {cols} codes_t")
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, pad_head_dim_to=dp, token_local_exponents=True)
    with pytest.raises(ValueError):
        quantize_fp8_blocks(x, pad_head_dim_to=dp, block_rows=1, block_columns=32)

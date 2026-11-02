"""flash_attn.cute.quant.fp4 against the eager references written from the format definitions (tests/cute/quant_reference.py): codes,
scale-factor bytes and every side output bitwise equal, for every option and for the cross-products the call sites use,
over N(0,1), logit-3, outlier, zero, tiny, subnormal, near-min-normal, huge, mixed-magnitude and SF-underflow inputs
(own-quant's families), several (b, s, h), bf16 / fp16 / fp32. Stochastic rounding: bitwise for a seed, and unbiased.

    pytest tests/cute/test_quant_fp4.py
Compilation dominates: FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 keeps the kernels on disk; CASES below can also be
compiled ahead (CPU, fake tensors)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # quant_reference, the test helper next to this file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DISTS = ("n01", "logit3", "outlier", "zeros", "tiny", "denorm", "near_min_normal", "huge", "mixed", "subnorm_sf")
SHAPES = ((1, 128, 1, 128), (2, 256, 3, 128), (1, 384, 4, 128))


def make(shape, dist, dtype, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float32)
    fmax = torch.finfo(dtype).max if dtype != torch.float32 else 1e36
    if dist == "logit3":
        x = x * 3.0
    elif dist == "outlier":
        m = torch.rand(*shape, generator=g) < 1e-3
        x = torch.where(m, x * 1000.0, x)
        x[..., 7] *= 50.0
    elif dist == "zeros":
        x = x * (torch.rand(*shape[:-1], 1, generator=g) < 0.5)
        x[..., :32] = 0.0
    elif dist == "tiny":
        x = x * (1e-6 if dtype == torch.float16 else 1e-30)
    elif dist == "denorm":
        x = x * (1e-7 if dtype == torch.float16 else 1e-39)
    elif dist == "near_min_normal":
        x = x * (1e-4 if dtype == torch.float16 else 1e-37)
    elif dist == "huge":
        x = (x.clamp(-4, 4) / 4.0) * (fmax * 0.99)
    elif dist == "mixed":
        lo, hi = (-24, 15) if dtype == torch.float16 else (-60, 60)
        e = torch.randint(lo, hi, (*shape[:-1], shape[-1] // 16), generator=g).float()
        x = x * torch.exp2(e).repeat_interleave(16, dim=-1)
    elif dist == "subnorm_sf":
        x = x * 1e-3 * (2.0 ** -torch.randint(0, 14, (*shape[:-1], 1), generator=g).float())
    x = x.clamp(-fmax, fmax).to(dtype)
    return x.to(device)


def rope_tables(s, layout, device="cuda", half=64):
    pos = torch.arange(s, dtype=torch.float32)
    ang = pos[:, None] / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))[None, :]
    c, sn = ang.cos(), ang.sin()
    if layout == "transposed":
        c, sn = c.t().contiguous(), sn.t().contiguous()
    return c.to(device), sn.to(device)


def build_kwargs(opts, x, device="cuda", fake=False):
    """Materialise the tensors an option set names ('scale', 'rowfac', 'slot', 'tiles', 'O', 'rope_*', 'gs_dev')."""
    b, s, h, d = x.shape
    kw = {}
    for k, v in opts.items():
        if k == "_d":
            continue
        if k == "rope":
            kw["rope_cos"], kw["rope_sin"] = rope_tables(s, v, device, half=d // 2)
        elif k == "e4m3":
            kw["also_make_e4m3_per_tensor_copy"] = True
            amax = 1.0 if fake else max(float(x.float().abs().amax()), 1e-30)
            kw["e4m3_copy_scale"] = torch.tensor([max(amax / 448.0, 1e-12)], dtype=torch.float32, device=device)
            if v == "unrotated":
                kw["e4m3_copy_of_unrotated_values"] = True
        elif k == "rowfac":
            g = torch.Generator(device="cpu").manual_seed(3)
            kw["per_32_row_power_of_two_factors"] = (torch.empty(b, h, s // 32, device=device) if fake else
                                                     torch.exp2(torch.randint(-4, 5, (b, h, s // 32), generator=g).float()).to(device))
        elif k == "norm":
            kw["output_row_norm_max"] = torch.zeros(1, device=device)
        elif k == "tiles":
            kw["output_tile_amax"] = torch.zeros(b * h * (s // 128), device=device)
        elif k == "O":
            odt = x.dtype if x.dtype != torch.float32 else torch.bfloat16
            kw["compute_d_with_output"] = (torch.empty(x.shape, dtype=odt, device=device) if fake
                                           else make(x.shape, "n01", odt, seed=9, device=device))
        elif k == "gs_dev":
            kw["global_scale"] = torch.tensor([v], dtype=torch.float32, device=device)
        else:
            kw[k] = v
    return kw


# (name, function "rows" | "2d", options, input dtypes, distributions)
ALL_D = DISTS
CORE_D = ("n01", "outlier", "zeros", "tiny", "denorm", "huge", "subnorm_sf")
FEW_D = ("n01", "outlier", "zeros", "tiny")
BF = (torch.bfloat16,)
BF_FP = (torch.bfloat16, torch.float16)
CASES = [
    ("nvfp4", "rows", {}, BF_FP, ALL_D),
    ("mxfp4", "rows", {"scale_group_size": 32}, BF_FP, ALL_D),
    ("nvfp4_trans_keyorder", "rows", {"make_transposed_copy": True}, BF_FP, ALL_D),
    ("nvfp4_trans_natural_headmajor", "rows", {"make_transposed_copy": True, "transposed_copy_in_kernel_key_order": False,
                                               "transposed_copy_layout": "batch_head_dim_seq"}, BF, CORE_D),
    ("mxfp4_kmajor_only", "rows", {"scale_group_size": 32, "make_row_copy": False, "make_transposed_copy": True,
                                   "transposed_copy_in_kernel_key_order": False,
                                   "transposed_copy_layout": "batch_head_dim_seq"}, BF_FP, CORE_D),
    ("fo6", "rows", {"use_four_over_six": True, "make_transposed_copy": True}, BF_FP, CORE_D),
    ("sr", "rows", {"rounding": "stochastic", "stochastic_rounding_seed": 1234, "make_transposed_copy": True}, BF_FP, CORE_D),
    ("mxfp4_sr", "rows", {"scale_group_size": 32, "rounding": "stochastic", "stochastic_rounding_seed": 5}, BF, FEW_D),
    ("unit", "rows", {"rounding": "unit", "make_transposed_copy": True}, BF, FEW_D),
    ("eden_d", "rows", {"rounding": "ms_eden", "stochastic_rounding_seed": 3, "O": 1}, BF, FEW_D),
    ("global_host_d", "rows", {"global_scale": 0.37, "O": 1}, BF, FEW_D),
    ("global_dev_sr_d", "rows", {"gs_dev": 2.5, "rounding": "stochastic", "stochastic_rounding_seed": 7, "O": 1}, BF_FP, FEW_D),
    ("rowfac_sr_d", "rows", {"rowfac": 1, "rounding": "stochastic", "stochastic_rounding_seed": 9, "O": 1}, BF_FP, FEW_D),
    ("e4m3", "rows", {"e4m3": "rotated"}, BF_FP, CORE_D),
    ("hadamard_e4m3_norm", "rows", {"apply_hadamard_rotation": True, "e4m3": "rotated", "norm": 1}, BF_FP, CORE_D),
    ("hadamard_mx_e4m3plain", "rows", {"apply_hadamard_rotation": True, "scale_group_size": 32, "e4m3": "unrotated"}, BF, FEW_D),
    ("amax_only", "rows", {"make_row_copy": False, "apply_hadamard_rotation": True, "tiles": 1}, BF_FP, FEW_D),
    ("rows_tiles_norm", "rows", {"tiles": 1, "norm": 1}, BF, FEW_D),
    ("rope_t_trans", "rows", {"rope": "transposed", "make_transposed_copy": True}, BF_FP, FEW_D),
    ("rope_r", "rows", {"rope": "rows"}, BF, FEW_D),
    ("fp32_trans", "rows", {"make_transposed_copy": True}, (torch.float32,), FEW_D),
    ("all_rows", "rows", {"rope": "transposed", "apply_hadamard_rotation": True, "e4m3": "rotated", "norm": 1,
                          "make_transposed_copy": True, "gs_dev": 0.5, "rounding": "stochastic",
                          "stochastic_rounding_seed": 11, "rowfac": 1, "tiles": 1}, BF, FEW_D),
    ("2d", "2d", {"make_transposed_copy": True, "transposed_copy_in_kernel_key_order": False}, BF_FP, ALL_D),
    ("2d_keyorder", "2d", {"make_transposed_copy": True}, BF, CORE_D),
    ("2d_sr", "2d", {"make_transposed_copy": True, "rounding": "stochastic", "stochastic_rounding_seed": 77}, BF_FP, CORE_D),
    ("2d_unit_fp32", "2d", {"make_transposed_copy": True, "transposed_copy_in_kernel_key_order": False,
                            "rounding": "unit"}, (torch.float32,), FEW_D),
    ("2d_mx", "2d", {"scale_group_size": 32, "make_transposed_copy": True}, BF_FP, CORE_D),
    ("2d_hadamard", "2d", {"make_transposed_copy": True, "apply_hadamard_rotation": True}, BF, FEW_D),
    ("2d_d_global", "2d", {"O": 1, "global_scale": 2.0}, BF, FEW_D),
    ("nvfp4_d64", "rows", {"_d": 64, "make_transposed_copy": True}, BF_FP, CORE_D),
    ("nvfp4_d64_pad128", "rows", {"_d": 64, "padded_head_dim": 128, "make_transposed_copy": True, "e4m3": "rotated"},
     BF, CORE_D),
    ("mxfp4_d64_pad128", "rows", {"_d": 64, "scale_group_size": 32, "padded_head_dim": 128}, BF, FEW_D),
    ("hadamard_d64_pad128", "rows", {"_d": 64, "padded_head_dim": 128, "apply_hadamard_rotation": True, "norm": 1},
     BF, FEW_D),
    ("rope_d64_sr_d", "rows", {"_d": 64, "rope": "transposed", "rounding": "stochastic", "stochastic_rounding_seed": 4,
                               "O": 1}, BF, FEW_D),
    ("2d_d64", "2d", {"_d": 64, "make_transposed_copy": True}, BF, CORE_D),
    ("nvfp4_d256", "rows", {"_d": 256}, BF_FP, CORE_D),
    ("mxfp4_d256_e4m3", "rows", {"_d": 256, "scale_group_size": 32, "e4m3": "rotated"}, BF, FEW_D),
    ("all_2d", "2d", {"rope": "transposed", "e4m3": "rotated", "norm": 1, "rowfac": 1, "rounding": "stochastic",
                      "stochastic_rounding_seed": 13, "make_transposed_copy": True, "gs_dev": 3.0}, BF, FEW_D),
]
PARAMS = [(n, f, o, dt, dist, shp) for n, f, o, dts, dists in CASES for dt in dts for dist in dists for shp in SHAPES]


def _equal(a, b):
    if a.dtype in (torch.float32, torch.float64):
        return torch.equal(a.contiguous().view(torch.int32 if a.dtype == torch.float32 else torch.int64),
                           b.contiguous().to(a.dtype).view(torch.int32 if a.dtype == torch.float32 else torch.int64))
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("name,fn,opts,dtype,dist,shape", PARAMS,
                         ids=[f"{p[0]}-{str(p[3])[6:]}-{p[4]}-{'x'.join(map(str, p[5]))}" for p in PARAMS])
def test_against_reference(name, fn, opts, dtype, dist, shape):
    from flash_attn.cute.quant import fp4
    import quant_reference as R
    if "_d" in opts:
        shape = shape[:3] + (opts["_d"],)
    x = make(shape, dist, dtype)
    kw = build_kwargs(opts, x)
    new = (fp4.quantize_fp4_rows if fn == "rows" else fp4.quantize_fp4_blocks_2d)(x, **kw)
    ref = (R.fp4_rows_reference if fn == "rows" else R.fp4_blocks_2d_reference)(x, **kw)
    for k in ("codes", "scale_factors", "transposed_codes", "transposed_scale_factors", "e4m3_copy", "d"):
        assert (k in new) == (k in ref), k
        if k in new:
            assert new[k].shape == ref[k].shape, (k, new[k].shape, ref[k].shape)
            ok = _equal(new[k], ref[k])
            if not ok:
                diff = (new[k].contiguous().view(torch.uint8) != ref[k].contiguous().view(torch.uint8))
                pytest.fail(f"{k}: {int(diff.sum())} / {diff.numel()} bytes differ")
    if "output_row_norm_max" in kw:
        assert _equal(kw["output_row_norm_max"], ref["row_norm_max"]), (kw["output_row_norm_max"], ref["row_norm_max"])
    if "output_tile_amax" in kw:
        assert _equal(kw["output_tile_amax"], ref["tile_amax"])


@pytest.mark.parametrize("fn,G", [("rows", 16), ("rows", 32), ("2d", 16)])
def test_stochastic_rounding_unbiased(fn, G):
    """E[dequant(SR(x))] = x. Given the (seed-independent) scale, each element rounds to one of its two E2M1 neighbours
    lo, lo + step with P(up) = (|x / SF| - lo) / step; over N seeds the sample mean must lie within 5 theoretical standard
    errors of x for >= 99.99 % of elements, and the mean signed error over all elements must be ~0."""
    from flash_attn.cute.quant import fp4
    import quant_reference as R
    x = make((1, 128, 2, 128), "n01", torch.bfloat16)
    f = fp4.quantize_fp4_rows if fn == "rows" else fp4.quantize_fp4_blocks_2d
    n = 256
    acc = torch.zeros(x.shape, device="cuda", dtype=torch.float64)
    sf_ref = None
    for seed in range(n):
        o = f(x, scale_group_size=G, rounding="stochastic", stochastic_rounding_seed=1000 + seed)
        c = o["codes"]
        nib = torch.stack([c & 0xF, c >> 4], -1).reshape(*c.shape[:-1], 128)
        tbl = R.table_from_sf_atoms(o["scale_factors"]).permute(0, 2, 1, 3)
        sfv = tbl.contiguous().view(torch.float8_e4m3fn).float() if G == 16 else torch.exp2(tbl.float() - 127)
        sfe = sfv.repeat_interleave(G, -1)
        sf_ref = sfe if sf_ref is None else sf_ref
        assert torch.equal(sfe, sf_ref)                      # the scale does not depend on the seed
        acc += (R.e2m1_decode(nib) * sfe).double()
    mean = acc / n
    a = (x.float() / sf_ref).abs()
    lo = torch.where(a < 2, torch.floor(a * 2) / 2, torch.where(a < 4, torch.floor(a), torch.full_like(a, 4.0)))
    step = torch.where(a < 2, torch.full_like(a, 0.5), torch.where(a < 4, torch.ones_like(a), torch.full_like(a, 2.0)))
    p = ((a - lo) / step).clamp(0, 1)
    sd = (step * sf_ref * (p * (1 - p) / n).sqrt()).double()
    err = (mean - x.double()).abs()
    frac = (err <= 5 * sd + 1e-7 * x.double().abs() + 1e-30).float().mean().item()
    assert frac >= 0.9999, frac
    assert abs((mean - x.double()).mean().item()) < 2e-3


LARGE = [c for c in CASES if c[0] in ("nvfp4_trans_keyorder", "mxfp4_kmajor_only", "2d", "sr", "all_2d", "fo6")]


@pytest.mark.parametrize("name,fn,opts", [(c[0], c[1], c[2]) for c in LARGE], ids=[c[0] for c in LARGE])
def test_large_multi_tile_per_cta(name, fn, opts):
    """1024 tiles: the persistent schedule (kernels with a transposed copy) gives every CTA several tiles."""
    from flash_attn.cute.quant import fp4
    import quant_reference as R
    x = make((2, 4096, 16, 128), "n01", torch.bfloat16, seed=5)
    kw = build_kwargs(opts, x)
    new = (fp4.quantize_fp4_rows if fn == "rows" else fp4.quantize_fp4_blocks_2d)(x, **kw)
    ref = (R.fp4_rows_reference if fn == "rows" else R.fp4_blocks_2d_reference)(x, **kw)
    for k in new:
        assert _equal(new[k], ref[k]), k


@pytest.mark.parametrize("G", [16, 32])
def test_dequantize_rows(G):
    """dequantize_fp4_rows inverts the row copy exactly (against the reference dequantiser)."""
    from flash_attn.cute.quant import fp4
    import quant_reference as R
    x = make((2, 256, 3, 128), "logit3", torch.bfloat16)
    r = fp4.quantize_fp4_rows(x, scale_group_size=G)
    for dt in (torch.bfloat16, torch.float32):
        y = fp4.dequantize_fp4_rows(r["codes"], r["scale_factors"], dt, scale_group_size=G)
        ref = R.fp4_dequantize_rows_reference(r["codes"], r["scale_factors"], dt, G)
        assert torch.equal(y, ref)


@pytest.mark.parametrize("key_channel", ["off", "cut", "all", "fold"])
@pytest.mark.parametrize("r_gqa", [1, 4])
def test_causal_diagonal_against_reference(key_channel, r_gqa):
    """quantize_fp4_causal_diagonal against its eager reference (quant_reference.fp4_causal_diagonal_reference)."""
    from flash_attn.cute.quant import fp4_diagonal
    import quant_reference as R
    b, s, hk = 2, 768, 2
    q = make((b, s, hk * r_gqa, 128), "n01", torch.bfloat16, seed=1)
    k = make((b, s, hk, 128), "outlier", torch.bfloat16, seed=2)
    v = make((b, s, hk, 128), "logit3", torch.bfloat16, seed=3)
    kw = dict(key_channel=key_channel, query=q, make_key_row_copy=True, make_option_a_key_factors=True,
              make_key_norm_prefix_maxima=True)
    new = fp4_diagonal.quantize_fp4_causal_diagonal(k, v, 0.0883883, **kw)
    ref = R.fp4_causal_diagonal_reference(k, v, 0.0883883, **kw)
    assert set(new) == set(ref), (set(new) ^ set(ref))
    for n in new:
        assert _equal(new[n], ref[n]), n

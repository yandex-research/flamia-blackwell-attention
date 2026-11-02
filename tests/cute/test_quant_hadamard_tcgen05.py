"""hadamard_rotation_method="tcgen05" (opt-in; quant._tcgen05, quant._fp8_tcgen05): the rotation on tcgen05 UMMA.

Contract.
  * Every output is BITWISE the same quantiser run on fp32 input without rotation, fed the rows rotated exactly as the
    tcgen05 kernels rotate them (quant._tcgen05.rotate_rows of the RoPE'd, rounded, zero-padded rows): the tcgen05
    kernels change the rotation and nothing else. (The scope="nvfp4_copy" E4M3 side: bitwise the unrotated launch.)
  * Against the default method (tensor_core) the fp32 rotated values differ in the last bits (the tensor core's single
    128-term accumulation vs H_8 on mma.sync + an fp32 butterfly), so the codes may differ at the value level: at most
    FLIP_RATE of the elements (measured: 0 .. 3e-6 of the codes), on the 128-token tiles
    whose rows cannot overflow (as test_quant_hadamard_tc).
  * Where tcgen05 has no form (fp32 input, the per-tensor kernel, the causal-diagonal kernel, rows rescaled before
    the rotation) the call is REFUSED with a ValueError naming hadamard_rotation_method="tensor_core"; nothing falls back.
  * The default (None) stays tensor_core.

    pytest tests/cute/test_quant_hadamard_tcgen05.py
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
S4 = dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand")


def make(shape, dist, dtype, seed=0):
    from test_quant_fp8 import make as make8
    return make8(shape, dist, dtype, seed)


def _helpers():
    from test_quant_hadamard_tc import assert_codes_close, safe_tiles
    return assert_codes_close, safe_tiles


def _rotate(x):
    from flash_attn.cute.quant import _tcgen05 as T5
    return T5.rotate_rows(x, ROT.hadamard_signs(ROT.DEFAULT_HADAMARD_SIGN_SEED))


def _clone(d):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def _assert_bitwise(got, want, tag, keys=None):
    keys = [k for k in got if isinstance(got[k], torch.Tensor) and isinstance(want.get(k), torch.Tensor)] \
        if keys is None else keys
    assert keys, tag
    for k in keys:
        a, b = got[k].contiguous().view(torch.uint8), want[k].contiguous().view(torch.uint8)
        assert a.shape == b.shape, (tag, k, a.shape, b.shape)
        n = int((a != b).sum())
        assert n == 0, f"{tag}: {k} differs in {n} bytes"


def _rope_tables(s, kind, seed=5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cos = torch.randn(s, 64, device="cuda", generator=g)
    sin = torch.randn(s, 64, device="cuda", generator=g)
    if kind == "transposed":
        cos, sin = cos.t().contiguous(), sin.t().contiguous()
    return cos, sin


# -------------------------------------------------------------------------------------------------- quantize_fp8_blocks
BLOCK_ROWS = [   # (name, kwargs; "_..." keys are built per call)
    ("tb128_fac_amax", dict(output_token_block_factors=True, output_amax=True)),
    ("tb64", dict(block_columns=64, output_token_block_factors=True)),
    ("tb32", dict(block_columns=32, output_amax=True)),
    ("tb128_e5m2_noexp", dict(fp8_format="e5m2", output_exponent_table=False)),
    ("tb128_noflush", dict(flush_subnormal_inputs=False, output_token_block_factors=True)),
    ("tb128_cudnn", dict(scale_factor_layout="cudnn")),
    ("s4_raw_qk", dict(S4, output_token_block_factors=True, output_amax=True)),
    ("s4_raw_dp", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw")),
    ("a_tb_dp_sr", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="token_block", nvfp4_copy_rounding="stochastic",
                        nvfp4_copy_seed=11)),
    ("a_tb_dp", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="token_block", output_token_block_factors=True)),
    ("pt_dp", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="per_tensor", _pt=True)),
    ("pt_qk_noflush", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="per_tensor", nvfp4_copy_encoding="qk_operand",
                           flush_subnormal_inputs=False, _pt=True)),
    ("scope_raw_qk", dict(S4, hadamard_rotation_scope="nvfp4_copy", output_token_block_factors=True, output_amax=True)),
    ("scope_pt", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="per_tensor", hadamard_rotation_scope="nvfp4_copy",
                      _pt=True)),
    ("rope_t", dict(output_token_block_factors=True, _rope="transposed")),
    ("rope_rows_s4", dict(S4, _rope="rows")),
    ("rope_scope", dict(S4, hadamard_rotation_scope="nvfp4_copy", output_amax=True, _rope="transposed")),
    ("stream_cm_d", dict(make_channel_major_copy=True, _d=True)),
    ("stream_lse_zero", dict(_lse=True, _zero=True, output_token_block_factors=True)),
    ("stream_rows1x32", dict(block_rows=1, block_columns=32, output_exponent_table=False)),
    ("stream_rows128", dict(block_rows=128, block_columns=128, output_exponent_table=False, output_amax=True)),
    ("stream_rows64_rope", dict(block_rows=64, block_columns=64, _rope="rows")),
    ("stream_d_nvfp4", dict(also_make_nvfp4_copy=True, nvfp4_copy_scaling="token_block", _d=True)),
    ("cols1_rope", dict(block_rows=32, block_columns=1, output_exponent_table=False, _rope="rows")),
    ("cols1_128_e5m2", dict(block_rows=128, block_columns=1, output_exponent_table=False, fp8_format="e5m2")),
]


def _block_kwargs(kw, x):
    kw = dict(kw)
    b, s, h, _ = x.shape
    rope = kw.pop("_rope", None)
    if rope is not None:
        kw["rope_cos"], kw["rope_sin"] = _rope_tables(s, rope)
    if kw.pop("_pt", False):
        kw["nvfp4_copy_tensor_scale"] = torch.tensor([0.25], device="cuda")
    if kw.pop("_d", False):
        kw["compute_d_with_output"] = make((b, s, h, 128), "n01", x.dtype, seed=9)
    if kw.pop("_lse", False):
        kw["shifted_log2_lse_from"] = torch.randn(b, h, s, device="cuda",
                                                  generator=torch.Generator(device="cuda").manual_seed(2))
    if kw.pop("_zero", False):
        kw["zero_accumulator"] = torch.ones(b * h * s * 128, dtype=torch.float16, device="cuda")
    return kw


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", [r[0] for r in BLOCK_ROWS])
def test_blocks_tcgen05_vs_emulation_and_tensor_core(name, dtype, dist):
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    assert_codes_close, safe_tiles = _helpers()
    x = make((2, 256, 3, 128), dist, dtype)
    kw = _block_kwargs(dict(next(r for r in BLOCK_ROWS if r[0] == name)[1]), x)
    tag = f"{name} {dtype} {dist}"
    got = _clone(quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="tcgen05", **kw))
    assert ROT.ROTATION_METHOD_LOG["quantize_fp8_blocks"] == ("tcgen05", "requested")
    # (1) bitwise the emulation
    ekw = {k: v for k, v in kw.items() if k not in ("rope_cos", "rope_sin", "hadamard_rotation_scope")}
    xr = x if "rope_cos" not in kw else R.apply_rope(x, kw["rope_cos"], kw["rope_sin"]).to(dtype)
    emu = _clone(quantize_fp8_blocks(_rotate(xr), **ekw))
    if kw.get("hadamard_rotation_scope") == "nvfp4_copy":
        raw = quantize_fp8_blocks(xr, **ekw)
        for k in ("codes", "sf_tok_raw", "sf_dim_raw", "sf_tok", "sf_dim", "e", "fac", "amax"):
            if k in raw:
                emu[k] = raw[k].clone()
    _assert_bitwise(got, emu, tag + " vs emulation")
    # (2) the tensor_core method at the value level, on the tiles that cannot overflow
    tc = _clone(quantize_fp8_blocks(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core", **kw))
    fin = safe_tiles(x, kw.get("rope_cos"), kw.get("rope_sin"))
    assert_codes_close(got["codes"][fin], tc["codes"][fin], "fp8", tag + " codes vs tensor_core")
    if bool(fin.all()):
        for k in ("sf_tok_raw", "sf_dim_raw", "e", "fac", "sf4"):
            if k in got:
                assert_codes_close(got[k], tc[k], "byte", f"{tag} {k} vs tensor_core")
        if "x4" in got:
            assert_codes_close(got["x4"], tc["x4"], "nib", tag + " x4 vs tensor_core")


# ------------------------------------------------------------------------------------------------- quant.fp4 tile kernel
FP4_ROWS = [
    ("rows", {}),
    ("mx", dict(scale_group_size=32)),
    ("rows_t", dict(make_transposed_copy=True)),
    ("rows_t_bhds_swap", dict(make_transposed_copy=True, transposed_copy_layout="batch_head_dim_seq",
                              make_transposed_sf_halves_swapped=True)),
    ("rows_t_only", dict(make_row_copy=False, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False)),
    ("e4_rot", dict(also_make_e4m3_per_tensor_copy=True, _e4=True)),
    ("e4_unrot", dict(also_make_e4m3_per_tensor_copy=True, e4m3_copy_of_unrotated_values=True, _e4=True)),
    ("fo6", dict(use_four_over_six=True)),
    ("gs_host", dict(global_scale=2.0)),
    ("sr", dict(rounding="stochastic", stochastic_rounding_seed=7)),
    ("rowfac", dict(_rowfac=True)),
    ("norm_amax", dict(_norm=True, _tamax=True)),
    ("rope_t", dict(_rope="transposed")),
    ("rope_rows_t", dict(_rope="rows", make_transposed_copy=True)),
    ("pad64", dict(padded_head_dim=128, _d=64)),
    ("2d", dict(make_transposed_copy=True, _2d=True)),
    ("2d_mx", dict(scale_group_size=32, make_transposed_copy=True, _2d=True)),
]


@pytest.mark.parametrize("dist", ["n01", "logit3", "zeros", "mixed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", [r[0] for r in FP4_ROWS])
def test_fp4_tcgen05_vs_emulation_and_tensor_core(name, dtype, dist):
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows, quantize_fp4_blocks_2d
    assert_codes_close, _ = _helpers()
    kw = dict(next(r for r in FP4_ROWS if r[0] == name)[1])
    din = kw.pop("_d", 128)
    x = make((2, 256, 3, din), dist, dtype)
    b, s, h, _ = x.shape
    fn = quantize_fp4_blocks_2d if kw.pop("_2d", False) else quantize_fp4_rows
    if kw.pop("_e4", False):
        kw["e4m3_copy_scale"] = torch.tensor([0.05], dtype=torch.float64, device="cuda")
    rowfac, norm, tamax = kw.pop("_rowfac", False), kw.pop("_norm", False), kw.pop("_tamax", False)
    rope = kw.pop("_rope", None)
    if rope is not None:
        kw["rope_cos"], kw["rope_sin"] = _rope_tables(s, rope)
    tag = f"{name} {dtype} {dist}"

    def call(xx, **k2):
        k = dict(k2)
        if rowfac:
            g = torch.Generator(device="cuda").manual_seed(3)
            k["per_32_row_power_of_two_factors"] = torch.exp2(torch.randint(-3, 4, (b, h, s // 32), device="cuda",
                                                                            generator=g).float())
        if norm:
            k["output_row_norm_max"] = torch.zeros(1, device="cuda")
        if tamax:
            k["output_tile_amax"] = torch.empty(b * h * (s // 128), device="cuda")
        o = _clone(fn(xx, **k))
        if norm:
            o["row_norm_max"] = k["output_row_norm_max"].clone()
        if tamax:
            o["tile_amax"] = k["output_tile_amax"].clone()
        return o

    got = call(x, apply_hadamard_rotation=True, hadamard_rotation_method="tcgen05", **kw)
    tc = call(x, apply_hadamard_rotation=True, hadamard_rotation_method="tensor_core", **kw)
    ekw = {k: v for k, v in kw.items() if k not in ("rope_cos", "rope_sin", "padded_head_dim")}
    xr = x if rope is None else R.apply_rope(x, kw["rope_cos"], kw["rope_sin"]).to(dtype)
    if din < 128:
        xr = torch.cat([xr, torch.zeros(b, s, h, 128 - din, dtype=dtype, device="cuda")], -1)
    emu = call(_rotate(xr), **ekw)
    if ekw.get("e4m3_copy_of_unrotated_values"):
        emu["e4m3_copy"] = tc["e4m3_copy"]
    _assert_bitwise(got, emu, tag + " vs emulation")
    for k, kind in (("codes", "nib"), ("transposed_codes", "nib"), ("e4m3_copy", "fp8")):
        if k in got:
            assert_codes_close(got[k], tc[k], kind, f"{tag} {k} vs tensor_core")


# ----------------------------------------------------------------------------------------------- refusals, resolution
def test_tcgen05_refusals_name_tensor_core():
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks, quantize_fp8_per_tensor
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    from flash_attn.cute.quant.fp4_diagonal import quantize_fp4_causal_diagonal
    x = make((1, 256, 2, 128), "n01", torch.bfloat16)
    kw = dict(apply_hadamard_rotation=True, hadamard_rotation_method="tcgen05")
    calls = [
        lambda: quantize_fp8_blocks(x.float(), **kw),                                   # fp32 input
        lambda: quantize_fp8_blocks(x.float(), block_rows=32, block_columns=1, output_exponent_table=False, **kw),
        lambda: quantize_fp4_rows(x.float(), **kw),
        lambda: quantize_fp8_per_tensor([x, x], **kw),                                  # the per-tensor kernel
        lambda: quantize_fp8_per_tensor([x, x], output_codes=False, **kw),
        lambda: quantize_fp4_causal_diagonal(x, x, 0.088, key_channel="off", **kw),      # the causal diagonal
        lambda: quantize_fp8_blocks(misaligned, **kw),                                   # not TMA-tileable
    ]
    flat = torch.zeros(256 * 2 * 128 + 8, dtype=torch.bfloat16, device="cuda")
    misaligned = flat[1:1 + 256 * 2 * 128].view(1, 256, 2, 128)
    for c in calls:
        with pytest.raises(ValueError, match="tensor_core"):
            c()
    # nothing rotated: the name is only validated
    quantize_fp8_per_tensor([x], hadamard_rotation_method="tcgen05")
    quantize_fp8_blocks(x, hadamard_rotation_method="tcgen05")
    # the default stays tensor_core
    quantize_fp8_blocks(x, apply_hadamard_rotation=True)
    assert ROT.ROTATION_METHOD_LOG["quantize_fp8_blocks"] == ("tensor_core", "default")
    assert ROT.DEFAULT_HADAMARD_ROTATION_METHOD == "tensor_core"


def test_tcgen05_graph_capture_one_launch_no_sync():
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    x = make((2, 512, 3, 128), "n01", torch.bfloat16)
    kw = dict(apply_hadamard_rotation=True, hadamard_rotation_method="tcgen05")
    fac = torch.empty(2, 3, 512 // 32, device="cuda")
    fac2 = torch.empty(2, 3, 512 // 32, device="cuda")
    calls = [lambda: quantize_fp8_blocks(x, output_token_block_factors=fac, output_amax=True, **kw),
             lambda: quantize_fp8_blocks(x, hadamard_rotation_scope="nvfp4_copy", output_token_block_factors=fac2,
                                         output_amax=True, **kw, **S4),
             lambda: quantize_fp8_blocks(x, make_channel_major_copy=True, **kw),
             lambda: quantize_fp4_rows(x, make_transposed_copy=True, **kw)]
    outs = [_clone(c()) for c in calls]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for c in calls:
            c()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    from torch.profiler import profile, ProfilerActivity
    for c in calls:   # one kernel launch per call
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            c()
            torch.cuda.synchronize()
        n = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
        assert n == 1, n
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = [c() for c in calls]
    g.replay()
    torch.cuda.synchronize()
    for o, r in zip(outs, res):
        _assert_bitwise(r, o, "graph replay")


# ------------------------------------------------------------------------------------------------- the public API
def test_api_tuning_tcgen05_runs_and_refuses():
    """KernelTuning(qk_hadamard_rotation_method="tcgen05"): the token-block routes run it (O / dK / dV equal the default
    method's on these inputs; the code-level difference is ~1e-7), the per-tensor operands and a call without the
    rotation are refused by the plan (E12), naming tensor_core."""
    from flash_attn.cute.lowprec import flash_attn_fp4, flash_attn_fp8, FP4AttnOptions, FP8AttnOptions
    from flash_attn.cute.lowprec.tuning import KernelTuning
    t5 = KernelTuning(qk_hadamard_rotation_method="tcgen05")
    g = torch.Generator().manual_seed(0)
    q, k, v, do = (torch.randn(1, 512, 2, 128, generator=g).bfloat16().cuda() for _ in range(4))
    for fn, Opt in ((flash_attn_fp4, FP4AttnOptions), (flash_attn_fp8, FP8AttnOptions)):
        outs = []
        for opt in (None, Opt(tuning=t5)):
            qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
            o = fn(qr, kr, vr, causal=True, qk_hadamard=True, deterministic=True, options=opt)
            o.backward(do)
            outs.append((o.detach(), kr.grad, vr.grad))
        for a, b in zip(*outs):
            assert torch.isfinite(a).all()
            assert (a.float() - b.float()).abs().max() <= 1e-2 * b.float().abs().max()
        with pytest.raises(ValueError, match="tensor_core"):
            fn(q, k, v, causal=True, qk_hadamard=False, options=Opt(tuning=t5))
        with pytest.raises(ValueError, match="tensor_core"):
            fn(q, k, v, causal=True, qk_hadamard=True, options=Opt(tuning=t5, operand_scaling="per_tensor"))

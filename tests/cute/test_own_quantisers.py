"""Our FP4 quantisers are bitwise FlashInfer's, and the library never needs FlashInfer.

1. Bitwise equality (codes and SF bytes) of the library's block-scaled FP4 quantiser (quant.fp4) against
   flashinfer.quantization nvfp4_quantize (global scale 1, layout_128x4, no shuffle) and mxfp4_quantize (layout_128x4),
   over N(0,1), logit-3, outlier and edge inputs (zeros, tiny, huge, mixed), several shapes, bf16 and fp16. Skipped when
   flashinfer is not importable: it is only the reference here.
2. The library imports and quantises with flashinfer hidden (a sys.modules block in a fresh process).

    pytest tests/cute/test_own_quantisers.py
"""
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DISTS = ("n01", "logit3", "outlier", "zeros", "tiny", "denorm", "near_min_normal", "huge", "mixed", "subnorm_sf")


def make(shape, dist, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)
    fmax = torch.finfo(dtype).max
    if dist == "logit3":
        x = x * 3.0
    elif dist == "outlier":  # ~0.1 % of the entries x 1000, one per some groups
        m = torch.rand(*shape, generator=g, device="cuda") < 1e-3
        x = torch.where(m, x * 1000.0, x)
    elif dist == "zeros":  # all-zero groups and rows mixed with normal ones
        x = x * (torch.rand(*shape[:-1], 1, generator=g, device="cuda") < 0.5)
        x[..., :32] = 0.0
    elif dist == "tiny":
        x = x * (1e-6 if dtype == torch.float16 else 1e-30)
    elif dist == "denorm":  # subnormal inputs of the storage type
        x = x * (1e-7 if dtype == torch.float16 else 1e-39)
    elif dist == "near_min_normal":  # around the smallest normal: amax / 6 and amax / 448 underflow, some x subnormal
        x = x * (1e-4 if dtype == torch.float16 else 1e-37)
    elif dist == "huge":
        x = (x.clamp(-4, 4) / 4.0) * (fmax * 0.99)
    elif dist == "mixed":  # every magnitude across the groups of a row (the scale's whole range)
        e = torch.randint(-24 if dtype == torch.float16 else -60, 15 if dtype == torch.float16 else 60,
                          (*shape[:-1], shape[-1] // 16), generator=g, device="cuda").float()
        x = x * torch.exp2(e).repeat_interleave(16, dim=-1)
    elif dist == "subnorm_sf":  # group maxima around the E4M3 subnormal and underflow range of amax / 6 (SF byte 0, 1, 2 ...)
        x = x * 1e-3 * (2.0 ** -torch.randint(0, 14, (*shape[:-1], 1), generator=g, device="cuda").float())
    x = x.clamp(-fmax, fmax).to(dtype)
    assert torch.isfinite(x).all()
    return x


def fi_2d(t, sf_vec):
    """FlashInfer's quantiser on a 2-D (M, K) matrix -> (codes uint8 (M, K/2), SF bytes flat)."""
    from flashinfer.quantization import SfLayout, mxfp4_quantize, nvfp4_quantize
    if sf_vec == 16:
        one = torch.ones(1, device=t.device, dtype=torch.float32)
        c, sf = nvfp4_quantize(t, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    else:
        c, sf = mxfp4_quantize(t, sfLayout=SfLayout.layout_128x4)
    return c.view(torch.uint8).reshape(t.shape[0], t.shape[1] // 2), sf.view(torch.uint8).reshape(-1)


def fi_rowwise(x, sf_vec):
    """The library's former FlashInfer-backed row-wise quantiser (interface_bwd_fp4.quantize_nvfp4_rowwise /
    mxfp4_quant.quantize_mxfp4_rowwise, since removed), verbatim apart from the call."""
    b, s, h, d = x.shape
    fp4, sf = fi_2d(x.reshape(b * s, h * d).contiguous(), sf_vec)
    fp4 = fp4.reshape(b, s, h, d // 2)
    rest_m, rest_k = s // 128, d // (sf_vec * 4)
    sf = sf.reshape(b * rest_m, h * rest_k, 32, 4, 4).reshape(b, rest_m, h, rest_k, 32, 4, 4)
    return fp4, sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)


def fi_kmajor_v(v):
    """The former mxfp4_quant.quantize_mxfp4_kmajor_v (FlashInfer-backed)."""
    b, s, h, d = v.shape
    fp4, sf = fi_2d(v.permute(0, 2, 3, 1).contiguous().reshape(b * h * d, s), 32)
    v4 = fp4.reshape(b, h, d, s // 2)
    sf = sf.reshape(b * h * (d // 128), s // 128, 32, 4, 4).reshape(b, h, d // 128, s // 128, 32, 4, 4)
    return v4, sf.contiguous().permute(4, 5, 2, 6, 3, 1, 0)


_TO7 = (4, 5, 2, 6, 3, 1, 0)
_FROM7 = (6, 5, 2, 4, 0, 1, 3)


def _width(d, sf_vec):
    return 128 if d % 128 == 0 else 64


def ours_rows(x, sf_vec):
    """quant.fp4's row copy for any d (multiple of 128, or of 64 for NVFP4): a row of D = r * w is r consecutive w-wide
    heads (a view), and the SF atoms are reordered to the (b, h, s / 128, D / (4 G)) layout (the old fp4_rowwise_quant)."""
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    b, s, h, d = x.shape
    w = _width(d, sf_vec)
    r_ = d // w
    out = quantize_fp4_rows(x.contiguous().view(b, s, h * r_, w), scale_group_size=sf_vec)
    c, sf_c = out["codes"].view(b, s, h, d // 2), out["scale_factors"].permute(*_FROM7)
    kk = w // (4 * sf_vec)
    sf_c = sf_c.reshape(b, h, r_, s // 128, kk, 32, 4, 4).transpose(2, 3).reshape(b, h, s // 128, r_ * kk, 32, 4, 4)
    return c, sf_c.permute(*_TO7)


def ours_2d(t, sf_vec):
    """quant.fp4's row copy of a 2-D (M, K) matrix in flashinfer's layout_128x4 order (atoms (M / 128, K / (4 G)))."""
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    m, k = t.shape
    w = _width(k, sf_vec)
    out = quantize_fp4_rows(t.contiguous().view(1, m, k // w, w), scale_group_size=sf_vec)
    sf_c = out["scale_factors"].permute(*_FROM7)            # (1, K / w, M / 128, w / (4 G), 32, 4, 4)
    return out["codes"].view(m, k // 2), sf_c[0].permute(1, 0, 2, 3, 4, 5).reshape(-1)


def ours_transposed(x, key_order):
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    r = quantize_fp4_rows(x, make_row_copy=False, make_transposed_copy=True, transposed_copy_in_kernel_key_order=key_order)
    return r["transposed_codes"], r["transposed_scale_factors"]


def ours_kmajor_v(v):
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    r = quantize_fp4_rows(v, scale_group_size=32, make_row_copy=False, make_transposed_copy=True,
                          transposed_copy_in_kernel_key_order=False, transposed_copy_layout="batch_head_dim_seq")
    return r["transposed_codes"], r["transposed_scale_factors"]


def ours_mxfp8_2d(t2d):
    """quant.fp8's MXFP8 (E4M3 + UE8M0 per 32 along K) of a 2-D (M, K) matrix in flashinfer's layout_128x4 SF order."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    m, k = t2d.shape
    w = 128 if k % 128 == 0 else 64
    r = quantize_fp8_blocks(t2d.contiguous().view(1, m, k // w, w), block_rows=1, block_columns=32,
                            output_exponent_table=False)
    return r["codes"].view(m, k), r["sf_tok_raw"][0].transpose(0, 1).reshape(-1)


def same(a, b, what):
    assert a.shape == b.shape, (what, a.shape, b.shape)
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    n = int((a != b).sum())
    assert n == 0, f"{what}: {n} of {a.numel()} bytes differ"


ROW_SHAPES = [(1, 128, 1, 128), (2, 256, 3, 128), (1, 384, 2, 256), (2, 1024, 16, 128), (2, 256, 3, 64), (1, 128, 2, 192)]
D2_SHAPES = [(128, 128), (256, 384), (512, 1024), (2 * 4 * 128, 640), (256, 64), (128, 320)]


@pytest.mark.parametrize("sf_vec", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("shape", ROW_SHAPES)
def test_rowwise_bitwise_flashinfer(shape, dist, dtype, sf_vec):
    pytest.importorskip("flashinfer")
    if shape[3] % 128 and sf_vec == 32:
        pytest.skip("MXFP4 rows are multiples of 128 (flashinfer's too)")
    x = make(shape, dist, dtype)
    c, sf = ours_rows(x, sf_vec)
    rc, rsf = fi_rowwise(x, sf_vec)
    same(c, rc, "codes")
    same(sf, rsf, "sf")


@pytest.mark.parametrize("sf_vec", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_2d_bitwise_flashinfer(shape, dist, dtype, sf_vec):
    pytest.importorskip("flashinfer")
    if shape[1] % 128 and sf_vec == 32:
        pytest.skip("MXFP4 rows are multiples of 128 (flashinfer's too)")
    t = make(shape, dist, dtype)
    c, sf = ours_2d(t, sf_vec)
    rc, rsf = fi_2d(t, sf_vec)
    same(c, rc, "codes")
    same(sf, rsf, "sf")


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("shape", [(1, 256, 2, 128), (2, 512, 3, 128)])
def test_library_entry_points_bitwise_flashinfer(shape, dist):
    """The quant.fp4 calls that replaced the library's former FlashInfer-backed entry points (row-wise NVFP4 / MXFP4, the
    K-major MXFP4 V, the transposed C3 copy) against those FlashInfer bodies."""
    pytest.importorskip("flashinfer")
    x = make(shape, dist, torch.bfloat16)
    for ours, ref in ((ours_rows(x, 16), fi_rowwise(x, 16)), (ours_rows(x, 32), fi_rowwise(x, 32)),
                      (ours_kmajor_v(x), fi_kmajor_v(x))):
        same(ours[0], ref[0], "codes")
        same(ours[1], ref[1], "sf")
    for perm in (True, False):
        xp = x.reshape(shape[0], shape[1] // 128, 4, 32, shape[2], shape[3])[:, :, [0, 2, 1, 3]].reshape(shape) if perm else x
        ref = fi_rowwise(xp.permute(0, 3, 2, 1).contiguous(), 16)
        ours = ours_transposed(x, perm)
        same(ours[0], ref[0], "codes_t")
        same(ours[1], ref[1], "sf_t")


@pytest.mark.parametrize("dist", DISTS)
def test_fused_quantisers_bitwise_flashinfer(dist):
    """quant.fp4's one-read quantiser the default routes call, row-wise and transposed copies."""
    pytest.importorskip("flashinfer")
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    shape = (2, 512, 3, 128)
    x = make(shape, dist, torch.bfloat16)
    xp = x.reshape(2, 4, 4, 32, 3, 128)[:, :, [0, 2, 1, 3]].reshape(shape)
    rc, rsf = fi_rowwise(x, 16)
    rct, rsft = fi_rowwise(xp.permute(0, 3, 2, 1).contiguous(), 16)
    r = quantize_fp4_rows(x, make_transposed_copy=True)
    c, sf, ct, sft = r["codes"], r["scale_factors"], r["transposed_codes"], r["transposed_scale_factors"]
    same(c, rc, "codes"), same(sf, rsf, "sf"), same(ct, rct, "codes_t"), same(sft, rsft, "sf_t")
    r = quantize_fp4_rows(x, scale_group_size=32)
    c, sf = r["codes"], r["scale_factors"]
    rc, rsf = fi_rowwise(x, 32)
    same(c, rc, "mx codes"), same(sf, rsf, "mx sf")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_mxfp8_2d_bitwise_flashinfer(shape, dist, dtype):
    """The MXFP8 (E4M3 + UE8M0 per 32) benchmark operands: quant.fp8 (block_rows 1, block_columns 32, flashinfer's 2-D SF order) vs flashinfer's mxfp8_quantize."""
    pytest.importorskip("flashinfer")
    from flashinfer.quantization import SfLayout, mxfp8_quantize
    if shape[1] % 128:
        pytest.skip("MXFP8 rows are multiples of 128 here")
    t = make(shape, dist, dtype)
    c, sf = ours_mxfp8_2d(t)
    rc, rsf = mxfp8_quantize(t, sf_swizzle_layout=SfLayout.layout_128x4)
    same(c, rc.view(torch.uint8).reshape(c.shape).view(c.dtype), "codes")
    same(sf, rsf.view(torch.uint8).reshape(-1), "sf")


_HIDE = r"""
import sys, importlib.abc
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ModuleNotFoundError("flashinfer hidden by test_own_quantisers")
sys.meta_path.insert(0, _Block())
for m in [m for m in sys.modules if m == "flashinfer" or m.startswith("flashinfer.")]:
    del sys.modules[m]
import torch
from flash_attn.cute import lowprec
from flash_attn.cute.lowprec import _routes
from flash_attn.cute import interface_bwd_fp4, interface_bwd_2cta8, interface_bwd_fp4q, interface_fwd_fp4, fp4_bwd_autograd
from flash_attn.cute.quant.fp4 import quantize_fp4_rows
x = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
for g in (16, 32):
    quantize_fp4_rows(x, scale_group_size=g, make_transposed_copy=True)
assert "flashinfer" not in sys.modules
print("NO_FLASHINFER_OK")
"""


def test_library_without_flashinfer():
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", _HIDE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=280)
    assert "NO_FLASHINFER_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_flashinfer_quant_knob_refused():
    """The FlashInfer quantiser choice is gone (one quantiser, quant.fp4): KernelTuning has no field
    that selects it."""
    from flash_attn.cute.lowprec import KernelTuning
    with pytest.raises((TypeError, ValueError)):
        KernelTuning(bwd_fp4_quant="flashinfer")

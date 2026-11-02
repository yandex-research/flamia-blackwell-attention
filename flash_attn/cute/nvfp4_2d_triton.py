# NOTE: the quantiser of this module (quantize_nvfp4_2d) is replaced by flash_attn.cute.quant.fp4
# (quantize_fp4_blocks_2d / quantize_fp4_rows). What remains is the head-dim Hadamard pass (fwht_rows, hadamard_signs) of
# the rotation fallbacks, and the Triton element helpers whose arithmetic the CuTe quantisers reproduce bitwise.
#
# 16x16-block NVFP4 quantiser (Triton) for the all-FP4 attention step: one E4M3 scale per 16x16 block of (rows, head dim),
# valid along the rows and along the head dim, so one set of E2M1 codes serves the GEMMs that contract either dim
# (the Megatron-Core NVFP4 2D scaling applied to attention operands).
#
# One read of a (b, s, h, d) tensor (d = 128, s % 128 == 0; one program per (b, h, 128-row tile)) writes
#   - the row-wise copy (groups of 16 along d): codes (b, s, h, d/2) + SF (32, 4, s/128, 4, d/64, h, b), the layout of
#     the row-wise NVFP4 copies (S, dP operands; now quant.fp4.quantize_fp4_rows);
#   - the transposed copy (groups of 16 along the rows): codes (b, d, h, s/2) + SF (32, 4, 1, 4, s/64, h, b), rows in the
#     C3 32-block order [0, 2, 1, 3] (perm=True: dV / dK B operands) or natural (perm=False: forward V^T, dQ K^T);
#   - optionally D (b, h, s) = sum_d o * x^ (the dequantised row-wise values; o rotated alike when rot is given).
# Both copies carry the SAME codes and the same block scale (replicated into the 1x16 layouts).
#
# Rounding (rnd): "rtn" (SF = E4M3(amax / 6), E2M1 round to nearest, ties to the even code as the cvt); "sr" (SF =
# E4M3(amax / (6 * 16 / 17)), stochastic rounding onto the E2M1 grid: unbiased, never clips); "eden" (MS-EDEN, Quartet II:
# RTN at SF = E4M3(amax / (6 * 16 / 17 / 0.93)) clipping at 6, then the block's EDEN factor <x, x> / <x, x^> folded into the
# E4M3 scale by stochastic rounding).
# Outer scale: the quantised tensor is x * inv_outer (per-tensor FP32 scale kept by the caller); dequantised = code SF outer.
# Rotation (signs: (d,) +-1 tensor): x -> (x * signs) H / sqrt(d) along the head dim before quantising; the same rotation of
# Q, K, V, dO cancels in S, dP and D.
import torch
import triton
import triton.language as tl



@triton.jit
def _pack_e2m1_pairs(lo, hi):
    """lo, hi fp32 -> one byte (lo in the low nibble) as int32."""
    # Triton passes fp32 operands in .b32 registers: move them into .f32 registers before the cvt; the cvt
    # writes an 8-bit register (as in blackwell_helpers.packed_float_to_e2m1); only the low byte is used
    return tl.inline_asm_elementwise(
        "{ .reg .f32 a, b; .reg .b8 t; mov.b32 a, $1; mov.b32 b, $2; "
        "cvt.rn.satfinite.e2m1x2.f32 t, b, a; mov.b32 $0, {t, t, t, t}; }",
        "=r,r,r", [lo, hi], dtype=tl.int32, is_pure=True, pack=1,
    )


@triton.jit
def _e2m1_code_val(c):
    """E2M1 code (sign bit 3, magnitude codes 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6) -> fp32 value."""
    m = c & 7
    mf = m.to(tl.float32)
    v = tl.where(m < 4, mf * 0.5, tl.where(m < 6, mf - 2.0, tl.where(m == 6, 4.0, 6.0)))
    return tl.where((c & 8) != 0, -v, v)



RND_ID = {"rtn": 0, "sr": 1, "eden": 2, "unit": 3}  # unit: SF = 1 exactly, RTN saturating at 6 (the vtok v~ = V / s_k)
SR_SMAX = 6.0 * 16.0 / 17.0            # SR: the grid max never clips an E4M3-rounded-down scale
EDEN_SMAX = 6.0 * 16.0 / 17.0 / 0.93   # MS-EDEN clipping RTN


@triton.jit
def _hash_u01(idx, seed):
    """uniform [0, 1) from an int64 index and seed (splitmix64 finaliser, logical shifts emulated by masks)."""
    z = idx * -7046029254386353131 + seed  # 0x9E3779B97F4A7C15
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * -4658895280553007687  # 0xBF58476D1CE4E5B9
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * -7723592293110705685  # 0x94D049BB133111EB
    z = z ^ ((z >> 31) & 0x1FFFFFFFF)
    return ((z >> 40) & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _e2m1_sr(y, u):
    """stochastic rounding of y onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (|y| clamped to 6) with uniform u."""
    a = tl.minimum(tl.abs(y), 6.0)
    lo = tl.where(a < 2.0, tl.floor(a * 2.0) * 0.5, tl.where(a < 4.0, tl.floor(a), 4.0))
    step = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    v = tl.where(u * step < a - lo, lo + step, lo)
    return tl.where(y < 0, -v, v)


@triton.jit
def _e4m3_sr(x, u):
    """stochastic rounding of x >= 0 onto the E4M3 grid (normal steps 2^(e-3) from 2^-6, subnormal step 2^-9, max 448)."""
    xc = tl.minimum(tl.maximum(x, 0.0), 448.0)
    e = tl.maximum(tl.floor(tl.log2(tl.maximum(xc, 0.015625))), -6.0)
    step = tl.exp(((e - 3.0) * 0.6931471805599453))
    lo = tl.floor(xc / step) * step
    return tl.minimum(tl.where(u * step < xc - lo, lo + step, lo), 448.0)


@triton.jit
def _wht_level(y, R: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    """one Walsh-Hadamard level: pairs (i, i + H) within blocks of 2H of the last dim -> (a + b, a - b)."""
    y4 = tl.permute(tl.reshape(y, [R, D // (2 * H), 2, H]), [0, 1, 3, 2])  # [R, D/2H, H, 2]
    a, b = tl.split(y4)
    return tl.reshape(tl.permute(tl.join(a + b, a - b), [0, 1, 3, 2]), [R, D])


@triton.jit
def _fwht(x, R: tl.constexpr, D: tl.constexpr):
    """orthonormal Walsh-Hadamard transform of the last dim (D = 128) of an [R, D] block."""
    y = _wht_level(x, R, D, 1)
    y = _wht_level(y, R, D, 2)
    y = _wht_level(y, R, D, 4)
    y = _wht_level(y, R, D, 8)
    y = _wht_level(y, R, D, 16)
    y = _wht_level(y, R, D, 32)
    y = _wht_level(y, R, D, 64)
    return y * 0.08838834764831845  # 1 / sqrt(128)


@triton.jit
def _perm_blocks(y, N: tl.constexpr, W: tl.constexpr):
    """rows of an [N, W] block in 4 equal row-blocks reordered [0, 2, 1, 3] (involution)."""
    return tl.reshape(tl.permute(tl.reshape(y, [2, 2, N // 4, W]), [1, 0, 2, 3]), [N, W])


@triton.jit
def _q_groups(xg, idx, seed, RND: tl.constexpr, A: tl.constexpr, B: tl.constexpr, N: tl.constexpr):
    """NVFP4 of groups xg [A, B, N] (N even): returns (pre-pack values y [A, B, N] (grid values for SR), packed codes
    [A, B, N / 2] int32 (low byte), E4M3 scales sf8 [A, B]). idx [A, B] int64: per-group random stream base."""
    amax = tl.max(tl.abs(xg), axis=2)
    if RND == 3:
        sf8 = (amax * 0.0 + 1.0).to(tl.float8e4nv)  # data-independent scale: nothing crosses group members
    elif RND == 0:
        sf8 = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
    elif RND == 1:
        sf8 = (amax * (17.0 / 96.0)).to(tl.float8e4nv)
    else:
        sf8 = (amax * (0.93 * 17.0 / 96.0)).to(tl.float8e4nv)
    sfr = sf8.to(tl.float32)
    inv = tl.where(sfr > 0.0, 1.0 / sfr, 0.0)
    y = xg * inv[:, :, None]
    if RND == 1:
        y = _e2m1_sr(y, _hash_u01(idx[:, :, None] * N + tl.arange(0, N)[None, None, :].to(tl.int64), seed))
    lo, hi = tl.split(tl.reshape(y, [A, B, N // 2, 2]))
    packed = _pack_e2m1_pairs(lo, hi)
    if RND == 2:
        cval = tl.reshape(tl.join(_e2m1_code_val(packed & 15), _e2m1_code_val((packed >> 4) & 15)), [A, B, N])
        num = tl.sum(xg * xg, axis=2)
        den = tl.sum(xg * cval, axis=2) * sfr
        fac = tl.where(den > 0.0, num / tl.where(den > 0.0, den, 1.0), 1.0)
        sf8 = _e4m3_sr(sfr * fac, _hash_u01(idx + (1 << 44), seed)).to(tl.float8e4nv)
    return y, packed, sf8


@triton.jit
def _fwht_tile(x_ptr, y_ptr, sgn_ptr, s, h, sxb, sxs, sxh, INV: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    tiles = s // 128
    bh = pid // tiles
    t = pid % tiles
    b = bh // h
    hh = bh % h
    rows = t * 128 + tl.arange(0, 128)
    cols = tl.arange(0, D)
    x = tl.load(x_ptr + b * sxb + hh * sxh + rows[:, None] * sxs + cols[None, :]).to(tl.float32)
    sg = tl.load(sgn_ptr + cols).to(tl.float32)
    if INV:
        y = _fwht(x, 128, D) * sg[None, :]
    else:
        y = _fwht(x * sg[None, :], 128, D)
    tl.store(y_ptr + ((b * s + rows[:, None]) * h + hh) * D + cols[None, :], y)


def fwht_rows(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False, dtype=None) -> torch.Tensor:
    """randomised Hadamard along the head dim of x (b, s, h, 128), s % 128 == 0: x -> (x * signs) H / sqrt(d) (inverse:
    x H / sqrt(d) * signs). Returns a contiguous (b, s, h, d) tensor in dtype (default x.dtype); computed in float32."""
    b, s, h, d = x.shape
    assert d == 128 and s % 128 == 0 and x.stride(3) == 1, (x.shape, x.stride())
    y = torch.empty(b, s, h, d, dtype=torch.float32, device=x.device)
    sg = signs.to(device=x.device, dtype=torch.float32).contiguous()
    _fwht_tile[(b * h * (s // 128),)](x, y, sg, s, h, x.stride(0), x.stride(1), x.stride(2), INV=inverse, D=d, num_warps=4)
    dt = dtype or x.dtype
    return y if dt == torch.float32 else y.to(dt)


# ------------------------------------------------------------------ host helpers (rotation, reference, dequantisers)
def hadamard_signs(d: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    return (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(device=device, dtype=torch.float32)


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    y = x.float().clone()
    hlen = 1
    while hlen < d:
        y = y.view(*y.shape[:-1], d // (2 * hlen), 2, hlen)
        a, bb = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + bb, a - bb), -2).reshape(*x.shape)
        hlen *= 2
    return y * (1.0 / d ** 0.5)


def rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return fwht_torch(x.float() * signs)


def unrotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return fwht_torch(x.float()) * signs

# Eager-PyTorch references of the quantisers in flash_attn.cute.quant, written from the format definitions, never by calling
# a kernel. The FP4 part is this file's fp4_* functions; the FP8 part the fp8_* functions.
#
# FP4 definitions (also in fp4.py's header):
#   E2M1     the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6} with a sign bit; code = magnitude index (0..7) | 8 * sign; round to
#            nearest, ties to the even code (cvt.rn.satfinite.e2m1x2.f32), |v| > 6 saturates to 6, NaN -> +6 (code 7).
#            Two codes per byte, element 0 in the low nibble.
#   E4M3     torch.float8_e4m3fn (round to nearest even) after clamping to +-448 (satfinite).
#   NVFP4    group of 16: amax over the group of x with subnormal values flushed to signed zero; SF = E4M3(amax * f32(c))
#            with c = 1/6 (nearest), 17/96 (stochastic), 0.93 * 17/96 (ms_eden); "unit": SF = 1. Codes of x * m with
#            m = 1/SF (SF > 0), +inf (SF rounds to 0 while amax * c > 0: nonzero -> +-6, zero -> NaN -> +6), 0 (all-zero
#            group: codes keep the sign of zero).
#   MXFP4    group of 32: SF = 2^e, the smallest power of two with 6 * 2^e >= amax; byte e + 127; byte 0 (multiplier 0)
#            when amax * f32(1/6) is below the smallest normal fp32 (subnormal or zero).
#   2D       one SF per 16 x 16 (NVFP4) / 32 x 32 (MXFP4) block of (tokens, d); the amax over the block.
#   stochastic rounding  y -> one of its two E2M1 neighbours lo, lo + step (|y| clamped to 6): lo + step iff u * step <
#            |y| - lo, u = splitmix64(element index * 0x9E3779B97F4A7C15 + seed) >> 40 as a 24-bit fraction. Element
#            indices: tile base B = ((b * H + h) * S + q0) * 4096 (q0 the tile's first token); row copy group (r, g):
#            B + r * (128 / G) + g, element (group) * G + j; transposed copy group (d, g): B + 2^40 + d * (128 / G) + g;
#            2D block (rb, g): B + rb * (128 / G) + g, element (block) * G^2 + (r % G) * G + (c % G).
#   SF layout  the SM100 128 x 4 atom: (row, K group) -> byte ((row // 128) * K_atoms + group // 4) * 512 + (row % 32) * 16
#            + ((row // 32) % 4) * 4 + group % 4, viewed (32, 4, rows / 128, 4, K_atoms, h, b).
#   Hadamard  (x * signs) through butterfly levels H = 1, 2, ..., 64 ((a, b) -> (a + b, a - b)), then * f32(1 / sqrt(128)).
#   RoPE      (x_j, x_{j+64}) -> (fma(x_j, c_j, -(x_{j+64} s_j)), fma(x_{j+64}, c_j, x_j s_j)) in fp32 (single rounding of
#            the fma; with (s, 64) tables the high half is fma(x_j, s_j, x_{j+64} c_j)), then rounded to the input dtype.
#   Sums     (D, norms, Four-over-Six errors, MS-EDEN): fp32 xor-butterfly trees in the orders of fp4.py (*_SUM_ORDER);
#            the row norm: per 64-column half a sequential fma chain, then lo + hi.
import math

import numpy as np
from typing import Optional

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
INV_SQRT_128_F32 = torch.tensor(1.0 / math.sqrt(128.0), dtype=torch.float32).item()
F32_MIN_NORMAL = 1.1754943508222875e-38


def f32(v: float) -> float:
    return torch.tensor(v, dtype=torch.float32).item()


# ---------------------------------------------------------------------------------------------------------------- scalars
def nanless_amax(x: torch.Tensor) -> torch.Tensor:
    """max |x| over the last dim ignoring NaN (fmax semantics: a NaN never wins)."""
    a = x.abs()
    return torch.where(torch.isnan(a), torch.zeros_like(a), a).amax(-1)


def ftz(x: torch.Tensor) -> torch.Tensor:
    """fp32 subnormals -> zero of the same sign."""
    return torch.where(x.abs() < F32_MIN_NORMAL, x * 0.0, x)


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E4M3 (satfinite, round to nearest even) -> (byte uint8, value fp32)."""
    q = x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), q.float()


def e2m1_encode_rn(y: torch.Tensor) -> torch.Tensor:
    """fp32 -> E2M1 code (uint8 0..15): round to nearest, ties to the even code, saturating; NaN -> 7 (+6)."""
    grid = E2M1_GRID.to(y.device)
    a = y.abs().clamp(max=6.0)
    nan = torch.isnan(y)
    a = torch.where(nan, torch.full_like(a, 6.0), a)
    d = (a[..., None] - grid).abs()
    best = d.min(-1, keepdim=True).values
    cand = (d == best)
    # ties: prefer the even code index (cand at two neighbours, one even)
    idx = torch.arange(8, device=y.device)
    score = cand.to(torch.int64) * (2 + (idx % 2 == 0).to(torch.int64))
    mag = score.argmax(-1)
    neg = torch.signbit(y) & ~nan
    return (mag + 8 * neg.to(torch.int64)).to(torch.uint8)


def e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    grid = E2M1_GRID.to(code.device)
    c = code.to(torch.int64)
    v = grid[c & 7]
    return torch.where((c & 8) != 0, -v, v)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """(..., 2n) codes -> (..., n) bytes, element 0 in the low nibble."""
    c = codes.to(torch.uint8)
    return c[..., 0::2] | (c[..., 1::2] << 4)


def splitmix_uniform(index: torch.Tensor, seed: int) -> torch.Tensor:
    """u in [0, 1): splitmix64 finaliser of index * golden + seed (int64 wrapping, logical shifts as masked shifts)."""
    z = index.to(torch.int64) * -7046029254386353131 + int(seed)
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * -4658895280553007687
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * -7723592293110705685
    z = z ^ ((z >> 31) & 0x1FFFFFFFF)
    return ((z >> 40) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)


def e2m1_stochastic(y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """the E2M1 grid value (fp32, signed) chosen by stochastic rounding of y with uniform u."""
    a = torch.nan_to_num(y.abs(), nan=6.0).clamp(max=6.0)
    lo = torch.where(a < 2.0, torch.floor(a * 2.0) * 0.5, torch.where(a < 4.0, torch.floor(a), torch.full_like(a, 4.0)))
    step = torch.where(a < 2.0, torch.full_like(a, 0.5), torch.where(a < 4.0, torch.ones_like(a), torch.full_like(a, 2.0)))
    v = torch.where(u * step < a - lo, lo + step, lo)
    return torch.where(y < 0, -v, v)


def fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """round_f32(a * b + c) via float64 (exact product; the float64 sum rounds only when exponents are > 29 apart, then
    double rounding needs an exact tie: not reached by the test inputs)."""
    return (a.double() * b.double() + c.double()).float()


def xor_tree_sum(v: torch.Tensor, order) -> torch.Tensor:
    """fp32 xor-butterfly sum over the last dim in `order` (the kernels' D_SUM_ORDER etc.)."""
    n = v.shape[-1]
    v = v.clone()
    done = 0
    for off in order:
        idx = torch.tensor([c for c in range(n) if (c & (done | off)) == 0], device=v.device)
        v[..., idx] = v[..., idx] + v[..., idx | off]
        done |= off
    return v[..., 0]


# ------------------------------------------------------------------------------------------------------------ transforms
def hadamard_rotate(x: torch.Tensor, signs) -> torch.Tensor:
    """(x * signs) H_128 / sqrt(128) along the last dim, fp32, the defined order of operations."""
    s = torch.tensor(signs, dtype=torch.float32, device=x.device)
    y = x.float() * s
    shape = y.shape
    h = 1
    while h < 128:
        y = y.view(*shape[:-1], 128 // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), -2).reshape(shape)
        h *= 2
    return y * INV_SQRT_128_F32


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (b, s, h, 128) -> RoPE in fp32 (the defined fma form), rounded to x.dtype, returned as fp32."""
    s = x.shape[1]
    half = x.shape[-1] // 2
    transposed = tuple(cos.shape) == (half, s) and tuple(cos.shape) != (s, half)
    if transposed:
        c, sn = cos.t(), sin.t()
    else:
        c, sn = cos, sin
    c = c[None, :, None, :].float()
    sn = sn[None, :, None, :].float()
    xf = x.float()
    half = xf.shape[-1] // 2
    x0, x1 = xf[..., :half], xf[..., half:]
    lo = fma_f32(x0, c, -(x1 * sn))
    hi = fma_f32(x1, c, x0 * sn) if transposed else fma_f32(x0, sn, x1 * c)
    return torch.cat([lo, hi], -1).to(x.dtype).float()


# ------------------------------------------------------------------------------------------------------------ SF layout
def sf_atoms_from_table(table: torch.Tensor) -> torch.Tensor:
    """table (b, h, R, K) bytes (row R, K group) -> the 7-D SF view (32, 4, R/128, 4, K_atoms, h, b) of contiguous atoms."""
    b, h, R, K = table.shape
    ka = K // 4
    t = table.reshape(b, h, R // 128, 4, 32, ka, 4)            # (b, h, Rt, j, i, ka, kk): row = Rt*128 + j*32 + i
    raw = t.permute(0, 1, 2, 5, 4, 3, 6).contiguous()          # (b, h, Rt, ka, i(32), j(4), kk(4))
    return raw.permute(4, 5, 2, 6, 3, 1, 0)


def table_from_sf_atoms(sf7: torch.Tensor) -> torch.Tensor:
    raw = sf7.permute(6, 5, 2, 4, 0, 1, 3)                     # (b, h, Rt, ka, i, j, kk)
    b, h, Rt, ka = raw.shape[:4]
    return raw.permute(0, 1, 2, 5, 4, 3, 6).reshape(b, h, Rt * 128, ka * 4)


# ------------------------------------------------------------------------------------------------------------ groups
def _nvfp4_scale(amax: torch.Tensor, rounding: str):
    if rounding == "unit":
        one = torch.ones_like(amax)
        return torch.full_like(amax, 0x38, dtype=torch.uint8), one, one
    c = {"nearest": f32(1.0 / 6.0), "stochastic": f32(17.0 / 96.0), "ms_eden": f32(0.93 * 17.0 / 96.0)}[rounding]
    u = amax * c
    byte, sfv = e4m3_round(u)
    mult = torch.where(sfv > 0, 1.0 / sfv, torch.where(u > 0, torch.full_like(u, float("inf")), torch.zeros_like(u)))
    return byte, sfv, mult


def _mxfp4_scale(amax: torch.Tensor):
    r = amax * f32(1.0 / 6.0)
    e = torch.ceil(torch.log2(amax.double() / 6.0)).clamp(-127, 127)
    # the smallest e with 6 * 2^e >= amax (exact comparisons in float64)
    e = torch.where(6.0 * torch.exp2(e) < amax.double(), e + 1, e)
    e = torch.where(6.0 * torch.exp2(e - 1) >= amax.double(), e - 1, e)
    ok = r >= F32_MIN_NORMAL
    byte = torch.where(ok, e + 127, torch.zeros_like(e)).to(torch.uint8)
    mult = torch.where(ok, torch.exp2(-e), torch.zeros_like(e)).float()
    return byte, mult


def _encode_groups(xg: torch.Tensor, G: int, rounding: str, amax: torch.Tensor, elem_index: Optional[torch.Tensor],
                   seed: int, four_over_six: bool, eden_index: Optional[torch.Tensor], transposed: bool = False):
    """xg (..., G) fp32 values of groups (FTZ applied here), amax (...,) -> (codes (..., G) uint8, SF byte (...,))."""
    xg = ftz(xg)
    if G == 32:
        byte, mult = _mxfp4_scale(amax)
        sfv = None
    else:
        byte, sfv, mult = _nvfp4_scale(amax, rounding)
    y = ftz(xg * mult[..., None]) if G == 16 else xg * mult[..., None]
    if rounding == "stochastic":
        yv = e2m1_stochastic(y, splitmix_uniform(elem_index, seed))
        codes = e2m1_encode_rn(yv)
    else:
        codes = e2m1_encode_rn(y)
    if four_over_six:
        from flash_attn.cute.quant.fp4 import FO6_ROW_SUM_ORDER, FO6_TRANSPOSED_SUM_ORDER
        FO6_SUM_ORDER = FO6_TRANSPOSED_SUM_ORDER if transposed else FO6_ROW_SUM_ORDER
        b4, sf4 = e4m3_round(amax * 0.25)
        inv4 = torch.where(sf4 > 0, 1.0 / sf4, torch.zeros_like(sf4))
        y4 = (xg * inv4[..., None]).clamp(-4.0, 4.0)
        a6 = e2m1_value_rn(y) * sfv[..., None] - xg
        a4 = e2m1_value_rn(y4) * sf4[..., None] - xg
        e6 = xor_tree_sum(a6 * a6, FO6_SUM_ORDER)
        e4 = xor_tree_sum(a4 * a4, FO6_SUM_ORDER)
        use4 = e4 < e6
        codes = torch.where(use4[..., None], e2m1_encode_rn(y4), codes)
        byte = torch.where(use4, b4, byte)
    if rounding == "ms_eden":
        from flash_attn.cute.quant.fp4 import EDEN_SUM_ORDER
        cval = e2m1_decode(codes)
        num = xor_tree_sum(xg * xg, EDEN_SUM_ORDER)
        den = xor_tree_sum(xg * cval, EDEN_SUM_ORDER) * sfv
        fac = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.ones_like(den))
        v = _e4m3_stochastic(sfv * fac, splitmix_uniform(eden_index + (1 << 44), seed))
        byte, _ = e4m3_round(v)
    return codes, byte


def e2m1_value_rn(y: torch.Tensor) -> torch.Tensor:
    """The value Four-over-Six's error uses: the E2M1 grid point with thresholds at the midpoints rounding UP
    (0.25 -> 0.5 ... the old Triton _e2m1_val), NaN -> 6."""
    a = torch.nan_to_num(y.abs(), nan=6.0).clamp(max=6.0)
    v = torch.where(a < 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a < 1.25, 1.0, torch.where(a < 1.75, 1.5,
        torch.where(a < 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a < 5.0, 4.0, 6.0)))))))
    return torch.where(y < 0, -v, v).float()


def _e4m3_stochastic(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    xc = x.clamp(0.0, 448.0)
    e = torch.floor(torch.log2(xc.clamp(min=0.015625))).clamp(min=-6.0)
    step = torch.exp2(e - 3.0)
    lo = torch.floor(xc / step) * step
    return torch.where(u * step < xc - lo, lo + step, lo).clamp(max=448.0)


# ------------------------------------------------------------------------------------------------------------ quantisers
def _prepare(x, rope_cos, rope_sin, apply_hadamard_rotation, hadamard_sign_seed):
    from flash_attn.cute.quant._rotations import hadamard_signs
    xf = x.float()
    if rope_cos is not None:
        xf = rope(x, rope_cos, rope_sin)
    plain = xf
    if apply_hadamard_rotation:
        xf = hadamard_rotate(xf, hadamard_signs(hadamard_sign_seed))
    return plain, xf


def fp4_rows_reference(x: torch.Tensor, *, scale_group_size: int = 16, make_row_copy: bool = True,
                       make_transposed_copy: bool = False, transposed_copy_in_kernel_key_order: bool = True,
                       transposed_copy_layout: str = "batch_dim_head_seq", use_four_over_six: bool = False,
                       global_scale=None, rounding: str = "nearest", stochastic_rounding_seed: int = 0,
                       per_32_row_power_of_two_factors=None, also_make_e4m3_per_tensor_copy: bool = False,
                       e4m3_copy_scale=None, e4m3_copy_of_unrotated_values: bool = False,
                       output_row_norm_max=None, output_tile_amax=None, apply_hadamard_rotation: bool = False,
                       hadamard_sign_seed: int = 4321, rope_cos=None, rope_sin=None, compute_d_with_output=None,
                       padded_head_dim=None, _blocks_2d: bool = False) -> dict:
    """The eager reference of fp4.quantize_fp4_rows (and of quantize_fp4_blocks_2d with _blocks_2d=True). Same arguments,
    same outputs; output_row_norm_max / output_tile_amax return "row_norm_max" / "tile_amax" tensors instead of reducing
    into the caller's slots."""
    from flash_attn.cute.quant.fp4 import D_SUM_ORDER
    if x.shape[-1] > 128:
        # head_dim 128 c: each 128-wide chunk quantised as a head; the SF table's K groups are the chunks' in order
        b0, s0, h0, d0 = x.shape
        c = d0 // 128
        r = fp4_rows_reference(x.reshape(b0, s0, h0 * c, 128), scale_group_size=scale_group_size,
                               make_row_copy=make_row_copy, rounding=rounding,
                               stochastic_rounding_seed=stochastic_rounding_seed,
                               also_make_e4m3_per_tensor_copy=also_make_e4m3_per_tensor_copy,
                               e4m3_copy_scale=e4m3_copy_scale, use_four_over_six=use_four_over_six,
                               global_scale=global_scale)
        out = {}
        if make_row_copy:
            out["codes"] = r["codes"].reshape(b0, s0, h0, d0 // 2)
            tbl = table_from_sf_atoms(r["scale_factors"])                    # (b, h c, s, ng)
            tbl = tbl.reshape(b0, h0, c, s0, -1).permute(0, 1, 3, 2, 4).reshape(b0, h0, s0, -1)
            out["scale_factors"] = sf_atoms_from_table(tbl.contiguous())
        if also_make_e4m3_per_tensor_copy:
            out["e4m3_copy"] = r["e4m3_copy"].reshape(b0, s0, h0, d0)
        return out
    b, s, h, d = x.shape
    G = scale_group_size
    dev = x.device
    seed = int(stochastic_rounding_seed) & 0x3FFFFFFFFFFFFFFF
    din = d
    d = din if padded_head_dim is None else int(padded_head_dim)       # the quantised width (zero padding)
    ng = d // G
    xin = x.float()
    if rope_cos is not None:
        xin = rope(x, rope_cos, rope_sin)
    xin = torch.nn.functional.pad(xin, (0, d - din))
    plain = xin
    xr = hadamard_rotate(xin, __import__("flash_attn.cute.quant._rotations", fromlist=["x"]).hadamard_signs(
        hadamard_sign_seed)) if apply_hadamard_rotation else xin
    out = {}
    if also_make_e4m3_per_tensor_copy:
        inv = 1.0 / e4m3_copy_scale.float().reshape(())   # float32 reciprocal (the kernel's rcp.rn)
        src = plain if (e4m3_copy_of_unrotated_values and apply_hadamard_rotation) else xr
        out["e4m3_copy"] = (src * inv).clamp(-448, 448).to(torch.float8_e4m3fn)
    if output_row_norm_max is not None:
        sq_lo = torch.zeros(b, s, h, device=dev)
        sq_hi = torch.zeros(b, s, h, device=dev)
        for j in range(min(64, d)):
            sq_lo = fma_f32(xr[..., j], xr[..., j], sq_lo)
        for j in range(64, d):
            sq_hi = fma_f32(xr[..., j], xr[..., j], sq_hi)
        out["row_norm_max"] = (sq_lo + sq_hi).max().reshape(1)
    if output_tile_amax is not None:
        out["tile_amax"] = nanless_amax(xr.permute(0, 2, 1, 3).reshape(b, h, s // 128, -1)).reshape(-1)
    xq = xr
    outer = 1.0
    if global_scale is not None:
        if isinstance(global_scale, torch.Tensor):
            od = global_scale.float().reshape(())
            inv_outer, outer = 1.0 / od, od   # float32 (rcp.rn): the FP64-free contract
        else:
            inv_outer, outer = f32(1.0 / float(global_scale)), f32(float(global_scale))
        xq = xq * inv_outer
    rf = None
    if per_32_row_power_of_two_factors is not None:
        rf = per_32_row_power_of_two_factors.float().repeat_interleave(32, -1).permute(0, 2, 1)[..., None]  # (b, s, h, 1)
        xq = xq * (1.0 / rf)
    tile = torch.arange(s, device=dev) // 128
    q0 = tile * 128
    r = torch.arange(s, device=dev) % 128
    bh = (torch.arange(b, device=dev)[:, None] * h + torch.arange(h, device=dev)[None, :])        # (b, h)
    base = ((bh[:, None, :].to(torch.int64) * s + q0[None, :, None]) * 4096)                    # (b, s, h)
    xg = xq.view(b, s, h, ng, G)
    amax = nanless_amax(ftz(xg))                                                                 # (b, s, h, ng)
    if _blocks_2d:
        a2 = amax.view(b, s // G, G, h, ng).amax(2, keepdim=True).expand(b, s // G, G, h, ng).reshape(b, s, h, ng)
        amax = a2
    gidx = None
    elem = None
    if rounding == "stochastic":
        j = torch.arange(G, device=dev)
        if _blocks_2d:
            gidx = base[..., None] + ((r // G)[None, :, None, None] * ng + torch.arange(ng, device=dev))
            elem = gidx[..., None] * (G * G) + ((r % G)[None, :, None, None, None] * G + j)
        else:
            gidx = base[..., None] + (r[None, :, None, None] * ng + torch.arange(ng, device=dev))
            elem = gidx[..., None] * G + j
    eden_idx = base[..., None] + (r[None, :, None, None] * ng + torch.arange(ng, device=dev)) if rounding == "ms_eden" else None
    codes, sfb = _encode_groups(xg, G, rounding, amax, elem, seed, use_four_over_six, eden_idx)
    codes = codes.view(b, s, h, d)
    if make_row_copy:
        out["codes"] = pack_nibbles(codes)
        out["scale_factors"] = sf_atoms_from_table(sfb.permute(0, 2, 1, 3))                    # table (b, h, s, ng)
    if compute_d_with_output is not None:
        sfv = e4m3_round(torch.zeros(1))[1]  # noqa: F841
        if G == 16:
            sfval = sfb.view(torch.float8_e4m3fn).float()
        else:
            sfval = torch.where(sfb > 0, torch.exp2(sfb.float() - 127.0), torch.zeros_like(sfb, dtype=torch.float32))
        deq = (e2m1_decode(codes).view(b, s, h, ng, G) * sfval[..., None]).view(b, s, h, d)
        prod = deq[..., :din] * compute_d_with_output.float()
        D = xor_tree_sum(torch.nn.functional.pad(prod, (0, 128 - din)), D_SUM_ORDER)
        if global_scale is not None:
            D = D * outer
        if rf is not None:
            D = D * rf[..., 0]
        out["d"] = D.permute(0, 2, 1).contiguous()
    if make_transposed_copy:
        perm = torch.arange(128, device=dev)
        if transposed_copy_in_kernel_key_order:
            blk = perm // 32
            perm = ((blk % 2) * 2 + blk // 2) * 32 + perm % 32                                 # output position -> source row
        src_rows = torch.arange(s // 128, device=dev)[:, None] * 128 + perm[None, :]
        if _blocks_2d:
            tc = codes[:, src_rows.reshape(-1)]                                                  # (b, s', h, 128 d)
            tcodes = tc.view(b, s // 128, 128, h, d).permute(0, 4, 3, 1, 2).reshape(b, d, h, s)       # (b, d, h, s)
            blk_sf = sfb[:, src_rows.reshape(-1)]                                               # (b, s', h, ng)
            tsf = blk_sf.view(b, s // G, G, h, ng)[:, :, 0]                                     # (b, s/G, h, ng)
            tsf = tsf.repeat_interleave(G, -1).permute(0, 2, 3, 1)                              # (b, h, d, s/G)
        else:
            xt = xq[:, src_rows.reshape(-1)]                                                    # (b, s', h, d)
            xt = xt.view(b, s // 128, 128, h, d).permute(0, 4, 3, 1, 2).reshape(b, d, h, s // G, G)      # (b, d, h, ng_s, G)
            amt = nanless_amax(ftz(xt))
            elem_t = None
            if rounding == "stochastic":
                dd = torch.arange(d, device=dev)
                gs = torch.arange(s // G, device=dev)
                tb = (gs * G) // 128                                                             # tile of the group
                gt = gs % (128 // G)
                bt = ((bh[:, None, :, None].to(torch.int64) * s + (tb * 128)[None, None, None, :]) * 4096)  # (b,1,h,ngs)
                gi = bt + (1 << 40) + (dd[None, :, None, None] * (128 // G) + gt[None, None, None, :])
                elem_t = gi[..., None] * G + torch.arange(G, device=dev)
            eden_t = None
            if rounding == "ms_eden":
                dd = torch.arange(d, device=dev)
                gs = torch.arange(s // G, device=dev)
                tb = (gs * G) // 128
                gt = gs % (128 // G)
                bt = ((bh[:, None, :, None].to(torch.int64) * s + (tb * 128)[None, None, None, :]) * 4096)
                eden_t = bt + (1 << 40) + (dd[None, :, None, None] * (128 // G) + gt[None, None, None, :])
            tcg, tsfb = _encode_groups(xt, G, rounding, amt, elem_t, seed, use_four_over_six, eden_t, transposed=True)
            tcodes = tcg.reshape(b, d, h, s)
            tsf = tsfb.permute(0, 2, 1, 3)                                                      # (b, h, d, s/G)
        packed = pack_nibbles(tcodes)                                                            # (b, d, h, s/2)
        if transposed_copy_layout == "batch_head_dim_seq":
            packed = packed.permute(0, 2, 1, 3).contiguous()
        out["transposed_codes"] = packed
        tsf = torch.nn.functional.pad(tsf.contiguous(), (0, 0, 0, 128 - d))                   # atom rows d..127: 0
        out["transposed_scale_factors"] = sf_atoms_from_table(tsf.contiguous())
    return out


def fp4_blocks_2d_reference(x: torch.Tensor, **kw) -> dict:
    """The eager reference of fp4.quantize_fp4_blocks_2d."""
    return fp4_rows_reference(x, _blocks_2d=True, **kw)


# ------------------------------------------------------------------------------------------------------- dequantisers
def fp4_dequantize_rows_reference(codes: torch.Tensor, scale_factors: torch.Tensor, dtype=torch.float32,
                                  scale_group_size: int = 16) -> torch.Tensor:
    """codes (b, R, h, n / 2) + the 7-D SF view -> (b, R, h, n) values code * SF (also the transposed copy, R = d)."""
    c = codes.contiguous().view(torch.uint8)
    nib = torch.stack([c & 0xF, c >> 4], -1).reshape(*c.shape[:-1], -1)
    tbl = table_from_sf_atoms(scale_factors).permute(0, 2, 1, 3)[:, :c.shape[1]]
    G = scale_group_size
    if G == 16:
        sfv = tbl.contiguous().view(torch.float8_e4m3fn).float()
    else:
        sfv = torch.where(tbl > 0, torch.exp2(tbl.float() - 127.0), torch.zeros_like(tbl, dtype=torch.float32))
    sfv = sfv[..., :nib.shape[-1] // G]
    return (e2m1_decode(nib).view(*nib.shape[:-1], -1, G) * sfv[..., None]).reshape(nib.shape).to(dtype)


def fp4_dequantize_transposed_reference(tcodes: torch.Tensor, tscale_factors: torch.Tensor, key_order: bool,
                                        scale_group_size: int = 16) -> torch.Tensor:
    """The transposed copy (b, d, h, s / 2) + its SF view -> (b, s, h, d) values in the original token order."""
    xt = fp4_dequantize_rows_reference(tcodes, tscale_factors, torch.float32, scale_group_size)   # (b, d, h, s)
    y = xt.permute(0, 3, 2, 1).contiguous()
    if key_order:
        b, s, h, d = y.shape
        y = y.reshape(b, s // 128, 4, 32, h, d)[:, :, [0, 2, 1, 3]].reshape(b, s, h, d)
    return y


def fp4_quantize_dequantize_2d_reference(t2d: torch.Tensor, scale_group_size: int = 16) -> torch.Tensor:
    """(M, K) -> fp32 values of its NVFP4 (16) / MXFP4 (32) quantisation along K (M % 128 == 0, K % 128 == 0): the
    reference of an operand's dequantised values (no kernel)."""
    m, k = t2d.shape
    x = t2d.contiguous().view(1, m, k // 128, 128)
    r = fp4_rows_reference(x, scale_group_size=scale_group_size)
    return fp4_dequantize_rows_reference(r["codes"], r["scale_factors"], torch.float32, scale_group_size).reshape(m, k)


# ------------------------------------------------------------------------------------------------- the causal diagonal
# Definitions (fp4_diagonal.py's header): per key row x (128 values, V as read, K after RoPE / Hadamard), e = ceil(log2(
# max(amax(x) * f32(1/6), 2^-120))) clamped to +-120 (-0.0 for a log in (-1, 0)); per 128-key tile t, the tile mean m_t
# = sum(e) / 128 and c_t = clamp(rint(fp32(m_0 + ... + m_{t-2}) / (t - 1)), +-7) (0 for t < 2); 2^n = bits (n + 127) << 23.
def pow2i(e: torch.Tensor) -> torch.Tensor:
    """2^e from the exponent bits ((int(e) + 127) << 23)."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def ceil_log2_amax6(x: torch.Tensor) -> torch.Tensor:
    """per row (last dim) e as defined above, fp32."""
    amax = nanless_amax(x.float())
    y = torch.maximum(amax * f32(1.0 / 6.0), torch.full_like(amax, 2.0 ** -120))
    m, ex = torch.frexp(y)                                          # y = m 2^ex, m in [0.5, 1)
    e = torch.where(m == 0.5, ex - 1, ex).clamp(-120, 120).float()
    return torch.where((e == 0) & (y < 1.0), torch.full_like(e, -0.0), e)


def _prefix_c(e: torch.Tensor) -> torch.Tensor:
    """e (b, s, hk) -> c (b, tiles, hk) as defined above (the sums are exact)."""
    b, s, hk = e.shape
    nt = s // 128
    means = e.reshape(b, nt, 128, hk).sum(2) * 0.0078125
    csum = torch.cumsum(means.double(), 1).float()
    c = torch.zeros(b, nt, hk, device=e.device)
    for t in range(2, nt):
        c[:, t] = torch.round(csum[:, t - 2] / float(t - 1)).clamp(-7.0, 7.0)
    return c


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0xF, packed >> 4), -1).reshape(*packed.shape[:-1], -1)


def _tile_table(byte: torch.Tensor, rows: int, cols_per_tile: int, per_key_rows: bool) -> torch.Tensor:
    """byte (b, tiles, hk) -> an SF table (b, hk, R, K) whose bytes of tile t are byte[t]: per_key_rows: R = s keys, K =
    cols_per_tile; else R = rows (d), K = tiles * cols_per_tile."""
    b, nt, hk = byte.shape
    bt = byte.permute(0, 2, 1)                                                       # (b, hk, nt)
    if per_key_rows:
        return bt.repeat_interleave(128, -1)[..., None].expand(b, hk, nt * 128, cols_per_tile).contiguous()
    return bt.repeat_interleave(cols_per_tile, -1)[:, :, None, :].expand(b, hk, rows, nt * cols_per_tile).contiguous()


def fp4_causal_diagonal_reference(key, value, softmax_scale, *, key_channel="fold", query=None, make_key_row_copy=False,
                                  make_option_a_key_factors=False, make_key_norm_prefix_maxima=False, query_seqlen=None,
                                  query_heads=None, apply_hadamard_rotation=False, hadamard_sign_seed=4321,
                                  rope_cos=None, rope_sin=None) -> dict:
    """The eager reference of fp4_diagonal.quantize_fp4_causal_diagonal (same arguments, same outputs)."""
    import numpy as np
    b, s, hk, d = value.shape
    nt = s // 128
    dev = value.device
    out = {}
    qk_kw = dict(apply_hadamard_rotation=apply_hadamard_rotation, hadamard_sign_seed=hadamard_sign_seed,
                 rope_cos=rope_cos, rope_sin=rope_sin)
    if query is not None:
        r = fp4_rows_reference(query, **qk_kw)
        out.update(q4=r["codes"], sfq=r["scale_factors"])
    if make_key_row_copy:
        r = fp4_rows_reference(key, **qk_kw)
        out.update(k4=r["codes"], sfk=r["scale_factors"])
    # ---------------------------------------------------------------- V
    vf = value.float()
    ev = ceil_log2_amax6(vf)                                                         # (b, s, hk)
    pl = fp4_blocks_2d_reference(value)                                              # plain: 16 x 16 blocks, nearest
    out["v4"], out["sfv"] = pl["codes"], pl["scale_factors"]
    plain_codes = _unpack_nibbles(pl["codes"])                                       # (b, s, hk, 128)
    local_codes = e2m1_encode_rn(vf * pow2i(-ev)[..., None])                        # unit scale, key-local
    out["v4b"] = pack_nibbles(local_codes)
    out["vt4"] = torch.cat([pack_nibbles(plain_codes.permute(0, 2, 3, 1)), pack_nibbles(local_codes.permute(0, 2, 3, 1))],
                           -1)                                                       # (b, hk, d, s): plain | local
    tbl_v = table_from_sf_atoms(pl["scale_factors"])                                 # (b, hk, s, 8)
    plain_t = tbl_v[:, :, ::16, :].repeat_interleave(16, -1).transpose(2, 3)        # (b, hk, 128 d, s / 16)
    cv = _prefix_c(ev)                                                               # (b, nt, hk)
    cbv, _ = e4m3_round(pow2i(cv))
    out["sfvt"] = sf_atoms_from_table(torch.cat([plain_t, _tile_table(cbv, 128, 8, False)], -1).contiguous())
    out["sfvb"] = sf_atoms_from_table(_tile_table(cbv, 128, 8, True))
    cv_k = cv.repeat_interleave(128, 1)                                              # (b, s, hk)
    inv = float(np.float32(1.0 / (softmax_scale * 1.4426950408889634)))
    bias = (ev - cv_k) * inv
    u = pow2i(-ev) * pow2i(cv_k)
    out.update(vtok_bias=bias, vtok_u=u, vtok_s=pow2i(ev), vtok_c=cv)
    if make_option_a_key_factors:
        out["rk"] = (pow2i(ev) * pow2i(-cv_k)).permute(0, 2, 1).contiguous()
    vn = vf.norm(dim=-1)
    vn = torch.maximum(vn, vn * u)
    rk = torch.ones_like(vn)
    # ---------------------------------------------------------------- K
    if key_channel != "off":
        kx = _prepare(key, rope_cos, rope_sin, apply_hadamard_rotation, hadamard_sign_seed)[1]
        ek = ceil_log2_amax6(kx)
        ck = _prefix_c(ek)
        ck_k = ck.repeat_interleave(128, 1)
        kl = e2m1_encode_rn(kx * pow2i(-ek)[..., None])                              # (b, s, hk, 128)
        perm = torch.arange(128, device=dev)
        blk = perm // 32
        perm = ((blk % 2) * 2 + blk // 2) * 32 + perm % 32
        src = (torch.arange(nt, device=dev)[:, None] * 128 + perm[None, :]).reshape(-1)
        out["xt4b"] = pack_nibbles(kl[:, src].permute(0, 3, 2, 1))                    # (b, d, hk, s / 2)
        cbk, _ = e4m3_round(pow2i(ck))
        out["sftb"] = sf_atoms_from_table(_tile_table(cbk, 128, 8, False))
        rk = pow2i(ek) * pow2i(-ck_k)                                                # (b, s, hk)
        rk_t = rk.permute(0, 2, 1)
        out["kr"] = torch.cat([torch.ones_like(rk_t) if key_channel == "cut" else rk_t, rk_t], -1).contiguous()
        out["vb2"] = torch.cat([torch.zeros_like(rk_t), bias.permute(0, 2, 1)], -1).contiguous()
        if key_channel == "fold":
            out["vu2"] = torch.cat([rk_t, (u * rk).permute(0, 2, 1)], -1).contiguous()
            a = torch.clamp(ek.reshape(b, nt, 128, hk).amax(2) - ck, min=0.0)         # (b, nt, hk)
            out["dva"] = (448.0 * pow2i(-a)).permute(0, 2, 1).contiguous()
            a_k = a.repeat_interleave(128, 1)
            out["fdk"] = pow2i(ck_k - ek).permute(0, 2, 1).contiguous()
            out["fdv"] = pow2i(a_k + ck_k - ek).permute(0, 2, 1).contiguous()
            r_t = rk_t[..., None]                                                    # (b, hk, s, 1)
            sfval = tbl_v.contiguous().view(torch.float8_e4m3fn).float()
            out["sfv2"] = sf_atoms_from_table(e4m3_round(torch.clamp(sfval * r_t, max=448.0))[0].contiguous())
            pc = pow2i(cv_k).permute(0, 2, 1)[..., None]
            out["sfvb2"] = sf_atoms_from_table(
                e4m3_round(torch.clamp(pc * r_t, max=448.0))[0].expand(b, hk, s, 8).contiguous())
            vn = vn * rk
        else:
            out["vu2"] = torch.cat([torch.ones_like(rk_t), u.permute(0, 2, 1)], -1).contiguous()
            rk = torch.ones_like(rk)
        if key_channel == "cut":
            r = fp4_rows_reference(kx.to(key.dtype), make_row_copy=False, make_transposed_copy=True)
            out["xt4"], out["sft"] = r["transposed_codes"], r["transposed_scale_factors"]
            t = table_from_sf_atoms(r["transposed_scale_factors"])                    # (b, hk, 128 d, s / 16)
            rows = torch.arange(128, device=dev)
            dst = (rows % 32) + 32 * ((rows // 32 + 2) % 4)
            t1 = torch.empty_like(t)
            t1[:, :, dst] = t
            out["sft1"] = sf_atoms_from_table(t1)
    if make_key_norm_prefix_maxima:
        s_q = query_seqlen if query_seqlen is not None else (query.shape[1] if query is not None else s)
        h_q = query_heads if query_heads is not None else (query.shape[2] if query is not None else hk)
        idx = torch.arange(s_q, device=dev) + (s - s_q)

        def pmax(x):
            cm = torch.cummax(x.float(), dim=1).values                               # (b, s, hk)
            return cm[:, idx].repeat_interleave(h_q // hk, dim=2).permute(0, 2, 1).contiguous()
        out["vmax_q"] = pmax(vn)
        if key_channel == "fold":
            out["rmax_q"] = pmax(rk)
    return out


# ======================================================================================================================
# FP8 references: fp8_* functions; the FP8 set's own Hadamard helpers are fp8_hadamard_signs / fp8_hadamard_rotate
# (signs as a float tensor from a seed), the FP4 set's hadamard_rotate takes the sign tuple.
# ======================================================================================================================
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
TORCH_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
FLT_MIN = 2.0 ** -126
LOG2E_F32 = torch.tensor(1.4426950408889634, dtype=torch.float32).item()


# ------------------------------------------------------------------------------------------------ element formats
def fp8_round(y: torch.Tensor, fmt: str) -> torch.Tensor:
    """fp32 -> FP8 codes: clamp to +-FP8_MAX, round to nearest even."""
    m = FP8_MAX[fmt]
    return y.float().clamp(-m, m).to(TORCH_FP8[fmt])


def fp8_hadamard_signs(seed: int = 4321, d: int = 128) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    return (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float()


def fp8_hadamard_rotate(x: torch.Tensor, seed: int = 4321) -> torch.Tensor:
    """(x * signs) H / sqrt(128) along the last dim, fp32 butterfly order."""
    y = x.float() * fp8_hadamard_signs(seed, x.shape[-1]).to(x.device)
    d = y.shape[-1]
    hl = 1
    while hl < d:
        y = y.reshape(*y.shape[:-1], d // (2 * hl), 2, hl)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), -2).reshape(*x.shape)
        hl *= 2
    return y * torch.tensor(1.0 / math.sqrt(d), dtype=torch.float32)


def hadamard_matrix_check(x: torch.Tensor, seed: int = 4321) -> float:
    """max |butterfly - float64 (x * signs) H / sqrt(d)|, relative to max |x| (should be ~1e-7)."""
    d = x.shape[-1]
    i = torch.arange(d)
    par = torch.zeros(d, d, dtype=torch.int64)
    m = i[:, None] & i[None, :]
    while bool(m.any()):
        par ^= m & 1
        m = m >> 1
    hmat = (1 - 2 * par).double() / math.sqrt(d)
    ref = (x.double().cpu() * fp8_hadamard_signs(seed, d).double()) @ hmat
    return ((fp8_hadamard_rotate(x).double().cpu() - ref).abs().max() / x.double().abs().max().clamp_min(1e-300)).item()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (b, s, h, 128); cos / sin fp32 (s, 64) or (64, s). GPT-NeoX halves in quant/_rotations.py's defined arithmetic:
    lo = fma(x0, c, -(x1 s)); hi = fma(x1, c, x0 s) with (64, s) tables, fma(x0, s, x1 c) with (s, 64) tables; the row
    rounded to x.dtype."""
    s = x.shape[1]
    transposed = tuple(cos.shape) == (64, s) and tuple(cos.shape) != (s, 64)
    if transposed:
        cos, sin = cos.t(), sin.t()
    c = cos[None, :, None, :].float()
    sn = sin[None, :, None, :].float()
    xf = x.float()
    x0, x1 = xf[..., :64], xf[..., 64:]
    lo = fma_f32(x0, c, -(x1 * sn))
    hi = fma_f32(x1, c, x0 * sn) if transposed else fma_f32(x0, sn, x1 * c)
    return torch.cat((lo, hi), dim=-1).to(x.dtype).float()


def flush_subnormals(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < FLT_MIN, x * 0.0, x)


# ------------------------------------------------------------------------------------------------ UE8M0 selection
def ue8m0_exponent(amax: torch.Tensor, fmt: str, flush: bool = True):
    """(byte uint8, scale fp32) of the round-up selector for block maxima amax (fp32): smallest e with
    FMAX 2^e >= amax; zero blocks (and, with flush, blocks with fp32(amax * fp32(1/FMAX)) < 2^-126) -> 0, 0."""
    fmax = FP8_MAX[fmt]
    a = amax.double()
    _, ex = torch.frexp(a / fmax)          # a / fmax = mant * 2^ex, mant in [0.5, 1): 2^(ex-1) <= q < 2^ex
    e = ex.to(torch.int64)
    for _ in range(3):                     # exact corrections (FMAX * 2^e is exact in float64)
        e = torch.where(fmax * torch.exp2(e.double() - 1) >= a, e - 1, e)
        e = torch.where(fmax * torch.exp2(e.double()) < a, e + 1, e)
    pos = a > 0
    if flush:
        r = (amax.float() * torch.tensor(1.0 / fmax, dtype=torch.float32)).double()
        pos = r >= FLT_MIN
    byte = torch.where(pos, e + 127, torch.zeros_like(e)).to(torch.uint8)
    scale = torch.where(pos, torch.exp2(e.double()), torch.zeros_like(a)).float()
    return byte, scale


def _atom_view(per_row: torch.Tensor) -> torch.Tensor:
    """(b, h, R, G) bytes (row r, K-group g; R % 128 == 0, G % 4 == 0) -> (b, h, R/128, G/4, 32, 4, 4) atoms."""
    b, h, R, G = per_row.shape
    t = per_row.reshape(b, h, R // 128, 4, 32, G // 4, 4)            # r = 32 j + i
    return t.permute(0, 1, 2, 5, 4, 3, 6).contiguous()                # (b, h, rt, kt, i, j, g)


# ------------------------------------------------------------------------------------------------ blocks
def fp8_blocks_reference(x: torch.Tensor, *, fp8_format="e4m3", block_rows=32, block_columns=128,
                         flush_subnormal_inputs=True, apply_hadamard_rotation=False, hadamard_sign_seed=4321,
                         rope_cos=None, rope_sin=None, compute_d_with_output=None, shifted_log2_lse_from=None,
                         log2_lse_shift=0.0, also_make_nvfp4_copy=False, nvfp4_copy_scaling="token_block",
                         nvfp4_copy_rounding="nearest", nvfp4_copy_seed=0, nvfp4_copy_tensor_scale=None,
                         token_local_exponents=False, token_axis_scale="block") -> dict:
    """The outputs of quantize_fp8_blocks from the definitions (keys as the kernel's; D as float64).
    token_local_exponents: codes / sf_tok / D from each token's own exponent (1 x d), sf_dim / e / fac from the block's,
    token_factors = 2^(e_t - 127) (1 for an all-zero token); token_axis_scale="unit": sf_dim bytes 127."""
    b, s, h, d = x.shape
    xf = x.float()
    if rope_cos is not None:
        xf = apply_rope(x, rope_cos, rope_sin)
    if apply_hadamard_rotation:
        xf = fp8_hadamard_rotate(xf, hadamard_sign_seed)
    if flush_subnormal_inputs:
        xf = flush_subnormals(xf)
    R, C = block_rows, min(block_columns, d)   # block_columns >= d: one exponent per block of the whole head dim
    a = xf.abs().permute(0, 2, 1, 3)                                  # (b, h, s, d)
    blk = a.reshape(b, h, s // R, R, d // C, C).amax(dim=(3, 5))      # (b, h, s/R, d/C)
    byte, scale = ue8m0_exponent(blk, fp8_format, flush_subnormal_inputs)
    # per element (b, h, s, d)
    sc_el = scale.repeat_interleave(R, 2).repeat_interleave(C, 3)
    if token_local_exponents:
        assert R == 32 and C == d
        tbyte, tscale = ue8m0_exponent(a.amax(dim=3), fp8_format, flush_subnormal_inputs)       # (b, h, s)
        sc_el = tscale[..., None].expand(b, h, s, d)
    inv = torch.where(sc_el > 0, 1.0 / sc_el.double(), torch.zeros_like(sc_el, dtype=torch.float64))
    y = (xf.permute(0, 2, 1, 3).double() * inv).float()
    codes = fp8_round(y, fp8_format).permute(0, 2, 1, 3).contiguous()
    out = dict(codes=codes, codes_t=codes.permute(0, 2, 3, 1).contiguous())
    b32 = byte.repeat_interleave(R, 2).repeat_interleave(C, 3)        # (b, h, s, d) byte of each element
    if C >= 32:
        # rows = tokens, K = d / 32; one atom along d -- d = 64 fills K-groups 2, 3 with groups 0, 1
        k = b32[..., ::32]
        if token_local_exponents:
            k = tbyte[..., None].expand(b, h, s, d // 32)
            out["token_factors"] = torch.where(tscale > 0, tscale, torch.ones_like(tscale))
        out["sf_tok_raw"] = _atom_view(k.repeat(1, 1, 1, 4 // k.shape[-1]).contiguous() if k.shape[-1] < 4 else k.contiguous())
    if R >= 32:
        # rows = d, K = tokens / 32; d = 64: atom rows 64..127 repeat rows 0..63
        t = b32.transpose(2, 3)[..., ::32]
        if d < 128:
            t = t.repeat(1, 1, 128 // d, 1)
        if token_axis_scale == "unit":
            t = torch.full_like(t, 127)
        out["sf_dim_raw"] = _atom_view(t.contiguous())
        if C >= 32:
            e = b32[:, :, ::32, ::32].contiguous()                     # (b, h, s/32, d/32)
            out["e"] = e
            # an all-zero block: its 128-token tile's largest factor, 1 for an all-zero tile (output_token_block_factors)
            e0 = e[..., 0].long()
            tile_max = e0.reshape(*e0.shape[:-1], -1, 4).amax(-1, keepdim=True).expand(*e0.shape[:-1], -1, 4)
            e0 = torch.where(e0 > 0, e0, tile_max.reshape(e0.shape))
            fac = torch.where(e0 > 0, torch.exp2(e0.double() - 127), torch.ones_like(e0, dtype=torch.float64))
            out["fac"] = fac.float()
    deq = codes.float().permute(0, 2, 1, 3).double() * sc_el.double()  # (b, h, s, d)
    if also_make_nvfp4_copy:
        if nvfp4_copy_scaling == "token_block":
            y4 = (xf.permute(0, 2, 1, 3).double() * inv).float()
        elif nvfp4_copy_scaling == "raw":
            y4 = xf.permute(0, 2, 1, 3)
        else:
            y4 = (xf.permute(0, 2, 1, 3) * (1.0 / nvfp4_copy_tensor_scale.reshape(())).float())
        x4, sf4, deq4 = nvfp4_dp_operand_reference(y4, nvfp4_copy_rounding == "stochastic", nvfp4_copy_seed)
        out["x4"] = x4.permute(0, 2, 1, 3).contiguous()
        out["sf4_raw"] = sf4
        if compute_d_with_output is not None:
            ofac = {"token_block": scale.repeat_interleave(R, 2)[..., 0].double(),
                    "raw": torch.ones(b, h, s, dtype=torch.float64, device=x.device),
                    "per_tensor": torch.full((b, h, s), float(nvfp4_copy_tensor_scale.reshape(()).item()),
                                             dtype=torch.float64, device=x.device)}[nvfp4_copy_scaling]
            o = compute_d_with_output.permute(0, 2, 1, 3).double()
            out["D"] = (deq4 * o).sum(-1) * ofac
    elif compute_d_with_output is not None:
        out["D"] = (deq * compute_d_with_output.permute(0, 2, 1, 3).double()).sum(-1)
    if shifted_log2_lse_from is not None:
        out["lse_log2"] = lse_log2_reference(shifted_log2_lse_from, log2_lse_shift)
    return out


def lse_log2_reference(lse: torch.Tensor, shift: float) -> torch.Tensor:
    """where(lse == -inf, 0, lse * fp32(log2 e)) - fp32(shift), fp32 ops."""
    prod = lse * torch.tensor(LOG2E_F32, dtype=torch.float32, device=lse.device)
    return torch.where(torch.isneginf(lse), torch.zeros_like(lse), prod) - torch.tensor(float(shift), dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ NVFP4 (option A)


def e2m1_round_nearest(y: torch.Tensor) -> torch.Tensor:
    """fp32 -> E2M1 code nibbles (sign bit 3): nearest grid point, ties to the even code, |y| > 6 saturates."""
    a = y.abs().double().clamp(max=6.0)
    grid = E2M1_GRID.double().to(y.device)
    dist = (a[..., None] - grid).abs()
    best = dist.min(-1, keepdim=True).values
    cand = (dist == best)
    # ties: prefer the even code (codes are the grid indices)
    even = cand & (torch.arange(8, device=y.device) % 2 == 0)
    code = torch.where(even.any(-1), even.float().argmax(-1), cand.float().argmax(-1))
    sign = torch.signbit(y).to(torch.int64)
    return (code + 8 * sign).to(torch.uint8)


def splitmix_u01(idx: torch.Tensor, seed: int) -> torch.Tensor:
    """uniform [0, 1): splitmix64 finaliser of idx * golden + seed (int64 wrap-around), top 24 bits."""
    z = idx * -7046029254386353131 + seed
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * -4658895280553007687
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * -7723592293110705685
    z = z ^ ((z >> 31) & 0x1FFFFFFFF)
    return ((z >> 40) & 0xFFFFFF).float() * (1.0 / 16777216.0)


def e2m1_round_stochastic(y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    a = y.abs().clamp(max=6.0)
    lo = torch.where(a < 2, torch.floor(a * 2) * 0.5, torch.where(a < 4, torch.floor(a), torch.full_like(a, 4.0)))
    step = torch.where(a < 2, torch.full_like(a, 0.5), torch.where(a < 4, torch.ones_like(a), torch.full_like(a, 2.0)))
    v = torch.where(u * step < a - lo, lo + step, lo)
    return torch.where(y < 0, -v, v)


def nvfp4_dp_operand_reference(y: torch.Tensor, stochastic: bool, seed: int):
    """NVFP4 1x16 along d of y (b, h, s, 128) fp32 -- option A's dO encoding (nvfp4_2d_triton._q_groups):
    SF = E4M3_rn(amax * fp32(1/6 | 17/96)); codes of y / SF (y * fl(1 / SF)), 0 for a zero SF; stochastic: the E2M1
    grid point chosen with u = splitmix(idx * 16 + n, seed), idx = ((b h + head) s + 128 t) 4096 + 8 r + g.
    Returns (codes packed (b, h, s, 64) uint8, SF atoms (b, h, s/128, 2, 32, 4, 4), dequantised values float64)."""
    b, h, s, d = y.shape
    ng = d // 16
    g = y.reshape(b, h, s, ng, 16)
    amax = g.abs().amax(-1)
    k = torch.tensor(17.0 / 96.0 if stochastic else 1.0 / 6.0, dtype=torch.float32)
    sf8 = (amax * k).to(torch.float8_e4m3fn)
    sfr = sf8.float()
    inv = torch.where(sfr > 0, (1.0 / sfr.double()).float(), torch.zeros_like(sfr))
    v = g * inv[..., None]
    if stochastic:
        bi = torch.arange(b, device=y.device).view(b, 1, 1, 1, 1)
        hi = torch.arange(h, device=y.device).view(1, h, 1, 1, 1)
        si = torch.arange(s, device=y.device).view(1, 1, s, 1, 1)
        gi = torch.arange(ng, device=y.device).view(1, 1, 1, ng, 1)
        ni = torch.arange(16, device=y.device).view(1, 1, 1, 1, 16)
        idx = ((bi * h + hi) * s + (si // 128) * 128) * 4096 + (si % 128) * 8 + gi
        u = splitmix_u01(idx.to(torch.int64) * 16 + ni, int(seed) & 0x3FFFFFFFFFFFFFFF)
        v = e2m1_round_stochastic(v, u)
    codes = e2m1_round_nearest(v.reshape(b, h, s, d))
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    sf_atoms = _atom_view(sf8.view(torch.uint8))
    val = E2M1_GRID.double().to(y.device)[(codes & 7).long()] * torch.where((codes & 8) > 0, -1.0, 1.0).double()
    deq = val * sfr.double().repeat_interleave(16, -1)
    return packed, sf_atoms, deq


# ------------------------------------------------------------------------------------------------ per-tensor
def fp8_per_tensor_reference(x: torch.Tensor, *, fp8_format="e4m3", scale=None, apply_hadamard_rotation=False,
                             hadamard_sign_seed=4321, rope_cos=None, rope_sin=None, row_divisor=None,
                             compute_d_with_output=None, shifted_log2_lse_from=None, log2_lse_shift=0.0,
                             ds_bound_from=None, power_of_two_scale=False) -> dict:
    """The outputs of quantize_fp8_per_tensor for one tensor from the definitions (D, row norms, bound in float64)."""
    xf = x.float()
    if row_divisor is not None:
        xf = (xf / row_divisor.permute(0, 2, 1)[..., None]).to(x.dtype).float()
    if rope_cos is not None:
        xf = apply_rope(xf.to(x.dtype), rope_cos, rope_sin)
    if apply_hadamard_rotation:
        xf = fp8_hadamard_rotate(xf, hadamard_sign_seed)
    fmax = FP8_MAX[fp8_format]
    # the scale in float32 (the FP64-free contract): s = max(amax / FP8_MAX, 1e-12) and 1 / s, each correctly rounded
    if scale is None:
        s = max(np.float32(xf.abs().max().item()) / np.float32(fmax), np.float32(1e-12))
        if power_of_two_scale:
            m_, e_ = math.frexp(float(s))           # s = m 2^e, m in [0.5, 1) (exact: s is a float32)
            s = s if m_ == 0.5 else np.float32(math.ldexp(1.0, e_))
    else:
        s = np.float32(scale)
    inv32 = torch.tensor(float(np.float32(1.0) / np.float32(s)), dtype=torch.float32)
    s = float(s)
    codes = fp8_round(xf * inv32, fp8_format)
    out = dict(codes=codes, scale=s, row_norm2_max=(xf.double() ** 2).sum(-1).max().item())
    s32 = torch.tensor(s, dtype=torch.float64).float().double()
    if compute_d_with_output is not None:
        dv = (codes.double() * s32 * compute_d_with_output.double()).sum(-1).permute(0, 2, 1)
        out["D"] = dv
        if ds_bound_from is not None:
            pmax, vn2 = ds_bound_from
            nrm = (xf.double() ** 2).sum(-1).sqrt().permute(0, 2, 1)
            out["ds_bound"] = (pmax.double().clamp(max=1.0) * (nrm * math.sqrt(float(vn2.reshape(())))
                                                               + dv.abs())).max().item()
    if shifted_log2_lse_from is not None:
        out["lse_log2"] = lse_log2_reference(shifted_log2_lse_from, log2_lse_shift)
    return out


def dequantize_fp8_blocks(codes: torch.Tensor, sf_tok_raw: torch.Tensor) -> torch.Tensor:
    """codes (b, s, h, d) x the (tokens, d / 32) UE8M0 atoms (b, h, s/128, d/128, 32, 4, 4) -> fp32 (b, s, h, d)."""
    b, s, h, d = codes.shape
    e = sf_tok_raw.to(torch.int32).reshape(b, h, s // 128, d // 128, 32, 4, 4)
    e = e.permute(0, 1, 2, 5, 4, 3, 6).reshape(b, h, s, d // 32)          # (row = 32 j + i, group = 4 kt + g)
    scale = torch.where(e == 0, torch.zeros_like(e, dtype=torch.float32), torch.exp2((e - 127).float()))
    x = codes.float().permute(0, 2, 1, 3).reshape(b, h, s, d // 32, 32) * scale[..., None]
    return x.reshape(b, h, s, d).permute(0, 2, 1, 3)

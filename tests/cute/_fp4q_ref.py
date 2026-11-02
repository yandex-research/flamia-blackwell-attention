"""Torch reference of the all-FP4 query-row backward's arithmetic (flash_bwd_sm100_fp4q.py) for tests/cute/test_fp4_all.py.

It models the kernel as SHIPPED, given the operands and host scalars `interface_bwd_fp4q.fp4q_prepare` hands the kernel
(it does not re-derive them: the reference checks the kernel, the driver's scale choices are checked by the accuracy
gates). Deterministic modes only -- RTN dS^ for dQ (ds_sr=False) and an RTN dO^ are what the test passes; the dK
operand's stochastic rounding is not modelled, and it is not reached at the shipped defaults (see below).

What it covers, and where each piece lives in the kernel / driver:

  * P^ reproduction: the forward's group max, E4M3 group SF, Direct-P Mitchell values (or exp2, as the call's
    BwdQSpec.exp2 says) and their E2M1 codes; w = code SF / Z, m = Mitchell SF / Z.
  * `key-local-v-scale` (vtok, the default for causal: causal_safe_v="auto"): the per-key raw-score bias before the
    group maxima and the per-key weight u_k in m; the all-FP4 route's dV operand is built from the unbiased m
    (`vtok_dv_u`), the hybrid route's from m x 448.
  * dS' = (w dP' - D' m) ds_mul, dP' = dO^ V^T.
  * per-row factors of the driver: the causal dS scale (fp4_ds_multiplier="auto" -> "causal" on a causal hybrid
    backward: dO row q carries F_q, dQ and Q row q in dK's B operand carry 1 / F_q) and the non-causal query-row boost
    ("global_row_boost", the default for non-causal: w and m of row q carry r_q, Q^ / dO^ rows of dK / dV carry 1 / r_q,
    dQ row q is divided by r_q). Both reach this reference as `info["r_bshd"]` and, for QROW, `info["ds_row"]` is None.
  * dQ: dS'^ x K^T. Hybrid (dkv="e4m3"): E2M1 codes with 1x16 E4M3 scales per query row (16 keys); all-FP4: 16x16
    (queries x keys) blocks.
  * dK, hybrid: E4M3(dS') x E4M3 Q (per tensor) with the BwdQSpec.mxds block exponent -- "blk32" (the shipped
    default, lowprec.specs.FP4Q_MXDS_DEFAULT): one UE8M0 exponent e = ceil(log2(amax / 448)) per 32 keys x 32 queries,
    codes E4M3(dS' 2^-e); "mx32": per 16 keys x 32 queries; "per_tensor": plain per-tensor E4M3 (with dk_sr off; the SR
    variant is refused). With MXDS on, the kernel never reaches its dK stochastic rounding.
  * dK / dV, all-FP4: dS'^T Q^T and P^v^T dO^T with 16x16 blocks (P^v boosted by 2^11 before its E4M3 scale).
  * epilogue scales: the kernel's dk_scale / dv_scale / ds_mul (the device `scl` vector when the driver uses one,
    otherwise `info`), dq_scale from `info`; GQA: dK / dV summed over each kv head's q heads.

Not modelled (the reference refuses): the diagonal-only route (causal_safety="diagonal"), dq8 (dq_operand="e4m3"), a
Hadamard `signs` operand set, SR on the dK operand without MXDS.

Derived from an earlier offline reference of the kernel arithmetic, extended for the causal dS scale, QROW, MXDS and
the vtok dV operand.
"""
import math

import torch

LOG2_E = math.log2(math.e)

_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e2m1_rtn(y):
    """Round to the nearest E2M1 value (saturating at 6), ties to the even code -- cvt.rn.satfinite.e2m1x2."""
    grid = torch.tensor(_GRID, dtype=torch.float32, device=y.device)
    mid = torch.tensor(_MID, dtype=torch.float32, device=y.device)
    a = y.abs().clamp(max=6.0)
    idx = torch.bucketize(a, mid)
    tie = (a[..., None] == mid).any(-1)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)  # a tie at MID[i] sits between GRID[i] and GRID[i + 1]
    mag = grid[idx]
    return torch.where(y < 0, -mag, mag)


def e4m3_round(x):
    """RN onto E4M3, saturating at +-448."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def kernel_value_knobs(spec):
    """(exp2, delta, window) exactly as flash_bwd_sm100_fp4q.py's __init__ resolves them from its BwdQSpec (`spec`, the
    call's plan.specs.bwdq)."""
    exp2 = bool(spec.exp2)
    from flash_attn.cute.flash_fwd_sm100_fp4 import DIRECTP_DELTA, DIRECTP_WINDOW
    delta = 0.0 if exp2 else DIRECTP_DELTA
    window = DIRECTP_WINDOW
    return exp2, delta, window


def max_offset():
    from flash_attn.cute.flash_bwd_sm100_fp4q import FP4_PV_RESCALE_THRESHOLD
    return math.log2(6) + math.log2(448) - FP4_PV_RESCALE_THRESHOLD


def mxds_mode(spec):
    """"blk32" | "mx32" | "per_tensor": the dK operand's block exponent, as the kernel's _mxds_init resolves it from its
    BwdQSpec (non-diagonal route)."""
    from flash_attn.cute.lowprec.specs import BwdQSpec
    m = spec.mxds if spec.mxds is not None else BwdQSpec.rule_mxds(False)
    assert m in ("per_tensor", "mx32", "blk32"), m
    return m


def mitchell(x, spec):
    """2^floor(x) (1 + frac(x)) inside the classifier window; below it, the clamp pattern's tiny value."""
    exp2, _, window = kernel_value_knobs(spec)
    if exp2:
        return torch.pow(2.0, x)
    lo = -(window - 1)
    xc = x.clamp(min=lo)
    f = torch.floor(xc)
    m = torch.pow(2.0, f) * (1.0 + (xc - f))
    return torch.where(x < lo, torch.full_like(x, 2.0 ** lo), m)


def _f64(x):
    return x.double() if isinstance(x, torch.Tensor) else float(x)


def kernel_scales(prep):
    """(dk_scale, dv_scale, ds_mul) as the kernel applies them: from the device `scl` vector when the driver built one
    (read as float32, as the kernel does), otherwise the driver's host values."""
    scl = prep["keep"][12]
    if scl is not None:
        s = scl.double()
        return s[0], s[1], s[2]
    info = prep["info"]
    return _f64(info["dk_scale"]), _f64(info["dv_scale"]), _f64(info["ds_mul"])


def _blocks(x, bq, bk):
    """(b, h, s_q, s_k) -> (b, h, s_q / bq, bq, s_k / bk, bk)"""
    b, h, sq, sk = x.shape
    return x.reshape(b, h, sq // bq, bq, sk // bk, bk)


def reference(q, k, ops, prep, frame, lse, scale, causal, spec):
    """-> dict(dq, dk, dv) in (b, s, h, d) float32, plus dS (b, h, s_q, s_k) float64 = the kernel's dS' before any
    quantiser and dq_exact = the dQ that unquantised dS' would give through the same K^T operand and scales.
    spec: the kernel's BwdQSpec (the call's plan.specs.bwdq, as handed to fp4q_prepare)."""
    from quant_reference import (fp4_dequantize_rows_reference as deq_rows,
                                 fp4_dequantize_transposed_reference as dequantize_transposed)

    info = prep["info"]
    dkv8 = info["dkv"] == "e4m3"
    vtok = "vtok_bias" in ops
    assert not ops.get("vtok_diag", False), "the diagonal-only route is not modelled"
    assert info.get("kd") is None, "the diagonal K route is not modelled"
    assert not spec.dq8, "dq8 is not modelled"
    b, s, h, d = q.shape
    hk = k.shape[2]
    r = h // hk
    dev = q.device
    dq_, qt, kt = info["dq_"], info["qt"], info["kt"]
    Qh = deq_rows(ops["q4"], ops["sfq"], torch.float32).permute(0, 2, 1, 3).double()
    Kh = deq_rows(ops["k4"], ops["sfk"], torch.float32).permute(0, 2, 1, 3).double().repeat_interleave(r, 1)
    Vh = deq_rows(ops["v4"], ops["sfv"], torch.float32).permute(0, 2, 1, 3).double().repeat_interleave(r, 1)
    dOh = deq_rows(dq_["x4"], dq_["sf"], torch.float32).permute(0, 2, 1, 3).double()  # dO' = dO F / so_do
    Kt = dequantize_transposed(kt["xt4"], kt["sft"], True).permute(0, 2, 1, 3).double().repeat_interleave(r, 1)

    # ---- P^ reproduction (the forward's frame, group max, group SF, Direct-P / exp2 values, E2M1 codes)
    S = Qh @ Kh.transpose(-1, -2)  # raw scores
    if vtok:
        S = S + ops["vtok_bias"].permute(0, 2, 1).double().repeat_interleave(r, 1)[:, :, None, :]
    if causal:
        S = S.masked_fill(torch.ones(s, s, device=dev, dtype=torch.bool).triu(1), float("-inf"))
    sc2 = scale * LOG2_E
    G = S.reshape(b, h, s, s // 16, 16)
    gm = G.amax(-1, keepdim=True).clamp(min=-1e9)
    kb = (max_offset() - frame.double()[..., None, None]) - math.log2(6)
    sfx = torch.pow(2.0, gm * sc2 + kb)
    sfr = e4m3_round(sfx.float()).double()
    mit = mitchell((G - gm) * sc2 + math.log2(6) - kernel_value_knobs(spec)[1], spec)
    code = e2m1_rtn(torch.relu(mit).float()).double()
    uinv = torch.pow(2.0, frame.double() - lse.double() * LOG2_E - max_offset())[..., None, None]
    r_row = info.get("r_bshd")
    qrow = r_row is not None and info.get("ds_row") is None  # QROW multiplies w and m of row q by r_q
    if qrow:
        uinv = uinv * r_row.double()[..., 0].permute(0, 2, 1)[..., None, None]
    w = (code * sfr * uinv).reshape(b, h, s, s)
    u = (ops["vtok_u"].permute(0, 2, 1).double().repeat_interleave(r, 1).reshape(b, h, 1, s // 16, 16)
         if vtok else 1.0)
    m = (mit * sfx * uinv * u).reshape(b, h, s, s)

    # ---- dS' = (w dP' - D' m) ds_mul
    dk_scale, dv_scale, ds_mul = kernel_scales(prep)
    so_do = _f64(info["so_do"])
    dP = dOh @ Vh.transpose(-1, -2)
    Dp = (info["D"].double() / so_do)[..., None]
    dS = (w * dP - Dp * m) * ds_mul

    # ---- dQ
    if dkv8:  # E2M1 dS^ with 1x16 E4M3 scales per query row
        g = dS.reshape(b, h, s, s // 16, 16)
        sf = e4m3_round((g.abs().amax(-1, keepdim=True) / 6.0).float()).double()
        dSh = (e2m1_rtn(g.float() * (1.0 / sf.float().clamp(min=1e-30))).double() * sf).reshape(b, h, s, s)
    else:  # 16 x 16 blocks (queries x keys)
        blk = _blocks(dS, 16, 16)
        sf = e4m3_round((blk.abs().amax((3, 5), keepdim=True) / 6.0).float()).double()
        dSh = (e2m1_rtn(blk.float() * (1.0 / sf.float().clamp(min=1e-30))).double() * sf).reshape(b, h, s, s)
    dq_scale = _f64(info["dq_scale"])
    dQ = (dSh @ Kt) * dq_scale
    dQx = (dS @ Kt) * dq_scale
    if r_row is not None:  # dQ row q carries its own row factor (F_q or r_q); undone after the kernel
        rr = r_row.double()[..., 0].permute(0, 2, 1)[..., None]
        dQ, dQx = dQ / rr, dQx / rr

    # ---- dK, dV
    if dkv8:
        mode = mxds_mode(spec)
        if mode == "per_tensor":
            assert not spec.dk_sr, "the dK operand's stochastic rounding is not modelled"
            dS8 = e4m3_round(dS.float()).double()
        else:
            blk = _blocks(dS, 32, 32 if mode == "blk32" else 16)
            am = blk.abs().amax((3, 5), keepdim=True)
            e = torch.ceil(torch.log2(am.clamp(min=1e-30) / 448.0)).clamp(min=-126.0)
            dS8 = (e4m3_round((blk * torch.pow(2.0, -e)).float()).double() * torch.pow(2.0, e)).reshape(b, h, s, s)
        Q8 = info["q8"].float().permute(0, 2, 1, 3).double()     # codes; the per-tensor scale is inside dk_scale
        dO8 = info["do8"].float().permute(0, 2, 1, 3).double()
        dK = (dS8.transpose(-1, -2) @ Q8) * dk_scale
        P8 = e4m3_round(((m if vtok else w) * 448.0).float()).double()
        dV = (P8.transpose(-1, -2) @ dO8) * dv_scale
    else:
        Qt = dequantize_transposed(qt["xt4"], qt["sft"], False).permute(0, 2, 1, 3).double()
        dOt = dequantize_transposed(dq_["xt4"], dq_["sft"], False).permute(0, 2, 1, 3).double()
        dK = (dSh.transpose(-1, -2) @ Qt) * dk_scale
        boost = 2.0 ** 11
        wb = _blocks((m if vtok else w) * boost, 16, 16)  # vtok: the unbiased m (`vtok_dv_u`)
        sfw = e4m3_round((wb.amax((3, 5), keepdim=True) / 6.0).float()).double()
        Pv = (e2m1_rtn(wb.float() * (1.0 / sfw.float().clamp(min=1e-30))).double() * sfw).reshape(b, h, s, s)
        dV = (Pv.transpose(-1, -2) @ dOt) * dv_scale
    dK = dK.reshape(b, hk, r, s, d).sum(2)
    dV = dV.reshape(b, hk, r, s, d).sum(2)
    t = lambda z: z.transpose(1, 2).float()  # noqa: E731
    return dict(dq=t(dQ), dk=t(dK), dv=t(dV), dS=dS, dq_exact=dQx.transpose(1, 2))

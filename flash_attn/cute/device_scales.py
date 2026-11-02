# Per-tensor scales that never leave the GPU.
#
# Every per-tensor scale of the low-precision entries (the E4M3 scales s = amax / 448 of Q, K, V, dO, the dS scale s_ds,
# and the kernel scalars derived from them) is computed and consumed on the device: no .item(), no .tolist(), no host
# float. That makes every entry sync-free and CUDA-graph capturable.
#
# Arithmetic contract (no FP64 anywhere: FP64 runs at 1:64 on B300). Every scalar is a float32 device value, derived
# with IEEE float32 operations that are each correctly rounded -- mul.rn (multiplications only: nothing an FMA
# contraction could fuse), div.rn (tl.math.div_rn: Triton's '/' is div.full, an approximation), sqrt.rn -- in a fixed,
# documented order. A host evaluation of the same formula in numpy float32, op by op in the same order, is therefore
# bitwise the device value (tests/cute/test_device_scales_fp32.py); the values differ from the earlier float64
# derivation (rounded to float32 once) by a few float32 ulps at most. Host constants (softmax_scale, 2^shift) enter as
# float32 (rounded to nearest once, on the host).
#
#   (the per-tensor E4M3 scales themselves: quant.fp8.quantize_fp8_per_tensor, in the quantiser's own launch)
#   dev32(x, device)           -> a float32 device scalar from a Python float or a device tensor (no host read)
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


def _f32(x: float) -> float:
    """x rounded to float32 (nearest even), as a Python float."""
    return float(np.float32(x))


@triton.jit
def _div(a, b):
    """a / b correctly rounded (div.rn.f32; Python constants, scalars and blocks broadcast)."""
    a, b = tl.broadcast(tl.cast(a, tl.float32), tl.cast(b, tl.float32))
    return tl.math.div_rn(a, b)


@triton.jit
def _pow2_floor_clamped(y, lo: tl.constexpr, hi: tl.constexpr):
    """The largest power of two <= y (y >= 0 float32), its exponent clamped to [lo, hi] (-126 < lo, hi < 127): the
    float32 exponent field (y = 0 / subnormal -> lo; inf / NaN -> hi)."""
    e = ((y.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    e = tl.minimum(tl.maximum(e, lo), hi)
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


def amax_abs(x: torch.Tensor) -> torch.Tensor:
    """max |x| as a 0-dim tensor of x's dtype, one pass, exact (bitwise x.abs().amax())."""
    return torch.linalg.vector_norm(x, ord=float("inf"))


def dev32(x, device) -> torch.Tensor:
    """A float32 device scalar (0-dim) from a Python float (a fill launch, no host read; rounded to float32 once) or a
    device tensor (a view; a float64 tensor -- accepted for compatibility -- is converted once, by torch)."""
    if isinstance(x, torch.Tensor):
        return x.reshape(()) if x.dtype == torch.float32 else x.reshape(()).float()
    return torch.full((), float(x), dtype=torch.float32, device=device)


def div32(a: torch.Tensor, c: float) -> torch.Tensor:
    """a / float32(c) in float32 rounded ONCE (div.rn): torch's CUDA `tensor / <Python float>` multiplies by the rounded
    reciprocal (two roundings), so the constant goes in as a 0-dim device tensor (one fill launch)."""
    return a.float() / torch.full((), float(c), dtype=torch.float32, device=a.device)


def rdiv32(c: float, a: torch.Tensor) -> torch.Tensor:
    """float32(c) / a in float32 rounded once (`<Python float> / tensor` is reciprocal(a) * c: two roundings). See div32."""
    return torch.full((), float(c), dtype=torch.float32, device=a.device) / a.float()


def is_dev(x) -> bool:
    return isinstance(x, torch.Tensor)


def f32_view(x: torch.Tensor) -> torch.Tensor:
    """float32 view / copy of a device scalar (a float32 one is returned as is)."""
    return x if x.dtype == torch.float32 else x.float()


def host_or_none(x) -> Optional[float]:
    """For log lines only: a Python float if x is one, else None (never reads a device value)."""
    return None if isinstance(x, torch.Tensor) else x


# ---------------------------------------------------------------------------------------------------------------------
# Backward kernel scalars. One single-program launch per backward replaces the host float arithmetic of the three
# drivers; each kernel is that driver's expressions, transcribed in the same order (see the driver for the meaning).
# Outputs are float32, in the order the kernel's mScalars expects, plus the D fold multiplier the driver applies with a
# torch multiply (which rounds a Python scalar to float32 just the same).
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def _ld(p):
    return tl.load(p).to(tl.float32)


@triton.jit
def _sds_from(sds_p, r_p, sv, sdo, SDS_MODE: tl.constexpr):
    """s_ds: 0 = a float32 value (the caller's, or the bound computed on the device), 1 = max(r, 1e-30) / 448 of a
    float32 device max r (the P_max bound), 2 = s_v s_dO (the uncalibrated control), 3 = 1.0 (block-scaled dS^)."""
    if SDS_MODE == 0:
        s_ds = _ld(sds_p)
    elif SDS_MODE == 1:
        r = _ld(r_p)
        s_ds = _div(tl.maximum(r, 1e-30), 448.0)
    elif SDS_MODE == 2:
        s_ds = sv * sdo
    else:
        s_ds = tl.full([], 1.0, tl.float32)
    return s_ds


@triton.jit
def _max_of(p, n, st):
    """max of n non-negative float32 values at p, stride st (a (1,) slot, or a quantiser's per-tile maxima: b h s / 128
    of them, one 4096-wide load up to s = 32768 at b h = 16)."""
    acc = tl.zeros([4096], tl.float32)
    for i in range(0, n, 4096):
        idx = i + tl.arange(0, 4096)
        acc = tl.maximum(acc, tl.load(p + idx * st, mask=idx < n, other=0.0))
    return tl.max(acc, 0)


@triton.jit
def _dq_bound_pow2(dn2, vn2, kmax, unit):
    """B = the largest power of two with 4 B bound <= 65504, in [2^-40, 2^40], bound = unit 2 sqrt(dn2) sqrt(vn2) kmax:
    float32, ((((4 unit) 2) sqrt.rn(dn2)) sqrt.rn(vn2)) kmax, then 65504 / bound (div.rn) and its exponent field (bound
    = 0 -> inf -> the cap; an intermediate that overflows -> 0 -> the floor; both are where the float64 form clamped
    too). B equals the float64 derivation's except where 65504 / bound lies within a float32 rounding of a power of two
    (then B can be twice it: the 4x headroom of the bound absorbs that)."""
    bnd = 4.0 * unit * 2.0 * tl.sqrt_rn(dn2) * tl.sqrt_rn(vn2) * kmax
    return _pow2_floor_clamped(_div(65504.0, bnd), -40, 40)


@triton.jit
def _dq_prescale(dn_p, vn_p, km_p, dn_n, dn_s, vn_n, vn_s, km_n, km_s, sk, unit, KMAX_FROM_SK: tl.constexpr):
    """`dq-f16-range`: the power-of-two pre-scale of the f16 dQ partials, sized from the data (every f16 dQ route).

    The partials hold B * unit * dS K (unit: the route's fixed factor, e.g. 2^p_shift), and a partial over ANY set of
    keys -- the final reduced value included -- is bounded by unit * 2 max_q ||dO_q|| max_k ||V_k|| max|K|
    (sum_k |dS_qk| <= sum_k P_qk (|dP_qk| + |D_q|) <= 2 ||dO_q|| max ||V||). B is the largest power of two with
    4 B bound <= 65504 (4x: the E4M3 / block rounding of dO, V, K, dS), in [2^-40, 2^40]: no overflow (a fixed 2^6 gave
    NaN dQ on K channels x20-50, the blk32 FP8 control) and no flush of small gradients (true units with no pre-scale
    flushed ~6 % of dQ to 0 at dO ~ 1e-6 on the 2-CTA and dk8 routes). A power of two is exact in f16's normal range, so
    the result moves only where a partial was subnormal or overflowed. Norms from the quantisers' reads (no statistic
    pass): dn_p max ||dO_q||^2, vn_p max ||V_k||^2, km_p max|K| -- float32 slots or the block quantisers' per-tile
    maxima, (count, stride) of each, reduced here; max|K| = 448 s_k for per-tensor E4M3 K (KMAX_FROM_SK; block-scaled
    codes carry no per-tensor scale). See _dq_bound_pow2 for the float32 arithmetic."""
    if KMAX_FROM_SK:
        kmax = 448.0 * sk
    else:
        kmax = _max_of(km_p, km_n, km_s)
    dn2 = _max_of(dn_p, dn_n, dn_s)
    vn2 = _max_of(vn_p, vn_n, vn_s)
    return _dq_bound_pow2(dn2, vn2, kmax, unit)


@triton.jit
def _dq_row_bound(dn_p, vn_p, km_p, bq, hq, hk, tq, tk, qr_lo, qr_hi, k_lo, k_hi, base, num, unit,
                  BLOCK: tl.constexpr):
    """_dq_tile_table's per-row values (kernel scale base B_r, postprocess scale num / (unit B_r)) from the row's query
    tiles [qr_lo, qr_hi) of every q head (dn_p) and key tiles [k_lo, k_hi) of every kv head (vn_p / km_p)."""
    # the row's (head, tile) entries as one flat range j = head * n_tiles + tile (one vector load per BLOCK of them:
    # the per-head loads in sequence were latency-bound)
    va = tl.zeros([BLOCK], tl.float32)
    ka = tl.zeros([BLOCK], tl.float32)
    nk = k_hi - k_lo
    for j0 in range(0, hk * nk, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        m = j < hk * nk
        off = (bq * hk + j // tl.maximum(nk, 1)) * tk + k_lo + j % tl.maximum(nk, 1)
        va = tl.maximum(va, tl.load(vn_p + off, mask=m, other=0.0))
        ka = tl.maximum(ka, tl.load(km_p + off, mask=m, other=0.0))
    da = tl.zeros([BLOCK], tl.float32)
    nq = qr_hi - qr_lo
    for j0 in range(0, hq * nq, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        m = j < hq * nq
        off = (bq * hq + j // tl.maximum(nq, 1)) * tq + qr_lo + j % tl.maximum(nq, 1)
        da = tl.maximum(da, tl.load(dn_p + off, mask=m, other=0.0))
    b = _dq_bound_pow2(tl.max(da, 0), tl.max(va, 0), tl.max(ka, 0), unit)
    kern = base * b
    post = _div(num, unit * b)
    return kern, post


@triton.jit
def _dq_doc_row(dn_p, vn_p, km_p, tab_p, out_p, n_tab, hq, hk, tq, tk, n_ent, base, num, unit, r,
                BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr):
    """One document row r of _dq_tile_table (DOCS): its per-row slot and the query tiles of its slot."""
    q_lo = tl.load(tab_p + r) // 128
    q_lo = tl.where(r == 0, 0, q_lo)
    q_hi = tl.load(tab_p + tl.minimum(r + 1, n_tab - 1)) // 128
    q_hi = tl.where(r == n_tab - 1, tq, q_hi)
    q0 = tl.load(tab_p + r)
    k0 = tl.load(tab_p + n_tab + r)
    kern, post = _dq_row_bound(dn_p, vn_p, km_p, 0, hq, hk, tq, tk, q0 // 128,
                               (q0 + tl.load(tab_p + 2 * n_tab + r) + 127) // 128, k0 // 128,
                               (k0 + tl.load(tab_p + 3 * n_tab + r) + 127) // 128, base, num, unit, ROW_BLOCK)
    tl.store(out_p + 8 + 2 * n_ent + r, kern)
    for t0 in range(q_lo, q_hi, BLOCK):
        idx = t0 + tl.arange(0, BLOCK)
        m = idx < q_hi
        tl.store(out_p + 8 + idx, kern + tl.zeros([BLOCK], tl.float32), mask=m)
        tl.store(out_p + 8 + n_ent + idx, post + tl.zeros([BLOCK], tl.float32), mask=m)


@triton.jit
def _dq_doc_tail(tab_p, n_cap, n_tab, BLOCK: tl.constexpr):
    """The first document row r < n_cap at or past the end of both sides' slots (tab rows 0 / 1 non-decreasing, so
    every row in [r, n_cap) is an empty document whose slot is empty: no tiles of either side). n_cap (every row a
    full one) when an end is not 128-aligned. A BLOCK-ary search: two vector loads at n_cap <= BLOCK^2."""
    e0 = tl.load(tab_p + n_cap)
    e1 = tl.load(tab_p + n_tab + n_cap)
    lo = tl.zeros([], tl.int32)
    n = n_cap + tl.zeros([], tl.int32)
    while n > BLOCK:
        st = (n + BLOCK - 1) // BLOCK
        idx = lo + tl.arange(0, BLOCK) * st
        m = idx < lo + n
        tail = (tl.load(tab_p + idx, mask=m, other=0) >= e0) & (tl.load(tab_p + n_tab + idx, mask=m, other=0) >= e1)
        c = tl.sum((m & ~tail).to(tl.int32), 0)   # probes before the boundary (a prefix: the predicate is monotone)
        new_lo = tl.where(c > 0, lo + (c - 1) * st + 1, lo)
        n = tl.where(c > 0, tl.minimum(st - 1, lo + n - new_lo), 0)
        lo = new_lo
    idx = lo + tl.arange(0, BLOCK)
    m = idx < lo + n
    tail = (tl.load(tab_p + idx, mask=m, other=0) >= e0) & (tl.load(tab_p + n_tab + idx, mask=m, other=0) >= e1)
    r = lo + tl.sum((m & ~tail).to(tl.int32), 0)
    return tl.where(((e0 % 128) == 0) & ((e1 % 128) == 0), r, n_cap)


@triton.jit
def _dq_tile_table(dn_p, vn_p, km_p, tab_p, out_p, n_tab, hq, hk, tq, tk, n_ent, base, num, unit,
                   DOCS: tl.constexpr, BLOCK: tl.constexpr = 1024, ROW_BLOCK: tl.constexpr = 256,
                   CAUSAL: tl.constexpr = 0):
    """`dq-f16-range-per-row` (dqfix): the f16 dQ pre-scale per row -- a document (DOCS: b = 1, doc_tab (6, n_tab)
    int32 = the aligned tables of lp_varlen, the last column the empty sentinel) or a batch row -- instead of ONE per
    call. Per row: _dq_prescale's bound from the row's own max ||dO_q||^2 (dn_p: the dO quantiser's
    (b * hq * tq,) per-128-row-tile maxima, over the row's query tiles and every q head) and max ||V_k||^2 / max|K|
    (vn_p / km_p: (b * hk * tk,), over the row's key tiles and every kv head): nothing outside the row reaches it, so a
    document's (batch row's) dQ never depends on another one's magnitude. Writes, with B_r the row's power of two and
    n_ent = b tq: out[8 + b tq + t] = base B_r (the kernel's dQ scale per query tile, the 2-CTA reduce),
    out[8 + n_ent + b tq + t] = num / (unit B_r) (the postprocess scale per query tile), out[8 + 2 n_ent + r] = base B_r
    (per row: the 1-CTA reduce, once per work tile). Every query tile of the allocation is written once: a document's
    tiles are its SLOT [start_d, start_{d+1}) (the sentinel's up to tq; document 0's from tile 0), so padding tiles get
    a finite scale too (their accumulator is 0).
    Batch rows: one program per row. DOCS: any grid (the host caps it; the capacity N can be T): the programs take
    the documents before the empty tail (_dq_doc_tail, found on the device) round robin; the tail's rows (no tile on
    either side: the zero bound's values) and the sentinel's tiles are vector fills split across the programs.

    CAUSAL (dq-causal, batched causal square rows): one program per (row, query tile) t; the pre-scale of tile t is sized
    from that tile's own max ||dO_q||^2 and the PREFIX of key tiles 0..t (max ||V_k||^2, max|K|) -- the only keys a
    causal query of tile t may see -- so a later key can no longer move an earlier query's dQ pre-scale. A power of
    two of a superset bound of the visible keys: the f16 headroom argument of _dq_prescale holds per tile."""
    if CAUSAL == 2:
        # dq-causal, per ROW i (1-CTA kernels): dn_p / vn_p / km_p are (n_ent,) per-row arrays -- max ||dO_i||^2 over the q
        # heads, the running (prefix) max ||V_j||^2 and max|K_j| over keys j <= i and kv heads -- so row i's pre-scale is
        # a function of keys the query may see only (exactly causal: no key tile granularity). tq = n_ent = b * s_q here.
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < n_ent
        dn = tl.load(dn_p + idx, mask=m, other=0.0)
        vn = tl.load(vn_p + idx, mask=m, other=0.0)
        km = tl.load(km_p + idx, mask=m, other=0.0)
        bnd = 4.0 * unit * 2.0 * tl.sqrt_rn(dn) * tl.sqrt_rn(vn) * km
        bb = _pow2_floor_clamped(_div(65504.0, bnd), -40, 40)
        tl.store(out_p + 8 + idx, base * bb, mask=m)
        tl.store(out_p + 8 + n_ent + idx, _div(num, unit * bb), mask=m)
    elif CAUSAL:
        pid = tl.program_id(0)
        r = pid // tq
        t = pid % tq
        kern, post = _dq_row_bound(dn_p, vn_p, km_p, r, hq, hk, tq, tk, t, t + 1, 0, tl.minimum(t + 1, tk), base, num,
                                   unit, ROW_BLOCK)
        tl.store(out_p + 8 + pid, kern)
        tl.store(out_p + 8 + n_ent + pid, post)
        if t == tq - 1:   # the per-row slot (read only by the non-causal kernels): the last tile's, the full prefix
            tl.store(out_p + 8 + 2 * n_ent + r, kern)
    elif DOCS:
        pid = tl.program_id(0)
        G = tl.num_programs(0)
        n_cap = n_tab - 1
        if n_tab <= G:   # a program per row (the sentinel's too): no tail search, BLOCK-wide reductions (long rows)
            _dq_doc_row(dn_p, vn_p, km_p, tab_p, out_p, n_tab, hq, hk, tq, tk, n_ent, base, num, unit, pid, BLOCK,
                        BLOCK)
        else:   # (ROW_BLOCK-wide reductions: many short rows per program, 1024 -> 256: all-1-token N = T 516 -> 300 us)
            n_hi = _dq_doc_tail(tab_p, n_cap, n_tab, BLOCK)
            for r in range(pid, n_hi, G):
                _dq_doc_row(dn_p, vn_p, km_p, tab_p, out_p, n_tab, hq, hk, tq, tk, n_ent, base, num, unit, r, BLOCK,
                            ROW_BLOCK)
            # the empty tail [n_hi, n_cap): no tiles -> the zero bound's values, no query tile of their (empty) slots
            z = tl.zeros([], tl.int32)
            kz, pz = _dq_row_bound(dn_p, vn_p, km_p, 0, hq, hk, tq, tk, z, z, z, z, base, num, unit, ROW_BLOCK)
            for r0 in range(n_hi + pid * BLOCK, n_cap, G * BLOCK):
                idx = r0 + tl.arange(0, BLOCK)
                tl.store(out_p + 8 + 2 * n_ent + idx, kz + tl.zeros([BLOCK], tl.float32), mask=idx < n_cap)
            # the sentinel row n_cap: its value (one program stores the slot), its slot's query tiles split
            q0 = tl.load(tab_p + n_cap)
            k0 = tl.load(tab_p + n_tab + n_cap)
            ks, ps = _dq_row_bound(dn_p, vn_p, km_p, 0, hq, hk, tq, tk, q0 // 128,
                                   (q0 + tl.load(tab_p + 2 * n_tab + n_cap) + 127) // 128, k0 // 128,
                                   (k0 + tl.load(tab_p + 3 * n_tab + n_cap) + 127) // 128, base, num, unit, ROW_BLOCK)
            if pid == 0:
                tl.store(out_p + 8 + 2 * n_ent + n_cap, ks)
            s_lo = tl.where(n_cap == 0, 0, q0 // 128)
            for t0 in range(s_lo + pid * BLOCK, tq, G * BLOCK):
                idx = t0 + tl.arange(0, BLOCK)
                m = idx < tq
                tl.store(out_p + 8 + idx, ks + tl.zeros([BLOCK], tl.float32), mask=m)
                tl.store(out_p + 8 + n_ent + idx, ps + tl.zeros([BLOCK], tl.float32), mask=m)
    else:
        r = tl.program_id(0)
        kern, post = _dq_row_bound(dn_p, vn_p, km_p, r, hq, hk, tq, tk, 0, tq, 0, tk, base, num, unit, BLOCK)
        tl.store(out_p + 8 + 2 * n_ent + r, kern)
        for t0 in range(0, tq, BLOCK):
            idx = t0 + tl.arange(0, BLOCK)
            m = idx < tq
            tl.store(out_p + 8 + r * tq + idx, kern + tl.zeros([BLOCK], tl.float32), mask=m)
            tl.store(out_p + 8 + n_ent + r * tq + idx, post + tl.zeros([BLOCK], tl.float32), mask=m)


@triton.jit
def _bwd1_fp4_scalars_kernel(sv_p, sdo_p, sq_p, sk_p, sds_p, r_p, out_p, softmax, sh,
                             DP8: tl.constexpr, DV8: tl.constexpr, DQ8: tl.constexpr, DK8: tl.constexpr,
                             SDS_MODE: tl.constexpr, dn_p=None, vn_p=None, km_p=None,
                             dn_n=1, dn_s=1, vn_n=1, vn_s=1, km_n=1, km_s=1, p2=1.0, DQ_RANGE: tl.constexpr = 0,
                             tab_p=None, n_tab=1, t_hq=1, t_hk=1, t_tq=1, t_tk=1, t_n=0, TILES: tl.constexpr = 0,
                             T_DOCS: tl.constexpr = 0, T_CAUSAL: tl.constexpr = 0):
    # interface_bwd_fp4._flash_attn_bwd_fp4 (1-CTA FP4 route): out = [softmax_scale, dpsum_scale, ds_boost, dv_scale,
    # dk_scale, dq_scale, D fold multiplier] (+ [7] the dQ postprocess scale softmax_scale / (2^p_shift B) with
    # DQ_RANGE: 1 = max|K| from the per-tensor s_k, 2 = from km_p; the f16 partials carry the pre-scale B, see
    # _dq_prescale). float32 (softmax, sh = 2^dv8_shift, p2 = 2^p_shift: float32 arguments); the order of _bwd1_fp4_ref.
    softmax = softmax.to(tl.float32)
    sh = sh.to(tl.float32)
    one = tl.full([], 1.0, tl.float32)
    sv = one
    sdo = one
    if DP8:
        sv = _ld(sv_p)
        sdo = _ld(sdo_p)
        dps = _div(1.0, sv * sdo)
        ds_b = sv * sdo
    else:
        dps = one
        ds_b = one
    if DV8:
        dv_s = _div(sdo, sh)
        ds_b = _div(ds_b, sh)
    else:
        dv_s = one
    dk_s = one
    dq_s = one
    if DQ8:
        s_ds = _sds_from(sds_p, r_p, sv, sdo, SDS_MODE)
        ds_b = _div(_div(sv * sdo, sh), s_ds)
        dk_s = s_ds
        dq_s = s_ds * _ld(sk_p)
        if DK8:
            dk_s = softmax * s_ds * _ld(sq_p)
        if DQ_RANGE > 0:
            b = _dq_prescale(dn_p, vn_p, km_p, dn_n, dn_s, vn_n, vn_s, km_n, km_s, _ld(sk_p), p2,
                             KMAX_FROM_SK=DQ_RANGE == 1)
            dq_s = dq_s * b
            tl.store(out_p + 7, _div(softmax, p2 * b))
    if TILES:  # (DQ_RANGE 0: dq_s is s_ds s_k here) the per-tile table, unit 2^p_shift
        _dq_tile_table(dn_p, vn_p, km_p, tab_p, out_p, n_tab, t_hq, t_hk, t_tq, t_tk, t_n, dq_s, softmax,
                       p2.to(tl.float32), T_DOCS, CAUSAL=T_CAUSAL)
    m = -ds_b * dps
    tl.store(out_p + 0, softmax)
    tl.store(out_p + 1, dps)
    tl.store(out_p + 2, ds_b)
    tl.store(out_p + 3, dv_s)
    tl.store(out_p + 4, dk_s)
    tl.store(out_p + 5, dq_s)
    tl.store(out_p + 6, m)


@triton.jit
def _bwd1_fp8_scalars_kernel(sq_p, sk_p, sv_p, sdo_p, sds_p, out_p, softmax, p_shift, dq_boost, BOOST_MODE: tl.constexpr,
                             vn_p=None, dn_p=None, km_p=None, dn_n=1, dn_s=1, vn_n=1, vn_s=1, km_n=1, km_s=1,
                             DQ_RANGE: tl.constexpr = 0, tab_p=None, n_tab=1, t_hq=1, t_hk=1, t_tq=1, t_tk=1,
                             t_n=0, TILES: tl.constexpr = 0, T_DOCS: tl.constexpr = 0,
                             T_CAUSAL: tl.constexpr = 0):
    # interface_bwd_fp4._flash_attn_bwd_fp8_control: out = [softmax_scale * s_q s_k, dpsum_scale, ds_boost / p_shift,
    # dv_scale, dk_scale, dq_scale, D fold multiplier]. BOOST_MODE 0: ds_boost = s_v s_dO / s_ds (s_ds float32 at sds_p);
    # 1: ds_boost = 1 (block-scaled dS^). float32, the order of _bwd1_fp8_ref: the ratio r = (s_v s_dO) / ds_boost first
    # (~ s_ds in mode 0), so no intermediate is a product of more than two scales.
    softmax = softmax.to(tl.float32)
    p_shift = p_shift.to(tl.float32)
    dq_boost = dq_boost.to(tl.float32)
    sq = _ld(sq_p)
    sk = _ld(sk_p)
    sv = _ld(sv_p)
    sdo = _ld(sdo_p)
    if DQ_RANGE > 0:
        # the f16 dQ partials hold dq_boost dQ / softmax_scale: dq_boost is the data-sized pre-scale (_dq_prescale; it
        # replaces the fixed interface_bwd_fp4.FP8_DQ_BOOST_LOG2, which stays the boost of a call without the norms). DQ_RANGE 1:
        # max|K| = 448 s_k (per-tensor K); 2: the K quantiser's amax (block-scaled K, unit descales)
        dq_boost = _dq_prescale(dn_p, vn_p, km_p, dn_n, dn_s, vn_n, vn_s, km_n, km_s, sk, 1.0,
                                KMAX_FROM_SK=DQ_RANGE == 1)
    if BOOST_MODE == 0:
        ds_boost = _div(sv * sdo, _ld(sds_p))
    else:
        ds_boost = tl.full([], 1.0, tl.float32)
    r = _div(sv * sdo, ds_boost)
    if TILES:  # dq_boost per tile: slot 5 = (r s_k) B, exactly (a power of two scales exactly)
        _dq_tile_table(dn_p, vn_p, km_p, tab_p, out_p, n_tab, t_hq, t_hk, t_tq, t_tk, t_n, r * sk,
                       softmax, tl.full([], 1.0, tl.float32), T_DOCS, CAUSAL=T_CAUSAL)
    m = -_div(_div(ds_boost, p_shift), sv * sdo)
    tl.store(out_p + 0, softmax * sq * sk)
    tl.store(out_p + 1, _div(1.0, sv * sdo))
    tl.store(out_p + 2, _div(ds_boost, p_shift))
    tl.store(out_p + 3, _div(sdo, p_shift))
    tl.store(out_p + 4, (softmax * sq) * r)
    tl.store(out_p + 5, (r * sk) * dq_boost)
    tl.store(out_p + 6, m)
    tl.store(out_p + 7, _div(softmax, dq_boost))  # the dQ postprocess scale


@triton.jit
def _bwd2_scalars_kernel(sq_p, sk_p, sv_p, sdo_p, sds_p, r_p, out_p, softmax, p2,
                         SDS_MODE: tl.constexpr, NVFP4_S: tl.constexpr, DQ_F16: tl.constexpr,
                         dn_p=None, vn_p=None, km_p=None, dn_n=1, dn_s=1, vn_n=1, vn_s=1, km_n=1, km_s=1,
                         DQ_RANGE: tl.constexpr = 0, tab_p=None, n_tab=1, t_hq=1, t_hk=1, t_tq=1, t_tk=1,
                         t_n=0, TILES: tl.constexpr = 0, T_DOCS: tl.constexpr = 0,
                             T_CAUSAL: tl.constexpr = 0):
    # interface_bwd_2cta8.flash_attn_bwd_2cta8: out = [softmax_scale_k, ds_boost, dv_scale, dk_scale, dq kernel scale,
    # D fold multiplier, dq_scale (fp32 postprocess)] (+ [7] the f16 dQ postprocess scale softmax_scale / B with
    # DQ_RANGE, see _dq_prescale: the f16 partials were in true units with no pre-scale). float32, the order of _bwd2_ref.
    softmax = softmax.to(tl.float32)
    p2 = p2.to(tl.float32)  # 2 ** P_SHIFT_LOG2
    sq = _ld(sq_p)
    sk = _ld(sk_p)
    sv = _ld(sv_p)
    sdo = _ld(sdo_p)
    s_ds = _sds_from(sds_p, r_p, sv, sdo, SDS_MODE)
    ds_b = _div(_div(sv * sdo, p2), s_ds)
    m = -_div(ds_b, sv * sdo)
    if NVFP4_S:
        ssk = softmax
    else:
        ssk = softmax * sq * sk
    if DQ_F16:
        dq_kern = s_ds * sk
        if DQ_RANGE > 0:
            b = _dq_prescale(dn_p, vn_p, km_p, dn_n, dn_s, vn_n, vn_s, km_n, km_s, sk, 1.0, KMAX_FROM_SK=DQ_RANGE == 1)
            dq_kern = dq_kern * b
            tl.store(out_p + 7, _div(softmax, b))
        if TILES:
            _dq_tile_table(dn_p, vn_p, km_p, tab_p, out_p, n_tab, t_hq, t_hk, t_tq, t_tk, t_n, s_ds * sk, softmax,
                           tl.full([], 1.0, tl.float32), T_DOCS, CAUSAL=T_CAUSAL)
    else:
        dq_kern = tl.full([], 1.0, tl.float32)
    tl.store(out_p + 0, ssk)
    tl.store(out_p + 1, ds_b)
    tl.store(out_p + 2, _div(sdo, p2))
    tl.store(out_p + 3, softmax * s_ds * sq)
    tl.store(out_p + 4, dq_kern)
    tl.store(out_p + 5, m)
    tl.store(out_p + 6, softmax * s_ds * sk)


def _ptr(x, dummy):
    return dummy if x is None else x


# ---------------------------------------------------------------------------------------------------------------------
# Constant scalars without launches. The token-block routes' operand descales are all 1, and with the
# block-scaled dS^ (sds_mode 3) or the uncalibrated control (2) every kernel scalar is then a function of host constants:
# the drivers take the unit descales as views of ONE per-device constant and the scalars come from a per-key cache of
# the kernels' float32 expressions evaluated on the host in numpy float32 (the same correctly rounded operations in the
# same order: bitwise what the kernel computes from unit descales). The first call of a key uploads its (7,) / (8,)
# vector; every later call is launch-free.
# ---------------------------------------------------------------------------------------------------------------------
_UNIT = {}
DQ_TAB_GRID = 1024  # the documents' grid cap of _dq_tile_table (N = T, T = 131072: 1024 -> 25 us, 4096 -> 50 us)
_CONST_SCALARS = {}
_F = np.float32


def unit_descales(device) -> torch.Tensor:
    """(4,) float32 ones on `device`, allocated once: the token-block drivers' unit descales (views of it)."""
    key = str(torch.device(device))
    u = _UNIT.get(key)
    if u is None:
        u = _UNIT[key] = torch.ones(4, dtype=torch.float32, device=device)
    return u


def _is_unit(t) -> bool:
    """t is a view of the unit-descale constant (so its value is 1.0 without reading it)."""
    if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.numel() != 1:
        return False
    u = _UNIT.get(str(t.device))
    if u is None:
        return False
    base = u.data_ptr()
    return base <= t.data_ptr() < base + 4 * 4


def _const_vector(key, values, device) -> torch.Tensor:
    k = (str(torch.device(device)),) + key
    t = _CONST_SCALARS.get(k)
    if t is None:
        t = _CONST_SCALARS[k] = torch.tensor([_f32(v) for v in values], dtype=torch.float32, device=device)
        t.host_values = tuple(_f32(v) for v in values)
    return t


def apply_d_fold(dpsum: torch.Tensor, scal: torch.Tensor, index: int, folded) -> None:
    """The drivers' D fold D'' = scal[index] D. folded: the multiplier the dO launch already applied (mx2d d_mul), or None.
    A folded D is accepted only when this call's scalars are host constants with exactly that multiplier (otherwise a
    loud error: the dO launch and the driver disagree); a launch-free no-op then."""
    if folded is None:
        dpsum.mul_(scal[index])
        return
    hv = getattr(scal, "host_values", None)
    if hv is None or hv[index] != _f32(folded):
        raise RuntimeError(f"D was folded by the dO launch with {folded!r}, but this backward's D multiplier is "
                           f"{'a device value' if hv is None else hv[index]!r}")


def _sds_ref(sds_mode, sv, sdo, s_ds=None, r=None):
    """_sds_from in numpy float32."""
    if sds_mode == 0:
        return _F(s_ds)
    if sds_mode == 1:
        return np.maximum(_F(r), _F(1e-30)) / _F(448.0)
    if sds_mode == 2:
        return _F(sv) * _F(sdo)
    return _F(1.0)


def _bwd1_fp4_ref(softmax, sh, dp8, dv8, dq8, dk8, sds_mode, sv=1.0, sdo=1.0, sq=1.0, sk=1.0, s_ds=None, r=None):
    """_bwd1_fp4_scalars_kernel's slots 0-6 in numpy float32 (same operations, same order); descales default to 1."""
    softmax, sh, one = _F(softmax), _F(sh), _F(1.0)
    sv_, sdo_ = (_F(sv), _F(sdo)) if dp8 else (one, one)
    if dp8:
        dps = one / (sv_ * sdo_)
        ds_b = sv_ * sdo_
    else:
        dps = ds_b = one
    if dv8:
        dv_s = sdo_ / sh
        ds_b = ds_b / sh
    else:
        dv_s = one
    dk_s = dq_s = one
    if dq8:
        s_ds_ = _sds_ref(sds_mode, sv_, sdo_, s_ds, r)
        ds_b = ((sv_ * sdo_) / sh) / s_ds_
        dk_s = s_ds_
        dq_s = s_ds_ * _F(sk)
        if dk8:
            dk_s = (softmax * s_ds_) * _F(sq)
    m = -ds_b * dps
    return (softmax, dps, ds_b, dv_s, dk_s, dq_s, m)


def _bwd2_ref(softmax, p2, nvfp4_s, dq_f16, sds_mode, sq=1.0, sk=1.0, sv=1.0, sdo=1.0, s_ds=None, r=None):
    """_bwd2_scalars_kernel's slots 0-6 in numpy float32 (no dQ range)."""
    softmax, p2 = _F(softmax), _F(p2)
    sq, sk, sv, sdo = _F(sq), _F(sk), _F(sv), _F(sdo)
    s_ds_ = _sds_ref(sds_mode, sv, sdo, s_ds, r)
    ds_b = ((sv * sdo) / p2) / s_ds_
    m = -(ds_b / (sv * sdo))
    ssk = softmax if nvfp4_s else (softmax * sq) * sk
    dq_kern = s_ds_ * sk if dq_f16 else _F(1.0)
    return (ssk, ds_b, sdo / p2, (softmax * s_ds_) * sq, dq_kern, m, (softmax * s_ds_) * sk)


def _host_sds(sds_mode, sv, sdo):
    return sv * sdo if sds_mode == 2 else 1.0


def _bwd1_fp4_host(softmax, sh, dp8, dv8, dq8, dk8, sds_mode):
    """_bwd1_fp4_scalars_kernel with every descale 1 (numpy float32, the kernel's order)."""
    return tuple(float(x) for x in _bwd1_fp4_ref(softmax, sh, dp8, dv8, dq8, dk8, sds_mode))


def _bwd2_host(softmax, p2, nvfp4_s, dq_f16, sds_mode):
    """_bwd2_scalars_kernel with every descale 1 (numpy float32)."""
    return tuple(float(x) for x in _bwd2_ref(softmax, p2, nvfp4_s, dq_f16, sds_mode))


def _dq_range_args(dq_range, out) -> dict:
    """dq_range: None, or (dn2, vn2, kmax): float32 device tensors, a (1,) slot or a block quantiser's per-tile maxima
    (1-D, any stride), kmax None for per-tensor K (max|K| = 448 s_k) -> the scalar kernels' keyword arguments."""
    if dq_range is None:
        return dict(dn_p=out, vn_p=out, km_p=out, DQ_RANGE=0)
    dn2, vn2, kmax = dq_range
    km = vn2 if kmax is None else kmax
    kw = dict(dn_p=dn2, vn_p=vn2, km_p=km, DQ_RANGE=1 if kmax is None else 2)
    for nm, x in (("dn", dn2), ("vn", vn2), ("km", km)):
        kw[nm + "_n"], kw[nm + "_s"] = x.numel(), (x.stride(-1) if x.dim() else 1)
    return kw


@triton.jit
def _row_stats_kernel(x_p, n2_p, am_p, S, H, sb, ss, sh, D: tl.constexpr, ROWS: tl.constexpr, WANT_AM: tl.constexpr):
    """dq-causal: per row (b, s) of x (b, S, H, D) -- max over the heads of sum_d x^2 (n2_p, (b * S,)) and, with WANT_AM,
    of max_d |x| (am_p). One program per ROWS rows; float32 arithmetic of the bf16 values."""
    pid = tl.program_id(0)
    nblk = (S + ROWS - 1) // ROWS
    b = pid // nblk
    rows = (pid % nblk) * ROWS + tl.arange(0, ROWS)
    m = rows < S
    d = tl.arange(0, D)
    n2 = tl.zeros([ROWS], tl.float32)
    am = tl.zeros([ROWS], tl.float32)
    for h in range(0, H):
        x = tl.load(x_p + b.to(tl.int64) * sb + rows[:, None].to(tl.int64) * ss + h * sh + d[None, :], mask=m[:, None],
                    other=0.0).to(tl.float32)
        n2 = tl.maximum(n2, tl.sum(x * x, 1))
        if WANT_AM:
            am = tl.maximum(am, tl.max(tl.abs(x), 1))
    tl.store(n2_p + b * S + rows, n2, mask=m)
    if WANT_AM:
        tl.store(am_p + b * S + rows, am, mask=m)


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _prefix_max_kernel(a_p, S, BLOCK: tl.constexpr):
    """dq-causal: in-place running maximum along s of each (S,) row of a (rows, S) float32 array (values >= 0); one
    program per row, a carry between BLOCK-wide chunks."""
    base = a_p + tl.program_id(0).to(tl.int64) * S
    carry = tl.zeros([], tl.float32)
    for i in range(0, S, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        x = tl.load(base + idx, mask=idx < S, other=0.0)
        y = tl.maximum(tl.associative_scan(x, 0, _max_combine), carry)
        tl.store(base + idx, y, mask=idx < S)
        carry = tl.max(y, 0)


def row_stats(x: torch.Tensor, want_amax: bool = False):
    """dq-causal: per row (b, s) of x (b, S, H, 128) -- (max over heads of sum_d x^2 (b*S,) float32, [max over heads and d of
    |x| (b*S,)]). One read of x, no host sync."""
    b, s, h, d = x.shape
    assert d == 128
    n2 = torch.empty(b * s, dtype=torch.float32, device=x.device)
    am = torch.empty(b * s, dtype=torch.float32, device=x.device) if want_amax else n2
    rows = 16
    _row_stats_kernel[(b * triton.cdiv(s, rows),)](x, n2, am, s, h, x.stride(0), x.stride(1), x.stride(2), D=d, ROWS=rows,
                                                   WANT_AM=want_amax, num_warps=4)
    return (n2, am) if want_amax else n2


def prefix_max_(a: torch.Tensor, s: int) -> torch.Tensor:
    """In-place running max along the last dim of the (rows, s) float32 array a (one launch)."""
    _prefix_max_kernel[(a.shape[0],)](a, s, BLOCK=1024, num_warps=4)
    return a


def _dq_tiles_args(dq_tiles, dq_range, out) -> dict:
    """dq_tiles (interface_bwd_fp4.dq_tiles_of): the per-(row, q head, query tile) f16 dQ pre-scale table -> the scalar
    kernels' keyword arguments (DQ_RANGE 0: no global pre-scale; the table replaces it) and their grid."""
    dn2, vn2, kmax = dq_range
    t = dq_tiles
    if t.get("rng") is not None:   # dq-causal rows: the per-row statistic arrays replace the quantisers' per-tile maxima
        dn2, vn2, kmax = t["rng"]
    tab = t.get("tab")
    kw = dict(dn_p=dn2, vn_p=vn2, km_p=kmax, DQ_RANGE=0, tab_p=out if tab is None else tab,
              n_tab=1 if tab is None else int(tab.shape[1]), t_hq=t["hq"], t_hk=t["hk"], t_tq=t["tq"], t_tk=t["tk"],
              t_n=t["n"], TILES=1, T_DOCS=int(tab is not None), T_CAUSAL=(2 if t.get("rowwise") else int(bool(t.get("causal")))))
    if t.get("rowwise"):   # dq-causal rows: one program per 1024 rows of (b * s_q)
        return kw, ((t["n"] + 1023) // 1024,)
    if t.get("causal"):   # dq-causal: one program per (row, query tile)
        return kw, (t["rows"] * t["tq"],)
    # documents: a capped grid (the kernel splits the document rows / fills; one program per document row of a capacity
    # N = T launched T programs, ~740 us at T = 131072); batch rows: one program each
    return kw, ((t["rows"] if tab is None else min(t["rows"], DQ_TAB_GRID)),)


def dq_tile_planes(scal: torch.Tensor, dq_tiles) -> torch.Tensor:
    """The postprocess plane of a per-row scalar vector: (b tq,) float32, index b tq + t (128-row query tiles)."""
    n = dq_tiles["n"]
    return scal[8 + n:8 + 2 * n]


def _known_slots(out: torch.Tensor, vals, device_slots) -> torch.Tensor:
    """A launched scalar vector whose other slots are the host constants vals (the unit-descale case): out.host_values
    holds them (None at the data-sized pre-scale's slots), so apply_d_fold checks a D folded by the dO launch exactly as
    on the launch-free path."""
    hv = [_f32(v) for v in vals] + [None] * (out.numel() - len(vals))
    for i in device_slots:
        hv[i] = None
    out.host_values = tuple(hv)
    return out


def bwd1_fp4_scalars(device, softmax_scale: float, dv8_shift: int, dp8: bool, dv8: bool, dq8: bool, dk8: bool,
                     sv=None, sdo=None, sq=None, sk=None, s_ds=None, r=None, sds_mode: int = 0, dq_range=None,
                     p_shift: int = 0, dq_tiles=None) -> torch.Tensor:
    """dq_range (dq8 with f16 partials): (dn2, vn2, kmax) -> the data-sized dQ pre-scale and out[7], the postprocess
    scale softmax_scale / (2^p_shift B) -- a launch even where the descales are the unit constants (the range is data)."""
    used = ([sv, sdo] if dp8 else []) + ([sk] if dq8 else []) + ([sq] if (dq8 and dk8) else [])
    unit = s_ds is None and r is None and (sds_mode in (2, 3) or not dq8) and all(_is_unit(t) for t in used)
    dq_range = dq_range if dq8 else None
    if dq_tiles is not None and dq_range is not None:
        # dqfix: the per-tile pre-scale table after the (8,) vector (out[7] unused: the postprocess reads the table)
        out = torch.empty(8 + 2 * dq_tiles["n"] + dq_tiles["rows"], dtype=torch.float32, device=device)
        d = out
        tk, grid = _dq_tiles_args(dq_tiles, dq_range, out)
        _bwd1_fp4_scalars_kernel[grid](_ptr(sv, d), _ptr(sdo, d), _ptr(sq, d), _ptr(sk, d), _ptr(s_ds, d), _ptr(r, d),
                                       out, _f32(softmax_scale), _f32(2.0 ** dv8_shift),
                                       DP8=dp8, DV8=dv8, DQ8=dq8, DK8=dk8, SDS_MODE=sds_mode,
                                       p2=_f32(2.0 ** p_shift), num_warps=4, **tk)
        if unit:
            return _known_slots(out, _bwd1_fp4_host(float(softmax_scale), float(2 ** dv8_shift), dp8, dv8, dq8, dk8,
                                                    sds_mode), ())
        return out
    if unit and dq_range is None:
        vals = _bwd1_fp4_host(float(softmax_scale), float(2 ** dv8_shift), dp8, dv8, dq8, dk8, sds_mode)
        return _const_vector(("bwd1_fp4", float(softmax_scale), dv8_shift, dp8, dv8, dq8, dk8, sds_mode), vals, device)
    out = torch.empty(8 if dq_range is not None else 7, dtype=torch.float32, device=device)
    d = out
    rk = _dq_range_args(dq_range if dq8 else None, out)
    _bwd1_fp4_scalars_kernel[(1,)](_ptr(sv, d), _ptr(sdo, d), _ptr(sq, d), _ptr(sk, d), _ptr(s_ds, d), _ptr(r, d), out,
                                   _f32(softmax_scale), _f32(2.0 ** dv8_shift),
                                   DP8=dp8, DV8=dv8, DQ8=dq8, DK8=dk8, SDS_MODE=sds_mode,
                                   p2=_f32(2.0 ** p_shift), num_warps=8 if rk["DQ_RANGE"] else 1, **rk)
    if unit:  # the data-sized pre-scale's slots (5, 7) are device values; the others are the host constants
        return _known_slots(out, _bwd1_fp4_host(float(softmax_scale), float(2 ** dv8_shift), dp8, dv8, dq8, dk8,
                                                sds_mode), (5, 7))
    return out


def _bwd1_fp8_ref(softmax, p_shift, dq_boost, sq=1.0, sk=1.0, sv=1.0, sdo=1.0, s_ds=None):
    """_bwd1_fp8_scalars_kernel's slots 0-7 in numpy float32 (same operations, same order; no dynamic boost);
    s_ds None = BOOST_MODE 1 (block-scaled dS^)."""
    softmax, p_shift, dq_boost = _F(softmax), _F(p_shift), _F(dq_boost)
    sq, sk, sv, sdo = _F(sq), _F(sk), _F(sv), _F(sdo)
    ds_boost = _F(1.0) if s_ds is None else (sv * sdo) / _F(s_ds)
    r = (sv * sdo) / ds_boost
    m = -((ds_boost / p_shift) / (sv * sdo))
    return ((softmax * sq) * sk, _F(1.0) / (sv * sdo), ds_boost / p_shift, sdo / p_shift,
            (softmax * sq) * r, (r * sk) * dq_boost, m, softmax / dq_boost)


def _bwd1_fp8_host(softmax, p_shift, dq_boost):
    """_bwd1_fp8_scalars_kernel with every descale 1, BOOST_MODE 1 (block-scaled dS^), no dynamic boost (float32)."""
    return tuple(float(x) for x in _bwd1_fp8_ref(softmax, p_shift, dq_boost))


def bwd1_fp8_scalars(device, softmax_scale: float, p_shift: float, dq_boost: float, sq, sk, sv, sdo, s_ds=None,
                     dq_range=None, dq_tiles=None) -> torch.Tensor:
    """out[0:7] as _bwd1_fp8_scalars_kernel documents, out[7] = softmax_scale / dq_boost (the dQ postprocess scale). With
    dq_range = (dn2, vn2, kmax) (float32 device slots: max ||dO_q||^2, max ||V_k||^2, max|K| or None for 448 s_k) the
    boost is the data-sized pre-scale (_dq_prescale) instead of the fixed dq_boost."""
    unit = s_ds is None and all(_is_unit(t) for t in (sq, sk, sv, sdo))
    if dq_tiles is not None and dq_range is not None:
        # dqfix: slots 5 / 7 keep the fixed dq_boost (unused by the f16 kernel and postprocess, which read the table)
        out = torch.empty(8 + 2 * dq_tiles["n"] + dq_tiles["rows"], dtype=torch.float32, device=device)
        tk, grid = _dq_tiles_args(dq_tiles, dq_range, out)
        _bwd1_fp8_scalars_kernel[grid](sq, sk, sv, sdo, _ptr(s_ds, out), out, _f32(softmax_scale),
                                       _f32(p_shift), _f32(dq_boost), BOOST_MODE=0 if s_ds is not None else 1,
                                       num_warps=4, **tk)
        if unit:
            return _known_slots(out, _bwd1_fp8_host(float(softmax_scale), float(p_shift), float(dq_boost)), ())
        return out
    if unit and dq_range is None:
        vals = _bwd1_fp8_host(float(softmax_scale), float(p_shift), float(dq_boost))
        return _const_vector(("bwd1_fp8", float(softmax_scale), float(p_shift), float(dq_boost)), vals, device)
    out = torch.empty(8, dtype=torch.float32, device=device)
    _bwd1_fp8_scalars_kernel[(1,)](sq, sk, sv, sdo, _ptr(s_ds, out), out, _f32(softmax_scale), _f32(p_shift),
                                   _f32(dq_boost), BOOST_MODE=0 if s_ds is not None else 1,
                                   num_warps=8 if dq_range is not None else 1, **_dq_range_args(dq_range, out))
    if unit:  # slots 5 (dq_scale) and 7 (the postprocess scale) carry the data-sized pre-scale
        return _known_slots(out, _bwd1_fp8_host(float(softmax_scale), float(p_shift), float(dq_boost)), (5, 7))
    return out


@triton.jit
def _doc_prescale(sk_p, sv_p, sdo_p, row, hk, m, rd, unit):
    """dqfix, per-(document, kv head) per-tensor operands: _dq_prescale's power of two per DOCUMENT (row, a vector),
    from that document's own scales only (|x| <= 448 s for E4M3 codes, so ||x|| <= sqrt(d) 448 s), the largest over its
    kv heads: max ||dO_q|| <= rd 448 s_dO, max ||V_k|| <= rd 448 s_V, max|K| <= 448 s_K (rd = float32(sqrt(d))). Looser
    than the row norms, but no other document reaches it. float32, the order of _doc_prescale_ref."""
    bnd = tl.zeros(row.shape, tl.float32)
    for kh in range(0, hk):
        e = row * hk + kh
        sk = tl.load(sk_p + e, mask=m, other=0.0).to(tl.float32)
        sv = tl.load(sv_p + e, mask=m, other=0.0).to(tl.float32)
        sdo = tl.load(sdo_p + e, mask=m, other=0.0).to(tl.float32)
        bnd = tl.maximum(bnd, 4.0 * unit * 2.0 * (rd * 448.0 * sdo) * (rd * 448.0 * sv) * (448.0 * sk))
    return _pow2_floor_clamped(_div(65504.0, bnd), -40, 40)


@triton.jit
def _doc_post_plane(sk_p, sv_p, sdo_p, post_p, tdq_p, n_ent, hk, tq, rd, num, unit, PT_DOCS: tl.constexpr,
                    BLOCK: tl.constexpr):
    """dqfix: the postprocess plane (b tq,) of the per-document pre-scale: query tile (b, t) -> its row (document
    tile_doc_q[t], or batch row b); num / (unit B) of that row's _doc_prescale."""
    for i0 in range(0, n_ent, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        m = idx < n_ent
        if PT_DOCS:
            row = tl.load(tdq_p + idx, mask=m, other=0)
        else:
            row = idx // tq
        b = _doc_prescale(sk_p, sv_p, sdo_p, row, hk, m, rd, unit)
        tl.store(post_p + idx, _div(num + tl.zeros([BLOCK], tl.float32), unit * b), mask=m)


@triton.jit
def _bwd1_fp8_doc_scalars_kernel(sq_p, sk_p, sv_p, sdo_p, out_p, post_p, n, softmax, p_shift, dq_boost, skg_p,
                                 vn_p=None, dn_p=None, dn_n=1, dn_s=1, vn_n=1, vn_s=1, DQ_RANGE: tl.constexpr = 0,
                                 BLOCK: tl.constexpr = 1024, tdq_p=None, rd=1.0, n_ent=0, t_hq=1, t_hk=1, t_tq=1,
                                 PT_DOCS: tl.constexpr = 0):
    # (dqfix) DQ_RANGE 2: the pre-scale per (document, kv head) from that document's own scales (_doc_prescale), and
    # post_p the (b tq,) postprocess plane (_doc_post_plane)
    # _bwd1_fp8_scalars_kernel (BOOST_MODE 1: the block-scaled dS^) for every (document, kv head) of the per-document
    # scale tables (n entries): out (n, 8) rows = [softmax_scale s_q s_k, dpsum_scale, ds_boost / p_shift, dv_scale,
    # dk_scale, dq_scale, D fold multiplier, -]; post_p (1,) = the dQ postprocess scale softmax_scale / dq_boost. The f16
    # dQ pre-scale is ONE data-sized power of two for all documents (_dq_prescale with max|K| = 448 s_k of the global K
    # scale -- the max over the documents -- and the global row norms), so the postprocess scale stays one number.
    # float32, _bwd1_fp8_scalars_kernel's order (ds_boost = 1: r = s_v s_dO).
    softmax = softmax.to(tl.float32)
    p_shift = p_shift.to(tl.float32)
    dq_boost = dq_boost.to(tl.float32)
    if DQ_RANGE == 1:
        dq_boost = _dq_prescale(dn_p, vn_p, vn_p, dn_n, dn_s, vn_n, vn_s, 1, 1, _ld(skg_p), 1.0, KMAX_FROM_SK=True)
    if DQ_RANGE == 2:
        _doc_post_plane(sk_p, sv_p, sdo_p, post_p, tdq_p, n_ent, t_hk, t_tq, rd, softmax, 1.0,
                        PT_DOCS, BLOCK)
    else:
        tl.store(post_p, _div(softmax, dq_boost))
    for i0 in range(0, n, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        m = idx < n
        sq = tl.load(sq_p + idx, mask=m, other=1.0).to(tl.float32)
        sk = tl.load(sk_p + idx, mask=m, other=1.0).to(tl.float32)
        sv = tl.load(sv_p + idx, mask=m, other=1.0).to(tl.float32)
        sdo = tl.load(sdo_p + idx, mask=m, other=1.0).to(tl.float32)
        dqb_e = dq_boost + tl.zeros([BLOCK], tl.float32)
        if DQ_RANGE == 2:
            dqb_e = _doc_prescale(sk_p, sv_p, sdo_p, idx // t_hk, t_hk, m, rd, 1.0)
        r = sv * sdo   # (s_v s_dO) / ds_boost, ds_boost = 1
        ds_p = _div(1.0 + tl.zeros([BLOCK], tl.float32), p_shift + tl.zeros([BLOCK], tl.float32))
        mm = -_div(ds_p, sv * sdo)
        o = out_p + idx * 8
        tl.store(o + 0, softmax * sq * sk, mask=m)
        tl.store(o + 1, _div(1.0 + tl.zeros([BLOCK], tl.float32), sv * sdo), mask=m)
        tl.store(o + 2, ds_p, mask=m)
        tl.store(o + 3, _div(sdo, p_shift + tl.zeros([BLOCK], tl.float32)), mask=m)
        tl.store(o + 4, (softmax * sq) * r, mask=m)
        tl.store(o + 5, (r * sk) * dqb_e, mask=m)
        tl.store(o + 6, mm, mask=m)
        tl.store(o + 7, _div(softmax + tl.zeros([BLOCK], tl.float32), dqb_e), mask=m)


@triton.jit
def _bwd1_fp4_doc_scalars_kernel(sq_p, sk_p, sv_p, sdo_p, out_p, post_p, n, softmax, sh, p2, skg_p,
                                 vn_p=None, dn_p=None, dn_n=1, dn_s=1, vn_n=1, vn_s=1, DQ_RANGE: tl.constexpr = 0,
                                 BLOCK: tl.constexpr = 1024, tdq_p=None, rd=1.0, n_ent=0, t_hq=1, t_hk=1, t_tq=1,
                                 PT_DOCS: tl.constexpr = 0):
    # (dqfix) DQ_RANGE 2: as _bwd1_fp8_doc_scalars_kernel's, unit 2^p_shift
    # _bwd1_fp4_scalars_kernel with DP8 = DV8 = DQ8 = DK8 and SDS_MODE 3 (the block-scaled dS^: s_ds = 1) for every
    # (document, kv head): out (n, 8) rows = [softmax_scale, dpsum_scale, ds_boost, dv_scale, dk_scale, dq_scale, D fold
    # multiplier, -]; post_p (1,) = softmax_scale / (2^p_shift B), B one data-sized pre-scale for all documents
    # (max|K| = 448 s_k of the global K scale). float32, _bwd1_fp4_scalars_kernel's order.
    softmax = softmax.to(tl.float32)
    sh = sh.to(tl.float32)
    p2 = p2.to(tl.float32)
    b = tl.full([], 1.0, tl.float32)
    if DQ_RANGE == 1:
        b = _dq_prescale(dn_p, vn_p, vn_p, dn_n, dn_s, vn_n, vn_s, 1, 1, _ld(skg_p), p2, KMAX_FROM_SK=True)
        tl.store(post_p, _div(softmax, p2 * b))
    if DQ_RANGE == 2:
        _doc_post_plane(sk_p, sv_p, sdo_p, post_p, tdq_p, n_ent, t_hk, t_tq, rd, softmax, p2,
                        PT_DOCS, BLOCK)
    for i0 in range(0, n, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        m = idx < n
        sq = tl.load(sq_p + idx, mask=m, other=1.0).to(tl.float32)
        sk = tl.load(sk_p + idx, mask=m, other=1.0).to(tl.float32)
        sv = tl.load(sv_p + idx, mask=m, other=1.0).to(tl.float32)
        sdo = tl.load(sdo_p + idx, mask=m, other=1.0).to(tl.float32)
        shv = sh + tl.zeros([BLOCK], tl.float32)
        dps = _div(1.0 + tl.zeros([BLOCK], tl.float32), sv * sdo)
        dv_s = _div(sdo, shv)
        ds_b = _div(sv * sdo, shv)   # (s_v s_dO / sh) / s_ds, s_ds = 1: the division by 1 is exact
        dk_s = softmax * sq          # (softmax s_ds) s_q, s_ds = 1
        dq_s = sk                    # s_ds s_k
        if DQ_RANGE == 1:
            dq_s = dq_s * b
        if DQ_RANGE == 2:
            dq_s = dq_s * _doc_prescale(sk_p, sv_p, sdo_p, idx // t_hk, t_hk, m, rd, p2)
        mm = -ds_b * dps
        o = out_p + idx * 8
        tl.store(o + 0, softmax + tl.zeros([BLOCK], tl.float32), mask=m)
        tl.store(o + 1, dps, mask=m)
        tl.store(o + 2, ds_b, mask=m)
        tl.store(o + 3, dv_s, mask=m)
        tl.store(o + 4, dk_s, mask=m)
        tl.store(o + 5, dq_s, mask=m)
        tl.store(o + 6, mm, mask=m)
        tl.store(o + 7, tl.zeros([BLOCK], tl.float32), mask=m)


def _doc_tiles_kw(dq_docs, sq) -> dict:
    """dq_docs (dqfix, interface_bwd_fp4.dq_docs_of): the per-document pre-scale's arguments (DQ_RANGE 2)."""
    t = dq_docs
    tdq = t.get("tile_doc_q")
    return dict(DQ_RANGE=2, tdq_p=sq if tdq is None else tdq, rd=_f32(float(t["d"]) ** 0.5), n_ent=t["n"],
                t_hq=t["hq"],
                t_hk=int(sq.shape[1]), t_tq=t["tq"], PT_DOCS=int(tdq is not None))


def bwd1_fp4_doc_scalars(softmax_scale: float, dv8_shift: int, p_shift: int, sq, sk, sv, sdo, sk_global, dq_range=None,
                         dq_docs=None):
    """Per-(document, kv head) twin of bwd1_fp4_scalars for the shipped dk8 chain (dp8 / dv8 / dq8 / dk8, the
    block-scaled dS^): the (n_rows, h_kv, 8) float32 table the 1-CTA kernel reads per work tile (doc_scaled) and the (1,)
    dQ postprocess scale (softmax_scale / 2^p_shift without the f16 range). One launch."""
    nr, hk = sq.shape
    out = torch.empty(nr, hk, 8, dtype=torch.float32, device=sq.device)
    post = torch.empty(1, dtype=torch.float32, device=sq.device)
    kw = dict(DQ_RANGE=0, vn_p=post, dn_p=post)
    if dq_range is not None:
        dn2, vn2, _kmax = dq_range
        kw = dict(DQ_RANGE=1, vn_p=vn2, dn_p=dn2, dn_n=dn2.numel(), dn_s=(dn2.stride(-1) if dn2.dim() else 1),
                  vn_n=vn2.numel(), vn_s=(vn2.stride(-1) if vn2.dim() else 1))
    else:
        post.fill_(float(_F(softmax_scale) / _F(2.0 ** p_shift)))
    if dq_docs is not None and dq_range is not None:
        # dqfix: the pre-scale per (document, kv head); post becomes the (b hq tq,) postprocess plane
        post = torch.empty(dq_docs["n"], dtype=torch.float32, device=sq.device)
        kw = dict(_doc_tiles_kw(dq_docs, sq), vn_p=post, dn_p=post)
    _bwd1_fp4_doc_scalars_kernel[(1,)](sq, sk, sv, sdo, out, post, nr * hk, _f32(softmax_scale),
                                       _f32(2.0 ** dv8_shift), _f32(2.0 ** p_shift),
                                       sk_global.reshape(-1), num_warps=8, **kw)
    return out, post


def bwd1_fp8_doc_scalars(softmax_scale: float, p_shift: float, dq_boost: float, sq, sk, sv, sdo, sk_global,
                         dq_range=None, dq_docs=None):
    """Per-(document, kv head) twin of bwd1_fp8_scalars (block-scaled dS^ only): sq / sk / sv / sdo are the float32
    (n_rows, h_kv) per-document scale tables (quant.fp8.quantize_fp8_per_tensor(documents=...)), sk_global the K
    launch's global scale (the max over the documents: max|K| <= 448 sk_global for the f16 dQ range). Returns
    ((n_rows, h_kv, 8) float32 -- the 1-CTA kernel's per-tile mScalars (flash_bwd_sm100_fp4 doc_scaled) --, (1,) float32
    postprocess scale). One launch, as the per-tensor path's scalar launch."""
    nr, hk = sq.shape
    out = torch.empty(nr, hk, 8, dtype=torch.float32, device=sq.device)
    post = torch.empty(1, dtype=torch.float32, device=sq.device)
    kw = dict(DQ_RANGE=0, vn_p=post, dn_p=post)
    if dq_range is not None:
        dn2, vn2, _kmax = dq_range
        kw = dict(DQ_RANGE=1, vn_p=vn2, dn_p=dn2, dn_n=dn2.numel(), dn_s=(dn2.stride(-1) if dn2.dim() else 1),
                  vn_n=vn2.numel(), vn_s=(vn2.stride(-1) if vn2.dim() else 1))
    if dq_docs is not None and dq_range is not None:
        # dqfix: the pre-scale per (document, kv head); post becomes the (b hq tq,) postprocess plane
        post = torch.empty(dq_docs["n"], dtype=torch.float32, device=sq.device)
        kw = dict(_doc_tiles_kw(dq_docs, sq), vn_p=post, dn_p=post)
    _bwd1_fp8_doc_scalars_kernel[(1,)](sq, sk, sv, sdo, out, post, nr * hk, _f32(softmax_scale), _f32(p_shift),
                                       _f32(dq_boost), sk_global.reshape(-1), num_warps=8, **kw)
    return out, post


def bwd2_scalars(device, softmax_scale: float, p_shift_log2: int, nvfp4_s: bool, dq_f16: bool, sq, sk, sv, sdo,
                 s_ds=None, r=None, sds_mode: int = 0, dq_range=None, dq_tiles=None) -> torch.Tensor:
    """dq_range (f16 partials): (dn2, vn2, kmax) -> the data-sized dQ pre-scale and out[7] = softmax_scale / B.
    dq_tiles (dqfix): the pre-scale per (row, q head, query tile) instead, a table after the (8,) vector."""
    dq_range = dq_range if dq_f16 else None
    unit = s_ds is None and r is None and sds_mode in (2, 3) and all(_is_unit(t) for t in (sq, sk, sv, sdo))
    if dq_tiles is not None and dq_range is not None:
        out = torch.empty(8 + 2 * dq_tiles["n"] + dq_tiles["rows"], dtype=torch.float32, device=device)
        tk, grid = _dq_tiles_args(dq_tiles, dq_range, out)
        _bwd2_scalars_kernel[grid](sq, sk, sv, sdo, _ptr(s_ds, out), _ptr(r, out), out, _f32(softmax_scale),
                                   _f32(2.0 ** p_shift_log2), SDS_MODE=sds_mode, NVFP4_S=nvfp4_s,
                                   DQ_F16=dq_f16, num_warps=4, **tk)
        if unit:
            return _known_slots(out, _bwd2_host(float(softmax_scale), float(2 ** p_shift_log2), nvfp4_s, dq_f16,
                                                sds_mode), ())
        return out
    if unit and dq_range is None:
        vals = _bwd2_host(float(softmax_scale), float(2 ** p_shift_log2), nvfp4_s, dq_f16, sds_mode)
        return _const_vector(("bwd2", float(softmax_scale), p_shift_log2, nvfp4_s, dq_f16, sds_mode), vals, device)
    out = torch.empty(8 if dq_range is not None else 7, dtype=torch.float32, device=device)
    _bwd2_scalars_kernel[(1,)](sq, sk, sv, sdo, _ptr(s_ds, out), _ptr(r, out), out, _f32(softmax_scale),
                               _f32(2.0 ** p_shift_log2), SDS_MODE=sds_mode, NVFP4_S=nvfp4_s, DQ_F16=dq_f16,
                               num_warps=8 if dq_range is not None else 1, **_dq_range_args(dq_range, out))
    if unit:  # slots 4 (the kernel's dQ scale) and 7 (the postprocess scale) carry the data-sized pre-scale
        return _known_slots(out, _bwd2_host(float(softmax_scale), float(2 ** p_shift_log2), nvfp4_s, dq_f16, sds_mode),
                            (4, 7))
    return out

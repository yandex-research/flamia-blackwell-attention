# The FP64-free scale contract: every per-tensor / per-document scale and every kernel scalar derived from them is a
# float32 device value computed with correctly rounded float32 operations in a fixed order. Each test below evaluates
# the same formula independently in numpy float32 (IEEE: each operation correctly rounded, no FMA) and requires the
# device value to be BITWISE equal; it also records the relative difference to the earlier float64 derivation (the same
# formula in Python doubles, rounded to float32 once), which must stay within a few float32 ulps.
import math

import numpy as np
import pytest
import torch

F = np.float32
DEV = "cuda"
F64_REL = {}   # name -> max relative difference to the float64 derivation (printed by test_zz_report)


def _bits_eq(dev_vals, ref_vals, name):
    a = np.asarray(dev_vals, dtype=np.float32).reshape(-1)
    b = np.asarray(ref_vals, dtype=np.float32).reshape(-1)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    bad = np.nonzero(a.view(np.int32) != b.view(np.int32))[0]
    assert bad.size == 0, f"{name}: {bad.size} of {a.size} differ, first {bad[:5]}: dev {a[bad[:5]]} ref {b[bad[:5]]}"


def _rel64(name, dev_vals, f64_vals):
    a = np.asarray(dev_vals, dtype=np.float64).reshape(-1)
    b = np.asarray(f64_vals, dtype=np.float64).reshape(-1)
    b32 = b.astype(np.float32).astype(np.float64)        # the old value: the float64 result rounded once
    m = np.isfinite(b32) & (b32 != 0)
    rel = float(np.max(np.abs(a[m] - b32[m]) / np.abs(b32[m]))) if m.any() else 0.0
    F64_REL[name] = max(F64_REL.get(name, 0.0), rel)
    assert rel <= 8 * 2.0 ** -23, f"{name}: {rel:.3g} relative to the float64 derivation"


def _scales(n, lo=-6.0, hi=1.0, seed=0):
    g = np.random.default_rng(seed)
    return (10.0 ** g.uniform(lo, hi, n)).astype(np.float32)


def _t(x):
    return torch.tensor(np.asarray(x, dtype=np.float32), device=DEV)


# ------------------------------------------------------------------------------- independent references (numpy f32)
def ref_pow2_floor(y, lo=-40, hi=40):
    y = np.asarray(y, dtype=np.float32)
    e = ((y.view(np.int32) >> 23) & 0xFF) - 127
    e = np.clip(e, lo, hi)
    return ((e + 127) << 23).astype(np.int32).view(np.float32)


def ref_B(dn2, vn2, kmax, unit):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        bnd = F(4.0) * F(unit) * F(2.0) * np.sqrt(F(dn2)) * np.sqrt(F(vn2)) * F(kmax)
        return ref_pow2_floor(F(65504.0) / bnd)


def f64_B(dn2, vn2, kmax, unit):
    bnd = 4.0 * unit * 2.0 * math.sqrt(dn2) * math.sqrt(vn2) * kmax
    y = 65504.0 / bnd if bnd > 0 else math.inf
    e = math.floor(math.log2(y)) if math.isfinite(y) else 40
    return 2.0 ** min(max(e, -40), 40)


def ref_bwd1_fp4(softmax, sh, sv, sdo, sq, sk, s_ds):
    softmax, sh, sv, sdo, sq, sk, s_ds = (F(x) for x in (softmax, sh, sv, sdo, sq, sk, s_ds))
    dps = F(1.0) / (sv * sdo)
    dv_s = sdo / sh
    ds_b = ((sv * sdo) / sh) / s_ds
    dk_s = (softmax * s_ds) * sq
    dq_s = s_ds * sk
    return [softmax, dps, ds_b, dv_s, dk_s, dq_s, -ds_b * dps]


def f64_bwd1_fp4(softmax, sh, sv, sdo, sq, sk, s_ds):
    dps = 1.0 / (sv * sdo)
    ds_b = (sv * sdo / sh) / s_ds
    return [softmax, dps, ds_b, sdo / sh, softmax * s_ds * sq, s_ds * sk, -ds_b * dps]


def ref_bwd1_fp8(softmax, p_shift, dq_boost, sq, sk, sv, sdo, s_ds=None):
    softmax, p_shift, dq_boost, sq, sk, sv, sdo = (F(x) for x in (softmax, p_shift, dq_boost, sq, sk, sv, sdo))
    ds_boost = F(1.0) if s_ds is None else (sv * sdo) / F(s_ds)
    r = (sv * sdo) / ds_boost
    return [(softmax * sq) * sk, F(1.0) / (sv * sdo), ds_boost / p_shift, sdo / p_shift, (softmax * sq) * r,
            (r * sk) * dq_boost, -((ds_boost / p_shift) / (sv * sdo)), softmax / dq_boost]


def f64_bwd1_fp8(softmax, p_shift, dq_boost, sq, sk, sv, sdo, s_ds=None):
    ds_boost = 1.0 if s_ds is None else (sv * sdo) / s_ds
    return [softmax * sq * sk, 1.0 / (sv * sdo), ds_boost / p_shift, sdo / p_shift, softmax * sv * sdo * sq / ds_boost,
            dq_boost * sv * sdo * sk / ds_boost, -(ds_boost / p_shift) / (sv * sdo), softmax / dq_boost]


def ref_bwd2(softmax, p2, sq, sk, sv, sdo, s_ds):
    softmax, p2, sq, sk, sv, sdo, s_ds = (F(x) for x in (softmax, p2, sq, sk, sv, sdo, s_ds))
    ds_b = ((sv * sdo) / p2) / s_ds
    return [(softmax * sq) * sk, ds_b, sdo / p2, (softmax * s_ds) * sq, s_ds * sk, -(ds_b / (sv * sdo)),
            (softmax * s_ds) * sk]


def f64_bwd2(softmax, p2, sq, sk, sv, sdo, s_ds):
    ds_b = (sv * sdo / p2) / s_ds
    return [softmax * sq * sk, ds_b, sdo / p2, softmax * s_ds * sq, s_ds * sk, -ds_b / (sv * sdo), softmax * s_ds * sk]


SOFTMAX = 1.0 / math.sqrt(128.0) * 1.0000001   # not a float32 value: exercises its rounding


# ------------------------------------------------------------------------------------------------- the scalar kernels
@pytest.mark.parametrize("sds_mode", [0, 1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_bwd1_fp4_scalars(sds_mode, seed):
    from flash_attn.cute.device_scales import bwd1_fp4_scalars
    sv, sdo, sq, sk, sds_in, r = _scales(6, seed=seed)
    if sds_mode == 1:
        s_ds = np.maximum(F(r), F(1e-30)) / F(448.0)
        s_ds64 = max(float(r), 1e-30) / 448.0
    elif sds_mode == 2:
        s_ds, s_ds64 = F(sv) * F(sdo), float(sv) * float(sdo)
    else:
        s_ds, s_ds64 = sds_in, float(sds_in)
    out = bwd1_fp4_scalars(DEV, SOFTMAX, 3, True, True, True, True, sv=_t([sv]), sdo=_t([sdo]), sq=_t([sq]),
                           sk=_t([sk]), s_ds=_t([sds_in]) if sds_mode == 0 else None,
                           r=_t([r]) if sds_mode == 1 else None, sds_mode=sds_mode).cpu().numpy()
    _bits_eq(out[:7], ref_bwd1_fp4(SOFTMAX, 8.0, sv, sdo, sq, sk, s_ds), "bwd1_fp4")
    _rel64("bwd1_fp4", out[:7], f64_bwd1_fp4(SOFTMAX, 8.0, float(sv), float(sdo), float(sq), float(sk), s_ds64))


@pytest.mark.parametrize("kmax_mode", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_bwd1_fp4_dq_range(kmax_mode, seed):
    from flash_attn.cute.device_scales import bwd1_fp4_scalars
    sv, sdo, sq, sk = _scales(4, seed=seed)
    dn2, vn2, km = _scales(3, -4, 4, seed=seed + 50)
    out = bwd1_fp4_scalars(DEV, SOFTMAX, 3, True, True, True, True, sv=_t([sv]), sdo=_t([sdo]), sq=_t([sq]),
                           sk=_t([sk]), sds_mode=3, p_shift=6,
                           dq_range=(_t([dn2]), _t([vn2]), None if kmax_mode == 1 else _t([km]))).cpu().numpy()
    kmax = F(448.0) * F(sk) if kmax_mode == 1 else F(km)
    B = ref_B(dn2, vn2, kmax, 64.0)
    base = ref_bwd1_fp4(SOFTMAX, 8.0, sv, sdo, sq, sk, 1.0)
    _bits_eq(out[5], base[5] * B, "bwd1_fp4 dq_s B")
    _bits_eq(out[7], F(SOFTMAX) / (F(64.0) * B), "bwd1_fp4 post")
    B64 = f64_B(float(dn2), float(vn2), 448.0 * float(sk) if kmax_mode == 1 else float(km), 64.0)
    F64_REL["B (x2 = a power-of-two step)"] = max(F64_REL.get("B (x2 = a power-of-two step)", 0.0),
                                                abs(float(B) / B64 - 1.0))


@pytest.mark.parametrize("ds_mode", ["blk", "per_tensor"])
@pytest.mark.parametrize("seed", range(4))
def test_bwd1_fp8_scalars(ds_mode, seed):
    from flash_attn.cute.device_scales import bwd1_fp8_scalars
    sq, sk, sv, sdo, sds = _scales(5, seed=seed)
    s_ds = sds if ds_mode == "per_tensor" else None
    out = bwd1_fp8_scalars(DEV, SOFTMAX, 256.0, 2.0 ** 6, _t([sq]), _t([sk]), _t([sv]), _t([sdo]),
                           s_ds=None if s_ds is None else _t([s_ds])).cpu().numpy()
    _bits_eq(out, ref_bwd1_fp8(SOFTMAX, 256.0, 64.0, sq, sk, sv, sdo, s_ds), "bwd1_fp8")
    _rel64("bwd1_fp8", out, f64_bwd1_fp8(SOFTMAX, 256.0, 64.0, *(float(x) for x in (sq, sk, sv, sdo)),
                                         None if s_ds is None else float(s_ds)))


@pytest.mark.parametrize("seed", range(4))
def test_bwd1_fp8_dq_range_and_tiles(seed):
    from flash_attn.cute.device_scales import bwd1_fp8_scalars
    sq, sk, sv, sdo = _scales(4, seed=seed)
    dn2, vn2 = _scales(2, -4, 4, seed=seed + 7)
    out = bwd1_fp8_scalars(DEV, SOFTMAX, 256.0, 64.0, _t([sq]), _t([sk]), _t([sv]), _t([sdo]),
                           dq_range=(_t([dn2]), _t([vn2]), None)).cpu().numpy()
    B = ref_B(dn2, vn2, F(448.0) * F(sk), 1.0)
    _bits_eq(out, ref_bwd1_fp8(SOFTMAX, 256.0, B, sq, sk, sv, sdo), "bwd1_fp8 dq_range")
    # the per-row tile table (dqfix, batch rows): b = 3, h = 4, h_kv = 2, 2 query / 3 key tiles
    b, h, hk, tq, tk = 3, 4, 2, 2, 3
    dn = _scales(b * h * tq, -4, 3, seed=seed + 1)
    vn = _scales(b * hk * tk, -4, 3, seed=seed + 2)
    km = _scales(b * hk * tk, -3, 3, seed=seed + 3)
    tiles = dict(tab=None, rows=b, hq=h, hk=hk, tq=tq, tk=tk, n=b * tq)
    out = bwd1_fp8_scalars(DEV, SOFTMAX, 256.0, 64.0, _t([sq]), _t([sk]), _t([sv]), _t([sdo]),
                           dq_range=(_t(dn), _t(vn), _t(km)), dq_tiles=tiles).cpu().numpy()
    base = (F(sv) * F(sdo) / F(1.0)) * F(sk)
    for r in range(b):
        Br = ref_B(dn.reshape(b, h, tq)[r].max(), vn.reshape(b, hk, tk)[r].max(), km.reshape(b, hk, tk)[r].max(), 1.0)
        n = b * tq
        _bits_eq(out[8 + r * tq:8 + (r + 1) * tq], np.full(tq, base * Br, np.float32), "tile kern")
        _bits_eq(out[8 + n + r * tq:8 + n + (r + 1) * tq], np.full(tq, F(SOFTMAX) / (F(1.0) * Br), np.float32),
                 "tile post")
        _bits_eq(out[8 + 2 * n + r], base * Br, "row kern")


@pytest.mark.parametrize("sds_mode", [0, 1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_bwd2_scalars(sds_mode, seed):
    from flash_attn.cute.device_scales import bwd2_scalars
    sq, sk, sv, sdo, sds_in, r = _scales(6, seed=seed)
    if sds_mode == 1:
        s_ds, s_ds64 = np.maximum(F(r), F(1e-30)) / F(448.0), max(float(r), 1e-30) / 448.0
    elif sds_mode == 2:
        s_ds, s_ds64 = F(sv) * F(sdo), float(sv) * float(sdo)
    else:
        s_ds, s_ds64 = sds_in, float(sds_in)
    out = bwd2_scalars(DEV, SOFTMAX, 8, False, True, _t([sq]), _t([sk]), _t([sv]), _t([sdo]),
                       s_ds=_t([sds_in]) if sds_mode == 0 else None, r=_t([r]) if sds_mode == 1 else None,
                       sds_mode=sds_mode).cpu().numpy()
    _bits_eq(out, ref_bwd2(SOFTMAX, 256.0, sq, sk, sv, sdo, s_ds), "bwd2")
    _rel64("bwd2", out, f64_bwd2(SOFTMAX, 256.0, *(float(x) for x in (sq, sk, sv, sdo)), s_ds64))


def ref_doc_B(sk, sv, sdo, rd, unit):
    """_doc_prescale over a row's kv heads (arrays (hk,))."""
    with np.errstate(divide="ignore", over="ignore"):
        t = F(4.0) * F(unit) * F(2.0) * (F(rd) * F(448.0) * sdo) * (F(rd) * F(448.0) * sv) * (F(448.0) * sk)
        return ref_pow2_floor(F(65504.0) / t.max())


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
@pytest.mark.parametrize("dq", [0, 1, 2])
def test_doc_scalars(kind, dq):
    from flash_attn.cute.device_scales import bwd1_fp4_doc_scalars, bwd1_fp8_doc_scalars
    nr, hk, h, d, tq = 5, 2, 4, 128, 3
    sq, sk, sv, sdo = (_scales(nr * hk, seed=s).reshape(nr, hk) for s in range(4))
    skg = sk.max()
    dn2, vn2 = _scales(2, -2, 3, seed=9)
    kw = {}
    if dq >= 1:
        kw["dq_range"] = (_t([dn2]), _t([vn2]), None)
    if dq == 2:
        kw["dq_docs"] = dict(tile_doc_q=None, hq=h, tq=tq, n=nr * tq, d=d)
    args = (_t(sq), _t(sk), _t(sv), _t(sdo), _t([skg]))
    if kind == "fp4":
        out, post = bwd1_fp4_doc_scalars(SOFTMAX, 3, 6, *args, **kw)
    else:
        out, post = bwd1_fp8_doc_scalars(SOFTMAX, 256.0, 64.0, *args, **kw)
    out, post = out.cpu().numpy().reshape(nr * hk, 8), post.cpu().numpy()
    rd = F(math.sqrt(d))
    unit = 64.0 if kind == "fp4" else 1.0
    Bg = ref_B(dn2, vn2, F(448.0) * F(skg), unit) if dq == 1 else (F(1.0) if kind == "fp4" else F(64.0))
    for i in range(nr * hk):
        row = i // hk
        Bd = ref_doc_B(sk[row], sv[row], sdo[row], rd, unit) if dq == 2 else Bg
        a = (sq.flat[i], sk.flat[i], sv.flat[i], sdo.flat[i])
        if kind == "fp4":
            ref = ref_bwd1_fp4(SOFTMAX, 8.0, a[2], a[3], a[0], a[1], 1.0)
            ref[5] = ref[5] * Bd if dq else ref[5]
            _bits_eq(out[i, :7], ref, f"fp4 doc {i}")
        else:
            _bits_eq(out[i], ref_bwd1_fp8(SOFTMAX, 256.0, Bd, *a), f"fp8 doc {i}")
    if dq == 2:
        for t in range(nr * tq):
            Bd = ref_doc_B(sk[t // tq], sv[t // tq], sdo[t // tq], rd, unit)
            _bits_eq(post[t], F(SOFTMAX) / (F(unit) * Bd), "doc post plane")
    elif dq == 1:
        _bits_eq(post[0], F(SOFTMAX) / (F(unit) * Bg) if kind == "fp4" else F(SOFTMAX) / Bg, "doc post")


def test_const_vectors_match_kernels():
    """The launch-free host constants (unit descales) equal the kernels' float32 values with unit descales."""
    from flash_attn.cute import device_scales as D
    u = D.unit_descales(torch.device("cuda", torch.cuda.current_device()))   # (keyed like the drivers: cuda:N)
    for dp8, dv8, dq8, dk8, mode in ((1, 1, 1, 1, 3), (1, 1, 1, 0, 2), (1, 0, 0, 0, 3), (0, 1, 1, 1, 3)):
        host = D.bwd1_fp4_scalars(DEV, SOFTMAX, 3, bool(dp8), bool(dv8), bool(dq8), bool(dk8), sv=u[2], sdo=u[3], sq=u[0],
                                  sk=u[1], sds_mode=mode)
        assert getattr(host, "host_values", None) is not None
        one = _t([1.0])
        dev = D.bwd1_fp4_scalars(DEV, SOFTMAX, 3, bool(dp8), bool(dv8), bool(dq8), bool(dk8), sv=one, sdo=one.clone(),
                                 sq=one.clone(), sk=one.clone(), sds_mode=mode)
        _bits_eq(host.cpu().numpy(), dev.cpu().numpy(), "bwd1_fp4 const")
    host = D.bwd1_fp8_scalars(DEV, SOFTMAX, 256.0, 64.0, u[0], u[1], u[2], u[3])
    dev = D.bwd1_fp8_scalars(DEV, SOFTMAX, 256.0, 64.0, *(_t([1.0]) for _ in range(4)))
    _bits_eq(host.cpu().numpy(), dev.cpu().numpy(), "bwd1_fp8 const")
    for mode in (2, 3):
        host = D.bwd2_scalars(DEV, SOFTMAX, 8, False, True, u[0], u[1], u[2], u[3], sds_mode=mode)
        dev = D.bwd2_scalars(DEV, SOFTMAX, 8, False, True, *(_t([1.0]) for _ in range(4)), sds_mode=mode)
        _bits_eq(host.cpu().numpy(), dev.cpu().numpy(), "bwd2 const")


# ------------------------------------------------------------------------------------------------- the quantisers
def ref_e4_scale(amax, fmax=448.0, pow2=False):
    s = np.maximum(F(amax) / F(fmax), F(1e-12))
    if pow2:
        b = np.asarray(s, np.float32).view(np.int32)
        e = (b >> 23) + ((b & 0x7FFFFF) != 0)
        s = np.asarray(e << 23, np.int32).view(np.float32)
    return F(s)


@pytest.mark.parametrize("pow2", [False, True])
@pytest.mark.parametrize("rows", [False, True])
def test_fp8_per_tensor_scales(pow2, rows):
    """quantize_fp8_per_tensor: s = max(amax / 448, 1e-12), 1 / s (the codes), the derived scale scalars."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    g = torch.Generator(device=DEV).manual_seed(3)
    xs = [torch.randn(2, 256, 4, 128, device=DEV, generator=g, dtype=torch.bfloat16) * m for m in (1e-3, 3.7, 211.0)]
    kw = dict(output_row_norm2_max=True, output_scale_scalars=((0, SOFTMAX, 1.0), (1, 1.0, 448.0), (None, 3.0, 7.0))) \
        if rows else {}
    r = quantize_fp8_per_tensor(xs, power_of_two_scale=pow2, **kw)
    amax = [float(x.abs().max().float()) for x in xs]
    ref_s = [ref_e4_scale(a, pow2=pow2) for a in amax]
    _bits_eq(r["scales"].cpu().numpy(), ref_s, "per-tensor scales")
    if not pow2:
        _rel64("fp8 per-tensor s", r["scales"].cpu().numpy(), [max(a / 448.0, 1e-12) for a in amax])
    for x, c, s in zip(xs, r["codes"], ref_s):
        inv = F(1.0) / s
        ref_c = (x.float() * float(inv)).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(c.view(torch.uint8), ref_c.view(torch.uint8))
    if rows:
        ref_d = [(F(SOFTMAX) * ref_s[0]) / F(1.0), (F(1.0) * ref_s[1]) / F(448.0), F(3.0) / F(7.0)]
        _bits_eq(r["scale_scalars"].cpu().numpy(), ref_d, "derived")


@pytest.mark.parametrize("kind", ["fp8", "dk8"])
def test_fp8_doc_tables(kind):
    """documents (batched rows): the per-(row, kv head) scale table, the derived score table and the D fold."""
    from flash_attn.cute.lowprec.prequant_fa3 import _batch_tabs
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    b, s, h, hk = 3, 256, 4, 2
    g = torch.Generator(device=DEV).manual_seed(5)
    mult = torch.tensor([1e-4, 1.0, 37.0], device=DEV).view(b, 1, 1, 1)
    x = (torch.randn(b, s, h, 128, device=DEV, generator=g) * mult).to(torch.bfloat16)
    o = torch.randn(b, s, h, 128, device=DEV, generator=g).to(torch.bfloat16)
    t = _batch_tabs(x.device, b, s, s)
    sv_tab = _t(_scales((b + 1) * hk, seed=11).reshape(b + 1, hk))
    c = 1.0 / 256.0 if kind == "fp8" else 256.0
    docs = dict(side=("q",), row_end_q=t["row_end_q"], row_end_k=t["row_end_k"], tile_doc_q=t["tile_doc_q"],
                tile_doc_k=t["tile_doc_k"], n_rows=b + 1, h_kv=hk, batched=True)
    r0 = quantize_fp8_per_tensor(x, compute_d_with_output=o, documents=dict(docs, derived=((0, 0, SOFTMAX), (0, -1, 2.0))))
    r1 = quantize_fp8_per_tensor(x, compute_d_with_output=o, documents=dict(docs, fold=(sv_tab, c, kind)))
    am = x.float().abs().view(b, s, hk, h // hk, 128).amax(dim=(1, 3, 4)).cpu().numpy()   # (b, hk)
    tab = r0["doc_scales"][0].cpu().numpy()
    ref = np.array([[ref_e4_scale(a) for a in row] for row in am], np.float32)
    _bits_eq(tab[:b], ref, "doc scale table")
    _rel64("fp8 doc s", tab[:b], am.astype(np.float64) / 448.0)
    der = r0["doc_derived"].cpu().numpy()
    _bits_eq(der[0][:b], (ref * F(SOFTMAX)) * ref, "doc derived (s m) s")
    _bits_eq(der[1][:b], ref * F(2.0), "doc derived s m")
    # the fold: D'' = D * m per (row, kv head) of the row's head
    m = np.zeros((b, hk), np.float32)
    svn = sv_tab.cpu().numpy()
    for i in range(b):
        for j in range(hk):
            tt = F(svn[i, j]) * ref[i, j]
            m[i, j] = -(F(c) / tt) if kind == "fp8" else -((tt / F(c)) * (F(1.0) / tt))
    mm = torch.tensor(m, device=DEV).repeat_interleave(h // hk, 1)[:, :, None]        # (b, h, 1)
    assert torch.equal(r1["D"], r0["D"] * mm), "D fold"


@pytest.mark.parametrize("docs", [False, True])
def test_fp4_global_scale(docs):
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    g = torch.Generator(device=DEV).manual_seed(1)
    x = (torch.randn(1, 512, 4, 128, device=DEV, generator=g) * 3.3e-3).to(torch.bfloat16)
    kw = {}
    if docs:
        kw = dict(row_documents=torch.tensor([0, 0, 1, 2], dtype=torch.int32, device=DEV), n_documents=4)
    div = 6.0 * 448.0
    r = quantize_fp4_rows(x, make_row_copy=False, output_tensor_max="abs", fp4_global_scale_divisor=div, **kw)
    gs = r["fp4_global_scale"].cpu().numpy()
    if docs:
        am = x.float().abs().view(4, 128, 4, 128).amax(dim=(1, 2, 3)).cpu().numpy()
        ref = [np.maximum(F(am[0:2].max()) / F(div), F(1e-30)), np.maximum(F(am[2]) / F(div), F(1e-30)),
               np.maximum(F(am[3]) / F(div), F(1e-30)), F(1e-30)]
    else:
        ref = [np.maximum(F(float(x.float().abs().max())) / F(div), F(1e-30))]
    _bits_eq(gs, ref, "fp4 global scale")
    # the E4M3 copy with that device scale: 1 / s as rcp.rn
    if not docs:
        r2 = quantize_fp4_rows(x, also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=r["fp4_global_scale"])
        inv = F(1.0) / F(gs[0])
        ref_c = (x.float() * float(inv)).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(r2["e4m3_copy"].view(torch.uint8), ref_c.view(torch.uint8))


def test_fp4q_e4m3_scl():
    """fp4q_diag_triton: E4M3 Q / dO scales and the dK / dV / dS / dQ epilogue scalars (causal and dense)."""
    from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_device, e4m3_q_do_dense
    g = torch.Generator(device=DEV).manual_seed(2)
    q = torch.randn(1, 256, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    do = (torch.randn(1, 256, 2, 128, device=DEV, generator=g) * 1e-5).to(torch.bfloat16)
    q8, do8, scl = e4m3_q_do_device(q, do, None, SOFTMAX)
    sq, sdo = ref_e4_scale(float(q.float().abs().max())), ref_e4_scale(float(do.float().abs().max()))
    _bits_eq(scl.cpu().numpy(), [F(SOFTMAX) * sq, sdo / F(448.0), F(1.0)], "fp4q scl")
    for x, c, s in ((q, q8, sq), (do, do8, sdo)):
        assert torch.equal(c.view(torch.uint8),
                           (x.float() * float(F(1.0) / s)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8))
    amax = torch.tensor([float(q.float().abs().max()), float(do.float().abs().max())], device=DEV)
    so_do, bound = _t([3.1e-7]), _t([5.5e3])
    rb = torch.ones(1, 2, 256, device=DEV)
    _q8, _do8, scl = e4m3_q_do_dense(q, do, rb, SOFTMAX, amax, so_do, bound)
    ds_mul = F(448.0) / np.maximum(F(5.5e3), F(1e-30))
    base = F(SOFTMAX) * F(3.1e-7)
    _bits_eq(scl.cpu().numpy(), [(base * sq) / ds_mul, sdo / F(448.0), ds_mul, base / ds_mul], "fp4q dense scl")


def test_fa3_tables_and_score_scale():
    from flash_attn.cute.lowprec.prequant_fa3 import fa3_tables
    from flash_attn.cute.fp4_bwd_autograd import score_scale_dev
    n, hk = 5, 3
    qd, kd, vd = (_scales(n * hk, seed=s).reshape(n, hk) for s in (1, 2, 3))
    for dt in (torch.float32, torch.float64):   # float64 descales are accepted (converted to float32 once)
        score, vdesc, dsc, kg, vn = fa3_tables(*(torch.tensor(x, dtype=dt, device=DEV) for x in (qd, kd, vd)),
                                               SOFTMAX, docs=True)
        _bits_eq(score.cpu().numpy(), (qd * F(SOFTMAX)) * kd, "fa3 score")
        _bits_eq(vdesc.cpu().numpy(), vd, "fa3 vdesc")
        _bits_eq(dsc.cpu().numpy()[:, :n], np.stack([qd, kd, vd]), "fa3 tables")
        _bits_eq(kg.cpu().numpy(), [kd.max()], "fa3 k global")
    _rel64("fa3 score", score.cpu().numpy(), (qd.astype(np.float64) * SOFTMAX) * kd.astype(np.float64))
    s = score_scale_dev(SOFTMAX, _t([qd[0, 0]]), _t([kd[0, 0]])).cpu().numpy()
    _bits_eq(s, [(qd[0, 0] * F(SOFTMAX)) * kd[0, 0]], "score_scale_dev")


def test_pow2_floor_bits():
    from flash_attn.cute.interface_bwd_fp4q import pow2_floor_bits
    x = np.concatenate([_scales(4000, -30, 30, seed=4), np.float32([1.0, 2.0, 0.99999994, 1.9999999, 3.0e38])])
    got = pow2_floor_bits(_t(x)).cpu().numpy()
    _bits_eq(got, ref_pow2_floor(x, -126, 127), "pow2 floor")
    got = pow2_floor_bits(_t(x), cap=6.0).cpu().numpy()
    _bits_eq(got, ref_pow2_floor(x, -126, 6), "pow2 floor cap")


def test_zz_report():
    """Prints the max relative differences to the float64 derivations (in float32 ulps at 2^-23)."""
    for k, v in sorted(F64_REL.items()):
        print(f"F64REL {k}: {v:.3e} ({v / 2.0 ** -23:.2f} ulp)")

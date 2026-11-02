# Triton helpers of the all-FP4 step's diagonal route that are not FP4 quantisers. The diagonal builders that lived here
# (v_diag_fused, k_diag_fused, k_diag_fused_v: four launches) are replaced by ONE launch of
# flash_attn.cute.quant.fp4_diagonal.quantize_fp4_causal_diagonal. What remains: e4m3_q_do_device (per-tensor E4M3 of
# Q / F_q and dO with device scales) and rows_scale_dk_dv (the K fold's dK / dV row unscale after the kernel).
import numpy as np
import torch
import triton
import triton.language as tl



# ------------------------------------------------------------------ per-tensor E4M3 of Q / F_q and dO with device scales
# interface_bwd_fp4.quantize_e4m3_per_tensor reads amax to the host (a sync) and derives s = max(amax / 448, 1e-12) and
# 1 / s. Here s and 1 / s are formed on the device in float32 (div.rn; bitwise quant.fp8's per-tensor scale, no FP64);
# the backward kernel's dK / dV epilogue scales (softmax_scale s_Q, s_dO / 448) go to a 2-float device buffer the
# kernel reads instead of its host arguments.
@triton.jit
def _e4_load(q_ptr, do_ptr, f_ptr, pid, tiles_one, s, h, sb, ss, sh, D: tl.constexpr, HAS_F: tl.constexpr,
             F_ON_DO: tl.constexpr = False):
    which = (pid >= tiles_one).to(tl.int32)
    p = pid - which * tiles_one
    tiles = s // 128
    bh = p // tiles
    t = p % tiles
    b = bh // h
    hh = bh % h
    rows = t * 128 + tl.arange(0, 128)
    cols = tl.arange(0, D)
    off = b * sb + hh * sh + rows[:, None] * ss + cols[None, :]
    if which == 0:
        x = tl.load(q_ptr + off).to(tl.float32)
        if HAS_F:
            # (q / F_q).to(bf16): F_q a power of two, so both steps are exact
            x = tl.math.div_rn(x, tl.load(f_ptr + bh * s + rows)[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(do_ptr + off).to(tl.float32)
        if F_ON_DO:
            # the dense route's row boost divides dO's rows too: (dO / r_q).to(bf16)
            x = tl.math.div_rn(x, tl.load(f_ptr + bh * s + rows)[:, None]).to(tl.bfloat16).to(tl.float32)
    return x, which, off


@triton.jit
def _div_rn(a, b):
    """a / b correctly rounded (div.rn.f32; Triton's '/' is div.full)."""
    a, b = tl.broadcast(tl.cast(a, tl.float32), tl.cast(b, tl.float32))
    return tl.math.div_rn(a, b)


@triton.jit
def _e4_scale(a):
    """(s, 1 / s) of a per-tensor E4M3 scale from its float32 amax: s = max(amax / 448, 1e-12) (div.rn, then the clamp),
    1 / s by div.rn -- bitwise quant.fp8's _scale_of (div.rn / rcp.rn)."""
    sc = tl.maximum(_div_rn(a, 448.0), 1e-12)
    return sc, _div_rn(1.0, sc)


@triton.jit
def _e4_amax(q_ptr, do_ptr, f_ptr, amax_ptr, tiles_one, s, h, sb, ss, sh, D: tl.constexpr, HAS_F: tl.constexpr):
    pid = tl.program_id(0)
    x, which, _ = _e4_load(q_ptr, do_ptr, f_ptr, pid, tiles_one, s, h, sb, ss, sh, D, HAS_F)
    m = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    tl.atomic_max(amax_ptr + which, m.to(tl.int32, bitcast=True))   # non-negative floats order as their bits


@triton.jit
def _e4_quant(q_ptr, do_ptr, f_ptr, amax_ptr, y_ptr, ydo_ptr, scl_ptr, tiles_one, s, h, sb, ss, sh, sm_f32,
              D: tl.constexpr, HAS_F: tl.constexpr, DENSE: tl.constexpr = False, so_do_ptr=None, bound_ptr=None):
    pid = tl.program_id(0)
    x, which, off = _e4_load(q_ptr, do_ptr, f_ptr, pid, tiles_one, s, h, sb, ss, sh, D, HAS_F, DENSE)
    sc, inv = _e4_scale(tl.load(amax_ptr + which).to(tl.float32, bitcast=True))
    y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
    if which == 0:
        tl.store(y_ptr + off, y)
    else:
        tl.store(ydo_ptr + off, y)
    if pid == 0:
        sq, _iq = _e4_scale(tl.load(amax_ptr).to(tl.float32, bitcast=True))
        sdo, _id = _e4_scale(tl.load(amax_ptr + 1).to(tl.float32, bitcast=True))
        sm = sm_f32.to(tl.float32)   # float32(softmax_scale)
        if DENSE:
            # fp4q_prepare's global dS multiplier, its float32 device expressions in order (torch's float32 ops):
            #   ds_mul = 448 / max(bound, 1e-30); dk = (sm * s_dO) * s_Q / ds_mul; dv = s_dO8 / 448;
            #   dq = (sm * s_dO) / ds_mul (scl, the postprocess scale)
            bound = tl.load(bound_ptr).to(tl.float32)
            so_do = tl.load(so_do_ptr).to(tl.float32)
            ds_mul = _div_rn(448.0, tl.maximum(bound, 1e-30))
            base = sm * so_do
            tl.store(scl_ptr, _div_rn(base * sq, ds_mul))
            tl.store(scl_ptr + 1, _div_rn(sdo, 448.0))
            tl.store(scl_ptr + 2, ds_mul)
            tl.store(scl_ptr + 3, _div_rn(base, ds_mul))
        else:
            tl.store(scl_ptr, sm * sq)
            tl.store(scl_ptr + 1, _div_rn(sdo, 448.0))
            tl.store(scl_ptr + 2, 1.0)   # ds_mul = 1 (the causal dS scale): written here, no fill launch


def e4m3_q_do_device(q: torch.Tensor, dout: torch.Tensor, f_row, softmax_scale: float,
                     amax: torch.Tensor = None):
    """(q8, do8, scl): E4M3(q / F_q / s_Q), E4M3(dO / s_dO) with s = max(amax / 448, 1e-12), bitwise
    quantize_e4m3_per_tensor((q / F).to(bf16)) and quantize_e4m3_per_tensor(dO); scl = [softmax_scale s_Q, s_dO / 448] as fp32
    (the dkv8 backward's dk_scale / dv_scale in the causal dS-scale mode). f_row (b, h, s) power-of-two fp32 or None. Two
    launches (the amax, the quantiser), no host read. amax: (2,) float32 [max |q / F_q|, max |dO|] already reduced (the
    dO launch's max slots, quant.fp4 causal_row_scale): then ONE launch (the quantiser)."""
    b, s, h, d = q.shape
    assert dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and s % 128 == 0 and d == 128
    tiles_one = b * h * (s // 128)
    given = amax is not None
    if given:
        assert amax.dtype == torch.float32 and amax.numel() == 2
        amax = amax.view(torch.int32)    # non-negative floats: the same bits the atomic path reads
    else:
        amax = torch.zeros(2, dtype=torch.int32, device=q.device)
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(3, dtype=torch.float32, device=q.device)  # [dk_scale, dv_scale, ds_mul = 1 (the causal dS scale)]
    fr = f_row.contiguous() if f_row is not None else amax
    args = (q, dout, fr)
    st = (s, h, q.stride(0), q.stride(1), q.stride(2))
    if not given:
        _e4_amax[(2 * tiles_one,)](*args, amax, tiles_one, *st, D=d, HAS_F=f_row is not None, num_warps=8)
    sm_f32 = float(np.float32(softmax_scale))   # the kernels' float32 softmax_scale
    _e4_quant[(2 * tiles_one,)](*args, amax, q8, do8, scl, tiles_one, *st, sm_f32, D=d,
                                HAS_F=f_row is not None, num_warps=8)
    return q8, do8, scl


# ------------------------------------------------------------------ the fold's dK / dV row unscale, after the kernel
@triton.jit
def _rows_scale2(dk_ptr, dv_ptr, fk_ptr, fv_ptr, s, h, sb, ss, sh, D: tl.constexpr):
    """dK[b, key, h, :] *= fk[b, h, key]; dV likewise with fv (in place). fk / fv powers of two, so each product is exact
    and equals scaling the fp32 accumulator before its bf16 rounding (outside the subnormal range)."""
    pid = tl.program_id(0)
    which = pid % 2
    p = pid // 2
    tiles = s // 128
    bh = p // tiles
    t = p % tiles
    b = bh // h
    hh = bh % h
    rows = t * 128 + tl.arange(0, 128)
    cols = tl.arange(0, D)
    off = b * sb + hh * sh + rows[:, None] * ss + cols[None, :]
    if which == 0:
        x = tl.load(dk_ptr + off).to(tl.float32) * tl.load(fk_ptr + bh * s + rows)[:, None]
        tl.store(dk_ptr + off, x.to(dk_ptr.dtype.element_ty))
    else:
        x = tl.load(dv_ptr + off).to(tl.float32) * tl.load(fv_ptr + bh * s + rows)[:, None]
        tl.store(dv_ptr + off, x.to(dv_ptr.dtype.element_ty))


def rows_scale_dk_dv(dk: torch.Tensor, dv: torch.Tensor, fk: torch.Tensor, fv: torch.Tensor):
    """In place: dK / dV rows (b, s_k, h_k, d) times their (b, h_k, s_k) factors. One launch."""
    b, s, h, d = dk.shape
    assert dv.shape == dk.shape and dk.stride() == dv.stride() and dk.stride(3) == 1 and s % 128 == 0
    _rows_scale2[(2 * b * h * (s // 128),)](dk, dv, fk, fv, s, h, dk.stride(0), dk.stride(1), dk.stride(2), D=d, num_warps=8)


def e4m3_q_do_dense(q: torch.Tensor, dout: torch.Tensor, row_boost: torch.Tensor, softmax_scale: float,
                    amax: torch.Tensor, so_do: torch.Tensor, bound: torch.Tensor):
    """The dense all-FP4 backward's per-tensor E4M3 copies, ONE launch: q8 = E4M3(bf16(q / r_q) / s_Q), do8 =
    E4M3(bf16(dO / r_q) / s_dO), s = max(amax / 448, 1e-12) from the row-boost launch's amax (2,) float32, and scl (4,)
    float32 = [dk_scale, dv_scale, ds_mul, dq_scale] of the global dS multiplier from so_do ((1,) float32, the dO global
    scale) and bound ((1,) float32, the dS bound): bitwise fp4q_prepare's torch / quantize_e4m3_dev path."""
    b, s, h, d = q.shape
    assert dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and s % 128 == 0 and d == 128
    assert amax.dtype == torch.float32 and amax.numel() == 2 and so_do.dtype == torch.float32 and bound.numel() >= 1
    tiles_one = b * h * (s // 128)
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(4, dtype=torch.float32, device=q.device)
    sm_f32 = float(np.float32(softmax_scale))   # the kernels' float32 softmax_scale
    _e4_quant[(2 * tiles_one,)](q, dout, row_boost, amax.view(torch.int32), q8, do8, scl, tiles_one, s, h, q.stride(0),
                                q.stride(1), q.stride(2), sm_f32, D=d, HAS_F=True, DENSE=True, so_do_ptr=so_do,
                                bound_ptr=bound, num_warps=8)
    return q8, do8, scl


# ------------------------------------------------------------------ aligned varlen documents (b = 1): per-document scales
@triton.jit
def _e4_quant_docs(q_ptr, do_ptr, f_ptr, amax_ptr, y_ptr, ydo_ptr, scl_ptr, tdoc_ptr, n_docs1, tiles_one, s, h, sb, ss,
                   sh, sm_f32, D: tl.constexpr, HAS_F: tl.constexpr, BD: tl.constexpr):
    """_e4_quant per document: the 128-row tile's document (tdoc) selects its pair of amax slots [max |q / F|, max |dO|]
    (the dO launch's per-document max slots), so a document's E4M3 Q / dO codes and its scl row [softmax_scale s_Q,
    s_dO / 448, 1] depend on its own rows only. Program 0 writes every document's scl row (n_docs1 = documents + the
    sentinel)."""
    pid = tl.program_id(0)
    x, which, off = _e4_load(q_ptr, do_ptr, f_ptr, pid, tiles_one, s, h, sb, ss, sh, D, HAS_F)
    p = pid - which * tiles_one
    doc = tl.load(tdoc_ptr + p % (s // 128))
    sc, inv = _e4_scale(tl.load(amax_ptr + 2 * doc + which).to(tl.float32, bitcast=True))
    y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
    if which == 0:
        tl.store(y_ptr + off, y)
    else:
        tl.store(ydo_ptr + off, y)
    if pid == 0:
        sm = sm_f32.to(tl.float32)
        for d0 in range(0, n_docs1, BD):
            dd = d0 + tl.arange(0, BD)
            md = dd < n_docs1
            sq, _iq = _e4_scale(tl.load(amax_ptr + 2 * dd, mask=md, other=0).to(tl.float32, bitcast=True))
            sdo, _id = _e4_scale(tl.load(amax_ptr + 2 * dd + 1, mask=md, other=0).to(tl.float32, bitcast=True))
            tl.store(scl_ptr + 3 * dd, sm * sq, mask=md)
            tl.store(scl_ptr + 3 * dd + 1, _div_rn(sdo, 448.0), mask=md)
            tl.store(scl_ptr + 3 * dd + 2, tl.full([BD], 1.0, tl.float32), mask=md)


def e4m3_q_do_device_docs(q: torch.Tensor, dout: torch.Tensor, f_row, softmax_scale: float, amax: torch.Tensor,
                          tile_doc_q: torch.Tensor):
    """e4m3_q_do_device on aligned varlen documents (b = 1): amax ((n_docs + 1) * 2,) float32 per-document max slots,
    tile_doc_q (s / 128,) int32. Returns (q8, do8, scl (n_docs + 1, 3) float32), one launch; per document bitwise
    e4m3_q_do_device on that document alone."""
    b, s, h, d = q.shape
    assert b == 1 and dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and s % 128 == 0 \
        and d == 128
    assert amax.dtype == torch.float32 and amax.numel() % 2 == 0 and tile_doc_q.numel() == s // 128
    n_docs1 = amax.numel() // 2
    tiles_one = h * (s // 128)
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(n_docs1, 3, dtype=torch.float32, device=q.device)
    sm_f32 = float(np.float32(softmax_scale))   # the kernels' float32 softmax_scale
    fr = f_row.contiguous() if f_row is not None else amax
    _e4_quant_docs[(2 * tiles_one,)](q, dout, fr, amax.view(torch.int32), q8, do8, scl, tile_doc_q, n_docs1, tiles_one,
                                     s, h, q.stride(0), q.stride(1), q.stride(2), sm_f32, D=d,
                                     HAS_F=f_row is not None, BD=128, num_warps=8)
    return q8, do8, scl


@triton.jit
def _e4_quant_dense_docs(q_ptr, do_ptr, f_ptr, amax_ptr, y_ptr, ydo_ptr, scl_ptr, dqs_ptr, tdoc_ptr, so_do_ptr, bound_ptr,
                         n_docs1,
                         tiles_one, s, h, sb, ss, sh, sm_f32, D: tl.constexpr, BD: tl.constexpr):
    """_e4_quant(DENSE) per document (b = 1): the 128-row tile's document selects its amax pair, s_dO and dS bound; program
    0 writes every document's scl row [dk_scale, dv_scale, ds_mul, dq_scale] (n_docs1 = documents + the sentinel)."""
    pid = tl.program_id(0)
    x, which, off = _e4_load(q_ptr, do_ptr, f_ptr, pid, tiles_one, s, h, sb, ss, sh, D, True, True)
    p = pid - which * tiles_one
    doc = tl.load(tdoc_ptr + p % (s // 128))
    sc, inv = _e4_scale(tl.load(amax_ptr + 2 * doc + which).to(tl.float32, bitcast=True))
    y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
    if which == 0:
        tl.store(y_ptr + off, y)
    else:
        tl.store(ydo_ptr + off, y)
    if pid == 0:
        sm = sm_f32.to(tl.float32)
        for d0 in range(0, n_docs1, BD):
            dd = d0 + tl.arange(0, BD)
            md = dd < n_docs1
            sq, _iq = _e4_scale(tl.load(amax_ptr + 2 * dd, mask=md, other=0).to(tl.float32, bitcast=True))
            sdo, _id = _e4_scale(tl.load(amax_ptr + 2 * dd + 1, mask=md, other=0).to(tl.float32, bitcast=True))
            bound = tl.load(bound_ptr + dd, mask=md, other=1.0).to(tl.float32)
            so_do = tl.load(so_do_ptr + dd, mask=md, other=1.0).to(tl.float32)
            ds_mul = _div_rn(448.0, tl.maximum(bound, 1e-30))
            base = sm * so_do
            tl.store(scl_ptr + 4 * dd, _div_rn(base * sq, ds_mul), mask=md)
            tl.store(scl_ptr + 4 * dd + 1, _div_rn(sdo, 448.0), mask=md)
            tl.store(scl_ptr + 4 * dd + 2, ds_mul, mask=md)
            tl.store(scl_ptr + 4 * dd + 3, _div_rn(base, ds_mul), mask=md)
            tl.store(dqs_ptr + dd, _div_rn(base, ds_mul), mask=md)   # (contiguous: the post's table)


def e4m3_q_do_dense_docs(q: torch.Tensor, dout: torch.Tensor, row_boost: torch.Tensor, softmax_scale: float,
                         amax: torch.Tensor, so_do: torch.Tensor, bound: torch.Tensor, tile_doc_q: torch.Tensor):
    """e4m3_q_do_dense on aligned varlen documents (b = 1): amax ((n_docs + 1) * 2,), so_do (n_docs + 1,) float32, bound
    (n_docs + 1,) float32, all per document; scl (n_docs + 1, 4) and dq_scale (n_docs + 1,). One launch; per document bitwise e4m3_q_do_dense on that
    document alone."""
    b, s, h, d = q.shape
    assert b == 1 and dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and s % 128 == 0 \
        and d == 128
    n_docs1 = bound.numel()
    assert amax.numel() == 2 * n_docs1 and so_do.numel() == n_docs1 and so_do.dtype == torch.float32
    tiles_one = h * (s // 128)
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(n_docs1, 4, dtype=torch.float32, device=q.device)
    dqs = torch.empty(n_docs1, dtype=torch.float32, device=q.device)
    sm_f32 = float(np.float32(softmax_scale))   # the kernels' float32 softmax_scale
    _e4_quant_dense_docs[(2 * tiles_one,)](q, dout, row_boost, amax.view(torch.int32), q8, do8, scl, dqs, tile_doc_q, so_do,
                                           bound, n_docs1, tiles_one, s, h, q.stride(0), q.stride(1), q.stride(2), sm_f32,
                                           D=d, BD=128, num_warps=8)
    return q8, do8, scl, dqs


# ------------------------------------------------------------------ cov5: the packed rows of compact DocMaps (b = 1)
@triton.jit
def _e4_quant_docs_packed(q_ptr, do_ptr, f_ptr, amax_ptr, y_ptr, ydo_ptr, scl_ptr, tdoc_ptr, vt_ptr, nt_ptr, n_docs1,
                          tiles_s, T, h, ss, sh, sm_f32, D: tl.constexpr, HAS_F: tl.constexpr, BD: tl.constexpr):
    """_e4_quant_docs on the packed rows: persistent over (q / dO, aligned tile, head) of the DEVICE tile count; tile t's
    rows the packed rows vt[0, t] + r, r < vt[1, t] (the codes written there only), its document tdoc[t]."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nt = tl.load(nt_ptr)
    tot = 2 * nt * h
    r = tl.arange(0, 128)
    cols = tl.arange(0, D)
    for it in range(pid, tot, nprog):
        which = (it >= nt * h).to(tl.int32)
        p = it - which * nt * h
        t = p % nt
        hh = p // nt
        pb = tl.load(vt_ptr + t)
        nvr = tl.load(vt_ptr + tiles_s + t) & 0xFFFF
        rows = pb + r
        m = r < nvr
        off = hh * sh + rows[:, None] * ss + cols[None, :]
        mk = m[:, None]
        if which == 0:
            x = tl.load(q_ptr + off, mask=mk, other=0.0).to(tl.float32)
            if HAS_F:
                x = tl.math.div_rn(x, tl.load(f_ptr + hh * T + rows, mask=m, other=1.0)[:, None]).to(tl.bfloat16).to(
                    tl.float32)
        else:
            x = tl.load(do_ptr + off, mask=mk, other=0.0).to(tl.float32)
        doc = tl.load(tdoc_ptr + t)
        sc, inv = _e4_scale(tl.load(amax_ptr + 2 * doc + which).to(tl.float32, bitcast=True))
        y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
        if which == 0:
            tl.store(y_ptr + off, y, mask=mk)
        else:
            tl.store(ydo_ptr + off, y, mask=mk)
    # (spd4) the per-document scl rows split over every program: program 0 alone walked all N + 1 rows, a serial tail
    # of ~0.3 ms at the fork's capacity N = T (T = 65536); every row is still written once, with the same values
    sm = sm_f32.to(tl.float32)
    for d0 in range(pid * BD, n_docs1, nprog * BD):
        dd = d0 + tl.arange(0, BD)
        md = dd < n_docs1
        sq, _iq = _e4_scale(tl.load(amax_ptr + 2 * dd, mask=md, other=0).to(tl.float32, bitcast=True))
        sdo, _id = _e4_scale(tl.load(amax_ptr + 2 * dd + 1, mask=md, other=0).to(tl.float32, bitcast=True))
        tl.store(scl_ptr + 3 * dd, sm * sq, mask=md)
        tl.store(scl_ptr + 3 * dd + 1, _div_rn(sdo, 448.0), mask=md)
        tl.store(scl_ptr + 3 * dd + 2, tl.full([BD], 1.0, tl.float32), mask=md)


def e4m3_q_do_device_packed(q: torch.Tensor, dout: torch.Tensor, f_row, softmax_scale: float, amax: torch.Tensor,
                            tile_doc_q: torch.Tensor, vt_tile_q: torch.Tensor, ntiles_q: torch.Tensor):
    """(cov5) e4m3_q_do_device_docs on the packed rows of compact DocMaps: q / dout (1, T, h, 128) packed, f_row (1, h, T),
    amax ((n_docs + 1) * 2,) the per-document max slots, tile_doc_q / vt_tile_q (2, tiles) / ntiles_q (1,) the maps'
    query tiles. Returns (q8, do8 (1, T, h, 128) E4M3 at the packed rows, scl (n_docs + 1, 3)); per row bitwise
    e4m3_q_do_device_docs on the aligned layout. One persistent launch."""
    b, s, h, d = q.shape
    assert b == 1 and dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and d == 128
    assert amax.dtype == torch.float32 and amax.numel() % 2 == 0 and tile_doc_q.numel() == vt_tile_q.shape[1]
    n_docs1 = amax.numel() // 2
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(n_docs1, 3, dtype=torch.float32, device=q.device)
    sm_f32 = float(np.float32(softmax_scale))   # the kernels' float32 softmax_scale
    fr = f_row.contiguous() if f_row is not None else amax
    tiles_s = vt_tile_q.shape[1]
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = max(1, min(2 * h * tiles_s, 4 * nsm))
    _e4_quant_docs_packed[(grid,)](q, dout, fr, amax.view(torch.int32), q8, do8, scl, tile_doc_q, vt_tile_q, ntiles_q,
                                   n_docs1, tiles_s, s, h, q.stride(1), q.stride(2), sm_f32, D=d,
                                   HAS_F=f_row is not None, BD=128, num_warps=8)
    return q8, do8, scl


@triton.jit
def _fp4q_post_packed(acc_ptr, dq_ptr, rf_ptr, dk_ptr, dv_ptr, fk_ptr, fv_ptr, ctr_ptr, scale, T, Tk, RT, h, hk,
                      sacc_h, sacc_r, sdq_r, sdq_h, sdk_r, sdk_h, TPK, nq_prog,
                      HAS_RF: tl.constexpr, FOLD: tl.constexpr, CTR: tl.constexpr, rs_ptr=None,
                      ROW_SCALE: tl.constexpr = False):
    """(cov5) fp4q_postprocess on the packed rows: dQ (1, T, h, 128) = bf16(acc * scale * (1 / F_q)) of the (1, h, T', 128)
    accumulator (the same fp32 operations as dq_postprocess_cute.FP4QPostprocess), and, with FOLD, dK / dV (1, T_k, h_k,
    128) rows times their (h_k, T') powers of two in place; the SR counter advanced by one."""
    pid = tl.program_id(0)
    if CTR:
        if pid == 0:
            tl.store(ctr_ptr, tl.load(ctr_ptr) + 1)
    cols = tl.arange(0, 128)
    r = tl.arange(0, 64)
    if pid < nq_prog:
        nb = tl.cdiv(T, 64)
        hh = pid // nb
        rows = (pid % nb) * 64 + r
        m = rows < T
        x = tl.load(acc_ptr + hh * sacc_h + rows[:, None] * sacc_r + cols[None, :], mask=m[:, None], other=0.0)
        if ROW_SCALE:   # (cov6 dense: the row's document's dq_scale, (T,))
            x = x.to(tl.float32) * tl.load(rs_ptr + rows, mask=m, other=1.0)[:, None]
        else:
            x = x.to(tl.float32) * scale
        if HAS_RF:
            x = x * (1.0 / tl.load(rf_ptr + hh * RT + rows, mask=m, other=1.0))[:, None]
        tl.store(dq_ptr + rows[:, None] * sdq_r + hh * sdq_h + cols[None, :], x.to(dq_ptr.dtype.element_ty),
                 mask=m[:, None])
    else:
        if FOLD:
            p = pid - nq_prog
            nb = tl.cdiv(Tk, 64)
            which = p % 2
            p2 = p // 2
            hh = p2 // nb
            rows = (p2 % nb) * 64 + r
            m = rows < Tk
            off = rows[:, None] * sdk_r + hh * sdk_h + cols[None, :]
            if which == 0:
                f = tl.load(fk_ptr + hh * TPK + rows, mask=m, other=1.0)
                x = tl.load(dk_ptr + off, mask=m[:, None], other=0.0).to(tl.float32) * f[:, None]
                tl.store(dk_ptr + off, x.to(dk_ptr.dtype.element_ty), mask=m[:, None])
            else:
                f = tl.load(fv_ptr + hh * TPK + rows, mask=m, other=1.0)
                x = tl.load(dv_ptr + off, mask=m[:, None], other=0.0).to(tl.float32) * f[:, None]
                tl.store(dv_ptr + off, x.to(dv_ptr.dtype.element_ty), mask=m[:, None])


def fp4q_postprocess_packed(dq_accum, T: int, scale, out_dtype, row_factor=None, dk=None, dv=None,
                            dk_row_factor=None, dv_row_factor=None, seed_counter=None):
    """(cov5) The packed rows' post launch: dq_accum (1, h, T', 128) f16, row_factor (1, h, T'') f32 powers of two (or None),
    dk / dv (1, T_k, h_k, 128) with (h_k, T_k') factors (or None). Returns dQ (1, T, h, 128). One launch."""
    _, h, Tr, d = dq_accum.shape
    assert d == 128 and dq_accum.is_contiguous() and Tr >= T
    dq = torch.empty(1, T, h, d, dtype=out_dtype, device=dq_accum.device)
    fold = dk is not None
    nq = h * ((T + 63) // 64)
    nkv = 2 * dk.shape[2] * ((dk.shape[1] + 63) // 64) if fold else 0
    rf = row_factor if row_factor is not None else dq_accum
    rs = isinstance(scale, torch.Tensor)   # (cov6: a per-packed-row scale (T,) float32)
    assert not rs or (scale.dtype == torch.float32 and scale.numel() >= T)
    _fp4q_post_packed[(nq + nkv,)](
        dq_accum, dq, rf, dk if fold else dq, dv if fold else dq, dk_row_factor if fold else rf,
        dv_row_factor if fold else rf, seed_counter if seed_counter is not None else rf, 1.0 if rs else float(scale), T,
        dk.shape[1] if fold else 0, row_factor.shape[-1] if row_factor is not None else 0, h, dk.shape[2] if fold else 0,
        dq_accum.stride(1), dq_accum.stride(2), dq.stride(1), dq.stride(2),
        dk.stride(1) if fold else 0, dk.stride(2) if fold else 0, dk_row_factor.shape[-1] if fold else 0, nq,
        HAS_RF=row_factor is not None, FOLD=fold, CTR=seed_counter is not None,
        **(dict(rs_ptr=scale, ROW_SCALE=True) if rs else {}), num_warps=4)
    return dq


# ------------------------------------------------------------------ cov6: the dense all-4-bit backward on packed rows
@triton.jit
def _bits_f32(i):
    return i.to(tl.float32, bitcast=True)


@triton.jit
def _dense_boost_packed(br_ptr, bound_ptr, ui_ptr, qa_ptr, da_ptr, r_ptr, u2_ptr, amax_ptr, tdoc_ptr, vt_ptr, nt_ptr,
                        tiles_s, T, h, cap):
    """fp4q_row_scales.dense_row_boost (docs) on the packed rows: persistent over (head, aligned tile) of the DEVICE tile
    count; the tile's rows vt[0, t] + r, r < vt[1, t] (written there only), its document tdoc[t]: r_q = 2^min(floor(log2
    max(bound[doc] / max(bound_q, 1e-30), 1)), cap), uinv r_q, and the document's max |bf16(Q / r)|, max |bf16(dO / r)|
    max-reduced into amax (zeroed by the caller; non-negative floats order as their int32 bits). The same fp32 ops."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nt = tl.load(nt_ptr)
    r = tl.arange(0, 128)
    for it in range(pid, nt * h, nprog):
        t = it % nt
        hh = it // nt
        pb = tl.load(vt_ptr + t)
        nvr = tl.load(vt_ptr + tiles_s + t) & 0xFFFF
        m = r < nvr
        rows = hh * T + pb + r
        doc = tl.load(tdoc_ptr + t)
        bnd = tl.load(bound_ptr + doc)
        bq = tl.load(br_ptr + rows, mask=m, other=1.0)
        ratio = tl.maximum(_div_rn(bnd, tl.maximum(bq, 1e-30)), 1.0)
        e = tl.minimum(((ratio.to(tl.int32, bitcast=True) >> 23) - 127).to(tl.float32), cap)
        rq = _bits_f32((e.to(tl.int32) + 127) << 23)
        tl.store(r_ptr + rows, rq, mask=m)
        tl.store(u2_ptr + rows, tl.load(ui_ptr + rows, mask=m, other=0.0) * rq, mask=m)
        qa = _div_rn(tl.load(qa_ptr + rows, mask=m, other=0.0), rq).to(tl.bfloat16).to(tl.float32)
        da = _div_rn(tl.load(da_ptr + rows, mask=m, other=0.0), rq).to(tl.bfloat16).to(tl.float32)
        qa = tl.max(tl.where(m, qa, 0.0), axis=0)
        da = tl.max(tl.where(m, da, 0.0), axis=0)
        tl.atomic_max(amax_ptr + 2 * doc, qa.to(tl.int32, bitcast=True))
        tl.atomic_max(amax_ptr + 2 * doc + 1, da.to(tl.int32, bitcast=True))


def dense_row_boost_packed(bound_rows, bound, uinv, query_row_amax, grad_row_amax, cap: float, tile_doc, vt_tile,
                           ntiles):
    """(cov6) fp4q_row_scales.dense_row_boost(tile_doc=) on the packed rows of compact DocMaps: (1, h, T) row vectors per
    packed row, bound (n_docs + 1,) per document. Returns (r_q, uinv r_q, amax ((n_docs + 1) * 2,)): per row bitwise the
    aligned kernel's. A zeroing memset + one persistent launch."""
    for t_ in (bound_rows, uinv, query_row_amax, grad_row_amax):
        assert t_.dtype == torch.float32 and t_.is_contiguous() and t_.shape == bound_rows.shape
    _, h, T = bound_rows.shape
    r = torch.empty_like(bound_rows)
    u2 = torch.empty_like(uinv)
    amax = torch.zeros(2 * bound.numel(), dtype=torch.float32, device=bound.device)
    tiles_s = vt_tile.shape[1]
    nsm = torch.cuda.get_device_properties(bound.device).multi_processor_count
    grid = max(1, min(h * tiles_s, 4 * nsm))
    _dense_boost_packed[(grid,)](bound_rows, bound, uinv, query_row_amax, grad_row_amax, r, u2, amax.view(torch.int32),
                                 tile_doc, vt_tile, ntiles, tiles_s, T, h, float(cap), num_warps=4)
    return r, u2, amax


@triton.jit
def _e4_quant_dense_packed(q_ptr, do_ptr, f_ptr, amax_ptr, y_ptr, ydo_ptr, scl_ptr, dqs_ptr, tdoc_ptr, vt_ptr, nt_ptr,
                           so_do_ptr, bound_ptr, n_docs1, tiles_s, T, h, ss, sh, sm_f32, D: tl.constexpr,
                           BD: tl.constexpr):
    """_e4_quant_dense_docs on the packed rows (persistent over (q / dO, aligned tile, head) of the device tile count):
    the codes E4M3(bf16(x / r_q) / s) at the tile's real packed rows; the dQ post's scale per packed row (dqs, (T,):
    the row's document's dq_scale, written by the head-0 items); program 0 the per-document scl rows."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nt = tl.load(nt_ptr)
    tot = 2 * nt * h
    r = tl.arange(0, 128)
    cols = tl.arange(0, D)
    sm = sm_f32.to(tl.float32)
    for it in range(pid, tot, nprog):
        which = (it >= nt * h).to(tl.int32)
        p = it - which * nt * h
        t = p % nt
        hh = p // nt
        pb = tl.load(vt_ptr + t)
        nvr = tl.load(vt_ptr + tiles_s + t) & 0xFFFF
        rows = pb + r
        m = r < nvr
        off = hh * sh + rows[:, None] * ss + cols[None, :]
        mk = m[:, None]
        fr = tl.load(f_ptr + hh * T + rows, mask=m, other=1.0)[:, None]
        if which == 0:
            x = tl.load(q_ptr + off, mask=mk, other=0.0).to(tl.float32)
        else:
            x = tl.load(do_ptr + off, mask=mk, other=0.0).to(tl.float32)
        x = tl.math.div_rn(x, fr).to(tl.bfloat16).to(tl.float32)
        doc = tl.load(tdoc_ptr + t)
        sc, inv = _e4_scale(tl.load(amax_ptr + 2 * doc + which).to(tl.float32, bitcast=True))
        y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
        if which == 0:
            tl.store(y_ptr + off, y, mask=mk)
            if hh == 0:
                bound = tl.load(bound_ptr + doc).to(tl.float32)
                so_do = tl.load(so_do_ptr + doc).to(tl.float32)
                ds_mul = _div_rn(448.0, tl.maximum(bound, 1e-30))
                dqs = _div_rn(sm * so_do, ds_mul)
                tl.store(dqs_ptr + rows, tl.zeros([128], tl.float32) + dqs, mask=m)
        else:
            tl.store(ydo_ptr + off, y, mask=mk)
    # (spd4) the per-document scl rows split over every program (as _e4_quant_docs_packed)
    for d0 in range(pid * BD, n_docs1, nprog * BD):
        dd = d0 + tl.arange(0, BD)
        md = dd < n_docs1
        sq, _iq = _e4_scale(tl.load(amax_ptr + 2 * dd, mask=md, other=0).to(tl.float32, bitcast=True))
        sdo, _id = _e4_scale(tl.load(amax_ptr + 2 * dd + 1, mask=md, other=0).to(tl.float32, bitcast=True))
        bound = tl.load(bound_ptr + dd, mask=md, other=1.0).to(tl.float32)
        so_do = tl.load(so_do_ptr + dd, mask=md, other=1.0).to(tl.float32)
        ds_mul = _div_rn(448.0, tl.maximum(bound, 1e-30))
        base = sm * so_do
        tl.store(scl_ptr + 4 * dd, _div_rn(base * sq, ds_mul), mask=md)
        tl.store(scl_ptr + 4 * dd + 1, _div_rn(sdo, 448.0), mask=md)
        tl.store(scl_ptr + 4 * dd + 2, ds_mul, mask=md)
        tl.store(scl_ptr + 4 * dd + 3, _div_rn(base, ds_mul), mask=md)


def e4m3_q_do_dense_packed(q: torch.Tensor, dout: torch.Tensor, row_boost: torch.Tensor, softmax_scale: float,
                           amax: torch.Tensor, so_do: torch.Tensor, bound: torch.Tensor, tile_doc_q: torch.Tensor,
                           vt_tile_q: torch.Tensor, ntiles_q: torch.Tensor):
    """(cov6) e4m3_q_do_dense_docs on the packed rows of compact DocMaps: q / dout (1, T, h, 128), row_boost (1, h, T),
    amax / so_do / bound per document. Returns (q8, do8 at the packed rows, scl (n_docs + 1, 4), dq_scale per packed row
    (T,)): per row bitwise the aligned launch. One persistent launch."""
    b, T, h, d = q.shape
    assert b == 1 and dout.shape == q.shape and q.stride() == dout.stride() and q.stride(3) == 1 and d == 128
    n_docs1 = bound.numel()
    assert amax.numel() == 2 * n_docs1 and so_do.numel() == n_docs1 and so_do.dtype == torch.float32
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    do8 = torch.empty_like(dout, dtype=torch.float8_e4m3fn)
    scl = torch.empty(n_docs1, 4, dtype=torch.float32, device=q.device)
    dqs = torch.empty(T, dtype=torch.float32, device=q.device)
    sm_f32 = float(np.float32(softmax_scale))
    tiles_s = vt_tile_q.shape[1]
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = max(1, min(2 * h * tiles_s, 4 * nsm))
    _e4_quant_dense_packed[(grid,)](q, dout, row_boost, amax.view(torch.int32), q8, do8, scl, dqs, tile_doc_q,
                                    vt_tile_q, ntiles_q, so_do, bound, n_docs1, tiles_s, T, h, q.stride(1), q.stride(2),
                                    sm_f32, D=d, BD=128, num_warps=8)
    return q8, do8, scl, dqs

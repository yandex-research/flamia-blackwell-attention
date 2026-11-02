"""The fused all-FP4 backward prep against the torch chains it replaced, bitwise:
  * quant.fp4.quantize_fp4_rows(causal_row_scale=...)  vs fp4q_prepare's former causal torch chain + the dO quantiser
  * quantize_fp4_rows(dense_row_bound=...), output_tensor_max, fp4_global_scale_divisor, output_row_amax
  * fp4q_row_scales.dense_row_boost, fp4q_diag_triton.e4m3_q_do_dense  vs the dense torch chain + quant.fp8's per-tensor E4M3
  * make_transposed_sf_halves_swapped  vs the torch.cat of the two d halves
  * quantize_fp4_causal_diagonal(make_query_row_amax=True)  vs |Q|.amax(-1) and the zeroed slots
  * dq_postprocess_cute.fp4q_postprocess  vs dq_postprocess_cute + (dq / F) + rows_scale_dk_dv

    pytest tests/cute/test_fuseall_prep.py
"""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LOG2_E = math.log2(math.e)
OFF = math.log2(6) + math.log2(448) - 8.0
SHAPES = ((1, 256, 2), (2, 512, 3))


def _inputs(b, s, h, seed, spread=6):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = lambda: torch.exp2(torch.randint(-spread, spread, (b, s, h, 1), generator=g).float())  # noqa: E731
    dout = (torch.randn(b, s, h, 128, generator=g) * sc()).bfloat16().cuda()
    out = (torch.randn(b, s, h, 128, generator=g) * 0.3).bfloat16().cuda()
    lse = (torch.rand(b, h, s, generator=g) * 12 - 2).cuda()
    lse[:, :, :3] = torch.tensor([float("-inf"), 0.0, 30.0])   # a row with no key, and extremes
    frame = torch.floor(lse.clamp(-50, 50) * LOG2_E + torch.randint(-3, 4, (b, h, s), generator=g).float().cuda())
    vmax = torch.rand(b, h, s, generator=g).cuda() * 20
    rmax = torch.exp2(torch.randint(-3, 4, (b, h, s), generator=g).float()).cuda()
    qa = torch.rand(b, h, s, generator=g).cuda() * 5
    return dout, out, lse, frame, vmax, rmax, qa


def _pow2_floor(x):
    # the exact floor (exponent field; the FP64-free kernels' pow2_floor_as_torch / interface_bwd_fp4q.pow2_floor_bits)
    return ((x.float().view(torch.int32) >> 23 & 0xFF) << 23).view(torch.float32)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_causal_row_scale(shape, fold, seed):
    from flash_attn.cute.quant import fp4 as F
    b, s, h = shape
    dout, out, lse, frame, vmax, rmax, qa = _inputs(b, s, h, seed)
    smax = F.SR_SCALE_DIVISOR
    # the former torch chain (interface_bwd_fp4q.fp4q_prepare, causal dS scale)
    lse2 = lse.float() * LOG2_E
    pmax_b = torch.pow(2.0, (frame.float() + 8.0 - lse2).clamp(max=0.0))
    do_n = dout.float().norm(dim=-1).permute(0, 2, 1)
    d_est = (out.float() * dout.float()).sum(-1).permute(0, 2, 1)
    if fold:
        d_est = d_est * rmax
    bound_rows = pmax_b * (do_n * vmax + d_est.abs())
    dmax_q = dout.float().abs().amax(-1).permute(0, 2, 1)
    f = torch.minimum(448.0 / bound_rows.clamp(min=1e-30), (smax * 448.0) / dmax_q.clamp(min=1e-30))
    f = torch.where((bound_rows > 0) & (dmax_q > 0), f, torch.ones_like(f))
    F_row = _pow2_floor(f.clamp(min=2.0 ** -100, max=2.0 ** 100))
    dout_q = (dout.float() * F_row.permute(0, 2, 1)[..., None]).to(dout.dtype)
    ref = F.quantize_fp4_rows(dout_q, rounding="stochastic", stochastic_rounding_seed=7, compute_d_with_output=out)
    uinv = torch.where(torch.isfinite(lse2), torch.pow(2.0, (frame.float() - lse2 - OFF).clamp(max=1000)),
                       torch.zeros_like(lse2))
    q_amax = (qa / F_row).to(torch.bfloat16).float().amax()
    acc = torch.full((b, h, s, 128), 3.0, dtype=torch.float16, device="cuda")
    slots = torch.zeros(2, dtype=torch.float32, device="cuda")
    crs = F.CausalRowScale(lse=lse, frame=frame, key_norm_prefix_max=vmax, query_row_amax=qa, max_slots=slots,
                           top=smax * 448.0, lse_log2_offset=OFF, key_factor_prefix_max=rmax if fold else None,
                           zero_accumulator=acc)
    new = F.quantize_fp4_rows(dout, rounding="stochastic", stochastic_rounding_seed=7, compute_d_with_output=out,
                              causal_row_scale=crs)
    assert torch.equal(new["row_factor"], F_row)
    assert torch.equal(new["uinv"], uinv)
    assert torch.equal(new["codes"], ref["codes"])
    assert torch.equal(new["scale_factors"], ref["scale_factors"])
    assert torch.equal(new["d"], ref["d"])
    assert torch.equal(slots[0], q_amax) and torch.equal(slots[1], dout.float().abs().amax())
    assert not acc.any()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_dense_row_chain(shape, seed):
    from flash_attn.cute.quant import fp4 as F
    from flash_attn.cute.device_scales import div32
    from flash_attn.cute.fp4q_row_scales import dense_row_boost
    from flash_attn.cute.fp4q_diag_triton import e4m3_q_do_dense
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor

    def quantize_e4m3_dev(t):  # the per-tensor E4M3 with a device scale (quant.fp8; was interface_bwd_fp4's)
        r = quantize_fp8_per_tensor(t)
        return r["codes"], r["scales"][0]
    b, s, h = shape
    dout, out, lse, frame, _, _, _ = _inputs(b, s, h, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 5)
    v = torch.randn(b, s, h, 128, generator=g).bfloat16().cuda()
    q = torch.randn(b, s, h, 128, generator=g).bfloat16().cuda()
    smax = F.SR_SCALE_DIVISOR
    c = smax * 448.0
    # the former torch chain (global dS multiplier with the row boost)
    so_do = div32(dout.abs().amax(), c).clamp_min(1e-30)
    ref = F.quantize_fp4_rows(dout, global_scale=so_do.reshape(1), rounding="stochastic", stochastic_rounding_seed=3,
                              compute_d_with_output=out)
    lse2 = lse.float() * LOG2_E
    pmax_b = torch.pow(2.0, (frame.float() + 8.0 - lse2).clamp(max=0.0))
    do_n = dout.float().norm(dim=-1).permute(0, 2, 1)
    v_max = v.float().norm(dim=-1).amax()
    D = ref["d"]
    p = so_do * 1.0
    bound_q = pmax_b * (do_n * v_max + D.abs()) * torch.reciprocal(p.float())
    bound = bound_q.amax()
    uinv = torch.where(torch.isfinite(lse2), torch.pow(2.0, (frame.float() - lse2 - OFF).clamp(max=1000)),
                       torch.zeros_like(lse2))
    e_q = ((bound / bound_q.clamp(min=1e-30)).clamp(min=1.0).view(torch.int32) >> 23 & 0xFF) - 127
    r_q = ((e_q.clamp(max=6) + 127) << 23).view(torch.float32)
    r_bshd = r_q.permute(0, 2, 1)[..., None]
    q8, sq8 = quantize_e4m3_dev((q.float() / r_bshd).to(q.dtype).contiguous())
    do8, sdo8 = quantize_e4m3_dev((dout.float() / r_bshd).to(dout.dtype).contiguous())
    ds_mul = torch.full((), 448.0, dtype=torch.float32, device="cuda") / bound.clamp_min(1e-30)
    sm = 1.0 / math.sqrt(128)
    dk_scale = sm * so_do * 1.0 * sq8 / ds_mul
    dv_scale = sdo8 / torch.full((), 448.0, dtype=torch.float32, device="cuda")
    dq_scale = sm * so_do * 1.0 * 1.0 / ds_mul
    # the fused launches
    vq = F.quantize_fp4_blocks_2d(v, make_transposed_copy=True, transposed_copy_in_kernel_key_order=False,
                                  transposed_copy_layout="batch_head_dim_seq", output_tensor_max="row_norm2")
    assert torch.equal(vq["tensor_max"].sqrt(), v_max.reshape(1))
    qq = F.quantize_fp4_rows(q, output_row_amax=True)
    assert torch.equal(qq["row_amax"], q.float().abs().amax(-1).permute(0, 2, 1))
    r0 = F.quantize_fp4_rows(dout, make_row_copy=False, output_tensor_max="abs", fp4_global_scale_divisor=c)
    assert torch.equal(r0["fp4_global_scale"], so_do.reshape(1))
    acc = torch.full((b, h, s, 128), 3.0, dtype=torch.float16, device="cuda")
    r1 = F.quantize_fp4_rows(dout, global_scale=r0["fp4_global_scale"], rounding="stochastic", stochastic_rounding_seed=3,
                             compute_d_with_output=out,
                             dense_row_bound=F.DenseRowBound(lse=lse, frame=frame, value_norm2_max=vq["tensor_max"],
                                                             lse_log2_offset=OFF, zero_accumulator=acc))
    for n in ("codes", "scale_factors", "d"):
        assert torch.equal(r1[n], ref[n]), n
    assert torch.equal(r1["bound_rows"], bound_q)
    assert torch.equal(r1["tensor_max"], bound_q.amax().reshape(1))
    assert torch.equal(r1["uinv"], uinv)
    assert torch.equal(r1["d_scaled"], D * torch.reciprocal(p.float()))
    assert not acc.any()
    rq, u2, amax = dense_row_boost(r1["bound_rows"], r1["tensor_max"], r1["uinv"], qq["row_amax"], r1["row_amax"], 6.0)
    assert torch.equal(rq, r_q) and torch.equal(u2, uinv * r_q)
    q8n, do8n, scl = e4m3_q_do_dense(q, dout, rq, sm, amax, r0["fp4_global_scale"], r1["tensor_max"])
    assert torch.equal(q8n.view(torch.uint8), q8.view(torch.uint8))
    assert torch.equal(do8n.view(torch.uint8), do8.view(torch.uint8))
    assert torch.equal(scl, torch.stack([dk_scale.reshape(()), dv_scale.reshape(()), ds_mul.reshape(()),
                                         dq_scale.reshape(())]).float())


def test_transposed_sf_halves_swapped():
    from flash_attn.cute.quant import fp4 as F
    k = torch.randn(2, 512, 3, 128, generator=torch.Generator().manual_seed(4)).bfloat16().cuda()
    r = F.quantize_fp4_rows(k, make_transposed_copy=True, make_transposed_sf_halves_swapped=True)
    ref = F.quantize_fp4_rows(k, make_row_copy=False, make_transposed_copy=True)
    assert torch.equal(r["transposed_codes"], ref["transposed_codes"])
    assert torch.equal(r["transposed_scale_factors"], ref["transposed_scale_factors"])
    c = ref["transposed_scale_factors"].permute(6, 5, 2, 4, 0, 1, 3)
    sf1 = torch.cat([c[..., 2:4, :], c[..., 0:2, :]], dim=-2).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    assert torch.equal(r["transposed_scale_factors_halves_swapped"], sf1)


@pytest.mark.parametrize("r_gqa", [1, 2])
def test_diagonal_query_row_amax(r_gqa):
    from flash_attn.cute.quant.fp4_diagonal import quantize_fp4_causal_diagonal
    g = torch.Generator().manual_seed(9)
    k = torch.randn(1, 512, 2, 128, generator=g).bfloat16().cuda()
    v = torch.randn(1, 512, 2, 128, generator=g).bfloat16().cuda()
    q = torch.randn(1, 512, 2 * r_gqa, 128, generator=g).bfloat16().cuda()
    a = quantize_fp4_causal_diagonal(k, v, 0.088, key_channel="fold", query=q, make_key_row_copy=True,
                                     make_key_norm_prefix_maxima=True)
    bq = quantize_fp4_causal_diagonal(k, v, 0.088, key_channel="fold", query=q, make_key_row_copy=True,
                                      make_key_norm_prefix_maxima=True, make_query_row_amax=True)
    for n in a:
        assert torch.equal(a[n], bq[n]), n
    assert torch.equal(bq["query_row_amax"], q.float().abs().amax(-1).permute(0, 2, 1))
    assert not bq["backward_amax_slots"].any()
    assert bq["vtok_bias"].permute(0, 2, 1).is_contiguous()


@pytest.mark.parametrize("device_scale", [False, True])
def test_fp4q_postprocess(device_scale):
    from flash_attn.cute.dq_postprocess_cute import dq_postprocess_cute, fp4q_postprocess
    from flash_attn.cute.fp4q_diag_triton import rows_scale_dk_dv
    g = torch.Generator().manual_seed(2)
    b, s, h, hk = 2, 512, 4, 2
    acc = (torch.randn(b, h, s, 128, generator=g) * 100).half().cuda()
    F_row = torch.exp2(torch.randint(-5, 6, (b, h, s), generator=g).float()).cuda()
    dk = torch.randn(b, s, hk, 128, generator=g).bfloat16().cuda()
    dv = torch.randn(b, s, hk, 128, generator=g).bfloat16().cuda()
    fk = torch.exp2(torch.randint(-5, 6, (b, hk, s), generator=g).float()).cuda()
    fv = torch.exp2(torch.randint(-5, 6, (b, hk, s), generator=g).float()).cuda()
    scale = 0.0123
    sc = torch.full((1,), scale, dtype=torch.float32, device="cuda") if device_scale else scale
    ref_dq = (dq_postprocess_cute(acc, s, float(torch.tensor(scale, dtype=torch.float32)) if device_scale else scale,
                                  torch.bfloat16).float() / F_row.permute(0, 2, 1)[..., None]).to(torch.bfloat16)
    dk_r, dv_r = dk.clone(), dv.clone()
    rows_scale_dk_dv(dk_r, dv_r, fk, fv)
    dq = fp4q_postprocess(acc, s, sc, torch.bfloat16, row_factor=F_row, dk=dk, dv=dv, dk_row_factor=fk, dv_row_factor=fv)
    assert torch.equal(dq, ref_dq)
    assert torch.equal(dk, dk_r) and torch.equal(dv, dv_r)

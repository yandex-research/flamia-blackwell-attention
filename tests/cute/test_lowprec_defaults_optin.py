"""The opt-ins of the default-decision evidence:

- options.sr_ds: stochastic rounding of the E4M3 dS-hat now also under the block-scaled dS (the default), on the 1-CTA
  kernel (the plan routes an sr_ds call there); before, the block-scaled path rounded to nearest whatever sr_ds said.
- KernelTuning.bwd_fp4q_dk_sr_blk32: SR of the all-4-bit backward's dK-side dS-hat under the default blk32 dS
  (sr_dk_operand, default True, reaches only the per-tensor dS).
- KernelTuning.fp4_qk_sf_vec = 32: MXFP4 Q / K (the S operand, UE8M0 per 32) on pv_precision="fp8", backward="e4m3";
  not API (rule E40 refuses what it does not serve).
The defaults are untouched: sr_ds=False and fp4_qk_sf_vec=None resolve and run exactly as before.
"""
import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, KernelTuning, flash_attn_fp4
from flash_attn.cute.lowprec.plan import resolve_fp4

BF16 = torch.bfloat16
GPU = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
needs_gpu = pytest.mark.skipif(not GPU, reason="NVFP4 kernels need SM 10.x")


def _m(b=1, s=1024, h=4, d=128, rg=True):
    t = torch.empty(b, s, h, d, dtype=BF16, device="meta", requires_grad=rg)
    return t, t, t


def test_sr_ds_takes_the_1cta_backward():
    p = resolve_fp4(*_m(), causal=True, options=FP4AttnOptions(sr_ds=True))
    assert p.bwd.kernel == "1cta" and p.bwd.sr_ds and p.specs.bwd1.ds_sr
    assert any("sr_ds" in r for f, v, r in p.resolution_log if f == "two_cta"), p.log()
    with pytest.raises(ValueError):
        resolve_fp4(*_m(), causal=True, options=FP4AttnOptions(sr_ds=True, two_cta="on"))
    assert resolve_fp4(*_m(), causal=True).bwd.kernel == "2cta"   # the default route is unchanged


def test_dk_sr_blk32_reaches_the_spec():
    ALL4 = dict(pv_precision="fp4", backward="fp4")
    p = resolve_fp4(*_m(), causal=True, options=FP4AttnOptions(tuning=KernelTuning(bwd_fp4q_dk_sr_blk32=True)), **ALL4)
    assert p.specs.bwdq.dk_sr_blk32
    assert not resolve_fp4(*_m(), causal=True, **ALL4).specs.bwdq.dk_sr_blk32


def test_mxfp4_qk_resolves_and_refuses():
    mx = FP4AttnOptions(tuning=KernelTuning(fp4_qk_sf_vec=32))
    p = resolve_fp4(*_m(), causal=True, options=mx)
    assert p.specs.quant.s_sf_vec == 32 and not p.specs.quant.qk_hadamard
    assert ("qk_hadamard", "off") in [(f, v) for f, v, _ in p.resolution_log]
    assert resolve_fp4(*_m(), causal=True).specs.quant.s_sf_vec == 16
    for kw in (dict(qk_hadamard=True), dict(window_size=(128, 0)), dict(pv_precision="fp4", backward="fp4")):
        with pytest.raises(ValueError) as ei:
            resolve_fp4(*_m(), causal=True, options=mx, **kw)
        assert ei.value.rule == "E40", (kw, ei.value)


def _ref(q, k, v, g, causal):
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    p = s.softmax(-1)
    o = p @ vf
    go = g.float().transpose(1, 2)
    dp = go @ vf.transpose(-1, -2)
    ds = p * (dp - (go * o).sum(-1, keepdim=True))
    dq = ds @ kf / q.shape[-1] ** 0.5
    dk = ds.transpose(-1, -2) @ qf / q.shape[-1] ** 0.5
    dv = p.transpose(-1, -2) @ go
    return [t.transpose(1, 2) for t in (o, dq, dk, dv)]


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def _step(q0, k0, v0, g, causal, opts):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q0, k0, v0))
    o = flash_attn_fp4(q, k, v, causal=causal, options=opts)
    return (o, *torch.autograd.grad(o, (q, k, v), g))


@needs_gpu
@pytest.mark.parametrize("causal", [True, False])
def test_sr_ds_rounds_the_block_scaled_ds(causal):
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(2, 1024, 4, 128, device="cuda", dtype=BF16) for _ in range(4))
    rtn = _step(q, k, v, g, causal, FP4AttnOptions(two_cta="off"))
    sr1 = _step(q, k, v, g, causal, FP4AttnOptions(sr_ds=True))
    sr2 = _step(q, k, v, g, causal, FP4AttnOptions(sr_ds=True))
    assert torch.equal(rtn[0], sr1[0]) and torch.equal(rtn[3], sr1[3])   # O and dV do not read dS
    assert not torch.equal(rtn[2], sr1[2]) and not torch.equal(sr1[2], sr2[2])   # a fresh dither per call
    ref = _ref(q, k, v, g, causal)
    for i in (1, 2):
        assert _cos(sr1[i], ref[i]) > 0.98, i
    # unbiased: the mean of 8 SR calls is closer to fp32 than one RTN call
    acc = None
    for _ in range(8):
        r = _step(q, k, v, g, causal, FP4AttnOptions(sr_ds=True))
        acc = [x.float() for x in r] if acc is None else [a + x.float() for a, x in zip(acc, r)]
    assert _cos(acc[2] / 8, ref[2]) > _cos(rtn[2], ref[2])


@needs_gpu
def test_dk_sr_blk32_changes_only_dk():
    """dk_sr_blk32 runs and stays accurate. (The all-4-bit SR seed is a device counter advanced per call, so two calls in
    one process differ in their dO / dS dither anyway: only O, which reads no dither, is compared bitwise.)"""
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(2, 1024, 4, 128, device="cuda", dtype=BF16) for _ in range(4))
    ALL4 = dict(pv_precision="fp4", backward="fp4")

    def run(opts):
        qq, kk, vv = (t.detach().clone().requires_grad_() for t in (q, k, v))
        o = flash_attn_fp4(qq, kk, vv, causal=True, options=opts, **ALL4)
        return (o, *torch.autograd.grad(o, (qq, kk, vv), g))
    d = run(FP4AttnOptions())
    s = run(FP4AttnOptions(tuning=KernelTuning(bwd_fp4q_dk_sr_blk32=True)))
    assert torch.equal(d[0], s[0])
    assert all(torch.isfinite(t).all() for t in s)
    ref = _ref(q, k, v, g, True)
    assert _cos(s[2], ref[2]) > 0.95


@needs_gpu
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("two_cta", ["on", "off"])
def test_mxfp4_qk_step(causal, two_cta):
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(2, 1024, 4, 128, device="cuda", dtype=BF16) for _ in range(4))
    mx = _step(q, k, v, g, causal, FP4AttnOptions(two_cta=two_cta, tuning=KernelTuning(fp4_qk_sf_vec=32)))
    nv = _step(q, k, v, g, causal, FP4AttnOptions(two_cta=two_cta))
    ref = _ref(q, k, v, g, causal)
    for i in range(4):
        assert torch.isfinite(mx[i]).all()
        assert _cos(mx[i], ref[i]) > 0.95, i
        assert not torch.equal(mx[i], nv[i])   # the S operand changed format

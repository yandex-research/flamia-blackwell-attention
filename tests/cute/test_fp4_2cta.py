r"""2-CTA low-precision backward (flash_bwd_sm100_2cta8) on B200 (SM 10.x).

Gates: the 2-CTA dk8 kernel (NVFP4 S, E4M3 dP / dV / dK / dQ, cta_group::2) equals the 1-CTA dk8 kernel on identical operands
and d\hat{S} scale (cosine >= 0.99999 on dQ / dK / dV: the dQ accumulation order differs, everything else is the same arithmetic);
the all-E4M3 2-CTA variant tracks BF16 (>= 0.99 on N(0,1) inputs); the plan routes by shape (options.two_cta "auto") and
the backward driver follows its decision. The drivers run with the specs the plan builds for the same call
(resolve_fp4 / resolve_fp8(...).specs); options.two_cta "on" / "off" pins the kernel.
Run:  pytest -x tests/cute/test_fp4_2cta.py
"""
import dataclasses

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, FP8AttnOptions, resolve_fp4, resolve_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="NVFP4 kernels need SM 10.x"
)

D = 128


def _e4m3_host_scale(x):
    """Per-tensor E4M3 of x and its scale as a host float (quant.fp8.quantize_fp8_per_tensor; one host read, as the
    harness always took): s = max(amax / 448, 1e-12)."""
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    r = quantize_fp8_per_tensor(x)
    return r["codes"], r["scales"][0].item()


def _meta(*shape):
    return torch.empty(*shape, dtype=torch.bfloat16, device="meta").requires_grad_(True)


def _plan(entry, q_shape, k_shape, causal, deterministic=False, qk_hadamard=None, **opts):
    """The plan of a flash_attn_fp4 / flash_attn_fp8 call of these shapes and options (meta tensors: no GPU work)."""
    resolve, O = (resolve_fp4, FP4AttnOptions) if entry == "fp4" else (resolve_fp8, FP8AttnOptions)
    q, k = _meta(*q_shape), _meta(*k_shape)
    return resolve(q, k, k, causal=causal, deterministic=deterministic, qk_hadamard=qk_hadamard,
                   options=O(**opts) if opts else None)


def _specs(entry, q, causal, two_cta):
    """The specs for the per-tensor E4M3 codes _case hands over (scaling-consistency rule; unrotated: qk_hadamard off)."""
    return _plan(entry, q.shape, q.shape, causal, qk_hadamard=False, operand_scaling="per_tensor", two_cta=two_cta).specs


def _cos(a, r):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), r.float().flatten(), dim=0).item()


def _case(b, s, h, causal, seed=0):
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.interface_bwd_fp4 import _pmax_bound

    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(b, s, h, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4))
    out, lse = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)[:2]
    sc = {}
    for nm, t in (("q8", q), ("k8", k), ("v8", v), ("do8", do)):
        sc[nm], sc["s" + nm[:-1]] = _e4m3_host_scale(t)
    dsum = (out.float() * do.float()).sum(-1).permute(0, 2, 1)
    sc["s_ds"] = _pmax_bound(q, k, do, v, lse, dsum, 1.0 / D ** 0.5, causal) / 448.0
    return q, k, v, out, do, lse, sc


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,s,h", [(1, 4096, 4), (2, 2048, 3)])
def test_2cta_dk8_equals_1cta(b, s, h, causal):
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4, _use_2cta_dk8

    q, k, v, out, do, lse, sc = _case(b, s, h, causal)
    route = lambda sp: _use_2cta_dk8(q, k, causal, False, 128, False, None, "all", None, None, None, None,  # noqa: E731
                                     quant=sp.quant, host=sp.host)
    sp1, sp2 = _specs("fp4", q, causal, "off"), _specs("fp4", q, causal, "on")
    assert not route(sp1)
    ref = _flash_attn_bwd_fp4(q, k, v, out, do, lse, causal=causal, fp8_scales=dict(sc), specs=sp1)
    assert route(sp2), "two_cta='on' does not route this shape to the 2-CTA kernel"
    new = _flash_attn_bwd_fp4(q, k, v, out, do, lse, causal=causal, fp8_scales=dict(sc), specs=sp2)
    for nm, a, r in zip(("dq", "dk", "dv"), new, ref):
        assert torch.isfinite(a.float()).all(), nm
        assert _cos(a, r) >= 0.99999, (nm, _cos(a, r))


@pytest.mark.parametrize("causal", [False, True])
def test_2cta_e4m3_tracks_bf16(causal):
    from flash_attn.cute.interface import _flash_attn_bwd
    from flash_attn.cute.interface_bwd_2cta8 import flash_attn_bwd_2cta8

    q, k, v, out, do, lse, sc = _case(1, 4096, 4, causal, seed=1)
    ref = _flash_attn_bwd(q, k, v, out, do, lse, causal=causal)
    sp = _specs("fp8", q, causal, "on")   # the all-E4M3 variant is the 8-bit route's 2-CTA kernel
    new = flash_attn_bwd_2cta8(q, k, v, out, do, lse, causal=causal, fp8_scales=dict(sc), nvfp4_s=False, spec=sp.bwd2,
                               quant=sp.quant, host=sp.host)
    for nm, a, r in zip(("dq", "dk", "dv"), new, ref):
        assert _cos(a, r) >= 0.99, (nm, _cos(a, r))


def test_2cta_dispatch_rule():
    """The 1-CTA / 2-CTA rule by shape, as the plan resolves it (lowprec.plan._resolve_two_cta; options.two_cta "auto"),
    and the backward driver's routing (_use_2cta_dk8) follows the plan's decision on every case."""
    from flash_attn.cute.interface_bwd_fp4 import _use_2cta_dk8

    def two(entry, s, causal, h=4, hkv=4, det=False, **opts):
        plan = _plan(entry, (1, s, h, D), (1, s, hkv, D), causal, deterministic=det, **opts)
        q, k = _meta(1, s, h, D), _meta(1, s, hkv, D)
        drv = _use_2cta_dk8(q, k, causal, det, 128, False, None, "all", None, None, None, None,
                            quant=plan.specs.quant, host=plan.specs.host)
        assert drv == (plan.bwd.kernel == "2cta"), (entry, s, causal, h, hkv, det, opts, plan.bwd.kernel, drv)
        return plan.bwd.kernel == "2cta"

    # per-tensor 4-bit operands: s >= 16384, or dense >= 8192
    pt = dict(operand_scaling="per_tensor")
    assert two("fp4", 32768, True, **pt)
    assert two("fp4", 8192, False, **pt)
    assert not two("fp4", 8192, True, **pt)
    assert not two("fp4", 4096, False, **pt)
    assert not two("fp4", 16384, True, h=32, hkv=8, **pt)  # GQA stays on the 1-CTA kernel below key length 32768
    assert two("fp4", 32768, True, h=32, hkv=8, **pt)      # ... and takes the pair from it (the long-sequence rule)
    # token-block operands (the default): the 2-CTA kernel wherever eligible (3-6 % faster at s2048-8192)
    assert two("fp4", 32768, True)
    assert two("fp4", 8192, True)
    assert two("fp4", 4096, False)
    assert two("fp4", 2048, True)
    assert not two("fp4", 16384, True, h=32, hkv=8)  # GQA stays on the 1-CTA kernel below key length 32768
    assert two("fp4", 32768, True, h=32, hkv=8)
    # deterministic takes the 1-CTA kernel (dQ issue / release warps) at every length (cta2: the 2-CTA ordered dQ
    # reduce measured no faster from s_k 32768, so causal s_k >= 32768 moved to the 1-CTA kernel)
    assert not two("fp4", 4096, True, det=True)
    assert not two("fp4", 32768, False, det=True)
    assert not two("fp4", 32768, True, det=True)
    # by design: deterministic 8-bit PER-TENSOR operands keep the 2-CTA kernel (the 1-CTA kernel differs there in dK);
    # 8-bit token blocks follow the 1-CTA rule above
    assert two("fp8", 4096, True, det=True, **pt)
    assert two("fp8", 2048, False, det=True, **pt)
    assert two("fp8", 32768, True, det=True, **pt)
    assert not two("fp8", 4096, True, h=32, hkv=8, det=True, **pt)  # GQA: ineligible
    assert not two("fp8", 4096, True, det=True)
    assert not two("fp8", 32768, False, det=True)
    assert not two("fp8", 32768, True, det=True)
    assert two("fp8", 4096, True, h=32, hkv=8, det=True, two_cta="on")  # deterministic GQA on the pair: opt-in
    # the same split with and without the Q/K Hadamard (deterministic MHA d128: 1-CTA on token blocks, both entries,
    # and on 4-bit per-tensor operands; 2-CTA on 8-bit per-tensor operands)
    for causal in (True, False):
        for had in (None, False):
            assert two("fp8", 4096, causal, det=True, qk_hadamard=had, **pt)
            assert not two("fp8", 4096, causal, det=True, qk_hadamard=had)
            assert not two("fp4", 4096, causal, det=True, qk_hadamard=had)
            assert not two("fp4", 4096, causal, det=True, qk_hadamard=had, **pt)
    # explicit choices
    assert not two("fp4", 32768, True, two_cta="off")
    assert two("fp4", 4096, True, two_cta="on")  # "always when eligible"
    with pytest.raises(ValueError) as ei:
        _plan("fp4", (1, 4096, 4, 64), (1, 4096, 4, 64), True, two_cta="on")   # head_dim 64: the 2-CTA kernel is d = 128
    assert ei.value.rule == 15


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pv", ["mxfp4", "e4m3"])
def test_mxfp4_forward(pv, causal):
    """MXFP4 QK (E2M1 + UE8M0 per 32) with MXFP4 or per-tensor E4M3 PV against the BF16 forward: O cosine and the LSE the
    backward consumes (the split pipeline folds an integer frame for the UE8M0 group SFs). The MXFP4 formats are not
    part of the public API; their forward-kernel paths stay in the tree and this is the
    driver-level probe of them, with the default 4-bit plan's forward specs."""
    from flash_attn.cute.interface import _flash_attn_fwd as _bf16_fwd
    from flash_attn.cute.interface_fwd_fp4 import _flash_attn_fwd  # the block-scaled driver: mSFQ / mSFK / mSFV
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows

    b, s, h = 1, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(b, s, h, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    o_ref, lse_ref = _bf16_fwd(q, k, v, causal=causal, return_lse=True)[:2]
    rq, rk = quantize_fp4_rows(q, scale_group_size=32), quantize_fp4_rows(k, scale_group_size=32)
    q4, sfq, k4, sfk = rq["codes"], rq["scale_factors"], rk["codes"], rk["scale_factors"]
    e2 = torch.float4_e2m1fn_x2
    sp = _plan("fp4", q.shape, k.shape, causal, qk_hadamard=False).specs
    kw = dict(mSFQ=sfq, mSFK=sfk, qk_sf_vec_size=32, causal=causal, return_lse=True, spec=sp.fwd, host_spec=sp.host,
              quant_spec=sp.quant)
    if pv == "mxfp4":
        rv = quantize_fp4_rows(v, scale_group_size=32, make_row_copy=False, make_transposed_copy=True,
                               transposed_copy_in_kernel_key_order=False, transposed_copy_layout="batch_head_dim_seq")
        v4, sfv = rv["transposed_codes"].view(torch.float4_e2m1fn_x2), rv["transposed_scale_factors"]
        if causal:
            # a causal 4-bit PV with key-grouped V scales leaks future V rows: refused by default (QuantSpec
            # causal_safe_v "auto"), reachable only through the explicit opt-out ("never", what causal_safety="off"
            # sets), which is what this numerics check runs
            with pytest.raises(NotImplementedError, match="future V rows"):
                _flash_attn_fwd(q4.view(e2), k4.view(e2), v4, mSFV=sfv, pv_sf_vec_size=32, **kw)
            kw["quant_spec"] = dataclasses.replace(sp.quant, causal_safe_v="never")
            o, lse = _flash_attn_fwd(q4.view(e2), k4.view(e2), v4, mSFV=sfv, pv_sf_vec_size=32, **kw)[:2]
        else:
            o, lse = _flash_attn_fwd(q4.view(e2), k4.view(e2), v4, mSFV=sfv, pv_sf_vec_size=32, **kw)[:2]
    else:
        v8, sv = _e4m3_host_scale(v)
        desc = torch.full((b, h), float(sv), device="cuda", dtype=torch.float32)
        o, lse = _flash_attn_fwd(q4.view(e2), k4.view(e2), v8, v_descale=desc, **kw)[:2]
    assert torch.isfinite(o.float()).all() and torch.isfinite(lse.float()).all()
    assert _cos(o, o_ref) >= 0.96, _cos(o, o_ref)
    assert (lse.float() - lse_ref.float()).abs().mean().item() <= 0.05


@pytest.mark.parametrize("causal", [False, True])
def test_2cta_training_scales(causal):
    r"""The training step hands the backward a consistent D (fp8_scales["dpsum"], from exactly the E4M3 dO) and the forward's
    per-query P_max instead of a precomputed d\hat{S} scale: the 2-CTA path must consume both like the 1-CTA kernel."""
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4

    b, s, h = 1, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(3)
    q, k, v, do = (torch.randn(b, s, h, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4))
    out, lse = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)[:2]
    sc = {}
    for nm, t in (("q8", q), ("k8", k), ("v8", v), ("do8", do)):
        sc[nm], sc["s" + nm[:-1]] = _e4m3_host_scale(t)
    # D from the E4M3 dO (consistent with the dP GEMM) and the forward's per-query P_max, as the training step passes them
    sc["dpsum"] = (out.float() * sc["do8"].float().mul(sc["sdo"])).sum(-1).permute(0, 2, 1).contiguous()
    # per-query P_max = exp(max_k S_qk - LSE_q), what the forward exports in the training step (host, chunked)
    scale = 1.0 / D ** 0.5
    pm = torch.empty(b, h, s, device="cuda")
    kk = k.float().permute(0, 2, 1, 3)
    for i in range(0, s, 1024):
        qi = q[:, i:i + 1024].float().permute(0, 2, 1, 3)
        S = torch.einsum("bhcd,bhkd->bhck", qi, kk) * scale
        if causal:
            qpos = torch.arange(i, i + qi.shape[2], device="cuda")[:, None]
            S = S.masked_fill(torch.arange(s, device="cuda")[None, :] > qpos, float("-inf"))
        pm[:, :, i:i + 1024] = (S.amax(-1) - lse[:, :, i:i + 1024].float()).exp()
    sc["pmax"] = pm
    outs = {}
    for cta in ("off", "on"):
        outs[cta] = _flash_attn_bwd_fp4(q, k, v, out, do, lse, causal=causal, fp8_scales=dict(sc),
                                        specs=_specs("fp4", q, causal, cta))
    for nm, a, r in zip(("dq", "dk", "dv"), outs["on"], outs["off"]):
        assert torch.isfinite(a.float()).all(), nm
        assert _cos(a, r) >= 0.9999, (nm, _cos(a, r))

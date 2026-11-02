"""All-FP4 attention step on B200 (SM 10.x): every attention matmul NVFP4 in both passes (interface_bwd_fp4q).

forward (flash_attn_fp4(pv_precision="fp4", backward="fp4")): S = Q^K^T, O = P^V^ (1x16 Direct-P P^, integer frames
exported); backward (query-row kernel): S, dP = dO^V^T, dQ = dS^K^T NVFP4; dK = dS^^T Q, dV = P^T dO per-tensor E4M3
(options.fp4_backward_dkdv="e4m3", default) or NVFP4 ("fp4"). Stabilisers: options.all_fp4_stabiliser = "sr" (default)
| "eden_rht" | "rtn".

Checks: the kernel equals its torch reference (tests/cute/_fp4q_ref.py, RTN dS^; the shipped operand modes: causal dS
scale, query-row boost, MXDS blk32 dK operand, key-local V under a causal mask); cosine gates against the BF16 kernels
(random inputs); the softmax invariant sum_k dK = 0; in-kernel SR is unbiased; the head-dim Hadamard round trip;
determinism for a fixed seed; a short training run stays finite and tracks BF16.
The kernel-level checks call the route's drivers with the specs the plan builds for the same call (resolve_fp4(...).specs);
the accuracy gates and the training run go through the public flash_attn_fp4. Nothing reads the environment.
Run:  pytest -x tests/cute/test_fp4_all.py   (-m "not slow" skips the training run)
"""
import math
import os
import sys

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, flash_attn_fp4, resolve_fp4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fp4q_ref  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="NVFP4 kernels need SM 10.x"
)

D = 128
# measured on random N(0, 1) inputs, minus ~0.01 margin
GATES = {
    "rtn": dict(o=0.97, dq=0.94, dk=0.94, dv=0.955),
    "sr": dict(o=0.97, dq=0.91, dk=0.91, dv=0.95),
    "eden_rht": dict(o=0.97, dq=0.915, dk=0.915, dv=0.955),
}


def _cos(a, r):
    return torch.nn.functional.cosine_similarity(a.double().flatten(), r.double().flatten(), dim=0).item()


def _inputs(b, s, h, hkv, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(b, s, n, D, device="cuda", generator=g).to(torch.bfloat16) for n in (h, hkv, hkv))
    do = torch.randn(b, s, h, D, device="cuda", generator=g).to(torch.bfloat16)
    return q, k, v, do


def _specs(q, k, causal, **opts):
    """The specs the plan builds for this all-4-bit call (the drivers' explicit spec arguments). Causal calls take the key-local
    V on every tile (causal_safety="key_local"), the configuration _fp4q_ref models; the public default under causal is its
    diagonal-only form (test_fp4_gqa.py::test_fp4all_diag_gqa_equals_repeated_mha, test_exact_mix.py)."""
    m = lambda t: torch.empty(t.shape, dtype=t.dtype, device="meta").requires_grad_(True)  # noqa: E731
    return resolve_fp4(m(q), m(k), m(k), causal=causal, pv_precision="fp4", backward="fp4",
                       causal_safety="key_local" if causal else "auto",
                       options=FP4AttnOptions(**opts) if opts else None).specs


def _bkw(sp):
    return dict(spec=sp.bwdq, qspec=sp.quant, fspec=sp.fwd)


# Kernel vs _fp4q_ref.reference. Re-derived 2026-10-01 from 6 seeds x the 8 cases of the test below (both dK / dV
# modes, dense and causal, MHA and GQA), shipped defaults, RTN dS^ / dO^:
#   * the bulk: the kernel's f32 / f16 / rcp.approx arithmetic against the float64 reference gives, on the rows left
#     after dropping the worst 1 % (min 2), 1 - cos <= 2.6e-5 (the all-FP4 route's vtok dV, whose operand is the f16 m;
#     every other output <= 5e-6);
#   * rare knife-edge flips: an E2M1 value within ~1e-4 of a rounding midpoint lands on the other grid point in the
#     kernel. Seed 0, all-FP4, causal, s512: dS'/sf = 4.99968 on ONE element of query row 2 -> kernel 6, reference 4; on
#     this tensor that single flip is whole-tensor 1 - cos = 3.1e-4 and norm ratio - 1 = 1.4e-3, all in one dQ row and
#     the dK row of its key (the float32-exp2 variant of the reference moves nothing: it is the dS' rounding itself).
# So the gate is two-part: a tight cosine on the trimmed rows (4x the worst measured), which a wrong operand mode or scale
# misses by orders of magnitude (the stale reference was 1 - cos = 7e-2 on the causal cases), and a whole-tensor
# cosine / norm bound (3x the worst measured) that a handful of knife-edge flips pass.
REF_TRIM_FRAC = 0.01
REF_TRIM_1MCOS = 1e-4
REF_1MCOS = 1e-3
REF_NORM = 5e-3


def _trimmed_1mcos(a, r, frac=REF_TRIM_FRAC):
    """1 - cos over the rows (every index but head_dim) left after dropping the max(2, frac) rows of largest error."""
    a2, r2 = a.double().reshape(-1, a.shape[-1]), r.double().reshape(-1, r.shape[-1])
    e = (a2 - r2).pow(2).sum(-1)
    keep = torch.ones_like(e, dtype=torch.bool)
    keep[torch.topk(e, max(2, int(frac * e.numel()))).indices] = False
    return 1 - _cos(a2[keep], r2[keep])


@pytest.mark.parametrize("dkv", ["e4m3", "fp4"])
@pytest.mark.parametrize("b,s,h,hkv,causal", [(1, 256, 2, 2, False), (1, 512, 4, 2, True), (2, 512, 2, 1, False), (1, 1024, 4, 4, True)])
def test_fp4all_bwd_matches_reference(b, s, h, hkv, causal, dkv):
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, fp4q_prepare, fp4q_run

    q, k, v, do = _inputs(b, s, h, hkv)
    sp = _specs(q, k, causal, fp4_backward_dkdv=dkv)
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, causal=causal, specs=sp)  # causal: the key-local V
    prep = fp4q_prepare(q, k, v, out, do, lse, frame, ops, causal=causal, do_rnd="rtn", ds_sr=False, dkv=dkv, **_bkw(sp))
    dq, dk, dv = fp4q_run(prep)
    ref = _fp4q_ref.reference(q, k, ops, prep, frame, lse, 1 / math.sqrt(D), causal, sp.bwdq)
    for name, a in (("dq", dq), ("dk", dk), ("dv", dv)):
        r = ref[name]
        c, nr, tc = _cos(a, r), a.float().norm().item() / r.norm().item(), _trimmed_1mcos(a, r)
        msg = f"{name}: 1-cos {1 - c:.2e}, trimmed 1-cos {tc:.2e}, norm ratio {nr:.6f}"
        assert tc <= REF_TRIM_1MCOS, msg
        assert 1 - c <= REF_1MCOS and abs(nr - 1) <= REF_NORM, msg


@pytest.mark.parametrize("mode", ["rtn", "sr", "eden_rht"])
@pytest.mark.parametrize("b,s,h,hkv,causal", [(1, 1024, 8, 2, True), (2, 512, 4, 4, False)])
def test_fp4all_gates_vs_bf16(mode, b, s, h, hkv, causal):
    """The public all-4-bit step at its shipped configuration (causal: the diagonal-only key-local V) per stabiliser."""
    from flash_attn.cute import flash_attn_func

    q, k, v, do = _inputs(b, s, h, hkv, seed=1)
    opts = FP4AttnOptions(all_fp4_stabiliser=mode)
    if causal and mode == "eden_rht":
        # eden_rht rotates Q / K / V / dO jointly, which neither causal-safe V payload (diagonal-only, rule 10; every
        # tile, rule 31) can share: refused by the plan, not served
        for cs, rule in (("auto", 10), ("key_local", 31)):
            with pytest.raises((NotImplementedError, ValueError)) as ei:
                flash_attn_fp4(*(t.clone().requires_grad_() for t in (q, k, v)), causal=causal, pv_precision="fp4",
                               backward="fp4", causal_safety=cs, options=opts)
            assert getattr(ei.value, "rule", None) == rule, (cs, ei.value)
        return
    ref = [t.clone().requires_grad_() for t in (q, k, v)]
    o_ref = flash_attn_func(*ref, causal=causal)
    o_ref = o_ref[0] if isinstance(o_ref, tuple) else o_ref
    o_ref.backward(do)
    ts = [t.clone().requires_grad_() for t in (q, k, v)]
    o = flash_attn_fp4(*ts, causal=causal, pv_precision="fp4", backward="fp4", options=opts)
    o.backward(do)
    g = GATES[mode]
    assert _cos(o, o_ref) >= g["o"]
    for name, i in (("dq", 0), ("dk", 1), ("dv", 2)):
        assert torch.isfinite(ts[i].grad.float()).all(), name
        assert _cos(ts[i].grad, ref[i].grad) >= g[name], name


@pytest.mark.parametrize("causal", [False, True])
def test_fp4all_softmax_consistent(causal):
    """dS rows sum to zero (w and D from the same P^ as the forward) -> sum over keys of dK vanishes."""
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, _flash_attn_bwd_fp4q

    q, k, v, do = _inputs(1, 1024, 4, 4, seed=2)
    sp = _specs(q, k, causal)
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, causal=causal, specs=sp)
    _, dk, _ = _flash_attn_bwd_fp4q(q, k, v, out, do, lse, frame, ops, causal=causal, do_rnd="sr", ds_sr=True, seed=3,
                                    **_bkw(sp))
    dk = dk.float()
    inv = (dk.sum(1).norm() / dk.norm()).item() / math.sqrt(dk.shape[1])
    assert inv < 0.03


def test_fp4all_sr_unbiased():
    """mean over seeds of the SR dQ converges to the gradient of the unquantised dS (same P^, dP, K^T): 1 - cos ~ 1 / n."""
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, fp4q_prepare, fp4q_run

    q, k, v, do = _inputs(1, 256, 2, 2, seed=4)
    sp = _specs(q, k, False)
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, specs=sp)
    acc, means, exact = None, {}, None
    for n in range(1, 17):
        prep = fp4q_prepare(q, k, v, out, do, lse, frame, ops, do_rnd="rtn", ds_sr=True, seed=n * 7919, **_bkw(sp))
        dq = fp4q_run(prep)[0]
        if exact is None:
            # the unquantised dS' through the same K^T operand, scales and per-row factors as the kernel's dQ
            exact = _fp4q_ref.reference(q, k, ops, prep, frame, lse, 1 / math.sqrt(D), False, sp.bwdq)["dq_exact"]
        acc = dq.double() if acc is None else acc + dq.double()
        means[n] = acc / n
    e1, e16 = 1 - _cos(means[1], exact), 1 - _cos(means[16], exact)
    assert e1 > 0.01  # SR noise is present in one draw
    assert e16 < e1 / 6  # unbiased: the error of the mean shrinks ~1 / n (measured 0.042 -> 0.0023)
    assert abs(means[16].norm().item() / exact.norm().item() - 1) < 0.01


def test_fp4all_hadamard_roundtrip():
    from flash_attn.cute.nvfp4_2d_triton import fwht_rows, fwht_torch, hadamard_signs

    x = torch.randn(2, 256, 3, D, device="cuda", dtype=torch.bfloat16)
    sg = hadamard_signs(D, 4321, x.device)
    r = fwht_rows(x, sg, dtype=torch.float32)
    assert torch.allclose(r, fwht_torch(x.float() * sg), atol=1e-5)
    assert torch.allclose(fwht_rows(r, sg, inverse=True), x.float(), atol=1e-5)


def test_fp4all_deterministic():
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, _flash_attn_bwd_fp4q

    q, k, v, do = _inputs(1, 1024, 4, 2, seed=5)
    sp = _specs(q, k, True)
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, causal=True, specs=sp)
    runs = [_flash_attn_bwd_fp4q(q, k, v, out, do, lse, frame, ops, causal=True, do_rnd="sr", ds_sr=True, seed=11,
                                 **_bkw(sp)) for _ in range(3)]
    for r in runs[1:]:
        for a, b_ in zip(runs[0], r):
            assert torch.equal(a, b_)


def _toy_losses(mode, steps=150):
    """2-layer causal transformer (width 4 x 128, 4 heads) on a copy-the-previous-token task, which needs attention."""
    import torch.nn as nn
    import torch.nn.functional as F

    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    B, S, HQ, V, L = 8, 256, 4, 64, 2
    W = HQ * D

    def attn(q, k, v):
        if mode == "fp4all":
            return flash_attn_fp4(q, k, v, causal=True, pv_precision="fp4", backward="fp4")
        o = flash_attn_func(q, k, v, causal=True)
        return o[0] if isinstance(o, tuple) else o

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(W), nn.LayerNorm(W)
            self.qkv = nn.Linear(W, 3 * W, bias=False)
            self.proj = nn.Linear(W, W, bias=False)
            self.mlp = nn.Sequential(nn.Linear(W, 2 * W), nn.GELU(), nn.Linear(2 * W, W))

        def forward(self, x):
            q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, HQ, D).unbind(2)
            a = attn(*(t.to(torch.bfloat16).contiguous() for t in (q, k, v)))
            x = x + self.proj(a.reshape(B, S, W).to(x.dtype))
            return x + self.mlp(self.ln2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, W)
            self.pos = nn.Parameter(torch.randn(1, S, W))
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.ln = nn.LayerNorm(W)
            self.head = nn.Linear(W, V, bias=False)

        def forward(self, idx):
            x = self.emb(idx) + self.pos
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln(x))

    model = Model().cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator(device="cuda").manual_seed(1)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, V, (B, S), generator=g, device="cuda")
        y = torch.roll(x, 1, dims=1)
        loss = F.cross_entropy(model(x).float().view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.mark.slow
def test_fp4all_trains():
    """The shipped all-FP4 step (flash_attn_fp4(pv_precision="fp4", backward="fp4"), default options) trains a small causal model: finite losses, it
    learns, and it tracks the BF16 kernels."""
    bf = _toy_losses("bf16")
    f4 = _toy_losses("fp4all")
    bf_tail, f4_tail = sum(bf[-20:]) / 20, sum(f4[-20:]) / 20
    assert all(math.isfinite(x) for x in f4)
    assert bf_tail < 0.5 * bf[0], f"reference did not learn: {bf[0]:.3f} -> {bf_tail:.3f}"
    assert f4_tail < 0.5 * f4[0], f"all-FP4 did not learn: {f4[0]:.3f} -> {f4_tail:.3f}"
    assert abs(f4_tail - bf_tail) <= 0.05 + 0.1 * bf_tail, f"all-FP4 tail loss {f4_tail:.4f} vs BF16 {bf_tail:.4f}"

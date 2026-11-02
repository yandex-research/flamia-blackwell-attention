"""`causal-4bit-pv-unguarded` and `query-row-key-local-v-scale-dv-halved`, as pytest.

A 4-bit PV groups V's scale factors along KEYS (V is K-major for the block-scaled MMA), so with key-grouped scales a
causal output can depend on future V rows through a shared group amax. The fix (`key-local-v-scale`, the default
under a mask: causal_safety="auto" / "key_local") gives every key its own power-of-two scale. Checks:

  * future-V leak (the direct test): perturb V rows K0+1..K0+8 without moving V's pinned amax and require the
    causal outputs of queries K0-8..K0 -- which may not see those keys -- to stay bitwise equal. Zero on the 8-bit
    P.V with per-tensor V (its amax pinned) and on the causal-safe 4-bit P.V; NON-zero on the leaky configuration
    (causal_safety="off"), which proves the probe is not blind.
  * the in-32-token-block channel of the 8-bit P.V (docs/lowprec.md, causal safety): the default token-block E4M3 V
    shares one UE8M0 exponent per 32 keys, so a later key of the same block that raises the block amax re-rounds the
    visible keys' subnormal V codes. The defaults are therefore NOT causally exact: on a stressed diagonal block (the
    visible keys of the block small, the future ones large) the default call of both entries leaks, bounded -- sub-ULP
    to ~1e-3, only rows of the perturbed keys' own 32-token block (< 32 rows). operand_scaling="token_local" closes
    it exactly (0). The channel does not depend on qk_hadamard (V is never rotated).
  * dV fold count: dL/dV does not depend on V, and V -> 2^3 V leaves the key-local codes bitwise unchanged, so
    n = log2(||dV(8V)|| / ||dV(V)||) / 3 must be 0 (n = +1: the per-key 1 / s_k fold is missing; -1: applied twice) on
    the query-row backward (hybrid and all-FP4 dK / dV).
  * accuracy with the norm ratio next to the cosine (a cosine alone missed a dV at half magnitude), on the default
    4-bit step (flash_attn_fp4: E4M3 P.V forward, dk8 backward).

Every call is the public flash_attn_fp4 / flash_attn_fp8, except the query-row backward rows, which call the all-4-bit route's drivers with
the specs the plan builds for the same call (resolve_fp4(...).specs).
"""
import math

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, FP8AttnOptions, flash_attn_fp4, flash_attn_fp8, resolve_fp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="SM 10.x kernels"
)

B, S, H, D = 1, 1024, 4, 128
SCALE = 1.0 / math.sqrt(D)
PIN_AMAX = 240.0  # pins V's per-tensor amax in both runs so the outer scale cannot move
K0 = 1000         # strictly inside a 16- and a 32-key scale group (1000 % 32 == 8)


def _inputs(vscale_log2=0, seed=20260928):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(B, S, H, D, generator=g, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    if vscale_log2:
        v = (v.float() * (2.0 ** vscale_log2)).to(torch.bfloat16).contiguous()
    return q, k, v, do


def _four(x, ref):
    x, r = x.double().flatten(), ref.double().flatten()
    return (torch.dot(x, r) / (x.norm() * r.norm()).clamp(min=1e-30)).item(), (x.norm() / r.norm().clamp(min=1e-30)).item()


def _ref_attn(q, k, v, do, causal, qchunk=256):
    """fp32 (o, dq, dk, dv), chunked over queries (no (b, h, s, s) tensor)."""
    q, k, v, do = (t.float() for t in (q, k, v, do))
    o, dq, dk, dv = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for i in range(0, S, qchunk):
        j = i + qchunk
        sc = torch.einsum("bqhd,bkhd->bhqk", q[:, i:j], k) * SCALE
        if causal:
            sc = sc.masked_fill(torch.arange(S, device=q.device)[None, :] > torch.arange(i, j, device=q.device)[:, None],
                                float("-inf"))
        p = torch.softmax(sc, -1)
        o[:, i:j] = torch.einsum("bhqk,bkhd->bqhd", p, v)
        dv += torch.einsum("bhqk,bqhd->bkhd", p, do[:, i:j])
        dp = torch.einsum("bqhd,bkhd->bhqk", do[:, i:j], v)
        dd = (do[:, i:j] * o[:, i:j]).sum(-1).permute(0, 2, 1)
        ds = p * (dp - dd[..., None]) * SCALE
        dq[:, i:j] = torch.einsum("bhqk,bkhd->bqhd", ds, k)
        dk += torch.einsum("bhqk,bqhd->bkhd", ds, q[:, i:j])
    return dict(o=o, dq=dq, dk=dk, dv=dv)


def _step(q, k, v, do, causal):
    """The default 4-bit step through the public entry (its default P.V is E4M3, 8-bit, with no 4-bit V group along
    keys; the 4-bit P.V is reached only by asking for it, pv_precision="fp4")."""
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = flash_attn_fp4(qg, kg, vg, causal=causal, softmax_scale=SCALE)
    dq, dk, dv = torch.autograd.grad(out, (qg, kg, vg), do)
    torch.cuda.synchronize()
    return dict(o=out.detach(), dq=dq, dk=dk, dv=dv)


def _query_row(q, k, v, do, dkv):
    """The all-4-bit route's key-local V forward and query-row backward (causal_safety="key_local"), kernel level."""
    from flash_attn.cute.interface_bwd_fp4q import flash_attn_fp4all_fwd, fp4all_quantize_qkv_vtok, fp4q_prepare, fp4q_run

    m = torch.empty(q.shape, dtype=q.dtype, device="meta").requires_grad_(True)
    sp = resolve_fp4(m, m, m, causal=True, softmax_scale=SCALE, pv_precision="fp4", backward="fp4",
                     causal_safety="key_local", options=FP4AttnOptions(fp4_backward_dkdv=dkv)).specs
    ops = fp4all_quantize_qkv_vtok(q, k, v, SCALE)
    out, lse, frame, ops = flash_attn_fp4all_fwd(q, k, v, softmax_scale=SCALE, causal=True, ops=ops, vtok=True, specs=sp)
    prep = fp4q_prepare(q, k, v, out, do, lse, frame, ops, softmax_scale=SCALE, causal=True, do_rnd="rtn", ds_sr=False,
                        seed=1, dkv=dkv, spec=sp.bwdq, qspec=sp.quant, fspec=sp.fwd)
    dq, dk, dv = fp4q_run(prep)
    torch.cuda.synchronize()
    return dict(o=out, dq=dq, dk=dk, dv=dv, vt4=ops["vt4"])


# ------------------------------------------------------------------------------------------ future-V leak
@pytest.mark.parametrize("route,four_bit_pv,csv,leaks", [
    # the 8-bit P.V on per-tensor E4M3 V with its amax pinned: exact. (The default token-block V is not exact -- its
    # in-block channel is test_future_v_in_block_channel's; an unstressed probe like this one sees it only by chance,
    # and it does not depend on qk_hadamard.)
    ("per_tensor_8bit_pv_pinned", False, None, False),
    ("causal_safe_4bit_pv", True, "auto", False),  # causal_safety="auto" -> the diagonal-only key-local V scale
    ("leaky_4bit_pv", True, "off", True),        # the explicit opt-out must still leak, or the probe is blind
])
def test_future_v_leak(route, four_bit_pv, csv, leaks):
    """The 4-bit P.V rows probe option A's shipped inference forward (flash_attn_fp4(pv_precision="fp4"), no grad): option A
    is inference-only. The 8-bit row is flash_attn_fp4's E4M3 P.V forward on per-tensor operands (V's amax pinned)."""
    q, k, v, do = _inputs()
    v = v.clone()
    v[:, 0, :, 0] = PIN_AMAX
    v2 = v.clone()
    g = torch.Generator(device="cuda").manual_seed(7)
    v2[:, K0 + 1:K0 + 9] = torch.randn(B, 8, H, D, generator=g, device="cuda", dtype=torch.bfloat16).sign() * (PIN_AMAX / 4)
    assert v.float().abs().max().item() == v2.float().abs().max().item(), "outer scale moved"
    kw = dict(causal=True, softmax_scale=SCALE)
    if route == "per_tensor_8bit_pv_pinned":
        kw.update(options=FP4AttnOptions(operand_scaling="per_tensor"))
    if four_bit_pv:
        kw.update(pv_precision="fp4", causal_safety=csv)
    o1, o2 = flash_attn_fp4(q, k, v, **kw), flash_attn_fp4(q, k, v2, **kw)
    torch.cuda.synchronize()
    diff = (o1[:, K0 - 8:K0 + 1].float() - o2[:, K0 - 8:K0 + 1].float()).abs().max().item()
    if leaks:
        assert diff > 0.0, f"{route}: the leaky configuration shows no leak -- the probe is blind"
    else:
        assert diff == 0.0, f"{route}: causal outputs of queries {K0 - 8}..{K0} moved by {diff:.3e} with future V rows"


# ------------------------------------------------------------------------------------------ the in-block channel
BLK0 = K0 - K0 % 32       # 992: the first key of the perturbed keys' 32-token block
STRESS = 1.0 / 16         # the visible keys of that block (BLK0..K0) made small: near the E4M3 subnormal edge
AMP = PIN_AMAX            # the future keys K0+1..K0+8 at the pinned amax: the block exponent rises by ~6
IN_BLOCK_BOUND = 2e-3     # the documented size: sub-ULP to ~1e-3 (measured 2.4e-4 at this stress)


@pytest.mark.parametrize("entry,scaling,leaks", [
    ("fp4", "token_block", True),    # flash_attn_fp4's default causal call: leaks, bounded
    ("fp4", "token_local", False),   # the exact closure
    ("fp8", "token_block", True),    # flash_attn_fp8's default causal call: the same MXFP8 V, the same channel
    ("fp8", "token_local", False),
])
def test_future_v_in_block_channel(entry, scaling, leaks):
    """A stressed diagonal block: every row before the block reads exactly 0; on
    token-block V the block's own rows move (> 0, < IN_BLOCK_BOUND, < 32 rows, none before BLK0); token_local reads 0
    everywhere. qk_hadamard is the entry default on token_block (on); token_local turns it off (a logged fallback)."""
    q, k, v, _ = _inputs()
    v = v.clone()
    v[:, 0, :, 0] = PIN_AMAX
    v[:, BLK0:K0 + 1] *= STRESS
    v2 = v.clone()
    g = torch.Generator(device="cuda").manual_seed(7)
    v2[:, K0 + 1:K0 + 9] = torch.randn(B, 8, H, D, generator=g, device="cuda", dtype=torch.bfloat16).sign() * AMP
    fn, O = (flash_attn_fp4, FP4AttnOptions) if entry == "fp4" else (flash_attn_fp8, FP8AttnOptions)
    opts = O(operand_scaling="token_local", ds_dq_rowlocal="all") if scaling == "token_local" else O()
    o1 = fn(q, k, v, causal=True, softmax_scale=SCALE, options=opts)
    o2 = fn(q, k, v2, causal=True, softmax_scale=SCALE, options=opts)
    torch.cuda.synchronize()
    per_row = (o1[:, :K0 + 1].float() - o2[:, :K0 + 1].float()).abs().amax(dim=(0, 2, 3))   # rows 0..K0
    moved = torch.nonzero(per_row).flatten().tolist()
    worst = per_row.max().item()
    if not leaks:
        assert worst == 0.0, f"{entry} {scaling}: rows {moved[:8]} moved by up to {worst:.3e} with future V rows"
        return
    assert worst > 0.0, f"{entry} {scaling}: the stressed block shows no in-block leak -- the probe is blind"
    assert worst < IN_BLOCK_BOUND, f"{entry} {scaling}: in-block leak {worst:.3e} >= {IN_BLOCK_BOUND:.0e}"
    assert min(moved) >= BLK0 and len(moved) < 32, f"{entry} {scaling}: moved rows {moved} outside the block"


# ------------------------------------------------------------------------------------------ dV fold count
# (no option A row: option A is inference-only)
@pytest.mark.parametrize("route", ["query_row_e4m3", "query_row_fp4"])
def test_dv_fold_exponent_is_zero(route):
    run = {"query_row_e4m3": lambda *a: _query_row(*a, "e4m3"), "query_row_fp4": lambda *a: _query_row(*a, "fp4")}[route]
    r0 = run(*_inputs(0))
    r3 = run(*_inputs(3))
    if "vt4" in r0:
        assert torch.equal(r0["vt4"], r3["vt4"]), "the 2^3 V rescale changed the key-local codes"
    n0, n3 = r0["dv"].float().norm().item(), r3["dv"].float().norm().item()
    n = math.log2(max(n3, 1e-30) / max(n0, 1e-30)) / 3.0
    assert abs(n) < 0.10, f"{route}: dV fold exponent {n:+.4f} (0 correct, +1 fold missing, -1 folded twice)"


# ------------------------------------------------------------------------------------------ accuracy, with norm ratio
# (no 4-bit P.V step rows: option A, the 4-bit P.V route, is inference-only)
@pytest.mark.parametrize("route,four_bit_pv,csv,causal", [
    ("step_default_8bit_pv", False, None, True),
])
def test_step_accuracy_with_norm_ratio(route, four_bit_pv, csv, causal):
    q, k, v, do = _inputs()
    got = _step(q, k, v, do, causal)
    ref = _ref_attn(q, k, v, do, causal)
    for nm in ("o", "dq", "dk", "dv"):
        c, nr = _four(got[nm], ref[nm])
        assert c >= COS_GATE[nm] and NORM_GATE[0] <= nr <= NORM_GATE[1], f"{route} {nm}: cos {c:.5f} norm ratio {nr:.5f}"


COS_GATE = dict(o=0.90, dq=0.90, dk=0.90, dv=0.90)
NORM_GATE = (0.95, 1.05)

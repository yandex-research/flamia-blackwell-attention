"""Lengths that are not multiples of 128 on the low-precision entries: the 128-row tile of every 4-bit operand.

  * test_ragged_batched_is_documents   batched (b, s, h, d) calls with s % 128 != 0 on the 4-bit routes (they used to
                                       be refused, rule 27): the batch rows run as documents -- bitwise the same rows
                                       passed as repacked varlen documents -- and match fp32 (O, LSE, dQ, dK, dV);
  * test_ragged_batched_seqused_k      batched seqused_k with s % 128 != 0 (refused by the same rule before);
  * test_partial_tile_documents        every varlen document compared with fp32 on its own rows, its partial last
                                       128-row tile separately: no document or tail falls below the accuracy of a
                                       whole-tile control document in the same call (the suite had only path-vs-path
                                       checks for partial tiles).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import interface_bwd_fp4q as iq
from flash_attn.cute.lowprec import flash_attn_fp4, flash_attn_fp8, resolve_fp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM100")
BF16 = torch.bfloat16
ROUTES = {  # (entry, kwargs, has backward, deterministic dQ)
    "fp8": (flash_attn_fp8, {"deterministic": True}, True, True),
    "fp4": (flash_attn_fp4, {"deterministic": True}, True, True),
    "all4": (flash_attn_fp4, {"pv_precision": "fp4", "backward": "fp4"}, True, False),
    "optA": (flash_attn_fp4, {"pv_precision": "fp4"}, False, True),
}
# cosine vs fp32 at N(0, 1) inputs (the routes' whole-tile accuracy, docs/lowprec.md: 0.992 / 0.981 / 0.945 ...)
GATE = {"fp8": (0.996,) * 4, "fp4": (0.985,) * 4, "all4": (0.975, 0.93, 0.955, 0.98), "optA": (0.975,)}


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _ref(q, k, v, do, causal, used_k=None):
    """fp32 reference of one (L, h, d) document or (b, s, h, d) batch: O, LSE (.., h, L), dQ, dK, dV;
    causal bottom-right aligned; used_k (b,): keys per batch row."""
    batched = q.dim() == 4
    if not batched:
        q, k, v, do = (t[None] for t in (q, k, v, do))
    h, hk = q.shape[2], k.shape[2]
    qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf.repeat_interleave(h // hk, 2)) / math.sqrt(q.shape[-1])
    Lq, Lk = q.shape[1], k.shape[1]
    keep = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device)
    if causal:
        keep = torch.arange(Lk, device=q.device)[None] <= torch.arange(Lq, device=q.device)[:, None] + Lk - Lq
    keep = keep.expand(q.shape[0], 1, Lq, Lk)
    if used_k is not None:
        keep = keep & (torch.arange(Lk, device=q.device) < used_k.view(-1, 1, 1, 1))
    s = s.masked_fill(~keep, float("-inf"))
    lse = s.logsumexp(-1)
    o = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), vf.repeat_interleave(h // hk, 2))
    g = torch.autograd.grad(o, (qf, kf, vf), do.float())
    out = [o.detach(), lse.detach(), *g]
    return out if batched else [t[0] for t in out]


def _call(route, q, k, v, do, causal, **kw):
    """[O, LSE, dQ, dK, dV] (no gradients without a backward) of one route call; the all-4-bit SR dither restarted."""
    fn, rkw, grad, _ = ROUTES[route]
    iq.reset_sr_counter()
    x = [t.detach().clone().requires_grad_(grad) for t in (q, k, v)]
    o, lse = fn(*x, causal=causal, return_lse=True, **rkw, **kw)
    return [o.detach(), lse] + (list(torch.autograd.grad(o, x, do)) if grad else [None] * 3)


def _rand(shape, g):
    return torch.randn(shape, generator=g, device="cuda", dtype=BF16)


@pytest.mark.parametrize("route", ["fp4", "all4", "optA"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("b,sq,sk", [(2, 1000, 1000), (3, 129, 129), (2, 1000, 1024)], ids=["s1000", "s129", "sq_ne_sk"])
def test_ragged_batched_is_documents(route, causal, b, sq, sk):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, do = _rand((b, sq, 8, 128), g), _rand((b, sq, 8, 128), g)
    k, v = _rand((b, sk, 2, 128), g), _rand((b, sk, 2, 128), g)
    p = resolve_fp4(q, k, v, causal=causal, **ROUTES[route][1])
    assert p.fwd.ragged_documents, p.log()
    A = _call(route, q, k, v, do, causal)                                       # the batched call
    ar = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    B = _call(route, q.reshape(b * sq, 8, 128), k.reshape(b * sk, 2, 128), v.reshape(b * sk, 2, 128),
              do.reshape(b * sq, 8, 128), causal, cu_seqlens_q=ar * sq, cu_seqlens_k=ar * sk, max_seqlen_q=sq,
              max_seqlen_k=sk)                                                  # the same rows as documents
    assert A[0].shape == (b, sq, 8, 128) and A[1].shape == (b, 8, sq)
    B = [B[0].view(b, sq, 8, 128), B[1].view(8, b, sq).permute(1, 0, 2)] + \
        [None if t is None else t.view(A[i + 2].shape) for i, t in enumerate(B[2:])]
    det = ROUTES[route][3]
    for i, n in enumerate(("O", "LSE", "dQ", "dK", "dV")):
        if A[i] is None:
            continue
        if n == "dQ" and not det:   # the all-4-bit backward reduces dQ with atomics across key tiles
            torch.testing.assert_close(A[i].float(), B[i].float(), rtol=0, atol=float(B[i].abs().max()) * 2e-2)
        else:
            assert torch.equal(A[i], B[i]), (route, n)
    R = _ref(q, k, v, do, causal)
    gates = GATE[route]
    for (n, a, r), gate in zip([("O", A[0], R[0]), ("dQ", A[2], R[2]), ("dK", A[3], R[3]), ("dV", A[4], R[4])], gates):
        assert torch.isfinite(a).all() and _cos(a, r) > gate, (route, n, _cos(a, r))
    assert (A[1] - R[1]).abs().max() < 0.5      # NVFP4 S: rows with few visible keys carry ~0.3 into the LSE


def test_ragged_batched_seqused_k():
    g = torch.Generator(device="cuda").manual_seed(1)
    q, k, v, do = (_rand((2, 1000, 4, 128), g) for _ in range(4))
    used = torch.tensor([700, 1000], dtype=torch.int32, device="cuda")
    A = _call("fp4", q, k, v, do, False, seqused_k=used)
    R = _ref(q, k, v, do, False, used_k=used)
    for n, a, r in (("O", A[0], R[0]), ("dQ", A[2], R[2]), ("dK", A[3], R[3]), ("dV", A[4], R[4])):
        assert _cos(a, r) > 0.985, (n, _cos(a, r))
    assert A[3][0, 700:].abs().max() == 0 and A[4][0, 700:].abs().max() == 0   # keys past seqused_k: no gradient


LENS = [512, 300, 77, 129, 255, 1000]   # 512: the whole-tile control document (one-token documents: see the ledger)


@pytest.mark.parametrize("route", ["fp8", "fp4", "all4", "optA"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_partial_tile_documents(route, causal):
    g = torch.Generator(device="cuda").manual_seed(2)
    starts = [0]
    for L in LENS:
        starts.append(starts[-1] + -(-L // 128) * 128)
    q, k, v, do = (torch.zeros(starts[-1], 8, 128, device="cuda", dtype=BF16) for _ in range(4))
    for s0, L in zip(starts, LENS):
        for t in (q, k, v, do):
            t[s0:s0 + L] = _rand((L, 8, 128), g)
    cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
    used = torch.tensor(LENS, dtype=torch.int32, device="cuda")
    A = _call(route, q, k, v, do, causal, cu_seqlens_q=cu, cu_seqlens_k=cu, seqused_q=used, seqused_k=used,
              max_seqlen_q=max(LENS), max_seqlen_k=max(LENS))
    tol = {"fp8": 0.003, "fp4": 0.01, "all4": 0.02, "optA": 0.015}[route]
    names = ("O", "dQ", "dK", "dV")[:4 if ROUTES[route][2] else 1]
    ctrl = None
    for s0, L in zip(starts, LENS):
        sl = slice(s0, s0 + L)
        R = _ref(q[sl], k[sl], v[sl], do[sl], causal)
        got = {"O": A[0][sl], "dQ": A[2] is not None and A[2][sl], "dK": A[3] is not None and A[3][sl],
               "dV": A[4] is not None and A[4][sl]}
        ref = {"O": R[0], "dQ": R[2], "dK": R[3], "dV": R[4]}
        cos = {n: _cos(got[n], ref[n]) for n in names}
        tail = L - L // 128 * 128
        if tail >= 16:   # the rows of the partial last tile (a 1-row tail is too few for a cosine)
            cos.update({n + "_tail": _cos(got[n][L - tail:], ref[n][L - tail:]) for n in names})
        lse_ok = (A[1][:, sl] - R[1]).abs().max() < 0.5
        assert lse_ok, (route, L, "LSE")
        if ctrl is None:   # the 512-token control: the route's accuracy on whole tiles
            ctrl = cos
            continue
        for n, c in cos.items():
            assert c > ctrl[n.split("_")[0]] - tol * (2 if n.endswith("_tail") else 1), (route, L, n, c, ctrl)

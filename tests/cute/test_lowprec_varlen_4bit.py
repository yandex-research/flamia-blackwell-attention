"""vall4: the 4-bit P.V routes (all-FP4 training, option A inference) on varlen documents.

Every statistic of these routes (the backward's E4M3 amax slots, max ||V||^2, max |dO|, the dS bound, the diagonal
quantiser's tile constants and prefix maxima) is per document, so
  * each 128-multiple document equals the same document run alone as a batched b = 1 call (bitwise; dense dQ up to the
    f16 dQ accumulator's reduce-add order when a document has several key tiles),
  * scaling one document's V or dO leaves every other document bitwise unchanged (documents of one key tile: dQ too),
  * repacked varlen equals the aligned form by hand."""
import pytest
import torch

from flash_attn.cute.lowprec import flash_attn_fp4, FP4AttnOptions
from flash_attn.cute import interface_bwd_fp4q as iq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM100")
BF = torch.bfloat16
ROUTES = {"all4": dict(pv_precision="fp4", backward="fp4"), "optA": dict(pv_precision="fp4")}


def _aligned(lens, h, hk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    st = [0]
    for L in lens:
        st.append(st[-1] + (L + 127) // 128 * 128)
    tot = st[-1]
    q, do = (torch.zeros(tot, h, 128, device="cuda", dtype=BF) for _ in range(2))
    k, v = (torch.zeros(tot, hk, 128, device="cuda", dtype=BF) for _ in range(2))
    for s0, L in zip(st[:-1], lens):
        for t in (q, k, v, do):
            t[s0:s0 + L] = torch.randn(L, t.shape[1], 128, device="cuda", dtype=BF, generator=g)
    cu = torch.tensor(st, dtype=torch.int32, device="cuda")
    su = torch.tensor(lens, dtype=torch.int32, device="cuda")
    kw = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, seqused_q=su, seqused_k=su, max_seqlen_q=max(lens),
              max_seqlen_k=max(lens))
    return (q, k, v, do), st, kw


def _run(route, q, k, v, do, causal, kw):
    grad = route == "all4"
    x = tuple(t.detach().clone().requires_grad_(grad) for t in (q, k, v))
    iq.reset_sr_counter()
    opts = FP4AttnOptions(all_fp4_stabiliser="rtn", sr_dk_operand=False) if grad else FP4AttnOptions()
    out = flash_attn_fp4(*x, causal=causal, options=opts, **ROUTES[route], **kw)
    return [out] + (list(torch.autograd.grad(out, x, do)) if grad else [])


@pytest.mark.parametrize("route", ["all4", "optA"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("hk", [4, 2])
def test_documents_equal_their_batched_calls(route, causal, hk):
    lens = [128, 300, 256]
    (q, k, v, do), st, kw = _aligned(lens, 4, hk, 3)
    A = _run(route, q, k, v, do, causal, kw)
    for i in (0, 2):
        sl = slice(st[i], st[i] + lens[i])
        B = _run(route, *(t[sl][None] for t in (q, k, v, do)), causal, {})
        for n, a, b in zip(("O", "dQ", "dK", "dV"), A, B):
            if n == "dQ" and not causal and lens[i] > 128:
                torch.testing.assert_close(a[sl].float(), b[0].float(), rtol=0, atol=float(b.abs().max()) * 1e-2)
            else:
                assert torch.equal(a[sl], b[0]), (route, causal, hk, i, n)


@pytest.mark.parametrize("route", ["all4", "optA"])
@pytest.mark.parametrize("causal", [True, False])
def test_documents_do_not_couple(route, causal):
    lens = [128, 700, 128]
    (q, k, v, do), st, kw = _aligned(lens, 4, 4, 5)
    base = _run(route, q, k, v, do, causal, kw)
    s1 = slice(st[1], st[1] + lens[1])
    for sv, sd in ((8.0, 1.0), (1.0, 8.0)):
        v2, do2 = v.clone(), do.clone()
        v2[s1] *= sv
        do2[s1] *= sd
        P = _run(route, q, k, v2, do2, causal, kw)
        for i in (0, 2):
            sl = slice(st[i], st[i] + lens[i])
            for a, b in zip(base, P):
                assert torch.equal(a[sl], b[sl])

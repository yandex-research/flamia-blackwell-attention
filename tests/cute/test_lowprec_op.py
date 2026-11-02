"""flash_attn.cute.lowprec._op: flash_attn_fp4 / flash_attn_fp8 as the torch.library ops lowbit_attn::fp4 / ::fp8.

  * test_op_registered          the ops (and their backward ops) are registered on `import flash_attn.cute.lowprec`;
  * test_op_matches_function    the op path == the autograd.Function path, bitwise (O, LSE, dQ, dK, dV; deterministic),
                                batched / ragged / repacked varlen / aligned varlen, causal / dense, MHA / GQA;
  * test_op_no_host_sync        fwd + bwd under torch.cuda.set_sync_debug_mode("error");
  * test_op_graph_replay        fwd + bwd captured in a CUDA graph, replayed on new inputs == eager, bitwise;
  * test_op_sac_saves           torch.utils.checkpoint with a selective policy that saves the op: the forward kernel
                                runs ONCE per step (twice when the policy recomputes it), gradients bitwise equal;
  * test_op_compile             torch.compile traces the op (fake implementation), bitwise vs eager.
"""
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from flash_attn.cute.lowprec import _op, flash_attn_fp4, flash_attn_fp8
from flash_attn.cute.lowprec import api as _api

BF16 = torch.bfloat16
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="Blackwell (SM100) only")
ENTRIES = {"fp4": flash_attn_fp4, "fp8": flash_attn_fp8}


def _rand(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=BF16) * scale


def _case(layout, h, hk, d, seed=0):
    """(q, k, v, kwargs, row mask or None) leaf tensors requiring grad."""
    kw = {}
    rows = None
    if layout == "batched":
        q, k, v = _rand((2, 1024, h, d), seed + 1), _rand((2, 1024, hk, d), seed + 2), _rand((2, 1024, hk, d), seed + 3)
    elif layout == "ragged":
        q, k, v = _rand((2, 1000, h, d), seed + 1), _rand((2, 1000, hk, d), seed + 2), _rand((2, 1000, hk, d), seed + 3)
    elif layout == "varlen":
        lens = [300, 1000, 77, 2048, 640]
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        t = sum(lens)
        q, k, v = _rand((t, h, d), seed + 1), _rand((t, hk, d), seed + 2), _rand((t, hk, d), seed + 3)
        kw = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=2048, max_seqlen_k=2048)
    else:  # aligned
        lens = [300, 1000, 77, 2048, 640]
        starts = [0]
        for n in lens:
            starts.append(starts[-1] + (n + 127) // 128 * 128)
        t = starts[-1]
        rows = torch.zeros(t, dtype=torch.bool, device="cuda")
        for s0, n in zip(starts, lens):
            rows[s0:s0 + n] = True
        m = rows.to(BF16).view(-1, 1, 1)
        q, k, v = _rand((t, h, d), seed + 1) * m, _rand((t, hk, d), seed + 2) * m, _rand((t, hk, d), seed + 3) * m
        cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
        used = torch.tensor(lens, dtype=torch.int32, device="cuda")
        kw = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, seqused_q=used, seqused_k=used, max_seqlen_q=2048,
                  max_seqlen_k=2048)
    return [x.requires_grad_(True) for x in (q, k, v)], kw, rows


def _step(fn, qkv, kw, go_seed=7):
    q, k, v = qkv
    for x in qkv:
        x.grad = None
    r = fn(q, k, v, **kw)
    out, lse = r if isinstance(r, tuple) else (r, None)
    out.backward(_rand(out.shape, go_seed))
    return out.detach(), (None if lse is None else lse.detach()), q.grad, k.grad, v.grad


def _function_path(monkeypatch):
    """Route every call through the pre-op autograd.Function (the 3dd18da path)."""
    monkeypatch.setattr(_op, "op_apply",
                        lambda plan, q, k, v, m: _api._LowPrecAttnFunc.apply(q, k, v, plan, None, m))


def _eq(a, b, rows=None, lse=False):
    if a is None or b is None:
        return a is None and b is None
    if rows is not None and lse:
        a, b = a[:, rows], b[:, rows]
    return torch.equal(a.view(torch.uint8) if a.dtype == BF16 else a, b.view(torch.uint8) if b.dtype == BF16 else b)


def test_op_registered():
    for n in ("fp4", "fp8", "fp4_bwd", "fp8_bwd"):
        assert hasattr(torch.ops.lowbit_attn, n), n
    assert isinstance(torch.ops.lowbit_attn.fp4.default, torch._ops.OpOverload)


CASES = [(e, lay, c, h, hk, d) for e in ("fp8", "fp4") for lay in ("batched", "ragged", "varlen", "aligned")
         for c in (False, True) for (h, hk) in ((4, 4), (8, 2)) for d in (64, 128)
         if not (e == "fp4" and lay == "ragged")]


@pytest.mark.parametrize("entry,layout,causal,h,hk,d", CASES)
def test_op_matches_function(monkeypatch, entry, layout, causal, h, hk, d):
    qkv, kw, rows = _case(layout, h, hk, d)
    kw = dict(kw, causal=causal, deterministic=True, return_lse=layout in ("batched", "aligned") and entry == "fp8")
    fn = ENTRIES[entry]
    calls = []
    real = _op.OPS[entry]
    monkeypatch.setitem(_op.OPS, entry, lambda *a: calls.append(1) or real(*a))
    new = _step(fn, qkv, kw)
    assert calls, "the call did not go through the op"
    monkeypatch.setitem(_op.OPS, entry, real)
    _function_path(monkeypatch)
    old = _step(fn, qkv, kw)
    for i, (a, b) in enumerate(zip(new, old)):
        assert _eq(a, b, rows, lse=i == 1), ("out", "lse", "dq", "dk", "dv")[i]


@pytest.mark.parametrize("entry,layout", [("fp8", "batched"), ("fp4", "batched"), ("fp4", "varlen"),
                                          ("fp8", "aligned"), ("fp8", "ragged")])
def test_op_no_host_sync(entry, layout):
    qkv, kw, _ = _case(layout, 8, 2, 128)
    kw = dict(kw, causal=True)
    _step(ENTRIES[entry], qkv, kw)   # warm: compile, plan, entry
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _step(ENTRIES[entry], qkv, kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("entry,layout", [("fp8", "batched"), ("fp4", "batched"), ("fp4", "varlen"),
                                          ("fp8", "aligned")])
def test_op_graph_replay(entry, layout):
    fn = ENTRIES[entry]
    qkv, kw, _ = _case(layout, 8, 2, 128)
    kw = dict(kw, causal=True, deterministic=True)
    go = _rand(qkv[0].shape[:-1] + (qkv[2].shape[-1],), 7)

    def body():
        out = fn(*qkv, **kw)
        out.backward(go)
        return out
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for x in qkv:
                x.grad = None
            body()
    torch.cuda.current_stream().wait_stream(s)
    for x in qkv:
        x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = body()
    new, _, _ = _case(layout, 8, 2, 128, seed=100)
    with torch.no_grad():
        for x, y in zip(qkv, new):
            x.copy_(y)
            x.grad.zero_()
    g.replay()
    torch.cuda.synchronize()
    got = (out_g.clone(), qkv[0].grad.clone(), qkv[1].grad.clone(), qkv[2].grad.clone())
    ref_qkv = [y.detach().clone().requires_grad_(True) for y in new]
    ref = fn(*ref_qkv, **kw)
    ref.backward(go)
    for a, b in zip(got, (ref, ref_qkv[0].grad, ref_qkv[1].grad, ref_qkv[2].grad)):
        assert _eq(a, b.detach())


def _fwd_launches(fn):
    """Number of forward attention kernel launches (the fork forwards: FlashAttentionForwardSm100*) in fn()."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sum(1 for n in names if "FlashAttentionForward" in n), sum(1 for n in names if "Backward" in n)


@pytest.mark.parametrize("entry,layout", [("fp8", "batched"), ("fp4", "batched"), ("fp4", "varlen"),
                                          ("fp8", "aligned")])
def test_op_sac_saves(entry, layout):
    fn = ENTRIES[entry]
    qkv, kw, _ = _case(layout, 8, 2, 128)
    kw = dict(kw, causal=True, deterministic=True)
    w = _rand((128, 128), 11).requires_grad_(True)
    save = {torch.ops.lowbit_attn.fp4.default, torch.ops.lowbit_attn.fp8.default}

    def block(q, k, v):
        out = fn(q @ w, k, v, **kw)    # a projection in front, recomputed under SAC
        return (out.float() ** 2).sum()

    def policy(saving):
        def pol(ctx, op, *a, **k):
            return CheckpointPolicy.MUST_SAVE if saving and op in save else CheckpointPolicy.PREFER_RECOMPUTE
        return pol

    def run(mode):
        for x in (*qkv, w):
            x.grad = None
        if mode == "eager":
            loss = block(*qkv)
        else:
            loss = checkpoint(block, *qkv, use_reentrant=False,
                              context_fn=partial(create_selective_checkpoint_contexts, policy(mode == "save")))
        loss.backward()
        return [x.grad.clone() for x in (*qkv, w)]

    run("save"), run("recompute"), run("eager")   # warm
    res = {}
    for mode in ("eager", "save", "recompute"):
        box = {}
        res[mode] = _fwd_launches(lambda: box.setdefault("g", run(mode)))
        res[mode + "_grads"] = box["g"]
    assert res["eager"][0] == 1 and res["save"][0] == 1, res
    assert res["recompute"][0] == 2, res
    assert res["eager"][1] == res["save"][1] == res["recompute"][1] >= 1, res
    for a, b in zip(res["save_grads"], res["eager_grads"]):
        assert torch.equal(a, b)
    for a, b in zip(res["recompute_grads"], res["eager_grads"]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("entry,layout,backend", [("fp8", "batched", "aot_eager"), ("fp4", "batched", "aot_eager"),
                                                  ("fp4", "varlen", "aot_eager"), ("fp4", "batched", "inductor")])
def test_op_compile(entry, layout, backend):
    import torch._dynamo
    fn = ENTRIES[entry]
    qkv, kw, _ = _case(layout, 8, 2, 128)
    kw = dict(kw, causal=True, deterministic=True)
    if layout != "batched":   # a document-table signature needs one eager call first (the fake reads its entry)
        _step(fn, qkv, kw)

    def f(q, k, v):
        return fn(q, k, v, **kw)
    torch._dynamo.reset()
    import torch._inductor.config as icfg
    with icfg.patch(force_disable_caches=True):   # the op's state layout is not part of inductor's cache key
        cf = torch.compile(f, backend=backend)
        got = _step(lambda q, k, v, **_: cf(q, k, v), qkv, {})
    ref = _step(fn, qkv, kw)
    for a, b in zip(got, ref):
        assert _eq(a, b)


@pytest.mark.parametrize("entry,kw", [("fp4", dict(pv_precision="fp4", backward="fp4")),
                                      ("fp4", dict(deterministic=True)), ("fp8", dict(deterministic=True))])
def test_op_aliased_inputs(entry, kw):
    """Same call signature, q / k / v separate or slices of one storage: same output, no state-layout refusal; the
    gradients equal by value (the stacked leaf's gradient goes through select_backward, which turns -0.0 into 0.0)."""
    fn = ENTRIES[entry]
    kw = dict(kw, causal=True)
    base = _rand((3, 2, 1024, 8, 128), 5)
    sep = [base[i].clone().requires_grad_(True) for i in range(3)]
    a = _step(fn, sep, kw)
    stacked = base.clone().requires_grad_(True)
    views = [stacked[i] for i in range(3)]
    assert all(tuple(x.stride()) == tuple(y.stride()) for x, y in zip(views, sep))
    r = fn(*views, **kw)
    r.backward(_rand(r.shape, 7))
    assert _eq(a[0], r.detach())
    if kw.get("deterministic"):
        for i in range(3):
            assert torch.equal(a[2 + i], stacked.grad[i])

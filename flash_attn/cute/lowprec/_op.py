"""``torch.library`` custom ops around the low-precision autograd node: ``lowbit_attn::fp4`` / ``lowbit_attn::fp8``.

Why: an opaque Python ``autograd.Function`` is invisible to selective activation checkpointing (SAC) and to
``torch.compile``. As a registered op the forward is ONE dispatcher call with tensor outputs, so a SAC policy can SAVE
it (``torch.ops.lowbit_attn.fp4.default``) instead of replaying the quantisers and the forward kernel in the backward,
and ``torch.compile`` traces it through its fake implementation.

The op boundary is the library's autograd node (``api._LowPrecAttnFunc``): q / k / v as the kernels see them (the
batched tensors, the aligned varlen views, or the repacked buffer of the repack fallback) plus the document tables.
The repacking glue around it (``_routes.run_documents``) stays ordinary differentiable torch ops.

Schema (both ops; ``fp8`` and ``fp4`` share the implementation, the plan picks the route)::

    lowbit_attn::fp4(Tensor q, Tensor k, Tensor v, Tensor[] docs, int plan, int docs_id)
        -> (Tensor out, Tensor lse, Tensor[] state)
    lowbit_attn::fp4_bwd(Tensor dout, Tensor[] state, int entry) -> (Tensor dq, Tensor dk, Tensor dv)

- ``plan``: an int handle into a process-local registry of resolved, immutable ``AttnPlan``s (options and
  ``KernelTuning`` included); the plan cache hands the same plan (and handle) to every call with the same signature.
- ``docs`` / ``docs_id``: the call's document tables (``lp_varlen.LPDocs``) flattened to their tensors plus an int
  handle of their (interned) non-tensor structure; ``docs_id = -1`` without documents. The masks are rebuilt in the
  op from the plan (causal / window / softcap) and the documents, exactly as the public path builds them.
- ``state``: everything the route's backward needs that is not an input or ``out`` / ``lse``: one tensor per
  distinct storage the route's saved state (``_routes.Saved``) holds (the first saved tensor on it; the others are
  views of it). The state's structure (which attribute is which view of which storage, every non-tensor value) is
  recorded once per call signature (``entry``) and must be the same on every later call with that signature: the
  tensor layout is checked on every call, the non-tensor values on the first ``_FULL_CHECKS`` calls (a mismatch
  raises).
  The autograd context saves exactly the inputs, outputs and state tensors the backward reads, so the backward state
  lives in the op's autograd context only: no host sync, nothing on the Python side per call, CUDA-graph safe.

Not through the op (they keep the ``autograd.Function`` path, ``api._run``): user ``score_mod`` / ``score_mod_bwd`` /
``mask_mod`` / ``aux_tensors`` (callables are not op arguments), and pre-quantised inputs (``flash_attn_fp8`` with
``q_quantized`` / ``k_quantized`` / ``q_scale_fused``, and the FA3-style E4M3 inputs with ``q_descale`` / ...
``grad_proxies``).

Fake implementation (``torch.compile``): the output shapes come from the signature's recorded entry. An entry is
recorded by the first real call; without documents a cold fake call records it by running the real op once on zero
inputs of the same metadata (``_probe``); with documents a cold fake call raises (run the call once eagerly first).
"""
from __future__ import annotations

import copy
import dataclasses
import threading
from typing import List, Tuple

import torch

__all__ = ["OPS", "plan_handle", "op_apply", "flatten_docs", "state_of"]

_LOCK = threading.RLock()

# ---------------------------------------------------------------------------------------------------------------------
# registries (process-local, append-only: a compiled graph may hold any handle)
# ---------------------------------------------------------------------------------------------------------------------
_PLANS: list = []            # handle -> AttnPlan (kept alive, so id(plan) is never reused)
_PLAN_HANDLE: dict = {}      # id(plan) -> handle
_DOCS: list = []             # docs_id -> skeleton (with templates)
_DOCS_ID: dict = {}          # structural key -> docs_id
_ENTRIES: list = []          # entry -> _Entry
_ENTRY_ID: dict = {}         # call-signature key -> entry


def plan_handle(plan) -> int:
    with _LOCK:
        h = _PLAN_HANDLE.get(id(plan))
        if h is None:
            h = len(_PLANS)
            _PLANS.append(plan)
            _PLAN_HANDLE[id(plan)] = h
        return h


# ---------------------------------------------------------------------------------------------------------------------
# generic flatten / unflatten of the route's Python objects (Saved, LPMasks, LPDocs, containers)
# skeleton nodes: ("T", i) tensor i of the flat list | ("tuple"|"list", (children)) | ("dict", keys, (children))
#                 ("obj", cls, names, (children), template) | ("leaf", value)
# ---------------------------------------------------------------------------------------------------------------------

def _descend_types():
    from flash_attn.cute.lp_masks import LPMasks
    from flash_attn.cute.lp_varlen import LPDocs
    from flash_attn.cute.lp_docmaps import DocMaps
    from ._routes import Saved
    return (Saved, LPMasks, LPDocs, DocMaps)


def _attrs(o):
    if hasattr(o, "__dict__"):
        return list(o.__dict__.keys())
    return [f.name for f in dataclasses.fields(o)]


def _flatten(o, out: list, types):
    """(skeleton, has_tensor); tensors appended to out."""
    if isinstance(o, torch.Tensor):
        out.append(o)
        return ("T", len(out) - 1), True
    if type(o) in (tuple, list):
        ch = [_flatten(x, out, types) for x in o]
        return (type(o).__name__, tuple(c[0] for c in ch)), any(c[1] for c in ch)
    if type(o) is dict:
        keys = tuple(o.keys())
        ch = [_flatten(o[kk], out, types) for kk in keys]
        return ("dict", keys, tuple(c[0] for c in ch)), any(c[1] for c in ch)
    if isinstance(o, types):
        names = tuple(_attrs(o))
        ch = [_flatten(getattr(o, n), out, types) for n in names]
        return ("obj", type(o), names, tuple(c[0] for c in ch), o), any(c[1] for c in ch)
    return ("leaf", o), False


def _unflatten(sk, get):
    kind = sk[0]
    if kind == "T":
        return get(sk[1])
    if kind == "tuple":
        return tuple(_unflatten(c, get) for c in sk[1])
    if kind == "list":
        return [_unflatten(c, get) for c in sk[1]]
    if kind == "dict":
        return {kk: _unflatten(c, get) for kk, c in zip(sk[1], sk[2])}
    if kind == "obj":
        new = copy.copy(sk[4])
        for n, c in zip(sk[2], sk[3]):
            object.__setattr__(new, n, _unflatten(c, get))
        return new
    return sk[1]


def _leaf_key(v):
    """A hashable stand-in of a leaf for the structural key (callables: their content hash)."""
    if callable(v) and not isinstance(v, type):
        from flash_attn.cute import utils
        return ("callable", utils.hash_callable(v))
    try:
        hash(v)
        return v
    except TypeError:
        return ("unhashable", type(v).__name__, id(v))


def _struct_key(sk):
    kind = sk[0]
    if kind == "T":
        return sk
    if kind in ("tuple", "list"):
        return (kind, tuple(_struct_key(c) for c in sk[1]))
    if kind == "dict":
        return ("dict", sk[1], tuple(_struct_key(c) for c in sk[2]))
    if kind == "obj":
        return ("obj", sk[1], sk[2], tuple(_struct_key(c) for c in sk[3]))
    return ("leaf", _leaf_key(sk[1]))


def _meta(t: torch.Tensor):
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device.type)


# ---------------------------------------------------------------------------------------------------------------------
# documents: the op's input form
# ---------------------------------------------------------------------------------------------------------------------

def flatten_docs(docs) -> Tuple[List[torch.Tensor], int]:
    """(tensors, docs_id) of an lp_varlen.LPDocs (([], -1) for None)."""
    if docs is None:
        return [], -1
    ts: list = []
    sk, _ = _flatten(docs, ts, _descend_types())
    key = (_struct_key(sk), tuple(_meta(t) for t in ts))
    with _LOCK:
        i = _DOCS_ID.get(key)
        if i is None:
            i = len(_DOCS)
            _DOCS.append(sk)
            _DOCS_ID[key] = i
    return ts, i


def _masks(plan, docs):
    from flash_attn.cute.lp_masks import LPMasks
    from . import _routes
    mask_kw = dict(window_size=plan.fwd.window or (None, None), softcap=plan.fwd.softcap)
    if docs is None:
        return _routes.masks_of(plan, mask_kw)
    return LPMasks.build(causal=plan.fwd.causal, docs=docs, **mask_kw)


# ---------------------------------------------------------------------------------------------------------------------
# the saved state: tensors -> (source, view) references
# ---------------------------------------------------------------------------------------------------------------------

class _Entry:
    """One call signature's recorded state layout."""
    __slots__ = ("key", "skel", "views", "n_in", "base_specs", "in_refs", "out_refs", "qkv", "out_meta", "lse_meta",
                 "checked")

    def __init__(self, **kw):
        for kk, vv in kw.items():
            setattr(self, kk, vv)


_FULL_CHECKS = 8   # calls per signature whose non-tensor saved values are compared with the first call's


def _sig_key(name, plan_h, docs_id, q, k, v, docs_ts):
    return (name, plan_h, docs_id, _meta(q), _meta(k), _meta(v), tuple(_meta(t) for t in docs_ts))


def _byte_off(t):
    return t.storage_offset() * t.element_size()


def _record_state(st, srcs):
    """Flatten the route's Saved state: every tensor becomes a view (dtype, size, stride, byte offset relative to its
    source) of an input / output (srcs: [("in", i, t)] + [("out", j, t)]) or of a base: the first saved tensor on each
    other storage, returned as an op output itself (the other saved tensors on that storage are views of it).
    Returns (skeleton, views, bases)."""
    ts: list = []
    sk, _ = _flatten(st, ts, _descend_types())
    by_ptr: dict = {}
    for kind, idx, s in srcs:
        if s.untyped_storage().nbytes() > 0:
            by_ptr.setdefault((s.device, s.untyped_storage().data_ptr()), []).append((kind, idx, s))
    bases, views = [], []
    for t in ts:
        dev = t.device
        stor = t.untyped_storage()
        if stor.nbytes() == 0:
            views.append(("E", None, t.dtype, tuple(t.shape), tuple(t.stride()), 0, dev))
            continue
        pk = (dev, stor.data_ptr())
        cands = by_ptr.get(pk)
        if cands is None:
            cands = by_ptr[pk] = [("base", len(bases), t)]
            bases.append(t)
        # inputs sharing one storage (q, k, v slices of one projection, or not, at the same signature): a tensor at an
        # input's own offset refers to that input, so the recorded layout does not depend on how the inputs alias
        src = next((c for c in cands if _byte_off(c[2]) == _byte_off(t)), cands[0])
        kind, idx, s = src
        rel = _byte_off(t) - _byte_off(s)
        views.append((kind, idx, t.dtype, tuple(t.shape), tuple(t.stride()), rel, dev))
    return sk, tuple(views), bases


def _rebuild_view(view, srcget):
    kind, idx, dtype, size, stride, rel, dev = view
    if kind == "E":
        return torch.empty_strided(size, stride, dtype=dtype, device=dev)
    s = srcget(kind, idx)
    if rel == 0 and s.dtype == dtype and tuple(s.shape) == size and tuple(s.stride()) == stride:
        return s   # the source itself (every base, and a saved input / output as it is)
    off = _byte_off(s) + rel
    isz = dtype.itemsize
    assert off % isz == 0, (off, isz)
    return torch.empty(0, dtype=dtype, device=s.device).set_(s.untyped_storage(), off // isz, size, stride)


def _same_leaves(a, b) -> bool:
    if a is b:
        return True
    if callable(a) and callable(b) and not isinstance(a, type):
        from flash_attn.cute import utils
        try:
            return utils.hash_callable(a) == utils.hash_callable(b)
        except Exception:
            return False
    try:
        r = a == b
        return bool(r) if not isinstance(r, torch.Tensor) else False
    except Exception:
        return False


def _same_skel(a, b, path="st") -> str:
    """'' if structurally equal (tensor slots by index, leaves by value), else the first differing path."""
    if a[0] != b[0]:
        return path
    kind = a[0]
    if kind == "T":
        return "" if a[1] == b[1] else path
    if kind in ("tuple", "list"):
        if len(a[1]) != len(b[1]):
            return path
        for i, (x, y) in enumerate(zip(a[1], b[1])):
            d = _same_skel(x, y, f"{path}[{i}]")
            if d:
                return d
        return ""
    if kind == "dict":
        if a[1] != b[1]:
            return path
        for kk, x, y in zip(a[1], a[2], b[2]):
            d = _same_skel(x, y, f"{path}[{kk!r}]")
            if d:
                return d
        return ""
    if kind == "obj":
        if a[1] is not b[1] or a[2] != b[2]:
            return path
        for n, x, y in zip(a[2], a[3], b[3]):
            d = _same_skel(x, y, f"{path}.{n}")
            if d:
                return d
        return ""
    return "" if _same_leaves(a[1], b[1]) else f"{path} ({a[1]!r} != {b[1]!r})"


# ---------------------------------------------------------------------------------------------------------------------
# the op bodies
# ---------------------------------------------------------------------------------------------------------------------

def _fwd_impl(name, q, k, v, docs, plan, docs_id):
    from . import _routes
    p = _PLANS[plan]
    dobj = None
    if docs_id >= 0:
        dts = list(docs)
        dobj = _unflatten(_DOCS[docs_id], lambda i: dts[i])
    masks = _masks(p, dobj)
    st = _routes.Saved()
    out, lse = _routes.forward(p, st, q, k, v, None, masks)
    if lse is None:
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    srcs = [("in", i, t) for i, t in enumerate([q, k, v] + list(docs))] + [("out", 0, out), ("out", 1, lse)]
    skel, views, bases = _record_state(st, srcs)
    key = _sig_key(name, plan, docs_id, q, k, v, docs)
    with _LOCK:
        e = _ENTRY_ID.get(key)
        if e is None:
            in_refs = tuple(sorted({vw[1] for vw in views if vw[0] == "in"}))
            out_refs = tuple(sorted({vw[1] for vw in views if vw[0] == "out"}))
            ent = _Entry(key=key, skel=skel, views=views, n_in=3 + len(docs),
                         base_specs=_base_specs(bases), in_refs=in_refs, out_refs=out_refs, checked=1,
                         qkv=tuple((tuple(t.shape), t.dtype) for t in (q, k, v)),
                         out_meta=(tuple(out.shape), tuple(out.stride()), out.dtype),
                         lse_meta=(tuple(lse.shape), tuple(lse.stride()), lse.dtype))
            _ENTRY_ID[key] = len(_ENTRIES)
            _ENTRIES.append(ent)
        else:
            ent = _ENTRIES[e]
            bad = ""
            if views != ent.views:
                bad = "the saved tensors' layout"
            elif _base_specs(bases) != ent.base_specs:
                bad = "the saved storages' sizes"
            elif ent.checked < _FULL_CHECKS:   # the non-tensor values: compared on the first calls of a signature
                ent.checked += 1
                bad = _same_skel(ent.skel, skel)
            if bad:
                raise RuntimeError(f"lowbit_attn::{name}: the route's saved state differs from the first call with the "
                                   f"same signature at {bad}; the op path needs a per-signature-static state.")
    return out, lse, bases


def _base_specs(bases):
    return tuple((tuple(b.shape), tuple(b.stride()), b.dtype, b.device) for b in bases)


def _entry_of(name, q, k, v, docs, plan, docs_id):
    return _ENTRY_ID.get(_sig_key(name, plan, docs_id, q, k, v, docs))


def _probe(name, q, k, v, docs, plan, docs_id):
    """Record a cold signature's entry for the fake implementation: the real op once on zero inputs of the same
    metadata (no documents only)."""
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    if docs_id >= 0:
        raise RuntimeError(f"lowbit_attn::{name}: torch.compile met a document-table call signature that never ran "
                           f"eagerly; run the call once outside torch.compile first (the fake implementation reads the "
                           f"saved-state layout the first real call records).")
    with unset_fake_temporarily(), torch.no_grad():
        def z(t):
            return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device).zero_()
        _fwd_impl(name, z(q), z(k), z(v), [], plan, docs_id)


def _fake(name, q, k, v, docs, plan, docs_id):
    e = _entry_of(name, q, k, v, docs, plan, docs_id)
    if e is None:
        _probe(name, q, k, v, docs, plan, docs_id)
        e = _entry_of(name, q, k, v, docs, plan, docs_id)
    ent = _ENTRIES[e]
    out = torch.empty_strided(ent.out_meta[0], ent.out_meta[1], dtype=ent.out_meta[2], device=q.device)
    lse = torch.empty_strided(ent.lse_meta[0], ent.lse_meta[1], dtype=ent.lse_meta[2], device=q.device)
    bases = [torch.empty_strided(sz, sd, dtype=dt, device=dev) for sz, sd, dt, dev in ent.base_specs]
    return out, lse, bases


def _rebuild_state(ent, state):
    """The route's Saved state of an entry from the op's saved tensors (inputs, outputs, state storages)."""
    n_in, n_out = len(ent.in_refs), len(ent.out_refs)
    ins = dict(zip(ent.in_refs, state[:n_in]))
    outs = dict(zip(ent.out_refs, state[n_in:n_in + n_out]))
    bases = state[n_in + n_out:]

    def srcget(kind, idx):
        return ins[idx] if kind == "in" else outs[idx] if kind == "out" else bases[idx]

    ts = [_rebuild_view(vw, srcget) for vw in ent.views]
    return _unflatten(ent.skel, lambda i: ts[i])


def _bwd_impl(dout, state, entry):
    from . import _routes
    ent = _ENTRIES[entry]
    st = _rebuild_state(ent, state)
    dq, dk, dv = _routes.backward(_PLANS[ent.key[1]], st, dout.contiguous())
    return dq, dk, dv


def state_of(grad_fn):
    """The route's saved state (lowprec._routes.Saved) behind an op node (``out.grad_fn``), rebuilt from the node's
    saved tensors, for inspection (the autograd.Function node keeps it as ``grad_fn.st``); valid until the backward
    frees them. Nothing is stored on the node (a back-reference would make a cycle that keeps the graph alive)."""
    st = getattr(grad_fn, "st", None)
    if st is not None:
        return st
    return _rebuild_state(_ENTRIES[grad_fn.lowbit_entry], list(grad_fn.saved_tensors))


def _bwd_fake(dout, state, entry):
    ent = _ENTRIES[entry]
    return tuple(dout.new_empty(s, dtype=dt) for s, dt in ent.qkv)


def _register():
    from torch.library import custom_op, register_autograd, register_fake
    ops = {}
    for name in ("fp4", "fp8"):
        def make(name):
            def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, docs: List[torch.Tensor], plan: int,
                    docs_id: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                return _fwd_impl(name, q, k, v, docs, plan, docs_id)

            def bwd(dout: torch.Tensor, state: List[torch.Tensor], entry: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return _bwd_impl(dout, state, entry)

            def fake(q, k, v, docs, plan, docs_id):
                return _fake(name, q, k, v, docs, plan, docs_id)

            def setup(ctx, inputs, output):
                q, k, v, docs, plan, docs_id = inputs
                e = _ENTRY_ID.get(_sig_key(name, plan, docs_id, q, k, v, docs))
                if e is None:
                    raise RuntimeError(f"lowbit_attn::{name}: no recorded entry for this call signature")
                ent = _ENTRIES[e]
                out, lse, bases = output
                ins = [q, k, v] + list(docs)
                outs = [out, lse]
                ctx.lowbit_entry = e
                ctx.lowbit_ndocs = len(docs)
                ctx.mark_non_differentiable(lse, *bases)
                # no zero-filled grads for lse / the bases (fp8 codes, scale factors: ~0.1 ms per backward);
                # backward reads only dout and handles it None
                ctx.set_materialize_grads(False)
                ctx.save_for_backward(*[ins[i] for i in ent.in_refs], *[outs[j] for j in ent.out_refs], *bases)

            fop = custom_op(f"lowbit_attn::{name}", fwd, mutates_args=())
            bop = custom_op(f"lowbit_attn::{name}_bwd", bwd, mutates_args=())
            register_fake(f"lowbit_attn::{name}_bwd")(_bwd_fake)
            register_fake(f"lowbit_attn::{name}")(fake)

            def backward(ctx, dout, dlse, dbases):
                nd = [None] * ctx.lowbit_ndocs
                if dout is None:
                    return None, None, None, nd, None, None
                dq, dk, dv = bop(dout, list(ctx.saved_tensors), ctx.lowbit_entry)
                return dq, dk, dv, nd, None, None

            register_autograd(f"lowbit_attn::{name}", backward, setup_context=setup)
            return fop
        ops[name] = make(name)
    return ops


OPS = _register()


def op_apply(plan, q, k, v, masks):
    """The op form of api._LowPrecAttnFunc.apply(q, k, v, plan, None, masks): masks is the call's LPMasks built from the
    plan's causal / window / softcap and the documents (no user score_mod / mask_mod / aux_tensors)."""
    docs = None if masks is None else masks.docs
    dts, did = flatten_docs(docs)
    out, lse, _ = OPS[plan.entry](q, k, v, dts, plan_handle(plan), did)
    if plan.fwd.return_lse:
        return out, lse
    return out

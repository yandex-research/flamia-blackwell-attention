"""Varlen benchmark: every configuration in BOTH formats on identical token content.

    python benchmarks/benchmark_packed_docs.py --set eq4x4096 --h 16 --hk 16 --d 128 --causal true --format varlen \
        --backend ours-fp8,ours-fp4,fa4-bf16 --mode fwd,bwd,step,infer --jsonl rows.jsonl [--no-acc] [--tag k=v]

Formats
  batched : (b, L, h, d). L = the longest document rounded up to a multiple of 128; every document is copied to the start
            of its row and the rest is zero padding (no key mask: a plain batched call, as a user without varlen would
            make). Causal: real rows are exact (the padding follows them). Dense: real rows also attend the zero padding
            keys, i.e. a DIFFERENT function; accuracy is then vs fp32 on the same padded inputs.
  varlen  : packed (total, h, d) + cu_seqlens (int32, (b + 1,)), each backend's own varlen API.
  aligned : the zero-copy form of ours: every document at a 128-aligned start of a (total_al, h, d)
            buffer (padding rows zero), cu_seqlens = the starts, seqused_q / seqused_k = the true lengths. Ours only.
Token content: one packed (total, h, d) draw (CPU generator, seed S for q, k, v, S + 7 for dO); batched is scattered from it.

Length sets (--set):  eq<N>x<L> | skew<L>+<N>x<l> | logu<lo>-<hi>-t<total>[-s<seed>]
  logu: log-uniform lengths in [lo, hi] (python random.Random(seed)), drawn until the sum reaches total; the last is cut
  so the sum is exactly total (a cut below 16 tokens is merged into the previous document).

Rows: TFLOPS on REAL tokens only (FA convention: 4 h d sum(len^2), causal len (len + 1) / 2; bwd 2.5x, step 3.5x),
ns per real token, padded token count, accuracy (cos vs a chunked fp32 reference over the real rows, same inputs).
Helpers (timing, quantisers, _Recorder, setup_public) come from benchmarks/benchmark_lowprec_attn.py.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("benchmark_lowprec_attn", os.path.join(_HERE, "benchmark_lowprec_attn.py"))
B = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_lowprec_attn"] = B
_spec.loader.exec_module(B)

FLOP_MULT = {"fwd": 1.0, "bwd": 2.5, "step": 3.5, "infer": 1.0}
MODES = ("fwd", "bwd", "step", "infer")


# -- length sets -------------------------------------------------------------------------------------------------------
def lengths_of(name):
    if name.startswith("eq"):
        n, l_ = name[2:].split("x")
        return [int(l_)] * int(n)
    if name.startswith("skew"):
        big, rest = name[4:].split("+")
        n, l_ = rest.split("x")
        return [int(big)] + [int(l_)] * int(n)
    if name.startswith("logu"):
        parts = name[4:].split("-")
        lo, hi = int(parts[0]), int(parts[1])
        total = int(parts[2][1:])
        seed = int(parts[3][1:]) if len(parts) > 3 else 0
        rng, out = random.Random(seed), []
        while sum(out) < total:
            out.append(int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))))))
        out[-1] -= sum(out) - total
        if out[-1] < 16:
            out[-2] += out.pop()
        return out
    raise SystemExit(f"unknown length set {name}")


def pairs(n, causal):
    return n * (n + 1) / 2 if causal else n * n


# -- fp32 reference over the real rows (per document; keys optionally zero-padded to kpad) ----------------------------
def reference_docs(q, k, v, g, lens, causal, bwd, kpad=None, chunk=512):
    """q, k, v, g packed (total, h|hk, d) real tokens. kpad: every document's keys are zero-padded to kpad (the dense
    batched function). Returns packed fp32 o, dq, dk, dv (real rows)."""
    h, hk, d = q.shape[1], k.shape[1], q.shape[2]
    grp, sc = h // hk, 1.0 / math.sqrt(d)
    o = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dq, dk, dv = (torch.zeros(t.shape, dtype=torch.float32, device=q.device) if bwd else None for t in (q, k, v))
    off = 0
    for n in lens:
        Q, K, V = (t[off:off + n].float().transpose(0, 1) for t in (q, k, v))  # (h, n, d)
        G = g[off:off + n].float().transpose(0, 1) if bwd else None
        m = n
        if kpad is not None and kpad > n:
            z = torch.zeros(hk, kpad - n, d, device=q.device)
            K, V, m = torch.cat([K, z], 1), torch.cat([V, z], 1), kpad
        K, V = K.repeat_interleave(grp, 0), V.repeat_interleave(grp, 0)
        dKf = torch.zeros(h, m, d, device=q.device) if bwd else None
        dVf = torch.zeros(h, m, d, device=q.device) if bwd else None
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            S = torch.einsum("hqd,hkd->hqk", Q[:, r0:r1], K) * sc
            if causal:
                rr = torch.arange(r0, r1, device=q.device)[:, None]
                cc = torch.arange(m, device=q.device)[None, :]
                S = S.masked_fill(cc > rr, float("-inf"))
            P = torch.softmax(S, -1)
            O = torch.einsum("hqk,hkd->hqd", P, V)
            o[off + r0:off + r1] = O.transpose(0, 1)
            if bwd:
                dO = G[:, r0:r1]
                dS = P * (torch.einsum("hqd,hkd->hqk", dO, V) - (dO * O).sum(-1, keepdim=True))
                dq[off + r0:off + r1] = (torch.einsum("hqk,hkd->hqd", dS, K) * sc).transpose(0, 1)
                dKf += torch.einsum("hqk,hqd->hkd", dS, Q[:, r0:r1]) * sc
                dVf += torch.einsum("hqk,hqd->hkd", P, dO)
            del S, P
        if bwd:
            dk[off:off + n] = dKf[:, :n].view(hk, grp, n, d).sum(1).transpose(0, 1)
            dv[off:off + n] = dVf[:, :n].view(hk, grp, n, d).sum(1).transpose(0, 1)
        off += n
    return dict(o=o, dq=dq, dk=dk, dv=dv)


# -- backends ----------------------------------------------------------------------------------------------------------
# setup(ctx) -> dict(mode -> closure returning {tensor: value in the format's layout}); missing mode = not offered.
# ctx: fmt, q, k, v, g (in the format's layout), causal, lens, cu (int32 cuda), cu_cpu, maxL, L (batched pad length),
#      total, b. A setup returning a string is a "unsupported" row with that reason.

OP_SCALING = None   # --op-scaling: the operand_scaling option of ours (None = the default)
# VL_OPTS='{"forward_two_cta": "on", ...}': more options fields of ours (both formats); VL_DET=1: deterministic
VL_OPTS = json.loads(os.environ.get("VL_OPTS", "{}") or "{}")
VL_DET = os.environ.get("VL_DET", "0") == "1"
# VL_KW='{"pv_precision": "fp4", "backward": "fp4"}': call keywords of ours (both formats)
VL_KW = json.loads(os.environ.get("VL_KW", "{}") or "{}")


def _ours_batched(ctx, entry):
    c = dict(q=ctx["q"], k=ctx["k"], v=ctx["v"], g=ctx["g"], causal=ctx["causal"], cli=None, window=None)
    _of = dict({"operand_scaling": OP_SCALING} if OP_SCALING else {}, **VL_OPTS)
    fns = B.setup_public(c, entry, dict(dict(deterministic=True) if VL_DET else {}, **VL_KW), _of, None)
    ctx["plan_info"], ctx["replay"], ctx["refused"] = c.get("plan_info"), c.get("replay", {}), c.get("refused", {})
    return dict(zip(("fwd", "bwd", "step", "infer"), fns[:4]))


def _first_tensors(r, n):
    if torch.is_tensor(r):
        return [r]
    if isinstance(r, (tuple, list)):
        return [x for x in r if torch.is_tensor(x)][:n]
    return []


def _ours_varlen(ctx, entry):
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    lp = B.lp
    fn = lp.flash_attn_fp4 if entry == "fp4" else lp.flash_attn_fp8
    resolve = lp.resolve_fp4 if entry == "fp4" else lp.resolve_fp8
    # VL3_TUNING="name=0,name=1" -> options.tuning (KernelTuning booleans), benchmark-side only
    _tn = {kv.split("=")[0]: kv.split("=")[1] == "1" for kv in os.environ.get("VL3_TUNING", "").split(",") if kv}
    _tn.update(json.loads(os.environ.get("VL_TUNING_JSON", "{}") or "{}"))   # string / bool tuning fields
    _tk = dict(tuning=lp.KernelTuning(**_tn)) if _tn else {}
    if os.environ.get("VL3_L2_MB"):  # the work lists' L2 budget (lp_varlen._L2_BYTES)
        import flash_attn.cute.lp_varlen as _LV
        _LV._L2_BYTES = int(float(os.environ["VL3_L2_MB"]) * 1024 * 1024)
    if OP_SCALING:
        _tk["operand_scaling"] = OP_SCALING
    _tk.update(VL_OPTS)
    opts = lp.FP4AttnOptions(**_tk) if entry == "fp4" else lp.FP8AttnOptions(**_tk)
    kw = dict(causal=causal, options=opts, cu_seqlens_q=ctx["cu"], cu_seqlens_k=ctx["cu"],
              max_seqlen_q=ctx["maxL"], max_seqlen_k=ctx["maxL"])
    if ctx["fmt"] == "docmaps":   # the packed-documents path of training: compact DocMaps, no cu_seqlens
        kw = dict(causal=causal, options=opts, doc_maps=ctx["maps"])
    if VL_DET:
        kw["deterministic"] = True
    kw.update(VL_KW)
    if ctx["fmt"] == "aligned":
        kw.update(seqused_q=ctx["used"], seqused_k=ctx["used"])
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    info, refused = {}, {}
    for tag, args in (("train", (qg, kg, vg)), ("infer", (q, k, v))):
        try:
            if ctx["fmt"] == "docmaps":   # the plan is the maps' aligned buffer's: resolve on its stand-ins
                _m = ctx["maps"]
                _st = [_m.stand_in(t, s_) for t, s_ in zip(args, ("q", "k", "k"))]
                if tag == "train":
                    _st = [t.detach().requires_grad_(True) for t in _st]
                _kr = {x: y for x, y in kw.items() if x != "doc_maps"}
                _kr.update(cu_seqlens_q=_m.cu_pad_q, cu_seqlens_k=_m.cu_pad_k, seqused_q=_m.len_q,
                           seqused_k=_m.len_k, max_seqlen_q=_m.max_seqlen_q, max_seqlen_k=_m.max_seqlen_k)
                p = resolve(*_st, **_kr)
            else:
                p = resolve(*args, **kw)
            info[tag] = dict(route=p.route, fwd_kernel=p.fwd.kernel, operand_scaling=p.fwd.operand_scaling,
                             bwd_kernel=p.bwd.kernel, ds_scaling=p.bwd.ds_scaling, pending=list(p.pending))
        except (NotImplementedError, ValueError) as e:
            refused[tag] = f"{type(e).__name__}(rule={getattr(e, 'rule', None)}): {e}"
    ctx["plan_info"], ctx["refused"] = info, refused
    call = lambda q_, k_, v_, causal=causal: fn(q_, k_, v_, **dict(kw, causal=causal))  # noqa: E731
    out = {}
    if "infer" in info:
        out["infer"] = lambda: dict(o=call(q, k, v))
    if "train" not in info:
        return out
    rec = B._Recorder()
    try:
        rec.phase = "fwd"
        o_keep = call(qg, kg, vg)
        o_api = o_keep.detach().clone()
        rec.phase = "bwd"
        api_grads = [x.clone() for x in torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True)]
        rec.phase = None
    finally:
        rec.restore()
    ctx["replay"], ctx["api_out"] = {}, dict(fwd=dict(o=o_api), bwd=dict(zip(("dq", "dk", "dv"), api_grads)))
    if ctx["fmt"] == "docmaps":
        # the recorder replays the aligned-layout kernels; the packed path launches different ones. fwd / bwd are
        # the public forward (autograd, saves for backward) and autograd.grad(retain_graph) of the same call: they include
        # the packed-row quantisers / scatters, unlike the kernel-only fwd / bwd of the aligned rows. step / infer are
        # directly comparable with the other formats.
        out["fwd"] = lambda: dict(o=call(qg, kg, vg))
        out["bwd"] = lambda: dict(zip(("dq", "dk", "dv"), torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True)))
        ctx["replay"] = dict(fwd=dict(calls=["public forward (docmaps)"]), bwd=dict(calls=["autograd.grad(retain_graph)"]))
        out["step"] = B._train_step(call, dict(ctx, q=q, k=k, v=v, g=g))
        return out
    for ph, n in (("fwd", 1), ("bwd", 3)):
        if not rec.calls[ph]:
            continue
        run, check = rec.replay(ph), rec.replay(ph, pin=True)
        r1 = [x.clone() for x in _first_tensors(check(), n)]
        r2 = _first_tensors(check(), n)
        drift = min((B._cos(a, b) for a, b in zip(r1, r2)), default=1.0)
        if drift < 0.999:
            for c in rec.calls[ph]:
                c["restore"] = bool(c["mut"])
        ctx["replay"][ph] = dict(calls=[c["label"] for c in rec.calls[ph]], repeat_cos=round(drift, 6),
                                 restored=[list(t.shape) for c in rec.calls[ph] if c["restore"] for t, _ in c["mut"]])
        # kernel-only replay: the outputs are in the repacked (1, s_pad, h, d) layout; accuracy comes from the API call
        out[ph] = (lambda run=run: (run(), {})[1])
    out["step"] = B._train_step(call, dict(ctx, q=q, k=k, v=v, g=g))
    return out


def setup_ours(ctx, entry):
    return _ours_batched(ctx, entry) if ctx["fmt"] == "batched" else _ours_varlen(ctx, entry)


def setup_fa4(ctx):
    if ctx["fmt"] == "aligned":
        return "aligned: ours only"
    fa4 = B.fa4
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    vk = {} if ctx["fmt"] == "batched" else dict(cu_seqlens_q=ctx["cu"], cu_seqlens_k=ctx["cu"],
                                                 max_seqlen_q=ctx["maxL"], max_seqlen_k=ctx["maxL"])
    out, lse = fa4._flash_attn_fwd(q, k, v, causal=causal, return_lse=True, **vk)[:2]
    if ctx["fmt"] == "batched":
        pub = fa4.flash_attn_func
    else:
        pub = lambda q_, k_, v_, causal: fa4.flash_attn_varlen_func(q_, k_, v_, causal=causal, **vk)  # noqa: E731

    def infer():
        with torch.no_grad():
            r = pub(q, k, v, causal=causal)
        return dict(o=r[0] if isinstance(r, tuple) else r)
    return dict(fwd=lambda: dict(o=fa4._flash_attn_fwd(q, k, v, causal=causal, return_lse=True, **vk)[0]),
                bwd=lambda: dict(zip(("dq", "dk", "dv"), fa4._flash_attn_bwd(q, k, v, out, g, lse, causal=causal, **vk)[:3])),
                step=B._train_step(pub, ctx), infer=infer)


def setup_sdpa(ctx):
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    if ctx["fmt"] == "batched":
        f, b_, st = B.setup_torch_sdpa(dict(q=q, k=k, v=v, g=g, causal=causal))
        return dict(fwd=f, bwd=b_, step=st, infer=f)
    # varlen: nested jagged tensors (torch.nested, layout=torch.jagged)
    gqa = dict(enable_gqa=True) if k.shape[1] != q.shape[1] else {}
    off = ctx["cu"].long()

    def nj(t):
        return torch.nested.nested_tensor_from_jagged(t, off, max_seqlen=ctx["maxL"])

    def sdpa(q_, k_, v_, causal):
        o = torch.nn.functional.scaled_dot_product_attention(nj(q_).transpose(1, 2), nj(k_).transpose(1, 2),
                                                             nj(v_).transpose(1, 2), is_causal=causal, **gqa)
        return o.transpose(1, 2).values()
    sdpa(q, k, v, causal)  # probe: raises if NJT SDPA refuses this call
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o_keep = sdpa(qg, kg, vg, causal)
    f = lambda: dict(o=sdpa(q, k, v, causal))  # noqa: E731
    return dict(fwd=f, infer=f,
                bwd=lambda: dict(zip(("dq", "dk", "dv"), torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True))),
                step=B._train_step(sdpa, ctx))


def setup_flashinfer(ctx, prec, backend):
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper, trtllm_ragged_attention_deepseek
    q, k, v, causal = ctx["q"], ctx["k"], ctx["v"], ctx["causal"]
    sc = 1 / math.sqrt(q.shape[-1])
    h, hk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    if hk != h and backend != "cutlass":
        return f"flashinfer {backend}: no GQA (h {h} != h_kv {hk}) in this wrapper"
    if ctx["fmt"] == "batched":  # FlashInfer has no 4-D API: the padded batch as uniform-length ragged rows
        b, L = q.shape[0], q.shape[1]
        lens_cpu = torch.full((b,), L, dtype=torch.int32)
        flat = lambda t: t.reshape(b * L, t.shape[2], d)  # noqa: E731
        ind = torch.arange(0, (b + 1) * L, L, device="cuda", dtype=torch.int32)
        maxL, shape_o = L, q.shape
    else:
        lens_cpu = torch.tensor(ctx["lens"], dtype=torch.int32)
        flat = lambda t: t  # noqa: E731
        ind, maxL, shape_o = ctx["cu"], ctx["maxL"], q.shape
    b = lens_cpu.numel()
    lens_dev = lens_cpu.cuda()
    src, sq, sk, sv, dsc = (q, k, v), 1.0, 1.0, 1.0, None
    qr, kr, vr = (flat(t) for t in (q, k, v))
    if prec == "fp8":  # FlashInfer takes host scales: read once here, outside the timed region
        qkv8 = [B.e4m3_dev(t) for t in (q, k, v)]
        dsc = [s_ for _, s_ in qkv8]
        (qr, sq), (kr, sk), (vr, sv) = ((flat(c), float(s_)) for c, s_ in qkv8)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    o = torch.empty(qr.shape[0], h, d, device="cuda", dtype=torch.bfloat16)
    st = dict(q=qr, k=kr, v=vr)
    if backend == "cutlass":
        w = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="cutlass")
        w.plan(ind, ind, h, hk, d, causal=causal, q_data_type=qr.dtype, kv_data_type=kr.dtype, o_data_type=torch.bfloat16)
        kw = dict(q_scale=sq, k_scale=sk, v_scale=sv) if prec == "fp8" else {}
        fn = lambda: w.run(st["q"], st["k"], st["v"], out=o, **kw)  # noqa: E731
    else:
        fn = lambda: trtllm_ragged_attention_deepseek(  # noqa: E731
            query=st["q"], key=st["k"], value=st["v"], workspace_buffer=ws, seq_lens=lens_dev, max_q_len=maxL,
            max_kv_len=maxL, bmm1_scale=sq * sk * sc, bmm2_scale=sv, o_sf_scale=1.0, batch_size=b, window_left=-1,
            cum_seq_lens_q=ind, cum_seq_lens_kv=ind, enable_pdl=False, is_causal=causal, return_lse=False, out=o,
            q_seq_lens_cpu=lens_cpu, kv_seq_lens_cpu=lens_cpu, backend={"cutedsl": "cute-dsl", "trtllmgen": "trtllm-gen"}[backend])
    res = dict(fwd=lambda: (fn(), dict(o=o.view(shape_o)))[1])
    if prec == "fp8":
        def infer():
            st["q"], st["k"], st["v"] = (flat(B.e4m3_dev(x, s_)[0]) for x, s_ in zip(src, dsc))
            fn()
            return dict(o=o.view(shape_o))
        res["infer"] = infer
    else:
        res["infer"] = res["fwd"]
    return res


def setup_meta(ctx):
    lpfa4 = B.lpfa4
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    if ctx["fmt"] != "varlen":
        return "meta-mxfp8: varlen API only (Mxfp8VarlenMeta); equal-length varlen = its batched call"
    if causal:
        return "meta-mxfp8: the public MXFP8 functions have no causal mode"
    if k.shape[1] != q.shape[1]:
        return "meta-mxfp8: no GQA"
    meta = lpfa4.Mxfp8VarlenMeta.from_lengths(ctx["lens"], device="cuda")
    quant = lambda t, o: lpfa4.quantize_mxfp8_varlen(t, meta, orientations=o, scaling_mode="rceil")  # noqa: E731
    qf, kf, vf = quant(q, "hdim"), quant(k, "hdim"), quant(v, "seq")
    lpfa4.mxfp8_flash_attn_varlen_forward(qf, kf, vf, meta, meta, return_lse=True)
    qb, kb, vb = (quant(t, "both") for t in (q, k, v))
    ob, lb = lpfa4.mxfp8_flash_attn_varlen_forward(qb, kb, vb, meta, meta, return_lse=True)
    grad = lambda r: dict(dq=r[0], dk=r[1], dv=r[2])  # noqa: E731

    def step():
        q2, k2, v2 = (quant(t, "both") for t in (q, k, v))
        o2, l2 = lpfa4.mxfp8_flash_attn_varlen_forward(q2, k2, v2, meta, meta, return_lse=True)
        return dict(o=o2, **grad(lpfa4.mxfp8_flash_attn_varlen_backward(q2, k2, v2, o2, g, l2, meta, meta)))

    def infer():
        q2, k2, v2 = quant(q, "hdim"), quant(k, "hdim"), quant(v, "seq")
        return dict(o=lpfa4.mxfp8_flash_attn_varlen_forward(q2, k2, v2, meta, meta, return_lse=False)[0])
    return dict(fwd=lambda: dict(o=lpfa4.mxfp8_flash_attn_varlen_forward(qf, kf, vf, meta, meta, return_lse=False)[0]),
                bwd=lambda: grad(lpfa4.mxfp8_flash_attn_varlen_backward(qb, kb, vb, ob, g, lb, meta, meta)),
                step=step, infer=infer)


def setup_cudnn(ctx, fmt):
    """cuDNN SDPA graphs. batched: the benchmark's adapter (B.setup_cudnn). varlen: the same graphs on packed THD
    tensors with ragged offsets (cu_seqlens * heads * d), padding mask and per-document seq_len_q / seq_len_kv."""
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    if ctx["fmt"] == "batched":
        r = B.setup_cudnn(dict(q=q, k=k, v=v, g=g, causal=causal), fmt)
        r = list(r) + [None] * (4 - len(r))
        return dict(fwd=r[0], bwd=r[1], step=r[2], infer=r[3] if r[3] is not None else (r[0] if fmt == "bf16" else None))
    return _cudnn_ragged(ctx, fmt)


def _cudnn_ragged(ctx, fmt):
    cudnn = B.cudnn
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    sc = 1 / math.sqrt(q.shape[-1])
    T, h, d = q.shape
    hk = k.shape[1]
    b, s = ctx["b"], ctx["maxL"]
    F32, BF, E4, I32 = cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.INT32
    dim, st = [b, h, s, d], [s * h * d, d, h * d, 1]
    kv = ([b, hk, s, d], [s * hk * d, d, hk * d, 1])
    # BF16: ragged (THD) stats. With padded (b, h, s) stats the BF16 ragged backward returns wrong gradients (measured
    # dQ cos 0.62 causal); the FP8 graphs are accurate with padded stats and keep them.
    rag_stats = fmt == "bf16"
    sdim, sst = [b, h, s, 1], ([s * h, 1, h, 1] if rag_stats else [h * s, s, 1, 1])
    one = [1, 1, 1, 1]
    mask = dict(diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT)
    cu = ctx["cu"]
    ro_q = (cu.long() * h * d).to(torch.int32).view(b + 1, 1, 1, 1).contiguous()
    ro_k = (cu.long() * hk * d).to(torch.int32).view(b + 1, 1, 1, 1).contiguous()
    lens = torch.tensor(ctx["lens"], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    hd = cudnn.create_handle()
    new = lambda: cudnn.pygraph(handle=hd, intermediate_data_type=F32, compute_data_type=F32)  # noqa: E731
    RQ, RK, SQ, SK, RS = 90, 91, 92, 93, 94
    ro_s = (cu.long() * h).to(torch.int32).view(b + 1, 1, 1, 1).contiguous()

    def rag(gr):
        return dict(q=gr.tensor(name="roq", dim=[b + 1, 1, 1, 1], stride=one, data_type=I32, uid=RQ),
                    k=gr.tensor(name="rok", dim=[b + 1, 1, 1, 1], stride=one, data_type=I32, uid=RK),
                    sq=gr.tensor(name="slq", dim=[b, 1, 1, 1], stride=one, data_type=I32, uid=SQ),
                    sk=gr.tensor(name="slk", dim=[b, 1, 1, 1], stride=one, data_type=I32, uid=SK),
                    s=gr.tensor(name="ros", dim=[b + 1, 1, 1, 1], stride=one, data_type=I32, uid=RS))

    def ten(gr, R, name, uid, dt, side="q", shape=None):
        shape = shape or ((dim, st) if side == "q" else kv)
        t = gr.tensor(name=name, dim=shape[0], stride=shape[1], data_type=dt, uid=uid)
        if side in ("q", "k") or (side == "s" and rag_stats):
            t.set_ragged_offset(R[side])
        return t

    def out(R, t, uid, dt, side="q"):
        shape = (dim, st) if side == "q" else (kv if side == "k" else (sdim, sst))
        t.set_uid(uid).set_output(True).set_dim(shape[0]).set_stride(shape[1]).set_data_type(dt)
        if side in ("q", "k") or (side == "s" and rag_stats):
            t.set_ragged_offset(R[side])

    def run(gr, pack):
        gr.validate(), gr.build_operation_graph(), gr.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        gr.check_support(), gr.build_plans()
        ws = torch.empty(max(gr.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        pack.update({RQ: ro_q, RK: ro_k, SQ: lens, SK: lens, RS: ro_s})

        def call():
            cudnn.set_stream(handle=hd, stream=torch.cuda.current_stream().cuda_stream)
            gr.execute(pack, ws, handle=hd)
        return call
    pad = dict(use_padding_mask=True)
    o = torch.zeros(T, h, d, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(((T + 1) * h,) if rag_stats else (b, h, s, 1), dtype=torch.float32, device="cuda")
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    grads = lambda: dict(dq=dq, dk=dk, dv=dv)  # noqa: E731
    if fmt == "bf16":
        gr = new()
        R = rag(gr)
        o_, l_ = gr.sdpa(name="f", q=ten(gr, R, "q", 1, BF), k=ten(gr, R, "k", 2, BF, "k"), v=ten(gr, R, "v", 3, BF, "k"),
                         generate_stats=True, attn_scale=sc, diagonal_band_right_bound=0 if causal else None,
                         seq_len_q=R["sq"], seq_len_kv=R["sk"], **pad, **mask)
        out(R, o_, 10, BF), out(R, l_, 11, F32, "s")
        fwd = run(gr, {1: q, 2: k, 3: v, 10: o, 11: lse})
        fwd()
        gr = new()
        R = rag(gr)
        r = gr.sdpa_backward(name="b", q=ten(gr, R, "q", 1, BF), k=ten(gr, R, "k", 2, BF, "k"), v=ten(gr, R, "v", 3, BF, "k"),
                             o=ten(gr, R, "o", 10, BF), dO=ten(gr, R, "dO", 12, BF),
                             stats=ten(gr, R, "l", 11, F32, "s", (sdim, sst)), attn_scale=sc,
                             diagonal_band_right_bound=0 if causal else None, seq_len_q=R["sq"], seq_len_kv=R["sk"],
                             **pad, **mask)
        [out(R, t, u, BF, sd) for t, u, sd in zip(r, (20, 21, 22), ("q", "k", "k"))]
        bwd = run(gr, {1: q, 2: k, 3: v, 10: o, 12: g, 11: lse, 20: dq, 21: dk, 22: dv})
        fo = lambda: (fwd(), dict(o=o))[1]  # noqa: E731
        return dict(fwd=fo, bwd=lambda: (bwd(), grads())[1], step=lambda: (fwd(), bwd(), dict(o=o, **grads()))[2],
                    infer=fo)
    # fp8: per-tensor E4M3 operands (the library quantiser), output scales calibrated as B.setup_cudnn does
    (q8, sq_), (k8, sk_), (v8, sv_), (do8, sdo) = (B.e4m3_dev(t) for t in (q, k, v, g))
    scal = lambda x: (x if torch.is_tensor(x) else torch.tensor(float(x), device="cuda")).float().reshape(1, 1, 1, 1)  # noqa: E731
    ref = reference_docs(*(t.float() * s_.float() for t, s_ in ((q8, sq_), (k8, sk_), (v8, sv_), (do8, sdo))),
                         ctx["lens"], causal, True)
    so, sdq, sdk, sdv = (448.0 / float(ref[t].abs().amax()) for t in ("o", "dq", "dk", "dv"))
    fsc = ["descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"]

    def fwd_graph(o_dt, o_t, so_, lse_):
        gr = new()
        R = rag(gr)
        Tn = {n: gr.tensor(name=n, dim=one, stride=one, data_type=F32, uid=40 + i) for i, n in enumerate(fsc)}
        r = gr.sdpa_fp8(name="f", q=ten(gr, R, "q", 1, E4), k=ten(gr, R, "k", 2, E4, "k"), v=ten(gr, R, "v", 3, E4, "k"),
                        generate_stats=True, attn_scale=sc, right_bound=0 if causal else None,
                        seq_len_q=R["sq"], seq_len_kv=R["sk"], **pad, **mask, **Tn)
        out(R, r[0], 10, o_dt), out(R, r[1], 11, F32, "s")
        for t, u in ((r[2], 12), (r[3], 13)):
            t.set_uid(u).set_output(True).set_dim(one).set_stride(one).set_data_type(F32)
        vals = [sq_, sk_, sv_, 1 / 448, 448.0, so_]
        pack = {1: q8, 2: k8, 3: v8, 10: o_t, 11: lse_, 12: scal(0), 13: scal(0),
                **{40 + i: scal(x) for i, x in enumerate(vals)}}
        return run(gr, pack), pack
    fwd, _ = fwd_graph(BF, o, 1.0, lse)
    lse8 = torch.zeros_like(lse)
    o8 = torch.zeros(T, h, d, dtype=torch.float8_e4m3fn, device="cuda")
    fwd8, pf = fwd_graph(E4, o8, so, lse8)
    fwd8()
    bsc = ["descale_q", "descale_k", "descale_v", "descale_o", "descale_dO", "descale_s", "descale_dP", "scale_s",
           "scale_dQ", "scale_dK", "scale_dV", "scale_dP"]
    gr = new()
    R = rag(gr)
    Tn = {n: gr.tensor(name=n, dim=one, stride=one, data_type=F32, uid=40 + i) for i, n in enumerate(bsc)}
    r = gr.sdpa_fp8_backward(name="b", q=ten(gr, R, "q", 1, E4), k=ten(gr, R, "k", 2, E4, "k"), v=ten(gr, R, "v", 3, E4, "k"),
                             o=ten(gr, R, "o", 10, E4), dO=ten(gr, R, "dO", 14, E4),
                             stats=ten(gr, R, "l", 11, F32, "s", (sdim, sst)), attn_scale=sc,
                             right_bound=0 if causal else None, seq_len_q=R["sq"], seq_len_kv=R["sk"], **pad, **mask, **Tn)
    g8 = [torch.zeros(T, n, d, dtype=torch.float8_e4m3fn, device="cuda") for n in (h, hk, hk)]
    [out(R, t, u, E4, sd) for t, u, sd in zip(r[:3], (20, 21, 22), ("q", "k", "k"))]
    [t.set_uid(60 + i).set_output(True).set_dim(one).set_stride(one).set_data_type(F32) for i, t in enumerate(r[3:])]
    amax = [scal(0) for _ in range(4)]
    vals = [sq_, sk_, sv_, 1 / so, sdo, 1 / 448, 1.0, 448.0, sdq, sdk, sdv, 1.0]
    pb = {1: q8, 2: k8, 3: v8, 10: o8, 14: do8, 11: lse8, 20: g8[0], 21: g8[1], 22: g8[2],
          **{60 + i: a for i, a in enumerate(amax)}, **{40 + i: scal(x) for i, x in enumerate(vals)}}
    bwd = run(gr, pb)
    bwd()
    sdp = 448.0 / float(amax[3]) if float(amax[3]) > 0 else 1.0
    pb[40 + 6].fill_(1 / sdp), pb[40 + 11].fill_(sdp)
    grads8 = lambda: dict(dq=(g8[0], sdq), dk=(g8[1], sdk), dv=(g8[2], sdv))  # noqa: E731

    def step():
        for uid, x, s_ in ((1, q, sq_), (2, k, sk_), (3, v, sv_), (14, g, sdo)):
            pf[uid] = pb[uid] = B.e4m3_dev(x, s_)[0]
        fwd8(), bwd()
        return dict(o=(o8, so), **grads8())
    fwdi, pi = fwd_graph(BF, o, 1.0, lse)

    def infer():
        for uid, x, s_ in ((1, q, sq_), (2, k, sk_), (3, v, sv_)):
            pi[uid] = B.e4m3_dev(x, s_)[0]
        fwdi()
        return dict(o=o)
    return dict(fwd=lambda: (fwd(), dict(o=o))[1], bwd=lambda: (bwd(), grads8())[1], step=step, infer=infer)


BACKENDS = {
    "ours-fp8": lambda c: setup_ours(c, "fp8"),
    "ours-fp4": lambda c: setup_ours(c, "fp4"),
    "fa4-bf16": setup_fa4,
    "cudnn-bf16": lambda c: setup_cudnn(c, "bf16"),
    "cudnn-fp8": lambda c: setup_cudnn(c, "fp8"),
    "flashinfer-bf16-cutedsl": lambda c: setup_flashinfer(c, "bf16", "cutedsl"),
    "flashinfer-fp8-cutedsl": lambda c: setup_flashinfer(c, "fp8", "cutedsl"),
    "flashinfer-bf16-cutlass": lambda c: setup_flashinfer(c, "bf16", "cutlass"),
    "flashinfer-fp8-cutlass": lambda c: setup_flashinfer(c, "fp8", "cutlass"),
    "torch-sdpa": setup_sdpa,
    "meta-mxfp8": setup_meta,
}


# -- main --------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--hk", type=int, default=None)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--causal", required=True, choices=["true", "false"])
    ap.add_argument("--format", required=True, choices=["batched", "varlen", "aligned", "docmaps"])
    ap.add_argument("--dm-n", default="docs", help="docmaps: max_num_documents = docs | T | <int>")
    ap.add_argument("--op-scaling", default=None, help="ours: options.operand_scaling (per_tensor / token_block)")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--mode", default="fwd,bwd,step,infer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-acc", action="store_true")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--row-predicate", action="store_true", help="aligned: the quantisers' in-kernel row predicate")
    a = ap.parse_args()
    global OP_SCALING
    if a.op_scaling:
        OP_SCALING = a.op_scaling
        a.tag.append(f"op_scaling={a.op_scaling}")
    sys.argv = sys.argv[:1]  # the CuTe DSL's diagnostic parser reads sys.argv at compile time ("--h 16" = --help)
    if a.row_predicate:
        import flash_attn.cute.lp_varlen as _LV
        _LV.ALIGNED_ROW_PREDICATE = True
        a.tag.append("row_predicate=1")
    import flash_attn
    print(f"flash_attn from {os.path.dirname(flash_attn.__file__)}", flush=True)
    tags = dict(t.split("=", 1) for t in a.tag)
    hk = a.hk or a.h
    causal = a.causal == "true"
    lens = lengths_of(a.set)
    total, nb, maxL = sum(lens), len(lens), max(lens)
    L = -(-maxL // 128) * 128
    gen = torch.Generator().manual_seed(a.seed)
    q, k, v = (torch.randn(total, n, a.d, generator=gen, dtype=torch.float32).to(torch.bfloat16) for n in (a.h, hk, hk))
    g = torch.randn(total, a.h, a.d, generator=torch.Generator().manual_seed(a.seed + 7),
                    dtype=torch.float32).to(torch.bfloat16)
    q, k, v, g = (t.cuda() for t in (q, k, v, g))
    cu_cpu = torch.zeros(nb + 1, dtype=torch.int32)
    cu_cpu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    cu = cu_cpu.cuda()
    # packed row -> (doc, pos) for the batched layout
    doc = torch.repeat_interleave(torch.arange(nb), torch.tensor(lens))
    pos = torch.arange(total) - cu_cpu[:-1].long()[doc]
    bidx = (doc * L + pos).cuda()

    def to_batched(t):
        z = torch.zeros(nb * L, t.shape[1], t.shape[2], dtype=t.dtype, device=t.device)
        z.index_copy_(0, bidx, t)
        return z.view(nb, L, t.shape[1], t.shape[2])

    def from_batched(t):
        return t.reshape(nb * L, t.shape[2], t.shape[3]).index_select(0, bidx)
    # aligned: every document at a 128-aligned start, the true lengths as seqused
    _al = int(os.environ.get("VL_ALIGN", "128"))   # 256: the 2-CTA backward's document starts
    al_lens = [-(-n // _al) * _al for n in lens]
    al_cu_cpu = torch.zeros(nb + 1, dtype=torch.int32)
    al_cu_cpu[1:] = torch.cumsum(torch.tensor(al_lens, dtype=torch.int32), 0)
    total_al = int(al_cu_cpu[-1])
    aidx = (al_cu_cpu[:-1].long()[doc] + pos).cuda()

    def to_aligned(t):
        z = torch.zeros(total_al, t.shape[1], t.shape[2], dtype=t.dtype, device=t.device)
        z.index_copy_(0, aidx, t)
        return z

    def from_aligned(t):
        return t.index_select(0, aidx)
    if a.format == "batched":
        X = dict(q=to_batched(q), k=to_batched(k), v=to_batched(v), g=to_batched(g))
        pad_tokens = nb * L
    elif a.format == "aligned":
        X = dict(q=to_aligned(q), k=to_aligned(k), v=to_aligned(v), g=to_aligned(g))
        pad_tokens = total_al
    else:
        X = dict(q=q, k=k, v=v, g=g)
        pad_tokens = total
    maps = None
    if a.format == "docmaps":
        from flash_attn.cute.lp_docmaps import DocMaps
        _n = nb if a.dm_n == "docs" else (total if a.dm_n == "T" else int(a.dm_n))
        maps = DocMaps.from_cu_seqlens(cu_cpu, max_num_documents=_n, max_seqlen=maxL).to("cuda")
        tags["dm_n"] = str(_n)
        tags["dm_compact"] = str(bool(maps.compact))
    real_pairs = sum(pairs(n, causal) for n in lens)
    pad_pairs = nb * pairs(L, causal) if a.format == "batched" else real_pairs
    shape_id = f"{a.set}-h{a.h}" + (f"-hk{hk}" if hk != a.h else "") + f"-d{a.d}-{'causal' if causal else 'dense'}"
    base = dict(shape_id=shape_id, length_set=a.set, format=a.format, causal=causal, h=a.h, h_kv=hk, d=a.d,
                n_docs=nb, total_tokens=total, max_len=maxL, pad_len=L if a.format == "batched" else None,
                computed_tokens=pad_tokens, seed=a.seed, bench="benchmark_packed_docs.py",
                dense_padded_unmasked=(a.format == "batched" and not causal and L * nb != total), **tags)
    refs = {}
    # calls per CUDA graph: 20, fewer when the computed (padded) batch is large -- the graph keeps every call's outputs
    n_calls = 20 if pad_tokens <= 70000 else max(2, int(20 * 70000 / pad_tokens))
    base["graph_calls"] = n_calls
    if a.format == "batched" and pad_tokens * max(a.h, hk) * a.d >= 2 ** 31:
        why = (f"padded batch not run: {nb} x {L} tokens ({pad_tokens / total:.1f}x padding), {pad_tokens * a.h * a.d:.2e} "
               "elements per tensor >= 2^31 (FlashInfer faults with an illegal address; the graph run OOMs on 178 GB)")
        for name in a.backend.split(","):
            emit_ = dict(base, config=name, status="unsupported", error=why)
            if a.jsonl:
                with open(a.jsonl, "a") as fh:
                    fh.write(json.dumps(emit_) + "\n")
        print(why)
        return

    def ref(bwd):
        if bwd not in refs:
            kp = L if (a.format == "batched" and not causal) else None
            refs[bwd] = reference_docs(q, k, v, g, lens, causal, bwd, kpad=kp)
        return refs[bwd]

    def emit(row):
        if a.jsonl:
            with open(a.jsonl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    modes = a.mode.split(",")
    for name in a.backend.split(","):
        ctx = dict(fmt=a.format, causal=causal, lens=lens, cu=cu, cu_cpu=cu_cpu, maxL=maxL, L=L, total=total, b=nb, maps=maps, **X)
        if a.format == "aligned":
            ctx.update(cu=al_cu_cpu.cuda(), cu_cpu=al_cu_cpu,
                       used=torch.tensor(lens, dtype=torch.int32, device="cuda"))
            if not name.startswith("ours"):
                emit(dict(base, config=name, status="unsupported", error="aligned: ours only"))
                continue
        try:
            fns = BACKENDS[name](ctx)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name} {a.format} {shape_id}: {type(e).__name__}: {str(e)[:400]}", flush=True)
            emit(dict(base, config=name, status="error", error=f"{type(e).__name__}: {str(e)[:1500]}"))
            continue
        if isinstance(fns, str):
            print(f"unsupported {name} {a.format} {shape_id}: {fns}", flush=True)
            emit(dict(base, config=name, status="unsupported", error=fns))
            continue
        for tag_, why in (ctx.get("refused") or {}).items():
            emit(dict(base, config=name, status="refused", refused_for=tag_, error=why))
        for mode in MODES:
            fn = fns.get(mode)
            if mode not in modes:
                continue
            if fn is None:
                emit(dict(base, config=name, kernel=mode, status="unsupported", error="mode not offered by this backend"))
                continue
            try:
                fn()
                torch.cuda.synchronize()
                ms, timing = B.bench(fn, mode, "graph", n=n_calls)
                outs = fn()
                outs = {t: x[0].float() / x[1] if isinstance(x, tuple) else x for t, x in outs.items()}
                acc_src = "closure"
                if not outs and ctx.get("api_out", {}).get(mode):
                    outs, acc_src = ctx["api_out"][mode], "api"
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"skip {name} {mode} {a.format} {shape_id}: {type(e).__name__}: {str(e)[:400]}", flush=True)
                emit(dict(base, config=name, kernel=mode, status="error", error=f"{type(e).__name__}: {str(e)[:1500]}"))
                torch.cuda.synchronize()
                continue
            acc = {}
            if not a.no_acc:
                r = ref(mode in ("bwd", "step"))
                for t, x in outs.items():
                    if r.get(t) is None or not torch.is_tensor(x):
                        continue
                    x = from_batched(x) if (a.format == "batched" and x.dim() == 4) else x
                    x = from_aligned(x) if (a.format == "aligned" and x.shape[0] == total_al) else x
                    acc[t] = B.accuracy(x, r[t])
            fl = FLOP_MULT[mode] * 4 * a.h * a.d * real_pairs
            row = dict(base, config=name, kernel=mode, status="ok", ms_median=round(ms, 6),
                       tflops=round(fl / ms * 1e-9, 1),
                       tflops_computed=round(FLOP_MULT[mode] * 4 * a.h * a.d * pad_pairs / ms * 1e-9, 1),
                       ns_per_real_token=round(ms * 1e6 / total, 3), timing=timing, acc_source=acc_src,
                       cos_o=acc.get("o", {}).get("cos"), cos_dq=acc.get("dq", {}).get("cos"),
                       cos_dk=acc.get("dk", {}).get("cos"), cos_dv=acc.get("dv", {}).get("cos"), accuracy=acc,
                       plan=ctx.get("plan_info"), replay=(ctx.get("replay") or {}).get(mode),
                       contaminated=False, foreign_procs=[], validated=True, source="measured")
            emit(row)
            print(f"{name:24s} {mode:5s} {a.format:7s} {shape_id:36s} {ms:9.4f} ms {row['tflops']:7.0f} TF {timing:16s} "
                  + " ".join(f"{t} {x['cos']:.5f}" for t, x in acc.items()), flush=True)
        del fns, ctx
        torch.cuda.empty_cache()



# -- forward-only and unsupported baselines: cuDNN MXFP8 (THD ragged), FA4 FP8, FlashInfer NVFP4 KV, HW-FP4 FA4 ----------


def _aligned(ctx):
    lens = ctx["lens"]
    al = [-(-n // 128) * 128 for n in lens]
    cu_al = [0]
    for a in al:
        cu_al.append(cu_al[-1] + a)
    nb = len(lens)
    cu = ctx["cu_cpu"].long()
    tot = ctx["total"]
    doc = torch.repeat_interleave(torch.arange(nb), torch.tensor(lens))
    pos = torch.arange(tot) - cu[:-1][doc]
    aidx = (torch.tensor(cu_al[:-1])[doc] + pos).cuda()
    return cu_al[-1], aidx


# ------------------------------------------------------------------------------------------------- cuDNN MXFP8
def setup_cudnn_mxfp8(ctx):
    if ctx["fmt"] == "batched":
        r = B.setup_cudnn(dict(q=ctx["q"], k=ctx["k"], v=ctx["v"], g=ctx["g"], causal=ctx["causal"]), "mxfp8")
        return dict(fwd=r[0], bwd=r[1], step=r[2], infer=r[3])
    cudnn = B.cudnn
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    sc = 1 / math.sqrt(q.shape[-1])
    T, h, d = q.shape
    hk = k.shape[1]
    nb, maxL = ctx["b"], ctx["maxL"]
    L = -(-maxL // 128) * 128
    F32, BF, E4, E8, I32 = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3,
                            cudnn.data_type.FP8_E8M0, cudnn.data_type.INT32)
    one = [1, 1, 1, 1]
    dim, st = [nb, h, L, d], [L * h * d, d, h * d, 1]
    kv = ([nb, hk, L, d], [L * hk * d, d, hk * d, 1])
    rows_ = lambda n: ([nb, n, L, d // 32], [n * L * d // 32, L * d // 32, d // 32, 1])  # noqa: E731
    cols_ = lambda n: ([nb, n, L // 32, d], [n * L // 32 * d, L // 32 * d, d, 1])  # noqa: E731
    sdim, sst = [nb, h, L, 1], [h * L, L, 1, 1]
    mask = dict(diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT)
    cu = ctx["cu"]
    ro_q = (cu.long() * h * d).to(torch.int32).view(nb + 1, 1, 1, 1).contiguous()
    ro_k = (cu.long() * hk * d).to(torch.int32).view(nb + 1, 1, 1, 1).contiguous()
    lens_t = torch.tensor(ctx["lens"], dtype=torch.int32, device="cuda").view(nb, 1, 1, 1)
    hd = cudnn.create_handle()
    gr = cudnn.pygraph(handle=hd, intermediate_data_type=F32, compute_data_type=F32)
    RQ, RK, SQ, SK = 90, 91, 92, 93
    R = dict(q=gr.tensor(name="roq", dim=[nb + 1, 1, 1, 1], stride=one, data_type=I32, uid=RQ),
             k=gr.tensor(name="rok", dim=[nb + 1, 1, 1, 1], stride=one, data_type=I32, uid=RK),
             sq=gr.tensor(name="slq", dim=[nb, 1, 1, 1], stride=one, data_type=I32, uid=SQ),
             sk=gr.tensor(name="slk", dim=[nb, 1, 1, 1], stride=one, data_type=I32, uid=SK))

    def ten(name, uid, dt, side="q", shape=None, reorder=False, rag=True):
        shape = shape or ((dim, st) if side == "q" else kv)
        t = gr.tensor(name=name, dim=shape[0], stride=shape[1], data_type=dt, uid=uid)
        if reorder:
            t.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        if rag:
            t.set_ragged_offset(R[side])
        return t

    ds = [ten("dq", 4, E8, "q", rows_(h), True, False), ten("dk", 5, E8, "k", rows_(hk), True, False),
          ten("dv", 6, E8, "k", cols_(hk), True, False)]
    r = gr.sdpa_mxfp8(name="f", q=ten("q", 1, E4), k=ten("k", 2, E4, "k"), v=ten("v", 3, E4, "k"), descale_q=ds[0],
                      descale_k=ds[1], descale_v=ds[2], generate_stats=True, attn_scale=sc,
                      diagonal_band_right_bound=0 if causal else None, seq_len_q=R["sq"], seq_len_kv=R["sk"],
                      use_padding_mask=True, **mask)
    for t, uid, dt, shp, rg in ((r[0], 10, BF, (dim, st), "q"), (r[1], 11, F32, (sdim, sst), None),
                                (r[2], 12, F32, (one, one), None)):
        t.set_uid(uid).set_output(True).set_dim(shp[0]).set_stride(shp[1]).set_data_type(dt)
        if rg:
            t.set_ragged_offset(R[rg])
    # operands: the library quantiser on the 128-aligned copy of the documents (batch 1): SF tiles come out per sequence,
    # tile-padded, in document order = the layout the THD graph documents; the codes are then packed to (T, h, d).
    tot_al, aidx = _aligned(ctx)

    def to_al(t):
        z = torch.zeros(1, tot_al, t.shape[1], t.shape[2], dtype=t.dtype, device="cuda")
        z[0].index_copy_(0, aidx, t)
        return z

    def quant_all():
        (qc, sfq), (kc, sfk), (vc, sfv) = B.mxfp8(to_al(q)), B.mxfp8(to_al(k)), B.mxfp8(to_al(v), along_seq=True)
        pk = lambda c: c[0].index_select(0, aidx).contiguous()  # noqa: E731
        return pk(qc), pk(kc), pk(vc), sfq, sfk, sfv
    qp, kp, vp, sfq, sfk, sfv = quant_all()
    o = torch.zeros(T, h, d, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(nb, h, L, 1, dtype=torch.float32, device="cuda")
    pf = {1: qp, 2: kp, 3: vp, 4: sfq, 5: sfk, 6: sfv, 10: o, 11: lse, 12: torch.zeros(1, 1, 1, 1, device="cuda"),
          RQ: ro_q, RK: ro_k, SQ: lens_t, SK: lens_t}
    gr.validate(), gr.build_operation_graph(), gr.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    gr.check_support(), gr.build_plans()
    ws = torch.empty(max(gr.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

    def call():
        cudnn.set_stream(handle=hd, stream=torch.cuda.current_stream().cuda_stream)
        gr.execute(pf, ws, handle=hd)

    def fwd():
        call()
        return dict(o=o)

    bw = {}

    def build_bwd():
        if bw:
            return
        g2 = cudnn.pygraph(handle=hd, intermediate_data_type=F32, compute_data_type=F32)
        R2 = dict(q=g2.tensor(name="roq", dim=[nb + 1, 1, 1, 1], stride=one, data_type=I32, uid=RQ),
                  k=g2.tensor(name="rok", dim=[nb + 1, 1, 1, 1], stride=one, data_type=I32, uid=RK),
                  sq=g2.tensor(name="slq", dim=[nb, 1, 1, 1], stride=one, data_type=I32, uid=SQ),
                  sk=g2.tensor(name="slk", dim=[nb, 1, 1, 1], stride=one, data_type=I32, uid=SK))

        def t2(name, uid, dt, side="q", shape=None, reorder=False, rag=True):
            shape = shape or ((dim, st) if side == "q" else kv)
            t = g2.tensor(name=name, dim=shape[0], stride=shape[1], data_type=dt, uid=uid)
            if reorder:
                t.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
            if rag:
                t.set_ragged_offset(R2[side])
            return t
        U = dict(q=1, q_T=2, k=3, k_T=4, v=5, o_f16=6, dO_f16=7, dO=8, dO_T=9, stats=10, descale_q=11, descale_q_T=12,
                 descale_k=13, descale_k_T=14, descale_v=15, descale_dO=16, descale_dO_T=17)
        T_ = {n: t2(n, U[n], E4, "k" if n[0] in "kv" else "q") for n in ("q", "q_T", "k", "k_T", "v", "dO", "dO_T")}
        T_.update(o_f16=t2("o_f16", 6, BF), dO_f16=t2("dO_f16", 7, BF), stats=t2("stats", 10, F32, "q", (sdim, sst), rag=False))
        for n, lay, sd in (("descale_q", rows_(h), "q"), ("descale_q_T", cols_(h), "q"), ("descale_k", rows_(hk), "k"),
                           ("descale_k_T", cols_(hk), "k"), ("descale_v", rows_(hk), "k"), ("descale_dO", rows_(h), "q"),
                           ("descale_dO_T", cols_(h), "q")):
            T_[n] = t2(n, U[n], E8, sd, lay, True, False)
        rr = g2.sdpa_mxfp8_backward(name="b", attn_scale=sc, right_bound=0 if causal else None, seq_len_q=R2["sq"],
                                    seq_len_kv=R2["sk"], use_padding_mask=True, **mask, **T_)
        for t, uid, shp, sd in zip(rr[:3], (20, 21, 22), ((dim, st), kv, kv), ("q", "k", "k")):
            t.set_uid(uid).set_output(True).set_dim(shp[0]).set_stride(shp[1]).set_data_type(BF)
            t.set_ragged_offset(R2[sd])
        for i_, t in enumerate(rr[3:]):
            t.set_uid(23 + i_).set_dim(one).set_stride(one).set_data_type(F32)
        g2.validate(), g2.build_operation_graph(), g2.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g2.check_support(), g2.build_plans()
        ws2 = torch.empty(max(g2.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        pack = {20: dq, 21: dk, 22: dv, RQ: ro_q, RK: ro_k, SQ: lens_t, SK: lens_t}
        bw.update(U=U, pack=pack, g2=g2, ws=ws2, grads=lambda: dict(dq=dq, dk=dk, dv=dv))

    def set_bwd_operands(x_q, x_k, x_v, x_g, outs=None):
        U, pack = bw["U"], bw["pack"]
        pk = lambda c: c[0].index_select(0, aidx).contiguous()  # noqa: E731
        for x, n in ((x_q, "q"), (x_k, "k"), (x_g, "dO")):
            (c, sf), (ct, sft) = B.mxfp8(to_al(x)), B.mxfp8(to_al(x), along_seq=True)
            pack.update({U[n]: pk(c), U["descale_" + n]: sf, U[n + "_T"]: pk(ct), U["descale_" + n + "_T"]: sft})
        c, sf = B.mxfp8(to_al(x_v))
        pack.update({U["v"]: pk(c), U["descale_v"]: sf})

    def call_bwd():
        cudnn.set_stream(handle=hd, stream=torch.cuda.current_stream().cuda_stream)
        bw["g2"].execute(bw["pack"], bw["ws"], handle=hd)

    def bwd():
        if not bw:  # bwd alone: operands quantised outside the timed region; O / stats are this forward kernel's own
            build_bwd()
            set_bwd_operands(q, k, v, g)
            fwd_c = call_fwd_once()
            bw["pack"].update({6: o, 7: g, 10: lse})
        call_bwd()
        return bw["grads"]()

    def call_fwd_once():
        call()

    def step():
        if not bw:
            build_bwd()
        (pf[1], pf[4]), (pf[2], pf[5]), (pf[3], pf[6]) = [(a_, b_) for a_, b_ in ((lambda r_: (pkq(r_[0]), r_[1]))(B.mxfp8(to_al(q))),
                                                                                  (lambda r_: (pkq(r_[0]), r_[1]))(B.mxfp8(to_al(k))),
                                                                                  (lambda r_: (pkq(r_[0]), r_[1]))(B.mxfp8(to_al(v), along_seq=True)))]
        set_bwd_operands(q, k, v, g)
        call()
        bw["pack"].update({6: o, 7: g, 10: lse})
        call_bwd()
        return dict(o=o, **bw["grads"]())
    pkq = lambda c: c[0].index_select(0, aidx).contiguous()  # noqa: E731
    return dict(fwd=fwd, bwd=bwd, step=step)


# ------------------------------------------------------------------------------------------------- FA4 FP8 (upstream)
def setup_fa4_fp8(ctx):
    fa4 = B.fa4
    q, k, v, causal = ctx["q"], ctx["k"], ctx["v"], ctx["causal"]
    hk = k.shape[-2]
    if ctx["fmt"] == "batched":
        nb, vk = q.shape[0], {}
    else:
        nb = ctx["b"]
        vk = dict(cu_seqlens_q=ctx["cu"], cu_seqlens_k=ctx["cu"], max_seqlen_q=ctx["maxL"], max_seqlen_k=ctx["maxL"])
    (q8, sq), (k8, sk), (v8, sv) = (B.e4m3_dev(t) for t in (q, k, v))
    dsc = [s.float().expand(nb, hk).contiguous() for s in (sq, sk, sv)]

    def fwd():
        return dict(o=fa4._flash_attn_fwd(q8, k8, v8, causal=causal, q_descale=dsc[0], k_descale=dsc[1], v_descale=dsc[2],
                                          return_lse=True, **vk)[0])

    def no_bwd():  # an ACTUAL training call: FA4's autograd entry on FP8 inputs that require grad
        qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q8, k8, v8))
        fa4._flash_attn_fwd(qg, kg, vg, causal=causal, q_descale=dsc[0], k_descale=dsc[1], v_descale=dsc[2], return_lse=True, **vk)
        raise RuntimeError("UNEXPECTED: FA4 FP8 training forward ran")
    return dict(fwd=fwd, bwd=no_bwd, step=no_bwd)


# ------------------------------------------------------------------------------------------------- FlashInfer NVFP4-KV
def setup_flashinfer_nvfp4kv(ctx, page=64):
    from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache
    q, k, v, causal = ctx["q"], ctx["k"], ctx["v"], ctx["causal"]
    if ctx["fmt"] != "varlen":
        return "varlen format only"
    T, h, d = q.shape
    hk = k.shape[1]
    lens = ctx["lens"]
    nb = len(lens)
    sm = 1 / math.sqrt(d)
    if d != 128:
        return "flashinfer nvfp4-kv: d != 128"
    npg = [-(-n // page) for n in lens]
    pbase = [0]
    for n in npg:
        pbase.append(pbase[-1] + n)
    tot_pages = pbase[-1]
    cu = ctx["cu_cpu"].long()
    doc = torch.repeat_interleave(torch.arange(nb), torch.tensor(lens))
    pos = torch.arange(T) - cu[:-1][doc]
    pidx = (torch.tensor(pbase[:-1])[doc] * page + pos).cuda()

    def paged(x):
        z = torch.zeros(tot_pages * page, hk, d, dtype=x.dtype, device="cuda")
        z.index_copy_(0, pidx, x)
        return z.view(tot_pages, page, hk, d).permute(0, 2, 1, 3).contiguous()  # (pages, hk, page, d) HND
    kc, vc = paged(k), paged(v)
    (kf, vf), (ksf, vsf), ks, vs = nvfp4_quantize_paged_kv_cache(kc, vc, "HND")
    q8, sq = B.e4m3_dev(q)
    sqf = float(sq)
    maxp = max(npg)
    bt = torch.zeros(nb, maxp, dtype=torch.int32)
    for i, n in enumerate(npg):
        bt[i, :n] = torch.arange(pbase[i], pbase[i] + n, dtype=torch.int32)
    bt = bt.cuda()
    lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cum = ctx["cu"]
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    o = torch.empty(T, h, d, device="cuda", dtype=torch.bfloat16)

    def fwd():
        trtllm_batch_context_with_kv_cache(query=q8, kv_cache=(kf, vf), workspace_buffer=ws, block_tables=bt,
                                           seq_lens=lens_t, max_q_len=ctx["maxL"], max_kv_len=ctx["maxL"],
                                           bmm1_scale=sqf * ks * sm, bmm2_scale=vs, batch_size=nb, cum_seq_lens_q=cum,
                                           cum_seq_lens_kv=cum, out=o, kv_layout="HND", kv_cache_sf=(ksf, vsf),
                                           causal=causal)
        return dict(o=o)

    def no_bwd():
        import flashinfer, flashinfer.prefill as fp_
        names = sorted(n for m in (flashinfer, fp_) for n in dir(m) if "bwd" in n.lower() or "backward" in n.lower() or "grad" in n.lower())
        getattr(fp_, "trtllm_batch_context_with_kv_cache_backward")  # the call a training step would need
        raise RuntimeError(f"UNEXPECTED backward entry points: {names}")
    return dict(fwd=fwd, bwd=no_bwd, step=no_bwd)


# ------------------------------------------------------------------------------------------------- Hu NVMX
def setup_hu(ctx):
    return ("hu-nvmx: no variable-length / ragged API. The Direct-P kernel is compiled per (B, S, H) with "
            "TK_HAO_DIRECT_SEQ_LEN / HEADS / BATCH as compile-time constants (hao_direct_config.inc), takes dense "
            "(B, S, H, D) tensors only (no cu_seqlens / seqused / padding mask anywhere in the TK sources), is non-causal "
            "and MHA only")


BACKENDS.update({"cudnn-mxfp8": setup_cudnn_mxfp8, "fa4-fp8": setup_fa4_fp8,
                 "flashinfer-nvfp4kv": setup_flashinfer_nvfp4kv, "hu-nvmx": setup_hu})


if __name__ == "__main__":
    main()

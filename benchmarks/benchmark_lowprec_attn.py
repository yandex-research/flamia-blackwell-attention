"""Benchmark low-precision attention (ours: NVFP4 / MXFP4 / FP8) against every baseline under one protocol.

    python benchmarks/benchmark_lowprec_attn.py [--backend ours,cudnn,...] [--mode fwd,bwd,step]
        [--shapes A1,A2 | b4-s4096-h16-d128[-hk4]] [--causal true|false|both] [--dist n01,logit3] [--seed 0]
        [--opt FIELD=VALUE ...] [--tune FIELD=VALUE ...] [--qk-hadamard] [--timer graph|do_bench] [--jsonl rows.jsonl]

Default cells (h16 d128, bf16): A1 b4 s4096 causal, A2 b4 s4096 dense, A3 b1 s32768 causal, A4 b1 s32768 dense.
Inputs come from CPU generators (seed S for q, k, v in that order, S + 7 for dO); logit3 multiplies q by 3.

Modes: fwd and bwd time the kernel alone (operands quantised outside the timed region); step times what a training
step calls, quantisation included (forward + autograd backward for ours, torch-sdpa and upstream FA4).
Timing (--timer graph, default): 10 warmups, then a CUDA graph of 20 calls; fwd / bwd report the median per-call time of
3 replays, step the median of a 2.5 s sustained soak of replays (eager if the closure cannot be captured).
--timer do_bench uses triton.testing.do_bench, as upstream benchmarks/benchmark_attn.py does. For stable numbers lock
the SM clock outside this script (e.g. `nvidia-smi -lgc 1700,1700`) and pass --clock-mhz to record it in the rows.
FLOPs: fwd 4 b h s^2 d, bwd 2.5x, step 3.5x, halved when causal (bench_utils.flops).
Accuracy: cosine, norm ratio |X| / |X_ref|, max and mean relative error (floor 1e-3 max|ref|) of O (fwd, step) and
dQ / dK / dV (bwd, step) against a chunked fp32 reference over the original bf16 inputs; sha256 of every output.

Backends are imported if present and skipped with a note if not. Each needs its own environment; run the script once
per environment and concatenate the --jsonl files:
  ours-fp4, ours-fp8, fa4-*   this repository (flash_attn.cute) with nvidia-cutlass-dsl >= 4.8; fa4-* is
                              flash_attn.cute.interface of whichever flash_attn is importable.
  cudnn-*                     nvidia-cudnn-frontend >= 1.29 (FP8 / MXFP8 SDPA need CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1).
  flashinfer-*                flashinfer-python >= 0.6 (forward only).
  meta-*                      Meta's lp_fa4 (facebookresearch/ads_model_kernel_library) in its own venv.
  hu-nvmx                     Hu et al. (github.com/MrHuff/fp4-fa4): build their Direct-P extension per (b, s, h) and put a
                              module `hu_entry` exposing build(b, s, h, d, causal, q, k, v, route) on PYTHONPATH.
                              Their extension is compiled per shape, so run one cell per process (--shapes A2).
  torch-sdpa                  any torch.
Our quantisers are always the library's (imported leaf modules, which need only torch and triton), in every
environment; put this repository on PYTHONPATH there too. ours-* and the baselines that take pre-quantised inputs
(cudnn-fp8 / cudnn-mxfp8, flashinfer-fp8, fa4-fp8) use only those; FlashInfer appears only as the flashinfer-*
attention baseline, never as a quantiser. meta-mxfp8 quantises inside its own pipeline (lp_fa4.quantize_mxfp8_varlen).

Our rows go through the public API (flash_attn.cute.lowprec): every lp-* mode is a fixed preset of flash_attn_fp4 /
flash_attn_fp8 arguments (LP_MODES); ours-fp4 / ours-fp8 / ours-fp4all are the entry with the options given here:
    --opt FIELD=VALUE   a field of FP4AttnOptions / FP8AttnOptions (e.g. --opt operand_scaling=per_tensor)
    --tune FIELD=VALUE  a field of KernelTuning (e.g. --tune token_block_forward=fold)
    --qk-hadamard       qk_hadamard=True (default: the API's None, on wherever it runs)
A field the entry's options class does not have refuses that backend's rows (a "refused" row naming it).
step / infer / fwdapi / bwdapi time the public call. fwd / bwd time the kernel calls the plan's route body
(lowprec._routes) makes, recorded at one public call and replayed on the operands the library's own quantisers
(quant.fp4 / quant.fp8) produced there, quantisation outside the timed region (_Recorder).
ours-mxfp4 is the one off-API row: the MXFP4 Q / K / V forward kernel (the MXFP4 formats are not part of the public
API), fwd only, on operands from quant.fp4 with 32-element scale groups.
"""
import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import time

import torch

_OURS_ERR = None
try:
    from flash_attn.cute import lowprec as lp
    from flash_attn.cute.lowprec.tuning import KernelTuning, TUNING_FIELDS  # undocumented, not API
    from flash_attn.cute.lowprec import specs as lp_specs
    from flash_attn.cute import interface_fwd_fp4 as ifwd4
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
except Exception as e:  # noqa: BLE001  (a missing or incompatible cutlass-dsl surfaces as other errors too)
    lp = None
    _OURS_ERR = f"{type(e).__name__}: {e}"
try:
    from flash_attn.cute import interface as fa4
except Exception:  # noqa: BLE001
    fa4 = None
try:
    import cudnn
except ImportError:
    cudnn = None
try:
    import flashinfer
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper, trtllm_ragged_attention_deepseek
except ImportError:
    flashinfer = None
try:
    import lp_fa4.cute as lpfa4
    from lp_fa4.cute.interface import _flash_attn_fwd as lpfa4_fwd, _flash_attn_bwd as lpfa4_bwd
    # lp_fa4's persistent JIT cache (FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1, inherited from our own tooling) cannot
    # hold its backward's entry, a tuple of compiled variants: JITPersistentCache.__setitem__ calls tuple.export_to_c
    # and every meta-bf16 / meta-mxfp8 bwd row errors. Their backward keeps an in-memory cache, whatever the caller sets.
    if type(lpfa4_bwd.compile_cache).__name__ == "JITPersistentCache":
        from lp_fa4.cute.cache_utils import JITCache as _LpJITCache
        lpfa4_bwd.compile_cache = _LpJITCache()
except ImportError:
    lpfa4 = None
try:
    import hu_entry
except ImportError:
    hu_entry = None
try:  # SageAttention3 (thu-ml/SageAttention sageattention3_blackwell), built in place and put on PYTHONPATH
    import sageattn3.api as sage3
except Exception:  # noqa: BLE001
    sage3 = None

CELLS = {"A1": (4, 4096, 16, 16, 128, True), "A2": (4, 4096, 16, 16, 128, False),
         "A3": (1, 32768, 16, 16, 128, True), "A4": (1, 32768, 16, 16, 128, False)}
# fwd / bwd: the attention kernel(s) alone, operands quantised outside the timed region. step: the training call.
# infer: the inference call (no input requires grad), quantisation included. fwdapi / bwdapi: the training call's forward
# alone (inputs require grad) and its autograd backward alone (retain_graph), each with its own quantisation and glue.
MODES = ("fwd", "bwd", "step", "infer", "fwdapi", "bwdapi")
FLOP_MULT = {"fwd": 1.0, "bwd": 2.5, "step": 3.5, "infer": 1.0, "fwdapi": 1.0, "bwdapi": 2.5}


@functools.lru_cache(maxsize=None)
def lib(name):
    """flash_attn.cute.<name> of this repository. If the package __init__ cannot run in this environment (another
    cutlass-dsl, e.g. cuDNN's), the quant package is imported under stub parent packages: it imports only torch,
    cutlass and its own files."""
    try:
        return importlib.import_module(f"flash_attn.cute.{name}")
    except Exception:  # noqa: BLE001
        import sys
        import types
        root = importlib.util.find_spec("flash_attn").submodule_search_locations[0]
        # leaf modules import siblings (`from flash_attn.cute.quant import ...`): stub the packages so those
        # resolve by path without running flash_attn.cute.__init__ (which needs this repository's cutlass-dsl)
        for pkg, path in (("flash_attn", root), ("flash_attn.cute", f"{root}/cute"),
                          ("flash_attn.cute.quant", f"{root}/cute/quant")):
            if not getattr(sys.modules.get(pkg), "_lowprec_bench_stub", False):
                stub = types.ModuleType(pkg)
                stub.__path__ = [path]
                stub._lowprec_bench_stub = True
                sys.modules[pkg] = stub
        for k in [k for k in sys.modules if k.startswith("flash_attn.cute.") and sys.modules[k] is None]:
            del sys.modules[k]
        return importlib.import_module(f"flash_attn.cute.{name}")


def e4m3_dev(x, scale=None):
    """Per-tensor E4M3 with the scale on the GPU (quant.fp8.quantize_fp8_per_tensor): (codes, float64 device scale)."""
    r = lib("quant.fp8").quantize_fp8_per_tensor(x, scales=None if scale is None else scale.reshape(1).double())
    return r["codes"], r["scales"][0]


def mxfp8(x, along_seq=False):
    """The library's MXFP8 quantiser (quant.fp8.quantize_fp8_blocks) -> (codes (b, s, h, d), SF bytes in cuDNN's
    F8_128x4 order): 1 x 32 blocks along d, or (along_seq) 32 x 1 blocks along the sequence, one launch either way."""
    q = lib("quant.fp8").quantize_fp8_blocks
    if not along_seq:
        r = q(x, block_rows=1, block_columns=32, output_exponent_table=False, scale_factor_layout="cudnn")
        return r["codes"], r["sf_tok"]
    r = q(x, block_rows=32, block_columns=1, output_exponent_table=False, scale_factor_layout="cudnn")
    return r["codes"], r["sf_dim"]


def dequantize_fp8_blocks(codes, sf_tok_raw):
    """The MXFP8 codes of quant.fp8.quantize_fp8_blocks (1 x 32 blocks along d) times their UE8M0 scales:
    codes (b, s, h, d) x the (tokens, d / 32) atoms (b, h, s/128, d/128, 32, 4, 4) -> fp32 (b, s, h, d)."""
    b, s, h, d = codes.shape
    e = sf_tok_raw.to(torch.int32).reshape(b, h, s // 128, d // 128, 32, 4, 4)
    e = e.permute(0, 1, 2, 5, 4, 3, 6).reshape(b, h, s, d // 32)          # (row = 32 j + i, group = 4 kt + g)
    scale = torch.where(e == 0, torch.zeros_like(e, dtype=torch.float32), torch.exp2((e - 127).float()))
    x = codes.float().permute(0, 2, 1, 3).reshape(b, h, s, d // 32, 32) * scale[..., None]
    return x.reshape(b, h, s, d).permute(0, 2, 1, 3)


def mxfp4_kmajor_v(v):
    """K-major MXFP4 V for the forward's PV MMA (quant.fp4, groups of 32 along the sequence, natural key order):
    (float4_e2m1fn_x2 (b, h, d, s / 2), SF (32, 4, d / 128, 4, s / 128, h, b))."""
    rv = quantize_fp4_rows(v, scale_group_size=32, make_row_copy=False, make_transposed_copy=True,
                           transposed_copy_in_kernel_key_order=False, transposed_copy_layout="batch_head_dim_seq")
    return rv["transposed_codes"].view(torch.float4_e2m1fn_x2), rv["transposed_scale_factors"]


# -- our options: --opt FIELD=VALUE (FP4AttnOptions / FP8AttnOptions) and --tune FIELD=VALUE (KernelTuning) ----------
def _parse_value(text, default):
    """A command-line value for an options field whose default is `default` (bools: 1/0, true/false, on/off)."""
    if text.lower() == "none":
        return None
    if isinstance(default, bool):
        if text.lower() in ("1", "true", "on", "yes"):
            return True
        if text.lower() in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"expected a bool, got {text!r}")
    return text


def _parse_tune(text, kind):
    """A command-line value for a KernelTuning field of value kind `kind` (options.TUNING_FIELDS)."""
    if text.lower() == "none":
        return None
    if kind == "bool":
        return _parse_value(text, False)
    tag = kind[0]
    if tag in ("int", "ints"):
        return int(text)
    if tag == "float":
        return float(text)
    return text  # enum, re


def options_from_cli(opts, tune):
    """{"fp4": FP4AttnOptions kwargs, "fp8": FP8AttnOptions kwargs} and the KernelTuning (or None) of --opt / --tune.
    A field neither options class has, or an unknown KernelTuning field, is an argument error; a field only one class
    has is kept for that entry (the other entry refuses: options_for)."""
    classes = {"fp4": lp.FP4AttnOptions, "fp8": lp.FP8AttnOptions}
    kw = {e: {} for e in classes}
    for name, text in opts.items():
        owners = [e for e, c in classes.items() if name in c.__dataclass_fields__ and name != "tuning"]
        if not owners:
            raise SystemExit(f"--opt {name}: not a field of FP4AttnOptions or FP8AttnOptions")
        for e in owners:
            kw[e][name] = _parse_value(text, classes[e].__dataclass_fields__[name].default)
    tk = {}
    for name, text in tune.items():
        if name not in TUNING_FIELDS:
            raise SystemExit(f"--tune {name}: not a KernelTuning field")
        tk[name] = _parse_tune(text, TUNING_FIELDS[name])
    return kw, (KernelTuning(**tk) if tk else None)


def options_for(entry, ofields, cli):
    """The options object of one of our backends: the mode's fixed fields, then --opt / --tune (ours-* only).
    Raises ValueError (a refusal) for a field the entry's options class does not have."""
    cls = lp.FP4AttnOptions if entry == "fp4" else lp.FP8AttnOptions
    kw = dict(ofields)
    if cli is not None:
        okw, tuning = cli
        other = "fp8" if entry == "fp4" else "fp4"
        foreign = sorted(set(okw[other]) - set(okw[entry]))
        if foreign:
            raise ValueError(f"--opt {', '.join(foreign)}: not a field of {cls.__name__}")
        kw.update(okw[entry])
        if tuning is not None:
            kw["tuning"] = tuning
    bad = sorted(n for n in kw if n not in cls.__dataclass_fields__)
    if bad:  # e.g. qk_format / pv_format: the MXFP4 formats are not part of the API
        raise ValueError(f"{', '.join(bad)}: not a field of {cls.__name__} (removed from the API)")
    return cls(**kw)


def _train_step(entry, ctx, **kw):
    q, k, v, g = ctx["q"], ctx["k"], ctx["v"], ctx["g"]
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    def step():
        o = entry(qg, kg, vg, causal=ctx["causal"], **kw)
        o = o[0] if isinstance(o, tuple) else o
        return dict(o=o, **dict(zip(("dq", "dk", "dv"), torch.autograd.grad(o, (qg, kg, vg), g))))
    return step


# -- backends: setup(ctx) -> (fwd_fn, bwd_fn, step_fn), each a zero-argument closure returning its outputs, or None --
def setup_ours_mxfp4(ctx):
    """The MXFP4 Q / K / V forward kernel (E2M1 with a UE8M0 scale per 32; K-major V, groups of 32 keys), kernel only.
    Off the public API (which offers no MXFP4 format): the driver is called directly with the
    specs' defaults (the measured rule of every field) on operands from the library's quant.fp4. Its 4-bit P.V has no
    causal-safe V here, so causal cells error as they always did."""
    q, k, v, causal = ctx["q"], ctx["k"], ctx["v"], ctx["causal"]
    e2 = torch.float4_e2m1fn_x2
    # V K-major: the transposed copy only, natural key order, (b, h, d, s / 2)
    rv = quantize_fp4_rows(v, scale_group_size=32, make_row_copy=False, make_transposed_copy=True,
                           transposed_copy_in_kernel_key_order=False, transposed_copy_layout="batch_head_dim_seq")
    va, sfv = rv["transposed_codes"].view(e2), rv["transposed_scale_factors"]
    rq, rk = quantize_fp4_rows(q, scale_group_size=32), quantize_fp4_rows(k, scale_group_size=32)
    q4, sfq, k4, sfk = rq["codes"], rq["scale_factors"], rk["codes"], rk["scale_factors"]
    sk = dict(spec=lp_specs.FwdKernelSpec(), host_spec=lp_specs.HostSpec(), quant_spec=lp_specs.QuantSpec())

    def fwd():
        o_, l_ = ifwd4._flash_attn_fwd(q4.view(e2), k4.view(e2), va, causal=causal, mSFQ=sfq, mSFK=sfk, mSFV=sfv,
                                       return_lse=True, qk_sf_vec_size=32, pv_sf_vec_size=32, **sk)[:2]
        return dict(o=o_, lse=l_)
    return fwd, None, None


def setup_fa4(ctx, fp8=False):
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    w = ctx.get("window")
    wk = dict(window_size_left=w[0], window_size_right=w[1]) if w else {}
    if not fp8:
        out, lse = fa4._flash_attn_fwd(q, k, v, causal=causal, return_lse=True, **wk)[:2]
        step = ((lambda q_, k_, v_, causal: fa4.flash_attn_func(q_, k_, v_, causal=causal, window_size=w)) if w
                else fa4.flash_attn_func)
        return (lambda: dict(zip(("o", "lse"), fa4._flash_attn_fwd(q, k, v, causal=causal, return_lse=True, **wk)[:2])),
                lambda: dict(zip(("dq", "dk", "dv"), fa4._flash_attn_bwd(q, k, v, out, g, lse, causal=causal, **wk)[:3])),
                _train_step(step, ctx))
    if w:
        return None, None, None
    (q8, sq), (k8, sk), (v8, sv) = (e4m3_dev(t) for t in (q, k, v))
    dsc = [s.float().expand(q.shape[0], k.shape[2]).contiguous() for s in (sq, sk, sv)]
    return (lambda: dict(zip(("o", "lse"), fa4._flash_attn_fwd(q8, k8, v8, causal=causal, q_descale=dsc[0], k_descale=dsc[1],
                                                               v_descale=dsc[2], return_lse=True)[:2])), None, None)


def setup_torch_sdpa(ctx, backend=None):
    """torch SDPA on bf16; backend None = torch's own choice, "cudnn" / "flash" pins the backend (sdpa_kernel)."""
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    if ctx.get("window"):
        return None, None, None
    gqa = dict(enable_gqa=True) if k.shape[2] != q.shape[2] else {}
    from torch.nn.attention import SDPBackend, sdpa_kernel
    pin = {None: None, "cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}[backend]

    def sdpa(q_, k_, v_, causal):
        with (sdpa_kernel([pin]) if pin is not None else contextlib.nullcontext()):
            return torch.nn.functional.scaled_dot_product_attention(
                q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2), is_causal=causal, **gqa).transpose(1, 2)
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o_keep = sdpa(qg, kg, vg, causal)
    return (lambda: dict(o=sdpa(q, k, v, causal)),
            lambda: dict(zip(("dq", "dk", "dv"), torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True))),
            _train_step(sdpa, ctx))


def setup_cudnn(ctx, fmt, fresh=False):
    """fresh (FP8 only): step / infer compute the per-tensor input scales of Q, K, V, dO on the GPU inside the timed
    region (the library's e4m3_scales, as our steps do) instead of reusing the scales calibrated at setup; the output
    scales (O, dQ, dK, dV, dP) stay calibrated, as cuDNN's FP8 SDPA needs them before the call."""
    q, k, v, g, causal, sc = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"], 1 / math.sqrt(ctx["q"].shape[-1])
    b, s, h, d = q.shape
    hk = k.shape[2]  # GQA: K / V / dK / dV carry hk heads
    if ctx.get("window"):
        return None, None, None
    F32, BF, E4, E8 = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3,
                       cudnn.data_type.FP8_E8M0)
    dim, st, sdim, one = [b, h, s, d], [s * h * d, d, h * d, 1], [b, h, s, 1], [1, 1, 1, 1]
    kv = ([b, hk, s, d], [s * hk * d, d, hk * d, 1])
    rows_ = lambda n: ([b, n, s, d // 32], [n * s * d // 32, s * d // 32, d // 32, 1])  # noqa: E731
    cols_ = lambda n: ([b, n, s // 32, d], [n * s // 32 * d, s // 32 * d, d, 1])  # noqa: E731
    rows, cols = rows_(h), cols_(h)
    mask = dict(diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT)
    hd = cudnn.create_handle()
    new = lambda: cudnn.pygraph(handle=hd, intermediate_data_type=F32, compute_data_type=F32)  # noqa: E731

    def ten(gr, name, uid, dt, shape=(dim, st), reorder=False):
        t = gr.tensor(name=name, dim=shape[0], stride=shape[1], data_type=dt, uid=uid)
        if reorder:
            t.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return t

    def out(t, uid, dt, shape=(dim, st)):
        t.set_uid(uid).set_output(True).set_dim(shape[0]).set_stride(shape[1]).set_data_type(dt)

    def run(gr, pack, built=False):
        if not built:
            gr.validate(), gr.build_operation_graph(), gr.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            gr.check_support(), gr.build_plans()
        ws = torch.empty(max(gr.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

        def call():
            cudnn.set_stream(handle=hd, stream=torch.cuda.current_stream().cuda_stream)
            gr.execute(pack, ws, handle=hd)
        return call
    o = torch.empty(b, s, h, d, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(b, h, s, 1, dtype=torch.float32, device="cuda")
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    grads = lambda: dict(dq=dq, dk=dk, dv=dv)  # noqa: E731
    if fmt == "bf16":
        gr = new()
        o_, l_ = gr.sdpa(name="f", q=ten(gr, "q", 1, BF), k=ten(gr, "k", 2, BF, kv), v=ten(gr, "v", 3, BF, kv), generate_stats=True,
                         attn_scale=sc, diagonal_band_right_bound=0 if causal else None, **mask)
        out(o_, 10, BF), out(l_, 11, F32, (sdim, [h * s, s, 1, 1]))
        fwd = run(gr, {1: q, 2: k, 3: v, 10: o, 11: lse})
        fwd()
        gr = new()
        r = gr.sdpa_backward(name="b", q=ten(gr, "q", 1, BF), k=ten(gr, "k", 2, BF, kv), v=ten(gr, "v", 3, BF, kv),
                             o=ten(gr, "o", 10, BF), dO=ten(gr, "dO", 12, BF), stats=ten(gr, "l", 11, F32, (sdim, [h * s, s, 1, 1])),
                             attn_scale=sc, diagonal_band_right_bound=0 if causal else None, **mask)
        [out(t, u, BF, sh) for t, u, sh in zip(r, (20, 21, 22), ((dim, st), kv, kv))]
        bwd = run(gr, {1: q, 2: k, 3: v, 10: o, 12: g, 11: lse, 20: dq, 21: dk, 22: dv})
        return ((lambda: (fwd(), dict(o=o, lse=lse[..., 0]))[1]), (lambda: (bwd(), grads())[1]),
                (lambda: (fwd(), bwd(), dict(o=o, **grads()))[2]))
    if fmt == "fp8":  # per-tensor scales on the GPU from the library quantiser; O in BF16 for fwd, E4M3 for bwd/step
        (q8, sq), (k8, sk), (v8, sv), (do8, sdo) = (e4m3_dev(t) for t in (q, k, v, g))
        scal = lambda x: (x if torch.is_tensor(x) else torch.tensor(float(x), device="cuda")).float().reshape(1, 1, 1, 1)  # noqa: E731
        ref = reference(*(t.float() * s_.float() for t, s_ in ((q8, sq), (k8, sk), (v8, sv), (do8, sdo))), causal, True)
        so, sdq, sdk, sdv = (448.0 / float(t.abs().amax()) for t in (ref[0], ref[2], ref[3], ref[4]))  # arena calibration
        fsc = ["descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"]

        def fwd_graph(o_dt, o_t, so_, lse):
            gr = new()
            T = {n: ten(gr, n, 40 + i, F32, (one, one)) for i, n in enumerate(fsc)}
            r = gr.sdpa_fp8(name="f", q=ten(gr, "q", 1, E4), k=ten(gr, "k", 2, E4, kv), v=ten(gr, "v", 3, E4, kv), generate_stats=True,
                            attn_scale=sc, right_bound=0 if causal else None, **mask, **T)
            out(r[0], 10, o_dt), out(r[1], 11, F32, (sdim, [h * s, s, 1, 1])), out(r[2], 12, F32, (one, one)), out(r[3], 13, F32, (one, one))
            vals = [sq, sk, sv, 1 / 448, 448.0, so_]
            pack = {1: q8, 2: k8, 3: v8, 10: o_t, 11: lse, 12: scal(0), 13: scal(0), **{40 + i: scal(x) for i, x in enumerate(vals)}}
            return run(gr, pack), pack
        fwd, _ = fwd_graph(BF, o, 1.0, lse)
        lse8 = torch.empty_like(lse)
        o8 = torch.empty(b, s, h, d, dtype=torch.float8_e4m3fn, device="cuda")
        fwd8, pf = fwd_graph(E4, o8, so, lse8)
        fwd8()
        bsc = ["descale_q", "descale_k", "descale_v", "descale_o", "descale_dO", "descale_s", "descale_dP", "scale_s",
               "scale_dQ", "scale_dK", "scale_dV", "scale_dP"]
        gr = new()
        T = {n: ten(gr, n, 40 + i, F32, (one, one)) for i, n in enumerate(bsc)}
        r = gr.sdpa_fp8_backward(name="b", q=ten(gr, "q", 1, E4), k=ten(gr, "k", 2, E4, kv), v=ten(gr, "v", 3, E4, kv),
                                 o=ten(gr, "o", 10, E4), dO=ten(gr, "dO", 14, E4), stats=ten(gr, "l", 11, F32, (sdim, [h * s, s, 1, 1])),
                                 attn_scale=sc, right_bound=0 if causal else None, **mask, **T)
        g8 = [torch.empty(b, s, n, d, dtype=torch.float8_e4m3fn, device="cuda") for n in (h, hk, hk)]
        [out(t, u, E4, sh) for t, u, sh in zip(r[:3], (20, 21, 22), ((dim, st), kv, kv))]
        [out(t, 60 + i, F32, (one, one)) for i, t in enumerate(r[3:])]
        amax = [scal(0) for _ in range(4)]
        vals = [sq, sk, sv, 1 / so, sdo, 1 / 448, 1.0, 448.0, sdq, sdk, sdv, 1.0]
        pb = {1: q8, 2: k8, 3: v8, 10: o8, 14: do8, 11: lse8, 20: g8[0], 21: g8[1], 22: g8[2],
              **{60 + i: a for i, a in enumerate(amax)}, **{40 + i: scal(x) for i, x in enumerate(vals)}}
        bwd = run(gr, pb)
        bwd()  # calibration run: scale_dP from its amax_dP (the arena's delayed-scaling setup)
        sdp = 448.0 / float(amax[3]) if float(amax[3]) > 0 else 1.0
        pb[40 + 6].fill_(1 / sdp), pb[40 + 11].fill_(sdp)
        grads8 = lambda: dict(dq=(g8[0], sdq), dk=(g8[1], sdk), dv=(g8[2], sdv))  # E4M3 codes, scale  # noqa: E731

        def step():  # quantisation inside, with the scales precomputed above (cuDNN ships no quantiser)
            if fresh:  # fresh per-tensor scales: amax + quantise inside, descales handed over on the GPU
                for uid, x, su in ((1, q, 40), (2, k, 41), (3, v, 42), (14, g, 44)):
                    c_, s_ = e4m3_dev(x)
                    pf[uid] = pb[uid] = c_
                    if uid != 14:
                        pf[su] = scal(s_)
                    pb[su] = scal(s_)
            else:
                for uid, x, s_ in ((1, q, sq), (2, k, sk), (3, v, sv), (14, g, sdo)):
                    pf[uid] = pb[uid] = e4m3_dev(x, s_)[0]
            fwd8(), bwd()
            return dict(o=(o8, so), **grads8())
        pi = dict(pf)

        def infer():  # the library quantiser (scales precomputed, as step; fresh: computed here) + the forward, BF16 O
            for uid, x, s_ in ((1, q, sq), (2, k, sk), (3, v, sv)):
                if fresh:
                    pi[uid], s2 = e4m3_dev(x)
                    pi[39 + uid] = scal(s2)
                else:
                    pi[uid] = e4m3_dev(x, s_)[0]
            fwdi()
            return dict(o=o)
        fwdi, pi = fwd_graph(BF, o, 1.0, lse)
        return (lambda: (fwd(), dict(o=o, lse=lse[..., 0]))[1]), (lambda: (bwd(), grads8())[1]), step, infer
    # mxfp8: the library's MXFP8 quantiser; its SF atoms are cuDNN's F8_128x4 (along the sequence: tile-row-major)
    (qp, sfq), (kp, sfk), (vp, sfv) = mxfp8(q), mxfp8(k), mxfp8(v, along_seq=True)
    gr = new()
    ds = [ten(gr, "dq", 4, E8, rows, True), ten(gr, "dk", 5, E8, rows_(hk), True), ten(gr, "dv", 6, E8, cols_(hk), True)]
    r = gr.sdpa_mxfp8(name="f", q=ten(gr, "q", 1, E4), k=ten(gr, "k", 2, E4, kv), v=ten(gr, "v", 3, E4, kv), descale_q=ds[0],
                      descale_k=ds[1], descale_v=ds[2], generate_stats=True, attn_scale=sc,
                      diagonal_band_right_bound=0 if causal else None, **mask)
    out(r[0], 10, BF), out(r[1], 11, F32, (sdim, [h * s, s, 1, 1])), out(r[2], 12, F32, (one, one))
    pf = {1: qp, 2: kp, 3: vp, 4: sfq, 5: sfk, 6: sfv, 10: o, 11: lse,
          12: torch.zeros(1, 1, 1, 1, device="cuda")}
    fwd = run(gr, pf)
    fwd()
    U = dict(q=1, q_T=2, k=3, k_T=4, v=5, o_f16=6, dO_f16=7, dO=8, dO_T=9, stats=10, descale_q=11, descale_q_T=12,
             descale_k=13, descale_k_T=14, descale_v=15, descale_dO=16, descale_dO_T=17)

    def bwd_operands(pack):
        for x, n in ((q, "q"), (k, "k"), (g, "dO")):
            (c, sf), (ct, sft) = mxfp8(x), mxfp8(x, along_seq=True)
            pack.update({U[n]: c, U["descale_" + n]: sf, U[n + "_T"]: ct, U["descale_" + n + "_T"]: sft})
        c, sf = mxfp8(v)
        pack.update({U["v"]: c, U["descale_v"]: sf})
    gr = new()
    T = {n: ten(gr, n, U[n], E4, kv if n[0] in "kv" else (dim, st)) for n in ("q", "q_T", "k", "k_T", "v", "dO", "dO_T")}
    T.update(o_f16=ten(gr, "o_f16", 6, BF), dO_f16=ten(gr, "dO_f16", 7, BF), stats=ten(gr, "stats", 10, F32, (sdim, [h * s, s, 1, 1])))
    for n, lay in (("descale_q", rows), ("descale_q_T", cols), ("descale_k", rows_(hk)), ("descale_k_T", cols_(hk)),
                   ("descale_v", rows_(hk)), ("descale_dO", rows), ("descale_dO_T", cols)):
        T[n] = ten(gr, n, U[n], E8, lay, True)
    r = gr.sdpa_mxfp8_backward(name="b", attn_scale=sc, right_bound=0 if causal else None, **mask, **T)
    [out(t, u, BF, sh) for t, u, sh in zip(r[:3], (20, 21, 22), ((dim, st), kv, kv))]
    [t.set_uid(23 + i).set_dim(one).set_stride(one).set_data_type(F32) for i, t in enumerate(r[3:])]
    pb = {6: o, 7: g, 10: lse, 20: dq, 21: dk, 22: dv}
    bwd_operands(pb)
    bwd = run(gr, pb)
    # bwd alone runs on O / stats of an fp32 reference over its own dequantised operands (as the arena did)
    def deq(x):
        r = lib("quant.fp8").quantize_fp8_blocks(x, block_rows=1, block_columns=32, output_exponent_table=False)
        return dequantize_fp8_blocks(r["codes"], r["sf_tok_raw"])
    ro, rl = reference(deq(q), deq(k), deq(v), None, causal, False)[:2]
    pr = {**pb, 6: ro.bfloat16(), 10: rl.unsqueeze(-1).contiguous()}
    bwd_ref = run(gr, pr, built=True)

    def step():  # the library's MXFP8 quantiser inside the timed region (cuDNN ships no quantiser)
        (pf[1], pf[4]), (pf[2], pf[5]), (pf[3], pf[6]) = mxfp8(q), mxfp8(k), mxfp8(v, along_seq=True)
        bwd_operands(pb)
        fwd(), bwd()
        return dict(o=o, **grads())

    def infer():  # the library's MXFP8 quantiser + the forward
        (pf[1], pf[4]), (pf[2], pf[5]), (pf[3], pf[6]) = mxfp8(q), mxfp8(k), mxfp8(v, along_seq=True)
        fwd()
        return dict(o=o)
    return (lambda: (fwd(), dict(o=o, lse=lse[..., 0]))[1]), (lambda: (bwd_ref(), grads())[1]), step, infer


def setup_flashinfer(ctx, prec, backend):
    q, k, v, causal, sc = ctx["q"], ctx["k"], ctx["v"], ctx["causal"], 1 / math.sqrt(ctx["q"].shape[-1])
    b, s, h, d = q.shape
    hk = k.shape[2]
    if (hk != h and backend != "cutlass") or ctx.get("window"):
        return None, None, None
    sq = sk = sv = 1.0
    src, dsc = (q, k, v), None
    if prec == "fp8":  # FlashInfer's API takes host scales: read once here, outside the timed region
        qkv8 = [e4m3_dev(t) for t in (q, k, v)]
        dsc = [s_ for _, s_ in qkv8]
        (q, sq), (k, sk), (v, sv) = ((c, float(s_)) for c, s_ in qkv8)
    qr, kr, vr = (t.reshape(b * s, t.shape[2], d) for t in (q, k, v))
    ind = torch.arange(0, (b + 1) * s, s, device="cuda", dtype=torch.int32)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    o = torch.empty(b * s, h, d, device="cuda", dtype=torch.bfloat16)
    if backend == "cutlass":
        w = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="cutlass")
        w.plan(ind, ind, h, hk, d, causal=causal, q_data_type=q.dtype, kv_data_type=k.dtype, o_data_type=torch.bfloat16)
        kw = dict(q_scale=sq, k_scale=sk, v_scale=sv) if prec == "fp8" else {}
        fn = lambda: w.run(qr, kr, vr, out=o, **kw)  # noqa: E731
    else:
        n, n_dev = torch.full((b,), s, dtype=torch.int32), torch.full((b,), s, dtype=torch.int32, device="cuda")
        fn = lambda: trtllm_ragged_attention_deepseek(  # noqa: E731
            query=qr, key=kr, value=vr, workspace_buffer=ws, seq_lens=n_dev, max_q_len=s, max_kv_len=s,
            bmm1_scale=sq * sk * sc, bmm2_scale=sv, o_sf_scale=1.0, batch_size=b, window_left=-1, cum_seq_lens_q=ind,
            cum_seq_lens_kv=ind, enable_pdl=False, is_causal=causal, return_lse=False, out=o, q_seq_lens_cpu=n,
            kv_seq_lens_cpu=n, backend={"cutedsl": "cute-dsl", "trtllmgen": "trtllm-gen"}[backend])
    infer = None
    if prec == "fp8":
        def infer():  # the library's E4M3 quantiser with the precomputed scales (as cuDNN's step) + the forward
            nonlocal qr, kr, vr
            qr, kr, vr = (e4m3_dev(x, s_)[0].view(b * s, x.shape[2], d) for x, s_ in zip(src, dsc))
            fn()
            return dict(o=o.view(b, s, h, d))
    return (lambda: (fn(), dict(o=o.view(b, s, h, d)))[1]), None, None, infer


def setup_flashinfer_nvfp4kv(ctx, page=64):
    """FlashInfer trtllm-gen context attention on an NVFP4 paged KV cache (FP8 E4M3 Q, NVFP4 K / V: E2M1 with an E4M3
    scale per 16 along d and a global FP32 scale; BF16 O): FlashInfer's 4-bit forward. Q: the library's per-tensor
    E4M3 quantiser; K / V: FlashInfer's nvfp4_quantize_paged_kv_cache (the swizzled paged layout is theirs, no library
    quantiser emits it). fwd: kernel on operands quantised outside; infer: Q / K / V quantisation + the kernel."""
    from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache
    q, k, v, causal, sm = ctx["q"], ctx["k"], ctx["v"], ctx["causal"], 1 / math.sqrt(ctx["q"].shape[-1])
    b, s, h, d = q.shape
    hk = k.shape[2]
    if ctx.get("window") or s % page or d != 128:
        return None, None, None
    npg = s // page
    paged = lambda x: x.view(b, npg, page, hk, d).permute(0, 1, 3, 2, 4).reshape(b * npg, hk, page, d).contiguous()  # noqa: E731
    kc, vc = paged(k), paged(v)
    (kf, vf), (ksf, vsf), ks, vs = nvfp4_quantize_paged_kv_cache(kc, vc, "HND")
    kg = torch.tensor(1.0 / ks, device="cuda", dtype=torch.float32)
    vg = torch.tensor(1.0 / vs, device="cuda", dtype=torch.float32)
    q8, sq = e4m3_dev(q)
    sqf = float(sq)
    bt = torch.arange(b * npg, dtype=torch.int32, device="cuda").view(b, npg)
    lens = torch.full((b,), s, dtype=torch.int32, device="cuda")
    cum = torch.arange(0, (b + 1) * s, s, device="cuda", dtype=torch.int32)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    o = torch.empty(b * s, h, d, device="cuda", dtype=torch.bfloat16)
    st = dict(q=q8.view(b * s, h, d), kv=(kf, vf), sf=(ksf, vsf))

    def run():
        trtllm_batch_context_with_kv_cache(query=st["q"], kv_cache=st["kv"], workspace_buffer=ws, block_tables=bt,
                                           seq_lens=lens, max_q_len=s, max_kv_len=s, bmm1_scale=sqf * ks * sm,
                                           bmm2_scale=vs, batch_size=b, cum_seq_lens_q=cum, cum_seq_lens_kv=cum,
                                           out=o, kv_layout="HND", kv_cache_sf=st["sf"], causal=causal)
        return dict(o=o.view(b, s, h, d))

    def infer():  # scales calibrated at setup (host scalars the API takes), quantisation inside
        st["q"] = e4m3_dev(q, sq)[0].view(b * s, h, d)
        st["kv"], st["sf"], _, _ = nvfp4_quantize_paged_kv_cache(kc, vc, "HND", k_global_sf=kg, v_global_sf=vg)
        return run()
    return run, None, None, infer


def setup_meta(ctx, mx):
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    b, s, h, d = q.shape
    if k.shape[2] != h:
        return None, None, None
    if not mx:
        out, lse = lpfa4_fwd(q, k, v, causal=causal, return_lse=True)[:2]
        return (lambda: dict(zip(("o", "lse"), lpfa4_fwd(q, k, v, causal=causal, return_lse=True)[:2])),
                lambda: dict(zip(("dq", "dk", "dv"), lpfa4_bwd(q, k, v, out, g, lse, causal=causal)[:3])),
                _train_step(lpfa4.flash_attn_func, ctx))
    if causal:  # the public MXFP8 functions have no causal mode
        return None, None, None
    # Meta's own MXFP8 quantiser (quantize_mxfp8_varlen) is part of this baseline's pipeline: left as is, not ours.
    meta = lpfa4.Mxfp8VarlenMeta.from_lengths([s] * b, device="cuda")
    qp, kp, vp, gp = (t.reshape(b * s, h, d) for t in (q, k, v, g))
    quant = lambda t, o: lpfa4.quantize_mxfp8_varlen(t, meta, orientations=o, scaling_mode="rceil")  # noqa: E731
    qf, kf, vf = quant(qp, "hdim"), quant(kp, "hdim"), quant(vp, "seq")
    lpfa4.mxfp8_flash_attn_varlen_forward(qf, kf, vf, meta, meta, return_lse=True)  # LSE-first: else +12 % at b4 s4096
    qb, kb, vb = (quant(t, "both") for t in (qp, kp, vp))
    ob, lb = lpfa4.mxfp8_flash_attn_varlen_forward(qb, kb, vb, meta, meta, return_lse=True)
    grad = lambda r: dict(dq=r[0].view(b, s, h, d), dk=r[1].view(b, s, h, d), dv=r[2].view(b, s, h, d))  # noqa: E731

    def step():
        q2, k2, v2 = (quant(t, "both") for t in (qp, kp, vp))
        o2, l2 = lpfa4.mxfp8_flash_attn_varlen_forward(q2, k2, v2, meta, meta, return_lse=True)
        return dict(o=o2.view(b, s, h, d), **grad(lpfa4.mxfp8_flash_attn_varlen_backward(q2, k2, v2, o2, gp, l2, meta, meta)))
    def infer():  # their MXFP8 quantiser (hdim Q / K, seq V) + the forward
        q2, k2, v2 = quant(qp, "hdim"), quant(kp, "hdim"), quant(vp, "seq")
        return dict(o=lpfa4.mxfp8_flash_attn_varlen_forward(q2, k2, v2, meta, meta, return_lse=False)[0].view(b, s, h, d))
    return (lambda: dict(o=lpfa4.mxfp8_flash_attn_varlen_forward(qf, kf, vf, meta, meta, return_lse=False)[0].view(b, s, h, d)),
            lambda: grad(lpfa4.mxfp8_flash_attn_varlen_backward(qb, kb, vb, ob, gp, lb, meta, meta)), step, infer)


def setup_hu(ctx):
    q, k, v = ctx["q"], ctx["k"], ctx["v"]
    if ctx["causal"] or k.shape[2] != q.shape[2]:
        return None, None, None
    sp = hu_entry.build(*q.shape[:3], q.shape[3], False, q, k, v, route="nvmx-fast-g148")  # grid capped at 148 SMs
    return (lambda: (sp["fn"](), dict(o=sp["collect"]()))[1]), None, None


# -- our public entry points, every mode: flash_attn_fp4 / flash_attn_fp8 (flash_attn.cute.lowprec) ---------------------
# name: (entry, public keyword arguments, options fields). Each call resolves its plan (lowprec.plan); a combination the
# plan refuses (NotImplementedError, or ValueError naming the rule) becomes a "refused" row, as does an options field
# the API no longer has (the MXFP4 formats).
LP_MODES = {
    "fp4": ("fp4", {}, {}),
    "fp4-pt": ("fp4", {}, dict(operand_scaling="per_tensor")),
    # (fp4-pt-ptds: the 4-bit per-tensor dS was removed, rule 33)
    "fp4-b32": ("fp4", {}, dict(operand_scaling="block_32x32")),
    "fp4-1cta": ("fp4", {}, dict(two_cta="off")),
    "fp4-2cta": ("fp4", {}, dict(two_cta="on")),
    "fp4-det": ("fp4", dict(deterministic=True), {}),
    "fp4-had": ("fp4", dict(qk_hadamard=True), {}),                     # = the default (qk_hadamard on), made explicit
    "fp4-mxfp4": ("fp4", {}, dict(qk_format="mxfp4")),                  # refused: MXFP4 Q.K not in the API
    "fp4-optA": ("fp4", dict(pv_precision="fp4"), {}),                  # option A: inference only
    "fp4-optA-mx": ("fp4", dict(pv_precision="fp4"), dict(pv_format="mxfp4")),  # refused: MXFP4 P.V not in the API
    "fp4-pv4-off": ("fp4", dict(pv_precision="fp4", causal_safety="off"), {}),  # 4-bit P.V, causal leak accepted
    "fp4-all": ("fp4", dict(pv_precision="fp4", backward="fp4"), {}),
    "fp4-all-had": ("fp4", dict(pv_precision="fp4", backward="fp4", qk_hadamard=True, causal_safety="key_local"), {}),
    "fp4-all-eden": ("fp4", dict(pv_precision="fp4", backward="fp4"), dict(all_fp4_stabiliser="eden_rht")),
    "fp8": ("fp8", {}, {}),
    "fp8-pt": ("fp8", {}, dict(operand_scaling="per_tensor")),
    # (fp8-pt-ptds: the 8-bit per-tensor dS was removed, rule 33)
    "fp8-b32": ("fp8", {}, dict(operand_scaling="block_32x32")),
    "fp8-1cta": ("fp8", {}, dict(two_cta="off")),
    "fp8-det": ("fp8", dict(deterministic=True), {}),
    "fp8-e5m2": ("fp8", {}, dict(grad_format="e5m2", operand_scaling="per_tensor")),  # E5M2: explicit per-tensor
    "fp8-had": ("fp8", dict(qk_hadamard=True), {}),                     # = the default (qk_hadamard on), made explicit
}
# ours-*: the entry and its public keyword arguments; the options come from --opt / --tune (options_from_cli)
OURS = {
    "fp4": ("fp4", {}),                                                 # NVFP4 Q.K, E4M3 P.V, E4M3 backward
    "fp8": ("fp8", {}),
    "fp4all": ("fp4", dict(pv_precision="fp4", backward="fp4")),        # the all-4-bit route
}


class _Recorder:
    """Records the attention driver calls one public call makes, as the route bodies (lowprec._routes) make them: the
    forward driver (interface_fwd_fp4._flash_attn_fwd, bound by name in _routes and interface_bwd_fp4q; upstream
    interface._flash_attn_fwd), the E4M3-operand backward (interface_bwd_fp4._flash_attn_bwd_fp4, bound in _routes) and
    the all-4-bit backward's kernel launch (interface_bwd_fp4q.fp4q_run). Every call carries its plan's specs as explicit
    arguments, so a replay runs exactly the kernels the plan runs, on the operands the plan's own quantisers produced
    (quantisation outside the timed region, as the published fwd / bwd rows).
    A tensor argument the call mutates in place (other than its outputs) is snapshotted and restored before each replay
    only if the replay would otherwise change the result (the dk8 backward's D fold); the restore is timed."""
    TARGETS = (("lowprec._routes", "_flash_attn_fwd"), ("interface_fwd_fp4", "_flash_attn_fwd"),
               ("interface_bwd_fp4q", "_flash_attn_fwd"), ("interface", "_flash_attn_fwd"),
               ("lowprec._routes", "_flash_attn_bwd_fp4"), ("interface_bwd_fp4", "_flash_attn_bwd_fp4"),
               ("interface_bwd_fp4q", "fp4q_run"))

    def __init__(self):
        self.phase, self.depth, self.calls, self.saved = None, 0, {"fwd": [], "bwd": []}, []
        # import every target module BEFORE wrapping: a module imported later would bind a wrapper by name
        mods = {mod: importlib.import_module(f"flash_attn.cute.{mod}") for mod, _ in self.TARGETS}
        for mod, name in self.TARGETS:
            m = mods[mod]
            if hasattr(m, name) and getattr(m, name).__name__ != "w":
                self.saved.append((m, name, getattr(m, name)))
                setattr(m, name, self._wrap(getattr(m, name), f"{mod}.{name}"))

    def restore(self):
        for m, name, f in self.saved:
            setattr(m, name, f)

    @staticmethod
    def _tensors(x, out=None):
        out = [] if out is None else out
        if torch.is_tensor(x):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                _Recorder._tensors(y, out)
        elif isinstance(x, dict):
            for y in x.values():
                _Recorder._tensors(y, out)
        return out

    def _wrap(self, fn, label):
        def w(*a, **kw):
            if self.phase is None or self.depth:
                return fn(*a, **kw)
            ins = [t for t in self._tensors((a, {k_: v_ for k_, v_ in kw.items() if k_ != "out"}))]
            snap = [t.clone() for t in ins]
            self.depth += 1
            try:
                r = fn(*a, **kw)
            finally:
                self.depth -= 1
            outs = {t.data_ptr() for t in self._tensors(r)}
            mut = [(t, s) for t, s in zip(ins, snap) if t.data_ptr() not in outs and not _same(t, s)]
            # the all-4-bit backward's SR counter (interface_bwd_fp4q.sr_counter, int64 (1,)) advances in the call's post
            # launch, so each replay draws a new dither, as each training step does. It is put back only for the
            # determinism checks (replay(pin=True)), never in a timed replay: the timed kernels are the training call's.
            pin = [(t, s) for t, s in mut if t.dtype == torch.int64 and t.numel() == 1]
            self.calls[self.phase].append(dict(fn=fn, a=a, kw=kw, mut=mut, pin=pin, label=label, restore=False))
            return r
        w.__dict__ = fn.__dict__  # shared: the drivers keep their compile caches as function attributes
        return w

    def replay(self, phase, pin=False):
        calls = self.calls[phase]

        def run():
            r = None
            for c in calls:
                for t, s in (c["mut"] if c["restore"] else (c["pin"] if pin else ())):
                    t.copy_(s)
                r = c["fn"](*c["a"], **c["kw"])
            return r
        return run

def _same(a, b):
    try:
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    except Exception:  # noqa: BLE001
        try:
            return torch.equal(a, b)
        except Exception:  # noqa: BLE001
            return True


def _lp_outputs(r, phase, q, k, v):
    """The tensors a replayed driver call returns that map onto O (fwd) or dQ / dK / dV (bwd)."""
    if not isinstance(r, (tuple, list)):
        return {}
    if phase == "fwd":
        return dict(o=r[0]) if torch.is_tensor(r[0]) and r[0].shape == q.shape else {}
    ok = len(r) >= 3 and all(torch.is_tensor(x) and x.shape == t.shape for x, t in zip(r[:3], (q, k, v)))
    return dict(zip(("dq", "dk", "dv"), r[:3])) if ok else {}


def _cos(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def setup_lp(ctx, mode):
    """One preset mode of our public API (LP_MODES)."""
    entry, kw, ofields = LP_MODES[mode]
    return setup_public(ctx, entry, kw, ofields, None)


def setup_ours(ctx, name):
    """ours-fp4 / ours-fp8 / ours-fp4all: the entry with the options of --opt / --tune (and --qk-hadamard)."""
    entry, kw = OURS[name]
    kw = dict(kw)
    if ctx.get("qk_hadamard"):
        kw["qk_hadamard"] = True
    return setup_public(ctx, entry, kw, {}, ctx["cli"])


def setup_public(ctx, entry, kw, ofields, cli):
    """One configuration of our public API: fwd / bwd = the recorded driver calls replayed; step / infer / fwdapi /
    bwdapi = the public calls. The resolved plan travels in ctx["plan_info"] for the rows."""
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    fn = lp.flash_attn_fp4 if entry == "fp4" else lp.flash_attn_fp8
    resolve = lp.resolve_fp4 if entry == "fp4" else lp.resolve_fp8
    info, refused = {}, {}
    try:
        opts = options_for(entry, ofields, cli)
    except (TypeError, ValueError) as e:  # an options field this entry does not have: both uses refused
        why = f"{type(e).__name__}(rule=options): {e}"
        ctx["plan_info"], ctx["refused"] = info, {"train": why, "infer": why}
        return None, None, None, None, None, None
    kw = dict(kw, causal=causal, options=opts)
    if ctx.get("window"):
        kw["window_size"] = ctx["window"]
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    for tag, args in (("train", (qg, kg, vg)), ("infer", (q, k, v))):
        try:
            p = resolve(*args, **kw)
            info[tag] = dict(route=p.route, fwd_kernel=p.fwd.kernel, operand_scaling=p.fwd.operand_scaling,
                             softmax_exp=p.fwd.softmax_exp,
                             causal_safety=p.fwd.causal_safety, pv_operand=p.fwd.pv_operand, qk_format=p.fwd.qk_format,
                             bwd_kernel=p.bwd.kernel, ds_scaling=p.bwd.ds_scaling, p_scaling=p.bwd.p_scaling,
                             grad_format=p.bwd.grad_format, deterministic=p.bwd.deterministic, pending=list(p.pending),
                             forward_only=getattr(p.fwd, "forward_only", False), head_dim_path=p.bwd.head_dim_path)
        except (NotImplementedError, ValueError) as e:
            refused[tag] = f"{type(e).__name__}(rule={getattr(e, 'rule', None)}, pending={getattr(e, 'pending', None)}): {e}"
    ctx["plan_info"], ctx["refused"] = info, refused
    call = lambda q_, k_, v_, causal=causal: fn(q_, k_, v_, **dict(kw, causal=causal))  # noqa: E731
    infer = (lambda: dict(o=call(q, k, v))) if "infer" in info else None
    if "train" not in info:
        return None, None, None, infer, None, None
    rec = _Recorder()
    try:
        rec.phase = "fwd"
        o_keep = call(qg, kg, vg)
        o_api = o_keep.detach().clone()
        rec.phase = "bwd"
        api_grads = [x.clone() for x in torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True)]
        rec.phase = None
    finally:
        rec.restore()
    ctx["replay"] = {}
    fns = {}
    for ph, api in (("fwd", dict(o=o_api)), ("bwd", dict(zip(("dq", "dk", "dv"), api_grads)))):
        if not rec.calls[ph]:
            continue
        run, check = rec.replay(ph), rec.replay(ph, pin=True)
        r1 = _lp_outputs(check(), ph, q, k, v)
        r1 = {t: x.clone() for t, x in r1.items()}
        r2 = _lp_outputs(check(), ph, q, k, v)
        drift = min((_cos(r2[t], r1[t]) for t in r1), default=1.0)
        off_api = min((_cos(r1[t], api[t]) for t in r1 if t in api), default=1.0)
        if drift < 0.999 or off_api < 0.999:  # a mutated input changes later calls: restore it before each replay
            for c in rec.calls[ph]:
                c["restore"] = bool(c["mut"])
        r3 = _lp_outputs(check(), ph, q, k, v)
        ctx["replay"][ph] = dict(calls=[c["label"] for c in rec.calls[ph]],
                                 restored=[list(t.shape) for c in rec.calls[ph] if c["restore"] for t, _ in c["mut"]],
                                 sr_counter=[list(t.shape) for c in rec.calls[ph] if not c["restore"] for t, _ in c["pin"]],
                                 vs_api_cos={t: round(_cos(r3[t], api[t]), 6) for t in r3 if t in api},
                                 repeat_cos=round(drift, 6))
        fns[ph] = (lambda run=run, ph=ph: _lp_outputs(run(), ph, q, k, v))

    def fwdapi():
        return dict(o=call(qg, kg, vg))

    def bwdapi():
        return dict(zip(("dq", "dk", "dv"), torch.autograd.grad(o_keep, (qg, kg, vg), g, retain_graph=True)))
    return fns.get("fwd"), fns.get("bwd"), _train_step(call, ctx), infer, fwdapi, bwdapi


def setup_sage3(ctx):
    """SageAttention3 (inference, forward only): NVFP4 Q / K / V (E2M1, E4M3 scale per 16) after their per-block Q mean
    and K mean smoothing, their own CUDA quantisers (the operand layouts are theirs; no library quantiser emits them).
    fwd: the attention kernel on operands quantised outside; infer: their whole public call."""
    q, k, v, causal = ctx["q"], ctx["k"], ctx["v"], ctx["causal"]
    if k.shape[2] != q.shape[2] or ctx.get("window"):
        return None, None, None
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    s = q.shape[1]
    qp, kp, vp, delta_s = sage3.preprocess_qkv(qt.clone(), kt.clone(), vt.clone(), True)  # it shifts K in place
    ql, kl, vl = sage3.scale_and_quant_fp4(qp), sage3.scale_and_quant_fp4_permute(kp), sage3.scale_and_quant_fp4_transpose(vp)

    def fwd():
        return dict(o=sage3.blockscaled_fp4_attn(ql, kl, vl, delta_s, s, causal, True, True)[0][:, :, :s].transpose(1, 2))

    k_own = kt.clone()  # their call subtracts K's mean in place: idempotent after the first call, O is invariant to it

    def infer():
        return dict(o=sage3.sageattn3_blackwell(qt, k_own, vt, is_causal=causal).transpose(1, 2))
    return fwd, None, None, infer


BACKENDS = [
    *[(f"ours-{m}", (lambda m: lambda c: setup_ours(c, m) if lp else None)(m)) for m in OURS],
    ("ours-mxfp4", lambda c: setup_ours_mxfp4(c) if lp else None),
    *[(f"lp-{m}", (lambda m: lambda c: setup_lp(c, m) if lp else None)(m)) for m in LP_MODES],
    ("fa4-bf16", lambda c: setup_fa4(c) if fa4 else None), ("fa4-fp8", lambda c: setup_fa4(c, fp8=True) if fa4 else None),
    *[(f"cudnn-{f}", (lambda f: lambda c: setup_cudnn(c, f) if cudnn else None)(f)) for f in ("bf16", "fp8", "mxfp8")],
    ("cudnn-fp8-fresh", lambda c: (lambda r: (None, None) + tuple(r[2:]))(setup_cudnn(c, "fp8", fresh=True)) if cudnn else None),
    *[(f"flashinfer-{p}-{bk}", (lambda p, bk: lambda c: setup_flashinfer(c, p, bk) if flashinfer else None)(p, bk))
      for p in ("bf16", "fp8") for bk in ("cutedsl", "trtllmgen", "cutlass")],
    ("flashinfer-nvfp4kv-trtllmgen", lambda c: setup_flashinfer_nvfp4kv(c) if flashinfer else None),
    ("meta-bf16", lambda c: setup_meta(c, False) if lpfa4 else None), ("meta-mxfp8", lambda c: setup_meta(c, True) if lpfa4 else None),
    ("hu-nvmx", lambda c: setup_hu(c) if hu_entry else None),
    ("sage3", lambda c: setup_sage3(c) if sage3 else None),
    ("torch-sdpa", setup_torch_sdpa),
    ("torch-sdpa-cudnn", lambda c: setup_torch_sdpa(c, "cudnn")), ("torch-sdpa-flash", lambda c: setup_torch_sdpa(c, "flash")),
]
OURS_CLI = {f"ours-{m}" for m in OURS}  # the backends --opt / --tune / --qk-hadamard configure


# -- inputs, reference, accuracy, timing --------------------------------------------------------------------------
def make_inputs(b, s, h, hk, d, dist, seed):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(b, s, n, d, generator=gen, dtype=torch.float32).to(torch.bfloat16) for n in (h, hk, hk))
    g = torch.randn(b, s, h, d, generator=torch.Generator().manual_seed(seed + 7), dtype=torch.float32).to(torch.bfloat16)
    if dist == "logit3":
        q = (q.float() * 3.0).to(torch.bfloat16)
    return [t.cuda() for t in (q, k, v, g)]


def reference(q, k, v, g, causal, bwd, chunk=1024, window=None):
    """Chunked fp32 attention: (o, lse, dq, dk, dv), (b, s, h, d), lse (b, h, s); GQA by repeating K / V.
    window (left, right): key j visible to query i iff i - left <= j <= i + right (None = unbounded), as upstream."""
    b, s, h, d = q.shape
    grp, sc = h // k.shape[2], 1.0 / math.sqrt(d)
    k, v = k.repeat_interleave(grp, 2), v.repeat_interleave(grp, 2)
    o, lse = torch.zeros_like(q), torch.zeros(b, h, s, device=q.device)
    dq, dk, dv = (torch.zeros_like(q) if bwd else None for _ in range(3))
    for bi in range(b):
        for hi in range(h):
            Q, K, V = q[bi, :, hi], k[bi, :, hi], v[bi, :, hi]
            for r0 in range(0, s, chunk):
                S = (Q[r0:r0 + chunk] @ K.T) * sc
                rr = torch.arange(r0, r0 + S.shape[0], device=q.device)[:, None]
                cc = torch.arange(s, device=q.device)[None, :]
                if causal:
                    S = S.masked_fill(cc > rr, float("-inf"))
                if window is not None:
                    if window[0] is not None:
                        S = S.masked_fill(cc < rr - window[0], float("-inf"))
                    if window[1] is not None:
                        S = S.masked_fill(cc > rr + window[1], float("-inf"))
                lse[bi, hi, r0:r0 + chunk] = torch.logsumexp(S, -1)
                P = torch.softmax(S, -1)
                o[bi, r0:r0 + chunk, hi] = O = P @ V
                if bwd:
                    dO = g[bi, r0:r0 + chunk, hi]
                    dS = P * (dO @ V.T - (dO * O).sum(-1, keepdim=True))
                    dq[bi, r0:r0 + chunk, hi] = (dS @ K) * sc
                    dk[bi, :, hi] += (dS.T @ Q[r0:r0 + chunk]) * sc
                    dv[bi, :, hi] += P.T @ dO
    if bwd and grp > 1:
        dk, dv = (t.view(b, s, -1, grp, d).sum(3) for t in (dk, dv))
    return o, lse, dq, dk, dv


def visible_pairs(s, causal, window):
    """Visible (query, key) pairs of one (batch, head): the FLOP count's attention area."""
    if window is None:
        return s * (s + 1) / 2 if causal else s * s
    i = torch.arange(s, dtype=torch.float64)
    lo = (i - window[0]).clamp_min(0) if window[0] is not None else torch.zeros_like(i)
    hi = i if causal else ((i + window[1]).clamp_max(s - 1) if window[1] is not None else torch.full_like(i, s - 1))
    return float((hi - lo + 1).clamp_min(0).sum())


def accuracy(x, ref):
    p, r = x.detach().float().flatten(), ref.float().flatten()
    rel = (p - r).abs() / r.abs().clamp_min(max(float(r.abs().max()) * 1e-3, 1e-30))
    return dict(cos=float(p @ r / (p.norm() * r.norm() + 1e-30)), norm_ratio=float(p.norm() / (r.norm() + 1e-30)),
                max_rel_err=float(rel.max()), mean_rel_err=float(rel.mean()))


def bench(fn, mode, timer, n=20, soak_s=2.5):
    """ms per call: CUDA-graph median of 3 replays of n calls (fwd / bwd / infer / fwdapi / bwdapi), sustained median of
    replays (step); eager calls where the closure cannot be captured. Returns (ms, how)."""
    if timer == "do_bench":
        from triton.testing import do_bench
        return do_bench(fn), "do_bench"
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            [fn() for _ in range(3)]
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            [fn() for _ in range(n)]
        run = graph.replay
    except Exception:  # noqa: BLE001  (not capturable: time eager calls)
        torch.cuda.synchronize()
        graph, run = None, lambda: [fn() for _ in range(n)]
    run()

    def timed():
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(), run(), e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n
    if mode != "step" and graph is not None:
        return sorted(timed() for _ in range(3))[1], "graph"
    t0, xs = time.time(), []
    while time.time() - t0 < soak_s:
        xs.append(timed())
    return sorted(xs)[len(xs) // 2], ("graph-sustained" if graph is not None else "eager-sustained")


_FI_QUANT = ("nvfp4_quantize", "mxfp4_quantize", "fp4_quantize", "mxfp8_quantize", "nvfp4_batched_quantize")


@contextlib.contextmanager
def flashinfer_quant_counter():
    """Counts FlashInfer quantiser calls (flashinfer.quantization.* / flashinfer.*) made inside the block: the library
    imports them inside functions, so patching the module attributes sees every call. Yields the {name: count} dict."""
    counts, saved = {}, []
    try:
        import flashinfer as _fi
        import flashinfer.quantization as _fiq
        mods = [_fi, _fiq]
    except Exception:  # noqa: BLE001
        mods = []
    for m in mods:
        for nm in _FI_QUANT:
            f = getattr(m, nm, None)
            if f is None or getattr(f, "_bench_cnt", False):
                continue

            def w(*a, _f=f, _nm=nm, **kw):
                counts[_nm] = counts.get(_nm, 0) + 1
                return _f(*a, **kw)
            w._bench_cnt = True
            saved.append((m, nm, f))
            setattr(m, nm, w)
    try:
        yield counts
    finally:
        for m, nm, f in saved:
            setattr(m, nm, f)


def _parse_cell(sh, causal_arg):
    """A1..A4, or b4-s4096-h16-d128[-hk4][-w1024] (w: a sliding window (1024, 0), i.e. (left, right))."""
    if sh in CELLS:
        return [(sh, *CELLS[sh], None)]
    p = {}
    for x in sh.split("-"):
        if x.startswith("hk"):
            p["hk"] = int(x[2:])
        else:
            p[x[0]] = int(x[1:])
    win = (p["w"], 0) if "w" in p else None
    return [(f"{sh}-{'causal' if c else 'dense'}", p["b"], p["s"], p["h"], p.get("hk", p["h"]), p["d"], c, win)
            for c in {"true": [True], "false": [False], "both": [True, False]}[causal_arg]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="all", help="comma-separated names or prefixes (ours, cudnn, cudnn-fp8, ...)")
    ap.add_argument("--exact", action="store_true", help="match --backend names exactly (no prefix matching)")
    ap.add_argument("--mode", default="fwd,bwd,step")
    ap.add_argument("--shapes", default="A1,A2,A3,A4", help="A1..A4 or b4-s4096-h16-d128[-hk4][-w1024], comma-separated")
    ap.add_argument("--causal", default="both", choices=["true", "false", "both"], help="for b-s-h-d shapes")
    ap.add_argument("--dist", default="n01", help="n01, logit3 (comma-separated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", action="append", default=[],
                    help="FIELD=VALUE: an FP4AttnOptions / FP8AttnOptions field for ours-fp4 / -fp8 / -fp4all (repeatable)")
    ap.add_argument("--tune", action="append", default=[],
                    help="FIELD=VALUE: a KernelTuning field for ours-fp4 / -fp8 / -fp4all (repeatable)")
    ap.add_argument("--timer", default="graph", choices=["graph", "do_bench"])
    ap.add_argument("--clock-mhz", type=int, default=None, help="record that the SM clock was locked at this value")
    ap.add_argument("--tag", action="append", default=[], help="KEY=VALUE recorded in every row (commit, round, ...)")
    ap.add_argument("--no-acc", action="store_true", help="skip the fp32 reference (timing-only rows)")
    ap.add_argument("--jsonl", default=None, help="append one row per measurement (arena schema)")
    ap.add_argument("--qk-hadamard", action="store_true",
                    help="ours-fp4 / -fp8 / -fp4all: qk_hadamard=True (default: the API's None, on wherever it runs)")
    a = ap.parse_args()
    opts = dict(o.split("=", 1) for o in a.opt)
    tune = dict(t.split("=", 1) for t in a.tune)
    cli = options_from_cli(opts, tune) if lp is not None and (opts or tune) else None
    ours_cfg = [*a.opt, *(f"tune.{t}" for t in a.tune), *(["qk_hadamard=True"] if a.qk_hadamard else [])]
    tags = dict(t.split("=", 1) for t in a.tag)
    want = [x.strip() for x in a.backend.split(",")]
    backends = [(n, f) for n, f in BACKENDS if "all" in want or n in want
                or (not a.exact and any(n.startswith(w + "-") for w in want))]
    cells = [c for sh in a.shapes.split(",") for c in _parse_cell(sh, a.causal)]
    modes = a.mode.split(",")
    results = []

    def emit(row):
        results.append(row)
        if a.jsonl:
            with open(a.jsonl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    for (sid, b, s, h, hk, d, causal, window) in cells:
        for dist in a.dist.split(","):
            q, k, v, g = make_inputs(b, s, h, hk, d, dist, a.seed)
            cid = sid + ("" if dist == "n01" else f"-{dist}") + ("" if a.seed == 0 else f"-seed{a.seed}")
            refs = {}
            base = dict(shape_id=cid, b=b, s_q=s, s_k=s, h=h, h_kv=hk, d=d, causal=causal, window=window, dist=dist,
                        seed=a.seed, bench="benchmark_lowprec_attn.py", **tags)
            for name, setup in backends:
                ctx = dict(q=q, k=k, v=v, g=g, causal=causal, cli=cli, qk_hadamard=a.qk_hadamard, window=window)
                try:
                    fns = setup(ctx)
                except Exception as e:  # noqa: BLE001
                    for f in ctx.get("cleanup", []):
                        f()
                    print(f"skip {name} {cid}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                    emit(dict(base, impl=name.replace("-", "_"), config=name, status="error",
                              error=f"{type(e).__name__}: {str(e)[:1500]}", validated=False))
                    continue
                for tag_, why in ctx.get("refused", {}).items():
                    print(f"refused {name} {tag_} {cid}: {why[:300]}", flush=True)
                    emit(dict(base, impl=name.replace("-", "_"), config=name, status="refused", refused_for=tag_,
                              error=why, validated=False))
                if fns is None:
                    print(f"skip {name}: not installed" + (f" ({_OURS_ERR})" if name.startswith(("ours", "lp-")) and not lp else ""))
                    continue
                for mode, fn in zip(MODES, fns):
                    if fn is None or mode not in modes:
                        continue
                    try:  # a backend may refuse a cell (e.g. ours-mxfp4's 4-bit-PV forward on causal)
                        with flashinfer_quant_counter() as fiq:
                            fn()
                            torch.cuda.synchronize()
                        fiq = dict(fiq)
                        ms, timing = bench(fn, mode, a.timer)
                        outs = fn()  # (codes, scale) pairs are dequantised here, outside the timed closure
                        outs = {t: x[0].float() / x[1] if isinstance(x, tuple) else x for t, x in outs.items()}
                        torch.cuda.synchronize()
                    except Exception as e:  # noqa: BLE001
                        print(f"skip {name} {mode} {cid}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                        emit(dict(base, impl=name.replace("-", "_") + "_" + mode, config=name, kernel=mode,
                                  status="error", error=f"{type(e).__name__}: {str(e)[:1500]}", validated=False))
                        torch.cuda.synchronize()
                        continue
                    want_bwd = mode in ("bwd", "step", "bwdapi")
                    acc = {}
                    if not a.no_acc:
                        if want_bwd not in refs:
                            refs[want_bwd] = dict(zip(("o", "lse", "dq", "dk", "dv"),
                                                      reference(q.float(), k.float(), v.float(), g.float(), causal,
                                                                want_bwd, window=window)))
                        acc = {t: accuracy(x, refs[want_bwd][t]) for t, x in outs.items() if refs[want_bwd].get(t) is not None}
                    prim = next((t for t in ("o", "dq", "dk", "dv") if t in acc), None)
                    fl = FLOP_MULT[mode] * 4 * b * h * d * visible_pairs(s, causal, window)
                    row = dict(base, impl=name.replace("-", "_") + "_" + mode, config=name + (f"[{','.join(ours_cfg)}]" if ours_cfg and name in OURS_CLI else ""),
                               kernel=mode, status="ok", ms_median=round(ms, 6), tflops=round(fl / ms * 1e-9, 1), timing=timing,
                               clock="locked" if a.clock_mhz else "free", clock_locked_mhz=a.clock_mhz, accuracy=acc,
                               acc_tensor=prim, cos_o=acc.get("o", {}).get("cos"),
                               cos_dq=acc.get("dq", {}).get("cos"), cos_dk=acc.get("dk", {}).get("cos"), cos_dv=acc.get("dv", {}).get("cos"),
                               **({k_: acc[prim][k_] for k_ in ("norm_ratio", "max_rel_err", "mean_rel_err")} if prim else {}),
                               cos_primary=acc[prim]["cos"] if prim else None,
                               dumps={t: hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                                      for t, x in outs.items()},
                               opts=dict(options=opts, tuning=tune, qk_hadamard=a.qk_hadamard) if name in OURS_CLI else {},
                               plan=ctx.get("plan_info"),
                               replay=ctx.get("replay", {}).get(mode), flashinfer_quant_calls=fiq,
                               contaminated=False, foreign_procs=[], validated=prim is not None or a.no_acc, source="measured")
                    emit(row)
                    print(f"{name:24s} {mode:6s} {cid:28s} {ms:9.4f} ms {row['tflops']:7.0f} TF {timing:16s} " + "  ".join(
                        f"{t} {x['cos']:.5f}/{x['norm_ratio']:.4f}" for t, x in acc.items() if t != "lse")
                        + (f"  FI-quant {fiq}" if fiq else ""), flush=True)
                for f in ctx.get("cleanup", []):
                    f()
                del fns, ctx
                torch.cuda.empty_cache()
    print(f"\n{len(results)} rows; columns: ms, TFLOP/s, timing, then per tensor cos / norm-ratio vs fp32.")


if __name__ == "__main__":
    main()

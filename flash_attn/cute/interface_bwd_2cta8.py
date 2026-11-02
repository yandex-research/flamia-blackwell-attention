# Host-side driver of the 2-CTA low-precision backward (flash_bwd_sm100_2cta8.py): per-tensor E4M3 operands, the BF16
# preprocess (D, LSE * log2 e, zeroed fp32 dQ accumulator), the kernel, and the BF16 dQ postprocess with the folded scale.
import math
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32
import cuda.bindings.driver as cuda

from flash_attn.cute.interface import torch2cute_dtype_map, _get_device_capability
from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_bwd_sm100_2cta8 import FlashAttentionBackwardSm100TwoCta8
from flash_attn.cute.interface_bwd_fp4 import (_pmax_bound, bwd_dq_f16,
                                               e4m3_p_shift_log2, P8_SHIFT_2CTA, row_norms_f32)
from flash_attn.cute.tile_scheduler import _L2_SIZE_MB
from flash_attn.cute.lowprec.specs import Bwd2Spec, HostSpec, QuantSpec


def _resolved_key(obj, names) -> tuple:
    """The named constexprs read off a CONSTRUCTED kernel, i.e. after any ``__init__`` override (the same tuple as
    lowprec.specs.resolved_key): an absent attribute contributes None, a value that is not a plain scalar its str()."""
    return tuple((n, v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
                 for n, v in ((n, getattr(obj, n, None)) for n in names))


def _lpt_l2_worth_it(s_k: int, d: int, d_v: int, element_size: int = 1) -> bool:
    """Should the backward use the L2-locality LPT tile scheduler?

    The scheduler groups `swizzle` consecutive (head, batch) pairs so their Q/dO/dQaccum stay
    L2-resident, sizing the group from the expression reproduced here (swizzle is the largest
    power of two <= budget/size_one_head). The threshold below is EMPIRICAL, not derived --
    measured on the FP4 2-CTA backward, d=128 h=32, B200, medians matched per GPU:

        s4096  swizzle 16   +1.0 .. +1.2%    s16384  swizzle 4   within noise (that shape's
        s8192  swizzle  8   +3.2 .. +3.4%                        spread is ~6%, and the two
                                                                 GPUs disagree on the sign)
        s32768 swizzle  2   -0.5%

    so it pays while the formula yields >= 4 and turns slightly negative at 2, where the grouping
    no longer offsets the LPT path's 1-D grid and divmod. s16384 sits at the threshold and is kept
    only because it is neutral, not because it was shown to win. Bigger is not better either: forcing the
    budget to 200 MB (swizzle 32 at s8192, i.e. every head in one group) measured 0.858 ms vs
    0.821 at the default -- and the true B200 L2 of 126 MiB measured the same as the 50 MB default,
    so `_L2_SIZE_MB` is a tuning constant, not a hardware figure to be "corrected".
    """
    size_one_head = s_k * (d + d_v) * element_size + s_k * d * 4
    return (_L2_SIZE_MB * 1024 * 1024) // max(size_one_head, 1) >= 4

def _resolve_lpt_l2(spec: Bwd2Spec, s_k: int, d: int, d_v: int) -> bool:
    """The LPT-L2 swizzle decision, resolved HERE so the key and the kernel read one variable: ``spec.lpt_l2`` (the
    KernelTuning.two_cta8_lpt_l2 override) when set, else the driver's rule.

    `lpt-l2-override-unkeyed`: the override used to be applied inside the kernel's ``__init__``
    while the compile key carried the pre-override `_lpt_l2_worth_it()` value, so an A/B of the setting
    in one process measured the first cubin twice. Rule: resolve an override where the key is built.
    """
    if spec.lpt_l2 is not None:
        return spec.lpt_l2
    return _lpt_l2_worth_it(s_k, d, d_v)


# P' = 2^8 P <= 256 < 448 (E4M3 max). `p-shift-two-names`: this is the value of `e4m3_p_shift_log2()` (the one name)
# that the 2-CTA route runs, not an independent constant; `_use_2cta_dk8` declines any other
# value and the driver below refuses a direct call that disagrees. Kept as a name because fp4_bwd_autograd imports it.
P_SHIFT_LOG2 = P8_SHIFT_2CTA


# SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
def _tb_mode(spec: Bwd2Spec) -> bool:
    """The token-block operand mode of this route (spec.op_scaling: the 4-bit route's E4M3 operand scaling under
    nvfp4_s, else the 8-bit route's, = blk32x128)."""
    return spec.op_scaling == "blk32x128"


def flash_attn_bwd_2cta8(q, k, v, out, dout, lse, softmax_scale: Optional[float] = None, causal: bool = False,
                         fp8_scales: Optional[dict] = None, return_accum: bool = False, nvfp4_s: Optional[bool] = None,
                         q4=None, sfq=None, k4=None, sfk=None, s_sf_vec_size: Optional[int] = None, masks=None,
                         ds_dq_rowlocal: str = "off", deterministic: bool = False, *, spec: Bwd2Spec, quant: QuantSpec,
                         host: HostSpec):
    """2-CTA backward with E4M3 operands for S / dP / dV / dK / dQ. q, k, v, out, dout: (b, s, h, 128) bf16; lse (b, h, s) fp32.
    fp8_scales: optional {q8, k8, v8, do8, sq, sk, sv, sdo, s_ds} (pre-quantised operands and the scales computed from
    this call's tensors). spec / quant / host: the call's plan.specs.bwd2 (the kernel's compile-time settings), .quant and
    .host (this driver's) -- lowprec/specs.py, required."""
    if spec is None or quant is None or host is None:
        raise TypeError("flash_attn_bwd_2cta8: spec (Bwd2Spec), quant (QuantSpec) and host (HostSpec) are required: the "
                        "call's plan.specs")
    from flash_attn.cute.interface_bwd_fp4 import check_codes_scaling
    # SCALING-CONSISTENCY RULE: the codes handed over were quantised in the plan's one operand scaling
    check_codes_scaling(nvfp4_s is not None and not nvfp4_s, fp8_scales, masks, quant, "flash_attn_bwd_2cta8")

    def _spec_for(nv: bool) -> Bwd2Spec:
        return spec
    assert _get_device_capability() == 10
    b, s_q, h, d = q.shape
    s_k = k.shape[1]
    from flash_attn.cute.lp_masks import lp_masks_or_none
    masks = lp_masks_or_none(masks, causal)  # masks / varlen (explicit LPMasks; None = the plain kernel, bitwise)
    doc_t128 = False
    if masks is not None:
        causal = masks.causal
        if masks.docs is not None and masks.docs.align % 256 != 0:
            # documents packed at 128: the key pair may straddle two documents (flash_bwd_sm100_2cta8 doc_t128); each
            # CTA masks in its own tile's document
            if masks.docs.align % 128 != 0 or masks.any_mod:
                raise ValueError("2-CTA backward with documents packed at 128: no score_mod / mask_mod (got align "
                                 f"{masks.docs.align}, mods {masks.any_mod})")
            doc_t128 = True
    gqa = k.shape[2] != h   # bwdk opt-in (HostSpec.bwd_2cta_gqa): fp32 dK / dV reduce-adds, converted after the kernel
    # GQA: deterministic orders the fp32 dK / dV reduce-adds by q head with upstream's dK / dV semaphores
    # (deterministic_KV, flash_bwd_sm100); documents reduce-add into the packed (1, h_k, total_k * d) accumulators
    assert d == 128 and (not gqa or (host.bwd_2cta_gqa and h % k.shape[2] == 0)), \
        "2-CTA E4M3 backward: head_dim 128, MHA (GQA: KernelTuning.bwd_2cta_gqa)"
    assert s_q % 128 == 0 and s_k % 128 == 0, "2-CTA E4M3 backward: seqlen_q % 128 == 0, seqlen_k % 128 == 0"
    # batched s_k % 256 == 128: the last key pair's upper half is past the keys (masked, its dK / dV not written)
    key_tail = (masks is None or masks.docs is None) and s_k % 256 != 0

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    from flash_attn.cute.interface_bwd_fp4 import contig
    q, k, v, out, dout, lse = [contig(t) for t in (q, k, v, out, dout, lse)]
    device = q.device
    sc = dict(fp8_scales or {})
    mxq = None
    _docs = masks is not None and masks.docs is not None
    # option A's research backward (NVFP4 dP on the forward's V codes, fp8_scales["pv4"]) is not shipped: option A is
    # inference-only
    if sc.get("pv4") is not None:
        raise NotImplementedError("option A's research backward (fp8_scales['pv4']) is not part of this build; option A is "
                                  "inference-only")
    # (nvfp4_s=None reads the 8-bit route's operand scaling here, before the None -> True default below)
    if "mx" in sc or _tb_mode(_spec_for(bool(nvfp4_s))) or _docs:  # varlen: token-block operands, as the 1-CTA driver
        # token-block operands (blk32x128): codes with unit scales, D from the dequantised dO,
        # the d-rows SF atoms of Q / K / dO and the (4, b, h, s / 32) factors
        from flash_attn.cute.interface_bwd_fp4 import mx_operands
        mxq = dict(sc.get("mx") or {})
        need = {nm: t for nm, t in (("q", q), ("k", k), ("v", v)) if nm not in mxq}
        # D is folded in place: owned only when quantised here or handed over (mx["owned"]); see interface_bwd_fp4.
        d_owned = "do" not in mxq or "do" in mxq.get("owned", ())
        if "do" not in mxq:
            mxq.pop("fac", None)  # a caller's factor table has no row for the dO quantised here: rebuilt from the "e"s
            mxq.update(mx_operands(dout=dout, out=out, masks=masks, quant=quant, host=host))
        if need:
            mxq.update(mx_operands(**need, masks=masks, quant=quant, host=host))
        # the four unit descales as views of ONE float32 device tensor (one fill launch instead of four dev32 fills)
        from flash_attn.cute.device_scales import unit_descales
        _u = unit_descales(device)   # a per-device constant: no fill launch
        # the f16 dQ range (dq_range_of): the caller's norms, else the statistics of the operands quantised here
        _fs = fp8_scales or {}
        _rng = {key: _fs[key] if _fs.get(key) is not None else mxq[nm].get(stat)
                for key, nm, stat in (("dn2", "do", "row_norm2_max"), ("vn2", "v", "row_norm2_max"),
                                      ("kmax", "k", "amax"))}
        from flash_attn.cute.interface_bwd_fp4 import dq_kv_rows_of
        _rng["dq_kv_rows"] = dq_kv_rows_of(_fs, k, v, causal)
        sc = dict(q8=mxq["q"]["codes"], k8=mxq["k"]["codes"], v8=mxq["v"]["codes"], do8=mxq["do"]["codes"], sq=_u[0],
                  sk=_u[1], sv=_u[2], sdo=_u[3], dpsum=mxq["do"]["D"], dpsum_owned=d_owned,
                  d_folded=mxq["do"].get("d_folded") if d_owned else None, **_rng)
        # the fused step glue (fp4_bwd_autograd.tb_bwd_glue): LSE * log2 e - P_SHIFT_LOG2 and the zeroed f16 dQ
        # accumulator from the autograd backward's own dO launch (its fresh, owned dict)
        _do = mxq["do"]
        if "do" in mxq.get("owned", ()):
            if _do.get("lse_log2") is not None and float(_do.get("lse_log2_shift", -1)) == float(P_SHIFT_LOG2):
                sc["lse_log2"] = _do["lse_log2"]
            if _do.get("dq_accum") is not None:
                sc["dq_accum"] = _do["dq_accum"]
        if (fp8_scales or {}).get("had_signs"):  # qk_hadamard: the codes are Q R / K R (kept for the back-rotation)
            sc["had_signs"] = fp8_scales["had_signs"]
    # every scale is a float32 device scalar from here on (device_scales): no host read in this driver
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    from flash_attn.cute.device_scales import dev32, bwd2_scalars

    def e4m3(t, s):  # a missing operand: per-tensor E4M3 with its (given or GPU-computed) float32 device scale
        r = quantize_fp8_per_tensor(t, scales=None if s is None else dev32(s, device),
                                    power_of_two_scale=s is None and quant.per_tensor_pow2)
        return r["codes"], r["scales"][0]
    q8, sq = (sc["q8"], dev32(sc["sq"], device)) if "q8" in sc else e4m3(q, sc.get("sq"))
    k8, sk = (sc["k8"], dev32(sc["sk"], device)) if "k8" in sc else e4m3(k, sc.get("sk"))
    v8, sv = (sc["v8"], dev32(sc["sv"], device)) if "v8" in sc else e4m3(v, sc.get("sv"))
    do8, sdo = (sc["do8"], dev32(sc["sdo"], device)) if "do8" in sc else e4m3(dout, sc.get("sdo"))
    # `2cta-e5m2-grad-mistyped`: every FP8 GEMM of this kernel is built with ONE element type,
    # `make_trivial_tiled_mma(self.do_dtype, ...)` for dP / dV / dK, and P^ is recast to `self.do_dtype` in TMEM, so an
    # E5M2 dO makes the E4M3 V, P^ and dS^ operands read as E5M2. Measured through flash_attn_fp8_func with
    # fp8_grad_dtype="e5m2" (b1 s2048 h4 d128, a fresh process): dQ and dK NaN, dV cosine 0.566 against 0.9986 for
    # E4M3. It was hidden until `2cta-key-missing-grad-dtype` was fixed, because the E4M3 kernel was reused for E5M2
    # data. Refuse it rather than return NaN; route E5M2 to a kernel that types dO separately.
    for _nm, _t in (("q8", q8), ("k8", k8), ("v8", v8), ("do8", do8)):
        if _t.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"flash_attn_bwd_2cta8: {_nm} is {_t.dtype}; the 2-CTA backward types every FP8 GEMM with one element "
                "type and supports E4M3 operands only (an E5M2 dO gives NaN dQ/dK). Use fp8_grad_dtype='e4m3', or the "
                "1-CTA backward (two_cta='off').")
    # dk8 configuration (default): S from NVFP4 row-wise Q / K; nvfp4_s=False is the all-E4M3 FP8 route. Every library
    # caller passes it explicitly.
    if nvfp4_s is None:
        nvfp4_s = True
    # S scale-factor format: 16 = NVFP4 (E4M3 per 16), 32 = MXFP4 (UE8M0 per 32)
    if s_sf_vec_size is None:
        s_sf_vec_size = 16
    kspec = _spec_for(nvfp4_s)
    if nvfp4_s and q4 is None:
        from flash_attn.cute.quant.fp4 import quantize_fp4_rows
        rq = quantize_fp4_rows(q, scale_group_size=s_sf_vec_size)
        rk = quantize_fp4_rows(k, scale_group_size=s_sf_vec_size)
        q4, sfq, k4, sfk = rq["codes"], rq["scale_factors"], rk["codes"], rk["scale_factors"]

    dtype = torch2cute_dtype_map[q.dtype]
    m_block_size = 128
    # f16 dQ partials (KernelTuning.bwd_dq_f16, default on; `dq-f16-two-names`): half the reduce-add bytes;
    # zeroed here (the preprocess zeroes fp32 only)
    dq_f16 = bwd_dq_f16(host)
    if e4m3_p_shift_log2() != P_SHIFT_LOG2:
        raise NotImplementedError(
            f"flash_attn_bwd_2cta8 runs E4M3 P^ at 2^{P_SHIFT_LOG2} only; a shift of "
            f"2^{e4m3_p_shift_log2()} is served by the 1-CTA backward (_use_2cta_dk8 routes it there)")
    # where the f16 accumulator's zeros come from (HostSpec.dq_zero, see dq_postprocess_cute): the caller's dO-quantiser
    # launch (fp8_scales["dq_accum"]), the pool re-zeroed by the previous call's postprocess, or a fill here
    from flash_attn.cute.dq_postprocess_cute import dq_zero_mode, take_zeroed_dq_accum_2cta
    # fp8_scales["dpsum"]: D from exactly the E4M3 dO the dP GEMM uses (the training default: softmax-consistent D); the
    # prologue is then skipped and LSE * log2 e comes from torch, as in the 1-CTA dk8 path -- or, with
    # fp8_scales["lse_log2"], from the same launch as D and already shifted by P_SHIFT_LOG2 (fp4_bwd_autograd);
    # fp8_scales["dpsum_owned"] says the D tensor is this call's to scale in place
    ext_d = sc.get("dpsum")
    lse_shifted = ext_d is not None and sc.get("lse_log2") is not None
    ext_acc = sc.get("dq_accum")
    acc_pooled = False
    # `acc_zeroed`: the accumulator already holds zeros. Otherwise it is allocated uninitialised and zeroed by the
    # prologue launch (Bwd2Cta8Prologue, the 2-CTA side-kernel fusion), or by torch where no prologue runs.
    acc_zeroed = True
    if dq_f16 and ext_acc is not None and ext_acc.shape == (b, h, s_q * d) and ext_acc.dtype == torch.float16 and ext_acc.is_contiguous():
        dq_accum = ext_acc
    elif dq_f16 and dq_zero_mode(host) == "pool" and not return_accum:
        dq_accum = take_zeroed_dq_accum_2cta(b, h, s_q * d, torch.float16, device)
        acc_pooled = True
    else:
        dq_accum = torch.empty(b, h, s_q * d, dtype=torch.float16 if dq_f16 else torch.float32, device=device)
        acc_zeroed = False
    if ext_d is not None:
        dpsum = ext_d if sc.get("dpsum_owned") else ext_d.float().contiguous().clone()  # the dS fold scales D in place
        lse_log2 = sc["lse_log2"] if lse_shifted else torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse * math.log2(math.e)).float().contiguous()
        assert lse_log2.shape == (b, h, s_q) and lse_log2.dtype == torch.float32 and lse_log2.is_contiguous(), lse_log2.shape
        assert dpsum.shape == (b, h, s_q) and dpsum.dtype == torch.float32 and dpsum.is_contiguous(), dpsum.shape
        if not acc_zeroed:
            dq_accum.zero_()
    else:
        dpsum = torch.empty(b, h, s_q, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(b, h, s_q, dtype=torch.float32, device=device)
    dq = torch.empty_like(q)
    # masks / varlen: a key pair with an empty query range (window / document tail) writes no dK / dV: zeros
    from flash_attn.cute.interface_bwd_fp4 import dkv_zero_fill
    dk = torch.zeros_like(k) if dkv_zero_fill(masks) else torch.empty_like(k)
    dv = torch.zeros_like(v) if dkv_zero_fill(masks) else torch.empty_like(v)
    if gqa:
        # the kernel's dKV_postprocess form: (b, h_k, s_k * d) fp32 accumulators the q heads reduce-add into
        # (pooled: the post launch writes zeros back over what it reads, so steady state has no fill launch)
        from flash_attn.cute.bwd_2cta8_gqa import take_zeroed_dkv_accum
        dk_out, dv_out = dk, dv
        dk, dv = take_zeroed_dkv_accum(b, k.shape[2], s_k * d, device)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # d\hat{S} = dS / s_ds with s_ds = max|dS| bound / 448, exactly as the 1-CTA dk8 path: a precomputed scale, the forward's
    # per-query P_max (training), or the exact host bound (O(s^2), accuracy runs only)
    # `2cta-blockscaled-ds-missing-fp8-guard`, host half. Without `not nvfp4_s` the dk8 route also took
    # s_ds = 1.0, i.e. it assumed in-kernel SFs that a 4-bit S never produces -- so deleting the kernel's assert
    # instead of guarding it would have given wrong dK/dQ rather than a loud failure. Must stay in step with
    # `self.mxds` in flash_bwd_sm100_2cta8.py.
    mxds = (not nvfp4_s) and kspec.ds_scaling_fp8 != "per_tensor"
    if nvfp4_s:  # Bwd2Spec.ds_scaling_fp4 = blk32: the kernel's `mxds_fp4` (same rung on the dk8 route; = ds_blk32_fp4())
        mxds = kspec.ds_scaling_fp4 == "blk32"
    # The D fold D'' = c D is always the `dpsum.mul_(c)` below, with c computed on the GPU: the prologue used to fold it
    # when s_ds was known early (block-scaled d\hat{S}, or a supplied scale), which needs c on the host. It is the same fp32
    # product either way (the prologue's D is the fp32 value the multiply reads), one elementwise launch more.
    if ext_d is None:
        # One launch for the accumulator fill, D and LSE * log2 e - P_SHIFT_LOG2: three launches before (torch.zeros,
        # upstream's preprocess, and a torch elementwise op).
        from flash_attn.cute.bwd_2cta8_side import Bwd2Cta8Prologue
        c_d = 1.0
        acc_t = None if acc_zeroed else dq_accum
        key_pre = ("prologue", dtype, d, None if acc_t is None else acc_t.dtype)
        if key_pre not in flash_attn_bwd_2cta8.cache_pre:
            pro = Bwd2Cta8Prologue(dtype, None if acc_t is None else torch2cute_dtype_map[acc_t.dtype], d, m_block_size)
            flash_attn_bwd_2cta8.cache_pre[key_pre] = cute.compile(
                pro, *[to_cute_tensor(t, divisibility=8) for t in (out, dout)], to_cute_tensor(lse, assumed_align=4),
                *[to_cute_tensor(t, divisibility=4) for t in (dpsum, lse_log2)],
                None if acc_t is None else to_cute_tensor(acc_t, divisibility=8),
                Float32(c_d), Float32(float(P_SHIFT_LOG2)), current_stream, options="--enable-tvm-ffi",
            )
        flash_attn_bwd_2cta8.cache_pre[key_pre](out, dout, lse, dpsum, lse_log2, acc_t, Float32(c_d),
                                                Float32(float(P_SHIFT_LOG2)), current_stream)

    if masks is not None:
        from flash_attn.cute.interface_bwd_fp4 import _check_masks_ds
        _check_masks_ds(masks, not mxds)
    if mxds:
        # block-scaled d\hat{S} (in-kernel UE8M0 scale factors): no global max|dS|; the SFs carry the range and
        # the MMA applies them, so the epilogue scales see s_ds = 1
        sds_kw = dict(sds_mode=3)
    elif "s_ds" in sc:
        sds_kw = dict(s_ds=dev32(sc["s_ds"], device), sds_mode=0)
    elif "s_ds_r" in sc:  # the P_max bound r, max-reduced by the dO / V quantisers of this step (no norm passes)
        sds_kw = dict(r=sc["s_ds_r"], sds_mode=1)
    elif "pmax" in sc:
        # see interface_bwd_fp4: the only read of v; row_norms_f32 takes the norms without the fp32 copies.
        # s_ds = max(r, 1e-30) / 448 of this float32 max, on the GPU
        vmax = sc["vmax"] if "vmax" in sc else row_norms_f32(v).amax()
        y = row_norms_f32(dout).permute(0, 2, 1) * vmax + dpsum[:, :, :s_q].abs().float()
        sds_kw = dict(r=(sc["pmax"].float().clamp(max=1.0) * y).amax().reshape(1), sds_mode=1)
    else:
        # a per-tensor dS with no scale source: there is no exact-amax fallback (it would be O(s^2) on the host); every
        # library caller passes s_ds, s_ds_r or the forward's pmax
        raise ValueError("flash_attn_bwd_2cta8: a per-tensor dS needs fp8_scales['s_ds'], ['s_ds_r'] or ['pmax'] "
                         "(the exact max|dS| fallback was removed)")
    if not lse_shifted and ext_d is not None:  # the prologue already subtracted it
        lse_log2.sub_(float(P_SHIFT_LOG2))
    # the kernel scalars on the GPU (device_scales._bwd2_scalars_kernel, the host expressions transcribed):
    #   ds_b = (s_v s_dO / 2^P) / s_ds, D'' = -ds_b D / (s_v s_dO), softmax_scale_k = softmax_scale [s_q s_k unless NVFP4 S,
    #   which carries its scale factors], dv = s_dO / 2^P, dk = softmax_scale s_ds s_q, dq = softmax_scale s_ds s_k,
    #   dq kernel scale = s_ds s_k with f16 partials (the postprocess applies softmax_scale), else 1
    # the f16 dQ partials' data-sized pre-scale (device_scales._dq_prescale): scal[7] is then the postprocess scale
    from flash_attn.cute.interface_bwd_fp4 import dq_range_of, dq_tiles_of
    from flash_attn.cute.qk_hadamard import fuse_backrot as _fb
    dq_range = dq_range_of(sc, block_scaled=mxq is not None) if (dq_f16 and not return_accum) else None
    # dqfix: the pre-scale per (row, q head, query tile) where the statistics are per tile (token-block operands)
    dq_tiles = (dq_tiles_of(dq_range, masks, b, h, k.shape[2], s_q, s_k, causal=causal,
                         rows_src=None if d != 128 else (sc.get("dq_kv_rows"), dout, _fb(sc))) if mxq is not None and dq_range is not None
                and dq_accum.shape[2] == s_q * 128 else None)
    scal = bwd2_scalars(device, softmax_scale, P_SHIFT_LOG2, nvfp4_s, dq_f16, sq, sk, sv, sdo, dq_range=dq_range,
                        dq_tiles=dq_tiles, **sds_kw)
    post_s = scal[7:8] if dq_range is not None else softmax_scale
    if dq_tiles is not None:
        from flash_attn.cute.device_scales import dq_tile_planes
        post_s = dq_tile_planes(scal, dq_tiles)
    from flash_attn.cute.device_scales import apply_d_fold
    apply_d_fold(dpsum, scal, 5, sc.get("d_folded"))  # D'' = -b D / (s_v s_dO) (or already, in the dO launch)

    # ---- compile key
    # An older hand-kept list of key entries (the root cause of `unkeyed-numerics-knobs`, `hand-rolled-compile-keys`:
    # settings that changed the kernel but not the key) became the kernel's spec:
    # spec.key() carries every compile-time setting the kernel reads, by construction (the kernel reads nothing else).
    # Added alongside it:
    #   do8.dtype        `2cta-key-missing-grad-dtype`: an in-process E4M3 -> E5M2 gradient switch
    #                    reused the first kernel, because `dp8` is a bool and not a dtype.
    #   lpt_l2           now the POST-override value (`_resolve_lpt_l2`), Sec. 3.8 group 3.
    #   deterministic, d, d_v, r_gqa          the 1-CTA key has all four; this one had none.
    #   tile_m/tile_n, cluster_size, use_2cta_instrs, is_persistent, spt
    #                    fixed today, so latent -- but `MX_XW="dq"` on a 1-CTA-instruction build hangs,
    #                    which is precisely a latent item earning its place in the key.
    #   score/mask mod hashes, has_aux_tensors, subtile_factor
    #                    Constexpr constructor arguments with no key entry; None/1 today.
    # The kernel object is constructed BEFORE the lookup (pure Python, microseconds) so the key can
    # carry `_resolved_key`, the constexprs as they stand after any in-`__init__` override.
    lpt_l2 = _resolve_lpt_l2(kspec, s_k, d, d)
    d_v = d
    r_gqa = h // k.shape[2]          # 1 today: the shape assert above is MHA-only
    deterministic = bool(deterministic)  # dQ semaphore per (b, h, m_block, CTA of the pair): upstream's ordered reduce-add
    tile_m, tile_n = 128, 128
    cluster_size, use_2cta_instrs = 2, True
    is_persistent, spt = False, None
    score_mod = score_mod_bwd = mask_mod = None
    has_aux_tensors, subtile_factor = False, 1
    # masks / varlen: explicit mask arguments (lp_masks.LPMasks); none -> the plain constructor
    mask_kw = {}
    if masks is not None:
        mask_kw = dict(is_local=masks.local, doc_varlen=masks.docs is not None)
        if doc_t128:
            mask_kw["doc_t128"] = True
        score_mod, score_mod_bwd, mask_mod = masks.score_mod, masks.score_mod_bwd, masks.mask_mod
        has_aux_tensors = masks.aux_tensors is not None
        spt = masks.spt
    kern = FlashAttentionBackwardSm100TwoCta8(
        d, d_v, is_causal=causal, qhead_per_kvhead=r_gqa, tile_m=tile_m, tile_n=tile_n,
        is_persistent=is_persistent, deterministic=deterministic, lpt_l2=lpt_l2, spt=spt,
        cluster_size=cluster_size, use_2cta_instrs=use_2cta_instrs,
        score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod,
        has_aux_tensors=has_aux_tensors, subtile_factor=subtile_factor,
        nvfp4_s=nvfp4_s, s_sf_vec_size=s_sf_vec_size, ds_dq_rowlocal=ds_dq_rowlocal, **mask_kw,
        token_local_operands=bool(mxq is not None and mxq.get("tok") is not None),
        spec=kspec, **({"key_tail": True} if key_tail else {}),
        **({"dq_tile_scaled": True} if dq_tiles is not None else {}),
        **({"dq_tile_causal": 2 if dq_tiles.get("rowwise") else 1} if (dq_tiles is not None and dq_tiles.get("causal")) else {}),
    )
    # qk_hadamard (fp8_scales["had_signs"]): q8 / k8 (and the NVFP4 S codes) are rotated. The dK epilogue undoes it
    # in registers; the dQ postprocess becomes the thread-per-row DQPostprocess2CTA with the rotation undone in registers
    from flash_attn.cute.qk_hadamard import fuse_backrot
    had = fuse_backrot(sc)
    if had and not gqa:   # GQA: the dK back-rotation runs in the dK / dV post launch (bwd_2cta8_gqa)
        from flash_attn.cute.qk_hadamard import signs_tuple
        kern.had_dk = signs_tuple(128)
    tok = mxq.get("tok") if mxq is not None else None  # token-local operands: K / V / dO factors
    # the kernel's compile-time settings are kern.spec (spec.key(): plain values, never the object's identity)
    key = (FlashAttentionBackwardSm100TwoCta8.__qualname__, kern.spec.key(),
           had, tok is not None, dtype, do8.dtype, causal, nvfp4_s, s_sf_vec_size, dq_f16, lpt_l2, deterministic,
           d, d_v, r_gqa, tile_m, tile_n, cluster_size, use_2cta_instrs, is_persistent, spt,
           has_aux_tensors, subtile_factor,
           getattr(score_mod, "__name__", None), getattr(score_mod_bwd, "__name__", None),
           getattr(mask_mod, "__name__", None),
           _get_device_capability(), cutlass.__version__,
           _resolved_key(kern, ("mxds", "mxds_row", "mxp", "ds_scaling", "p_scaling",
                                     "mx_xw", "mx_xw_all", "lpt_l2",
                                     "warp_sync", "ld_rep", "ds_dq_rowlocal")))
    if masks is not None:
        key = key + (masks.key(),)  # every compile-time mask item (window values and the tables are runtime)
    if doc_t128:
        key = key + ("doc_t128",)
    if key_tail:
        key = key + ("key_tail",)
    if dq_tiles is not None:
        key = key + ("dq_tiles",) + (("dq_causal", bool(dq_tiles.get("rowwise"))) if dq_tiles.get("causal") else ())

    if kern._test_skew_peer:  # the park-race test's private hook (tests/cute/test_2cta_dp_park_race.py); never set here
        key = key + ("_test_skew_peer",)
    wl = wr = aux_c = aux_r = tab = ndoc = tab_c = ndoc_c = None
    if masks is not None:
        wl, wr = masks.window_args()
        aux_r = masks.aux_list()
        if aux_r is not None:
            from flash_attn.cute.lp_masks import to_cute_aux
            aux_c = to_cute_aux(aux_r)
        if masks.docs is not None:
            tab, ndoc = masks.docs.doc_tab, masks.docs.tile_doc(128 if doc_t128 else 256)
            tab_c = to_cute_tensor(tab, assumed_align=4, leading_dim=1)
            ndoc_c = to_cute_tensor(ndoc, assumed_align=4, leading_dim=0)

    wl_c = None if wl is None else cutlass.Int32(0)
    wr_c = None if wr is None else cutlass.Int32(0)
    fp4_args_c = fp4_args = (None, None, None, None, (), ())
    tb_args_c = tb_args = (None, None, None, None, None)
    if mxq is not None:
        from flash_attn.cute.interface_bwd_fp4 import mx_factors
        fac = mx_factors(mxq)
        # token-local operands: the quantisers wrote unit dQ-side K (and, 8-bit, dK-side Q) scales
        _sfkd = mxq["k"]["sf_dim_raw"]
        _sfqd = mxq["q"]["sf_dim_raw"]
        tb_args = (_sfqd.permute(4, 5, 2, 6, 3, 1, 0), _sfkd.permute(4, 5, 2, 6, 3, 1, 0),
                   mxq["do"]["sf_dim_raw"].permute(4, 5, 2, 6, 3, 1, 0), fac, tok)
        tb_args_c = tuple(to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in tb_args[:3]) + (
            to_cute_tensor(fac, assumed_align=16), None if tok is None else to_cute_tensor(tok, assumed_align=16))
    if nvfp4_s:
        from flash_attn.cute.interface_bwd_fp4 import _make_fp4_ptr
        sfq_t, sfk_t = [to_cute_tensor(t, leading_dim=3, assumed_align=16) for t in (sfq, sfk)]
        z4 = tuple(cutlass.Int32(0) for _ in range(4))
        fp4_args_c = (_make_fp4_ptr(None), _make_fp4_ptr(None), sfq_t, sfk_t, z4, z4)
        fp4_args = (_make_fp4_ptr(q4), _make_fp4_ptr(k4), sfq, sfk, (b, s_q, h, d), (b, s_k, k.shape[2], d))
    dQ_semaphore = dK_semaphore = dV_semaphore = None
    if deterministic and gqa:
        # one zero fill for the three semaphore sets: dQ (b, h, m blocks, CTA of the pair) and upstream's dK / dV
        # semaphores (b, h_k, key tiles of 128, compute warpgroup) that order the q heads' fp32 reduce-adds
        _nq, _nk = b * h * (s_q // tile_m) * cluster_size, b * k.shape[2] * (s_k // tile_n) * 2
        _sem = torch.zeros(_nq + 2 * _nk, dtype=torch.int32, device=device)
        dQ_semaphore = _sem[:_nq].view(b, h, s_q // tile_m, cluster_size)
        dK_semaphore = _sem[_nq:_nq + _nk].view(b, k.shape[2], s_k // tile_n, 2)
        dV_semaphore = _sem[_nq + _nk:].view(b, k.shape[2], s_k // tile_n, 2)
    elif deterministic:
        dQ_semaphore = torch.zeros(b, h, s_q // tile_m, cluster_size, dtype=torch.int32, device=device)
    if key not in flash_attn_bwd_2cta8.cache:
        q_t, k_t, v_t, do_t = [to_cute_tensor(t, divisibility=16) for t in (q8, k8, v8, do8)]
        dk_t, dv_t = [to_cute_tensor(t, divisibility=8) for t in (dk, dv)]
        dqa_t, dps_t, lsel_t = [to_cute_tensor(t, divisibility=4) for t in (dq_accum, dpsum, lse_log2)]
        flash_attn_bwd_2cta8.cache[key] = cute.compile(
            kern, q_t, k_t, v_t, do_t, lsel_t, dps_t, dqa_t, dk_t, dv_t,
            *[Float32(1.0)] * 5,  # ignored: the kernel reads mScalars
            None, None, None, None, wl_c, wr_c,
            to_cute_tensor(dQ_semaphore, assumed_align=4) if deterministic else None,
            *[None if t is None else to_cute_tensor(t, assumed_align=4) for t in (dK_semaphore, dV_semaphore)],
            aux_c, None,
            *fp4_args_c, *tb_args_c,
            to_cute_tensor(torch.empty(8, dtype=torch.float32, device=device), assumed_align=4, leading_dim=0),
            tab_c, ndoc_c, current_stream,
            options="--enable-tvm-ffi",
        )
    flash_attn_bwd_2cta8.cache[key](
        q8, k8, v8, do8, lse_log2, dpsum, dq_accum, dk, dv, 1.0, 1.0, 1.0, 1.0, 1.0,
        None, None, None, None, wl, wr, dQ_semaphore, dK_semaphore, dV_semaphore, aux_r, None, *fp4_args, *tb_args,
        scal, tab, ndoc,
        current_stream,
    )
    if gqa:
        # the fp32 accumulators hold the epilogue's tile order (the epilogue already applied dk_scale / dv_scale):
        # one launch converts dK and dV (bitwise upstream's FlashAttentionBackwardPostprocess), undoes the qk_hadamard
        # rotation on dK and re-zeroes both accumulators for the pool
        from flash_attn.cute.bwd_2cta8_gqa import dkv_postprocess_2cta_gqa
        dkv_postprocess_2cta_gqa(dk, dv, dk_out, dv_out, had=had)
        dk, dv = dk_out, dv_out

    if return_accum:
        return dq_accum, dk, dv, scal[6]
    if had and not gqa:
        dk._had_unrotated = True
    if dq_f16 and had and dq_accum.shape[2] != s_q * 128:  # a rounded accumulator: the thread-per-row form
        from flash_attn.cute.dq_postprocess_cute import dq_postprocess_2cta_cute
        return dq_postprocess_2cta_cute(dq_accum, s_q, post_s, q.dtype,
                                        zero_after=acc_pooled, had=True), dk, dv
    if dq_f16:
        # Bwd2Cta8Postprocess: the same per-element arithmetic as dq_postprocess_cute.DQPostprocess2CTA with coalesced
        # stores. A pooled accumulator is re-zeroed by the postprocess as it is read, and goes back to the pool.
        # qk_hadamard: the same launch undoes the rotation (lane-pair form, bitwise DQPostprocess2CTA(had_signs)).
        from flash_attn.cute.bwd_2cta8_side import dq_postprocess_2cta8
        return dq_postprocess_2cta8(dq_accum, s_q, post_s, q.dtype,
                                    zero_after=acc_pooled, had=had, tile_scale=dq_tiles is not None,
                                    row_scale=bool(dq_tiles is not None and dq_tiles.get("rowwise"))), dk, dv
    key_post = (dtype, d)
    if key_post not in flash_attn_bwd_2cta8.cache_post:
        post = FlashAttentionBackwardPostprocess(dtype, d, 100, m_block_size, 128, 1, False, use_2cta_instrs=True)
        # upstream's postprocess gained `sink_tensors` and `mCuTotalMBlocks` before the stream; this fp32-dQ path
        # (bwd_dq_f16 off) still passed the stream in the sink slot and died with ARG_ANNOTATION_MISMATCH
        flash_attn_bwd_2cta8.cache_post[key_post] = cute.compile(
            post, to_cute_tensor(dq_accum, divisibility=4), to_cute_tensor(dq, divisibility=8), Float32(1.0), None, None,
            None, None, current_stream, options="--enable-tvm-ffi",
        )
    if return_accum:
        return dq_accum, dk, dv, scal[6]
    dq_accum.mul_(scal[6])  # the fp32 postprocess's dq_s, applied on the GPU (acc * dq_s, the same fp32 product)
    flash_attn_bwd_2cta8.cache_post[key_post](dq_accum, dq, 1.0, None, None, None, None, current_stream)
    return dq, dk, dv


flash_attn_bwd_2cta8.cache = {}
flash_attn_bwd_2cta8.cache_pre = {}
flash_attn_bwd_2cta8.cache_post = {}

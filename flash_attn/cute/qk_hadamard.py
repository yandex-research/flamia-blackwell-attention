# Random-sign Hadamard rotation of Q and K along the head dimension, for the low-precision routes (the qk_hadamard option).
#
#   R = diag(signs) H_d / sqrt(d)   (orthogonal; signs = nvfp4_2d_triton.hadamard_signs(d, 4321), the all-FP4 route's)
#   Q~ = Q R, K~ = K R              S = Q~ K~^T = Q K^T exactly, so O and LSE need nothing
#   dQ = dQ~ R^T, dK = dK~ R^T      the backward's GEMMs against rotated operands give rotated gradients
#
# Where it happens (nothing here adds a pass over Q or K):
#   * the rotation is applied inside the quantiser launch that reads Q / K (quant.fp8 / quant.fp4);
#   * the per-tensor E4M3 scale of a rotated operand is the amax of the ROTATED tensor: quant.fp8.quantize_fp8_per_tensor
#     rotates each row in registers and reduces |x| in the same launch as the scale (apply_hadamard_rotation);
#   * the back-rotation of dQ is fused into the 2-CTA dQ postprocess and that of dK into the 2-CTA kernel's dK
#     epilogue; every other backward kernel (1-CTA dk8 / FP8 control: GQA, ragged, fp32 dQ) falls back to
#     `unrotate_rows` (one extra read + write of dQ and dK), reported as such.
#
# Settings (the call's lowprec.specs.QuantSpec, from its plan; nothing here reads any other configuration):
#   QuantSpec.qk_hadamard        the public qk_hadamard on the E4M3-operand routes
#   QuantSpec.qk_hadamard_e4m3   "rotated" | "plain" -- 4-bit routes only (None: plain on block scaling, rotated per
#                                tensor: E4M3_DEFAULT): whether the E4M3 copies of Q / K that the dk8 backward multiplies
#                                (dQ = dS K8, dK = dS^T Q8) are the rotated tensors (then dQ / dK are rotated back) or the
#                                plain ones (then no back-rotation at all: S is basis-free, and a plain K8 gives dQ in the
#                                true basis directly).
import torch

SEED = 4321
_SIGNS = {}


def enabled(qspec) -> bool:
    """QuantSpec.qk_hadamard (the public qk_hadamard on the E4M3-operand routes)."""
    return qspec.qk_hadamard


# The E4M3 copies' basis when KernelTuning.qk_hadamard_e4m3 is not set, per 4-bit operand path: on token blocks /
# 32 x 32 blocks (the default 4-bit route) the copies are of the UNROTATED Q / K (the S codes stay rotated; the
# gradients need no back-rotation; measured neutral against the rotated copies in accuracy, in a causal leak probe and
# in two-seed training arms); the per-tensor path keeps the rotated copies.
E4M3_DEFAULT = {"blocks": "plain", "per_tensor": "rotated"}


def e4m3_mode(path: str = "per_tensor", *, qspec) -> str:
    """QuantSpec.qk_hadamard_e4m3 (KernelTuning.qk_hadamard_e4m3), None = E4M3_DEFAULT[path]
    (QuantSpec.rule_qk_hadamard_e4m3)."""
    qs = qspec
    v = qs.qk_hadamard_e4m3 if qs.qk_hadamard_e4m3 is not None else E4M3_DEFAULT[path]
    if v not in ("rotated", "plain"):
        raise ValueError(f"qk_hadamard_e4m3 must be 'rotated' or 'plain', got {v!r}")
    return v


def rotation_method(qspec):
    """QuantSpec.qk_hadamard_rotation_method (KernelTuning.qk_hadamard_rotation_method): the hadamard_rotation_method every
    rotated Q / K quantiser call of the routes passes -- None (the quantisers' default, "tensor_core") or "tcgen05" (the
    opt-in UMMA rotation, quant._tcgen05; the plan refuses it where a route's quantiser has no tcgen05 form).
    Measured (B200): step time with qk_hadamard on vs off, 4-bit routes -0.6 .. +0.3 %
    (tensor_core: -0.2 .. +1.0 %), 8-bit +0.1 .. +0.5 % (tensor_core +0.2 .. +1.6 %); end to end O / LSE / dK / dV
    bitwise the default at A1 / GQA / A3 / S2; training (byte-GPT, 6 layers, 6000 steps, 2 seeds): 4-bit e4pv final loss
    -0.0026 .. +0.0020 vs tensor_core (seed spread), 8-bit +0.0024 / +0.0029 (both seeds higher, within ~1.5x the seed
    spread: unresolved with 2 seeds)."""
    v = qspec.qk_hadamard_rotation_method
    if v not in (None, "tensor_core", "tcgen05"):
        raise ValueError(f"qk_hadamard_rotation_method must be 'tensor_core' or 'tcgen05', got {v!r}")
    return v


def fuse_backrot(sc) -> bool:
    """Whether a backward driver should undo the rotation in its own kernels (fp8_scales carries "had_signs"). The
    forced-unfused validation mode was removed; drivers without the fused
    back-rotation still fall back to `unrotate_rows` through finish_grads."""
    return bool((sc or {}).get("had_signs"))


def signs(device, d: int = 128) -> torch.Tensor:
    """(d,) float32 +-1 on `device`; for d = 128 the same vector as the all-FP4 route's interface_bwd_fp4q._signs."""
    key = (str(device), d)
    if key not in _SIGNS:
        from flash_attn.cute.nvfp4_2d_triton import hadamard_signs
        _SIGNS[key] = hadamard_signs(d, SEED, device)
    return _SIGNS[key]


def signs_tuple(d: int = 128) -> tuple:
    """The signs as a Python tuple of +-1 ints (compile-time constants of a CuTe kernel)."""
    from flash_attn.cute.nvfp4_2d_triton import hadamard_signs
    return tuple(int(x) for x in hadamard_signs(d, SEED, "cpu").tolist())


def refuse_q_blk2d(qspec) -> None:
    """The 16 x 16 Q scales (q_block_2d) re-quantise the UNROTATED Q after the Q / K quantisers run; with K rotated that
    gives a wrong S. Refused until a rotated-Q 2D path exists (the plan refuses the combination first, rule 3)."""
    if qspec.q_block_2d:
        raise ValueError("qk_hadamard with q_block_2d: the 16 x 16 Q override would re-quantise the unrotated Q while K "
                         "is rotated (wrong S); no rotated-Q 2D path exists yet")


def check_shape(x: torch.Tensor, name: str) -> None:
    d = x.shape[-1]
    if d != 128:
        raise NotImplementedError(
            f"qk_hadamard: {name} has head_dim {d}; the fused rotation is built for d = 128 (the 4-bit routes are "
            "d = 128 only, and the fused back-rotation kernels are d = 128)")


# ------------------------------------------------------------------ back-rotation fallback (unfused)
def unrotate_rows(x: torch.Tensor) -> torch.Tensor:
    """x R^T = (x H / sqrt(d)) * signs for a (b, s, h, 128) gradient; one read + one write (the unfused fallback)."""
    from flash_attn.cute.nvfp4_2d_triton import fwht_rows
    return fwht_rows(x.contiguous(), signs(x.device, x.shape[-1]), inverse=True, dtype=x.dtype)


def rotate_rows(x: torch.Tensor) -> torch.Tensor:
    from flash_attn.cute.nvfp4_2d_triton import fwht_rows
    return fwht_rows(x.contiguous(), signs(x.device, x.shape[-1]), inverse=False, dtype=x.dtype)


def finish_grads(dq: torch.Tensor, dk: torch.Tensor):
    """Rotate dQ / dK back unless the backward already did it in its epilogue / postprocess (the tensors then carry the
    `_had_unrotated` attribute). Returns (dq, dk, where), where names what ran, for the logs."""
    fused_q = getattr(dq, "_had_unrotated", False)
    fused_k = getattr(dk, "_had_unrotated", False)
    if not fused_q:
        dq = unrotate_rows(dq)
    if not fused_k:
        dk = unrotate_rows(dk)
    where = ("fused" if fused_q else "fallback", "fused" if fused_k else "fallback")
    LAST_BACKROT["dq"], LAST_BACKROT["dk"] = where
    return dq, dk, where


LAST_BACKROT = {"dq": None, "dk": None}  # what the last finish_grads did (for the harness logs)


# ------------------------------------------------------------------ forward operand builders of the routes
def tb_qk_operands(q: torch.Tensor, k: torch.Tensor, *, cols: int, fac: torch.Tensor = None, fp4: bool = True,
                   k_statistics: bool = False, e4m3_of_unrotated: bool = False, qspec, row_end_q=None,
                   row_end_k=None):
    """The block-scaled routes' Q / K operands with the rotation fused into ONE quantiser launch per tensor
    (quant.fp8.quantize_fp8_blocks(apply_hadamard_rotation=True), bitwise the earlier separate rotated token-block quantiser): NVFP4 1 x 16
    codes of Q R / K R (fp4: the 4-bit routes' S operand, the S operand's encoding) and, from the same read, the E4M3
    token-block (cols 128) or 32 x 32 (cols 32) copies with every SF table, the exponent table and the token-block factors
    -- the statistics taken on the ROTATED rows. No per-tensor statistic, no extra pass.
    SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (by design) -- the returned codes
    are the only Q / K codes of the call: the forward multiplies them and the backward re-reads exactly these (saved).
    fac: the (4, b, h, s / 32) factor table (rows 0 / 1 written here) or None. Returns (mq, mk): quantize_fp8_blocks
    dicts, plus x4 / sf when fp4. k_statistics: K's per-tile max |K R| (amax) from the same launch (the
    backward's f16 dQ range).
    e4m3_of_unrotated (4-bit routes, qk_hadamard_e4m3="plain"): only the NVFP4 S codes are of Q R / K R; the E4M3
    copies (with their tables, factors and amax) are of the unrotated Q / K, from the same read on token blocks
    (quantize_fp8_blocks(hadamard_rotation_scope="nvfp4_copy")). S is basis-free and the dQ / dK GEMMs then multiply
    unrotated operands: the gradients come out in the true basis and the backward rotates nothing back.
    row_end_q / row_end_k: the aligned varlen padding masks (quant.fp8.quantize_fp8_blocks row_end), or None."""
    check_shape(q, "q")
    check_shape(k, "k")
    qs = qspec
    refuse_q_blk2d(qs)
    from flash_attn.cute.quant.fp8 import quantize_fp8_blocks
    from flash_attn.cute.fp4_bwd_autograd import FAC_K, FAC_Q, fac_row
    assert fp4 or not e4m3_of_unrotated, "e4m3_of_unrotated: the 4-bit routes (the 8-bit S operand is the E4M3 copy)"
    kw = dict(block_columns=cols, apply_hadamard_rotation=True, hadamard_sign_seed=SEED,
              hadamard_rotation_method=rotation_method(qs))
    # the NVFP4 S codes come from the same launch on token blocks (cols 128); quantize_fp8_blocks builds that dual output
    # for 32 x 128 blocks only, so the 32 x 32 MXFP8 route (cols 32) takes them from quant.fp4's rotated row copy, a
    # second read (the same rotation and the same S-operand encoding)
    fused4 = fp4 and cols == 128
    if fused4:
        kw.update(also_make_nvfp4_copy=True, nvfp4_copy_scaling="raw", nvfp4_copy_encoding="qk_operand")
        if e4m3_of_unrotated:
            kw.update(hadamard_rotation_scope="nvfp4_copy")
    elif e4m3_of_unrotated:
        kw = dict(block_columns=cols)  # the E4M3 copies of the unrotated rows; the rotated NVFP4 codes below
    rq = {} if row_end_q is None else dict(row_end=row_end_q)
    rk = {} if row_end_k is None else dict(row_end=row_end_k)
    if (rq or rk) and fp4 and not fused4:
        raise NotImplementedError("aligned varlen: the rotated NVFP4 row copy is built on the token-block launch only")
    mq = quantize_fp8_blocks(q, output_token_block_factors=fac_row(fac, FAC_Q, q), **kw, **rq)
    mk = quantize_fp8_blocks(k, output_token_block_factors=fac_row(fac, FAC_K, k),
                             output_amax=k_statistics, **kw, **rk)
    if fused4:
        mq["sf"], mk["sf"] = mq["sf4"], mk["sf4"]
    elif fp4:
        from flash_attn.cute.quant.fp4 import quantize_fp4_rows
        for m, x in ((mq, q), (mk, k)):
            r = quantize_fp4_rows(x, apply_hadamard_rotation=True, hadamard_sign_seed=SEED,
                                  hadamard_rotation_method=rotation_method(qs))
            m["x4"], m["sf"] = r["codes"], r["scale_factors"]
    return mq, mk


def fp4_qk_operands(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None, sf_vec: int = 16, *, qspec):
    """The 4-bit routes' Q / K operands with the rotation fused into the quantiser launch: NVFP4 (sf_vec 16) or MXFP4 (32)
    codes of Q R / K R plus, from the same read, the per-tensor E4M3 copies the dk8 backward multiplies -- rotated
    (qk_hadamard_e4m3="rotated": scales from the rotated amax, the backward rotates dQ / dK back) or plain (the plain
    tensors' scales; no back-rotation). Returns dict(q4, sfq, q8, k4, sfk, k8, s (n,) float32 scales of q, k[, v]
    (v's is plain), back: whether dQ / dK need the back-rotation)."""
    from flash_attn.cute.quant.fp4 import quantize_fp4_rows
    check_shape(q, "q")
    qs = qspec
    refuse_q_blk2d(qs)
    plain = e4m3_mode(qspec=qs) == "plain"
    ts = (q, k) if v is None else (q, k, v)
    # the per-tensor E4M3 scales of the copies: of the rotated Q / K (plain V), or of the plain tensors -- one launch, the
    # amax only (quant.fp8; its rotation is bitwise this module's)
    from flash_attn.cute.quant.fp8 import quantize_fp8_per_tensor
    rot = (False,) * len(ts) if plain else (True, True, False)[:len(ts)]
    s = quantize_fp8_per_tensor(list(ts), apply_hadamard_rotation=rot, output_codes=False,
                                power_of_two_scale=qs.per_tensor_pow2,
                                hadamard_rotation_method=rotation_method(qs))["scales"]
    # one quant.fp4 launch per tensor: the codes of the rotated rows and, from the same read, the E4M3 copy (of the rotated
    # or the plain values); forward and backward read these same codes (scaling-consistency rule)
    oq = quantize_fp4_rows(q, scale_group_size=sf_vec, apply_hadamard_rotation=True, hadamard_sign_seed=SEED,
                           hadamard_rotation_method=rotation_method(qs),
                           also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=s[0:1],
                           e4m3_copy_of_unrotated_values=plain)
    ok = quantize_fp4_rows(k, scale_group_size=sf_vec, apply_hadamard_rotation=True, hadamard_sign_seed=SEED,
                           hadamard_rotation_method=rotation_method(qs),
                           also_make_e4m3_per_tensor_copy=True, e4m3_copy_scale=s[1:2],
                           e4m3_copy_of_unrotated_values=plain)
    return dict(q4=oq["codes"], sfq=oq["scale_factors"], q8=oq["e4m3_copy"], k4=ok["codes"], sfk=ok["scale_factors"],
                k8=ok["e4m3_copy"], s=s, back=not plain)

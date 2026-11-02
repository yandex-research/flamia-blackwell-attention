"""flash_attn.cute.lowprec: the public low-precision entry points.

  * test_rule_*          every invalid-combination rule of the plan raises its exception and message,
                         before any CUDA work (meta tensors: a missed raise would reach the bridge and fail differently);
  * test_pending_*       the combinations whose implementation is still being built raise NotImplementedError naming
                         the missing piece and its branch;
  * test_equivalence     the entry vs a recorded golden of the same rows (skipped without it), on the small
                         cells (S1, A1, A2): O, LSE, dK, dV bitwise; dQ bitwise where the route is deterministic, else
                         its norm. The golden was recorded from the earlier entry points under equivalent settings,
                         where this test proved new == old bitwise;
  * test_plan_reaches_backward   a backward-only option set on the forward reaches the traced backward kernel on another
                         thread, including interleaved calls (the plan is the only configuration);
  * test_no_host_sync / test_graph_capture   the default paths.

GPU tests are grouped by the first word of their id (fp4_*, pv4_*, all4_*, fp8_*) so a run can take one group per
process: pytest tests/cute/test_lowprec_api.py -k "equivalence and fp4_default".
"""
import hashlib
import json
import math
import os
import pathlib
import threading

import pytest
import torch

from flash_attn.cute.lowprec import (PENDING, FP4AttnOptions as O4, FP8AttnOptions as O8, KernelTuning,
                                     flash_attn_fp4, flash_attn_fp8, resolve_fp4, resolve_fp8)
from flash_attn.cute.lowprec import plan as P

BF16 = torch.bfloat16


# ================================================================================================================
# CPU: the invalid-combination rules
# ================================================================================================================

def M(b=1, s=1024, h=4, hk=None, d=128, rg=True):
    """A meta tensor: shape, dtype and requires_grad only; no data, no device work."""
    return torch.empty(b, s, h if hk is None else hk, d, dtype=BF16, device="meta").requires_grad_(rg)


def qkv(b=1, s=1024, h=4, hk=None, d=128, s_k=None, rg=True):
    return M(b, s, h, None, d, rg), M(b, s if s_k is None else s_k, h, hk or h, d, rg), \
        M(b, s if s_k is None else s_k, h, hk or h, d, rg)


F4, F8 = flash_attn_fp4, flash_attn_fp8
ALL = dict(pv_precision="fp4", backward="fp4")
PV4 = dict(pv_precision="fp4")
_I32 = lambda n: torch.empty(n, dtype=torch.int32, device="meta")  # noqa: E731

# (id, entry, kwargs, qkv kwargs, exception, rule, message prefix)
RULES = [
    ("r01", F4, dict(backward="fp4"), {}, ValueError, 1, "backward='fp4' is the query-row backward"),
    ("r02", F4, dict(causal=True, options=O4(q_block_2d=True)), {}, ValueError, 2, "q_block_2d shares one scale"),
    ("r03", F4, dict(qk_hadamard=True, options=O4(q_block_2d=True)), {}, ValueError, 3, "q_block_2d re-quantises"),
    ("r04", F4, dict(qk_hadamard=True), dict(d=256), NotImplementedError, 4, "qk_hadamard is built for head_dim 128"),
    ("r04b", F8, dict(qk_hadamard=True), dict(d=64), NotImplementedError, 4, "qk_hadamard is built for head_dim 128"),
    ("r05", F4, dict(causal=True, qk_hadamard=True, **ALL), {}, NotImplementedError, 5,
     "On the all-4-bit route the Q/K Hadamard"),
    # (r06, r24, r25, rE1: the API offers no MXFP4 format -- test_removed_options)
    ("r07", F4, dict(causal_safety="key_local"), {}, ValueError, 7, "causal_safety='key_local' applies to a 4-bit P.V"),
    ("r08", F4, dict(causal=True, causal_safety="key_local", **PV4), {}, NotImplementedError, 8,
     "Option A (pv_precision='fp4', backward='e4m3') has the diagonal-only key-local V"),
    ("r09", F4, dict(causal_safety="diagonal", **ALL), {}, ValueError, 9, "causal_safety='diagonal' classifies"),
    ("r10", F4, dict(causal=True, options=O4(fp4_backward_dkdv="fp4"), **ALL), {}, NotImplementedError, 10,
     "causal_safety='diagonal' is built for the hybrid backward"),
    ("r11", F4, dict(options=O4(k_channel="cut"), **ALL), {}, ValueError, 11, "k_channel selects how the diagonal"),
    ("r12", F4, dict(causal=True, options=O4(k_channel="cut"), **ALL), dict(h=4, hk=1), NotImplementedError, 12,
     "k_channel='cut' cannot close dQ's K channel"),
    ("r12b", F4, dict(causal=True, options=O4(dq_operand="e4m3"), **ALL), {}, NotImplementedError, 12,
     "k_channel='fold' cannot close dQ's K channel"),
    ("r13", F4, dict(deterministic=True, **ALL), {}, NotImplementedError, 13,
     "The all-4-bit backward reduces dQ with atomics"),
    # (GQA, s_q != s_k and s_k % 256 == 128 take the 2-CTA backward since cta2: head_dim is the remaining shape rule)
    ("r15", F4, dict(options=O4(two_cta="on")), dict(d=64), ValueError, 15,
     "two_cta='on' but the 2-CTA backward cannot serve this call: head_dim 64"),
    ("r15b", F8, dict(options=O8(two_cta="on")), dict(d=64), ValueError, 15,
     "two_cta='on' but the 2-CTA backward cannot serve this call: head_dim 64"),
    ("r16", F4, dict(options=O4(operand_scaling="token_block")), dict(d=256), ValueError, 16,
     "operand_scaling='token_block' cannot serve this call: head_dim 256 (token blocks are built for 64 and 128)"),
    ("r16c", F8, dict(options=O8(operand_scaling="block_32x32"), seqused_k=_I32(1)), {}, ValueError, 16,
     "operand_scaling='block_32x32' cannot serve this call: varlen / seqused_k documents take token-block operands"),
    ("r16d", F4, dict(options=O4(operand_scaling="per_tensor"), **PV4), dict(rg=False), ValueError, 16,
     "operand_scaling='per_tensor' cannot serve this call: option A's E4M3 copies of Q / K / dO are token blocks"),
    ("r16b", F8, dict(options=O8(operand_scaling="token_block", grad_format="e5m2")), {}, ValueError, 16,
     "operand_scaling='token_block' cannot serve this call: E5M2 gradients"),
    ("r17", F4, dict(options=O4(operand_scaling="block_32x32", two_cta="on")), {}, ValueError, 17,
     "block_32x32 operands are 1-CTA only"),
    ("r18", F4, dict(options=O4(operand_scaling="per_tensor", ds_scaling="block_32x32")), dict(d=256), ValueError, 18,
     "block_32x32 dS is built into the single-pass d = 128 kernel"),
    ("r19", F8, dict(options=O8(operand_scaling="per_tensor", ds_scaling="row_local", two_cta="on")), {}, ValueError, 19,
     "ds_scaling='row_local' exists on the 1-CTA kernel only"),
    ("r19b", F8, dict(options=O8(operand_scaling="per_tensor", ds_scaling="tile_128", two_cta="off")), {}, ValueError, 19,
     "ds_scaling='tile_128' exists on the 2-CTA kernel only"),
    ("r20", F8, dict(options=O8(operand_scaling="per_tensor", grad_format="e5m2", two_cta="on")), {}, ValueError, 20,
     "The 2-CTA backward types every FP8 GEMM"),
    ("r21", F4, dict(options=O4(softmax_exp="mitchell")), {}, ValueError, 21, "Mitchell P-hat on an E4M3 P.V"),
    ("r21b", F8, dict(options=O8(softmax_exp="mitchell", two_cta="off")), {}, ValueError, 21,
     "Mitchell P-hat on an E4M3 P.V"),
    # the 8-bit Mitchell P-hat was removed: refused without grad too, on both entries and every 8-bit route
    ("r21c", F4, dict(options=O4(softmax_exp="mitchell")), dict(rg=False), ValueError, 21,
     "Mitchell P-hat on an E4M3 P.V was removed"),
    ("r21d", F8, dict(options=O8(softmax_exp="mitchell")), dict(rg=False), ValueError, 21,
     "Mitchell P-hat on an E4M3 P.V was removed"),
    ("r21e", F8, dict(options=O8(softmax_exp="mitchell"), causal=True), {}, ValueError, 21,
     "Mitchell P-hat on an E4M3 P.V was removed"),
    ("r23", F4, dict(options=O4(p_frame="per_tile")), {}, ValueError, 23, "p_frame='per_tile' is not defined on the "
     "integer-frame all-FP4 step"),
    ("r23b", F4, dict(options=O4(p_frame="per_tile"), **PV4), dict(rg=False), ValueError, 23,
     "p_frame='per_tile' is not defined on the integer-frame all-FP4 step"),
    ("r26", F4, dict(options=O4(seqused_k_group_aligned=True), **PV4), {}, ValueError, 26,
     "seqused_k_group_aligned applies to seqused_k with a 4-bit P.V, which no route takes"),
    ("r27", F4, {}, dict(s=1000), ValueError, 27, "flash_attn_fp4 needs seqlen % 128 == 0"),
    ("r27b", F4, dict(**ALL), dict(d=64), ValueError, 27, "backward='fp4' needs head_dim 128"),
    ("r27c", F4, dict(causal=True, **PV4), dict(d=64), NotImplementedError, 27,
     "pv_precision='fp4' (the 4-bit P.V, option A) is built for head_dim 128"),
    ("r27d", F4, dict(**PV4), dict(d=64), NotImplementedError, 27,
     "pv_precision='fp4' (the 4-bit P.V, option A) is built for head_dim 128"),
    # masks / varlen: the masking / varlen features run forward and backward on pv_precision="fp8"; the 4-bit P.V routes'
    # V scale is key-local for causal only
    ("r28", F4, dict(window_size=(256, 0), **PV4), dict(rg=False), NotImplementedError, 28,
     "The 4-bit P.V routes take no window_size"),
    ("r28b", F4, dict(softcap=30.0, **ALL), {}, NotImplementedError, 28, "The 4-bit P.V routes take no softcap"),
    ("r28c", F8, dict(score_mod=lambda *a: a[0]), {}, ValueError, 28, "score_mod needs score_mod_bwd"),
    ("r28d", F8, dict(mask_mod=lambda *a: True, causal=True), {}, ValueError, 28, "mask_mod with causal / window_size"),
    ("r28e", F4, dict(softcap=30.0, score_mod=lambda *a: a[0], score_mod_bwd=lambda *a: a[0]), {}, ValueError, 28,
     "softcap is a score_mod"),
    # (the only per-tensor dS left is head_dim 256's automatic one, rule 33)
    ("r28g", F4, dict(score_mod=lambda *a: a[0], score_mod_bwd=lambda *a: a[0],
                      options=O4(operand_scaling="per_tensor")), dict(d=256), NotImplementedError, 28,
     "score_mod with a per-tensor dS scale"),
    # (vall4: the 4-bit P.V routes take varlen / seqused_k documents; causal batched seqused_k stays refused -- the
    # first s - seqused_k queries of a row would see no key)
    ("r28h", F4, dict(seqused_k=_I32(1), causal=True, **PV4), dict(rg=False), NotImplementedError, 28,
     "causal batched seqused_k on the 4-bit P.V routes"),
    # (m4: causal s_q > s_k leaves whole query tiles without a key; the 4-bit forward hung there)
    ("r28i", F4, dict(causal=True, **ALL), dict(s=1024, s_k=512), NotImplementedError, 28,
     "The 4-bit P.V routes need a key under every query: causal s_q > s_k (1024 vs 512)"),
    ("r28j", F4, dict(causal=True, **PV4), dict(s=1024, s_k=512, rg=False), NotImplementedError, 28,
     "The 4-bit P.V routes need a key under every query"),
    # token-block fallback shapes: d = 96 runs padded on token blocks (forward and backward); per-tensor operands keep 64 / 128
    ("rE13", F8, dict(options=O8(operand_scaling="per_tensor")), dict(d=96), NotImplementedError, "E13",
     "the 8-bit backward on per-tensor operands serves head_dim 64 and 128"),
    ("rE13c", F8, {}, dict(d=256, rg=False), NotImplementedError, "E13", "flash_attn_fp8 serves head_dim <= 128"),
    ("rE13b", F4, {}, dict(d=192), NotImplementedError, "E13", "flash_attn_fp4 serves head_dim 64, 128 and 256"),
    ("rE12", F8, dict(options=O8(operand_scaling="per_tensor", tuning=KernelTuning(token_block_forward="fold"))), {},
     ValueError, "E12", "KernelTuning.token_block_forward selects the forward of token-block E4M3 operands"),
    ("rE12b", F4, dict(qk_hadamard=False, options=O4(tuning=KernelTuning(qk_hadamard_e4m3="plain"))), {}, ValueError,
     "E12", "KernelTuning.qk_hadamard_e4m3 configures the E4M3 copies of Q / K on the rotated"),
    ("r29", F4, dict(options=O4(dq_operand="e4m3")), {}, ValueError, 29,
     "dq_operand configures the all-4-bit backward"),
    ("r29b", F4, dict(options=O4(fp4_ds_multiplier="global")), {}, ValueError, 29,
     "fp4_ds_multiplier configures the all-4-bit backward"),
    ("r30", F4, dict(options=O4(sr_ds=True), **ALL), {}, ValueError, 30, "On backward='fp4' dS rounding follows"),
    ("r31", F4, dict(causal=True, causal_safety="key_local", options=O4(all_fp4_stabiliser="eden_rht"), **ALL), {},
     ValueError, 31, "eden_rht rotates V jointly"),
    ("r32", F8, dict(q_quantized=(torch.empty(1, device="meta"), 1.0)), {}, ValueError, 32,
     "flash_attn_fp8: q_quantized and k_quantized must be supplied together"),
    ("r32b", F8, dict(q_scale_fused=True), {}, ValueError, 32, "q_scale_fused=True needs q_quantized"),
    ("r33", F8, dict(options=O8(ds_scaling="per_tensor", operand_scaling="per_tensor")), {}, ValueError, 33,
     "ds_scaling='per_tensor' was removed from flash_attn_fp8: sizing a per-tensor dS needs either an extra full pass"),
    ("r33b", F8, dict(options=O8(per_tensor_ds_source="recompute")), {}, ValueError, 33,
     "per_tensor_ds_source='recompute' was removed with the per-tensor dS: the exact max|dS| costs an extra full pass"),
    ("r33c", F8, dict(options=O8(per_tensor_ds_source="forward_bound")), {}, ValueError, 33,
     "per_tensor_ds_source='forward_bound' was removed with the per-tensor dS: the forward's P_max bound is 4.7x "
     "loose (+0.029 val)"),
    ("r33d", F8, dict(options=O8(ds_scaling="per_tensor")), {}, ValueError, 33,
     "ds_scaling='per_tensor' was removed from flash_attn_fp8"),
    ("r33e", F4, dict(options=O4(ds_scaling="per_tensor", operand_scaling="per_tensor")), {}, ValueError, 33,
     "ds_scaling='per_tensor' was removed from flash_attn_fp4: one dS scale per tensor, sized from the forward's P_max "
     "bound"),
    ("r33f", F4, dict(options=O4(ds_scaling="per_tensor")), {}, ValueError, 33,
     "ds_scaling='per_tensor' was removed from flash_attn_fp4"),
    ("r33g", F4, dict(options=O4(ds_scaling="per_tensor"), **ALL), {}, ValueError, 33,
     "ds_scaling='per_tensor' was removed from flash_attn_fp4"),
    ("r34", F4, dict(options=O4(save_bf16_inputs=False), **PV4), {}, ValueError, 34,
     "save_bf16_inputs=False cannot serve this call"),
    ("r36", F4, dict(**PV4), {}, ValueError, 36,
     "pv_precision='fp4' with backward='e4m3' (option A: the 4-bit P-hat.V forward) is an inference-only option"),
    # extra refusals of this implementation (no silent no-op)
    ("rE3", F4, dict(options=O4(q_block_2d=True), **PV4), {}, ValueError, "E3", "q_block_2d is wired on"),
    ("rE7", F8, dict(options=O8(forward_kernel="fork", operand_scaling="per_tensor")), dict(s=1000), ValueError, "E7",
     "forward_kernel='fork' needs"),
    # by design: operand_scaling='auto' never falls back to per-tensor scales silently
    ("r16e", F4, {}, dict(d=256), ValueError, 16,
     "flash_attn_fp4: operand_scaling='auto' resolves to token blocks, which cannot serve this call: head_dim 256"),
    ("r16f", F8, dict(options=O8(grad_format="e5m2")), {}, ValueError, 16, "grad_format='e5m2' is a per-tensor option"),
    ("r16h", F8, dict(options=O8(forward_kernel="upstream")), {}, ValueError, 16,
     "flash_attn_fp8: operand_scaling='auto' resolves to token blocks, which cannot serve this call: "
     "forward_kernel='upstream'"),
    ("r16i", F4, dict(options=O4(p_scaling="block_1x32")), {}, ValueError, 16,
     "flash_attn_fp4: operand_scaling='auto' resolves to token blocks, which cannot serve this call: "
     "p_scaling='block_1x32'"),
    ("r16j", F8, dict(options=O8(operand_scaling="token_local", ds_dq_rowlocal="all"), seqused_k=_I32(1),
                      window_size=(64, 0), causal=True), {},
     ValueError, 16, "operand_scaling='token_local' runs the fold forward, which takes no window"),
    ("r16k", F8, dict(causal=True, options=O8(operand_scaling="token_local", ds_dq_rowlocal="all")),
     dict(s=1024, s_k=512), ValueError, 16, "operand_scaling='token_local' with causal s_q > s_k"),
    ("r16l", F8, dict(options=O8(operand_scaling="token_local", ds_dq_rowlocal="all")), dict(d=96), ValueError, 16,
     "operand_scaling='token_local' is built for head_dim 64 and 128"),
    # the kernels' scale-factor byte index is the x32 TMEM-load mapping: refused at plan time, not by a bare assert in
    # the kernel constructor (flash_bwd_sm100_fp4: `assert ... tmem_ld_rep == 32` had no message)
    ("rE11", F8, dict(options=O8(two_cta="off", tuning=KernelTuning(bwd_fp4_ld_rep=64))), {}, ValueError, "E11",
     "KernelTuning bwd_fp4_ld_rep=64 (the 1-CTA kernel) cannot serve the block-scaled scale factors of "
     "ds_scaling='block_32x32'"),
    ("rE11b", F4, dict(options=O4(operand_scaling="block_32x32", tuning=KernelTuning(bwd_fp4_ld_rep=64))), {},
     ValueError, "E11", "KernelTuning bwd_fp4_ld_rep=64 (the 1-CTA kernel)"),
    ("rE11d", F8, dict(options=O8(tuning=KernelTuning(two_cta8_ld_rep=64))), {}, ValueError, "E11",
     "KernelTuning two_cta8_ld_rep=64 (the 2-CTA kernel)"),
]


@pytest.mark.parametrize("rid,entry,kw,shape,exc,rule,msg", RULES, ids=[r[0] for r in RULES])
def test_rule(rid, entry, kw, shape, exc, rule, msg):
    q, k, v = qkv(**shape)
    with pytest.raises(exc) as ei:
        entry(q, k, v, **kw)
    assert getattr(ei.value, "rule", None) == rule, (ei.value, getattr(ei.value, "rule", None))
    assert str(ei.value).startswith(msg), str(ei.value)


KT = KernelTuning
# (id, entry, kwargs, qkv kwargs, rule, message prefix): calls that used to fail inside a kernel or driver assert,
# each now refused before anything runs, by the rule its message names
ASSERT_REFUSALS = [
    ("dq8_dense", F4, dict(options=O4(dq_operand="e4m3"), **ALL), {}, "E16", "dq_operand='e4m3' (the E4M3 dS"),
    ("mult_row_local_dense", F4, dict(options=O4(fp4_ds_multiplier="row_local"), **ALL), {}, "E15",
     "fp4_ds_multiplier='row_local' sizes each query row's dS"),
    ("mult_global_diag", F4, dict(causal=True, options=O4(fp4_ds_multiplier="global"), **ALL), {}, "E15",
     "fp4_ds_multiplier='global' needs max|dO|"),
    ("mult_global_row_boost_diag", F4, dict(causal=True, options=O4(fp4_ds_multiplier="global_row_boost"), **ALL), {},
     "E15", "fp4_ds_multiplier='global_row_boost' needs max|dO|"),
    ("p_frame_per_tile_optA_dense", F4, dict(options=O4(p_frame="per_tile"), **PV4), dict(rg=False), 23,
     "p_frame='per_tile' is not defined"),
] + [(f"{n}@{r}", F4, dict(causal=True, options=O4(tuning=KT(**{n: v})), **kw), sh, "E18", "KernelTuning")
     for n, v in (("qk_split_pipe", False), ("qk_split_causal", False), ("split_lean", False), ("corr_empty_early", False),
                  ("fp4_q_stage", 1), ("fp4_pv_log2_quant", False))
     for r, kw, sh in (("all4", ALL, {}), ("optA", PV4, dict(rg=False)))] + [
    ("fp4_pv_log2_quant@fp4", F4, dict(options=O4(tuning=KT(fp4_pv_log2_quant=False))), {}, "E19", "KernelTuning"),
    ("fp4_pv_log2_quant@fp8", F8, dict(options=O8(tuning=KT(fp4_pv_log2_quant=False))), {}, "E19", "KernelTuning"),
    ("fp8_pv_p_log2_offset@fp4", F4, dict(options=O4(tuning=KT(fp8_pv_p_log2_offset=2.0))), {}, "E20", "KernelTuning"),
    ("fp8_pv_p_log2_offset@all4", F4, dict(options=O4(tuning=KT(fp8_pv_p_log2_offset=2.0)), **ALL), {}, "E20",
     "KernelTuning"),
    ("fp8_pv_p_log2_offset@optA", F4, dict(options=O4(tuning=KT(fp8_pv_p_log2_offset=2.0)), **PV4), dict(rg=False),
     "E20", "KernelTuning"),
    ("fp8_pv_rescale_threshold@fp8", F8, dict(options=O8(tuning=KT(fp8_pv_rescale_threshold=6.0))), {}, "E20",
     "KernelTuning"),
    ("bwd_fp4_do_stage@fp4_1cta", F4, dict(options=O4(two_cta="off", tuning=KT(bwd_fp4_do_stage=2))), {}, "E11",
     "KernelTuning bwd_fp4_do_stage=2"),
    ("bwd_fp4_do_stage@fp8_1cta", F8, dict(options=O8(two_cta="off", tuning=KT(bwd_fp4_do_stage=2))), {}, "E11",
     "KernelTuning bwd_fp4_do_stage=2"),
    ("two_cta8_q_stage@fp4_2cta", F4, dict(options=O4(two_cta="on", tuning=KT(two_cta8_q_stage=2))), {}, "E11",
     "KernelTuning two_cta8_q_stage=2"),
    ("two_cta8_q_stage@fp8_2cta", F8, dict(options=O8(two_cta="on", tuning=KT(two_cta8_q_stage=2))), {}, "E11",
     "KernelTuning two_cta8_q_stage=2"),
    ("two_cta8_warp_sync@fp4_2cta", F4, dict(options=O4(two_cta="on", tuning=KT(two_cta8_warp_sync=False))), {}, "E11",
     "KernelTuning two_cta8_warp_sync=False"),
    ("two_cta8_warp_sync@fp8_2cta", F8, dict(options=O8(two_cta="on", tuning=KT(two_cta8_warp_sync=False))), {}, "E11",
     "KernelTuning two_cta8_warp_sync=False"),
    # never run: suspected hang (a probe ran > 300 s) -- refused on every route, the 1-CTA ones included
    ("warp_sync_mx_xw_none@fp8_1cta", F8, dict(options=O8(two_cta="off", tuning=KT(two_cta8_warp_sync=False,
                                                                                    two_cta8_mx_xw=""))), {}, "E11",
     "KernelTuning two_cta8_warp_sync=False with two_cta8_mx_xw=''"),
    ("ldred_rowmax@fp4", F4, dict(options=O4(tuning=KT(ldred_rowmax=True))), {}, "E21", "KernelTuning ldred_rowmax=True"),
]


@pytest.mark.parametrize("rid,entry,kw,shape,rule,msg", ASSERT_REFUSALS, ids=[r[0] for r in ASSERT_REFUSALS])
def test_kernel_asserts_are_refusals(rid, entry, kw, shape, rule, msg):
    """The 32 public-reachable kernel / driver asserts of the option matrix are plan refusals
    (raised on meta tensors: before any CUDA work), each message naming its rule."""
    with pytest.raises((ValueError, NotImplementedError)) as ei:
        entry(*qkv(**shape), **kw)
    assert getattr(ei.value, "rule", None) == rule, (rid, ei.value)
    assert str(ei.value).startswith(msg), (rid, str(ei.value))


def test_kernel_asserts_leave_servable_values():
    """The same KernelTuning values resolve where the kernels serve them."""
    pt4, pt8 = dict(operand_scaling="per_tensor"), dict(operand_scaling="per_tensor")
    for entry, O, kw in (
            (resolve_fp4, O4, dict(options=O4(**pt4, tuning=KT(fp4_pv_log2_quant=False)))),
            (resolve_fp8, O8, dict(options=O8(**pt8, tuning=KT(fp4_pv_log2_quant=False)))),
            (resolve_fp4, O4, dict(options=O4(two_cta="off", **pt4, tuning=KT(bwd_fp4_do_stage=2)))),
            (resolve_fp8, O8, dict(options=O8(two_cta="off", **pt8, tuning=KT(bwd_fp4_do_stage=2)))),
            (resolve_fp4, O4, dict(options=O4(tuning=KT(fp8_pv_rescale_threshold=6.0)))),
            (resolve_fp8, O8, dict(options=O8(tuning=KT(fp8_pv_p_log2_offset=2.0)))),
            (resolve_fp8, O8, dict(options=O8(tuning=KT(ldred_rowmax=True)))),
            (resolve_fp4, O4, dict(options=O4(tuning=KT(ldred_rowmax=True)), **ALL)),
            (resolve_fp4, O4, dict(options=O4(tuning=KT(qk_split_causal=False)), **ALL)),   # dense: unread
            (resolve_fp4, O4, dict(causal=True, options=O4(tuning=KT(qk_split_pipe=False)))),
            (resolve_fp8, O8, dict(options=O8(two_cta="off", tuning=KT(two_cta8_warp_sync=False))))):
        entry(*qkv(), **kw)


def test_p_hat_frame():
    """options.p_hat_frame: the row-frame E4M3 P-hat of the token-block MXFP8 P.V forward -- on by
    default on flash_attn_fp8, opt-in on flash_attn_fp4 -- reaches HostSpec; an explicit value where the forward has no
    such choice is refused (E14)."""
    q, k, v = qkv()
    assert resolve_fp8(q, k, v).specs.host.fwd8_row_p is True
    assert resolve_fp4(q, k, v).specs.host.fwd4_row_p is False
    assert resolve_fp8(q, k, v, options=O8(p_hat_frame="group")).specs.host.fwd8_row_p is False
    p = resolve_fp4(q, k, v, options=O4(p_hat_frame="row"))
    assert p.specs.host.fwd4_row_p is True and ("p_hat_frame", "row", "explicit") in p.resolution_log
    assert resolve_fp4(q, k, v, options=O4(p_hat_frame="group", operand_scaling="block_32x32")).specs.host.fwd4_row_p \
        is False
    for entry, O, kw in ((resolve_fp4, O4, dict(operand_scaling="per_tensor")),
                         (resolve_fp8, O8, dict(operand_scaling="per_tensor")),
                         (resolve_fp8, O8, dict(operand_scaling="token_local", ds_dq_rowlocal="all")),
                         (resolve_fp4, O4, dict(tuning=KT(token_block_forward="fold")))):
        for val in ("row", "group"):
            with pytest.raises(ValueError) as ei:
                entry(q, k, v, options=O(p_hat_frame=val, **kw))
            assert ei.value.rule == "E14", (kw, val)
        assert entry(q, k, v, options=O(**kw)).route   # auto: no choice there, no refusal
    with pytest.raises(ValueError):
        resolve_fp4(*qkv(rg=False), options=O4(p_hat_frame="row"), **PV4)
    with pytest.raises(ValueError, match="p_hat_frame must be one of"):
        O4(p_hat_frame="tile")


def test_rule_e11_leaves_servable_tuning():
    """E11 refuses only what the resolved kernel cannot compile: x64 loads with a per-tensor dS and row P^ (head_dim
    256's automatic one: no explicit per-tensor dS is left, rule 33), and the 1-CTA field on a call that resolves to the
    2-CTA kernel, still resolve."""
    q, k, v = qkv()
    p = resolve_fp4(*qkv(d=256), options=O4(operand_scaling="per_tensor", two_cta="off",
                                            tuning=KernelTuning(bwd_fp4_ld_rep=64)))
    assert p.bwd.kernel == "1cta" and p.bwd.ds_scaling == "per_tensor"
    p = resolve_fp8(q, k, v, options=O8(tuning=KernelTuning(bwd_fp4_ld_rep=64)))
    assert p.bwd.kernel == "2cta"


def test_rule28_varlen():
    """varlen resolves on the E4M3 P.V routes (token-block documents, any lengths) and (vall4) on the 4-bit P.V routes
    (the all-4-bit training route, option A inference); window / mods stay refused on the 4-bit routes."""
    q = torch.empty(2000, 4, 128, dtype=BF16, device="meta", requires_grad=True)
    vl = dict(cu_seqlens_q=_I32(3), cu_seqlens_k=_I32(3), max_seqlen_q=1024, max_seqlen_k=1024)
    for res in (resolve_fp4, resolve_fp8):
        p = res(q, q, q, causal=True, **vl)
        assert p.fwd.varlen and p.fwd.operand_scaling == "token_block" and p.bwd.kernel == "1cta", p.log()
    qi = q.detach()
    for causal in (True, False):
        assert resolve_fp4(q, q, q, causal=causal, pv_precision="fp4", backward="fp4", **vl).route == "fp4.all_fp4"
        assert resolve_fp4(qi, qi, qi, causal=causal, pv_precision="fp4", **vl).route == "fp4.fp4_pv"
    with pytest.raises(NotImplementedError) as ei:
        flash_attn_fp4(q, q, q, pv_precision="fp4", backward="fp4", causal=True, window_size=(256, 0), **vl)
    assert ei.value.rule == 28 and "window_size" in str(ei.value)


def test_varlen_coverage_resolves():
    """varlen everywhere (vcov): return_lse and per-tensor operands on the repack fallback / batched seqused_k,
    token-local operands on documents, q_block_2d on dense documents, the 2-CTA forward ('on') on documents and the
    2-CTA backward ('on') deterministic / total_q != total_k on aligned documents; 'auto' keeps documents 1-CTA."""
    q = torch.empty(2048, 4, 128, dtype=BF16, device="meta", requires_grad=True)
    q2 = torch.empty(2560, 4, 128, dtype=BF16, device="meta", requires_grad=True)
    rp = dict(cu_seqlens_q=_I32(3), cu_seqlens_k=_I32(3), max_seqlen_q=1024, max_seqlen_k=1024)
    al = dict(rp, seqused_q=_I32(2), seqused_k=_I32(2))
    su = dict(seqused_k=_I32(1))
    for res, O in ((resolve_fp4, O4), (resolve_fp8, O8)):
        for docs in (rp, al):
            assert res(q, q, q, causal=True, return_lse=True, **docs).fwd.return_lse
            p = res(q, q, q, causal=True, options=O(operand_scaling="per_tensor"), **docs)
            assert p.fwd.operand_scaling == "per_tensor" and p.bwd.kernel == "1cta"
            p = res(q, q, q, causal=True, options=O(operand_scaling="token_local", ds_dq_rowlocal="all"), **docs)
            assert p.fwd.operand_scaling == "token_local"
            assert res(q, q, q, causal=True, options=O(forward_two_cta="on"), **docs).fwd.two_cta
            assert not res(q, q, q, causal=False, **docs).fwd.two_cta     # auto: documents stay 1-CTA
        assert res(*qkv(), return_lse=True, **su).fwd.return_lse
        assert res(*qkv(), options=O(operand_scaling="per_tensor"), **su).fwd.operand_scaling == "per_tensor"
        p = res(q, q, q, causal=True, deterministic=True, options=O(two_cta="on"), **al)
        assert p.bwd.kernel == "2cta" and p.bwd.deterministic
        p = res(q2, q, q, causal=True, options=O(two_cta="on"), **dict(al, max_seqlen_q=1280))
        assert p.bwd.kernel == "2cta"
        assert res(q, q, q, causal=True, **al).bwd.kernel == "1cta"     # auto: aligned documents stay 1-CTA
    for docs in (rp, al):
        assert resolve_fp4(q, q, q, causal=False, options=O4(q_block_2d=True), **docs).fwd.q_block_2d
        with pytest.raises(ValueError, match="q_block_2d shares one scale"):
            resolve_fp4(q, q, q, causal=True, options=O4(q_block_2d=True), **docs)


def test_masked_features_resolve():
    """masks / varlen through the API: every feature resolves with grad on the E4M3 P.V routes (no PENDING row is left
    for a merged feature); the features stay out of the 4-bit P.V routes."""
    sm = lambda *a: a[0]  # noqa: E731
    feats = (dict(window_size=(256, 0), causal=True), dict(softcap=30.0), dict(score_mod=sm, score_mod_bwd=sm),
             dict(mask_mod=lambda *a: True), dict(seqused_k=_I32(1)))
    for res in (resolve_fp4, resolve_fp8):
        for kw in feats:
            p = res(*qkv(), **kw)
            assert p.bwd.kernel in ("1cta", "2cta") and p.pending == (), (kw, p.log())
    for kw in ({}, dict(qk_hadamard=True)):
        assert resolve_fp4(*qkv(), window_size=(256, 0), **kw).fwd.window == (256, 0)
    p = resolve_fp4(*qkv(rg=False), causal=True, **PV4)   # option A causal (inference): the diagonal-only key-local V
    assert p.fwd.causal_safety == "diagonal" and p.pending == ()


def test_rule35_tuning():
    for kw, msg in ((dict(fwd_mask_loops=5), "KernelTuning.fwd_mask_loops must be one of"),
                    (dict(two_cta8_regs="1x2"), "KernelTuning.two_cta8_regs must match"),
                    (dict(two_cta8_regs="96x56"), "KernelTuning.two_cta8_regs must match"),
                    (dict(fwd_max_tmem=1), "KernelTuning.fwd_max_tmem must be a bool"),
                    (dict(fp8_pv_p_log2_offset=4.0, fp8_pv_rescale_threshold=8.0), "KernelTuning.fp8_pv_p_log2_offset")):
        with pytest.raises(ValueError) as ei:
            KernelTuning(**kw)
        assert str(ei.value).startswith(msg), str(ei.value)
    assert KernelTuning(fwd_mask_loops=0).non_default() == {"fwd_mask_loops": 0}


def test_option_values_validated():
    with pytest.raises(ValueError, match="FP4AttnOptions.ds_scaling must be one of"):
        O4(ds_scaling="row_local")
    with pytest.raises(ValueError, match="FP8AttnOptions.grad_format must be one of"):
        O8(grad_format="e3m4")
    with pytest.raises(ValueError, match="pv_precision must be one of"):
        flash_attn_fp4(*qkv(), pv_precision="fp6")


def test_scaling_consistency_rule():
    """The one check: a plan whose forward and backward scale a shared operand differently is refused."""
    log = P._Log()
    with pytest.raises(ValueError) as ei:
        P._consistency("fp8", [("V", "e4m3 per_tensor", "e4m3 token_block")], True, log)
    assert ei.value.rule == "consistency" and str(ei.value).startswith("scaling-consistency rule")
    # every route resolves consistent pairs; option A shares its NVFP4 V (no exemption is left)
    for kw in ({}, dict(causal=True), dict(**ALL, causal=True), dict(**ALL)):
        p = resolve_fp4(*qkv(), **kw)
        assert all(f == b for _, f, b in p.consistency) and p.pending == (), kw
    # option A (pv_precision="fp4", backward="e4m3") is inference only: no backward, nothing shared (rule 36 refuses grad)
    p = resolve_fp4(*qkv(rg=False), **PV4)
    assert p.pending == () and dict((n, f) for n, f, _ in p.consistency)["V"] == "nvfp4 16x16"
    assert p.fwd.operand_scaling == "token_block"


def test_removed_options():
    """The MXFP4 formats are not part of the API -- the options fields are gone, so a
    value is refused when the options object is built; every 4-bit operand is NVFP4."""
    for kw in (dict(qk_format="mxfp4"), dict(pv_format="mxfp4"), dict(qk_format="nvfp4")):
        with pytest.raises(TypeError):
            O4(**kw)
    p = resolve_fp4(*qkv(rg=False), causal=True, **PV4)
    assert (p.fwd.qk_format, p.fwd.pv_operand) == ("nvfp4", "nvfp4")
# KernelTuning fields that were deleted: the dead
# disable_2cta_fwd, the unwireable clc_scheduler / num_sms (formerly refused as PENDING tuning_unbridged), the readerless
# bwd_fp4_rcp_approx / qk_hadamard_impl / qk_hadamard_lanes, and bwd_fp4_compute_wgs / bwd_fp4_regs4 (no value but the
# default builds). A value for any of them is refused when the object is built.
DELETED_TUNING = dict(disable_2cta_fwd=True, clc_scheduler=True, num_sms=100, bwd_fp4_rcp_approx=False,
                      qk_hadamard_impl="mma", qk_hadamard_lanes="8", bwd_fp4_compute_wgs=1, bwd_fp4_regs4="88,64,64")


@pytest.mark.parametrize("name", sorted(DELETED_TUNING))
def test_deleted_tuning_fields_refused(name):
    with pytest.raises(TypeError):
        KernelTuning(**{name: DELETED_TUNING[name]})


def test_pending_rows_name_their_piece():
    for key, row in PENDING.items():
        assert row["feature"] and row["missing"] and row["branch"], key


def test_defaults_resolved():
    """The documented defaults, resolved (the table of the report)."""
    p = resolve_fp4(*qkv(b=4, s=4096, h=16), causal=True)
    assert (p.route, p.fwd.operand_scaling, p.bwd.ds_scaling, p.bwd.p_scaling, p.fwd.softmax_exp, p.fwd.causal_safety,
            p.bwd.kernel) == ("fp4.e4m3_pv", "token_block", "block_32x32", "row", "exp2", "none", "2cta")
    # token-block operands take the 2-CTA backward wherever eligible; per-tensor operands keep the threshold
    assert resolve_fp4(*qkv(b=4, s=4096, h=16), causal=True,
                       options=O4(operand_scaling="per_tensor")).bwd.kernel == "1cta"
    assert resolve_fp4(*qkv(b=1, s=32768, h=16), causal=True).bwd.kernel == "2cta"
    assert resolve_fp4(*qkv(b=1, s=8192, h=16)).bwd.kernel == "2cta"
    # d = 256 (the three-pass kernels): per-tensor operands and dS, by explicit opt-in only
    pt = O4(operand_scaling="per_tensor")
    assert resolve_fp4(*qkv(d=256), options=pt).bwd.ds_scaling == "per_tensor"
    assert resolve_fp4(*qkv(d=256), options=pt).fwd.operand_scaling == "per_tensor"
    p = resolve_fp4(*qkv(), causal=True, **ALL)
    assert (p.fwd.causal_safety, p.bwd.k_channel, p.bwd.fp4_ds_multiplier, p.fwd.softmax_exp) == \
        ("diagonal", "fold", "row_local", "mitchell")
    p = resolve_fp4(*qkv(), **ALL)
    assert (p.fwd.causal_safety, p.bwd.fp4_ds_multiplier) == ("none", "global_row_boost")
    p = resolve_fp4(*qkv(rg=False), **PV4)
    assert (p.fwd.softmax_exp, p.fwd.p_frame) == ("mitchell", "two_level")
    p = resolve_fp8(*qkv(), causal=True)
    assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.ds_scaling, p.bwd.kernel, p.bwd.grad_format) == \
        ("fork_fp8", "token_block", "block_32x32", "2cta", "e4m3")
    # ragged lengths: the batch rows as documents, token blocks, the fork forward (upstream's per-tensor
    # forward only by explicit opt-in)
    p = resolve_fp8(*qkv(s=1000), causal=True)
    assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.kernel, p.fwd.ragged_documents) == \
        ("fork_fp8", "token_block", "1cta", True)
    p = resolve_fp8(*qkv(s=1000), causal=True, options=O8(operand_scaling="per_tensor"))
    assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.kernel, p.fwd.ragged_documents) == \
        ("upstream_fp8", "per_tensor", "1cta", False)
    # s_q != s_k: token blocks (the factor table indexes Q / dO rows by query block, K / V by key block)
    for entry in (resolve_fp4, resolve_fp8):
        for sq, sk in ((1024, 2048), (2048, 1024)):
            assert entry(*qkv(s=sq, s_k=sk), causal=True).fwd.operand_scaling == "token_block", (entry, sq, sk)
    p = resolve_fp8(*qkv(h=4, hk=1))
    assert (p.fwd.operand_scaling, p.bwd.kernel) == ("token_block", "1cta")


def test_resolve_d64():
    """head_dim 64 runs natively: the fork forward, token-block operands, the single-pass 1-CTA backward."""
    p = resolve_fp8(*qkv(d=64), causal=True)
    assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.kernel, p.bwd.head_dim_path) == \
        ("fork_fp8", "token_block", "1cta", "single_pass")
    p = resolve_fp4(*qkv(d=64), causal=True)
    assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.kernel, p.bwd.head_dim_path) == \
        ("fork_fp4", "token_block", "1cta", "single_pass")
    # d = 96 on the 8-bit entry: the token-block kernels run it padded to 128, forward AND backward (the
    # quantisers stage the zero columns); per-tensor operands keep upstream's forward and no backward (rule E13)
    for d, dp in ((96, 128), (32, 64), (48, 64), (80, 128), (112, 128), (16, 64)):
        p = resolve_fp8(*qkv(d=d), causal=True)
        assert (p.fwd.kernel, p.fwd.operand_scaling, p.bwd.kernel, p.bwd.head_dim_path) == \
            ("fork_fp8", "token_block", "1cta", f"zero_pad_{dp}"), d
        assert ("head_dim_padding", str(dp)) in [(f, v) for f, v, _ in p.resolution_log], d
    p = resolve_fp8(*qkv(d=96, rg=False), causal=True)
    assert (p.fwd.kernel, p.fwd.operand_scaling) == ("fork_fp8", "token_block")
    with pytest.raises(NotImplementedError) as ei:
        resolve_fp8(*qkv(d=96), causal=True, options=O8(operand_scaling="per_tensor"))
    assert ei.value.rule == "E13"
    p = resolve_fp8(*qkv(d=96, rg=False), causal=True, options=O8(operand_scaling="per_tensor"))
    assert (p.fwd.kernel, p.fwd.operand_scaling) == ("upstream_fp8", "per_tensor")


def _settings(p):
    """The settings a plan states for its kernels and drivers (the fields of its specs that differ from the specs'
    defaults), as {spec.field: value} -- what the former bridge expressed as tuning-map knobs."""
    out = {}
    for name in ("fwd", "quant", "host", "bwd1", "bwd2", "bwdq"):
        s = getattr(p.specs, name)
        if s is not None:
            out.update({f"{name}.{k}": v for k, v in s.non_default().items()})
    return out


def test_plan_specs_per_route():
    """The settings the plan states, per default route (the report's list; formerly the bridge's knobs). qk_hadamard is
    on by default (2026-10-01): QuantSpec.qk_hadamard on the E4M3-operand routes; the all-4-bit route falls back to off."""
    kn = _settings
    H = {"quant.qk_hadamard": True}
    TB4 = {"quant.operand_scaling_fp4": "blk32x128"}   # the E4M3 copies' token blocks (OP_SCALING_DEFAULT on the kernels)
    assert kn(resolve_fp4(*qkv(), causal=True)) == {**TB4, "host.bwd_two_cta": "on", **H}
    assert kn(resolve_fp4(*qkv(rg=False), causal=True)) == {**TB4, **H}
    assert kn(resolve_fp8(*qkv(), causal=True)) == {
        "quant.operand_scaling_fp8": "blk32x128", "host.bwd_op_mode": "fp8", "host.bwd_two_cta": "on",
        "bwd1.fp8": True, **H}
    assert kn(resolve_fp4(*qkv(), causal=True, **ALL)) == {
        "fwd.fp4all": True, "quant.vtok_diag": True}   # (the shipped all-4-bit configuration is the specs' defaults)
    assert kn(resolve_fp4(*qkv(), options=O4(tuning=KernelTuning(fwd_mask_loops=0, q_dbuf=True)))) == {
        **TB4, "host.bwd_two_cta": "on", "fwd.mask_loops": 0, "fwd.q_dbuf": True, **H}
    # the same calls with qk_hadamard=False state exactly what they stated before the default flip
    assert kn(resolve_fp4(*qkv(), causal=True, qk_hadamard=False)) == {**TB4, "host.bwd_two_cta": "on"}
    assert kn(resolve_fp8(*qkv(), causal=True, qk_hadamard=False)) == {
        "quant.operand_scaling_fp8": "blk32x128", "host.bwd_op_mode": "fp8", "host.bwd_two_cta": "on", "bwd1.fp8": True}
    # operand_scaling="per_tensor" reaches the kernels (left unset, the drivers fell back to token blocks at d = 128)
    p = resolve_fp8(*qkv(), causal=True, options=O8(operand_scaling="per_tensor"))
    assert (kn(p)["quant.operand_scaling_fp8"], p.specs.bwd1.op_scaling, p.specs.bwd2.op_scaling) == ("per_tensor",) * 3
    p = resolve_fp4(*qkv(), causal=True, options=O4(operand_scaling="per_tensor"))
    assert (kn(p)["quant.operand_scaling_fp4"], p.specs.bwd1.op_scaling, p.specs.bwd2.op_scaling) == ("per_tensor",) * 3
    # every spec the plan builds is a legal spec (the strict parsers would refuse anything else), KernelTuning copied in
    p = resolve_fp4(*qkv(), options=O4(tuning=KernelTuning(bwd_fp4_dq_post="torch", two_cta8_lpt_l2=True)))
    assert p.specs.host.dq_post == "torch" and p.specs.bwd2.lpt_l2 is True
    # option A's forward carries its reproduce-forward-p settings and the all-FP4 step; its copies are token blocks
    p = resolve_fp4(*qkv(rg=False), causal=True, **PV4)
    assert (p.specs.fwd.fp4all, p.specs.fwd.fp4all_rowsum, p.specs.fwd.fp4all_sf_decode, p.specs.fwd.fp4all_exp2,
            p.specs.quant.operand_scaling_fp4) == (True, "codes", True, False, "blk32x128")


def test_qk_hadamard_resolves_on_block_scaled_defaults():
    """qk_hadamard keeps the block-scaled default operands (no fallback to per_tensor) and reaches the route bodies
    through QuantSpec.qk_hadamard; every other setting is the call's without the rotation."""
    kn = _settings
    for entry, kw in ((resolve_fp4, dict(causal=True)), (resolve_fp4, {}), (resolve_fp8, dict(causal=True)),
                      (resolve_fp4, dict(options=O4(operand_scaling="block_32x32"))),
                      (resolve_fp8, dict(options=O8(operand_scaling="block_32x32"))),
                      (resolve_fp4, dict(options=O4(operand_scaling="per_tensor"))), (resolve_fp4, dict(**PV4))):
        sh = dict(rg=False) if kw.get("pv_precision") == "fp4" else {}  # option A is inference-only
        p0, p1 = entry(*qkv(**sh), qk_hadamard=False, **kw), entry(*qkv(**sh), qk_hadamard=True, **kw)
        pd = entry(*qkv(**sh), **kw)  # the default (None): on wherever it runs -- the same plan as True here
        assert p1.fwd.qk_hadamard and p1.fwd.operand_scaling == p0.fwd.operand_scaling, (kw, p1.fwd.operand_scaling)
        assert kn(p1) == dict(kn(p0), **{"quant.qk_hadamard": True}), kw
        assert pd.fwd.qk_hadamard and kn(pd) == kn(p1), kw
        assert any(f == "qk_hadamard" for f, _, _ in p1.resolution_log)
        assert ("qk_hadamard", "off", "explicit") in p0.resolution_log
    assert resolve_fp4(*qkv(), causal=True, qk_hadamard=True).fwd.operand_scaling == "token_block"
    # ragged lengths run as documents: the rotation runs there; per-tensor operands keep the refusal
    assert resolve_fp8(*qkv(s=1000), qk_hadamard=True).fwd.qk_hadamard
    with pytest.raises(NotImplementedError, match="s % 128 == 0"):
        resolve_fp8(*qkv(s=1000), qk_hadamard=True, options=O8(operand_scaling="per_tensor"))


# the plan's per_tensor resolutions (automatic and explicit) vs what the driver resolves under the plan's config
# (`bridge-auto-per-tensor-dropped`: s_q != s_k, the upstream 8-bit forward and the 8-bit Mitchell forward resolved
# per_tensor in the plan and ran token blocks)
PER_TENSOR_CASES = [
    # (fp8_sq_ne_sk / fp4_sq_ne_sk: token blocks now; the per-tensor-only options below by explicit opt-in)
    ("fp8_upstream", resolve_fp8, dict(options=O8(forward_kernel="upstream", operand_scaling="per_tensor")), {}),
    # (fp8_mitchell: gone with del-mitch8 -- softmax_exp='mitchell' on an E4M3 P.V is refused, rule 21)
    ("fp8_e5m2", resolve_fp8, dict(options=O8(grad_format="e5m2", operand_scaling="per_tensor")), {}),
    # (fp8_ds_per_tensor: the 8-bit per-tensor dS was removed, rule 33)
    ("fp4_p_block", resolve_fp4, dict(options=O4(p_scaling="block_1x32", operand_scaling="per_tensor")), {}),
    ("fp4_d256", resolve_fp4, dict(options=O4(operand_scaling="per_tensor")), dict(d=256)),
    ("fp8_prequant", resolve_fp8, dict(q_quantized=(torch.empty(1, device="meta"), 1.0),
                                       k_quantized=(torch.empty(1, device="meta"), 1.0)), {}),
    ("fp8_explicit", resolve_fp8, dict(options=O8(operand_scaling="per_tensor")), {}),
    ("fp4_explicit", resolve_fp4, dict(options=O4(operand_scaling="per_tensor")), {}),
]


@pytest.mark.parametrize("rid,res,kw,shape", PER_TENSOR_CASES, ids=[c[0] for c in PER_TENSOR_CASES])
def test_plan_per_tensor_reaches_the_driver(rid, res, kw, shape):
    """Every per_tensor resolution of the plan is what the drivers and kernels read: the plan's specs state it (the
    drivers no longer re-resolve the operand scaling), and the drivers' scaling-consistency check accepts the per-tensor
    codes a forward hands over."""
    import flash_attn.cute.interface_bwd_fp4 as ib
    q, k, v = qkv(**shape)
    p = res(q, k, v, **kw)
    assert p.fwd.operand_scaling == "per_tensor", (rid, p.log())
    fp8 = res is resolve_fp8
    qs = p.specs.quant
    got = qs.operand_scaling_fp8 if fp8 else qs.operand_scaling_fp4
    assert got == "per_tensor" and p.specs.bwd1.op_scaling == "per_tensor", (rid, got)
    assert p.specs.bwd2 is None or p.specs.bwd2.op_scaling == "per_tensor", rid
    ib.check_codes_scaling(fp8, dict(q8=None, k8=None, v8=None), None, qs, rid)


def test_qk_hadamard_default_falls_back_with_a_log():
    """The default (None) never refuses: where the rotation cannot run the call runs without it and the resolution log
    names why; an explicit True still refuses there."""
    cases = [(resolve_fp4, dict(d=64), {}, "head_dim 64"), (resolve_fp8, dict(d=64), {}, "head_dim 64"),
             (resolve_fp8, dict(s=1000), dict(options=O8(operand_scaling="per_tensor")), "s % 128 != 0"),
             (resolve_fp4, {}, dict(options=O4(q_block_2d=True)), "q_block_2d"),
             (resolve_fp8, {}, dict(options=O8(operand_scaling="token_local", ds_dq_rowlocal="all")), "token_local"),
             (resolve_fp4, {}, dict(causal=True, **ALL), "all-4-bit")]
    for entry, shape, kw, why in cases:
        p = entry(*qkv(**shape), **kw)
        assert not p.fwd.qk_hadamard, (shape, kw)
        rows = [r for r in p.resolution_log if r[0] == "qk_hadamard"]
        assert len(rows) == 1 and rows[0][1] == "off" and why in rows[0][2], (shape, kw, rows)
        assert not p.specs.quant.qk_hadamard and not p.specs.quant.vtok_hadqk, (shape, kw)
    for entry, shape, kw, _ in cases:  # explicit True keeps the refusals
        with pytest.raises((NotImplementedError, ValueError)):
            entry(*qkv(**shape), qk_hadamard=True, **kw)
    with pytest.raises(ValueError, match="qk_hadamard must be None"):
        resolve_fp4(*qkv(), qk_hadamard=1)
    for entry in (resolve_fp4, resolve_fp8):  # window: rotated, forward and backward
        assert entry(*qkv(), window_size=(256, 0)).fwd.qk_hadamard


# upstream's files that keep upstream's own environment reads on the BF16 path (by design): FA_LOG_LEVEL, FLASH_ATTENTION_ARCH / _NUM_SMS, the compile cache, CUTE_DSL_*, FA_CLC / FA_DISABLE_2CTA
UPSTREAM_ENV_FILES = {"fa_logging.py", "cute_dsl_utils.py", "cute_dsl_ptxas.py", "cache_utils.py", "utils.py",
                      "interface.py"}
LINT = ("lowprec_config", "lpcfg", "os.environ", "getenv", "threading.local")


def test_no_configuration_state():
    """No fork-owned library file names the deleted configuration module, the
    environment or thread-local state -- code, docstrings and comments alike. The plan (its specs) is the only
    configuration of the low-precision path. Benchmarks are not library."""
    root = pathlib.Path(P.__file__).parents[1]   # flash_attn/cute
    bad = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root)
        if rel.parts[0] in ("debug", "benchmarks") or (len(rel.parts) == 1 and f.name in UPSTREAM_ENV_FILES):
            continue
        src = f.read_text()
        bad += [(str(rel), t) for t in LINT if t in src]
    assert not bad, bad
    assert not (root / "lowprec_config.py").exists()


# ================================================================================================================
# GPU
# ================================================================================================================

gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="SM100 kernels")

CELLS = {"S1": dict(b=1, s=1024, h=4, causal=True), "A1": dict(b=4, s=4096, h=16, causal=True),
         "A2": dict(b=4, s=4096, h=16, causal=False)}


def _inputs(cell):
    c = CELLS[cell]
    g = torch.Generator().manual_seed(0)
    q, k, v, do = (torch.randn(c["b"], c["s"], c["h"], 128, generator=g).to(BF16) for _ in range(4))
    return [x.cuda() for x in (q, k, v, do)], c["causal"]


def _sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _reset_sr():
    import flash_attn.cute.interface_bwd_fp4 as ib
    import flash_attn.cute.interface_bwd_fp4q as iq
    ib.reset_ds_sr_counter()   # the dk8 sr_ds seed: a device counter where live (and the host counter elsewhere)
    iq.reset_sr_counter()   # the all-4-bit SR seed: a device counter (one fixed seed sequence per run)
    import flash_attn.cute.pv4_route as pr  # option A's per-call dO4 SR seed has its own counter
    pr._SR_CALL[0] = 0


def _run_new(entry, ins, causal, kw):
    q, k, v, do = ins
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    _reset_sr()
    o, lse = entry(qr, kr, vr, causal=causal, return_lse=True, **kw)
    o.backward(do)
    return [o, lse, qr.grad, kr.grad, vr.grad]


# a recorded golden capture (a gbit.jsonl named by LOWPREC_GOLDEN_GBIT; test_equivalence skips without it): its
# equiv/<rid>/<cell> rows are this table's new-entry side, recorded where test_equivalence proved them bitwise the
# entry points they replaced
GOLDEN = pathlib.Path(os.environ.get("LOWPREC_GOLDEN_GBIT", "/nonexistent/gbit.jsonl"))


def _golden(rid, cell):
    if not GOLDEN.exists():
        pytest.skip(f"no golden capture (set LOWPREC_GOLDEN_GBIT; got {GOLDEN})")
    if not hasattr(_golden, "rows"):
        _golden.rows = {json.loads(ln)["id"]: json.loads(ln) for ln in open(GOLDEN)}
    rec = _golden.rows.get(f"equiv/{rid}/{cell}")
    assert rec is not None and rec["status"] == "ok", (rid, cell, rec and rec.get("status"))
    return rec


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()  # float32 dot products round identical tensors to 1 +- 1e-7
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-30)).item()


# (id, entry, kwargs, cells, dQ bitwise). The reference side of each row is the golden record
# equiv/<id>/<cell>, where the row was proven bitwise against the legacy entry under its equivalent settings.
EQUIV = [
    ("fp4_default", F4, {}, ("S1", "A1", "A2"), False),
    ("fp4_per_tensor", F4, dict(options=O4(operand_scaling="per_tensor")), ("S1", "A1", "A2"), False),
    ("fp4_block32", F4, dict(options=O4(operand_scaling="block_32x32")), ("S1", "A2"), False),
    # qk_hadamard: the API option, on the block-scaled defaults
    ("fp4_had", F4, dict(qk_hadamard=True), ("S1", "A1"), False),
    ("fp4_had_per_tensor", F4, dict(qk_hadamard=True, options=O4(operand_scaling="per_tensor")), ("S1",), False),
    # (no option A row: option A is inference-only, and this table runs the backward)
    ("fp4_deterministic", F4, dict(deterministic=True), ("A1", "A2"), True),
    ("fp4_scale", F4, dict(softmax_scale=0.07), ("S1",), False),
    # (fp4_ds_per_tensor: the explicit 4-bit per-tensor dS was removed, rule 33)
    ("fp4_p_block", F4, dict(options=O4(p_scaling="block_1x32", operand_scaling="per_tensor")), ("S1",), False),
    ("fp4_sr_ds", F4, dict(options=O4(sr_ds=True)), ("S1",), False),
    ("fp4_two_cta_on", F4, dict(options=O4(two_cta="on")), ("A1", "A2"), False),
    ("fp4_lean", F4, dict(options=O4(operand_scaling="per_tensor", save_bf16_inputs=False)), ("S1",), False),
    ("all4_dense", F4, dict(**ALL), ("A2",), False),
    ("all4_diagonal", F4, dict(**ALL), ("S1", "A1"), False),
    ("all4_key_local", F4, dict(causal_safety="key_local", **ALL), ("S1",), False),
    ("all4_off", F4, dict(causal_safety="off", **ALL), ("S1",), False),
    ("all4_dq8", F4, dict(options=O4(dq_operand="e4m3"), **ALL), ("A2",), False),
    ("fp8_default", F8, {}, ("S1", "A1", "A2"), False),
    ("fp8_had", F8, dict(qk_hadamard=True), ("S1", "A1"), False),
    ("fp8_per_tensor", F8, dict(options=O8(operand_scaling="per_tensor")), ("S1", "A1"), False),
    # (fp8_ds_per_tensor: the 8-bit per-tensor dS was removed, rule 33)
    ("fp8_e5m2", F8, dict(options=O8(grad_format="e5m2", operand_scaling="per_tensor")), ("S1",), False),
    ("fp8_two_cta_off", F8, dict(options=O8(two_cta="off")), ("A1",), False),
    ("fp8_deterministic", F8, dict(deterministic=True), ("A1",), True),
    # (`bridge-auto-per-tensor-dropped`: the upstream forward is per-tensor by construction)
    ("fp8_upstream", F8, dict(options=O8(forward_kernel="upstream", operand_scaling="per_tensor")), ("S1",), False),
]
# qk_hadamard defaults ON: the rows above that meant "today's call without the rotation"
# keep that meaning with an explicit qk_hadamard=False (rows with qk_hadamard given, and the all-4-bit rows -- where the
# default falls back to off -- run as written); the default itself is checked by the *_default_on rows
EQUIV = [r if ("qk_hadamard" in r[2] or r[0].startswith("all4")) else (r[0], r[1], dict(r[2], qk_hadamard=False), *r[3:])
         for r in EQUIV] + [
    ("fp4_default_on", F4, {}, ("S1", "A1", "A2"), False),
    ("fp4_deterministic_on", F4, dict(deterministic=True), ("A1",), True),
    ("fp8_default_on", F8, {}, ("S1", "A1", "A2"), False),
    ("fp8_deterministic_on", F8, dict(deterministic=True), ("A1",), True),
]
EQUIV_PARAMS = [(r[0] + "-" + c, r[1], r[2], c, r[4]) for r in EQUIV for c in r[3]]
# A defect of the earlier tree: its all-4-bit entry with the E4M3 dQ operand failed the same
# way on a dense call ("sA must be provided when a_src is not TMEM" in the query-row kernel's build).
# (all4_dq8: the old entry failed identically inside the kernel build; the plan refuses it, rule E16 --
# test_equivalence checks the refusal against the golden record of the failure)


@gpu
@pytest.mark.parametrize("rid,entry,kw,cell,dq_bitwise", EQUIV_PARAMS,
                         ids=[p.values[0] if hasattr(p, "values") else p[0] for p in EQUIV_PARAMS])
def test_equivalence(rid, entry, kw, cell, dq_bitwise):
    """The entry against the earlier entries (proven bitwise against the legacy entry and its settings when the golden
    was recorded), through the golden record of the same row: O, LSE, dK, dV bitwise; dQ bitwise where the
    route is deterministic, else its norm (an atomic reduction: the run-to-run spread moves it by < 1e-6 relative)."""
    ins, causal = _inputs(cell)
    if rid.startswith("all4_dq8"):
        with pytest.raises(NotImplementedError) as ei:
            _run_new(entry, ins, causal, kw)
        assert ei.value.rule == "E16", ei.value
        if GOLDEN.exists():   # the earlier entry failed in the kernel build (the golden's error record)
            rows = {json.loads(ln)["id"]: json.loads(ln) for ln in open(GOLDEN)}
            assert rows[f"equiv/all4_dq8/{cell}"]["status"] == "error"
        return
    ref = _golden(rid.rsplit("-", 1)[0], cell)
    new = _run_new(entry, ins, causal, kw)
    for i, name in ((0, "o"), (1, "lse"), (3, "dk"), (4, "dv")):
        assert _sha(new[i]) == ref[name], (rid, name)
    if dq_bitwise:
        assert _sha(new[2]) == ref["dq"], (rid, "dQ")
        return
    n = new[2].double().norm().item()
    assert abs(n - ref["dq_norm"]) <= 1e-4 * abs(ref["dq_norm"]), (rid, "dQ norm", n, ref["dq_norm"])


@gpu
def test_return_lse_does_not_move_o():
    ins, causal = _inputs("S1")
    q, k, v, _ = ins
    with torch.no_grad():
        o1 = flash_attn_fp4(q, k, v, causal=causal)
        o2, _ = flash_attn_fp4(q, k, v, causal=causal, return_lse=True)
        o3 = flash_attn_fp8(q, k, v, causal=causal)
        o4, _ = flash_attn_fp8(q, k, v, causal=causal, return_lse=True)
    assert torch.equal(o1, o2) and torch.equal(o3, o4)


# ---------------------------------------------------------------------------------------------------------------
# the plan reaches the backward
# ---------------------------------------------------------------------------------------------------------------

@gpu
def test_plan_reaches_backward(monkeypatch):
    """p_scaling is backward-only (ds_scaling was, until the explicit per-tensor dS was removed, rule 33). The backward
    runs on another Python thread (and autograd's device thread); the backward driver and kernel must use the plan's
    value, per call, interleaved. (There is no ambient configuration to contradict it: the plan's specs are the only
    configuration.)"""
    import flash_attn.cute.interface_bwd_fp4 as ib
    from flash_attn.cute.lowprec import _routes
    from flash_attn.cute.flash_bwd_sm100_fp4 import FlashAttentionBackwardSm100FP4
    seen = []
    orig_init = FlashAttentionBackwardSm100FP4.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        seen.append(dict(thread=threading.get_ident(), p_scaling=self.p_scaling_fp4))
    monkeypatch.setattr(FlashAttentionBackwardSm100FP4, "__init__", init)
    drv = []
    orig_drv = ib._flash_attn_bwd_fp4

    def spy(*a, **kw):
        drv.append(kw["specs"].quant.p_scaling_fp4)
        return orig_drv(*a, **kw)
    for attr in ("cache", "cache_pre"):
        if hasattr(orig_drv, attr):
            setattr(spy, attr, getattr(orig_drv, attr))
    monkeypatch.setattr(ib, "_flash_attn_bwd_fp4", spy)
    monkeypatch.setattr(_routes, "_flash_attn_bwd_fp4", spy)

    ins, causal = _inputs("S1")
    q, k, v, do = ins
    main = threading.get_ident()

    def fwd(opts):
        qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
        return flash_attn_fp4(qr, kr, vr, causal=causal, options=opts)

    o_a = fwd(O4(p_scaling="block_1x32", operand_scaling="per_tensor"))
    o_b = fwd(O4(two_cta="off"))  # the 1-CTA kernel this test spies on (token-block defaults take the 2-CTA one)

    def bwd(o):
        t = threading.Thread(target=lambda: o.backward(do))
        t.start()
        t.join()

    bwd(o_b)  # interleaved: forward A, forward B, backward B, backward A
    rec_b, drv_b = list(seen), list(drv)
    seen.clear()
    drv.clear()
    bwd(o_a)
    rec_a, drv_a = list(seen), list(drv)
    assert rec_a and rec_b and drv_a and drv_b
    assert all(r["thread"] != main for r in rec_a + rec_b)
    assert drv_a == ["mx32"] * len(drv_a) and all(r["p_scaling"] == "mx32" for r in rec_a)
    assert drv_b == ["row"] * len(drv_b) and all(r["p_scaling"] == "row" for r in rec_b)


# ---------------------------------------------------------------------------------------------------------------
# host syncs and graph capture on the default paths
# ---------------------------------------------------------------------------------------------------------------

DEFAULTS = [("fp4", F4, {}), ("fp8", F8, {}),  # the defaults: qk_hadamard on (2026-10-01)
            ("fp4_off", F4, dict(qk_hadamard=False)), ("fp8_off", F8, dict(qk_hadamard=False)),
            # qk_hadamard on the block-scaled defaults: the rotated quantisers add no host read, capture
            ("fp4_had", F4, dict(qk_hadamard=True)), ("fp8_had", F8, dict(qk_hadamard=True)),
            ("fp4_b32_had", F4, dict(qk_hadamard=True, options=O4(operand_scaling="block_32x32"))),
            # the 1-CTA routes whose dQ pre-scale is read from the quantisers' statistics on the device (dk8 / FP8
            # control, token blocks and 32x32): no host read, and the statistics are captured with the step
            ("fp4_1cta", F4, dict(options=O4(two_cta="off"))), ("fp8_1cta", F8, dict(options=O8(two_cta="off"))),
            ("fp4_b32", F4, dict(options=O4(operand_scaling="block_32x32"))),
            ("fp8_b32", F8, dict(options=O8(operand_scaling="block_32x32"))),
            # the forward on CTA pairs: no host read, captures like the 1-CTA forward
            ("fp4_fwd2cta", F4, dict(options=O4(forward_two_cta="on"))),
            ("fp8_fwd2cta", F8, dict(options=O8(forward_two_cta="on"))),
            ("fp8_pt_fwd2cta", F8, dict(options=O8(forward_two_cta="on", operand_scaling="per_tensor")))]


@gpu
@pytest.mark.parametrize("cell", ["S1", "A2"])
@pytest.mark.parametrize("entry,okw", [(F4, {}), (F8, {}), (F8, dict(operand_scaling="per_tensor"))],
                         ids=["fp4", "fp8", "fp8_pt"])
def test_forward_two_cta_bitwise(cell, entry, okw):
    """The forward on CTA pairs gives O and LSE bitwise those of the 1-CTA forward (and so the same gradients),
    and two calls agree (deterministic)."""
    ins, causal = _inputs(cell)
    O = O4 if entry is F4 else O8
    one = _run_new(entry, ins, causal, dict(options=O(forward_two_cta="off", **okw)))
    two = _run_new(entry, ins, causal, dict(options=O(forward_two_cta="on", **okw)))
    two_b = _run_new(entry, ins, causal, dict(options=O(forward_two_cta="on", **okw)))
    for i, name in enumerate(("O", "LSE", "dQ", "dK", "dV")):
        if name in ("O", "LSE", "dK", "dV"):
            assert torch.equal(one[i], two[i]), name
        assert torch.equal(two[i], two_b[i]) or name == "dQ", name


def test_forward_two_cta_plan():
    """The plan resolves forward_two_cta; 'on' on an ineligible call raises (rule 15)."""
    q = torch.empty(1, 1024, 4, 128, dtype=BF16, device="meta")
    p = resolve_fp8(q, q, q, causal=True, options=O8(forward_two_cta="on"))
    assert p.fwd.two_cta
    p = resolve_fp8(q, q, q, causal=True, options=O8(forward_two_cta="off"))
    assert not p.fwd.two_cta
    # auto: the measured rule -- never on the 8-bit entry, never causal
    assert not resolve_fp8(q, q, q, causal=False).fwd.two_cta
    assert not resolve_fp4(q, q, q, causal=True).fwd.two_cta
    q64 = torch.empty(1, 1024, 4, 64, dtype=BF16, device="meta")
    with pytest.raises(ValueError):
        resolve_fp8(q64, q64, q64, causal=True, options=O8(forward_two_cta="on"))
    with pytest.raises(ValueError):
        resolve_fp4(q, q, q, causal=False, pv_precision="fp4", options=O4(forward_two_cta="on"))


@gpu
@pytest.mark.parametrize("entry,kw", [(F4, {}), (F8, {}), (F4, dict(options=O4(operand_scaling="block_32x32"))),
                                      (F8, dict(options=O8(operand_scaling="block_32x32")))],
                         ids=["fp4", "fp8", "fp4_b32", "fp8_b32"])
def test_qk_hadamard_deterministic(entry, kw):
    """deterministic=True with qk_hadamard: two calls bitwise on every output (the rotation adds no atomics)."""
    ins, causal = _inputs("S1")
    a = _run_new(entry, ins, causal, dict(kw, qk_hadamard=True, deterministic=True))
    b = _run_new(entry, ins, causal, dict(kw, qk_hadamard=True, deterministic=True))
    for i, name in enumerate(("O", "LSE", "dQ", "dK", "dV")):
        assert torch.equal(a[i], b[i]), name


@gpu
@pytest.mark.parametrize("name,entry,kw", DEFAULTS, ids=[d[0] for d in DEFAULTS])
def test_no_host_sync(name, entry, kw):
    ins, causal = _inputs("S1")
    q, k, v, do = ins

    def step():
        qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
        o = entry(qr, kr, vr, causal=causal, **kw)
        o.backward(do)
        return o
    step()  # compile
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
        step()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()


@gpu
@pytest.mark.parametrize("name,entry,kw", DEFAULTS, ids=[d[0] for d in DEFAULTS])
def test_graph_capture(name, entry, kw):
    ins, causal = _inputs("S1")
    q, k, v, do = ins
    eager = _run_new(entry, ins, causal, kw)
    sq, sk, sv = (x.clone().requires_grad_(True) for x in (q, k, v))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            o = entry(sq, sk, sv, causal=causal, **kw)
            o.backward(do)
            sq.grad = sk.grad = sv.grad = None
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        go, glse = entry(sq, sk, sv, causal=causal, return_lse=True, **kw)
        go.backward(do)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(go, eager[0]) and torch.equal(glse, eager[1])
    assert torch.equal(sk.grad, eager[3]) and torch.equal(sv.grad, eager[4])
    assert _cos(sq.grad, eager[2]) > 0.99999, _cos(sq.grad, eager[2])


@gpu
def test_all4_sr_graph_replay_new_dither():
    """The all-4-bit backward's stochastic-rounding seed is a device counter: every replay
    of a captured step draws a new dither (dQ / dK / dV change, the forward does not), the counter advances by one per
    replay (inside the step's own post launch: no added launch), and resetting the counter replays the eager sequence
    bitwise. The round-to-nearest configuration without
    the SR operands keeps identical replays."""
    import flash_attn.cute.interface_bwd_fp4q as iq
    ins, causal = _inputs("S1")
    q, k, v, do = ins

    def capture(kw):
        sq, sk, sv = (x.clone().requires_grad_(True) for x in (q, k, v))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                F4(sq, sk, sv, causal=causal, **ALL, **kw).backward(do)
                sq.grad = sk.grad = sv.grad = None
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            go = F4(sq, sk, sv, causal=causal, **ALL, **kw)
            go.backward(do)
        return g, go, sq, sk, sv

    g, go, sq, sk, sv = capture({})
    c = iq.sr_counter(q.device)
    outs = []
    iq.reset_sr_counter()   # 1: the first step's seed
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append([t.clone() for t in (go, sq.grad, sk.grad, sv.grad)])
    assert int(c.item()) == 4
    for a, b in ((0, 1), (1, 2)):
        assert torch.equal(outs[a][0], outs[b][0])                       # the forward has no dither
        assert not torch.equal(outs[a][1], outs[b][1]) and not torch.equal(outs[a][2], outs[b][2]), (a, b)
    _reset_sr()
    eager = _run_new(F4, ins, causal, dict(**ALL))
    assert torch.equal(eager[3], outs[0][2]) and torch.equal(eager[4], outs[0][3])   # replay 1 == the eager step 1
    g2, go2, sq2, sk2, sv2 = capture(dict(options=O4(all_fp4_stabiliser="rtn", sr_dk_operand=False)))
    rr = []
    for _ in range(2):
        g2.replay()
        torch.cuda.synchronize()
        rr.append([t.clone() for t in (sk2.grad, sv2.grad)])
    assert torch.equal(rr[0][0], rr[1][0]) and torch.equal(rr[0][1], rr[1][1])


@gpu
@pytest.mark.parametrize("hk,d", [(4, 256), (2, 256), (4, 128), (2, 128)],
                         ids=["mha", "gqa", "mha_blk32_d128", "gqa_blk32_d128"])
def test_dk8_sr_ds_graph_replay_new_dither(hk, d):
    """The dk8 backward's sr_ds seed is a device counter (interface_bwd_fp4.ds_sr_counter) where the SR is live -- the
    per-tensor dS^ of head_dim 256's three passes, and (defaults opt-in, merged) the block-scaled dS^ of d = 128's
    single pass (default operands; sr_ds takes the call to the 1-CTA kernel): every replay of a captured step draws a
    new dither (dQ / dK change; the forward and dV do not), the counter advances by one per pass (inside each pass's dQ
    postprocess: no added launch), and resetting the counter replays the eager sequence bitwise. sr_ds off keeps
    identical replays."""
    import flash_attn.cute.interface_bwd_fp4 as ib
    g0 = torch.Generator().manual_seed(0)
    q, do = (torch.randn(1, 1024, 4, d, generator=g0).to(BF16).cuda() for _ in range(2))
    k, v = (torch.randn(1, 1024, hk, d, generator=g0).to(BF16).cuda() for _ in range(2))
    ins, causal = [q, k, v, do], True
    passes = 3 if d == 256 else 1
    opk = dict(operand_scaling="per_tensor") if d == 256 else {}
    on = dict(options=O4(sr_ds=True, **opk))
    off = dict(options=O4(**opk))

    def capture(kw):
        sq, sk, sv = (x.clone().requires_grad_(True) for x in (q, k, v))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                F4(sq, sk, sv, causal=causal, **kw).backward(do)
                sq.grad = sk.grad = sv.grad = None
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            go = F4(sq, sk, sv, causal=causal, **kw)
            go.backward(do)
        return g, go, sq, sk, sv

    g, go, sq, sk, sv = capture(on)
    c = ib.ds_sr_counter(q.device)
    outs = []
    ib.reset_ds_sr_counter()   # 1: the first pass's seed
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append([t.clone() for t in (go, sq.grad, sk.grad, sv.grad)])
    assert int(c.item()) == 1 + 3 * passes, int(c.item())
    for a, b in ((0, 1), (1, 2)):
        assert torch.equal(outs[a][0], outs[b][0])                       # the forward has no dither
        assert torch.equal(outs[a][3], outs[b][3])                       # dV = P^T dO: no dS
        for i in (1, 2):
            assert not torch.equal(outs[a][i], outs[b][i]), (a, b, i)
    eager = _run_new(F4, ins, causal, on)
    for i in (2, 3, 4):
        assert torch.equal(eager[i], outs[0][i - 1]), i               # replay 1 == the eager step 1
    g2, go2, sq2, sk2, sv2 = capture(off)
    rr = []
    for _ in range(2):
        g2.replay()
        torch.cuda.synchronize()
        rr.append([t.clone() for t in (sq2.grad, sk2.grad, sv2.grad)])
    assert all(torch.equal(x, y) for x, y in zip(*rr))


# ---------------------------------------------------------------------------------------------------------------
# the masking / varlen features through the public entries: forward AND backward
# ---------------------------------------------------------------------------------------------------------------

def _ref_attn(q, k, v, causal, window=None, softcap=0.0):
    b, s, h, d = q.shape
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    sc = qf @ kf.transpose(-1, -2) / math.sqrt(d)
    if softcap:
        sc = softcap * torch.tanh(sc / softcap)
    i = torch.arange(s, device=q.device)
    keep = torch.ones(s, s, dtype=torch.bool, device=q.device)
    if causal:
        keep &= i[None, :] <= i[:, None]
    if window is not None:
        keep &= i[None, :] >= i[:, None] - window[0]
    sc = sc.masked_fill(~keep, float("-inf"))
    return (sc.softmax(-1) @ vf).transpose(1, 2)


FWD_ONLY = [("fp4_window", F4, dict(window_size=(256, 0)), dict(window=(256, 0))),
            ("fp8_window", F8, dict(window_size=(256, 0)), dict(window=(256, 0))),
            ("fp4_softcap", F4, dict(softcap=30.0), dict(softcap=30.0)),
            ("fp8_softcap", F8, dict(softcap=30.0), dict(softcap=30.0)),
            ("fp8_window_had", F8, dict(window_size=(256, 0), qk_hadamard=True), dict(window=(256, 0))),
            ("fp8_window_b32", F8, dict(window_size=(256, 0), options=O8(operand_scaling="block_32x32")),
             dict(window=(256, 0)))]


@gpu
@pytest.mark.parametrize("name,entry,kw,ref", FWD_ONLY, ids=[f[0] for f in FWD_ONLY])
def test_forward_only(name, entry, kw, ref):
    """No input requires grad: the same entries as training (token-block operands where the plan says so)."""
    ins, causal = _inputs("S1")
    q, k, v, _ = ins
    o, lse = entry(q, k, v, causal=causal, return_lse=True, **kw)
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    c = _cos(o, _ref_attn(q, k, v, causal, **ref))
    assert c > 0.99, (name, c)


@gpu
@pytest.mark.parametrize("name,entry,kw,ref", FWD_ONLY, ids=[f[0] for f in FWD_ONLY])
def test_masked_backward(name, entry, kw, ref):
    """`stale-pending-rows`: the masked backward is reachable through the API -- dQ / dK / dV vs the
    fp32 reference's (these combinations raised NotImplementedError rule 28 / PENDING before)."""
    ins, causal = _inputs("S1")
    q, k, v, do = ins
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    entry(qr, kr, vr, causal=causal, **kw).backward(do)
    qf, kf, vf = (x.float().requires_grad_(True) for x in (q, k, v))
    _ref_attn(qf, kf, vf, causal, **ref).backward(do.float())
    gate = 0.985 if entry is F4 else 0.995
    for n, a, b in (("dQ", qr.grad, qf.grad), ("dK", kr.grad, kf.grad), ("dV", vr.grad, vf.grad)):
        assert torch.isfinite(a).all() and _cos(a, b) > gate, (name, n, _cos(a, b))


@gpu
@pytest.mark.parametrize("entry", [F4, F8], ids=["fp4", "fp8"])
def test_varlen_and_seqused_backward(entry):
    """varlen (two packed documents) and seqused_k through the API, forward and backward: each document equals the
    reference on its own rows; keys past seqused_k get no gradient."""
    ins, causal = _inputs("S1")
    q, k, v, do = (x[0] for x in ins)          # (1024, h, d) packed
    cu = torch.tensor([0, 600, 1024], dtype=torch.int32, device="cuda")
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    o = entry(qr, kr, vr, causal=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=600, max_seqlen_k=600)
    o.backward(do)
    for a, e in ((0, 600), (600, 1024)):
        qf, kf, vf = (x[None, a:e].float().requires_grad_(True) for x in (q, k, v))
        r = _ref_attn(qf, kf, vf, True)
        r.backward(do[None, a:e].float())
        assert _cos(o[a:e], r[0]) > 0.985, (a, e)
        assert _cos(qr.grad[a:e], qf.grad[0]) > 0.98 and _cos(kr.grad[a:e], kf.grad[0]) > 0.98, (a, e)
    q2, k2, v2, do2 = (x.view(2, 512, *x.shape[1:]) for x in (q, k, v, do))
    used = torch.tensor([300, 512], dtype=torch.int32, device="cuda")
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q2, k2, v2))
    entry(qr, kr, vr, seqused_k=used).backward(do2)
    assert torch.isfinite(qr.grad).all() and kr.grad[0, 300:].abs().max() == 0 and vr.grad[0, 300:].abs().max() == 0


# ---------------------------------------------------------------------------------------------------------------
# the f16 dQ accumulator's range (`dq-f16-range`, device_scales._dq_prescale)
# ---------------------------------------------------------------------------------------------------------------

def _ref_grads(q, k, v, do, causal):
    qf, kf, vf = (x.float().detach().requires_grad_(True) for x in (q, k, v))
    _ref_attn(qf, kf, vf, causal).backward(do.float())
    return qf.grad, kf.grad, vf.grad


# (id, entry, options, dist): outk = K channels 3 / 37 / 64 / 101 x 20 / 30 / 40 / 50 (a fixed 2^6 pre-scale
# overflowed f16: NaN dQ); small = dO x 1e-6, trained-magnitude gradients (true units with no pre-scale flushed ~6 % of
# dQ to 0)
DQ_RANGE = [
    ("fp8_b32_outk", F8, O8(operand_scaling="block_32x32"), "outk"),
    ("fp8_tb_1cta_outk", F8, O8(two_cta="off"), "outk"),
    ("fp8_tb_small", F8, O8(), "small"),
    ("fp8_pt_small", F8, O8(operand_scaling="per_tensor"), "small"),
    ("fp8_b32_small", F8, O8(operand_scaling="block_32x32"), "small"),
    ("fp4_tb_small", F4, O4(), "small"),
    ("fp4_tb_1cta_small", F4, O4(two_cta="off"), "small"),
    ("fp4_pt_small", F4, O4(operand_scaling="per_tensor"), "small"),
]


@gpu
@pytest.mark.parametrize("name,entry,opts,dist", DQ_RANGE, ids=[r[0] for r in DQ_RANGE])
def test_dq_f16_range(name, entry, opts, dist):
    """The f16 dQ partials carry a data-sized power-of-two pre-scale on every route: finite on outlier keys, no flushed
    elements on small gradients, the dQ of the N(0,1) cell's quality either way."""
    (q, k, v, do), causal = _inputs("S1")
    if dist == "outk":
        f = torch.ones(128, device=k.device)
        f[[3, 37, 64, 101]] = torch.tensor([20.0, 30.0, 40.0, 50.0], device=k.device)
        k = (k.float() * f).to(BF16)
    scale = 1e-6 if dist == "small" else 1.0
    do = (do.float() * scale).to(BF16)
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    entry(qr, kr, vr, causal=causal, options=opts).backward(do)
    dq = qr.grad.float() / scale
    assert torch.isfinite(dq).all(), name
    rq = _ref_grads(q, k, v, do.float() / scale, causal)[0]
    # flushed elements: exactly 0 where the reference is not negligible (causal row 0 is legitimately 0: one key,
    # dS = 0)
    flushed = ((qr.grad == 0) & (rq.abs() > 1e-3 * rq.abs().max())).float().mean().item()
    assert flushed < 1e-4, (name, flushed)
    c = _cos(dq, rq)
    assert c > {"outk": 0.95, "small": 0.995 if name.startswith("fp8") else 0.985}[dist], (name, c)


@gpu
@pytest.mark.parametrize("rid", ["fp8_sq_ne_sk", "fp4_sq_ne_sk", "fp8_upstream"])
def test_per_tensor_fallback_runs_per_tensor(rid, monkeypatch):
    """The plan's operand scaling reaches the kernels (`bridge-auto-per-tensor-dropped`): s_q != s_k runs token blocks
    (it used to fall back to per-tensor), the explicit per-tensor upstream forward runs per-tensor operands
    and is the forward that runs."""
    import flash_attn.cute.interface as up
    import functools
    calls = []
    orig = up._flash_attn_fwd

    @functools.wraps(orig)  # its function attributes (compile_cache) come along
    def spy(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)
    monkeypatch.setattr(up, "_flash_attn_fwd", spy)
    g = torch.Generator(device="cuda").manual_seed(0)
    s_k = 2048 if rid.endswith("sq_ne_sk") else 1024
    q = torch.randn(1, 1024, 4, 128, device="cuda", generator=g).to(BF16).requires_grad_(True)
    k, v = (torch.randn(1, s_k, 4, 128, device="cuda", generator=g).to(BF16).requires_grad_(True) for _ in range(2))
    entry = F4 if rid.startswith("fp4") else F8
    kw = dict(options=O8(forward_kernel="upstream", operand_scaling="per_tensor")) if rid == "fp8_upstream" else {}
    o = entry(q, k, v, causal=True, **kw)
    from flash_attn.cute.lowprec._op import state_of
    blocks = getattr(state_of(o.grad_fn), "e4blk" if entry is F4 else "mx")   # the route's saved state (_routes)
    assert blocks is (rid != "fp8_upstream"), (rid, blocks)
    assert (len(calls) > 0) == (rid == "fp8_upstream"), (rid, len(calls))
    o.backward(torch.randn_like(o))
    S = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(128)
    S = S.masked_fill(torch.ones(1024, s_k, device="cuda", dtype=torch.bool).triu(s_k - 1024 + 1), float("-inf"))
    ref = torch.einsum("bhqk,bkhd->bqhd", S.softmax(-1), v.float())
    assert _cos(o.detach(), ref) > (0.98 if entry is F4 else 0.99), rid


def test_option_a_inference_only_message():
    """Option A is inference only: the refusal names both reasons and the notes; requires_grad=False resolves."""
    with pytest.raises(ValueError) as ei:
        resolve_fp4(*qkv(), **PV4)
    m = str(ei.value)
    assert ei.value.rule == 36 and "consistency pass" in m and "rebuild" in m and "trained ~0.008 worse" in m
    assert ".md" not in m and "NOTES" not in m   # self-contained: the message cites no document
    assert resolve_fp4(*qkv(rg=False), **PV4).route == "fp4.fp4_pv"
    with torch.no_grad():
        assert resolve_fp4(*qkv(), **PV4).route == "fp4.fp4_pv"


@gpu
@pytest.mark.parametrize("had", [False, True])
def test_pv4_inference_forward(had):
    """The option-A forward through the API (requires_grad=False) equals pv4_route.pv4_forward bit for bit, called
    here directly with reproduce-forward-p, the Mitchell value function and the API's auto exp_mix EMMEMMEM; had: with
    qk_hadamard (the rotated token-block Q / K). The default call (had on) also equals the recorded golden of
    optA_infer_lse at A2 when one is given."""
    from flash_attn.cute.pv4_route import pv4_forward
    (q, k, v, _do), causal = _inputs("A2")
    o_new = flash_attn_fp4(q, k, v, causal=causal, pv_precision="fp4", qk_hadamard=had)
    p = resolve_fp4(q, k, v, causal=causal, pv_precision="fp4", qk_hadamard=had)
    st = pv4_forward(q.contiguous(), k.contiguous(), v.contiguous(), 1.0 / math.sqrt(q.shape[-1]), causal,
                     p_consistency="reproduce_forward_p", softmax_exp="mitchell", p_mix="EMMEMMEM", qk_had=had,
                     specs=p.specs)
    assert _sha(o_new) == _sha(st["out"])
    if had and GOLDEN.exists():
        rows = {json.loads(ln)["id"]: json.loads(ln) for ln in open(GOLDEN)}
        assert _sha(o_new) == rows["optA/optA_infer_lse/A2"]["o"]


@gpu
@pytest.mark.parametrize("entry,kw", [
    (flash_attn_fp8, dict(options=O8(operand_scaling="per_tensor", tuning=KernelTuning(qk_split_pipe=True)))),
    (flash_attn_fp8, dict(options=O8(tuning=KernelTuning(qk_split_pipe=True)))),
    (flash_attn_fp4, dict(options=O4(tuning=KernelTuning(qk_split_pipe=True)))),
], ids=["fp8_per_tensor", "fp8_token_block", "fp4_e4m3_pv"])
def test_split_pipe_refuses_e4m3_p(entry, kw):
    """The 8-bit split step was removed: forcing the split-QK pipeline onto an E4M3 P-hat is refused by name."""
    (q, k, v, _do), _causal = _inputs("A2")
    with pytest.raises(NotImplementedError, match="split-QK pipeline hosts the 4-bit"):
        with torch.no_grad():
            entry(q, k, v, causal=False, **kw)


@gpu
@pytest.mark.parametrize("entry", [F4, F8], ids=["fp4", "fp8"])
def test_token_local_two_cta_backward(entry):
    """operand_scaling='token_local' (leak-fix) on a 2-CTA-eligible call (MHA, d = 128, s % 256 == 0): the plan takes
    the 2-CTA backward and it runs. Merge regression: the 2-CTA kernel's token_local assert read `mitchell_p`, which
    del-mitch8 deleted -- every such backward raised AttributeError."""
    ins, causal = _inputs("S1")
    O = O4 if entry is F4 else O8
    kw = dict(options=O(operand_scaling="token_local", ds_dq_rowlocal="all"))
    res = resolve_fp4 if entry is F4 else resolve_fp8
    p = res(*(x.detach().requires_grad_(True) for x in ins[:3]), causal=causal, **kw)
    assert p.bwd.kernel == "2cta" and p.fwd.operand_scaling == "token_local", (p.bwd.kernel, p.fwd.operand_scaling)
    o, lse, dq, dk, dv = _run_new(entry, ins, causal, kw)
    for name, t in (("O", o), ("dQ", dq), ("dK", dk), ("dV", dv)):
        assert torch.isfinite(t).all(), name


# ---------------------------------------------------------------------------------------------------------------
# token-block fallback shapes: token blocks on the shapes that fell back to per-tensor operands
# ---------------------------------------------------------------------------------------------------------------

def _ref_bottom_right(q, k, v, do, causal, seqused=None):
    """fp32 attention, bottom-right causal (upstream's convention for s_q != s_k), optional per-row key count; returns
    (o, lse, dq, dk, dv) with dK / dV summed over the q heads of each kv head."""
    b, sq, h, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qf, kf, vf = (x.float().detach().requires_grad_(True) for x in (q, k, v))
    kr, vr = (x.repeat_interleave(h // hk, dim=2) for x in (kf, vf))
    sc = torch.einsum("bqhd,bkhd->bhqk", qf, kr) / math.sqrt(d)
    keep = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
    if causal:
        keep &= torch.arange(sk, device=q.device)[None, :] <= torch.arange(sq, device=q.device)[:, None] + (sk - sq)
    keep = keep.expand(b, 1, sq, sk)
    if seqused is not None:
        keep = keep & (torch.arange(sk, device=q.device)[None, :] < seqused[:, None])[:, None, None, :]
    sc = sc.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(sc, dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", torch.nan_to_num(sc.softmax(-1)), vr)
    if do is not None:
        o.backward(do.float())
    return o.detach(), lse.detach(), qf.grad, kf.grad, vf.grad


TB_SHAPES = [  # (id, entry, b, s_q, s_k, h, h_kv, d, causal)
    ("fp4_sq_lt_sk_causal", F4, 2, 512, 1024, 4, 4, 128, True),
    ("fp4_sq_gt_sk_causal", F4, 2, 1024, 512, 4, 4, 128, True),
    ("fp4_sq_lt_sk_dense_gqa", F4, 2, 512, 1024, 8, 2, 128, False),
    ("fp4_sq_lt_sk_d64", F4, 2, 512, 1024, 4, 4, 64, True),
    ("fp8_sq_lt_sk_causal", F8, 2, 512, 1024, 4, 4, 128, True),
    ("fp8_sq_gt_sk_dense", F8, 2, 1024, 512, 4, 4, 128, False),
    ("fp8_sq_lt_sk_gqa_d64", F8, 2, 512, 1024, 8, 2, 64, True),
    ("fp8_ragged_causal", F8, 2, 1000, 1000, 4, 4, 128, True),
    ("fp8_ragged_sq_ne_sk", F8, 2, 700, 1000, 4, 4, 128, True),
    ("fp8_d96_causal", F8, 2, 1024, 1024, 4, 4, 96, True),
    ("fp8_d32_dense", F8, 2, 1024, 1024, 4, 4, 32, False),
    ("fp8_d80_ragged_gqa", F8, 2, 1000, 1000, 8, 2, 80, True),
]


@gpu
@pytest.mark.parametrize("name,entry,b,sq,sk,h,hk,d,causal", TB_SHAPES, ids=[t[0] for t in TB_SHAPES])
def test_token_block_fallback_shapes(name, entry, b, sq, sk, h, hk, d, causal):
    """The shapes that fell back to per-tensor operands run token blocks (the plan says so and the autograd context
    records block-scaled operands); O, LSE, dQ, dK, dV against fp32 (the bounds are those of the per-tensor path's
    measured cosines on these shapes)."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, sq, h, d, generator=g, device="cuda").to(BF16)
    k, v = (torch.randn(b, sk, hk, d, generator=g, device="cuda").to(BF16) for _ in range(2))
    do = torch.randn(b, sq, h, d, generator=g, device="cuda").to(BF16)
    res = resolve_fp4 if entry is F4 else resolve_fp8
    p = res(q.requires_grad_(True), k, v, causal=causal)
    assert p.fwd.operand_scaling == "token_block", p.log()
    qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    o, lse = entry(qr, kr, vr, causal=causal, return_lse=True)
    if not p.fwd.ragged_documents:
        from flash_attn.cute.lowprec._op import state_of
        assert getattr(state_of(o.grad_fn), "e4blk" if entry is F4 else "mx") is True, name
    o.backward(do)
    ro, rlse, rdq, rdk, rdv = _ref_bottom_right(q, k, v, do, causal)
    gate = 0.985 if entry is F4 else 0.996
    for n, a, r in (("O", o, ro), ("dQ", qr.grad, rdq), ("dK", kr.grad, rdk), ("dV", vr.grad, rdv)):
        assert a.shape == r.shape and torch.isfinite(a).all(), (name, n)
        assert _cos(a, r) > gate, (name, n, _cos(a, r))
    fin = torch.isfinite(rlse)  # (rows that see no key: -inf in the reference; the kernels' convention is not checked)
    # (NVFP4 S: a row with one or two visible keys carries its 4-bit score error straight into the LSE, ~0.3 measured)
    lse_gate = 0.5 if entry is F4 else 0.25
    assert (lse[fin] - rlse[fin]).abs().max() < lse_gate, (name, (lse[fin] - rlse[fin]).abs().max())


@gpu
@pytest.mark.parametrize("name,entry,b,sq,sk,h,hk,d,causal", [TB_SHAPES[i] for i in (0, 1, 5, 7, 9)],
                         ids=[TB_SHAPES[i][0] for i in (0, 1, 5, 7, 9)])
def test_token_block_fallback_shapes_graph_sync_det(name, entry, b, sq, sk, h, hk, d, causal):
    """The new token-block shapes: no host sync, CUDA-graph capture bitwise the eager step, deterministic=True bitwise
    across calls."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, sq, h, d, generator=g, device="cuda").to(BF16)
    k, v = (torch.randn(b, sk, hk, d, generator=g, device="cuda").to(BF16) for _ in range(2))
    do = torch.randn(b, sq, h, d, generator=g, device="cuda").to(BF16)

    def step(xs):
        qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in xs)
        o = entry(qr, kr, vr, causal=causal, deterministic=True)
        o.backward(do)
        return [o.detach(), qr.grad, kr.grad, vr.grad]
    a = step((q, k, v))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        bb = step((q, k, v))
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    for i, n in enumerate(("O", "dQ", "dK", "dV")):
        assert torch.equal(a[i], bb[i]), (name, "deterministic", n)
    sq_, sk_, sv_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            entry(sq_, sk_, sv_, causal=causal, deterministic=True).backward(do)
            sq_.grad = sk_.grad = sv_.grad = None
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        go = entry(sq_, sk_, sv_, causal=causal, deterministic=True)
        go.backward(do)
    graph.replay()
    torch.cuda.synchronize()
    for n, x, y in (("O", go, a[0]), ("dQ", sq_.grad, a[1]), ("dK", sk_.grad, a[2]), ("dV", sv_.grad, a[3])):
        assert torch.equal(x, y), (name, "graph", n)


@gpu
@pytest.mark.parametrize("entry", [F4, F8], ids=["fp4", "fp8"])
def test_sq_ne_sk_causal_channel_is_token_local(entry):
    """s_q != s_k (bottom-right causal): a future key moves only queries in its own 32-token block's reach -- the
    per-tensor global channel is gone -- and with operand_scaling='token_local' (+ ds_dq_rowlocal='all') O / LSE / dQ of
    the queries that do not see the key are exactly unchanged."""
    g = torch.Generator(device="cuda").manual_seed(0)
    sq, sk = 256, 512
    q = torch.randn(1, sq, 2, 128, generator=g, device="cuda").to(BF16)
    k, v = (torch.randn(1, sk, 2, 128, generator=g, device="cuda").to(BF16) for _ in range(2))
    do = (torch.randn(1, sq, 2, 128, generator=g, device="cuda") * 1e-3).to(BF16)
    O = O4 if entry is F4 else O8

    def run(vv, op):
        qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, vv))
        o, lse = entry(qr, kr, vr, causal=True, deterministic=True, return_lse=True,
                       options=O(operand_scaling=op, ds_dq_rowlocal="all"))
        o.backward(do)
        return o.detach(), lse, qr.grad
    p = 400  # key 400 is seen by queries >= 400 - (512 - 256) = 144
    v2 = v.clone()
    v2[:, p] = 128.0 * torch.sign(torch.randn(1, 2, 128, generator=g, device="cuda")).to(BF16)  # moves block exponents
    for op in ("auto", "token_local"):
        a, b_ = run(v, op), run(v2, op)
        rows = p - (sk - sq)
        d_o = (a[0][:, :rows] - b_[0][:, :rows]).float().abs().amax(dim=(0, 2, 3))
        d_q = (a[2][:, :rows] - b_[2][:, :rows]).float().abs().amax(dim=(0, 2, 3))
        if op == "token_local":
            assert d_o.max() == 0 and d_q.max() == 0 and torch.equal(a[1][..., :rows], b_[1][..., :rows]), op
        else:  # only the queries whose 32-key block straddles p can move (subnormal re-rounding of the block's V codes)
            moved = (d_o > 0).nonzero().flatten()
            assert moved.numel() == 0 or rows - int(moved.min()) <= 32, (op, rows - int(moved.min()))


def test_fp8_per_tensor_ds_removed():
    """Block-scaled dS is the only explicit 8-bit dS. The per-tensor dS and its sources are
    refused by name (rule 33) on every operand scaling (the 4-bit entry: test_fp4_per_tensor_ds_removed)."""
    q, k, v = qkv()
    for op in ("auto", "token_block", "per_tensor", "block_32x32"):
        for kw in (dict(ds_scaling="per_tensor"), dict(per_tensor_ds_source="recompute"),
                   dict(per_tensor_ds_source="forward_bound")):
            with pytest.raises(ValueError) as ei:
                resolve_fp8(q, k, v, options=O8(operand_scaling=op, **kw))
            assert getattr(ei.value, "rule", None) == 33, (op, kw, ei.value)
    for ds in ("auto", "block_32x32"):
        p = resolve_fp8(q, k, v, options=O8(ds_scaling=ds))
        assert p.bwd.ds_scaling == "block_32x32" and p.bwd.per_tensor_ds_source == "n/a"


def test_fp4_per_tensor_ds_removed():
    """The explicit per-tensor dS is gone from flash_attn_fp4 too -- refused by name (rule 33)
    on every operand scaling and route, before any other rule. It stays only where the plan resolves it itself: head_dim
    256 (the three-pass kernels) under ds_scaling='auto' (and, inside the all-4-bit kernel, the diagonal route's r_k
    pairs; the plan states block_32x32 for that route's dK-side dS)."""
    q, k, v = qkv()
    for op in ("auto", "token_block", "per_tensor", "block_32x32", "token_local"):
        for route in ({}, dict(pv_precision="fp4"), ALL):
            for shape in ({}, dict(d=256), dict(d=96)):
                with pytest.raises(ValueError) as ei:
                    resolve_fp4(*qkv(**shape), options=O4(operand_scaling=op, ds_scaling="per_tensor"), **route)
                assert getattr(ei.value, "rule", None) == 33, (op, route, shape, ei.value)
    for ds in ("auto", "block_32x32"):
        assert resolve_fp4(q, k, v, options=O4(ds_scaling=ds)).bwd.ds_scaling == "block_32x32"
        assert resolve_fp4(q, k, v, causal=True, options=O4(ds_scaling=ds), **ALL).bwd.ds_scaling == "block_32x32"
    p = resolve_fp4(*qkv(d=256), options=O4(operand_scaling="per_tensor"))
    assert p.bwd.ds_scaling == "per_tensor" and p.bwd.head_dim_path == "three_pass"
    # the per-tensor dS of head_dim 256 is the three-pass kernels' own; the specs state it (they used to keep blk32 and
    # leave the choice to the driver's out_pass), so the spec is what the kernels run
    assert p.specs.quant.ds_scaling_fp4 == "per_tensor" and p.specs.bwd1.ds_scaling_fp4 == "per_tensor"
    assert p.specs.bwd2 is None or p.specs.bwd2.ds_scaling_fp4 == "per_tensor"

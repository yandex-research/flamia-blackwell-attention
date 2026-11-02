# Low-precision attention (FP4 / FP8) on Blackwell

This fork adds 4-bit (NVFP4) and 8-bit (E4M3) attention to FlashAttention-4, forward and backward, for the SM100
family (tested on B200). The kernels are written in CuTe DSL like the rest of `flash_attn/cute`; the BF16 / FP16 entry
points of FlashAttention-4 are unchanged.

```python
import torch
from flash_attn.cute import flash_attn_fp4, flash_attn_fp8

q, k, v = (torch.randn(4, 4096, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
out = flash_attn_fp4(q, k, v, causal=True)        # NVFP4 Q.K, E4M3 P.V; E4M3-operand backward
out.sum().backward()
out8 = flash_attn_fp8(q, k, v, causal=True)       # E4M3 everywhere
```

The inputs stay BF16 / FP16: each call quantises them itself (fused quantiser kernels, no host synchronisation) and
returns BF16 / FP16 outputs and gradients.

New here? Start with [lowprec_usage.md](lowprec_usage.md) (install, examples, recipes); the code layout is in
[lowprec_code_map.md](lowprec_code_map.md).

Contents: [API](#api) | [Defaults](#defaults) | [Precision routes](#precision-routes) |
[Causal safety](#causal-safety) | [Determinism](#determinism) | [Caveats](#caveats-the-fastest-path-is-not-always-right) |
[Requirements](#requirements) | [Performance](#performance)

---

## API

### Entry points

Both are exported from `flash_attn.cute` and from `flash_attn.cute.lowprec`.

```python
flash_attn_fp4(q, k, v, *, softmax_scale=None, causal=False, window_size=(None, None), softcap=0.0,
               score_mod=None, score_mod_bwd=None, mask_mod=None, aux_tensors=None,
               cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, seqused_k=None,
               seqused_q=None, deterministic=False, return_lse=False,
               pv_precision="fp8", backward="e4m3", causal_safety="auto", qk_hadamard=None, options=None)

flash_attn_fp8(q, k, v, *, softmax_scale=None, causal=False, window_size=(None, None), softcap=0.0,
               score_mod=None, score_mod_bwd=None, mask_mod=None, aux_tensors=None,
               cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, seqused_k=None,
               seqused_q=None, deterministic=False, return_lse=False, qk_hadamard=None,
               q_quantized=None, k_quantized=None, q_scale_fused=False, options=None,
               q_descale=None, k_descale=None, v_descale=None, grad_proxies=None)
```

**Shapes.** `q` is `(b, s_q, h, d)`, `k` and `v` are `(b, s_k, h_kv, d)` with `h % h_kv == 0` (MHA, GQA, MQA), in
bfloat16 or float16. With `cu_seqlens_q` / `cu_seqlens_k` the tensors are packed `(total, h, d)`. The call returns
`out` with the dtype and shape of `q`; with `return_lse=True` it returns `(out, lse)`, `lse` being `(b, h, s_q)`
float32 in nats and not differentiable.

**Common arguments** (the same meaning as in `flash_attn_func`):

| argument | meaning |
|---|---|
| `softmax_scale` | default `1 / sqrt(d)` |
| `causal` | bottom-right aligned causal mask |
| `window_size` | `(left, right)` sliding window |
| `softcap` | tanh soft-capping of the scores, `0` = off |
| `score_mod`, `mask_mod`, `aux_tensors` | CuTe DSL callables with upstream's signatures; `score_mod_bwd` is the score_mod's backward on dS and is required with a `score_mod` when an input requires grad |
| `cu_seqlens_*`, `max_seqlen_*` | variable length. With `seqused_q` and `seqused_k` (aligned, zero-copy): `cu_seqlens_*` are document starts, multiples of 128, and the kernels run in place on the caller's tensors. Without them (repacked): the documents are copied to aligned starts on the GPU and back (warned once per process) |
| `seqused_q` | aligned varlen only: each document's true query length (FA3 convention) |
| `seqused_k` | without `cu_seqlens_*`: per-batch used key length, served as documents of the batch rows; with them: each document's true key length |
| `deterministic` | bitwise-reproducible dQ (see [Determinism](#determinism)) |
| `return_lse` | also return the log-sum-exp; `(h, total_q)` under varlen (aligned and repacked), `(b, h, s)` under batched `seqused_k` |

`window_size`, `softcap`, `score_mod`, `mask_mod`, varlen and `seqused_k` run forward and backward on the E4M3 P.V
routes (`flash_attn_fp8` and `flash_attn_fp4` with its default `pv_precision="fp8"`). The 4-bit P.V routes take
`causal` only among the masks and mods; they run on varlen documents too (see [Varlen coverage](#varlen-coverage)).

**`flash_attn_fp4` arguments.**

| argument | values | meaning |
|---|---|---|
| `pv_precision` | `"fp8"` (default), `"fp4"` | the P.V product: E4M3 P-hat times E4M3 V, or NVFP4 P-hat times NVFP4 V |
| `backward` | `"e4m3"` (default), `"fp4"` | the backward: S recomputed from the forward's 4-bit Q/K codes with dP / dV / dK / dQ on E4M3 operands, or the all-4-bit query-row backward (needs `pv_precision="fp4"`, d = 128) |
| `causal_safety` | `"auto"`, `"key_local"`, `"diagonal"`, `"off"` | how a 4-bit P.V keeps future keys out of V's shared scale; see [Causal safety](#causal-safety) |
| `qk_hadamard` | `None` (default), `True`, `False` | random-sign Hadamard rotation of Q and K along d (d = 128), fused into the quantisers; `None` = on wherever it can run, with a resolution-log line where it cannot; `True` = required (refused where it cannot run); `False` = off |
| `options` | `FP4AttnOptions` | the advanced options below |

**`flash_attn_fp8` arguments.** `qk_hadamard` and `options` (`FP8AttnOptions`) as above, plus:

| argument | meaning |
|---|---|
| `q_quantized`, `k_quantized` | caller-quantised per-tensor E4M3 Q and K (`flash_attn.cute.prequant.QuantizedOperand` or `(codes, descale)`), supplied together; they skip the forward's quantiser and force per-tensor operands, whose global causal channel is then the caller's responsibility |
| `q_scale_fused` | the supplied Q descale already carries `log2(e) * softmax_scale` |
| `q_descale`, `k_descale`, `v_descale` | FA3-style pre-quantised input: `q`, `k`, `v` are `torch.float8_e4m3fn` codes and these are their float32 `(b, h_kv)` dequantisation scales (`b` = batch row, or document under varlen); see [Pre-quantised E4M3 inputs](#pre-quantised-e4m3-inputs-fa3-convention) |
| `grad_proxies` | with E4M3 inputs: bf16 `(q, k, v)` whose `.grad` receives dQ / dK / dV (straight-through) |

`flash_attn_fp4` refuses `q_descale` / `k_descale` / `v_descale` and E4M3 inputs (rule 32): its Q.K^T is NVFP4, so E4M3
codes would need a re-quantisation to NVFP4 -- a quantiser pass and a second rounding -- or an E4M3 Q.K^T, which is
`flash_attn_fp8`'s forward.

### Pre-quantised E4M3 inputs (FA3 convention)

`flash_attn_fp8(q8, k8, v8, q_descale=dq, k_descale=dk, v_descale=dv, ...)` takes the operands already in E4M3, as
FlashAttention-3 does (`hopper/flash_attn_interface.py`): `x ~= code * descale[b, kv_head]`, one float32 scale per
(batch row, kv head) -- per (document, kv head) under varlen -- and the q heads of a GQA group share their kv head's
`q_descale`. No Q / K / V quantiser runs. The output is bf16.

- **Served.** Batched `(b, s, h, d)` with `s_q % 128 == 0` and `s_k % 128 == 0`, and the aligned varlen form
  (`cu_seqlens` at multiples of 128 + `seqused_q` / `seqused_k`; the documents run in place); head_dim 64 / 128;
  causal or dense; window / softcap / score_mod / mask_mod (the forward's score_mod applies the per-(row or document,
  kv head) score scale before the mod; the doc-scaled backward applies masks and mods); MHA / GQA / MQA; `return_lse`;
  `deterministic`; `forward_two_cta="on"`. Batched `seqused_k` runs as aligned documents of the batch rows (zero-copy
  views; the K / V proxies' gradients past `seqused_k` are zero); repacked varlen (`cu_seqlens` without `seqused_*`)
  copies the codes (one byte per element) and the gradient proxies into an aligned buffer and back. Refused (rule 32,
  each with its reason): other head dims or ragged batched lengths, block-scaled `operand_scaling`,
  `forward_kernel="upstream"`, `grad_format="e5m2"`, `qk_hadamard=True` (the codes are unrotated; `None` resolves to
  off), and `q_quantized` / `q_scale_fused` together with the descales.
- **Validation.** Host-side only (dtype, shape `(b or n_docs, h_kv)`, device; any strides): nothing is read back, no
  host sync; the call is CUDA-graph capturable. float64 descales are accepted too (an exact hand-over of this library's
  own float64 scales).
- **Forward.** The fork forward on the caller's codes reads, per work tile, the score scale
  `float32((softmax_scale * q_descale) * k_descale)` and the V descale `float32(v_descale)` of its (row or document,
  kv head): the per-(document, kv head) tables of the per-tensor varlen path, built from the descales by one small
  launch. Batched calls use the batched kernel with a `(b, h_kv)` table (2-CTA forward where the plan picks it).
- **Equivalence.** Codes and scales made by this library's per-tensor quantiser and handed over give outputs bitwise
  equal to the internal per-tensor path (`operand_scaling="per_tensor"`, `qk_hadamard=False`): batched (one scale per
  tensor, broadcast to `(b, h_kv)`) and aligned varlen (one per (document, kv head)), forward and, on aligned varlen,
  the backward (`tests/cute/test_prequant_fa3.py`). With float32 descales the result is bitwise where the scales are
  powers of two (`per_tensor_scale="power_of_two"`) and within the last bits of the score scale otherwise.
- **Backward (training).** PyTorch casts every gradient to its input's dtype, so an E4M3 leaf could only receive E4M3
  codes on no scale; E4M3 inputs that require grad are refused. Pass detached codes and `grad_proxies=(q, k, v)`
  (bf16, the codes' shapes -- usually the pre-quantisation activations): their `.grad` receives the straight-through
  gradient (the gradient at the dequantised codes). The backward reuses the caller's codes (no E4M3 copies are made);
  dO is quantised by the library, one scale per (document, kv head), in the launch that also writes D, the shifted LSE
  and the zeroed f16 dQ accumulator; the 1-CTA per-(document, kv head) kernel reads a scalar table built from the
  descales; the f16 dQ pre-scale is sized from `max ||V_k||^2` of the dequantised V codes (one read of the V codes).
  A batched call with grad keeps the batched forward (the `(b, h_kv)` table) and runs the same 1-CTA kernel with every
  batch row as a document, after one batched dO launch (dO's scale per (batch row, kv head), D `(b, h, s)` folded per
  row, the shifted LSE, the zeroed accumulator); it is bitwise the aligned varlen path on the same rows as documents.
- **Causality.** A per-tensor scale is a causal channel inside its row / document (a later token's max moves the
  rounding of earlier tokens); with pre-quantised input that scaling is the caller's.

Every invalid combination raises before anything is quantised or compiled. The exception names the rule it breaks
(`exc.rule`) and says which argument to change. An option that the call would not read is refused rather than ignored.

### `FP4AttnOptions`

A frozen dataclass; every default reproduces the default 4-bit configuration. `"auto"` fields are resolved per call
from the shapes, and the plan records each decision and its reason (see [Plans](#plans-resolve_fp4--resolve_fp8--attnplan)).

| field | values (default first) | meaning |
|---|---|---|
| `operand_scaling` | `"auto"`, `"token_block"`, `"block_32x32"`, `"per_tensor"`, `"token_local"` | scaling of the E4M3 operand copies (V on the 8-bit P.V; the backward's Q / K / V / dO). `"token_block"`: one UE8M0 exponent per 32 tokens x the whole head dim. `"block_32x32"`: per 32 tokens x 32 d (1-CTA kernels only). `"per_tensor"`: one GPU-computed scale per tensor (fastest; a global causal channel); on varlen one per (document, kv head), 1-CTA backward, head_dim 128 on this entry. `"token_local"`: one power-of-two exponent per token (exactly causal). `"auto"` resolves to `"token_block"` and never falls back to `"per_tensor"` silently; a call token blocks cannot serve (head_dim 256, `p_scaling="block_1x32"`) raises and names the opt-in |
| `per_tensor_scale` | `"exact"`, `"power_of_two"` | per-tensor scales are `amax / 448`, or that rounded up to a power of two (a 20-60x smaller, but non-zero, causal channel at no cost); per-tensor operands only |
| `ds_scaling` | `"auto"`, `"block_32x32"` | the E4M3 score gradient dS-hat: one UE8M0 exponent per 32 keys x 32 queries. At head_dim 256 `"auto"` resolves to a per-tensor dS (the three-pass kernels have no block-scaled dS). An explicit `"per_tensor"` is refused |
| `ds_dq_rowlocal` | `"off"`, `"diagonal"`, `"all"` | per-query-row exponents for the dQ side of the block dS; `"all"` closes the in-block dS causal channel exactly (+21-38 % backward kernel time); `"diagonal"` does not close it (see [Causal safety](#causal-safety)) |
| `fp4_ds_multiplier` | `"auto"`, `"global"`, `"global_row_boost"`, `"row_local"` | `backward="fp4"` only: the multiplier under the 4-bit dS. `"auto"` = `"row_local"` when causal, `"global_row_boost"` when dense |
| `p_scaling` | `"row"`, `"block_1x32"` | E4M3 P-hat of the dV GEMM: `P * 2^8` with no scale factor, or one UE8M0 per key row and 32 queries (1-CTA only on the 4-bit route) |
| `p_frame` | `"two_level"`, `"single_level"`, `"per_tile"` | option A only: how the forward frames its 4-bit P-hat; `"per_tile"` is refused |
| `p_hat_frame` | `"auto"`, `"row"`, `"group"` | the E4M3 P-hat of the token-block forward: a row frame with a constant P scale tile, or one UE8M0 per 32 keys. `"auto"` = `"group"` on `flash_attn_fp4` |
| `softmax_exp` | `"auto"`, `"exp2"`, `"mitchell"` | how P-hat = 2^x is formed: `"auto"` = Mitchell (Direct-P, the exact clamped classifier) where P-hat is 4-bit, exp2 where it is E4M3 |
| `k_channel` | `"fold"`, `"cut"`, `"all"`, `"open"` | `causal_safety="diagonal"` on the all-4-bit route: how dQ's K operand avoids future keys' scales (`"fold"`, the default, is training-certified; `"open"` leaves the channel open) |
| `q_block_2d` | `False`, `True` | NVFP4 Q with one scale per 16 queries x 16 d; worse everywhere measured and refused under any mask |
| `v_hadamard` | `False`, `True` | Hadamard on V, all-4-bit key-local route only; no recorded win |
| `all_fp4_stabiliser` | `"sr"`, `"rtn"`, `"eden_rht"` | `backward="fp4"`: stochastic rounding of dO-hat and the 4-bit dS, round-to-nearest, or a joint head-dim Hadamard with an MS-EDEN dO (dense only) |
| `sr_ds` | `False`, `True` | `backward="e4m3"`: stochastic rounding of the E4M3 dS-hat (per-tensor and block-scaled dS; takes the call to the 1-CTA backward, `two_cta="on"` is refused). Measured in training: neutral under the default block-scaled dS (d128, +0.0013 +- 0.0027 val loss, 4 seeds), -0.04 at head_dim 256 (per-tensor dS); +1-2 % step |
| `sr_dk_operand` | `True`, `False` | `backward="fp4"`: SR of the dK operand's E4M3 dS-hat |
| `sr_dv_operand` | `False`, `True` | `backward="fp4"`: SR of the dV operand's P-hat |
| `fp4_backward_dkdv` | `"e4m3"`, `"fp4"` | `backward="fp4"`: the dK / dV operands; `"fp4"` makes every backward GEMM 4-bit (gradients fall below the accuracy gate) |
| `dq_operand` | `"fp4"`, `"e4m3"` | `backward="fp4"`: the dQ GEMM's dS operand; `"e4m3"` is refused (not built on these routes) |
| `two_cta` | `"auto"`, `"on"`, `"off"` | whether the backward may use the 2-CTA cluster kernel; `"auto"` follows the measured rule (from key length 32768: 2-CTA wherever supported, deterministic excepted); `"on"` on an ineligible call raises |
| `forward_two_cta` | `"auto"`, `"on"`, `"off"` | whether the forward runs on CTA pairs (O and LSE bitwise the 1-CTA forward's); `"auto"` = on only where measured faster |
| `seqused_k_group_aligned` | `False` | kept for compatibility; `True` is refused (no 4-bit P.V route takes `seqused_k`) |
| `save_bf16_inputs` | `True`, `False` | `False` keeps only the quantised codes for the backward (1560 -> 792 bytes per (token, head) at d = 128, bitwise); E4M3 P.V with per-tensor operands |
| `p_consistency` | `"reproduce_forward_p"`, `"none"` | option A only: the forward variant whose P-hat export fixes its O / LSE |
| `do_rounding` | `"rtn"`, `"sr"` | option A's backward dO rounding; option A is inference-only, so `"sr"` is refused |
| `exp_mix` | `"auto"`, `"none"`, 8 characters over `M` / `E` / `2` / `3` | the value function of each 16-key group position of the 4-bit P-hat (option A inference: `"auto"` = `"EMMEMMEM"`, the all-4-bit route: `"none"` = Mitchell everywhere) |
| `tuning` | `KernelTuning` or `None` | see below |

### `FP8AttnOptions`

| field | values (default first) | meaning |
|---|---|---|
| `operand_scaling` | `"auto"`, `"token_block"`, `"block_32x32"`, `"per_tensor"`, `"token_local"` | as for FP4, for E4M3 Q / K / V and dO. `"auto"` = `"token_block"` on every served shape (head dims up to 128 in steps of 16, `s_q != s_k`, varlen, any length); calls that need per-tensor operands (a non-block dS, E5M2 gradients, `forward_kernel="upstream"`) raise and name the opt-in |
| `per_tensor_scale` | `"exact"`, `"power_of_two"` | as for FP4 |
| `ds_scaling` | `"auto"`, `"block_32x32"`, `"row_local"`, `"row_1x32"`, `"block_32x64"`, `"tile_128"` | the E4M3 dS-hat; `"auto"` = `"block_32x32"`. `"row_local"` / `"row_1x32"` are exactly leak-free (1-CTA only, 1.4-1.5x backward); `"block_32x64"` / `"tile_128"` are 2-CTA only, slower and not convergence-tested. `"per_tensor"` is refused |
| `ds_dq_rowlocal` | `"off"`, `"diagonal"`, `"all"` | as for FP4 (with `ds_scaling="block_32x32"`) |
| `per_tensor_ds_source` | `None` | kept for compatibility; any value is refused |
| `p_scaling` | `"row"`, `"block_1x32"` | the E4M3 P-hat of the dV GEMM |
| `grad_format` | `"e4m3"`, `"e5m2"` | element type of dO and dS; E5M2 needs explicit per-tensor operands (1-CTA) |
| `softmax_exp` | `"exp2"` | `"mitchell"` is refused on the 8-bit route |
| `forward_kernel` | `"auto"`, `"fork"`, `"upstream"` | the fork's forward on token-block operands; upstream's FP8 forward only for explicit per-tensor operands it serves |
| `two_cta`, `forward_two_cta` | `"auto"`, `"on"`, `"off"` | as for FP4 (`"auto"` backward: 2-CTA wherever eligible) |
| `p_hat_frame` | `"auto"`, `"row"`, `"group"` | as for FP4; `"auto"` = `"row"` on `flash_attn_fp8` |
| `tuning` | `KernelTuning` or `None` | see below |

### `KernelTuning` (advanced; may change without notice)

`flash_attn.cute.lowprec.tuning.KernelTuning` holds the kernels' measured performance constants: pipeline depths,
TMEM load repetitions, register budgets, exp2-emulation interleaves, persistent-grid thresholds and the like (91
fields; `TUNING_FIELDS` lists each with its value kind). Every field defaults to `None`, meaning "the measured rule".
A non-`None` value is validated when the object is built and copied into the kernel spec that reads it, so it is part
of the compile key. It is for benchmark campaigns and is not a stable API: fields may be renamed or removed, and some
values are refused on some routes. Pass it as `options=FP4AttnOptions(tuning=KernelTuning(...))`.

**Fields added on 2026-10-06/07 (advanced).** Value = what the measured rule does when the field is left at `None`.
Changes are paired medians of 3 rounds on one B200 (h16 d128; cells as in [Performance](#performance)); "bitwise" means
O / LSE (and dK / dV where stated) are bit-identical to the default.

*Aligned varlen scheduling* (all scheduling only: O / LSE / dK / dV bitwise either way):

| field | `None` = | what it does | measured effect | status |
|---|---|---|---|---|
| `varlen_fwd_list` | `True` | forward: persistent grid over a GPU-built longest-first work list of the documents' tiles (empty tiles pruned); `False` = one CTA per tile | off: forward +10 to +65 % on causal documents, +1 to +31 % dense | default |
| `varlen_fwd_tma_o` | `True` | causal forward: batched sO + TMA O epilogue (dense keeps the correction warps' predicated copies) | off: causal forward +1 to +16 % (with the stage skip) | default |
| `varlen_fwd_stage_skip` | `True` | causal forward: the 1-CTA stage skip (needs `varlen_fwd_tma_o`) | off: causal forward +2.5 to +13 % | default |
| `varlen_l2_sections` | `True` | work lists ordered in L2 head sections (forward) and head / document sections (backward); `False` = one global longest-first order | off: causal backward +0.2 to +22.5 %; document sections on the forward measured -7 to -9 % slower, so the forward keeps head sections only | default |
| `varlen_bwd_list` | `True` | 1-CTA causal non-deterministic backward: document-aware longest-first key-tile list; `False` = the plain (key tile, head) grid | off: causal backward up to +9.7 % (skewed lengths), -1.3 to +3.6 % on equal lengths | default |
| `varlen_fwd_sched` | `"auto"` | how CTAs take work-list entries: `"auto"` = `"clc"` with the just-in-time claim on causal calls, `"static"` on dense ones (logged as `varlen_fwd_sched` in `plan.log()`); `"static"` = boustrophedon deal over a persistent grid; `"atomic"` = dynamic claiming by a global counter; `"clc"` = dynamic claiming with Cluster Launch Control | `clc` + jit vs static: causal forward -1.7 to -5.7 % on 13 of 16 cells (fp8 and fp4, MHA and GQA), -0.6 to +0.5 % on the other 3 (64k log-uniform fp8, an adversarial mix); dense -8.5 to +2.3 % (wins on skewed lengths, +0.6 to +2.3 % on equal ones, hence static there); `atomic`: -6.4 to +5.3 % | `"auto"` default (causal CLC since 2026-10-07) |
| `varlen_fwd_sched_stages` | `2` | SMEM ring depth of the dynamic claiming (1-4) | with a producer-warp claim, claiming ahead costs up to +14.5 % on skewed causal lists | opt-in |
| `varlen_fwd_sched_jit` | `False` (forced `True` by `"auto"` on causal calls) | the load warp claims the next entry right after issuing the current tile's last K / V load, instead of a producer warp filling the ring | the jit claim is what makes CLC win (claim lag is the first-order effect) | used by the default |
| `varlen_fwd_sched_early` | `False` | with the jit claim: claim before the last K / V pair's loads | within +-1 % of the plain jit claim | not recommended |

*Forward experiments* (`fwd_exp`: comma-separated switches; both fields default to the certified build):

| field / value | `None` = | what it does | measured effect | status |
|---|---|---|---|---|
| `fwd_exp="sfi"` | `""` | 4-bit group-SF P-hat step: the scale's ceil / 2^ceil on the FMA / ALU pipes instead of FRND + MUFU | 4-bit forward -2.8 to -4.3 % on every cell, bitwise | candidate default |
| `fwd_exp="schm3"` (`schmK`, `scha`, `sche`) | `""` | 8-bit P-hat step: Schraudolph exp2 on 8 - K of every 8 pairs (`scha`: all, `sche`: only the polynomial pairs) | `schm3`: -9.9 to -17.3 % forward (ahead of cuDNN FP8 on every cell), O cosine -6e-5; training NaN at step 500 in 4 / 4 seeds | inference-only (NaN in training) |
| `fwd_exp="sfi,schg"` (with `fp4_exp2_emulation_every=3`) | `""` | 4-bit: the emulated exp2 pairs take the Schraudolph read-out | -15.4 to -19.8 % forward; training NaN at steps 750-1500 in 4 / 4 seeds | inference-only (NaN in training) |
| `fwd_exp="gmx"`, `"zzlong"` | `""` | group maxes from raw scores (bitwise) / balanced persistent causal grid above s_k 16384 | `gmx` +1.3 / +2.6 % alone; `zzlong` +2.0 % (8-bit A3) / -0.9 % (4-bit) | not recommended |
| `fwd_ptxas` | `""` | extra ptxas options of the forward compile (e.g. `--register-usage-level=N`, a beta ptxas option) | reshuffles the schedule by up to +-5 %, no level wins everywhere | not recommended |

*Backward*:

| field | `None` = | what it does | measured effect | status |
|---|---|---|---|---|
| `bwd_fp4q_post` | `"legacy"` | all-4-bit backward's post launch (dQ convert + the K fold's dK / dV rows): `"rows"` coalesced, `"epi"` rows with the dK / dV factors applied in the kernel's epilogue | post 104 -> 62 (`rows`) / 21 us (`epi`) at A1; backward A1 -3.8 / -6.2 %, A2 -1.1 / -1.3 %, A3-A4 0 to -0.6 %; bitwise | candidate default (`"epi"`) |
| `bwd_2cta_gqa` | `None` (= on with `two_cta="on"` or from key length 32768, else off) | GQA on the 2-CTA backward: one q head's dK / dV per pair, reduce-added in fp32 (ordered by semaphores when `deterministic=True`); batched and varlen | -28.6 % (b2 s8192 h16/2 causal) to +21.3 % (b8 s2048 h32/8 dense) below 32K; part of the >= 32K default | `True` / `False` force it |
| `bwd_fp4_det_gqa_rot` | `False` | deterministic dense GQA with a rotated walk (each key tile starts its query walk at its own block) | +39 to +60 % slower than the default deterministic walk: every step waits one release hop (~2.7 us) | not recommended |
| `bwd_fp4_q_stage` | `2` | 1-CTA E4M3 backward: a third Q / LSE ring stage (+16 KB SMEM), implies `bwd_fp4_q_dyn_issue` | -2.2 to -3.3 % (GQA causal, deterministic, varlen), +0.5 % at d64; bitwise | candidate default (not at d64) |
| `bwd_fp4_q_dyn_issue` | `False` | S / dK GEMMs issued on the dynamically indexed Q stage (one issue instead of one static copy per stage) | -0.8 to -2.5 %, +0.5 % at d64; spills 92 -> 28 B; bitwise | candidate default (not at d64) |
| `bwd_fp4_stat_shfl` | `"off"` | LSE / dPsum held one per lane and broadcast by SHFL instead of broadcast LDS (`"lse"`, `"dpsum"`, `"both"`) | +2.0 to +11.4 % slower (4x the MIO issue slots for the same data) | not recommended |
| `bwd_fp4q_dk_sr_blk32` | `False` | stochastic rounding of the all-4-bit dK operand's dS-hat under the default block-scaled dS (`sr_dk_operand` reaches only the per-tensor dS) | training +0.0004 val loss (4 seeds, neutral), step +1.5 to +1.7 % | opt-in |

*Formats*:

| field | `None` = | what it does | measured effect | status |
|---|---|---|---|---|
| `fp4_qk_sf_vec` | `16` (NVFP4) | `32` = MXFP4 Q / K (UE8M0 per 32) on `pv_precision="fp8"`, `backward="e4m3"`, no Hadamard / `q_block_2d` / window / mods / varlen (plan rule E40); not part of the API | O rel. L2 +20 to +22 %; forward +3.6 to +6.6 % at s4096, -0.3 to -0.5 % at s32768; never trained | not recommended |

**2-CTA backward default (`two_cta="auto"`, both entries) [2026-10-07].** From key length 32768 (`s_k`, or
`max_seqlen_k` on varlen; `lowprec.plan.TWO_CTA_LONG_KEYS`) the backward takes the 2-CTA kernel wherever it serves the
call: MHA and GQA, batched (incl. `s_q != s_k` and `s_k % 256 == 128` tails) and varlen at the normal 128-aligned
document starts. Measured 2-CTA vs 1-CTA per full step at >= 32K: batched -2.3 to -6.4 %, varlen dense -3.9 to
-9.2 %, varlen causal -1.2 to -4.7 % (one cell +1.1 %). Deterministic calls keep the 1-CTA kernel (2-CTA measured up
to +5 % batched / +14 % varlen there), except 8-bit explicit per-tensor operands as before; below 32K the earlier
rules are unchanged. With 2-CTA GQA, dK / dV are summed over q heads with fp32 atomics (not bitwise run to run unless
`deterministic=True`). `KernelTuning.bwd_2cta_gqa`: `None` (default) = GQA on 2-CTA with `two_cta="on"` or the
long-sequence default; `True` / `False` force it. No 2-CTA kernel exists for the all-FP4 route (measured ceiling
<= 0.4 %) or head_dim != 128.

### Plans: `resolve_fp4`, `resolve_fp8`, `AttnPlan`

```python
from flash_attn.cute import resolve_fp4
plan = resolve_fp4(q, k, v, causal=True)   # same arguments as flash_attn_fp4; launches nothing
print(plan.route)      # fp4.e4m3_pv
print(plan.log())      # every resolved choice and its reason, one per line
```

`resolve_*` are pure functions of the arguments and of tensor metadata (shape, dtype, `requires_grad`): they read no
tensor values, no environment variable and no global state. The returned `AttnPlan` is immutable: `entry`, `route`,
`softmax_scale`, `shape`, `fwd` / `bwd` (what each pass runs), `consistency` (the scaling every shared operand has in
each pass), `resolution_log` / `log()`, and `specs` (the kernels' compile-time settings). Each call resolves its plan
once (plans are cached per argument signature) and the autograd context carries it to the backward; nothing else
configures the kernels.

## Defaults

| | `flash_attn_fp4` | `flash_attn_fp8` |
|---|---|---|
| route | `fp4.e4m3_pv` | `fp8` |
| Q, K in the S GEMM | NVFP4: E2M1, one E4M3 scale per 16 along d | E4M3, token-block scales |
| P-hat, V in the P.V GEMM | E4M3 P-hat x E4M3 V (MXFP8 GEMM); P-hat with one UE8M0 per 32 keys | E4M3 P-hat (row frame) x E4M3 V |
| P-hat value function | exp2 (MUFU + FMA-pipe emulation) | exp2 |
| operand scaling of the E4M3 copies | token blocks: one UE8M0 exponent per 32 tokens x d | token blocks |
| Q / K Hadamard | on (d = 128) | on (d = 128) |
| backward | S recomputed from the forward's 4-bit Q / K codes; dP, dV, dK, dQ on E4M3 operands | all E4M3 |
| dS-hat | E4M3, one UE8M0 per 32 keys x 32 queries | same |
| dV's P-hat | E4M3, fixed `2^8` lift | same |
| backward kernel | 1-CTA, or 2-CTA by the measured rule; from key length 32768 2-CTA wherever supported (GQA, varlen; deterministic stays 1-CTA) | 2-CTA wherever eligible; from key length 32768 also GQA and varlen (deterministic stays 1-CTA) |
| forward on CTA pairs | off (`"auto"` takes it only for dense calls with the opt-in row-frame P-hat, where it measured 3-4 % faster) | off (measured slower on every 8-bit cell) |
| causal safety | no 4-bit V group; the in-32-token-block channel of the token-block V remains (below) | same |
| deterministic dQ | off | off |
| stochastic rounding | off | off |
| head dims | 64, 128, 256 (256: per-tensor operands and dS, opt-in) | up to 128 in steps of 16 (64 / 128 native, others padded) |
| sequence lengths | `s % 128 == 0`, or varlen | any |
| aligned varlen forward scheduler | persistent longest-first work list over the documents' tiles (L2 head sections); causal: tiles claimed dynamically with Cluster Launch Control (`KernelTuning.varlen_fwd_sched="auto"`), dense: dealt statically | same |

## Precision routes

| route | how to select it | forward | backward | status |
|---|---|---|---|---|
| **8-bit** | `flash_attn_fp8(...)` | E4M3 Q.K, E4M3 P.V | E4M3 | training and inference |
| **4-bit, E4M3 P.V** | `flash_attn_fp4(...)` | NVFP4 Q.K, E4M3 P.V | E4M3 operands, S recomputed from the 4-bit codes | training and inference; the default 4-bit route |
| **all-4-bit** | `flash_attn_fp4(..., pv_precision="fp4", backward="fp4")` | NVFP4 Q.K, NVFP4 P.V | the query-row backward: 4-bit S / dP / dQ GEMMs, E4M3 dK / dV operands (`fp4_backward_dkdv`), stochastic rounding | training; d = 128; not deterministic |
| **option A** | `flash_attn_fp4(..., pv_precision="fp4")` with no input requiring grad | NVFP4 Q.K, NVFP4 P.V | none | inference only |

**8-bit.** Q, K, V and dO are quantised to E4M3 with token-block scales, P-hat is E4M3. The backward runs on E4M3
operands throughout and reduces dQ in f16.

**How the 8-bit forward uses the token-block scales.** A token-block copy stores `x[t, :] = code[t, :] * 2^e_B`, one
power-of-two exponent `e_B` per 32 tokens `B`, shared by the whole head dimension.
- **S = Q K^T: the scale factors are folded, not fed to the MMA.** The reduction axis of S is d, and both factors are
  constant along d, so they come out of the dot product exactly:
  `S[i, j] = 2^(e_Q(i) + e_K(j)) * sum_d codeQ[i, d] * codeK[j, d]`. The forward therefore runs a plain E4M3 GEMM on
  the codes and applies the factors in the softmax, where the score is scaled anyway: the row factor `2^e_Q(i)` joins
  the softmax scale of that row and the column factor `2^e_K(j)` multiplies each score column. Powers of two multiply
  exactly, so these scores are bitwise the scores of the block-scaled S GEMM; the fold is a speed choice
  (forward -4 % at s4096 causal), not a numerical one. It runs on batched calls without window / softcap / mods; with
  those, and on varlen documents, the same scores come from the block-scaled S GEMM (`mSFQ` / `mSFK`). The 4-bit
  entry's NVFP4 S has one scale per 16 elements along
  d, which does not come out of the sum, so it stays a block-scaled NVFP4 GEMM.
- **O = P V: block-scaled (MXFP8), not folded.** The reduction axis of P.V is the keys, and V's factor changes every
  32 keys, so it cannot leave the sum. Folding it would mean multiplying P-hat by V's factor before P-hat is rounded
  to E4M3; that extra rounding leaks 2.3e-2 inside a 32-key block (`KernelTuning.token_block_forward="fold"`, opt-in).
  The default keeps V's exponents as UE8M0 scale factors of a block-scaled MXFP8 MMA (one per 32 keys), and P-hat
  has a constant row-frame scale (`p_hat_frame="row"`) on `flash_attn_fp8` or one UE8M0 per 32 keys on
  `flash_attn_fp4`.
- The backward reads the same codes and exponents (one quantisation per operand serves both passes), and its GEMMs are
  block-scaled E4M3 MMAs.

**4-bit with an E4M3 P.V.** Only the S GEMM is 4-bit. A 4-bit P-hat times the backward's own E4M3 V would make
the backward's `D = rowsum(dO * O)` inconsistent with its P (the softmax identity sum_k dS = 0 then fails), and a
4-bit P-hat forward with this backward collapsed training on every seed tried. So the training forward multiplies an
E4M3 P-hat by an E4M3 V.

**All-4-bit.** Every forward GEMM is NVFP4. The backward rebuilds the forward's 4-bit P-hat bit for bit, so dS is
consistent with the forward. Under `causal=True` the default `causal_safety="auto"` resolves to `"diagonal"`: V gets a
second, key-local payload on the tiles the causal diagonal crosses, and dQ's K operand is folded (`k_channel="fold"`).
Both read exactly 0 on every leak probe and are training-certified (two seeds of a byte-level GPT).

**Option A: 4-bit P.V for inference.** `pv_precision="fp4"` with `backward="e4m3"` is the 4-bit P-hat . 4-bit V
forward without a training backward. Any input that requires grad is refused (rule 36). A training backward for this
forward exists in the research history of this fork and is *not shipped*: keeping D consistent needs either an extra
pass over the scores or a backward that rebuilds the forward's 4-bit P-hat bit for bit. The rebuild variant was the
best of these and still lost to the default route on both time and loss:

| | step, b4 s4096 h16 d128 causal | dQ cosine | validation loss (2 seeds, lower is better) |
|---|---|---|---|
| option A with the P-hat-rebuild backward (not shipped) | 1.60-1.64 ms | 0.968 | 0.870 |
| option A with a fused D pre-pass (not shipped) | 1.30 ms | 0.973 | 0.862 |
| default 4-bit route (E4M3 P.V) | 1.00 ms | 0.991 | 0.857 |

With D taken from the forward's O (no extra pass) option A diverged in every training run. Its inference forward is
faster than the E4M3 P.V forward only on well-conditioned inputs; see the [caveats](#caveats-the-fastest-path-is-not-always-right).

## Causal safety

A causal kernel must not let O or dQ at position t depend on any input at a position after t. Block scales break
this whenever one scale is shared across positions: a later token that moves the shared scale re-rounds the earlier
tokens' codes. The table states what the shipped routes guarantee, measured with perturbation probes (change one
future row, compare the past rows bitwise).

| channel | where | size | how to close it |
|---|---|---|---|
| **4-bit V groups along keys** | 4-bit P.V (all-4-bit, option A) without `causal_safety` | up to ~8e-2 in O | `causal_safety="auto"` (default) closes it: `"diagonal"` (diagonal-only key-local V) or `"key_local"`; both read exactly 0 |
| **K into dQ** | all-4-bit route, diagonal V | 1.2e-1 in dQ unfixed | `k_channel="fold"` (default) reads 0 |
| **per-tensor scales** | `operand_scaling="per_tensor"` (explicit opt-in only), caller-quantised Q / K, head_dim 256 on `flash_attn_fp4` | up to 0.23 absolute in O, reaching the whole prefix | token-block operands (the default); `per_tensor_scale="power_of_two"` shrinks it 20-60x but not to zero |
| **in-32-token-block channel of the E4M3 copies** | the default operands of both entries (`token_block`, also `block_32x32`) | sub-ULP to ~2e-3 absolute in O, about 1e-6 in dQ; reach < 32 rows, always inside the perturbed token's own 32-token block | `operand_scaling="token_local"` |
| **in-block dS channel** | the default `ds_scaling="block_32x32"` | up to 3.8e-6 absolute in dQ (3e-4 of max\|dQ\|), reach <= 31 rows | `ds_dq_rowlocal="all"` |

**The in-32-token-block channel.** The token-block E4M3 copies share one UE8M0 exponent per 32 tokens x d. E4M3
absorbs a power-of-two rescale exactly *except* on subnormal codes: when a later token in the same block raises the
block's maximum, the exponent rises and earlier tokens' subnormal codes are re-rounded or flushed. For V this moves O
by `sum_k p_tk dV_k`: sub-ULP in fp32, sometimes a 1-ULP bf16 flip, up to about 1e-3 (2e-3 through Q / K on the 8-bit
route). It is independent of `qk_hadamard` (V is never rotated; the hit rate is the same with the rotation on or off),
affects the default causal call of both entries, and never reaches outside the perturbed token's 32-token block, so
always inside the causal diagonal tile.

**Exact causality** holds only under `operand_scaling="token_local"`: every token of the E4M3 copies carries its own
power-of-two exponent, so no token's codes or scale depend on another token. It reads exactly 0 on every probe. It
needs `ds_dq_rowlocal="all"`, head_dim 64 or 128, and no window / softcap / score_mod / mask_mod (it runs a
folded forward; varlen / `seqused_k` documents run on it, the fold tables read at each document's offsets). Its cost on top of `ds_dq_rowlocal="all"`: 4-bit forward +10-17 %, backward +22-30 %. In training it
measured +0.002 to +0.0035 validation loss (two seeds), not shown neutral. (Per-tensor V with an amax that cannot
move also reads 0, but a real per-tensor scale moves.)

**`token_local` turns `qk_hadamard` off.** No rotated token-local quantiser exists. With the default
`qk_hadamard=None` the call runs unrotated and the resolution log says so; `qk_hadamard=True` with `token_local` is
refused.

**dS and dQ.** The block dS-hat shares one exponent per 32 keys x 32 queries between the dQ and the dK GEMM. The
exponent is a maximum over the block's 32 queries, so a later query's dS (which carries its dO, its Q and, through LSE
and D, every key up to it) rounds an earlier query's dS codes. The coupling runs along the block's query axis in
*every* block, not only on the diagonal, which is why `ds_dq_rowlocal="diagonal"` reads identical to `"off"` on every
probe. `ds_dq_rowlocal="all"` gives the dQ side its own exponent per query row on every block (the dK side keeps the
block exponent: exact dK legitimately depends on later queries) and reads exactly 0; dK, dV, O and LSE are bitwise
unchanged. It costs +21-23 % of the 1-CTA backward kernel and +29-38 % of the 2-CTA one, about +20 % of a
long-sequence training step, and was training-neutral (+0.0001 mean validation loss over two seeds, spread 0.009). It
is off by default because of that cost.

**Varlen and documents.** Documents (varlen, `seqused_k`, ragged lengths) run on token-block operands by default: a
per-tensor scale over all documents would be a channel *between* documents. Explicit per-tensor operands run with one
scale per (document, kv head) of every operand, reduced over the document's real rows (FA3's granularity) -- on the
aligned path, and on repacked varlen / batched `seqused_k` as the aligned path on the repacked documents. Explicit
`token_local` operands run on documents too (exactly causal). Its causal channel is the per-tensor one, confined to the document (and kv head): a later token
of a document can move the rounding of that document's earlier tokens, never another document's.

**Aligned varlen (zero-copy).** With `cu_seqlens_*` plus `seqused_q` and `seqused_k`, the caller's packed tensors
already hold every document at a start that is a multiple of 128, so a 128-token tile,
a 32-token scale block, an NVFP4 128-row SF atom and a 16-key V group never straddle two documents. The document tables
(`lp_varlen.LPDocs.aligned_from`) come straight from the caller's tensors (padded start = caller's start =
`cu_seqlens`, true length = `seqused`, allocation = `total`) in one prep launch, and the kernels run on
`(1, total, h, d)` views. Path selection reads only the arguments; the starts (`% 128`), `0 <= seqused <= slot` and
`cu[-1] <= total` are checked by a device-side assert, `total % 128` on the host. The 2-CTA backward's key pair (256
keys, two 128-key tiles) may straddle two documents there: both CTAs iterate the union of the two halves' query
ranges (the pair's MMAs share the Q tile), each masks in its own key tile's document, and a query tile of the other
document is masked whole; the deterministic dQ order counts a pair only at the query tiles one of its halves meets.
Under `two_cta="auto"` the aligned path takes it at `max_seqlen_k >= 32768` (non-deterministic), below that the 1-CTA
backward.

The rows after a document's true length are padding, and they are a **contract**, not something the library
checks: they must be zero, or at least finite and no larger than the document's real rows (no check or zeroing pass is
run). The Q / K / V / dO quantisers read the padding rows of a document's last 128-row tile into its 32-token E4M3
blocks and statistics; under the contract that changes nothing (zero padding: the codes, O, dQ, dK, dV are bitwise those
of the repacked call). The forward writes the whole last Q tile of a document, the rows past `seqused_q` as zeros
(their scale is 0: no extra pass), and the backward writes zero dK / dV for the padding keys of a real key tile (P = 0
there) and zero dQ on every padding row; so a layer stack keeps the padding neutral. Whole padding tiles after a
document's last real tile are never computed by the attention kernels (the document's tile range ends at its true
length); the quantisers do read them, and O / dK / dV there are left as allocated (no zero fill: the aligned path
drops the repacked path's `zeros_like` on O, dK and dV). A document with `seqused_q = 0` and `seqused_k > 0` therefore
gets undefined dK / dV (no query reaches its keys); with a window or a mask_mod the dK / dV zero fill stays.

The aligned backward takes the batched path's fused prologue (`fp4_bwd_autograd.tb_bwd_glue`): the dO quantiser
writes D, the shifted LSE (`+inf` on padding rows, from the document table: `quantize_fp8_blocks(lse_row_end=...)`)
and the zeroed f16 dQ accumulator in its one launch, so the bf16 preprocess kernel, the torch LSE chain and the
accumulator fill do not run.

An in-kernel row predicate exists as an A/B option (`flash_attn.cute.lp_varlen.ALIGNED_ROW_PREDICATE = True`; off by
default): the quantisers (`quant.fp8.quantize_fp8_blocks` / `quant.fp4.quantize_fp4_rows`, argument `row_end`, the
end of the real rows of each 128-token tile) stage padding rows as zero rows (`cp.async` zero-fill or a register
select, no torch op and no extra pass), which makes any padding content -- NaN and inf included -- give exactly the
zero-padding result.

**What can be claimed.** At the standard shapes the default routes carry no key-grouped and no global channel, only
the in-block channels above (reach < 32 rows, about 1e-6 in dQ to 2e-3 in O). The defaults are *not* exactly causal;
`token_local` + `ds_dq_rowlocal="all"` is.

## Varlen coverage

What runs on documents (aligned varlen = `cu_seqlens` + `seqused_*`, in place; repacked = `cu_seqlens` only, copied to
an aligned buffer; batched `seqused_k` = the batch rows as documents). Training and inference unless noted.

| route / option | batched | aligned varlen | repacked varlen | batched `seqused_k` |
|---|---|---|---|---|
| E4M3 P.V routes, token-block operands (default): causal / dense / window / softcap / score_mod / mask_mod / deterministic / `return_lse` | yes | yes | yes | yes |
| `operand_scaling="per_tensor"` (one scale per (document, kv head) on documents) | yes | yes | yes (aligned path on the repacked buffer) | yes |
| `operand_scaling="token_local"` (exactly causal) | yes | yes | yes | yes (no window / softcap / mods on any form: the fold forward) |
| `q_block_2d=True` (`flash_attn_fp4`) | dense | dense | dense | dense (refused under causal / window / mask_mod: rule 2) |
| `operand_scaling="block_32x32"` | yes | no (documents take token blocks; wiring) | no | no |
| `two_cta="on"` (2-CTA backward; `"auto"` from key length 32768, not deterministic) | yes, GQA, `s_q != s_k`, `s_k % 256 == 128` | yes, GQA, deterministic and `total_q != total_k` too (starts % 128) | yes (repacked at 256) | yes |
| `forward_two_cta="on"` (2-CTA forward, bitwise the 1-CTA one) | yes | yes | yes | yes |
| pre-quantised E4M3 (`q_descale` / ...) incl. window / softcap / score_mod / mask_mod | yes | yes | yes (codes repacked) | yes (aligned rows) |
| 4-bit P.V routes (`pv_precision="fp4"`: all-FP4 training, option A inference): causal / dense / `return_lse` | yes | yes | yes (aligned path on the repacked buffer) | dense only (causal refused: queries with no key) |
| 4-bit P.V routes: window / softcap / mods / deterministic | no | no | no | no |
| all-FP4 non-default variants (`causal_safety="key_local"`, `k_channel="cut"`, `v_hadamard`, `all_fp4_stabiliser="eden_rht"`, `fp4_backward_dkdv="fp4"`) | yes | no (refused, named) | no | no |

`two_cta="auto"` keeps documents on the 1-CTA backward below `max_seqlen_k` 32768 and takes the 2-CTA one from there
(non-deterministic: 1.2-9 % faster per step on long documents).
`forward_two_cta="auto"` keeps documents on the 1-CTA forward (the 2-CTA forward measured 1.3-1.6x
slower than the 1-CTA forward with its persistent work list on 16 x 1024 documents). [2026-10-07] The 4-bit P.V
routes run on documents: every whole-tensor statistic of these routes (the all-FP4 backward's E4M3 amax slots,
`max ||V||^2`, `max |dO|`, the dS bound, the diagonal quantiser's tile constants and prefix maxima) is kept per
document, inside the launches that already exist. Each 128-multiple document then equals the same document run alone
as a batched b = 1 call (bitwise; the dense all-FP4 dQ up to its f16 reduce-add order when a document has several key
tiles), and scaling one document's V or dO by 8 leaves every other document bitwise unchanged. (In one batched call the
batch rows do couple: with the stochastic-rounding counter reset, scaling row 1's V or dO by 8 moves row 0's dense dQ by
up to 0.16 of its max and dK by up to 0.055; an earlier figure of 0.49 was mostly SR dither from an un-reset counter.)
On equal-length documents the aligned all-FP4 step costs +9 % over batched (eq16x1024), on skewed lengths it is
22-25x faster than padding to the longest. Causal batched `seqused_k` stays refused on these routes (the first
`s - seqused_k` queries would have no key, as batched `s_q > s_k`), and document rows need `seqused_q <= seqused_k`
under causal (device-side assert).

## Determinism

- The forward is deterministic: the same inputs and plan give bitwise-identical O and LSE.
- `deterministic=False` (default): dQ is accumulated with atomics across key tiles, so its last bits vary from run to
  run; dK and dV are deterministic -- except GQA on the 2-CTA backward (the default from key length 32768, or
  `two_cta="on"`), where each pair computes one q head's dK / dV and the q heads are summed by fp32 atomics.
- `deterministic=True` fixes the dQ reduction order, so dQ is bitwise reproducible as well (and GQA's dK / dV on the
  2-CTA kernel, ordered by q head with upstream's dK / dV semaphores). It takes the 1-CTA kernel with dedicated dQ
  issue / release warps at every length (the 2-CTA ordered reduce measured 0-6 % slower from s_k 32768), except 8-bit
  explicit per-tensor operands, which keep the 2-CTA kernel's ordered reduction; `two_cta="on"` takes the 2-CTA
  kernel (GQA and documents included). The 1-CTA and 2-CTA kernels are each deterministic but reduce in different
  orders, so their dQ differs in the last bits.
- The all-4-bit backward (`backward="fp4"`) has no deterministic dQ; `deterministic=True` is refused there.
- Stochastic rounding (all-4-bit route by default; `sr_ds` on the E4M3 backward) draws a new dither on every call.
  Two calls with identical inputs then differ by the dither, as intended for training. Both the all-4-bit route
  and `sr_ds` take their seeds from counters on the device, so CUDA-graph replays draw new dithers too.
- The default routes make no host synchronisation and can be captured in a CUDA graph.

## Caveats: the fastest path is not always right

Every row below is a configuration that was the fastest, or looked the cheapest, and turned out to be wrong, leaky,
or only conditionally right. The shipped defaults avoid them; several remain available as explicit options. Training
numbers are final validation losses of a byte-level GPT on enwik8 (6000 steps, two to four seeds); lower is better.

| fastest path | what it buys | why it is not the default |
|---|---|---|
| per-tensor operand scales (`operand_scaling="per_tensor"`) | the 4-bit step 5-23 % faster, the 8-bit step 6-10 % | a global causal channel (0.23 in O over the whole prefix); trains no worse, so rejected for causality, not loss. Opt-in only |
| the row-frame P-hat on the 4-bit entry (`p_hat_frame="row"`) | -17 / -21 % of the token-block forward kernel | +0.0015 validation loss, worse on 4 of 4 seeds. Opt-in on `flash_attn_fp4`; on `flash_attn_fp8` the default (+0.0011, within noise) |
| option A, the 4-bit P.V forward | -12 % / -17 % forward on N(0,1) inputs | on trained activations slower and less accurate (O cosine 0.975 vs 0.981); its training backward costs more than it saves (table above). Inference only |
| the 4-bit inference call | 4-bit Q.K | the forward is exp-bound and the 4-bit call pays its quantisation: the 8-bit call measured up to 33 % faster on short causal shapes. Compare both on your shapes ([Performance](#performance)) |
| a 4-bit P-hat with the E4M3 backward | the cheapest training forward | D from the stored O is inconsistent with the backward's P; training collapsed on 3 of 3 seeds. Not offered |
| every backward GEMM in 4-bit (`fp4_backward_dkdv="fp4"`) | the 4-bit tensor-core rate | the slowest backward measured (geomean 571 TFLOP/s against 1329 for the default E4M3 backward and 1015 for BF16), gradients below the accuracy gate |
| a per-tensor dS bound | no in-kernel dS amax | 13x loose on median, flushes small dS: +0.024 / +0.031 validation loss, the largest effect measured. Removed; block dS is the default |
| `ds_dq_rowlocal="diagonal"` | +2-8 % backward instead of +21-38 % | closes nothing (the dS coupling is in every block). Use `"all"` |
| the folded token-block forward (`KernelTuning.token_block_forward="fold"`) | +10-16 % forward cost instead of +21-37 % | leaks 2.3e-2 inside a 32-key block (P times the V factor is rounded to E4M3); untrained. Not the default |
| a 2D (16 x 16) Q scale (`q_block_2d=True`) | one quantisation for both GEMM orientations | leaks in the query direction (Q->O 0.27); worse at every cell measured. Refused under any mask |
| 2-CTA everywhere | half the MMA issue | wins only part of the cells; the shipped rule (`two_cta="auto"`) is measured per route and shape |
| Hadamard off | cheapest and safe on one step | on is the default: 0.1-0.6 % quantiser cost, training within seed noise; note that on synthetic outlier channels it lowers the NVFP4 one-step cosine (0.912 -> 0.581) while helping MXFP4 and 8-bit |
| stochastic rounding everywhere | "safer" rounding | SR on option A's dO bought nothing; SR on the all-4-bit dK operand helps training while lowering the one-step cosine. Only where measured useful |

Two general lessons from the same measurements:
- **One-step accuracy does not certify training.** A backward that tiled its "diagonal" decision differently from
  the forward passed every one-shot gradient gate (dQ cosine 0.945, norm ratios 1.02) and diverged in training
  (gradient norm 0.5 -> 2e5). A per-row identity check, sum_k dS = 0, catches that class; a training run is the only
  certificate.
- **A speed number is a claim only together with its inputs and numerics.** N(0,1) inputs flatter some paths
  (option A, window-only classifiers) and synthetic outliers mislead the other way; calibrated scales flatter
  baselines that need them; free-running clocks, power caps and neighbouring jobs move results by several percent;
  and the CuTe DSL version alone moved the 4-bit dense forward by ~19 % with bitwise-identical output.

## Requirements

- **GPU:** NVIDIA Blackwell, SM100 family (`sm_100a`; developed and tested on B200). The low-precision kernels use
  tcgen05 block-scaled MMAs and TMEM and do not run on Hopper or older GPUs.
- **CuTe DSL:** `nvidia-cutlass-dsl` 4.8 (the version tested). Codegen is version-sensitive: other versions may
  change performance (by up to ~20 % on some kernels) while producing identical results.
- **PyTorch:** tested with 2.14 (CUDA 13.0). Triton (3.8 tested) for the Hadamard fallback pass.
- `quack-kernels`, `apache-tvm-ffi` and `einops`, as listed in `flash_attn/cute/pyproject.toml`.
- Install from the repository: `pip install -e "flash_attn/cute[dev]"`.
- Tests: `tests/cute/test_lowprec_*.py`, `test_fp4_*.py`, `test_quant_*.py`, `test_prequant.py`, `test_prequant_fa3.py`,
  `test_own_quantisers.py`, `test_row_p_defaults.py`, `test_exact_mix.py`, `test_scaling_consistency.py`,
  `test_2cta_dp_park_race.py`, `test_fuseall_prep.py` and `test_bwd_input_immutability.py`. Compilation dominates
  their run time; `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` keeps compiled kernels between runs. `test_equivalence`
  and the recorded-key test compare against a golden capture when `LOWPREC_GOLDEN_GBIT` / `LOWPREC_GOLDEN_GKEY` point
  at one, and skip otherwise.

## Performance

Measured on one B200 (SM100, free-running clocks, median of 3 paired rounds after a warm-up round; each cell's jobs
run back to back on one card). Times are milliseconds; "best baseline" is the fastest of FlashAttention-4 BF16 / FP8,
cuDNN BF16 / FP8 / MXFP8, FlashInfer BF16 / FP8 (all backends), PyTorch SDPA and the other published low-precision
kernels at that cell. Baselines quantise outside the timed region for forward / backward and inside it for step /
inference; the low-precision entries always include their own quantisation in step / inference.

Cells: A1 = b4 s4096 h16 d128 causal, A2 = the same dense, A3 = b1 s32768 h16 d128 causal, A4 = the same dense;
S8c / S8d = b2 s8192 h16 d128 causal / dense; S16c / S16d = b1 s16384; D64c / D64d = head_dim 64 at s4096;
GQAc / GQAd = h16 / hk4 at s4096; W1k = sliding window 1024.

**Training step (forward + backward, quantisation included), default entries:**

| cell | `flash_attn_fp4` | `flash_attn_fp8` | best baseline | fastest low-precision / best baseline |
|---|---|---|---|---|
| A1 | 0.866 | 0.865 | cuDNN FP8 0.873 | 0.986 |
| A2 | 1.422 | 1.416 | cuDNN FP8 1.409 | 1.005 |
| A3 | 9.766 | 9.745 | cuDNN FP8 10.549 | 0.866 |
| A4 | 18.982 | 19.141 | cuDNN FP8 19.997 | 0.895 |
| S8c | 1.444 | 1.449 | cuDNN FP8 1.527 | 0.946 |
| S16c | 2.584 | 2.603 | cuDNN FP8 2.797 | 0.895 |
| D64c | 0.697 | 0.650 | PyTorch SDPA (cuDNN) 0.671 | 0.968 |
| GQAc | 0.815 | 0.797 | cuDNN FP8 0.951 | 0.838 |
| W1k | 0.666 | 0.647 | FlashAttention-4 BF16 0.652 | 0.993 |

Across all 13 cells the fastest low-precision configuration beats every baseline on the training step in 12 cells
(0.84-0.99) and ties cuDNN FP8 at A2; the backward alone is ahead in all 13 cells (0.72-0.94). The forward alone is
ahead in 9 cells; it trails the 4-bit forward of Hu et al. on dense cells, FlashInfer FP8 at S8d (+3.7 %) and cuDNN
FP8 at head_dim 64 causal (+5.5 %). Inference with quantisation included is ahead only at long sequences: at short
sequences the quantisers' fixed cost dominates (A1 +12 %, W1k +48 %).

**Accuracy at A1** (cosine similarity to an fp32 reference on the same bf16 inputs, O / dQ / dK / dV):

| entry | O / dQ / dK / dV |
|---|---|
| `flash_attn_fp8` | 0.9988 / 0.9979 / 0.9979 / 0.9987 |
| `flash_attn_fp4` | 0.9920 / 0.9906 / 0.9906 / 0.9919 |
| `flash_attn_fp4(backward="fp4")` (all-4-bit) | 0.9810 / 0.945 / 0.968 / 0.9918 |
| cuDNN FP8 (reference point) | 0.9985 / 0.9975 / 0.9975 / 0.9984 |

The all-4-bit backward's dQ / dK vary slightly from call to call because each call draws fresh stochastic-rounding
dither.

Reproduce with the benchmark script in this repository:

```sh
python benchmarks/benchmark_lowprec_attn.py --backend lp-fp4,lp-fp8,lp-fp4-all,fa4-bf16 --exact \
    --mode fwd,bwd,step,infer --shapes A1,A2,A3,A4
```

`step` and `infer` time the public call including quantisation; `fwd` / `bwd` time the kernels alone on operands
quantised outside the timed region. Lock the SM clock for stable numbers and quote every number with its inputs,
toolchain and clock.

# Where the low-precision attention code lives

All paths are under `flash_attn/cute/`. The BF16 / FP16 FlashAttention-4 code next to it is upstream's and is
unchanged apart from small additive helpers. How to use the functions: [lowprec_usage.md](lowprec_usage.md).

## Public API: `lowprec/`

| file | contents |
|---|---|
| `lowprec/api.py` | `flash_attn_fp4`, `flash_attn_fp8`: argument checks, plan lookup, the single autograd function |
| `lowprec/options.py` | `FP4AttnOptions`, `FP8AttnOptions` |
| `lowprec/plan.py` | `resolve_fp4`, `resolve_fp8`, `AttnPlan`: every rule, refusal and default decision, with its log |
| `lowprec/specs.py` | the kernels' compile-time settings (one frozen spec per kernel family), built once per plan |
| `lowprec/tuning.py` | `KernelTuning` (advanced performance constants) |
| `lowprec/_routes.py` | the forward / backward bodies of each precision route, called by the autograd function |
| `__init__.py` | re-exports the public names from `flash_attn.cute` |

## Kernels (CuTe DSL)

| file | kernel |
|---|---|
| `flash_fwd_sm100_fp4.py` | the forward: block-scaled tcgen05 Q.K and P.V, NVFP4 / E4M3 operands, optional CTA-pair forward |
| `flash_bwd_sm100_fp4.py` | the 1-CTA backward (E4M3 operands, S recomputed from the forward's codes) |
| `flash_bwd_sm100_2cta8.py`, `bwd_2cta8_side.py` | the 2-CTA cluster backward and its side kernels |
| `flash_bwd_sm100_fp4q.py` | the all-4-bit query-row backward (`backward="fp4"`) |
| `dq_postprocess_cute.py` | dQ accumulation post-processing |

## Drivers (host side: compile, cache, launch)

| file | drives |
|---|---|
| `interface_fwd_fp4.py` | the forward kernel |
| `interface_bwd_fp4.py` | the 1-CTA backward |
| `interface_bwd_2cta8.py` | the 2-CTA backward |
| `interface_bwd_fp4q.py` | the all-4-bit backward |
| `fp4_bwd_autograd.py` | shared host helpers of the backward routes |
| `pv4_route.py`, `pv4_repro.py` | the 4-bit P.V inference forward (`pv_precision="fp4"` without grad) |

## Quantisers and helpers

| file | contents |
|---|---|
| `quant/fp8.py`, `quant/fp4.py` | fused E4M3 / NVFP4 quantisers (token-block, block 32x32, per-tensor scaling; RoPE; Hadamard) |
| `quant/fp4_diagonal.py` | the causal-diagonal quantiser of the all-4-bit route |
| `quant/_rotations.py`, `quant/_tcgen05.py`, `quant/_fp8_tcgen05.py`, `quant/_fp8_nvfp4.py` | Hadamard rotations (tensor-core and opt-in tcgen05), mixed-format outputs |
| `qk_hadamard.py`, `qk_hadamard_cute.py` | the Q/K Hadamard rotation and its backward un-rotation |
| `prequant.py` | `QuantizedOperand` and helpers for caller-quantised Q / K (`flash_attn_fp8`) |
| `device_scales.py`, `mx_scale_utils.py` | GPU-side scale computation, UE8M0 / UE4M3 scale-factor layouts |
| `lp_masks.py`, `lp_varlen.py`, `lp_bwd_tiles.py` | masks, varlen repacking, backward tile scheduling |
| `lp_docmaps.py` | `DocMaps`: precomputed packed-document maps (`doc_maps=`), the CPU / GPU builders, the fused-gather autograd node |
| `fp4q_diag_triton.py`, `fp4q_row_scales.py`, `nvfp4_2d_triton.py` | Triton helpers of the all-4-bit route |

## Tests and benchmark

| path | covers |
|---|---|
| `tests/cute/test_lowprec_api.py` | API rules and refusals, plans, equivalence, graph capture, no host sync, causal-leak probes |
| `tests/cute/test_lowprec_spec_keys.py`, `test_lowprec_compile_keys.py`, `test_lowprec_options_reach.py` | every option reaches its kernel and its compile key |
| `tests/cute/test_fp4_*.py`, `test_lowprec_causal_safe_v.py`, `test_scaling_consistency.py` | route accuracy, GQA, 2-CTA, all-4-bit, causal safety |
| `tests/cute/test_quant_*.py`, `test_own_quantisers.py` | quantiser bit-exactness |
| `tests/cute/test_lowprec_docmaps.py` | `doc_maps=`: tables, bitwise the repacking path, graph capture, no host sync, data-loader workers |
| `benchmarks/benchmark_lowprec_attn.py` | timing and accuracy against BF16 FlashAttention-4, cuDNN, FlashInfer and other baselines |
| `docs/lowprec.md` | API reference, defaults, routes, causal safety, determinism, caveats, performance |

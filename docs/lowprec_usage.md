# Using the low-precision attention (FP4 / FP8)

A practical guide: install, check the setup, replace an attention call, train, run inference, and pick options.
The full argument reference is [lowprec.md](lowprec.md#api); where the code lives is in [lowprec_code_map.md](lowprec_code_map.md).

## 1. Requirements

- An SM100-family GPU (developed and tested on B200). The low-precision kernels do not run on Hopper or older.
- CUDA 13.0, PyTorch 2.14, `nvidia-cutlass-dsl` 4.8 (tested versions), Triton.
- Inputs in bfloat16 or float16, layout `(batch, seqlen, heads, head_dim)`, last dimension contiguous.

## 2. Install

From the root of this repository:

```sh
pip install -e "flash_attn/cute[dev]"
```

This installs the `flash-attn-4` package (`flash_attn.cute`), including the `lowprec/` and `quant/` subpackages. The
FA2 CUDA extension is not needed to import `flash_attn.cute`.

## 3. Check the setup

```python
import torch
from flash_attn.cute import flash_attn_fp4, flash_attn_fp8

q, k, v = (torch.randn(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
out = flash_attn_fp4(q, k, v, causal=True)
out.sum().backward()
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), is_causal=True).transpose(1, 2)
print(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0))   # about 0.99
```

The first call of each configuration compiles its kernels (seconds to a minute); later calls with the same
configuration reuse them. To keep compiled kernels across processes, set `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
(cache in `/tmp/$USER/flash_attention_cute_dsl_cache/`).

## 4. Use it in place of an attention call

The functions take and return the same dtypes as their inputs and are differentiable, so they replace an attention
call directly. Note the layout: `(b, s, h, d)`, whereas `torch.nn.functional.scaled_dot_product_attention` takes
`(b, h, s, d)`.

```python
import torch
from torch import nn
from flash_attn.cute import flash_attn_fp8


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.h, self.h_kv = n_heads, n_kv_heads or n_heads
        self.d = dim // n_heads
        self.wq = nn.Linear(dim, self.h * self.d, bias=False)
        self.wk = nn.Linear(dim, self.h_kv * self.d, bias=False)
        self.wv = nn.Linear(dim, self.h_kv * self.d, bias=False)
        self.wo = nn.Linear(self.h * self.d, dim, bias=False)

    def forward(self, x):                                   # x: (b, s, dim), bf16
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.h, self.d)
        k = self.wk(x).view(b, s, self.h_kv, self.d)
        v = self.wv(x).view(b, s, self.h_kv, self.d)
        o = flash_attn_fp8(q, k, v, causal=True)            # (b, s, h, d), bf16; or flash_attn_fp4(...)
        return self.wo(o.reshape(b, s, self.h * self.d))
```

Quantisation happens inside the call (fused kernels, no host synchronisation), so the call can be captured in a CUDA
graph. Both entries support MHA, GQA and MQA (`h % h_kv == 0`, as above).

### Which entry to use

| entry | forward | backward | accuracy at b4 s4096 h16 d128 causal (cos O / dQ) |
|---|---|---|---|
| `flash_attn_fp8(q, k, v, ...)` | E4M3 everywhere | E4M3 | 0.9988 / 0.9979 |
| `flash_attn_fp4(q, k, v, ...)` | NVFP4 Q.K, E4M3 P.V | E4M3 operands, S from the forward's 4-bit codes | 0.9920 / 0.9906 |
| `flash_attn_fp4(..., pv_precision="fp4", backward="fp4")` | all 4-bit | 4-bit query-row backward with stochastic rounding (d = 128) | 0.9810 / 0.945 |
| `flash_attn_fp4(..., pv_precision="fp4")`, no grad | 4-bit P.V | (inference only) | O 0.980 |

Speed against cuDNN, FlashAttention-4 BF16 and FlashInfer is in [lowprec.md, Performance](lowprec.md#performance).

## 5. Recipes

### Training step

```python
out = flash_attn_fp4(q, k, v, causal=True)
loss = loss_fn(out)
loss.backward()                         # dq, dk, dv in q / k / v dtype
```

### Inference

```python
with torch.inference_mode():
    out = flash_attn_fp8(q, k, v, causal=True)
    out4 = flash_attn_fp4(q, k, v, causal=True, pv_precision="fp4")   # 4-bit P.V; refused if an input requires grad
```

### Grouped-query attention

```python
q = torch.randn(4, 4096, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(4, 4096, 4, 128, device="cuda", dtype=torch.bfloat16)   # 4 KV heads shared by 16 query heads
v = torch.randn(4, 4096, 4, 128, device="cuda", dtype=torch.bfloat16)
out = flash_attn_fp8(q, k, v, causal=True)
```

### Variable-length batches (packed sequences)

**Aligned varlen (zero-copy, the fast path).** Lay every document out at a start that is a multiple of 128 in the
packed `(total, h, d)` tensors and pass the starts as `cu_seqlens_*` and the true lengths as `seqused_q` / `seqused_k`
(FlashAttention-3's convention). The kernels then run on views of your tensors: nothing is copied.

```python
def aligned_layout(lens, align=128, device="cuda"):
    """Document lengths (host ints, known to the data loader) -> (starts (b + 1,), seqused (b,), total), int32."""
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + -(-n // align) * align)     # each slot rounded up to `align` tokens
    return (torch.tensor(starts, dtype=torch.int32, device=device),
            torch.tensor(lens, dtype=torch.int32, device=device), starts[-1])

lens = [1000, 3000, 96]
cu, used, total = aligned_layout(lens)        # starts [0, 1024, 4096, 4224], total 4224
# q, k, v: (total, h, d); document i's tokens at rows cu[i] .. cu[i] + lens[i] - 1, anything in the rows after them
out = flash_attn_fp8(q, k, v, causal=True, cu_seqlens_q=cu, cu_seqlens_k=cu, seqused_q=used, seqused_k=used,
                     max_seqlen_q=3000, max_seqlen_k=3000)
```

If your documents arrive packed back to back, move them to the aligned layout once, where the batch is built (the
library's repacking does exactly this copy on every call, forward and backward):

```python
def pad_to_aligned(x, cu_packed, lens, align=128):
    """packed (sum(lens), ...) with boundaries cu_packed -> (the aligned (total, ...) tensor, starts, seqused)."""
    cu, used, total = aligned_layout(lens, align, x.device)
    doc = torch.repeat_interleave(torch.arange(len(lens), device=x.device), used.long(), output_size=x.shape[0])
    rows = cu[:-1].long()[doc] + torch.arange(x.shape[0], device=x.device) - cu_packed[:-1].long()[doc]
    out = x.new_zeros((total,) + tuple(x.shape[1:]))
    return out.index_copy_(0, rows, x), cu, used
```

The rules of the aligned form:
- `cu_seqlens_q` / `cu_seqlens_k`: `(b + 1,)` int32, every entry a multiple of 128 (the 2-CTA backward too),
  `cu[-1] <= total`; `total % 128 == 0` (checked on the host from the shapes).
- `seqused_q` / `seqused_k`: `(b,)` int32, `0 <= seqused[i] <= cu[i + 1] - cu[i]`. Pass both or neither.
- The starts and lengths are checked by a device-side assert (`torch._assert_async`): no host sync, so a violation
  surfaces as a CUDA device-side assert at the next synchronising call (and the CUDA context is lost).
- **The padding contract.** Rows between `cu[i] + seqused[i]` and `cu[i + 1]` are padding, and they must be zero --
  or at least finite and no larger than the real rows of the document. Nothing checks this (a check or a zeroing pass
  would cost a pass over the tensors). The quantisers read the padding rows of a document's last 128-row tile into its
  32-token scale blocks, so a NaN / inf there poisons that document, and a value larger than the real ones coarsens the
  scale of the real rows that share its block. Zero padding gives results bitwise equal to the repacked call.
- The library keeps the contract on its own outputs where that is free: `out`, `dk`, `dv` are written as zeros on the
  padding rows of each document's last 128-row tile, and `dq` is zero on every padding row (so a stack of layers keeps
  the padding neutral). Whole padding tiles after a document's last real tile are never computed: `out`, `dk`, `dv`
  there are left as allocated (undefined; no fill pass is run). A document with `seqused_q = 0` and `seqused_k > 0`
  gets undefined `dk` / `dv` on its key rows (no query reaches them, and nothing zero-fills). The returned `lse` is
  unspecified on padding rows.
- The backward (`two_cta="auto"`) is 2-CTA when `max_seqlen_k >= 32768` and the call is not deterministic, else
  1-CTA; `two_cta="on"` takes the 2-CTA kernel at any length (d = 128; MHA and GQA; deterministic too, with a
  document-local dQ order). Its key pair (256 keys) may straddle two documents: each CTA of the pair masks in its own
  key tile's document, so the 128-aligned starts suffice. The forward is 1-CTA by default; `forward_two_cta="on"` runs
  it on CTA pairs (bitwise the 1-CTA forward).
- `return_lse=True` works and returns `lse` as `(h, total_q)`.
- `operand_scaling="token_local"` (exact causality) and, on dense calls, `q_block_2d=True` run on documents (no
  16-query block spans two documents; `q_block_2d` stays refused under causal / window / mask_mod).
- **Per-tensor operands per document.** `options.operand_scaling="per_tensor"` (explicit opt-in, as for batched calls)
  runs on the aligned path with one E4M3 scale per (document, kv head) of every operand -- Q, K, V, dO (FA3's
  `(b, h_kv)` granularity), never one scale across documents. The scales are reduced on the GPU over each document's
  real rows only (rows at or past `seqused` are left out of every scale), in the same quantiser launches as the
  batched per-tensor call; the forward and backward kernels read each work tile's (document, kv head) scales. The
  backward is 1-CTA (`two_cta="on"` is refused); `flash_attn_fp4` serves it at head_dim 128. On `flash_attn_fp8` the
  per-tensor quantisers read every padding row as a zero row (so it also tolerates NaN / inf padding); the 4-bit route's
  NVFP4 / E4M3 Q / K codes follow the padding contract. Repacked varlen (no `seqused_*`) and batched `seqused_k` run
  per-tensor operands as the aligned path on the repacked (zero-padded) documents.

**Repacked varlen.** `cu_seqlens_*` without `seqused_*` take any document boundaries: the documents are copied into
an aligned buffer on the GPU and copied back (one extra read and write of every input, output and gradient, up to
255 padding rows per document). The first such call in a process warns once and shows the aligned call. Everything the
aligned form serves runs here too, `return_lse` included (`(h, total_q)`, gathered back to the packed rows);
the 2-CTA backward (`two_cta="on"`, or `"auto"` at `max_seqlen_k >= 32768`) repacks at 256-aligned starts.

```python
# three documents of lengths 1000, 3000, 96, packed along dim 0: q, k, v are (total, h, d)
lens = torch.tensor([1000, 3000, 96], device="cuda", dtype=torch.int32)
cu = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))      # [0, 1000, 4000, 4096]
out = flash_attn_fp8(q, k, v, causal=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=3000, max_seqlen_k=3000)
```

**Packed documents with precomputed maps (`doc_maps=`).** A training loop that packs documents back to back can keep
its token buffer packed and still run the aligned kernels, without the per-call index building of the repacked form:
build a `DocMaps` once per microbatch -- on the CPU, in the data loader's worker (numpy, no CUDA) -- move it to the GPU
with the batch, and pass it to every layer. Each call runs the aligned path with the precomputed tables (no prep
launch, no device-side assert, no host sync, the scheduling work lists built once per microbatch). On the default
token-block routes nothing is copied: the quantisers read Q / K / V / dO through the row map, the forward writes O and
the backward's epilogues write dQ / dK / dV straight into the packed rows. The other routes (per-tensor operands, the
4-bit P.V routes, pre-quantised inputs, padded head dims) gather the rows into the maps' aligned buffer and back.
Outputs are bitwise the repacked form's either way.

```python
from flash_attn.cute.lowprec import DocMaps, flash_attn_fp8

# data loader worker (CPU): cu = packed document boundaries of this microbatch, (n + 1,) with cu[-1] = T
maps = DocMaps.from_cu_seqlens(cu, max_num_documents=256, max_seqlen=2048)  # fixed shapes for every microbatch
# main process: one non-blocking copy (DataLoader(pin_memory=True) pins it: DocMaps has pin_memory())
maps = maps.to("cuda", non_blocking=True)
out = flash_attn_fp8(q, k, v, causal=True, doc_maps=maps)       # q, k, v: (T, h, d) packed; every layer, fwd + bwd
```

- `max_num_documents` (N): a fixed capacity; zero-length trailing documents fill it, so every microbatch's maps have
  the same shapes. `padded_size` (default `round_up(T + N * 127, 256)`, a bound that always holds): the rows of the
  aligned buffer; the quantisers and the gathers touch all of them, so a tighter bound (e.g. from the data's
  document-count distribution) is cheaper. `max_seqlen` (default the longest document): the kernels' grid bound; pass
  a fixed value for CUDA graphs. All three are host facts: equal values give the same plan, compile keys and graph.
- CUDA graphs: capture with one static `DocMaps` on the GPU and refresh it in place before each replay
  (`static.copy_(next_maps)`); the work lists derived from the maps are built inside the graph.
- `DocMaps.from_cu_seqlens` also takes CUDA `cu_seqlens` (built with GPU ops and one launch, no host read; pass
  `total_q` and `max_seqlen`).
- Not with `seqused_k` or the other varlen arguments (the maps carry the documents); the pre-quantised E4M3 inputs
  take `doc_maps` (see "Pre-quantised E4M3 inputs").
- `forward_two_cta="on"` (the forward on CTA pairs) runs on the packed rows of compact maps too: each CTA of the pair
  builds its scale-factor atoms from the dense per-row tables, O is stored by TMA tiles plus a document's partial last
  tile by row copies; O / LSE bitwise the fused pair's (and the 1-CTA forward's). At a large capacity (N > 3 T / 128,
  e.g. N = T) the 1-CTA packed forward runs instead (same O / LSE): the pair's grid and tile decode scale with N.
  `return_lse=True` returns `(h, T)`.
- Stochastic rounding (the all-4-bit backward, `sr_ds`) draws its dither from each element's position in the aligned
  buffer: it equals the repacked form's when `padded_size` is the repacked allocation, `round_up(T + n * 127, 128)`
  for `n` documents, and is an equally distributed, different dither otherwise.

Varlen, `seqused_k`, `window_size`, `softcap`, `score_mod` and `mask_mod` run on `flash_attn_fp8` and on
`flash_attn_fp4` with its default `pv_precision="fp8"`; the 4-bit P.V routes (`pv_precision="fp4"`) take `causal`
only (no window / softcap / mods), on batched calls and on documents (aligned and repacked; batched `seqused_k` dense
only). Coverage on documents (details in `lowprec.md`, "Varlen coverage"):

| route / option | batched | aligned varlen | repacked varlen | batched `seqused_k` |
|---|---|---|---|---|
| E4M3 P.V routes, token-block operands (default): causal / dense / window / softcap / score_mod / mask_mod / deterministic / `return_lse` | yes | yes | yes | yes |
| `operand_scaling="per_tensor"` (one scale per (document, kv head) on documents) | yes | yes | yes (aligned path on the repacked buffer) | yes |
| `operand_scaling="token_local"` (exactly causal) | yes | yes | yes | yes (no window / softcap / mods on any form: the fold forward) |
| `q_block_2d=True` (`flash_attn_fp4`) | dense | dense | dense | dense (refused under causal / window / mask_mod: rule 2) |
| `operand_scaling="block_32x32"` | yes | no (documents take token blocks; wiring) | no | no |
| `two_cta="on"` (2-CTA backward; `"auto"` at key length >= 32768, not deterministic) | yes, GQA, `s_q != s_k` and `s_k % 256 == 128` too | yes, GQA, deterministic and `total_q != total_k` too (starts % 128: the key pair straddles documents) | yes (repacked at 256) | yes |
| `forward_two_cta="on"` (2-CTA forward, bitwise the 1-CTA one) | yes | yes | yes | yes |
| pre-quantised E4M3 (`q_descale` / ...) incl. window / softcap / score_mod / mask_mod | yes | yes | yes (codes repacked) | yes (aligned rows) |
| 4-bit P.V routes (`pv_precision="fp4"`: all-FP4 training, option A inference): causal / dense / `return_lse` | yes | yes | yes (aligned path on the repacked buffer) | dense only (causal refused: queries with no key) |
| 4-bit P.V routes: window / softcap / mods / deterministic | no | no | no | no |
| all-FP4 non-default variants (`causal_safety="key_local"`, `k_channel="cut"`, `v_hadamard`, `all_fp4_stabiliser="eden_rht"`, `fp4_backward_dkdv="fp4"`) | yes | no (refused, named) | no | no |

### Sliding window, soft-capping

```python
out = flash_attn_fp8(q, k, v, causal=True, window_size=(1024, 0))   # each query sees the previous 1024 keys
out = flash_attn_fp8(q, k, v, causal=True, softcap=30.0)
```

### Bitwise-reproducible gradients

```python
out = flash_attn_fp8(q, k, v, causal=True, deterministic=True)   # fixed dQ summation order (not on backward="fp4")
```

### Log-sum-exp

```python
out, lse = flash_attn_fp8(q, k, v, causal=True, return_lse=True)   # lse: (b, h, s_q) float32, not differentiable
```

### Options

Advanced settings go in an options object; every field and value is in [lowprec.md](lowprec.md#fp4attnoptions).

```python
from flash_attn.cute import FP4AttnOptions, FP8AttnOptions

# exact causality: per-token operand scales and row-local dS exponents (slower; see lowprec.md, Causal safety)
opts = FP8AttnOptions(operand_scaling="token_local", ds_dq_rowlocal="all")
out = flash_attn_fp8(q, k, v, causal=True, options=opts)

# turn the Q/K Hadamard rotation off, or keep the backward on the 1-CTA kernel
out = flash_attn_fp4(q, k, v, causal=True, qk_hadamard=False, options=FP4AttnOptions(two_cta="off"))

# the all-4-bit training route
out = flash_attn_fp4(q, k, v, causal=True, pv_precision="fp4", backward="fp4")
```

Pre-quantised Q / K for `flash_attn_fp8` (`q_quantized=`, `k_quantized=`, from `flash_attn.cute.prequant`) are
described in [lowprec.md, Entry points](lowprec.md#entry-points); they force per-tensor operands.

### Pre-quantised E4M3 inputs (FA3's `q_descale` / `k_descale` / `v_descale`)

When the projections already emit E4M3, hand the codes over with one float32 dequantisation scale per (batch row,
kv head) -- FA3's convention, `x ~= code * descale` -- and the Q / K / V quantisers do not run:

```python
import torch
from flash_attn.cute import flash_attn_fp8

def to_e4m3(x, h_kv):
    """(b, s, h, d) -> E4M3 codes + (b, h_kv) float32 descales; the q heads of a GQA group share a scale."""
    b, s, h, d = x.shape
    amax = x.float().abs().view(b, s, h_kv, h // h_kv, d).amax(dim=(1, 3, 4))          # (b, h_kv)
    descale = (amax / 448.0).clamp_min(1e-12)                                           # x ~= code * descale
    codes = (x.float() / descale.repeat_interleave(h // h_kv, 1)[:, None, :, None]).clamp(-448, 448)
    return codes.to(torch.float8_e4m3fn), descale

h_kv = k.shape[2]
(q8, dq), (k8, dk), (v8, dv) = (to_e4m3(t, h_kv) for t in (q, k, v))
out = flash_attn_fp8(q8, k8, v8, causal=True, q_descale=dq, k_descale=dk, v_descale=dv)   # bf16 (b, s, h, d)
```

- Varlen: the aligned form (documents at starts that are multiples of 128, `seqused_q` / `seqused_k`), descales
  `(n_docs, h_kv)` reduced over each document's real rows; padding rows: the usual contract (zero, or finite and no
  larger than the real rows). The library's own quantiser makes exactly the internal path's codes and scales:
  `flash_attn.cute.quant.fp8.quantize_fp8_per_tensor([q, k, v])` (batched: `scales[i]` broadcast to `(b, h_kv)`), or
  with `documents=LPDocs.aligned_from(...).doc_quant_args(("q", "k", "k"), h_kv)` on the `(1, total, h, d)` views
  (varlen: `doc_scales[i][:n_docs]`); pass its float64 scales as they are for a bitwise hand-over.
- Served: batched with `s % 128 == 0`, aligned varlen, repacked varlen (the codes are copied to an aligned buffer)
  and batched `seqused_k`; head_dim 64 / 128; causal / dense; window / softcap / score_mod / mask_mod; MHA / GQA. Not
  on `flash_attn_fp4` (its Q.K^T is NVFP4).
- Packed documents (`doc_maps=`): the codes stay in the caller's packed rows -- `q8` `(T, h, d)`, `k8` /
  `v8` `(T, h_kv, d)`, row for row the maps' documents, no alignment, no channel-major V -- with descales of shape
  `(max_num_documents, h_kv)` (row i: document i; rows past the microbatch's document count are not read). Nothing is
  copied: the forward reads every tile at its document's packed start (V as its row codes) and writes O / the LSE
  there; outputs are bitwise the repacked form's. Plain causal / dense (no window / softcap / mods); the codes must be
  finite. `grad_proxies` train on compact maps (no `padded_size`): the backward reads the codes, quantises dO and
  writes dQ / dK / dV in the packed rows.
- Training: autograd cannot hand a bf16 gradient to an E4M3 tensor, so pass detached codes plus the bf16 tensors that
  should receive the gradients:

```python
out = flash_attn_fp8(q8, k8, v8, causal=True, q_descale=dq, k_descale=dk, v_descale=dv,
                     grad_proxies=(q, k, v))   # q, k, v: bf16, requires_grad; codes made from them without autograd
out.backward(dout)                             # q.grad, k.grad, v.grad: the gradients at the dequantised codes
```

  The backward reuses the caller's codes and quantises dO itself (one scale per (document, kv head)).

### Activation checkpointing and `torch.compile` (the registered ops)

`flash_attn_fp4` / `flash_attn_fp8` run their attention node as the `torch.library` ops `lowbit_attn::fp4` /
`lowbit_attn::fp8` (registered on `import flash_attn.cute.lowprec`; backward ops `lowbit_attn::fp4_bwd` / `fp8_bwd`).
A selective activation-checkpointing policy can therefore save the attention instead of re-running it in the
backward:

```python
from functools import partial
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

SAVE = {torch.ops.lowbit_attn.fp4.default, torch.ops.lowbit_attn.fp8.default}
policy = lambda ctx, op, *a, **k: CheckpointPolicy.MUST_SAVE if op in SAVE else CheckpointPolicy.PREFER_RECOMPUTE
y = checkpoint(block, x, use_reentrant=False, context_fn=partial(create_selective_checkpoint_contexts, policy))
```

The saved outputs are the attention output, its LSE and the state the backward reads (the same tensors the eager
autograd node keeps); the forward kernel then runs once per step. `torch.compile` traces the ops through their fake
implementation; a varlen call (document tables) must have run once eagerly with the same shapes first. The op is the
library's autograd node: the varlen repacking around it stays ordinary torch ops. Not through the op (the previous
`autograd.Function` path, same results): `score_mod` / `score_mod_bwd` / `mask_mod` / `aux_tensors`, and
pre-quantised inputs (`q_quantized` / `k_quantized` / `q_scale_fused`, `q_descale` / `k_descale` / `v_descale` /
`grad_proxies`). The op arguments name the resolved plan by a process-local integer handle, so a traced graph is
not portable to another process; the op's saved-state outputs follow the library's internals, which inductor's
on-disk cache does not key on: after upgrading the library, clear that cache (a stale entry fails its metadata
assert, it does not run silently).

### See what a call will run

```python
from flash_attn.cute import resolve_fp4

plan = resolve_fp4(q, k, v, causal=True)   # same arguments as flash_attn_fp4; launches nothing
print(plan.route)                          # e.g. fp4.e4m3_pv
print(plan.log())                          # every resolved choice and why
```

## 6. All arguments and their defaults

Every argument except `q`, `k`, `v` is keyword-only. **Default** is what you get when you leave the argument out;
for `"auto"` defaults, the column also says what `"auto"` resolves to. `resolve_fp4(...)` / `resolve_fp8(...)` print
the resolved value of every `"auto"` for a concrete call (`plan.log()`).

### 6.1 `flash_attn_fp4(q, k, v, ...)` and `flash_attn_fp8(q, k, v, ...)`: common arguments

| argument | default | other values | what it does |
|---|---|---|---|
| `q`, `k`, `v` | (required) | | bf16 / fp16 tensors `(b, s, h, d)` (`k`, `v`: `h_kv` heads, `h % h_kv == 0`); packed `(total, h, d)` with `cu_seqlens_*` |
| `softmax_scale` | `None` = `1 / sqrt(d)` | any float | score scale |
| `causal` | `False` | `True` | bottom-right aligned causal mask |
| `window_size` | `(None, None)` = no window | `(left, right)` ints | sliding-window attention (E4M3 P.V routes) |
| `softcap` | `0.0` = off | float > 0 | tanh soft-capping of the scores (E4M3 P.V routes) |
| `score_mod` | `None` | CuTe DSL callable | score modifier, upstream signature (E4M3 P.V routes) |
| `score_mod_bwd` | `None` | CuTe DSL callable | backward of `score_mod` on dS; required with `score_mod` when an input requires grad |
| `mask_mod` | `None` | CuTe DSL callable | mask modifier, upstream signature (E4M3 P.V routes) |
| `aux_tensors` | `None` | tuple of tensors | extra tensors for `score_mod` / `mask_mod` |
| `cu_seqlens_q`, `cu_seqlens_k` | `None` | int32 `(n_docs + 1,)` | variable-length (packed) batches: document starts, multiples of 128, with `seqused_q` / `seqused_k` (aligned, zero-copy); cumulative lengths without them (repacked, warned once) |
| `max_seqlen_q`, `max_seqlen_k` | `None` | int | longest document, with `cu_seqlens_*` |
| `doc_maps` | `None` | `DocMaps` | packed `(T, h, d)` documents with precomputed maps (instead of `cu_seqlens_*` / `seqused_*` / `max_seqlen_*`): the aligned path on a gathered buffer, no per-call index building ("Packed documents with precomputed maps") |
| `seqused_q` | `None` | int32 `(n_docs,)` | aligned varlen: each document's true query length (with `cu_seqlens_*` and `seqused_k` only) |
| `seqused_k` | `None` | int32 `(b,)` / `(n_docs,)` | without `cu_seqlens_*`: number of used keys per batch row; with them (and `seqused_q`): each document's true key length |
| `deterministic` | `False` | `True` | bitwise-reproducible dQ (fixed summation order); not on `backward="fp4"` |
| `return_lse` | `False` | `True` | also return the log-sum-exp `(b, h, s_q)` float32 (`(h, total_q)` under varlen; aligned: padding rows unspecified), not differentiable |
| `qk_hadamard` | `None` = **on** wherever it can run (d = 128, s % 128 = 0, not `token_local`, not the all-4-bit route), with a log line where it falls back | `True` (required; refused where it cannot run), `False` (off) | random-sign Hadamard rotation of Q and K along d, fused into the quantisers |

### 6.2 Arguments only of `flash_attn_fp4`

| argument | default | other values | what it does |
|---|---|---|---|
| `pv_precision` | `"fp8"` (E4M3 P-hat x E4M3 V) | `"fp4"` (NVFP4 P-hat x NVFP4 V) | precision of the P.V product. `"fp4"` without `backward="fp4"` is inference-only (refused when an input requires grad) |
| `backward` | `"e4m3"` (S recomputed from the forward's 4-bit Q/K codes, dP / dV / dK / dQ on E4M3 operands) | `"fp4"` (all-4-bit query-row backward; needs `pv_precision="fp4"`, d = 128, not deterministic) | the backward route |
| `causal_safety` | `"auto"` = `"diagonal"` on the 4-bit P.V routes under `causal=True`; nothing to do on dense calls and on the default E4M3 P.V | `"key_local"`, `"diagonal"`, `"off"` | how a 4-bit P.V keeps future keys out of V's shared scales |
| `options` | `None` = `FP4AttnOptions()` | `FP4AttnOptions(...)` | advanced options, section 6.4 |

### 6.3 Arguments only of `flash_attn_fp8`

| argument | default | other values | what it does |
|---|---|---|---|
| `q_quantized`, `k_quantized` | `None` (the call quantises Q and K itself) | `flash_attn.cute.prequant.QuantizedOperand` or `(codes, descale)`, both together | caller-quantised per-tensor E4M3 Q and K; forces per-tensor operands |
| `q_scale_fused` | `False` | `True` | the supplied Q descale already carries `log2(e) * softmax_scale` |
| `q_descale`, `k_descale`, `v_descale` | `None` (q, k, v are bf16 / fp16 and quantised by the call) | float32 (or float64) `(b, h_kv)` CUDA tensors, all three, with `torch.float8_e4m3fn` q / k / v | FA3-style pre-quantised input: `x ~= code * descale`, per (batch row or document, kv head); no Q / K / V quantiser; bf16 output |
| `grad_proxies` | `None` | bf16 `(q, k, v)` of the codes' shapes | with E4M3 inputs: the tensors whose `.grad` receives dQ / dK / dV (straight-through) |
| `options` | `None` = `FP8AttnOptions()` | `FP8AttnOptions(...)` | advanced options, section 6.5 |

### 6.4 `FP4AttnOptions` (pass as `flash_attn_fp4(..., options=FP4AttnOptions(...))`)

| field | default | other values | what it does |
|---|---|---|---|
| `operand_scaling` | `"auto"` = `"token_block"` (raises, naming the opt-in, where token blocks cannot serve, e.g. head_dim 256); the all-4-bit route (`backward="fp4"`) always uses per-tensor E4M3 copies | `"token_block"`, `"block_32x32"`, `"per_tensor"`, `"token_local"` | scaling of the E4M3 operand copies: one UE8M0 exponent per 32 tokens x d (`token_block`), per 32 tokens x 32 d, one scale per tensor (fastest, global causal channel; on varlen one scale per (document, kv head), head_dim 128), or per token (exactly causal, slower, turns `qk_hadamard` off) |
| `per_tensor_scale` | `"exact"` (amax / 448) | `"power_of_two"` | per-tensor scales only: round the scale up to a power of two (much smaller causal channel, no cost) |
| `ds_scaling` | `"auto"` = `"block_32x32"` (per-tensor dS at head_dim 256) | `"block_32x32"` | scaling of the E4M3 score gradient dS-hat; explicit `"per_tensor"` is refused |
| `ds_dq_rowlocal` | `"off"` | `"diagonal"`, `"all"` | per-query-row dS exponents on the dQ side; `"all"` closes the in-block dS causal channel exactly (+21-38 % backward) |
| `fp4_ds_multiplier` | `"auto"` = `"row_local"` when causal, `"global_row_boost"` when dense | `"global"`, `"global_row_boost"`, `"row_local"` | `backward="fp4"`: multiplier under the 4-bit dS |
| `p_scaling` | `"row"` (P x 2^8, no scale factor) | `"block_1x32"` | E4M3 P-hat of the dV GEMM |
| `p_frame` | `"two_level"` | `"single_level"` (`"per_tile"` is refused) | `pv_precision="fp4"` inference forward: how the 4-bit P-hat is framed |
| `p_hat_frame` | `"auto"` = `"group"` (one UE8M0 per 32 keys) | `"row"`, `"group"` | the E4M3 P-hat of the token-block forward; `"row"` (constant P scale tile) is opt-in on 4-bit |
| `softmax_exp` | `"auto"` = Mitchell (Direct-P) where P-hat is 4-bit, exp2 where it is E4M3 | `"exp2"`, `"mitchell"` | how P-hat = 2^x is formed |
| `k_channel` | `"fold"` | `"cut"`, `"all"`, `"open"` | `causal_safety="diagonal"` on the all-4-bit route: how dQ's K operand avoids future keys' scales |
| `q_block_2d` | `False` | `True` | NVFP4 Q with one scale per 16 x 16 block (refused under causal / window / mask_mod; dense varlen / `seqused_k` documents take it) |
| `v_hadamard` | `False` | `True` | Hadamard on V, all-4-bit key-local route only |
| `all_fp4_stabiliser` | `"sr"` (stochastic rounding of dO-hat and the 4-bit dS) | `"rtn"`, `"eden_rht"` (dense only) | `backward="fp4"`: rounding of the 4-bit backward operands |
| `sr_ds` | `False` | `True` | `backward="e4m3"`: stochastic rounding of the E4M3 dS-hat (1-CTA backward; helps at head_dim 256, neutral at 128) |
| `sr_dk_operand` | `True` | `False` | `backward="fp4"`: stochastic rounding of the dK operand's dS-hat |
| `sr_dv_operand` | `False` | `True` | `backward="fp4"`: stochastic rounding of the dV operand's P-hat |
| `fp4_backward_dkdv` | `"e4m3"` | `"fp4"` | `backward="fp4"`: dK / dV operands (`"fp4"` = every backward GEMM 4-bit, below the accuracy gate) |
| `dq_operand` | `"fp4"` | (`"e4m3"` is refused) | `backward="fp4"`: the dQ GEMM's dS operand |
| `two_cta` | `"auto"`: key length (s_k batched, `max_seqlen_k` on varlen) >= 32768 -> 2-CTA wherever supported (d = 128; MHA and GQA; batched any s_q / s_k multiple of 128; aligned varlen at 128-aligned starts; repacked varlen / `seqused_k` repacked at 256), except deterministic calls, which keep the 1-CTA kernel (its ordered dQ measured faster), and 8-bit explicit per-tensor operands, which keep the 2-CTA kernel. Below 32768: 2-CTA when MHA, d = 128, s_q = s_k, s % 256 = 0, no varlen; per-tensor operands only from s_k 16384 causal / 8192 dense; deterministic uses 1-CTA; varlen / `seqused_k`: 1-CTA | `"on"` (raises if ineligible; GQA, deterministic GQA, documents at 128-aligned starts, `s_q != s_k` and `s_k % 256 == 128` included), `"off"` | backward on the 2-CTA cluster kernel. GQA on the pair (one q head per pair, fp32 dK / dV reduce-adds over the q heads) makes dK / dV non-bitwise run to run unless deterministic (then ordered by q head) |
| `forward_two_cta` | `"auto"` = on only for dense batched calls with `p_hat_frame="row"` at s_k >= 4096 (varlen / `seqused_k`: off) | `"on"`, `"off"` | forward on CTA pairs (bitwise the 1-CTA forward; varlen documents too) |
| `seqused_k_group_aligned` | `False` | (`True` is refused) | kept for compatibility |
| `save_bf16_inputs` | `True` | `False` | `False` keeps only the quantised codes for the backward (1560 -> 792 bytes per token-head at d = 128, bitwise); E4M3 P.V with per-tensor operands |
| `p_consistency` | `"reproduce_forward_p"` | `"none"` | `pv_precision="fp4"` inference forward only: which forward variant fixes O / LSE |
| `do_rounding` | `"rtn"` | (`"sr"` is refused: the 4-bit P.V forward is inference-only) | kept for that route's API |
| `exp_mix` | `"auto"` = `"EMMEMMEM"` on the 4-bit P.V inference forward, `"none"` (Mitchell everywhere) on the all-4-bit route | `"none"`, or 8 characters over `M` / `E` / `2` / `3` | value function per 16-key group position of the 4-bit P-hat |
| `tuning` | `None` (measured rules) | `KernelTuning(...)` | advanced performance constants; not a stable API |

### 6.5 `FP8AttnOptions` (pass as `flash_attn_fp8(..., options=FP8AttnOptions(...))`)

| field | default | other values | what it does |
|---|---|---|---|
| `operand_scaling` | `"auto"` = `"token_block"` on every served shape (head dims up to 128 in steps of 16, `s_q != s_k`, varlen); raises, naming the opt-in, where per-tensor operands are needed | `"token_block"`, `"block_32x32"`, `"per_tensor"`, `"token_local"` | scaling of the E4M3 Q / K / V / dO, as for FP4 (varlen per-tensor: one scale per (document, kv head), head_dim 64 / 128) |
| `per_tensor_scale` | `"exact"` | `"power_of_two"` | as for FP4 |
| `ds_scaling` | `"auto"` = `"block_32x32"` | `"block_32x32"`, `"row_local"`, `"row_1x32"` (exactly leak-free, 1-CTA, 1.4-1.5x backward), `"block_32x64"`, `"tile_128"` (2-CTA only) | scaling of the E4M3 dS-hat; `"per_tensor"` is refused |
| `ds_dq_rowlocal` | `"off"` | `"diagonal"`, `"all"` | as for FP4 (with `ds_scaling="block_32x32"`) |
| `per_tensor_ds_source` | `None` | (any value is refused) | kept for compatibility |
| `p_scaling` | `"row"` | `"block_1x32"` | E4M3 P-hat of the dV GEMM |
| `grad_format` | `"e4m3"` | `"e5m2"` (needs explicit per-tensor operands, 1-CTA) | element type of dO and dS |
| `softmax_exp` | `"exp2"` | (`"mitchell"` is refused on 8-bit) | how P-hat = 2^x is formed |
| `forward_kernel` | `"auto"` = the fork's forward | `"fork"`, `"upstream"` (explicit per-tensor operands only) | which forward kernel runs |
| `two_cta` | `"auto"` (2-CTA backward wherever eligible -- below key length 32768: MHA, s_q = s_k, s % 256 = 0, no varlen; from 32768: also GQA, varlen and the batched tails; deterministic uses 1-CTA except for explicit per-tensor operands) | `"on"`, `"off"` | as for FP4 |
| `forward_two_cta` | `"auto"` = off | `"on"`, `"off"` | as for FP4 |
| `p_hat_frame` | `"auto"` = `"row"` (constant P scale tile) | `"row"`, `"group"` | the E4M3 P-hat frame of the forward |
| `tuning` | `None` | `KernelTuning(...)` | as for FP4 |

### 6.6 `KernelTuning` (advanced; pass as `options=FP4AttnOptions(tuning=KernelTuning(...))` / `FP8AttnOptions(...)`)

Every field defaults to `None` = the measured rule; fields are validated, part of the compile key and not a stable API
(91 fields, listed in `flash_attn.cute.lowprec.tuning.TUNING_FIELDS`). The fields added on 2026-10-06/07, with what
`None` does (effects and status in [lowprec.md](lowprec.md#kerneltuning-advanced-may-change-without-notice)):

| field | `None` = | other values | what it does |
|---|---|---|---|
| `varlen_fwd_list` | `True` | `False` | aligned varlen forward: persistent longest-first work list (False: one CTA per tile) |
| `varlen_fwd_tma_o` | `True` | `False` | aligned varlen causal forward: TMA O epilogue |
| `varlen_fwd_stage_skip` | `True` | `False` | aligned varlen causal forward: the 1-CTA stage skip |
| `varlen_l2_sections` | `True` | `False` | aligned varlen work lists in L2 sections |
| `varlen_bwd_list` | `True` | `False` | aligned varlen causal non-deterministic backward: longest-first key-tile list |
| `varlen_fwd_sched` | `"auto"` = `"clc"` with the just-in-time claim on causal calls, `"static"` on dense calls (`plan.log()` line `varlen_fwd_sched`) | `"static"`, `"atomic"`, `"clc"` | how the forward's CTAs take work-list entries (O / LSE bitwise either way) |
| `varlen_fwd_sched_stages` | `2` | `1`-`4` | claim ring depth (dynamic claiming) |
| `varlen_fwd_sched_jit` | `False` (`True` under `"auto"` on causal calls) | `True` | the load warp claims the next entry just in time |
| `varlen_fwd_sched_early` | `False` | `True` | with the jit claim: claim before the last K / V loads |
| `fwd_exp` | `""` (the certified kernel) | comma list: `"sfi"`, `"gmx"`, `"sche"`, `"scha"`, `"schmK"`, `"schg"`, `"zzlong"` | experimental forward switches; `"sfi"` is bitwise, the Schraudolph ones (`sch*`) are inference-only (NaN in training) |
| `fwd_ptxas` | `""` | ptxas option string | extra ptxas options of the forward compile |
| `bwd_fp4q_post` | `"legacy"` | `"rows"`, `"epi"` | all-4-bit backward's post launch (all three bitwise) |
| `bwd_2cta_gqa` | `None` (on with `two_cta="on"` or from key length 32768) | `True`, `False` | GQA on the 2-CTA backward (dK / dV summed over q heads in fp32; ordered when deterministic) |
| `bwd_fp4_det_gqa_rot` | `False` | `True` | deterministic dense GQA with a rotated walk (slower) |
| `bwd_fp4_q_stage` | `2` | `3` | 1-CTA E4M3 backward: Q ring depth |
| `bwd_fp4_q_dyn_issue` | `False` | `True` | 1-CTA E4M3 backward: GEMMs on the dynamically indexed Q stage |
| `bwd_fp4_stat_shfl` | `"off"` | `"lse"`, `"dpsum"`, `"both"` | LSE / dPsum broadcast by shuffles (slower) |
| `bwd_fp4q_dk_sr_blk32` | `False` | `True` | all-4-bit: SR of the dK operand's dS-hat under the block-scaled dS |
| `fp4_qk_sf_vec` | `16` (NVFP4) | `32` (MXFP4) | 4-bit Q / K scale group; MXFP4 is not part of the API (less accurate, not faster) |

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

The tables follow `flash_attn/cute/lowprec/api.py` and `lowprec/options.py`; the reasons behind each default and the
measurements are in [lowprec.md](lowprec.md#defaults).

## 7. Troubleshooting

- **A call raises `ValueError` (or `NotImplementedError`)** before running: the combination of arguments is not
  served. The message says which argument to change and `exc.rule` names the rule; nothing was quantised or compiled.
  `resolve_fp4` / `resolve_fp8` raise the same errors, so you can check a configuration ahead of time.
- **An option is "refused rather than ignored"**: options that the chosen route would not read raise instead of being
  silently dropped. Remove the option or switch route.
- **First call is slow**: kernel compilation; see section 3 for the persistent cache.
- **Out of memory in tests or benchmarks**: pick a free GPU with `CUDA_VISIBLE_DEVICES=<id>`.
- **Causality**: the default token-block operands keep a small, documented in-block channel; use
  `operand_scaling="token_local"` (plus `ds_dq_rowlocal="all"` for the backward) when a model must not see future
  tokens in any bit. Details and measured sizes: [lowprec.md, Causal safety](lowprec.md#causal-safety).

## 8. Tests and benchmark

```sh
pytest tests/cute/test_lowprec_api.py              # API rules, plans, equivalence, graph capture, no host sync
pytest tests/cute/test_fp4_gqa.py tests/cute/test_fp4_2cta.py tests/cute/test_fp4_all.py
python benchmarks/benchmark_lowprec_attn.py --help # forward / backward / step / inference timing vs baselines
```

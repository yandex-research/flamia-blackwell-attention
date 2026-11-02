# FP4 backward for FA4 on Blackwell (SM100). Derived from flash_bwd_sm100.py (Ted Zadouri, Markus
# Hoehnerbach, Jay Shah, Tri Dao) -- the 1-CTA, dense, MHA subset -- with the two producer GEMMs
# (S^T = K Q^T and dP^T = V dO^T) as NVFP4 block-scaled tcgen05 MMAs (kind::mxf4nvf4) whose scale
# factors live in TMEM, and the dQ GEMM computed transposed (dQ^T = K^T dS^T, M = head_dim) so that
# 64-row query tiles keep an M=128 accumulator. This was the first stage ("C1") of the FP4 backward: dV / dK / dQ
# stayed BF16 GEMMs; the shipped operand configuration ("dk8", E4M3 dP / dV / dK / dQ) is described below.
#
# Per work tile (one KV block of tile_n keys, stationary): stream query tiles of tile_m rows.
#   inputs per query tile: Q bf16 (dK B-operand, MN-major), Q e2m1 + SFQ (S), dO bf16 (dV B-operand),
#                          dO e2m1 + SFdO (dP), LSE, dPsum
#   inputs per KV block:   K bf16 (dQ^T A-operand, MN-major), K e2m1 + SFK (S), V e2m1 + SFV (dP)
# TMEM (tile_m = 64, d = 128): S [0,64) (P bf16 overlays [0,32)), dV [64,192), dP [192,256)
#   (dS bf16 overlays [192,224), dQ^T [192,256)), dK [256,384), SFK [384,392), SFV [392,400),
#   SFQ per Q stage [400,416), SFdO [416,424).
import math
import os
from typing import Callable, Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr, Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Float4E2M1FN, BFloat16
from cutlass.utils import LayoutEnum
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.pipeline import PipelineAsync

from quack import layout_utils
from flash_attn.cute import utils
from flash_attn.cute.lowprec.specs import Bwd1Spec
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import mx_scale_utils as mxs
from flash_attn.cute.blackwell_helpers import (
    gemm_w_idx,
    gemm_ptx_w_idx,
    make_smem_layout_sfa,
    make_smem_layout_sfb,
)
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.utils import AuxData
from flash_attn.cute.softmax import apply_score_mod_inner, apply_score_mod_bwd_inner


def _as_aux_data(aux_tensors):
    """Upstream replaced the bare `aux_tensors` list with an AuxData(tensors, scalars) tuple in
    apply_mask_sm100* / apply_score_mod*_inner. This kernel threads the list internally, so it is
    converted at the call boundaries rather than rewritten throughout."""
    if aux_tensors is None:
        return AuxData()
    if isinstance(aux_tensors, AuxData):
        return aux_tensors
    return AuxData(tensors=aux_tensors)

from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.lp_bwd_tiles import LPTileInfo
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,
    VarlenWorkListScheduler,
)
from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100
import cutlass.utils as utils_basic
from flash_attn.cute.qk_hadamard_cute import dk_unrotated_half

_SMEM_CAPACITY_SM100 = utils_basic.get_smem_capacity_in_bytes("sm_100")


famax_tree = mxs.famax_tree  # shared with the 2-CTA kernel


class FlashAttentionBackwardSm100FP4:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        deterministic: bool = False,
        sf_dtype=Float8E4M3FN,
        sf_vec_size: int = 16,
        dq_reduce_dtype=Float32,
        qhead_per_kvhead: int = 1,
        seqused_k: bool = False,
        out_pass: str = "all",
        # masks / varlen (explicit arguments; all compile-time, all in the driver's compile key):
        mask_mod=None,  # upstream mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors) on every tile of the range
        score_mod=None,  # upstream score_mod(score * softmax_scale, b, h, q_idx, kv_idx, seqlen_info, aux_tensors)
        score_mod_bwd=None,  # its derivative, applied to dS before the dS quantisation (softcap arrives as this pair)
        has_aux_tensors: bool = False,
        doc_varlen: bool = False,  # aligned-packed varlen (lp_bwd_tiles): b = 1, per-key-tile document tables
        # lp_docmaps (with doc_varlen): dK / dV written to the caller's PACKED key rows -- row r of a document's key tile to
        # row (doc_tab[5][doc] + local key) of mdK / mdV (T_k, h_k, d), the real rows only, straight from the registers
        # (no sdK / sdV staging, no TMA store). Compile-time (the driver's key).
        dkv_rows_packed: bool = False,
        # lp_docmaps packed rows (with dkv_rows_packed): the operand codes are the caller's PACKED rows too -- a query tile's
        # rows (Q / dO codes, the dQ accumulator) start at doc_tab[4][doc] + local, a key tile's (K / V codes) at
        # doc_tab[5][doc] + local; the shifted LSE / D rows and the scale factors stay per aligned row / tile (the
        # statistics' padding rows +inf / +2^120, the dO launch). A tile's rows past its document's end are the next
        # document's: P = +0 there (masked keys, +inf LSE queries), so dS = +0, its dQ rows exact zeros, its dK / dV
        # rows not stored -- every output bitwise the aligned layout's. Requires ds_fold. Compile-time (the driver's key).
        codes_packed: bool = False,
        # sfd (SF redesign; with codes_packed, token-block operands): no per-aligned-tile table at all -- the dK / dQ / dV
        # B operands' d-rows SF atoms are built in SMEM by the load warp from the DENSE per-row exponent tables (mSfd:
        # (b h, R) uint8 per operand, quant.fp8 dense_out), the token-block factors f = 2^(e - 127) come from the same
        # bytes, the fp4 route's NVFP4 S-operand SF atoms from the dense per-row SF words (4 coalesced loads + one
        # 16-byte SMEM store per lane), and the statistics (shifted LSE, D'') are dense per packed row (b, h, R): the
        # load warp bulk-copies an aligned 132-row superset one stage AHEAD into a staging slot and writes the stage
        # (rows past the document +inf / +2^120), so no aligned (S) row or tile exists. Compile-time (the driver's key).
        sf_dense: bool = False,
        sf_persist: bool = False,
        spt: Optional[bool] = None,  # deterministic dQ order: None = upstream's rule ((causal or local) and deterministic)
        # non-deterministic key-tile order: False = the plain (n_block, head, batch) grid; True = the L2-grouped LPT order
        # (SingleTileLPTBwdScheduler without spt: head groups that fit L2, longest key tile first); None = True for
        # causal (the tail: -4 % kernel time at b4 s4096 h16, -21 % at GQA 16/4 where the kv-stationary tiles are 4x
        # longer), False otherwise (dense tiles are equal; neutral). Deterministic runs always take the LPT scheduler
        # (spt as above). dK / dV do not depend on it; the dQ partials' arrival order does (unordered either way).
        lpt: Optional[bool] = None,
        # deterministic dQ, staged path: the reduce warps convert each (q head, m block) dQ tile to f16 in SMEM and
        # hand it to idle warp 0 (the issue warp), which waits for this key tile's turn on the semaphore, issues the TMA
        # reduce-adds, frees the SMEM buffer once they are read and waits for the writes; idle warp 1 (the release warp)
        # then increments the semaphore with release semantics. The reduce warps no longer wait for the turn, the
        # write drain or the release fence (MEMBAR.ALL.GPU, ~0.4 us per (q head, m block) on the old path). Lock values,
        # f16 partials and reduce-adds are those of the staged path: dQ / dK / dV bits unchanged. None = on wherever
        # the staged deterministic path runs. Compile-time.
        det_issue_warp: Optional[bool] = None,
        # det_issue_warp: f16 dQ tiles staged between the reduce warps and the issue warp (1 or 2); None = 2 when the
        # SMEM budget allows it, else 1
        det_issue_bufs: Optional[int] = None,
        # blk32 dS^ only (dk8 / FP8 routes): the dQ side of a 32-key x 32-query dS block that a causal diagonal
        # crosses gets one UE8M0 exponent PER QUERY ROW over its 32 keys (the native MX layout of the dQ GEMM) and its
        # own E4M3 codes in the SMEM dS^ tile; the dK side keeps the block exponent. "off" (default) = the blk32 rung
        # bitwise; "diagonal" = only the 32 x 32 blocks the causal diagonal crosses (4 of 16 per diagonal 128 x 128 tile);
        # "all" = every block (the dQ side becomes the mx32 rung's, the dK side stays blk32). Compile-time, in the
        # driver's compile key. dS is backward-only, so the scaling-consistency rule (forward and
        # backward share the scaling of every operand they SHARE) is unaffected: no forward operand changes.
        ds_dq_rowlocal: str = "off",
        # token-local operands: E4M3 operand copies (FP4AttnOptions / FP8AttnOptions operand_scaling="token_local"). Every
        # token of K, V, dO (and Q on the 8-bit route) carries its own power-of-two factor (the kernel argument
        # mTokFac, (6, b, h, s) fp32 rows Q, K, V, dO, ...): per-key factors are per-thread scalars here (dP^T rows are keys),
        # per-query factors are 32-vectors per stage staged through a per-warp SMEM slot; dQ = dS K and dK = dS^T Q
        # take the factor in their dS side (unit K / Q scales), dV = P^T dO takes f_dO in the block-scaled P^ (forced
        # on: one UE8M0 per key x 32 queries; dO's dV-side scale is unit). Needs ds_dq_rowlocal="all". Compile-time.
        token_local_operands: bool = False,
        # per-tensor operands on aligned varlen documents: every (document, kv head) carries its own E4M3 scales, so the
        # kernel scalars are a (n_docs + 1, h_kv, 8) table (mScalars) read per work tile (the key tile's document and kv
        # head) instead of one vector. Needs doc_varlen. Compile-time (the driver's key).
        doc_scaled: bool = False,
        # dqfix: the f16 dQ pre-scale per row (document / batch row): mScalars is the (8,) vector, then the tables of
        # device_scales._dq_tile_table; the dQ reduce reads its work tile's row's scale instead of the one dq_scale.
        # Compile-time (the driver's key).
        dq_tile_scaled: bool = False,
        # dq-causal: batched causal square: the dQ reduce reads the scale of the query ROW it reduces (1: the 128-row
        # tile's plane b tq + m_block; 2: the per-row plane b s_q + 128 m_block + row, sized from the keys the row may
        # see) instead of the work tile's one row scale
        dq_tile_causal: int = 0,
        *,
        # the compile-time settings (lowprec.specs.Bwd1Spec): every knob this constructor reads; no default, so every
        # construction site states one (the driver builds it from the active configuration, the plan later)
        spec: Bwd1Spec,
    ):
        self.spec = spec
        self.token_local_arg = bool(token_local_operands)
        if self.token_local_arg:
            assert ds_dq_rowlocal == "all", "token_local_operands needs ds_dq_rowlocal='all' (the dQ side's own codes)"
        self.mask_mod = mask_mod
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        assert (score_mod is None) == (score_mod_bwd is None), "score_mod needs score_mod_bwd (the backward's dS)"
        self.has_aux_tensors = bool(has_aux_tensors)
        self.doc_varlen = bool(doc_varlen)
        self.dkv_rows_packed = bool(dkv_rows_packed)
        assert not self.dkv_rows_packed or self.doc_varlen, "dkv_rows_packed: aligned documents (doc_varlen) only"
        self.codes_packed = bool(codes_packed)
        assert not self.codes_packed or (self.dkv_rows_packed and tile_m == 128 and tile_n == 128), \
            "codes_packed: with dkv_rows_packed, 128 x 128 tiles"
        self.sf_dense = bool(sf_dense)
        # sf_dense staging packet per query tile (bytes): shifted LSE rows (132 f32 from the 4-aligned row), D'' rows,
        # the Q / dO exponent bytes (144 from the 16-aligned row), the fp4 route's Q NVFP4 SF words (132 rows)
        self.sfd_off_e = 2 * 528
        self.sfd_off_sf4 = 2 * 528 + 2 * 144
        # sfd: the work-list backward runs persistently (grid = min(SMs, entries x heads), several work tiles per CTA)
        self.bwd_persist = bool(self.sf_dense and sf_persist)
        assert not self.sf_dense or self.codes_packed, "sf_dense: with codes_packed"
        self.doc_scaled = bool(doc_scaled)
        self.dq_tile_scaled = bool(dq_tile_scaled)
        self.dq_tile_causal = int(dq_tile_causal)
        assert not self.dq_tile_causal or (self.dq_tile_scaled and not self.codes_packed)
        assert not (self.dq_tile_scaled and self.doc_scaled), "dq_tile_scaled: the (8,) scalar vector form"
        # (without doc_varlen: a batched call whose every batch row carries its own (row, kv head) scales -- the table is
        # read at the work tile's batch index, block_info.batch_for_mods; lowprec.prequant_fa3's batched training path)
        assert not (mask_mod is not None and (is_causal or is_local)), \
            "mask_mod with causal / window: express the causal / window part inside the mask_mod"
        # the key tile's query range is data dependent (may be empty): the local-window tile handling
        self.dyn_tiles = bool(is_local or doc_varlen)
        # any tile may hold masked (-inf) scores beyond the causal diagonal
        self.masked = bool(is_causal or is_local or mask_mod is not None or doc_varlen or seqused_k)
        # score_mod / mask_mod read one element at a time (aux tensors); upstream's vec_size rule
        self.mod_vec_size = 1 if self.has_aux_tensors else 4
        if lpt is None:
            lpt = bool(is_causal) and not is_local and mask_mod is None and not doc_varlen
        self.lpt = (bool(lpt) or bool(deterministic)) and not (
            bool(spec.det_gqa_rot) and bool(deterministic) and int(qhead_per_kvhead) > 1 and not is_causal
            and not is_local and mask_mod is None and not doc_varlen)
        self.det_issue_warp_arg = True if det_issue_warp is None else bool(det_issue_warp)
        self.det_issue_bufs_req = det_issue_bufs
        self.det_issue_bufs = 2 if det_issue_bufs is None else int(det_issue_bufs)
        assert self.det_issue_bufs in (1, 2), "det_issue_bufs: 1 or 2"
        # spt (shortest key tile first) needs the LPT scheduler; its default stays upstream's rule
        self.spt = (((is_causal or is_local) and bool(deterministic)) if spt is None else bool(spt)) and self.lpt
        # deterministic causal GQA: the kv-stationary tile walks its (q head, m block) pairs m-major with m
        # DESCENDING (iteration (M-1-m) r + qh), so every key tile of a kv head reaches a given (q head, m block) at the
        # same iteration; with the LPT order (longest key tile first, no spt) and the ascending lock (lock = n_block)
        # each key tile waits on the one dispatched before it by one reduce hop, not on a later-dispatched one. The
        # q-head-major walk needed spt, whose shortest-first order leaves the 4x-long GQA tiles for the tail
        # (det GQAc 0.83 ms against 0.55 non-deterministic).
        self.gqa_mmajor = (bool(deterministic) and int(qhead_per_kvhead) > 1 and bool(is_causal) and not is_local
                           and mask_mod is None and not doc_varlen and spt is None)
        if self.gqa_mmajor:
            self.spt = False
        # bwdk opt-in (Bwd1Spec.det_gqa_rot): deterministic dense GQA, rotated walk. Key tile n walks q head qh's m blocks
        # starting at block (n mod nm): at walk step j it writes block m_min + (j + n) mod nm, the j-th writer of that
        # block (the lock value), right after key tile n + 1 (mod nm) wrote it at step j - 1. No start-up chain; the
        # dependency points at the NEXT key tile, so the head's key tiles must be co-resident: the natural grid (n
        # fastest, head-contiguous) instead of the LPT order, s_q == s_k and s_k / 128 <= SMs (the driver checks).
        self.det_gqa_rot = (bool(spec.det_gqa_rot) and bool(deterministic) and int(qhead_per_kvhead) > 1
                            and not is_causal and not is_local and mask_mod is None and not doc_varlen)
        if self.det_gqa_rot:
            self.spt = False
        self.lse_tile = tile_m  # the per-query stats tile: LSE, one word per query
        # seqused_k: per-sequence key length (b,) int32 -- keys >= seqused_k[b] are masked on the key-boundary tile
        # (varlen via per-sequence calls on 128-padded tensors); causal tiles need no extra mask for padded keys
        self.seqused_k = bool(seqused_k)
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        # out_pass (head_dim 256): one kernel call per output -- "dk" / "dq" / "dv" issue only that output's GEMM (and the dQ
        # reduce for "dq"); the other outputs' epilogues store into caller-provided dummies. TMEM per pass: S [0,128) |
        # output accumulator [128,384) (256 columns) | dP bank [384,512) (E4M3 d\hat{S} overlay, SF parks). Needs the E4M3 \hat{P} / d\hat{S}
        # configurations (dk8 or the FP8 control), whose 32-column \hat{P} / d\hat{S} leave the S / dP banks' parks free.
        assert out_pass in ("all", "dk", "dq", "dv"), out_pass
        self.out_pass = out_pass
        if out_pass == "all":
            # d = 64 (native head_dim 64): the same single pass with N = 64 dK / dV / dQ accumulators and K = 64 S / dP GEMMs
            assert (self.tile_hdim, self.tile_hdimv) in ((128, 128), (64, 64)), \
                "FP4 backward: head_dim 128 or 64 (256: out_pass dk / dq / dv)"
        else:
            assert self.tile_hdim in (128, 256) and self.tile_hdimv == self.tile_hdim, (self.tile_hdim, self.tile_hdimv)
        assert tile_n == 128
        assert tile_m == 128, "FP4 backward: tile_m = 128 (N=64 MMAs run at half rate; see notes)"
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.is_causal = is_causal
        self.is_local = is_local
        self.deterministic = deterministic
        self.qhead_per_kvhead = int(qhead_per_kvhead)  # GQA: r q heads per kv head (kv-stationary tiles)
        # sf_vec_size 16 = NVFP4 S (E4M3 SFs per 16, scale_vec::4X); 32 = MXFP4 S (UE8M0 per 32, scale_vec::2X). Only the S
        # operands are block-scaled in the shipped dk8 configuration, so 32 requires it (dP / dV / dK / dQ are E4M3).
        self.sf_dtype = sf_dtype if sf_vec_size == 16 else Float8E8M0FNU
        self.sf_vec_size = sf_vec_size
        assert sf_vec_size in (16, 32)
        self.dq_reduce_dtype = dq_reduce_dtype
        self.mma_inst_bits_k = 256
        # one E2M1 tcgen05.mma instruction covers K = 64 elements (x4 SF groups of 16)
        self.mma_inst_tile_k = self.tile_hdim // (self.mma_inst_bits_k // 8 * 2)
        self.cta_group_size = 1
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.acc_dtype = Float32

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S^T = K Q^T (M = keys, N = queries, K = d)
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP^T = V dO^T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV += P^T dO (M = keys, N = dv, K = queries)
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK += dS^T Q (M = keys, N = d, K = queries)
        self.mma_tiler_dsq = (tile_n, self.tile_hdim, tile_m)
        # dQ = dS K (M = queries, N = d, K = keys)
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n)

        # Bwd1Spec.compute_wgs = 4: 16 compute warps at 32 columns each (halves every per-WG serial chain;
        # a 64-query K-tile then spans two WGs, whose SF halves are exchanged through smem), 24 warps in all
        self.num_compute_wgs = int(spec.compute_wgs)
        assert self.num_compute_wgs in (2, 4)
        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = tuple(range(4, 4 + 4 * self.num_compute_wgs))
        self.mma_warp_id = self.compute_warp_ids[-1] + 1
        self.load_warp_id = self.mma_warp_id + 1
        self.empty_warp_ids = (self.load_warp_id + 1, self.load_warp_id + 2)
        self.threads_per_cta = (self.empty_warp_ids[-1] + 1) * cute.arch.WARP_SIZE
        # named barrier ids: the dK/dV epilogue uses base + wg_idx; 4 WGs would collide with Compute (3) and
        # dQaccReduce (4), so it moves to 8..11; the SF exchange of WG pairs uses 12, 13
        self.epi_barrier_base = int(NamedBarrierBwdSm100.EpilogueWG1) if self.num_compute_wgs == 2 else 8
        self.sfx_barrier_base = 12
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM map (columns), tile_m = 128: the four accumulators fill all 512 columns as in the BF16
        # kernel (S [0,128) with P bf16 overlaying [0,64); dV [128,256); dP [256,384) with dS bf16
        # overlaying [256,320) and dQ overlaying [256,384); dK [384,512)). The scale factors are parked in
        # the accumulator banks' free windows: SFV / SFdO (operands of dP) in the S bank [64,88) after the
        # compute warps have loaded S, SFK / SFQ (operands of S) in the dP bank [320,344) after they have
        # loaded dP; the in-order tensor pipe then overwrites them only after the MMA that read them.
        # (The earlier research operand modes -- BF16 dV / dK / dQ GEMMs and E2M1 P^ / dS^ GEMMs -- are not part of this
        # kernel: the 4-bit route is dk8 below.)
        # FP8 control mode (Bwd1Spec.fp8, the 8-bit route): the same kernel with E4M3 operands and per-tensor descales
        # (kind::f8f6f4): S / dP from K-major E4M3 Q, K, V, dO tiles (no scale factors, no S2T), \hat{P} / d\hat{S} as
        # E4M3 in TMEM (32 columns), dK / dV / dQ on the bf16 (C1) code paths with E4M3 B operands: the E4M3
        # Q / dO / K tiles loaded for S / dP double as the MN-major B operands of dK / dV / dQ (same bytes,
        # same 128 B swizzle). Per-tensor descales are folded on the
        # host into softmax_scale, the dPsum scale and the dV / dK / dQ epilogue scales (runtime arguments).
        self.fp8 = bool(spec.fp8)
        self.sfd_slot = (self.sfd_off_sf4 + (0 if self.fp8 else 528 * (self.tile_hdim // 64)) + 127) // 128 * 128
        # OPERAND CONFIGURATION (the 4-bit route, "dk8"): S from NVFP4 Q / K; dP from E4M3 V / dO; dV = E4M3 P^ x the E4M3
        # dO tile; dQ / dK from one E4M3 dS^ x E4M3 K / Q. Other operand modes (NVFP4 dP, E2M1 P^ / dS^ for dV / dK,
        # bf16 / smem dK and dQ) were measured during development and are not shipped.
        self.op_dtype = Float8E4M3FN if self.fp8 else Float4E2M1FN
        self.dp_dtype = Float8E4M3FN
        # the E4M3 chain of the 4-bit route: E4M3 dP operands (dp8), E4M3 P^ x the dO tile of dP for dV (dv8, the dO tile
        # of dP IS the dV B operand: dv8_alias), E4M3 dS^ x K8 for dQ (dq8), the same dS^ words x Q8 for dK (dk8). All
        # True on the 4-bit route and False on the FP8 control (which has its own branches); kept as names for the
        # compile key and the readability of the branches.
        self.dp8 = self.dp_chain = self.dv8 = self.dv8_alias = self.dq8 = self.dk8 = not self.fp8
        # ds_fold (KernelTuning.bwd_fp4_ds_fold): dS = P (d\hat{P} b - D b s) as FMA(d\hat{P}, b, D'') with D'' = -s b D pre-scaled on
        # the host (mdPsum): one MUL per dS element fewer than FMA(D, -s, d\hat{P}) -> MUL P -> MUL b
        self.ds_fold = bool(spec.ds_fold)
        # (cov3: sf_dense also unfolded -- the load warp writes D = -2^100 into a padding row, so x = dP - s D > 0)
        assert not self.codes_packed or self.ds_fold or self.sf_dense, "codes_packed: the folded D'' (padding rows +2^120)"
        # Bwd1Spec.ds_sr (options sr_ds): stochastic rounding of the E4M3 dS^ operand (the query-row kernel's dK operand needed it for
        # 0.022 of val loss: the bound's slack makes round-to-nearest flush every softmax row's tail to zero)
        self.ds_sr = bool(spec.ds_sr)
        # S / dP TMEM load repetition (columns per tcgen05.ld). copy_utils.make_tmem_copy hands the WGs
        # alternating atom-sized column blocks, so with x32 loads WG w holds queries {32w..32w+31, 64+32w..}.
        # C3 makes that the MMA's K order: the host quantises dO^T / Q^T with the 32-query blocks of every
        # 128-query tile permuted [0, 2, 1, 3] (interface_bwd_fp4.permute_queries_c3), so K-tile w of the
        # dV / dK GEMMs is exactly WG w's two blocks and the E2M1 words / SF word of a thread stay local.
        # x64 (KernelTuning.bwd_fp4_ld_rep = 64: one contiguous K-tile per WG, no permutation) costs 7 % in spills.
        # FP8 mode is the other way round: x32 is +3 % at s4096 and +8 % at
        # s32768 causal -- 168 compute registers and no SF fragments, so nothing spills, and the
        # E4M3 dO^T / Q^T tiles need no host permutation (same bytes viewed MN-major).
        # Bwd1Spec.ds_scaling_fp4 (dk8 route; token-block scales): "per_tensor" = the s_ds scale (P_max bound or
        # supplied); "blk32" (default) = the FP8 route's in-kernel UE8M0 exponent per 32 keys x 32 queries on the E4M3 dS^ (one
        # quantisation for the dK and dQ GEMMs, both on kind::mxf8f6f4 with constant 2^0 SF tiles for Q8 / K8): no s_ds,
        # no P_max, no host read. The mxds machinery needs the x32 TMEM loads, so that becomes this mode's default.
        self.ds_scaling_fp4 = spec.ds_scaling_fp4
        assert self.ds_scaling_fp4 in ("per_tensor", "blk32"), self.ds_scaling_fp4
        # head_dim > 128 (out_pass dk / dq / dv, three kernel calls): blk32 resolves to per_tensor -- the block-scaled dS
        # machinery (TMEM parks, x32 loads) is laid out for the single-pass d = 128 kernel only, and dV came back NaN
        # there. Resolved here and in interface_bwd_fp4.ds_blk32_fp4(out_pass); `mxds` is in the compile key.
        self.mxds_fp4 = bool(self.dk8 and self.ds_scaling_fp4 == "blk32" and out_pass == "all")
        # Bwd1Spec.p_scaling_fp4 (dk8): "row" (default) = the fixed 2^8 lift of P^ (no scale factor); "mx32" = the FP8
        # route's block-scaled P^ (one UE8M0 exponent per key row x 32 queries, dV on mxf8f6f4)
        self.p_scaling_fp4 = spec.p_scaling_fp4
        assert self.p_scaling_fp4 in ("row", "mx32"), self.p_scaling_fp4
        self.mxp_fp4 = bool(self.dk8 and (self.p_scaling_fp4 == "mx32" or self.token_local_arg))
        # Block-scaled E4M3 operands (token-block scales): the 8-bit copies of Q, K, V, dO carry one UE8M0 exponent per
        # 32 tokens x 32 head-dim elements (flash_attn.cute.quant.fp8), one set of codes for both GEMMs each operand
        # enters. FP8 route: Bwd1Spec.op_scaling = per_tensor | blk32 -- S, dP, dV, dK, dQ all on
        # kind::mxf8f6f4. dk8 route: Bwd1Spec.op_scaling = per_tensor | blk32 -- dP, dV, dK, dQ (S stays
        # NVFP4). The scale-factor tiles (512 B per 128 x 128 operand tile, precomputed in both layouts by the quantiser)
        # arrive by 1D bulk copies on the operand's own pipeline stage; the MMA warp S2Ts them into TMEM parks.
        # SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
        if self.fp8:
            self.op_scaling = spec.op_scaling  # the 8-bit route's operand scaling, resolved by the plan per call
        else:
            self.op_scaling = spec.op_scaling  # the 4-bit route's E4M3 operand scaling, resolved by the plan per call
        if self.doc_varlen and not self.doc_scaled:
            # aligned-packed varlen: a per-tensor scale over all documents is a channel between documents, so the E4M3
            # operands are token-block scaled (32 tokens x d, inside one document) -- or (doc_scaled) per-tensor with
            # one scale per (document, kv head); the driver quantises / reuses the same way
            self.op_scaling = "blk32x128"
        if self.doc_scaled:
            assert self.op_scaling == "per_tensor", "doc_scaled: per-(document, kv head) scales of per-tensor operands"
        assert self.op_scaling in ("per_tensor", "blk32", "blk32x128"), self.op_scaling
        self.mxop = self.op_scaling in ("blk32", "blk32x128")
        # blk32x128 (token blocks): the S / dP scales are constant along d, so those GEMMs stay plain E4M3 and the compute
        # warps fold the per-(key block, query block) factor into their scalar; only dV / dK / dQ read SF tiles
        self.mxop_tb = self.op_scaling == "blk32x128"
        self.mx_dp = self.mxop and not self.mxop_tb        # dP on mxf8f6f4
        self.mxop_s = self.mxop and self.fp8 and not self.mxop_tb  # S on mxf8f6f4 too (the dk8 route's S is NVFP4)
        self.tmem_ld_rep = int(spec.ld_rep if spec.ld_rep is not None
                               else Bwd1Spec.rule_ld_rep(self.fp8, self.mxds_fp4, self.mxp_fp4, self.mxop))
        # signal "S(t) loaded" right after the TMEM load (1) instead of after stage 0's exps (0)
        self.s_loaded_early = bool(spec.sload_early)
        # per-warp hand-offs: every compute warp fences and arrives on its own (the S_P / dP / dS pipelines
        # count one arrival per warp anyway) instead of a 256-thread named barrier before each release; the
        # two WGs may then drift and overlap their MUFU-bound (exp) and FMA-bound (dS) phases
        self.warp_sync = bool(spec.warp_sync)  # +4 % at every shape
        self.causal_diag_only = bool(spec.causal_diag)  # mask only the diagonal tiles
        # 1/SF of the E2M1 quantiser: approximate reciprocal (MUFU.RCP) instead of an IEEE division. No shipped code path
        # reads it (the E2M1 P^ / dS^ quantisers are not part of this kernel); kept as a constant attribute
        self.quant_rcp_approx = True
        # issue both x32 dP loads before the first stage's dS math (one exposed TMEM latency instead of two;
        # +32 live registers during stage 0)
        self.dp_ld2 = bool(spec.dp_ld2)
        # Block-scaled d\hat{S} (Bwd1Spec.ds_scaling_fp8, FP8 mode): the dK and dQ GEMMs run on kind::mxf8f6f4
        # with UE8M0 scale factors computed in-kernel -- no host-side max|dS| pass, no ds_boost. Granularity along
        # each GEMM's contraction axis: mx32 = one exponent per 32 queries per key row (dK side: in-thread abs-max
        # over the stage's 32 values) and one per 32 keys per query (dQ side: the warp's 32 lanes are one key
        # group, one redux.sync.max.abs per query column); blk32 = the dQ side takes the warp's 32 x 32 block
        # abs-max instead (one redux per stage). The bytes go to two 512 B smem tiles the MMA warp S2Ts into TMEM
        # before each GEMM; the per-tensor E4M3 B operands (Q, K) carry constant 2^0 SF tiles.
        # row = one exponent per row per tile on each GEMM's own row axis: dK side per key row over the tile's 128
        # queries (both warpgroups' halves, exchanged through smem behind the compute barrier; quantised after both
        # stages, so the dP loads are hoisted); dQ side per query over the tile's 128 keys -- constant along that
        # GEMM's contraction, so it needs no scale factor in the MMA: the reduce warps apply it per query row.
        # Its per-query abs-max is a column reduction: a shuffle reduce-scatter per warp + a 4-warp max in smem.
        self.ds_scaling = spec.ds_scaling_fp8
        assert self.ds_scaling in ("per_tensor", "mx32", "blk32", "row"), \
            f"{self.ds_scaling}: blk32x64 / tile128 exist on the 2-CTA kernel only (this shape took the 1-CTA route)"
        if self.mxds_fp4:
            self.ds_scaling = "blk32"  # dk8: the FP8 route's blk32 rung, selected by Bwd1Spec.ds_scaling_fp4
        self.mxds = (self.fp8 and self.ds_scaling != "per_tensor") or self.mxds_fp4
        self.mxds_row = self.mxds and self.ds_scaling == "row"
        # Bwd1Spec.ds_sr takes effect on the per-tensor E4M3 dS^ branch (fp8tp) and on the block-scaled dS^ (mx32 /
        # blk32: the defaults opt-in; the row rung quantises after the loop and stays round-to-nearest). Wherever it is
        # live the seed comes from the step's device counter (mSeed), so a graph replay draws a new dither.
        self.ds_sr_live = self.ds_sr and not self.mxds_row
        self.mxds_dq_sf = self.mxds and not self.mxds_row  # dQ on mxf8f6f4 with an SF tile (mx32 / blk32)
        if self.mxds:
            # the block-scaled dS^ SF bytes are indexed 2 * stage + wg_idx, the x32 TMEM-load mapping (as the 2-CTA
            # kernel's `ld-rep-64-sf-index-swap` guard): at x64 query blocks 1 and 2 would swap exponents
            assert self.tmem_ld_rep == 32, (
                f"KernelTuning.bwd_fp4_ld_rep={self.tmem_ld_rep} with a block-scaled dS "
                f"(ds_scaling {self.ds_scaling!r}): the dS^ scale-factor byte index is the x32 TMEM-load mapping. Use "
                "bwd_fp4_ld_rep=32, or a per-tensor dS.")
            assert self.num_compute_wgs == 2, "a block-scaled dS needs two compute warpgroups (Bwd1Spec.compute_wgs=2)"
        # row-local dQ-side dS exponents (the constructor argument). "diag" needs a causal diagonal to cross: refused on a
        # call without one rather than compiled to a silent no-op. Both modes refine only the blk32 rung (mx32 / row are
        # row-local already; per-tensor has no block exponent).
        assert ds_dq_rowlocal in ("off", "diagonal", "all"), ds_dq_rowlocal
        if ds_dq_rowlocal != "off":
            assert self.mxds and self.ds_scaling == "blk32", (
                f"ds_dq_rowlocal={ds_dq_rowlocal!r} refines the blk32 dS^ (ds_scaling={self.ds_scaling!r}, mxds={self.mxds})")
        if ds_dq_rowlocal == "diagonal":
            assert is_causal and not is_local and mask_mod is None, "ds_dq_rowlocal='diagonal': causal (no window / mask_mod)"
        self.ds_dq_rowlocal = ds_dq_rowlocal
        # blk32 SF tiles assembled by the MMA warp: under blk32 every key row of a warp's 32-key block
        # gets the same dK-side byte and every query row of a 32-query block the same dQ-side byte, so both 512 B SF tiles
        # are 32 copies of one 16-byte vector -- the 16 exponents (query block x key block), as is for dQ and 4 x 4
        # byte-transposed for dK. The compute warps write their exponent ONCE per stage into a 16-byte scratch (one
        # broadcast STS, conflict-free) instead of 32 lanes x 2 byte stores at a 16-byte stride (4-way bank conflict, on
        # the dS chain); the MMA warp expands the scratch into the tiles before their S2T (after the dS wait: the 1-stage
        # dS pipeline keeps the compute warps off the scratch until the GEMMs released it). Bitwise the per-lane stores.
        # The dQ side stays per lane under ds_dq_rowlocal (its rows differ there). FP8 route only: measured paired
        # (1-CTA kernel, 5 rounds) -0.4 .. -2.6 % on FP8 but +1.4 .. +2.3 % on the dk8 / dp_fp4
        # routes, whose MMA warp also S2Ts the NVFP4 S scale factors: the expansion sits on its dK issue path there.
        self.sf_asm = bool(self.mxds and self.ds_scaling == "blk32" and self.fp8)
        self.sf_asm_q = bool(self.sf_asm and ds_dq_rowlocal == "off")
        if self.mxds_row:
            self.dp_ld2 = True  # both stages' dS in registers before the row quantisation
        # Block-scaled \hat{P} (Bwd1Spec.p_scaling_fp8 = mx32, FP8 mode): the backward counterpart of the forward's
        # MXFP8 \hat{P} route -- one UE8M0 exponent per (key row, 32 queries) on the E4M3 \hat{P} the dV GEMM consumes from TMEM
        # (in-thread abs-max of the stage's 32 P values, codes = P * 2^-e), dV on kind::mxf8f6f4 with a constant SF
        # tile for the per-tensor dO. row (default) = the fixed 2^8 shift through LSE, i.e. per-row scaled \hat{P}.
        self.p_scaling = spec.p_scaling_fp8
        assert self.p_scaling in ("row", "mx32"), self.p_scaling
        self.mxp = (self.fp8 and (self.p_scaling == "mx32" or self.token_local_arg)) or self.mxp_fp4
        if self.mxop:
            assert self.mxds and self.mxds_dq_sf, (
                "block-scaled E4M3 operands need the block-scaled dS (FP8: Bwd1Spec.ds_scaling_fp8 in mx32 / blk32; "
                "dk8: Bwd1Spec.ds_scaling_fp4=blk32): the dK / dQ GEMMs are already mxf8f6f4 there")
        # dV on kind::mxf8f6f4: the block-scaled P^ (mxp), or the block-scaled dO B operand (mxop, P^ SF = constant 2^0)
        self.mxdv = self.mxp or self.mxop
        # `mxp-guard-precedence-dead`: this guard used to read
        #     assert A and B and C if hasattr(self, "dO_stage") else True
        # which Python parses as `assert (A and B and C) if hasattr(...) else True`; self.dO_stage is assigned
        # further down __init__, so hasattr was False and the whole statement compiled to `assert True`. What it
        # protects is the block-scaled \hat{P} SF byte store, which hardcodes `2*stage + wg_idx` exactly as
        # `ld-rep-64-sf-index-swap` describes: at tmem_ld_rep == 64 it swaps query blocks 1 and 2's exponents
        # (silently wrong dV), and at num_compute_wgs == 4 wg_idx runs 0..3 and overflows the 4-byte group into the
        # next row's word (silent corruption). Split into asserts that each name what they refuse. The dO_stage
        # conjunct is dropped here because it is already caught loudly at its own use site.
        if self.mxp:
            assert self.tmem_ld_rep == 32, (
                r"Bwd1Spec.p_scaling_fp8=mx32 needs KernelTuning.bwd_fp4_ld_rep=32: the \hat{P} SF byte index is hardcoded "
                "2*stage + wg_idx, which is the x32 mapping, and at 64 it swaps query blocks 1 and 2"
            )
            assert self.num_compute_wgs == 2, (
                r"Bwd1Spec.p_scaling_fp8=mx32 needs Bwd1Spec.compute_wgs=2: the \hat{P} SF store packs wg_idx into a "
                "4-byte group, so wg_idx in 0..3 overflows into the next row's word"
            )
        self.sf_dtype_mx = Float8E8M0FNU
        self.sf_vec_mx = 32
        # The f16-range shift of P (used only by the former bf16 dV path) is a constant 0: P is not shifted in the kernel
        # (the E4M3 P^ lift is the host's e4m3_p_shift_log2)
        self.p_shift = 1.0
        self.tmem_alloc_cols = 512
        sf_cols = (tile_n // 32) * self.mma_inst_tile_k  # 8 for d = 128
        self.tmem_S_offset = 0
        self.tmem_P_offset = 0
        self.tmem_dV_offset = tile_m
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dS_offset = self.tmem_dP_offset
        self.tmem_dQ_offset = self.tmem_dP_offset
        self.tmem_dK_offset = self.tmem_dP_offset + tile_m
        if out_pass != "all":
            self.tmem_dV_offset = self.tmem_dK_offset = self.tmem_dQ_offset = tile_m
            self.tmem_dP_offset = self.tmem_dS_offset = tile_m + self.tile_hdim
        self.skip_dk = out_pass in ("dq", "dv")
        self.skip_dv = out_pass in ("dk", "dq")
        self.skip_dq = out_pass in ("dk", "dv")
        # qk_hadamard: the Q operand of the dK GEMM is rotated, so the dK accumulator is dK R; the epilogue stores
        # (dK R) R^T. Set by the driver (a tuple of the 128 signs, compile-time) and keyed there.
        self.had_dk = None
        # Bwd1Spec.q_stage (KernelTuning.bwd_fp4_q_stage, item C): 2, or 3 on the FP8 kernel (+16 KB SMEM per extra
        # 128 x 128 E4M3 Q tile, + its LSE row and SF atom). Not on the 32 x 32 block-scaled operands (mxop_s: their
        # per-stage SFQ TMEM parks) nor on the dk8 route (NVFP4 Q, per-stage SFQ parks).
        self.Q_stage = int(spec.q_stage)
        assert self.Q_stage == 2 or (self.Q_stage == 3 and self.fp8 and not self.mxop_s), (
            "Bwd1Spec.q_stage=3: the FP8 kernel with per-tensor or token-block operands only")
        self.dO_stage = int(spec.do_stage)
        # Bwd1Spec.q_dyn_issue (KernelTuning.bwd_fp4_q_dyn_issue): the MMA warp issues S(t) / the block-scaled dK(t) on
        # the dynamically indexed Q stage (one GEMM issue) instead of one static-stage copy per stage behind a branch.
        # q_stage = 3 always does: its 3-way static form spilled (ptxas: GQA 92 -> 264 / 312 B, d64 0 -> 56 B) and lost
        # 0.2-2.9 %; the dynamic form spills less than q_stage 2's static one (GQA 28 B, d64 0).
        self.q_dyn = self.Q_stage == 3 or bool(spec.q_dyn_issue)
        assert not self.q_dyn or (self.fp8 and not self.mxop_s), (
            "the dynamic Q-stage issue: the FP8 kernel with per-tensor or token-block operands only")
        # Bwd1Spec.stat_shfl (U4 prototype): which of LSE / dPsum is broadcast by warp shuffles instead of LDS
        self.lse_shfl = spec.stat_shfl in ("lse", "both")
        self.dpsum_shfl = spec.stat_shfl in ("dpsum", "both")
        # pp (Bwd1Spec.pp, d = 64, the E4M3-operand backwards: FP8 and dk8): at d = 64 the GEMMs no longer hide
        # the elementwise work and the cross-role waits of the lockstep pipeline are exposed. pp decouples them: dQ gets a TMEM bank of its own
        # (dP(t + 1) no longer waits for the reduce warps to drain dQ(t); dQ(t + 1) waits for it instead), its SF parks
        # move out of the S bank (dP(t + 1) no longer waits for every warp to load S(t + 1)), S(t + 1) no longer waits for
        # dP(t) to be loaded (only for the S bank, freed by dV(t)), and WG0 stores its P^ / dS^ words without waiting for
        # WG1's TMEM loads (they land in WG0's own columns). Same arithmetic, codes and GEMMs: -9 % at D64c.
        # (An exp-phase token variant measured slower, +7 % / +7 %, and is not shipped.)
        self.pp = bool(self.tile_hdim == 64 and out_pass == "all" and spec.pp
                       and self.ds_scaling in ("per_tensor", "blk32") and not self.mxp
                       and self.op_scaling in ("per_tensor", "blk32x128") and self.num_compute_wgs == 2
                       and self.tmem_ld_rep == 32)
        if self.mxop:
            assert self.dO_stage == 1 and self.Q_stage in (2, 3) and self.num_compute_wgs == 2
        p_cols_bf16 = tile_m // 32 * 16  # P / dS columns reserved in the S / dP banks (64)
        # S bank: P^ [0, 64) | SFV [64,72) | SFdO [72,80)
        self.tmem_SFV_offset = self.tmem_S_offset + p_cols_bf16
        self.tmem_SFdO_offset = [self.tmem_SFV_offset + sf_cols + i * sf_cols for i in range(self.dO_stage)]
        assert self.tmem_SFdO_offset[-1] + sf_cols <= self.tmem_S_offset + tile_m
        # dP bank: dS^ [256, 320) | SFK [320,328) | SFQ [328,344)
        self.tmem_SFK_offset = self.tmem_dP_offset + p_cols_bf16
        self.tmem_SFQ_offset = [self.tmem_SFK_offset + sf_cols + i * sf_cols for i in range(self.Q_stage)]
        assert self.tmem_SFQ_offset[-1] + sf_cols <= self.tmem_dP_offset + tile_m
        # block-scaled d\hat{S}: SF parks (8 columns each). dK side in the dP bank after the E4M3 d\hat{S} columns: S2T'd once
        # dS(t) is ready (dP(t) loaded), read by dK(t), overwritten by dQ(t). dQ side in the S bank: S2T'd after
        # S(t+1) is loaded, read by dQ(t), overwritten by S(t+2).
        self.tmem_SFdS_mx_offset = self.tmem_dS_offset + tile_m // 32 * 8
        self.tmem_SFQc_offset = self.tmem_SFdS_mx_offset + 8
        self.tmem_SFdSq_offset = self.tmem_S_offset + 64
        self.tmem_SFKc_offset = self.tmem_SFdSq_offset + 8
        # block-scaled \hat{P}: SF parks in the S bank [96, 112), S2T'd right before dV(t) (\hat{P}(t) stored => S(t) loaded;
        # dQ(t-1)'s parks at [64, 80) were consumed earlier in the tensor pipe's order; S(t+1) overwrites later)
        self.tmem_SFP_mx_offset = self.tmem_S_offset + 96
        self.tmem_SFdOc_offset = self.tmem_SFP_mx_offset + 8
        # block-scaled E4M3 operands: SFV / SFdO (dP) in the S bank after the 32 E4M3 P^ columns, S2T'd once S(t+1) is
        # loaded (the dQ-side parks [64, 80) keep theirs); SFK / SFQ (FP8 S) in the dP bank at the dk8 NVFP4 parks
        # [dP+64, dP+88) (after the E4M3 dS^ [dP, dP+32) and the dK-side parks [dP+32, dP+48))
        self.tmem_SFV_mx_offset = self.tmem_S_offset + 32
        self.tmem_SFdO_mx_offset = self.tmem_SFV_mx_offset + 8
        self.tmem_SFK_mx_offset = self.tmem_dP_offset + 64
        self.tmem_SFQ_mx_offset = [self.tmem_SFK_mx_offset + 8 + 8 * i for i in range(self.Q_stage)]
        self.tmem_total = self.tmem_dK_offset + self.tile_hdim if out_pass == "all" else self.tmem_dP_offset + tile_m
        if self.pp:
            # pp: dQ [dK + d, + d) (384 at d = 64) instead of over the dP bank, and the dQ-side SF parks after it
            self.tmem_dQ_offset = self.tmem_dK_offset + self.tile_hdim
            self.tmem_SFdSq_offset = self.tmem_dQ_offset + self.tile_hdim
            self.tmem_SFKc_offset = self.tmem_SFdSq_offset + 8
            self.tmem_total = self.tmem_SFKc_offset + 8
            if not self.fp8:
                # dk8 (NVFP4 S): S(t + 1) is issued before every warp has loaded dP(t), so its SFK / SFQ parks leave the
                # dP bank for the free columns after the dQ-side parks
                self.tmem_SFK_offset = self.tmem_total
                self.tmem_SFQ_offset = [self.tmem_SFK_offset + sf_cols + i * sf_cols for i in range(self.Q_stage)]
                self.tmem_total = self.tmem_SFQ_offset[-1] + sf_cols
        assert self.tmem_total <= self.tmem_alloc_cols, f"TMEM {self.tmem_total} > 512"
        self.sf_cols = sf_cols
        # K per E2M1 instruction of the dV / dK GEMMs (K = queries): tile_m / 64 K-tiles
        self.mma_inst_tile_k_q = tile_m // (self.mma_inst_bits_k // 8 * 2)

        self.reduce_off = out_pass in ("dk", "dv")  # no dQ reduce copies
        # register budgets (setmaxnreg): 4 reduce + 8 compute + load + mma + 2 idle warps, 65536 / 32 = 2048
        # per-warp total. KernelTuning.bwd_fp4_reduce_chunked: the reduce warps load dQ one chunk at a time (64 regs)
        # and give the compute warps +40 registers (the load-overlap levers need them)
        self.reduce_chunked = bool(spec.reduce_chunked)  # +2.5 / +3 / +0.7 %
        # setmaxnreg is warpgroup-scoped: warpgroup 3 (mma, load, 2 idle) must agree on one value for the
        # freed registers to reach the pool (idle warps at 24 with mma/load at 88 -> the compute inc hangs)
        if self.num_compute_wgs == 4:
            # 768 threads -> launch count 80. setmaxnreg.inc only draws on registers released by the CTA's own
            # warpgroups (uniform value per warpgroup): 16 x (C - 80) <= 4 x (80 - R) + 4 x (80 - W), i.e.
            # C = 96 needs R + W <= 96 (the MMA warp spills at 40), C = 88 allows 64 / 64.
            # Bwd1Spec.regs4 = (C, R, W) overrides.
            self.reduce_chunked = True
            c, r, w = (int(x) for x in spec.regs4)
            assert 16 * (c - 80) <= 4 * (80 - r) + 4 * (80 - w), (c, r, w)
            self.num_regs_reduce = r
            self.num_regs_compute = c
            self.num_regs_load = self.num_regs_mma = self.num_regs_empty = w
        else:
            self.num_regs_reduce = 80 if self.reduce_chunked else 152
            self.num_regs_compute = 168 if self.reduce_chunked else 136
            self.num_regs_load = 88
            self.num_regs_mma = 88
            self.num_regs_empty = 88 if self.reduce_chunked else 24
        # per-warp register pool: 4 reduce + 8 compute + load + mma + 2 idle warps <= 65536 / 32 (setmaxnreg is
        # applied per warp inside warpgroup 3: load 88, mma 88, idle 24)
        assert (4 * self.num_regs_reduce + 8 * self.num_regs_compute + self.num_regs_load + self.num_regs_mma
                + 2 * self.num_regs_empty) <= 2048
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.single_stage = 1
        self.sdKVaccum_stage = 2
        # dQ reduce: the accumulator is (tile_m lanes x hdim cols); per m_block we flush
        # hdim // dQ_reduce_ncol chunks of (tile_m x dQ_reduce_ncol) fp32 through TMA tensor reduce-add
        # into the row-major dQ accumulator (b, h, s, d).
        self.dQ_reduce_ncol = int(self.spec.dq_ncol if self.spec.dq_ncol is not None
                                  else Bwd1Spec.rule_dq_ncol(self.num_compute_wgs))
        self.dq_stages = 1  # one dQ accumulator stage (a two-halves dQ variant measured slower and is not shipped)
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.sdQaccum_stage = int(self.spec.dq_stages if self.spec.dq_stages is not None
                                  else Bwd1Spec.rule_dq_stages(self.num_compute_wgs))
        # deterministic dQ, staged: the whole tile is parked in SMEM (one stage per chunk) before the
        # semaphore wait; see dQacc_reduce. The other deterministic configurations keep the per-chunk path.
        self.det_staged = (bool(self.deterministic) and self.reduce_chunked and not self.reduce_off
                           and self.dq_reduce_dtype.width == 16 and self.sdQaccum_stage >= self.dQaccum_reduce_stage)
        self.det_issue = self.det_staged and self.det_issue_warp_arg
        if self.det_issue:
            # one SMEM stage per chunk per hand-off buffer (the issue warp drains buffer b while the reduce warps fill
            # the other)
            self.sdQaccum_stage = self.dQaccum_reduce_stage * self.det_issue_bufs
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)

    def _get_tiled_mma(self):
        if self.fp8:
            tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
                Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.acc_dtype, self.cta_group, self.mma_tiler_kq[:2],
            ) if not self.mxop_s else sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.sf_dtype_mx, self.sf_vec_mx, self.cta_group, self.mma_tiler_kq[:2],
            )
            tiled_mma_dP = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.sf_dtype_mx, self.sf_vec_mx, self.cta_group, self.mma_tiler_vdo[:2],
            ) if self.mx_dp else sm100_utils_basic.make_trivial_tiled_mma(
                Float8E4M3FN, self.do_dtype,  # A = V (E4M3), B = dO (gradient dtype)
                tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.acc_dtype, self.cta_group, self.mma_tiler_vdo[:2],
            )
        else:
            tiled_mma_S = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float4E2M1FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_kq[:2],
            )
            tiled_mma_dP = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.sf_dtype_mx, self.sf_vec_mx, self.cta_group, self.mma_tiler_vdo[:2],
            ) if self.mx_dp else sm100_utils_basic.make_trivial_tiled_mma(
                Float8E4M3FN, self.do_dtype,  # A = V (E4M3), B = dO (gradient dtype)
                tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.acc_dtype, self.cta_group, self.mma_tiler_vdo[:2],
            )
        if self.mxdv:
            # dV += \hat{P}^T dO: A = E4M3 \hat{P} (TMEM) with UE8M0 SFs per 32 queries, B = the E4M3 dO tile MN-major, constant SFs
            # (mxop: dO's own 32 x 32 block SFs, P^ SF constant 2^0 unless mxp)
            tiled_mma_dV = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.MN,
                self.sf_dtype_mx,
                self.sf_vec_mx,
                self.cta_group,
                self.mma_tiler_pdo[:2],
                tcgen05.OperandSource.TMEM,
            )
        else:
            tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
                Float8E4M3FN, self.do_dtype,  # A = P^ (E4M3, TMEM), B = dO (gradient dtype)
                tcgen05.OperandMajorMode.K,  # P (TMEM)
                tcgen05.OperandMajorMode.MN,  # dO
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_pdo[:2],
                a_source=tcgen05.OperandSource.TMEM,
            )
        if self.mxds:
            # dK += d\hat{S}^T Q: A = E4M3 d\hat{S} (TMEM) with UE8M0 SFs per 32 queries, B = the E4M3 Q tile MN-major with
            # constant SFs (kind::mxf8f6f4 scale_vec::1X; transposed B is legal for this kind, PTX 9.7.16)
            tiled_mma_dK = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.MN,
                self.sf_dtype_mx,
                self.sf_vec_mx,
                self.cta_group,
                self.mma_tiler_dsq[:2],
                tcgen05.OperandSource.TMEM,
            )
        else:
            tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,  # bf16 dS x bf16 Q (do_dtype is E4M3 in dp8)
                tcgen05.OperandMajorMode.K,  # dS (TMEM)
                tcgen05.OperandMajorMode.MN,  # Q
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dsq[:2],
                a_source=tcgen05.OperandSource.TMEM,
            )
        # SFB helper mma for the TMA of the dO^T / Q^T scale factors (N = d = 128 rows, K = queries)
        self.mma_tiler_sfb_q = (self.tile_n, cute.round_up(self.tile_hdim, 128), self.tile_m)
        self.tiled_mma_dQh = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            (self.tile_m, 64),
        )
        self.tiled_mma_sfb_q = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_sfb_q[:2],
            tcgen05.OperandSource.TMEM,
        )
        # dQ = dS K: A = dS (M = queries, K = keys) read from the (keys, queries) dS^T smem tile -> MN-major;
        # B = K (N = d, K = keys) read from the (keys, d) K tile -> MN-major (as the BF16 kernel).
        if self.mxds_dq_sf:
            # dQ = d\hat{S} K: A = the E4M3 d\hat{S}^T smem tile viewed MN-major with UE8M0 SFs per 32 keys per query,
            # B = the E4M3 K tile MN-major with constant SFs (the row rung keeps the plain GEMM: its per-query
            # scale is applied by the reduce warps)
            tiled_mma_dQ = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float8E4M3FN,
                tcgen05.OperandMajorMode.MN,
                tcgen05.OperandMajorMode.MN,
                self.sf_dtype_mx,
                self.sf_vec_mx,
                self.cta_group,
                self.mma_tiler_dsk[:2],
            )
        else:
            tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
                self.k_dtype,
                tcgen05.OperandMajorMode.MN,
                tcgen05.OperandMajorMode.MN,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dsk[:2],
            )
        # SFB tiled mma helpers (128-row padded N for the TMA of the B-side scale factors)
        self.mma_tiler_sfb_kq = (self.tile_n, cute.round_up(self.tile_m, 128), self.tile_hdim)
        tiled_mma_sfb_S = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_sfb_kq[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ, tiled_mma_sfb_S

    def _setup_smem_layout(self):
        e2m1 = Float4E2M1FN
        op = self.op_dtype  # E2M1 (FP4) or E4M3 (FP8 control)
        # S^T = K Q^T
        sK4_layout = sm100_utils_basic.make_smem_layout_a(self.tiled_mma_S, self.mma_tiler_kq, op, 1)
        self.sK4_layout = cute.slice_(sK4_layout, (None, None, None, 0))
        self.sQ4_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S, self.mma_tiler_kq, op, self.Q_stage
        )
        # dP^T = V dO^T
        sV4_layout = sm100_utils_basic.make_smem_layout_a(self.tiled_mma_dP, self.mma_tiler_vdo, self.dp_dtype, 1)
        self.sV4_layout = cute.slice_(sV4_layout, (None, None, None, 0))
        self.sdO4_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP, self.mma_tiler_vdo, self.dp_dtype, self.dO_stage
        )
        # scale factors (smem, staged) -- K-side of the two FP4 GEMMs (none in FP8 mode)
        sfk = {"mma_tile_inst_k": self.mma_inst_tile_k}
        self.sfk_smem_layout_staged = self.sfq_smem_layout_staged = None
        self.sfv_smem_layout_staged = self.sfdo_smem_layout_staged = None
        if not self.fp8:
            self.sfk_smem_layout_staged = make_smem_layout_sfa(
                self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, 1, **sfk
            )
            self.sfq_smem_layout_staged = make_smem_layout_sfb(
                self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, self.Q_stage, **sfk
            )
            self.sfv_smem_layout_staged = make_smem_layout_sfa(
                self.tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_size, 1, **sfk
            )
            self.sfdo_smem_layout_staged = make_smem_layout_sfb(
                self.tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_size, self.dO_stage, **sfk
            )
            self.sfv_smem_layout_staged = self.sfdo_smem_layout_staged = None
        # dV += P^T dO (bf16 P in TMEM, bf16 dO MN-major)  or  \hat{P}^T d\hat{O}^T (E2M1 \hat{P} in TMEM, dO^T K-major)
        p_dtype = self.do_dtype
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV, self.mma_tiler_pdo, p_dtype, 1
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV, self.mma_tiler_pdo, p_dtype, self.dO_stage
        )
        sfq_kw = {"mma_tile_inst_k": self.mma_inst_tile_k_q}
        # dK += dS^T Q  or  d\hat{S}^T \hat{Q}^T
        ds_dtype_mma = self.ds_dtype
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_dsq, ds_dtype_mma, 1
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK, self.mma_tiler_dsq, ds_dtype_mma, self.Q_stage
        )
        self.sfds_mx_layout = self.sfqc_layout = self.sfdsq_layout = self.sfkc_layout = None
        if self.mxds:
            # block-scaled d\hat{S}: one (32 x 4)-byte SF atom per 128 x 128 operand (4 E4M3 K-tiles of 32). dK: A = d\hat{S}
            # (rows = keys), B = Q (rows = d); dQ: A = d\hat{S}^T (rows = queries), B = K (rows = d)
            mx_kw = {"mma_tile_inst_k": self.tile_m // 32}
            self.sfds_mx_layout = make_smem_layout_sfa(self.tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, 1, **mx_kw)
            # mxop: Q's own SFs (rows = d, groups along queries), one tile per Q stage
            self.sfqc_layout = make_smem_layout_sfb(self.tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx,
                                                    self.Q_stage if self.mxop else 1, **mx_kw)
            if self.mxds_dq_sf:
                mx_kw_k = {"mma_tile_inst_k": self.tile_n // 32}
                self.sfdsq_layout = make_smem_layout_sfa(self.tiled_mma_dQ, self.mma_tiler_dsk, self.sf_vec_mx, 1, **mx_kw_k)
                self.sfkc_layout = make_smem_layout_sfb(self.tiled_mma_dQ, self.mma_tiler_dsk, self.sf_vec_mx, 1, **mx_kw_k)
        # block-scaled E4M3 operands: the S (FP8) and dP operands' SF tiles (rows = tokens, 4 groups of 32 along d)
        self.sfk_mx_layout = self.sfq_mx_layout = self.sfv_mx_layout = self.sfdo_mx_layout = None
        if self.mx_dp:
            mx_kw_d = {"mma_tile_inst_k": self.tile_hdim // 32}
            if self.mxop_s:
                self.sfk_mx_layout = make_smem_layout_sfa(self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_mx, 1, **mx_kw_d)
                self.sfq_mx_layout = make_smem_layout_sfb(self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_mx, self.Q_stage, **mx_kw_d)
            self.sfv_mx_layout = make_smem_layout_sfa(self.tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_mx, 1, **mx_kw_d)
            self.sfdo_mx_layout = make_smem_layout_sfb(self.tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_mx, 1, **mx_kw_d)
            for nm_, lay_, nst_ in (("sfk_mx", self.sfk_mx_layout, 1), ("sfq_mx", self.sfq_mx_layout, self.Q_stage),
                                    ("sfv_mx", self.sfv_mx_layout, 1), ("sfdo_mx", self.sfdo_mx_layout, 1),
                                    ("sfqc", self.sfqc_layout, self.Q_stage)):  # (the Q^c check covers mxop_tb below)
                if lay_ is not None:  # one 512 B SF atom per stage: the bulk copies write whole atoms
                    assert cute.cosize(lay_) == 512 * nst_, (nm_, lay_, cute.cosize(lay_))
        self.sfp_mx_layout = self.sfdoc_layout = None
        if self.mxdv:
            # block-scaled \hat{P}: A = \hat{P} (128 keys x 4 query groups), B = dO (128 d x 4 groups, constant)
            mx_kw_p = {"mma_tile_inst_k": self.tile_m // 32}
            self.sfp_mx_layout = make_smem_layout_sfa(self.tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, 1, **mx_kw_p)
            self.sfdoc_layout = make_smem_layout_sfb(self.tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, 1, **mx_kw_p)
        # dQ = dS K: A = dS (MN-major view of the (keys, queries) dS^T tile), B = K (MN-major view of K)
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.ds_dtype, 1
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.k_dtype, 1
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        # N = 64 half of the same bytes (dq_halves): canonical B layout of the (tile_m, 64, tile_n) MMA
        sK_half_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQh, (self.tile_m, 64, self.tile_n), self.k_dtype, 1
        )
        self.sK_half_layout = cute.slice_(sK_half_layout, (None, None, None, 0))
        # dS^T as written by the compute warps: (tile_n keys, tile_m queries) row-major (same bytes)
        self.sdSt_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        # dQ reduce staging: (tile_m rows, dQ_reduce_ncol cols) fp32, swizzled, sdQaccum_stage deep
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dq_reduce_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.dQ_reduce_ncol), self.sdQaccum_stage
        )
        # sf_dense: each statistics stage also carries the token-block factors after its 128 rows (LSE stage: f_Q of the
        # 4 query blocks, f_K, f_V of the 4 key blocks; D'' stage: f_dO), written by the load warp
        self.stat_stride = 192 if self.sf_dense else cute.round_up(self.lse_tile, 64)
        self.sLSE_layout = cute.make_layout(
            shape=(self.lse_tile + (16 if self.sf_dense else 0), self.Q_stage), stride=(1, self.stat_stride)
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m + (16 if self.sf_dense else 0), self.dO_stage), stride=(1, self.stat_stride)
        )
        # dK / dV epilogue: each compute WG stores its hdim / num_wg columns through its own smem buffer
        nwg = self.num_compute_wgs
        self.sdK_epi_tile = (self.tile_n, math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // nwg))
        self.sdV_epi_tile = (self.tile_n, math.gcd(128 // (self.dv_dtype.width // 8), self.tile_hdimv // nwg))
        self.num_epi_stages = max(1, (self.tile_hdim // nwg) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // nwg) // self.sdV_epi_tile[1])
        self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_dtype, LayoutEnum.ROW_MAJOR, self.sdK_epi_tile, nwg
        )
        self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dv_dtype, LayoutEnum.ROW_MAJOR, self.sdV_epi_tile, nwg
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) bf16
        mK: cute.Tensor,  # (b, s_k, h, d) bf16
        mdO: cute.Tensor,  # (b, s_q, h, dv) bf16
        mQ4,  # cute.Pointer, e2m1 (b, s_q, h, d)
        mK4,  # cute.Pointer, e2m1 (b, s_k, h, d)
        mV4,  # cute.Pointer, e2m1 (b, s_k, h, dv)
        mdO4,  # cute.Pointer, e2m1 (b, s_q, h, dv)
        mSFQ: Optional[cute.Tensor],
        mSFK: Optional[cute.Tensor],
        mSFV: Optional[cute.Tensor],
        mSFdO: Optional[cute.Tensor],
        mLSE: cute.Tensor,  # (b, h, s_q) f32 (log2 domain)
        mdPsum: cute.Tensor,  # (b, h, s_q) f32
        mdQaccum: cute.Tensor,  # (b, h, s_q_rounded, d) f32 (row-major dQ accumulator)
        mdK: cute.Tensor,  # (b, s_k, h, d) bf16
        mdV: cute.Tensor,  # (b, s_k, h, dv) bf16
        softmax_scale: Float32,
        # FP8 control mode: folded per-tensor descales (ignored in FP4 mode)
        dpsum_scale: Float32,  # dS' = P' (d\hat{P} - dpsum_scale * D)
        ds_boost: Float32,  # d\hat{S} = e4m3(ds_boost * dS')
        sr_seed: Int32,  # Bwd1Spec.ds_sr: per-call seed for the dS^ dither (0 = round to nearest)
        dv_scale: Float32,  # dV = dv_scale * dV_acc
        dk_scale: Float32,  # dK = dk_scale * dK_acc
        dq_scale: Float32,  # dQ partials * dq_scale before the f16 reduce-add
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        q4_shape: tuple = (),
        k4_shape: tuple = (),
        v4_shape: tuple = (),
        do4_shape: tuple = (),
        # C3 operands: transposed NVFP4 copies (b, d, h, s) with SF along the queries
        mQt4=None,  # cute.Pointer, e2m1 (b, d, h, s_q)
        mdOt4=None,  # cute.Pointer, e2m1 (b, dv, h, s_q)
        mSFQt: Optional[cute.Tensor] = None,
        mSFdOt: Optional[cute.Tensor] = None,
        qt4_shape: tuple = (),
        dot4_shape: tuple = (),
        mSeqUsedK: Optional[cute.Tensor] = None,
        # mxop: the UE8M0 SF atoms of the E4M3 operands, (b, h, s / 128, 512) uint8, one 512 B atom per 128 x 128 tile.
        # *mx = rows are tokens, groups of 32 along d (S / dP operands); *d = rows are d, groups of 32 along tokens (the
        # MN-major B operands of dK (Q), dQ (K), dV (dO)). Same exponent table, two replications (quant.fp8).
        mSFQmx: Optional[cute.Tensor] = None,
        mSFKmx: Optional[cute.Tensor] = None,
        mSFVmx: Optional[cute.Tensor] = None,
        mSFdOmx: Optional[cute.Tensor] = None,
        mSFQd: Optional[cute.Tensor] = None,
        mSFKd: Optional[cute.Tensor] = None,
        mSFdOd: Optional[cute.Tensor] = None,
        mFac: Optional[cute.Tensor] = None,  # mxop_tb: (4, b, h, s / 32) fp32 = 2^(e - 127) of Q, K (h_kv), V (h_kv), dO
        # token-local operands: (4, b, h, s) fp32 = 2^(e_t - 127), each token's OWN exponent, of the
        # token-local E4M3 copies (rows Q, K, V, dO; the fp4 route's Q row is unused: its Q8 copy stays token-block, it
        # feeds dK only). None = today's token blocks (bitwise).
        mTokFac: Optional[cute.Tensor] = None,
        # GPU-only scales (device_scales): float32 [softmax_scale, dpsum_scale, ds_boost, dv_scale, dk_scale, dq_scale]
        # in device memory. When given, the kernel reads these instead of the scalar arguments (then ignored), so a
        # data-dependent scale never has to be read back to the host.
        mScalars: Optional[cute.Tensor] = None,
        # masks / varlen: aligned-packed varlen tables (lp_bwd_tiles) and the mods' aux tensors
        mDocTab: Optional[cute.Tensor] = None,  # (4, n_docs + 1) int32
        mNDoc: Optional[cute.Tensor] = None,  # (n_key_tiles,) int32
        # aligned documents, non-deterministic: the document-aware longest-first key-tile list (lp_varlen.varlen_worklist
        # 'bwd'; tile_scheduler.VarlenWorkListScheduler, one CTA per position) instead of the plain grid
        mWork: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        stream: cuda.CUstream = None,
        # Bwd1Spec.ds_sr where live (ds_sr_live: per-tensor or block-scaled dS): the step's SR counter (int64 (1,), interface_bwd_fp4.
        # ds_sr_counter). The kernel derives the dither seed from it on the GPU, so a CUDA-graph replay draws a fresh
        # dither; the dQ postprocess advances it. None = the scalar sr_seed (then 0 or a host seed).
        mSeed: Optional[cute.Tensor] = None,
        # codes_packed: (2, S_q / 128) int32 (DocMaps.vt_tile_q), read for its shape only: its tile count gives the scale
        # factors' aligned S_q (S_k: mNDoc's)
        mVtQ: Optional[cute.Tensor] = None,
        # sf_dense: [EvQ, EvK, EvV, EvdO (b h, R) uint8, SF4Q, SF4K (b h, R d / 64) int32 or None] (see the constructor)
        mSfd: Optional[list] = None,
    ):
        assert (mVtQ is not None) == self.codes_packed, "codes_packed <-> mVtQ"
        assert (mSfd is not None) == self.sf_dense, "sf_dense <-> mSfd"
        assert mSeed is None or self.ds_sr_live, "mSeed: only with a live dS SR (ds_sr_live)"
        assert (mDocTab is not None) == self.doc_varlen and (mNDoc is not None) == self.doc_varlen
        assert (aux_tensors is not None) == self.has_aux_tensors
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = Float8E4M3FN if self.dq8 else self.q_dtype
        if const_expr(self.dq8):
            assert mK.element_type == Float8E4M3FN, "dq8: mK is the per-tensor E4M3 K"
        assert self.dqaccum_dtype == self.dq_reduce_dtype

        # FP4 tensors from pointer + shape, row-major (b, s, h, d)
        def _fp4_tensor(ptr, shape):
            return cute.make_tensor(
                ptr, cute.make_ordered_layout(shape, order=tuple(range(len(shape) - 1, -1, -1)))
            )

        if const_expr(self.fp8):
            # E4M3 tensors: mQ / mK / mdO are the E4M3 copies (b, s, h, d); mV4 is the E4M3 V tensor
            assert mQ.element_type == Float8E4M3FN and mK.element_type == Float8E4M3FN
            # dO is a gradient operand and may be E5M2; V is a forward operand and stays E4M3.
            assert mdO.element_type in (Float8E4M3FN, Float8E5M2)
            assert mV4.element_type == Float8E4M3FN
            mQ4, mK4, mdO4 = mQ, mK, mdO
        else:
            mQ4 = _fp4_tensor(mQ4, q4_shape)
            mK4 = _fp4_tensor(mK4, k4_shape)
            # dO is a gradient operand and may be E5M2; V is a forward operand and stays E4M3.
            assert mdO.element_type in (Float8E4M3FN, Float8E5M2)
            assert mV4.element_type == Float8E4M3FN
            mdO4 = mdO

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]

        # (b, s, h, d) -> (s, d, h, b)
        T = [1, 3, 2, 0]
        mQ, mdO, mK, mQ4, mK4, mV4, mdO4 = [layout_utils.select(t, mode=T) for t in (mQ, mdO, mK, mQ4, mK4, mV4, mdO4)]
        # (b, d, h, s) -> (d, s, h, b): the B operands of dK / dV (N = d, K = queries), K-major
        # (b, h, s) -> (s, h, b)
        mLSE, mdPsum = [layout_utils.select(t, mode=[2, 1, 0]) for t in (mLSE, mdPsum)]
        # (b, h, s, d) -> (s, d, h, b)
        mdQaccum = layout_utils.select(mdQaccum, mode=[2, 3, 1, 0])
        mdK, mdV = [layout_utils.select(t, mode=T) for t in (mdK, mdV)]
        # B operands of dK / dV (N = d, K = queries): (d, s, h, b) views of Q / dO
        mQt = layout_utils.select(mQ, mode=[1, 0, 2, 3])
        mdOt = layout_utils.select(mdO, mode=[1, 0, 2, 3])
        # B operand of dQ (N = d, K = keys): (d, s_k, h, b) view of K
        mKt = layout_utils.select(mK, mode=[1, 0, 2, 3])
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])  # (block, 1, h, b)

        # scale factor tensors: rebuild the swizzled (32,4,rest_m)x(16,4,rest_k) layout from the operand shape
        def _sf_tensor(sf, op_shape):
            lay = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, op_shape, (2, 1, 3, 4))
            return cute.make_tensor(cute.recast_ptr(sf.iterator, dtype=self.sf_dtype), lay)

        if const_expr(not self.fp8):
            if const_expr(self.codes_packed):   # (packed codes: T rows; the SFs: the aligned S rows)
                mSFQ = _sf_tensor(mSFQ, (mVtQ.shape[1] * 128,) + tuple(mQ4.shape[1:]))
                mSFK = _sf_tensor(mSFK, (mNDoc.shape[0] * 128,) + tuple(mK4.shape[1:]))
            else:
                mSFQ = _sf_tensor(mSFQ, mQ4.shape)
                mSFK = _sf_tensor(mSFK, mK4.shape)

        if const_expr(self.mxop):
            assert mSFQd is not None and mSFKd is not None and mSFdOd is not None
            if const_expr(self.mx_dp):
                assert mSFVmx is not None and mSFdOmx is not None
            else:
                assert mFac is not None, "mxop_tb: the per-(token block) factors of Q, K, V, dO"
            assert mdO.element_type == Float8E4M3FN, "mxop: dO must be E4M3 (the dP GEMM is typed on one E4M3 element type)"
            if const_expr(self.mxop_s):
                assert mSFQmx is not None and mSFKmx is not None
        self.token_local = const_expr(self.token_local_arg)
        assert (mTokFac is not None) == self.token_local, "token_local_operands <-> mTokFac"
        if const_expr(self.token_local):
            assert self.mxop_tb and self.mxds and self.mxp and self.ds_dq_rowlocal == "all", (
                "token_local_operands: the token-block E4M3 routes, block-scaled P^, ds_dq_rowlocal='all'")
        mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd = [
            layout_utils.select(t, mode=[3, 2, 1, 0]) if const_expr(t is not None) else None
            for t in (mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd)]
        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
            self.tiled_mma_sfb_S,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_S.thr_id.shape,)
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        cl = self.cluster_layout_vmnk.shape
        # FP4 operands of S^T = K Q^T
        tma_atom_K4, tma_tensor_K4 = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mK4, cute.select(self.sK4_layout, mode=[0, 1, 2]), self.mma_tiler_kq, self.tiled_mma_S, cl
        )
        tma_atom_Q4, tma_tensor_Q4 = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mQ4, cute.select(self.sQ4_layout, mode=[0, 1, 2]), self.mma_tiler_kq, self.tiled_mma_S, cl
        )
        # FP4 operands of dP^T = V dO^T
        tma_atom_V4, tma_tensor_V4 = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mV4, cute.select(self.sV4_layout, mode=[0, 1, 2]), self.mma_tiler_vdo, self.tiled_mma_dP, cl
        )
        tma_atom_dO4, tma_tensor_dO4 = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mdO4, cute.select(self.sdO4_layout, mode=[0, 1, 2]), self.mma_tiler_vdo, self.tiled_mma_dP, cl
        )
        tma_atom_SFK = tma_tensor_SFK = tma_atom_SFV = tma_tensor_SFV = None
        tma_atom_SFQ = tma_tensor_SFQ = tma_atom_SFdO = tma_tensor_SFdO = None
        tma_atom_Q = tma_tensor_Q = tma_atom_dO = tma_tensor_dO = tma_atom_K = tma_tensor_K = None
        tma_atom_SFQt = tma_tensor_SFQt = tma_atom_SFdOt = tma_tensor_SFdOt = None
        if const_expr(not self.fp8):
            # scale factors: A-side (K, V) like the forward's SFQ; B-side (Q, dO) like the forward's SFK
            sfa_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op, mSFK, cute.select(self.sfk_smem_layout_staged, mode=[0, 1, 2]), self.mma_tiler_kq, self.tiled_mma_S, cl,
                internal_type=cutlass.Int16,
            )
            sfb_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
            mma_tiler_sfb = (self.tile_n, cute.round_up(self.tile_m, 128), (self.mma_inst_bits_k // 4) * self.mma_inst_tile_k)
            self.mma_tiler_sfb = mma_tiler_sfb
            cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_sfb_S.thr_id.shape,)
            )
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, mSFQ, cute.select(self.sfq_smem_layout_staged, mode=[0, 1, 2]), mma_tiler_sfb, self.tiled_mma_sfb_S,
                cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
            )
            # B operands of dK / dV: bf16 Q / dO (MN-major views) in C1, NVFP4 Q^T / dO^T (K-major) in C3
            tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mQt,
                cute.select(self.sQ_layout, mode=[0, 1, 2]), self.mma_tiler_dsq, self.tiled_mma_dK, cl
            )
            tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mdOt,
                cute.select(self.sdO_layout, mode=[0, 1, 2]), self.mma_tiler_pdo, self.tiled_mma_dV, cl
            )
            cluster_layout_sfb_q_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_sfb_q.thr_id.shape,)
            )
            sfb_q_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_dK.thr_id)
            tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mKt, cute.select(self.sK_layout, mode=[0, 1, 2]), self.mma_tiler_dsk, self.tiled_mma_dQ, cl
            )
        # dK / dV TMA stores
        tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op, mdK, cute.select(self.sdK_layout, mode=[0, 1]), self.sdK_epi_tile, 1
        )
        tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op, mdV, cute.select(self.sdV_layout, mode=[0, 1]), self.sdV_epi_tile, 1
        )
        # dQ TMA reduce-add: smem (tile_m, ncol) tile -> gmem (s, d) window
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        tma_atom_dQ, mdQaccum_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_reduce_op, mdQaccum, cute.select(self.sdQ_layout, mode=[0, 1]), (self.tile_m, self.dQ_reduce_ncol), 1
        )

        thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))
        val_layout_r2s_dKV = cute.make_ordered_layout((1, 128 // self.dk_dtype.width), order=(1, 0))
        copy_atom_r2s_dKV = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dk_dtype, num_bits_per_copy=128)
        tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV)

        def _bytes(dtype, layout):
            return cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1, 2]))

        q_b_dtype = self.q_dtype
        do_b_dtype = self.do_dtype
        if const_expr(self.fp8):
            # one E4M3 tile per operand: the S / dP tiles double as the dK / dV / dQ B operands
            self.tma_copy_bytes = {
                "Q": _bytes(Float8E4M3FN, self.sQ4_layout),
                "dO": _bytes(Float8E4M3FN, self.sdO4_layout),
                "K": _bytes(Float8E4M3FN, self.sK4_layout),
                "V": _bytes(Float8E4M3FN, self.sV4_layout),
            }
            assert _bytes(Float8E4M3FN, self.sQ4_layout) == _bytes(q_b_dtype, self.sQ_layout)
            assert _bytes(Float8E4M3FN, self.sdO4_layout) == _bytes(do_b_dtype, self.sdO_layout)
            assert _bytes(Float8E4M3FN, self.sK4_layout) == _bytes(self.k_dtype, self.sK_layout)
        else:
            self.tma_copy_bytes = {
                "Q": (0 if self.skip_dk else _bytes(q_b_dtype, self.sQ_layout)) + _bytes(Float4E2M1FN, self.sQ4_layout) + _bytes(self.sf_dtype, self.sfq_smem_layout_staged),
                "dO": _bytes(self.dp_dtype, self.sdO4_layout),  # the E4M3 dO tile of dP is also the dV B operand
                "K": (0 if self.skip_dq else _bytes(self.k_dtype, self.sK_layout)) + _bytes(Float4E2M1FN, self.sK4_layout) + _bytes(self.sf_dtype, self.sfk_smem_layout_staged),
                "V": _bytes(self.dp_dtype, self.sV4_layout),
            }
        if const_expr(self.mxop):  # 512 B SF atoms arriving on the operand's own stage (bulk copies)
            self.tma_copy_bytes["Q"] += 512 * (2 if self.mxop_s else 1)   # Q dim (dK B) [+ Q tok (S)]
            self.tma_copy_bytes["K"] += 512 * (2 if self.mxop_s else 1)   # K dim (dQ B) [+ K tok (S)]
            self.tma_copy_bytes["dO"] += 512 * (2 if self.mx_dp else 1)   # [dO tok (dP) +] dO dim (dV B)
            self.tma_copy_bytes["V"] += 512 if self.mx_dp else 0          # V tok (dP)
        self.tma_copy_bytes["LSE"] = self.lse_tile * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        if const_expr(self.sf_dense):
            # the SF atoms and the statistics are written by the load warp's generic stores (no tx bytes); cov1: also
            # the per-(document, kv head) per-tensor operands (doc_scaled: no E4M3 SF atom, the statistics packet only)
            assert (self.mxop_tb or (self.doc_scaled and not self.mxop)) and not self.mxop_s and not self.mx_dp \
                and self.tile_m == 128, "sf_dense: token blocks or per-(document, kv head) per-tensor operands"
            assert self.Q_stage == 2 and not self.q_dyn, "sf_dense: two Q stages, static issue"
            if const_expr(self.mxop_tb):
                self.tma_copy_bytes["Q"] -= 512
                self.tma_copy_bytes["K"] -= 512
                self.tma_copy_bytes["dO"] -= 512
            if const_expr(not self.fp8):
                self.tma_copy_bytes["Q"] -= _bytes(self.sf_dtype, self.sfq_smem_layout_staged)
                self.tma_copy_bytes["K"] -= _bytes(self.sf_dtype, self.sfk_smem_layout_staged)
            self.tma_copy_bytes["LSE"] = 0
            self.tma_copy_bytes["dPsum"] = 0

        TileScheduler = SingleTileLPTBwdScheduler if const_expr(self.lpt) else SingleTileScheduler
        if const_expr(self.bwd_persist and mWork is None):
            self.bwd_persist = False   # (no work list: one CTA per tile of the plain grid, the per-position code)
        if const_expr(mWork is not None):
            assert self.doc_varlen and not self.deterministic, "mWork: aligned documents, non-deterministic"
            TileScheduler = VarlenWorkListScheduler
        tile_sched_args = TileSchedulerArguments(
            # (codes_packed: the key tiles are the aligned layout's -- mNDoc's -- not the packed codes' rows)
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]) if const_expr(not self.codes_packed)
            else cute.size(mNDoc.shape[0]),
            cute.size(mK.shape[2]) if const_expr(self.qhead_per_kvhead > 1) else cute.size(mQ.shape[2]),
            cute.size(mK.shape[3]),
            1,
            cute.size(mQ.shape[0]),
            # the LPT scheduler's L2 group size counts one head's Q / dO / dQ accumulator: a GQA work tile streams r
            # q heads, so its footprint is r of them
            mQ.shape[1] * self.qhead_per_kvhead,
            mdO.shape[1] * self.qhead_per_kvhead,
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,  # upstream flash_bwd_sm100: spt = (causal or local) and deterministic, overridable
            head_swizzle=self.deterministic,
        )
        sched_over = None
        if const_expr(mWork is not None and self.bwd_persist):
            # sfd (compact maps, a static bound above the capped grid): TWO launches instead of one CTA per position of
            # the static bound (up to T x heads; an empty CTA costs ~1 us of its SM).
            # main: the capped grid min(5 / 4 ceil(T / 128), n_max) x heads, one position per CTA in the position
            #   table's order, the per-position code -- every position while the documents' tiles fit (documents
            #   averaging >= 512 tokens);
            # overflow: the positions past it, persistent over the SMs (its CTAs exit at once when there are none).
            # (One persistent kernel looping over the list cost +8 % at d128 / +4 % at fp4 d64: its loop-carried MMA
            # descriptors spill the MMA warp; a device-side branch between the two codes in one kernel cost 1-4 %.)
            _nh = cute.size(mK.shape[2]) if const_expr(self.qhead_per_kvhead > 1) else cute.size(mQ.shape[2])
            _n_max = (cute.size(mWork.shape[0]) - 2) // (2 + _nh)
            _tq = (cute.size(mK.shape[0]) + 127) // 128   # (codes_packed: mK holds the T packed rows)
            _cap = Int32((_tq + (_tq + 3) // 4) * _nh)
            tile_sched_params = VarlenWorkListScheduler.Params(mWork, _nh, _n_max * _nh, False, 2 + 2 * _n_max, True,
                                                               _cap)
            sched_over = VarlenWorkListScheduler.Params(mWork, _nh, _n_max * _nh, True, 2 + 2 * _n_max, True, None,
                                                        _cap)
        elif const_expr(mWork is not None):
            # mWork = [header, n_max entries, n_max x heads position table] (lp_varlen.varlen_worklist pos_heads)
            _nh = cute.size(mK.shape[2]) if const_expr(self.qhead_per_kvhead > 1) else cute.size(mQ.shape[2])
            _n_max = (cute.size(mWork.shape[0]) - 2) // (2 + _nh)
            tile_sched_params = VarlenWorkListScheduler.Params(
                mWork, _nh, _n_max * _nh, False, 2 + 2 * _n_max, True)
        else:
            tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        sQ_alloc_bytes = 16 if self.skip_dk else max(cute.size_in_bytes(q_b_dtype, self.sQ_layout), cute.size_in_bytes(self.dk_dtype, self.sdK_layout))
        sdO_alloc_bytes = max(cute.size_in_bytes(do_b_dtype, self.sdO_layout), cute.size_in_bytes(self.dv_dtype, self.sdV_layout))
        sfqt_size = 1
        sfdot_size = 1
        self.q_b_dtype = q_b_dtype
        self.do_b_dtype = do_b_dtype
        # FP8 mode: sQ4 / sK4 / sdO4 alias sQ / sK / sdO (one E4M3 tile per operand); no scale factors
        alias = self.fp8
        sQ4_size = 1 if alias else cute.cosize(self.sQ4_layout)
        sK4_size = 1 if alias else cute.cosize(self.sK4_layout)
        sdO4_size = 1 if (alias or self.dv8_alias) else cute.cosize(self.sdO4_layout)
        if const_expr(self.dv8_alias):  # the E4M3 dO tile of dP doubles as the MN-major dV B operand (same bytes)
            assert _bytes(Float8E4M3FN, self.sdO4_layout) == _bytes(do_b_dtype, self.sdO_layout)
        sfq_size = 1 if self.fp8 else cute.cosize(self.sfq_smem_layout_staged)
        sfk_size = 1 if self.fp8 else cute.cosize(self.sfk_smem_layout_staged)
        sfv_size = 1
        sfdo_size = 1

        # SF exchange (4 compute WGs): [phase P/dS][wg][row][2 floats]
        sfx_size = 2 * self.num_compute_wgs * self.tile_n * 2 if self.num_compute_wgs == 4 else 4
        # block-scaled d\hat{S}: the four UE8M0 tiles (d\hat{S} for dK, constant Q, d\hat{S}^T for dQ, constant K), 512 B each
        sfmx_sizes = [cute.cosize(l) if l is not None else 1 for l in (self.sfds_mx_layout, self.sfqc_layout, self.sfdsq_layout, self.sfkc_layout)]
        # row rung exchange buffers (parity double-buffered): per key row abs-max per WG, per column per-warp abs-max,
        # per column multiplier, and the per-query scale the reduce warps apply
        # (ds_dq_rowlocal: sColE holds the per-query dQ multipliers, [compute warp 8][stage 2][query 32])
        rowx_sizes = ([2 * 2 * self.tile_n, 2 * 2 * 4 * 2 * 32, 2 * 2 * 2 * 32, self.tile_m] if self.mxds_row else
                      [1, 1, (8 * 2 * 32 * (3 if self.token_local_arg else 1)) if self.ds_dq_rowlocal != "off" else 1, 1])
        sfp_sizes = [cute.cosize(l) if l is not None else 1 for l in (self.sfp_mx_layout, self.sfdoc_layout)]
        sfo_sizes = [cute.cosize(l) if l is not None else 1
                     for l in (self.sfk_mx_layout, self.sfq_mx_layout, self.sfv_mx_layout, self.sfdo_mx_layout)]

        def _make_shared_storage():
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dq_stages]
                S_loaded_mbar_ptr: cutlass.Int64
                # det_issue: [0, B) staged (reduce -> issue) per buffer, [B, 2B) free per buffer, [2B] drained (issue ->
                # release), [2B + 1] release slot free
                dQ_stage_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.det_issue_bufs + 2]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: Int64
                sQ: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes], self.buffer_align_bytes]
                sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, 1 if self.skip_dq else cute.cosize(self.sK_layout)], self.buffer_align_bytes]
                sdO: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes], self.buffer_align_bytes]
                sQ4: cute.struct.Align[cute.struct.MemRange[self.op_dtype, sQ4_size], self.buffer_align_bytes]
                sK4: cute.struct.Align[cute.struct.MemRange[self.op_dtype, sK4_size], self.buffer_align_bytes]
                sV4: cute.struct.Align[cute.struct.MemRange[self.dp_dtype, cute.cosize(self.sV4_layout)], self.buffer_align_bytes]
                sdO4: cute.struct.Align[cute.struct.MemRange[self.dp_dtype, sdO4_size], self.buffer_align_bytes]
                sSFQ: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfq_size], self.buffer_align_bytes]
                sSFK: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfk_size], self.buffer_align_bytes]
                sSFV: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfv_size], self.buffer_align_bytes]
                sSFdO: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfdo_size], self.buffer_align_bytes]
                sV4d: cute.struct.Align[cute.struct.MemRange[self.dp_dtype, 1], self.buffer_align_bytes]
                sSFVd: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, 1], self.buffer_align_bytes]
                sSFQt: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfqt_size], self.buffer_align_bytes]
                sSFdOt: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sfdot_size], self.buffer_align_bytes]
                sdS: cute.struct.Align[cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)], self.buffer_align_bytes]
                sSFx: cute.struct.Align[cute.struct.MemRange[Float32, sfx_size], 128]
                sSFdS_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx_sizes[0]], self.buffer_align_bytes]
                sSFQc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx_sizes[1]], self.buffer_align_bytes]
                sSFdSq: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx_sizes[2]], self.buffer_align_bytes]
                sSFKc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx_sizes[3]], self.buffer_align_bytes]
                sRowX: cute.struct.Align[cute.struct.MemRange[Float32, rowx_sizes[0]], 128]
                sColX: cute.struct.Align[cute.struct.MemRange[Float32, rowx_sizes[1]], 128]
                sColE: cute.struct.Align[cute.struct.MemRange[Float32, rowx_sizes[2]], 128]
                sQscale: cute.struct.Align[cute.struct.MemRange[Float32, rowx_sizes[3]], 128]
                sSFP_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfp_sizes[0]], self.buffer_align_bytes]
                sSFdOc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfp_sizes[1]], self.buffer_align_bytes]
                sSFK_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfo_sizes[0]], self.buffer_align_bytes]
                sSFQ_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfo_sizes[1]], self.buffer_align_bytes]
                sSFV_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfo_sizes[2]], self.buffer_align_bytes]
                sSFdO_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfo_sizes[3]], self.buffer_align_bytes]
                sLSE: cute.struct.Align[cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)], 128]
                sdPsum: cute.struct.Align[cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)], 128]
                sdQaccum: cute.struct.Align[cute.struct.MemRange[self.dqaccum_dtype, 1 if self.reduce_off else cute.cosize(self.sdQ_layout)], self.buffer_align_bytes]
                sDsScr: cute.struct.Align[cute.struct.MemRange[cute.Uint8, 16], 16]  # sf_asm: the 16 blk32 exponents
                # p_repro: one 32 x TR_STRIDE f32 transpose tile per compute warp (the cross-lane 16-key group maxes)
                sTr: cute.struct.Align[cute.struct.MemRange[Float32, 1], 128]
            if const_expr(self.sf_dense):
                # sf_dense only (the other kernels' storage is unchanged): the staging slots (statistics, exponent
                # bytes, fp4 SF words; see the constructor) and their mbarriers
                SharedStorage.__annotations__["sStg"] = cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, 4 * self.sfd_slot], 128]
                SharedStorage.__annotations__["Stg_mbar_ptr"] = cute.struct.MemRange[cutlass.Int64, 4]
            return cute.struct(SharedStorage)

        SharedStorage = _make_shared_storage()
        if const_expr(self.det_issue and self.det_issue_bufs_req is None and self.det_issue_bufs > 1
                      and SharedStorage.size_in_bytes() > _SMEM_CAPACITY_SM100):
            # det_issue_bufs auto: the second hand-off buffer does not fit; one buffer
            self.det_issue_bufs = 1
            self.sdQaccum_stage = self.dQaccum_reduce_stage
            self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dq_reduce_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.dQ_reduce_ncol), self.sdQaccum_stage
            )
            SharedStorage = _make_shared_storage()
        self.shared_storage = SharedStorage
        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        for launch in cutlass.range_constexpr(1 if const_expr(sched_over is None) else 2):
            if const_expr(sched_over is not None):
                # (trace time) main: the per-position code; overflow: the persistent code
                self._set_bwd_persist(launch == 1)
                if const_expr(launch == 1):
                    tile_sched_params = sched_over
                    grid_dim = TileScheduler.get_grid_shape(sched_over)
            self.kernel(
                tma_tensor_Q, tma_tensor_Q4, tma_tensor_SFQ, tma_tensor_SFQt, tma_tensor_SFdOt,
                tma_tensor_K, tma_tensor_K4, tma_tensor_SFK,
                tma_tensor_V4, tma_tensor_SFV,
                tma_tensor_dO, tma_tensor_dO4, tma_tensor_SFdO,
                mLSE, mdPsum,
                mdQaccum_tma_tensor, mdK_tma_tensor, mdV_tma_tensor, mdK, mdV,
                mdQ_semaphore,
                tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_SFQt, tma_atom_SFdOt,
                tma_atom_K, tma_atom_K4, tma_atom_SFK,
                tma_atom_V4, tma_atom_SFV,
                tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                tma_atom_dQ, tma_atom_dK, tma_atom_dV,
                self.sQ_layout, self.sQ4_layout, self.sfq_smem_layout_staged,
                None,
                None,
                None,
                None,
                self.sfds_mx_layout, self.sfqc_layout, self.sfdsq_layout, self.sfkc_layout,
                self.sfp_mx_layout, self.sfdoc_layout,
                self.sK_layout, self.sK4_layout, self.sfk_smem_layout_staged, self.sK_half_layout,
                self.sV4_layout, self.sfv_smem_layout_staged,
                self.sdO_layout, self.sdO4_layout, self.sfdo_smem_layout_staged,
                self.sLSE_layout, self.sdPsum_layout,
                self.sdSt_layout, self.sdS_layout, self.sdQ_layout, self.sdK_layout, self.sdV_layout,
                self.tP_layout, self.tdS_layout,
                self.tiled_mma_S, self.tiled_mma_dP, self.tiled_mma_dV, self.tiled_mma_dK, self.tiled_mma_dQ, self.tiled_mma_sfb_S,
                self.tiled_mma_sfb_q, self.tiled_mma_dQh,
                tiled_copy_r2s_dKV,
                softmax_scale, softmax_scale_log2,
                dpsum_scale, ds_boost, sr_seed, dv_scale, dk_scale, dq_scale,
                window_size_left, window_size_right,
                tile_sched_params,
                mSeqUsedK,
                mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd,
                self.sfk_mx_layout, self.sfq_mx_layout, self.sfv_mx_layout, self.sfdo_mx_layout, mFac, mTokFac,
                mScalars, mDocTab, mNDoc, aux_tensors, mSeed,
                *((mVtQ,) if const_expr(self.codes_packed) else ()),
                *((mSfd,) if const_expr(self.sf_dense) else ()),
            ).launch(
                grid=grid_dim,
                block=[self.threads_per_cta, 1, 1],
                cluster=None,
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=1,
            )

    def _set_bwd_persist(self, v: bool):
        """(trace time) the two launches of a capped work-list backward: per-position code / persistent code."""
        self.bwd_persist = v

    @cute.kernel
    def kernel(
        self,
        mQ: Optional[cute.Tensor], mQ4: cute.Tensor, mSFQ: Optional[cute.Tensor],
        mSFQt: Optional[cute.Tensor], mSFdOt: Optional[cute.Tensor],
        mK: Optional[cute.Tensor], mK4: cute.Tensor, mSFK: Optional[cute.Tensor],
        mV4: cute.Tensor, mSFV: Optional[cute.Tensor],
        mdO: Optional[cute.Tensor], mdO4: cute.Tensor, mSFdO: Optional[cute.Tensor],
        mLSE: cute.Tensor, mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor, mdK_tma_tensor: cute.Tensor, mdV_tma_tensor: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor,
        mdQ_semaphore: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom], tma_atom_Q4: cute.CopyAtom, tma_atom_SFQ: Optional[cute.CopyAtom],
        tma_atom_SFQt: Optional[cute.CopyAtom], tma_atom_SFdOt: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom], tma_atom_K4: cute.CopyAtom, tma_atom_SFK: Optional[cute.CopyAtom],
        tma_atom_V4: cute.CopyAtom, tma_atom_SFV: Optional[cute.CopyAtom],
        tma_atom_dO: Optional[cute.CopyAtom], tma_atom_dO4: cute.CopyAtom, tma_atom_SFdO: Optional[cute.CopyAtom],
        tma_atom_dQ: cute.CopyAtom, tma_atom_dK: cute.CopyAtom, tma_atom_dV: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout, sQ4_layout: cute.ComposedLayout, sfq_layout: Optional[cute.Layout],
        sfqt_layout: Optional[cute.Layout], sfdot_layout: Optional[cute.Layout],
        sfp_layout: Optional[cute.Layout], sfds_layout: Optional[cute.Layout],
        sfds_mx_layout: Optional[cute.Layout], sfqc_layout: Optional[cute.Layout],
        sfdsq_layout: Optional[cute.Layout], sfkc_layout: Optional[cute.Layout],
        sfp_mx_layout: Optional[cute.Layout], sfdoc_layout: Optional[cute.Layout],
        sK_layout: cute.ComposedLayout, sK4_layout: cute.ComposedLayout, sfk_layout: Optional[cute.Layout],
        sK_half_layout: cute.ComposedLayout,
        sV4_layout: cute.ComposedLayout, sfv_layout: Optional[cute.Layout],
        sdO_layout: cute.ComposedLayout, sdO4_layout: cute.ComposedLayout, sfdo_layout: Optional[cute.Layout],
        sLSE_layout: cute.Layout, sdPsum_layout: cute.Layout,
        sdSt_layout: cute.ComposedLayout, sdS_layout: cute.ComposedLayout, sdQ_layout: cute.ComposedLayout,
        sdK_layout: cute.ComposedLayout, sdV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout, tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma, tiled_mma_dP: cute.TiledMma, tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma, tiled_mma_dQ: cute.TiledMma, tiled_mma_sfb_S: cute.TiledMma,
        tiled_mma_sfb_q: cute.TiledMma, tiled_mma_dQh: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: Float32, softmax_scale_log2: Float32,
        dpsum_scale: Float32, ds_boost: Float32, sr_seed: Int32, dv_scale: Float32, dk_scale: Float32, dq_scale: Float32,
        window_size_left: Optional[Int32], window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mSFQmx: Optional[cute.Tensor] = None, mSFKmx: Optional[cute.Tensor] = None,
        mSFVmx: Optional[cute.Tensor] = None, mSFdOmx: Optional[cute.Tensor] = None,
        mSFQd: Optional[cute.Tensor] = None, mSFKd: Optional[cute.Tensor] = None, mSFdOd: Optional[cute.Tensor] = None,
        sfk_mx_layout: Optional[cute.Layout] = None, sfq_mx_layout: Optional[cute.Layout] = None,
        sfv_mx_layout: Optional[cute.Layout] = None, sfdo_mx_layout: Optional[cute.Layout] = None,
        mFac: Optional[cute.Tensor] = None,
        mTokFac: Optional[cute.Tensor] = None,
        mScalars: Optional[cute.Tensor] = None,
        mDocTab: Optional[cute.Tensor] = None,
        mNDoc: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        mSeed: Optional[cute.Tensor] = None,
        mVtQ: Optional[cute.Tensor] = None,
        mSfd: Optional[list] = None,
    ):
        if const_expr(self.doc_scaled):
            # per-(document, kv head) scalars: the (n_docs + 1, h_kv, 8) table, read per work tile (_doc_scalars)
            self._doc_scal = mScalars
        elif const_expr(mScalars is not None):
            # GPU-only scales: the float32 values the host would have passed, read from device memory; the log2 scale is
            # derived exactly as __call__ derives it
            softmax_scale = mScalars[0]
            softmax_scale_log2 = softmax_scale * math.log2(math.e)
            dpsum_scale = mScalars[1]
            ds_boost = mScalars[2]
            dv_scale = mScalars[3]
            dk_scale = mScalars[4]
            dq_scale = mScalars[5]
            if const_expr(self.dq_tile_scaled):
                self._dq_tab = mScalars   # dqfix: the per-row dQ scales after the vector (dQacc_reduce)
                if const_expr(self.codes_packed):
                    self._pk_ntq = cute.size(mVtQ.shape[1])   # (the aligned query tiles: b = 1)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                for atom in (tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_K, tma_atom_K4, tma_atom_SFK,
                             tma_atom_V4, tma_atom_SFV, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                             tma_atom_dQ, tma_atom_dK, tma_atom_dV, tma_atom_SFQt, tma_atom_SFdOt):
                    if const_expr(atom is not None):
                        cpasync.prefetch_descriptor(atom)

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_S.thr_id.shape,))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # "S loaded" mbarrier: the compute warps (one thread after their post-load barrier) tell the MMA
        # warp that the S bank's free columns may receive the parked SFV / SFdO. Initialised here; the
        # dO pipeline's create() below performs the CTA-wide sync that publishes it.
        S_loaded_mbar_ptr = storage.S_loaded_mbar_ptr
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(S_loaded_mbar_ptr, len(self.compute_warp_ids) if self.warp_sync else 1)
                if const_expr(self.det_issue):
                    _nb = self.det_issue_bufs
                    for _b in cutlass.range_constexpr(_nb):
                        cute.arch.mbarrier_init(storage.dQ_stage_mbar_ptr.data_ptr() + _b,
                                                cute.arch.WARP_SIZE * len(self.reduce_warp_ids))
                        cute.arch.mbarrier_init(storage.dQ_stage_mbar_ptr.data_ptr() + _nb + _b, 1)
                    cute.arch.mbarrier_init(storage.dQ_stage_mbar_ptr.data_ptr() + 2 * _nb, 1)
                    cute.arch.mbarrier_init(storage.dQ_stage_mbar_ptr.data_ptr() + 2 * _nb + 1, 1)
                if const_expr(self.sf_dense):
                    for _b in cutlass.range_constexpr(4):
                        cute.arch.mbarrier_init(storage.Stg_mbar_ptr.data_ptr() + _b, 1)
        cute.arch.mbarrier_init_fence()

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # pipelines (as in the BF16 kernel, 1-CTA)
        producer_group_MMA = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_compute = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids))
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.S_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=producer_group_MMA, consumer_group=consumer_group_compute,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        consumer_group_reduce = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids))
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dq_stages, producer_group=producer_group_MMA, consumer_group=consumer_group_reduce,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)),
            consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
            barrier_storage=storage.dS_mbar_ptr.data_ptr(), cta_layout_vmnk=cluster_layout_vmnk,
        )
        producer_group_load = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_mma = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        consumer_group_compute_tma = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids))
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(), num_stages=self.Q_stage,
            producer_group=producer_group_load, consumer_group=consumer_group_compute_tma,
            tx_count=self.tma_copy_bytes["LSE"], defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(), num_stages=self.dO_stage,
            producer_group=producer_group_load, consumer_group=consumer_group_compute_tma,
            tx_count=self.tma_copy_bytes["dPsum"], defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(), num_stages=self.Q_stage,
            producer_group=producer_group_load, consumer_group=consumer_group_mma,
            tx_count=self.tma_copy_bytes["Q"], cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(), num_stages=self.dO_stage,
            producer_group=producer_group_load, consumer_group=consumer_group_mma,
            tx_count=self.tma_copy_bytes["dO"], cta_layout_vmnk=cluster_layout_vmnk, defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_b_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_b_dtype)
        sV4 = storage.sV4.get_tensor(sV4_layout.outer, swizzle=sV4_layout.inner)
        if const_expr(self.fp8):
            # the E4M3 tiles loaded K-major for S / dP, viewed MN-major for dK / dV / dQ (same bytes)
            sQ4 = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQ4_layout.inner), sQ4_layout.outer)
            sK4 = cute.make_tensor(cute.recast_ptr(sK.iterator, sK4_layout.inner), sK4_layout.outer)
            sdO4 = cute.make_tensor(cute.recast_ptr(sdO.iterator, sdO4_layout.inner), sdO4_layout.outer)
            sSFQ = sSFK = sSFV = sSFdO = None
        else:
            sQ4 = storage.sQ4.get_tensor(sQ4_layout.outer, swizzle=sQ4_layout.inner)
            sK4 = storage.sK4.get_tensor(sK4_layout.outer, swizzle=sK4_layout.inner)
            sdO4 = cute.make_tensor(cute.recast_ptr(sdO.iterator, sdO4_layout.inner), sdO4_layout.outer)
            sSFQ = storage.sSFQ.get_tensor(sfq_layout)
            sSFK = storage.sSFK.get_tensor(sfk_layout)
            sSFV = None
            sSFdO = None
        sSFQt = None
        sSFdOt = None
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        # block-scaled d\hat{S}: typed (UE8M0, SF atom layout) views for the S2T copies and flat byte views for the
        # compute warps' scale stores
        sSFdS_mx = sSFQc = sSFdSq = sSFKc = sSFdS_mx_u8 = sSFdSq_u8 = None
        sRowX = sColX = sColE = sQscale = None
        if const_expr(self.mxds):
            sSFdS_mx = storage.sSFdS_mx.get_tensor(sfds_mx_layout, dtype=self.sf_dtype_mx)
            sSFQc = storage.sSFQc.get_tensor(sfqc_layout, dtype=self.sf_dtype_mx)
            sSFdS_mx_u8 = storage.sSFdS_mx.get_tensor(cute.make_layout(cute.cosize(sfds_mx_layout)))
            if const_expr(self.mxds_dq_sf):
                sSFdSq = storage.sSFdSq.get_tensor(sfdsq_layout, dtype=self.sf_dtype_mx)
                sSFKc = storage.sSFKc.get_tensor(sfkc_layout, dtype=self.sf_dtype_mx)
                sSFdSq_u8 = storage.sSFdSq.get_tensor(cute.make_layout(cute.cosize(sfdsq_layout)))
        sSFP_mx = sSFdOc = sSFP_mx_u8 = None
        sSFK_mx = sSFQ_mx = sSFV_mx = sSFdO_mx = None
        sSFK_mx_u8 = sSFQ_mx_u8 = sSFV_mx_u8 = sSFdO_mx_u8 = sSFQc_u8 = sSFKc_u8 = sSFdOc_u8 = None
        if const_expr(self.mxop):
            if const_expr(self.mx_dp):
                sSFV_mx = storage.sSFV_mx.get_tensor(sfv_mx_layout, dtype=self.sf_dtype_mx)
                sSFdO_mx = storage.sSFdO_mx.get_tensor(sfdo_mx_layout, dtype=self.sf_dtype_mx)
                sSFV_mx_u8 = storage.sSFV_mx.get_tensor(cute.make_layout(cute.cosize(sfv_mx_layout)))
                sSFdO_mx_u8 = storage.sSFdO_mx.get_tensor(cute.make_layout(cute.cosize(sfdo_mx_layout)))
            sSFQc_u8 = storage.sSFQc.get_tensor(cute.make_layout(cute.cosize(sfqc_layout)))
            sSFKc_u8 = storage.sSFKc.get_tensor(cute.make_layout(cute.cosize(sfkc_layout)))
            sSFdOc_u8 = storage.sSFdOc.get_tensor(cute.make_layout(cute.cosize(sfdoc_layout)))
            if const_expr(self.mxop_s):
                sSFK_mx = storage.sSFK_mx.get_tensor(sfk_mx_layout, dtype=self.sf_dtype_mx)
                sSFQ_mx = storage.sSFQ_mx.get_tensor(sfq_mx_layout, dtype=self.sf_dtype_mx)
                sSFK_mx_u8 = storage.sSFK_mx.get_tensor(cute.make_layout(cute.cosize(sfk_mx_layout)))
                sSFQ_mx_u8 = storage.sSFQ_mx.get_tensor(cute.make_layout(cute.cosize(sfq_mx_layout)))
        if const_expr(self.mxdv):
            sSFP_mx = storage.sSFP_mx.get_tensor(sfp_mx_layout, dtype=self.sf_dtype_mx)
            sSFdOc = storage.sSFdOc.get_tensor(sfdoc_layout, dtype=self.sf_dtype_mx)
            sSFP_mx_u8 = storage.sSFP_mx.get_tensor(cute.make_layout(cute.cosize(sfp_mx_layout)))
        if const_expr(self.mxds_row):
            sRowX = storage.sRowX.get_tensor(cute.make_layout((2, 2, self.tile_n), stride=(2 * self.tile_n, self.tile_n, 1)))
            sColX = storage.sColX.get_tensor(cute.make_layout((2, 2, 4, 2, 32), stride=(2 * 4 * 2 * 32, 4 * 2 * 32, 2 * 32, 32, 1)))
            sColE = storage.sColE.get_tensor(cute.make_layout((2, 2, 2, 32), stride=(2 * 2 * 32, 2 * 32, 32, 1)))
            sQscale = storage.sQscale.get_tensor(cute.make_layout(self.tile_m))
        if const_expr(self.ds_dq_rowlocal != "off"):
            sColE = storage.sColE.get_tensor(cute.make_layout((32, 2, 8), stride=(1, 32, 64)))
        sTokV = None
        if const_expr(self.token_local_arg):
            # token-local per-query factors: [query 32][stage 2][compute warp 8][vector: f_Q, f_dO], after sColE's 512
            sTokV = cute.make_tensor(sColE.iterator + 512, cute.make_layout((32, 2, 8, 2), stride=(1, 32, 64, 512)))
        sSFx = storage.sSFx.get_tensor(cute.make_layout((2, self.num_compute_wgs, self.tile_n, 2),
                                                        stride=(self.num_compute_wgs * self.tile_n * 2, self.tile_n * 2, 2, 1)))
        # MN-major (queries, keys) view of the same bytes: A operand of dQ = dS K
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        sdV = storage.sdO.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype)
        sdK = storage.sQ.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype)
        sdQ = storage.sdQaccum.get_tensor(sdQ_layout.outer, swizzle=sdQ_layout.inner)
        sDsScr = storage.sDsScr.get_tensor(cute.make_layout(16))

        # TMEM tensors (the allocation always covers 512 columns at address 0)
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        thr_mma_S = tiled_mma_S.get_slice(0)
        tStS = thr_mma_S.make_fragment_C(thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2]))
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        tdPtdP = thr_mma_dP.make_fragment_C(thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2]))
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        tdVtdV = thr_mma_dV.make_fragment_C(thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2]))
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        p_mma_dtype = self.do_dtype
        tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=p_mma_dtype), tP_layout.outer)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        tdKtdK = thr_mma_dK.make_fragment_C(thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2]))
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        ds_mma_dtype = self.ds_dtype
        tdS = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=ds_mma_dtype), tdS_layout.outer)
        # dS store target of the compute warps: the dP-shaped fp32 view at the dS region (= dP bank)
        tdStdS_f32 = cute.make_tensor(tmem_ptr + self.tmem_dS_offset, tdPtdP.layout)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        tdQtdQ = thr_mma_dQ.make_fragment_C(thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2]))
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)
        tdQtdQ_h = None
        sK_h = None
        # scale-factor TMEM tensors
        sf_align = 16

        def _sf_tmem(col, layout):
            return cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=sf_align), dtype=self.sf_dtype),
                layout,
            )

        tCtSFK = tCtSFV = tCtSFQs = tCtSFdOs = None
        if const_expr(not self.fp8):
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, cute.slice_(sfk_layout, (None, None, None, 0))
            )
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, cute.slice_(sfq_layout, (None, None, None, 0))
            )
            tCtSFK = _sf_tmem(self.tmem_SFK_offset, tCtSFK_layout)
            tCtSFV = None
            tCtSFQs = [_sf_tmem(self.tmem_SFQ_offset[s], tCtSFQ_layout) for s in range(self.Q_stage)]
            tCtSFdOs = None
        # C3: SF tensors of the dV / dK GEMMs (A = \hat{P} / d\hat{S} written by the compute warps, B = dO^T / Q^T)
        tCtSFP = tCtSFdOts = tCtSFdS = tCtSFQts = None
        # block-scaled d\hat{S}: SF TMEM tensors of the dK (A = d\hat{S}, B = Q) and dQ (A = d\hat{S}^T, B = K) GEMMs
        tCtSFdS_mx = tCtSFQc = tCtSFdSq = tCtSFKc = None
        if const_expr(self.mxds):
            def _sf_tmem_mx(col, layout):
                return cute.make_tensor(
                    cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=sf_align), dtype=self.sf_dtype_mx),
                    layout,
                )
            tCtSFdS_mx = _sf_tmem_mx(self.tmem_SFdS_mx_offset, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, cute.slice_(sfds_mx_layout, (None, None, None, 0))))
            tCtSFQc = _sf_tmem_mx(self.tmem_SFQc_offset, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, cute.slice_(sfqc_layout, (None, None, None, 0))))
            if const_expr(self.mxds_dq_sf):
                tCtSFdSq = _sf_tmem_mx(self.tmem_SFdSq_offset, blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma_dQ, self.mma_tiler_dsk, self.sf_vec_mx, cute.slice_(sfdsq_layout, (None, None, None, 0))))
                tCtSFKc = _sf_tmem_mx(self.tmem_SFKc_offset, blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_dQ, self.mma_tiler_dsk, self.sf_vec_mx, cute.slice_(sfkc_layout, (None, None, None, 0))))
        tCtSFK_mx = tCtSFV_mx = tCtSFdO_mx = None
        tCtSFQ_mx = [None, None]
        if const_expr(self.mx_dp):
            def _sf_tmem_o(col, layout):
                return cute.make_tensor(
                    cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=sf_align), dtype=self.sf_dtype_mx),
                    layout,
                )
            tCtSFV_mx = _sf_tmem_o(self.tmem_SFV_mx_offset, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_mx, cute.slice_(sfv_mx_layout, (None, None, None, 0))))
            tCtSFdO_mx = _sf_tmem_o(self.tmem_SFdO_mx_offset, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dP, self.mma_tiler_vdo, self.sf_vec_mx, cute.slice_(sfdo_mx_layout, (None, None, None, 0))))
            if const_expr(self.mxop_s):
                tCtSFK_mx = _sf_tmem_o(self.tmem_SFK_mx_offset, blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma_S, self.mma_tiler_kq, self.sf_vec_mx, cute.slice_(sfk_mx_layout, (None, None, None, 0))))
                tCtSFQ_mx = [_sf_tmem_o(self.tmem_SFQ_mx_offset[st], blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_S, self.mma_tiler_kq, self.sf_vec_mx, cute.slice_(sfq_mx_layout, (None, None, None, 0))))
                    for st in range(self.Q_stage)]
        tCtSFP_mx = tCtSFdOc = None
        if const_expr(self.mxdv):
            def _sf_tmem_p(col, layout):
                return cute.make_tensor(
                    cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=sf_align), dtype=self.sf_dtype_mx),
                    layout,
                )
            tCtSFP_mx = _sf_tmem_p(self.tmem_SFP_mx_offset, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, cute.slice_(sfp_mx_layout, (None, None, None, 0))))
            tCtSFdOc = _sf_tmem_p(self.tmem_SFdOc_offset, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, cute.slice_(sfdoc_layout, (None, None, None, 0))))

        block_info = BlockInfo(
            self.tile_m, self.tile_n, self.is_causal, self.is_local, False,
            window_size_left, window_size_right, qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create, seqlen_q_static=mQ4.shape[0], seqlen_k_static=mK4.shape[0],
            mCuSeqlensQ=None, mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=mSeqUsedK if const_expr(self.seqused_k) else None,
            tile_m=self.tile_m, tile_n=self.tile_n,
        )
        # every role takes a key tile's query range from here (lp_bwd_tiles: batched, or the aligned-packed documents)
        block_info = LPTileInfo(
            block_info, mQ4.shape[0], mK4.shape[0], self.tile_m, self.tile_n,
            mSeqUsedK=mSeqUsedK if const_expr(self.seqused_k) else None, mDocTab=mDocTab, mNDoc=mNDoc,
            docs=mDocTab is not None,
        )
        self.aux_data = _as_aux_data(aux_tensors)
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)
        AttentionMaskCls = partial(
            AttentionMask, self.tile_m, self.tile_n, swap_AB=True,
            window_size_left=window_size_left, window_size_right=window_size_right,
        )

        if warp_idx >= self.empty_warp_ids[0]:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)
            if const_expr(self.det_issue):
                if warp_idx == self.empty_warp_ids[0]:
                    self.dQ_issue(mdQaccum, sdQ, tma_atom_dQ, block_info, TileSchedulerCls, mdQ_semaphore,
                                  storage.dQ_stage_mbar_ptr.data_ptr())
                if warp_idx == self.empty_warp_ids[1]:
                    self.dQ_release(block_info, TileSchedulerCls, mdQ_semaphore, storage.dQ_stage_mbar_ptr.data_ptr())

        if const_expr(self.sf_dense):
            if warp_idx == self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.load_sfd(
                    thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ, tiled_mma_sfb_S.get_slice(0),
                    mQ, mQ4, mSFQ, mK, mK4, mSFK, mV4, mSFV, mdO, mdO4, mSFdO, mLSE, mdPsum,
                    sQ, sQ4, sSFQ, sK, sK4, sSFK, sV4, sSFV, sdO, sdO4, sSFdO, sLSE, sdPsum,
                    tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_K, tma_atom_K4, tma_atom_SFK,
                    tma_atom_V4, tma_atom_SFV, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                    pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum,
                    block_info, SeqlenInfoCls, TileSchedulerCls,
                    mSFQt=mSFQt, mSFdOt=mSFdOt, sSFQt=sSFQt, sSFdOt=sSFdOt,
                    tma_atom_SFQt=tma_atom_SFQt, tma_atom_SFdOt=tma_atom_SFdOt,
                    thr_mma_sfb_q=tiled_mma_sfb_q.get_slice(0),
                    mx=(mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd,
                        sSFQ_mx_u8, sSFK_mx_u8, sSFV_mx_u8, sSFdO_mx_u8, sSFQc_u8, sSFKc_u8, sSFdOc_u8),
                    mSfd=mSfd, sStg=storage.sStg.get_tensor(cute.make_layout(4 * self.sfd_slot)),
                    stg_mbar=storage.Stg_mbar_ptr.data_ptr(), pipeline_dKV=pipeline_dKV,
                )
        if const_expr(not self.sf_dense):
            if warp_idx == self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.load(
                    thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ, tiled_mma_sfb_S.get_slice(0),
                    mQ, mQ4, mSFQ, mK, mK4, mSFK, mV4, mSFV, mdO, mdO4, mSFdO, mLSE, mdPsum,
                    sQ, sQ4, sSFQ, sK, sK4, sSFK, sV4, sSFV, sdO, sdO4, sSFdO, sLSE, sdPsum,
                    tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_K, tma_atom_K4, tma_atom_SFK,
                    tma_atom_V4, tma_atom_SFV, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                    pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum,
                    block_info, SeqlenInfoCls, TileSchedulerCls,
                    mSFQt=mSFQt, mSFdOt=mSFdOt, sSFQt=sSFQt, sSFdOt=sSFdOt,
                    tma_atom_SFQt=tma_atom_SFQt, tma_atom_SFdOt=tma_atom_SFdOt,
                    thr_mma_sfb_q=tiled_mma_sfb_q.get_slice(0),
                    mx=(mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd,
                        sSFQ_mx_u8, sSFK_mx_u8, sSFV_mx_u8, sSFdO_mx_u8, sSFQc_u8, sSFKc_u8, sSFdOc_u8),
                    **({"mVtQ": mVtQ} if const_expr(self.codes_packed) else {}),
                )

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.mma(
                tiled_mma_S, tiled_mma_dP, tiled_mma_dV, tiled_mma_dK, tiled_mma_dQ,
                sQ, sQ4, sK, sK4, sV4, sdO, sdO4, sSFQ, sSFK, sSFV, sSFdO,
                tP, sdS, tdS, tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ,
                tCtSFK, tCtSFV, tCtSFQs, tCtSFdOs,
                pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dKV, pipeline_dP, pipeline_dQ,
                S_loaded_mbar_ptr,
                block_info, SeqlenInfoCls, TileSchedulerCls,
                sSFQt=sSFQt, sSFdOt=sSFdOt, tCtSFP=tCtSFP, tCtSFdOts=tCtSFdOts, tCtSFdS=tCtSFdS, tCtSFQts=tCtSFQts,
                tiled_mma_dQh=tiled_mma_dQh, tdQtdQ_h=tdQtdQ_h, sK_h=sK_h,
                sSFdS_mx=sSFdS_mx, sSFQc=sSFQc, sSFdSq=sSFdSq, sSFKc=sSFKc, sDsScr=sDsScr,
                tCtSFdS_mx=tCtSFdS_mx, tCtSFQc=tCtSFQc, tCtSFdSq=tCtSFdSq, tCtSFKc=tCtSFKc,
                sSFP_mx=sSFP_mx, sSFdOc=sSFdOc, tCtSFP_mx=tCtSFP_mx, tCtSFdOc=tCtSFdOc,
                sSFK_mx=sSFK_mx, sSFQ_mx=sSFQ_mx, sSFV_mx=sSFV_mx, sSFdO_mx=sSFdO_mx,
                tCtSFK_mx=tCtSFK_mx, tCtSFQ_mx=tCtSFQ_mx, tCtSFV_mx=tCtSFV_mx, tCtSFdO_mx=tCtSFdO_mx,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
                tStS, tdPtdP, tdStdS_f32, tdVtdV, tdKtdK, sLSE, sdPsum, sdSt, S_loaded_mbar_ptr,
                pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dKV, pipeline_dP, pipeline_dQ,
                softmax_scale, softmax_scale_log2, block_info, SeqlenInfoCls, AttentionMaskCls, TileSchedulerCls,
                sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK, tiled_copy_r2s_dKV,
                sSFx=sSFx,
                dpsum_scale=dpsum_scale, ds_boost=ds_boost, sr_seed=sr_seed, dv_scale_in=dv_scale, dk_scale_in=dk_scale,
                sr_ctr=mSeed,
                sSFdS_mx_u8=sSFdS_mx_u8, sSFdSq_u8=sSFdSq_u8, sDsScr=sDsScr,
                sRowX=sRowX, sColX=sColX, sColE=sColE, sQscale=sQscale, sSFP_mx_u8=sSFP_mx_u8, mFac=mFac, mTokFac=mTokFac, sTokV=sTokV, mSfd=mSfd,
                **({"mdK_g": mdK, "mdV_g": mdV} if const_expr(self.dkv_rows_packed) else {}),
            )
            tmem_alloc_barrier.arrive()

        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            if const_expr(self.num_regs_reduce >= 128):
                cute.arch.setmaxregister_increase(self.num_regs_reduce)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum, sdQ, tma_atom_dQ, thr_mma_dQ, tdQtdQ, pipeline_dQ,
                block_info, SeqlenInfoCls, TileSchedulerCls, mdQ_semaphore,
                dq_scale=dq_scale, sQscale=sQscale,
                stage_mbar=storage.dQ_stage_mbar_ptr.data_ptr() if const_expr(self.det_issue) else None,
            )
            tmem_alloc_barrier.arrive()
        return

    # ------------------------------------------------------------------ load warp
    @cute.jit
    def load(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ, thr_mma_sfb,
        mQ, mQ4, mSFQ, mK, mK4, mSFK, mV4, mSFV, mdO, mdO4, mSFdO, mLSE, mdPsum,
        sQ, sQ4, sSFQ, sK, sK4, sSFK, sV4, sSFV, sdO, sdO4, sSFdO, sLSE, sdPsum,
        tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_K, tma_atom_K4, tma_atom_SFK,
        tma_atom_V4, tma_atom_SFV, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum,
        block_info, SeqlenInfoCls, TileSchedulerCls,
        mSFQt=None, mSFdOt=None, sSFQt=None, sSFdOt=None, tma_atom_SFQt=None, tma_atom_SFdOt=None, thr_mma_sfb_q=None,
        mx=None,
        mVtQ=None,
    ):
        (mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd,
         sSFQ_mx_u8, sSFK_mx_u8, sSFV_mx_u8, sSFdO_mx_u8, sSFQc_u8, sSFKc_u8, sSFdOc_u8) = mx
        producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        one = cute.make_layout(1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            pk = None   # codes_packed: (Q-side row shift, K-side row shift) of this work tile
            if const_expr(self.codes_packed):
                pk = self._pk_shifts(seqlen, m_blk_off, n_block, n_blk_loc)

            if const_expr(self.qhead_per_kvhead > 1):
                # ---- GQA: the work tile is (n_block, kv head); the q side streams r q heads x m blocks
                r_q = self.qhead_per_kvhead
                mK4_cur = self._pk_rows(mK4[None, None, head_idx, batch_idx], pk, 1, 0)  # kv head
                mV4_cur = self._pk_rows(mV4[None, None, head_idx, batch_idx], pk, 1, 0)
                gK4 = cute.local_tile(mK4_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
                load_K4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K4, 0, one, thr_mma_S.partition_A(gK4), sK4, single_stage=True)
                gV4 = cute.local_tile(mV4_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
                load_V4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4, 0, one, thr_mma_dP.partition_A(gV4), sV4, single_stage=True)
                load_SFK = load_SFV = load_K = self._load_noop
                if const_expr(not self.fp8):
                    mK_cur = self._pk_rows(mK[None, None, head_idx, batch_idx], pk, 1, 1)
                    mSFK_cur = mSFK[None, None, head_idx, batch_idx]
                    gSFK = cute.local_tile(mSFK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
                    load_SFK, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_SFK, 0, one, thr_mma_S.partition_A(gSFK), sSFK, filter_zeros=True, single_stage=True
                    )
                    if const_expr(not self.skip_dq):
                        gKt = cute.local_tile(mK_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block))
                        load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, one, thr_mma_dQ.partition_B(gKt), sK, single_stage=True)
                qa = (pk, one, mQ, mQ4, mSFQ, mdO, mdO4, mSFdO, mLSE, mdPsum, sQ, sQ4, sSFQ, sdO, sdO4, sSFdO,
                      tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                      mSFQt, mSFdOt, sSFQt, sSFdOt, tma_atom_SFQt, tma_atom_SFdOt,
                      thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_sfb, thr_mma_sfb_q)
                copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
                copy_stats = partial(cute.copy, copy_atom_stats)
                process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
                if process_tile:
                    nm = m_block_max - m_block_min
                    _, m0 = self._gqa_qm(Int32(0), nm, m_block_min, n_block)
                    (lQ4, ldO4, lQ, ldO, lSFQ, lSFdO, lSFQt, lSFdOt, gLSE, gdPsum) = self._q_loaders(head_idx * r_q, batch_idx, *qa)
                    pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                    bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                    load_K(tma_bar_ptr=bar)
                    load_K4(tma_bar_ptr=bar)
                    load_SFK(tma_bar_ptr=bar)
                    self._mx_load_kv(mSFKmx, mSFKd, sSFK_mx_u8, sSFKc_u8, head_idx, batch_idx, n_block, bar)
                    self._mx_load_q(mSFQmx, mSFQd, sSFQ_mx_u8, sSFQc_u8, head_idx * r_q, batch_idx, m0, producer_state_Q.index, bar)
                    lQ(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lQ4(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lSFQ(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lSFQt(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    pipeline_Q.producer_commit(producer_state_Q)
                    pipeline_LSE.producer_acquire(producer_state_Q)
                    with cute.arch.elect_one():
                        copy_stats(gLSE[None, m0], sLSE[None, producer_state_Q.index], mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q))
                    producer_state_Q.advance()
                    pipeline_dO.producer_acquire(producer_state_dO, extra_tx_count=self.tma_copy_bytes["V"])
                    bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                    load_V4(tma_bar_ptr=bar)
                    load_SFV(tma_bar_ptr=bar)
                    self._mx_load_kv(mSFVmx, None, sSFV_mx_u8, None, head_idx, batch_idx, n_block, bar)
                    self._mx_load_q(mSFdOmx, mSFdOd, sSFdO_mx_u8, sSFdOc_u8, head_idx * r_q, batch_idx, m0, producer_state_dO.index, bar)
                    ldO(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    ldO4(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    lSFdO(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    lSFdOt(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    pipeline_dO.producer_commit(producer_state_dO)
                    pipeline_dPsum.producer_acquire(producer_state_dO)
                    with cute.arch.elect_one():
                        copy_stats(gdPsum[None, m0], sdPsum[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                    producer_state_dO.advance()
                    for j in cutlass.range(1, r_q * nm, unroll=1):
                        qh, m_blk = self._gqa_qm(j, nm, m_block_min, n_block)
                        (jQ4, jdO4, jQ, jdO, jSFQ, jSFdO, jSFQt, jSFdOt, jLSE, jdPsum) = self._q_loaders(head_idx * r_q + qh, batch_idx, *qa)
                        pipeline_Q.producer_acquire(producer_state_Q)
                        bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                        self._mx_load_q(mSFQmx, mSFQd, sSFQ_mx_u8, sSFQc_u8, head_idx * r_q + qh, batch_idx, m_blk, producer_state_Q.index, bar)
                        jQ(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        jQ4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        jSFQ(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        jSFQt(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        pipeline_Q.producer_commit(producer_state_Q)
                        pipeline_LSE.producer_acquire(producer_state_Q)
                        with cute.arch.elect_one():
                            copy_stats(jLSE[None, m_blk], sLSE[None, producer_state_Q.index], mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q))
                        producer_state_Q.advance()
                        pipeline_dO.producer_acquire(producer_state_dO)
                        bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                        self._mx_load_q(mSFdOmx, mSFdOd, sSFdO_mx_u8, sSFdOc_u8, head_idx * r_q + qh, batch_idx, m_blk, producer_state_dO.index, bar)
                        jdO(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        jdO4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        jSFdO(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        jSFdOt(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        pipeline_dO.producer_commit(producer_state_dO)
                        pipeline_dPsum.producer_acquire(producer_state_dO)
                        with cute.arch.elect_one():
                            copy_stats(jdPsum[None, m_blk], sdPsum[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                        producer_state_dO.advance()
                    pipeline_Q.producer_tail(producer_state_Q.clone())
                    pipeline_LSE.producer_tail(producer_state_Q)
                    pipeline_dO.producer_tail(producer_state_dO.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO)
            else:
                mQ4_cur = self._pk_rows(mQ4[None, None, head_idx, batch_idx], pk, 0, 0)  # (s_q, d)
                mK4_cur = self._pk_rows(mK4[None, None, head_idx, batch_idx], pk, 1, 0)  # (s_k, d)
                mV4_cur = self._pk_rows(mV4[None, None, head_idx, batch_idx], pk, 1, 0)
                mdO4_cur = self._pk_rows(mdO4[None, None, head_idx, batch_idx], pk, 0, 0)  # (s_q, dv)
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                mdPsum_cur = mdPsum[None, head_idx, batch_idx]

                # S^T = K4 Q4^T
                gK4 = cute.local_tile(mK4_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
                load_K4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K4, 0, one, thr_mma_S.partition_A(gK4), sK4, single_stage=True)
                gQ4 = cute.local_tile(mQ4_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
                load_Q4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q4, 0, one, thr_mma_S.partition_B(gQ4), sQ4)
                # dP^T = V4 dO4^T
                gV4 = cute.local_tile(mV4_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
                load_V4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4, 0, one, thr_mma_dP.partition_A(gV4), sV4, single_stage=True)
                gdO4 = cute.local_tile(mdO4_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0))
                load_dO4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO4, 0, one, thr_mma_dP.partition_B(gdO4), sdO4)

                # FP4 mode: scale factors of the four operands, the dK / dV B operands (bf16 Q / dO or NVFP4 Q^T / dO^T
                # with their SFs) and the bf16 K of dQ. FP8 mode: none of these (the E4M3 tiles above serve every GEMM).
                # (partial objects, not local closures: the DSL forbids closures with captures inside dynamic control flow)
                load_SFK = load_SFV = load_Q = load_dO = load_K = self._load_noop
                load_SFQ = load_SFdO = load_SFQt = load_SFdOt = self._load_noop
                if const_expr(not self.fp8):
                    mQ_cur = self._pk_rows(mQ[None, None, head_idx, batch_idx], pk, 0, 1)  # (d, s_q) view (dK B operand)
                    mSFQ_cur = mSFQ[None, None, head_idx, batch_idx]
                    mK_cur = self._pk_rows(mK[None, None, head_idx, batch_idx], pk, 1, 1)  # (d, s_k) view (dQ^T A operand)
                    mSFK_cur = mSFK[None, None, head_idx, batch_idx]
                    mdO_cur = self._pk_rows(mdO[None, None, head_idx, batch_idx], pk, 0, 1)  # (dv, s_q)
                    gSFK = cute.local_tile(mSFK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
                    load_SFK, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_SFK, 0, one, thr_mma_S.partition_A(gSFK), sSFK, filter_zeros=True, single_stage=True
                    )
                    # SFQ: the SF atom spans 128 queries, so the tile index along the queries is m_block * tile_m / 128
                    gSFQ = cute.local_tile(mSFQ_cur, cute.select(self.mma_tiler_sfb, mode=[1, 2]), (None, 0))
                    tSgSFQ = thr_mma_sfb.partition_B(gSFQ)
                    tQsSFQ, tQgSFQ = cpasync.tma_partition(
                        tma_atom_SFQ, 0, one, cute.group_modes(sSFQ, 0, 3), cute.group_modes(tSgSFQ, 0, 3)
                    )
                    load_SFQ = partial(self._load_sf_stage, tma_atom_SFQ, cute.filter_zeros(tQgSFQ), cute.filter_zeros(tQsSFQ), self.tile_m // 128)
                    # dK B operand: bf16 Q (MN-major) or NVFP4 Q^T (K-major); dV B: bf16 dO or NVFP4 dO^T
                    if const_expr(not self.skip_dk):
                        gQt = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None))
                        load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, one, thr_mma_dK.partition_B(gQt), sQ)
                    # dQ B operand: bf16 K (MN-major view)
                    if const_expr(not self.skip_dq):
                        gKt = cute.local_tile(mK_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block))
                        load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, one, thr_mma_dQ.partition_B(gKt), sK, single_stage=True)

                gLSE = cute.local_tile(mLSE_cur, (self.lse_tile,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
                copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
                copy_stats = partial(cute.copy, copy_atom_stats)

                process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
                if process_tile:
                    # prologue: K (+ SF / bf16 K in FP4 mode) with Q(first); V (+ SF) with dO(first)
                    m0 = m_block_min
                    pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                    bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                    load_K(tma_bar_ptr=bar)
                    load_K4(tma_bar_ptr=bar)
                    load_SFK(tma_bar_ptr=bar)
                    self._mx_load_kv(mSFKmx, mSFKd, sSFK_mx_u8, sSFKc_u8, head_idx, batch_idx, n_block, bar)
                    self._mx_load_q(mSFQmx, mSFQd, sSFQ_mx_u8, sSFQc_u8, head_idx, batch_idx, m0, producer_state_Q.index, bar)
                    load_Q(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    load_Q4(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    load_SFQ(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    load_SFQt(src_idx=m0, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    pipeline_Q.producer_commit(producer_state_Q)
                    pipeline_LSE.producer_acquire(producer_state_Q)
                    with cute.arch.elect_one():
                        copy_stats(gLSE[None, m0], sLSE[None, producer_state_Q.index], mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q))
                    producer_state_Q.advance()

                    pipeline_dO.producer_acquire(producer_state_dO, extra_tx_count=self.tma_copy_bytes["V"])
                    bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                    load_V4(tma_bar_ptr=bar)
                    load_SFV(tma_bar_ptr=bar)
                    self._mx_load_kv(mSFVmx, None, sSFV_mx_u8, None, head_idx, batch_idx, n_block, bar)
                    self._mx_load_q(mSFdOmx, mSFdOd, sSFdO_mx_u8, sSFdOc_u8, head_idx, batch_idx, m0, producer_state_dO.index, bar)
                    load_dO(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    load_dO4(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    load_SFdO(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    load_SFdOt(src_idx=m0, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    pipeline_dO.producer_commit(producer_state_dO)
                    pipeline_dPsum.producer_acquire(producer_state_dO)
                    with cute.arch.elect_one():
                        copy_stats(gdPsum[None, m0], sdPsum[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                    producer_state_dO.advance()

                    for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                        pipeline_Q.producer_acquire(producer_state_Q)
                        bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                        self._mx_load_q(mSFQmx, mSFQd, sSFQ_mx_u8, sSFQc_u8, head_idx, batch_idx, m_block, producer_state_Q.index, bar)
                        load_Q(src_idx=m_block, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        load_Q4(src_idx=m_block, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        load_SFQ(src_idx=m_block, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        load_SFQt(src_idx=m_block, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                        pipeline_Q.producer_commit(producer_state_Q)
                        pipeline_LSE.producer_acquire(producer_state_Q)
                        with cute.arch.elect_one():
                            copy_stats(gLSE[None, m_block], sLSE[None, producer_state_Q.index], mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q))
                        producer_state_Q.advance()

                        pipeline_dO.producer_acquire(producer_state_dO)
                        bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                        self._mx_load_q(mSFdOmx, mSFdOd, sSFdO_mx_u8, sSFdOc_u8, head_idx, batch_idx, m_block, producer_state_dO.index, bar)
                        load_dO(src_idx=m_block, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        load_dO4(src_idx=m_block, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        load_SFdO(src_idx=m_block, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        load_SFdOt(src_idx=m_block, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                        pipeline_dO.producer_commit(producer_state_dO)
                        pipeline_dPsum.producer_acquire(producer_state_dO)
                        with cute.arch.elect_one():
                            copy_stats(gdPsum[None, m_block], sdPsum[None, producer_state_dO.index], mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO))
                        producer_state_dO.advance()

                    pipeline_Q.producer_tail(producer_state_Q.clone())
                    pipeline_LSE.producer_tail(producer_state_Q)
                    pipeline_dO.producer_tail(producer_state_dO.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ sf_dense (SF redesign): the load warp
    @cute.jit
    def _sfd_word(self, mEv, bh, ps, nr):
        """The 4 32-row blocks' exponents of the tile at packed row ps (nr real rows) as one word (block i in byte i;
        a block past the document 0, as the aligned layout's all-zero block). Every lane loads the same bytes."""
        w = Int32(0)
        for i in cutlass.range_constexpr(4):
            if Int32(32 * i) < nr:
                w = w | ((Int32(mEv[bh, ps + 32 * i]) & 0xFF) << (8 * i))
        return w

    @cute.jit
    def _sfd_dim_atom(self, w, dst_u8, stage):
        """A d-rows SF atom (token blocks: every row's word is the tile's 4 block exponents) into stage `stage` of
        dst_u8: one 16-byte store per lane (words 4 lane .. 4 lane + 3)."""
        lane = cute.arch.lane_idx()
        v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
        for i in cutlass.range_constexpr(4):
            v[i] = w
        cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, dst_u8.iterator.toint() + stage * 512 + 16 * lane,
                                                            cute.AddressSpace.smem, assumed_align=16),
                                              cute.make_layout(4)))

    @cute.jit
    def _sfd_sf4_load(self, mSF4, bh, ps, nr, regs):
        """The NVFP4 S operand's SF atom(s) of the tile at packed row ps from the dense per-row words: lane l holds
        rows l, l + 32, l + 64, l + 96 (the 32 x 4 atom order: words 4 l .. 4 l + 3), 0 past the document."""
        lane = cute.arch.lane_idx()
        nk = self.tile_hdim // 64
        for k in cutlass.range_constexpr(nk):
            for j in cutlass.range_constexpr(4):
                row = lane + 32 * j
                v = Int32(0)
                if row < nr:
                    v = mSF4[bh, (ps + row) * nk + k]
                regs[4 * k + j] = v

    @cute.jit
    def _sfd_sf4_store(self, regs, s_sf, stage, stage_bytes: cutlass.Constexpr[int]):
        lane = cute.arch.lane_idx()
        for k in cutlass.range_constexpr(self.tile_hdim // 64):
            cute.autovec_copy(cute.make_tensor(regs.iterator + 4 * k, cute.make_layout(4)),
                              cute.make_tensor(cute.make_ptr(Int32, s_sf.iterator.toint() + stage * stage_bytes + 512 * k
                                                             + 16 * lane, cute.AddressSpace.smem, assumed_align=16),
                                               cute.make_layout(4)))

    @cute.jit
    def _sfd_packet_issue(self, mLSE, mdPsum, mEvQ, mEvdO, mSF4Q, hq, batch_idx, ps, sStg, slot, mbar):
        """One staging packet (the query tile at packed row ps of q head hq): bulk copies of the shifted LSE and D''
        rows from (ps & ~3), the Q / dO exponent bytes from (ps & ~15), the fp4 route's Q NVFP4 SF words from
        (ps & ~3), all on the slot's mbarrier."""
        nk = self.tile_hdim // 64
        base = sStg.iterator + slot * self.sfd_slot
        p4 = ps - ps % 4
        p16 = ps - ps % 16
        # (per-tensor operands, not mxop_tb: no exponent bytes in the packet)
        # (cov3: token-local operands with a unit d-rows scale -- dO, and Q on the 8-bit route -- need no exponent bytes)
        ev_q = self.mxop_tb and not (self.token_local and self.fp8)
        ev_do = self.mxop_tb and not self.token_local
        nbytes = 2 * 528 + (144 if ev_q else 0) + (144 if ev_do else 0) + (0 if self.fp8 else 528 * nk)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar + slot, nbytes)
            copy_utils.cpasync_bulk_g2s(mLSE[None, hq, batch_idx].iterator + p4, base, mbar + slot, 528)
            copy_utils.cpasync_bulk_g2s(mdPsum[None, hq, batch_idx].iterator + p4, base + 528, mbar + slot, 528)
            if const_expr(ev_q):
                copy_utils.cpasync_bulk_g2s(mEvQ[hq, None].iterator + p16, base + self.sfd_off_e, mbar + slot, 144)
            if const_expr(ev_do):
                copy_utils.cpasync_bulk_g2s(mEvdO[hq, None].iterator + p16, base + self.sfd_off_e + 144, mbar + slot,
                                            144)
            if const_expr(not self.fp8):
                copy_utils.cpasync_bulk_g2s(mSF4Q[hq, None].iterator + p4 * nk, base + self.sfd_off_sf4, mbar + slot,
                                            528 * nk)

    @cute.jit
    def _sfd_packet_word(self, sStg, slot, which: cutlass.Constexpr[int], ps, nr):
        """The tile's 4 block exponents (Q: which 0, dO: 1) from a landed packet as one word (0 past the document)."""
        base = sStg.iterator + slot * self.sfd_slot + self.sfd_off_e + 144 * which + ps % 16
        w = Int32(0)
        for i in cutlass.range_constexpr(4):
            # (& 0xFF: the byte -> Int32 conversion sign-extends exponents >= 128, which would fill the higher bytes)
            b = Int32(cute.make_tensor(base + 32 * i, cute.make_layout(1))[0]) & 0xFF
            w = w | (cutlass.select_(Int32(32 * i) < nr, b, Int32(0)) << (8 * i))
        return w

    @cute.jit
    def _sfd_packet_sf4(self, sStg, slot, ps, nr, s_sf, stage, stage_bytes: cutlass.Constexpr[int]):
        """The Q NVFP4 SF atom(s) of the stage from a landed packet: lane l takes rows l, l + 32, l + 64, l + 96 (one
        16-byte store per atom), 0 past the document."""
        lane = cute.arch.lane_idx()
        nk = self.tile_hdim // 64
        src = cute.make_tensor(cute.recast_ptr(sStg.iterator + slot * self.sfd_slot + self.sfd_off_sf4, dtype=Int32),
                               cute.make_layout(132 * nk))
        off = ps % 4
        for k in cutlass.range_constexpr(nk):
            v = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for j in cutlass.range_constexpr(4):
                r = lane + 32 * j
                v[j] = cutlass.select_(r < nr, src[(off + r) * nk + k], Int32(0))
            cute.autovec_copy(v, cute.make_tensor(cute.make_ptr(Int32, s_sf.iterator.toint() + stage * stage_bytes
                                                                + 512 * k + 16 * lane, cute.AddressSpace.smem,
                                                                assumed_align=16), cute.make_layout(4)))

    @cute.jit
    def _sfd_packet_stat(self, sStg, slot, which: cutlass.Constexpr[int], ps, nr, dst, pad: cutlass.Constexpr[float],
                         fw0, fw1, fw2):
        """Write a statistics stage from a landed packet (LSE: which 0, D'': 1): row r < 128 = staged row (ps & 3) + r,
        `pad` past the document; rows 128 + 4 i + b = the factor of byte b of word fw_i (None: not written)."""
        lane = cute.arch.lane_idx()
        src = cute.make_tensor(cute.recast_ptr(sStg.iterator + slot * self.sfd_slot + 528 * which, dtype=Float32),
                               cute.make_layout(132))
        off = ps % 4
        for j in cutlass.range_constexpr(4):
            r = lane + 32 * j
            dst[r] = cutlass.select_(r < nr, src[off + r], Float32(pad))
        ws = [w for w in (fw0, fw1, fw2) if w is not None]
        if const_expr(len(ws) > 0):   # (per-tensor operands: no factors)
            if lane < 4 * len(ws):
                w = ws[0]
                for i in cutlass.range_constexpr(1, len(ws)):
                    w = cutlass.select_(lane // 4 == i, ws[i], w)
                dst[self.tile_m + lane] = self._sfd_fac((w >> (8 * (lane % 4))) & 0xFF)

    @cute.jit
    def _sfd_qtile(self, seqlen, it, nm, m_min, n_block, m_blk_off, head_idx):
        """(q head, m block, packed start, real rows) of work-tile iteration `it`."""
        qh, m = self._sfd_qm(it, nm, m_min, n_block)
        ps = seqlen.offset_q + (m - m_blk_off) * self.tile_m
        nr = cutlass.min(seqlen.seqlen_q - (m - m_blk_off) * self.tile_m, Int32(self.tile_m))
        return head_idx * self.qhead_per_kvhead + qh, m, ps, nr

    @cute.jit
    def load_sfd(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ, thr_mma_sfb,
        mQ, mQ4, mSFQ, mK, mK4, mSFK, mV4, mSFV, mdO, mdO4, mSFdO, mLSE, mdPsum,
        sQ, sQ4, sSFQ, sK, sK4, sSFK, sV4, sSFV, sdO, sdO4, sSFdO, sLSE, sdPsum,
        tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_K, tma_atom_K4, tma_atom_SFK,
        tma_atom_V4, tma_atom_SFV, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
        pipeline_Q, pipeline_dO, pipeline_LSE, pipeline_dPsum,
        block_info, SeqlenInfoCls, TileSchedulerCls,
        mSFQt=None, mSFdOt=None, sSFQt=None, sSFdOt=None, tma_atom_SFQt=None, tma_atom_SFdOt=None, thr_mma_sfb_q=None,
        mx=None, mSfd=None, sStg=None, stg_mbar=None, pipeline_dKV=None,
    ):
        """sf_dense: the load warp of load()'s token-block path with every per-tile table replaced by the dense per-row
        tables (see the constructor). Per query tile one staging packet (statistics rows, exponent bytes, fp4 SF words;
        4 slots, issued TWO iterations ahead) feeds the stage writes; per work tile the K / V exponents (and K's SF
        words) are loaded once. Each operand stage: the TMA loads first, then the generic writes (SF atoms, statistics,
        the factors after them), a proxy fence and the arrive (release) with the stage's tx bytes (the TMA may land
        before the expect-tx: the phase still needs the arrival)."""
        (mSFQmx, mSFKmx, mSFVmx, mSFdOmx, mSFQd, mSFKd, mSFdOd,
         sSFQ_mx_u8, sSFK_mx_u8, sSFV_mx_u8, sSFdO_mx_u8, sSFQc_u8, sSFKc_u8, sSFdOc_u8) = mx
        mEvQ, mEvK, mEvV, mEvdO = mSfd[0], mSfd[1], mSfd[2], mSfd[3]
        mSF4Q = mSF4K = None
        if const_expr(not self.fp8):
            mSF4Q, mSF4K = mSfd[4], mSfd[5]
        dkv_phase_ld = Int32(1)
        producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        one = cute.make_layout(1)
        r_q = self.qhead_per_kvhead
        nk4 = self.tile_hdim // 64
        sfq_stage_bytes = 512 * nk4
        tx_Q = pipeline_Q.sync_object_full.tx_count
        tx_dO = pipeline_dO.sync_object_full.tx_count
        sf4k = cute.make_rmem_tensor(cute.make_layout(4 * nk4), Int32)
        n_p = Int32(0)   # staging packets issued / consumed (slot = n % 4)
        c_p = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            pk = self._pk_shifts(seqlen, m_blk_off, n_block, n_blk_loc)
            mK4_cur = self._pk_rows(mK4[None, None, head_idx, batch_idx], pk, 1, 0)  # kv head
            mV4_cur = self._pk_rows(mV4[None, None, head_idx, batch_idx], pk, 1, 0)
            gK4 = cute.local_tile(mK4_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            load_K4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K4, 0, one, thr_mma_S.partition_A(gK4), sK4, single_stage=True)
            gV4 = cute.local_tile(mV4_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            load_V4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V4, 0, one, thr_mma_dP.partition_A(gV4), sV4, single_stage=True)
            load_K = self._load_noop
            if const_expr(not self.fp8):
                mK_cur = self._pk_rows(mK[None, None, head_idx, batch_idx], pk, 1, 1)
                if const_expr(not self.skip_dq):
                    gKt = cute.local_tile(mK_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block))
                    load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, one, thr_mma_dQ.partition_B(gKt), sK, single_stage=True)
            qa = (pk, one, mQ, mQ4, mSFQ, mdO, mdO4, mSFdO, mLSE, mdPsum, sQ, sQ4, sSFQ, sdO, sdO4, sSFdO,
                  tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                  mSFQt, mSFdOt, sSFQt, sSFdOt, tma_atom_SFQt, tma_atom_SFdOt,
                  thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_sfb, thr_mma_sfb_q)
            process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
            if process_tile:
                if const_expr(self.bwd_persist):
                    # persistent CTA: sdK / sdV alias sQ / sdO -- the previous work tile's dK / dV epilogue (its TMA
                    # stores read back included) must have released both dKV stages before this tile's Q / dO land
                    # (a CTA's first tile: the barriers' initial phase, no wait)
                    pipeline_dKV.sync_object_empty.wait(0, dkv_phase_ld)
                    pipeline_dKV.sync_object_empty.wait(1, dkv_phase_ld)
                    dkv_phase_ld ^= 1
                nm = m_block_max - m_block_min
                total = r_q * nm
                psk = seqlen.offset_k + n_blk_loc * self.tile_n
                nrk = cutlass.min(seqlen.seqlen_k - n_blk_loc * self.tile_n, Int32(self.tile_n))
                # packets of iterations 0 and 1
                hq, m_blk, psq, nrq = self._sfd_qtile(seqlen, Int32(0), nm, m_block_min, n_block, m_blk_off, head_idx)
                hq1, m1, psq1, nrq1 = self._sfd_qtile(seqlen, Int32(1), nm, m_block_min, n_block, m_blk_off, head_idx)
                self._sfd_packet_issue(mLSE, mdPsum, mEvQ, mEvdO, mSF4Q, hq, batch_idx, psq, sStg, n_p % 4, stg_mbar)
                n_p += 1
                if total > 1:
                    self._sfd_packet_issue(mLSE, mdPsum, mEvQ, mEvdO, mSF4Q, hq1, batch_idx, psq1, sStg, n_p % 4,
                                           stg_mbar)
                    n_p += 1
                # the key tile's block exponents (K, V) and K's NVFP4 SF words: loaded once, after the first TMA issue
                ewk = Int32(0)
                ewv = Int32(0)
                for j in cutlass.range(total, unroll=1):
                    (lQ4, ldO4, lQ, ldO, lSFQ, lSFdO, lSFQt, lSFdOt, gLSE, gdPsum) = self._q_loaders(hq, batch_idx, *qa)
                    slot = c_p % 4
                    # ---- Q stage (+ K on the first): TMA first, then the generic writes, the fence and the arrive
                    pipeline_Q.sync_object_empty.wait(producer_state_Q.index, producer_state_Q.phase)
                    bar = pipeline_Q.producer_get_barrier(producer_state_Q)
                    if j == 0:
                        load_K(tma_bar_ptr=bar)
                        load_K4(tma_bar_ptr=bar)
                    lQ(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lQ4(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    lSFQt(src_idx=m_blk, dst_idx=producer_state_Q.index, tma_bar_ptr=bar)
                    if j == 0:
                        if const_expr(self.mxop_tb and not self.token_local):
                            ewk = self._sfd_word(mEvK, head_idx, psk, nrk)
                            ewv = self._sfd_word(mEvV, head_idx, psk, nrk)
                        if const_expr(not self.fp8):
                            self._sfd_sf4_load(mSF4K, head_idx, psk, nrk, sf4k)
                    cute.arch.mbarrier_wait(stg_mbar + slot, (c_p // 4) % 2)
                    ewq = Int32(0)
                    if const_expr(self.token_local and self.fp8):
                        # (cov3: token-local Q on the 8-bit route has a unit d-rows scale -- the aligned 0x7F atoms)
                        self._sfd_dim_atom(Int32(0x7F7F7F7F), sSFQc_u8, producer_state_Q.index)
                    elif const_expr(self.mxop_tb):
                        ewq = self._sfd_packet_word(sStg, slot, 0, psq, nrq)
                        self._sfd_dim_atom(ewq, sSFQc_u8, producer_state_Q.index)
                    if const_expr(not self.fp8):
                        self._sfd_packet_sf4(sStg, slot, psq, nrq, sSFQ, producer_state_Q.index, sfq_stage_bytes)
                    if j == 0:
                        if const_expr(self.token_local):   # (cov3: token-local K: unit d-rows scale)
                            self._sfd_dim_atom(Int32(0x7F7F7F7F), sSFKc_u8, Int32(0))
                        elif const_expr(self.mxop_tb):
                            self._sfd_dim_atom(ewk, sSFKc_u8, Int32(0))
                        if const_expr(not self.fp8):
                            self._sfd_sf4_store(sf4k, sSFK, Int32(0), sfq_stage_bytes)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if j == 0:
                        pipeline_Q.sync_object_full.arrive_and_expect_tx(producer_state_Q.index,
                                                                         tx_Q + self.tma_copy_bytes["K"])
                    else:
                        pipeline_Q.sync_object_full.arrive_and_expect_tx(producer_state_Q.index, tx_Q)
                    # ---- LSE stage (+ f_Q, f_K, f_V)
                    pipeline_LSE.sync_object_empty.wait(producer_state_Q.index, producer_state_Q.phase)
                    self._sfd_packet_stat(sStg, slot, 0, psq, nrq, sLSE[None, producer_state_Q.index], math.inf,
                                          *((ewq, ewk, ewv) if const_expr(self.mxop_tb and not self.token_local)
                                            else (None, None, None)))
                    cute.arch.sync_warp()
                    pipeline_LSE.sync_object_full.arrive(producer_state_Q.index, pipeline_LSE.producer_mask)
                    producer_state_Q.advance()
                    # ---- dO stage (+ V on the first)
                    pipeline_dO.sync_object_empty.wait(producer_state_dO.index, producer_state_dO.phase)
                    bar = pipeline_dO.producer_get_barrier(producer_state_dO)
                    if j == 0:
                        load_V4(tma_bar_ptr=bar)
                    ldO(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    ldO4(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    lSFdO(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    lSFdOt(src_idx=m_blk, dst_idx=producer_state_dO.index, tma_bar_ptr=bar)
                    ewdo = Int32(0)
                    if const_expr(self.token_local):   # (cov3: token-local dO: unit d-rows scale)
                        self._sfd_dim_atom(Int32(0x7F7F7F7F), sSFdOc_u8, producer_state_dO.index)
                    elif const_expr(self.mxop_tb):
                        ewdo = self._sfd_packet_word(sStg, slot, 1, psq, nrq)
                        self._sfd_dim_atom(ewdo, sSFdOc_u8, producer_state_dO.index)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if j == 0:
                        pipeline_dO.sync_object_full.arrive_and_expect_tx(producer_state_dO.index,
                                                                          tx_dO + self.tma_copy_bytes["V"])
                    else:
                        pipeline_dO.sync_object_full.arrive_and_expect_tx(producer_state_dO.index, tx_dO)
                    # ---- D'' stage (+ f_dO)
                    pipeline_dPsum.sync_object_empty.wait(producer_state_dO.index, producer_state_dO.phase)
                    self._sfd_packet_stat(sStg, slot, 1, psq, nrq, sdPsum[None, producer_state_dO.index],
                                          (2.0 ** 120) if self.ds_fold else -(2.0 ** 100),
                                          ewdo if const_expr(self.mxop_tb and not self.token_local) else None, None, None)
                    cute.arch.sync_warp()
                    pipeline_dPsum.sync_object_full.arrive(producer_state_dO.index, pipeline_dPsum.producer_mask)
                    producer_state_dO.advance()
                    c_p += 1
                    # ---- iteration j + 2's packet (its slot was consumed two iterations ago)
                    hq2, m2, psq2, nrq2 = self._sfd_qtile(seqlen, j + 2, nm, m_block_min, n_block, m_blk_off, head_idx)
                    if j + 2 < total:
                        self._sfd_packet_issue(mLSE, mdPsum, mEvQ, mEvdO, mSF4Q, hq2, batch_idx, psq2, sStg, n_p % 4,
                                               stg_mbar)
                        n_p += 1
                    hq = hq1
                    m_blk = m1
                    psq = psq1
                    nrq = nrq1
                    hq1 = hq2
                    m1 = m2
                    psq1 = psq2
                    nrq1 = nrq2

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # the tails once, after the last work tile (per tile they would advance the states: a persistent CTA's next
        # tile would then run a ring of stages out of phase)
        pipeline_Q.producer_tail(producer_state_Q.clone())
        pipeline_LSE.producer_tail(producer_state_Q.clone())
        pipeline_dO.producer_tail(producer_state_dO.clone())
        pipeline_dPsum.producer_tail(producer_state_dO.clone())

    @cute.jit
    def _sfd_fac(self, e):
        """A dense exponent byte -> its token-block factor: 2^(e - 127) (exact, e >= 1), 1 for e = 0 (an all-zero
        tile; the quantiser's rule)."""
        from flash_attn.cute.quant.fp8 import bits_f32
        ei = Int32(e)
        return cutlass.select_(ei == 0, Float32(1.0), bits_f32(ei << 23))

    @cute.jit
    def _sfd_qm(self, it, nm, m_min, n_block):
        """sf_dense: iteration -> (q head within the kv head, m block), the consumers' order (MHA: ascending m)."""
        if const_expr(self.qhead_per_kvhead > 1):
            return self._gqa_qm(it, nm, m_min, n_block)
        return Int32(0), m_min + it

    @cute.jit
    def _mx_bulk(self, mSF, tile, head, batch, dst_u8, dst_stage, bar):
        """One 512 B SF atom (tile `tile` of (b, h)) gmem -> smem stage dst_stage, on the stage's mbarrier (tx counted in
        tma_copy_bytes)."""
        with cute.arch.elect_one():
            copy_utils.cpasync_bulk_g2s(mSF[None, tile, head, batch].iterator, dst_u8.iterator + dst_stage * 512, bar, 512)

    @cute.jit
    def _mx_load_q(self, m_tok, m_dim, s_tok_u8, s_dim_u8, hq, batch_idx, m_blk, stage, bar):
        """mxop, query side (Q or dO) of stage `stage`: the token-rows atom (S / dP operand; staged for Q, single for dO)
        and the d-rows atom (MN-major B operand of dK / dV)."""
        if const_expr(self.mxop):
            if const_expr(m_tok is not None):
                self._mx_bulk(m_tok, m_blk * (self.tile_m // 128), hq, batch_idx, s_tok_u8, stage, bar)
            if const_expr(m_dim is not None):
                self._mx_bulk(m_dim, m_blk * (self.tile_m // 128), hq, batch_idx, s_dim_u8, stage, bar)

    @cute.jit
    def _mx_load_kv(self, m_tok, m_dim, s_tok_u8, s_dim_u8, hk, batch_idx, n_block, bar):
        """mxop, key side (K or V), once per work tile."""
        if const_expr(self.mxop):
            if const_expr(m_tok is not None):
                self._mx_bulk(m_tok, n_block, hk, batch_idx, s_tok_u8, 0, bar)
            if const_expr(m_dim is not None):
                self._mx_bulk(m_dim, n_block, hk, batch_idx, s_dim_u8, 0, bar)

    @cute.jit
    def _pk_shifts(self, seqlen, m_blk_off, n_block, n_blk_loc):
        """codes_packed: (Q-side, K-side) row shifts of a work tile: packed row = aligned row + shift (seqlen.offset_q
        / offset_k: the documents' packed starts, LPTileInfo). The statistics rows (LSE / D) are the aligned rows."""
        return (seqlen.offset_q - m_blk_off * self.tile_m, seqlen.offset_k - (n_block - n_blk_loc) * self.tile_n)

    def _pk_rows(self, t, pk, which: int, mode: int):
        """codes_packed: t (a rank-2 (rows, d) / (d, rows) or rank-1 (rows,) view) with its row mode `mode` shifted by
        pk[which] (the packed / stats row of the aligned tile coordinates); t itself otherwise."""
        if pk is None:
            return t
        off = [0] * cute.rank(t)
        off[mode] = pk[which]
        return cute.domain_offset(tuple(off), t)

    def _q_loaders(self, hq, batch_idx, pk, one, mQ, mQ4, mSFQ, mdO, mdO4, mSFdO, mLSE, mdPsum, sQ, sQ4, sSFQ, sdO, sdO4, sSFdO,
                   tma_atom_Q, tma_atom_Q4, tma_atom_SFQ, tma_atom_dO, tma_atom_dO4, tma_atom_SFdO,
                   mSFQt, mSFdOt, sSFQt, sSFdOt, tma_atom_SFQt, tma_atom_SFdOt,
                   thr_mma_S, thr_mma_dP, thr_mma_dK, thr_mma_dV, thr_mma_sfb, thr_mma_sfb_q):
        """GQA load warp: the q-side TMA loaders of q head hq (trace-time Python, no dynamic control flow). Same
        objects as the MHA path builds once per tile."""
        mQ4_cur = self._pk_rows(mQ4[None, None, hq, batch_idx], pk, 0, 0)
        mdO4_cur = self._pk_rows(mdO4[None, None, hq, batch_idx], pk, 0, 0)
        gQ4 = cute.local_tile(mQ4_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
        load_Q4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q4, 0, one, thr_mma_S.partition_B(gQ4), sQ4)
        gdO4 = cute.local_tile(mdO4_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0))
        load_dO4, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO4, 0, one, thr_mma_dP.partition_B(gdO4), sdO4)
        load_Q = load_dO = load_SFQ = load_SFdO = load_SFQt = load_SFdOt = self._load_noop
        if self.fp8 is False:
            mQ_cur = self._pk_rows(mQ[None, None, hq, batch_idx], pk, 0, 1)
            mSFQ_cur = mSFQ[None, None, hq, batch_idx]
            mdO_cur = self._pk_rows(mdO[None, None, hq, batch_idx], pk, 0, 1)
            gSFQ = cute.local_tile(mSFQ_cur, cute.select(self.mma_tiler_sfb, mode=[1, 2]), (None, 0))
            tQsSFQ, tQgSFQ = cpasync.tma_partition(
                tma_atom_SFQ, 0, one, cute.group_modes(sSFQ, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(gSFQ), 0, 3)
            )
            load_SFQ = partial(self._load_sf_stage, tma_atom_SFQ, cute.filter_zeros(tQgSFQ), cute.filter_zeros(tQsSFQ), self.tile_m // 128)
            if not self.skip_dk:
                gQt = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None))
                load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, one, thr_mma_dK.partition_B(gQt), sQ)
        gLSE = cute.local_tile(mLSE[None, hq, batch_idx], (self.lse_tile,), (None,))
        gdPsum = cute.local_tile(mdPsum[None, hq, batch_idx], (self.tile_m,), (None,))
        return load_Q4, load_dO4, load_Q, load_dO, load_SFQ, load_SFdO, load_SFQt, load_SFdOt, gLSE, gdPsum

    @cute.jit
    def _load_noop(self, **kwargs):
        pass


    @cute.jit
    def _load_sf_stage(self, tma_atom, g, s, mul: cutlass.Constexpr[int], src_idx, dst_idx, tma_bar_ptr):
        """Staged scale-factor TMA load: gmem tile src_idx * mul -> smem stage dst_idx."""
        cute.copy(tma_atom, g[None, src_idx * mul], s[None, dst_idx], tma_bar_ptr=tma_bar_ptr)

    # ------------------------------------------------------------------ S2T helper
    @cute.jit
    def s2t(self, tiled_copy, src, dst):
        """One scale-factor S2T copy."""
        cute.copy(tiled_copy, src, dst)

    def s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor, dtype=None):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), dtype if dtype is not None else self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # ------------------------------------------------------------------ MMA warp
    @cute.jit
    def mma(
        self,
        tiled_mma_S, tiled_mma_dP, tiled_mma_dV, tiled_mma_dK, tiled_mma_dQ,
        sQ, sQ4, sK, sK4, sV4, sdO, sdO4, sSFQ, sSFK, sSFV, sSFdO,
        tP, sdS, tdS, tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ,
        tCtSFK, tCtSFV, tCtSFQs, tCtSFdOs,
        pipeline_Q, pipeline_dO, pipeline_S_P, pipeline_dS, pipeline_dKV, pipeline_dP, pipeline_dQ,
        S_loaded_mbar_ptr,
        block_info, SeqlenInfoCls, TileSchedulerCls,
        sSFQt=None, sSFdOt=None, tCtSFP=None, tCtSFdOts=None, tCtSFdS=None, tCtSFQts=None,
        tiled_mma_dQh=None, tdQtdQ_h=None, sK_h=None,
        sSFdS_mx=None, sSFQc=None, sSFdSq=None, sSFKc=None, tCtSFdS_mx=None, tCtSFQc=None, tCtSFdSq=None, tCtSFKc=None,
        sDsScr=None,
        sSFP_mx=None, sSFdOc=None, tCtSFP_mx=None, tCtSFdOc=None,
        sSFK_mx=None, sSFQ_mx=None, sSFV_mx=None, sSFdO_mx=None,
        tCtSFK_mx=None, tCtSFQ_mx=None, tCtSFV_mx=None, tCtSFdO_mx=None,
    ):
        # fragments
        tSrK4 = tiled_mma_S.make_fragment_A(sK4)
        tSrQ4 = tiled_mma_S.make_fragment_B(sQ4)
        tdPrV4 = tiled_mma_dP.make_fragment_A(sV4)
        tdPrdO4 = tiled_mma_dP.make_fragment_B(sdO4)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQ)
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sK)

        def _noop(*args, **kwargs):
            pass

        # S2T copies of the scale factors (FP4 mode only; FP8 mode has no scale factors)
        s2t_K = s2t_V = _noop
        s2t_Q = [_noop] * self.Q_stage
        s2t_dO = [_noop, _noop]
        s2t_Qt = [_noop, _noop]
        s2t_dOt = [_noop, _noop]
        assert self.Q_stage in (2, 3) and self.dO_stage in (1, 2)
        if const_expr(not self.fp8):
            s2t_sfk, tCsSFK_s2t, tCtSFK_s2t = self.s2t_copy_and_partition(sSFK, tCtSFK)
            s2t_sfq_staged = [self.s2t_copy_and_partition(sSFQ, tCtSFQs[s]) for s in range(self.Q_stage)]
            s2t_sfq, tCsSFQ_s2t, _ = s2t_sfq_staged[0]
            s2t_K = partial(self.s2t, s2t_sfk, tCsSFK_s2t[None, None, None, None, 0], tCtSFK_s2t)
            s2t_Q = [partial(self.s2t, s2t_sfq, tCsSFQ_s2t[None, None, None, None, s], s2t_sfq_staged[s][2]) for s in range(self.Q_stage)]
            # C3: S2T of the transposed operands' scale factors; block-scaled dV / dK ops
        # block-scaled d\hat{S}: S2T copies of the four SF tiles (d\hat{S} / const Q before dK; d\hat{S}^T / const K before dQ)
        s2t_dS_mx = s2t_Qc = s2t_dSq = s2t_Kc = _noop
        if const_expr(self.mxds):
            _pairs = [(sSFdS_mx, tCtSFdS_mx), (sSFQc, tCtSFQc)]
            if const_expr(self.mxds_dq_sf):
                _pairs += [(sSFdSq, tCtSFdSq), (sSFKc, tCtSFKc)]
            _mxp = [self.s2t_copy_and_partition(s, t, dtype=self.sf_dtype_mx) for s, t in _pairs]
            _s2ts = [partial(self.s2t, tc, src[None, None, None, None, 0], dst) for (tc, src, dst) in _mxp]
            s2t_dS_mx, s2t_Qc = _s2ts[0], _s2ts[1]
            # mxop: Q's d-rows SF atom is staged with the Q tile; the S2T reads the dK(t) stage
            s2t_Qc_st = [s2t_Qc] * self.Q_stage
            qdyn_qc = s2t_Qc
            if const_expr(self.mxop):
                _tcq, _srcq, _dstq = _mxp[1]
                qdyn_qc = (_tcq, _srcq, _dstq)
                s2t_Qc_st = [partial(self.s2t, _tcq, _srcq[None, None, None, None, st], _dstq) for st in range(self.Q_stage)]
            if const_expr(self.mxds_dq_sf):
                s2t_dSq, s2t_Kc = _s2ts[2], _s2ts[3]
            # constant 2^0 SF tiles for the per-tensor E4M3 B operands (Q of dK, K of dQ): written once by this warp
            lane = cute.arch.lane_idx()
            for buf in ([] if const_expr(self.mxop) else ([sSFQc, sSFKc] if const_expr(self.mxds_dq_sf) else [sSFQc])):
                nbytes = cute.cosize(buf.layout)
                w32 = cute.make_tensor(cute.recast_ptr(buf.iterator, dtype=Int32), cute.make_layout(nbytes // 4))
                for i in cutlass.range_constexpr(nbytes // 128):
                    w32[lane + 32 * i] = Int32(0x7F7F7F7F)
            cute.arch.fence_view_async_shared()
            mma_dK_mx_op = tiled_mma_dK.op
        mma_S_op = tiled_mma_S.op
        mma_dP_op = tiled_mma_dP.op
        # stage-static producer GEMMs: S^T(stage) and dP^T(stage); FP8 mode passes no scale tensors
        sf_Q = [None] * self.Q_stage if const_expr(self.fp8) else tCtSFQs
        sf_dO = ([None, None])
        sf_K, sf_V = tCtSFK, tCtSFV
        if const_expr(self.mx_dp):
            # block-scaled E4M3 operands: SFV / SFdO(t) S2T'd into the S bank [32, 48) right before dP(t) (the callers'
            # s2t_V / s2t_dO slots, after "S(t) loaded"); FP8 S: SFK / SFQ(t) into the dP bank [dP+64, dP+88) before S(t)
            _pv = [self.s2t_copy_and_partition(sSFV_mx, tCtSFV_mx, dtype=self.sf_dtype_mx),
                   self.s2t_copy_and_partition(sSFdO_mx, tCtSFdO_mx, dtype=self.sf_dtype_mx)]
            s2t_V = partial(self.s2t, _pv[0][0], _pv[0][1][None, None, None, None, 0], _pv[0][2])
            s2t_dO = [partial(self.s2t, _pv[1][0], _pv[1][1][None, None, None, None, 0], _pv[1][2]) for _ in range(2)]
            sf_V = tCtSFV_mx
            sf_dO = [tCtSFdO_mx, tCtSFdO_mx]
            if const_expr(self.mxop_s):
                _pk = self.s2t_copy_and_partition(sSFK_mx, tCtSFK_mx, dtype=self.sf_dtype_mx)
                s2t_K = partial(self.s2t, _pk[0], _pk[1][None, None, None, None, 0], _pk[2])
                _pq = [self.s2t_copy_and_partition(sSFQ_mx, tCtSFQ_mx[st], dtype=self.sf_dtype_mx) for st in range(self.Q_stage)]
                s2t_Q = [partial(self.s2t, _pq[st][0], _pq[st][1][None, None, None, None, st], _pq[st][2]) for st in range(self.Q_stage)]
                sf_K = tCtSFK_mx
                sf_Q = tCtSFQ_mx
        mma_S = [partial(self.mma_S_stage, mma_S_op, tSrK4, tSrQ4, sK4, sQ4, sf_K, sf_Q[s], s) for s in range(self.Q_stage)]
        mma_dP = [partial(self.mma_S_stage, mma_dP_op, tdPrV4, tdPrdO4, sV4, sdO4, sf_V, sf_dO[s], s, acc_offset=self.tmem_dP_offset)
                  for s in (0, self.dO_stage - 1)]

        if const_expr(self.skip_dv):
            mma_pdo_fn = _noop
        else:
            mma_pdo_fn = partial(
                gemm_ptx_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO, sA=None, sB=sdO,
                tA_addr=self.tmem_P_offset, cta_group=1,
            )
        if const_expr(self.skip_dk):
            mma_dsq_fn = _noop
        else:
            mma_dsq_fn = partial(
                gemm_ptx_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ, sA=None, sB=sQ,
                tA_addr=self.tmem_dS_offset, cta_group=1,
            )
        if const_expr(self.mxdv and not self.skip_dv):
            # block-scaled \hat{P}: S2T of the \hat{P} SFs / constant dO SFs into their S-bank parks, then the mxf8f6f4 dV GEMM
            _pp = [self.s2t_copy_and_partition(s, t, dtype=self.sf_dtype_mx) for s, t in ((sSFP_mx, tCtSFP_mx), (sSFdOc, tCtSFdOc))]
            s2t_P_mx, s2t_dOc = [partial(self.s2t, tc, src[None, None, None, None, 0], dst) for (tc, src, dst) in _pp]
            _lane_p = cute.arch.lane_idx()
            # constant 2^0 tiles: dO's SFs unless mxop loads them; P^'s SFs unless mxp computes them
            for _cbuf in ([] if const_expr(self.mxop) else [sSFdOc]) + ([] if const_expr(self.mxp) else [sSFP_mx]):
                _nb = cute.cosize(_cbuf.layout)
                _w32 = cute.make_tensor(cute.recast_ptr(_cbuf.iterator, dtype=Int32), cute.make_layout(_nb // 4))
                for i in cutlass.range_constexpr(_nb // 128):
                    _w32[_lane_p + 32 * i] = Int32(0x7F7F7F7F)
            cute.arch.fence_view_async_shared()
            assert self.dO_stage == 1
            mma_pdo_fn = partial(self._mma_dV_mx, tiled_mma_dV.op, tdVrP, tdVrdO, sdO, tCtSFP_mx, tCtSFdOc, s2t_P_mx, s2t_dOc)
        if const_expr(self.skip_dq):
            mma_dq_fn = _noop
            mma_dq_h = [_noop, _noop]
        elif const_expr(self.mxds_dq_sf):
            mma_dq_fn = partial(
                sm100_utils.gemm_ptx_partial_fp4, tiled_mma_dQ.op, self.tmem_dQ_offset, tdQrdS, tdQrK,
                sA=sdS, sB=sK, tScaleA=tCtSFdSq, tScaleB=tCtSFKc, zero_init=True,
            )
            mma_dq_h = [_noop, _noop]
        else:
            mma_dq_fn = partial(gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True)
            mma_dq_h = [_noop, _noop]

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_phase_acc = Int32(1)  # S_P empty (P stored), producer style
        producer_phase_dKV = Int32(1)
        phase_dPe = Int32(0)  # dP empty: compute warps loaded dP(t) (consumer-style: first wait blocks)
        phase_dQe = Int32(0)  # dQ empty: reduce warps drained dQ(t)
        phase_Sl = Int32(0)  # S loaded (compute warps)
        cta_group = pipeline_S_P.cta_group
        sfb_half_mul = 0

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
            if process_tile:
                accumulate_dK = False
                sfb_off = Int32(0)
                # ---- prologue: S(0) with SFK/SFQ(0) parked in the (free) dP bank; dP(0) with SFV/SFdO(0)
                # parked in the S bank once the compute warps have loaded S(0); dV(0) once P(0) is stored.
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                sm100_utils.tcgen05_after_thread_sync()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                s2t_K()
                if const_expr(self.q_dyn):
                    self._mma_S_dyn(mma_S_op, tSrK4, tSrQ4, sK4, sQ4, handle_Q.index)
                elif handle_Q.index == 0:
                    s2t_Q[0]()
                    mma_S[0](sfb_off)
                else:
                    s2t_Q[1]()
                    mma_S[1](sfb_off)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                pipeline_dO.consumer_wait(consumer_state_dO)
                if const_expr(not self.pp):  # pp: no SF parks in the S bank, dP(0) needs no "S(0) loaded"
                    cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                    phase_Sl ^= 1
                sm100_utils.tcgen05_after_thread_sync()
                s2t_V()
                if consumer_state_dO.index == 0:
                    s2t_dO[0]()
                    mma_dP[0](sfb_off)
                else:
                    s2t_dO[1]()
                    mma_dP[1](sfb_off)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                if const_expr(self.bwd_persist):
                    # persistent CTA: the previous work tile's dK / dV must have left TMEM (its epilogue's release)
                    # before this tile's first dV / dK GEMM overwrites them (one tile per CTA never needs this)
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # P(0) stored
                sm100_utils.tcgen05_after_thread_sync()
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                # ---- main loop, iteration t (t = m_block - 1):
                #  (1) [dP(t) loaded, dQ(t-1) drained] park SFK/SFQ(t+1) in the dP bank -> S(t+1)
                #  (2) [dS(t)] dK(t)   (3) dQ(t) [overwrites the dP bank incl. dS(t) / SFK / SFQ]
                #  (4) [dO(t+1), S(t+1) loaded, dQ(t) drained] park SFV/SFdO(t+1) in the S bank -> dP(t+1)
                #  (5) [P(t+1) stored] dV(t+1)
                main_loop_iters = (m_block_max - m_block_min) * self.qhead_per_kvhead - 1  # GQA: r q heads x m blocks
                handle_Q_next = handle_Q
                dq_pending = Int32(0)  # pp: a dQ issued and not yet waited drained
                for it in cutlass.range(main_loop_iters, unroll=1):
                    # (1) S(t+1)
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    # dP(t) loaded -> the dP bank's free columns may take SFK / SFQ(t+1); dQ(t-1) was
                    # drained before dP(t) was issued (step 4 of the previous iteration). pp: S(t + 1) needs only the S
                    # bank, free since dV(t) waited for every P^(t) word; the dP-bank wait moves to dP(t + 1)
                    if const_expr(not self.pp):
                        pipeline_dP.sync_object_empty.wait(0, phase_dPe)
                        phase_dPe ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    s2t_K()
                    if const_expr(self.q_dyn):
                        self._mma_S_dyn(mma_S_op, tSrK4, tSrQ4, sK4, sQ4, handle_Q_next.index)
                    elif handle_Q_next.index == 0:
                        s2t_Q[0]()
                        mma_S[0](sfb_off)
                    else:
                        s2t_Q[1]()
                        mma_S[1](sfb_off)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    # (2) dK += dS(t)^T Q(t)
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    sm100_utils.tcgen05_after_thread_sync()
                    if const_expr(self.mxds):
                        # dS(t) ready => dP(t) loaded: the dP bank's free columns take SFd\hat{S} / const SFQ for dK(t)
                        if const_expr(self.sf_asm):
                            self._sf_assemble(sDsScr, sSFdS_mx, sSFdSq)
                        s2t_dS_mx()
                        if const_expr(self.q_dyn):
                            self._s2t_mma_dk_dyn(qdyn_qc, mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, not accumulate_dK,
                                                 handle_Q.index)
                        elif handle_Q.index == 0:
                            s2t_Qc_st[0]()
                            self.mma_ts_stage(mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, 0, not accumulate_dK, self.tmem_dK_offset, self.tmem_dS_offset)
                        else:
                            s2t_Qc_st[1]()
                            self.mma_ts_stage(mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, 1, not accumulate_dK, self.tmem_dK_offset, self.tmem_dS_offset)
                    else:
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    accumulate_dK = True
                    handle_Q.release()
                    # (3) dQ(t) = dS(t) K
                    if const_expr(self.mxds_dq_sf):
                        # the dQ-side SFs park in the S bank: wait until the compute warps have loaded S(t+1)
                        # (the wait step (4) would otherwise do), then S2T SFd\hat{S}^T / const SFK for dQ(t)
                        # (nodqwait: timing-only ablation of that wait, races S(t+1)'s load)
                        if const_expr(not self.pp):  # pp: parks outside S
                            cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                            phase_Sl ^= 1
                            sm100_utils.tcgen05_after_thread_sync()
                    if const_expr(self.pp):
                        # pp: dQ has its own bank; dQ(t - 1) drained by the reduce warps before dQ(t) (and its parks)
                        if dq_pending != 0:
                            pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                            phase_dQe ^= 1
                            sm100_utils.tcgen05_after_thread_sync()
                        dq_pending = Int32(1)
                    if const_expr(self.mxds_dq_sf):
                        s2t_dSq()
                        s2t_Kc()
                    mma_dq_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()
                    # (4) dP(t+1)
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    if const_expr(not self.mxds_dq_sf and not self.pp):
                        cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                        phase_Sl ^= 1
                    if const_expr(self.pp):
                        pipeline_dP.sync_object_empty.wait(0, phase_dPe)  # dP(t) loaded (implied by dS(t) ready)
                        phase_dPe ^= 1
                    else:
                        pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                        phase_dQe ^= 1
                    sm100_utils.tcgen05_after_thread_sync()
                    s2t_V()
                    if consumer_state_dO.index == 0:
                        s2t_dO[0]()
                        mma_dP[0](sfb_off)
                    else:
                        s2t_dO[1]()
                        mma_dP[1](sfb_off)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                    # (5) dV += P(t+1)^T dO(t+1)
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    sm100_utils.tcgen05_after_thread_sync()
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()
                    handle_Q = handle_Q_next

                if const_expr(not self.bwd_persist):
                    # (one tile per CTA: a final S-full phase nobody waits for. A persistent CTA must not leave it: the
                    # compute warps' first S wait of the next tile would pass on it)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                if const_expr(not self.bwd_persist):
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                if const_expr(not self.bwd_persist):
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                # ---- tail: dK, dQ of the last tile (dQ(last-1) was drained before dP(last))
                pipeline_dP.sync_object_empty.wait(0, phase_dPe)  # dP(last) loaded
                phase_dPe ^= 1
                pipeline_dS.consumer_wait(consumer_state_dS)
                sm100_utils.tcgen05_after_thread_sync()
                if const_expr(self.mxds):
                    if const_expr(self.sf_asm):
                        self._sf_assemble(sDsScr, sSFdS_mx, sSFdSq)
                    s2t_dS_mx()
                    if const_expr(self.q_dyn):
                        self._s2t_mma_dk_dyn(qdyn_qc, mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, not accumulate_dK,
                                             handle_Q.index)
                    elif handle_Q.index == 0:
                        s2t_Qc_st[0]()
                        self.mma_ts_stage(mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, 0, not accumulate_dK, self.tmem_dK_offset, self.tmem_dS_offset)
                    else:
                        s2t_Qc_st[1]()
                        self.mma_ts_stage(mma_dK_mx_op, tdKrdS, tdKrQ, sQ, tCtSFdS_mx, tCtSFQc, 1, not accumulate_dK, self.tmem_dK_offset, self.tmem_dS_offset)
                else:
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                producer_phase_dKV ^= 1
                if const_expr(self.pp):
                    if dq_pending != 0:
                        pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                        phase_dQe ^= 1
                        sm100_utils.tcgen05_after_thread_sync()
                if const_expr(self.mxds_dq_sf):
                    # no S(t+1) in flight: the S bank is free once dS(last) is ready
                    s2t_dSq()
                    s2t_Kc()
                mma_dq_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()
                if const_expr(not self.bwd_persist):
                    producer_phase_acc ^= 1
                # dQ(last) drained: the next tile may start
                pipeline_dQ.sync_object_empty.wait(0, phase_dQe)
                phase_dQe ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _sf_assemble(self, sScr, sSFdS_mx, sSFdSq):
        """sf_asm (MMA warp, after the dS wait, before the S2T): expand the 16 blk32 exponents e[query block qb][key
        block kb] (scratch byte qb * 4 + kb) into the two 512 B SF tiles, 32 rows x 16 bytes, every row the same:
        dQ side row = the scratch as is (byte (q // 32) * 4 + kb); dK side row word kb = bytes e[0..3][kb] (byte
        (k // 32) * 4 + qb). Each lane writes one row (two 8-byte stores per tile), then fences for the async proxy."""
        lane = cute.arch.lane_idx()
        s32 = cute.make_tensor(cute.recast_ptr(sScr.iterator, dtype=Int32), cute.make_layout(4))
        w = [s32[i] for i in range(4)]
        dk = []
        for kb in cutlass.range_constexpr(4):
            v = Int32(0)
            for qb in cutlass.range_constexpr(4):
                v = v | (((w[qb] >> (8 * kb)) & Int32(0xFF)) << (8 * qb))
            dk.append(v)
        t_dk = cute.make_tensor(cute.recast_ptr(sSFdS_mx.iterator, dtype=Int64), cute.make_layout(64))
        t_dk[lane * 2] = (dk[0].to(Int64) & Int64(0xFFFFFFFF)) | (dk[1].to(Int64) << 32)
        t_dk[lane * 2 + 1] = (dk[2].to(Int64) & Int64(0xFFFFFFFF)) | (dk[3].to(Int64) << 32)
        if const_expr(self.sf_asm_q):
            s64 = cute.make_tensor(cute.recast_ptr(sScr.iterator, dtype=Int64), cute.make_layout(2))
            t_dq = cute.make_tensor(cute.recast_ptr(sSFdSq.iterator, dtype=Int64), cute.make_layout(64))
            t_dq[lane * 2] = s64[0]
            t_dq[lane * 2 + 1] = s64[1]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _ds_dq_rowlocal_words_x(self, vals, words, lane, wps: cutlass.Constexpr[int], s_mul, cm=None):
        """ds_dq_rowlocal: the dQ side of one 32-key x 32-query dS block with one UE8M0 exponent per query (the mx32 rung's
        dQ side). A lane's row holds one key x the block's 32 queries, so a query's abs-max is a column reduction: a warp
        reduce-scatter (mx_scale_utils.warp_col_max_scatter: 31 SHFL + 62 SEL + 31 FMNMX3; `cm` when the caller already
        has it) leaves query `lane`'s abs-max in lane `lane`, its multiplier goes through this warp's 32-float smem slot
        `s_mul` to every lane (one STS, 8 broadcast LDS.128), the codes go to `words` (the SMEM dS^ tile's E4M3 words),
        and the exponent of query `lane` is returned (that lane's row of the dQ SF tile). Measured against 32 redux.sync
        (the mx32 rung's form, bitwise the same codes): +37 % vs +22 % of the kernel at b4 s4096 h16 d128 causal."""
        if const_expr(cm is None):
            cm = mxs.warp_col_max_scatter([vals[i] for i in range(cute.size(vals))], lane)
        e_q = mxs.ue8m0_exp_448(cm)
        s_mul[lane] = mxs.ue8m0_mul(e_q)
        cute.arch.sync_warp()
        mulc = cute.make_rmem_tensor(cute.size(vals), Float32)
        cute.autovec_copy(s_mul, mulc)
        sq = cute.make_rmem_tensor(cute.size(vals), Float32)
        for v in cutlass.range_constexpr(cute.size(vals) // 2):
            sq[2 * v], sq[2 * v + 1] = cute.arch.mul_packed_f32x2((vals[2 * v], vals[2 * v + 1]), (mulc[2 * v], mulc[2 * v + 1]))
        for j in cutlass.range_constexpr(wps):
            words[j] = sm100_utils.packed_float_to_ue4m3(sq[4 * j], sq[4 * j + 1], sq[4 * j + 2], sq[4 * j + 3])
        return e_q

    @cute.jit
    def _tl_query_factors(self, mTokFac, row: cutlass.Constexpr[int], batch_idx, head_idx, m_block, tScS_stage,
                          slot, lane, load: cutlass.Constexpr[bool], pre=None):
        """token-local operands: the per-query factors f(q) of this stage's 32 queries as a 32-float fragment
        (element i is query q0 + i, q0 = the stage's first query: the x32 TMEM load gives each lane 32 consecutive
        columns). load=True: lane j fetches query q0 + j (one coalesced 128-byte load per warp) into the warp's 32-float
        SMEM slot; then every lane reads the slot (8 broadcast LDS.128). load=False: the slot written earlier in this
        iteration is re-read."""
        if const_expr(load):
            v = pre
            if const_expr(pre is None):
                v = mTokFac[row, batch_idx, head_idx, m_block * self.tile_m + tScS_stage[0][1] + lane]
            cute.arch.sync_warp()
            slot[lane] = v
            cute.arch.sync_warp()
        f = cute.make_rmem_tensor(32, Float32)
        cute.autovec_copy(slot, f)
        return f

    def _gqa_qm(self, it, nm, m_min, rot=None):
        """GQA work-tile iteration -> (q head within the kv head, m block): q-head-major ascending m, or (gqa_mmajor)
        m-major with m descending, or (det_gqa_rot, rot = n_block) q-head-major with m rotated by the key tile."""
        r = self.qhead_per_kvhead
        if const_expr(self.det_gqa_rot):
            qh = it // nm
            return qh, m_min + (it - qh * nm + rot) % nm
        if const_expr(self.gqa_mmajor):
            mo = it // r
            return it - mo * r, m_min + nm - 1 - mo
        qh = it // nm
        return qh, m_min + (it - qh * nm)



    @cute.jit
    def mma_S_stage(
        self, op, tCrA, tCrB_staged, sA, sB_staged, tScaleA, tScaleB,
        stage: cutlass.Constexpr[int], sfb_off: Int32, acc_offset: cutlass.Constexpr[int] = 0,
    ):
        """One producer GEMM (S^T or dP^T) from a static operand stage: block-scaled (FP4) or plain E4M3 (FP8)."""
        if const_expr(tScaleB is None):  # E4M3 operands (FP8 control; dP in the dp8 accuracy mode)
            sm100_utils.gemm_ptx_partial(
                op, acc_offset, tCrA, tCrB_staged[None, None, None, stage],
                sA=sA, sB=sB_staged[None, None, None, stage], zero_init=True,
            )
        else:
            sm100_utils.gemm_ptx_partial_fp4(
                op, acc_offset, tCrA, tCrB_staged[None, None, None, stage],
                sA=sA, sB=sB_staged[None, None, None, stage], tScaleA=tScaleA, tScaleB=tScaleB,
                zero_init=True, sfb_col_offset=sfb_off,
            )

    @cute.jit
    def _mma_dV_mx(self, op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB, s2t_a, s2t_b, B_idx, zero_init):
        r"""block-scaled dV += \hat{P}^T dO: S2T of the \hat{P} SFs / constant dO SFs (S bank parks; \hat{P}(t) stored => S(t) loaded,
        and the dQ-side parks were consumed earlier in the tensor pipe's order), then the mxf8f6f4 GEMM (dO stage 0)."""
        sm100_utils.tcgen05_after_thread_sync()
        s2t_a()
        s2t_b()
        self.mma_ts_stage(op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB, 0, zero_init, self.tmem_dV_offset, self.tmem_P_offset)

    @cute.jit
    def _stat_shfl(self, src: cute.Tensor, dst: cute.Tensor):
        """U4 prototype (Bwd1Spec.stat_shfl): the 32 per-query statistics of this thread's stage are the same 32 smem
        words for every lane of the warp (the lanes are keys). Lane l loads word l only (one LDS.32) and the row is
        rebuilt by 32 SHFL.IDX -- the register-resident form, without the broadcast LDS.128s."""
        n = cute.size(src)
        assert n == 32, "stat_shfl: one 32-query stage per thread (tmem_ld_rep 32)"
        lane = cute.arch.lane_idx()
        mine = src[lane]
        for i in cutlass.range_constexpr(n):
            dst[i] = cute.arch.shuffle_sync(mine, i)

    @cute.jit
    def _mma_S_dyn(self, op, tCrA, tCrB_staged, sA, sB_staged, idx):
        """q_dyn: S^T on the dynamically indexed Q stage (E4M3 operands, no scale factors: the q_dyn scope)."""
        sm100_utils.gemm_ptx_partial(op, 0, tCrA, tCrB_staged[None, None, None, idx], sA=sA,
                                     sB=sB_staged[None, None, None, idx], zero_init=True)

    @cute.jit
    def _s2t_mma_dk_dyn(self, qc, op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB, zero_init, idx):
        """q_dyn: the block-scaled dK(t) GEMM on the dynamically indexed Q stage, after the S2T of that stage's Q SF
        atom (mxop) or of the constant SF tile."""
        if const_expr(isinstance(qc, tuple)):
            cute.copy(qc[0], qc[1][None, None, None, None, idx], qc[2])
        else:
            qc()
        sm100_utils.gemm_ptx_partial_fp4(
            op, self.tmem_dK_offset, tCrA, tCrB_staged[None, None, None, idx],
            sA=None, sB=sB_staged[None, None, None, idx], tScaleA=tScaleA, tScaleB=tScaleB,
            zero_init=zero_init, tA_addr=self.tmem_dS_offset,
        )

    @cute.jit
    def mma_ts_stage(
        self, op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB,
        stage: cutlass.Constexpr[int], zero_init, acc_offset: cutlass.Constexpr[int], ta_addr: cutlass.Constexpr[int],
    ):
        r"""One block-scaled consumer GEMM (dV or dK): A (\hat{P} / d\hat{S}) from TMEM, B (dO^T / Q^T) from a static smem stage."""
        sm100_utils.gemm_ptx_partial_fp4(
            op, acc_offset, tCrA, tCrB_staged[None, None, None, stage],
            sA=None, sB=sB_staged[None, None, None, stage], tScaleA=tScaleA, tScaleB=tScaleB,
            zero_init=zero_init, tA_addr=ta_addr,
        )

    # ------------------------------------------------------------------ compute warps
    @cute.jit
    def split_wg(self, t: cute.Tensor, wg_idx: cutlass.Int32, num_wg: cutlass.Constexpr[int]):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            if const_expr(rank == 3):
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg))
                coord = (None, None, (None, wg_idx)) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2], reduced_shape[3] // num_wg))
                coord = (None, None, None, (None, wg_idx)) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
        tStS, tdPtdP, tdStdS_f32, tdVtdV, tdKtdK, sLSE, sdPsum, sdS_buf, S_loaded_mbar_ptr,
        pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dKV, pipeline_dP, pipeline_dQ,
        softmax_scale, softmax_scale_log2, block_info, SeqlenInfoCls, AttentionMaskCls, TileSchedulerCls,
        sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK, tiled_copy_r2s_dKV,
        sSFx=None,
        dpsum_scale=None, ds_boost=None, sr_seed=None, dv_scale_in=None, dk_scale_in=None, sr_ctr=None,
        sSFdS_mx_u8=None, sSFdSq_u8=None, sRowX=None, sColX=None, sColE=None, sQscale=None, sSFP_mx_u8=None, sDsScr=None,
        mFac=None, mTokFac=None, sTokV=None, mdK_g=None, mdV_g=None, mSfd=None,
    ):
        sLSE_2D = layout_utils.transpose_view(cute.make_tensor(
            sLSE.iterator, cute.make_layout((self.tile_m, self.tile_n, self.Q_stage), stride=(1, 0, self.stat_stride))
        ))
        sdPsum_2D = layout_utils.transpose_view(cute.make_tensor(
            sdPsum.iterator, cute.make_layout((self.tile_m, self.tile_n, self.dO_stage), stride=(1, 0, self.stat_stride))
        ))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)

        tileP_f32_like = self.tile_m // 32 * self.ds_dtype.width
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPtdS = cute.composition(tdStdS_f32, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPtdS = cute.make_tensor(tdStdS_f32.iterator, tdPtdS.layout)
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        # x32 loads / x16 stores (default): WG w's two 32-query blocks form K-tile w of the dV / dK GEMMs
        # (see tmem_ld_rep in __init__); the E2M1 words of stage st go to columns [8w + 4st, +4).
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_ld_rep)), Float32)
        # bf16 P / dS: 16 columns per 32 queries; E4M3 (FP8 mode): 8
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(self.tmem_ld_rep * self.ds_dtype.width // 32)), Float32
        )
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        if const_expr(self.dv8):
            # E4M3 \hat{P} (4 queries per word): 32 columns of the S bank, x8 stores (dS stays bf16 for dQ)
            tileP8 = self.tile_m // 32 * 8
            tmem_store_atom_p8 = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(self.tmem_ld_rep * 8 // 32)), Float32
            )
            thr_copy_r2t_p8 = copy_utils.make_tmem_copy(tmem_store_atom_p8, num_wg).get_slice(tidx)
            tStP8 = cute.make_tensor(tStS.iterator, cute.composition(tStS, (cute.make_layout((self.tile_n, tileP8)), 1, 1)).layout)
            tScP8 = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP8)), 1, 1))
            tScP8_r2t = thr_copy_r2t_p8.partition_S(tScP8)
            tStP8_r2t = thr_copy_r2t_p8.partition_D(tStP8)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r)
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)
        # C3: E2M1 \hat{P} (16 columns, 8 per WG) + SF\hat{P} (8 columns, 4 per WG) in the S bank; d\hat{S} / SFd\hat{S} in the dP bank.
        # A thread (key row) packs its 64 queries into 8 words and its 4 SFs into one word; the SF word is
        # stored to all 4 columns of the WG's K-tile group (only column row//32 is this row's slot).
        p4_cols = self.tile_m // 32 * 4
        sf_cols_a = 8
        tmem_store_atom_p4 = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p4_cols // num_wg)), Float32)
        tmem_store_atom_sf = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(sf_cols_a // num_wg)), Float32)
        thr_copy_r2t_p4 = copy_utils.make_tmem_copy(tmem_store_atom_p4, num_wg).get_slice(tidx)
        thr_copy_r2t_sf = copy_utils.make_tmem_copy(tmem_store_atom_sf, num_wg).get_slice(tidx)
        tScP4 = cute.composition(tScS, (cute.make_layout((self.tile_n, p4_cols)), 1, 1))
        tScSF = cute.composition(tScS, (cute.make_layout((self.tile_n, sf_cols_a)), 1, 1))
        tScP4_r2t = thr_copy_r2t_p4.partition_S(tScP4)
        tScSF_r2t = thr_copy_r2t_sf.partition_S(tScSF)
        # (built inside the kernel: host-side layouts cannot be used in the kernel region)
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(cute.recast_ptr(sdS_buf.iterator, sdS_epi_layout.inner), sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        # per-warp hand-offs let one WG run ahead; its E2M1 \hat{P} / d\hat{S} words and SF words land in TMEM columns that
        # hold the other WG's not-yet-loaded S / dP data (cols 0-39 of each bank), so the stores wait until
        # every compute warp has loaded S(t) / dP(t) (the mbarriers the MMA warp also waits on; no cost in step)
        phase_Sl_c = Int32(0)
        phase_dPe_c = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dPsum = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            if const_expr(self.doc_scaled):
                # this key tile's (document, kv head) scalars (device_scales.bwd1_fp8_doc_scalars)
                _dsc = block_info.batch_for_mods(batch_idx, n_block)
                softmax_scale = self._doc_scal[_dsc, head_idx, 0]
                softmax_scale_log2 = softmax_scale * math.log2(math.e)
                dpsum_scale = self._doc_scal[_dsc, head_idx, 1]
                ds_boost = self._doc_scal[_dsc, head_idx, 2]
                dv_scale_in = self._doc_scal[_dsc, head_idx, 3]
                dk_scale_in = self._doc_scal[_dsc, head_idx, 4]
            mask = AttentionMaskCls(seqlen)
            # the mods see (batch, q head, q_idx, kv_idx) in the mask's coordinates: the document under docs
            mod_batch = block_info.batch_for_mods(batch_idx, n_block)
            fastdiv_mods = (None, None)
            if const_expr(self.has_aux_tensors and (self.mask_mod is not None or self.score_mod is not None)):
                # upstream: indices wrapped modulo the sequence lengths so a padded row never reads an aux tensor OOB
                fastdiv_mods = (FastDivmodDivisorV2(cutlass.max(seqlen.seqlen_q, 1)),
                                FastDivmodDivisorV2(cutlass.max(seqlen.seqlen_k, 1)))
            mask_fn = partial(
                mask.apply_mask_sm100_transposed, tScS_t2r=tScS_t2r, t0ScS_t2r=t0ScS_t2r, n_block=n_blk_loc,
                mask_seqlen=True, mask_causal=self.is_causal, mask_local=self.is_local, mask_mod=self.mask_mod,
                batch_idx=mod_batch, aux_data=self.aux_data, fastdiv_mods=fastdiv_mods,
            )
            nm_c = m_block_max - m_block_min
            loop_count = nm_c * self.qhead_per_kvhead
            process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
            if const_expr(sr_ctr is not None):
                # Bwd1Spec.ds_sr on a device counter: the seed, once per work tile, through the opaque load of the
                # all-4-bit kernel (an ordinary load is hoisted to the kernel entry and held in a register by every
                # warp: LOWBIT 8.20). Same formula as the former host counter, so the sequence is unchanged.
                from flash_attn.cute.flash_bwd_sm100_fp4q import sr_seed_from_counter
                sr_seed_t = sr_seed_from_counter(sr_ctr.iterator)
            else:
                sr_seed_t = sr_seed
            for iter_idx in cutlass.range(loop_count, unroll=1):
                if const_expr(self.qhead_per_kvhead > 1):
                    _qh_sr, m_block = self._gqa_qm(iter_idx, nm_c, m_block_min, n_block)
                else:
                    m_block = m_block_min + iter_idx
                    # the q-head of this iteration; part of the SR dither hash (`sr-dither-missing-batch-head`)
                    _qh_sr = Int32(0)
                tb_s = [None, None]
                tb_dp = [None, None]
                if const_expr(self.mxop_tb):
                    # token-block factors (mxop_tb): S = f_K(key block) f_Q(query block) S^, dP = f_V f_dO dP^; a warp's 32
                    # lanes are one key block and stage st of WG w is query block 2 st + w (the x32 TMEM loads)
                    _hq = head_idx * self.qhead_per_kvhead + _qh_sr if const_expr(self.qhead_per_kvhead > 1) else head_idx
                    _kb = n_block * (self.tile_n // 32) + dp_idx // 32
                    if const_expr(self.sf_dense):
                        _fk = _fv = None   # (sf_dense: the factors arrive in the statistics stages)
                    else:
                        _fk = mFac[1, batch_idx, head_idx, _kb]
                        _fv = mFac[2, batch_idx, head_idx, _kb]
                    if const_expr(self.token_local):
                        # (cov3, codes_packed: the token table is per PACKED row -- the work tile's row shifts)
                        _kr = n_block * self.tile_n + dp_idx
                        _tsq = None
                        if const_expr(self.codes_packed):
                            _tsq, _tsk = self._pk_shifts(seqlen, m_blk_off, n_block, n_blk_loc)
                            _kr = _kr + _tsk
                        # token-local: this thread's key's own factors (dP^T rows are keys: a per-thread scalar)
                        _fv = mTokFac[2, batch_idx, head_idx, _kr]
                        tl_fk = mTokFac[1, batch_idx, head_idx, _kr]
                        # this lane's query of each stage, f_Q / f_dO: issued here, staged into the warp slots later
                        tl_pre = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                        for _st in cutlass.range_constexpr(2):
                            _q = m_block * self.tile_m + tScS_t2r[None, _st, 0, 0][0][1] + dp_idx % 32
                            if const_expr(self.codes_packed):
                                _q = _q + _tsq
                            if const_expr(self.fp8):
                                tl_pre[_st] = mTokFac[0, batch_idx, _hq, _q]
                            tl_pre[2 + _st] = mTokFac[3, batch_idx, _hq, _q]
                    for _st in cutlass.range_constexpr(2):
                        _qb = m_block * (self.tile_m // 32) + 2 * _st + wg_idx
                        # an all-zero 32-token block (varlen padding, zero rows) has factor 2^-127: the product flushes
                        # to 0 and a masked -inf score would become -inf x 0 = NaN. Floored at the smallest normal
                        # (S^ is 0 on such a block, so the unmasked value is 0 either way)
                        if const_expr(self.sf_dense and not self.token_local):
                            pass   # (sf_dense: from the statistics stages below; token-local: its own factors)
                        else:
                            if const_expr(self.token_local):
                                tb_s[_st] = softmax_scale_log2 * tl_fk  # token-local: f_Q per element (S -> P loop)
                            else:
                                tb_s[_st] = cutlass.max(softmax_scale_log2 * (_fk * mFac[0, batch_idx, _hq, _qb]), Float32(1.1754944e-38))
                            if const_expr(self.token_local):
                                tb_dp[_st] = ds_boost * _fv  # f_dO is per query: applied per element in the dS fold
                            else:
                                tb_dp[_st] = ds_boost * (_fv * mFac[3, batch_idx, _hq, _qb])
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                if const_expr(self.mxop_tb and self.sf_dense and not self.token_local):
                    # the factors the load warp wrote after this stage's LSE rows: f_Q (4 query blocks), f_K, f_V
                    _fk = sLSE[self.tile_m + 4 + dp_idx // 32, consumer_state_LSE.index]
                    _fv = sLSE[self.tile_m + 8 + dp_idx // 32, consumer_state_LSE.index]
                    for _st in cutlass.range_constexpr(2):
                        tb_s[_st] = cutlass.max(
                            softmax_scale_log2 * (_fk * sLSE[self.tile_m + 2 * _st + wg_idx, consumer_state_LSE.index]),
                            Float32(1.1754944e-38))
                tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
                # (codes_packed: a key / query row past the document is the NEXT document's -- finite codes where the
                # aligned layout has zero rows. Nothing to emulate for S / P / dP: a key past the document is masked
                # (P = 0; its tile is always a masked one), a query past it has LSE = +inf (the aligned statistics row's
                # padding value), so P = +0 exactly either way, and the block exponents (abs-maxes) see the same zeros.
                # dS = P x = +0 needs x > 0: the folded D'' of a padding row is +2^120 (the dO launch; codes_packed
                # requires ds_fold), so x = b dP + 2^120 > 0. The dQ side's key rows past the document: zero codes below.)
                # (`s-loaded-double-arrive`: S_loaded must be arrived once per tile; a former split-stage load variant
                # that also arrived it is not shipped.)
                if const_expr(self.s_loaded_early):
                    # every compute warp has S(t) in registers: the S bank's free columns may now receive the
                    # parked SFV / SFdO (MMA warp) and dP(t) may be issued -- signalled before the exps
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(self.warp_sync):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                    else:
                        self.compute_sync_barrier.arrive_and_wait()
                        if tidx == 0:
                            cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                # the mask / mods run in the mask's coordinates (document-local under docs; batched: the same numbers)
                m_loc = m_block - m_blk_off if const_expr(self.doc_varlen) else m_block
                n_loc = n_blk_loc if const_expr(self.doc_varlen) else n_block
                _hq_m = head_idx * self.qhead_per_kvhead + _qh_sr if const_expr(self.qhead_per_kvhead > 1) else head_idx
                tSrSpre = None
                if const_expr(self.score_mod is not None):
                    # upstream order: score_mod on the scaled score, then the mask; P = exp2(S' log2 e - LSE). The
                    # scaled pre-mod score is kept for score_mod_bwd. Scale = the kernel's own score scale (NVFP4 codes
                    # carry their SFs; FP8 / token-block: the folded descales, per stage for token blocks).
                    tSrSpre = cute.make_rmem_tensor(tSrS_t2r.shape, Float32)
                    cS_mod = cute.domain_offset((n_loc * self.tile_n, m_loc * self.tile_m),
                                                cute.make_identity_tensor(self.mma_tiler_kq[:2]))
                    tScS_mod = thr_copy_t2r.partition_D(thr_mma_S.partition_C(cS_mod))
                    for _st in cutlass.range_constexpr(cute.size(tScS_t2r, mode=[1])):
                        _sl2m = tb_s[_st] if const_expr(self.mxop_tb and self.fp8) else softmax_scale_log2
                        _scn = _sl2m * Float32(math.log(2.0))
                        _cur = tSrS_t2r[None, _st, 0, 0]
                        _pre = tSrSpre[None, _st, 0, 0]
                        for _i in cutlass.range_constexpr(cute.size(_cur)):
                            _pre[_i] = _cur[_i] * _scn
                        apply_score_mod_inner(
                            _cur, tScS_mod[None, _st, 0, 0], self.score_mod, mod_batch, _hq_m, _scn,
                            self.mod_vec_size, Float32, self.aux_data, fastdiv_mods, seqlen,
                            constant_q_idx=None, qhead_per_kvhead=1, transpose_indices=True,
                        )
                check_m_boundary = (m_loc + 1) * self.tile_m > seqlen.seqlen_q
                needs_mask = False  # this tile may hold -inf scores (Direct-P clamps them)
                if const_expr(self.is_causal and not self.is_local and self.causal_diag_only):
                    # query q sees key k iff k <= q + (seqlen_k - seqlen_q): the S^T tile (keys of n_block, queries
                    # of m_block) is fully visible once its smallest query reaches its largest key, so only the
                    # diagonal tiles (and a ragged last query tile) pay the 128 compare + select per thread
                    needs_mask = (m_loc * self.tile_m + (seqlen.seqlen_k - seqlen.seqlen_q)
                                  < (n_loc + 1) * self.tile_n - 1) or check_m_boundary
                    if const_expr(self.seqused_k or self.doc_varlen):
                        needs_mask = needs_mask or ((n_loc + 1) * self.tile_n > seqlen.seqlen_k)
                    if needs_mask:
                        mask_fn(tSrS_t2r, m_block=m_loc, head_idx=_hq_m, is_full_block=False, check_m_boundary=check_m_boundary)
                elif const_expr(self.is_causal or self.is_local or self.mask_mod is not None):
                    needs_mask = True
                    mask_fn(tSrS_t2r, m_block=m_loc, head_idx=_hq_m, is_full_block=False, check_m_boundary=check_m_boundary)
                elif const_expr(self.seqused_k or self.doc_varlen):
                    # non-causal varlen: only the key-boundary tile holds keys >= seqused_k / the document's length
                    needs_mask = (n_loc + 1) * self.tile_n > seqlen.seqlen_k
                    if needs_mask:
                        mask_fn(tSrS_t2r, m_block=m_loc, head_idx=_hq_m, is_full_block=False, check_m_boundary=check_m_boundary)
                # non-causal: the transposed mask only zeroes keys beyond seqlen_k (uniform predicate, 64 SELs
                # per step); the interface requires seqlen_k % 128 == 0, so there is nothing to mask
                num_stages = cute.size(tScS_t2r, mode=[1])
                # P = exp2(S * scale_log2 - LSE)
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.ds_dtype)
                tSrP_u32 = cute.recast_tensor(tSrP_r2t_f32, cutlass.Int32)
                wps = cute.size(tSrP_r2t_f32) // num_stages  # 32-bit words per stage (8 for E4M3)
                tSrP4_r2t_f32 = cute.make_rmem_tensor(tScP4_r2t.shape, Float32)
                tSrP4_u32 = cute.recast_tensor(tSrP4_r2t_f32, cutlass.Int32)
                tSrSF_r2t_f32 = cute.make_rmem_tensor(tScSF_r2t.shape, Float32)
                tSrSF_r2t_u32 = cute.recast_tensor(tSrSF_r2t_f32, cutlass.Int32)
                if const_expr(self.dv8):
                    tSrP8_f32 = cute.make_rmem_tensor(tScP8_r2t.shape, Float32)
                    tSrP8_u32 = cute.recast_tensor(tSrP8_f32, cutlass.Int32)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    p_dv_src = tSrS_cur
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(self.lse_shfl):
                        self._stat_shfl(tSsLSE_cur, tSrLSE_s2r)
                    else:
                        cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                    tSrLSE = tSrLSE_s2r
                    tl_fq = None
                    if const_expr(self.token_local and self.fp8):
                        # token-local: S = f_K(this key) f_Q(query) S^ -- the query factor per element
                        tl_fq = self._tl_query_factors(mTokFac, 0, batch_idx, _hq, m_block, tScS_t2r[None, stage, 0, 0],
                                                       sTokV[None, stage, wg_idx * 4 + dp_idx // 32 % 4, 0], dp_idx % 32,
                                                       True, pre=tl_pre[stage])
                    for v in cutlass.range_constexpr((cute.size(tSrS_t2r, mode=[0]) // 2)):
                        _sl2 = tb_s[stage] if const_expr(self.mxop_tb and self.fp8) else softmax_scale_log2  # dk8: S is NVFP4
                        if const_expr(self.score_mod is not None):
                            _sl2 = Float32(math.log2(math.e))  # S' is already the scaled, modified score
                        if const_expr(self.token_local and self.fp8):
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                (_sl2 * tl_fq[2 * v], _sl2 * tl_fq[2 * v + 1]),
                                (-tSrLSE[2 * v], -tSrLSE[2 * v + 1]),
                            )
                        else:
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                (_sl2, _sl2),
                                (-tSrLSE[2 * v], -tSrLSE[2 * v + 1]),
                            )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    if const_expr(self.dv8 and self.mxp):
                        # dk8 block-scaled P^ (Bwd1Spec.p_scaling_fp4=mx32): as the FP8 route's mx32 branch below, into
                        # the dv8 P^ words
                        wps8 = cute.size(tSrP8_f32) // num_stages
                        _pv = tSrS_cur
                        if const_expr(self.token_local):
                            # token-local: dV = P^T dO against token-local dO codes with a unit dV-side scale: P^ carries
                            # f_dO(q) (its block exponent absorbs the range; dV is not a causal output)
                            tl_fdo = self._tl_query_factors(mTokFac, 3, batch_idx, _hq, m_block, tScS_t2r[None, stage, 0, 0],
                                                            sTokV[None, stage, wg_idx * 4 + dp_idx // 32 % 4, 1],
                                                            dp_idx % 32, True, pre=tl_pre[2 + stage])
                            _pv = cute.make_rmem_tensor(cute.size(tSrS_cur), Float32)
                            for v in cutlass.range_constexpr(cute.size(tSrS_cur) // 2):
                                _pv[2 * v], _pv[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]), (tl_fdo[2 * v], tl_fdo[2 * v + 1]))
                        am_p = mxs.famax_tree([_pv[i] for i in range(cute.size(_pv))])
                        e_p = mxs.ue8m0_exp_448(am_p)
                        mul_p = mxs.ue8m0_mul(e_p)
                        pk = cute.make_rmem_tensor(cute.size(tSrS_cur), Float32)
                        for v in cutlass.range_constexpr(cute.size(tSrS_cur) // 2):
                            pk[2 * v], pk[2 * v + 1] = cute.arch.mul_packed_f32x2((_pv[2 * v], _pv[2 * v + 1]), (mul_p, mul_p))
                        for j in cutlass.range_constexpr(wps8):
                            tSrP8_u32[wps8 * stage + j] = sm100_utils.packed_float_to_ue4m3(pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3])
                        sSFP_mx_u8[(dp_idx % 32) * 16 + (dp_idx // 32) * 4 + (2 * stage + wg_idx)] = e_p.to(cute.Uint8)
                    elif const_expr(self.dv8):
                        wps8 = cute.size(tSrP8_f32) // num_stages
                        for j in cutlass.range_constexpr(wps8):
                            tSrP8_u32[wps8 * stage + j] = sm100_utils.packed_float_to_ue4m3(
                                p_dv_src[4 * j], p_dv_src[4 * j + 1], p_dv_src[4 * j + 2], p_dv_src[4 * j + 3]
                            )
                    elif const_expr(self.fp8 and self.mxp):
                        # block-scaled \hat{P}: one UE8M0 exponent per (this key row, this stage's 32 queries),
                        # codes = P' * 2^-e (P' = P * 2^8 through LSE; the shift is absorbed by the exponent); the SF byte
                        # into the (32 x 4)-byte SF atom before "P stored" (the fence below covers it)
                        _pv = tSrS_cur
                        if const_expr(self.token_local):
                            # token-local: f_dO(q) into P^ (token-local dO codes, unit dV-side scale)
                            tl_fdo = self._tl_query_factors(mTokFac, 3, batch_idx, _hq, m_block, tScS_t2r[None, stage, 0, 0],
                                                            sTokV[None, stage, wg_idx * 4 + dp_idx // 32 % 4, 1],
                                                            dp_idx % 32, True, pre=tl_pre[2 + stage])
                            _pv = cute.make_rmem_tensor(cute.size(tSrS_cur), Float32)
                            for v in cutlass.range_constexpr(cute.size(tSrS_cur) // 2):
                                _pv[2 * v], _pv[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]), (tl_fdo[2 * v], tl_fdo[2 * v + 1]))
                        am_p = mxs.famax_tree([_pv[i] for i in range(cute.size(_pv))])
                        e_p = mxs.ue8m0_exp_448(am_p)
                        mul_p = mxs.ue8m0_mul(e_p)
                        pk = cute.make_rmem_tensor(cute.size(tSrS_cur), Float32)
                        for v in cutlass.range_constexpr(cute.size(tSrS_cur) // 2):
                            pk[2 * v], pk[2 * v + 1] = cute.arch.mul_packed_f32x2((_pv[2 * v], _pv[2 * v + 1]), (mul_p, mul_p))
                        for j in cutlass.range_constexpr(wps):
                            tSrP_u32[wps * stage + j] = sm100_utils.packed_float_to_ue4m3(pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3])
                        sSFP_mx_u8[(dp_idx % 32) * 16 + (dp_idx // 32) * 4 + (2 * stage + wg_idx)] = e_p.to(cute.Uint8)
                    elif const_expr(self.fp8):
                        # E4M3 \hat{P} (P shifted by 2^p_shift on the host through LSE): 4 queries per word
                        for j in cutlass.range_constexpr(wps):
                            tSrP_u32[wps * stage + j] = sm100_utils.packed_float_to_ue4m3(
                                tSrS_cur[4 * j], tSrS_cur[4 * j + 1], tSrS_cur[4 * j + 2], tSrS_cur[4 * j + 3]
                            )
                    else:
                        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    if const_expr(stage == 0 and not self.s_loaded_early):
                        cute.arch.fence_view_async_tmem_load()
                        # every compute warp has S(t) in registers: the S bank's free columns may now
                        # receive the parked SFV / SFdO (MMA warp) and P may be stored
                        if const_expr(self.warp_sync):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                        else:
                            self.compute_sync_barrier.arrive_and_wait()
                            if tidx == 0:
                                cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                # \hat{P} columns [0, 32) overlap the other WG's S columns: store once every warp has loaded S.
                # pp: WG0's P^ words land in its own S columns (query blocks 0, 2 -> columns [0, 8), [16, 24) of
                # block 0), so only WG1 waits
                if const_expr(self.warp_sync and self.pp):
                    if wg_idx != 0:
                        cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl_c)
                        phase_Sl_c ^= 1
                elif const_expr(self.warp_sync):
                    cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl_c)
                    phase_Sl_c ^= 1
                if const_expr(self.dv8):
                    cute.copy(thr_copy_r2t_p8, tSrP8_f32, tStP8_r2t)
                else:
                    cute.copy(thr_copy_r2t, tSrP_r2t_f32, tStP_r2t)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                if const_expr(not self.warp_sync):
                    self.compute_sync_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                # dS^T = P^T * (dP^T - D)
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                if const_expr(self.mxop_tb and self.sf_dense and not self.token_local):
                    for _st in cutlass.range_constexpr(2):
                        tb_dp[_st] = ds_boost * (_fv * sdPsum[self.tile_m + 2 * _st + wg_idx, consumer_state_dPsum.index])
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                tSrdS4_r2t_f32 = cute.make_rmem_tensor(tScP4_r2t.shape, Float32)
                tSrdS4_u32 = cute.recast_tensor(tSrdS4_r2t_f32, cutlass.Int32)
                tSrdSSF = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                tSrdS8_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # FP8 mode: E4M3 d\hat{S} words of both stages
                tSrdS8_u32 = cute.recast_tensor(tSrdS8_f32, cutlass.Int32)
                row_am = [None, None]  # row rung: per-stage in-thread abs-max / per-column warp abs-max
                col_am = [None, None]
                if const_expr(self.dp_ld2):
                    tdPrdP_all = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                    cute.copy(thr_copy_t2r, tdPtdP_t2r, tdPrdP_all)
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(not self.warp_sync):
                        self.compute_sync_barrier.arrive_and_wait()
                for stage in cutlass.range_constexpr(num_stages):
                    if const_expr(self.dp_ld2):
                        tdPrdP_t2r = tdPrdP_all[None, stage, None, None]
                    else:
                        tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                        cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(not self.warp_sync):
                            self.compute_sync_barrier.arrive_and_wait()
                    if const_expr(stage == num_stages - 1):
                        # dP(t) is in registers: the dP bank's free columns may receive SFK / SFQ(t+1)
                        with cute.arch.elect_one():
                            pipeline_dP.consumer_release(consumer_state_S_P_dP)
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                    if const_expr(self.dpsum_shfl):
                        self._stat_shfl(tSsdPsum_cur, tSrdPsum)
                    else:
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    tl_fdo_ds = None
                    if const_expr(self.token_local):
                        tl_fdo_ds = self._tl_query_factors(mTokFac, 3, batch_idx, _hq, m_block, tScS_t2r[None, stage, 0, 0],
                                                           sTokV[None, stage, wg_idx * 4 + dp_idx // 32 % 4, 1],
                                                           dp_idx % 32, False)
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        if const_expr(self.ds_fold and self.token_local):
                            # token-local: b f_V(this key) f_dO(this query) d\hat{P} + D''
                            _dsb = tb_dp[stage]
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (_dsb * tl_fdo_ds[2 * v], _dsb * tl_fdo_ds[2 * v + 1]),
                                (tSrdPsum[2 * v], tSrdPsum[2 * v + 1]),
                            )
                        elif const_expr(self.ds_fold):
                            # b d\hat{P} + D'' (D'' = -b D / (s_v s_dO) from the host); mxop_tb: b x f_V f_dO of this block
                            _dsb = tb_dp[stage] if const_expr(self.mxop_tb) else ds_boost
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (_dsb, _dsb),
                                (tSrdPsum[2 * v], tSrdPsum[2 * v + 1]),
                            )
                        else:
                            # d\hat{P} - D / (s_v s_dO): the descales of the dP GEMM folded into D
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                (tSrdPsum[2 * v], tSrdPsum[2 * v + 1]), (-dpsum_scale, -dpsum_scale),
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                            )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]), (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1])
                        )
                        if const_expr((self.fp8 or self.dp_chain) and not self.ds_fold):  # dp8: ds_boost = s_v s_dO restores dS
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (ds_boost, ds_boost)
                            )
                    if const_expr(self.score_mod_bwd is not None):
                        # dS w.r.t. the scaled pre-mod score (upstream apply_score_mod_bwd). score_mod_bwd is linear in
                        # its grad (chain rule), so the kernel's scaled dS units (ds_boost, P x 2^shift) carry through;
                        # the dS quantisation below sees the final dS (its block scale follows it)
                        apply_score_mod_bwd_inner(
                            tdPrdP_cur, tSrSpre[None, stage, 0, 0], tScS_mod[None, stage, 0, 0], self.score_mod_bwd,
                            mod_batch, _hq_m, Float32(1.0), self.mod_vec_size, Float32, self.aux_data, fastdiv_mods,
                            seqlen, constant_q_idx=None, qhead_per_kvhead=1, transpose_indices=True,
                        )
                    if const_expr(self.mxds_row):
                        # row rung: only the statistics here; both stages are quantised after the loop once the
                        # other WG's row abs-max and the other warps' column abs-max are in (see below)
                        _vals = [tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))]
                        row_am[stage] = mxs.famax_tree(_vals)
                        col_am[stage] = mxs.warp_col_max_scatter(_vals, dp_idx % 32)
                    elif const_expr(self.mxds):
                        # Block-scaled d\hat{S}: two quantisations of this stage's 32 dS values, one per consumer
                        # GEMM. dK side (K = queries): this key row's 32 queries are one UE8M0 group -- in-thread
                        # abs-max, codes to the TMEM words.
                        # abs-max of the 32 values as a tree of 3-input abs-max instructions (17 FMNMX3)
                        cm_q = None  # ds_dq_rowlocal="all": this lane's query column abs-max over the warp's 32 keys
                        if const_expr(self.ds_scaling == "blk32" and self.ds_dq_rowlocal == "all" and not self.token_local):
                            # the column abs-maxes first (reduce-scatter); the block's is their warp max (exact, so e_r
                            # is bitwise the tree + redux form's)
                            cm_q = mxs.warp_col_max_scatter([tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))], dp_idx % 32)
                            e_r = mxs.ue8m0_exp_448(mxs.redux_max_abs_f32(cm_q))
                        elif const_expr(self.token_local and self.fp8):
                            # token-local Q in the dK GEMM (unit Q scale): f_Q(query) into the dK side's values
                            tl_fq_dk = self._tl_query_factors(mTokFac, 0, batch_idx, _hq, m_block,
                                                              tScS_t2r[None, stage, 0, 0],
                                                              sTokV[None, stage, wg_idx * 4 + dp_idx // 32 % 4, 0],
                                                              dp_idx % 32, False)
                            tl_dsk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                            for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                tl_dsk[2 * v], tl_dsk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (tl_fq_dk[2 * v], tl_fq_dk[2 * v + 1]))
                            e_r = mxs.ue8m0_exp_448(mxs.redux_max_abs_f32(famax_tree([tl_dsk[i] for i in range(cute.size(tl_dsk))])))
                        else:
                            am_r = famax_tree([tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))])
                            if const_expr(self.ds_scaling == "blk32"):
                                # one UE8M0 exponent per (32 keys x 32 queries) block serves both GEMMs (the SF tiles
                                # replicate it along each contraction axis): a single quantisation and pack
                                e_r = mxs.ue8m0_exp_448(mxs.redux_max_abs_f32(am_r))
                            else:
                                e_r = mxs.ue8m0_exp_448(am_r)
                        mul_r = mxs.ue8m0_mul(e_r)
                        tdPrdSk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                        for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                            if const_expr(self.token_local and self.fp8):
                                tdPrdSk[2 * v], tdPrdSk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tl_dsk[2 * v], tl_dsk[2 * v + 1]), (mul_r, mul_r))
                            else:
                                tdPrdSk[2 * v], tdPrdSk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (mul_r, mul_r)
                                )
                        for j in cutlass.range_constexpr(wps):
                            if const_expr(self.ds_sr):
                                # sr_ds under the block-scaled dS (opt-in): the same dither hash as the per-tensor
                                # branch below, on the block-scaled codes (the dK words; under blk32 the dQ tile takes
                                # the same words unless ds_dq_rowlocal == "all", whose dQ codes stay RTN); the seed is
                                # the device counter's (sr_seed_t, as the per-tensor branch)
                                xr = (sr_seed_t ^ (n_block * Int32(-1640531535)) ^ (m_block * Int32(-2048144777))
                                      ^ (tidx * Int32(-1028477379))
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))
                                      ^ (_qh_sr * Int32(-1509342103))
                                      + Int32(2 * j + 1) * Int32(-1640531527)) * Int32(-2048144789)
                                xr = xr ^ ((xr >> 13) & Int32(0x7FFFF))
                                xr = xr * Int32(-1028477387)
                                xr = xr ^ ((xr >> 16) & Int32(0xFFFF))
                                tSrdS8_u32[wps * stage + j] = sm100_utils.packed_float_to_e4m3x4_rs(
                                    tdPrdSk[4 * j], tdPrdSk[4 * j + 1], tdPrdSk[4 * j + 2], tdPrdSk[4 * j + 3], xr
                                )
                            else:
                                tSrdS8_u32[wps * stage + j] = sm100_utils.packed_float_to_ue4m3(
                                    tdPrdSk[4 * j], tdPrdSk[4 * j + 1], tdPrdSk[4 * j + 2], tdPrdSk[4 * j + 3]
                                )
                        # dQ side (K = keys): the warp's 32 lanes are keys 32k..32k+31 -- one group per query column
                        # (redux.sync.max.abs per column, uniform result), codes to the smem d\hat{S}^T tile.
                        lane = dp_idx % 32
                        kgrp = dp_idx // 32
                        tdPrdSq_w = cute.make_rmem_tensor(wps, cutlass.Int32)
                        if const_expr(self.ds_scaling == "blk32" and self.ds_dq_rowlocal != "all"):
                            # blk32: the block exponent is the dQ-side exponent of every query in the block, so the
                            # smem tile takes the same words
                            e_q = e_r
                            for j in cutlass.range_constexpr(wps):
                                tdPrdSq_w[j] = tSrdS8_u32[wps * stage + j]
                            if const_expr(self.ds_dq_rowlocal == "diagonal"):
                                # ds_dq_rowlocal="diagonal": on a 32 x 32 block the causal diagonal crosses, the shared
                                # exponent would carry later queries' dS (hence keys / dO / Q after a query) into that
                                # query's dQ codes. Such a block gets the per-query exponent below instead (warp-uniform
                                # branch: a warp's 32 lanes are one key block, a stage one query block). Query q sees key
                                # k iff k <= q + (s_k - s_q); the block holds a masked pair iff its last key exceeds its
                                # first query, and a visible pair iff its first key reaches its last query.
                                _qb0 = m_loc * self.tile_m + tScS_t2r[None, stage, 0, 0][0][1]
                                _kb0 = n_loc * self.tile_n + kgrp * 32
                                _qb0 = _qb0 + (seqlen.seqlen_k - seqlen.seqlen_q)
                                if _kb0 + 31 > _qb0:
                                    if _kb0 <= _qb0 + 31:
                                        e_q = self._ds_dq_rowlocal_words_x(tdPrdP_cur, tdPrdSq_w, lane, wps,
                                                                           sColE[None, stage, wg_idx * 4 + kgrp])
                        elif const_expr(self.ds_dq_rowlocal == "all" and self.token_local):
                            # token-local: the dQ GEMM's K is token-local (K_k = f_K(k) c_k, unit dQ-side SF): this key's
                            # factor goes into the dQ side's values before their per-query exponent
                            tl_vk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                            for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                tl_vk[2 * v], tl_vk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (tl_fk, tl_fk))
                            e_q = self._ds_dq_rowlocal_words_x(tl_vk, tdPrdSq_w, lane, wps,
                                                               sColE[None, stage, wg_idx * 4 + kgrp])
                        elif const_expr(self.ds_dq_rowlocal == "all"):
                            e_q = self._ds_dq_rowlocal_words_x(tdPrdP_cur, tdPrdSq_w, lane, wps,
                                                               sColE[None, stage, wg_idx * 4 + kgrp], cm=cm_q)
                        else:
                            e_q = Int32(1)
                            tdPrdSq = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                            for i in cutlass.range_constexpr(cute.size(tdPrdP_cur)):
                                e_i = mxs.ue8m0_exp_448(mxs.redux_max_abs_f32(tdPrdP_cur[i]))
                                tdPrdSq[i] = tdPrdP_cur[i] * mxs.ue8m0_mul(e_i)
                                if lane == i:
                                    e_q = e_i
                            for j in cutlass.range_constexpr(wps):
                                tdPrdSq_w[j] = sm100_utils.packed_float_to_ue4m3(
                                    tdPrdSq[4 * j], tdPrdSq[4 * j + 1], tdPrdSq[4 * j + 2], tdPrdSq[4 * j + 3]
                                )
                        tdPrdS_cvt = cute.recast_tensor(tdPrdSq_w, self.ds_dtype)
                    else:
                        # E4M3 d\hat{S} words: this stage's 8 words go to smem (dQ A operand) now and to TMEM after the wait
                        tdPrdS_w = cute.make_rmem_tensor(wps, cutlass.Int32)
                        for j in cutlass.range_constexpr(wps):
                            if const_expr(self.ds_sr):
                                # one dither word per four values, hashed from the per-call seed and this tile / thread
                                # `sr-dither-missing-batch-head`: batch_idx, head_idx and the q-head belong
                                # in this hash too. Without them the rounding error at a given (row-in-tile,
                                # key-tile, element) position is identical across every batch and head of the step,
                                # and under GQA that is the coherent case -- m_block repeats for each of the r
                                # q-heads sharing a kv head (m_block = m_block_min + iter_idx mod nm_c) and their
                                # dK / dV are summed inside the kernel, so the errors add in phase and the variance
                                # grows as r^2 instead of r, which negates most of what
                                # `stochastic-rounding-ds` buys. The host quantiser always included b and h.
                                xr = (sr_seed_t ^ (n_block * Int32(-1640531535)) ^ (m_block * Int32(-2048144777))
                                      ^ (tidx * Int32(-1028477379))
                                      ^ (batch_idx * Int32(-1252004669)) ^ (head_idx * Int32(-1861952465))
                                      ^ (_qh_sr * Int32(-1509342103))
                                      + Int32(2 * j + 1) * Int32(-1640531527)) * Int32(-2048144789)
                                xr = xr ^ ((xr >> 13) & Int32(0x7FFFF))
                                xr = xr * Int32(-1028477387)
                                xr = xr ^ ((xr >> 16) & Int32(0xFFFF))
                                tdPrdS_w[j] = sm100_utils.packed_float_to_e4m3x4_rs(
                                    tdPrdP_cur[4 * j], tdPrdP_cur[4 * j + 1], tdPrdP_cur[4 * j + 2], tdPrdP_cur[4 * j + 3], xr
                                )
                            else:
                                tdPrdS_w[j] = sm100_utils.packed_float_to_ue4m3(
                                    tdPrdP_cur[4 * j], tdPrdP_cur[4 * j + 1], tdPrdP_cur[4 * j + 2], tdPrdP_cur[4 * j + 3]
                                )
                            tSrdS8_u32[wps * stage + j] = tdPrdS_w[j]
                        tdPrdS_cvt = cute.recast_tensor(tdPrdS_w, self.ds_dtype)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                    if const_expr(self.mxds and not self.mxds_row):
                        # SF bytes into the (32 x 4)-byte SF atoms (offset (m % 32) * 16 + (m // 32) * 4 + k_group), after
                        # the dS buffer is ours: dK side keyed by this key row and query group 2 * stage + wg; dQ side
                        # keyed by this lane's query column (32 consecutive queries per stage) and the warp's key group
                        q0 = tScS_t2r[None, stage, 0, 0][0][1]
                        if const_expr(self.sf_asm):
                            # one broadcast byte per (query block, key block) -> the MMA warp builds both tiles
                            sDsScr[(2 * stage + wg_idx) * 4 + kgrp] = e_r.to(cute.Uint8)
                        else:
                            sSFdS_mx_u8[(dp_idx % 32) * 16 + (dp_idx // 32) * 4 + (2 * stage + wg_idx)] = e_r.to(cute.Uint8)
                        if const_expr(not self.sf_asm_q):
                            sSFdSq_u8[lane * 16 + (q0 // 32) * 4 + kgrp] = e_q.to(cute.Uint8)
                    if const_expr(not self.mxds_row):
                        tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                    if const_expr(not self.mxds_row):
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                if const_expr(self.mxds_row):
                    # row rung, after both stages: (1) exchange the row abs-max with the other WG and the column
                    # abs-max with the WG's other 3 warps; (2) row exponent (dK side) and column exponent (dQ side,
                    # applied by the reduce warps: 2^e per query in sQscale); (3) the column multipliers back through
                    # smem so every lane has its stage's 32; (4) quantise both stages: row-scaled words to TMEM (dK),
                    # column-scaled words to the smem d\hat{S}^T tile (dQ), the row exponent into the dK SF tile
                    _par = iter_idx % 2
                    _lane = dp_idx % 32
                    _wq = (tidx // 32) % 4  # warp within the WG = this warp's 32-key block
                    sRowX[_par, wg_idx, dp_idx] = utils.fmax(row_am[0], row_am[1])
                    for st in cutlass.range_constexpr(num_stages):
                        sColX[_par, wg_idx, _wq, st, _lane] = col_am[st]
                    self.compute_sync_barrier.arrive_and_wait()
                    e_row = mxs.ue8m0_exp_448(utils.fmax(utils.fmax(row_am[0], row_am[1]), sRowX[_par, wg_idx ^ 1, dp_idx]))
                    mul_row = mxs.ue8m0_mul(e_row)
                    for st in cutlass.range_constexpr(num_stages):
                        cm = utils.fmax(utils.fmax(sColX[_par, wg_idx, 0, st, _lane], sColX[_par, wg_idx, 1, st, _lane]),
                                        utils.fmax(sColX[_par, wg_idx, 2, st, _lane], sColX[_par, wg_idx, 3, st, _lane]))
                        e_c = mxs.ue8m0_exp_448(cm)
                        sColE[_par, wg_idx, st, _lane] = mxs.ue8m0_mul(e_c)
                        q0_st = tScS_t2r[None, st, 0, 0][0][1]
                        sc_val = mxs.ue8m0_val(e_c)
                        if _wq == 0:
                            sQscale[q0_st + _lane] = sc_val
                    self.compute_sync_barrier.arrive_and_wait()
                    for st in cutlass.range_constexpr(num_stages):
                        dS_st = tdPrdP_all[None, st, None, None][None, 0, 0]
                        mulc = cute.make_rmem_tensor(32, Float32)
                        cute.autovec_copy(sColE[_par, wg_idx, st, None], mulc)
                        wk = cute.make_rmem_tensor(cute.size(dS_st), Float32)
                        wq_ = cute.make_rmem_tensor(cute.size(dS_st), Float32)
                        for v in cutlass.range_constexpr(cute.size(dS_st) // 2):
                            wk[2 * v], wk[2 * v + 1] = cute.arch.mul_packed_f32x2((dS_st[2 * v], dS_st[2 * v + 1]), (mul_row, mul_row))
                            wq_[2 * v], wq_[2 * v + 1] = cute.arch.mul_packed_f32x2((dS_st[2 * v], dS_st[2 * v + 1]), (mulc[2 * v], mulc[2 * v + 1]))
                        tdPrdSq_w = cute.make_rmem_tensor(wps, cutlass.Int32)
                        for j in cutlass.range_constexpr(wps):
                            tSrdS8_u32[wps * st + j] = sm100_utils.packed_float_to_ue4m3(wk[4 * j], wk[4 * j + 1], wk[4 * j + 2], wk[4 * j + 3])
                            tdPrdSq_w[j] = sm100_utils.packed_float_to_ue4m3(wq_[4 * j], wq_[4 * j + 1], wq_[4 * j + 2], wq_[4 * j + 3])
                        cute.autovec_copy(cute.recast_tensor(tdPrdSq_w, self.ds_dtype), tRS_sdS[None, st])
                        sSFdS_mx_u8[_lane * 16 + (dp_idx // 32) * 4 + (2 * st + wg_idx)] = e_row.to(cute.Uint8)
                # d\hat{S} columns [0, 32) of the dP bank overlap the other WG's dP columns: store after dP empty
                # (pp: WG0's words land in its own dP columns, only WG1 waits)
                if const_expr(self.warp_sync and self.pp):
                    if wg_idx != 0:
                        pipeline_dP.sync_object_empty.wait(0, phase_dPe_c)
                        phase_dPe_c ^= 1
                elif const_expr(self.warp_sync):
                    pipeline_dP.sync_object_empty.wait(0, phase_dPe_c)
                    phase_dPe_c ^= 1
                cute.copy(thr_copy_r2t, tSrdS8_f32, tdPtdS_r2t)
                cute.arch.fence_view_async_tmem_store()
                consumer_state_S_P_dP.advance()
                cute.arch.fence_view_async_shared()
                if const_expr(not self.warp_sync):
                    self.compute_sync_barrier.arrive_and_wait()
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            if process_tile:
                thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                if const_expr(self.fp8):
                    dv_scale = dv_scale_in
                    dk_scale = dk_scale_in
                else:
                    dv_scale = dv_scale_in  # s_dO / 2^8 (host)
                    dk_scale = dk_scale_in  # softmax_scale s_ds s_q (host)
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx, batch_idx, head_idx, n_block, seqlen, thr_mma_dV, tdVtdV, mdV_tma_tensor, sdV,
                    tma_atom_dV, thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, dv_scale,
                    self.epi_barrier_base, "V",
                    **({"mdKV_g": mdV_g, "n_blk_loc": n_blk_loc} if const_expr(self.dkv_rows_packed) else {}),
                )
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx, batch_idx, head_idx, n_block, seqlen, thr_mma_dK, tdKtdK, mdK_tma_tensor, sdK,
                    tma_atom_dK, thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, dk_scale,
                    self.epi_barrier_base, "K",
                    **({"mdKV_g": mdK_g, "n_blk_loc": n_blk_loc} if const_expr(self.dkv_rows_packed) else {}),
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    # ------------------------------------------------------------------ reduce warps (dQ^T)
    @cute.jit
    def dQacc_reduce(
        self, mdQaccum, sdQ, tma_atom_dQ, thr_mma_dQ, tdQtdQ, pipeline_dQ,
        block_info, SeqlenInfoCls, TileSchedulerCls, mdQ_semaphore, dq_scale=None, sQscale=None, stage_mbar=None,
    ):
        stage_k = Int32(0)  # det_issue: running hand-off index (buffer k % bufs, phase (k / bufs) % 2)
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        cdQ = cute.make_identity_tensor(self.mma_tiler_dsk[:2])  # (tile_m, hdim)
        tdQcdQ = thr_mma_dQ.partition_C(cdQ)
        tdQrdQ_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        num_chunks = cute.size(tdQrdQ_shape, mode=[1])
        assert num_chunks == self.dQaccum_reduce_stage
        # registers -> smem (hdim, ncol) tile: thread t (TMEM lane t = d row t) holds ncol consecutive
        # columns of its row, written as 16-byte vectors (same pattern as the dK/dV epilogue)
        atom_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dqaccum_dtype, num_bits_per_copy=128)
        thr_layout_r2s = cute.make_ordered_layout((num_reduce_threads, 1), order=(1, 0))
        val_layout_r2s = cute.make_ordered_layout((1, 128 // self.dqaccum_dtype.width), order=(1, 0))
        tiled_copy_r2s = cute.make_tiled_copy_tv(atom_r2s, thr_layout_r2s, val_layout_r2s)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        one = cute.make_layout(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dq_stages)
        smem_stage = Int32(0)
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            if const_expr(self.doc_scaled):
                dq_scale = self._doc_scal[block_info.batch_for_mods(batch_idx, n_block), head_idx, 5]
            if const_expr(self.dq_tile_scaled):
                # dqfix: this work tile's row (its key tile's document, or the batch row): one scale per work tile,
                # after the vector and the two (b tq,) planes of device_scales._dq_tile_table (codes_packed: the
                # planes count the ALIGNED query tiles, mVtQ's -- the accumulator has the packed T rows)
                if const_expr(self.codes_packed):
                    _nt = self._pk_ntq
                else:
                    _nt = mdQaccum.shape[3] * (mdQaccum.shape[0] // self.tile_m)
                dq_scale = self._dq_tab[8 + 2 * _nt + block_info.batch_for_mods(batch_idx, n_block)]
            pkq = None   # codes_packed: the dQ rows are the packed query rows (a partial tile's extra rows add zeros)
            if const_expr(self.codes_packed):
                pkq = (seqlen.offset_q - m_blk_off * self.tile_m,)
            mdQaccum_cur = self._pk_rows(mdQaccum[None, None, head_idx, batch_idx], pkq, 0, 0)  # (s_q_rounded, d)
            gdQ = cute.local_tile(mdQaccum_cur, (self.tile_m, self.dQ_reduce_ncol), (None, None))  # (tile_m, ncol, m_blocks, chunks)
            tdQsdQ, tdQgdQ = cpasync.tma_partition(
                tma_atom_dQ, 0, one, cute.group_modes(sdQ, 0, 2), cute.group_modes(gdQ, 0, 2)
            )
            if const_expr(self.deterministic):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]
            nm_r = m_block_max - m_block_min
            loop_count = nm_r * self.qhead_per_kvhead
            process_tile = const_expr(not self.dyn_tiles) or m_block_min < m_block_max
            for iter_idx in cutlass.range(loop_count, unroll=1):
                if const_expr(self.qhead_per_kvhead > 1):
                    qh_r, m_block = self._gqa_qm(iter_idx, nm_r, m_block_min, n_block)
                    hq_r = head_idx * self.qhead_per_kvhead + qh_r
                    gdQ_j = cute.local_tile(self._pk_rows(mdQaccum[None, None, hq_r, batch_idx], pkq, 0, 0),
                                            (self.tile_m, self.dQ_reduce_ncol), (None, None))
                    tdQsdQ_c, tdQgdQ_c = cpasync.tma_partition(
                        tma_atom_dQ, 0, one, cute.group_modes(sdQ, 0, 2), cute.group_modes(gdQ_j, 0, 2)
                    )
                    if const_expr(self.deterministic):
                        mdQ_sem_c = mdQ_semaphore[None, None, hq_r, batch_idx]
                else:
                    m_block = m_block_min + iter_idx
                    tdQsdQ_c, tdQgdQ_c = tdQsdQ, tdQgdQ
                    if const_expr(self.deterministic):
                        mdQ_sem_c = mdQ_semaphore_cur
                dq_scale_tab = dq_scale
                if const_expr(self.dq_tile_causal == 2):
                    # dq-causal: this row's scale, loaded BEFORE the wait for the dQ tile (its latency hides behind it)
                    dq_scale_tab = self._dq_tab[8 + batch_idx * mdQaccum.shape[0] + m_block * self.tile_m + tidx]
                elif const_expr(self.dq_tile_causal == 1):
                    dq_scale_tab = self._dq_tab[8 + batch_idx * (mdQaccum.shape[0] // self.tile_m) + m_block]
                if const_expr(not self.det_staged):
                    pipeline_dQ.consumer_wait(dQ_consumer_state)
                    dq_scale_row = dq_scale_tab if const_expr(self.dq_tile_causal != 0) else dq_scale
                    if const_expr(self.mxds_row):
                        # row rung: this query row's d\hat{S}^T codes carry 2^-e_q; the compute warps left 2^e_q in sQscale
                        dq_scale_row = dq_scale_row * sQscale[tidx]
                    if const_expr(not self.reduce_chunked):
                        tdQrdQ = cute.make_rmem_tensor(tdQrdQ_shape, Float32)
                        cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            pipeline_dQ.consumer_release(dQ_consumer_state)
                        dQ_consumer_state.advance()
                    if const_expr(self.deterministic):
                        # lock value = this key tile's rank among the key tiles that meet m_block (upstream
                        # _dq_semaphore_lock_value): ascending from the first one (window / document start), or descending
                        # under spt. Batched causal / dense: n_block, as before.
                        if const_expr(self.spt or self.dyn_tiles):
                            _m_loc_r = m_block - m_blk_off if const_expr(self.doc_varlen) else m_block
                            _n_loc_r = n_blk_loc if const_expr(self.doc_varlen) else n_block
                            _n_lo_r, _n_hi_r = block_info.block_info.get_n_block_min_max(seqlen, _m_loc_r)
                            _lock = _n_hi_r - 1 - _n_loc_r if const_expr(self.spt) else _n_loc_r - _n_lo_r
                        else:
                            _lock = n_block
                        if const_expr(self.det_gqa_rot):
                            _lock = self._rot_lock(iter_idx, nm_r)
                        barrier.wait_eq(mdQ_sem_c[(m_block, None)].iterator, tidx, 0, _lock)
                    for chunk in cutlass.range_constexpr(num_chunks if const_expr(not self.reduce_off or self.reduce_chunked) else 0):
                        sdQ_cur = sdQ[None, None, smem_stage]
                        tdQsdQ_r2s = thr_copy_r2s.partition_D(sdQ_cur)
                        if const_expr(self.reduce_chunked):
                            # one chunk of dQ at a time (64 registers); dQ is released after the last chunk's load
                            # (dq_halves: each 64-column chunk is a half with its own full / empty barrier)
                            tdQrdQ_c = cute.make_rmem_tensor(tdQrdQ_shape[:1] + (1,) + tuple(tdQrdQ_shape[2:]), Float32)
                            cute.copy(thr_copy_t2r, tdQtdQ_t2r[None, chunk, None, None], tdQrdQ_c[None, 0, None, None])
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(chunk == num_chunks - 1):
                                cute.arch.sync_warp()
                                with cute.arch.elect_one():
                                    pipeline_dQ.consumer_release(dQ_consumer_state)
                                dQ_consumer_state.advance()
                            tdQrdQ = tdQrdQ_c
                            chunk_sel = 0
                        else:
                            chunk_sel = chunk
                        if const_expr(self.dqaccum_dtype.width == 16):
                            # f16 / bf16 partials: half the reduce-add bytes (the 50 GB/s-per-SM reduce-add wall)
                            tdQrdQ_f32 = cute.make_tensor(tdQrdQ[None, chunk_sel, None, None].iterator, tdQsdQ_r2s.shape)
                            # FP8 mode: d\hat{S} \hat{K} carries the E4M3 code magnitudes (up to 448 x 448 x 128); rescale
                            # to the natural dS K magnitude (x 2^6 like the FP4 path) before the f16 partials
                            for i in cutlass.range_constexpr(cute.size(tdQrdQ_f32) // 2):
                                tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1] = cute.arch.mul_packed_f32x2(
                                    (tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1]), (dq_scale_row, dq_scale_row)
                                )
                            tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dqaccum_dtype)
                            utils.cvt_f16(tdQrdQ_f32, tdQrdQ_r2s)
                        else:
                            tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, chunk_sel, None, None].iterator, tdQsdQ_r2s.shape)
                        if const_expr(not self.reduce_off):
                            cute.copy(thr_copy_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                        cute.arch.fence_view_async_shared()
                        self.reduce_sync_barrier.arrive_and_wait()
                        if const_expr(not self.reduce_off) and is_tma_warp:
                            cute.copy(tma_atom_dQ, tdQsdQ_c[None, smem_stage], tdQgdQ_c[None, m_block, chunk])
                            cute.arch.cp_async_bulk_commit_group()
                            # the slot written next is the one committed (stages - 1) groups ago
                            cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=True)
                        self.reduce_sync_barrier.arrive_and_wait()
                        smem_stage = (smem_stage + 1) % self.sdQaccum_stage
                    if const_expr(self.deterministic):
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                        self.reduce_sync_barrier.arrive_and_wait()
                        barrier.arrive_inc(mdQ_sem_c[(m_block, None)].iterator, tidx, 0, 1)
                if const_expr(self.det_staged):
                    # deterministic, staged: the whole dQ tile goes TMEM -> f16 -> SMEM and the TMEM
                    # accumulator is released BEFORE the semaphore wait, so the MMA's next dQ never waits on this key
                    # tile's turn or on the write drain. The semaphore then gates only the TMA reduce-adds; they are
                    # drained and released at once (no lag added to the next key tile's turn). Same lock values,
                    # same f16 partials, same TMA reduce-add per (m_block, chunk): the dQ bits are those of the
                    # unstaged deterministic path.
                    pipeline_dQ.consumer_wait(dQ_consumer_state)
                    dq_scale_row = dq_scale_tab if const_expr(self.dq_tile_causal != 0) else dq_scale
                    if const_expr(self.mxds_row):
                        dq_scale_row = dq_scale_row * sQscale[tidx]
                    stage_buf = stage_k % self.det_issue_bufs if const_expr(self.det_issue) else 0
                    for chunk in cutlass.range_constexpr(num_chunks):
                        tdQsdQ_r2s = thr_copy_r2s.partition_D(sdQ[None, None, stage_buf * num_chunks + chunk])
                        tdQrdQ_c = cute.make_rmem_tensor(tdQrdQ_shape[:1] + (1,) + tuple(tdQrdQ_shape[2:]), Float32)
                        cute.copy(thr_copy_t2r, tdQtdQ_t2r[None, chunk, None, None], tdQrdQ_c[None, 0, None, None])
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(chunk == num_chunks - 1):
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                pipeline_dQ.consumer_release(dQ_consumer_state)
                            dQ_consumer_state.advance()
                        tdQrdQ_f32 = cute.make_tensor(tdQrdQ_c[None, 0, None, None].iterator, tdQsdQ_r2s.shape)
                        for i in cutlass.range_constexpr(cute.size(tdQrdQ_f32) // 2):
                            tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1] = cute.arch.mul_packed_f32x2(
                                (tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1]), (dq_scale_row, dq_scale_row)
                            )
                        tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dqaccum_dtype)
                        utils.cvt_f16(tdQrdQ_f32, tdQrdQ_r2s)
                        if const_expr(self.det_issue and chunk == 0):
                            # the issue warp's TMA has read this buffer (a fresh buffer passes: parity of phase -1)
                            cute.arch.mbarrier_wait(stage_mbar + self.det_issue_bufs + stage_buf,
                                                    ((stage_k // self.det_issue_bufs) % 2) ^ 1)
                        cute.copy(thr_copy_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    cute.arch.fence_view_async_shared()
                    if const_expr(self.det_issue):
                        cute.arch.mbarrier_arrive(stage_mbar + stage_buf)
                        stage_k += 1
                if const_expr(self.det_staged and not self.det_issue):
                    if const_expr(self.spt or self.dyn_tiles):
                        _m_loc_s = m_block - m_blk_off if const_expr(self.doc_varlen) else m_block
                        _n_loc_s = n_blk_loc if const_expr(self.doc_varlen) else n_block
                        _n_lo_s, _n_hi_s = block_info.block_info.get_n_block_min_max(seqlen, _m_loc_s)
                        _lock_s = _n_hi_s - 1 - _n_loc_s if const_expr(self.spt) else _n_loc_s - _n_lo_s
                    else:
                        _lock_s = n_block
                    if const_expr(self.det_gqa_rot):
                        _lock_s = self._rot_lock(iter_idx, nm_r)
                    barrier.wait_eq(mdQ_sem_c[(m_block, None)].iterator, tidx, 0, _lock_s)
                    self.reduce_sync_barrier.arrive_and_wait()
                    if is_tma_warp:
                        for chunk in cutlass.range_constexpr(num_chunks):
                            cute.copy(tma_atom_dQ, tdQsdQ_c[None, chunk], tdQgdQ_c[None, m_block, chunk])
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                    self.reduce_sync_barrier.arrive_and_wait()
                    barrier.arrive_inc(mdQ_sem_c[(m_block, None)].iterator, tidx, 0, 1)
            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.reduce_sync_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        if is_tma_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=False)

    # ------------------------------------------------------------------ deterministic dQ: issue / release warps
    @cute.jit
    def _rot_lock(self, iter_idx, nm):
        """det_gqa_rot: the walk step within the q head's pass = this key tile's rank among the block's writers."""
        qh = iter_idx // nm
        return iter_idx - qh * nm

    @cute.jit
    def _det_dq_coords(self, iter_idx, nm_r, m_block_min, head_idx, n_block=None):
        """(q head, m block) of iteration iter_idx of a work tile, in the reduce warps' walk."""
        hq_r = head_idx
        m_block = m_block_min + iter_idx
        if const_expr(self.qhead_per_kvhead > 1):
            qh_r, m_block_g = self._gqa_qm(iter_idx, nm_r, m_block_min, n_block)
            hq_r = head_idx * self.qhead_per_kvhead + qh_r
            m_block = m_block_g
        return hq_r, m_block

    @cute.jit
    def dQ_issue(self, mdQaccum, sdQ, tma_atom_dQ, block_info, TileSchedulerCls, mdQ_semaphore, stage_mbar):
        """det_issue, idle warp 0: per (q head, m block) of every work tile, in the reduce warps' order: wait for the staged
        f16 dQ tile (stage_mbar[buf]), wait for this key tile's turn, issue the TMA reduce-adds, free the SMEM buffer once
        the TMA has read it (stage_mbar[B + buf]), wait for the writes and hand the release to the release warp
        (stage_mbar[2B], slot stage_mbar[2B + 1]). Lock values and reduce-adds as the staged path of dQacc_reduce."""
        lane = cute.arch.lane_idx()
        one = cute.make_layout(1)
        num_chunks = self.dQaccum_reduce_stage
        nb = self.det_issue_bufs
        stage_k = Int32(0)
        rel_free_phase = Int32(1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            nm_r = m_block_max - m_block_min
            loop_count = nm_r * self.qhead_per_kvhead
            pkq = None   # codes_packed: the dQ rows are the packed query rows
            if const_expr(self.codes_packed):
                pkq = (seqlen.offset_q - m_blk_off * self.tile_m,)
            for iter_idx in cutlass.range(loop_count, unroll=1):
                hq_r, m_block = self._det_dq_coords(iter_idx, nm_r, m_block_min, head_idx, n_block)
                gdQ_j = cute.local_tile(self._pk_rows(mdQaccum[None, None, hq_r, batch_idx], pkq, 0, 0),
                                        (self.tile_m, self.dQ_reduce_ncol), (None, None))
                tdQsdQ_c, tdQgdQ_c = cpasync.tma_partition(
                    tma_atom_dQ, 0, one, cute.group_modes(sdQ, 0, 2), cute.group_modes(gdQ_j, 0, 2)
                )
                sem_ptr = mdQ_semaphore[None, None, hq_r, batch_idx][(m_block, None)].iterator
                # lock value: as dQacc_reduce (upstream _dq_semaphore_lock_value)
                if const_expr(self.spt or self.dyn_tiles):
                    _m_loc_s = m_block - m_blk_off if const_expr(self.doc_varlen) else m_block
                    _n_loc_s = n_blk_loc if const_expr(self.doc_varlen) else n_block
                    _n_lo_s, _n_hi_s = block_info.block_info.get_n_block_min_max(seqlen, _m_loc_s)
                    _lock_s = _n_hi_s - 1 - _n_loc_s if const_expr(self.spt) else _n_loc_s - _n_lo_s
                else:
                    _lock_s = n_block
                if const_expr(self.det_gqa_rot):
                    _lock_s = self._rot_lock(iter_idx, nm_r)
                stage_buf = stage_k % nb
                cute.arch.mbarrier_wait(stage_mbar + stage_buf, (stage_k // nb) % 2)
                stage_k += 1
                barrier.wait_eq(sem_ptr, lane, 0, _lock_s)
                cute.arch.sync_warp()
                for chunk in cutlass.range_constexpr(num_chunks):
                    cute.copy(tma_atom_dQ, tdQsdQ_c[None, stage_buf * num_chunks + chunk], tdQgdQ_c[None, m_block, chunk])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(stage_mbar + nb + stage_buf)
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
                cute.arch.mbarrier_wait(stage_mbar + 2 * nb + 1, rel_free_phase)
                rel_free_phase ^= 1
                # every lane has seen this slot's release before lane 0 hands over the next drain: otherwise the
                # release warp can complete the slot's phase twice while a lagging lane still polls the old parity
                # (it then waits for a third completion that needs lane 0, which waits for it at the next sync_warp --
                # a hang seen on the packed rows with slower compute steps: softcap / mask_mod, fp8 d128 dense
                # deterministic, ~1 in 600-8000 calls; the same hand-off runs in every det_issue kernel)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(stage_mbar + 2 * nb)
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def dQ_release(self, block_info, TileSchedulerCls, mdQ_semaphore, stage_mbar):
        """det_issue, idle warp 1: per (q head, m block), after the issue warp has drained the reduce-adds
        (stage_mbar[2B]), increment the semaphore with release semantics (barrier.arrive_inc), then free the slot
        (stage_mbar[2B + 1]). The gpu-scope fence of the release stays off the issue warp's path."""
        lane = cute.arch.lane_idx()
        nb = self.det_issue_bufs
        drained_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block)
            nm_r = m_block_max - m_block_min
            loop_count = nm_r * self.qhead_per_kvhead
            for iter_idx in cutlass.range(loop_count, unroll=1):
                hq_r, m_block = self._det_dq_coords(iter_idx, nm_r, m_block_min, head_idx, n_block)
                sem_ptr = mdQ_semaphore[None, None, hq_r, batch_idx][(m_block, None)].iterator
                cute.arch.mbarrier_wait(stage_mbar + 2 * nb, drained_phase)
                drained_phase ^= 1
                barrier.arrive_inc(sem_ptr, lane, 0, 1)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(stage_mbar + 2 * nb + 1)
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ------------------------------------------------------------------ dK / dV epilogue (TMA store)
    @cute.jit
    def _dkv_row_store(self, tdKVrdKV, tdKVcdKV_t2r, n_blk_loc, seqlen, mdKV_g, head_idx, batch_idx):
        """The thread's dK / dV row segment straight to the caller's packed key row (real rows only)."""
        r_loc = tdKVcdKV_t2r[0][0]
        c_0 = tdKVcdKV_t2r[0][1]
        if n_blk_loc * self.tile_n + r_loc < seqlen.seqlen_k:
            prow = seqlen.offset_k + n_blk_loc * self.tile_n + r_loc
            nel = cute.size(tdKVrdKV.shape)
            cute.autovec_copy(cute.make_tensor(tdKVrdKV.iterator, cute.make_layout(nel)),
                              cute.make_tensor(mdKV_g[prow, None, head_idx, batch_idx].iterator + c_0,
                                               cute.make_layout(nel)))

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self, tidx, batch_idx, head_idx, n_block, seqlen, thr_mma, tdKVtdKV, mdKV, sdKV, tma_atom_dKV,
        thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, scale, barrier_id, K_or_V: cutlass.Constexpr[str],
        mdKV_g=None, n_blk_loc=None,
    ):
        assert K_or_V in ("K", "V")
        if const_expr((K_or_V == "K" and self.skip_dk) or (K_or_V == "V" and self.skip_dv)):
            # out_pass: this output belongs to another pass -- consume the dKV stage only
            pipeline_dKV.consumer_wait(consumer_state_dKV)
            with cute.arch.elect_one():
                pipeline_dKV.consumer_release(consumer_state_dKV)
            consumer_state_dKV.advance()
        else:
            tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
            dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
            epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
            num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
            num_wg = num_compute_threads // 128
            leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
            sdKV = sdKV[None, None, wg_idx]
            tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)
            mdKV_cur = mdKV[None, None, head_idx, batch_idx]
            if const_expr(self.dkv_rows_packed):
                # lp_docmaps: a key tile wholly inside its document goes out by the TMA store at the caller's packed
                # rows (the tile's coordinates shifted from the aligned start to the packed one); the document's last,
                # partial tile by the threads' row stores below (a whole tile would overrun into the next document)
                pk_full = (n_blk_loc + 1) * self.tile_n <= seqlen.seqlen_k
                mdKV_cur = cute.domain_offset(
                    (seqlen.offset_k + n_blk_loc * self.tile_n - n_block * self.tile_n, 0), mdKV_cur)
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0))
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
            gdKV_epi = cute.local_tile(gdKV, epi_tile, (0, None))
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV, 0, cute.make_layout(1), cute.group_modes(sdKV, 0, 2), cute.group_modes(gdKV_epi, 0, 2)
            )
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            # d > 128 (out_pass): the WG's hdim / 2 columns are staged in epi_tile[1]-column stages; load one stage per copy
            epi_rep = self.dK_reduce_ncol if const_expr(tile_hdim <= 128) else epi_tile[1]
            tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(epi_rep)), Float32)
            pipeline_dKV.consumer_wait(consumer_state_dKV)
            had = const_expr(K_or_V == "K" and self.had_dk is not None)
            if const_expr(had):
                # qk_hadamard: both halves of the row from TMEM, the rotation undone in registers (u, 64 values of
                # this warpgroup's half, already scaled); the stages below store from u instead of loading TMEM
                assert num_wg == 2 and tile_hdim == 128, "fused dK back-rotation: two compute warpgroups, d = 128"
                thr_copy_h = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
                src_h = thr_copy_h.partition_S(tdKVtdKV)
                cd_h = self.split_wg(thr_copy_h.partition_D(thr_mma.partition_C(cute.make_identity_tensor(
                    (self.tile_n, tile_hdim)))), wg_idx, num_wg)[None, None, 0, 0]
                if const_expr(num_epi_stages > 1):
                    cd_h = cd_h[None, 0]
                had_n = cute.size(cd_h.shape)  # this warpgroup's columns per stage
                assert num_epi_stages * had_n == 64, "fused dK back-rotation: 64 columns per warpgroup"
                had_u = cute.make_rmem_tensor(cute.make_layout(64), Float32)
                dk_unrotated_half(thr_copy_h, self.split_wg(src_h, 0, num_wg)[None, None, 0, 0],
                                  self.split_wg(src_h, 1, num_wg)[None, None, 0, 0], cd_h, had_u, wg_idx, scale,
                                  num_epi_stages, had_n, self.had_dk)
            for epi_stage in cutlass.range_constexpr(num_epi_stages):
                thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
                tdKVtdKV_t2r = self.split_wg(thr_copy_t2r.partition_S(tdKVtdKV), wg_idx, num_wg)[None, None, 0, 0]
                if const_expr(num_epi_stages > 1):
                    tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]
                cdKV = cute.make_identity_tensor((self.tile_n, tile_hdim))
                tdKVcdKV = thr_mma.partition_C(cdKV)
                tdKVcdKV_t2r = self.split_wg(thr_copy_t2r.partition_D(tdKVcdKV), wg_idx, num_wg)[None, None, 0, 0]
                if const_expr(num_epi_stages > 1):
                    tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]
                tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)
                if const_expr(had):
                    for i in cutlass.range_constexpr(had_n):
                        tdKVrdKV_t2r[i] = had_u[epi_stage * had_n + i]
                else:
                    cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
                    cute.arch.fence_view_async_tmem_load()
                if const_expr(scale is not None and not had):
                    for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                        tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                            (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                        )
                tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)
                tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))
                if const_expr(self.dkv_rows_packed):
                    if not pk_full:
                        # lp_docmaps: the document's partial last tile -- the thread's row segment straight to the
                        # caller's packed key row (real rows only)
                        self._dkv_row_store(tdKVrdKV, tdKVcdKV_t2r, n_blk_loc, seqlen, mdKV_g, head_idx, batch_idx)
                    else:
                        tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
                        cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
                        cute.arch.fence_view_async_shared()
                        cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
                        if leader_warp:
                            cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                            if const_expr(epi_stage < num_epi_stages - 1 or self.bwd_persist):
                                # (persistent CTA: the last stage's read waited too -- sdK / sdV alias sQ / sdO, which
                                # the load warp refills for the CTA's next work tile once this epilogue released)
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                            cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx,
                                                     number_of_threads=128 + cute.arch.WARP_SIZE)
                        cute.arch.fence_view_async_shared()
                        cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)
                else:
                    tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
                    cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
                    if leader_warp:
                        cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                        if const_expr(epi_stage < num_epi_stages - 1):
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx,
                                                 number_of_threads=128 + cute.arch.WARP_SIZE)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_dKV.consumer_release(consumer_state_dKV)
            consumer_state_dKV.advance()
        return consumer_state_dKV

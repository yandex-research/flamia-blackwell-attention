# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# 2-CTA low-precision backward (SM100, head_dim 128): the BF16 2-CTA kernel (flash_bwd_sm100.py) with per-tensor
# E4M3 operands for every GEMM (FP8 control, kind::f8f6f4, cta_group::2). \hat{P} = e4m3(2^8 P) (32 TMEM columns, shift through
# LSE on the host), d\hat{S} = e4m3(P' (b d\hat{P} + D'')) with D'' = -b D / (s_v s_dO) pre-scaled on the host (32 TMEM columns + the
# half-exchanged SMEM tile of dQ). Host-folded scales: softmax_scale s_q s_k (S), dv_scale / dk_scale (epilogues), dQ
# post-scale on the host. Stage 2 (nvfp4_s): S from NVFP4 Q / K with their scale factors (dk8 configuration).
import math
import os
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync

import quack.activation
from quack import layout_utils
from flash_attn.cute import utils
from flash_attn.cute.lowprec.specs import Bwd2Spec
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute.blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import blackwell_helpers as fp4_helpers
from flash_attn.cute import mx_scale_utils as mxs
from flash_attn.cute.qk_hadamard_cute import dk_unrotated_half
import cutlass.utils.blockscaled_layout as blockscaled_utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.utils import AuxData


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
from flash_attn.cute.lp_bwd_tiles import LPTileInfo, LPPairTileInfo
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    SingleTileVarlenScheduler,
)

from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100
from flash_attn.cute.softmax import apply_score_mod_inner, apply_score_mod_bwd_inner
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd,
    # Upstream split this into _default / _2cta_hdim192 variants; this kernel wants the default one.
    produce_block_sparse_q_loads_bwd_sm100_default as produce_block_sparse_q_loads_bwd_sm100,
)

class FlashAttentionBackwardSm100TwoCta8:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        lpt_l2: bool = False,
        spt: Optional[bool] = None,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
        nvfp4_s: bool = False,
        s_sf_vec_size: int = 16,
        doc_varlen: bool = False,  # masks / varlen: aligned-packed varlen (lp_bwd_tiles), documents 256-aligned (key pairs)
        # documents packed at 128 (lp_bwd_tiles.LPPairTileInfo): the key pair may straddle two documents; each CTA masks
        # in its own tile's document, both iterate the hull of the two query ranges. Needs doc_varlen; no mods.
        doc_t128: bool = False,
        # batched s_k % 256 == 128: the last key pair's upper half lies past the keys (masked; see key_tail)
        key_tail: bool = False,
        # blk32 dS^ only: row-local dQ-side exponents (as the 1-CTA kernel's argument). "diagonal" = on the
        # 32-key x 32-query blocks the causal diagonal crosses, the dQ side (this CTA's dQ SF tile rows + its SMEM dS^
        # copy, own half and the half shipped to the peer) takes one UE8M0 exponent per query over the block's 32 keys and
        # its own E4M3 codes; "all" = every block; the dK side (TMEM dS^, its SF tile) keeps the block exponent. dS is
        # backward-only: the scaling-consistency rule (shared operands) is unaffected.
        ds_dq_rowlocal: str = "off",
        # token-local operands: E4M3 operand copies (as the 1-CTA kernel's argument): per-key factors per thread, per-query
        # 32-vectors per stage through a per-warp SMEM slot, the dQ / dK dS sides and the block-scaled dV P^ (forced on)
        # carry the factors. Needs ds_dq_rowlocal="all". Compile-time.
        token_local_operands: bool = False,
        _test_skew_peer: bool = False,
        # dqfix: the f16 dQ pre-scale per row (document / batch row): mScalars is the (8,) vector, then the per-query-
        # tile plane of device_scales._dq_tile_table, read by the dQ reduce per tile instead of the one dq_scale
        dq_tile_scaled: bool = False,
        # dq-causal: batched causal square: the dQ reduce reads the scale of the query tile it reduces (the plane's
        # b tq + m_block), sized from the causal prefix of keys, instead of the batch row's one scale
        dq_tile_causal: int = 0,
        *,
        # every compile-time setting of this kernel (lowprec/specs.py; the driver builds it): no default, so a construction
        # site cannot silently run the defaults instead of the active configuration
        spec: Bwd2Spec,
    ):
        assert isinstance(spec, Bwd2Spec), type(spec)
        self.spec = spec
        self.token_local_arg = bool(token_local_operands)
        if self.token_local_arg:
            assert ds_dq_rowlocal == "all", "token_local_operands needs ds_dq_rowlocal='all'"
        # masks / varlen: window / softcap / score_mod / mask_mod / documents on the cluster (explicit constructor
        # arguments; all compile-time and in the driver's compile key)
        self.doc_varlen = bool(doc_varlen)
        self.doc_t128 = bool(doc_t128)
        self.dq_tile_scaled = bool(dq_tile_scaled)
        self.dq_tile_causal = int(dq_tile_causal)
        assert not self.dq_tile_causal or (self.dq_tile_scaled and not self.doc_varlen)
        # the last key pair may hold a half past the keys (doc_t128: an odd number of 128-key tiles): its GQA fp32
        # dK / dV reduce-add and dK / dV semaphore are skipped (its dK / dV are zero)
        self.key_tail = self.doc_t128 or bool(key_tail)
        assert not (self.key_tail and token_local_operands), "key_tail: not with token-local operands"
        assert not self.doc_t128 or (self.doc_varlen and mask_mod is None and score_mod is None
                                     and not token_local_operands), \
            "doc_t128: documents packed at 128 without mask_mod / score_mod / token-local operands"
        assert not (mask_mod is not None and (is_causal or is_local)), \
            "mask_mod with causal / window: express the causal / window part inside the mask_mod"
        assert (score_mod is None) == (score_mod_bwd is None), "score_mod needs score_mod_bwd (the backward's dS)"
        # a key pair's query range is data dependent (may be empty): the local-window tile handling
        self.dyn_tiles = bool(is_local or doc_varlen)
        # nvfp4_s (dk8 configuration): S^T = \hat{K} \hat{Q}^T from row-wise FP4 Q / K, block-scaled kind::mxf4nvf4 cta_group::2 with the
        # scale factors S2T-parked in the dP bank; dP / dV / dK / dQ stay per-tensor E4M3. s_sf_vec_size 16 = NVFP4 (E4M3 SFs
        # per 16, scale_vec::4X), 32 = MXFP4 (UE8M0 SFs per 32, scale_vec::2X).
        self.nvfp4_s = bool(nvfp4_s)
        # qk_hadamard: set by the driver to the 128 signs (compile-time) when Q / K are rotated; the dK epilogue then
        # stores dK = (dK R) R^T. Keyed by the driver.
        self.had_dk = None
        assert s_sf_vec_size in (16, 32)
        self.sf_dtype = cutlass.Float8E4M3FN if s_sf_vec_size == 16 else cutlass.Float8E8M0FNU
        self.sf_vec_size = s_sf_vec_size
        # Block-scaled d\hat{S} (Bwd2Spec.ds_scaling_fp8 = blk32, ported from the 1-CTA kernel): the dK and dQ
        # GEMMs run on kind::mxf8f6f4 with one UE8M0 exponent per (32 keys x 32 queries) block, computed by the compute
        # warps (in-thread abs-max + one redux.sync per stage), one quantisation for both GEMMs. SF tiles in smem:
        # dK side 128 own keys x 4 query groups; dQ side this CTA's 64 queries x 8 key groups of the cluster -- the
        # peer's 4 key groups arrive by DSMEM together with its dS half (xchg warp). The MMA warp S2Ts the tiles
        # into TMEM parks before each GEMM; the per-tensor B operands (Q^T, K^T) get constant 2^0 SF tiles.
        self.ds_scaling = spec.ds_scaling_fp8
        # `2cta-blockscaled-ds-missing-fp8-guard`: block-scaled dS is an FP8-operand feature. It parks its SF
        # tiles in the dP bank at the byte offsets nvfp4_s already uses for SFK / SFQ (tmem_SFK_offset 288 and
        # tmem_SFQ_offset 296 against tmem_SFdS_mx_offset 288 and tmem_SFQc_offset 296), which is what the assert
        # below protects. Deriving mxds from ds_scaling alone meant `block32-ds-scale`'s default turned that assert
        # into a trace-time failure on the SHIPPED dk8 configuration, so `cluster-2cta-backward` was dead at every
        # length the host chooses it (s_k >= 16384, or dense s_k >= 8192) -- the whole +22 % causal / +15 % dense
        # that route is worth. The 1-CTA kernel has always had the guard (flash_bwd_sm100_fp4.py:
        # `self.mxds = self.fp8 and ...`); this kernel has no `fp8` attribute, so `not nvfp4_s` states the same
        # condition. Written as a conjunct rather than an enumeration of ds_scaling values, so adding a new
        # granularity cannot reintroduce it.
        self.mxds = (not self.nvfp4_s) and self.ds_scaling != "per_tensor"
        if self.mxds:
            assert self.ds_scaling in ("blk32", "blk32x64", "tile128"), "2-CTA block-scaled dS: blk32 / blk32x64 / tile128"
            assert not self.nvfp4_s, "block-scaled dS shares the dP-bank SF parks with nvfp4_s"
        # Bwd2Spec.ds_scaling_fp4 = blk32 (dk8, token-block scales): the same blk32 rung on the nvfp4_s route. Its dK-side parks
        # move to [dP+48, dP+64) (see the TMEM map below), clear of SFK / SFQ at [dP+32, dP+48).
        self.ds_scaling_fp4 = spec.ds_scaling_fp4
        assert self.ds_scaling_fp4 in ("per_tensor", "blk32"), self.ds_scaling_fp4
        self.mxds_fp4 = bool(self.nvfp4_s and self.ds_scaling_fp4 == "blk32")
        if self.mxds_fp4:
            self.ds_scaling = "blk32"
            self.mxds = True
        self.sf_dtype_mx = cutlass.Float8E8M0FNU
        self.sf_vec_mx = 32
        # Block-scaled \hat{P} (Bwd2Spec.p_scaling_fp8 = mx32): one UE8M0 exponent per (key row, 32 queries) on the
        # E4M3 \hat{P} the dV GEMM reads from TMEM, dV on mxf8f6f4 with a constant SF tile for the per-tensor dO
        self.p_scaling = spec.p_scaling_fp8
        assert self.p_scaling in ("row", "mx32"), self.p_scaling
        self.mxp = self.p_scaling == "mx32" or self.token_local_arg
        # token-block E4M3 operands: Bwd2Spec.op_scaling = blk32x128 on the FP8 route and on the dk8 route. The 32 x 32 variant (blk32) is 1-CTA only: its dP SFs would need TMEM columns
        # that are all live when dP(t+1) issues here.
        # SCALING-CONSISTENCY RULE: fwd and bwd share operand scaling (the forward's codes are the backward's codes)
        _op = spec.op_scaling  # (the route's E4M3 operand scaling: the 4-bit route's under nvfp4_s, else the 8-bit route's)
        if self.doc_varlen:
            _op = "blk32x128"  # varlen: token-block operands (a per-tensor scale is a channel between documents)
        assert _op in ("per_tensor", "blk32x128"), f"2-CTA backward: operand scaling {_op!r} (blk32 is 1-CTA only)"
        self.mxop_tb = _op == "blk32x128"
        if self.mxop_tb:
            assert self.mxds, "blk32x128 needs the block-scaled dS (the dK / dQ GEMMs are then mxf8f6f4 already)"
        self.mxdv = self.mxp or self.mxop_tb  # dV on mxf8f6f4 (P^ SF computed under mxp, constant 2^0 otherwise)
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        assert self.tile_hdim <= 128 or (self.tile_hdim == 192 and self.tile_hdimv == 128)
        assert self.tile_hdimv <= 128

        # masks / varlen: the mods run on the cluster as in upstream flash_bwd_sm100 (which has no mod restriction on
        # use_2cta_instrs): index tensors over the 256-key cluster tile, each CTA its half
        self.use_2cta_instrs = bool(use_2cta_instrs and cluster_size == 2)
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        assert self.tile_hdim != 192 or self.use_2cta_instrs, "Must use 2CTA for hdim 192"

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        # 2-CTA: reduction dim is cluster-wide (tile_n * cta_group_size).
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * self.cta_group_size)

        self.acc_dtype = Float32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.deterministic = deterministic
        self.spt_override = spt

        # Score mod and mask mod support
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        # For score_mod, use vec_size=1 (like forward) to handle per-element indices
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        # Test-only (tests/cute/test_2cta_dp_park_race.py): skew the peer CTA's WG 1 dP loads to provoke the SF-park race
        # (fixed: the S2T of the next SFK / SFQ into the peer's park waits for the peer's dP_loaded pair barrier).
        # Private and compile-time; the driver never sets it.
        self._test_skew_peer = bool(_test_skew_peer)
        # per-warp hand-offs in the compute warps (mbarriers instead of the 256-thread named barriers): -10 % step
        self.warp_sync = spec.warp_sync
        # Two-pass dS quantisation: pass 0 does both stages' dS arithmetic and in-thread
        # abs-max, the two redux go back to back, pass 1 multiplies / packs / stores both stages; both dP stages are held
        # in registers (dp_hoist). The coarser rungs blk32x64 (one exponent per warp tile = both stages) and tile128 (one
        # per WG x stage, the 4 warps' redux max'd through smem) are built on the two-pass order. (Other measured levers --
        # two-pass blk32, shuffle / hoist / f16 P / redux / SF-select / u16 variants -- did not pay and are not shipped.)
        self.mx_both = self.mxds and self.ds_scaling in ("blk32x64", "tile128")
        self.dp_hoist = self.mx_both
        self.mx_tile128 = self.mxds and self.ds_scaling == "tile128"
        # MX_XW (dq | all): the SF tiles are assembled by the exchange warp from a 16-byte scratch that each compute warp
        # writes one byte per stage into (its exponent is warp-uniform), instead of 32-lane byte stores at a 16 B stride
        # (4-way bank conflicts, 4 LSU wavefronts each, on the dS chain). dq = the dQ-side tiles (own K-atom + the
        # exchange atom); all = the dK-side tile too, whose 32 rows are identical 16-byte permutations of the scratch --
        # the exchange warp then arrives on sf_ready, which the MMA warp waits on before the dK S2T.
        # default "all": the exchange warp assembles both SF tiles off the compute warps, bitwise
        # blk32, -1..-3.8 % of the kernel; "" = compute warps store the SF bytes themselves, "dq" = dQ side only
        self.mx_xw = spec.mx_xw if spec.mx_xw is not None else Bwd2Spec.rule_mx_xw(self.ds_scaling)
        assert self.mx_xw in ("", "dq", "all"), self.mx_xw
        self.mx_xw = self.mx_xw if self.mxds else ""
        self.mx_xw_all = self.mx_xw == "all"
        self.mx_xw_any = bool(self.mx_xw)  # the DSL's `if const_expr(...)` wants a bool, not the mode string
        assert not self.mx_xw or self.warp_sync, "MX_XW needs WARP_SYNC=1"
        # L2-locality CTA swizzle. The LPT scheduler is a pure permutation of work tiles (one tile per
        # CTA either way); determinism needs its `spt` reverse-order flag, not the swizzle, so the
        # swizzle can be had without it. Every backward in the tree gates it on deterministic=True,
        # which interface_bwd_2cta8 passes as False -- so it had never been active on our path.
        # Worth it only while the swizzle it computes exceeds 1; see _lpt_l2_worth_it().
        # KernelTuning.two_cta8_lpt_l2 forces either way for A/B, and is applied by the DRIVER
        # (`interface_bwd_2cta8._resolve_lpt_l2`), not here: `lpt-l2-override-unkeyed` was exactly this
        # override running AFTER the compile key had been built from the pre-override value, so the two
        # A/B builds silently shared one cubin. Rule: resolve an override where the key is built and pass the
        # constructor one value.
        self.lpt_l2 = bool(lpt_l2)
        # Generally slower to use store dS in smem for dK, and doesn't work for 2cta
        self.use_smem_dS_for_mma_dK = False

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        # self.epilogue_sync_barrier = pipeline.NamedBarrier(
        #     barrier_id=2,
        #     num_threads=self.num_compute_warps * self.threads_per_warp,
        # )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        # self.tmem_dK_offset = 0
        # self.tmem_dV_offset = self.tmem_dK_offset + self.tile_hdim
        # self.tmem_dQ_offset = self.tmem_dV_offset + self.tile_hdimv
        # self.tmem_dP_offset = self.tmem_dQ_offset  # overlap with dQ
        # self.tmem_S_offset = self.tmem_dQ_offset + max(self.tile_m, self.tile_hdim)
        # self.tmem_P_offset = self.tmem_S_offset  # overlap with S
        # self.tmem_total = self.tmem_S_offset + self.tile_n
        # assert self.tmem_total <= self.tmem_alloc_cols

        if self.use_2cta_instrs and self.tile_hdim == 192 and self.tile_hdimv == 128:
            assert self.tile_m == 128
            assert self.tile_n == 128
            self.tmem_dV_offset = 0
            self.tmem_dK_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_S_offset = self.tmem_dK_offset + self.tile_hdim
            self.tmem_P_offset = self.tmem_S_offset  # overlap with S
            self.tmem_dP_offset = 512 - self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
            self.tmem_dQ_offset = 512 - self.tile_hdim // 2
        else:
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # overlap with S
            self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
            self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_dQ_offset = (
                (self.tmem_S_offset + (self.tile_hdim // 2))
                if self.use_2cta_instrs
                else self.tmem_dP_offset
            )
            self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP
            # nvfp4_s: SFK / SFQ (8 columns each) in the dP bank's WG-1 stage-0 load block [288, 320): after d\hat{S} [256, 288)
            self.tmem_SFK_offset = self.tmem_dP_offset + 32
            self.tmem_SFQ_offset = self.tmem_SFK_offset + 8
            # block-scaled d\hat{S} parks: dK side in the dP bank after d\hat{S} (S2T'd once dP(t) is loaded, read by dK(t),
            # overwritten by dP(t+1)); dQ side in the S bank [32, 64) (\hat{P} at [0, 32), the dQ accumulator at [64, 128);
            # free once S(t+1) is loaded, which pipeline_dS already implies when dQ(t) is issued)
            self.tmem_SFdS_mx_offset = self.tmem_dS_offset + (48 if self.mxds_fp4 else 32)
            self.tmem_SFQc_offset = self.tmem_SFdS_mx_offset + 8
            self.tmem_SFdSq_offset = self.tmem_S_offset + 32
            self.tmem_SFKc_offset = self.tmem_SFdSq_offset + 8
            # block-scaled \hat{P} parks: S bank [48, 64) (the dQ-side parks [32, 48) are consumed by dQ(t) before dV(t+1)
            # in the tensor pipe's order; the dQ accumulator starts at 64)
            self.tmem_SFP_mx_offset = self.tmem_S_offset + 48
            self.tmem_SFdOc_offset = self.tmem_SFP_mx_offset + 8

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
            self.num_regs_compute = 136
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        else:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 136
            self.num_regs_compute = 136 if self.use_2cta_instrs else 144
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        self.num_regs_empty = 24
        if self.mx_xw_all:
            self.num_regs_empty = 64  # the exchange warp assembles the dK-side tile: 12 x 136 + 3 x 104 + 64 = 2008 <= 2048
        # Bwd2Spec.regs = (reduce, compute, load) (load = mma): per-warp register budgets (reduce + 2 compute + load <= 512)
        if spec.regs is not None:
            r_, c_, l_ = spec.regs
            self.num_regs_reduce, self.num_regs_compute, self.num_regs_load, self.num_regs_mma = r_, c_, l_, l_
        self.causal_diag_only = spec.causal_diag
        # Bwd2Spec.ld_rep = 64: one contiguous 64-column S / dP load per compute WG (WG w = query half w), x16 \hat{P} / d\hat{S} stores
        self.ld_rep = spec.ld_rep
        assert self.ld_rep in (32, 64)
        # `ld-rep-64-sf-index-swap`. The block-scaled SF byte stores hardcode the x32 mapping from
        # (warpgroup, stage) to query block as `2*stage + wg_idx` -- three sites in this file, plus the exchange
        # warp's assembly, which reads the same convention. At ld_rep == 64 a warpgroup holds the whole contiguous
        # query half, so the true index is `2*wg_idx + stage` and query blocks 1 and 2 SWAP EXPONENTS: wrong dK and
        # dQ, no error and no hang. The author handled ld_rep == 64 for the dS SMEM tile and for sdS_own / sdS_xchg
        # but not for the SF bytes. Guarded rather than re-indexed on purpose: re-indexing needs the same change at
        # three store sites and inside the exchange warp's packing, with no gate that would catch getting it wrong,
        # and `x32-tmem-loads` is a tuning lever rather than a shipped default -- so the combination is refused
        # until someone does the re-index with a bitwise gate. Mirrors the 1-CTA kernel's guard
        # (flash_bwd_sm100_fp4.py: `if self.mxds: assert self.tmem_ld_rep == 32`).
        if self.mxds or self.mxp:
            assert self.ld_rep == 32, (
                "Bwd2Spec.ld_rep=64 with a block-scaled dS (Bwd2Spec.ds_scaling_fp8 != per_tensor) swaps the SF "
                "exponents of query blocks 1 and 2: the SF byte index is hardcoded 2*stage + wg_idx, which is the "
                "x32 mapping. Use ld_rep=32, or a per-tensor dS."
            )
        # row-local dQ-side dS exponents: the blk32 rung only (blk32x64 / tile128 are coarser rungs of the same tile);
        # "diag" needs a causal diagonal to cross (refused elsewhere rather than compiled to a silent no-op)
        assert ds_dq_rowlocal in ("off", "diagonal", "all"), ds_dq_rowlocal
        if ds_dq_rowlocal != "off":
            assert self.mxds and self.ds_scaling == "blk32" and self.ld_rep == 32, (
                f"ds_dq_rowlocal={ds_dq_rowlocal!r} refines the blk32 dS^ (ds_scaling={self.ds_scaling!r}, mxds={self.mxds})")
        if ds_dq_rowlocal == "diagonal":
            assert is_causal and not is_local and mask_mod is None, "ds_dq_rowlocal='diagonal': causal (no window / mask_mod)"
        self.ds_dq_rowlocal = ds_dq_rowlocal
        # sMxSF bytes (resolved here: the struct body is traced, keep its sizes plain values)
        self.mxsf_bytes = (32 + 1024 + 2048) if ds_dq_rowlocal != "off" else (32 if self.mx_xw_any else 16)
        if self.token_local_arg:
            self.mxsf_bytes += 2 * 2 * 8 * 32 * 4   # [3104, 7200): token-local per-query factors (see compute_loop)

        if const_expr(self.tile_hdim == 192):
            if not is_causal and not is_local:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load
            else:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        # Bwd2Spec.q_stage (2-CTA): Q / Q^T / LSE ring depth; 1 is the only value built:
        # +16 KB sQ + 16 KB sQt + 0.5 KB sLSE per stage on the 147 KB footprint
        self.Q_stage = self.spec.q_stage if self.use_2cta_instrs else 2
        assert self.Q_stage in (1, 2)
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # todo: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.use_2cta_instrs and self.tile_hdim == 192:
            self.dQ_reduce_ncol_t2r = 32
            self.dQ_reduce_ncol = 24 if not self.is_causal else 32
            self.sdQaccum_stage = 2 if not self.is_causal else 1
        else:
            if self.use_2cta_instrs:
                # Bwd2Spec.dq_ncol / dq_stages: reduce chunk columns (of the CTA's 64) and TMA commit depth
                self.dQ_reduce_ncol = (self.spec.dq_ncol if self.spec.dq_ncol is not None
                                       else Bwd2Spec.rule_dq_ncol(self.deterministic))
                self.sdQaccum_stage = (self.spec.dq_stages if self.spec.dq_stages is not None
                                       else Bwd2Spec.rule_dq_stages(self.deterministic))
                self.dQ_reduce_ncol_t2r = 32
            else:
                self.dQ_reduce_ncol = 32
                self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
                self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue (must divide hdim_per_wg)
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        if const_expr(self.nvfp4_s):
            tiled_mma_S = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float4E2M1FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_kq[:2],
            )
            # SFB helper (per SM, 128-row padded N) for the multicast TMA of the Q scale factors
            self.mma_tiler_sfb = (self.tile_n, cute.round_up(self.tile_m, 128), self.tile_hdim)
            self.tiled_mma_sfb_S = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float4E2M1FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.mma_tiler_sfb[:2],
            )
        else:
            tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_kq[:2],
            )
            self.tiled_mma_sfb_S = None
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        if const_expr(self.mxdv):
            tiled_mma_dV = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
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
                self.do_dtype,
                tcgen05.OperandMajorMode.K,  # P_major_mode
                tcgen05.OperandMajorMode.MN,  # dO_major_mode
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_pdo[:2],
                a_source=tcgen05.OperandSource.TMEM,
            )
        # dK += dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK):
            mma_dK_a_src = tcgen05.OperandSource.SMEM
        else:
            mma_dK_a_src = tcgen05.OperandSource.TMEM
        if const_expr(self.mxds):
            # block-scaled d\hat{S}: E4M3 d\hat{S} (TMEM) with UE8M0 SFs x the E4M3 Q^T tile MN-major with constant SFs
            tiled_mma_dK = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
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
                self.do_dtype,
                tcgen05.OperandMajorMode.K,  # dS_major_mode
                tcgen05.OperandMajorMode.MN,  # Q_major_mode
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dsq[:2],
                a_source=mma_dK_a_src,
            )
        # dQ = dS @ K
        if const_expr(self.mxds):
            tiled_mma_dQ = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
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
                tcgen05.OperandMajorMode.MN,  # dS_major_mode
                tcgen05.OperandMajorMode.MN,  # Kt_major_mode
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dsk[:2],
            )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        self.s_op_dtype = cutlass.Float4E2M1FN if const_expr(self.nvfp4_s) else self.q_dtype
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.s_op_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.s_op_dtype,
            self.Q_stage,
        )
        self.sfk_layout = self.sfq_layout = None
        if const_expr(self.nvfp4_s):
            inst_k = self.tile_hdim // 64  # K64 block-scaled instructions per K-tile
            self.sfk_layout = fp4_helpers.make_smem_layout_sfa(
                self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, 1, mma_tile_inst_k=inst_k
            )
            self.sfq_layout = fp4_helpers.make_smem_layout_sfb(
                self.tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, self.Q_stage, mma_tile_inst_k=inst_k
            )
        # dP.T = V @ dO.T
        self.dp_op_dtype = self.do_dtype
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.dp_op_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))
        # block-scaled d\hat{S}: SF tiles. dK: A = d\hat{S} (128 own keys x 4 query groups), B = Q^T (128 d x 4); dQ: A = d\hat{S}^T
        # (64 own queries x 8 cluster key groups, two 512 B K-atoms), B = K^T (128 d x 8)
        self.sfds_mx_layout = self.sfqc_layout = self.sfdsq_layout = self.sfkc_layout = None
        if const_expr(self.mxds):
            kq = self.tile_m // 32
            kk = self.tile_n * self.cta_group_size // 32
            self.sfds_mx_layout = fp4_helpers.make_smem_layout_sfa(self.tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, 1, mma_tile_inst_k=kq)
            self.sfqc_layout = fp4_helpers.make_smem_layout_sfb(self.tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, 1, mma_tile_inst_k=kq)
            # dQ side: the CTA's 64 query rows are rows 0-63 of a 128-row SF atom (row r at lane r % 32, column r // 32;
            # the DSL's cta_group::2 / M = 64 layouts split the atom in a way the S2T partition rejects), so the
            # tiles are built from a 1-CTA helper MMA over (128 queries, d, 256 cluster keys): 1024 B each, rows
            # 64-127 unused, K-atom 1 at +512 B
            self.mma_tiler_dsk_sf = (self.tile_m, self.tile_hdim, self.tile_n * self.cta_group_size)
            self.tiled_mma_dq_sf = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN,
                self.sf_dtype_mx, self.sf_vec_mx, tcgen05.CtaGroup.ONE, self.mma_tiler_dsk_sf[:2],
            )
            self.sfdsq_layout = fp4_helpers.make_smem_layout_sfa(self.tiled_mma_dq_sf, self.mma_tiler_dsk_sf, self.sf_vec_mx, 1, mma_tile_inst_k=kk)
            self.sfkc_layout = fp4_helpers.make_smem_layout_sfb(self.tiled_mma_dq_sf, self.mma_tiler_dsk_sf, self.sf_vec_mx, 1, mma_tile_inst_k=kk)
        self.sfp_mx_layout = self.sfdoc_layout = None
        if const_expr(self.mxdv):
            kq = self.tile_m // 32
            self.sfp_mx_layout = fp4_helpers.make_smem_layout_sfa(self.tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, 1, mma_tile_inst_k=kq)
            self.sfdoc_layout = fp4_helpers.make_smem_layout_sfb(self.tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, 1, mma_tile_inst_k=kq)

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        if const_expr(not self.dKV_postprocess):
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                2,  # num compute wgs
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                2,  # num compute wgs
            )
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        ds_boost: Float32,  # b: dS' = P' (b d\hat{P} + D'')
        dv_scale: Float32,  # dV = dv_scale * acc
        dk_scale: Float32,  # dK = dk_scale * acc
        dq_scale: Float32,  # f16 dQ accumulator: partials * dq_scale before the f16 reduce-add
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # nvfp4_s: NVFP4 row-wise Q / K (pointers + (b, s, h, d) shapes) and their swizzled E4M3 scale factors
        mQ4=None,
        mK4=None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        q4_shape: tuple = (),
        k4_shape: tuple = (),
        # blk32x128: the d-rows SF atoms of Q (dK B), K (dQ B), dO (dV B) and the (4, b, h, s / 32) fp32 token-block factors
        mSFQd: Optional[cute.Tensor] = None,
        mSFKd: Optional[cute.Tensor] = None,
        mSFdOd: Optional[cute.Tensor] = None,
        mFac: Optional[cute.Tensor] = None,
        # token-local operands: (4, b, h, s) fp32 per-token factors 2^(e_t - 127) of the token-local
        # E4M3 copies (rows Q, K, V, dO; Q unused on the dk8 route). None = token blocks (bitwise today).
        mTokFac: Optional[cute.Tensor] = None,
        # GPU-only scales (device_scales): float32 [softmax_scale, ds_boost, dv_scale, dk_scale, dq_scale] in device
        # memory. When given, the kernel reads these instead of the five scalar arguments (which are then ignored), so a
        # data-dependent scale never has to be read back to the host.
        mScalars: Optional[cute.Tensor] = None,
        # masks / varlen: aligned-packed varlen tables (lp_bwd_tiles; key-pair tiles of 256)
        mDocTab: Optional[cute.Tensor] = None,
        mNDoc: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        assert (mDocTab is not None) == self.doc_varlen and (mNDoc is not None) == self.doc_varlen
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype
        self.fp8 = self.q_dtype is cutlass.Float8E4M3FN
        assert self.fp8 and self.use_2cta_instrs and self.tile_hdim == 128
        # GQA (bwdk, opt-in KernelTuning.bwd_2cta_gqa): upstream's q-head form -- each CTA pair computes one q head's dK / dV
        # on its kv head's K / V and reduce-adds them in fp32 into (b, h_k, s_k * d) accumulators (dKV_postprocess)
        assert self.qhead_per_kvhead == 1 or not self.token_local_arg, "2-CTA E4M3 backward GQA: not token-local"

        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        self.use_tma_store = not (self.qhead_per_kvhead == 1 and mCuSeqlensK is not None)
        # self.use_tma_store = not self.qhead_per_kvhead == 1
        self.dKV_postprocess = self.qhead_per_kvhead > 1

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]
        if const_expr(self.nvfp4_s):
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            rev4 = (3, 2, 1, 0)
            mQ4 = layout_utils.select(cute.make_tensor(mQ4, cute.make_ordered_layout(q4_shape, order=rev4)), mode=[1, 3, 2, 0])
            mK4 = layout_utils.select(cute.make_tensor(mK4, cute.make_ordered_layout(k4_shape, order=rev4)), mode=[1, 3, 2, 0])
            sf_chunk = blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout
            mSFQ = cute.make_tensor(cute.recast_ptr(mSFQ.iterator, dtype=self.sf_dtype), cute.tile_to_shape(sf_chunk, mQ4.shape, (2, 1, 3, 4)))
            mSFK = cute.make_tensor(cute.recast_ptr(mSFK.iterator, dtype=self.sf_dtype), cute.tile_to_shape(sf_chunk, mK4.shape, (2, 1, 3, 4)))

        if const_expr(self.mxop_tb):
            assert mSFQd is not None and mSFKd is not None and mSFdOd is not None and mFac is not None
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            # the d-rows replications as (d, s, h, b) operands' SF tensors (rows = d, K = tokens), UE8M0 per 32 tokens
            _sfc32 = blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_mx).layout
            mSFQd = cute.make_tensor(cute.recast_ptr(mSFQd.iterator, dtype=self.sf_dtype_mx),
                                     cute.tile_to_shape(_sfc32, (mQ.shape[1], mQ.shape[0], mQ.shape[2], mQ.shape[3]), (2, 1, 3, 4)))
            mSFdOd = cute.make_tensor(cute.recast_ptr(mSFdOd.iterator, dtype=self.sf_dtype_mx),
                                      cute.tile_to_shape(_sfc32, (mQ.shape[1], mQ.shape[0], mQ.shape[2], mQ.shape[3]), (2, 1, 3, 4)))
            mSFKd = cute.make_tensor(cute.recast_ptr(mSFKd.iterator, dtype=self.sf_dtype_mx),
                                     cute.tile_to_shape(_sfc32, (mK.shape[1], mK.shape[0], mK.shape[2], mK.shape[3]), (2, 1, 3, 4)))
        self.token_local = const_expr(self.token_local_arg)
        assert (mTokFac is not None) == self.token_local, "token_local_operands <-> mTokFac"
        if const_expr(self.token_local):
            assert self.mxop_tb and self.mxp and self.mxds \
                and self.ds_dq_rowlocal == "all", "token_local_operands: blk32x128, block-scaled P^, ds_dq_rowlocal='all'"

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK4 if const_expr(self.nvfp4_s) else mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ4 if const_expr(self.nvfp4_s) else mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_SFK = tma_tensor_SFK = tma_atom_SFQ = tma_tensor_SFQ = None
        if const_expr(self.nvfp4_s):
            sfa_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op, mSFK, cute.select(self.sfk_layout, mode=[0, 1, 2]), self.mma_tiler_kq, self.tiled_mma_S,
                self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
            )
            sfb_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_S.thr_id)
            self.cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_sfb_S.thr_id.shape,)
            )
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, mSFQ, cute.select(self.sfq_layout, mode=[0, 1, 2]), self.mma_tiler_sfb, self.tiled_mma_sfb_S,
                self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
            )
        tma_atom_SFQd = tma_tensor_SFQd = tma_atom_SFdOd = tma_tensor_SFdOd = tma_atom_SFKd = tma_tensor_SFKd = None
        self.tiled_mma_sfb_dim = None
        if const_expr(self.mxop_tb):
            # helper (1-CTA, N = d = 128 padded) whose B partition moves one (128 d x 4 token groups) SF atom per tile
            self.mma_tiler_sfb_dim = (self.tile_n, 128, self.tile_m)
            self.tiled_mma_sfb_dim = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                self.sf_dtype_mx, self.sf_vec_mx, tcgen05.CtaGroup.ONE, self.mma_tiler_sfb_dim[:2],
            )
            self.cluster_layout_sfb_dim = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_sfb_dim.thr_id.shape,)
            )
            _sfb_dk = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_dK.thr_id)
            _sfb_dv = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_dV.thr_id)
            _sfb_dq = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma_dQ.thr_id)
            tma_atom_SFQd, tma_tensor_SFQd = cute.nvgpu.make_tiled_tma_atom_B(
                _sfb_dk, mSFQd, cute.select(self.sfqc_layout, mode=[0, 1, 2]), self.mma_tiler_sfb_dim, self.tiled_mma_sfb_dim,
                self.cluster_layout_sfb_dim.shape, internal_type=cutlass.Int16,
            )
            tma_atom_SFdOd, tma_tensor_SFdOd = cute.nvgpu.make_tiled_tma_atom_B(
                _sfb_dv, mSFdOd, cute.select(self.sfdoc_layout, mode=[0, 1, 2]), self.mma_tiler_sfb_dim, self.tiled_mma_sfb_dim,
                self.cluster_layout_sfb_dim.shape, internal_type=cutlass.Int16,
            )
            self.cluster_layout_sfb_dq = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_dq_sf.thr_id.shape,)
            )
            tma_atom_SFKd, tma_tensor_SFKd = cute.nvgpu.make_tiled_tma_atom_B(
                _sfb_dq, mSFKd, cute.select(self.sfkc_layout, mode=[0, 1, 2]), self.mma_tiler_dsk_sf, self.tiled_mma_dq_sf,
                self.cluster_layout_sfb_dq.shape, internal_type=cutlass.Int16,
            )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
            )
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                layout_utils.select(mK, mode=transpose_sh_k),
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["Q"] = self.cta_group_size * cute.size_in_bytes(self.s_op_dtype, cute.select(self.sQ_layout, mode=[0, 1, 2]))
        self.tma_copy_bytes["K"] = self.cta_group_size * cute.size_in_bytes(self.s_op_dtype, cute.select(self.sK_layout, mode=[0, 1, 2]))
        # the transposed E4M3 tiles of dK (Q^T) / dQ (K^T) ride their own pipelines with these counts
        self.tma_copy_bytes["Qt"] = self.cta_group_size * cute.size_in_bytes(self.q_dtype, cute.select(self.sQt_layout, mode=[0, 1, 2]))
        self.tma_copy_bytes["Kt"] = self.cta_group_size * cute.size_in_bytes(self.k_dtype, cute.select(self.sKt_layout, mode=[0, 1, 2]))
        if const_expr(self.nvfp4_s):
            self.tma_copy_bytes["Q"] += self.cta_group_size * cute.size_in_bytes(self.sf_dtype, cute.select(self.sfq_layout, mode=[0, 1, 2]))
            self.tma_copy_bytes["K"] += self.cta_group_size * cute.size_in_bytes(self.sf_dtype, cute.select(self.sfk_layout, mode=[0, 1, 2]))
        # the dP B tile (dO^T) rides the dO pipeline as extra tx; E4M3 it is the dO tile's size
        self.tma_copy_bytes["dOt"] = self.tma_copy_bytes["dO"]
        # blk32x128: the three multicast SF tiles, counted on their operand's pipeline (both CTAs' bytes on the leader)
        self.tma_copy_bytes["Qt_sf"] = self.cta_group_size * cute.cosize(self.sfqc_layout) if const_expr(self.mxop_tb) else 0
        self.tma_copy_bytes["dO_sf"] = self.cta_group_size * cute.cosize(self.sfdoc_layout) if const_expr(self.mxop_tb) else 0
        self.tma_copy_bytes["Kt_sf"] = self.cta_group_size * cute.cosize(self.sfkc_layout) if const_expr(self.mxop_tb) else 0
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * self.dqaccum_dtype.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        # TileScheduler = SingleTileScheduler
        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic or self.lpt_l2):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = (self.is_causal or self.is_local) and self.deterministic
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,  # pack_gqa disabled for bwd
            # clamp: k_dtype is 4-bit on the FP4 path, so width // 8 == 0 would zero size_one_head
            # and break the scheduler's L2 swizzle sizing (same fix as flash_fwd_sm100_fp4.py:1481).
            element_size=max(self.k_dtype.width // 8, 1),
            is_persistent=self.is_persistent,  # persistent mode not tested
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.s_op_dtype, self.sK_layout)
        sQ_bytes = cute.size_in_bytes(self.s_op_dtype, self.sQ_layout)
        sfk_size = cute.cosize(self.sfk_layout) if const_expr(self.nvfp4_s) else 1
        sfq_size = cute.cosize(self.sfq_layout) if const_expr(self.nvfp4_s) else 1
        # block-scaled d\hat{S}: the four SF tiles plus the staging tile of the dQ-side bytes sent to the peer
        sfmx = [cute.cosize(l) if self.mxds else 1 for l in (self.sfds_mx_layout, self.sfqc_layout, self.sfdsq_layout, self.sfkc_layout)]
        sfp_sz = [cute.cosize(l) if self.mxdv else 1 for l in (self.sfp_mx_layout, self.sfdoc_layout)]
        # E4M3 V / K tiles are smaller than the bf16 dV / dK epilogue tiles that reuse their buffers
        sV_alloc_bytes = max(sV_bytes, sdV_bytes)
        sK_alloc_bytes = max(sK_bytes, sdK_bytes)

        if const_expr(self.use_2cta_instrs):
            sQt_size = cute.cosize(self.sQt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdS_xchg_size = (
                cute.cosize(self.sdS_xchg_layout) if const_expr(self.tile_hdim <= 128) else 0
            )

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64
                # per-warp hand-offs of the compute warps (warp_sync): S / dP loaded, d\hat{S} stored to smem
                S_loaded_mbar_ptr: cutlass.Int64
                dP_loaded_mbar_ptr: cutlass.Int64
                # nvfp4_s: the leader's MMA warp parks SFK / SFQ(t+1) in BOTH CTAs' dP banks (tcgen05.cp cta_group::2), so it
                # waits for both CTAs' compute warps (count 16, cluster-scope arrives on the leader's copy)
                dP_loaded_pair_mbar_ptr: cutlass.Int64
                dS_stored_mbar_ptr: cutlass.Int64
                sf_ready_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sK_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sSFQ: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, sfq_size],
                    self.buffer_align_bytes,
                ]
                sSFK: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, sfk_size],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sV_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sSFdS_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx[0]], self.buffer_align_bytes]
                sSFQc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx[1]], self.buffer_align_bytes]
                sSFdSq: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx[2]], self.buffer_align_bytes]
                sSFKc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx[3]], self.buffer_align_bytes]
                sSFq_xchg: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfmx[2]], self.buffer_align_bytes]
                sMxRed: cute.struct.Align[cute.struct.MemRange[Float32, 32 if self.mx_tile128 else 1], 16]
                # ds_dq_rowlocal: [32, 1056) = [parity 2][stage 2][lane 32][wg 2 x key group 4] per-query dQ-side bytes (the
                # exchange warp's MX_XW source), [1056, 3104) = the per-query multipliers, f32 [compute warp 8][stage 2][32]
                sMxSF: cute.struct.Align[cute.struct.MemRange[
                    cute.Uint8, self.mxsf_bytes], 16]
                sSFP_mx: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfp_sz[0]], self.buffer_align_bytes]
                sSFdOc: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sfp_sz[1]], self.buffer_align_bytes]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]

        else:

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        # masks / varlen: always the baked log2 scale; with a score_mod the compute warps derive the natural scale per stage
        # (token-block factors included) and switch the P exponent to LOG2_E after the mod
        softmax_scale_log2 = softmax_scale * LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(self.use_2cta_instrs):
            assert blocksparse_tensors is None, (
                "2-CTA mode does not support block sparsity. "
                "Please create kernel with use_2cta_instrs=False for block sparse attention."
            )
        # The static footprint (measured 147456 B; the 231424 / 232448 figures this file
        # inherited from flash_bwd_sm100.py are stale, E4M3 K / V are smaller than bf16). At 144 KB two CTAs would need 288 KB
        # against the SM's 233472, so this alone pins occupancy at one CTA per SM -- as does the unconditional 512-column TMEM
        # allocation below. Note that is NOT special to low precision: TMEM holds f32 accumulators, so its column count depends
        # on tile geometry only, and the bf16 kernel allocates the same 512. One CTA per SM is the tcgen05 design point (warp
        # specialisation hides latency inside the CTA), not a wall this fork hit. See notes 13.4 reading 6.
        if const_expr(self.use_block_sparsity or aux_tensors is not None):
            assert all(x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)), (
                "Variable sequence length is not supported yet for blocksparse or aux tensors in bwd"
            )

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            ds_boost,
            dv_scale,
            dk_scale,
            dq_scale,
            window_size_left,
            window_size_right,
            tile_sched_params,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
            tma_tensor_SFK,
            tma_tensor_SFQ,
            tma_atom_SFK,
            tma_atom_SFQ,
            self.sfk_layout,
            self.sfq_layout,
            self.tiled_mma_sfb_S,
            self.sfds_mx_layout,
            self.sfqc_layout,
            self.sfdsq_layout,
            self.sfkc_layout,
            self.tiled_mma_dq_sf if self.mxds else None,
            self.sfp_mx_layout,
            self.sfdoc_layout,
            tma_tensor_SFQd, tma_tensor_SFdOd, tma_tensor_SFKd, tma_atom_SFQd, tma_atom_SFdOd, tma_atom_SFKd,
            self.tiled_mma_sfb_dim, mFac, mTokFac,
            mScalars, mDocTab, mNDoc,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        ds_boost: cutlass.Float32,
        dv_scale: cutlass.Float32,
        dk_scale: cutlass.Float32,
        dq_scale: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFQ: Optional[cute.Tensor] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        sfk_layout: Optional[cute.Layout] = None,
        sfq_layout: Optional[cute.Layout] = None,
        tiled_mma_sfb_S: Optional[cute.TiledMma] = None,
        sfds_mx_layout: Optional[cute.Layout] = None,
        sfqc_layout: Optional[cute.Layout] = None,
        sfdsq_layout: Optional[cute.Layout] = None,
        sfkc_layout: Optional[cute.Layout] = None,
        tiled_mma_dq_sf: Optional[cute.TiledMma] = None,
        sfp_mx_layout: Optional[cute.Layout] = None,
        sfdoc_layout: Optional[cute.Layout] = None,
        mSFQd: Optional[cute.Tensor] = None, mSFdOd: Optional[cute.Tensor] = None, mSFKd: Optional[cute.Tensor] = None,
        tma_atom_SFQd: Optional[cute.CopyAtom] = None, tma_atom_SFdOd: Optional[cute.CopyAtom] = None,
        tma_atom_SFKd: Optional[cute.CopyAtom] = None, tiled_mma_sfb_dim: Optional[cute.TiledMma] = None,
        mFac: Optional[cute.Tensor] = None,
        mTokFac: Optional[cute.Tensor] = None,
        mScalars: Optional[cute.Tensor] = None,
        mDocTab: Optional[cute.Tensor] = None,
        mNDoc: Optional[cute.Tensor] = None,
    ):
        if const_expr(mScalars is not None):
            # GPU-only scales: the same float32 values the host would have passed, read from device memory; the log2
            # scale is derived exactly as __call__ derives it
            softmax_scale = mScalars[0]
            softmax_scale_log2 = softmax_scale * math.log2(math.e)
            ds_boost = mScalars[1]
            dv_scale = mScalars[2]
            dk_scale = mScalars[3]
            dq_scale = mScalars[4]
            if const_expr(self.dq_tile_scaled):
                self._dq_tab = mScalars   # dqfix: the per-tile dQ scales after the vector
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                cpasync.prefetch_descriptor(tma_atom_V)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)
                cute.arch.mbarrier_init(storage.S_loaded_mbar_ptr, len(self.compute_warp_ids))
                cute.arch.mbarrier_init(storage.dP_loaded_mbar_ptr, len(self.compute_warp_ids))
                cute.arch.mbarrier_init(storage.dP_loaded_pair_mbar_ptr, 2 * len(self.compute_warp_ids))
                cute.arch.mbarrier_init(storage.dS_stored_mbar_ptr, len(self.compute_warp_ids))
                cute.arch.mbarrier_init(storage.sf_ready_mbar_ptr, 2)  # both CTAs' exchange warps arrive on the leader's

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Qt"] + self.tma_copy_bytes["Qt_sf"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["Kt"] + self.tma_copy_bytes["Kt_sf"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dO_sf"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.s_op_dtype)
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sQt = storage.sQt.get_tensor(
                sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
            )
        else:
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner, dtype=self.s_op_dtype)
        sSFK = sSFQ = tCtSFK = tCtSFQ = None
        if const_expr(self.nvfp4_s):
            sSFK = storage.sSFK.get_tensor(sfk_layout)
            sSFQ = storage.sSFQ.get_tensor(sfq_layout)
            # SFK / SFQ TMEM parks in the dP bank (free once every compute warp has loaded stage 0 of dP)
            tCtSFK = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, self.tmem_SFK_offset, mem_space=cute.AddressSpace.tmem, assumed_align=16), dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfa(tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, cute.slice_(sfk_layout, (None, None, None, 0))),
            )
            tCtSFQ = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, self.tmem_SFQ_offset, mem_space=cute.AddressSpace.tmem, assumed_align=16), dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfb(tiled_mma_S, self.mma_tiler_kq, self.sf_vec_size, cute.slice_(sfq_layout, (None, None, None, 0))),
            )
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=self.v_dtype)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim <= 128):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None
        # block-scaled d\hat{S}: typed SF tiles (S2T sources), flat byte views (compute-warp stores), TMEM parks
        sSFdS_mx = sSFQc = sSFdSq = sSFKc = sSFq_xchg = None
        sSFdS_mx_u8 = sSFdSq_u8 = sSFq_xchg_u8 = None
        sMxRed = storage.sMxRed.get_tensor(cute.make_layout(32)) if const_expr(self.mx_tile128) else None
        sMxSF = None
        if const_expr(self.ds_dq_rowlocal != "off"):
            sMxSF = storage.sMxSF.get_tensor(cute.make_layout(self.mxsf_bytes))
        elif const_expr(self.mx_xw_any):
            sMxSF = storage.sMxSF.get_tensor(cute.make_layout(32))
        tCtSFdS_mx = tCtSFQc = tCtSFdSq = tCtSFKc = None
        if const_expr(self.mxds):
            sSFdS_mx = storage.sSFdS_mx.get_tensor(sfds_mx_layout, dtype=self.sf_dtype_mx)
            sSFQc = storage.sSFQc.get_tensor(sfqc_layout, dtype=self.sf_dtype_mx)
            sSFdSq = storage.sSFdSq.get_tensor(sfdsq_layout, dtype=self.sf_dtype_mx)
            sSFKc = storage.sSFKc.get_tensor(sfkc_layout, dtype=self.sf_dtype_mx)
            sSFdS_mx_u8 = storage.sSFdS_mx.get_tensor(cute.make_layout(cute.cosize(sfds_mx_layout)))
            sSFdSq_u8 = storage.sSFdSq.get_tensor(cute.make_layout(cute.cosize(sfdsq_layout)))
            sSFq_xchg = storage.sSFq_xchg.get_tensor(cute.make_layout(cute.cosize(sfdsq_layout)))
            sSFq_xchg_u8 = sSFq_xchg

            def _sf_tmem_mx(col, layout):
                return cute.make_tensor(
                    cute.recast_ptr(cute.make_ptr(Float32, col, mem_space=cute.AddressSpace.tmem, assumed_align=16), dtype=self.sf_dtype_mx),
                    layout,
                )
            tCtSFdS_mx = _sf_tmem_mx(self.tmem_SFdS_mx_offset, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, cute.slice_(sfds_mx_layout, (None, None, None, 0))))
            tCtSFQc = _sf_tmem_mx(self.tmem_SFQc_offset, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dK, self.mma_tiler_dsq, self.sf_vec_mx, cute.slice_(sfqc_layout, (None, None, None, 0))))
            tCtSFdSq = _sf_tmem_mx(self.tmem_SFdSq_offset, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dq_sf, self.mma_tiler_dsk_sf, self.sf_vec_mx, cute.slice_(sfdsq_layout, (None, None, None, 0))))
            tCtSFKc = _sf_tmem_mx(self.tmem_SFKc_offset, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dq_sf, self.mma_tiler_dsk_sf, self.sf_vec_mx, cute.slice_(sfkc_layout, (None, None, None, 0))))
        sSFP_mx = sSFdOc = sSFP_mx_u8 = tCtSFP_mx = tCtSFdOc = None
        if const_expr(self.mxdv):
            sSFP_mx = storage.sSFP_mx.get_tensor(sfp_mx_layout, dtype=self.sf_dtype_mx)
            sSFdOc = storage.sSFdOc.get_tensor(sfdoc_layout, dtype=self.sf_dtype_mx)
            sSFP_mx_u8 = storage.sSFP_mx.get_tensor(cute.make_layout(cute.cosize(sfp_mx_layout)))
            tCtSFP_mx = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, self.tmem_SFP_mx_offset, mem_space=cute.AddressSpace.tmem, assumed_align=16), dtype=self.sf_dtype_mx),
                blockscaled_utils.make_tmem_layout_sfa(tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, cute.slice_(sfp_mx_layout, (None, None, None, 0))))
            tCtSFdOc = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(Float32, self.tmem_SFdOc_offset, mem_space=cute.AddressSpace.tmem, assumed_align=16), dtype=self.sf_dtype_mx),
                blockscaled_utils.make_tmem_layout_sfb(tiled_mma_dV, self.mma_tiler_pdo, self.sf_vec_mx, cute.slice_(sfdoc_layout, (None, None, None, 0))))

        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sdOt = storage.sdOt.get_tensor(
                sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.dp_op_dtype
            )
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(
                    sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
                )
                sdK = storage.sK.get_tensor(
                    sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
                )
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)
        # masks / varlen: every role takes a key pair's query range from here (lp_bwd_tiles: batched, or the aligned-packed
        # documents at key-pair granularity 256); the roles keep SeqlenInfoCls(batch) for tensor addressing (b = 1 there)
        if const_expr(self.doc_t128):
            # documents packed at 128: the key pair's halves in their own documents (128-key-tile geometry)
            block_info = LPPairTileInfo(
                BlockInfo(self.tile_m, self.tile_n, self.is_causal, self.is_local, False, window_size_left,
                          window_size_right, qhead_per_kvhead_packgqa=1),
                mQ.shape[0], mK.shape[0], self.tile_m, mDocTab, mNDoc,
            )
        else:
            block_info = LPTileInfo(
                block_info, mQ.shape[0], mK.shape[0], self.tile_m, self.tile_n * self.cluster_shape_mnk[0],
                mSeqUsedK=mSeqUsedK, mDocTab=mDocTab, mNDoc=mNDoc, docs=mDocTab is not None,
            )

        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            # doc_t128: the mask runs on this CTA's 128-key tile (n_block = own local tile - rank: the tile's key
            # coordinates already carry rank * 128)
            self.tile_n * (1 if const_expr(self.doc_t128) else self.cta_group_size),
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)
            if const_expr(self.warp_sync):
                self.xchg_warp(
                    sdS, sdS_xchg, dS_cluster_full_mbar_ptr, storage.dS_stored_mbar_ptr,
                    block_info, SeqlenInfoCls, TileSchedulerCls,
                    sSFq_xchg=sSFq_xchg_u8, sSFdSq=sSFdSq_u8, sMxSF=sMxSF, sSFdS_mx=sSFdS_mx_u8,
                    sf_ready_mbar_ptr=storage.sf_ready_mbar_ptr,
                )

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(
                self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty
            )
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_empty_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    cluster_layout_vmnk,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
                mSFK=mSFK,
                mSFQ=mSFQ,
                sSFK=sSFK,
                sSFQ=sSFQ,
                tma_atom_SFK=tma_atom_SFK,
                tma_atom_SFQ=tma_atom_SFQ,
                tiled_mma_sfb_S=tiled_mma_sfb_S,
                tb=(mSFQd, mSFdOd, mSFKd, tma_atom_SFQd, tma_atom_SFdOd, tma_atom_SFKd, tiled_mma_sfb_dim, tiled_mma_dq_sf,
                    sSFQc, sSFdOc, sSFKc),
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdSt,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_full_mbar_ptr,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_leader_mbar_ptr,
                storage.sf_ready_mbar_ptr,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
                blocksparse_tensors,
                sSFK=sSFK,
                sSFQ=sSFQ,
                tCtSFK=tCtSFK,
                tCtSFQ=tCtSFQ,
                # the SFK / SFQ(t+1) parks write both CTAs' dP banks: wait for both CTAs' compute warps (the pair barrier)
                dP_loaded_mbar_ptr=storage.dP_loaded_pair_mbar_ptr,
                sSFdS_mx=sSFdS_mx, sSFQc=sSFQc, sSFdSq=sSFdSq, sSFKc=sSFKc,
                tCtSFdS_mx=tCtSFdS_mx, tCtSFQc=tCtSFQc, tCtSFdSq=tCtSFdSq, tCtSFKc=tCtSFKc,
                sSFP_mx=sSFP_mx, sSFdOc=sSFdOc, tCtSFP_mx=tCtSFP_mx, tCtSFdOc=tCtSFdOc,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            _compute = partial(
                self.compute_loop,
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                sdS_xchg,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                ds_boost,
                dv_scale,
                dk_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                aux_tensors,
                fastdiv_mods,
                blocksparse_tensors,
                S_loaded_mbar_ptr=storage.S_loaded_mbar_ptr,
                dP_loaded_mbar_ptr=storage.dP_loaded_mbar_ptr,
                dP_loaded_pair_mbar_ptr=storage.dP_loaded_pair_mbar_ptr,
                dS_stored_mbar_ptr=storage.dS_stored_mbar_ptr,
                sSFdS_mx_u8=sSFdS_mx_u8, sSFdSq_u8=sSFdSq_u8, sSFq_xchg_u8=sSFq_xchg_u8, sSFP_mx_u8=sSFP_mx_u8,
                sMxRed=sMxRed, sMxSF=sMxSF, mFac=mFac, mTokFac=mTokFac,
            )
            _compute()
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                blocksparse_tensors,
                dq_scale=dq_scale,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def _noop(self, **kwargs):
        pass

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

    @cute.jit
    def _mma_S_fp4(self, op, tCrA, tCrB, sA, sB, tScaleA, tScaleB, s2t_k, s2t_k_src, s2t_k_dst, s2t_q, s2t_q_src, s2t_q_dst, B_idx):
        r"""nvfp4_s: S2T of SFK / SFQ into their dP-bank parks, then the block-scaled S^T GEMM (A = \hat{K} SMEM, B = \hat{Q} SMEM)."""
        sm100_utils.tcgen05_after_thread_sync()
        cute.copy(s2t_k, s2t_k_src, s2t_k_dst)
        cute.copy(s2t_q, s2t_q_src, s2t_q_dst)
        sm100_utils.gemm_ptx_partial_fp4(
            op, self.tmem_S_offset, tCrA, tCrB[None, None, None, 0], sA=sA, sB=sB[None, None, None, 0],
            tScaleA=tScaleA, tScaleB=tScaleB, zero_init=True,
        )

    @cute.jit
    def _mma_dK_mx(self, op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB, s2t_a, s2t_a_src, s2t_a_dst,
                   s2t_b, s2t_b_src, s2t_b_dst, B_idx, zero_init):
        r"""block-scaled dK += d\hat{S}^T Q: S2T of the d\hat{S} SFs / constant Q SFs into their dP-bank parks (dP(t) loaded,
        which the dP-empty wait before this call guarantees), then the mxf8f6f4 GEMM (A = d\hat{S} TMEM, B = Q^T stage 0)."""
        sm100_utils.tcgen05_after_thread_sync()
        cute.copy(s2t_a, s2t_a_src, s2t_a_dst)
        cute.copy(s2t_b, s2t_b_src, s2t_b_dst)
        sm100_utils.gemm_ptx_partial_fp4(
            op, self.tmem_dK_offset, tCrA, tCrB_staged[None, None, None, 0], sA=None, sB=sB_staged[None, None, None, 0],
            tScaleA=tScaleA, tScaleB=tScaleB, zero_init=zero_init, tA_addr=self.tmem_dS_offset,
        )

    @cute.jit
    def _mma_dV_mx(self, op, tCrA, tCrB_staged, sB_staged, tScaleA, tScaleB, s2t_a, s2t_a_src, s2t_a_dst,
                   s2t_b, s2t_b_src, s2t_b_dst, B_idx, zero_init):
        r"""block-scaled dV += \hat{P}^T dO: S2T of the \hat{P} SFs / constant dO SFs into their S-bank parks (\hat{P}(t) stored => S(t)
        loaded; the dQ-side parks were consumed earlier in the tensor pipe's order), then the mxf8f6f4 GEMM (dO stage 0)."""
        sm100_utils.tcgen05_after_thread_sync()
        cute.copy(s2t_a, s2t_a_src, s2t_a_dst)
        cute.copy(s2t_b, s2t_b_src, s2t_b_dst)
        sm100_utils.gemm_ptx_partial_fp4(
            op, self.tmem_dV_offset, tCrA, tCrB_staged[None, None, None, 0], sA=None, sB=sB_staged[None, None, None, 0],
            tScaleA=tScaleA, tScaleB=tScaleB, zero_init=zero_init, tA_addr=self.tmem_P_offset,
        )

    @cute.jit
    def _mma_dQ_mx(self, op, tCrA, tCrB, sA, sB, tScaleA, tScaleB, s2t_a, s2t_a_src, s2t_a_dst, s2t_b, s2t_b_src, s2t_b_dst):
        r"""block-scaled dQ = d\hat{S} K over the cluster's 256 keys: S2T of the d\hat{S}^T SFs (own + the peer's half, both landed
        with the dS exchange) / constant K SFs into their S-bank parks, then the mxf8f6f4 GEMM (A = d\hat{S}^T smem MN-major)."""
        sm100_utils.tcgen05_after_thread_sync()
        cute.copy(s2t_a, s2t_a_src, s2t_a_dst)
        cute.copy(s2t_b, s2t_b_src, s2t_b_dst)
        sm100_utils.gemm_ptx_partial_fp4(
            op, self.tmem_dQ_offset, tCrA, tCrB, sA=sA, sB=sB, tScaleA=tScaleA, tScaleB=tScaleB, zero_init=True,
        )

    @cute.jit
    def _noop_q(self, src_idx, **kwargs):
        pass

    @cute.jit
    def _mx_redux(self, x: Float32) -> Float32:
        """warp abs-max of the blk32 exponent (redux.sync.max.abs.f32)."""
        return mxs.redux_max_abs_f32(x)

    @cute.jit
    def _csb(self):
        """Compute-warp named barrier (all 8 warps)."""
        self.compute_sync_barrier.arrive_and_wait()

    def _tl_slot(self, sMxSF, stage, warp_slot, vec: cutlass.Constexpr[int]):
        """token-local operands: this warp's 32-float SMEM slot of a per-query factor vector (vec 0 = f_Q, 1 = f_dO) at sMxSF byte
        3104: [query 32][stage 2][compute warp 8][vector 2]."""
        t = cute.make_tensor(cute.recast_ptr(sMxSF.iterator + 3104, dtype=Float32),
                             cute.make_layout((32, 2, 8, 2), stride=(1, 32, 64, 512)))
        return t[None, stage, warp_slot, vec]

    @cute.jit
    def _tl_query_factors(self, mTokFac, row: cutlass.Constexpr[int], batch_idx, head_idx, m_block, tScS_stage,
                          slot, lane, load: cutlass.Constexpr[bool], pre=None):
        """token-local operands: the per-query factors of this stage's 32 queries (element i = query q0 + i) as
        a 32-float fragment; load=True: lane j fetches query q0 + j into the warp's SMEM slot (one coalesced load per
        warp), then every lane reads the slot (broadcast); load=False re-reads it (as the 1-CTA kernel's)."""
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

    @cute.jit
    def _ds_dq_rowlocal_words(self, vals, words, lane, wps: cutlass.Constexpr[int], s_mul, cm=None):
        """ds_dq_rowlocal: the dQ side of one 32-key x 32-query dS block with one UE8M0 exponent per query (as the 1-CTA
        kernel's _ds_dq_rowlocal_words_x): query `lane`'s abs-max over the warp's 32 keys by a warp reduce-scatter, its
        multiplier to every lane through this warp's 32-float smem slot `s_mul`, the E4M3 codes into `words`, and the
        exponent of query `lane` returned (that lane's row of the dQ SF tile)."""
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
    def xchg_warp(
        self, sdS, sdS_xchg, dS_cluster_full_mbar_ptr, dS_stored_mbar_ptr, block_info, SeqlenInfoCls, TileSchedulerCls,
        sSFq_xchg=None, sSFdSq=None, sMxSF=None, sSFdS_mx=None, sf_ready_mbar_ptr=None,
    ):
        r"""warp_sync: the half-exchange DSMEM copy of every query tile, issued once all 8 compute warps have written their
        part of the exchange buffer (keeps the compute warps free of cross-warp waits). Block-scaled d\hat{S}: a second copy
        carries this CTA's 512 B K-atom of the peer's dQ-side SF tile on the same barrier."""
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
        stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
        sf_copy_bytes = const_expr(512 if self.mxds else 0)
        phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block // self.cluster_shape_mnk[0])
            process_tile = const_expr(not self.dyn_tiles and not self.is_varlen_q) or m_block_min < m_block_max
            if process_tile:
                for _ in cutlass.range(m_block_max - m_block_min, unroll=1):
                    cute.arch.mbarrier_wait(dS_stored_mbar_ptr, phase)
                    # every lane has seen this phase before lane 0 hands the tile to the peer: otherwise the next
                    # tile's dS_stored phase can complete while a lagging lane still polls this parity (the
                    # det_issue hand-off hang of flash_bwd_sm100_fp4.dQ_issue)
                    cute.arch.sync_warp()
                    phase ^= 1
                    peer = cta_rank_in_cluster ^ 1
                    if const_expr(self.mx_xw_any):
                        # the 8 KB dS half-exchange goes first; the SF tiles are assembled while it is in flight
                        if tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(dS_cluster_full_mbar_ptr, stage_copy_bytes + sf_copy_bytes, peer_cta_rank_in_cluster=peer)
                            copy_utils.cpasync_bulk_s2cluster(
                                sdS_xchg.iterator, sdS.iterator + cta_rank_in_cluster * stage_copy_elems, dS_cluster_full_mbar_ptr,
                                stage_copy_bytes, peer_cta_rank_in_cluster=peer,
                            )
                    if const_expr(self.mx_xw_any):
                        # MX_XW: assemble this tile's dQ-side SF tiles from the compute warps' scratch (parity = phase after
                        # the flip = the compute warps' phase_Sl during this tile). Row r of the 128-row atom sits at
                        # (r % 32) * 16 + (r // 32) * 4 + k: rows 0-31 are WG 0's queries, 32-63 WG 1's, so every 16 B row
                        # chunk (lane) takes the same 8 bytes [wg 0: key groups 0-3][wg 1: 0-3] -- the own-half stage
                        # (stage == rank) into K-atom `rank` of the own tile, the peer-half stage into the exchange atom
                        _s8 = cute.make_tensor(cute.recast_ptr(sMxSF.iterator, dtype=cutlass.Int64), cute.make_layout(4))
                        _own = _s8[phase * 2 + cta_rank_in_cluster]
                        _xch = _s8[phase * 2 + (cta_rank_in_cluster ^ 1)]
                        if const_expr(self.ds_dq_rowlocal != "off"):
                            # row-local dQ side: every query row (lane) has its own 8 bytes [wg 0: key groups 0-3][wg 1: 0-3],
                            # written by the compute warps at [32 + ((parity * 2 + stage) * 32 + lane) * 8]
                            _q8 = cute.make_tensor(cute.recast_ptr(sMxSF.iterator + 32, dtype=cutlass.Int64),
                                                   cute.make_layout(128))
                            _own = _q8[(phase * 2 + cta_rank_in_cluster) * 32 + tidx]
                            _xch = _q8[(phase * 2 + (cta_rank_in_cluster ^ 1)) * 32 + tidx]
                        _dq8 = cute.make_tensor(cute.recast_ptr(sSFdSq.iterator, dtype=cutlass.Int64), cute.make_layout(128))
                        _xc8 = cute.make_tensor(cute.recast_ptr(sSFq_xchg.iterator, dtype=cutlass.Int64), cute.make_layout(128))
                        _dq8[cta_rank_in_cluster * 64 + tidx * 2] = _own
                        _xc8[cta_rank_in_cluster * 64 + tidx * 2] = _xch
                        if const_expr(self.mx_xw_all):
                            # the dK-side tile: key row r at (r % 32) * 16 + (r // 32) * 4 + g (g = query group 2 s + w), and
                            # the exponent is uniform over the 32 keys of a warp, so all 32 rows are the same 16 bytes:
                            # word kb = [e(s0, wg0, kb), e(s0, wg1, kb), e(s1, wg0, kb), e(s1, wg1, kb)] (a 4 x 4 byte transpose
                            # of the scratch words [s][wg] = [kb0..kb3])
                            _dk32 = cute.make_tensor(cute.recast_ptr(sSFdS_mx.iterator, dtype=Int32), cute.make_layout(128))
                            for kb in cutlass.range_constexpr(4):
                                _w = Int32(0)
                                for g in cutlass.range_constexpr(4):  # g = 2 s + w -> scratch [s][w][kb]
                                    _w = _w | (sMxSF[phase * 16 + (g // 2) * 8 + (g % 2) * 4 + kb].to(Int32) << (8 * g))
                                _dk32[tidx * 4 + kb] = _w
                        cute.arch.fence_view_async_shared()
                        # the leader's S2T (cta_group::2) reads both CTAs' tiles: a cluster-scope arrive on the leader's
                        # sf_ready from each CTA's exchange warp orders these writes before it (count 2)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(sf_ready_mbar_ptr, Int32(0))
                        if tidx == 0:
                            copy_utils.cpasync_bulk_s2cluster(
                                sSFq_xchg.iterator + cta_rank_in_cluster * 512, sSFdSq.iterator + cta_rank_in_cluster * 512,
                                dS_cluster_full_mbar_ptr, sf_copy_bytes, peer_cta_rank_in_cluster=peer,
                            )
                    if const_expr(not self.mx_xw_any):
                        if tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(dS_cluster_full_mbar_ptr, stage_copy_bytes + sf_copy_bytes, peer_cta_rank_in_cluster=peer)
                            copy_utils.cpasync_bulk_s2cluster(
                                sdS_xchg.iterator, sdS.iterator + cta_rank_in_cluster * stage_copy_elems, dS_cluster_full_mbar_ptr,
                                stage_copy_bytes, peer_cta_rank_in_cluster=peer,
                            )
                            if const_expr(self.mxds):
                                copy_utils.cpasync_bulk_s2cluster(
                                    sSFq_xchg.iterator + cta_rank_in_cluster * 512, sSFdSq.iterator + cta_rank_in_cluster * 512,
                                    dS_cluster_full_mbar_ptr, sf_copy_bytes, peer_cta_rank_in_cluster=peer,
                                )
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block // self.cluster_shape_mnk[0])
            head_idx_kv = head_idx // self.qhead_per_kvhead

            process_tile = (
                const_expr(not self.dyn_tiles and not self.is_varlen_q) or m_block_min < m_block_max
            )

            if process_tile:
                num_iters = m_block_max - m_block_min
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.atom.ThrMma,
        thr_mma_dP: cute.atom.ThrMma,
        thr_mma_dV: cute.atom.ThrMma,
        thr_mma_dK: cute.atom.ThrMma,
        thr_mma_dQ: cute.atom.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
        mSFK: Optional[cute.Tensor] = None,
        mSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        sSFQ: Optional[cute.Tensor] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tiled_mma_sfb_S: Optional[cute.TiledMma] = None,
        tb=None,
    ):
        (mSFQd, mSFdOd, mSFKd, tma_atom_SFQd, tma_atom_SFdOd, tma_atom_SFKd, tiled_mma_sfb_dim, tiled_mma_dq_sf,
         sSFQc, sSFdOc, sSFKc) = tb
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block // self.cluster_shape_mnk[0])
            head_idx_kv = head_idx // self.qhead_per_kvhead
            n_block_cta_group = n_block // self.cta_group_size

            # GMEM tensors (varlen-aware)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset((0, seqlen.offset_q), mdO[None, None, head_idx])
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]

            if const_expr(self.use_2cta_instrs):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQt_cur = mQt[None, None, head_idx, batch_idx]
                    mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                else:
                    mQt_cur = cute.domain_offset((0, seqlen.offset_q, 0), mQt)[None, None, head_idx]
                    mdOt_cur = cute.domain_offset((seqlen.offset_q, 0, 0), mdOt)[
                        None, None, head_idx
                    ]
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
                else:
                    mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[
                        None, None, head_idx_kv
                    ]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            load_SFK = self._noop
            load_SFQ = self._noop_q
            if const_expr(self.nvfp4_s):
                mSFK_cur = mSFK[None, None, head_idx_kv, batch_idx]
                gSFK = cute.local_tile(mSFK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0))
                load_SFK, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_SFK, block_in_cluster_coord_vmnk[2], a_cta_layout, thr_mma_S.partition_A(gSFK), sSFK,
                    filter_zeros=True, single_stage=True,
                )
                thr_mma_sfb = tiled_mma_sfb_S.get_slice(0)
                cl_sfb = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_sfb_S.thr_id.shape,))
                coord_sfb = cl_sfb.get_flat_coord(cta_rank_in_cluster)
                mSFQ_cur = mSFQ[None, None, head_idx, batch_idx]
                gSFQ = cute.local_tile(mSFQ_cur, cute.select(self.mma_tiler_sfb, mode=[1, 2]), (None, 0))
                load_SFQ, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_SFQ, cta_coord=coord_sfb[1], cta_layout=cute.make_layout(cute.slice_(cl_sfb, (0, None, 0, 0)).shape),
                    src_tensor=thr_mma_sfb.partition_B(gSFQ), dst_tensor=sSFQ, filter_zeros=True,
                    mcast_mask=cpasync.create_tma_multicast_mask(cl_sfb, coord_sfb, mcast_mode=1),
                )
                load_SFQ = copy_utils.tma_producer_copy_fn(load_SFQ, pipeline_Q)

            load_SFQd = self._noop_q
            load_SFdOd = self._noop_q
            load_SFKd = self._noop
            if const_expr(self.mxop_tb):
                thr_sfb_d = tiled_mma_sfb_dim.get_slice(0)
                cl_d = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_sfb_dim.thr_id.shape,))
                coord_d = cl_d.get_flat_coord(cta_rank_in_cluster)
                lay_d = cute.make_layout(cute.slice_(cl_d, (0, None, 0, 0)).shape)
                mask_d = cpasync.create_tma_multicast_mask(cl_d, coord_d, mcast_mode=1)
                # Q^T / dO d-rows atoms: (N = d tile 0, K = query tile m)
                gSFQd = cute.local_tile(mSFQd[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_sfb_dim, mode=[1, 2]), (0, None))
                load_SFQd, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_SFQd, cta_coord=coord_d[1], cta_layout=lay_d, src_tensor=thr_sfb_d.partition_B(gSFQd),
                    dst_tensor=sSFQc, filter_zeros=True, mcast_mask=mask_d,
                )
                load_SFQd = copy_utils.tma_producer_copy_fn(load_SFQd, pipeline_Qt)
                gSFdOd = cute.local_tile(mSFdOd[None, None, head_idx, batch_idx], cute.select(self.mma_tiler_sfb_dim, mode=[1, 2]), (0, None))
                load_SFdOd, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_SFdOd, cta_coord=coord_d[1], cta_layout=lay_d, src_tensor=thr_sfb_d.partition_B(gSFdOd),
                    dst_tensor=sSFdOc, filter_zeros=True, mcast_mask=mask_d,
                )
                load_SFdOd = copy_utils.tma_producer_copy_fn(load_SFdOd, pipeline_dO)
                # K^T d-rows atoms of the cluster's 256 keys (two K-atoms)
                thr_sfb_q = tiled_mma_dq_sf.get_slice(0)
                cl_q = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_dq_sf.thr_id.shape,))
                coord_q = cl_q.get_flat_coord(cta_rank_in_cluster)
                gSFKd = cute.local_tile(mSFKd[None, None, head_idx_kv, batch_idx], cute.select(self.mma_tiler_dsk_sf, mode=[1, 2]),
                                        (0, n_block_cta_group))
                load_SFKd, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_SFKd, cta_coord=coord_q[1], cta_layout=cute.make_layout(cute.slice_(cl_q, (0, None, 0, 0)).shape),
                    src_tensor=thr_sfb_q.partition_B(gSFKd), dst_tensor=sSFKc, filter_zeros=True, single_stage=True,
                    mcast_mask=cpasync.create_tma_multicast_mask(cl_q, coord_q, mcast_mode=1),
                )

            # (2) dP = V @ dO.T
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(
                    mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                )
                tdPgdO = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdO,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(
                    mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                )
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # (5) dQ = dS @ K
            if const_expr(self.use_2cta_instrs):
                gKt = cute.local_tile(
                    mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block_cta_group)
                )
                tdQgK = thr_mma_dQ.partition_B(gKt)

                load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tdQgK,
                    sKt,
                    single_stage=True,
                )


            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

            # some tiles might be empty due to block sparsity
            if const_expr(self.use_block_sparsity):
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)
            else:
                process_tile = (
                    const_expr(not self.dyn_tiles and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )

            if process_tile:
                if const_expr(self.use_block_sparsity):
                    producer_state_Q_LSE, producer_state_dO_dPsum = (
                        produce_block_sparse_q_loads_bwd_sm100(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q_LSE,
                            producer_state_dO_dPsum,
                            pipeline_Q,
                            pipeline_LSE,
                            pipeline_dO,
                            pipeline_dPsum,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            copy_stats,
                            gLSE,
                            sLSE,
                            gdPsum,
                            sdPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            should_load_Q=should_load_Q,
                            should_load_dO=should_load_dO,
                            q_subtile_factor=self.subtile_factor,
                            m_block_max=m_block_max,
                        )
                    )
                else:
                    first_m_block = m_block_min
                    if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                        #### Prologue ####
                        assert should_load_Q and should_load_dO
                        # K & Q (for S)
                        pipeline_Q.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                        load_Q(first_m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Q.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()
                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, first_m_block],
                                sLSE[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                            )
                        producer_state_LSE.advance()

                        # dOt + V, for dP.T = V @ dO.T
                        pipeline_dO.producer_acquire(
                            producer_state_O_Ot,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                        load_dOt(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()
                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, first_m_block],
                                sdPsum[None, producer_state_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                            )
                        producer_state_dPsum.advance()

                        # Qt, for dK = dS.T @ Q
                        pipeline_Qt.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_Qt(first_m_block, producer_state=producer_state_Q_Qt)
                        load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                        pipeline_Qt.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dO, for dV = P.T @ dO
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dO(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                        #### Mainloop ####
                        # 2CTA: [lse | Q | dOt | dPsum | Qt | dO]
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                            producer_state_LSE.advance()

                            # Q
                            pipeline_Q.producer_acquire(producer_state_Q_Qt)
                            load_Q(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Q.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dPsum
                                    ),
                                )
                            producer_state_dPsum.advance()

                            # dOt, for dP.T = V @ dO.T
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dOt(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                            load_Qt(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Qt.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dO, for dV = P.T @ dO
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dO(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                    else:
                        #### Prologue ####
                        if const_expr(should_load_Q):
                            # K & Q (for S)
                            pipeline_Q.producer_acquire(
                                producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                            )
                            load_K(
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                            )
                            load_SFK(
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                            )
                            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                            load_SFQ(first_m_block, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, first_m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                            producer_state_Q_LSE.advance()

                        if const_expr(should_load_dO):
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dOt"]
                                if const_expr(tma_atom_dOt is not None)
                                else self.tma_copy_bytes["V"],
                            )
                            load_V(
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                )
                            )
                            load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
                            load_SFdOd(first_m_block, producer_state=producer_state_dO_dPsum)
                            if const_expr(tma_atom_dOt is not None):
                                load_dOt(first_m_block, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, first_m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                            producer_state_dO_dPsum.advance()

                        if const_expr(self.use_2cta_instrs):
                            pipeline_Kt.producer_acquire(producer_state_Kt)
                            load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                            load_SFKd(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                            pipeline_Kt.producer_commit(producer_state_Kt)
                            producer_state_Kt.advance()
                        #### Main Loop ####
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            if const_expr(should_load_Q):
                                if const_expr(tma_atom_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(m_block - 1, producer_state=producer_state_Qt)
                                    load_SFQd(m_block - 1, producer_state=producer_state_Qt)
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                # Q (for S)
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                load_Q(m_block, producer_state=producer_state_Q_LSE)
                                load_SFQ(m_block, producer_state=producer_state_Q_LSE)
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gLSE[None, m_block],
                                        sLSE[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                    )
                                producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["dOt"]
                                    if const_expr(tma_atom_dOt is not None)
                                    else 0,
                                )
                                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                                load_SFdOd(m_block, producer_state=producer_state_dO_dPsum)
                                if const_expr(tma_atom_dOt is not None):
                                    load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gdPsum[None, m_block],
                                        sdPsum[None, producer_state_dO_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                    )
                                producer_state_dO_dPsum.advance()

                        #### Tail ####
                        if const_expr(should_load_Q):
                            if const_expr(tma_atom_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                                load_SFQd(m_block_max - 1, producer_state=producer_state_Qt)
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                    pipeline_Q.producer_tail(producer_state_Q_Qt)
                    pipeline_LSE.producer_tail(producer_state_LSE)
                    pipeline_dO.producer_tail(producer_state_O_Ot)
                    pipeline_dPsum.producer_tail(producer_state_dPsum)
                else:
                    if const_expr(should_load_Q):
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        if const_expr(tma_atom_Qt is not None):
                            pipeline_Qt.producer_tail(producer_state_Qt)
                    if const_expr(should_load_dO):
                        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tP: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        sf_ready_mbar_ptr: cute.Pointer,
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        is_leader_cta: cutlass.Boolean,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        sSFK=None,
        sSFQ=None,
        tCtSFK=None,
        tCtSFQ=None,
        dP_loaded_mbar_ptr=None,
        sSFdS_mx=None, sSFQc=None, sSFdSq=None, sSFKc=None,
        tCtSFdS_mx=None, tCtSFQc=None, tCtSFdSq=None, tCtSFKc=None,
        sSFP_mx=None, sSFdOc=None, tCtSFP_mx=None, tCtSFdOc=None,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        # For 2-CTA, dS (dK mma) MUST come from TMEM (cannot use SMEM)
        if const_expr(self.use_smem_dS_for_mma_dK and not self.use_2cta_instrs):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)  # From SMEM
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM

        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        if const_expr(self.nvfp4_s):
            s2t_sfk, tCsSFK_s2t, tCtSFK_s2t = self.s2t_copy_and_partition(sSFK, tCtSFK)
            s2t_sfq, tCsSFQ_s2t, tCtSFQ_s2t = self.s2t_copy_and_partition(sSFQ, tCtSFQ)
            mma_qk_fn = partial(
                self._mma_S_fp4, tiled_mma_S.op, tSrK, tSrQ, sK, sQ, tCtSFK, tCtSFQ,
                s2t_sfk, tCsSFK_s2t[None, None, None, None, 0], tCtSFK_s2t,
                s2t_sfq, tCsSFQ_s2t[None, None, None, None, 0], tCtSFQ_s2t,
            )
        phase_dPl_m = Int32(0)
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
            cta_group=self.cta_group_size,
        )
        if const_expr(self.mxdv):
            # block-scaled \hat{P}: S2T of the \hat{P} SFs / constant dO SFs into their S-bank parks, then the mxf8f6f4 dV GEMM
            # (blk32x128: dO's SF tile arrives by TMA with the dO stage; P^'s is constant 2^0 unless mxp)
            assert self.dO_stage == 1
            _pp = [self.s2t_copy_and_partition(s, t, dtype=self.sf_dtype_mx) for s, t in ((sSFP_mx, tCtSFP_mx), (sSFdOc, tCtSFdOc))]
            _lane_p = cute.arch.lane_idx()
            for _cbuf in ([] if const_expr(self.mxop_tb) else [sSFdOc]) + ([] if const_expr(self.mxp) else [sSFP_mx]):
                _nb = cute.cosize(_cbuf.layout)
                _w32 = cute.make_tensor(cute.recast_ptr(_cbuf.iterator, dtype=Int32), cute.make_layout(_nb // 4))
                for i in cutlass.range_constexpr(_nb // 128):
                    _w32[_lane_p + 32 * i] = Int32(0x7F7F7F7F)
            cute.arch.fence_view_async_shared()
            mma_pdo_fn = partial(
                self._mma_dV_mx, tiled_mma_dV.op, tdVrP, tdVrdO, sdO, tCtSFP_mx, tCtSFdOc,
                _pp[0][0], _pp[0][1][None, None, None, None, 0], _pp[0][2], _pp[1][0], _pp[1][1][None, None, None, None, 0], _pp[1][2],
            )
        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        # E4M3 x 2-CTA: the DSL gemm op leaves the dQ accumulator NaN; the inline-PTX path works
        mma_dsk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        if const_expr(self.use_smem_dS_for_mma_dK and not self.use_2cta_instrs):
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ)
        else:
            # Need to explicitly pass in tA_addr for correctness
            mma_dsq_fn = partial(
                gemm_ptx_w_idx,
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                sA=None,
                sB=sQt,
                tA_addr=self.tmem_dS_offset,
                cta_group=self.cta_group_size,
            )
        if const_expr(self.mxds):
            # block-scaled d\hat{S}: S2T partitions of the four SF tiles; constant 2^0 SF tiles for Q^T / K^T written once
            assert self.Q_stage == 1
            _p = [self.s2t_copy_and_partition(s, t, dtype=self.sf_dtype_mx)
                  for s, t in ((sSFdS_mx, tCtSFdS_mx), (sSFQc, tCtSFQc), (sSFdSq, tCtSFdSq), (sSFKc, tCtSFKc))]
            _lane = cute.arch.lane_idx()
            for buf in (() if const_expr(self.mxop_tb) else (sSFQc, sSFKc)):
                nbytes = cute.cosize(buf.layout)
                w32 = cute.make_tensor(cute.recast_ptr(buf.iterator, dtype=Int32), cute.make_layout(nbytes // 4))
                for i in cutlass.range_constexpr(nbytes // 128):
                    w32[_lane + 32 * i] = Int32(0x7F7F7F7F)
            cute.arch.fence_view_async_shared()
            mma_dsq_fn = partial(
                self._mma_dK_mx, tiled_mma_dK.op, tdKrdS, tdKrQ, sQt, tCtSFdS_mx, tCtSFQc,
                _p[0][0], _p[0][1][None, None, None, None, 0], _p[0][2], _p[1][0], _p[1][1][None, None, None, None, 0], _p[1][2],
            )
            mma_dsk_fn = partial(
                self._mma_dQ_mx, tiled_mma_dQ.op, tdQrdS, tdQrK, sdS, sKt, tCtSFdSq, tCtSFKc,
                _p[2][0], _p[2][1][None, None, None, None, 0], _p[2][2], _p[3][0], _p[3][1][None, None, None, None, 0], _p[3][2],
            )

        pipeline_Q_consumer = pipeline_Q.make_consumer()

        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block // self.cluster_shape_mnk[0])

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                block_iter_count = m_block_max - m_block_min
                process_tile = (
                    const_expr(not self.dyn_tiles and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )

            if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    accumulate_dV = False

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dP.T = V    @ dO.T
                    # 3. dK   = dS.T @ Q
                    # 4. dV   = P.T  @ dO
                    # 5. dQ   = dS   @ K

                    main_loop_iters = m_block_max - m_block_min

                    # empty waits
                    # pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dQ tmem overlaps with S
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_S_P.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP tmem overlaps with S
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Q.index, zero_init=not accumulate_dK)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        if const_expr(self.mx_xw_any and not self.mx_xw_all):  # both exchange warps assembled the dQ-side SF tiles
                            cute.arch.mbarrier_wait(sf_ready_mbar_ptr, consumer_state_dS.phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1
            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dK   = dS.T @ Q
                    # 3. dP.T = V    @ dO.T
                    # 4. dQ   = dS   @ K
                    # 5. dV   = P.T  @ dO

                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min - 1
                    )

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T (next)
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        if const_expr(self.nvfp4_s):
                            # SFK / SFQ(t+1) are parked in the dP bank: every compute warp has loaded stage 0 of dP(t)
                            cute.arch.mbarrier_wait(dP_loaded_mbar_ptr, phase_dPl_m)
                            phase_dPl_m ^= 1
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # pipeline_dS.consumer_wait(consumer_state_dS)
                        # (2) dK += dS.T @ Q (cur)
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        if const_expr(self.mx_xw_all):  # both exchange warps assembled this tile's dK-side SF tile
                            cute.arch.mbarrier_wait(sf_ready_mbar_ptr, consumer_state_dS.phase)
                        mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        # (3) dP.T = V @ dO.T (next)
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # (5) dQ = dS @ K (cur)
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        if const_expr(self.mx_xw_any and not self.mx_xw_all):  # both exchange warps assembled the dQ-side SF tiles
                            cute.arch.mbarrier_wait(sf_ready_mbar_ptr, consumer_state_dS.phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1

                        # (4) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S -> P
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # pipeline_dS.consumer_wait(consumer_state_dS)
                    # dK += dS.T @ Q
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                    if const_expr(self.nvfp4_s):
                        cute.arch.mbarrier_wait(dP_loaded_mbar_ptr, phase_dPl_m)  # consume dP(last)'s wave
                        phase_dPl_m ^= 1
                    if const_expr(self.mx_xw_all):
                        cute.arch.mbarrier_wait(sf_ready_mbar_ptr, consumer_state_dS.phase)
                    mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    if const_expr(self.mx_xw_any and not self.mx_xw_all):
                        cute.arch.mbarrier_wait(sf_ready_mbar_ptr, consumer_state_dS.phase)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_dS.advance()
                    consumer_state_Kt.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1

                    producer_phase_acc ^= 1
            else:
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    handle_Q = pipeline_Q_consumer.wait_and_advance()
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=handle_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    producer_phase_acc ^= 1
                    # 3) dV = P.T @ dO
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S  = K    @ Q.T
                    # 2. dQ = dS   @ K
                    # 3. dK = dS.T @ Q
                    # 4. dP = V    @ dOt.T
                    # 5. dV = P.T  @ dO

                    # For block sparsity, we use block_iter_count; for dense, use m_block range
                    # MMA doesn't need actual m_block indices, just the iteration count
                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min - 1
                    )

                    handle_Q_next = handle_Q
                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                        mma_qk_fn(B_idx=handle_Q_next.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                        # (2) dK += dS.T @ Q
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()

                        # (3) dQ = dS @ K
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()

                        # (4) dP = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # (5) dV += P.T @ dO
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        handle_Q = handle_Q_next

                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # signal to the epilogue that dV is ready
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    # pipeline_dKV.producer_commit(producer_state_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # producer_state_dKV.advance()
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # 1) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # 2) dQ = dS @ K
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    handle_Q.release()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply forward score modification for SM100 backward pass."""
        # In bwd, S is computed as K @ Q.T so dimensions are (tile_n, tile_m)
        cS = cute.make_identity_tensor((self.tile_n, self.tile_m))
        cS = cute.domain_offset((n_block * self.tile_n, m_block * self.tile_m), cS)
        tScS = thr_mma_S.partition_C(cS)
        tScS_idx = thr_copy_t2r.partition_D(tScS)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_idx,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply backward score modification (joint graph) for SM100."""
        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            index_tensor,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.atom.ThrMma,
        thr_mma_dP: cute.atom.ThrMma,
        thr_mma_dV: cute.atom.ThrMma,
        thr_mma_dK: cute.atom.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdS_xchg: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: cute.Pointer,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        ds_boost: cutlass.Float32,
        dv_scale: cutlass.Float32,
        dk_scale: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        S_loaded_mbar_ptr=None,
        dP_loaded_mbar_ptr=None,
        dP_loaded_pair_mbar_ptr=None,
        dS_stored_mbar_ptr=None,
        sSFdS_mx_u8=None, sSFdSq_u8=None, sSFq_xchg_u8=None, sSFP_mx_u8=None,
        sMxRed=None, sMxSF=None,
        mFac=None,
        mTokFac=None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.cta_tiler[1] // 32 * self.do_dtype.width  # E4M3 P^ / dS^
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        # 2-CTA assumes: repetiton should always be 32 & 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.ld_rep)), Float32
        )
        # \hat{P} / d\hat{S} columns per (WG, stage) block: 16 for bf16, 8 for E4M3
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(self.ld_rep * self.ds_dtype.width // 32)), Float32
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        if const_expr(self.use_2cta_instrs):
            sdS_xchg_epi = cute.make_tensor(
                cute.recast_ptr(sdS_xchg.iterator, sdS_epi_layout.inner), sdS_layout
            )
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        # E4M3 d\hat{S} SMEM stores. The 2-CTA dQ A tile (M = 64 queries of this CTA's half, K = 256 keys) is laid out
        # q + 64 key on the swizzled pointer; WG w / stage s of a key row holds queries 64 s + 32 w + [0, 32).
        atom_ds_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.ds_dtype, num_bits_per_copy=128)
        thr_ds_r2s = cute.make_tiled_copy_tv(
            atom_ds_r2s,
            cute.make_ordered_layout((128, 1), order=(1, 0)),
            cute.make_ordered_layout((1, 128 // self.ds_dtype.width), order=(1, 0)),
        ).get_slice(dp_idx)
        sdS_kq = cute.make_tensor(sdS.iterator, cute.make_layout((2 * self.tile_n, self.tile_m // 2), stride=(self.tile_m // 2, 1)))
        if const_expr(self.ld_rep == 64):
            # x64 loads: WG w holds the whole query half w (one stage)
            sdS_own = cute.logical_divide(sdS_kq, (self.tile_n, self.tile_m // 2))[((None, cta_rank_in_cluster), None)]
        else:
            sdS_own = cute.logical_divide(sdS_kq, (self.tile_n, self.tile_m // 4))[((None, cta_rank_in_cluster), (None, wg_idx))]
        tSdS_own_r2s = thr_ds_r2s.partition_D(sdS_own)
        sdS_xchg_kq = cute.make_tensor(
            cute.recast_ptr(sdS_xchg.iterator, utils.parse_swizzle_from_pointer(sdS.iterator), dtype=self.ds_dtype),
            cute.make_layout((self.tile_n, self.tile_m // 2), stride=(self.tile_m // 2, 1)),
        )
        if const_expr(self.ld_rep == 64):
            sdS_xchg_w = sdS_xchg_kq
        else:
            sdS_xchg_w = cute.logical_divide(sdS_xchg_kq, (self.tile_n, self.tile_m // 4))[(None, (None, wg_idx))]
        tSdS_xchg_r2s = thr_ds_r2s.partition_D(sdS_xchg_w)
        dS_cluster_empty_phase = Int32(1)
        phase_Sl = Int32(0)
        phase_dPl = Int32(0)
        phase_dSs = Int32(0)
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block // self.cluster_shape_mnk[0])
            # masks / varlen: the mask and the mods run in the mask's coordinates (mseqlen, document-local key pairs under
            # docs; batched: the same numbers as before); the mods see the document as their batch
            mask = AttentionMaskCls(mseqlen)
            n_block_for_cluster = n_blk_loc if const_expr(self.doc_varlen) else n_block // self.cta_group_size
            # this CTA's first key in the mask's coordinates (doc_t128: n_blk_loc is the own local 128-key tile)
            k_tile_loc = n_blk_loc if const_expr(self.doc_t128) else None
            mask_n_block = n_blk_loc - n_block % self.cta_group_size if const_expr(self.doc_t128) else n_block_for_cluster
            mod_batch = block_info.batch_for_mods(batch_idx, n_block // self.cta_group_size)
            fdm_t = (None, None)
            if const_expr(aux_tensors is not None and (self.mask_mod is not None or self.score_mod is not None)):
                # upstream: indices wrapped modulo the sequence lengths so a padded row never reads an aux tensor OOB
                fdm_t = (FastDivmodDivisorV2(cutlass.max(mseqlen.seqlen_q, 1)),
                                FastDivmodDivisorV2(cutlass.max(mseqlen.seqlen_k, 1)))
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=mask_n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=self.mask_mod,
                batch_idx=mod_batch,
                head_idx=head_idx,
                aux_data=_as_aux_data(aux_tensors),
                fastdiv_mods=fdm_t,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            loop_count = m_block_max - m_block_min
            process_tile = (
                const_expr(not self.dyn_tiles and not self.is_varlen_q) or m_block_min < m_block_max
            )
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)

            # Mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                m_block_oob = False
                is_full_block = False
                if const_expr(self.use_block_sparsity):
                    m_block, is_full_block = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    m_block_oob = m_block >= m_block_max
                tb_s = [None, None]
                tb_dp = [None, None]
                if const_expr(self.mxop_tb):
                    # token-block factors: S = f_K f_Q S^ (FP8 route; the dk8 route's S is NVFP4), dP = f_V f_dO dP^, one
                    # value per (warp = key block, stage = query block 2 st + wg)
                    _kb = n_block * (self.tile_n // 32) + dp_idx // 32
                    if const_expr(self.key_tail):
                        # the odd last pair's upper half lies past the keys (its document is the empty sentinel /
                        # past s_k): read the last real factor (its P is zero) instead of past the table
                        _kb = cutlass.min(_kb, cute.size(mFac.shape[3]) - 1)
                    if const_expr(self.qhead_per_kvhead > 1):
                        _fk = mFac[1, batch_idx, head_idx // self.qhead_per_kvhead, _kb]
                        _fv = mFac[2, batch_idx, head_idx // self.qhead_per_kvhead, _kb]
                    else:
                        _fk = mFac[1, batch_idx, head_idx, _kb]
                        _fv = mFac[2, batch_idx, head_idx, _kb]
                    if const_expr(self.token_local):
                        # token-local: this thread's key's own factors (dP^T rows are keys)
                        _fv = mTokFac[2, batch_idx, head_idx, n_block * self.tile_n + dp_idx]
                        tl_fk = mTokFac[1, batch_idx, head_idx, n_block * self.tile_n + dp_idx]
                        # this lane's query of each stage, f_Q / f_dO: issued here, staged into the warp slots later
                        tl_pre = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                        for _st in cutlass.range_constexpr(2):
                            _q = m_block * self.tile_m + tScS_t2r[None, _st, 0, 0][0][1] + dp_idx % 32
                            if const_expr(not self.nvfp4_s):
                                tl_pre[_st] = mTokFac[0, batch_idx, head_idx, _q]
                            tl_pre[2 + _st] = mTokFac[3, batch_idx, head_idx, _q]
                    for _st in cutlass.range_constexpr(2):
                        _qb = m_block * (self.tile_m // 32) + 2 * _st + wg_idx
                        if const_expr(self.token_local):
                            tb_s[_st] = softmax_scale_log2 * tl_fk  # token-local: f_Q per element (S -> P loop)
                        else:
                            tb_s[_st] = cutlass.max(softmax_scale_log2 * (_fk * mFac[0, batch_idx, head_idx, _qb]), Float32(1.1754944e-38))  # masks / varlen: an all-zero (padding) token block has factor 2^-127; -inf x 0 would be NaN
                        if const_expr(self.token_local):
                            tb_dp[_st] = ds_boost * _fv  # f_dO is per query: applied per element in the dS fold
                        else:
                            tb_dp[_st] = ds_boost * (_fv * mFac[3, batch_idx, head_idx, _qb])
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if const_expr(prefetch_LSE):
                    cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                if const_expr(self.tile_hdim == 192):
                    # Signal S tmem load completion using pipeline_S_P when hdim 192
                    # dP is overlapped with S
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                elif const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
                    # Signal S tmem load completion using pipeline_dS when 2cta hdim 128
                    # dQ is overlapped with S
                    if const_expr(self.warp_sync):
                        # every warp arrives once S(t) is in its registers: the \hat{P} stores into the other WG's S columns wait on it
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(S_loaded_mbar_ptr)
                    if iter_idx > 0:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            pipeline_dS.producer_commit(producer_state_dS)
                        producer_state_dS.advance()

                # masks / varlen: mask / mod coordinates (document-local under docs)
                m_loc = m_block - m_blk_off if const_expr(self.doc_varlen) else m_block
                tSrS_pre = None
                tScS_mod = None
                if const_expr(self.score_mod is not None):
                    # upstream order: score_mod on the scaled score, then the mask; P = exp2(S' log2 e - LSE). The scaled
                    # pre-mod score is kept for score_mod_bwd. The index tensor spans the 256-key cluster tile (each CTA
                    # partitions its half; the inherited version used 128-key coordinates). Scale = the kernel's own (the
                    # folded descales; per stage for token blocks on the FP8 route).
                    tSrS_pre = cute.make_rmem_tensor(tSrS_t2r.shape, Float32)
                    cS_mod = cute.domain_offset((n_block_for_cluster * self.tile_n * self.cta_group_size, m_loc * self.tile_m),
                                                cute.make_identity_tensor(self.mma_tiler_kq[:2]))
                    tScS_mod = thr_copy_t2r.partition_D(thr_mma_S.partition_C(cS_mod))
                    for _st in cutlass.range_constexpr(cute.size(tScS_t2r, mode=[1])):
                        _sl2m = tb_s[_st] if const_expr(self.mxop_tb and not self.nvfp4_s) else softmax_scale_log2
                        _scn = _sl2m * Float32(math.log(2.0))
                        _cur = tSrS_t2r[None, _st, 0, 0]
                        _pre = tSrS_pre[None, _st, 0, 0]
                        for _i in cutlass.range_constexpr(cute.size(_cur)):
                            _pre[_i] = _cur[_i] * _scn
                        apply_score_mod_inner(
                            _cur, tScS_mod[None, _st, 0, 0], self.score_mod, mod_batch, head_idx, _scn,
                            self.vec_size, Float32, _as_aux_data(aux_tensors), fdm_t, mseqlen,
                            constant_q_idx=None, qhead_per_kvhead=1, transpose_indices=True,
                        )

                #### APPLY MASK (after score_mod, matching forward pass order)
                check_m_boundary = (m_loc + 1) * self.tile_m > mseqlen.seqlen_q
                if const_expr(self.doc_t128):
                    # documents packed at 128: the own tile's mask (keys of this CTA's document). A query tile of the
                    # other document (the pair straddles a document boundary) is masked whole through the same mask:
                    # its key tile is moved past seqlen_k (an explicit -inf loop measured +4 % on the causal step)
                    out_doc = m_loc < 0 or m_loc * self.tile_m >= mseqlen.seqlen_q
                    needs_mask = cutlass.Boolean(True)
                    if const_expr(self.causal_diag_only and self.is_causal and not self.is_local):
                        needs_mask = ((m_loc * self.tile_m + (mseqlen.seqlen_k - mseqlen.seqlen_q)
                                       < (k_tile_loc + 1) * self.tile_n - 1) or check_m_boundary
                                      or (k_tile_loc + 1) * self.tile_n > mseqlen.seqlen_k or out_doc)
                    elif const_expr(not self.is_causal and not self.is_local):
                        # dense: the mask only cuts keys past seqlen_k (and the other document's query tiles)
                        needs_mask = (k_tile_loc + 1) * self.tile_n > mseqlen.seqlen_k or out_doc
                    mask_nb = mask_n_block
                    if out_doc:
                        mask_nb = Int32(1 << 20)
                    if needs_mask:
                        mask_fn(tSrS_t2r, m_block=m_loc, n_block=mask_nb, is_full_block=is_full_block,
                                check_m_boundary=check_m_boundary)
                elif const_expr(self.causal_diag_only and self.is_causal and not self.is_local and self.mask_mod is None):
                    # causal: the (256-key cluster, 128-query) tile is fully visible once its smallest query reaches the cluster's
                    # largest key; only the diagonal tiles (and a ragged last query tile) pay the compare + select
                    needs_mask = (m_loc * self.tile_m + (mseqlen.seqlen_k - mseqlen.seqlen_q)
                                  < (n_block_for_cluster + 1) * self.tile_n * self.cta_group_size - 1) or check_m_boundary
                    if const_expr(self.doc_varlen):
                        # the document's last key pair holds padding keys
                        needs_mask = needs_mask or ((n_block_for_cluster + 1) * self.tile_n * self.cta_group_size > mseqlen.seqlen_k)
                    if needs_mask:
                        mask_fn(tSrS_t2r, m_block=m_loc, is_full_block=is_full_block, check_m_boundary=check_m_boundary)
                elif const_expr(not self.is_causal and not self.is_local and self.mask_mod is None):
                    # dense, one key sequence per batch row: the mask only cuts keys past seqlen_k, which only the last
                    # key pair of an s_k % 256 != 0 row holds (noncausal-lean: no per-step -inf select over the fragment)
                    if (n_block_for_cluster + 1) * self.tile_n * self.cta_group_size > mseqlen.seqlen_k:
                        mask_fn(tSrS_t2r, m_block=m_loc, is_full_block=is_full_block, check_m_boundary=check_m_boundary)
                else:
                    mask_fn(
                        tSrS_t2r,
                        m_block=m_loc,
                        is_full_block=is_full_block,
                        check_m_boundary=check_m_boundary,
                    )
                num_stages = cute.size(tScS_t2r, mode=[1])
                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                lane_idx = cute.arch.lane_idx()
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                tSrP_u32 = cute.recast_tensor(tSrP_r2t_f32, cutlass.Int32)
                wps = cute.size(tSrP_r2t_f32) // num_stages  # E4M3: 4 queries per 32-bit word
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(stage > 0 or not prefetch_LSE):
                        cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                    tSrLSE = tSrLSE_s2r
                    tl_fq = None
                    if const_expr(self.token_local and not self.nvfp4_s):
                        tl_fq = self._tl_query_factors(mTokFac, 0, batch_idx, head_idx, m_block,
                                                       tScS_t2r[None, stage, 0, 0],
                                                       self._tl_slot(sMxSF, stage, wg_idx * 4 + dp_idx // 32 % 4, 0),
                                                       dp_idx % 32, True, pre=tl_pre[stage])
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        _sl2 = tb_s[stage] if const_expr(self.mxop_tb and not self.nvfp4_s) else softmax_scale_log2
                        if const_expr(self.score_mod is not None):
                            _sl2 = Float32(math.log2(math.e))  # S' is already the scaled, modified score
                        if const_expr(self.token_local and not self.nvfp4_s):
                            # token-local: S = f_K(this key) f_Q(query) S^ -- the query factor per element
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                                (_sl2 * tl_fq[2 * v], _sl2 * tl_fq[2 * v + 1]),
                                (-lse_pair[0], -lse_pair[1]),
                            )
                        else:
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                                (_sl2, _sl2),
                                (-lse_pair[0], -lse_pair[1]),
                            )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    if const_expr(self.mxp):
                        # block-scaled \hat{P}: one UE8M0 exponent per (this key row, this stage's 32 queries),
                        # codes = P' 2^-e; the SF byte into the (32 x 4)-byte SF atom before "P stored"
                        _pv = tSrS_cur
                        if const_expr(self.token_local):
                            # token-local: f_dO(q) into P^ (token-local dO codes, unit dV-side scale; dV is not causal)
                            tl_fdo = self._tl_query_factors(mTokFac, 3, batch_idx, head_idx, m_block,
                                                            tScS_t2r[None, stage, 0, 0],
                                                            self._tl_slot(sMxSF, stage, wg_idx * 4 + dp_idx // 32 % 4, 1),
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
                    else:
                        # E4M3 \hat{P} (P' = 2^8 P through LSE): 4 queries per word
                        for j in cutlass.range_constexpr(wps):
                            tSrP_u32[wps * stage + j] = sm100_utils.packed_float_to_ue4m3(
                                tSrS_cur[4 * j], tSrS_cur[4 * j + 1], tSrS_cur[4 * j + 2], tSrS_cur[4 * j + 3]
                            )
                    if const_expr(stage == 0):
                        cute.arch.fence_view_async_tmem_load()
                        # Without this barrier, we could have 1 warp writing to P in tmem while
                        # another warp is still reading S from tmem.
                        if const_expr(self.warp_sync):
                            cute.arch.mbarrier_wait(S_loaded_mbar_ptr, phase_Sl)
                            phase_Sl ^= 1
                        else:
                            self._csb()
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                if const_expr(not self.warp_sync):
                    self._csb()
                if const_expr(not self.tile_hdim == 192):
                    # Signal tmem store P completion with pipeline_S_P
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                        # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                if const_expr(self._test_skew_peer):
                    # test-only skew (_test_skew_peer; tests/cute/test_2cta_dp_park_race.py): the peer CTA's WG 1 loads
                    # dP(t) late, so the leader's SFK / SFQ(t+1) parks into its stage-0 block [288, 320) would land first
                    # unless the MMA warp waits for both CTAs' compute warps (dP_loaded_pair_mbar_ptr)
                    if cta_rank_in_cluster == 1 and wg_idx == 1:
                        _t0 = cute.arch.clock64()
                        while cute.arch.clock64() - _t0 < Int64(20000):
                            pass
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                ### Now delayed to after loop
                # consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                ##### dS.T = P.T * (dP.T - Psum)
                tSrdS8_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # E4M3 d\hat{S} words of both stages
                tSrdS8_u32 = cute.recast_tensor(tSrdS8_f32, cutlass.Int32)
                # dP_hoist: issue BOTH stages' dP TMEM loads up front (as S already is, :3463-3464) instead of one
                # per stage. Serialised, each stage pays the full TMEM load latency before its own compute;
                # hoisted, the second load overlaps the first stage's arithmetic. long_scoreboard is the dominant
                # stall (notes 13.6) and the kernel is latency-bound, not traffic-bound -- reducing traffic
                # measured slower. Costs one extra 32-register fragment: on only under the two-pass order (mx_both).
                if const_expr(self.dp_hoist):
                    tdPrdP_all = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                    cute.copy(thr_copy_t2r, tdPtdP_t2r, tdPrdP_all)
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(self.warp_sync):
                        # both stages are in registers, so dP's TMEM columns are free one arrive earlier
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(dP_loaded_mbar_ptr)
                            if const_expr(self.nvfp4_s):
                                cute.arch.mbarrier_arrive(dP_loaded_pair_mbar_ptr, Int32(0))
                    else:
                        self._csb()
                # two-pass order (mx_both: blk32x64 / tile128, see __init__): pass 0 = the dS arithmetic and the in-thread abs-max of both
                # stages, then the exponents (the redux of both stages back to back; blk32x64 = one exponent for both;
                # tile128 = the WG's 4 warps max'd through smem behind a 128-thread barrier), pass 1 = multiply / pack / stores
                mx_am = [None] * num_stages
                mx_e = [None] * num_stages
                for pass_ in cutlass.range_constexpr(2 if self.mx_both else 1):
                    if const_expr(self.mx_both and pass_ == 1):
                        if const_expr(self.ds_scaling == "blk32x64"):
                            _e = mxs.ue8m0_exp_448(self._mx_redux(mxs.famax3_op(mx_am[0], mx_am[1], Float32(0.0))))
                            for _s in cutlass.range_constexpr(num_stages):
                                mx_e[_s] = _e
                        elif const_expr(self.mx_tile128):
                            _base = wg_idx * 16 + phase_Sl * 8  # parity-double-buffered per tile (phase_Sl flips per tile)
                            _warp = dp_idx // 32
                            for _s in cutlass.range_constexpr(num_stages):
                                sMxRed[_base + _s * 4 + _warp] = self._mx_redux(mx_am[_s])
                            cute.arch.barrier(barrier_id=8 + wg_idx, number_of_threads=128)
                            for _s in cutlass.range_constexpr(num_stages):
                                _m = mxs.famax3_op(sMxRed[_base + _s * 4], sMxRed[_base + _s * 4 + 1], sMxRed[_base + _s * 4 + 2])
                                mx_e[_s] = mxs.ue8m0_exp_448(mxs.famax3_op(_m, sMxRed[_base + _s * 4 + 3], Float32(0.0)))
                        else:
                            for _s in cutlass.range_constexpr(num_stages):
                                mx_e[_s] = mxs.ue8m0_exp_448(self._mx_redux(mx_am[_s]))
                    for stage in cutlass.range_constexpr(num_stages):
                        if const_expr(self.dp_hoist):
                            tdPrdP_t2r = tdPrdP_all[None, stage, None, None]
                        else:
                            tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                            cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(self.warp_sync):
                                if const_expr(stage == 0):
                                    # stage 0 of dP(t) in registers: the d\hat{S} stores (columns inside WG 0's stage-0 load block) may go
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive(dP_loaded_mbar_ptr)
                                        if const_expr(self.nvfp4_s):
                                            # and, cluster-wide, the leader's SFK / SFQ(t+1) parks in this CTA's WG-1
                                            # stage-0 block [288, 320) (the peer's compute warps were not waited for:
                                            # a slow WG 1 of the peer read SF bytes as dP)
                                            cute.arch.mbarrier_arrive(dP_loaded_pair_mbar_ptr, Int32(0))
                            else:
                                self._csb()
                        tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                        tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                        if const_expr(pass_ == 0):
                            tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                            cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                            tl_fdo_ds = None
                            if const_expr(self.token_local):
                                tl_fdo_ds = self._tl_query_factors(
                                    mTokFac, 3, batch_idx, head_idx, m_block, tScS_t2r[None, stage, 0, 0],
                                    self._tl_slot(sMxSF, stage, wg_idx * 4 + dp_idx // 32 % 4, 1), dp_idx % 32, False)
                            for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                                dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                                # b d\hat{P} + D'' (D'' = -b D / (s_v s_dO) from the host)
                                _dsb = tb_dp[stage] if const_expr(self.mxop_tb) else ds_boost
                                if const_expr(self.token_local):
                                    # token-local: b f_V(this key) f_dO(this query)
                                    tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                        (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                        (_dsb * tl_fdo_ds[2 * v], _dsb * tl_fdo_ds[2 * v + 1]), dPsum_pair
                                    )
                                else:
                                    tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                        (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (_dsb, _dsb), dPsum_pair
                                    )
                                p_pair = (tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                    p_pair,
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                )

                            if const_expr(self.score_mod_bwd is not None):
                                # dS w.r.t. the scaled pre-mod score (upstream apply_score_mod_bwd, cluster-tile indices).
                                # score_mod_bwd is linear in its grad, so the kernel's scaled dS units carry through; the
                                # dS quantisation below sees the final dS
                                tScS_idx_cur = tScS_mod[None, stage, 0, 0]
                                apply_score_mod_bwd_inner(
                                    tdPrdP_cur, tSrS_pre[None, stage, 0, 0], tScS_idx_cur, self.score_mod_bwd,
                                    mod_batch, head_idx, Float32(1.0), self.vec_size, Float32,
                                    _as_aux_data(aux_tensors), fdm_t, mseqlen,
                                    constant_q_idx=None, qhead_per_kvhead=1, transpose_indices=True,
                                )
                                # Zero out OOB positions (kv_idx >= seqlen_k) after score_mod_bwd
                                for i in cutlass.range(cute.size(tdPrdP_cur), unroll_full=True):
                                    kv_idx = tScS_idx_cur[i][0]
                                    tdPrdP_cur[i] = 0.0 if kv_idx >= mseqlen.seqlen_k else tdPrdP_cur[i]

                        if const_expr(self.mx_both and pass_ == 0):
                            mx_am[stage] = mxs.famax_tree([tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))])
                        else:
                            # E4M3 d\hat{S} words (4 values per word)
                            tdPrdS_w = cute.make_rmem_tensor(wps, cutlass.Int32)
                            if const_expr(self.mxds):
                                # block-scaled d\hat{S} (blk32): one UE8M0 exponent per (this warp's 32 keys x this stage's 32 queries),
                                # in-thread abs-max tree + one redux.sync; the codes serve both GEMMs.
                                cm_q = None  # ds_dq_rowlocal="all": this lane's query column abs-max over the warp's 32 keys
                                if const_expr(self.mx_both):
                                    e_r = mx_e[stage]
                                elif const_expr(self.token_local and not self.nvfp4_s):
                                    # token-local Q in the dK GEMM (unit Q scale): f_Q(query) into the dK side
                                    tl_fq_dk = self._tl_query_factors(
                                        mTokFac, 0, batch_idx, head_idx, m_block, tScS_t2r[None, stage, 0, 0],
                                        self._tl_slot(sMxSF, stage, wg_idx * 4 + dp_idx // 32 % 4, 0), dp_idx % 32, False)
                                    tl_dsk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                                    for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                        tl_dsk[2 * v], tl_dsk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (tl_fq_dk[2 * v], tl_fq_dk[2 * v + 1]))
                                    e_r = mxs.ue8m0_exp_448(self._mx_redux(mxs.famax_tree([tl_dsk[i] for i in range(cute.size(tl_dsk))])))
                                elif const_expr(self.ds_dq_rowlocal == "all" and not self.token_local):
                                    # the column abs-maxes first (reduce-scatter); the block's is their warp max (exact: e_r
                                    # is bitwise the tree + redux form's)
                                    cm_q = mxs.warp_col_max_scatter([tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))],
                                                                    dp_idx % 32)
                                    e_r = mxs.ue8m0_exp_448(self._mx_redux(cm_q))
                                else:
                                    am_r = mxs.famax_tree([tdPrdP_cur[i] for i in range(cute.size(tdPrdP_cur))])
                                    e_r = mxs.ue8m0_exp_448(self._mx_redux(am_r))
                                mul_r = mxs.ue8m0_mul(e_r)
                                tdPrdSk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                                for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                    if const_expr(self.token_local and not self.nvfp4_s):
                                        tdPrdSk[2 * v], tdPrdSk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                            (tl_dsk[2 * v], tl_dsk[2 * v + 1]), (mul_r, mul_r))
                                    else:
                                        tdPrdSk[2 * v], tdPrdSk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (mul_r, mul_r)
                                        )
                                for j in cutlass.range_constexpr(wps):
                                    tdPrdS_w[j] = sm100_utils.packed_float_to_ue4m3(
                                        tdPrdSk[4 * j], tdPrdSk[4 * j + 1], tdPrdSk[4 * j + 2], tdPrdSk[4 * j + 3]
                                    )
                                    tSrdS8_u32[wps * stage + j] = tdPrdS_w[j]
                            else:
                                for j in cutlass.range_constexpr(wps):
                                    tdPrdS_w[j] = sm100_utils.packed_float_to_ue4m3(
                                        tdPrdP_cur[4 * j], tdPrdP_cur[4 * j + 1], tdPrdP_cur[4 * j + 2], tdPrdP_cur[4 * j + 3]
                                    )
                                    tSrdS8_u32[wps * stage + j] = tdPrdS_w[j]
                            # the dQ side's words (the SMEM dS^ copy, own half and peer half) and its exponent byte: the dK
                            # side's unless ds_dq_rowlocal gives this block one exponent per query
                            tdPrdSq_w = tdPrdS_w
                            if const_expr(self.ds_dq_rowlocal != "off"):
                                e_q = e_r
                                tdPrdSq_w = cute.make_rmem_tensor(wps, cutlass.Int32)
                                for j in cutlass.range_constexpr(wps):
                                    tdPrdSq_w[j] = tdPrdS_w[j]
                                _s_mul = cute.make_tensor(cute.recast_ptr(sMxSF.iterator + 1056, dtype=Float32),
                                                          cute.make_layout((32, 2, 8), stride=(1, 32, 64)))
                                _s_mul = _s_mul[None, stage, wg_idx * 4 + dp_idx // 32]
                                if const_expr(self.ds_dq_rowlocal == "all" and self.token_local):
                                    # token-local K in the dQ GEMM (unit K scale): this key's factor into the dQ
                                    # side's values before their per-query exponent
                                    tl_vk = cute.make_rmem_tensor(cute.size(tdPrdP_cur), Float32)
                                    for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                        tl_vk[2 * v], tl_vk[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (tl_fk, tl_fk))
                                    e_q = self._ds_dq_rowlocal_words(tl_vk, tdPrdSq_w, dp_idx % 32, wps, _s_mul)
                                elif const_expr(self.ds_dq_rowlocal == "all"):
                                    e_q = self._ds_dq_rowlocal_words(tdPrdP_cur, tdPrdSq_w, dp_idx % 32, wps, _s_mul, cm=cm_q)
                                else:
                                    # a 32 x 32 block the causal diagonal crosses (warp-uniform: a warp's lanes are one key
                                    # block, a (stage, WG) one query block): query q sees key k iff k <= q + (s_k - s_q)
                                    _qb0 = (m_loc * self.tile_m + 32 * (2 * stage + wg_idx)
                                            + (mseqlen.seqlen_k - mseqlen.seqlen_q))
                                    _kb0 = ((k_tile_loc if const_expr(self.doc_t128) else
                                             n_block_for_cluster * self.cta_group_size + n_block % self.cta_group_size)
                                            * self.tile_n + (dp_idx // 32) * 32)
                                    if _kb0 + 31 > _qb0:
                                        if _kb0 <= _qb0 + 31:
                                            e_q = self._ds_dq_rowlocal_words(tdPrdP_cur, tdPrdSq_w, dp_idx % 32, wps, _s_mul)
                            tdPrdS_cvt = cute.recast_tensor(tdPrdS_w, self.ds_dtype)
                            if const_expr(stage == 0):
                                pipeline_dS.producer_acquire(producer_state_dS)
                            if const_expr(self.mxds):
                                # SF bytes into the (32 x 4)-byte SF atoms, offset (m % 32) * 16 + (m // 32) * 4 + k_group, K-atom 1 at +512.
                                # dK side: row = this key, group = this stage's 32 queries (2 * stage + wg). dQ side: row = the query
                                # (lane) within the 64-query half, group = this CTA's key block (rank * 4 + this warp's key group); the
                                # own-half stage goes to the own tile, the peer's half to the staging tile the xchg warp ships
                                _lane = dp_idx % 32
                                _kgrp = dp_idx // 32
                                e_u8 = e_r.to(cute.Uint8)
                                eq_u8 = e_u8  # the dQ side's byte
                                if const_expr(self.ds_dq_rowlocal != "off"):
                                    eq_u8 = e_q.to(cute.Uint8)
                                _offq = _lane * 16 + wg_idx * 4 + _kgrp + cta_rank_in_cluster * 512
                                if const_expr(not self.mx_xw_all):
                                    sSFdS_mx_u8[_lane * 16 + _kgrp * 4 + (2 * stage + wg_idx)] = e_u8
                                if const_expr(self.mx_xw_any):
                                    # one byte per (tile parity, stage, wg, key group); every lane writes the same byte
                                    sMxSF[phase_Sl * 16 + stage * 8 + wg_idx * 4 + _kgrp] = e_u8
                                    if const_expr(self.ds_dq_rowlocal != "off"):
                                        # the per-query bytes the exchange warp assembles the dQ-side tiles from
                                        sMxSF[32 + ((phase_Sl * 2 + stage) * 32 + _lane) * 8 + wg_idx * 4 + _kgrp] = eq_u8
                                elif exchange_stage == stage:
                                    sSFq_xchg_u8[_offq] = eq_u8
                                else:
                                    sSFdSq_u8[_offq] = eq_u8

                            # RMEM->TMEM: always write to TMEM for MMA
                            if const_expr(not self.warp_sync):
                                cute.copy(thr_copy_r2t, tSrdS8_f32[None, stage, None, None], tdPtdS_r2t[None, stage, None, None])

                            # RMEM->SMEM: the own query half (stage == rank) into keys block `rank` of the dQ A tile, the other
                            # half into the exchange buffer (same byte arrangement as the peer's keys block `rank`)
                            tdPrdS_src = cute.make_tensor(cute.recast_ptr(tdPrdSq_w.iterator, dtype=self.ds_dtype), tSdS_own_r2s.shape)
                            if (wg_idx != cta_rank_in_cluster) if const_expr(self.ld_rep == 64) else (exchange_stage == stage):
                                cute.copy(thr_ds_r2s, tdPrdS_src, tSdS_xchg_r2s)
                            else:
                                cute.copy(thr_ds_r2s, tdPrdS_src, tSdS_own_r2s)

                if const_expr(self.warp_sync):
                    cute.arch.mbarrier_wait(dP_loaded_mbar_ptr, phase_dPl)
                    phase_dPl ^= 1
                    cute.copy(thr_copy_r2t, tSrdS8_f32, tdPtdS_r2t)
                if const_expr(not self.use_smem_dS_for_mma_dK):
                    cute.arch.fence_view_async_tmem_store()

                if const_expr(self.use_2cta_instrs):
                    # use pipeline_dP to signal tmem store of dS
                    with cute.arch.elect_one():
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                consumer_state_S_P_dP.advance()

                # After the loop: copy exchange registers to sdS_xchg buffer
                if const_expr(self.use_2cta_instrs):
                    # when hdim 192, sdQaccum overlapped with sdS_xchg
                    if const_expr(self.tile_hdim == 192):
                        cute.arch.mbarrier_wait(
                            dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                        )

                cute.arch.fence_view_async_shared()
                if const_expr(self.warp_sync):
                    # the exchange warp issues the DSMEM copy once every warp has written its half
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_stored_mbar_ptr)
                else:
                    self._csb()
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                # when 2cta hdim 128, pipeline_dS also signals S tmem load completion so is deferred
                if const_expr(not (self.use_2cta_instrs and self.tile_hdim == 128)):
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                # 2-CTA: DSMEM copy from sdS_xchg to peer's sdS buffer
                if const_expr(self.use_2cta_instrs):
                    stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                    stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                    if const_expr(not self.warp_sync) and tidx == 0:
                        peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                        smem_src_ptr = sdS_xchg.iterator
                        # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                        smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )
                        copy_utils.cpasync_bulk_s2cluster(
                            smem_src_ptr,
                            smem_dst_ptr,
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )

            # Final signal for dS smem store completion
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                if process_tile:
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                    )
                else:
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        dv_scale,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        "V",
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        dk_scale,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        "K",
                    )
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity, no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_local or self.is_varlen_q):
                    should_zero_dKV = m_block_min >= m_block_max
                if const_expr(self.use_block_sparsity):
                    # For block sparsity, zero when no m_blocks contribute to this n_block
                    if not process_tile:
                        should_zero_dKV = True

                if should_zero_dKV:
                    # For 2-CTA: use cluster-wide tile size (cta_group_size * tile_n)
                    cluster_tile_n = self.tile_n * self.cta_group_size
                    n_block_for_tile = n_block // self.cta_group_size
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]
                    gdK = cute.local_tile(
                        mdK_cur, (cluster_tile_n, self.tile_hdim), (n_block_for_tile, 0)
                    )
                    gdV = cute.local_tile(
                        mdV_cur, (cluster_tile_n, self.tile_hdimv), (n_block_for_tile, 0)
                    )
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dK, zero, tdKgdK[None, i, j])
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dV, zero, tdVgdV[None, i, j])

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _dq_semaphore_lock_value(
        self,
        iter_idx: Int32,
        curr_q_cnt: Int32,
        curr_dq_write_order: Optional[cute.Tensor],
        curr_dq_write_order_full: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        seqlen,
        m_block: Int32,
        n_block: Int32,
        n_block_global_max: Int32,
        mseqlen=None,
        m_blk_off=None,
        n_blk_loc=None,
    ) -> Int32:
        lock_value = n_block
        if const_expr(self.doc_varlen):
            # documents: this key pair's rank among the key pairs of ITS document that meet m_block (the 1-CTA
            # kernel's doc-local lock, flash_bwd_sm100_fp4 dQacc_reduce): ascending from the document's first pair, or
            # descending under spt. The global pair index would wait on pairs of earlier documents that never arrive.
            _n_lo, _n_hi = block_info.block_info.get_n_block_min_max(mseqlen, m_block - m_blk_off)
            lock_value = _n_hi - 1 - n_blk_loc if const_expr(self.spt) else n_blk_loc - _n_lo
        elif const_expr(self.spt):
            # the roles carry lp_bwd_tiles.LPTileInfo; the key-pair range of this query tile comes from the
            # wrapped BlockInfo (key pairs of 256), as in the 1-CTA kernel's lock
            bi = block_info.block_info if const_expr(isinstance(block_info, (LPTileInfo, LPPairTileInfo))) else block_info
            _, n_block_max_for_m_block = bi.get_n_block_min_max(seqlen, m_block)
            lock_value = n_block_max_for_m_block - 1 - n_block
        if const_expr(self.use_block_sparsity):
            assert blocksparse_tensors is not None
            if const_expr(blocksparse_tensors.dq_write_order is not None):
                sparse_iter = iter_idx // self.subtile_factor
                if sparse_iter < curr_q_cnt:
                    assert curr_dq_write_order is not None
                    lock_value = curr_dq_write_order[sparse_iter]
                else:
                    assert curr_dq_write_order_full is not None
                    lock_value = curr_dq_write_order_full[sparse_iter - curr_q_cnt]
        return lock_value

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.atom.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        dq_scale=None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # For 2-CTA: reduce_stage = dQaccum_reduce_stage_t2r / cta_group_size
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // self.cta_group_size  # for 2cta
            seqlen = SeqlenInfoCls(batch_idx)
            mseqlen, m_block_min, m_block_max, m_blk_off, n_blk_loc = block_info.get(batch_idx, n_block_cta_group)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx]
                )
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )

            if const_expr(self.deterministic):
                assert mdQ_semaphore is not None
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]
            dq_row = dq_scale
            if const_expr(self.dq_tile_scaled and not self.doc_varlen):
                # dqfix, batched: the batch row's scale, after the vector and the two (b tq,) planes
                _nt = mdQaccum.shape[2] * (mdQaccum.shape[0] // (self.tile_m * self.tile_hdim))
                dq_row = self._dq_tab[8 + 2 * _nt + batch_idx]

            # doc_t128: a straddling pair skips hull tiles it does not meet (no wait, no count). The previous ACTIVE
            # tile (dq_prev) is released at the start of the next tile's reduce, BEFORE that tile's lock wait: its
            # reduce-add has overlapped this tile's TMEM load, and the release never waits on the next lock (measured
            # equal to the immediate release on documents; the lock's document lookups are per work tile, dq_lock_prep)
            # (on batched tiles the same early release measured +3..+14 % against upstream's delayed one: not used)
            delay_t128 = (self.deterministic and self.doc_t128 and not self.tile_hdim == 192
                          and not self.use_block_sparsity)
            delay_semaphore_release = (not self.tile_hdim == 192 and not self.use_block_sparsity
                                       and not self.doc_t128 and not delay_t128)
            dq_prev = None
            if const_expr(delay_t128):
                dq_prev = Int32(-1)
            dq_prep = None
            if const_expr(self.doc_t128 and self.deterministic):
                dq_prep = block_info.dq_lock_prep(n_block_cta_group)
            n_block_global_max = cute.ceil_div(seqlen.seqlen_k, self.tile_n)

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            curr_dq_write_order = None
            curr_dq_write_order_full = None
            loop_count = m_block_max - m_block_min
            process_tile = (
                const_expr(not self.dyn_tiles and not self.is_varlen_q) or m_block_min < m_block_max
            )
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)
            if const_expr(self.deterministic and self.use_block_sparsity):
                assert blocksparse_tensors is not None
                if const_expr(blocksparse_tensors.dq_write_order is not None):
                    assert blocksparse_tensors.dq_write_order is not None
                    curr_dq_write_order = blocksparse_tensors.dq_write_order[
                        batch_idx, head_idx, n_block, None
                    ]
                    if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                        assert blocksparse_tensors.dq_write_order_full is not None
                        curr_dq_write_order_full = blocksparse_tensors.dq_write_order_full[
                            batch_idx, head_idx, n_block, None
                        ]

            # dQacc_reduce mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                m_block_oob_upper = False
                if const_expr(self.use_block_sparsity):
                    m_block, _ = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    m_block_oob_upper = m_block >= m_block_max
                dq_scale_pre = dq_scale
                if const_expr(self.dq_tile_causal != 0):
                    # dq-causal: the scale table entry, loaded BEFORE the wait for the dQ tile (its latency hides behind it)
                    _mb = m_block
                    if m_block_max > 0:
                        _mb = cutlass.min(m_block, m_block_max - 1)
                    _tq = mdQaccum.shape[0] // (self.tile_m * self.tile_hdim)
                    if const_expr(self.dq_tile_causal == 2):
                        # per query ROW (the plane b s_q + 128 m_block + row): CTA rank r of the pair reduces rows
                        # 64 r .. 64 r + 63 of the tile, thread u of the 128 holds row u % 64 (columns by u // 64)
                        dq_scale_pre = self._dq_tab[8 + batch_idx * _tq * self.tile_m + _mb * self.tile_m
                                                    + 64 * cta_rank_in_cluster + tidx % 64]
                    else:
                        dq_scale_pre = self._dq_tab[8 + batch_idx * _tq + _mb]   # per 128-row tile (b tq + m_block)
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                if m_block_max > 0:
                    m_block = cutlass.min(m_block, m_block_max - 1)
                gdQaccum_cur = gdQaccum[None, None, m_block]
                dq_scale_t = dq_scale
                if const_expr(self.dq_tile_scaled and self.doc_varlen):
                    # dqfix: documents: this query tile's row scale (device_scales._dq_tile_table's per-tile plane,
                    # b tq + m_block; a straddling pair's tile belongs to one document)
                    assert not seqlen.has_cu_seqlens_q
                    _tq = mdQaccum.shape[0] // (self.tile_m * self.tile_hdim)
                    dq_scale_t = self._dq_tab[8 + batch_idx * _tq + m_block]
                if const_expr(self.dq_tile_causal != 0):
                    dq_scale_t = dq_scale_pre   # dq-causal: loaded before the wait for the dQ tile
                elif const_expr(self.dq_tile_scaled and not self.doc_varlen):
                    dq_scale_t = dq_row   # batched: the work tile's batch row (read once per work tile)
                dq_active = None
                dq_lock = None
                if const_expr(self.doc_t128 and self.deterministic):
                    dq_active, dq_lock = block_info.dq_lock(dq_prep, n_block_cta_group, m_block, self.spt)

                tdQrdQ_shape = (
                    self.dQ_reduce_ncol,
                    self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
                )
                tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                    if const_expr(self.dqaccum_dtype.width == 16):
                        # f16 partials (half the reduce-add bytes): E4M3 code products -> dS K units first
                        tdQrdQ_f32 = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                        for i in cutlass.range_constexpr(cute.size(tdQrdQ_f32) // 2):
                            tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1] = cute.arch.mul_packed_f32x2(
                                (tdQrdQ_f32[2 * i], tdQrdQ_f32[2 * i + 1]), (dq_scale_t, dq_scale_t)
                            )
                        tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dqaccum_dtype)
                        utils.cvt_f16(tdQrdQ_f32, tdQrdQ_r2s)
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_view_async_shared()
                    # semaphore acquire
                    if const_expr(delay_t128 and stage == 0):
                        if dq_prev >= 0:
                            if is_tma_warp:
                                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            self.reduce_sync_barrier.arrive_and_wait()
                            barrier.arrive_inc(mdQ_semaphore_cur[(dq_prev, None)].iterator, tidx,
                                               cta_rank_in_cluster, 1)
                            dq_prev = Int32(-1)
                    if const_expr(self.deterministic and stage == 0 and self.doc_t128):
                        if dq_active:
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator, tidx, cta_rank_in_cluster, dq_lock,
                            )
                    elif const_expr(self.deterministic and stage == 0):
                        if not m_block_oob_upper:
                            lock_value = self._dq_semaphore_lock_value(
                                iter_idx,
                                curr_q_cnt,
                                curr_dq_write_order,
                                curr_dq_write_order_full,
                                blocksparse_tensors,
                                block_info,
                                seqlen,
                                m_block,
                                n_block_cta_group,
                                n_block_global_max,
                                **({"mseqlen": mseqlen, "m_blk_off": m_blk_off, "n_blk_loc": n_blk_loc}
                                   if const_expr(self.doc_varlen) else {}),
                            )
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                lock_value,
                            )
                    self.reduce_sync_barrier.arrive_and_wait()
                    # Copy from shared memory to global memory
                    if const_expr(self.doc_t128 and self.deterministic):
                        dq_issue = is_tma_warp and not m_block_oob_upper and dq_active
                    else:
                        dq_issue = is_tma_warp and not m_block_oob_upper
                    if dq_issue:
                        with cute.arch.elect_one():
                            if const_expr(self.dqaccum_dtype.width == 16):
                                copy_utils.cpasync_reduce_bulk_add_f16(
                                    sdQaccum[None, smem_idx].iterator,
                                    gdQaccum_cur[None, stage + stage_offset].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            else:
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, smem_idx].iterator,
                                    gdQaccum_cur[None, stage + stage_offset].iterator,
                                    self.tma_copy_bytes["dQ"] // 1,
                                )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
                    elif is_tma_warp:
                        # Drain pending TMA stores so SMEM buffers are safe to reuse
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()

                    if const_expr(delay_t128 and stage == 0 and self.doc_t128):
                        if dq_active:
                            dq_prev = m_block
                    if const_expr(self.deterministic and stage == 0 and delay_semaphore_release):
                        if m_block > m_block_min:
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                1,
                            )

                if const_expr(self.tile_hdim == 192):
                    if const_expr(self.sdQaccum_stage > 1):
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dQaccum_empty_mbar_ptr)

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic and not delay_semaphore_release and not delay_t128):
                    if const_expr(self.sdQaccum_stage > 1 and not self.tile_hdim == 192):
                        if is_tma_warp and not m_block_oob_upper:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    if const_expr(self.doc_t128):
                        dq_rel = not m_block_oob_upper and dq_active
                    else:
                        dq_rel = not m_block_oob_upper
                    if dq_rel:
                        barrier.arrive_inc(
                            mdQ_semaphore_cur[m_block, None].iterator, tidx, cta_rank_in_cluster, 1
                        )

            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
                # final semaphore release
                if const_expr(delay_t128):
                    if dq_prev >= 0:
                        barrier.arrive_inc(mdQ_semaphore_cur[(dq_prev, None)].iterator, tidx, cta_rank_in_cluster, 1)
                if const_expr(self.deterministic and delay_semaphore_release):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block_max - 1, None)].iterator,
                        tidx,
                        cta_rank_in_cluster,
                        1,
                    )

            if const_expr(
                self.deterministic
                and not self.spt
                and not self.use_block_sparsity
                and (block_info.block_info if const_expr(isinstance(block_info, (LPTileInfo, LPPairTileInfo))) else block_info).window_size_left
                is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator, tidx, cta_rank_in_cluster, 1
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.atom.ThrMma,
        thr_mma_dK: cute.atom.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.mma_tiler_pdo[0], self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block // self.cta_group_size]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        if tidx < seqlen.seqlen_k - self.tile_n * n_block:
            cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.mma_tiler_dsq[0], self.tile_hdim), (None, 0))
        gdK_tile = gdK[None, None, n_block // self.cta_group_size]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        if tidx < seqlen.seqlen_k - self.tile_n * n_block:
            cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma: cute.atom.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        )
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
            )  # (tile_n, hdim) - per CTA
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV, epi_tile, (0, None)
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            # n_block_group = n_block // self.cta_group_size
            if const_expr(not seqlen.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset(
                    (seqlen.padded_offset_k * tile_hdim,), mdKV[None, head_idx_kv]
                )
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * tile_hdim,), (n_block,)
            )  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * tile_hdim // num_wg,))[
                ((None, wg_idx),)
            ]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV, (flat_epi_tile,)
            )  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            if const_expr(K_or_V == "K"):
                assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong (K)"
            else:
                assert num_epi_stages == self.num_epi_stages_v, "Epi stage calculation is wrong (V)"
        else:
            num_epi_stages = (
                self.num_epi_stages if const_expr(K_or_V == "K") else self.num_epi_stages_v
            )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32
        )

        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # qk_hadamard: dK came from the rotated Q operand (dK R); load both halves of the row and undo the rotation in
        # registers (qk_hadamard_cute.dk_unrotated_half), then store from those values below
        had = const_expr(K_or_V == "K" and self.had_dk is not None)
        if const_expr(had):
            assert num_wg == 2 and tile_hdim == 128 and not self.dKV_postprocess, (
                "fused dK back-rotation: MHA, two compute warpgroups, d = 128")
            thr_copy_h = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            src_h = thr_copy_h.partition_S(tdKVtdKV)
            cd_h = self.split_wg(thr_copy_h.partition_D(thr_mma.partition_C(cute.make_identity_tensor(
                (cta_group_tile_n, tile_hdim)))), wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                cd_h = cd_h[None, 0]
            had_n = cute.size(cd_h.shape)  # this warpgroup's columns per stage
            assert num_epi_stages * had_n == 64, "fused dK back-rotation: 64 columns per warpgroup"
            had_u = cute.make_rmem_tensor(cute.make_layout(64), Float32)
            dk_unrotated_half(thr_copy_h, self.split_wg(src_h, 0, num_wg)[None, None, 0, 0],
                              self.split_wg(src_h, 1, num_wg)[None, None, 0, 0], cd_h, had_u, wg_idx, scale,
                              num_epi_stages, had_n, self.had_dk)

        # semaphore acquire
        tile_live = None
        if const_expr(self.key_tail and self.dKV_postprocess):
            tile_live = n_block * self.tile_n < seqlen.seqlen_k
        if const_expr(deterministic_KV and self.key_tail):
            if tile_live:
                barrier.wait_eq(
                    mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
                )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
        elif const_expr(deterministic_KV):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            if const_expr(had):
                for i in cutlass.range_constexpr(had_n):
                    tdKVrdKV_t2r[i] = had_u[epi_stage * had_n + i]
            else:
                cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
                cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None and not had):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                elif const_expr(self.key_tail):
                    with cute.arch.elect_one():
                        if tile_live:
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdKV.iterator,
                                gdKV_epi[None, epi_stage].iterator,
                                self.tma_copy_bytes["dKacc"],
                            )
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            if const_expr(self.key_tail):
                if tile_live:
                    barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)
            else:
                barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

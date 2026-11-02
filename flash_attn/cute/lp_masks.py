# The masking / score-modification arguments of the low-precision backwards, as one explicit, immutable
# value. It is built once at the entry point (the autograd forward), saved in the autograd context, and handed to every
# backward driver as an argument -- never through a global configuration or the environment.
#
#   window        (left, right) sliding window, upstream semantics (_resolve_causal_local_window): causal + window
#                 becomes a local window with right = 0.
#   softcap       folded into the score_mod pair (upstream: utils.create_softcap_scoremod / _bwd).
#   score_mod     upstream score_mod(score * softmax_scale, b, h, q_idx, kv_idx, seqlen_info, aux_tensors) and its
#                 score_mod_bwd (grad, score, ...) -- the backward's dS.
#   mask_mod      upstream mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors), evaluated on every tile of the
#                 range (no block sparsity in the low-precision backwards). Causal / window go inside the mask_mod.
#   docs          aligned-packed varlen (lp_varlen.LPDocs): the document tables of the kernel.
#   spt           deterministic dQ order override (upstream flash_bwd_sm100 `spt`).
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch

from flash_attn.cute import utils


def resolve_window(causal: bool, window_size: Tuple[Optional[int], Optional[int]]):
    """-> (causal, local, left, right), upstream's rule."""
    left, right = window_size if window_size is not None else (None, None)
    if left is not None and right is not None and left < 0 and right < 0:
        left = right = None
    if left is not None and left < 0:
        left = None
    if right is not None and right < 0:
        right = None
    if causal:
        right = 0
    if left is not None or right is not None:
        if left is None and right == 0:
            return True, False, None, None
        return False, True, left, right
    return causal, False, None, None


@dataclass(frozen=True)
class LPMasks:
    causal: bool = False
    local: bool = False
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    softcap: float = 0.0
    score_mod: Optional[Callable] = None
    score_mod_bwd: Optional[Callable] = None
    mask_mod: Optional[Callable] = None
    aux_tensors: Optional[tuple] = field(default=None, compare=False, hash=False)
    docs: Optional[object] = field(default=None, compare=False, hash=False)  # lp_varlen.LPDocs
    spt: Optional[bool] = None

    @staticmethod
    def build(causal: bool = False, window_size=(None, None), softcap: float = 0.0, score_mod=None, score_mod_bwd=None,
              mask_mod=None, aux_tensors=None, docs=None, spt=None) -> "LPMasks":
        if softcap is None:
            softcap = 0.0
        if softcap != 0.0:
            if score_mod is not None or score_mod_bwd is not None:
                raise ValueError("softcap is a score_mod: pass one or the other")
            score_mod = utils.create_softcap_scoremod(float(softcap))
            # (the packed forward's S / c: a tag for flash_fwd_sm100_fp4's padding-row zeroing, cov4)
            score_mod._lp_softcap = True
            score_mod_bwd = utils.create_softcap_scoremod_bwd(float(softcap))
        if (score_mod is None) != (score_mod_bwd is None):
            raise ValueError("score_mod needs score_mod_bwd (the backward applies it to dS), as upstream")
        causal, local, wl, wr = resolve_window(causal, window_size)
        if mask_mod is not None and (causal or local):
            raise ValueError("mask_mod with causal / window_size: express the causal / window condition inside the "
                             "mask_mod (the low-precision kernels, like upstream's backward, apply one or the other)")
        if aux_tensors is not None:
            aux_tensors = tuple(aux_tensors)
        return LPMasks(causal=causal, local=local, window_left=wl, window_right=wr, softcap=float(softcap),
                       score_mod=score_mod, score_mod_bwd=score_mod_bwd, mask_mod=mask_mod, aux_tensors=aux_tensors,
                       docs=docs, spt=spt)

    @property
    def any_mod(self) -> bool:
        return self.score_mod is not None or self.mask_mod is not None

    @property
    def is_trivial(self) -> bool:
        """Nothing beyond plain causal / dense: the existing kernels and keys, bitwise."""
        return not (self.local or self.any_mod or self.docs is not None or self.spt is not None)

    def ctor_kwargs(self) -> dict:
        """Constructor arguments of the 1-CTA / 2-CTA low-precision backward kernels."""
        return dict(is_causal=self.causal, is_local=self.local, mask_mod=self.mask_mod, score_mod=self.score_mod,
                    score_mod_bwd=self.score_mod_bwd, has_aux_tensors=self.aux_tensors is not None,
                    doc_varlen=self.docs is not None, spt=self.spt)

    def key(self) -> tuple:
        """Every compile-time item (the window values and the tables are runtime)."""
        h = lambda f: None if f is None else utils.hash_callable(f)  # noqa: E731
        aux_meta = None if self.aux_tensors is None else tuple((t.dtype, t.dim()) for t in self.aux_tensors)
        return ("lpmasks", self.causal, self.local, self.window_left is not None, self.window_right is not None,
                h(self.score_mod), h(self.score_mod_bwd), h(self.mask_mod), aux_meta, self.docs is not None, self.spt)

    def window_args(self, placeholder: bool = False):
        from cutlass import Int32
        wl = None if self.window_left is None else (Int32(0) if placeholder else self.window_left)
        wr = None if self.window_right is None else (Int32(0) if placeholder else self.window_right)
        return wl, wr

    def table_args(self):
        if self.docs is None:
            return None, None
        return self.docs.doc_tab, self.docs.tile_doc(128)

    def aux_list(self):
        return None if self.aux_tensors is None else list(self.aux_tensors)


def lp_masks_or_none(masks: Optional[LPMasks], causal: bool) -> Optional[LPMasks]:
    """None for the plain causal / dense call (the existing code path, bitwise), else the masks with causal
    consistent with the call's own flag."""
    if masks is None or masks.is_trivial:
        return None
    return masks


def to_cute_aux(aux_tensors):
    """The aux tensors as cute tensors (compile-time stand-ins keep dtype / rank; upstream to_cute_aux_tensor)."""
    if aux_tensors is None:
        return None
    from flash_attn.cute.cute_dsl_utils import to_cute_aux_tensor
    return [to_cute_aux_tensor(t) for t in aux_tensors]


def check_torch_int32(name: str, t: Optional[torch.Tensor]):
    if t is not None and t.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {t.dtype}")


def fwd_mask_kwargs(masks: Optional[LPMasks], b: int, h: int, s_q: int, s_k: int, device) -> dict:
    """The low-precision forward driver's (interface_fwd_fp4._flash_attn_fwd) arguments for these masks. The forward and
    the backward see the same window / score_mod / mask_mod; the forward's mask_mod runs through block sparsity (its
    only mask_mod path), computed on the GPU from the mask_mod itself (compute_block_sparsity, no host read)."""
    if masks is None:
        return {}
    kw = dict(causal=masks.causal)
    if masks.local:
        kw.update(window_size_left=masks.window_left, window_size_right=masks.window_right)
    if masks.score_mod is not None:
        kw["score_mod"] = masks.score_mod
    if masks.aux_tensors is not None:
        kw["aux_tensors"] = list(masks.aux_tensors)
    if masks.mask_mod is not None:
        # evaluated on every tile by the fork forward (its block-sparse path is stale against block_sparse_utils)
        kw["mask_mod"] = masks.mask_mod
        kw["pack_gqa"] = False  # the fork forward's mask_mod sees one q head per row (pack_gqa is refused with it)
    return kw

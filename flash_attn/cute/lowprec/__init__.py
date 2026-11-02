"""Low-precision attention for Blackwell: flash_attn_fp4 (NVFP4) and flash_attn_fp8 (E4M3).

    from flash_attn.cute.lowprec import flash_attn_fp4, flash_attn_fp8, FP4AttnOptions, FP8AttnOptions

Every option is an explicit argument; nothing is read from the environment or a global configuration. Each call
resolves one immutable plan (``resolve_fp4`` / ``resolve_fp8``, public for inspection: ``plan.log()`` names every
resolved choice and its reason, ``plan.specs`` the kernels' settings) that the autograd context carries to the
backward. Invalid combinations raise before anything runs, with the rule they break (``exc.rule``).

``KernelTuning`` (the kernels' measured constants, ``options.tuning=``) stays importable from here and from
``flash_attn.cute.lowprec.tuning``; it is not documented API and carries no stability promise.
"""
from .api import flash_attn_fp4, flash_attn_fp8
from .options import FP4AttnOptions, FP8AttnOptions, KernelTuning  # noqa: F401  (KernelTuning: undocumented)
from .plan import PENDING, AttnPlan, resolve_fp4, resolve_fp8  # noqa: F401  (PENDING: the not-yet-built table)
from ..lp_docmaps import DocMaps  # noqa: F401  (precomputed packed-varlen document maps: doc_maps=)

__all__ = ["flash_attn_fp4", "flash_attn_fp8", "FP4AttnOptions", "FP8AttnOptions", "AttnPlan", "resolve_fp4",
           "resolve_fp8", "DocMaps"]

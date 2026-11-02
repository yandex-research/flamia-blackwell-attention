"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

# Low-precision entry points resolve on first access rather than being imported here: they pull
# in the quantisers and the block-scaled drivers, which a caller who only wants the BF16 kernels
# has no reason to pay for -- or to have fail on.
_LAZY = {
    "flash_attn_fp4": ("flash_attn.cute.lowprec", "flash_attn_fp4"),
    "flash_attn_fp8": ("flash_attn.cute.lowprec", "flash_attn_fp8"),
    "FP4AttnOptions": ("flash_attn.cute.lowprec", "FP4AttnOptions"),
    "FP8AttnOptions": ("flash_attn.cute.lowprec", "FP8AttnOptions"),
    # plan inspection (public): the resolved plan of a call, without running it
    "resolve_fp4": ("flash_attn.cute.lowprec", "resolve_fp4"),
    "resolve_fp8": ("flash_attn.cute.lowprec", "resolve_fp8"),
    "AttnPlan": ("flash_attn.cute.lowprec", "AttnPlan"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    *_LAZY,
]

from pkgutil import extend_path

# look for every subdir with flash_attn base name such that fa2 and fa4 can be co-installed
__path__ = extend_path(__path__, __name__)

__version__ = "2.8.4"

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    # FA2's compiled extension (flash_attn_2_cuda) is optional: the FA4 CuTe kernels in
    # flash_attn.cute are pure Python/JIT and must remain importable in an environment where
    # only FA4 is installed. Anything that needs the FA2 entry points will fail at its own
    # import of flash_attn.flash_attn_interface with the original error.
    pass

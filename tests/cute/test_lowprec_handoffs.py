"""The Direct-P window guard that bypassed the configuration is gone for good.
No GPU needed.

(Options are explicit arguments; the library reads no environment variables, so there is no environment boundary
left to test.)
"""
import inspect


def test_directp_guard_is_gone():
    """The window-only classifier, its guard (host / GPU pair / runtime branch) and their knobs were removed: the 4-bit
    P^ forwards always run the clamped Mitchell classifier. No entry point of the guard may come back."""
    import flash_attn.cute.fp4_bwd_autograd as a
    import flash_attn.cute.interface_fwd_fp4 as f
    for mod, names in ((f, ("directp_clamp_for", "flash_attn_fwd_fp4_guarded", "quantize_fp4_qk_guarded")),
                       (a, ("directp_guard", "gpu_guard_on"))):
        for n in names:
            assert not hasattr(mod, n), n
    assert "directp_clamp" not in inspect.signature(f._flash_attn_fwd).parameters

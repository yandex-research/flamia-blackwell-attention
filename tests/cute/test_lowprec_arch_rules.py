"""SM100 / SM103 (B200 / B300) wiring of the low-precision forward, CPU only.

Every SM103 rule is resolved from one value per call -- FLASH_ATTENTION_ARCH, else the current device -- which is in the
compile key and the kernel's constructor argument; the JIT caches (in memory and on disk) are keyed by the target
architecture, so sm_100a and sm_103a binaries never share an entry.
"""
import pytest

from flash_attn.cute.lowprec import specs as S


@pytest.fixture
def arch(monkeypatch):
    from flash_attn.cute.interface import _get_device_arch   # process-cached: re-read after every override

    def set_(a):
        monkeypatch.setenv("FLASH_ATTENTION_ARCH", a)
        _get_device_arch.cache_clear()
    yield set_
    _get_device_arch.cache_clear()


@pytest.mark.parametrize("a, want", [("sm_100a", "sm100"), ("sm_103a", "sm103"), ("sm_103f", "sm103"), ("103", "sm103"),
                                     ("sm100", "sm100")])
def test_target_arch_override(arch, a, want):
    from flash_attn.cute.cache_utils import target_arch
    arch(a)
    assert target_arch() == want


def test_jit_cache_is_keyed_by_arch(arch):
    from flash_attn.cute.cache_utils import JITCache
    c = JITCache()
    arch("sm_100a")
    c[("k", 1)] = "sm100-binary"
    assert ("k", 1) in c and c[("k", 1)] == "sm100-binary"
    arch("sm_103a")
    assert ("k", 1) not in c          # a B300 call never gets the B200 binary
    c[("k", 1)] = "sm103-binary"
    assert c[("k", 1)] == "sm103-binary"
    arch("sm_100a")
    assert c[("k", 1)] == "sm100-binary"


def test_persistent_cache_hash_is_keyed_by_arch(arch, tmp_path):
    from flash_attn.cute.cache_utils import JITPersistentCache
    c = JITPersistentCache(tmp_path)
    arch("sm_100a")
    h100 = c._key_to_hash(("k", 1))
    arch("sm_103a")
    h103 = c._key_to_hash(("k", 1))
    assert h100 != h103


def test_is_sm103_override(arch):
    from flash_attn.cute.flash_fwd_sm100_fp4 import _is_sm103
    arch("sm_103a")
    assert _is_sm103()
    arch("sm_100a")
    assert not _is_sm103()


def test_rules():
    assert S.FwdKernelSpec.rule_ldred_rowmax(True) and not S.FwdKernelSpec.rule_ldred_rowmax(False)
    # SM103: no FMA-pipe exp2 emulation on the pure-E4M3 generic step; SM100: every 8th pair on the short dense window
    assert S.FwdKernelSpec.rule_fp8_generic_e2e(True, True) == 0
    assert S.FwdKernelSpec.rule_fp8_generic_e2e(True, False) == 0
    assert S.FwdKernelSpec.rule_fp8_generic_e2e(False, True) == 8
    assert S.FwdKernelSpec.rule_fp8_generic_e2e(False, False) == 6


def test_tuning_table_per_arch():
    from flash_attn.cute.flash_fwd_sm100_fp4 import _fp4_tuning_config
    for causal in (False, True):
        assert _fp4_tuning_config(False)[(causal, 128)]["enable_e2e"] is True    # SM100: BF16-P.V exp2 emulation on
        assert _fp4_tuning_config(True)[(causal, 128)]["enable_e2e"] is False    # SM103: off (2x MUFU.EX2)


@pytest.mark.parametrize("sm103", [False, True])
def test_kernel_takes_the_arch_argument(arch, sm103):
    """The constructor argument decides, whatever the process's device or override says."""
    try:
        import cutlass
        from flash_attn.cute.flash_fwd_sm100_fp4 import FlashAttentionForwardSm100 as FW
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CuTe DSL not importable: {e}")
    arch("sm_100a" if sm103 else "sm_103a")   # the opposite: must be ignored
    k = FW(128, 128, is_causal=False, sf_dtype=cutlass.Float8E4M3FN, sf_vec_size=16, spec=S.FwdKernelSpec(),
           is_sm103=sm103)
    assert k.is_sm103 is sm103
    assert k.use_ldred_rowmax is sm103
    assert k.fp8_generic_e2e_default == (0 if sm103 else 6)
    assert k._tune.get("enable_e2e") is (not sm103)

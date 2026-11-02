"""The p_hat_frame defaults: "auto" resolves to the row-frame P^ on the 8-bit entry and to
the group frame (one UE8M0 per 32 keys; the row frame opt-in) on the 4-bit entry.

Evidence (four training seeds per entry): the row frame is 17-21 % faster in the forward kernel; its validation-loss
cost is noise on the 8-bit entry (+0.0011) but consistent on the 4-bit entry (+0.0015, worse in 4 of 4 seeds).
CPU only (meta tensors).
"""
import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, FP8AttnOptions, resolve_fp4, resolve_fp8
from flash_attn.cute.lowprec import specs as S


def _qkv(rg=True):
    return [torch.empty(1, 1024, 2, 128, dtype=torch.bfloat16, device="meta").requires_grad_(rg) for _ in range(3)]


def _frame(plan):
    hits = [(v, why) for f, v, why in plan.resolution_log if f == "p_hat_frame"]
    assert len(hits) == 1, plan.log()
    return hits[0]


def test_option_defaults_are_auto():
    assert FP4AttnOptions().p_hat_frame == "auto" and FP8AttnOptions().p_hat_frame == "auto"
    assert S.FWD8_ROW_P_DEFAULT is True and S.FWD4_ROW_P_DEFAULT is False
    assert S.HostSpec().fwd8_row_p is S.FWD8_ROW_P_DEFAULT and S.HostSpec().fwd4_row_p is S.FWD4_ROW_P_DEFAULT


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rg", [True, False])
def test_auto_is_row_on_fp8_and_group_on_fp4(causal, rg):
    p8 = resolve_fp8(*_qkv(rg), causal=causal)
    v, why = _frame(p8)
    assert v == "row" and why.startswith("auto") and p8.specs.host.fwd8_row_p is True
    p4 = resolve_fp4(*_qkv(rg), causal=causal)
    v, why = _frame(p4)
    assert v == "group" and why.startswith("auto") and p4.specs.host.fwd4_row_p is False


def test_explicit_values_override_the_defaults():
    p = resolve_fp8(*_qkv(), options=FP8AttnOptions(p_hat_frame="group"))
    assert _frame(p) == ("group", "explicit") and p.specs.host.fwd8_row_p is False
    p = resolve_fp4(*_qkv(), options=FP4AttnOptions(p_hat_frame="row"))
    assert _frame(p) == ("row", "explicit") and p.specs.host.fwd4_row_p is True
    # the explicit default value is the same plan as "auto" (the same specs, one compiled kernel)
    assert resolve_fp8(*_qkv(), options=FP8AttnOptions(p_hat_frame="row")).specs == resolve_fp8(*_qkv()).specs
    assert resolve_fp4(*_qkv(), options=FP4AttnOptions(p_hat_frame="group")).specs == resolve_fp4(*_qkv()).specs

"""Two-value proof of the low-precision compile keys, as pytest, on the public entries.

The property a compile key has to have is not "it contains item X". It is:

  1. two distinct values of a setting that specialises the kernel produce **two distinct cache entries** in one
     process, and
  2. an **A then B then A** sequence reproduces A -- bitwise wherever A reproduces itself bitwise, and within A's own
     run-to-run spread where it does not (the dense dQ reduce-add is not reproducible; stochastic rounding draws a new
     dither per call) -- so the second A found A's entry and not B's, and nothing leaked between them.

(1) alone passes if the key is a nonce; (2) alone passes if the setting does nothing. Together they say the key
discriminates exactly the setting. Settings are given as options / KernelTuning to flash_attn_fp4 / flash_attn_fp8;
the plan builds the kernels' specs from them and the drivers key on spec.key() (CPU-side proof for every field:
test_lowprec_spec_keys.py).

  fwd            KernelTuning.fwd_nc_persistent True/False -> the forward's `is_persistent`
  bwd-2cta       KernelTuning.two_cta8_lpt_l2 True/False -> `lpt_l2` after the in-__init__ override
  bwd-1cta-fp4   options.sr_ds False/True -> Bwd1Spec.ds_sr, the 1-CTA backward of flash_attn_fp4
  bwd-1cta-fp8   KernelTuning.bwd_fp4_sload_early True/False -> Bwd1Spec.sload_early, flash_attn_fp8's 1-CTA
  bwd-fp4q       options.sr_dk_operand True/False -> BwdQSpec.dk_sr, the query-row backward

The options that reach a kernel only as an explicit driver argument (not through any spec field, so spec.key() cannot
cover them) get the same two-value proof on the driver cache that receives them:

  opt-ds_dq_rowlocal@bwd-1cta  plan.bwd.ds_dq_rowlocal -> _flash_attn_bwd_fp4(ds_dq_rowlocal=) -> resolved_key
                               "ds_dq_rowlocal" (flash_attn_fp4, two_cta="off")
  opt-ds_dq_rowlocal@bwd-2cta  the same argument -> flash_attn_bwd_2cta8 -> resolved_key "ds_dq_rowlocal" (flash_attn_fp8)
  opt-exp_mix@fwd              plan.bwd.exp_mix -> pv4_forward(p_mix=) -> the forward key's p_mix item (option A
                               inference)
  opt-exp_mix@bwd-fp4q         plan.bwd.exp_mix -> the query-row backward (p_mix=) -> resolved_key "p_mix_key" (all-4-bit)
  opt-p_consistency@fwd        option A inference: the forward always runs the reproduce-forward-p variant; the option
                               reaches it through exp_mix's auto ("EMMEMMEM" vs "none") -> the forward key's p_mix item
  opt-token_local@fwd          operand_scaling token_block / token_local -> the forward's tb_tok_v argument -> the
                               forward key's `tb_tok_v is not None` item (inference)
  opt-token_local@bwd-1cta     -> fp8_scales["mx"]["tok"] -> the 1-CTA key's `tok is not None` item (flash_attn_fp4)
  opt-token_local@bwd-2cta     -> the 2-CTA key's `tok is not None` item (flash_attn_fp8)

do_rounding has no driver: option A is inference only (plan rule 36), so an explicit "sr" is refused (rule E1;
test_lowprec_options_reach.py) rather than accepted as a no-op.
"""
import math

import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions as O4
from flash_attn.cute.lowprec import FP8AttnOptions as O8
from flash_attn.cute.lowprec import KernelTuning, flash_attn_fp4, flash_attn_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="SM 10.x kernels"
)

B, S, H, D = 1, 1024, 2, 128
SCALE = 1.0 / math.sqrt(D)
SLACK = 4.0  # room over the control's own sampled spread (a bitwise control stays bitwise: 0 * SLACK = 0)
ALL4 = dict(pv_precision="fp4", backward="fp4")
OPTA = dict(pv_precision="fp4")          # option A (inference only)
NOHAD = dict(qk_hadamard=False)
ROWQ = dict(ds_dq_rowlocal="all")        # token_local needs it


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(4242)
    return tuple(torch.randn(B, S, H, D, generator=g, device="cuda", dtype=torch.bfloat16) for _ in range(4))


def _max_abs_diff(xs, ys):
    return [(x.float() - y.float()).abs().max().item() for x, y in zip(xs, ys)]


# --------------------------------------------------------------------------- runners: (q, k, v, do) -> outputs
def _infer(entry, options, **kw):
    def run(q, k, v, do):
        out, lse = entry(q, k, v, softmax_scale=SCALE, causal=False, return_lse=True, options=options, **kw)
        torch.cuda.synchronize()
        return out.clone(), lse.clone()
    return run


def _train(entry, options, **kw):
    def run(q, k, v, do):
        qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        out = entry(qg, kg, vg, softmax_scale=SCALE, causal=False, options=options, **kw)
        grads = torch.autograd.grad(out, (qg, kg, vg), do)
        torch.cuda.synchronize()
        return tuple(t.clone() for t in grads)
    return run


def _cache(name):
    name = name.split("@")[-1]
    if name == "fwd":
        from flash_attn.cute.interface_fwd_fp4 import _flash_attn_fwd
        return _flash_attn_fwd.compile_cache
    if name == "bwd-2cta":
        from flash_attn.cute.interface_bwd_2cta8 import flash_attn_bwd_2cta8
        return flash_attn_bwd_2cta8.cache
    if name == "bwd-fp4q":
        from flash_attn.cute.interface_bwd_fp4q import _flash_attn_bwd_fp4q
        return _flash_attn_bwd_fp4q.cache
    from flash_attn.cute.interface_bwd_fp4 import _flash_attn_bwd_fp4
    return _flash_attn_bwd_fp4.cache


def _cases():
    kt = lambda **f: KernelTuning(**f)  # noqa: E731
    return {
        "fwd": (_infer(flash_attn_fp4, O4(tuning=kt(fwd_nc_persistent=True))),
                _infer(flash_attn_fp4, O4(tuning=kt(fwd_nc_persistent=False)))),
        "bwd-2cta": (_train(flash_attn_fp8, O8(two_cta="on", tuning=kt(two_cta8_lpt_l2=True))),
                     _train(flash_attn_fp8, O8(two_cta="on", tuning=kt(two_cta8_lpt_l2=False)))),
        # two_cta="off" pins the 1-CTA kernels: the test measures this key's item, not route selection
        "bwd-1cta-fp4": (_train(flash_attn_fp4, O4(two_cta="off", sr_ds=False)),
                         _train(flash_attn_fp4, O4(two_cta="off", sr_ds=True))),
        "bwd-1cta-fp8": (_train(flash_attn_fp8, O8(two_cta="off", tuning=kt(bwd_fp4_sload_early=True))),
                         _train(flash_attn_fp8, O8(two_cta="off", tuning=kt(bwd_fp4_sload_early=False)))),
        "bwd-fp4q": (_train(flash_attn_fp4, O4(sr_dk_operand=True), **ALL4),
                     _train(flash_attn_fp4, O4(sr_dk_operand=False), **ALL4)),
        # ---- options passed to the drivers as explicit arguments (no spec field)
        "opt-ds_dq_rowlocal@bwd-1cta": (_train(flash_attn_fp4, O4(two_cta="off", ds_dq_rowlocal="off")),
                                        _train(flash_attn_fp4, O4(two_cta="off", ds_dq_rowlocal="all"))),
        "opt-ds_dq_rowlocal@bwd-2cta": (_train(flash_attn_fp8, O8(two_cta="on", ds_dq_rowlocal="off")),
                                        _train(flash_attn_fp8, O8(two_cta="on", ds_dq_rowlocal="all"))),
        # (non-default mixes on both arms: "EMMEMMEM" / "none" are opt-p_consistency@fwd's two forwards. Every B arm must
        # be a kernel no earlier case of this process compiled -- hence also the order of CASES)
        "opt-exp_mix@fwd": (_infer(flash_attn_fp4, O4(exp_mix="MMMMEEEE"), **OPTA),
                            _infer(flash_attn_fp4, O4(exp_mix="EEEEMMMM"), **OPTA)),
        "opt-exp_mix@bwd-fp4q": (_train(flash_attn_fp4, O4(exp_mix="none"), **ALL4),
                                 _train(flash_attn_fp4, O4(exp_mix="MMMMEEEE"), **ALL4)),
        "opt-p_consistency@fwd": (_infer(flash_attn_fp4, O4(p_consistency="reproduce_forward_p"), **OPTA),
                                  _infer(flash_attn_fp4, O4(p_consistency="none"), **OPTA)),
        # qk_hadamard=False on both arms: token_local turns the default rotation off, so the arms differ in one item
        "opt-token_local@fwd": (_infer(flash_attn_fp4, O4(operand_scaling="token_block", **ROWQ), **NOHAD),
                                _infer(flash_attn_fp4, O4(operand_scaling="token_local", **ROWQ), **NOHAD)),
        "opt-token_local@bwd-1cta": (
            _train(flash_attn_fp4, O4(two_cta="off", operand_scaling="token_block", **ROWQ), **NOHAD),
            _train(flash_attn_fp4, O4(two_cta="off", operand_scaling="token_local", **ROWQ), **NOHAD)),
        "opt-token_local@bwd-2cta": (
            _train(flash_attn_fp8, O8(two_cta="on", operand_scaling="token_block", **ROWQ), **NOHAD),
            _train(flash_attn_fp8, O8(two_cta="on", operand_scaling="token_local", **ROWQ), **NOHAD)),
    }


CASES = ["fwd", "bwd-2cta", "bwd-1cta-fp4", "bwd-1cta-fp8", "bwd-fp4q",
         "opt-ds_dq_rowlocal@bwd-1cta", "opt-ds_dq_rowlocal@bwd-2cta", "opt-exp_mix@fwd", "opt-exp_mix@bwd-fp4q",
         "opt-p_consistency@fwd", "opt-token_local@fwd", "opt-token_local@bwd-1cta", "opt-token_local@bwd-2cta"]


@pytest.mark.parametrize("case", CASES)
def test_two_values_two_entries_and_a_again_finds_a(case):
    run_a, run_b = _cases()[case]
    cache = _cache(case)
    q, k, v, do = _inputs()
    # the reproducibility control first: A three times (these runs also populate A's entry)
    a_runs = [run_a(q, k, v, do) for _ in range(3)]
    spread = [0.0] * len(a_runs[0])
    for i in range(3):
        for j in range(i + 1, 3):
            spread = [max(s_, d_) for s_, d_ in zip(spread, _max_abs_diff(a_runs[i], a_runs[j]))]
    keys_a = set(cache.keys())
    run_b(q, k, v, do)
    keys_ab = set(cache.keys())
    a2 = run_a(q, k, v, do)
    keys_aba = set(cache.keys())
    assert keys_a < keys_ab and len(keys_ab) == len(keys_a) + 1, f"{case}: B did not get its own cache entry"
    assert keys_aba == keys_ab, f"{case}: the second A compiled again instead of finding A's entry"
    for n, (d_, sp) in enumerate(zip(_max_abs_diff(a_runs[0], a2), spread)):
        assert d_ <= sp * SLACK, f"{case}: output {n} after A-B-A differs by {d_:.3e} (A's own spread {sp:.3e})"

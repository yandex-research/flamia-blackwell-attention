"""options.save_bf16_inputs=False (the lean save): the 4-bit step without saved bf16 Q / K / V.

- gradients and output are bitwise identical to the default path (deterministic backward, where the default path
  reproduces itself bitwise);
- saved activation bytes drop from 1560 to 792 per (token, head) at head_dim 128.
These run on the per-tensor E4M3 operands (options.operand_scaling="per_tensor"), the route lean mode is for; the
default token-block route saves no bf16 Q / K / V at all (test_token_block_default_saves_no_bf16).

Every call is the public flash_attn_fp4 (E4M3 P.V forward, dk8 backward); its plan travels to the backward with the
autograd context.
"""
import pytest
import torch

from flash_attn.cute.lowprec import FP4AttnOptions, KernelTuning, flash_attn_fp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="NVFP4 kernels need SM 10.x"
)


def _run(q0, k0, v0, do, causal, lean, deterministic=True, operand_scaling="per_tensor", tuning=None):
    """operand_scaling defaults to per_tensor: the lean mode belongs to the per-tensor route."""
    q, k, v = (t.detach().clone().requires_grad_() for t in (q0, k0, v0))
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        seen[(st.data_ptr(), st.nbytes())] = st.nbytes()
        return t

    opts = FP4AttnOptions(operand_scaling=operand_scaling, save_bf16_inputs=not lean, tuning=tuning)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = flash_attn_fp4(q, k, v, causal=causal, deterministic=deterministic, options=opts)
    out.backward(do)
    return (out.detach(), q.grad, k.grad, v.grad), sum(seen.values())


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,s,h", [(1, 4096, 8), (2, 2048, 4)])
def test_save_lean_bitwise_and_smaller(b, s, h, causal):
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    ref, ref_bytes = _run(q, k, v, do, causal, lean=False)
    new, new_bytes = _run(q, k, v, do, causal, lean=True)
    for name, a, c in zip(("out", "dq", "dk", "dv"), ref, new):
        assert torch.equal(a, c), f"{name} differs with save_bf16_inputs=False"
    per = lambda nb: nb / (b * s * h)  # noqa: E731
    assert per(ref_bytes) == pytest.approx(1560.0, abs=1.0)
    assert per(new_bytes) == pytest.approx(792.0, abs=1.0)


def test_save_lean_keeps_bf16_outside_default_modes():
    """A non-default operand mode (D from the bf16 dO: KernelTuning(fp4_d_do8=False)) reads bf16 data: lean must fall back
    to saving q / k / v."""
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    _, nbytes = _run(q, k, v, do, False, lean=True, tuning=KernelTuning(fp4_d_do8=False))
    assert nbytes / (2048 * 4) == pytest.approx(1560.0, abs=1.0)


def test_token_block_default_saves_no_bf16():
    """The default operands (token blocks) save codes and factors only: 800.375 bytes per (token, head) at head_dim 128,
    with or without save_bf16_inputs, and the same output / gradients either way."""
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    tb = dict(operand_scaling="token_block")
    ref, ref_bytes = _run(q, k, v, do, False, lean=False, **tb)
    new, new_bytes = _run(q, k, v, do, False, lean=True, **tb)
    for name, a, c in zip(("out", "dq", "dk", "dv"), ref, new):
        assert torch.equal(a, c), f"{name} differs with save_bf16_inputs=False on the token-block route"
    assert ref_bytes / (2048 * 4) == pytest.approx(800.375, abs=1.0)
    assert new_bytes / (2048 * 4) == pytest.approx(800.375, abs=1.0)

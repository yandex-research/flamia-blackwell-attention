"""GQA on the 2-CTA E4M3 backward (bwdk, opt-in KernelTuning.bwd_2cta_gqa): the dK / dV post launch.

flash_bwd_sm100_2cta8 with qhead_per_kvhead > 1 (upstream's dKV_postprocess form) reduce-adds every q head's dK / dV
partial in fp32 into flat (b, h_k, s_k * 128) accumulators, in the epilogue's tile order: per 128-key tile (16384
floats), per warpgroup w (columns 64 w ..), per epilogue stage st (32 columns), the 128 threads' (= key rows') 32
values as 4-float chunks: offset(row, d) = 8192 (d // 64) + 4096 ((d % 64) // 32) + 512 ((d % 32) // 4) + 4 row + d % 4.
DKVPostprocess2CTA converts dK and dV in ONE launch (grid: key tiles x h_k x b x {K, V}), one thread per key row (32
16-byte loads per row, coalesced across the warp's rows; one 256-byte row store), optionally undoes the qk_hadamard
rotation on dK in registers (the fused dK back-rotation of the MHA epilogue needs the final dK) and writes zeros back
over what it read, so a pooled accumulator starts the next call clean without a fill launch."""
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
import cuda.bindings.driver as cuda

from flash_attn.cute.interface import torch2cute_dtype_map
from flash_attn.cute.interface_fwd_fp4 import to_cute_tensor
from flash_attn.cute.qk_hadamard_cute import unrotate_row128

_cache = {}
_POOL = {}


class DKVPostprocess2CTA:
    def __init__(self, out_dtype, zero_after: bool = True, had_signs=None):
        self.out_dtype = out_dtype
        self.zero_after = zero_after
        self.had_signs = had_signs

    @cute.jit
    def __call__(self, mAccK: cute.Tensor, mAccV: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor,
                 stream: cuda.CUstream):
        grid = [cute.ceil_div(mdK.shape[1], 128) * 2, mdK.shape[2], mdK.shape[0]]
        self.kernel(mAccK, mAccV, mdK, mdV).launch(grid=grid, block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mAccK: cute.Tensor, mAccV: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        tw, hh, bb = cute.arch.block_idx()
        t = tw // 2
        if tw % 2 == 0:
            self.row(mAccK, mdK, t, hh, bb, tidx, const_expr(self.had_signs is not None))
        else:
            self.row(mAccV, mdV, t, hh, bb, tidx, False)

    @cute.jit
    def row(self, mAcc: cute.Tensor, mOut: cute.Tensor, t, hh, bb, tidx, had: cutlass.Constexpr[bool]):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        atom_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=128)
        x = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        base = mAcc[bb, hh, None].iterator + (t * 16384 + 4 * tidx)
        for g in cutlass.range_constexpr(32):   # g = d // 4: chunk offset 8192 (g // 16) + 4096 ((g // 8) % 2) + 512 (g % 8)
            acc = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            off = 8192 * (g // 16) + 4096 * ((g // 8) % 2) + 512 * (g % 8)
            ptr = cute.make_ptr(Float32, (base + off).toint(), cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(4)), acc)
            for j in cutlass.range_constexpr(4):
                x[4 * g + j] = acc[j]
        if const_expr(self.zero_after):
            zero = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            zero.fill(0.0)
            for g in cutlass.range_constexpr(32):
                off = 8192 * (g // 16) + 4096 * ((g // 8) % 2) + 512 * (g % 8)
                ptr = cute.make_ptr(Float32, (base + off).toint(), cute.AddressSpace.gmem, assumed_align=16)
                cute.copy(atom, zero, cute.make_tensor(ptr, cute.make_layout(4)))
        if const_expr(had):
            unrotate_row128(x, Float32(1.0), self.had_signs)
        o = cute.make_rmem_tensor(cute.make_layout(128), self.out_dtype)
        o.store(x.load().to(self.out_dtype))
        out_it = mOut[bb, t * 128 + tidx, hh, None].iterator
        for k in cutlass.range_constexpr(16):
            optr = cute.make_ptr(self.out_dtype, (out_it + 8 * k).toint(), cute.AddressSpace.gmem, assumed_align=16)
            cute.copy(atom_o, cute.make_tensor(o.iterator + 8 * k, cute.make_layout(8)),
                      cute.make_tensor(optr, cute.make_layout(8)))


def take_zeroed_dkv_accum(b: int, hk: int, n: int, device):
    """Two zero (b, h_k, n) fp32 accumulators: the pair the previous call's post launch re-zeroed, else fresh. Sequential
    use on one stream (eager or graph-captured), as dq_postprocess_cute's "pool" mode."""
    t = _POOL.pop((b, hk, n, str(device)), None)
    if t is None:
        t = (torch.zeros(b, hk, n, dtype=torch.float32, device=device),
             torch.zeros(b, hk, n, dtype=torch.float32, device=device))
    return t


def dkv_postprocess_2cta_gqa(acc_k: torch.Tensor, acc_v: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                             had: bool = False) -> None:
    """acc_k / acc_v (b, h_k, s_k * 128) fp32 in the epilogue order -> dk / dv (b, s_k, h_k, 128) in place; the
    accumulators are zeroed as read and returned to the pool. had: dK carries the qk_hadamard rotation (undone here)."""
    b, hk, n = acc_k.shape
    assert acc_v.shape == acc_k.shape and n % 16384 == 0 and dk.shape[1] * 128 == n and dk.stride(3) == 1
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (dk.dtype, bool(had))
    if key not in _cache:
        from flash_attn.cute.qk_hadamard import signs_tuple
        obj = DKVPostprocess2CTA(torch2cute_dtype_map[dk.dtype], True, signs_tuple(128) if had else None)
        _cache[key] = cute.compile(obj, to_cute_tensor(acc_k, divisibility=4), to_cute_tensor(acc_v, divisibility=4),
                                   to_cute_tensor(dk, divisibility=8), to_cute_tensor(dv, divisibility=8), stream,
                                   options="--enable-tvm-ffi")
    _cache[key](acc_k, acc_v, dk, dv, stream)
    _POOL[(b, hk, n, str(acc_k.device))] = (acc_k, acc_v)
    if had:
        dk._had_unrotated = True

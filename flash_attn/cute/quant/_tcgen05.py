# hadamard_rotation_method="tcgen05" (opt-in, never a default): the random-sign Hadamard
# rotation of a 128-row x 128-column 16-bit tile on tcgen05 UMMA, the fp32 accumulator in TMEM.
#
#   y = (x @ B^T) * f32(1 / sqrt(128)),   B[n, k] = signs[k] H_128[k, n]   (exact +-1 in bf16 / f16)
#
# One kind::f16 128 x 128 x 128 UMMA (8 accumulating K = 16 instructions, A and B K-major in 128-byte-swizzled SMEM);
# thread r of a 128-thread group reads TMEM lane r = its own row (tcgen05.ld 32x32b). The products are exact; the 128-term
# sum is the tensor core's single accumulation, so the fp32 values are NOT the tensor_core method's (H_8 on mma.sync,
# then a 4-level fp32 round-to-nearest butterfly) and not the butterfly's: they differ in the last bits of ~0.4 % (bf16)
# / ~3.6 % (f16) of N(0, 1) values, and the accumulation is less accurate on wide-range rows (log-normal rows: twice the
# tensor_core method's max error). Inputs must be bf16 / f16 (an fp32
# input would be rounded to a 16-bit operand: refused).
#
# Layouts (CuTe's make_smem_layout_a / _b for this MMA, SW128 K-major; checked bitwise):
#   A / B element (m, k) at byte 16384 (k / 64) + 128 m + 16 ((k % 64 / 8) ^ (m & 7)) + 2 (k % 8)
#   SMEM descriptor = ((addr >> 4) & 0x3FFF) | SW128 (2 << 61) | version (1 << 46) | SBO 1024 B (64 << 32); K step k:
#   + 2 (k % 4) + 1024 (k / 4)
#   instruction descriptor kind::f16: D fp32 (bit 4), A / B bf16 (bits 7, 10; 0 = f16), K-major, N >> 3 at bit 17,
#   M >> 4 at bit 24
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import tcgen05

from flash_attn.cute.quant._rotations import INV_SQRT_128, _u32_as_i32

TILE_BYTES = 32768                          # one 128 x 128 16-bit operand image
_DESC_HI = 0x4000404000000000
_IDESC = {"bf16": 0x08200490, "f16": 0x08200010}


@dsl_user_op
def sw128_desc(smem_addr, *, loc=None, ip=None) -> Int64:
    """The UMMA SMEM descriptor of a 1024-byte-aligned SW128 K-major operand image at smem_addr (a 32-bit address)."""
    return Int64(llvm.inline_asm(
        T.i64(), [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a;\n\t.reg .b64 w;\n\tshr.u32 a, $1, 4;\n\tand.b32 a, a, 16383;\n\tcvt.u64.u32 w, a;\n\t"
        f"or.b64 $0, w, {_DESC_HI};\n\t}}\n", "=l,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def umma_f16_ss(tmem_addr, a_desc, b_desc, idesc: cutlass.Constexpr[int], accumulate: cutlass.Constexpr[bool], *,
                loc=None, ip=None) -> None:
    """tcgen05.mma.cta_group::1.kind::f16 D[tmem] (+)= A[smem] B[smem]^T (one elected thread issues it)."""
    llvm.inline_asm(
        None, [Int32(tmem_addr).ir_value(loc=loc, ip=ip), Int64(a_desc).ir_value(loc=loc, ip=ip),
               Int64(b_desc).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        f"setp.ne.b32 p, {1 if accumulate else 0}, 0;\n\t"
        f"tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, {idesc}, p;\n\t}}\n",
        "r,l,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tmem_ld_row32(taddr, *, loc=None, ip=None):
    """32 consecutive fp32 columns of the thread's TMEM lane (tcgen05.ld 32x32b.x32) and the wait for them, one asm."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32), [Int32(taddr).ir_value(loc=loc, ip=ip)],
        "{\n\ttcgen05.ld.sync.aligned.32x32b.x32.b32 {" + ", ".join(f"${i}" for i in range(32)) + "}, [$32];\n\t"
        "tcgen05.wait::ld.sync.aligned;\n\t}\n",
        ",".join(["=f"] * 32) + ",r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return [Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(32)]


def b_sign_words(signs) -> tuple:
    """Per 16-byte vector (c = 0..7, word w = 0..3) the sign-bit words of columns 8 c + 2 w (+1) of each 64-column half."""
    sw = lambda k: _u32_as_i32(((1 if signs[k] < 0 else 0) << 15) | ((1 if signs[k + 1] < 0 else 0) << 31))  # noqa: E731
    return tuple(tuple((sw(8 * c + 2 * w), sw(64 + 8 * c + 2 * w)) for w in range(4)) for c in range(8))


@cute.jit
def build_b_image(b_addr, n, sign_words: cutlass.Constexpr, is_fp16: cutlass.Constexpr):
    """Thread n (0..127) writes row n of B = diag(signs) H_128 (both 64-column halves: 16 vectors) into the SW128 image at
    b_addr: element (n, k) is +-1 with sign parity(n & k) ^ signs[k]."""
    one = 0x3C00 if const_expr(is_fp16) else 0x3F80
    r0 = n & 1
    r1 = (n >> 1) & 1
    r2 = (n >> 2) & 1
    q = (Int32(0), r0, r1, r0 ^ r1, r2, r0 ^ r2, r1 ^ r2, r0 ^ r1 ^ r2)
    m = ((q[0] << 15) | (q[1] << 31), (q[2] << 15) | (q[3] << 31), (q[4] << 15) | (q[5] << 31),
         (q[6] << 15) | (q[7] << 31))
    n3 = n >> 3
    for half in cutlass.range_constexpr(2):
        base = b_addr + Int32(16384 * half) + Int32(128) * n
        for c in cutlass.range_constexpr(8):
            p = (n3 & c) ^ ((n3 >> 3) & half)            # parity(n & k0) over k0 = 64 half + 8 c, bits 3..6
            p = p ^ (p >> 2)
            p = (p ^ (p >> 1)) & 1
            flip = p * Int32(_u32_as_i32(0x80008000))
            vec = cute.make_rmem_tensor(cute.make_layout(4), Int32)
            for w in cutlass.range_constexpr(4):
                vec[w] = Int32(_u32_as_i32(one | (one << 16))) | (m[w] ^ flip ^ Int32(sign_words[c][w][half]))
            dst = cute.make_ptr(Int32, base + Int32(16) * (c ^ (n & 7)), cute.AddressSpace.smem, assumed_align=16)
            cute.autovec_copy(vec, cute.make_tensor(dst, cute.make_layout(4)))


@cute.jit
def write_a_row(a_addr, r, words: cute.Tensor):
    """Thread r writes its row (64 packed 16-bit words) into the SW128 A image at a_addr."""
    for c16 in cutlass.range_constexpr(16):
        dst = cute.make_ptr(Int32, a_addr + Int32(16384 * (c16 // 8)) + Int32(16) * ((c16 % 8) ^ (r & 7))
                            + Int32(128) * r, cute.AddressSpace.smem, assumed_align=16)
        cute.autovec_copy(cute.make_tensor(words.iterator + 4 * c16, cute.make_layout(4)),
                          cute.make_tensor(dst, cute.make_layout(4)))


@cute.jit
def read_a_row(a_addr, r, words: cute.Tensor):
    """The inverse of write_a_row (row r of a SW128 image, e.g. a TMA'd tile, into 64 packed words)."""
    for c16 in cutlass.range_constexpr(16):
        src = cute.make_ptr(Int32, a_addr + Int32(16384 * (c16 // 8)) + Int32(16) * ((c16 % 8) ^ (r & 7))
                            + Int32(128) * r, cute.AddressSpace.smem, assumed_align=16)
        cute.autovec_copy(cute.make_tensor(src, cute.make_layout(4)),
                          cute.make_tensor(words.iterator + 4 * c16, cute.make_layout(4)))


@cute.jit
def issue_rotation(tmem_acc, a_addr, b_addr, idesc: cutlass.Constexpr[int]):
    """The 8 K = 16 UMMAs of one tile (call from ONE thread)."""
    ad = sw128_desc(a_addr)
    bd = sw128_desc(b_addr)
    for k in cutlass.range_constexpr(8):
        step = 2 * (k % 4) + 1024 * (k // 4)
        umma_f16_ss(tmem_acc, ad + Int64(step), bd + Int64(step), idesc, k != 0)


@cute.jit
def load_rotated_row(tmem_acc, r, out: cute.Tensor):
    """Thread r (warp r / 32 owns TMEM lanes 32 (r / 32) ..) reads its 128 accumulator columns and applies the final
    * f32(1 / sqrt(128)) (packed, as the other methods)."""
    lane_base = Int32((r // 32) * 32) << 16
    for q in cutlass.range_constexpr(4):
        vals = tmem_ld_row32(tmem_acc + lane_base + Int32(32 * q))
        for e in cutlass.range_constexpr(0, 32, 2):
            out[32 * q + e], out[32 * q + e + 1] = cute.arch.mul_packed_f32x2(
                (vals[e], vals[e + 1]), (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))


class RowRotation:
    """The drop-in tcgen05 rotation for the thread-per-row quantiser kernels (128 threads, thread r = tile row r):
        st = rot.allocate(smem); rot.setup(st, tidx)            (B image, mbarrier, 128 TMEM columns; one CTA barrier)
        rot.write_row(st, r, words)                            (the row's 64 packed 16-bit words, RoPE already applied)
        rot.rotate(st, r, out, phase)                          (barrier, one thread issues, all wait; out = 128 fp32)
        rot.teardown(st, tidx)                                 (barrier, TMEM freed)
    phase: the mbarrier parity of this rotation (0, 1, 0, ... for a persistent CTA's successive tiles).
    SMEM: 64 KB + 16 B (smem_bytes), TMEM: 128 columns per CTA."""

    smem_bytes = 2 * TILE_BYTES + 1024 + 64

    def __init__(self, dtype, signs):
        self.dtype = dtype
        self.is_fp16 = dtype is cutlass.Float16
        self.sign_words = b_sign_words(signs)
        self.idesc = _IDESC["f16" if self.is_fp16 else "bf16"]

    @cute.jit
    def allocate(self, smem):
        sa = smem.allocate_tensor(Int32, cute.make_layout(TILE_BYTES // 4), 1024)
        sb = smem.allocate_tensor(Int32, cute.make_layout(TILE_BYTES // 4), 1024)
        bar = smem.allocate_tensor(Int64, cute.make_layout(1), 8)
        hold = smem.allocate_tensor(Int32, cute.make_layout(1), 8)
        return sa, sb, bar, hold

    @cute.jit
    def setup(self, st, tidx):
        sa, sb, bar, hold = st
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if tidx < 128:
            build_b_image(sb.iterator.toint(), tidx, self.sign_words, self.is_fp16)
        if warp == 0:
            cute.arch.alloc_tmem(128, hold.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
            if tidx == 0:
                cute.arch.mbarrier_init(bar.iterator, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

    @cute.jit
    def write_row(self, st, r, words: cute.Tensor):
        sa, sb, bar, hold = st
        write_a_row(sa.iterator.toint(), r, words)

    @cute.jit
    def rotate(self, st, r, out: cute.Tensor, phase):
        sa, sb, bar, hold = st
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        tmem = hold[0]
        if r == 0:
            tcgen05_after_sync()
            issue_rotation(tmem, sa.iterator.toint(), sb.iterator.toint(), self.idesc)
            tcgen05.commit(bar.iterator)
        cute.arch.mbarrier_wait(bar.iterator, phase)
        tcgen05_after_sync()
        if r < 128:
            load_rotated_row(tmem, r, out)
        tcgen05_before_sync()

    @cute.jit
    def teardown(self, st, tidx):
        sa, sb, bar, hold = st
        cute.arch.sync_threads()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            tcgen05_after_sync()
            ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=hold.iterator)
            cute.arch.dealloc_tmem(ptr, 128)


@dsl_user_op
def tcgen05_after_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_before_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


# ---------------------------------------------------------------------------------------------- host-side rotation
class _RotateRowsKernel:
    """(n, 128) 16-bit rows -> (n, 128) fp32 rows rotated on tcgen05: the exact values every tcgen05 quantiser
    quantises (the references and tests use it; n % 128 == 0)."""

    def __init__(self, dtype, signs):
        self.rot = RowRotation(dtype, signs)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, stream):
        self.kernel(mX, mY).launch(grid=[mX.shape[0] // 128, 1, 1], block=[128, 1, 1],
                                   smem=RowRotation.smem_bytes + 1024, stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mY: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        st = self.rot.allocate(smem)
        self.rot.setup(st, tidx)
        row = cute.make_rmem_tensor(cute.make_layout(64), Int32)
        cute.autovec_copy(cute.make_tensor(cute.recast_ptr(mX[bid * 128 + tidx, None].iterator, dtype=Int32),
                                           cute.make_layout(64)), row)
        self.rot.write_row(st, tidx, row)
        v = cute.make_rmem_tensor(cute.make_layout(128), Float32)
        self.rot.rotate(st, tidx, v, Int32(0))
        cute.autovec_copy(v, cute.make_tensor(mY[bid * 128 + tidx, None].iterator, cute.make_layout(128)))
        self.rot.teardown(st, tidx)


_ROWS_CACHE = {}


def rotate_rows(x, signs):
    """x (..., 128) bf16 / fp16 -> fp32 (x * signs) H_128 / sqrt(128) computed exactly as the tcgen05 quantisers do."""
    import torch
    import cuda.bindings.driver as cuda
    from flash_attn.cute.quant.fp8 import to_cute_tensor, _cute_dtype
    assert x.dtype in (torch.bfloat16, torch.float16) and x.shape[-1] == 128
    shape = x.shape
    x2 = x.reshape(-1, 128).contiguous()
    n = x2.shape[0]
    pad = (-n) % 128
    if pad:
        x2 = torch.cat([x2, torch.zeros(pad, 128, dtype=x.dtype, device=x.device)])
    y = torch.empty(x2.shape, dtype=torch.float32, device=x.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (x.dtype, tuple(signs))
    if key not in _ROWS_CACHE:
        _ROWS_CACHE[key] = cute.compile(_RotateRowsKernel(_cute_dtype(x.dtype), tuple(signs)),
                                        to_cute_tensor(x2, divisibility=8), to_cute_tensor(y, divisibility=4), stream,
                                        options="--enable-tvm-ffi")
    _ROWS_CACHE[key](x2, y, stream)
    return y[:n].reshape(shape)

# Register-resident rotations shared by the FP4 and FP8 quantisers (flash_attn.cute.quant): the random-sign Hadamard
# rotation of a head-dim row and RoPE. Every helper works on a row (or half a row) that one thread already holds in
# registers, so applying it inside a quantiser costs arithmetic only: no extra read or write of the tensor.
#
# Hadamard:  x -> (x * signs) H_128 / sqrt(128)   (orthogonal; signs = hadamard_signs(seed), compile-time +-1 constants)
#            Order of operations, which fixes the bits: the sign flips, the seven butterfly levels H = 1, 2, ..., 64 (each
#            pair (a, b) -> (a + b, a - b)), then one multiply by f32(1 / sqrt(128)). This is the order of every earlier
#            implementation (nvfp4_2d_triton._fwht, qk_hadamard_quant.rotate_row128 / half_rotate_packed), so the rotated
#            values are bitwise theirs.
# RoPE:      GPT-NeoX "halved" convention on a 128-wide row: for j < 64, (x_j, x_{j+64}) ->
#            (x_j cos_j - x_{j+64} sin_j, x_{j+64} cos_j + x_j sin_j), cos / sin fp32 tables indexed by the token position,
#            stored (d / 2, s) ("transposed": a warp's lanes read consecutive addresses) or (s, d / 2). The products are
#            each output one fma of the other product: fma(x0, c, -(x1 s)) and fma(x1, c, x0 s) with (64, s) tables,
#            fma(x0, s, x1 c) for the high half with (s, 64) tables -- what nvfp4_quant_cute's two RoPE paths compiled
#            to (measured); the quantiser then rounds the
#            rotated row to the input dtype (the row a standalone RoPE kernel would have written).
# Rotation method (hadamard_rotation_method, a compile-time option of every rotating quantiser):
#   "tensor_core"  the same R on mma.sync (quad_hadamard_rotate below): the same sign flips and the same final
#                  * f32(1 / sqrt(128)); an 8-point stage on the tensor cores, then a 4-level register butterfly. Equal to
#                  the butterfly up to the fp32 summation order. Needs 16-bit inputs (bf16 / f16 are exact mma operands).
#   "butterfly"    the register butterfly above, bitwise every earlier implementation.
#   "tcgen05"      OPT-IN (never the default, never a fallback): the same R as ONE tcgen05
#                  UMMA per 128-row tile (quant._tcgen05; the token-block kernel quant._fp8_tcgen05): exact products,
#                  the 128-term sum is the tensor core's single accumulation, then the same * f32(1 / sqrt(128)). NOT
#                  bitwise the other methods (0.4 % / 3.6 % of N(0, 1) bf16 / f16 rotated values differ in the last
#                  bits; less accurate than tensor_core on wide-range rows). Where a quantiser has no tcgen05 form the
#                  call is refused with a ValueError naming "tensor_core" (resolve_rotation_method's tcgen05_unsupported).
#   None           the default: tensor_core where it measured faster (the lane-split kernels: quant.fp8's token blocks,
#                  streaming blocks and per-tensor rows), the butterfly in the thread-per-row kernels (quant.fp4's tile
#                  kernel, quant.fp8's block_columns=1 kernel, the per-tensor amax-only launch) and where the tensor-core
#                  form does not exist (fp32 input; quant.fp4_diagonal).
# Every butterfly fallback names its reason (logger "flash_attn.cute.quant", once per quantiser and reason;
# ROTATION_METHOD_LOG keeps each quantiser's last resolution). Nothing is ever refused for the method.
import logging
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

INV_SQRT_128 = 1.0 / math.sqrt(128.0)

HADAMARD_ROTATION_METHODS = ("tensor_core", "butterfly", "tcgen05")
DEFAULT_HADAMARD_ROTATION_METHOD = "tensor_core"
ROTATION_METHOD_LOG = {}      # quantiser name -> (method run, reason): the last resolution of each quantiser
_LOGGED = set()
_LOGGER = logging.getLogger("flash_attn.cute.quant")
TCGEN05_NOT_BUILT = "is not built for this quantiser"


def resolve_rotation_method(requested: Optional[str], quantiser: str, unsupported: Optional[str] = None,
                            butterfly_faster: Optional[str] = None,
                            tcgen05_unsupported: Optional[str] = TCGEN05_NOT_BUILT) -> str:
    """The rotation method a quantiser runs. requested: "tensor_core" | "butterfly" | "tcgen05" | None (the default,
    DEFAULT_HADAMARD_ROTATION_METHOD, "tensor_core where it wins"). unsupported: why this call's kernel cannot run the
    tensor-core form (None: it can) -- then the butterfly runs, never a refusal. butterfly_faster: why the default is the
    butterfly in this kernel (measured slower on the tensor cores); an explicit "tensor_core" still runs there. Every
    fallback reason is logged once per (quantiser, reason).
    "tcgen05" (opt-in, quant._tcgen05; NOT bitwise the other methods) is never the default and never falls back:
    tcgen05_unsupported (None = this call runs it; default: the quantiser has no tcgen05 form) makes it a ValueError
    that names hadamard_rotation_method="tensor_core"."""
    if requested is not None and requested not in HADAMARD_ROTATION_METHODS:
        raise ValueError(f"hadamard_rotation_method must be one of {HADAMARD_ROTATION_METHODS} or None, got {requested!r}")
    if requested == "tcgen05":
        if tcgen05_unsupported is not None:
            raise ValueError(f"{quantiser}: hadamard_rotation_method='tcgen05' {tcgen05_unsupported}; use "
                             "hadamard_rotation_method='tensor_core' (the default) for this call")
        ROTATION_METHOD_LOG[quantiser] = ("tcgen05", "requested")
        return "tcgen05"
    want = DEFAULT_HADAMARD_ROTATION_METHOD if requested is None else requested
    reason = None
    if want == "tensor_core" and unsupported is not None:
        reason = f"tensor_core -> butterfly: {unsupported}"
    elif want == "tensor_core" and requested is None and butterfly_faster is not None:
        reason = f"default -> butterfly: {butterfly_faster}"
    if reason is not None:
        if (quantiser, reason) not in _LOGGED:
            _LOGGED.add((quantiser, reason))
            _LOGGER.info("%s: hadamard_rotation_method %s", quantiser, reason)
        ROTATION_METHOD_LOG[quantiser] = ("butterfly", reason)
        return "butterfly"
    ROTATION_METHOD_LOG[quantiser] = (want, "requested" if requested is not None else "default")
    return want


def _asm(op, *args):
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T
    n = len(args)
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value() for a in args], f"{op} $0, " + ", ".join(f"${i + 1}" for i in range(n)) + ";",
        "=f," + ",".join(["f"] * n), has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def _mul(a, b):
    return _asm("mul.rn.f32", a, b)


def _fma(a, b, c):
    return _asm("fma.rn.f32", a, b, c)
DEFAULT_HADAMARD_SIGN_SEED = 4321  # the seed every route has used (qk_hadamard.SEED, interface_bwd_fp4q._signs)

_SIGNS_CACHE = {}


def hadamard_signs(seed: int = DEFAULT_HADAMARD_SIGN_SEED, d: int = 128) -> tuple:
    """The (d,) random +-1 signs of the rotation as a tuple of Python ints (compile-time constants of a kernel).
    Same generator as every earlier implementation: torch's CPU generator seeded with seed & 0x7FFFFFFF, randint(0, 2)."""
    key = (int(seed), int(d))
    if key not in _SIGNS_CACHE:
        import torch
        g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        _SIGNS_CACHE[key] = tuple(int(v) for v in (torch.randint(0, 2, (d,), generator=g) * 2 - 1).tolist())
    return _SIGNS_CACHE[key]


def hadamard_signs_tensor(device, seed: int = DEFAULT_HADAMARD_SIGN_SEED, d: int = 128):
    """hadamard_signs as a float32 (d,) tensor on `device` (for host-side code and references)."""
    import torch
    return torch.tensor(hadamard_signs(seed, d), dtype=torch.float32, device=device)


@cute.jit
def hadamard_rotate_row(x: cute.Tensor, signs: cutlass.Constexpr):
    """x[0:128] <- (x * signs) H / sqrt(128) in place, one thread holding the whole row. The signs are compile-time
    negations; level 1 is scalar (its pair is one register pair), levels 2..64 are packed FADD2 / FSUB2 on (c, c + 1),
    the scale one FMUL2 per pair. Bitwise nvfp4_2d_triton.fwht_rows."""
    for d in cutlass.range_constexpr(128):
        if const_expr(signs[d] < 0):
            x[d] = -x[d]
    for k in cutlass.range_constexpr(0, 128, 2):
        a = x[k]
        b = x[k + 1]
        x[k] = a + b
        x[k + 1] = a - b
    for lvl in cutlass.range_constexpr(1, 7):
        hs = 1 << lvl
        for blk in cutlass.range_constexpr(0, 128, 2 * hs):
            for j in cutlass.range_constexpr(0, hs, 2):
                a0, a1 = x[blk + j], x[blk + j + 1]
                b0, b1 = x[blk + j + hs], x[blk + j + hs + 1]
                x[blk + j], x[blk + j + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                x[blk + j + hs], x[blk + j + hs + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
    for d in cutlass.range_constexpr(0, 128, 2):
        x[d], x[d + 1] = cute.arch.mul_packed_f32x2((x[d], x[d + 1]), (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))


@cute.jit
def hadamard_rotate_half_row(x: cute.Tensor, signs: cutlass.Constexpr, lane_half):
    """Two lanes per row (lane_half = 0 holds columns 0..63, 1 holds 64..127; the partner is lane ^ 1): x[0:64] <- this
    lane's half of (row * signs) H_128 / sqrt(128). Levels H = 1..32 in registers, level 64 with one shfl.xor(1) per value
    folded as fma(mine, +-1, partner) (exactly a + b / a - b). Bitwise hadamard_rotate_row."""
    pm = Float32(1.0) - Float32(2.0) * Float32(lane_half)
    for c in cutlass.range_constexpr(0, 64, 2):
        f0 = Float32(signs[c]) if const_expr(signs[c] == signs[64 + c]) else Float32(signs[c]) * pm
        f1 = Float32(signs[c + 1]) if const_expr(signs[c + 1] == signs[65 + c]) else Float32(signs[c + 1]) * pm
        x[c], x[c + 1] = cute.arch.mul_packed_f32x2((x[c], x[c + 1]), (f0, f1))
    for k in cutlass.range_constexpr(0, 64, 2):
        a = x[k]
        b = x[k + 1]
        x[k] = a + b
        x[k + 1] = a - b
    for lvl in cutlass.range_constexpr(1, 6):
        hs = 1 << lvl
        for blk in cutlass.range_constexpr(0, 64, 2 * hs):
            for j in cutlass.range_constexpr(0, hs, 2):
                a0, a1 = x[blk + j], x[blk + j + 1]
                b0, b1 = x[blk + j + hs], x[blk + j + hs + 1]
                x[blk + j], x[blk + j + 1] = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
                x[blk + j + hs], x[blk + j + hs + 1] = cute.arch.sub_packed_f32x2((a0, a1), (b0, b1))
    for c in cutlass.range_constexpr(0, 64, 2):
        o0 = cute.arch.shuffle_sync_bfly(x[c], offset=1)
        o1 = cute.arch.shuffle_sync_bfly(x[c + 1], offset=1)
        x[c], x[c + 1] = cute.arch.fma_packed_f32x2((x[c], x[c + 1]), (pm, pm), (o0, o1))
        x[c], x[c + 1] = cute.arch.mul_packed_f32x2((x[c], x[c + 1]), (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))


@cute.jit
def rope_rotate_pair(x0: Float32, x1: Float32, cos: Float32, sin: Float32, tables_transposed: cutlass.Constexpr = True):
    """One RoPE pair, explicitly the forms nvfp4_quant_cute's expressions compiled to (measured, probe_rope), so the result
    never depends on the compiler's contraction choices: low half fma(x0, cos, -(x1 sin)); high half fma(x1, cos, x0 sin)
    with (64, s) tables, fma(x0, sin, x1 cos) with (s, 64) tables."""
    if const_expr(tables_transposed):
        return _fma(x0, cos, -_mul(x1, sin)), _fma(x1, cos, _mul(x0, sin))
    return _fma(x0, cos, -_mul(x1, sin)), _fma(x0, sin, _mul(x1, cos))


@cute.jit
def rope_rotate_row(x: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, position,
                    tables_transposed: cutlass.Constexpr, head_dim: cutlass.Constexpr = 128):
    """x[0:128] <- RoPE(x) at `position`, one thread holding the whole row (GPT-NeoX halved pairs (j, j + 64)).
    tables_transposed: cos / sin stored (64, s) -- lane r reads column position, so a warp's 32 consecutive positions are
    32 consecutive addresses (coalesced); otherwise (s, 64), read 16 at a time with vector loads."""
    half = head_dim // 2
    if const_expr(tables_transposed):
        for j in cutlass.range_constexpr(half):
            x[j], x[j + half] = rope_rotate_pair(x[j], x[j + half], mCos[j, position], mSin[j, position])
    else:
        for blk in cutlass.range_constexpr(half // 16):
            cosb = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            sinb = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            cute.autovec_copy(cute.make_tensor(mCos[position, None].iterator + blk * 16, cute.make_layout(16)), cosb)
            cute.autovec_copy(cute.make_tensor(mSin[position, None].iterator + blk * 16, cute.make_layout(16)), sinb)
            for jj in cutlass.range_constexpr(16):
                j = blk * 16 + jj
                x[j], x[j + half] = rope_rotate_pair(x[j], x[j + half], cosb[jj], sinb[jj], False)


@cute.jit
def rope_rotate_half_row(x: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, position, lane_half,
                         tables_transposed: cutlass.Constexpr):
    """Two lanes per row: lane_half 0 holds x_j (j < 64), lane 1 holds x_{j+64}; the partner's value comes by shfl.xor(1).
    Lane 0 keeps x_j cos - x_{j+64} sin, lane 1 keeps x_{j+64} cos + x_j sin (bitwise rope_rotate_row)."""
    for j in cutlass.range_constexpr(64):
        c = Float32(0.0)
        s_ = Float32(0.0)
        if const_expr(tables_transposed):
            c = mCos[j, position]
            s_ = mSin[j, position]
        else:
            c = mCos[position, j]
            s_ = mSin[position, j]
        other = cute.arch.shuffle_sync_bfly(x[j], offset=1)
        if lane_half == 0:
            x[j] = _fma(x[j], c, -_mul(other, s_))
        elif const_expr(tables_transposed):
            x[j] = _fma(x[j], c, _mul(other, s_))
        else:
            x[j] = _fma(other, s_, _mul(x[j], c))


# ---------------------------------------------------------------------------------------------------------------------
# The same rotation on the tensor cores (warp-level mma.sync). H_128 = H_8 (x) H_16 over a split of the
# column bits: c = 32 m + 8 t + 2 j + v (m: bits 6-5, t: bits 4-3, j: bits 2-1, v: bit 0). One warp rotates 16 rows;
# lane (g = lane / 4, t = lane % 4) holds rows g and g + 8 and, of each, the four 8-column chunks 4 m + t (m = 0..3) --
# the "quad" layout: a lane quad reads / writes 64 contiguous bytes of a 16-bit row per instruction, and the RoPE partner
# of column c (c + 64) is chunk m + 2 of the same lane.
#   1. signs: an exact flip of the 16-bit inputs' sign bits (lane-dependent xor masks from compile-time words);
#   2. H_8 over bits {0, 3, 4}: one mma.sync.m16n8k8 per (m, j) (16 per warp); A = the 16 rows' signed inputs (the k
#      slots 2t, 2t + 1 are the lane's own packed pair (t, v) of chunk m: no data moves between lanes), B = the fixed +-1
#      matrix (exact in bf16 / f16), fp32 accumulation; the C fragment returns output columns (t, v) of rows g / g + 8 to
#      the same lane -- the lane's own columns again;
#   3. H_16 over bits {1, 2, 5, 6}: a 4-level register butterfly across the 16 products (j bit 0, j bit 1, m bit 0,
#      m bit 1);
#   4. * f32(1 / sqrt(128)).
# Products of a 16-bit float and +-1 are exact, so the result equals the butterfly (hadamard_rotate_row) up to the fp32
# summation order: bitwise wherever every partial sum is exact in fp32 (and up to the sign of an exact zero). The other
# split (H_16 on m16n8k16 + a 3-level butterfly) was measured too: the same speed for the 8-bit token blocks, 1-2 %
# slower for the 4-bit ones (twice the tensor-pipe time: the legacy mma.sync runs about one m16n8k16 per 32 cycles per
# SM sub-partition on this part) and more code differences on fp16 inputs.
_MMA_ACC_INIT = "0f80000000"  # C = -0.0: a column of -0 products stays -0, as an IEEE sum would


def _u32_as_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def quad_sign_words(signs) -> tuple:
    """The signs as one compile-time word per lane class t = lane % 4: bit 4 m + p of word t is set when column
    32 m + 8 t + 2 p is negated, bit 16 + 4 m + p when column 32 m + 8 t + 2 p + 1 is -- so the xor mask of the packed
    register a[.. 4 m + p] (low half = the even column) is (word << (15 - (4 m + p))) & 0x80008000."""
    words = []
    for t in range(4):
        w = 0
        for m in range(4):
            for p in range(4):
                j = 4 * m + p
                c = 32 * m + 8 * t + 2 * p
                if signs[c] < 0:
                    w |= 1 << j
                if signs[c + 1] < 0:
                    w |= 1 << (16 + j)
        words.append(_u32_as_i32(w))
    return tuple(words)


def mma_m16n8k8_f32(a0, a1, b0, half_kind: str):
    """D = A B + (-0.0), A 16 x 8 / B 8 x 8 of bf16 or f16 (half_kind "bf16" | "f16") in their mma.sync fragments, fp32
    accumulation; returns the 4 fp32 D registers of the lane."""
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(a0).ir_value(), Int32(a1).ir_value(), Int32(b0).ir_value()],
        "{\n\t.reg .f32 z0, z1, z2, z3;\n\t"
        f"mov.b32 z0, {_MMA_ACC_INIT};\n\tmov.b32 z1, {_MMA_ACC_INIT};\n\t"
        f"mov.b32 z2, {_MMA_ACC_INIT};\n\tmov.b32 z3, {_MMA_ACC_INIT};\n\t"
        f"mma.sync.aligned.m16n8k8.row.col.f32.{half_kind}.{half_kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5}, {$6}, {z0, z1, z2, z3};\n\t}\n",
        "=f,=f,=f,=f,r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(4))


@cute.jit
def quad_apply_signs(a: cute.Tensor, lane, sign_words: cutlass.Constexpr):
    """The sign flips of the quad layout's packed 16-bit inputs a (32 Int32), in place (None: nothing)."""
    if const_expr(sign_words is not None):
        t = lane % 4
        w = Int32(cutlass.select_(t == 0, Int32(sign_words[0]), Int32(sign_words[1])))
        w = Int32(cutlass.select_(t == 2, Int32(sign_words[2]), w))
        w = Int32(cutlass.select_(t == 3, Int32(sign_words[3]), w))
        for j in cutlass.range_constexpr(16):
            mask = (w << (15 - j)) & Int32(_u32_as_i32(0x80008000))
            for i in cutlass.range_constexpr(2):
                a[16 * i + j] = a[16 * i + j] ^ mask


@cute.jit
def quad_hadamard_b_operand(lane, is_fp16: cutlass.Constexpr):
    """The lane's B register of the fixed H_8 operand: B[k][n] = (-1)^(popc(t_k & t_n) + v_k v_n) with k = 2 t + v_k
    (the lane's own t), n = g -> (t_n = g / 2, v_n = g % 2); low half v_k = 0, high half v_k = 1. +-1 is 0x3F80 /
    0xBF80 in bf16, 0x3C00 / 0xBC00 in f16."""
    one = 0x3C00 if const_expr(is_fp16) else 0x3F80
    t = lane % 4
    g = lane // 4
    tt = t & (g >> 1)
    flip = ((tt ^ (tt >> 1)) & 1) * Int32(_u32_as_i32(0x80008000))
    vpart = Int32(cutlass.select_((g & 1) == 1, Int32(_u32_as_i32((one | (one << 16)) | (1 << 31))),
                                  Int32(one | (one << 16))))
    return vpart ^ flip


@cute.jit
def quad_hadamard_rotate(a: cute.Tensor, out: cute.Tensor, lane, sign_words: cutlass.Constexpr,
                         is_fp16: cutlass.Constexpr):
    """Rotate the warp's 16 rows on the tensor cores (see above). a: 32 Int32 registers of packed 16-bit inputs (bf16, or
    f16 when is_fp16): a[16 i + 4 m + p] = row g + 8 i, columns 32 m + 8 t + 2 p (low half) and + 1; out: 64 Float32,
    out[32 i + 8 m + e] = the rotated column 32 m + 8 t + e of row g + 8 i. sign_words: quad_sign_words(signs), or None
    for H alone. a is modified (the sign flips are applied in place)."""
    kind = "f16" if const_expr(is_fp16) else "bf16"
    quad_apply_signs(a, lane, sign_words)
    b = quad_hadamard_b_operand(lane, is_fp16)
    for m in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(4):
            c0, c1, c2, c3 = mma_m16n8k8_f32(a[4 * m + j], a[16 + 4 * m + j], b, kind)
            out[8 * m + 2 * j] = c0
            out[8 * m + 2 * j + 1] = c1
            out[32 + 8 * m + 2 * j] = c2
            out[32 + 8 * m + 2 * j + 1] = c3
    for lvl in cutlass.range_constexpr(4):          # H_16: offsets 2, 4 (j bits), 8, 16 (m bits); packed over v pairs
        hs = 2 << lvl
        for i in cutlass.range_constexpr(2):
            for base in cutlass.range_constexpr(0, 32, 2):
                if const_expr((base & hs) == 0):
                    lo = 32 * i + base
                    a_ = (out[lo], out[lo + 1])
                    b_ = (out[lo + hs], out[lo + hs + 1])
                    out[lo], out[lo + 1] = cute.arch.add_packed_f32x2(a_, b_)
                    out[lo + hs], out[lo + hs + 1] = cute.arch.sub_packed_f32x2(a_, b_)
    for e in cutlass.range_constexpr(0, 64, 2):
        out[e], out[e + 1] = cute.arch.mul_packed_f32x2((out[e], out[e + 1]),
                                                        (Float32(INV_SQRT_128), Float32(INV_SQRT_128)))
    return out


@cute.jit
def quad_rope(a: cute.Tensor, tq, pos0, pos1, mCos, mSin, rope_t: cutlass.Constexpr, dt: cutlass.Constexpr):
    """RoPE on the quad layout's packed 16-bit registers (quad_hadamard_rotate's a; rows at positions pos0, pos1):
    column j < 64 of chunk m (0, 1) pairs with column j + 64, chunk m + 2 of the same lane. _pair_transform's fma forms
    in fp32, then the row rounded through the storage dtype (bitwise rope_rotate_row / fp8._pair_transform's RoPE)."""
    av = cute.make_tensor(cute.recast_ptr(a.iterator, dtype=dt), cute.make_layout(64))
    x = cute.make_rmem_tensor(cute.make_layout(64), Float32)
    x.store(av.load().to(Float32))
    for i in cutlass.range_constexpr(2):
        pos = pos0 if const_expr(i == 0) else pos1
        for m in cutlass.range_constexpr(2):
            j0 = 32 * m + 8 * tq
            cs = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            sn = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            if const_expr(rope_t):
                for e in cutlass.range_constexpr(8):
                    cs[e] = mCos[j0 + e, pos]
                    sn[e] = mSin[j0 + e, pos]
            else:
                cute.autovec_copy(cute.make_tensor(cute.make_ptr(Float32, (mCos[pos, None].iterator + j0).toint(),
                                                                 cute.AddressSpace.gmem, assumed_align=16),
                                                   cute.make_layout(8)), cs)
                cute.autovec_copy(cute.make_tensor(cute.make_ptr(Float32, (mSin[pos, None].iterator + j0).toint(),
                                                                 cute.AddressSpace.gmem, assumed_align=16),
                                                   cute.make_layout(8)), sn)
            for e in cutlass.range_constexpr(8):
                lo = x[32 * i + 8 * m + e]
                hi = x[32 * i + 8 * (m + 2) + e]
                x[32 * i + 8 * m + e] = _fma(lo, cs[e], -_mul(hi, sn[e]))
                if const_expr(rope_t):
                    x[32 * i + 8 * (m + 2) + e] = _fma(hi, cs[e], _mul(lo, sn[e]))
                else:
                    x[32 * i + 8 * (m + 2) + e] = _fma(lo, sn[e], _mul(hi, cs[e]))
    av.store(x.load().to(dt))

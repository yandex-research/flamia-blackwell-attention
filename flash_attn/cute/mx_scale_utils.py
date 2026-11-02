# In-kernel UE8M0 block scales for E4M3 intermediates (dS^, P^) on SM100: the warp-wide abs-max
# (one CREDUX.MAXABS.F32 on sm_100a) and the exponent / multiplier of a power-of-two scale that
# lands a group's amax inside E4M3 (<= 448). Used by the block-scaled dS path of the FP8 backward
# (flash_bwd_sm100_fp4.py, Bwd1Spec.ds_scaling_fp8).
from cutlass import Int32, Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm


@dsl_user_op
def famax3_op(a, b, c, *, loc=None, ip=None) -> Float32:
    """max(|a|, |b|, |c|): one 3-input FMNMX in abs mode (the binding's positional form follows the DSL
    build, not the toolkit: try the result-type-inferring form first)."""
    args = [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)]
    try:
        return Float32(nvvm.fmax(args[0], args[1], c=args[2], abs=True, loc=loc, ip=ip))
    except TypeError:
        return Float32(nvvm.fmax(T.f32(), args[0], args[1], c=args[2], abs=True, loc=loc, ip=ip))


def famax_tree(vals):
    """max |v| over a Python list of Float32 values as 3-input abs-max instructions (plain Python at trace time).
    FIFO order: the leaves are consumed three at a time first, each result joins the back of the queue, and the last
    pair (if any) takes one 2-operand op, so every op removes two values: ceil((n - 1) / 2) ops -- 16 for 32 values
    (the level-by-level form spent 18: two of its pads were abs-max of a single, already non-negative value). Every
    leaf passes through an abs op; max is exact, so the result is bitwise the old tree's. Empty / all-masked -> 0."""
    m = list(vals)
    if len(m) == 1:
        return famax3_op(m[0], Float32(0.0), Float32(0.0))
    i = 0
    while len(m) - i > 1:
        if len(m) - i >= 3:
            m.append(famax3_op(m[i], m[i + 1], m[i + 2]))
            i += 3
        else:
            m.append(famax3_op(m[i], m[i + 1], Float32(0.0)))
            i += 2
    return m[i]


@dsl_user_op
def redux_max_abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """max |x| over the full warp (redux.sync.max.abs.f32, sm_100a); the result is warp-uniform."""
    out = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
        "=f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def redux_max_abs_f32_pure(x: Float32, *, loc=None, ip=None) -> Float32:
    """redux_max_abs_f32 without the side-effect flag: the warp-collective is still a single instruction, but LLVM may
    schedule other work across it (an opt-in variant of the 2-CTA backward's block-scaled dS reduction)."""
    out = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def ue8m0_exp_448(amax: Float32, *, loc=None, ip=None) -> Int32:
    """UE8M0 byte e (scale 2^(e - 127)) with amax / 2^(e - 127) in (224, 448]: amax = 1.m x 2^E needs
    e - 127 = E - 8 + [1.m > 1.75]; adding 0x00200000 to the fp32 bits carries into the exponent field
    exactly when m >= 0.75. amax = 0 (a fully masked group) gives e = 1 and codes 0; e is clamped to >= 1."""
    out = llvm.inline_asm(
        T.i32(),
        [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b;\n\t"
        "mov.b32 b, $1;\n\t"
        "add.u32 b, b, 0x00200000;\n\t"
        "shr.u32 b, b, 23;\n\t"
        "sub.s32 b, b, 8;\n\t"
        "max.s32 $0, b, 1;\n\t"
        "}\n",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


@dsl_user_op
def ue8m0_val(e: Int32, *, loc=None, ip=None) -> Float32:
    """The scale itself, 2^(e - 127), as fp32 (bits e << 23)."""
    out = llvm.inline_asm(
        T.f32(),
        [Int32(e).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b;\n\t"
        "shl.b32 b, $1, 23;\n\t"
        "mov.b32 $0, b;\n\t"
        "}\n",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def selp_f32(flag: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """flag != 0 ? a : b (one SEL, no branch)."""
    out = llvm.inline_asm(
        T.f32(),
        [Int32(flag).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.s32 p, $1, 0;\n\t"
        "selp.f32 $0, $2, $3, p;\n\t"
        "}\n",
        "=f,r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


def warp_col_max_scatter(vals, lane):
    """Column abs-max across the warp with a reduce-scatter: `vals` is this lane's row of 32 values (one per
    column); on return lane l holds max_lanes |vals[l]| (column l's abs-max over the warp's 32 rows). Five
    butterfly rounds (offsets 16..1): in each, the lane keeps the half of its current list selected by its
    lane bit, sends the other half to its partner and folds what it receives with a 3-input abs-max
    (31 SHFL + 62 SEL + 31 FMNMX3 for 32 columns). Plain Python at trace time."""
    import cutlass.cute as cute
    cur = list(vals)
    h = len(cur) // 2
    while h >= 1:
        flag = lane & Int32(h)  # nonzero iff this lane keeps the upper half
        nxt = []
        for i in range(h):
            lo, hi = cur[i], cur[i + h]
            send = selp_f32(flag, lo, hi)
            keep = selp_f32(flag, hi, lo)
            recv = cute.arch.shuffle_sync_op(send, offset=h, kind=nvvm.ShflKind.bfly)
            nxt.append(famax3_op(keep, recv, Float32(0.0)))
        cur = nxt
        h //= 2
    return cur[0]


@dsl_user_op
def ue8m0_mul(e: Int32, *, loc=None, ip=None) -> Float32:
    """The multiplier 2^-(e - 127) that maps a group onto its E4M3 codes (fp32 bits (254 - e) << 23)."""
    out = llvm.inline_asm(
        T.f32(),
        [Int32(e).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b;\n\t"
        "sub.u32 b, 254, $1;\n\t"
        "shl.b32 b, b, 23;\n\t"
        "mov.b32 $0, b;\n\t"
        "}\n",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)

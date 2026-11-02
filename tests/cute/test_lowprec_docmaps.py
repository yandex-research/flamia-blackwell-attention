"""Precomputed document maps (lp_docmaps.DocMaps, doc_maps=): packed varlen documents on the aligned path with tables
built once per microbatch (docs/lowprec_usage.md, 'Packed documents with precomputed maps').

  * the tables: CPU (numpy) build == GPU build == lp_varlen's own (LPDocs.build + aligned_from on the padded starts);
  * outputs bitwise the repacking path's (cu_seqlens only): fp8 / fp4, causal / dense, MHA / GQA, zero-length
    documents in the middle and a zero-length tail up to the fixed capacity; deterministic (all of O, dQ, dK, dV) and
    not (O, dK, dV);
  * return_lse = the aligned path's LSE rows;
  * no host sync; CUDA-graph capture with the maps as static inputs, replayed on another microbatch (copy_);
  * the CPU helper in DataLoader workers (pin_memory) and a non-blocking copy.

run: pytest tests/cute/test_lowprec_docmaps.py -q
"""
import pytest
import torch

from flash_attn.cute.lowprec import DocMaps, FP4AttnOptions as O4, FP8AttnOptions as O8, flash_attn_fp4, flash_attn_fp8

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                         reason="needs an SM100 GPU")
LENS = [300, 1000, 0, 45, 700, 1, 128, 257]
CAP = 12   # four zero-length trailing documents


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _packed(lens, h, hk, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, do = (torch.randn(T, h, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    k, v = (torch.randn(T, hk, d, device="cuda", dtype=BF16, generator=g) for _ in range(2))
    return q, k, v, do


def _run(fn, q, k, v, do, **kw):
    x = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v))
    r = fn(*x, **kw)
    o, lse = r if isinstance(r, tuple) else (r, None)
    return [o, lse] + list(torch.autograd.grad(o, x, do))


@gpu
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), ([5, 0, 130, 4000, 0, 0, 77], 7), ([128] * 4, 4)])
def test_tables(lens, cap):
    from flash_attn.cute.lp_varlen import LPDocs
    cu = _cu(lens)
    T = int(cu[-1])
    from flash_attn.cute.lp_docmaps import default_padded_size
    # (the full maps: an explicit padded size; without one the maps are compact -- test_compact_tables)
    P = default_padded_size(T, cap)
    m_cpu = DocMaps.from_cu_seqlens(cu.numpy(), max_num_documents=cap, padded_size=P).to("cuda")
    m_gpu = DocMaps.from_cu_seqlens(cu.cuda(), max_num_documents=cap, total_q=T, max_seqlen=max(lens), padded_size=P)
    assert torch.equal(m_cpu.buf, m_gpu.buf)
    # lp_varlen's own tables of the same documents (the capacity's zero-length tail included)
    cuc = torch.cat([cu, cu[-1:].expand(cap - len(lens))]).cuda()
    rp = LPDocs.build(cuc, cuc, T, T, align=128)
    S = m_cpu.padded_size_q
    ref = LPDocs.aligned_from(rp.cu_pad_q, rp.cu_pad_k, rp.len_q, rp.len_k, S, S, align=128)
    d = m_cpu.lpdocs()
    assert torch.equal(d.cu_pad_q, rp.cu_pad_q) and torch.equal(d.len_q, rp.len_q)
    assert torch.equal(m_cpu.dst_q.long(), rp.dst_q)
    tab = ref.doc_tab.clone()
    tab[4:] = rp.doc_tab[4:]
    assert torch.equal(d.doc_tab, tab)
    assert torch.equal(d.tile_doc(128), ref.tile_doc(128)) and torch.equal(d.tile_doc_q, ref.tile_doc_q)
    assert torch.equal(d.row_end_tab_q, ref.row_end_tab_q) and torch.equal(d.row_end_tab_k, ref.row_end_tab_k)
    assert torch.equal(d.q_row_valid(), ref.q_row_valid())
    src = torch.full((S,), -1, dtype=torch.int32, device="cuda")
    src[rp.dst_q] = torch.arange(T, dtype=torch.int32, device="cuda")
    assert torch.equal(m_cpu.src_q, src)


@gpu
@pytest.mark.parametrize("lens,cap", [(LENS, CAP), ([5, 0, 130, 4000, 0, 0, 77], 7), ([128] * 4, 4), ([1] * 300, 300)])
def test_compact_tables(lens, cap):
    """Maps without a padded size (compact, the SF redesign): CPU == GPU build, nothing sized by a padded size or by
    128 x the document count -- the per-tile tables hold min(T, ceil(T / 128) + N) tiles, each tile's packed start and
    real rows in document order, and the device tile counts."""
    from flash_attn.cute.lp_docmaps import tile_bound
    cu = _cu(lens)
    T = int(cu[-1])
    m_cpu = DocMaps.from_cu_seqlens(cu.numpy(), max_num_documents=cap).to("cuda")
    m_gpu = DocMaps.from_cu_seqlens(cu.cuda(), max_num_documents=cap, total_q=T, max_seqlen=max(lens))
    assert m_cpu.compact and m_gpu.compact
    assert torch.equal(m_cpu.buf, m_gpu.buf)
    nb = tile_bound(T, cap)
    assert m_cpu.padded_size_q == m_cpu.padded_size_k == 0 and m_cpu.rows_q == 128 * nb
    assert m_cpu.buf.numel() < 64 * (T + cap) + 64 * nb
    # the constructor fields (a caller's pytree round trip: shape_key + buf) rebuild compact maps
    re = DocMaps(*m_cpu.shape_key(), buf=m_cpu.buf)
    assert re.compact and torch.equal(re.vt_tile_q, m_cpu.vt_tile_q)
    starts, rows = [], []
    for i, L in enumerate(lens):
        for t in range(0, L, 128):
            starts.append(int(cu[i]) + t)
            rows.append(min(128, L - t))
    vt = m_cpu.vt_tile_q.cpu()
    n = len(starts)
    assert m_cpu.ntiles.cpu().tolist() == [n, n]
    assert vt[0, :n].tolist() == starts and vt[1, :n].tolist() == rows
    assert torch.equal(m_cpu.vt_tile_q, m_cpu.vt_tile)


@gpu
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("hk", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "gathered"])
def test_bitwise_repack(fn, causal, hk, d, fused, monkeypatch):
    from flash_attn.cute import lp_docmaps
    monkeypatch.setattr(lp_docmaps, "FUSED_GATHER", fused)
    q, k, v, do = _packed(LENS, 4, hk, d)
    cu = _cu(LENS)
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=CAP, max_seqlen=1024).to("cuda", non_blocking=True)
    cuc = cu.cuda()
    for det in (True, False):
        kw = dict(causal=causal, deterministic=det)
        r = _run(fn, q, k, v, do, cu_seqlens_q=cuc, cu_seqlens_k=cuc, max_seqlen_q=1000, max_seqlen_k=1000, **kw)
        m = _run(fn, q, k, v, do, doc_maps=maps, **kw)
        for i, (x, y) in enumerate(zip(r, m)):
            if i == 1 or (i == 2 and not det):   # (no LSE on the repacking path; dQ atomics without deterministic)
                continue
            assert torch.equal(x, y), (det, i)


@gpu
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
def test_return_lse(fn):
    q, k, v, do = _packed(LENS, 4, 2)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP).to("cuda")
    o, lse = fn(q, k, v, causal=True, return_lse=True, doc_maps=maps)
    S = maps.rows_q   # (the aligned row space: the padded size, or 128 x the tile bound of compact maps)
    qa, ka, va = (torch.zeros((S,) + t.shape[1:], dtype=t.dtype, device="cuda").index_copy_(0, maps.dst_q.long(), t)
                  for t in (q, k, v))
    oa, la = fn(qa, ka, va, causal=True, return_lse=True, cu_seqlens_q=maps.cu_pad_q, cu_seqlens_k=maps.cu_pad_k,
                seqused_q=maps.len_q, seqused_k=maps.len_k, max_seqlen_q=maps.max_seqlen_q,
                max_seqlen_k=maps.max_seqlen_k)
    dst = maps.dst_q.long()
    assert torch.equal(o, oa[dst]) and torch.equal(lse, la[:, dst])


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
def test_no_sync_and_graph(fn, O):
    lens_a, lens_b = LENS, [700, 3, 300, 129, 0, 1000, 299]
    assert sum(lens_a) == sum(lens_b)
    q, k, v, do = _packed(lens_a, 4, 2)
    q2, k2, v2, do2 = _packed(lens_b, 4, 2, seed=1)
    ma = DocMaps.from_cu_seqlens(_cu(lens_a), max_num_documents=CAP, max_seqlen=1024)
    mb = DocMaps.from_cu_seqlens(_cu(lens_b), max_num_documents=CAP, max_seqlen=1024)
    static = ma.to("cuda")
    kw = dict(causal=True, deterministic=True)
    ref_a = _run(fn, q, k, v, do, doc_maps=ma.to("cuda"), **kw)
    ref_b = _run(fn, q2, k2, v2, do2, doc_maps=mb.to("cuda"), **kw)
    xs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    gdo = do.clone()
    torch.cuda.set_sync_debug_mode("error")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):   # warm-up (compiles, plan cache)
                for x in xs:
                    x.grad = None
                fn(*xs, doc_maps=static, **kw).backward(gdo)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        for x in xs:
            x.grad = None
        with torch.cuda.graph(g):
            o = fn(*xs, doc_maps=static, **kw)
            o.backward(gdo)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    for src, dst in (((q, k, v, do), (*xs, gdo)),):
        for a, b in zip(src, dst):
            b.data.copy_(a)
    g.replay()
    assert torch.equal(o, ref_a[0]) and all(torch.equal(x.grad, r) for x, r in zip(xs, ref_a[2:]))
    static.copy_(mb.to("cuda"))
    for a, b in zip((q2, k2, v2, do2), (*xs, gdo)):
        b.data.copy_(a)
    g.replay()
    assert torch.equal(o, ref_b[0]) and all(torch.equal(x.grad, r) for x, r in zip(xs, ref_b[2:]))


class _Docs(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        lens = torch.randint(1, 900, (int(torch.randint(2, 9, (1,), generator=g)),), generator=g).tolist()
        lens[-1] += 4096 - sum(lens) if sum(lens) < 4096 else 0
        cu = _cu(lens)
        return cu, DocMaps.from_cu_seqlens(cu.numpy(), max_num_documents=16, max_seqlen=4096)


def test_dataloader_worker_cpu():
    dl = torch.utils.data.DataLoader(_Docs(), batch_size=None, num_workers=2,
                                     pin_memory=torch.cuda.is_available())
    for cu, maps in dl:
        ref = DocMaps.from_cu_seqlens(cu, max_num_documents=16, max_seqlen=4096)
        assert maps.shape_key() == ref.shape_key() and torch.equal(maps.buf, ref.buf)
        if torch.cuda.is_available():
            assert maps.buf.is_pinned()
            dev = maps.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            assert torch.equal(dev.buf.cpu(), ref.buf)


@gpu
@pytest.mark.parametrize("fn,O", [(flash_attn_fp8, O8), (flash_attn_fp4, O4)], ids=["fp8", "fp4"])
@pytest.mark.parametrize("case", ["two_cta", "fwd_two_cta", "window_softcap", "per_tensor"])
def test_bitwise_repack_options(fn, O, case):
    """doc_maps == the repacking path under the options that change the kernels / route: the 2-CTA backward (MHA:
    dK / dV bitwise the 1-CTA kernel's either way), the 2-CTA forward, window + softcap (the zero-filled dK / dV),
    per-tensor operands (the plain-gather path)."""
    q, k, v, do = _packed(LENS, 4, 4)
    cu = _cu(LENS)
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=CAP).to("cuda")
    cuc = cu.cuda()
    kw = dict(causal=True, deterministic=True)
    if case == "two_cta":
        kw["options"] = O(two_cta="on")
    elif case == "fwd_two_cta":
        kw["options"] = O(forward_two_cta="on", two_cta="off")
    elif case == "window_softcap":
        kw.update(window_size=(200, 0), softcap=30.0)
    else:
        kw["options"] = O(operand_scaling="per_tensor")
    r = _run(fn, q, k, v, do, cu_seqlens_q=cuc, cu_seqlens_k=cuc, max_seqlen_q=1000, max_seqlen_k=1000, **kw)
    m = _run(fn, q, k, v, do, doc_maps=maps, **kw)
    idx = (0, 3, 4) if case == "two_cta" else (0, 2, 3, 4)   # (2-CTA: dQ's order follows the key-pair packing)
    for i in idx:
        assert torch.equal(r[i], m[i]), (case, i)


@gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_bitwise_repack_all_fp4(causal):
    """The 4-bit P.V routes (the plain-gather path, the key check from the maps): bitwise the repacking path."""
    lens = [300, 1000, 45, 700, 128, 257]
    q, k, v, do = _packed(lens, 4, 4)
    cu = _cu(lens)
    # (the all-4-bit backward's stochastic rounding hashes each element's position in the aligned buffer: the
    # repacking path's allocation, round_up(T + n 127, 128), gives the same dither)
    maps = DocMaps.from_cu_seqlens(cu, padded_size=-(-(sum(lens) + len(lens) * 127) // 128) * 128).to("cuda")
    cuc = cu.cuda()
    kw = dict(causal=causal, pv_precision="fp4", backward="fp4" if causal else "e4m3")
    if not causal:
        with torch.no_grad():
            a = flash_attn_fp4(q, k, v, cu_seqlens_q=cuc, cu_seqlens_k=cuc, max_seqlen_q=1000, max_seqlen_k=1000,
                               **kw)
            b = flash_attn_fp4(q, k, v, doc_maps=maps, **kw)
        assert torch.equal(a, b)
        return
    import flash_attn.cute.interface_bwd_fp4q as iq
    iq.reset_sr_counter()   # (the all-4-bit backward's stochastic-rounding seed: one per call, the same for both)
    r = _run(flash_attn_fp4, q, k, v, do, cu_seqlens_q=cuc, cu_seqlens_k=cuc, max_seqlen_q=1000, max_seqlen_k=1000,
             **kw)
    iq.reset_sr_counter()
    m = _run(flash_attn_fp4, q, k, v, do, doc_maps=maps, **kw)
    for i in (0, 2, 3, 4):
        assert torch.equal(r[i], m[i]), i


@gpu
def test_refusals():
    q, k, v, do = _packed(LENS, 4, 4)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP).to("cuda")
    with pytest.raises(ValueError, match="do not pass"):
        flash_attn_fp8(q, k, v, doc_maps=maps, cu_seqlens_q=maps.cu_q, cu_seqlens_k=maps.cu_q)
    with pytest.raises(ValueError, match="total_q"):
        flash_attn_fp8(q[:-1], k[:-1], v[:-1], doc_maps=maps)
    with pytest.raises(ValueError, match="max_num_documents"):
        DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=3)
    with pytest.raises(ValueError, match="padded size"):
        DocMaps.from_cu_seqlens(_cu(LENS), padded_size=2048 + 64)
    with pytest.raises(ValueError, match="max_seqlen"):
        DocMaps.from_cu_seqlens(_cu(LENS), max_seqlen=100)


@gpu
@pytest.mark.parametrize("fn", [flash_attn_fp8, flash_attn_fp4], ids=["fp8", "fp4"])
@pytest.mark.parametrize("d", [128, 64])
def test_fused_no_copies(fn, d):
    """The fused path copies nothing (no row gather of an input or output) and builds its work lists once per maps
    (none on a repeated call); the call goes through the registered op lowbit_attn::fp4 / ::fp8."""
    from torch.profiler import ProfilerActivity, profile
    q, k, v, do = _packed(LENS, 4, 2, d)
    maps = DocMaps.from_cu_seqlens(_cu(LENS), max_num_documents=CAP).to("cuda")
    xs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    o = fn(*xs, causal=True, doc_maps=maps)
    assert "lowbit_attn" in type(o.grad_fn.next_functions[0][0]).__name__
    o.backward(do)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(2):
            fn(*xs, causal=True, doc_maps=maps).backward(do)
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    assert not [n for n in names if "row_gather" in n or "_worklist" in n or "gather_kernel" in n], set(names)

# spd3: the 1-CTA packed causal forward's TMA-tile O epilogue (flash_fwd_sm100_fp4 o_rows_tma1, the driver's
# interface_fwd_fp4.O_PACKED_TILE) -- O / LSE bitwise the correction warps' row copies (O_PACKED_TILE = False) on
# documents with partial last tiles, 1-token documents and whole tiles, for every packed route it reaches.
import pytest
import torch

from flash_attn.cute import interface_fwd_fp4 as IF
from flash_attn.cute import lp_docmaps as DM
from flash_attn.cute.lowprec import DocMaps, flash_attn_fp4, flash_attn_fp8

E4 = torch.float8_e4m3fn
LENS = [300, 1000, 0, 45, 700, 1, 128, 257, 384, 3]


def _bits(a, b):
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def _call(route, d, hk, cap):
    cu = [0]
    for L in LENS:
        cu.append(cu[-1] + L)
    T = cu[-1]
    maps = DocMaps.from_cu_seqlens(cu, max_num_documents=cap or T, max_seqlen=2048).to("cuda")
    g = torch.Generator(device="cuda").manual_seed(d + hk)
    if route == "pq":
        q8, k8, v8 = ((torch.randn(T, h, d, device="cuda", generator=g) * 40).to(E4) for h in (4, hk, hk))
        ds = [torch.rand(maps.max_num_documents, hk, device="cuda", generator=g) * 0.05 + 0.01 for _ in range(3)]
        return lambda: flash_attn_fp8(q8, k8, v8, q_descale=ds[0], k_descale=ds[1], v_descale=ds[2], doc_maps=maps,
                                      causal=True, return_lse=True)
    q = torch.randn(T, 4, d, device="cuda", generator=g).to(torch.bfloat16)
    k, v = (torch.randn(T, hk, d, device="cuda", generator=g).to(torch.bfloat16) for _ in range(2))
    if route == "optA":
        return lambda: flash_attn_fp4(q, k, v, causal=True, doc_maps=maps, return_lse=True, pv_precision="fp4")
    fn = flash_attn_fp8 if route == "fp8" else flash_attn_fp4
    return lambda: fn(q, k, v, causal=True, doc_maps=maps, return_lse=True)


@pytest.mark.parametrize("route,d,hk,cap", [
    ("pq", 128, 4, 16), ("pq", 64, 2, 16), ("pq", 128, 2, None),
    ("fp8", 128, 2, 16), ("fp8", 64, 4, 16), ("fp4", 128, 2, 16),
    ("optA", 128, 1, 16),
])
def test_packed_tile_epilogue_bitwise(route, d, hk, cap, monkeypatch):
    run = _call(route, d, hk, cap)
    with torch.no_grad():
        c0 = DM.PACKED_CALLS[0]
        monkeypatch.setattr(IF, "O_PACKED_TILE", False)
        ref = [t.clone() for t in run()]
        monkeypatch.setattr(IF, "O_PACKED_TILE", True)
        got = run()
    torch.cuda.synchronize()
    assert DM.PACKED_CALLS[0] == c0 + 2, "packed path"
    assert _bits(ref[0], got[0]) and _bits(ref[1], got[1])

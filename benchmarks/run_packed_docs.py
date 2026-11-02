"""Run the packed-document benchmark of the paper (Tables 1-3 and their non-causal / B300 variants).

    python benchmarks/run_packed_docs.py --out results/ [--groups ours,baselines] [--cells eq4x4096-c,...]
        [--rounds 0,1,2,3] [--python /path/to/python] [--gpu 0]

Workloads: 16 query heads, d = 128, every length set causal (-c) and non-causal (-d), MHA, plus the 16k log-uniform set
with GQA 16/4 (-hk4).

Method: one fresh process per job (timeout 330 s), a discarded warm-up round 0 that fills the compile cache, timed
rounds 1-3 (round 1 also records accuracy against an FP32 reference), all jobs of a workload back to back, job order
reversed on even rounds. Rows go to <out>/r<round>.jsonl; summarize_packed_docs.py takes the median over the timed
rounds. Use an otherwise idle GPU.

Groups (each needs its own environment; run the script once per environment with --python):
  ours       ours-fp8 / ours-fp4 at 128-aligned document starts (forward, backward, training call) and on the packed
             rows through DocMaps (training call, capacity = number of documents and = T).
  baselines  every baseline in its own varlen / ragged API: FlashAttention-4 BF16 / FP8, cuDNN BF16 / FP8 / MXFP8,
             FlashInfer BF16 / FP8 (CuTe DSL and CUTLASS) and NVFP4 KV cache, PyTorch SDPA on nested tensors, Low
             Precision FA4 (meta-mxfp8) and HW-FP4 FA4 (hu-nvmx). A backend that is not installed, or that has no
             packed-document path, gives a row with the reason.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark_packed_docs.py")

SETS = ("eq4x4096", "eq16x1024", "eq64x1024", "eq16x4096", "skew8192+32x256", "skew8192+224x256",
        "logu128-8192-t16384-s0", "logu128-8192-t65536-s0")
CELLS = {}
for _s in SETS:
    for _c in "cd":
        CELLS[f"{_s}-{_c}"] = (_s, _c == "c", 16)
for _c in "cd":
    CELLS[f"logu128-8192-t16384-s0-{_c}-hk4"] = ("logu128-8192-t16384-s0", _c == "c", 4)

BASELINES = ("fa4-bf16,fa4-fp8,cudnn-bf16,cudnn-fp8,cudnn-mxfp8,flashinfer-bf16-cutedsl,flashinfer-fp8-cutedsl,"
             "flashinfer-bf16-cutlass,flashinfer-fp8-cutlass,flashinfer-nvfp4kv,torch-sdpa,meta-mxfp8,hu-nvmx")


def jobs(cell, groups):
    set_, causal, hk = CELLS[cell]
    base = ["--set", set_, "--causal", "true" if causal else "false", "--hk", str(hk), "--tag", f"cell={cell}"]
    out = []
    if "ours" in groups:
        out += [("ours-aligned", base + ["--format", "aligned", "--backend", "ours-fp8,ours-fp4", "--mode", "fwd,bwd,step"]),
                ("ours-docmaps-docs", base + ["--format", "docmaps", "--dm-n", "docs", "--backend", "ours-fp8,ours-fp4",
                                              "--mode", "step"]),
                ("ours-docmaps-T", base + ["--format", "docmaps", "--dm-n", "T", "--backend", "ours-fp8,ours-fp4",
                                           "--mode", "step"])]
    if "baselines" in groups:
        out.append(("baselines", base + ["--format", "varlen", "--backend", BASELINES, "--mode", "fwd,bwd,step"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="ours,baselines")
    ap.add_argument("--cells", default="", help="comma-separated subset of: " + ", ".join(CELLS))
    ap.add_argument("--rounds", default="0,1,2,3")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    a = ap.parse_args()
    groups = set(a.groups.split(","))
    cells = [c for c in CELLS if not a.cells or c in a.cells.split(",")]
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1")
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    for r in (int(x) for x in a.rounds.split(",")):
        for cell in cells:
            js = jobs(cell, groups)
            if r % 2 == 0:
                js = js[::-1]
            for name, args in js:
                jsonl = os.path.join(a.out, f"r{r}.jsonl")
                cmd = ["timeout", "330", a.python, BENCH] + args + ["--jsonl", jsonl, "--tag", f"round={r}",
                                                                      "--tag", f"job={name}"]
                if r != 1:
                    cmd.append("--no-acc")
                rc = subprocess.run(cmd, env=env).returncode
                print(f"round {r} {cell} {name}: rc {rc}", flush=True)


if __name__ == "__main__":
    main()

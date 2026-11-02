"""Summarize run_packed_docs.py rows: median time over the timed rounds, one markdown table per mode.

    python benchmarks/summarize_packed_docs.py results/ [--rounds 1,2,3]

Columns are the configurations (backend, plus the layout for ours: aligned, docmaps-docs, docmaps-T); cells are
"ms / TFLOP/s" (TFLOP/s on within-document query-key pairs only), or the reason a backend did not run.
Speedup = the fastest baseline's time over our fastest time on that workload.
"""
import argparse
import collections
import glob
import json
import os
import statistics


def label(r):
    if r["config"].startswith("ours-"):
        lay = r.get("format")
        if lay == "docmaps":
            lay = f"docmaps-{r.get('job', '').rsplit('-', 1)[-1]}"
        return f"{r['config']} ({lay})"
    return r["config"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--rounds", default="1,2,3")
    a = ap.parse_args()
    rows = []
    for rd in a.rounds.split(","):
        for f in glob.glob(os.path.join(a.out, f"r{rd}.jsonl")):
            rows += [json.loads(line) for line in open(f)]
    ok = collections.defaultdict(list)
    why = {}
    cells, cols = [], []
    for r in rows:
        c, col = r.get("cell"), label(r)
        if c not in cells:
            cells.append(c)
        if col not in cols:
            cols.append(col)
        if r.get("status") == "ok":
            ok[(r["kernel"], c, col)].append((r["ms_median"], r["tflops"]))
        else:
            for k in (("fwd", "bwd", "step") if r.get("kernel") is None else (r["kernel"],)):
                why.setdefault((k, c, col), r.get("error", r.get("status", ""))[:60])
    for mode in ("step", "fwd", "bwd"):
        print(f"\n## {dict(step='Training call (forward + backward, quantization included)', fwd='Forward', bwd='Backward')[mode]}\n")
        used = [col for col in cols if any((mode, c, col) in ok for c in cells)]
        print("| workload | " + " | ".join(used) + " | speedup |")
        print("|---" * (len(used) + 2) + "|")
        for c in cells:
            med = {col: statistics.median(m for m, _ in ok[(mode, c, col)]) for col in used if (mode, c, col) in ok}
            tf = {col: statistics.median(t for _, t in ok[(mode, c, col)]) for col in med}
            cell = [f"{med[col]:.3f} / {tf[col]:.0f}" if col in med else why.get((mode, c, col), "") for col in used]
            ours = [m for col, m in med.items() if col.startswith("ours-")]
            base = [m for col, m in med.items() if not col.startswith("ours-")]
            sp = f"{min(base) / min(ours):.2f}x" if ours and base else ""
            print(f"| {c} | " + " | ".join(cell) + f" | {sp} |")


if __name__ == "__main__":
    main()

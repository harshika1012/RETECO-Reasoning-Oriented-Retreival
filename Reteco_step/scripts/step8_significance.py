"""Step 8: paired significance tests between saved runs, on a fixed query subset.

Each run is compared with the BASELINE (first --runs entry) and with the run just above it in the list.
Test = paired bootstrap, stratified by domain, on the macro (domain-averaged) nDCG@10 difference:
resample queries with replacement inside each domain, recompute both macro averages, B times.
Reports the difference, its 95% interval, and a two-sided p-value.

Usage:
  python3 scripts/step8_significance.py --root ../reteco_data --subset dev
  python3 scripts/step8_significance.py --root ../reteco_data --subset tune    # held-out 20% of train
  python3 scripts/step8_significance.py --root ../reteco_data --runs bm25=runs/bm25_official ce=runs/ce_...
Run specs: label=<run dir with <domain>/top200_<split>.jsonl>, or label=rrf:<pool dir> for the Step 4 RRF ranking.
Output: printed table + results/significance_<subset>.csv (one row per run).
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import ndcg_at_k
from reteco_io import TRACK1_DOMAINS, load_qrels
from run_io import load_topk
from splits import subset_qids

DEFAULT_RUNS = [
    "bm25=runs/bm25_official",
    "dense=runs/dense_BAAI__bge-base-en-v1.5_L512",
    "rrf=rrf:runs/pool",
    "rrf+cos=runs/graph_BAAI__bge-base-en-v1.5_L512",
    "ce-base=runs/ce_BAAI__bge-reranker-base_L512",
    "ce-v2m3=runs/ce_BAAI__bge-reranker-v2-m3_L512",
]


def rrf_from_pool(path):
    run = {}
    for line in open(path):
        p = json.loads(line)
        def s(c): return sum(1.0 / (60 + c[r]) for r in ("bm25_rank", "dense_rank") if c[r])
        run[p["qid"]] = [c["doc"] for c in sorted(p["candidates"], key=lambda c: (-s(c), c["doc"]))]
    return run


def load_run(spec, dom, split):
    if spec.startswith("rrf:"):
        return rrf_from_pool(Path(spec[4:]) / dom / f"pool_{split}.jsonl")
    return {q: [d for d, _ in v] for q, v in load_topk(Path(spec) / dom / f"top200_{split}.jsonl").items()}


def per_query(args, specs):
    """{label: {dom: np.array of per-query nDCG@10}}, all runs aligned on the same queries."""
    split = "dev" if args.subset == "dev" else "train"
    scores = {lab: {} for lab, _ in specs}
    for dom in args.domains:
        qrels = load_qrels(args.root, dom, split)
        qids = sorted(subset_qids(args.root, dom, args.subset))
        for lab, spec in specs:
            run = load_run(spec, dom, split)
            scores[lab][dom] = np.array([ndcg_at_k(run.get(q, []), qrels[q], 10) or 0.0 for q in qids])
    return scores


def bootstrap(a, b, B, rng):
    """a, b: {dom: per-query arrays}. Returns (macro diff b - a, ci_low, ci_high, p two-sided)."""
    doms = list(a)
    obs = np.mean([b[d].mean() - a[d].mean() for d in doms])
    boot = np.zeros(B)
    for d in doms:
        diff = b[d] - a[d]
        idx = rng.integers(0, len(diff), size=(B, len(diff)))
        boot += diff[idx].mean(axis=1)
    boot /= len(doms)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p: how often the resampled difference falls on the other side of zero
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return obs, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", default="dev", choices=["dev", "tune", "fit", "train"])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="label=path, first one is the baseline")
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    specs = [tuple(r.split("=", 1)) for r in args.runs]
    rng = np.random.default_rng(args.seed)

    scores = per_query(args, specs)
    n_q = sum(len(v) for v in scores[specs[0][0]].values())
    print(f"subset={args.subset} | {len(args.domains)} domains | {n_q} queries | B={args.B} | diffs in nDCG points (x100)\n")
    head = f"{'run':<12}{'nDCG@10':>9}  {'vs ' + specs[0][0]:<34}{'vs previous row':<34}"
    print(head); print("-" * len(head))
    rows = []
    for i, (lab, spec) in enumerate(specs):
        macro = float(np.mean([v.mean() for v in scores[lab].values()]))
        row = {"run": lab, "path": spec, "subset": args.subset, "ndcg10": round(macro, 4)}
        cells = []
        for key, ref in (("base", specs[0][0]), ("prev", specs[i - 1][0] if i else None)):
            if not i or ref is None:
                cells.append(""); continue
            d, lo, hi, p = bootstrap(scores[ref], scores[lab], args.B, rng)
            row.update({f"{key}_diff": round(100 * d, 2), f"{key}_ci_low": round(100 * lo, 2),
                        f"{key}_ci_high": round(100 * hi, 2), f"{key}_p": round(p, 4)})
            star = "*" if p < 0.05 else " "
            cells.append(f"{100 * d:+6.2f} [{100 * lo:+6.2f},{100 * hi:+6.2f}] p={p:.3f}{star}")
        print(f"{lab:<12}{macro:>9.4f}  {cells[0]:<34}{cells[1]:<34}")
        rows.append(row)
    print("\n* = p < 0.05 (two-sided paired bootstrap, stratified by domain)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"significance_{args.subset}.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "path", "subset", "ndcg10"), k))
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"saved {f}")


if __name__ == "__main__":
    main()

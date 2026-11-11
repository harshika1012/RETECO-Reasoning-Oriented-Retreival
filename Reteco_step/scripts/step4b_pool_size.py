"""Step 4b: how big does the pool need to be? Recall of the top-N of the BM25+dense RRF list, and distinct docs.
Usage: python3 scripts/step4b_pool_size.py --root ../reteco_data
Reads runs/pool/<domain>/pool_<split>.jsonl (from Step 4). Pick the smallest N whose recall is close to the full pool's."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import macro_average
from reteco_io import TRACK1_DOMAINS, load_qrels

NS = [20, 50, 100, 150, 200, 300, 400]


def rrf_order(cands, k=60):
    def sc(c):
        return sum(1.0 / (k + c[r]) for r in ("bm25_rank", "dense_rank") if c[r])
    return [c["doc"] for c in sorted(cands, key=lambda c: (-sc(c), c["doc"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool-dir", default="runs/pool")
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    args = ap.parse_args()
    for split in args.splits:
        rec = {n: {} for n in NS}; dist = {n: 0 for n in NS}
        for dom in args.domains:
            qrels = load_qrels(args.root, dom, split)
            seen = {n: set() for n in NS}; r = {n: [] for n in NS}
            for line in open(Path(args.pool_dir) / dom / f"pool_{split}.jsonl"):
                p = json.loads(line); gold = {d for d, v in qrels.get(p["qid"], {}).items() if v > 0}
                if not gold:
                    continue
                order = rrf_order(p["candidates"])
                for n in NS:
                    top = order[:n]; seen[n].update(top); r[n].append(len(gold & set(top)) / len(gold))
            for n in NS:
                rec[n][dom] = float(np.mean(r[n])); dist[n] += len(seen[n])
        print(f"\n{split}: RRF top-N recall (macro over domains) and distinct docs (sum over domains)")
        for n in NS:
            print(f"  N={n:<4} recall {macro_average(rec[n]):.3f}  distinct docs {dist[n]}")


if __name__ == "__main__":
    main()

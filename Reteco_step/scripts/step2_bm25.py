"""Step 2: BM25 top-200 for every domain/split, saved to disk and evaluated.

Usage:
  python scripts/step2_bm25.py --root ../reteco_data                      # all 13 domains, train+dev
  python scripts/step2_bm25.py --root ../reteco_data --domains iota law   # quick smoke test
Outputs (per domain/split):
  runs/bm25/<domain>/top200_<split>.jsonl   {qid, docs, scores}   <- used by later steps
  runs/bm25/<domain>/run_1a_<split>.trec    TREC run file (final packaging adapted after Step 2 gate)
  runs/bm25/summary.json                    per-domain + macro nDCG@10 / Recall@200
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bm25 import BM25Index
from metrics import evaluate_run, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import write_trec, save_topk, ids_only

# Reference BM25 for 1a on data v1.1 (RETECO README), macro nDCG@10 over domains
REFERENCE = {"train": 0.1075, "dev": 0.1147}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--out", default="runs/bm25")
    ap.add_argument("--cache", default="cache/bm25_index")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    summary = {s: {"ndcg10": {}, "recall": {}} for s in args.splits}
    for dom in args.domains:
        t0 = time.time()
        idx_path = Path(args.cache) / f"{dom}_k1{args.k1}_b{args.b}.pkl"
        if idx_path.exists() and not args.rebuild:
            index = BM25Index.load(idx_path)
        else:
            docs = load_documents(args.root, dom)
            index = BM25Index(args.k1, args.b).build(list(docs), list(docs.values()))
            index.save(idx_path)
            del docs
        t_index = time.time() - t0
        for split in args.splits:
            examples = load_examples(args.root, dom, split)
            qrels = load_qrels(args.root, dom, split)
            run = {e["id"]: index.search(e["query"], args.k) for e in examples}
            out = Path(args.out) / dom
            save_topk(out / f"top{args.k}_{split}.jsonl", run)
            write_trec(out / f"run_1a_{split}.trec", run, "bm25")
            nd, _ = evaluate_run(ids_only(run), qrels, 10)
            rc = [recall_at_k([d for d, _ in run.get(q, [])], qr, args.k) for q, qr in qrels.items()]
            rc = [r for r in rc if r is not None]
            summary[split]["ndcg10"][dom] = nd
            summary[split]["recall"][dom] = sum(rc) / len(rc)
        print(f"{dom:<10} index {t_index:6.1f}s | " + " | ".join(
            f"{s}: nDCG@10 {summary[s]['ndcg10'][dom]:.4f} R@{args.k} {summary[s]['recall'][dom]:.3f}"
            for s in args.splits), flush=True)

    print("\nMacro average over domains")
    for s in args.splits:
        m = macro_average(summary[s]["ndcg10"]); r = macro_average(summary[s]["recall"])
        summary[s]["macro_ndcg10"], summary[s]["macro_recall"] = m, r
        ref = REFERENCE.get(s)
        note = f" (reference {ref:.4f}, diff {m - ref:+.4f}; only comparable when all 13 domains are run)" if ref else ""
        print(f"  {s}: nDCG@10 {m:.4f}{note} | Recall@{args.k} {r:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()

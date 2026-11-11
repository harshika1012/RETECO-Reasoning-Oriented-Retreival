"""Step 2 (Lucene route): BM25 top-200 via Pyserini/Lucene, to match the official starter-kit baseline.

Why: the official baseline (pyserini/Lucene) scores 0.1147 dev / 0.1075 train; our pure-Python BM25 scored lower,
most likely because of tokenizer differences. This script uses Lucene's own default analyzer + BM25.

Usage:
  python scripts/step2_bm25_pyserini.py --root ../reteco_data --domains iota law     # check vs official per-domain numbers
  python scripts/step2_bm25_pyserini.py --root ../reteco_data                        # all 13 domains

Outputs mirror Step 2 (same formats), under runs/bm25_lucene/ by default.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate_run, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import write_trec, save_topk, ids_only

REFERENCE = {"train": 0.1075, "dev": 0.1147}
# Official starter-kit per-domain 1a nDCG@10 (train, dev), from your official_baseline.py run
OFFICIAL = {
    "bitcoin": (0.0856, 0.0370), "cardano": (0.1848, 0.1027), "economics": (0.0382, 0.0467),
    "genealogy": (0.1242, 0.1949), "history": (0.0654, 0.0873), "hsm": (0.1736, 0.2489),
    "iota": (0.1563, 0.3309), "law": (0.0946, 0.0574), "monero": (0.0382, 0.0320),
    "politics": (0.2810, 0.2625), "quant": (0.0255, 0.0237), "travel": (0.0506, 0.0442),
    "workplace": (0.0798, 0.0230),
}


def build_index(root, domain, index_dir, threads):
    """Write {id, contents} JSONL for the domain, then build a Lucene index with Pyserini's CLI."""
    if (index_dir / "segments_1").exists() or any(index_dir.glob("segments_*")):
        return
    src = index_dir.parent / f"{domain}_src"
    src.mkdir(parents=True, exist_ok=True)
    docs = load_documents(root, domain)
    with open(src / "docs.jsonl", "w", encoding="utf-8") as f:
        for did, text in docs.items():
            f.write(json.dumps({"id": did, "contents": text}) + "\n")
    del docs
    cmd = [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
           "--input", str(src), "--index", str(index_dir), "--generator", "DefaultLuceneDocumentGenerator",
           "--threads", str(threads)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(src, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/bm25_lucene")
    ap.add_argument("--cache", default="cache/pyserini_index")
    args = ap.parse_args()

    # pyserini.encode builds an OpenAI client at import time; newer openai raises
    # without a key even though BM25 never calls it. A placeholder is enough.
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    from pyserini.search.lucene import LuceneSearcher  # needs Java; same stack as the official baseline

    summary = {s: {"ndcg10": {}, "recall": {}} for s in args.splits}
    for dom in args.domains:
        t0 = time.time()
        index_dir = Path(args.cache) / dom
        build_index(args.root, dom, index_dir, args.threads)
        searcher = LuceneSearcher(str(index_dir))
        searcher.set_bm25(args.k1, args.b)
        t_index = time.time() - t0
        for split in args.splits:
            examples = load_examples(args.root, dom, split)
            qrels = load_qrels(args.root, dom, split)
            qids = [e["id"] for e in examples]
            hits = searcher.batch_search([e["query"] for e in examples], qids, k=args.k, threads=args.threads)
            run = {q: [(h.docid, float(h.score)) for h in hits.get(q, [])] for q in qids}
            out = Path(args.out) / dom
            save_topk(out / f"top{args.k}_{split}.jsonl", run)
            write_trec(out / f"run_1a_{split}.trec", run, "bm25_lucene")
            nd, _ = evaluate_run(ids_only(run), qrels, 10)
            rc = [recall_at_k([d for d, _ in run.get(q, [])], qr, args.k) for q, qr in qrels.items()]
            rc = [r for r in rc if r is not None]
            summary[split]["ndcg10"][dom] = nd
            summary[split]["recall"][dom] = sum(rc) / len(rc)
        parts = []
        for s in args.splits:
            off = OFFICIAL.get(dom, (None, None))[0 if s == "train" else 1]
            off_txt = f" (official {off:.4f})" if off is not None else ""
            parts.append(f"{s}: nDCG@10 {summary[s]['ndcg10'][dom]:.4f}{off_txt} R@{args.k} {summary[s]['recall'][dom]:.3f}")
        print(f"{dom:<10} index {t_index:6.1f}s | " + " | ".join(parts), flush=True)

    print("\nMacro average over domains")
    for s in args.splits:
        m = macro_average(summary[s]["ndcg10"]); r = macro_average(summary[s]["recall"])
        summary[s]["macro_ndcg10"], summary[s]["macro_recall"] = m, r
        ref = REFERENCE.get(s)
        print(f"  {s}: nDCG@10 {m:.4f} (reference {ref:.4f}, diff {m - ref:+.4f}; comparable only with all 13 domains) | Recall@{args.k} {r:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()

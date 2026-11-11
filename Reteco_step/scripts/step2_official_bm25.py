"""Step 2 (final): BM25 top-k using the OFFICIAL starter-kit retriever (import, not re-implementation).

The official baseline = pyserini Lucene *analyzer* + gensim LuceneBM25Model (k1=0.9, b=0.4). We import its
`OfficialBM25`, `load_corpus`, `topics_1a` unchanged and only add: top-k saving, our metrics, deeper recall.

Usage (from the project folder):
  python3 scripts/step2_official_bm25.py --root ../reteco_data --starter-kit ../RETECO/starter_kit --domains iota law
  python3 scripts/step2_official_bm25.py --root ../reteco_data --starter-kit ../RETECO/starter_kit
Outputs per domain/split:
  runs/bm25_official/<domain>/top200_<split>.jsonl   (used by later steps)
  runs/bm25_official/<domain>/run_1a_<split>.trec
  runs/bm25_official/summary.json
Finished domains are skipped on re-run (use --force to redo).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate_run, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_qrels
from run_io import write_trec, save_topk, ids_only

REFERENCE = {"train": 0.1075, "dev": 0.1147}
OFFICIAL = {  # per-domain 1a nDCG@10 (train, dev) printed by your official_baseline.py run
    "bitcoin": (0.0856, 0.0370), "cardano": (0.1848, 0.1027), "economics": (0.0382, 0.0467),
    "genealogy": (0.1242, 0.1949), "history": (0.0654, 0.0873), "hsm": (0.1736, 0.2489),
    "iota": (0.1563, 0.3309), "law": (0.0946, 0.0574), "monero": (0.0382, 0.0320),
    "politics": (0.2810, 0.2625), "quant": (0.0255, 0.0237), "travel": (0.0506, 0.0442),
    "workplace": (0.0798, 0.0230),
}
RECALL_KS = [100, 200, 500, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="reteco_data folder")
    ap.add_argument("--starter-kit", required=True, help="path to RETECO/starter_kit")
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--k", type=int, default=200, help="how many docs to save per query")
    ap.add_argument("--out", default="runs/bm25_official")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.starter_kit).resolve()))
    import official_baseline as ob  # the official file; its main() is guarded by __name__

    out_root = Path(args.out)
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for s in args.splits:
        summary.setdefault(s, {"ndcg10": {}, "recall": {}})

    for dom in args.domains:
        done = all((out_root / dom / f"top{args.k}_{s}.jsonl").exists() for s in args.splits)
        if done and not args.force and all(dom in summary[s]["ndcg10"] for s in args.splits):
            print(f"[skip] {dom} already done", flush=True)
            continue
        t0 = time.time()
        ddir = Path(args.root) / "track1_tempo" / dom
        doc_ids, texts = ob.load_corpus(str(ddir / "documents.jsonl"), "id")
        bm25 = ob.OfficialBM25(texts, doc_ids)
        del texts
        t_index = time.time() - t0
        parts = []
        for split in args.splits:
            queries, qids, _ = ob.topics_1a(str(ddir), split)
            scores = bm25.search(queries, qids, desc=f"  {dom}/{split}")       # top-1000, sorted desc
            full = {q: list(v.items()) for q, v in scores.items()}              # [(doc, score)] rank order
            run = {q: items[:args.k] for q, items in full.items()}
            save_topk(out_root / dom / f"top{args.k}_{split}.jsonl", run)
            write_trec(out_root / dom / f"run_1a_{split}.trec", run, "bm25_official")
            qrels = load_qrels(args.root, dom, split)
            nd, _ = evaluate_run(ids_only(full), qrels, 10)
            rec = {}
            for kk in RECALL_KS:
                vals = [recall_at_k([d for d, _ in full.get(q, [])], qr, kk) for q, qr in qrels.items()]
                vals = [v for v in vals if v is not None]
                rec[str(kk)] = sum(vals) / len(vals)
            summary[split]["ndcg10"][dom] = nd
            summary[split]["recall"][dom] = rec
            off = OFFICIAL.get(dom, (None, None))[0 if split == "train" else 1]
            parts.append(f"{split}: nDCG@10 {nd:.4f}" + (f" (official {off:.4f})" if off else "")
                         + f" R@200 {rec['200']:.3f}")
        print(f"{dom:<10} {t_index:6.0f}s index | " + " | ".join(parts), flush=True)
        out_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

    print("\nMacro average over the domains in summary.json")
    for s in args.splits:
        nd = macro_average(summary[s]["ndcg10"])
        ref = REFERENCE[s]
        line = f"  {s}: nDCG@10 {nd:.4f} (reference {ref:.4f}, diff {nd - ref:+.4f}; n_domains={len(summary[s]['ndcg10'])})"
        rc = {k: macro_average({d: r[k] for d, r in summary[s]["recall"].items()}) for k in map(str, RECALL_KS)}
        print(line + " | recall " + ", ".join(f"@{k} {v:.3f}" for k, v in rc.items()))


if __name__ == "__main__":
    main()

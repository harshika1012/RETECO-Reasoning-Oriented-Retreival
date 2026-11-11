"""Step 1: load data, print stats, and verify our nDCG@10 against pytrec_eval.

Usage:
  python scripts/step1_verify.py --root /path/to/reteco_data --domains bitcoin cardano law
  python scripts/step1_verify.py --root /path/to/reteco_data          # all 13 domains
"""
import argparse, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytrec_eval
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from metrics import evaluate_run, recall_at_k


def make_runs(doc_ids, qrels, seed=0):
    rng = random.Random(seed)
    perfect, noisy, rand = {}, {}, {}
    for qid, qrel in qrels.items():
        rel = [d for d, r in qrel.items() if r > 0]
        others = rng.sample(doc_ids, min(len(doc_ids), 50))
        perfect[qid] = rel + [d for d in others if d not in qrel]
        mixed = others[:5] + rel + others[5:]          # relevant docs pushed down
        noisy[qid] = list(dict.fromkeys(mixed))
        rand[qid] = rng.sample(doc_ids, min(len(doc_ids), 50))
    return {"perfect": perfect, "noisy": noisy, "random": rand}


def pytrec_mean(run, qrels, k=10):
    rd = {q: {d: float(len(docs) - i) for i, d in enumerate(docs)} for q, docs in run.items()}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{k}"})
    res = ev.evaluate(rd)
    vals = [v[f"ndcg_cut_{k}"] for v in res.values()]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    args = ap.parse_args()

    ok = True
    print(f"{'domain':<12}{'docs':>9}{'train_q':>9}{'dev_q':>7}{'gold/q':>8}")
    for dom in args.domains:
        docs = load_documents(args.root, dom)
        ids = list(docs)
        for split in ("train", "dev"):
            ex = load_examples(args.root, dom, split)
            qrels = load_qrels(args.root, dom, split)
            # sanity: every gold id exists in the corpus; qrels agree with examples
            missing = sum(1 for q in qrels.values() for d in q if d not in docs)
            ex_ids = {e["id"] for e in ex}
            if missing or ex_ids != set(qrels):
                print(f"  WARNING {dom}/{split}: {missing} qrel docs not in corpus; "
                      f"example/qrel query sets equal: {ex_ids == set(qrels)}")
            if split == "train":
                n_train = len(ex)
                gold = sum(len(q) for q in qrels.values()) / max(len(qrels), 1)
            else:
                n_dev = len(ex)
            for name, run in make_runs(ids, qrels).items():
                ours, _ = evaluate_run(run, qrels, 10)
                ref = pytrec_mean(run, qrels, 10)
                if abs(ours - ref) > 1e-6:
                    ok = False
                    print(f"  MISMATCH {dom}/{split}/{name}: ours={ours:.8f} pytrec={ref:.8f}")
        print(f"{dom:<12}{len(docs):>9}{n_train:>9}{n_dev:>7}{gold:>8.2f}")
    print("\nnDCG@10 verification vs pytrec_eval:", "PASSED (<=1e-6)" if ok else "FAILED")
    print("Next: also compare against the starter kit scorer.py on one BM25 run (Step 2).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

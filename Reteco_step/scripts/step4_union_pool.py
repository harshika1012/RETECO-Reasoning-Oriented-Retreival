"""Step 4: union candidate pool + the Recall gate.

Usage:
  python3 scripts/step4_union_pool.py --bm25-dir runs/bm25_official --dense-dir runs/dense_<tag>
Reads top-200 jsonl from both runs, builds BM25 U dense pools, reports Recall (BM25 / dense / union), pool size,
extraction cost proxy (distinct docs per domain), and RRF nDCG@10. Writes runs/pool/<domain>/pool_<split>.jsonl.
Optional: per-reasoning-class recall if guidance_<split>.jsonl exposes the class field (evaluation only).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate_run, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, domain_dir, load_qrels
from run_io import load_topk

CLASS_KEYS = ("temporal_reasoning_class_primary", "temporal_reasoning_class", "reasoning_class")


def rrf(rank_lists, k=60):
    sc = defaultdict(float)
    for lst in rank_lists:
        for r, d in enumerate(lst, 1):
            sc[d] += 1.0 / (k + r)
    return [d for d, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]


def load_classes(root, dom, split):
    p = domain_dir(root, dom) / f"guidance_{split}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for key in CLASS_KEYS:
            v = r.get(key) or (r.get("metadata") or {}).get(key)
            if v:
                out[r.get("id") or r.get("query_id")] = v
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bm25-dir", default="runs/bm25_official")
    ap.add_argument("--dense-dir", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--out", default="runs/pool")
    args = ap.parse_args()

    stats = {s: defaultdict(dict) for s in args.splits}
    cls_hits = {s: defaultdict(lambda: defaultdict(list)) for s in args.splits}
    print(f"{'domain':<10}{'split':<6}{'R_bm25':>8}{'R_dense':>8}{'R_union':>8}{'pool/q':>8}{'distinct':>9}{'nDCG_rrf':>9}")
    for dom in args.domains:
        for split in args.splits:
            b = load_topk(Path(args.bm25_dir) / dom / f"top{args.k}_{split}.jsonl")
            d = load_topk(Path(args.dense_dir) / dom / f"top{args.k}_{split}.jsonl")
            qrels = load_qrels(args.root, dom, split)
            classes = load_classes(args.root, dom, split)
            pool, rrf_run, distinct, sizes = {}, {}, set(), []
            r_b, r_d, r_u = [], [], []
            for qid, qrel in qrels.items():
                bl = [x for x, _ in b.get(qid, [])]; dl = [x for x, _ in d.get(qid, [])]
                bm = {x: (i + 1, s) for i, (x, s) in enumerate(b.get(qid, []))}
                dn = {x: (i + 1, s) for i, (x, s) in enumerate(d.get(qid, []))}
                union = list(dict.fromkeys(bl + dl))
                pool[qid] = [{"doc": x, "bm25_rank": bm.get(x, (None, None))[0], "bm25_score": bm.get(x, (None, None))[1],
                              "dense_rank": dn.get(x, (None, None))[0], "dense_score": dn.get(x, (None, None))[1]} for x in union]
                rrf_run[qid] = rrf([bl, dl])
                distinct.update(union); sizes.append(len(union))
                for lst, acc in ((bl, r_b), (dl, r_d), (union, r_u)):
                    v = recall_at_k(lst, qrel, 10 ** 6)
                    if v is not None:
                        acc.append(v)
                if qid in classes:
                    cls_hits[split][classes[qid]]["bm25"].append(recall_at_k(bl, qrel, 10 ** 6))
                    cls_hits[split][classes[qid]]["dense"].append(recall_at_k(dl, qrel, 10 ** 6))
                    cls_hits[split][classes[qid]]["union"].append(recall_at_k(union, qrel, 10 ** 6))
            out = Path(args.out) / dom / f"pool_{split}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                for q, c in pool.items():
                    f.write(json.dumps({"qid": q, "candidates": c}) + "\n")
            nd, _ = evaluate_run(rrf_run, qrels, 10)
            row = dict(R_bm25=np.mean(r_b), R_dense=np.mean(r_d), R_union=np.mean(r_u),
                       pool=float(np.mean(sizes)), distinct=len(distinct), ndcg_rrf=nd)
            stats[split][dom] = row
            print(f"{dom:<10}{split:<6}{row['R_bm25']:>8.3f}{row['R_dense']:>8.3f}{row['R_union']:>8.3f}"
                  f"{row['pool']:>8.0f}{row['distinct']:>9}{nd:>9.4f}")
    for s in args.splits:
        m = {k: macro_average({d: r[k] for d, r in stats[s].items()}) for k in ("R_bm25", "R_dense", "R_union", "ndcg_rrf")}
        tot = sum(r["distinct"] for r in stats[s].values())
        print(f"\nMACRO {s}: recall bm25 {m['R_bm25']:.3f} | dense {m['R_dense']:.3f} | UNION {m['R_union']:.3f} | "
              f"RRF nDCG@10 {m['ndcg_rrf']:.4f} | distinct docs to extract (sum over domains): {tot}")
        if cls_hits[s]:
            print(f"  by reasoning class ({s}): union recall (n queries)")
            for c, v in sorted(cls_hits[s].items()):
                print(f"    {c:<45} {np.mean(v['union']):.3f} (bm25 {np.mean(v['bm25']):.3f}, dense {np.mean(v['dense']):.3f}) n={len(v['union'])}")
    if not any(cls_hits[s] for s in args.splits):
        print("\n(note: no reasoning-class field found in guidance files; class breakdown skipped)")


if __name__ == "__main__":
    main()

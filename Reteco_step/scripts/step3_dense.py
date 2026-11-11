"""Step 3: dense retrieval for every domain/split (resumable, cached), top-200 saved + evaluated.

Usage:
  python3 scripts/step3_dense.py --root ../reteco_data --model BAAI/bge-base-en-v1.5 --domains iota law   # smoke test
  python3 scripts/step3_dense.py --root ../reteco_data --model BAAI/bge-large-en-v1.5                      # all 13 domains
Caches:  cache/dense/<model_tag>/<domain>/docs_XXXX.npy (fp16 chunks), queries_<split>.npy
Outputs: runs/dense_<model_tag>/<domain>/top200_<split>.jsonl, run_1a_<split>.trec, summary.json
Doc embedding is the expensive part: the script prints throughput and an ETA. Interrupted runs resume by chunk.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense
from metrics import evaluate_run, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import write_trec, save_topk, ids_only

RECALL_KS = [100, 200, 500, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-docs", type=int, default=5000)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--doc-prefix", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--progress", action="store_true", help="show a progress bar while encoding")
    ap.add_argument("--cache", default="cache/dense")
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    tag = args.model.replace("/", "__") + f"_L{args.max_len}"
    out_root = Path(args.out_root) / f"dense_{tag}"
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for s in args.splits:
        summary.setdefault(s, {"ndcg10": {}, "recall": {}})

    enc = dense.Encoder(args.model, args.max_len, args.batch_size, fp16=not args.no_fp16,
                        query_prefix=args.query_prefix, doc_prefix=args.doc_prefix)
    print(f"model={args.model} device={enc.device} fp16={enc.fp16} max_len={args.max_len} prefixes q={enc.qp!r} d={enc.dp!r}", flush=True)

    for dom in args.domains:
        cdir = Path(args.cache) / tag / dom
        cdir.mkdir(parents=True, exist_ok=True)
        docs = load_documents(args.root, dom)
        ids, texts = list(docs), list(docs.values())
        del docs
        n = len(ids)
        print(f"{dom}: encoding {n} docs in chunks of {args.chunk_docs} (cached chunks are skipped)", flush=True)
        t0 = time.time(); done_docs = 0
        for ci, start in enumerate(range(0, n, args.chunk_docs)):
            f = cdir / f"docs_{ci:04d}.npy"
            if f.exists():
                continue
            e = enc.encode(texts[start:start + args.chunk_docs], is_query=False, progress=args.progress)
            np.save(f, e)
            done_docs += e.shape[0]
            rate = done_docs / (time.time() - t0)
            left = n - start - e.shape[0]
            print(f"  {dom}: {start + e.shape[0]}/{n} docs | {rate:.1f} docs/s | ETA {left / rate / 60:.1f} min", flush=True)
        del texts
        n_chunks = (n + args.chunk_docs - 1) // args.chunk_docs
        D = np.concatenate([np.load(cdir / f"docs_{ci:04d}.npy") for ci in range(n_chunks)])
        assert D.shape[0] == n, (D.shape, n)

        parts = []
        for split in args.splits:
            examples = load_examples(args.root, dom, split)
            qf = cdir / f"queries_{split}.npy"
            if qf.exists():
                Q = np.load(qf)
            else:
                Q = enc.encode([e["query"] for e in examples], is_query=True)
                np.save(qf, Q)
            idx, sc = dense.search_topk(Q, D, k=max(RECALL_KS))
            full = {e["id"]: [(ids[j], float(s)) for j, s in zip(idx[r], sc[r])] for r, e in enumerate(examples)}
            run = {q: v[:args.k] for q, v in full.items()}
            save_topk(out_root / dom / f"top{args.k}_{split}.jsonl", run)
            write_trec(out_root / dom / f"run_1a_{split}.trec", run, "dense")
            qrels = load_qrels(args.root, dom, split)
            nd, _ = evaluate_run(ids_only(full), qrels, 10)
            rec = {}
            for kk in RECALL_KS:
                vals = [recall_at_k([d for d, _ in full.get(q, [])], qr, kk) for q, qr in qrels.items()]
                rec[str(kk)] = float(np.mean([v for v in vals if v is not None]))
            summary[split]["ndcg10"][dom] = nd
            summary[split]["recall"][dom] = rec
            parts.append(f"{split}: nDCG@10 {nd:.4f} R@200 {rec['200']:.3f}")
        print(f"{dom:<10} {n:>7} docs | " + " | ".join(parts), flush=True)
        out_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        del D

    print("\nMacro average over the domains in summary.json")
    for s in args.splits:
        nd = macro_average(summary[s]["ndcg10"])
        rc = {k: macro_average({d: r[k] for d, r in summary[s]["recall"].items()}) for k in map(str, RECALL_KS)}
        print(f"  {s}: nDCG@10 {nd:.4f} (n_domains={len(summary[s]['ndcg10'])}) | recall " +
              ", ".join(f"@{k} {v:.3f}" for k, v in rc.items()))
    print(f"\nrun dir for Step 4: {out_root}")


if __name__ == "__main__":
    main()

"""Step 7: cross-encoder rerank of the best first-stage run (Step 6: RRF + dense cosine).

The cross-encoder scores (query, doc) jointly for the top `--depth` docs of each query (cached, resumable).
Then (top_n, lambda) are tuned on TRAIN macro nDCG@10, reported on DEV:
  for the first top_n docs: final = minmax(CE) + lambda * minmax(first stage); docs below top_n keep first-stage order.

Usage: python3 scripts/step7_cross_encoder.py --root ../reteco_data [--model BAAI/bge-reranker-base] [--domains iota law]
Cache:   cache/ce/<model_tag>_L<max_len>/<domain>/<split>.jsonl   ({qid, docs, scores} per query, appended as it goes)
Outputs: runs/ce_<model_tag>_L<max_len>/<domain>/{top200_<split>.jsonl, run_1a_<split>.trec}, summary.json
"""
import argparse, html, itertools, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import ndcg_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import load_topk, save_topk, write_trec

TOP_N = [20, 50, 100]
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
TAG_RE = re.compile(r"<[^>]+>")


def clean(text):
    """Queries are StackExchange HTML; strip tags so markup doesn't eat the 512-token budget."""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", text))).strip()


def load_model(name, max_len, batch_size):
    import torch
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(name, device=dev, max_length=max_len)
    # pre-Volta GPUs (e.g. GTX 1080 Ti, CC 6.1) run fp16 far slower than fp32
    fp16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    if fp16:
        model.model.half()
    print(f"model={name} device={dev} fp16={fp16} max_len={max_len} batch={batch_size}", flush=True)
    return model


def score_split(model, args, dom, split, first, cache_file, docs_text, queries):
    """Cross-encoder scores for the top `depth` docs of every query; resumes from cache_file."""
    done = {}
    if cache_file.exists():
        for line in open(cache_file):
            r = json.loads(line)
            done[r["qid"]] = dict(zip(r["docs"], r["scores"]))
    todo = [q for q in first if q not in done and q in queries]
    t0 = time.time(); n_pairs = 0
    with open(cache_file, "a") as f:
        for b in range(0, len(todo), args.query_block):
            block = todo[b:b + args.query_block]
            pairs, owner = [], []
            for q in block:
                for d, _ in first[q][:args.depth]:
                    pairs.append((queries[q], docs_text[d])); owner.append((q, d))
            # raw logits: sigmoid saturates near 0/1 and squashes the min-max interpolation
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False,
                                   activation_fn=torch.nn.Identity())
            per_q = {q: ([], []) for q in block}
            for (q, d), s in zip(owner, scores):
                per_q[q][0].append(d); per_q[q][1].append(float(s))
            for q, (ds, ss) in per_q.items():
                f.write(json.dumps({"qid": q, "docs": ds, "scores": ss}) + "\n")
                done[q] = dict(zip(ds, ss))
            f.flush()
            n_pairs += len(pairs)
            rate = n_pairs / (time.time() - t0)
            left = sum(min(len(first[q]), args.depth) for q in todo[b + len(block):])
            print(f"    {dom}/{split}: {b + len(block)}/{len(todo)} queries | {rate:.0f} pairs/s | "
                  f"ETA {left / rate / 60:.1f} min", flush=True)
    return done


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rerank(first_list, ce, top_n, lam):
    """first_list: [(doc, first_stage_score)] in first-stage order; ce: {doc: score}."""
    head = [(d, s) for d, s in first_list[:top_n] if d in ce]
    tail = [x for x in first_list if x[0] not in {d for d, _ in head}]
    if not head:
        return first_list
    fs = minmax(np.array([s for _, s in head], dtype=np.float64))
    cs = minmax(np.array([ce[d] for d, _ in head], dtype=np.float64))
    final = cs + lam * fs
    order = np.lexsort((np.arange(len(head)), -final))
    top_score = float(final.max()) + 1.0          # keep reranked head above the untouched tail
    head_ranked = [(head[i][0], float(final[i]) + top_score) for i in order]
    return head_ranked + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--first-stage", default="runs/graph_BAAI__bge-base-en-v1.5_L512")
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--depth", type=int, default=max(TOP_N), help="docs per query scored by the cross-encoder")
    ap.add_argument("--query-block", type=int, default=20, help="queries per predict() call / cache flush")
    ap.add_argument("--k", type=int, default=200, help="depth of the saved runs")
    ap.add_argument("--cache", default="cache/ce")
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    tag = args.model.replace("/", "__") + f"_L{args.max_len}"
    top_ns = [n for n in TOP_N if n <= args.depth]

    model = load_model(args.model, args.max_len, args.batch_size)
    data = {s: {} for s in ("train", "dev")}            # split -> dom -> [(qid, first_list, ce, qrel)]
    for dom in args.domains:
        t0 = time.time()
        docs_text = load_documents(args.root, dom)
        for split in ("train", "dev"):
            first = load_topk(Path(args.first_stage) / dom / f"top200_{split}.jsonl")
            queries = {e["id"]: clean(e["query"]) for e in load_examples(args.root, dom, split)}
            cfile = Path(args.cache) / tag / dom / f"{split}.jsonl"
            cfile.parent.mkdir(parents=True, exist_ok=True)
            ce = score_split(model, args, dom, split, first, cfile, docs_text, queries)
            qrels = load_qrels(args.root, dom, split)
            data[split][dom] = [(q, first[q], ce[q], qrel) for q, qrel in qrels.items()
                                if q in first and q in ce and any(v > 0 for v in qrel.values())]
        del docs_text
        print(f"  {dom}: done ({time.time() - t0:.0f}s)", flush=True)

    def evaluate(split, top_n, lam):
        per_dom = {}
        for dom, rows in data[split].items():
            vals = [ndcg_at_k([d for d, _ in rerank(fl, ce, top_n, lam)], qrel, 10) or 0.0 for _, fl, ce, qrel in rows]
            per_dom[dom] = float(np.mean(vals)) if vals else 0.0
        return macro_average(per_dom), per_dom

    def first_stage(split):
        per_dom = {dom: float(np.mean([ndcg_at_k([d for d, _ in fl], qrel, 10) or 0.0 for _, fl, _, qrel in rows]))
                   for dom, rows in data[split].items()}
        return macro_average(per_dom), per_dom

    fs_tr, _ = first_stage("train"); fs_dv, fs_dom = first_stage("dev")
    grid = list(itertools.product(top_ns, LAMBDAS))
    res = {(n, l): evaluate("train", n, l)[0] for n, l in grid}
    best = max(grid, key=lambda c: res[c])
    ce_only = max(top_ns, key=lambda n: res[(n, 0.0)])
    dv_best, best_dom = evaluate("dev", *best)

    print(f"\n{'':<40}{'train':>8}{'dev':>8}")
    print(f"{'FIRST STAGE (Step 6: RRF + dense cos)':<40}{fs_tr:>8.4f}{fs_dv:>8.4f}")
    print(f"{f'CE only, top_n={ce_only}':<40}{res[(ce_only, 0.0)]:>8.4f}{evaluate('dev', ce_only, 0.0)[0]:>8.4f}")
    print(f"{f'BEST (tuned on train) top_n={best[0]} lambda={best[1]}':<40}{res[best]:>8.4f}{dv_best:>8.4f}")
    print("\nTrain nDCG@10 by (top_n, lambda):")
    print("  top_n " + "".join(f"{l:>8}" for l in LAMBDAS))
    for n in top_ns:
        print(f"  {n:<6}" + "".join(f"{res[(n, l)]:>8.4f}" for l in LAMBDAS))
    print("\nDev per-domain nDCG@10 (first stage -> best)")
    for d in data["dev"]:
        print(f"  {d:<10} {fs_dom[d]:.4f} -> {best_dom[d]:.4f}")

    out_root = Path(args.out_root) / f"ce_{tag}"
    for split in ("train", "dev"):
        for dom, rows in data[split].items():
            run = {q: rerank(fl, ce, *best)[:args.k] for q, fl, ce, _ in rows}
            save_topk(out_root / dom / f"top{args.k}_{split}.jsonl", run)
            write_trec(out_root / dom / f"run_1a_{split}.trec", run, "ce")
    summary = {"model": args.model, "first_stage": args.first_stage, "best": {"top_n": best[0], "lambda": best[1]},
               "train_ndcg": res[best], "dev_ndcg": dv_best, "dev_ndcg_first_stage": fs_dv,
               "dev_per_domain": best_dom, "train_grid": {f"{n}_{l}": v for (n, l), v in res.items()}}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nruns + summary: {out_root}")


if __name__ == "__main__":
    main()

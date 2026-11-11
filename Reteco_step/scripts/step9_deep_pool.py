"""Step 9: deeper candidate pool + bge-reranker-base, with every choice made on the held-out TUNE split.

Pool(depth) = BM25 top-200 (unchanged) U dense top-`depth` (300/500/1000; 200 = the Step 4 pool).
First stage (as Step 6): minmax-free RRF/max + w * minmax(dense cosine), cosine computed for every pool doc.
Rerank (as Step 7): bge-reranker-base on the first-stage top_n, final = minmax(CE) + lambda * minmax(first stage).
(depth, w, top_n, lambda) are chosen on TUNE (242 held-out train queries); DEV is scored only for the chosen config
and for BASELINE (= the Step 7 bge-reranker-base setting on the old pool), so the two can be compared with Step 8.

Usage: python3 scripts/step9_deep_pool.py --root ../reteco_data [--domains iota law] [--max-top-n 100]
Cache:   cache/ce/<ce_tag>/<domain>/<split>_extra.jsonl   (new CE scores; Step 7's <split>.jsonl is reused, not changed)
Outputs: runs/step9_best/<domain>/top200_{train,dev}.jsonl   (train file = tune queries only)
         runs/step9_baseline/<domain>/top200_{train,dev}.jsonl
         runs/pool_d<depth>/<domain>/pool_{train,dev}.jsonl   (chosen depth, ALL queries: negatives for fine-tuning)
         runs/step9_best/summary.json
"""
import argparse, itertools, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import dense
from metrics import ndcg_at_k, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import load_topk, save_topk, write_trec
from splits import subset_qids
from step7_cross_encoder import load_model, clean, rerank, LAMBDAS

DEPTHS = [200, 300, 500, 1000]
W_DENSE = [0.5, 1.0, 2.0]
TOP_N = [50, 100, 200]
BASELINE = (200, 1.0, 100, 1.0)          # Step 7, bge-reranker-base: old pool, w=1.0, top_n=100, lambda=1.0
NAMES = ("depth", "w_dense", "top_n", "lambda")


# ---------- first stage ----------

def pool_arrays(args, dom):
    """Per query (all queries with a positive qrel): candidate rows at max depth + dense/BM25 ranks + cosine."""
    tag = args.model_tag
    fids = Path(args.graph_cache) / tag / dom / "ids.json"
    ids = json.loads(fids.read_text()) if fids.exists() else list(load_documents(args.root, dom))
    D = np.concatenate([np.load(f) for f in sorted((Path(args.dense_cache) / tag / dom).glob("docs_*.npy"))])
    assert len(D) == len(ids), (len(D), len(ids))
    D = D.astype(np.float32)
    id2row = {d: i for i, d in enumerate(ids)}
    out = {}
    for split in ("train", "dev"):
        examples = load_examples(args.root, dom, split)
        qrow = {e["id"]: i for i, e in enumerate(examples)}
        qs = [q for q in sorted(subset_qids(args.root, dom, split)) if q in qrow]
        Q = np.load(Path(args.dense_cache) / tag / dom / f"queries_{split}.npy").astype(np.float32)[[qrow[q] for q in qs]]
        idx, _ = dense.search_topk(Q, D, k=max(DEPTHS))
        bm = load_topk(Path(args.bm25_dir) / dom / f"top200_{split}.jsonl")
        per = {}
        for r, q in enumerate(qs):
            drows = idx[r].tolist()
            brank = {id2row[d]: i + 1 for i, (d, _) in enumerate(bm.get(q, []))}
            rows = np.array(list(dict.fromkeys(drows + list(brank))), dtype=np.int64)
            dr = np.full(len(rows), np.inf, dtype=np.float32); dr[:len(drows)] = np.arange(1, len(drows) + 1)
            br = np.array([brank.get(x, np.inf) for x in rows.tolist()], dtype=np.float32)
            per[q] = (rows.astype(np.int32), dr, br, (D[rows] @ Q[r]).astype(np.float32))
        out[split] = per
    return ids, out


def first_stage(entry, depth, w, head):
    """Returns (rows of the top `head` docs, their scores, pool rows) for one pool depth and dense weight."""
    rows, dr, br, cos = entry
    m = (dr <= depth) | np.isfinite(br)
    rows, dr, br, cos = rows[m], np.where(dr[m] <= depth, dr[m], np.inf), br[m], cos[m]
    rrf = 1.0 / (60 + dr) + 1.0 / (60 + br)
    s = rrf / rrf.max() + w * (cos - cos.min()) / max(float(cos.max() - cos.min()), 1e-9)
    s = s / s.max()
    order = np.lexsort((rows, -s))[:head]
    return rows[order], s[order], rows


# ---------- cross-encoder cache ----------

def load_ce(cdir, split):
    ce = defaultdict(dict)
    for f in (cdir / f"{split}.jsonl", cdir / f"{split}_extra.jsonl"):
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                ce[r["qid"]].update(zip(r["docs"], r["scores"]))
    return ce


def score_missing(args, state, dom, split, need, ce):
    """need: {qid: set(doc ids)}. Scores the pairs not yet in `ce`, appends them to <split>_extra.jsonl."""
    todo = {q: sorted(ds - set(ce[q])) for q, ds in need.items()}
    todo = {q: ds for q, ds in todo.items() if ds}
    n_pairs = sum(map(len, todo.values()))
    if not n_pairs:
        return
    if state.get("model") is None:
        state["model"] = load_model(args.model, args.max_len, args.batch_size)
    model = state["model"]
    docs = load_documents(args.root, dom)
    queries = {e["id"]: clean(e["query"]) for e in load_examples(args.root, dom, split)}
    cfile = Path(args.cache) / state["ce_tag"] / dom / f"{split}_extra.jsonl"
    cfile.parent.mkdir(parents=True, exist_ok=True)
    qs, t0, done = list(todo), time.time(), 0
    print(f"    {dom}/{split}: scoring {n_pairs} new pairs for {len(qs)} queries", flush=True)
    with open(cfile, "a") as f:
        for b in range(0, len(qs), args.query_block):
            block = qs[b:b + args.query_block]
            pairs = [(queries[q], docs[d]) for q in block for d in todo[q]]
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False,
                                   activation_fn=torch.nn.Identity())
            k = 0
            for q in block:
                s = [float(x) for x in scores[k:k + len(todo[q])]]; k += len(todo[q])
                f.write(json.dumps({"qid": q, "docs": todo[q], "scores": s}) + "\n")
                ce[q].update(zip(todo[q], s))
            f.flush()
            done += len(pairs)
            rate = done / (time.time() - t0)
            print(f"    {dom}/{split}: {done}/{n_pairs} pairs | {rate:.0f} pairs/s | ETA {(n_pairs - done) / rate / 60:.1f} min",
                  flush=True)
    del docs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--query-block", type=int, default=20)
    ap.add_argument("--max-top-n", type=int, default=max(TOP_N), help="cap rerank depth to save GPU time")
    ap.add_argument("--model-tag", default="BAAI__bge-base-en-v1.5_L512")
    ap.add_argument("--dense-cache", default="cache/dense")
    ap.add_argument("--graph-cache", default="cache/graph")
    ap.add_argument("--bm25-dir", default="runs/bm25_official")
    ap.add_argument("--cache", default="cache/ce")
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    top_ns = [n for n in TOP_N if n <= args.max_top_n]
    head = max(top_ns)
    state = {"ce_tag": args.model.replace("/", "__") + f"_L{args.max_len}"}

    # 1. pools for every depth (CPU)
    t0 = time.time()
    data, ids, qrels, tune = {}, {}, {}, {}
    for dom in args.domains:
        ids[dom], data[dom] = pool_arrays(args, dom)
        qrels[dom] = {s: load_qrels(args.root, dom, s) for s in ("train", "dev")}
        tune[dom] = sorted(subset_qids(args.root, dom, "tune") & set(data[dom]["train"]))
        print(f"  {dom}: pools ready ({time.time() - t0:.0f}s)", flush=True)

    def fs(dom, split, q, depth, w):
        rows, sc, pool = first_stage(data[dom][split][q], depth, w, head)
        return [(ids[dom][r], float(s)) for r, s in zip(rows, sc)], pool

    print(f"\n[tune] pool recall and first-stage nDCG@10 by depth ({sum(map(len, tune.values()))} queries)")
    print(f"  {'depth':>6}{'recall':>9}{'pool/q':>8}" + "".join(f"{f'nDCG w={w}':>12}" for w in W_DENSE))
    heads = {}
    for depth in DEPTHS:
        rec, size, nd = defaultdict(list), defaultdict(list), {w: defaultdict(list) for w in W_DENSE}
        for dom in args.domains:
            for q in tune[dom]:
                for w in W_DENSE:
                    fl, pool = fs(dom, "train", q, depth, w)
                    heads[(dom, q, depth, w)] = fl
                    nd[w][dom].append(ndcg_at_k([d for d, _ in fl], qrels[dom]["train"][q], 10) or 0.0)
                rec[dom].append(recall_at_k([ids[dom][r] for r in pool], qrels[dom]["train"][q], 10 ** 6))
                size[dom].append(len(pool))
        m = lambda x: macro_average({d: float(np.mean(v)) for d, v in x.items()})
        print(f"  {depth:>6}{m(rec):>9.3f}{m(size):>8.0f}" + "".join(f"{m(nd[w]):>12.4f}" for w in W_DENSE), flush=True)

    # 2. cross-encoder scores for every tune head (GPU), then the grid on tune
    print("\n[tune] cross-encoder scoring", flush=True)
    ce = {}
    for dom in args.domains:
        ce[dom] = {"train": load_ce(Path(args.cache) / state["ce_tag"] / dom, "train"),
                   "dev": load_ce(Path(args.cache) / state["ce_tag"] / dom, "dev")}
        need = defaultdict(set)
        for (d_, q, depth, w), fl in heads.items():
            if d_ == dom:
                need[q].update(d for d, _ in fl)
        score_missing(args, state, dom, "train", need, ce[dom]["train"])

    grid = list(itertools.product(DEPTHS, W_DENSE, top_ns, LAMBDAS))
    def tune_score(cfg):
        depth, w, n, lam = cfg
        per = {dom: float(np.mean([ndcg_at_k([d for d, _ in rerank(heads[(dom, q, depth, w)], ce[dom]["train"][q], n, lam)],
                                             qrels[dom]["train"][q], 10) or 0.0 for q in tune[dom]])) for dom in args.domains}
        return macro_average(per)
    res = {c: tune_score(c) for c in grid}
    best = max(grid, key=lambda c: (res[c], c == BASELINE))
    base = BASELINE if BASELINE in res else max((c for c in grid if c[0] == 200), key=res.get)
    print(f"\n[tune] reranked nDCG@10, best setting per pool depth")
    for depth in DEPTHS:
        c = max((c for c in grid if c[0] == depth), key=res.get)
        print(f"  depth {depth:>4}: {res[c]:.4f}  ({', '.join(f'{n}={v}' for n, v in zip(NAMES, c))})")
    print(f"  BASELINE {', '.join(f'{n}={v}' for n, v in zip(NAMES, base))}: {res[base]:.4f}")
    print(f"  BEST     {', '.join(f'{n}={v}' for n, v in zip(NAMES, best))}: {res[best]:.4f}", flush=True)

    # 3. dev, scored once, for BASELINE and BEST only
    print("\n[dev] cross-encoder scoring for BASELINE and BEST", flush=True)
    dev_runs = {"baseline": {}, "best": {}}
    dev_rec = {"baseline": {}, "best": {}}
    for dom in args.domains:
        need, lists = defaultdict(set), {}
        for name, (depth, w, n, lam) in (("baseline", base), ("best", best)):
            rec = []
            for q in data[dom]["dev"]:
                fl, pool = fs(dom, "dev", q, depth, w)
                lists[(name, q)] = fl
                need[q].update(d for d, _ in fl[:n])
                rec.append(recall_at_k([ids[dom][r] for r in pool], qrels[dom]["dev"][q], 10 ** 6))
            dev_rec[name][dom] = float(np.mean(rec))
        score_missing(args, state, dom, "dev", need, ce[dom]["dev"])
        for name, (depth, w, n, lam) in (("baseline", base), ("best", best)):
            dev_runs[name][dom] = {q: rerank(lists[(name, q)], ce[dom]["dev"][q], n, lam)[:200] for q in data[dom]["dev"]}

    def dev_ndcg(name):
        per = {dom: float(np.mean([ndcg_at_k([d for d, _ in run[q]], qrels[dom]["dev"][q], 10) or 0.0 for q in run]))
               for dom, run in dev_runs[name].items()}
        return macro_average(per), per
    (b_dv, b_dom), (x_dv, x_dom) = dev_ndcg("baseline"), dev_ndcg("best")
    print(f"\n{'':<10}{'config':<44}{'tune':>8}{'dev':>8}{'dev recall':>11}")
    for name, cfg, dv in (("BASELINE", base, b_dv), ("BEST", best, x_dv)):
        desc = ", ".join(f"{n}={v}" for n, v in zip(NAMES, cfg))
        print(f"{name:<10}{desc:<44}{res[cfg]:>8.4f}{dv:>8.4f}{macro_average(dev_rec[name.lower()]):>11.3f}")
    print("\nDev per-domain nDCG@10 (baseline -> best)")
    for dom in args.domains:
        print(f"  {dom:<10} {b_dom[dom]:.4f} -> {x_dom[dom]:.4f}")

    # 4. outputs: runs (tune + dev) for both configs, pools at the chosen depth for ALL queries
    out = Path(args.out_root)
    for name, cfg in (("baseline", base), ("best", best)):
        depth, w, n, lam = cfg
        for dom in args.domains:
            tr = {q: rerank(heads[(dom, q, depth, w)], ce[dom]["train"][q], n, lam)[:200] for q in tune[dom]}
            for split, run in (("train", tr), ("dev", dev_runs[name][dom])):
                save_topk(out / f"step9_{name}" / dom / f"top200_{split}.jsonl", run)
                write_trec(out / f"step9_{name}" / dom / f"run_1a_{split}.trec", run, f"step9_{name}")
    depth = best[0]
    for dom in args.domains:
        for split in ("train", "dev"):
            p = out / f"pool_d{depth}" / dom / f"pool_{split}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                for q, (rows, dr, br, cos) in data[dom][split].items():
                    m = (dr <= depth) | np.isfinite(br)
                    cands = [{"doc": ids[dom][r], "bm25_rank": None if np.isinf(b) else int(b),
                              "dense_rank": None if np.isinf(d) or d > depth else int(d), "cos": round(float(c), 5)}
                             for r, d, b, c in zip(rows[m], dr[m], br[m], cos[m])]
                    f.write(json.dumps({"qid": q, "candidates": cands}) + "\n")
    summary = {"best": dict(zip(NAMES, best)), "baseline": dict(zip(NAMES, base)), "tune_best": res[best],
               "tune_baseline": res[base], "dev_best": x_dv, "dev_baseline": b_dv, "dev_per_domain_best": x_dom,
               "tune_grid": [{**dict(zip(NAMES, c)), "tune": v} for c, v in res.items()]}
    (out / "step9_best" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nruns: {out}/step9_best, {out}/step9_baseline | pools: {out}/pool_d{depth}")
    print("significance: python3 scripts/step8_significance.py --root ../reteco_data --subset dev "
          "--runs baseline=runs/step9_baseline best=runs/step9_best")


if __name__ == "__main__":
    main()

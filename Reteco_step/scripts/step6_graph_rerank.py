"""Step 6: graph-based rerank + pool expansion over a corpus kNN graph (GAR-style, content-only).

1. Corpus graph (cached): per domain, each doc -> its K nearest docs by cosine of the Step-3 dense embeddings.
2. Per query, starting from the Step-4 pool:
   - expand: add the graph neighbours of the top-`seeds` docs (by RRF) to the candidate set (fixes missed recall)
   - propagate: score smoothing over a local kNN graph of the candidates (label spreading), seeded with
     s0 = RRF(BM25, dense) + w * dense cosine (expanded docs have no RRF score; they rank via cosine + neighbours)
Edges come from document content only (no doc-ID heuristics). (seeds, nbrs, k_local, w, alpha) tuned on TRAIN
macro nDCG@10, reported on DEV.

Usage: python3 scripts/step6_graph_rerank.py --root ../reteco_data [--domains bitcoin law]
Caches:  cache/graph/<tag>/<domain>/{knn_idx.npy, knn_sim.npy}
Outputs: runs/graph_<tag>/<domain>/{top200_<split>.jsonl, run_1a_<split>.trec}, summary.json
"""
import argparse, itertools, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import ndcg_at_k, recall_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import save_topk, write_trec

SEEDS = [0, 10, 25, 50]          # 0 = no expansion
NBRS = [8, 16]                   # neighbours added per seed
K_LOCAL = [5, 10]                # degree of the per-query candidate graph
W_DENSE = [0.0, 0.5, 1.0]        # weight of dense cosine in the seed score
ALPHAS = [0.0, 0.2, 0.4, 0.6]    # 0 = no propagation
ITERS = 20


def load_doc_embeddings(cache_dir, n):
    files = sorted(Path(cache_dir).glob("docs_*.npy"))
    D = np.concatenate([np.load(f) for f in files]).astype(np.float32)
    assert D.shape[0] == n, (D.shape, n)
    return D


def build_knn(D, k, block=2048):
    """Exact cosine kNN (rows of D are L2-normalised). Returns (idx int32 (n,k), sim fp16 (n,k)), self excluded."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Dt = torch.from_numpy(D).to(dev)
    n = D.shape[0]
    idx = np.empty((n, k), dtype=np.int32); sim = np.empty((n, k), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, n, block):
            S = Dt[s:s + block] @ Dt.T
            r = torch.arange(S.shape[0], device=dev)
            S[r, r + s] = -float("inf")
            v, i = torch.topk(S, k, dim=1)
            idx[s:s + block] = i.cpu().numpy(); sim[s:s + block] = v.cpu().numpy()
    del Dt
    return idx, sim


def rrf_score(c):
    return sum(1.0 / (60 + c[r]) for r in ("bm25_rank", "dense_rank") if c[r])


def local_operator(E, k):
    """Symmetric-normalised kNN affinity over the candidates (label-spreading operator)."""
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(E) - 1)
    nb = np.argpartition(-S, k - 1, axis=1)[:, :k]
    W = np.zeros_like(S)
    rows = np.arange(len(E))[:, None]
    W[rows, nb] = np.maximum(S[rows, nb], 0.0)
    W = np.maximum(W, W.T)
    d = W.sum(1); d[d == 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    return W * dinv[:, None] * dinv[None, :]


def spread(A, s0, alpha):
    if alpha == 0:
        return s0
    f = s0.copy()
    for _ in range(ITERS):
        f = (1 - alpha) * s0 + alpha * (A @ f)
    return f


def rank(docs, score):
    order = np.lexsort((np.arange(len(docs)), -score))
    return [docs[i] for i in order], score[order]


def configs():
    for seeds in SEEDS:
        for nbrs in (NBRS if seeds else [0]):
            for kl, w, a in itertools.product(K_LOCAL, W_DENSE, ALPHAS):
                if a == 0 and kl != K_LOCAL[0]:
                    continue  # k_local is irrelevant without propagation
                yield (seeds, nbrs, kl, w, a)


def score_query(pool_cands, q, D, knn_idx, id2row, ids, only=None):
    """Yields (cfg, ranked_docs, ranked_scores, n_candidates) for every config (or just `only`)."""
    base_docs = [c["doc"] for c in pool_cands]
    base_rrf = np.array([rrf_score(c) for c in pool_cands], dtype=np.float32)
    top_rows = [id2row[d] for d in base_docs[:max(SEEDS)]]  # pool_cands is sorted by RRF desc
    for seeds in SEEDS:
        for nbrs in (NBRS if seeds else [0]):
            if only and (seeds, nbrs) != only[:2]:
                continue
            extra = []
            if seeds:
                seen = set(base_docs)
                for r in top_rows[:seeds]:
                    for j in knn_idx[r, :nbrs]:
                        d = ids[j]
                        if d not in seen:
                            seen.add(d); extra.append(d)
            docs = base_docs + extra
            rrf = np.concatenate([base_rrf, np.zeros(len(extra), np.float32)])
            rrf = rrf / rrf.max()
            E = D[[id2row[d] for d in docs]]
            cos = E @ q
            cos = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-9)
            ops = {}
            for kl, w, a in itertools.product(K_LOCAL, W_DENSE, ALPHAS):
                if (a == 0 and kl != K_LOCAL[0]) or (only and (seeds, nbrs, kl, w, a) != only):
                    continue
                s0 = rrf + w * cos
                s0 = s0 / s0.max()
                if a and kl not in ops:
                    ops[kl] = local_operator(E, kl)
                f = spread(ops.get(kl), s0, a)
                rd, rs = rank(docs, f)
                yield (seeds, nbrs, kl, w, a), rd, rs, len(docs)


def load_domain(args, dom, tag):
    """(ids, id2row, D, knn_idx); builds and caches the corpus kNN graph on first use."""
    gdir = Path(args.graph_cache) / tag / dom
    gdir.mkdir(parents=True, exist_ok=True)
    fids = gdir / "ids.json"
    if fids.exists():
        ids = json.loads(fids.read_text())
    else:
        ids = list(load_documents(args.root, dom))   # same order as the Step-3 embedding chunks
        fids.write_text(json.dumps(ids))
    D = load_doc_embeddings(Path(args.dense_cache) / tag / dom, len(ids))
    fi, fs = gdir / "knn_idx.npy", gdir / "knn_sim.npy"
    if fi.exists():
        knn_idx = np.load(fi)
    else:
        knn_idx, knn_sim = build_knn(D, args.knn_k)
        np.save(fi, knn_idx); np.save(fs, knn_sim)
    return ids, {d: i for i, d in enumerate(ids)}, D, knn_idx


def iter_queries(args, dom, tag, split):
    """Yields (qid, sorted pool candidates, query embedding, qrel) for queries with a pool and a positive qrel."""
    examples = load_examples(args.root, dom, split)
    Q = np.load(Path(args.dense_cache) / tag / dom / f"queries_{split}.npy").astype(np.float32)
    qrow = {e["id"]: i for i, e in enumerate(examples)}
    pool = {}
    for line in open(Path(args.pool_dir) / dom / f"pool_{split}.jsonl"):
        p = json.loads(line)
        pool[p["qid"]] = sorted(p["candidates"], key=lambda c: (-rrf_score(c), c["doc"]))
    for qid, qrel in load_qrels(args.root, dom, split).items():
        if qid in pool and any(v > 0 for v in qrel.values()):
            yield qid, pool[qid], Q[qrow[qid]], qrel


def run_domain(args, dom, tag):
    t0 = time.time()
    ids, id2row, D, knn_idx = load_domain(args, dom, tag)
    print(f"  {dom}: {len(ids)} docs, graph ready ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for split in ("train", "dev"):
        ndcg, recall, sizes = defaultdict(list), defaultdict(list), defaultdict(list)
        for qid, cands, q, qrel in iter_queries(args, dom, tag, split):
            for cfg, rd, rs, n in score_query(cands, q, D, knn_idx, id2row, ids):
                ndcg[cfg].append(ndcg_at_k(rd, qrel, 10) or 0.0)
                recall[cfg].append(recall_at_k(rd, qrel, 10 ** 6))
                sizes[cfg].append(n)
        out[split] = {m: {c: float(np.mean(v)) for c, v in x.items()}
                      for m, x in (("ndcg", ndcg), ("recall", recall), ("size", sizes))}
    print(f"  {dom}: scored {len(out['train']['ndcg'])} configs ({time.time() - t0:.0f}s)", flush=True)
    return out


def write_best_runs(args, dom, tag, cfg, out_root):
    ids, id2row, D, knn_idx = load_domain(args, dom, tag)
    for split in ("train", "dev"):
        run = {}
        for qid, cands, q, qrel in iter_queries(args, dom, tag, split):
            for _, rd, rs, _ in score_query(cands, q, D, knn_idx, id2row, ids, only=cfg):
                run[qid] = list(zip(rd[:args.k], map(float, rs[:args.k])))
        save_topk(out_root / dom / f"top{args.k}_{split}.jsonl", run)
        write_trec(out_root / dom / f"run_1a_{split}.trec", run, "graph")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--model-tag", default="BAAI__bge-base-en-v1.5_L512")
    ap.add_argument("--dense-cache", default="cache/dense")
    ap.add_argument("--graph-cache", default="cache/graph")
    ap.add_argument("--pool-dir", default="runs/pool")
    ap.add_argument("--knn-k", type=int, default=32)
    ap.add_argument("--k", type=int, default=200, help="depth of the saved runs")
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    tag = args.model_tag

    res = {}
    for dom in args.domains:
        res[dom] = run_domain(args, dom, tag)

    def macro(split, key, cfg):
        return macro_average({d: r[split][key][cfg] for d, r in res.items()})

    base = (0, 0, K_LOCAL[0], 0.0, 0.0)
    cfgs = list(configs())
    best = max(cfgs, key=lambda c: (macro("train", "ndcg", c), c == base))
    no_exp = max((c for c in cfgs if c[0] == 0), key=lambda c: macro("train", "ndcg", c))
    names = "seeds nbrs k_local w_dense alpha".split()

    print(f"\n{'config':<55}{'train':>8}{'dev':>8}{'dev recall':>11}{'dev pool':>9}")
    for label, c in (("BASE (RRF of Step-4 pool)", base), ("best without expansion", no_exp), ("BEST (tuned on train)", best)):
        desc = " ".join(f"{n}={v}" for n, v in zip(names, c))
        print(f"{label:<28}{desc:<27}{macro('train', 'ndcg', c):>8.4f}{macro('dev', 'ndcg', c):>8.4f}"
              f"{macro('dev', 'recall', c):>11.3f}{macro('dev', 'size', c):>9.0f}")

    families = (("expansion only", lambda c: c[0] > 0 and c[4] == 0),
                ("propagation only", lambda c: c[0] == 0 and c[4] > 0),
                ("expansion + propagation", lambda c: c[0] > 0 and c[4] > 0))
    print("\nBest config per graph family (chosen on train):")
    for label, keep in families:
        c = max((c for c in cfgs if keep(c)), key=lambda c: macro("train", "ndcg", c))
        desc = " ".join(f"{n}={v}" for n, v in zip(names, c))
        print(f"  {label:<26}{desc:<27}{macro('train', 'ndcg', c):>8.4f}{macro('dev', 'ndcg', c):>8.4f}"
              f"{macro('dev', 'recall', c):>11.3f}{macro('dev', 'size', c):>9.0f}")
    widest = (max(SEEDS), max(NBRS), K_LOCAL[0], 0.0, 0.0)
    print(f"  recall ceiling with the widest expansion (seeds={widest[0]} nbrs={widest[1]}): "
          f"train {macro('train', 'recall', base):.3f} -> {macro('train', 'recall', widest):.3f} | "
          f"dev {macro('dev', 'recall', base):.3f} -> {macro('dev', 'recall', widest):.3f} "
          f"(pool {macro('dev', 'size', base):.0f} -> {macro('dev', 'size', widest):.0f})")

    print("\nDev per-domain nDCG@10 (base -> best) | candidate recall (base -> best)")
    for d, r in res.items():
        print(f"  {d:<10} {r['dev']['ndcg'][base]:.4f} -> {r['dev']['ndcg'][best]:.4f} | "
              f"{r['dev']['recall'][base]:.3f} -> {r['dev']['recall'][best]:.3f}")

    out_root = Path(args.out_root) / f"graph_{tag}"
    for d in res:
        write_best_runs(args, d, tag, best, out_root)
    summary = {"best_config": dict(zip(names, best)), "base_config": dict(zip(names, base)),
               **{f"{s}_{m}": {d: r[s][m][best] for d, r in res.items()} for s in ("train", "dev") for m in ("ndcg", "recall")},
               **{f"{s}_ndcg_base": {d: r[s]["ndcg"][base] for d, r in res.items()} for s in ("train", "dev")},
               "all_configs": [{**dict(zip(names, c)), **{f"{s}_{m}": macro(s, m, c) for s in ("train", "dev")
                                                          for m in ("ndcg", "recall", "size")}} for c in cfgs]}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nruns + summary: {out_root}")


if __name__ == "__main__":
    main()

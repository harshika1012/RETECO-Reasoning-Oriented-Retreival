"""Step 5: cheap temporal baseline (no LLM): year regex on queries/docs, rerank the Step-4 pool.

final score = RRF(BM25, dense) normalised to [0,1] + lambda * temporal score.
Compares: base RRF | binary temporal | graded temporal. (mode, sigma, lambda, agg) are tuned on TRAIN, reported on DEV.
Usage: python3 scripts/step5_regex_temporal.py --root ../reteco_data [--topn 200]
Also prints how many queries/docs have a parsable year, and whether gold docs match the query period more often than
non-gold pool docs (the key signal check).
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import ndcg_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from tempo_simple import parse_query_intervals, parse_doc_years, temporal_score

LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
SIGMAS = [1.0, 3.0, 10.0]
AGGS = ["mean", "max"]


def load_pool(path, topn):
    out = {}
    for line in open(path):
        p = json.loads(line)
        cands = p["candidates"]
        def rrf(c): return sum(1.0 / (60 + c[r]) for r in ("bm25_rank", "dense_rank") if c[r])
        cands = sorted(cands, key=lambda c: (-rrf(c), c["doc"]))[:topn]
        out[p["qid"]] = [(c["doc"], rrf(c)) for c in cands]
    return out


def prepare(root, domains, split, topn, cache_dir):
    """Per domain: list of (qid, docs, base_scores, intervals, years_per_doc, qrel)."""
    data = {}
    for dom in domains:
        pool = load_pool(Path("runs/pool") / dom / f"pool_{split}.jsonl", topn)
        examples = {e["id"]: e["query"] for e in load_examples(root, dom, split)}
        qrels = load_qrels(root, dom, split)
        cache = Path(cache_dir) / f"{dom}.json"
        years = {k: set(v) for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
        need = {d for p in pool.values() for d, _ in p} - set(years)
        if need:
            docs = load_documents(root, dom)
            for d in need:
                years[d] = parse_doc_years(docs[d])
            del docs
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({k: sorted(v) for k, v in years.items()}))
        rows = []
        for qid, qrel in qrels.items():
            if qid not in pool or not any(v > 0 for v in qrel.values()):
                continue
            docs_, base = zip(*pool[qid])
            base = np.array(base) / max(base)
            rows.append((qid, list(docs_), base, parse_query_intervals(examples[qid]), [years[d] for d in docs_], qrel))
        data[dom] = rows
    return data


def evaluate(data, mode, sigma, lam, agg):
    per_dom = {}
    for dom, rows in data.items():
        vals = []
        for qid, docs, base, ivs, yrs, qrel in rows:
            t = np.array([temporal_score(y, ivs, mode, sigma, agg) for y in yrs]) if ivs else 0.0
            final = base + lam * t
            order = np.lexsort((np.arange(len(docs)), -final))
            v = ndcg_at_k([docs[i] for i in order], qrel, 10)
            vals.append(v if v is not None else 0.0)
        per_dom[dom] = float(np.mean(vals)) if vals else 0.0
    return macro_average(per_dom), per_dom


def signal_check(data):
    g_hit = g_n = o_hit = o_n = q_with = q_all = 0
    d_with_year = d_all = 0
    for rows in data.values():
        for qid, docs, base, ivs, yrs, qrel in rows:
            q_all += 1
            d_all += len(docs); d_with_year += sum(1 for y in yrs if y)
            if not ivs:
                continue
            q_with += 1
            for d, y in zip(docs, yrs):
                hit = temporal_score(y, ivs, "binary", agg="max") > 0
                if qrel.get(d, 0) > 0:
                    g_hit += hit; g_n += 1
                else:
                    o_hit += hit; o_n += 1
    return dict(queries=q_all, with_time=q_with, docs_with_year=d_with_year / max(d_all, 1),
                gold_match=g_hit / max(g_n, 1), other_match=o_hit / max(o_n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--cache", default="cache/doc_years")
    args = ap.parse_args()

    train = prepare(args.root, args.domains, "train", args.topn, args.cache)
    dev = prepare(args.root, args.domains, "dev", args.topn, args.cache)
    for name, d in (("train", train), ("dev", dev)):
        s = signal_check(d)
        print(f"[{name}] queries {s['queries']} | with parsable time {s['with_time']} ({s['with_time']/max(s['queries'],1):.0%}) | "
              f"pool docs with a year {s['docs_with_year']:.0%} | exact-period match: gold docs {s['gold_match']:.1%} vs other pool docs {s['other_match']:.1%}")

    base_tr, _ = evaluate(train, "binary", 1.0, 0.0, "max"); base_dv, _ = evaluate(dev, "binary", 1.0, 0.0, "max")
    print(f"\nBASE (RRF of BM25+dense, top-{args.topn} pool): train {base_tr:.4f} | dev {base_dv:.4f}")

    best = {}
    for mode in ("binary", "graded"):
        grid = itertools.product(AGGS, LAMBDAS[1:], SIGMAS if mode == "graded" else [1.0])
        res = [(evaluate(train, mode, sg, lam, agg)[0], agg, lam, sg) for agg, lam, sg in grid]
        sc, agg, lam, sg = max(res)
        dv, per_dom = evaluate(dev, mode, sg, lam, agg)
        best[mode] = (sc, dv, agg, lam, sg)
        print(f"{mode.upper():<7} best on train: agg={agg} lambda={lam} sigma={sg} -> train {sc:.4f} | dev {dv:.4f}")
    print("\nDev per-domain (graded best vs base):")
    _, base_dom = evaluate(dev, "binary", 1.0, 0.0, "max")
    _, g_dom = evaluate(dev, "graded", best["graded"][4], best["graded"][3], best["graded"][2])
    for d in args.domains:
        print(f"  {d:<10} base {base_dom[d]:.4f} -> graded {g_dom[d]:.4f}")


if __name__ == "__main__":
    main()

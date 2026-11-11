"""nDCG@k and Recall@k, matching trec_eval's ndcg_cut definition:
DCG = sum rel_i / log2(rank_i + 1) (rank starts at 1); IDCG from the qrels.
Queries with no relevant docs are skipped (as pytrec_eval does)."""
import math


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_ids, qrel, k=10):
    """ranked_ids: list of doc ids in rank order. qrel: {docid: rel}."""
    gains = [qrel.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg([g if g > 0 else 0 for g in gains]) / idcg if idcg > 0 else None


def recall_at_k(ranked_ids, qrel, k=200):
    rel_docs = {d for d, r in qrel.items() if r > 0}
    if not rel_docs:
        return None
    return len(rel_docs & set(ranked_ids[:k])) / len(rel_docs)


def evaluate_run(run, qrels, k=10):
    """run: {qid: [docid,...] ordered}. Returns (mean_ndcg, {qid: ndcg})."""
    per_q = {}
    for qid, qrel in qrels.items():
        v = ndcg_at_k(run.get(qid, []), qrel, k)
        if v is not None:
            per_q[qid] = v
    mean = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return mean, per_q


def macro_average(domain_scores):
    """domain_scores: {domain: mean_ndcg} -> macro average over domains."""
    return sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0.0

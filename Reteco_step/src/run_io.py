"""Reading/writing run files: TREC format and our own top-k JSONL format."""
import json
from pathlib import Path


def write_trec(path, run, tag):
    """run: {qid: [(docid, score), ...]} already in rank order. Format: qid Q0 docid rank score tag"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, items in run.items():
            for rank, (docid, score) in enumerate(items, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def save_topk(path, run):
    """run: {qid: [(docid, score), ...]} -> JSONL {qid, docs, scores}"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, items in run.items():
            f.write(json.dumps({"qid": qid, "docs": [d for d, _ in items],
                                "scores": [float(s) for _, s in items]}) + "\n")


def load_topk(path):
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            run[r["qid"]] = list(zip(r["docs"], r["scores"]))
    return run


def ids_only(run):
    """{qid: [(docid, score)]} -> {qid: [docid]} for the metrics functions."""
    return {q: [d for d, _ in items] for q, items in run.items()}

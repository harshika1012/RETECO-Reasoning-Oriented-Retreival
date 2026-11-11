"""Loaders for the RETECO Track 1 (TEMPO) data layout.

Expected layout (verify against your download):
  <root>/track1_tempo/<domain>/documents.jsonl        {id, content}
  <root>/track1_tempo/<domain>/examples_{train,dev}.jsonl   {id, query, gold_ids, ...}
  <root>/track1_tempo/<domain>/qrels_{train,dev}.txt   TREC: qid 0 docid rel
"""
import json
from pathlib import Path

TRACK1_DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]


def domain_dir(root, domain):
    return Path(root) / "track1_tempo" / domain


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_documents(root, domain):
    """Returns {doc_id: content}. Corpus is the FULL domain corpus (never split)."""
    return {d["id"]: d["content"] for d in _read_jsonl(domain_dir(root, domain) / "documents.jsonl")}


def load_examples(root, domain, split):
    """Returns list of dicts with at least: id, query, gold_ids."""
    assert split in ("train", "dev")
    return list(_read_jsonl(domain_dir(root, domain) / f"examples_{split}.jsonl"))


def load_qrels(root, domain, split):
    """Returns {qid: {docid: rel}} from TREC qrels (qid 0 docid rel)."""
    assert split in ("train", "dev")
    qrels = {}
    with open(domain_dir(root, domain) / f"qrels_{split}.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels

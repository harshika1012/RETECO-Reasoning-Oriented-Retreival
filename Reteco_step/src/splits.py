"""Fixed held-out split of TRAIN queries, used for all tuning from Step 8 on.

  fit  (80%) -> anything that learns from train (reranker fine-tuning, hard negatives)
  tune (20%) -> choosing settings (depths, blend weights, per-domain weights)
  dev        -> final check only, once per step

The split is drawn once (seeded, per domain) and frozen in splits/train_heldout.json, so every step uses the
same queries even if this code changes. Only queries with at least one positive qrel are split.
"""
import json, zlib
from pathlib import Path
import numpy as np

from reteco_io import TRACK1_DOMAINS, load_qrels

SPLIT_FILE = Path(__file__).resolve().parents[1] / "splits" / "train_heldout.json"
SEED, TUNE_FRAC = 2027, 0.2


def _draw(root, domains):
    out = {}
    for dom in domains:
        qids = sorted(q for q, qr in load_qrels(root, dom, "train").items() if any(v > 0 for v in qr.values()))
        rng = np.random.default_rng([SEED, zlib.crc32(dom.encode())])
        perm = rng.permutation(len(qids))
        n_tune = max(1, round(TUNE_FRAC * len(qids)))
        tune = {qids[i] for i in perm[:n_tune]}
        out[dom] = {"fit": [q for q in qids if q not in tune], "tune": sorted(tune)}
    return out


def load_split(root):
    """{domain: {"fit": [qid...], "tune": [qid...]}}; creates and freezes the file on first use."""
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    split = _draw(root, TRACK1_DOMAINS)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({"seed": SEED, "tune_frac": TUNE_FRAC, "domains": split}, indent=1))
    return json.loads(SPLIT_FILE.read_text())


def subset_qids(root, domain, subset):
    """subset: 'fit' | 'tune' (train queries) | 'train' | 'dev' (all queries with a positive qrel)."""
    if subset in ("fit", "tune"):
        return set(load_split(root)["domains"][domain][subset])
    return {q for q, qr in load_qrels(root, domain, subset).items() if any(v > 0 for v in qr.values())}

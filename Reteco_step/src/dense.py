"""Dense retrieval helpers: sentence-transformers encoder (with per-model prefixes) + exact top-k search."""
import numpy as np

# (substring of model name, query prefix, doc prefix)
PRESETS = [
    ("bge", "Represent this sentence for searching relevant passages: ", ""),
    ("e5", "query: ", "passage: "),
]


def prefixes_for(model_name, query_prefix=None, doc_prefix=None):
    qp, dp = "", ""
    for key, q, d in PRESETS:
        if key in model_name.lower():
            qp, dp = q, d
            break
    return (qp if query_prefix is None else query_prefix), (dp if doc_prefix is None else doc_prefix)


class Encoder:
    def __init__(self, model_name, max_len=512, batch_size=64, device=None, fp16=True,
                 query_prefix=None, doc_prefix=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_len
        # pre-Volta GPUs (e.g. GTX 1080 Ti, CC 6.1) run fp16 far slower than fp32
        self.fp16 = self.device == "cuda" and fp16 and torch.cuda.get_device_capability()[0] >= 7
        if self.fp16:
            self.model.half()
        self.batch_size = batch_size
        self.qp, self.dp = prefixes_for(model_name, query_prefix, doc_prefix)

    def encode(self, texts, is_query, progress=False):
        p = self.qp if is_query else self.dp
        emb = self.model.encode([p + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=progress)
        return emb.astype(np.float16)


def search_topk(Q, D, k=1000, chunk=20000):
    """Exact inner-product top-k. Q: (nq,d), D: (nd,d) (fp16/fp32). Returns (idx, scores), each (nq,k), sorted desc."""
    Q = Q.astype(np.float32)
    nq = Q.shape[0]
    best_s = np.full((nq, 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    for start in range(0, D.shape[0], chunk):
        S = Q @ D[start:start + chunk].astype(np.float32).T
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        sc = np.take_along_axis(S, idx, axis=1)
        best_s = np.concatenate([best_s, sc], axis=1)
        best_i = np.concatenate([best_i, idx + start], axis=1)
        if best_s.shape[1] > k:
            keep = np.argpartition(-best_s, k - 1, axis=1)[:, :k]
            best_s = np.take_along_axis(best_s, keep, axis=1)
            best_i = np.take_along_axis(best_i, keep, axis=1)
    order = np.lexsort((best_i, -best_s), axis=-1) if best_s.ndim == 1 else \
        np.stack([np.lexsort((best_i[r], -best_s[r])) for r in range(nq)])
    return np.take_along_axis(best_i, order, axis=1), np.take_along_axis(best_s, order, axis=1)

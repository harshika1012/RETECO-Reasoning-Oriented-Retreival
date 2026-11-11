"""Self-contained BM25 (no Java needed), Lucene-style:
  - analyzer: lowercase, possessive removal, Lucene English stopwords, Porter stemming
  - idf = log(1 + (N - df + 0.5) / (df + 0.5)),  k1=0.9, b=0.4 by default
  - repeated query terms count multiple times (like a Lucene boolean query)
Index = term-major sparse matrix of precomputed BM25 weights; search = sum of posting lists.
"""
import pickle
import re
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer

LUCENE_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


class Tokenizer:
    def __init__(self):
        self._stem = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM).stem
        self._cache = {}

    def __call__(self, text):
        out = []
        for t in _TOKEN_RE.findall(text.lower()):
            if t.endswith("'s"):
                t = t[:-2]
            if not t or t in LUCENE_STOPWORDS:
                continue
            s = self._cache.get(t)
            if s is None:
                s = self._cache[t] = self._stem(t)
            out.append(s)
        return out


class BM25Index:
    def __init__(self, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.doc_ids = None
        self.vocab = None
        self.indptr = self.indices = self.data = None  # term-major CSR of BM25 weights
        self.tok = None

    # ---------- build ----------
    def build(self, doc_ids, texts):
        self.tok = Tokenizer()
        n = len(doc_ids)
        vocab, indptr, indices, data = {}, array("q", [0]), array("i"), array("f")
        doc_len = np.zeros(n, dtype=np.float32)
        for i, text in enumerate(texts):
            toks = self.tok(text)
            doc_len[i] = len(toks)
            for t, c in Counter(toks).items():
                j = vocab.setdefault(t, len(vocab))
                indices.append(j)
                data.append(c)
            indptr.append(len(indices))
        X = sp.csr_matrix((np.frombuffer(data, dtype=np.float32),
                           np.frombuffer(indices, dtype=np.int32),
                           np.frombuffer(indptr, dtype=np.int64)),
                          shape=(n, len(vocab)))
        Xt = X.T.tocsr()                               # term x doc
        df = np.diff(Xt.indptr).astype(np.float64)
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        avgdl = float(doc_len.mean()) if n else 1.0
        tf = Xt.data.astype(np.float64)
        dl = doc_len[Xt.indices].astype(np.float64)
        w = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        w *= np.repeat(idf, np.diff(Xt.indptr))
        self.doc_ids = list(doc_ids)
        self.vocab = vocab
        self.indptr, self.indices, self.data = Xt.indptr, Xt.indices, w.astype(np.float32)
        return self

    # ---------- search ----------
    def search(self, query, k=200):
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        for t, c in Counter(self.tok(query)).items():
            j = self.vocab.get(t)
            if j is None:
                continue
            s, e = self.indptr[j], self.indptr[j + 1]
            scores[self.indices[s:e]] += c * self.data[s:e]
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]     # score desc, doc index asc (deterministic)
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    # ---------- persistence ----------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "doc_ids": self.doc_ids, "vocab": self.vocab,
                         "indptr": self.indptr, "indices": self.indices, "data": self.data}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls(d["k1"], d["b"])
        o.doc_ids, o.vocab = d["doc_ids"], d["vocab"]
        o.indptr, o.indices, o.data = d["indptr"], d["indices"], d["data"]
        o.tok = Tokenizer()
        return o

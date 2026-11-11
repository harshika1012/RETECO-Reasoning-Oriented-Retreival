"""Compare our sparse BM25 against a naive per-document implementation."""
import math, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bm25 import BM25Index, Tokenizer

random.seed(0)
vocab = [f"word{i}" for i in range(300)] + ["running", "runs", "bitcoin's", "the", "of"]
docs = {f"d{i}": " ".join(random.choices(vocab, k=random.randint(20, 80))) for i in range(400)}
idx = BM25Index().build(list(docs), list(docs.values()))

tok = Tokenizer()
toks = {d: tok(t) for d, t in docs.items()}
N = len(docs); avgdl = sum(len(t) for t in toks.values()) / N
df = Counter(w for t in toks.values() for w in set(t))

def naive(query, k1=0.9, b=0.4):
    q = Counter(tok(query)); out = {}
    for d, t in toks.items():
        tf = Counter(t); s = 0.0
        for w, c in q.items():
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += c * idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        out[d] = s
    return sorted(out.items(), key=lambda x: -x[1])

for _ in range(25):
    q = " ".join(random.choices(vocab, k=8))
    ours = idx.search(q, 10); ref = naive(q)[:10]
    assert all(abs(a[1] - b[1]) < 1e-3 * max(1, b[1]) for a, b in zip(ours, ref)), (q, ours[:3], ref[:3])
assert tok("Bitcoin's Running runs the") == ["bitcoin", "run", "run"], tok("Bitcoin's Running runs the")
idx.save("/tmp/idx.pkl"); idx2 = BM25Index.load("/tmp/idx.pkl")
assert idx2.search("word1 word2", 5) == idx.search("word1 word2", 5)
print("test_bm25: OK")

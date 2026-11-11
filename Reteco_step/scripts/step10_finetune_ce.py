"""Step 10: fine-tune bge-reranker-base on the FIT split (80% of train); pick the epoch on TUNE; check DEV once.

Training: listwise softmax loss over groups of 1 gold doc + `--negatives` hard negatives, sampled each epoch from the
non-gold docs in the first-stage top `--neg-depth` of the depth-300 pool (runs/pool_d300, written by Step 9).
Evaluation keeps the Step 7 setting unchanged (no re-tuning): rerank the first-stage top 100 of runs/graph_...,
final = minmax(CE) + 1.0 * minmax(first stage). The only choice made is the number of epochs, on TUNE.

Usage: python3 scripts/step10_finetune_ce.py --root ../reteco_data [--epochs 3] [--domains iota law --max-steps 20]
Outputs: models/step10/best/ (HF checkpoint, loadable with CrossEncoder or AutoModelForSequenceClassification)
         runs/step10_ft/<domain>/top200_{train,dev}.jsonl   (train file = tune queries only) + summary.json
"""
import argparse, json, math, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from metrics import ndcg_at_k, macro_average
from reteco_io import TRACK1_DOMAINS, load_documents, load_examples, load_qrels
from run_io import load_topk, save_topk, write_trec
from splits import subset_qids
from step7_cross_encoder import clean, rerank


def pool_ranking(cands, w=1.0):
    """Step 6 first stage on a pool file: RRF/max + w * minmax(cosine), best first."""
    rrf = np.array([sum(1.0 / (60 + c[k]) for k in ("bm25_rank", "dense_rank") if c[k]) for c in cands])
    cos = np.array([c["cos"] for c in cands])
    s = rrf / rrf.max() + w * (cos - cos.min()) / max(float(cos.max() - cos.min()), 1e-9)
    return [cands[i]["doc"] for i in np.lexsort((np.arange(len(cands)), -s))]


def prepare(args):
    """Training groups (fit), eval heads (tune, dev), and the texts they need."""
    groups, heads, texts, queries = [], {"tune": {}, "dev": {}}, {}, {}
    for dom in args.domains:
        need = set()
        qrels_tr = load_qrels(args.root, dom, "train")
        q_tr = {e["id"]: clean(e["query"]) for e in load_examples(args.root, dom, "train")}
        q_dv = {e["id"]: clean(e["query"]) for e in load_examples(args.root, dom, "dev")}
        fit = sorted(subset_qids(args.root, dom, "fit"))[:args.limit_queries or None]
        pool = {}
        for line in open(Path(args.pool_dir) / dom / "pool_train.jsonl"):
            p = json.loads(line)
            if p["qid"] in fit:
                pool[p["qid"]] = p["candidates"]
        for q in fit:
            if q not in pool:
                continue
            gold = {d for d, v in qrels_tr[q].items() if v > 0}
            negs = [d for d in pool_ranking(pool[q])[:args.neg_depth] if d not in gold]
            for g in sorted(gold):
                groups.append((dom, q, g, negs))
            need |= gold | set(negs)
            queries[(dom, q)] = q_tr[q]
        for split, qs, qtext in (("tune", subset_qids(args.root, dom, "tune"), q_tr),
                                 ("dev", subset_qids(args.root, dom, "dev"), q_dv)):
            first = load_topk(Path(args.first_stage) / dom / f"top200_{'dev' if split == 'dev' else 'train'}.jsonl")
            for q in sorted(qs)[:args.limit_queries or None]:
                if q in first:
                    heads[split][(dom, q)] = first[q]
                    need |= {d for d, _ in first[q][:args.top_n]}
                    queries[(dom, q)] = qtext[q]
        docs = load_documents(args.root, dom)
        texts.update({d: docs[d] for d in need if d in docs})
        del docs
        print(f"  {dom}: {sum(1 for g in groups if g[0] == dom)} training groups, {len(need)} docs needed", flush=True)
    groups = [g for g in groups if g[2] in texts]
    return groups, heads, texts, queries


class Reranker:
    def __init__(self, path, max_len):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev)
        # fp16 autocast only on Volta or newer; a GTX 1080 Ti (CC 6.1) runs fp16 far slower than fp32
        self.amp = self.dev == "cuda" and torch.cuda.get_device_capability()[0] >= 7
        self.max_len = max_len

    def batch(self, pairs):
        enc = self.tok([q for q, _ in pairs], [d for _, d in pairs], truncation="longest_first",
                       max_length=self.max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.dev) for k, v in enc.items()}

    def logits(self, pairs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.model(**self.batch(pairs)).logits.view(-1).float()

    @torch.no_grad()
    def score(self, pairs, bs):
        self.model.eval()
        out = []
        for i in range(0, len(pairs), bs):
            out.append(self.logits(pairs[i:i + bs]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def evaluate(rr, args, split, heads, texts, queries, lams=(1.0, 0.0)):
    """Rerank the first-stage top_n with the current model. Returns {lam: macro nDCG@10}, per-domain, runs."""
    keys = list(heads[split])
    pairs, owner = [], []
    for k in keys:
        for d, _ in heads[split][k][:args.top_n]:
            if d in texts:
                pairs.append((queries[k], texts[d])); owner.append((k, d))
    t0 = time.time()
    scores = rr.score(pairs, args.eval_batch)
    ce = {k: {} for k in keys}
    for (k, d), s in zip(owner, scores):
        ce[k][d] = float(s)
    split_qrels = "dev" if split == "dev" else "train"
    qrels = {dom: load_qrels(args.root, dom, split_qrels) for dom in {k[0] for k in keys}}
    res, per_dom, runs = {}, {}, {}
    for lam in lams:
        vals = {}
        for k in keys:
            ranked = rerank(heads[split][k], ce[k], args.top_n, lam)
            if lam == lams[0]:
                runs[k] = ranked[:200]
            vals.setdefault(k[0], []).append(ndcg_at_k([d for d, _ in ranked], qrels[k[0]][k[1]], 10) or 0.0)
        pd = {d: float(np.mean(v)) for d, v in vals.items()}
        res[lam] = macro_average(pd)
        if lam == lams[0]:
            per_dom = pd
    print(f"    scored {len(pairs)} {split} pairs in {time.time() - t0:.0f}s ({len(pairs) / max(time.time() - t0, 1e-9):.0f}/s)",
          flush=True)
    return res, per_dom, runs


def train_epoch(rr, args, groups, texts, queries, opt, sched, scaler, rng, epoch):
    rr.model.train()
    order = list(range(len(groups))); rng.shuffle(order)
    G, n_micro, t0, losses = args.groups_per_batch, 0, time.time(), []
    total = math.ceil(len(order) / G)
    for b in range(0, len(order), G):
        pairs = []
        for i in order[b:b + G]:
            dom, q, g, negs = groups[i]
            pool = [d for d in negs if d in texts]
            chosen = rng.sample(pool, args.negatives) if len(pool) >= args.negatives else rng.choices(pool, k=args.negatives)
            pairs += [(queries[(dom, q)], texts[g])] + [(queries[(dom, q)], texts[d]) for d in chosen]
        logits = rr.logits(pairs).view(-1, 1 + args.negatives)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=rr.dev))
        scaler.scale(loss / args.accum).backward()
        losses.append(loss.item()); n_micro += 1
        if n_micro % args.accum == 0 or b + G >= len(order):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(rr.model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        if n_micro % 50 == 0 or n_micro == total:
            rate = n_micro / (time.time() - t0)
            print(f"    epoch {epoch}: {n_micro}/{total} batches | loss {np.mean(losses[-50:]):.4f} | "
                  f"ETA {(total - n_micro) / rate / 60:.1f} min", flush=True)
        if args.max_steps and n_micro >= args.max_steps:
            break
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="*", default=TRACK1_DOMAINS)
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--negatives", type=int, default=7, help="hard negatives per gold doc")
    ap.add_argument("--neg-depth", type=int, default=100, help="negatives come from the first-stage top N")
    ap.add_argument("--groups-per-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = groups x accum)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--top-n", type=int, default=100, help="Step 7 setting, not tuned here")
    ap.add_argument("--first-stage", default="runs/graph_BAAI__bge-base-en-v1.5_L512")
    ap.add_argument("--pool-dir", default="runs/pool_d300")
    ap.add_argument("--model-dir", default="models/step10")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--baseline-run", default="runs/step9_baseline", help="dev comparison run (not fine-tuned)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="testing: stop each epoch after N micro-batches")
    ap.add_argument("--limit-queries", type=int, default=0, help="testing: use the first N queries per domain/split")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("preparing data", flush=True)
    groups, heads, texts, queries = prepare(args)
    rr = Reranker(args.model, args.max_len)
    print(f"model={args.model} device={rr.dev} amp_fp16={rr.amp} | {len(groups)} groups x (1 + {args.negatives}) | "
          f"tune {len(heads['tune'])} / dev {len(heads['dev'])} queries", flush=True)

    hist = []
    print("\n[tune] epoch 0 (off-the-shelf model, sanity check vs Step 9 baseline)", flush=True)
    r0, _, _ = evaluate(rr, args, "tune", heads, texts, queries)
    hist.append({"epoch": 0, "tune": r0[1.0], "tune_ce_only": r0[0.0]})
    print(f"  epoch 0: tune nDCG@10 {r0[1.0]:.4f} (CE only {r0[0.0]:.4f})", flush=True)

    steps = math.ceil(math.ceil(len(groups) / args.groups_per_batch) / args.accum) * args.epochs
    opt = torch.optim.AdamW(rr.model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=rr.amp)
    mdir = Path(args.model_dir)
    for ep in range(1, args.epochs + 1):
        print(f"\n[train] epoch {ep}", flush=True)
        loss = train_epoch(rr, args, groups, texts, queries, opt, sched, scaler, rng, ep)
        rr.model.save_pretrained(mdir / f"epoch{ep}"); rr.tok.save_pretrained(mdir / f"epoch{ep}")
        r, _, _ = evaluate(rr, args, "tune", heads, texts, queries)
        hist.append({"epoch": ep, "train_loss": loss, "tune": r[1.0], "tune_ce_only": r[0.0]})
        print(f"  epoch {ep}: loss {loss:.4f} | tune nDCG@10 {r[1.0]:.4f} (CE only {r[0.0]:.4f})", flush=True)

    best = max(hist, key=lambda h: (h["tune"], -h["epoch"]))["epoch"]
    print(f"\n[tune] best epoch = {best}" + (" (fine-tuning did not beat the off-the-shelf model)" if best == 0 else ""))
    path = args.model if best == 0 else mdir / f"epoch{best}"
    rr = Reranker(str(path), args.max_len)
    print("\n[dev] scoring the chosen model once", flush=True)
    rd, dom_dev, run_dev = evaluate(rr, args, "dev", heads, texts, queries)
    _, _, run_tune = evaluate(rr, args, "tune", heads, texts, queries, lams=(1.0,))

    base = {}
    for dom in args.domains:
        b = load_topk(Path(args.baseline_run) / dom / "top200_dev.jsonl")
        qr = load_qrels(args.root, dom, "dev")
        base[dom] = float(np.mean([ndcg_at_k([d for d, _ in b[q]], qr[q], 10) or 0.0
                                   for (d_, q) in heads["dev"] if d_ == dom and q in b]))
    print(f"\n{'epoch':<7}{'train loss':>11}{'tune':>9}{'tune CE only':>14}")
    for h in hist:
        print(f"{h['epoch']:<7}{h.get('train_loss', float('nan')):>11.4f}{h['tune']:>9.4f}{h['tune_ce_only']:>14.4f}")
    print(f"\nDEV nDCG@10: fine-tuned (epoch {best}) {rd[1.0]:.4f} (CE only {rd[0.0]:.4f}) | "
          f"Step 9 baseline {macro_average(base):.4f}")
    print("Dev per-domain nDCG@10 (baseline -> fine-tuned)")
    for dom in args.domains:
        print(f"  {dom:<10} {base[dom]:.4f} -> {dom_dev.get(dom, float('nan')):.4f}")

    out = Path(args.out_root) / "step10_ft"
    for split, run in (("train", run_tune), ("dev", run_dev)):
        for dom in args.domains:
            r = {q: v for (d_, q), v in run.items() if d_ == dom}
            save_topk(out / dom / f"top200_{split}.jsonl", r)
            write_trec(out / dom / f"run_1a_{split}.trec", r, "step10_ft")
    if best:
        if (mdir / "best").exists():
            shutil.rmtree(mdir / "best")
        (mdir / f"epoch{best}").rename(mdir / "best")
    for ep in range(1, args.epochs + 1):                     # checkpoints are ~1 GB each; keep only the chosen one
        if (mdir / f"epoch{ep}").exists():
            shutil.rmtree(mdir / f"epoch{ep}")
    (out / "summary.json").write_text(json.dumps({"best_epoch": best, "history": hist, "dev": rd[1.0],
                                                  "dev_ce_only": rd[0.0], "dev_per_domain": dom_dev,
                                                  "dev_baseline": macro_average(base), "args": vars(args)}, indent=2))
    print(f"\nmodel: {mdir / 'best' if best else args.model} | runs: {out}")
    print("significance: python3 scripts/step8_significance.py --root ../reteco_data --subset dev "
          "--runs baseline=runs/step9_baseline finetuned=runs/step10_ft")


if __name__ == "__main__":
    main()

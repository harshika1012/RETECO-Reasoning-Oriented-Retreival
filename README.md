# RETECO: Reasoning-Oriented Retrieval (SemEval-2027 Task 1, Track 1a)

Retrieval for long StackExchange-style questions that need temporal reasoning, across 13 domains of the
[RETECO](https://github.com/DataScienceUIBK/RETECO) benchmark. The metric is nDCG@10, macro-averaged over domains.

**Current best: dev nDCG@10 = 0.3540** (official BM25 baseline: 0.1147). The pipeline retrieves with BM25 and dense
embeddings, fuses them, and reranks the top 100 with a `bge-reranker-base` cross-encoder fine-tuned on the train split.

## Results

Macro nDCG@10 over 13 domains. *tune* = 242 held-out train queries, *dev* = 519 queries. Significance is a paired
bootstrap stratified by domain (`scripts/step8_significance.py`).

| Step | Method | tune | dev | Change on dev | Verdict |
|---|---|---|---|---|---|
| 2 | BM25, official starter kit | 0.1130 | 0.1147 | baseline | kept as retriever |
| 3 | Dense retrieval, `bge-base-en-v1.5` | 0.2059 | 0.2481 | +13.3, p<0.001 | kept |
| 4 | RRF fusion of BM25 + dense top-200 pools | 0.2005 | 0.2587 | +1.1, not significant | kept for recall |
| 5 | + regex year-match bonus | n/a | 0.2579 | -0.1 | dropped |
| 6a | + raw dense cosine added to the RRF score | 0.2185 | 0.2769 | +1.8, p<0.001 | kept |
| 6b | + kNN graph: pool expansion, score spreading | n/a | 0.2643 | -1.3 | dropped |
| 7 | + `bge-reranker-base` on top 100, blended | 0.2310 | 0.3030 | +2.6, p=0.001 | kept |
| 7 | + `bge-reranker-v2-m3` instead | 0.2363 | 0.3098 | +0.7 vs base, not significant | not adopted |
| 9 | Deeper pool (dense to depth 300-1000) | 0.2381 | 0.2855 | -1.8, p=0.007 | dropped |
| **10** | **Fine-tuned `bge-reranker-base`, 3 epochs** | **0.2897** | **0.3540** | **+5.1, p<0.001** | **current best** |

Key findings:

- Dense retrieval is the largest single gain. RRF fusion helps recall, not ranking; adding the raw dense cosine helps both splits.
- A regex year signal and a graph built on the same dense embeddings added no new information.
- A deeper pool raises recall (0.70 to 0.89 on tune) but not nDCG: half of the gold documents already sit in the
  first-stage top 50, and the main loss is the reranker's ordering there.
- Fine-tuning the reranker on the train split fixes that: +5.1 dev, improving 12 of 13 domains.

## Repository layout

```text
RETECO-Reasoning-Oriented-Retreival/
├── README.md
├── reteco_data/            # NOT committed: download it (Setup, step 2)
├── RETECO/                 # NOT committed: official starter kit, git clone (Setup, step 3)
└── Reteco_step/            # all project code; run every command from here
    ├── requirements.txt
    ├── src/                # shared modules: data loading, metrics, BM25, dense encoder, run I/O, splits
    ├── scripts/            # stepN_*.py (one per experiment) + stepN_*.sbatch (Slurm job scripts)
    ├── tests/
    ├── splits/train_heldout.json   # frozen 80/20 fit/tune split of train queries (needed to reproduce Steps 8-10)
    ├── results/            # significance tables (CSV)
    ├── cache/              # generated: embeddings, BM25 indexes, cross-encoder scores   (ignored by git)
    ├── runs/               # generated: per-step rankings, top200_<split>.jsonl + TREC    (ignored by git)
    ├── models/             # generated: fine-tuned reranker checkpoints, ~1 GB each       (ignored by git)
    └── logs/               # generated: Slurm job logs                                     (ignored by git)
```

The scripts default to `--root ../reteco_data` and `--starter-kit ../RETECO/starter_kit`, so `reteco_data/` and
`RETECO/` must sit next to `Reteco_step/`, as above.

## Setup

1. **Python environment** (Python 3.12):

   ```bash
   conda create -y -n reteco312 python=3.12 && conda activate reteco312
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r Reteco_step/requirements.txt
   ```

   The CUDA 12.6 build of torch supports both GTX 1080 Ti (sm_61) and newer GPUs; CUDA 12.8+ builds do not.

2. **Data** (about 1.9 GB for Track 1), from the repository root:

   ```bash
   pip install huggingface_hub
   hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir reteco_data
   ```

3. **Official starter kit** (only needed for Step 2, the official BM25 baseline):

   ```bash
   git clone https://github.com/DataScienceUIBK/RETECO.git
   pip install gensim pyserini     # pyserini needs a JDK (21 works)
   ```

## Running the pipeline

Run from `Reteco_step/`. On the cluster, submit the `.sbatch` version of a step instead
(`sbatch scripts/stepN_*.sbatch`); logs go to `logs/stepN_<jobid>.out`.
Each step reads the outputs of the steps before it.

| Step | Command | Needs | Writes |
|---|---|---|---|
| 1 | `python3 scripts/step1_verify.py --root ../reteco_data` | data | checks our nDCG@10 against pytrec_eval |
| 2 | `python3 scripts/step2_official_bm25.py --root ../reteco_data --starter-kit ../RETECO/starter_kit` | starter kit | `runs/bm25_official` |
| 3 | `python3 scripts/step3_dense.py --root ../reteco_data` (GPU) | data | `cache/dense`, `runs/dense_*` |
| 4 | `python3 scripts/step4_union_pool.py --root ../reteco_data --dense-dir runs/dense_BAAI__bge-base-en-v1.5_L512` | 2, 3 | `runs/pool` |
| 5 | `python3 scripts/step5_regex_temporal.py --root ../reteco_data` | 4 | printed results only |
| 6 | `python3 scripts/step6_graph_rerank.py --root ../reteco_data` | 3, 4 | `runs/graph_*` (= RRF + dense cosine first stage) |
| 7 | `python3 scripts/step7_cross_encoder.py --root ../reteco_data [--model BAAI/bge-reranker-v2-m3]` (GPU) | 6 | `runs/ce_*` |
| 8 | `python3 scripts/step8_significance.py --root ../reteco_data --subset dev` | runs | `results/`, creates `splits/` if missing |
| 9 | `python3 scripts/step9_deep_pool.py --root ../reteco_data` (GPU) | 2, 3, 8 | `runs/step9_*`, `runs/pool_d<depth>` |
| 10 | `python3 scripts/step10_finetune_ce.py --root ../reteco_data` (GPU, 2080 Ti or newer) | 6, 8, 9 | `models/step10/best`, `runs/step10_ft` |

To compare any two runs: `python3 scripts/step8_significance.py --root ../reteco_data --subset dev --runs a=<run dir> b=<run dir>`.

## Evaluation protocol

- Train queries are split once, per domain, into **fit** (80%, 969 queries) and **tune** (20%, 242 queries),
  frozen in `splits/train_heldout.json`.
- **fit** is used only for training (Step 10). Settings are chosen on **tune**, and only the setting being tested
  is tuned; everything else stays fixed. **dev** is scored once per step, as the final check.
- Why: in Step 9, choosing among 216 settings on 242 queries looked +0.7 better on tune but was significantly worse
  on dev (-1.8).


## Data and licence

RETECO text is CC BY-SA 4.0 and its annotations CC BY 4.0
([dataset card](https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027)). The data and the official
starter kit are not redistributed here; download them as described above.

#!/bin/bash
# Step 10 reranker fine-tuning on an RTX 2080 Ti node (fp16; a 1080 Ti would be several times slower).
# Submit from the repo root:   sbatch scripts/step10_finetune_ce.sbatch [extra step10_finetune_ce.py args]
#   quick test: sbatch scripts/step10_finetune_ce.sbatch --domains iota law --epochs 1 --max-steps 20 --limit-queries 10
# Watch:   tail -f logs/step10_<jobid>.out      Status: squeue -u $USER      Stop: scancel <jobid>
#SBATCH --job-name=step10_ft
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=29G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step10_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reteco312
# compute nodes may lack the login node's CA bundle; fall back to certifi's, and use cached HF models only
[ -f "${SSL_CERT_FILE:-}" ] || export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID on $(hostname) started $(date)"
nvidia-smi -L
python3 scripts/step10_finetune_ce.py --root ../reteco_data "$@"
echo "finished $(date)"

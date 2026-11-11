#!/bin/bash
# Step 7 cross-encoder rerank as a batch job (GPU; CE scores are cached in cache/ce, so resubmitting resumes).
# Submit from the repo root:   sbatch scripts/step7_cross_encoder.sbatch [extra step7_cross_encoder.py args]
#   e.g. sbatch scripts/step7_cross_encoder.sbatch --domains iota law        (quick test)
# Watch:   tail -f logs/step7_<jobid>.out      Status: squeue -u $USER      Stop: scancel <jobid>
#SBATCH --job-name=step7_ce
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gnode004
#SBATCH --output=logs/step7_%j.out

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
python3 scripts/step7_cross_encoder.py --root ../reteco_data "$@"
echo "finished $(date)"

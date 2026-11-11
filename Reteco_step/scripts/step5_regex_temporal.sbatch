#!/bin/bash
# Step 5 regex temporal rerank as a batch job (CPU only; doc years are cached in cache/doc_years).
# Submit from the repo root:   sbatch scripts/step5_regex_temporal.sbatch [extra step5_regex_temporal.py args]
# Watch:   tail -f logs/step5_<jobid>.out      Status: squeue -u $USER      Stop: scancel <jobid>
#SBATCH --job-name=step5_temporal
#SBATCH --partition=u22
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step5_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reteco312
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname) started $(date)"
python3 scripts/step5_regex_temporal.py --root ../reteco_data "$@"
echo "finished $(date)"

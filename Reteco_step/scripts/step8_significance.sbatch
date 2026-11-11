#!/bin/bash
# Step 8 significance tests (CPU, a few minutes). Runs dev and the held-out train split ("tune").
# Submit from the repo root:   sbatch scripts/step8_significance.sbatch [extra step8_significance.py args]
# Watch:   tail -f logs/step8_<jobid>.out      Status: squeue -u $USER      Stop: scancel <jobid>
#SBATCH --job-name=step8_sig
#SBATCH --partition=u22
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=1:00:00
#SBATCH --output=logs/step8_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reteco312
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname) started $(date)"
for subset in tune dev; do
  python3 scripts/step8_significance.py --root ../reteco_data --subset "$subset" "$@"
  echo
done
echo "finished $(date)"

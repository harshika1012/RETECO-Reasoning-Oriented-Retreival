#!/bin/bash
# Step 6 graph rerank + pool expansion as a batch job (CPU is enough; the kNN graph is cached in cache/graph).
# Submit from the repo root:   sbatch scripts/step6_graph_rerank.sbatch [extra step6_graph_rerank.py args]
# Watch:   tail -f logs/step6_<jobid>.out      Status: squeue -u $USER      Stop: scancel <jobid>
#SBATCH --job-name=step6_graph
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --mem=29G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step6_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reteco312
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname) started $(date)"
python3 scripts/step6_graph_rerank.py --root ../reteco_data "$@"
echo "finished $(date)"

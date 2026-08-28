#!/bin/bash
#SBATCH --job-name=pathloss-cpu
#SBATCH --clusters=htc
#SBATCH --partition=short           # devel (10 min) while testing, then short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/logs/slurm-%j.out
#SBATCH --error=results/logs/slurm-%j.err

# --- environment -----------------------------------------------------------
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="${1:-configs/baseline_mse.yaml}"
RUN_ID="${SLURM_JOB_ID}_$(basename "$CONFIG" .yaml)"

# --- stage to per-job scratch ---------------------------------------------
cd "$SCRATCH" || exit 1
rsync -a "$PROJECT_DIR/src" "$PROJECT_DIR/scripts" "$PROJECT_DIR/configs" ./
mkdir -p results

echo "job $SLURM_JOB_ID on $(hostname); config=$CONFIG"
echo "git SHA: $(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"

# --- run -------------------------------------------------------------------
PYTHONPATH=./src python scripts/run_reconstruction.py \
    --config "$CONFIG" \
    --out results/

# --- copy results back BEFORE the job exits ($SCRATCH is destroyed) --------
rsync -a results/ "$PROJECT_DIR/results/runs/$RUN_ID/"
echo "results -> $PROJECT_DIR/results/runs/$RUN_ID/"

#!/bin/bash
#SBATCH --job-name=node-closeout
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-41
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/logs/node-closeout-%A_%a.out
#SBATCH --error=results/logs/node-closeout-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path_closeout_10k.yaml"
OUT="$PROJECT_DIR/results/runs/neural_ode_fixed_path_closeout_10k"
mkdir -p "$OUT" "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

python scripts/run_fixed_path_study.py \
    --config "$CONFIG" \
    --out "$OUT" \
    --stage configured \
    --task-id "$SLURM_ARRAY_TASK_ID"


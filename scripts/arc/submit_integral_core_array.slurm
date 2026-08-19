#!/bin/bash
#SBATCH --job-name=integral-core
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/integral-%A_%a.out
#SBATCH --error=results/logs/integral-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
RUN_DIR="$PROJECT_DIR/results/runs/integral_core"
mkdir -p "$RUN_DIR" "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

python scripts/run_integral_study.py \
    --config configs/integral_core_study.yaml \
    --out "$RUN_DIR" \
    --index "$SLURM_ARRAY_TASK_ID"

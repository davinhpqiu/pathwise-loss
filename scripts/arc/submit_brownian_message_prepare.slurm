#!/bin/bash
#SBATCH --job-name=bm-prepare
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/logs/bm-prepare-%j.out
#SBATCH --error=results/logs/bm-prepare-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

PROJECT_DIR="$DATA/pathwise-loss"
cd "$PROJECT_DIR"

python scripts/run_brownian_message_study.py \
    --config configs/brownian_message.yaml \
    --out results/runs/brownian_message \
    --stage prepare \
    --ensemble main

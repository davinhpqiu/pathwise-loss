#!/bin/bash
#SBATCH --job-name=ou-signature
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/ou-signature-%A_%a.out
#SBATCH --error=results/logs/ou-signature-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_cde_brownian_ou.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

seed=$((SLURM_ARRAY_TASK_ID / 2))
loss_index=$((SLURM_ARRAY_TASK_ID % 2))
losses=(sig_global sig_local)
loss="${losses[$loss_index]}"
run_dir="$PROJECT_DIR/results/runs/neural_cde_brownian_ou/seed$seed/uniform/$loss"

python scripts/run_ou_operator.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed "$seed" \
    --condition uniform \
    --loss "$loss" \
    --evaluate-test

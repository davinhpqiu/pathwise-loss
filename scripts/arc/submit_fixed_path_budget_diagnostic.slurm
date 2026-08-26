#!/bin/bash
#SBATCH --job-name=node-budget-10k
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/node-budget-10k-%A_%a.out
#SBATCH --error=results/logs/node-budget-10k-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path_budget_10k.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

losses=(mse j2)
loss="${losses[$SLURM_ARRAY_TASK_ID]}"
run_dir="$PROJECT_DIR/results/runs/neural_ode_fixed_path_budget_10k/expressive/seed0/clustered/$loss"

python scripts/run_fixed_path.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed 0 \
    --capacity expressive \
    --condition clustered \
    --loss "$loss"

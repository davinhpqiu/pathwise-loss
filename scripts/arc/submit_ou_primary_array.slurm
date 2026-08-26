#!/bin/bash
#SBATCH --job-name=ou-primary
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/ou-primary-%A_%a.out
#SBATCH --error=results/logs/ou-primary-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_cde_brownian_ou.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

seed=$((SLURM_ARRAY_TASK_ID / 4))
case_index=$((SLURM_ARRAY_TASK_ID % 4))
conditions=(uniform clustered uniform clustered)
losses=(mse mse j2 j2)
condition="${conditions[$case_index]}"
loss="${losses[$case_index]}"
run_dir="$PROJECT_DIR/results/runs/neural_cde_brownian_ou/seed$seed/$condition/$loss"

python scripts/run_ou_operator.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed "$seed" \
    --condition "$condition" \
    --loss "$loss" \
    --evaluate-test

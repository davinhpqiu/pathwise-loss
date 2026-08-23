#!/bin/bash
#SBATCH --job-name=node-fixed-path
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/node-fit-%A_%a.out
#SBATCH --error=results/logs/node-fit-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

seed=$((SLURM_ARRAY_TASK_ID / 10))
within_seed=$((SLURM_ARRAY_TASK_ID % 10))
capacity_index=$((within_seed / 5))
case_index=$((within_seed % 5))

capacities=(restricted expressive)
conditions=(uniform clustered uniform clustered uniform)
losses=(mse mse j2 j2 h1)

capacity="${capacities[$capacity_index]}"
condition="${conditions[$case_index]}"
loss="${losses[$case_index]}"
run_dir="$PROJECT_DIR/results/runs/neural_ode_fixed_path/$capacity/seed$seed/$condition/$loss"

python scripts/run_fixed_path.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed "$seed" \
    --capacity "$capacity" \
    --condition "$condition" \
    --loss "$loss"

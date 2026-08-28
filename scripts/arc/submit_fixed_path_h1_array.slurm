#!/bin/bash
#SBATCH --job-name=node-fixed-h1
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/node-h1-%A_%a.out
#SBATCH --error=results/logs/node-h1-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

seed=$((SLURM_ARRAY_TASK_ID / 2))
capacity_index=$((SLURM_ARRAY_TASK_ID % 2))
capacities=(restricted expressive)
capacity="${capacities[$capacity_index]}"
run_dir="$PROJECT_DIR/results/runs/neural_ode_fixed_path/$capacity/seed$seed/uniform/h1"

python scripts/run_fixed_path.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed "$seed" \
    --capacity "$capacity" \
    --condition uniform \
    --loss h1

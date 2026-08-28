#!/bin/bash
#SBATCH --job-name=node-sig-pilot
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/logs/node-sig-pilot-%A_%a.out
#SBATCH --error=results/logs/node-sig-pilot-%A_%a.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path_signature_10k.yaml"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

capacity_index=$((SLURM_ARRAY_TASK_ID / 4))
loss_index=$((SLURM_ARRAY_TASK_ID % 4))
capacities=(restricted expressive)
losses=(mse j2 sig_global sig_local)
capacity="${capacities[$capacity_index]}"
loss="${losses[$loss_index]}"
run_dir="$PROJECT_DIR/results/runs/neural_ode_fixed_path_signature_10k/$capacity/seed0/uniform/$loss"

python scripts/run_fixed_path.py \
    --config "$CONFIG" \
    --out "$run_dir" \
    --seed 0 \
    --capacity "$capacity" \
    --condition uniform \
    --loss "$loss"

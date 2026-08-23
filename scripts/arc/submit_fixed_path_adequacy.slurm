#!/bin/bash
#SBATCH --job-name=node-adequacy
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/node-adequacy-%j.out
#SBATCH --error=results/logs/node-adequacy-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
RUN_DIR="$PROJECT_DIR/results/runs/neural_ode_fixed_path/adequacy"
mkdir -p "$RUN_DIR" "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

python scripts/run_fixed_path.py \
    --config configs/neural_ode_fixed_path.yaml \
    --out "$RUN_DIR" \
    --adequacy

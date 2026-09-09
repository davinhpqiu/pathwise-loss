#!/bin/bash
#SBATCH --job-name=node-local100-audit
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/logs/node-local100-audit-%j.out
#SBATCH --error=results/logs/node-local100-audit-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="$PROJECT_DIR/configs/neural_ode_fixed_path_local_fine_10k.yaml"
OUT="$PROJECT_DIR/results/runs/neural_ode_fixed_path_local_fine_10k/signature_audit"
mkdir -p "$OUT" "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

python scripts/run_fixed_path.py \
    --config "$CONFIG" \
    --out "$OUT" \
    --signature-audit

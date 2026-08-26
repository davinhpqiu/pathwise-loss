#!/bin/bash
#SBATCH --job-name=ou-sig-audit
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/ou-sig-audit-%j.out
#SBATCH --error=results/logs/ou-sig-audit-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
mkdir -p "$PROJECT_DIR/results/logs"
cd "$PROJECT_DIR"

python scripts/run_ou_operator.py \
    --config configs/neural_cde_brownian_ou.yaml \
    --out results/runs/neural_cde_brownian_ou/signature_audit \
    --signature-audit

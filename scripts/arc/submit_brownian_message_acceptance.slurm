#!/bin/bash
#SBATCH --job-name=bm-accept
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=results/logs/bm-accept-%j.out
#SBATCH --error=results/logs/bm-accept-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
cd "$PROJECT_DIR"
mkdir -p results/logs

python scripts/run_brownian_message_study.py \
    --config configs/brownian_message.yaml \
    --out results/runs/brownian_message \
    --stage acceptance

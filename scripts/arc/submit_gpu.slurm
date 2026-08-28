#!/bin/bash
#SBATCH --job-name=pathloss-gpu
#SBATCH --clusters=htc               # GPUs exist only on htc
#SBATCH --partition=short            # co-investment GPU nodes are capped at short (12h)
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # dataloader workers
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
##SBATCH --constraint='gpu_mem:32GB' # uncomment if the signature Gram matrix OOMs
#SBATCH --output=results/logs/slurm-%j.out
#SBATCH --error=results/logs/slurm-%j.err

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$DATA/pathwise-loss/.venv/bin/activate"

PROJECT_DIR="$DATA/pathwise-loss"
CONFIG="${1:-configs/sigkernel_loss.yaml}"
RUN_ID="${SLURM_JOB_ID}_$(basename "$CONFIG" .yaml)"

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd "$SCRATCH" || exit 1
rsync -a "$PROJECT_DIR/src" "$PROJECT_DIR/scripts" "$PROJECT_DIR/configs" "$PROJECT_DIR/data" ./
mkdir -p results

# Periodically copy partial results back, in case of a timeout.
copy_back() { rsync -a --partial results/ "$PROJECT_DIR/results/runs/$RUN_ID/"; }
trap 'echo "caught signal - saving"; copy_back; exit 2' SIGINT SIGTERM

PYTHONPATH=./src python scripts/run_reconstruction.py --config "$CONFIG" --out results/ &
jobpid=$!
while kill -0 $jobpid 2>/dev/null; do sleep 600; copy_back; done
wait $jobpid

copy_back
echo "results -> $PROJECT_DIR/results/runs/$RUN_ID/"

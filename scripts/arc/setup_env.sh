#!/bin/bash
# One-time environment build on ARC. RUN THIS FROM AN INTERACTIVE NODE:
#
#     srun -p interactive --pty /bin/bash
#     cd $DATA/pathwise-loss && bash scripts/arc/setup_env.sh
#
# Login nodes are capped at 1 hour CPU time and are not for builds.

set -euo pipefail

# Keep package caches off $HOME (15 GiB quota) -- the classic ARC footgun.
export PIP_CACHE_DIR="$DATA/.cache/pip"
export CONDA_PKGS_DIRS="$DATA/.cache/conda"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

module purge
# Check the exact name first with:  module spider python
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="$DATA/pathwise-loss"
cd "$PROJECT_DIR"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# Core first, so a failure in the torch stack never blocks notebook/test work
pip install -r requirements.txt
pip install -e .

# Torch: match the CUDA build to the module you loaded; check with nvidia-smi on a GPU node
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# iisignature imports NumPy while building but does not declare NumPy in its
# isolated build environment. Core requirements above already installed NumPy;
# Cython must also be present before installing the modelling stack without
# isolation.
python -m pip install "cython>=3.0"
python -m pip install --no-build-isolation -r requirements-ml.txt

# sigkernel needs Cython at build time but does not declare it, so pip's isolated
# build fails. cython came from requirements-ml.txt above; bypass isolation here.
python -m pip install --no-build-isolation \
    git+https://github.com/crispitagorico/sigkernel.git || \
    echo "WARNING: sigkernel install failed -- continue without it for now"

echo
echo "Environment built at $PROJECT_DIR/.venv"
echo "Activate in job scripts with: source \$DATA/pathwise-loss/.venv/bin/activate"

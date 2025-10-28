#!/usr/bin/env zsh
# Setup script for venv + PyTorch + project install
# Usage:
#   zsh inf2070-studies/setup.sh         # CPU-only (recommended)
#   USE_GPU=1 zsh inf2070-studies/setup.sh   # try GPU build (requires matching torch wheel)

# activate venv

if [ ! -d venv ]; then
  echo "Creating virtual environment in ./venv"
  python3 -m venv venv
fi

source venv/bin/activate

# ensure CUDA env (update if different)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# verify nvcc when present
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc -> $(command -v nvcc)"
  nvcc --version | head -n 1
else
  echo "nvcc not found in PATH (ok for CPU-only install)"
fi

# upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Default: CPU-only PyTorch to avoid building flash-attn/rl_square CUDA extensions.
# If you want GPU build, set USE_GPU=1 (may still fail unless torch wheel matches rl_square).
if [ "${USE_GPU:-0}" -eq 1 ]; then
  echo "GPU mode: installing GPU-capable torch (user must ensure correct CUDA/PyTorch pairing)"
  # You should replace the following line with the exact pip command from https://pytorch.org
  # matching your CUDA driver (example commented out):
  # pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.8.0 torchvision==0.18.0
  echo "NOTE: please run the correct torch install command for your CUDA version (see https://pytorch.org/get-started/locally/)"
else
  echo "CPU mode: installing CPU-only torch to avoid building GPU extensions"
  pip install --index-url https://download.pytorch.org/whl/cpu "torch" torchvision || \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" torchvision || true
fi

# Install project without forcing build of all optional deps that require CUDA.
# Install the local package (editable not required); avoid auto-building some dependency wheels by installing without deps,
# then install known runtime deps explicitly.
pip install . --no-deps || pip install -e . --no-deps

# Now install common runtime/test deps (excluding rl_square/flash-attn which trigger CUDA builds).
pip install pytest fire fastmcp ipykernel matplotlib-inline jupyter_client jupyter_core ipython jedi pandas numpy openpyxl jinja2 gem-llm || true

echo "Done. If you need oat-llm (GPU) install them separately after ensuring a matching torch wheel:"
echo "  pip install oat-llm    # will try to build flash-attn"

echo "Done. If you need rl_square or flash-attn (GPU) install them separately after ensuring a matching torch wheel:"
echo "  pip install rl_square    # will try to build flash-attn"



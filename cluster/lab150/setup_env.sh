#!/bin/bash
# Create the project venv ON TOP of the image's PyTorch and install the project into it.
#
# Why --system-site-packages: torch/CUDA come from the lab image (built for the lab's driver).
# The venv only adds what is missing. A constraints file pins torch (and friends, if present)
# to the image versions so pip can never replace them with a PyPI build.
set -euo pipefail
source cluster/lab150/env.sh

python --version
# The image's Python (Ubuntu 24.04) has no ensurepip, so `python -m venv` cannot put pip inside
# the venv. We create it --without-pip: the image's own pip is visible through
# --system-site-packages and, run by the venv's interpreter, installs into the venv.
if [[ ! -f "$FFC_VENV/pyvenv.cfg" ]]; then
  mkdir -p "$(dirname "$FFC_VENV")"
  python -m venv --system-site-packages --without-pip "$FFC_VENV"
  echo "created venv at $FFC_VENV"
fi
activate_venv
if ! python -m pip --version > /dev/null 2>&1; then
  echo "no pip in the image; bootstrapping pip into the venv"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$FFC_VENV/get-pip.py"
  python "$FFC_VENV/get-pip.py" --quiet
fi
python -m pip --version
python -c "import sys; assert sys.prefix != sys.base_prefix, 'not inside the venv'; print('installing into', sys.prefix)"

CONSTRAINTS="$FFC_VENV/image-constraints.txt"
python - > "$CONSTRAINTS" <<'PY'
import importlib.metadata as md
for pkg in ("torch", "torchvision", "numpy", "pillow"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
PY
echo "pinned from image:"; cat "$CONSTRAINTS"

python -m pip install --constraint "$CONSTRAINTS" --editable ".[features]"

python - <<'PY'
import importlib.metadata as md
import torch
for pkg in ("torch", "torchvision", "transformers", "scikit-learn", "umap-learn", "numpy", "ffc"):
    print(f"{pkg:14s} {md.version(pkg)}")
print("cuda build", torch.version.cuda)
PY
ffc --help > /dev/null && echo "setup OK"

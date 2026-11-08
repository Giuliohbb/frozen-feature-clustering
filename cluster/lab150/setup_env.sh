#!/bin/bash
# Create the project venv ON TOP of the image's PyTorch and install the project into it.
#
# Why --system-site-packages: torch/CUDA come from the lab image (built for the lab's driver).
# The venv only adds what is missing. A constraints file pins torch (and friends, if present)
# to the image versions so pip can never replace them with a PyPI build.
set -euo pipefail
source cluster/lab150/env.sh

python --version
if [[ ! -x "$FFC_VENV/bin/python" ]]; then
  mkdir -p "$(dirname "$FFC_VENV")"
  python -m venv --system-site-packages "$FFC_VENV"
  echo "created venv at $FFC_VENV"
fi
activate_venv
python -m pip install --quiet --upgrade pip

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

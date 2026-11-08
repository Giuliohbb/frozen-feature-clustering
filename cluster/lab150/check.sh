#!/bin/bash
# Inspect the job environment: Python, torch/CUDA, GPU, CPUs, disk and internet access.
set -uo pipefail
source cluster/lab150/env.sh

section() { printf '\n== %s\n' "$1"; }

section "host";        hostname; date
section "python";      command -v python; python --version
section "torch";       python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
section "packages in image"
for pkg in torchvision transformers numpy pillow scikit-learn umap-learn; do
  python - "$pkg" <<'PY'
import importlib.metadata as md, sys
pkg = sys.argv[1]
try:
    print(f"{pkg:14s} {md.version(pkg)}")
except md.PackageNotFoundError:
    print(f"{pkg:14s} NOT INSTALLED")
PY
done
section "venv";        [[ -x "$FFC_VENV/bin/python" ]] && echo "present at $FFC_VENV" || echo "not created yet"
section "cpus/memory"; echo "nproc=$(nproc) SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"; free -g | head -2
section "disk";        df -h "$HOME" | tail -1; du -sh "$HOME/dados" 2>/dev/null || true
section "internet from inside the job (HTTP status, 000 = blocked)"
for url in https://pypi.org/simple/ \
           https://huggingface.co/api/models/facebook/dinov2-base \
           https://download.pytorch.org/models/ \
           https://www.cs.toronto.edu/~kriz/cifar.html \
           http://ai.stanford.edu/~acoates/stl10/; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url")" "$url"
done

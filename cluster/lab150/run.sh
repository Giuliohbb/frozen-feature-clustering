#!/bin/bash
# Run the clustering grid on cached features (CPU only, resumable). Usage: run.sh CONFIG
set -euo pipefail
source cluster/lab150/env.sh
activate_venv
CONFIG="$1"
NAME="$(basename "$CONFIG" .yaml)"

# Match thread pools to the CPUs Slurm gave us, to avoid oversubscribing the node.
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS"

ffc run --config "$CONFIG" --cache-dir "$FFC_STORE/cache" --results-dir "$FFC_RUNS/$NAME"
ffc report --config "$CONFIG" --results-dir "$FFC_RUNS/$NAME"

#!/bin/bash
#SBATCH --job-name=ffc-run
#SBATCH --partition=lab150
#SBATCH --account=lab150-alunos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/runs/%x-%j.out
#SBATCH --error=/home/%u/runs/%x-%j.err
# Clustering grid on cached features. CPU only (does not hold the GPU). Resumable: if the job
# is interrupted or times out, submitting it again continues from the last finished run.
# Usage: sbatch cluster/lab150/run.sbatch [CONFIG]                (default: configs/smoke.yaml)
set -euo pipefail
# %u is not expanded in --chdir on this cluster, so change directory here instead.
cd "$HOME/frozen-feature-clustering"
export PATH=/usr/bin:/bin
CONFIG="${1:-configs/smoke.yaml}"
source cluster/lab150/container.sh
in_container -- bash cluster/lab150/run.sh "$CONFIG"

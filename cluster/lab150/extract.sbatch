#!/bin/bash
#SBATCH --job-name=ffc-extract
#SBATCH --partition=lab150
#SBATCH --account=lab150-alunos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/%u/frozen-feature-clustering
#SBATCH --output=/home/%u/runs/%x-%j.out
#SBATCH --error=/home/%u/runs/%x-%j.err
# Feature extraction on the RTX 4090. Skips (source, encoder) pairs already cached.
# Usage: sbatch cluster/lab150/extract.sbatch [CONFIG]            (default: configs/smoke.yaml)
#        sbatch --time=03:00:00 cluster/lab150/extract.sbatch configs/grid.yaml
set -euo pipefail
export PATH=/usr/bin:/bin
CONFIG="${1:-configs/smoke.yaml}"
source cluster/lab150/container.sh
in_container --gres=gpu:rtx4090:1 -- bash cluster/lab150/extract.sh "$CONFIG"

#!/bin/bash
#SBATCH --job-name=ffc-setup
#SBATCH --partition=lab150
#SBATCH --account=lab150-alunos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/%u/frozen-feature-clustering
#SBATCH --output=/home/%u/runs/%x-%j.out
#SBATCH --error=/home/%u/runs/%x-%j.err
# Create/update the venv on top of the image's PyTorch. No GPU needed. Idempotent.
# Usage: sbatch cluster/lab150/setup_env.sbatch
set -euo pipefail
export PATH=/usr/bin:/bin
source cluster/lab150/container.sh
in_container -- bash cluster/lab150/setup_env.sh

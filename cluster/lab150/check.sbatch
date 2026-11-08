#!/bin/bash
#SBATCH --job-name=ffc-check
#SBATCH --partition=lab150
#SBATCH --account=lab150-alunos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/%u/frozen-feature-clustering
#SBATCH --output=/home/%u/runs/%x-%j.out
#SBATCH --error=/home/%u/runs/%x-%j.err
# Sanity check of the job environment (GPU, image packages, internet). ~1 min.
# Usage: sbatch cluster/lab150/check.sbatch
set -euo pipefail
export PATH=/usr/bin:/bin
source cluster/lab150/container.sh
in_container --gres=gpu:rtx4090:1 -- bash cluster/lab150/check.sh

# Container side (sourced by the .sh scripts): paths and environment for every job.
# Storage follows the LAB150 manual: data read often -> ~/dados, results you read over SSH -> ~/runs.

export FFC_REPO="${HOME}/frozen-feature-clustering"
export FFC_VENV="${HOME}/venvs/ffc"
export FFC_STORE="${HOME}/dados/ffc"   # raw datasets, model weights, feature cache
export FFC_RUNS="${HOME}/runs/ffc"     # results (runs.jsonl, runs.parquet)

export HF_HOME="${FFC_STORE}/hf"       # Hugging Face weights (DINOv2, CLIP)
export TORCH_HOME="${FFC_STORE}/torch" # torchvision weights (ResNet-50)
export PYTHONUNBUFFERED=1              # logs show up in ~/runs while the job runs

mkdir -p "$FFC_STORE" "$FFC_RUNS"

activate_venv() {
  if [[ ! -x "$FFC_VENV/bin/python" ]]; then
    echo "venv not found at $FFC_VENV. Run: sbatch cluster/lab150/setup_env.sbatch" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$FFC_VENV/bin/activate"
}

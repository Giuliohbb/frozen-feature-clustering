# Host side (sourced by the .sbatch scripts): run a command inside the lab's PyTorch image.
# Image and mounts come from the LAB150 student manual, section 7.2. Do not change IMG.

IMG=/var/lib/lab150-images/pytorch-devtools-2.14.1-cuda12.6-cudnn9-2a0c31112ec3.sqsh
MNT="/etc/slurm/taskprolog.xdg:/etc/slurm/taskprolog.xdg:ro,${HOME}:${HOME}"

# Usage: in_container [srun options...] -- command [args...]
in_container() {
  local srun_opts=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do srun_opts+=("$1"); shift; done
  shift  # drop the "--"
  srun --container-image="$IMG" \
       --container-mounts="$MNT" \
       --container-workdir="$PWD" \
       "${srun_opts[@]+"${srun_opts[@]}"}" \
       bash -lc "$(printf '%q ' "$@")"
}

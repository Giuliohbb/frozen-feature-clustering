# Running on LAB150 (UFG/CEIA Slurm cluster)

LAB150 is a Slurm cluster: you log into an **access machine** to edit code and read results,
and heavy work goes to the queue as **jobs** that land on a lab PC with an RTX 4090.
Nothing in this guide runs Python on the access machine.

## Design

| Concern | Choice |
|---|---|
| PyTorch / CUDA | The lab's PyTorch image (2.14.1, CUDA 12.6), via `srun --container-image`. No torch downloads. |
| Isolation | A venv at `~/venvs/ffc` created **inside the container** with `--system-site-packages`: it reuses the image's torch and installs only what is missing. A constraints file pins torch, torchvision, numpy and pillow to the image versions, so pip can never replace them. |
| Storage | Datasets, model weights and the feature cache in `~/dados/ffc` (read often). Results in `~/runs/ffc` (read from SSH). Job logs in `~/runs/ffc-*.out`. |
| GPU usage | Only feature extraction holds the GPU. The clustering grid is a CPU-only job. |
| Interruptions | Extraction skips cached `(source, encoder)` pairs; the grid skips finished runs. Resubmitting a job continues where it stopped. |

Layout: each step is a `*.sbatch` (resources, runs on the host) that calls a `*.sh` of the same name inside the container. `env.sh` holds the paths, `container.sh` the image and mounts.

## One-time setup

```bash
ssh -p 22150 user_<firstname><lastname>@200.137.197.131
mkdir -p ~/runs ~/dados
git clone https://github.com/Giuliohbb/frozen-feature-clustering.git ~/frozen-feature-clustering
cd ~/frozen-feature-clustering
```

The repo **must** live at `~/frozen-feature-clustering`: every job starts with `cd "$HOME/frozen-feature-clustering"`.

### 1. Check the environment (~1 min, GPU)

```bash
sbatch cluster/lab150/check.sbatch
squeue -u $USER                       # wait until it leaves the queue
cat ~/runs/ffc-check-*.out
```

Shows Python and torch versions, GPU name, which packages the image already has, CPU/memory and whether jobs can reach PyPI, Hugging Face and the dataset hosts.

### 2. Create the venv (~5 min, CPU)

```bash
sbatch cluster/lab150/setup_env.sbatch
cat ~/runs/ffc-setup-*.out            # ends with "setup OK"
```

Run it again after changing dependencies in `pyproject.toml`. It is idempotent.

## Experiments

**One job at a time.** The lab allows one job per user (a GPU Jupyter kernel counts). Submit the next step only after `squeue -u $USER` is empty, and check its log first.

```bash
# Smoke test: CIFAR-10, pixels + DINOv2, 3k images
sbatch cluster/lab150/extract.sbatch configs/smoke.yaml
sbatch cluster/lab150/run.sbatch configs/smoke.yaml
cat ~/runs/ffc-run-*.out              # summary table at the end

# Full grid (1,280 runs)
sbatch --time=03:00:00 cluster/lab150/extract.sbatch configs/grid.yaml
sbatch cluster/lab150/run.sbatch configs/grid.yaml
```

Results land in `~/runs/ffc/<config name>/runs.jsonl` and `runs.parquet`.

## Monitoring

```bash
squeue -u $USER                       # PD = waiting, R = running
tail -f ~/runs/ffc-run-<JOBID>.out    # live log
scancel <JOBID>                       # cancel
df -h $HOME                           # 200 GiB quota
```

The first job that reads a large `~/dados` can sit in `R` for a few minutes while the lab prepares the files. That is expected; do not cancel it.

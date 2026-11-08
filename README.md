# Frozen-Feature Clustering

[![CI](https://github.com/Giuliohbb/frozen-feature-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/Giuliohbb/frozen-feature-clustering/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Representation, dimensionality reduction or algorithm? What drives clustering quality on frozen pretrained features, and can we pick a good configuration without labels?**

> **Status: work in progress.** Final project for *Unsupervised Machine Learning* at the Universidade Federal de Goiás (UFG), Nov–Dec 2026. The pipeline, metrics and tests are in place; experiments and analysis are under way (see [Roadmap](#roadmap)).

## Motivation

Image clustering has moved from learning representations and clusters jointly to clustering the **frozen features** of pretrained models, with k-means on top of those features as the standard baseline [1, 2]. In practice, whoever clusters data makes three choices: the **representation**, whether and how to **reduce dimensionality**, and the **clustering algorithm**. Papers usually report the best combination *as measured with labels*. In a real unsupervised problem there are no labels to make that choice. This project measures how much each choice matters and tests whether label-free (internal) metrics can make it for us.

## Research questions

| | Question | Hypothesis |
|---|---|---|
| **RQ1** | How much of the variation in clustering quality is explained by representation, reduction and algorithm? | Representation explains more ARI variance than the algorithm. |
| **RQ2** | Do internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) rank configurations consistently with external ones (ARI, NMI, ACC)? | Yes within a representation; no across representations, because internal indices depend on the geometry of the space they are computed in. |
| **RQ3** | Does UMAP before clustering help, and for which algorithms? | The ARI gain from UMAP is larger for HDBSCAN than for k-means [3]. |

## Experimental design

A full factorial grid, **1,280 runs** in total:

| Factor | Levels |
|---|---|
| Label set | CIFAR-10 (10), CIFAR-100 (20 superclasses and 100 classes), STL-10 (10). Official test splits. |
| Representation | Raw pixels (32×32, lower bound) · ResNet-50 supervised on ImageNet (reference, saw labels) · DINOv2 ViT-B/14 (self-supervised) · CLIP ViT-B/16 (image-text) |
| Reduction | None · PCA (50) · UMAP (10-d, `n_neighbors` ∈ {15, 50}, spectral init) |
| Algorithm | k-means · GMM (diagonal) · Ward agglomerative · HDBSCAN (infers k) |
| Seeds | 5 per cell |

**Metrics.** External: ARI (primary, chance-corrected), NMI, Hungarian accuracy. Internal: Silhouette, Davies-Bouldin, Calinski-Harabasz, computed in the space where clustering ran. For HDBSCAN, noise is its own group in external metrics and coverage is reported separately.

**Planned analyses.** (A1) ANOVA of ARI with η² per factor; (A2) Kendall's τ between internal and external rankings, plus the *regret* of choosing by an internal metric; (A3) UMAP effect per algorithm and across seeds; (A4) qualitative failure analysis.

The full proposal (in Portuguese) is in [`docs/proposal-pt.md`](docs/proposal-pt.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/Giuliohbb/frozen-feature-clustering.git
cd frozen-feature-clustering

make setup       # core + dev dependencies (CPU only, no torch)
make check       # lint, type check, tests

make setup-gpu   # adds torch, torchvision and transformers for feature extraction
make smoke       # small end-to-end run: CIFAR-10, pixels + DINOv2, 3k images, 1 seed

make extract     # download datasets, cache features for all encoders (GPU)
make run         # run the full grid on cached features (resumable)
uv run ffc report -c configs/grid.yaml   # export runs.parquet and print a summary
```

Running on a Slurm cluster? See [`cluster/lab150/README.md`](cluster/lab150/README.md) for job scripts that run inside a PyTorch container with an isolated venv.

## How it works

```
images ──► encoder (frozen) ──► cache/<source>/features_<encoder>.npy
                                        │  L2-normalize
                                        ▼
                     reduce (none | PCA | UMAP)   ◄── once per (source, encoder, reducer, seed)
                                        │
                                        ▼
                cluster (k-means | GMM | Ward | HDBSCAN) ──► metrics ──► results/<exp>/runs.jsonl
```

- **Two stages, two dependency sets.** Feature extraction needs torch and a GPU; everything after it reads `.npy` files and runs on CPU with scikit-learn and umap-learn. CI tests the second stage end to end on synthetic features.
- **Resumable.** Every run appends one JSON line with its id, metrics, timings and library versions. Re-running skips finished ids, so the grid survives interruptions and can be extended (new seeds, new methods) without recomputation.
- **Config-driven.** The whole grid is a YAML file ([`configs/grid.yaml`](configs/grid.yaml)). Adding an encoder, reducer or clusterer means registering one function.

## Repository layout

```
cluster/lab150/     Slurm job scripts for the UFG/CEIA LAB150 cluster
configs/            experiment grids (full and smoke)
docs/               proposal and, later, the paper-style write-up
src/ffc/
  config.py         YAML loading and grid expansion
  data.py           dataset registry and loaders (test splits)
  features.py       frozen encoders and the extraction stage
  store.py          on-disk feature/label cache
  reduce.py         none | PCA | UMAP
  cluster.py        k-means | GMM | Ward | HDBSCAN
  metrics.py        external and internal metrics
  runner.py         resumable grid execution
  cli.py            `ffc extract | run | report`
tests/              unit and end-to-end tests (CPU, synthetic data)
```

## Reproducibility

- Fixed seeds for every stochastic step (PCA solver, UMAP, k-means, GMM). Seeded UMAP runs single-threaded by design.
- Official test splits only; labels never touch the clustering stage.
- Library versions are recorded in every result row; dependencies are locked in `uv.lock`.
- Datasets, features and results are never committed (see `.gitignore`).

## Roadmap

- [x] Pipeline, configuration grid, metrics, tests and CI
- [ ] Feature extraction for all encoders (RTX 4090)
- [ ] Full grid (1,280 runs)
- [ ] Analyses A1–A4 and figures
- [ ] Paper-style write-up and presentation

## Key references

1. Adaloglou et al. *Exploring the Limits of Deep Image Clustering using Pretrained Models.* BMVC 2023.
2. Gadetsky, Jiang, Brbić. *Let Go of Your Labels with Unsupervised Transfer.* ICML 2024.
3. Herrmann et al. *Enhancing cluster analysis via topological manifold learning.* Data Mining and Knowledge Discovery 38, 2024.
4. Kobak, Linderman. *Initialization is critical for preserving global data structure in both t-SNE and UMAP.* Nature Biotechnology 39, 2021.
5. Lause, Berens, Kobak. *The art of seeing the elephant in the room: 2D embeddings of single-cell data do make sense.* PLoS Computational Biology 20, 2024.

Full reference list in the [proposal](docs/proposal-pt.md#referências).

## License

[MIT](LICENSE) © 2026 Giulio Henrique Borges Basso

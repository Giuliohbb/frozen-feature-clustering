"""A synthetic results grid with known structure, for testing the analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ffc.config import ExperimentConfig, MethodSpec, Paths

ENCODER_QUALITY = {"pixels": 0.05, "resnet50": 0.45, "dinov2_vitb14": 0.75, "clip_vitb16": 0.6}
# Internal-metric offset per representation: unrelated to ARI, so comparing silhouettes across
# representations is misleading while comparing within one is informative (hypothesis H2).
SILHOUETTE_OFFSET = {"pixels": 0.55, "resnet50": 0.05, "dinov2_vitb14": 0.0, "clip_vitb16": 0.3}


def grid_config(results_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=["cifar10", "stl10"],
        encoders=list(ENCODER_QUALITY),
        reducers=[
            MethodSpec("none"),
            MethodSpec.from_obj({"name": "pca", "n_components": 50}),
            MethodSpec.from_obj({"name": "umap", "n_components": 10, "n_neighbors": 15}),
        ],
        clusterers=[
            MethodSpec("kmeans"),
            MethodSpec("ward"),
            MethodSpec.from_obj({"name": "hdbscan", "min_cluster_size": 50}),
        ],
        seeds=[0, 1, 2],
        paths=Paths(results_dir=results_dir),
    )


def true_ari(encoder: str, reducer: str, clusterer: str) -> float:
    q = ENCODER_QUALITY[encoder]
    algo = {"kmeans": 0.0, "ward": -0.03, "hdbscan": -0.5}[clusterer]
    red = {"none": 0.0, "pca": 0.04, "umap": 0.12}[reducer]
    if clusterer == "hdbscan" and reducer == "umap":
        red += 0.45  # UMAP rescues density-based clustering
    return float(np.clip(q + algo + red, 0.0, 1.0))


def write_grid(results_dir: Path, drop: int = 0, seed: int = 0) -> ExperimentConfig:
    """Write a results file for the whole grid (minus the last ``drop`` runs)."""
    cfg = grid_config(results_dir)
    rng = np.random.default_rng(seed)
    lines = []
    for run in cfg.iter_runs():
        deterministic = run.reducer.name == "none" and run.clusterer.name in {"ward", "hdbscan"}
        noise = 0.0 if deterministic else rng.normal(0, 0.01)
        ari = float(
            np.clip(true_ari(run.encoder, run.reducer.name, run.clusterer.name) + noise, 0, 1)
        )
        sil = 0.3 * ari + SILHOUETTE_OFFSET[run.encoder] + rng.normal(0, 0.005)
        lines.append(json.dumps({
            "run_id": run.run_id, "dataset": run.dataset, "encoder": run.encoder,
            "reducer": run.reducer.id, "clusterer": run.clusterer.id, "seed": run.seed,
            "status": "ok", "timestamp": f"2026-11-16T00:00:{len(lines) % 60:02d}",
            "ari": ari, "nmi": ari, "acc": ari, "coverage": 1.0, "n_clusters_found": 10.0,
            "silhouette": sil, "davies_bouldin": 2.0 - sil, "calinski_harabasz": 100 * sil,
        }))  # fmt: skip
    results_dir.mkdir(parents=True, exist_ok=True)
    keep = lines[: len(lines) - drop] if drop else lines
    (results_dir / "runs.jsonl").write_text("\n".join(keep) + "\n")
    return cfg

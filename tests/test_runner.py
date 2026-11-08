"""End-to-end test of the run stage on a synthetic feature cache (no torch needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

from ffc.config import ExperimentConfig, MethodSpec, Paths
from ffc.runner import load_results, read_done, run_grid
from ffc.store import FeatureStore


def _make_cache(cache_dir: Path) -> None:
    """Fake cifar10 source: 10 well-separated blobs, one 'good' and one 'noise' encoder."""
    X, y = make_blobs(n_samples=500, centers=10, n_features=16, cluster_std=0.3, random_state=0)
    store = FeatureStore(cache_dir)
    store.save_labels("cifar10", "fine", y)
    store.save_features("cifar10", "good", X)
    store.save_features("cifar10", "noise", np.random.default_rng(0).normal(size=X.shape))


def _config(tmp_path: Path, seeds: list[int]) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=["cifar10"],
        encoders=["good", "noise"],
        reducers=[MethodSpec("none"), MethodSpec.from_obj({"name": "pca", "n_components": 8})],
        clusterers=[MethodSpec("kmeans"), MethodSpec("ward")],
        seeds=seeds,
        paths=Paths(cache_dir=tmp_path / "cache", results_dir=tmp_path / "results"),
    )


def test_run_grid_end_to_end_and_resume(tmp_path: Path) -> None:
    _make_cache(tmp_path / "cache")

    cfg = _config(tmp_path, seeds=[0])
    assert run_grid(cfg) == cfg.n_runs() == 8
    assert run_grid(cfg) == 0  # everything cached -> nothing re-run

    cfg2 = _config(tmp_path, seeds=[0, 1])  # extend the grid: only new seed runs
    assert run_grid(cfg2) == 8
    assert len(read_done(cfg2.paths.results_dir)) == 16

    df = load_results(cfg2.paths.results_dir)
    expected_cols = {"run_id", "ari", "nmi", "acc", "silhouette", "coverage", "time_cluster_s"}
    assert expected_cols <= set(df.columns)

    mean_ari = df.groupby("encoder")["ari"].mean()
    assert mean_ari["good"] > 0.9
    assert mean_ari["noise"] < 0.1

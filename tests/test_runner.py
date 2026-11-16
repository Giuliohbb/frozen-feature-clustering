"""End-to-end test of the run stage on a synthetic feature cache (no torch needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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


def test_failing_cell_is_recorded_and_grid_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ffc.runner as runner_mod
    from ffc.cluster import cluster as real_cluster

    _make_cache(tmp_path / "cache")
    cfg = _config(tmp_path, seeds=[0])
    fail = {"on": True}

    def maybe_failing(X, spec, n_clusters, seed):  # type: ignore[no-untyped-def]
        if spec.name == "ward" and fail["on"]:
            raise ValueError("boom")
        return real_cluster(X, spec, n_clusters, seed)

    monkeypatch.setattr(runner_mod, "cluster", maybe_failing)
    assert run_grid(cfg) == 8  # every cell is attempted, none aborts the job

    df = load_results(cfg.paths.results_dir)
    errors = df[df["status"] == "error"]
    assert len(errors) == 4 and set(errors["clusterer"]) == {"ward"}
    assert errors["error"].str.contains("ValueError: boom").all()
    assert errors["ari"].isna().all()
    assert (df[df["status"] == "ok"]["ari"].notna()).all()

    assert run_grid(cfg) == 0  # errors count as done by default...
    fail["on"] = False
    assert run_grid(cfg, retry_errors=True) == 4  # ...and are re-run on request

    df = load_results(cfg.paths.results_dir)
    assert len(df) == 8  # retried runs replace their error records
    assert (df["status"] == "ok").all()


def test_load_results_accepts_records_without_status(tmp_path: Path) -> None:
    """Results written before the status field existed (the first 901 grid runs)."""
    import json

    (tmp_path / "runs.jsonl").write_text(
        json.dumps({"run_id": "a", "ari": 0.5})
        + "\n"
        + json.dumps({"run_id": "b", "ari": 0.1})
        + "\n"
    )
    df = load_results(tmp_path)
    assert list(df["status"]) == ["ok", "ok"]
    assert read_done(tmp_path, include_errors=False) == {"a", "b"}


def test_corrupted_lines_are_skipped_or_recovered(tmp_path: Path) -> None:
    """runs.jsonl damaged by a job killed mid-write (seen on the LAB150 NFS home)."""
    import json

    good = json.dumps({"run_id": "a", "ari": 0.5})
    nul_prefixed = "\x00" * 64 + json.dumps({"run_id": "b", "ari": 0.2})
    torn = '{"run_id": "c", "ari": 0.'
    (tmp_path / "runs.jsonl").write_text(
        "\n".join([good, "\x00\x00\x00", nul_prefixed, torn, "not json", "[1, 2]"]) + "\n"
    )
    assert read_done(tmp_path) == {"a", "b"}  # "c" is torn, so it will simply run again
    assert set(load_results(tmp_path)["run_id"]) == {"a", "b"}


def test_grid_resumes_after_a_torn_last_line(tmp_path: Path) -> None:
    _make_cache(tmp_path / "cache")
    cfg = _config(tmp_path, seeds=[0])
    assert run_grid(cfg) == 8
    runs = cfg.paths.results_dir / "runs.jsonl"
    lines = runs.read_text().splitlines()
    runs.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:30])  # tear the last record
    assert run_grid(cfg) == 1  # only the torn cell is re-run
    assert len(load_results(cfg.paths.results_dir)) == 8

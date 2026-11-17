"""Grid execution: reduce once per (source, encoder, reducer, seed), then cluster and evaluate.

Results are appended to a per-process JSONL file in ``<results_dir>`` (see ``ffc.results``),
one JSON object per run, flushed immediately. Re-running skips run ids already recorded, so an
interrupted grid resumes.

A run that raises is recorded with ``status="error"`` (metrics NaN) and the grid continues:
one failing cell is a result to report, not a reason to lose the rest of the job. Errored
runs are skipped on resume unless ``retry_errors=True``.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ffc import __version__
from ffc.cluster import cluster
from ffc.config import ExperimentConfig, RunSpec
from ffc.data import DATASETS
from ffc.metrics import external_metrics, internal_metrics, partition_stats
from ffc.reduce import reduce
from ffc.results import (
    ensure_trailing_newline,
    load_results,
    provenance,
    read_done,
    writer_path,
)
from ffc.store import FeatureStore

__all__ = ["load_results", "read_done", "run_grid", "summarize"]

log = logging.getLogger(__name__)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def subsample_indices(n: int, max_samples: int | None, seed: int = 0) -> np.ndarray:
    """Fixed random subset shared by every run, so all methods see the same points."""
    if max_samples is None or max_samples >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n)[:max_samples])


METRIC_KEYS = (
    "n_clusters_found", "coverage", "ari", "nmi", "acc",
    "silhouette", "davies_bouldin", "calinski_harabasz",
)  # fmt: skip


def _environment() -> dict[str, str]:
    import umap

    return {
        "ffc_version": __version__,
        "sklearn_version": sklearn.__version__,
        "umap_version": umap.__version__,
        "python_version": platform.python_version(),
    }


def run_grid(cfg: ExperimentConfig, retry_errors: bool = False) -> int:
    """Execute all pending runs of the grid. Returns the number of runs executed."""
    store = FeatureStore(cfg.paths.cache_dir)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(results_dir, include_errors=not retry_errors)
    pending = {r.run_id for r in cfg.iter_runs()} - done
    log.info("%d runs in grid, %d already done, %d pending", cfg.n_runs(), len(done), len(pending))
    if not pending:
        return 0

    env = {**_environment(), **provenance()}
    n_executed = 0
    out_path = writer_path(results_dir)
    ensure_trailing_newline(out_path)  # same process calling run_grid again
    log.info("writing results to %s", out_path)
    with open(out_path, "a") as out:
        for source in cfg.sources:
            datasets = [d for d in cfg.datasets if DATASETS[d].source == source]
            for encoder in cfg.encoders:
                X_full = store.load_features(source, encoder)
                idx = subsample_indices(len(X_full), cfg.max_samples)
                X = X_full[idx]
                if cfg.normalize == "l2":
                    X = l2_normalize(X)
                labels = {
                    d: store.load_labels(source, DATASETS[d].label_key)[idx] for d in datasets
                }

                for reducer in cfg.reducers:
                    for seed in cfg.seeds:
                        group = [
                            RunSpec(d, encoder, reducer, c, seed)
                            for d in datasets
                            for c in cfg.clusterers
                        ]
                        todo = [r for r in group if r.run_id in pending]
                        if not todo:
                            continue

                        t0 = time.perf_counter()
                        X_red = reduce(X, reducer, seed)
                        t_reduce = time.perf_counter() - t0
                        log.info(
                            "[%s|%s|%s|seed=%d] reduced %s -> %s in %.1fs",
                            source, encoder, reducer.id, seed, X.shape, X_red.shape, t_reduce,
                        )  # fmt: skip

                        for run in todo:
                            record = _evaluate_run(run, X_red, labels[run.dataset], cfg)
                            record.update(
                                source=source,
                                n_samples=len(X),
                                dim_in=X.shape[1],
                                dim_reduced=X_red.shape[1],
                                time_reduce_s=t_reduce,
                                **env,
                            )
                            out.write(json.dumps(record) + "\n")
                            out.flush()
                            n_executed += 1
                            if record["status"] == "error":
                                log.warning("  %-28s FAILED: %s", run.clusterer.id, record["error"])
                                continue
                            log.info(
                                "  %-28s ARI=%.3f NMI=%.3f ACC=%.3f (%.1fs)",
                                run.clusterer.id, record["ari"], record["nmi"], record["acc"],
                                record["time_cluster_s"],
                            )  # fmt: skip
    return n_executed


def _evaluate_run(
    run: RunSpec, X_red: np.ndarray, y_true: np.ndarray, cfg: ExperimentConfig
) -> dict[str, Any]:
    k = DATASETS[run.dataset].n_classes
    record: dict[str, Any] = {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "encoder": run.encoder,
        "reducer": run.reducer.id,
        "clusterer": run.clusterer.id,
        "seed": run.seed,
        "k_true": k,
    }
    t0 = time.perf_counter()
    try:
        result = cluster(X_red, run.clusterer, n_clusters=k, seed=run.seed)
        record["time_cluster_s"] = time.perf_counter() - t0
        record.update(result.info)
        record.update(partition_stats(result.labels))
        record.update(external_metrics(y_true, result.labels))
        record.update(internal_metrics(X_red, result.labels, cfg.internal_metrics_sample, run.seed))
        record["status"] = "ok"
    except Exception as err:  # one failing cell must not kill the whole grid
        record["time_cluster_s"] = time.perf_counter() - t0
        record.update(dict.fromkeys(METRIC_KEYS, math.nan))
        record["status"] = "error"
        record["error"] = f"{type(err).__name__}: {err}"[:500]
    record["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record


def summarize(
    df: pd.DataFrame, metric: str = "ari", by: Iterable[str] = ("encoder",)
) -> pd.DataFrame:
    """Mean of a metric over seeds (successful runs only), pivoted for the terminal."""
    cols = ["dataset", *by]
    ok = df[df["status"] == "ok"] if "status" in df else df
    return ok.groupby([*cols, "clusterer"])[metric].mean().unstack("clusterer").round(3)

from __future__ import annotations

import math

import numpy as np
import pytest

from ffc.metrics import clustering_accuracy, external_metrics, internal_metrics, partition_stats


def test_accuracy_is_invariant_to_label_permutation() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])
    assert clustering_accuracy(y_true, y_pred) == 1.0


def test_accuracy_counts_unmatched_clusters_as_errors() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 2, 1, 1, 1])  # extra cluster gets no class
    assert clustering_accuracy(y_true, y_pred) == pytest.approx(5 / 6)


def test_external_metrics_perfect_partition() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    m = external_metrics(y, (y + 1) % 3)
    assert m == {"ari": 1.0, "nmi": 1.0, "acc": 1.0}


def test_external_metrics_random_partition_near_zero_ari() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 5000)
    y_pred = rng.integers(0, 10, 5000)
    assert abs(external_metrics(y_true, y_pred)["ari"]) < 0.01


def test_partition_stats_excludes_noise() -> None:
    stats = partition_stats(np.array([-1, -1, 0, 0, 1, 1, 1, 2]))
    assert stats == {"n_clusters_found": 3.0, "coverage": 0.75}


def test_internal_metrics_prefer_true_partition(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = blobs
    good = internal_metrics(X, y)
    bad = internal_metrics(X, np.random.default_rng(0).permutation(y))
    assert good["silhouette"] > bad["silhouette"]
    assert good["davies_bouldin"] < bad["davies_bouldin"]
    assert good["calinski_harabasz"] > bad["calinski_harabasz"]


def test_internal_metrics_degenerate_partition_is_nan(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    single = np.zeros(len(X), dtype=int)
    single[:10] = -1  # noise plus one cluster
    assert all(math.isnan(v) for v in internal_metrics(X, single).values())

"""External (label-based) and internal (label-free) clustering metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm
from sklearn.metrics.cluster import contingency_matrix

from ffc.cluster import NOISE


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy under the best one-to-one mapping between clusters and classes (Hungarian).

    Works when the number of clusters differs from the number of classes: unmatched clusters
    (and HDBSCAN noise) count as errors.
    """
    C = contingency_matrix(y_true, y_pred)  # classes x clusters
    rows, cols = linear_sum_assignment(C, maximize=True)
    return float(C[rows, cols].sum() / len(y_true))


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """ARI (primary), NMI and Hungarian accuracy. Noise is kept as its own group."""
    return {
        "ari": float(skm.adjusted_rand_score(y_true, y_pred)),
        "nmi": float(skm.normalized_mutual_info_score(y_true, y_pred)),
        "acc": clustering_accuracy(y_true, y_pred),
    }


def partition_stats(y_pred: np.ndarray) -> dict[str, float]:
    """Number of clusters found (noise excluded) and coverage (fraction not labeled noise)."""
    assigned = y_pred != NOISE
    return {
        "n_clusters_found": float(len(np.unique(y_pred[assigned]))),
        "coverage": float(assigned.mean()),
    }


def internal_metrics(
    X: np.ndarray,
    y_pred: np.ndarray,
    sample_size: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz in the space where clustering ran.

    Noise points are excluded. Returns NaN when the partition is degenerate
    (fewer than 2 clusters, or as many clusters as points).
    """
    nan = {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    mask = y_pred != NOISE
    Xc, yc = X[mask], y_pred[mask]
    k = len(np.unique(yc))
    if k < 2 or k >= len(yc):
        return nan

    sil_n = None if sample_size is None or sample_size >= len(yc) else sample_size
    return {
        "silhouette": float(skm.silhouette_score(Xc, yc, sample_size=sil_n, random_state=seed)),
        "davies_bouldin": float(skm.davies_bouldin_score(Xc, yc)),
        "calinski_harabasz": float(skm.calinski_harabasz_score(Xc, yc)),
    }

"""Clustering algorithms. All return integer labels; HDBSCAN marks noise as -1."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from ffc.config import MethodSpec

CLUSTERERS = ("kmeans", "gmm", "ward", "hdbscan")
NOISE = -1


def cluster(X: np.ndarray, spec: MethodSpec, n_clusters: int, seed: int) -> np.ndarray:
    """Cluster ``X``. ``n_clusters`` is ignored by HDBSCAN, which infers it from density."""
    kw = spec.kwargs

    if spec.name == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=int(kw.get("n_init", 10)),
            random_state=seed,
        )
        return model.fit_predict(X)

    if spec.name == "gmm":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=str(kw.get("covariance_type", "diag")),
            reg_covar=float(kw.get("reg_covar", 1e-6)),
            random_state=seed,
        )
        return gmm.fit_predict(X)

    if spec.name == "ward":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)

    if spec.name == "hdbscan":
        min_samples = kw.get("min_samples")
        return HDBSCAN(
            min_cluster_size=int(kw.get("min_cluster_size", 50)),
            min_samples=None if min_samples is None else int(min_samples),
            copy=True,
        ).fit_predict(X)

    raise ValueError(f"Unknown clusterer {spec.name!r}; available: {CLUSTERERS}")

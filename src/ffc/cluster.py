"""Clustering algorithms. All return integer labels; HDBSCAN marks noise as -1."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from ffc.config import MethodSpec

CLUSTERERS = ("kmeans", "gmm", "ward", "hdbscan")
NOISE = -1

# GMM covariance regularization tried in order when a fit collapses (see _fit_gmm).
GMM_REG_COVAR_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


class ClusterResult(NamedTuple):
    labels: np.ndarray
    info: dict[str, Any]  # method-specific details worth recording (e.g. GMM reg_covar used)


def _fit_gmm(
    X: np.ndarray, n_components: int, covariance_type: str, base_reg: float, seed: int
) -> ClusterResult:
    """Fit a GMM, increasing ``reg_covar`` only if the fit collapses.

    With many components on tightly packed data (e.g. 100 components on a UMAP embedding with
    min_dist=0), some components can end up on (near-)identical points and their variance
    underflows. We fit in float64 and, if sklearn still reports an ill-defined covariance,
    retry with the next larger ``reg_covar``. The value actually used is returned so the
    analysis can flag these runs.
    """
    X64 = np.asarray(X, dtype=np.float64)
    ladder = [base_reg, *(r for r in GMM_REG_COVAR_LADDER if r > base_reg)]
    last_error: ValueError | None = None
    for reg in ladder:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg,
            random_state=seed,
        )
        try:
            labels = gmm.fit_predict(X64)
        except ValueError as err:
            if "ill-defined empirical covariance" not in str(err):
                raise
            last_error = err
            continue
        return ClusterResult(labels, {"gmm_reg_covar": reg})
    assert last_error is not None
    raise last_error


def cluster(X: np.ndarray, spec: MethodSpec, n_clusters: int, seed: int) -> ClusterResult:
    """Cluster ``X``. ``n_clusters`` is ignored by HDBSCAN, which infers it from density."""
    kw = spec.kwargs

    if spec.name == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=int(kw.get("n_init", 10)),
            random_state=seed,
        )
        return ClusterResult(model.fit_predict(X), {})

    if spec.name == "gmm":
        return _fit_gmm(
            X,
            n_components=n_clusters,
            covariance_type=str(kw.get("covariance_type", "diag")),
            base_reg=float(kw.get("reg_covar", 1e-6)),
            seed=seed,
        )

    if spec.name == "ward":
        ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return ClusterResult(ward.fit_predict(X), {})

    if spec.name == "hdbscan":
        min_samples = kw.get("min_samples")
        hdb = HDBSCAN(
            min_cluster_size=int(kw.get("min_cluster_size", 50)),
            min_samples=None if min_samples is None else int(min_samples),
            copy=True,
        )
        return ClusterResult(hdb.fit_predict(X), {})

    raise ValueError(f"Unknown clusterer {spec.name!r}; available: {CLUSTERERS}")

"""Dimensionality reduction applied before clustering."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from sklearn.decomposition import PCA

from ffc.config import MethodSpec

REDUCERS = ("none", "pca", "umap")


def reduce(X: np.ndarray, spec: MethodSpec, seed: int) -> np.ndarray:
    """Return a reduced copy of ``X`` (or ``X`` itself for ``none``)."""
    kw = spec.kwargs
    if spec.name == "none":
        return X

    if spec.name == "pca":
        n_components = min(int(kw.get("n_components", 50)), X.shape[1], X.shape[0])
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)

    if spec.name == "umap":
        import umap  # slow import, keep local

        params: dict[str, object] = {
            "n_components": int(kw.get("n_components", 10)),
            "n_neighbors": int(kw.get("n_neighbors", 15)),
            # min_dist=0 packs points densely, which is the recommended setting for clustering
            "min_dist": float(kw.get("min_dist", 0.0)),
            "metric": str(kw.get("metric", "euclidean")),
            "init": str(kw.get("init", "spectral")),
            "random_state": seed,
        }
        # umap-learn 0.6 will change its default layout pipeline. Pin the classic (0.5.x)
        # pipeline, which is what the 0.6.0 default currently runs, so results do not change
        # under a library upgrade. Older versions have no such parameter and already use it.
        if "compatibility_layout" in inspect.signature(umap.UMAP.__init__).parameters:
            params["compatibility_layout"] = True
        model = umap.UMAP(**params)
        with warnings.catch_warnings():
            # Seeding UMAP forces single-threaded execution; we accept that for reproducibility.
            warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
            return np.asarray(model.fit_transform(X), dtype=np.float32)

    raise ValueError(f"Unknown reducer {spec.name!r}; available: {REDUCERS}")

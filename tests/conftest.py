from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated Gaussian blobs: any sane method should recover them."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=8, cluster_std=0.5, random_state=0)
    return X.astype(np.float32), y.astype(np.int64)

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from ffc.cluster import cluster
from ffc.config import MethodSpec
from ffc.features import downsample, encode_pixels
from ffc.reduce import reduce


@pytest.mark.parametrize(
    "spec",
    [
        MethodSpec("kmeans"),
        MethodSpec("gmm"),
        MethodSpec("ward"),
        MethodSpec.from_obj({"name": "hdbscan", "min_cluster_size": 20}),
    ],
    ids=lambda s: s.id,
)
def test_clusterers_recover_separated_blobs(
    spec: MethodSpec, blobs: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = blobs
    labels = cluster(X, spec, n_clusters=3, seed=0)
    assert labels.shape == y.shape
    assert adjusted_rand_score(y, labels) > 0.9


def test_kmeans_is_deterministic_given_seed(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    a = cluster(X, MethodSpec("kmeans"), 3, seed=1)
    b = cluster(X, MethodSpec("kmeans"), 3, seed=1)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("spec", "expected_dim"),
    [
        (MethodSpec("none"), 8),
        (MethodSpec.from_obj({"name": "pca", "n_components": 5}), 5),
        (MethodSpec.from_obj({"name": "umap", "n_components": 2, "n_neighbors": 10}), 2),
    ],
    ids=["none", "pca", "umap"],
)
def test_reducers_output_shape(
    spec: MethodSpec, expected_dim: int, blobs: tuple[np.ndarray, np.ndarray]
) -> None:
    X, _ = blobs
    assert reduce(X, spec, seed=0).shape == (len(X), expected_dim)


def test_unknown_methods_raise(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    with pytest.raises(ValueError, match="Unknown reducer"):
        reduce(X, MethodSpec("tsne"), seed=0)
    with pytest.raises(ValueError, match="Unknown clusterer"):
        cluster(X, MethodSpec("dbscan"), 3, seed=0)


def test_downsample_block_average() -> None:
    img = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    img[:, :2, :2, :] = 255  # one 2x2 block -> one bright output pixel
    out = downsample(img, 32)
    assert out.shape == (1, 32, 32, 3)
    assert out[0, 0, 0, 0] == 255.0
    assert out[0, 1, 1, 0] == 0.0


def test_encode_pixels_shape_and_range() -> None:
    images = np.random.default_rng(0).integers(0, 256, (4, 96, 96, 3), dtype=np.uint8)
    X = encode_pixels(images)
    assert X.shape == (4, 32 * 32 * 3)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0

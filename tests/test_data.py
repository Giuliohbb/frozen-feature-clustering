from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIL = pytest.importorskip("PIL.Image")

from ffc import data  # noqa: E402


def _write_hf_image_parquet(path: Path, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Mimic the Hugging Face layout: image column = {"bytes": PNG, "path": None}."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, n)
    cells = []
    for img in images:
        buf = io.BytesIO()
        PIL.fromarray(img).save(buf, format="PNG")
        cells.append({"bytes": buf.getvalue(), "path": None})
    pd.DataFrame({"image": cells, "label": labels}).to_parquet(path)
    return images, labels


def test_read_image_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "test.parquet"
    images, labels = _write_hf_image_parquet(path)
    got_images, got_labels = data.read_image_parquet(path)
    assert got_images.shape == (6, 96, 96, 3)
    assert got_images.dtype == np.uint8
    np.testing.assert_array_equal(got_images, images)  # PNG is lossless
    np.testing.assert_array_equal(got_labels, labels)


def test_load_source_stl10_uses_pinned_hf_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "test.parquet"
    _, labels = _write_hf_image_parquet(path)
    monkeypatch.setattr(data, "_download_stl10_test", lambda: path)
    images, label_sets = data.load_source("stl10", tmp_path / "data")
    assert images.shape == (6, 96, 96, 3)
    np.testing.assert_array_equal(label_sets["fine"], labels)
    assert len(data.STL10_HF_REVISION) == 40  # pinned to a commit, not a branch

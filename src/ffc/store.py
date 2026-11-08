"""On-disk cache of extracted features and labels.

Layout::

    cache/<source>/features_<encoder>.npy   float32 (N, D)
    cache/<source>/labels_<label_key>.npy   int64   (N,)

The clustering stage reads only from here, so it never needs torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)

    def features_path(self, source: str, encoder: str) -> Path:
        return self.root / source / f"features_{encoder}.npy"

    def labels_path(self, source: str, label_key: str) -> Path:
        return self.root / source / f"labels_{label_key}.npy"

    def has_features(self, source: str, encoder: str) -> bool:
        return self.features_path(source, encoder).exists()

    def save_features(self, source: str, encoder: str, X: np.ndarray) -> Path:
        path = self.features_path(source, encoder)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X.astype(np.float32, copy=False))
        return path

    def save_labels(self, source: str, label_key: str, y: np.ndarray) -> Path:
        path = self.labels_path(source, label_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, y.astype(np.int64, copy=False))
        return path

    def load_features(self, source: str, encoder: str) -> np.ndarray:
        path = self.features_path(source, encoder)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `ffc extract` first.")
        return np.load(path)

    def load_labels(self, source: str, label_key: str) -> np.ndarray:
        path = self.labels_path(source, label_key)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `ffc extract` first.")
        return np.load(path)

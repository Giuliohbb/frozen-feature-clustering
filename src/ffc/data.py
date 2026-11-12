"""Dataset registry and loaders.

Each *dataset* is a label set over an image *source*. CIFAR-100 is one source evaluated with two
label sets (20 superclasses and 100 classes), so features are extracted once per source.
We use the official test split of every source: equal n across methods and tractable O(n^2)
agglomerative clustering. Labels are used only for evaluation.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    label_key: str
    n_classes: int


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", "cifar10", "fine", 10),
    "cifar100_coarse": DatasetSpec("cifar100_coarse", "cifar100", "coarse", 20),
    "cifar100_fine": DatasetSpec("cifar100_fine", "cifar100", "fine", 100),
    "stl10": DatasetSpec("stl10", "stl10", "fine", 10),
}

SOURCES: tuple[str, ...] = ("cifar10", "cifar100", "stl10")

# STL-10's official host (ai.stanford.edu) is often unreachable, including from the LAB150 cluster.
# We read the same official test split (8,000 images, same class order) from a Hugging Face
# mirror, pinned to a commit for reproducibility.
STL10_HF_REPO = "tanganke/stl10"
STL10_HF_FILE = "data/test-00000-of-00001.parquet"
STL10_HF_REVISION = "49ae7f94508f7feae62baf836db284306eab0b0f"


def read_image_parquet(
    path: str | Path, image_col: str = "image", label_col: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a Hugging Face image parquet (image = {bytes, path}) into NHWC uint8 + labels."""
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(path, columns=[image_col, label_col])
    images = np.stack(
        [np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB")) for cell in df[image_col]]
    )
    return images.astype(np.uint8), df[label_col].to_numpy(dtype=np.int64)


def _download_stl10_test() -> Path:
    from huggingface_hub import hf_hub_download  # cache location follows HF_HOME

    return Path(
        hf_hub_download(
            repo_id=STL10_HF_REPO,
            filename=STL10_HF_FILE,
            repo_type="dataset",
            revision=STL10_HF_REVISION,
        )
    )


def load_source(source: str, data_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Download (if needed) and load the test split of an image source.

    Returns:
        images: uint8 array of shape (N, H, W, 3).
        labels: mapping from label key ("fine", "coarse") to int64 arrays of shape (N,).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if source == "stl10":
        stl_images, stl_labels = read_image_parquet(_download_stl10_test())
        return stl_images, {"fine": stl_labels}

    from torchvision import datasets as tvd  # heavy, optional dependency

    if source == "cifar10":
        ds10 = tvd.CIFAR10(root=data_dir, train=False, download=True)
        return np.asarray(ds10.data), {"fine": np.asarray(ds10.targets, dtype=np.int64)}

    if source == "cifar100":
        ds100 = tvd.CIFAR100(root=data_dir, train=False, download=True)
        # torchvision exposes only fine labels; superclasses live in the original pickle.
        with open(data_dir / "cifar-100-python" / "test", "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        labels = {
            "fine": np.asarray(ds100.targets, dtype=np.int64),
            "coarse": np.asarray(raw["coarse_labels"], dtype=np.int64),
        }
        return np.asarray(ds100.data), labels

    raise ValueError(f"Unknown source {source!r}; available: {SOURCES}")

"""Feature extractors (frozen encoders) and the extraction stage.

Encoders:
    pixels          raw pixels downsampled to 32x32, flattened (lower bound, no learning)
    resnet50        torchvision ResNet-50 trained *with labels* on ImageNet (reference)
    dinov2_vitb14   DINOv2 ViT-B/14, self-supervised (CLS token after final layer norm)
    clip_vitb16     CLIP ViT-B/16, image-text contrastive (projected image embedding)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ffc.data import load_source
from ffc.store import FeatureStore

if TYPE_CHECKING:
    from ffc.config import ExperimentConfig

log = logging.getLogger(__name__)

Encoder = Callable[[np.ndarray, str, int], np.ndarray]


def downsample(images: np.ndarray, size: int = 32) -> np.ndarray:
    """Antialiased downsampling by block averaging. Requires H and W divisible by ``size``."""
    n, h, w, c = images.shape
    if h % size or w % size:
        raise ValueError(f"Image size {h}x{w} is not a multiple of {size}")
    fh, fw = h // size, w // size
    blocks = images.reshape(n, size, fh, size, fw, c).astype(np.float32)
    return blocks.mean(axis=(2, 4))


def encode_pixels(images: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    del device, batch_size  # unused: no model involved
    x = downsample(images, 32) / 255.0
    return x.reshape(len(x), -1).astype(np.float32)


def _batches(images: np.ndarray, batch_size: int) -> Any:
    from tqdm import tqdm

    for i in tqdm(range(0, len(images), batch_size), unit="batch", leave=False):
        yield images[i : i + batch_size]


def _resolve_device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU (this will be slow)")
        return "cpu"
    return device


def encode_resnet50(images: np.ndarray, device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    device = _resolve_device(device)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()

    out = []
    with torch.inference_mode():
        for batch in _batches(images, batch_size):
            x = torch.from_numpy(batch).permute(0, 3, 1, 2)
            x = preprocess(x).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def _encode_hf(
    images: np.ndarray, device: str, batch_size: int, model_id: str, kind: str
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel

    device = _resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model: Any = (CLIPModel if kind == "clip" else AutoModel).from_pretrained(model_id)
    model.eval().to(device)

    out = []
    with torch.inference_mode():
        for batch in _batches(images, batch_size):
            inputs = processor(images=list(batch), return_tensors="pt").to(device)
            if kind == "clip":
                emb = model.get_image_features(pixel_values=inputs["pixel_values"])
                if not isinstance(emb, torch.Tensor):  # newer transformers return an output object
                    emb = emb.image_embeds if hasattr(emb, "image_embeds") else emb.pooler_output
            else:
                emb = model(pixel_values=inputs["pixel_values"]).pooler_output
            out.append(emb.float().cpu().numpy())
    return np.concatenate(out)


def encode_dinov2(images: np.ndarray, device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    return _encode_hf(images, device, batch_size, "facebook/dinov2-base", kind="dinov2")


def encode_clip(images: np.ndarray, device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    return _encode_hf(images, device, batch_size, "openai/clip-vit-base-patch16", kind="clip")


ENCODERS: dict[str, Encoder] = {
    "pixels": encode_pixels,
    "resnet50": encode_resnet50,
    "dinov2_vitb14": encode_dinov2,
    "clip_vitb16": encode_clip,
}


def extract_all(cfg: ExperimentConfig, overwrite: bool = False) -> None:
    """Cache labels and features for every (source, encoder) pair the config needs."""
    store = FeatureStore(cfg.paths.cache_dir)
    for encoder in cfg.encoders:
        if encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder {encoder!r}; available: {sorted(ENCODERS)}")

    for source in cfg.sources:
        pending = [e for e in cfg.encoders if overwrite or not store.has_features(source, e)]
        if not pending:
            log.info("[%s] all features cached, skipping", source)
            continue
        images, labels = load_source(source, cfg.paths.data_dir)
        for key, y in labels.items():
            store.save_labels(source, key, y)
        for encoder in pending:
            log.info("[%s] extracting %s on %d images", source, encoder, len(images))
            X = ENCODERS[encoder](images, cfg.device, cfg.batch_size)
            path = store.save_features(source, encoder, X)
            log.info("[%s] saved %s %s", source, path, X.shape)

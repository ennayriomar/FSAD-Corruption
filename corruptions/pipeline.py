"""Corrupt at native resolution, then resize to the backbone input (same steps as the clean loader)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from corruptions.operators import apply
from datasets.mvtec import Record, load_image


def read_native(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode not in ("L", "RGB"):
            raise ValueError(f"Only 8-bit L/RGB sources are supported, received {source.mode}")
        return np.asarray(source, dtype=np.uint8).copy()


def model_input(native: np.ndarray, size: int) -> np.ndarray:
    """RGB conversion, Pillow bilinear resize, float32 CHW / 255."""
    if native.ndim == 3 and native.shape[2] == 1:
        native = native[..., 0]
    image = Image.fromarray(native).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0))


def clean_input(record: Record, size: int) -> np.ndarray:
    return load_image(record.path, size=size)


def corrupted_input(record: Record, rng_id: str, name: str, severity: int, size: int) -> np.ndarray:
    """`rng_id` selects the corruption realisation: record.image_id for queries, train_rng_id(record) for
    training images. Ground-truth masks are never corrupted."""
    return model_input(apply(read_native(record.path), name, severity, category=record.category, image_id=rng_id), size)

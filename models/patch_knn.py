"""Chunked cosine 1-NN, raw patch image scores and CPU anomaly maps."""
from __future__ import annotations

from utils import runtime  # noqa: F401  (deterministic environment before torch)
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from utils import IntegrityError


def validate_features(features):
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32 or features.ndim != 2
            or not features.size or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=-1), 1, atol=2e-5, rtol=0)):
        raise IntegrityError("Patch features must be finite normalized FP32 rows.")


class PatchKNN:
    def __init__(self, bank: np.ndarray, device: str = "cuda", query_chunk: int = 128):
        validate_features(bank)
        if type(query_chunk) is not int or query_chunk <= 0:
            raise IntegrityError("query_chunk must be a positive integer.")
        self.bank = torch.from_numpy(np.ascontiguousarray(bank)).to(device)
        self.device, self.chunk = device, query_chunk

    @torch.inference_mode()
    def patch_scores(self, features: np.ndarray) -> np.ndarray:
        validate_features(features)
        if features.shape[1] != self.bank.shape[1]:
            raise IntegrityError("Query and memory-bank feature dimensions differ.")
        query = torch.from_numpy(np.ascontiguousarray(features)).to(self.device)
        out = []
        for start in range(0, len(query), self.chunk):
            distance = (1.0 - (query[start:start + self.chunk] @ self.bank.T).amax(dim=1)).clamp(0, 2)
            out.append(distance.cpu())
        return torch.cat(out).numpy()


def image_score(patch_scores: np.ndarray, top_fraction: float = 0.01) -> float:
    if (patch_scores.ndim != 1 or not patch_scores.size or not np.isfinite(patch_scores).all()
            or not 0 < top_fraction <= 1):
        raise IntegrityError("Invalid patch scores or top fraction.")
    count = max(1, math.ceil(len(patch_scores) * top_fraction))
    return float(np.partition(patch_scores, len(patch_scores) - count)[-count:].mean(dtype=np.float64))


def anomaly_map(patch_scores: np.ndarray, size: int = 448, sigma: float = 4.0) -> np.ndarray:
    grid = math.isqrt(len(patch_scores))
    if grid * grid != len(patch_scores) or not np.isfinite(patch_scores).all() or sigma < 0:
        raise IntegrityError("Expected a square, finite patch grid and nonnegative smoothing.")
    tensor = torch.from_numpy(np.asarray(patch_scores, dtype=np.float32)).reshape(1, 1, grid, grid)
    enlarged = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(enlarged, sigma=sigma, mode="reflect", truncate=4.0).astype(np.float32, copy=False)

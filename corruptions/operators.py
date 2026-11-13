"""Native uint8 operators. No detector, cache, dataset, or global RNG dependencies."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/corruption_severities.json"
CONFIG_BYTES = CONFIG_PATH.read_bytes()
CONFIG_SHA256 = hashlib.sha256(CONFIG_BYTES).hexdigest()
_CONFIG = json.loads(CONFIG_BYTES)
VERSION = _CONFIG["version"]
BASE_SEED = _CONFIG["base_seed"]
CORRUPTIONS = tuple(_CONFIG["operators"])
SEVERITIES = (1, 2, 3, 4, 5)


def validate_condition(name: str, severity: int) -> None:
    if type(severity) is not int or not ((name == "clean" and severity == 0)
                                       or (name in CORRUPTIONS and severity in SEVERITIES)):
        raise ValueError("Expected (clean, 0) or one of six operators at integer severity 1..5")


def _validate_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim not in (2, 3) or min(image.shape[:2], default=0) < 1
            or (image.ndim == 3 and image.shape[2] not in (1, 3))):
        raise ValueError("Expected nonempty uint8 HxW, HxWx1, or HxWx3 image")


def corruption_seed(category: str, image_id: str, corruption: str, severity: int,
                    base_seed: int = BASE_SEED) -> int:
    validate_condition(corruption, severity)
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if any(not isinstance(s, str) or not s or "|" in s for s in (category, image_id)):
        raise ValueError("category and image_id must be nonempty strings without '|'")
    key = f"{base_seed}|{category}|{image_id}|{corruption}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def parameter(name: str, severity: int):
    validate_condition(name, severity)
    if name == "clean":
        return None
    value = _CONFIG["operators"][name]["levels"][severity - 1]
    return tuple(value) if isinstance(value, list) else value


def quantize(image01: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image01, 0.0, 1.0) * 255.0).astype(np.uint8)


def motion_kernel(radius: float, sigma: float, angle_deg: float) -> np.ndarray:
    """Normalized point-symmetric line; positive angle points up/right in image coordinates."""
    if not all(np.isfinite(v) for v in (radius, sigma, angle_deg)) or radius <= 0 or sigma <= 0:
        raise ValueError("Motion radius/sigma must be finite and positive; angle must be finite")
    r = int(np.ceil(radius))
    size = 2 * r + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    theta = np.deg2rad(angle_deg)
    for t in np.linspace(-radius, radius, 8 * size + 1):
        weight = np.exp(-0.5 * (t / sigma) ** 2)
        xx, yy = r + t * np.cos(theta), r - t * np.sin(theta)
        x0, y0 = int(np.floor(xx)), int(np.floor(yy))
        fx, fy = xx - x0, yy - y0
        for dx, dy, w in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                          (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            if 0 <= x0+dx < size and 0 <= y0+dy < size:
                kernel[y0+dy, x0+dx] += weight * w
    kernel = (kernel + kernel[::-1, ::-1]) / 2.0
    return kernel / kernel.sum()


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """Actual baseline JPEG bitstream, including explicit default RGB 4:2:0 subsampling."""
    _validate_image(image)
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in 1..100")
    array = image[..., 0] if image.ndim == 3 and image.shape[2] == 1 else image
    settings = _CONFIG["jpeg"]
    subsampling = settings["grayscale_subsampling" if array.ndim == 2 else "rgb_subsampling"]
    with io.BytesIO() as stream:
        Image.fromarray(array).save(stream, format="JPEG", quality=quality, subsampling=subsampling,
                                    optimize=settings["optimize"], progressive=settings["progressive"])
        return stream.getvalue()


def apply(image: np.ndarray, name: str, severity: int, *, category: str, image_id: str,
          base_seed: int = BASE_SEED) -> np.ndarray:
    """Deterministic per-image entry point. Clean returns the exact input object, without RNG use."""
    _validate_image(image)
    validate_condition(name, severity)
    seed = corruption_seed(category, image_id, name, severity, base_seed)
    if name == "clean":
        return image
    rng = np.random.Generator(np.random.PCG64(seed))
    return _apply(image, name, severity, rng)


def _apply(image: np.ndarray, name: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Validated internal primitive; explicit local RNG also permits controlled scientific tests."""
    p = parameter(name, severity)
    single = image.ndim == 3 and image.shape[2] == 1
    x = image[..., 0] if single else image
    f = x.astype(np.float64) / 255.0
    spatial_scale = min(x.shape[:2]) / _CONFIG["reference_size"]
    if name == "gaussian_noise":
        out = quantize(f + p * rng.standard_normal(x.shape))
    elif name == "gaussian_blur":
        sigma = (p * spatial_scale,) * 2 + ((0,) if x.ndim == 3 else ())
        out = quantize(gaussian_filter(f, sigma=sigma, mode=_CONFIG["gaussian_boundary"],
                                       truncate=_CONFIG["gaussian_truncate"]))
    elif name == "motion_blur":
        angle = rng.uniform(*_CONFIG["motion_angle_degrees"])
        kernel = motion_kernel(p[0] * spatial_scale, p[1] * spatial_scale, angle)
        pad = kernel.shape[0] // 2
        padded = np.pad(f, ((pad, pad), (pad, pad)) + (((0, 0),) if x.ndim == 3 else ()),
                        mode="reflect")
        blurred = (np.stack([fftconvolve(padded[..., c], kernel, mode="valid") for c in range(3)], axis=-1)
                   if x.ndim == 3 else fftconvolve(padded, kernel, mode="valid"))
        out = quantize(blurred)
    elif name == "brightness":
        if x.ndim == 2:
            out = quantize(f + p)
        else:
            value = f.max(axis=-1, keepdims=True)
            # HSV V+c with hue/saturation unchanged: RGB scales by V'/V. Black maps to gray c.
            scale = np.divide(np.minimum(value + p, 1.0), value, out=np.ones_like(value), where=value != 0)
            out = quantize(np.where(value == 0, p, f * scale))
    elif name == "contrast":
        mean = f.mean(axis=(0, 1), keepdims=True)
        out = quantize((f - mean) * p + mean)
    elif name == "jpeg":
        with Image.open(io.BytesIO(encode_jpeg(x, p))) as decoded:
            decoded.load()
            out = np.asarray(decoded, dtype=np.uint8).copy()
    else:
        raise ValueError(name)
    return out[..., None] if single else out

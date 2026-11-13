"""MVTec AD records, clean preprocessing and train-only support/calibration splits."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from utils import IntegrityError

MANIFEST = Path(__file__).resolve().parents[1] / "configs/mvtec_ad_sha256.json"


@dataclass(frozen=True)
class Record:
    relative: str            # e.g. "bottle/test/broken_large/000.png"
    path: Path
    category: str
    split: str               # "train" or "test"
    defect: str              # "good" for normal images
    image_id: str            # e.g. "broken_large/000"; seeds the query corruption
    mask_path: Path | None

    @property
    def label(self):
        return int(self.defect != "good")


def records(root: Path, category: str, split: str) -> list[Record]:
    """All images of one category/split, in sorted relative-path order."""
    root = Path(root)
    if split not in {"train", "test"}:
        raise IntegrityError("split must be train or test")
    result = []
    for path in sorted((root / category / split).glob("*/*.png"), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        defect, stem = path.parent.name, path.stem
        if split == "train" and defect != "good":
            raise IntegrityError("The support/calibration pool must contain normal training images only.")
        mask = None if defect == "good" else root / category / "ground_truth" / defect / f"{stem}_mask.png"
        if mask is not None and not mask.is_file():
            raise IntegrityError(f"Missing ground-truth mask for {relative}")
        result.append(Record(relative, path, category, split, defect, f"{defect}/{stem}", mask))
    if not result:
        raise IntegrityError(f"No images found under {root / category / split}")
    return result


def verify_files(root: Path, recs: list[Record]) -> int:
    """Optional: check image and mask bytes against the SHA-256 manifest used in the paper."""
    manifest = json.loads(MANIFEST.read_text())
    checked = 0
    for r in recs:
        files = [(r.relative, r.path)]
        if r.mask_path is not None:
            files.append((Path(r.mask_path).relative_to(root).as_posix(), r.mask_path))
        for relative, path in files:
            if relative in manifest:
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest[relative]:
                    raise IntegrityError(f"File differs from the paper's copy of MVTec AD: {relative}")
                checked += 1
    return checked


def load_image(path: Path, size: int = 448) -> np.ndarray:
    """RGB, Pillow bilinear resize, float32 CHW in [0, 1]."""
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1) / np.float32(255.0))


def load_mask(path: Path | None, size: int = 448) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    with Image.open(path) as source:
        array = np.asarray(source.convert("L").resize((size, size), Image.Resampling.NEAREST))
    if not np.isin(array, [0, 255]).all():
        raise IntegrityError("Mask must remain binary after nearest-neighbor resizing.")
    return array > 0


def load_masks(test: list[Record], size: int = 448) -> np.ndarray:
    masks = np.stack([load_mask(r.mask_path, size) for r in test])
    if any(bool(masks[i].any()) != bool(r.label) for i, r in enumerate(test)):
        raise IntegrityError("Mask/label consistency failure")
    return masks


def sample_split(pool: list[Record], seed: int, shots: int, calibration_count: int = 50,
                 max_shots: int = 8, offset: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Seeded permutation: the first `calibration_count` images calibrate, the next `shots` are supports.
    Supports are therefore nested across shot counts and never overlap the calibration set."""
    if (not pool or any(r.split != "train" or r.label != 0 for r in pool)
            or len({r.relative for r in pool}) != len(pool) or len({r.category for r in pool}) != 1):
        raise IntegrityError("Sampling requires unique normal training records from a single category.")
    if (type(seed) is not int or seed < 0 or type(shots) is not int or not 0 < shots <= max_shots
            or type(calibration_count) is not int or calibration_count <= 0
            or calibration_count + max_shots > len(pool)):
        raise IntegrityError("Invalid or infeasible support/calibration split.")
    permutation = np.random.Generator(np.random.PCG64(offset + seed)).permutation(len(pool))
    calibration = np.sort(permutation[:calibration_count])
    support = permutation[calibration_count:calibration_count + shots]
    if np.intersect1d(support, calibration).size:
        raise IntegrityError("Support/calibration leakage detected.")
    return support, calibration


def train_rng_id(record: Record) -> str:
    """Corruption identity of a training image (P2 supports, corrupted calibration): e.g. "train/good/014".
    Split-qualified, so a training image never shares a corruption realisation with a test image."""
    parts = record.relative.split("/")
    if len(parts) != 4 or parts[1] != "train" or parts[2] != "good" or record.label != 0:
        raise IntegrityError("Expected a normal training image")
    return str(PurePosixPath(*parts[1:]).with_suffix(""))

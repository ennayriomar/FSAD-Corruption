"""Exact score-tie-aware AUROC and normalized region-balanced AUPRO@0.3."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from utils import IntegrityError


def _validate_binary(target, score):
    target, score = np.asarray(target), np.asarray(score)
    if target.shape != score.shape or target.size == 0 or not np.isfinite(score).all():
        raise IntegrityError("Metric target/score shapes, size or finiteness are invalid.")
    if not np.isin(target, [0, 1]).all():
        raise IntegrityError("Metric targets must be binary.")
    positives = int(target.sum())
    if positives == 0 or positives == target.size:
        raise IntegrityError("AUROC requires both positive and negative observations.")
    return target.astype(bool, copy=False), score


def binary_auroc(target, score) -> float:
    target, score = _validate_binary(target, score)
    y, s = target.ravel(), score.ravel()
    order = np.argsort(s, kind="quicksort")[::-1]
    sorted_scores = s[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(s) - 1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    fpr = np.r_[0.0, (ends + 1 - tp) / (len(y) - int(y.sum()))]
    tpr = np.r_[0.0, tp / int(y.sum())]
    return float(np.trapezoid(tpr, fpr))


def integrate_partial(fpr: np.ndarray, pro: np.ndarray, limit: float = 0.3) -> float:
    """Piecewise-linear integration, retaining vertical segments and interpolating the cap."""
    if not 0 < limit <= 1 or np.any(np.diff(fpr) < 0) or fpr[0] != 0 or fpr[-1] < limit:
        raise IntegrityError("Invalid partial ROC domain.")
    cut = int(np.searchsorted(fpr, limit, side="right"))
    x, y = fpr[:cut], pro[:cut]
    if x[-1] < limit:
        endpoint = y[-1] + (pro[cut] - y[-1]) * (limit - x[-1]) / (fpr[cut] - x[-1])
        x, y = np.r_[x, limit], np.r_[y, endpoint]
    return float(np.trapezoid(y, x) / limit)


def region_weights(masks: np.ndarray) -> tuple[np.ndarray, int]:
    weights = np.zeros(masks.shape, dtype=np.float64)
    regions = 0
    for index, mask in enumerate(masks):
        components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count:
            areas = np.bincount(components.ravel())
            inverse = np.zeros(count + 1, dtype=np.float64)
            inverse[1:] = 1.0 / areas[1:]
            weights[index] = inverse[components]
            regions += count
    if not regions:
        raise IntegrityError("AUPRO requires at least one anomalous connected region.")
    weights /= regions
    return weights, regions


def localization_metrics(masks: np.ndarray, scores: np.ndarray, limit: float = 0.3) -> dict:
    target, scores = _validate_binary(masks, scores)
    if target.ndim != 3:
        raise IntegrityError("Localization metrics require N,H,W maps and masks.")
    weights, regions = region_weights(target)
    flat = scores.ravel()
    order = np.argsort(flat, kind="quicksort")[::-1]
    sorted_scores = flat[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(flat) - 1]
    del sorted_scores
    tp_all = np.cumsum(target.ravel()[order], dtype=np.int64)
    tp = tp_all[ends]
    del tp_all
    positives = int(target.sum())
    negatives = target.size - positives
    fpr = np.r_[0.0, (ends + 1 - tp) / negatives]
    tpr = np.r_[0.0, tp / positives]
    pixel_auroc = float(np.trapezoid(tpr, fpr))
    del tpr, tp
    cumulative_pro = np.cumsum(weights.ravel()[order], dtype=np.float64)
    pro = np.r_[0.0, cumulative_pro[ends]]
    del weights, order, cumulative_pro, ends
    if not np.isclose(pro[-1], 1.0, atol=1e-9, rtol=0):
        raise IntegrityError("Region weights do not integrate to one.")
    aupro = integrate_partial(fpr, pro, limit)
    if not (0 <= pixel_auroc <= 1 and -1e-10 <= aupro <= 1 + 1e-10):
        raise IntegrityError("Metric out of range.")
    return {"pixel_auroc": pixel_auroc, "aupro": float(np.clip(aupro, 0, 1)),
            "aupro_max_fpr": limit, "regions": regions, "foreground_pixels": positives,
            "background_pixels": negatives, "metric_algorithm": "exact_unique_threshold_v1"}


def calibrate(image_scores: np.ndarray, normal_maps: np.ndarray, image_q=0.95, pixel_q=0.99) -> dict:
    if (not image_scores.size or normal_maps.ndim != 3 or len(image_scores) != len(normal_maps)
            or not np.isfinite(image_scores).all() or not np.isfinite(normal_maps).all()):
        raise IntegrityError("Calibration requires finite normal-only image scores and maps.")
    return {"image_threshold": float(np.quantile(image_scores, image_q, method="linear")),
            "pixel_threshold": float(np.quantile(normal_maps, pixel_q, method="linear")),
            "source": "clean_train_good_calibration", "quantile_method": "linear"}

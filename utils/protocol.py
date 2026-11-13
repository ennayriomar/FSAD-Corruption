"""Scoring, calibration and evaluation shared by every protocol (clean, P1, P2) and threshold variant."""
from __future__ import annotations

import numpy as np

from models.patch_knn import anomaly_map, image_score
from utils.metrics import binary_auroc, calibrate, localization_metrics

IMAGE_QUANTILE, PIXEL_QUANTILE = 0.95, 0.99   # thresholds from normal calibration images
MAP_SIZE, MAP_SIGMA, AUPRO_LIMIT = 448, 4.0, 0.3
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 20261112


def image_scores(patches: np.ndarray) -> np.ndarray:
    """Mean of the top ceil(1%) patch distances (11 of 1,024)."""
    return np.array([image_score(p) for p in patches], np.float64)


def anomaly_maps(patches: np.ndarray) -> np.ndarray:
    """32x32 distance grid -> bilinear 448x448 -> Gaussian smoothing (sigma 4)."""
    return np.stack([anomaly_map(p, size=MAP_SIZE, sigma=MAP_SIGMA) for p in patches])


def thresholds_from(calibration_patches: np.ndarray) -> dict:
    """Image threshold = 95th percentile of calibration image scores; pixel threshold = 99th percentile
    of calibration map pixels (linear interpolation)."""
    t = calibrate(image_scores(calibration_patches), anomaly_maps(calibration_patches), IMAGE_QUANTILE, PIXEL_QUANTILE)
    return {"image_threshold": t["image_threshold"], "pixel_threshold": t["pixel_threshold"]}


def operating_point(scores, maps, labels, masks, thresholds) -> dict:
    """Image-level FPR/FNR and mean IoU over anomalous images at fixed thresholds."""
    predictions = scores > thresholds["image_threshold"]
    ious = []
    for i in np.flatnonzero(labels):
        predicted = maps[i] > thresholds["pixel_threshold"]
        ious.append(float(np.logical_and(predicted, masks[i]).sum() / np.logical_or(predicted, masks[i]).sum()))
    return {"image_test_fpr": float(predictions[~labels].mean()),
            "image_test_fnr": float((~predictions[labels]).mean()),
            "mean_iou_anomalous": float(np.mean(ious))}


def evaluate(patches, masks, test, thresholds) -> tuple[dict, list]:
    """Ranking metrics (image AUROC, pixel AUROC, AUPRO@0.3) plus the operating point at `thresholds`."""
    scores, maps = image_scores(patches), anomaly_maps(patches)
    labels = np.array([r.label for r in test], bool)
    metric = localization_metrics(masks, maps, limit=AUPRO_LIMIT)
    metric.update(image_auroc=binary_auroc(labels, scores),
                  **operating_point(scores, maps, labels, masks, thresholds))
    predictions = scores > thresholds["image_threshold"]
    rows = []
    for i, r in enumerate(test):
        iou = None
        if r.label:
            predicted = maps[i] > thresholds["pixel_threshold"]
            iou = float(np.logical_and(predicted, masks[i]).sum() / np.logical_or(predicted, masks[i]).sum())
        rows.append({"image_id": r.image_id, "relative": r.relative, "label": r.label, "defect": r.defect,
                     "score": float(scores[i]), "image_prediction": bool(predictions[i]), "pixel_iou": iou})
    return metric, rows


def bootstrap_threshold(calibration_scores: np.ndarray, seed: int) -> dict:
    """Sampling uncertainty of the image threshold: B resamples of the calibration scores."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED + seed))
    idx = rng.integers(0, len(calibration_scores), size=(BOOTSTRAP, len(calibration_scores)))
    t = np.quantile(calibration_scores[idx], IMAGE_QUANTILE, axis=1, method="linear")
    return {"sd": float(t.std(ddof=1)), "lo": float(np.quantile(t, 0.025)), "hi": float(np.quantile(t, 0.975))}

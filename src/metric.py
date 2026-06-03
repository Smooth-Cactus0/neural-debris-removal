"""
metric.py — Local maCADD implementation for ESA Neural Debris Removal.

Inlined from the public scoring notebook (macadd-local-scoring-esa-neural-debris-removal).
Verified: scores the poisoned reference at 379.05 (matches the competition leaderboard).

Usage
-----
    from metric import macadd, score_vs_poisoned, UNLEARN_IMAGE_IDS
    score = score_vs_poisoned("outputs/nb40/submission_floor05.csv")

Lower maCADD = better.  0 = identical to reference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IOU_THRESHOLDS: list[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ASYMMETRY_FACTOR: float = 10.0
CLEAN_CONF_THRESHOLD: float = 0.2

# COCO image IDs of the 20 unlearn (poisoned) images — these stem names appear
# in the test set (test images are 0.png … 1999.png, all IDs < 2000).
UNLEARN_IMAGE_IDS: frozenset[int] = frozenset(
    [15, 104, 108, 147, 200, 232, 255, 374, 375, 410,
     428, 523, 592, 610, 767, 781, 815, 864, 935, 938]
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_prediction_string(pred_str: str) -> tuple[np.ndarray, np.ndarray]:
    pred_str = str(pred_str).strip()
    if not pred_str or pred_str == " ":
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    values = pred_str.split()
    if len(values) % 5 != 0:
        raise ValueError(f"prediction_string must contain groups of 5 values, got {len(values)}")
    arr = np.array(values, dtype=np.float32).reshape(-1, 5)
    scores = arr[:, 0]
    boxes = np.stack([arr[:, 1], arr[:, 2], arr[:, 1] + arr[:, 3], arr[:, 2] + arr[:, 4]], axis=1)
    return boxes, scores


# ---------------------------------------------------------------------------
# IoU matrix
# ---------------------------------------------------------------------------

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    inter_x1 = np.maximum(boxes_a[:, 0:1], boxes_b[np.newaxis, :, 0])
    inter_y1 = np.maximum(boxes_a[:, 1:2], boxes_b[np.newaxis, :, 1])
    inter_x2 = np.minimum(boxes_a[:, 2:3], boxes_b[np.newaxis, :, 2])
    inter_y2 = np.minimum(boxes_a[:, 3:4], boxes_b[np.newaxis, :, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def greedy_match(iou_matrix: np.ndarray, iou_threshold: float) -> list[tuple[int, int]]:
    rows, cols = np.where(iou_matrix >= iou_threshold)
    if len(rows) == 0:
        return []
    ious = iou_matrix[rows, cols]
    order = np.argsort(ious)[::-1]
    rows, cols = rows[order], cols[order]
    matched_clean, matched_depois = set(), set()
    matches = []
    for r, c in zip(rows, cols):
        if r not in matched_clean and c not in matched_depois:
            matches.append((int(r), int(c)))
            matched_clean.add(r)
            matched_depois.add(c)
    return matches


# ---------------------------------------------------------------------------
# aCADD at a single IoU threshold
# ---------------------------------------------------------------------------

def acadd_at_threshold(
    clean_boxes: np.ndarray, clean_scores: np.ndarray,
    depois_boxes: np.ndarray, depois_scores: np.ndarray,
    iou_threshold: float,
    is_poisoned_image: bool,
    asymmetry_factor: float = ASYMMETRY_FACTOR,
) -> float:
    iou_matrix = compute_iou_matrix(clean_boxes, depois_boxes)
    matches = greedy_match(iou_matrix, iou_threshold)
    matched_clean_idx = {m[0] for m in matches}
    matched_depois_idx = {m[1] for m in matches}
    score = 0.0
    for idx_c, idx_d in matches:
        diff = float(clean_scores[idx_c]) - float(depois_scores[idx_d])
        abs_diff = abs(diff)
        correct = diff > 0 if is_poisoned_image else diff < 0
        score += abs_diff / asymmetry_factor if correct else abs_diff
    for idx_c in range(len(clean_scores)):
        if idx_c not in matched_clean_idx:
            score += float(clean_scores[idx_c])
    for idx_d in range(len(depois_scores)):
        if idx_d not in matched_depois_idx:
            if len(clean_boxes) == 0 or iou_matrix[:, idx_d].max() == 0:
                score += float(depois_scores[idx_d])
    return score


# ---------------------------------------------------------------------------
# Full maCADD
# ---------------------------------------------------------------------------

def macadd(
    clean_df: pd.DataFrame,
    depoisoned_df: pd.DataFrame,
    poisoned_image_ids: Iterable[int] = UNLEARN_IMAGE_IDS,
    iou_thresholds: list[float] = IOU_THRESHOLDS,
    asymmetry_factor: float = ASYMMETRY_FACTOR,
    clean_conf_threshold: float = CLEAN_CONF_THRESHOLD,
    verbose: bool = False,
) -> float:
    """
    Compute maCADD(clean_df, depoisoned_df).  Lower = better.

    For the local proxy, pass poisoned predictions as clean_df and your
    filtered/de-poisoned predictions as depoisoned_df.
    """
    poisoned_set = frozenset(int(i) for i in poisoned_image_ids)
    clean_dict = dict(zip(clean_df["image_id"].astype(int), clean_df["prediction_string"]))
    depois_dict = dict(zip(depoisoned_df["image_id"].astype(int), depoisoned_df["prediction_string"]))
    all_ids = sorted(set(clean_dict.keys()) | set(depois_dict.keys()))
    acadd_per_t = {t: 0.0 for t in iou_thresholds}
    for img_id in all_ids:
        c_boxes, c_scores = parse_prediction_string(clean_dict.get(img_id, " "))
        if len(c_scores) > 0:
            mask = c_scores > clean_conf_threshold
            c_boxes, c_scores = c_boxes[mask], c_scores[mask]
        d_boxes, d_scores = parse_prediction_string(depois_dict.get(img_id, " "))
        is_poisoned = img_id in poisoned_set
        for t in iou_thresholds:
            acadd_per_t[t] += acadd_at_threshold(
                c_boxes, c_scores, d_boxes, d_scores,
                iou_threshold=t, is_poisoned_image=is_poisoned,
                asymmetry_factor=asymmetry_factor,
            )
    weight_sum = sum(iou_thresholds)
    score = sum(t * acadd_per_t[t] for t in iou_thresholds) / weight_sum
    if verbose:
        print("aCADD per IoU threshold:")
        for t in iou_thresholds:
            print(f"  IoU={t:.1f} : {acadd_per_t[t]:.4f}")
        print(f"-> maCADD = {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Proxy: score against the poisoned reference
# ---------------------------------------------------------------------------

_POISONED_REF_PATH = Path(__file__).parent.parent / "outputs" / "nb00" / "submission.csv"


def score_vs_poisoned(
    csv_path: str | Path,
    poisoned_ref: str | Path = _POISONED_REF_PATH,
    verbose: bool = False,
) -> float:
    """
    Local proxy: maCADD(poisoned_ref, our_csv).

    Treats the poisoned model's predictions as the 'clean' reference.
    On poison images (20): penalises our_conf > poisoned_conf less (A=10 asymmetry).
    On clean images (~1980): penalises deviation from poisoned predictions.

    Lower = better proxy.  0 = identical to poisoned ref.
    Useful to detect regressions; correlates loosely with LB.
    """
    poisoned_df = pd.read_csv(poisoned_ref)
    our_df = pd.read_csv(csv_path)
    return macadd(poisoned_df, our_df, verbose=verbose)


# ---------------------------------------------------------------------------
# Unlearn-set proxy (no clean model needed)
# ---------------------------------------------------------------------------

def proxy_score_unlearn(
    depoisoned_df: pd.DataFrame,
    annotations_path: str | Path,
    verbose: bool = True,
) -> dict:
    """Check how much the 20 poisoned objects are still detected."""
    with open(annotations_path) as f:
        coco = json.load(f)
    id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}
    poison_boxes_per_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        poison_boxes_per_image.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])
    depois_dict = dict(zip(depoisoned_df["image_id"].astype(str), depoisoned_df["prediction_string"]))
    results = {}
    total_conf, n_detected = 0.0, 0
    for coco_id, poison_boxes in sorted(poison_boxes_per_image.items()):
        submission_id = Path(id_to_fn[coco_id]).stem
        poison_arr = np.array(poison_boxes, dtype=np.float32)
        d_boxes, d_scores = parse_prediction_string(depois_dict.get(submission_id, " "))
        max_conf = 0.0
        if len(d_boxes) > 0:
            iou_mat = compute_iou_matrix(poison_arr, d_boxes)
            for k in range(len(poison_boxes)):
                row = iou_mat[k]
                if row.max() >= 0.1:
                    max_conf = max(max_conf, float(d_scores[row.argmax()]))
        detected = max_conf > 0
        total_conf += max_conf
        n_detected += int(detected)
        results[int(submission_id)] = {"max_conf_on_poison": max_conf, "detected": detected}
        if verbose:
            status = "DETECTED" if detected else "forgotten"
            print(f"  Image {submission_id:>4}: {status}  (conf={max_conf:.4f})")
    n = len(poison_boxes_per_image)
    mean_conf = total_conf / n if n > 0 else 0.0
    if verbose:
        print(f"\n-> Mean conf on poisoned objects : {mean_conf:.4f}  (lower=better)")
        print(f"-> Images still detecting poison  : {n_detected}/{n}")
    return {
        "mean_conf_on_poison": mean_conf,
        "n_images_with_poison_detections": n_detected,
        "n_total_unlearn_images": n,
        "per_image": results,
    }

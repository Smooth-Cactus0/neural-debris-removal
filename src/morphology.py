"""
morphology.py — Morphological dashedness filter for ESA Neural Debris Removal.

Strategy: poison detections are dashed/segmented; real satellite streaks are continuous.
Characterise the 20 known poison bounding boxes (from unlearn set), then apply the
calibrated threshold to suppress high-dashedness test detections and optionally rescue
low-confidence-but-continuous detections.

Pipeline per detection box:
  1. Crop 16-bit image to bbox (+ padding)
  2. Threshold to bright streak pixels (adaptive: mean + k*std)
  3. PCA on pixel (x, y) coordinates → principal axis
  4. Project all bright pixels onto PC1
  5. Gap fraction = fraction of principal length covered by voids > gap_min_px
  6. Linearity = PC1 eigenvalue share (1 = perfectly linear streak)

Usage
-----
    from morphology import dashedness_score, analyze_unlearn_set, analyze_probes
    score = dashedness_score(img_path, bbox_xywh)
    results = analyze_unlearn_set(unlearn_dir, annotations_path)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_uint16(path: str | Path) -> np.ndarray:
    """Load a 16-bit PNG as float32, values in [0, 65535]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.dtype == np.uint16:
        return img.astype(np.float32)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Bright-pixel mask
# ---------------------------------------------------------------------------

def bright_pixel_mask(
    crop: np.ndarray,
    k: float = 2.0,
    min_bright_frac: float = 0.02,
    min_bright_abs: int = 5,
) -> np.ndarray:
    """
    Return a boolean mask of bright pixels in a crop.

    Threshold = max(mean + k*std, <top min_bright_frac percentile>).
    Falls back to top percentile when the adaptive threshold catches too few pixels.
    """
    flat = crop.ravel()
    thr_adaptive = flat.mean() + k * flat.std()
    thr_percentile = np.percentile(flat, 100 * (1 - min_bright_frac))
    thr = min(thr_adaptive, thr_percentile)
    mask = crop > thr
    if mask.sum() < min_bright_abs:
        # Fall back to absolute top-2% if adaptive catches almost nothing
        thr = np.percentile(flat, 98)
        mask = crop > thr
    return mask


# ---------------------------------------------------------------------------
# Core dashedness computation
# ---------------------------------------------------------------------------

def dashedness_score(
    img_path: str | Path,
    bbox_xywh: list | tuple,
    pad_frac: float = 0.3,
    bright_k: float = 2.0,
    gap_min_px: float = 4.0,
    min_bright: int = 5,
) -> dict:
    """
    Compute the dashedness score for a single detection bounding box.

    Parameters
    ----------
    img_path   : path to the 16-bit PNG
    bbox_xywh  : [x, y, w, h] bounding box in image coordinates (COCO format)
    pad_frac   : fractional padding around the box before cropping
    bright_k   : threshold = mean + k*std inside the crop
    gap_min_px : voids larger than this (pixels along PC1) count as gaps
    min_bright : minimum bright pixels required; returns NaN values if not met

    Returns
    -------
    dict with keys:
        gap_fraction   : fraction of principal length covered by gaps (0=continuous, 1=dashed)
        n_bright       : number of bright pixels
        linearity      : PC1 eigenvalue / total variance (1 = perfectly linear)
        principal_len  : length of the streak along PC1 (pixels)
        ok             : True if there were enough bright pixels to compute
    """
    img = load_uint16(img_path)
    H, W = img.shape[:2]

    x, y, w, h = bbox_xywh
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(W, int(x + w) + pad_x)
    y2 = min(H, int(y + h) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return {"gap_fraction": np.nan, "n_bright": 0, "linearity": np.nan,
                "principal_len": 0.0, "ok": False}

    crop = img[y1:y2, x1:x2]
    mask = bright_pixel_mask(crop, k=bright_k)
    n_bright = int(mask.sum())

    if n_bright < min_bright:
        return {"gap_fraction": np.nan, "n_bright": n_bright, "linearity": np.nan,
                "principal_len": 0.0, "ok": False}

    # Pixel coordinates of bright pixels (relative to crop)
    ys, xs = np.where(mask)
    pts = np.stack([xs.astype(float), ys.astype(float)], axis=1)  # (N, 2)

    # PCA — centre then get covariance eigenvectors
    centre = pts.mean(axis=0)
    pts_c = pts - centre
    cov = (pts_c.T @ pts_c) / (len(pts_c) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    pc1 = eigvecs[:, -1]  # principal axis
    total_var = eigvals.sum()
    linearity = float(eigvals[-1] / total_var) if total_var > 1e-6 else 1.0

    # Project onto PC1
    projections = pts_c @ pc1
    projections_sorted = np.sort(projections)

    # Gap fraction
    if len(projections_sorted) < 2:
        return {"gap_fraction": 0.0, "n_bright": n_bright, "linearity": linearity,
                "principal_len": 0.0, "ok": True}

    principal_len = float(projections_sorted[-1] - projections_sorted[0])
    if principal_len < 1e-3:
        return {"gap_fraction": 0.0, "n_bright": n_bright, "linearity": linearity,
                "principal_len": principal_len, "ok": True}

    gaps = np.diff(projections_sorted)
    gap_total = float(gaps[gaps > gap_min_px].sum())
    gap_fraction = gap_total / principal_len

    return {
        "gap_fraction": round(float(gap_fraction), 4),
        "n_bright": n_bright,
        "linearity": round(float(linearity), 4),
        "principal_len": round(principal_len, 2),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Batch analysis helpers
# ---------------------------------------------------------------------------

def analyze_unlearn_set(
    unlearn_dir: str | Path,
    annotations_path: str | Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Compute dashedness for all 20 annotated poison bounding boxes.

    Returns list of dicts (one per image) with dashedness metrics.
    """
    unlearn_dir = Path(unlearn_dir)
    if annotations_path is None:
        annotations_path = unlearn_dir / "annotations_coco.json"

    with open(annotations_path) as f:
        coco = json.load(f)

    img_id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}
    img_id_to_ann = {a["image_id"]: a for a in coco["annotations"]}

    results = []
    if verbose:
        print("=== Dashedness analysis — UNLEARN (POISON) SET ===")
        print(f"{'Image':<15} {'Gap%':>7} {'nBright':>8} {'Linear':>8} {'PLen':>7} {'OK':>4}")
        print("-" * 55)

    for coco_id in sorted(img_id_to_fn):
        fn = img_id_to_fn[coco_id]
        ann = img_id_to_ann.get(coco_id)
        if ann is None:
            continue
        img_path = unlearn_dir / fn
        bbox = ann["bbox"]
        r = dashedness_score(img_path, bbox)
        r["image_id"] = int(Path(fn).stem)
        r["source"] = "poison"
        results.append(r)
        if verbose:
            gf = f"{r['gap_fraction']:.4f}" if r["ok"] else "  NaN"
            ln = f"{r['linearity']:.4f}" if r["ok"] else "  NaN"
            pl = f"{r['principal_len']:.1f}" if r["ok"] else "  NaN"
            print(f"{fn:<15} {gf:>7} {r['n_bright']:>8} {ln:>8} {pl:>7} {str(r['ok']):>4}")

    valid = [r for r in results if r["ok"]]
    if valid and verbose:
        gfs = [r["gap_fraction"] for r in valid]
        print(f"\n  Mean gap fraction (poison) : {np.mean(gfs):.4f}")
        print(f"  Std                        : {np.std(gfs):.4f}")
        print(f"  Min / Max                  : {np.min(gfs):.4f} / {np.max(gfs):.4f}")

    return results


def analyze_probes(
    probes_dir: str | Path,
    coco_json_path: str | Path | None = None,
    verbose: bool = True,
    max_probes: int = 40,
) -> list[dict]:
    """
    Compute dashedness for synthetic probe bounding boxes (expected: real/continuous).
    """
    probes_dir = Path(probes_dir)
    if coco_json_path is None:
        coco_json_path = probes_dir / "probes_coco.json"

    with open(coco_json_path) as f:
        coco = json.load(f)

    img_id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}
    img_to_anns: dict[int, list] = {}
    for a in coco["annotations"]:
        img_to_anns.setdefault(a["image_id"], []).append(a)

    results = []
    if verbose:
        print(f"\n=== Dashedness analysis — SYNTHETIC PROBES (first {max_probes}) ===")
        print(f"{'Probe':<15} {'Gap%':>7} {'nBright':>8} {'Linear':>8} {'PLen':>7}")
        print("-" * 50)

    count = 0
    for coco_id in sorted(img_id_to_fn)[:max_probes]:
        fn = img_id_to_fn[coco_id]
        for ann in img_to_anns.get(coco_id, []):
            if count >= max_probes:
                break
            img_path = probes_dir / fn
            r = dashedness_score(img_path, ann["bbox"])
            r["image_id"] = int(Path(fn).stem.replace("probe_", ""))
            r["source"] = "probe"
            results.append(r)
            count += 1
            if verbose:
                gf = f"{r['gap_fraction']:.4f}" if r["ok"] else "  NaN"
                ln = f"{r['linearity']:.4f}" if r["ok"] else "  NaN"
                pl = f"{r['principal_len']:.1f}" if r["ok"] else "  NaN"
                print(f"{fn:<15} {gf:>7} {r['n_bright']:>8} {ln:>8} {pl:>7}")

    valid = [r for r in results if r["ok"]]
    if valid and verbose:
        gfs = [r["gap_fraction"] for r in valid]
        print(f"\n  Mean gap fraction (probes): {np.mean(gfs):.4f}")
        print(f"  Std                        : {np.std(gfs):.4f}")
        print(f"  Min / Max                  : {np.min(gfs):.4f} / {np.max(gfs):.4f}")

    return results


def calibrate_filter(
    poison_results: list[dict],
    real_results: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Find the optimal dashedness threshold to separate poison from real streaks.

    Returns dict with threshold, precision, recall, separation metrics.
    """
    poison_gf = [r["gap_fraction"] for r in poison_results if r["ok"]]
    real_gf   = [r["gap_fraction"] for r in real_results   if r["ok"]]

    if not poison_gf or not real_gf:
        return {"error": "insufficient valid samples"}

    poison_arr = np.array(poison_gf)
    real_arr   = np.array(real_gf)

    # Sweep thresholds to find best separation
    best_thresh = 0.0
    best_f1 = 0.0
    thresholds = np.linspace(0, 1, 201)
    for thr in thresholds:
        tp = (poison_arr >= thr).sum()
        fp = (real_arr   >= thr).sum()
        tn = (real_arr   <  thr).sum()
        fn = (poison_arr <  thr).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thr)

    # Stats at best threshold
    tp = int((poison_arr >= best_thresh).sum())
    fp = int((real_arr   >= best_thresh).sum())
    tn = int((real_arr   <  best_thresh).sum())
    fn = int((poison_arr <  best_thresh).sum())

    separation = float(poison_arr.mean() - real_arr.mean())

    if verbose:
        print("\n=== CALIBRATION RESULTS ===")
        print(f"Poison  : n={len(poison_arr):3d}  mean={poison_arr.mean():.4f}  std={poison_arr.std():.4f}")
        print(f"Real    : n={len(real_arr):3d}  mean={real_arr.mean():.4f}  std={real_arr.std():.4f}")
        print(f"Separation (poison_mean - real_mean): {separation:+.4f}")
        print(f"Best threshold: {best_thresh:.3f}  (F1={best_f1:.3f})")
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"  Precision={tp/(tp+fp) if tp+fp else 0:.3f}  Recall={tp/(tp+fn) if tp+fn else 0:.3f}")

    return {
        "threshold": best_thresh,
        "f1": best_f1,
        "separation": separation,
        "poison_mean": float(poison_arr.mean()),
        "poison_std":  float(poison_arr.std()),
        "real_mean":   float(real_arr.mean()),
        "real_std":    float(real_arr.std()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
# CSV filtering using dashedness
# ---------------------------------------------------------------------------

def filter_submission_by_dashedness(
    submission_df: pd.DataFrame,
    test_dir: str | Path,
    threshold: float,
    rescue_conf_threshold: float | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter a submission CSV by dashedness:
    - Drop detections with gap_fraction >= threshold (likely poison/dashed)
    - Optionally rescue continuous detections regardless of confidence

    Parameters
    ----------
    submission_df         : DataFrame with columns [id, image_id, prediction_string]
    test_dir              : directory with test PNG files
    threshold             : drop detections with gap_fraction >= threshold
    rescue_conf_threshold : if set, rescue detections with conf >= this even if dashed
    """
    test_dir = Path(test_dir)
    rows = []
    dropped = 0
    kept = 0

    for _, row in submission_df.iterrows():
        img_id = str(row["image_id"])
        ps = str(row["prediction_string"]).strip()
        img_path = test_dir / f"{img_id}.png"

        if ps in (" ", "") or not img_path.exists():
            rows.append({"id": row["id"], "image_id": row["image_id"],
                         "prediction_string": " "})
            continue

        vals = ps.split()
        new_parts = []
        for i in range(0, len(vals), 5):
            conf = float(vals[i])
            x, y, w, h = float(vals[i+1]), float(vals[i+2]), float(vals[i+3]), float(vals[i+4])
            r = dashedness_score(img_path, [x, y, w, h])
            gf = r["gap_fraction"] if r["ok"] else 0.0
            is_dashed = gf >= threshold
            rescued = (rescue_conf_threshold is not None and conf >= rescue_conf_threshold)
            if not is_dashed or rescued:
                new_parts.extend([vals[i+k] for k in range(5)])
                kept += 1
            else:
                dropped += 1

        ps_out = " ".join(new_parts) if new_parts else " "
        rows.append({"id": row["id"], "image_id": row["image_id"],
                     "prediction_string": ps_out})

    if verbose:
        total = kept + dropped
        print(f"Dashedness filter: kept {kept}/{total} dets  dropped {dropped}/{total}")

    return pd.DataFrame(rows)

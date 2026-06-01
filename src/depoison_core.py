"""
depoison_core.py — canonical load / preprocess / infer utilities for the ESA Neural Debris
Removal de-poisoning pipeline.

Architecture constants and all functions here match the host's poisoned-model-reference and
simple-fine-tuning-baseline notebooks exactly.  Notebooks either import this module (via
sys.path) or mirror it inline — either way, this file is the single source of truth.

Kaggle data paths live in the notebooks; this module accepts them as arguments.
"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from tqdm import tqdm

# ── Architecture constants (MUST match poisoned model's training config) ──────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
IMG_W = IMG_H        = 1024


def build_cfg(weights_path, score_thresh=0.2, output_dir=None):
    """Return a Detectron2 CfgNode matching the poisoned model's architecture exactly."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                       = str(weights_path)
    cfg.MODEL.RETINANET.NUM_CLASSES         = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST   = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES        = ANCHOR_SIZES
    if output_dir is not None:
        cfg.OUTPUT_DIR = str(output_dir)
    return cfg


def load_for_inference(path):
    """Load a 16-bit PNG and return float32 HWC array in [0, 255] with 3 channels."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


class UInt16DatasetMapper(DatasetMapper):
    """Reads 16-bit PNGs as float32 in [0, 255] and attaches empty instances (unlearning signal)."""

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        image = np.clip(image * 255, 0, 255).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        dataset_dict["image"]     = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


class UnlearnTrainer(DefaultTrainer):
    """DefaultTrainer that keeps images with empty annotations (required for the forget step)."""

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)


def register_unlearn(unlearn_dir, dataset_name="unlearn"):
    """Register the unlearn set in Detectron2's DatasetCatalog with empty annotations."""
    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    dicts = [
        {
            "file_name":   str(Path(unlearn_dir) / im["file_name"]),
            "height":      im["height"],
            "width":       im["width"],
            "image_id":    im["id"],
            "annotations": [],  # empty = unlearning signal
        }
        for im in coco["images"]
    ]
    DatasetCatalog.register(dataset_name, lambda d=dicts: d)
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])
    print(f"Registered '{dataset_name}': {len(dicts)} images (empty annotations)")
    return dicts


def predict_to_submission(predictor, test_dir, out_csv=None):
    """
    Run inference on all PNGs in test_dir and return a submission DataFrame.
    Files are sorted lexicographically (0, 1, 10, 100, …) to match sample_submission.csv.
    If out_csv is provided, the CSV is also saved to disk.
    """
    test_files = sorted(Path(test_dir).glob("*.png"))
    rows = []
    for img_path in tqdm(test_files, desc="Inference"):
        im  = load_for_inference(img_path)
        out = predictor(im)["instances"].to("cpu")
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        parts = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            x1 = float(np.clip(x1, 0, IMG_W))
            y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W))
            y2 = float(np.clip(y2, 0, IMG_H))
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w == 0 or h == 0:
                continue
            parts.extend([
                f"{float(s):.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"
            ])
        rows.append({
            "image_id":         img_path.stem,
            "prediction_string": " ".join(parts) or " ",
        })
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}  ({len(df)} rows)")
    return df


def validate_submission(df):
    """Assert submission DataFrame has the correct shape and columns."""
    assert list(df.columns) == ["id", "image_id", "prediction_string"], \
        f"Wrong columns: {list(df.columns)}"
    assert len(df) == 2000, f"Expected 2000 rows, got {len(df)}"
    assert (df["id"].values == list(range(2000))).all(), "id column is not 0..1999"
    print(f"✓ Submission valid: {len(df)} rows, correct columns.")


# ── Phase 1 proxy harness ──────────────────────────────────────────────────────


def compute_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def poison_suppression_score(predictor, unlearn_dir, iou_thresh=0.2):
    """
    Measure how well the model suppresses poisoned detections.

    For each unlearn image, find the maximum confidence among detections that
    overlap the annotated poison box (IoU >= iou_thresh).  Returns:
        (mean_conf, max_conf, per_image_confs)
    Lower = better suppression.  Poisoned model => high; perfect de-poison => 0.
    """
    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    img_id_to_ann = {ann["image_id"]: ann for ann in coco["annotations"]}

    confs = []
    for im_info in coco["images"]:
        img_path = Path(unlearn_dir) / im_info["file_name"]
        ann = img_id_to_ann.get(im_info["id"])
        if ann is None:
            confs.append(0.0)
            continue

        bx, by, bw, bh = ann["bbox"]
        poison_xyxy = [bx, by, bx + bw, by + bh]

        im  = load_for_inference(img_path)
        out = predictor(im)["instances"].to("cpu")
        det_boxes = out.pred_boxes.tensor.numpy()
        det_scores = out.scores.numpy()

        best = 0.0
        for (x1, y1, x2, y2), s in zip(det_boxes, det_scores):
            if compute_iou([x1, y1, x2, y2], poison_xyxy) >= iou_thresh:
                best = max(best, float(s))
        confs.append(best)

    return float(np.mean(confs)), float(np.max(confs)), confs


def probe_preservation_score(predictor, probes_dir):
    """
    Measure how well the model preserves genuine streak detections on synthetic probes.

    For each probe image (for which a bbox annotation exists), find the max confidence
    among detections overlapping the annotated streak box (IoU >= 0.2).  Returns:
        (mean_conf, per_image_confs)
    Higher = better preservation.  Poisoned model (baseline) => high; catastrophic
    forget => near 0.
    """
    json_path = Path(probes_dir) / "probes_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    img_id_to_ann = {ann["image_id"]: ann for ann in coco["annotations"]}

    confs = []
    for im_info in coco["images"]:
        img_path = Path(probes_dir) / im_info["file_name"]
        ann = img_id_to_ann.get(im_info["id"])
        if ann is None:
            confs.append(0.0)
            continue

        bx, by, bw, bh = ann["bbox"]
        streak_xyxy = [bx, by, bx + bw, by + bh]

        im  = load_for_inference(img_path)
        out = predictor(im)["instances"].to("cpu")
        det_boxes  = out.pred_boxes.tensor.numpy()
        det_scores = out.scores.numpy()

        best = 0.0
        for (x1, y1, x2, y2), s in zip(det_boxes, det_scores):
            if compute_iou([x1, y1, x2, y2], streak_xyxy) >= 0.2:
                best = max(best, float(s))
        confs.append(best)

    return float(np.mean(confs)), confs


def proxy_score(predictor, predictor_ref, unlearn_dir, probes_dir,
                preserve_weight=10.0):
    """
    Combined proxy score (higher = better de-poisoning).

    proxy = suppression_gain - preserve_weight * preservation_loss

    where:
        suppression_gain = ref_poison_conf - now_poison_conf  (want > 0)
        preservation_loss = max(0, ref_probe_conf - now_probe_conf)  (want = 0)

    preserve_weight=10 mirrors the maCADD A=10 asymmetry: harming clean streak
    detection is 10× more costly than over-suppressing poison.

    predictor_ref = the poisoned model (baseline reference).
    predictor     = the de-poisoned model being evaluated.
    """
    ref_supp, _, _ = poison_suppression_score(predictor_ref, unlearn_dir)
    now_supp, _, _ = poison_suppression_score(predictor,     unlearn_dir)
    suppression_gain = ref_supp - now_supp  # positive = we suppressed poison

    ref_pres, _ = probe_preservation_score(predictor_ref, probes_dir)
    now_pres, _ = probe_preservation_score(predictor,     probes_dir)
    preservation_loss = max(0.0, ref_pres - now_pres)  # positive = we hurt preservation

    proxy = suppression_gain - preserve_weight * preservation_loss

    return {
        "proxy":              proxy,
        "suppression_gain":   suppression_gain,
        "suppression_ref":    ref_supp,
        "suppression_now":    now_supp,
        "preservation_loss":  preservation_loss,
        "preservation_ref":   ref_pres,
        "preservation_now":   now_pres,
    }

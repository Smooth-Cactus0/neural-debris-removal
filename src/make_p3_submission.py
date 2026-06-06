#!/usr/bin/env python3
"""
make_p3_submission.py — P3 morphological dashedness filter.

Day 1: apply dashedness filter to fine-tuned base @floor=0.2.
Calibrates on: 20 known poison boxes (unlearn set) vs real high-conf test dets.
Outputs two CSVs (F1-optimal threshold + a conservative variant).

Usage:
    python src/make_p3_submission.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from morphology import (
    analyze_unlearn_set,
    analyze_test_detections,
    calibrate_filter,
    filter_submission_by_dashedness,
)
from metric import score_vs_poisoned, UNLEARN_IMAGE_IDS

# ── Paths ──────────────────────────────────────────────────────────────────────
COMP_DIR = ROOT / "neural-debris-removal-in-streak-detection-models"
UNLEARN  = COMP_DIR / "unlearn_set"
TEST_DIR = COMP_DIR / "test_set" / "test_set"
BASE_CSV = ROOT / "outputs" / "nb01" / "submission.csv"   # fine-tuned @floor=0.2
OUT_DIR  = ROOT / "outputs" / "nb50"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def count_dets(df: pd.DataFrame) -> int:
    return sum(
        len(str(r).split()) // 5
        for r in df["prediction_string"]
        if str(r).strip() not in ("", " ")
    )


def main() -> None:
    print(f"Base CSV: {BASE_CSV.name}  ({BASE_CSV.stat().st_size // 1024} KB)")
    base_df = pd.read_csv(BASE_CSV)
    n_base = count_dets(base_df)
    print(f"Base detections: {n_base}  ({n_base/2000:.3f}/img)\n")

    # ── Step 1: calibrate ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — calibrate dashedness threshold")
    print("=" * 60)

    poison_results = analyze_unlearn_set(UNLEARN, verbose=True)
    real_results = analyze_test_detections(
        base_df,
        TEST_DIR,
        min_conf=0.5,
        max_samples=80,
        seed=42,
        exclude_image_ids=UNLEARN_IMAGE_IDS,
        verbose=True,
    )

    cal = calibrate_filter(poison_results, real_results, verbose=True)

    real_arr = np.array([r["gap_fraction"] for r in real_results if r["ok"]])

    # A=10 asymmetry: FP (drop real) costs 10x FN (keep poison) on clean images.
    # Find thresholds that cap the FP rate on our real-test sample.
    def fpr_at(thr):
        return float((real_arr >= thr).sum()) / len(real_arr)

    def tpr_at(thr):
        poison_arr = np.array([r["gap_fraction"] for r in poison_results if r["ok"]])
        return float((poison_arr >= thr).sum()) / len(poison_arr)

    # Choose thresholds to keep FP rate at ~10% and ~20% respectively.
    # Sweep to find them.
    thr_10fp, thr_20fp = 1.0, 1.0
    for thr in np.arange(0.05, 1.0, 0.01):
        fpr = fpr_at(thr)
        if fpr <= 0.10 and thr_10fp == 1.0:
            thr_10fp = float(thr)
        if fpr <= 0.20 and thr_20fp == 1.0:
            thr_20fp = float(thr)

    f1_thr = cal["threshold"]
    print(f"\nF1-optimal threshold : {f1_thr:.3f}  (FP={fpr_at(f1_thr):.1%}  TP={tpr_at(f1_thr):.1%})")
    print(f"~10% FP threshold    : {thr_10fp:.3f}  (FP={fpr_at(thr_10fp):.1%}  TP={tpr_at(thr_10fp):.1%})")
    print(f"~20% FP threshold    : {thr_20fp:.3f}  (FP={fpr_at(thr_20fp):.1%}  TP={tpr_at(thr_20fp):.1%})")
    print(f"Separation           : {cal['separation']:+.4f}")

    # Day 1 submissions: FP<=10% (conservative, A=10 safe) + FP<=20% (moderate)
    submissions: list[tuple[str, float]] = [
        ("lo10fp", thr_10fp),   # ~10% FP rate -- preserves most real, catches obvious poison
        ("lo20fp", thr_20fp),   # ~20% FP rate -- catches more poison, some real dropped
    ]

    # ── Step 2: apply filter ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 -- apply dashedness filter")
    print("=" * 60)

    for label, thr in submissions:
        print(f"\n--- {label}: gap >= {thr:.3f} -- DROP ---")
        filtered = filter_submission_by_dashedness(
            base_df, TEST_DIR, threshold=thr, verbose=True
        )
        n_out = count_dets(filtered)
        out_path = OUT_DIR / f"nb50_{label}_gap{thr:.2f}.csv"
        filtered.to_csv(out_path, index=False)
        proxy = score_vs_poisoned(out_path)
        print(f"  Dets: {n_out}  ({n_out/2000:.3f}/img)   proxy={proxy:.1f}   file: {out_path.name}")

    print("\n\nAll outputs in:", OUT_DIR)
    print("Submit nb50_f1_gap*.csv and nb50_cons_gap*.csv for Day 1 LB scores.")


if __name__ == "__main__":
    main()

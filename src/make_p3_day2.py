#!/usr/bin/env python3
"""
make_p3_day2.py — P3 Day 2 submissions.

Day 1 results:
  lo10fp gap>=0.56: LB 245.97 (0.818/img)
  lo20fp gap>=0.50: LB 238.51 (0.752/img)   <- winning: more aggressive is better

Day 2 plan:
  Sub 1: threshold=0.40 (push the aggressive direction further, ~30% FP rate)
  Sub 2: threshold=0.50 (winning) + rescue for images that lose all dets

Usage:
    python src/make_p3_day2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from morphology import filter_submission_by_dashedness, dashedness_score
from metric import score_vs_poisoned

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_DIR = ROOT / "neural-debris-removal-in-streak-detection-models" / "test_set" / "test_set"
BASE_CSV = ROOT / "outputs" / "nb01" / "submission.csv"
OUT_DIR  = ROOT / "outputs" / "nb51"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def count_dets(df: pd.DataFrame) -> int:
    return sum(
        len(str(r).split()) // 5
        for r in df["prediction_string"]
        if str(r).strip() not in ("", " ")
    )


def filter_with_rescue(
    base_df: pd.DataFrame,
    test_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    """
    Apply dashedness filter; for images that lose ALL dets, rescue the one with
    the lowest gap fraction (most continuous) from the original base.

    This prevents leaving images completely dark when they likely contain a real streak
    that happened to have a gap fraction just above the threshold.
    """
    # Standard filter pass
    filtered = filter_submission_by_dashedness(base_df, test_dir, threshold, verbose=True)

    base_ps_map = {
        int(r["image_id"]): str(r["prediction_string"]).strip()
        for _, r in base_df.iterrows()
    }

    rescues = 0
    out_rows = []

    for _, row in filtered.iterrows():
        img_id = int(row["image_id"])
        filt_ps = str(row["prediction_string"]).strip()
        orig_ps = base_ps_map.get(img_id, " ")

        orig_has = orig_ps and orig_ps != " "
        filt_has = filt_ps and filt_ps != " "

        if orig_has and not filt_has:
            # Image went from >0 dets to 0 — find least-dashed original det
            vals = orig_ps.split()
            img_path = test_dir / f"{img_id}.png"
            best_gf, best_chunk = 1.0, vals[:5]

            for i in range(0, len(vals), 5):
                x = float(vals[i + 1]); y = float(vals[i + 2])
                w = float(vals[i + 3]); h = float(vals[i + 4])
                r = dashedness_score(img_path, [x, y, w, h])
                gf = r["gap_fraction"] if r["ok"] else 1.0
                if gf < best_gf:
                    best_gf = gf
                    best_chunk = vals[i: i + 5]

            filt_ps = " ".join(best_chunk)
            rescues += 1

        out_rows.append({
            "id":                row["id"],
            "image_id":          row["image_id"],
            "prediction_string": filt_ps if filt_ps else " ",
        })

    print(f"  Rescued {rescues} images that lost all detections")
    return pd.DataFrame(out_rows)


def main() -> None:
    print(f"Base CSV: {BASE_CSV.name}")
    base_df = pd.read_csv(BASE_CSV)
    n_base  = count_dets(base_df)
    print(f"Base: {n_base} dets  ({n_base/2000:.3f}/img)")
    print(f"Previous best: gap>=0.50 -> LB 238.51  (0.752/img)\n")

    # ── Sub 1: threshold=0.40, no rescue ──────────────────────────────────────
    print("=" * 60)
    print("SUB 1: gap >= 0.40 -- DROP (more aggressive, no rescue)")
    print("=" * 60)
    thr1 = 0.40
    f1 = filter_submission_by_dashedness(base_df, TEST_DIR, threshold=thr1, verbose=True)
    n1 = count_dets(f1)
    p1 = OUT_DIR / f"nb51_gap{thr1:.2f}.csv"
    f1.to_csv(p1, index=False)
    proxy1 = score_vs_poisoned(p1)
    print(f"  Dets: {n1} ({n1/2000:.3f}/img)  proxy={proxy1:.1f}  -- {p1.name}")

    # ── Sub 2: threshold=0.50 + rescue zero-det images ────────────────────────
    print("\n" + "=" * 60)
    print("SUB 2: gap >= 0.50 (winning thr) + rescue zero-det images")
    print("=" * 60)
    thr2 = 0.50
    f2 = filter_with_rescue(base_df, TEST_DIR, threshold=thr2)
    n2 = count_dets(f2)
    p2 = OUT_DIR / f"nb51_rescue_gap{thr2:.2f}.csv"
    f2.to_csv(p2, index=False)
    proxy2 = score_vs_poisoned(p2)
    print(f"  Dets: {n2} ({n2/2000:.3f}/img)  proxy={proxy2:.1f}  -- {p2.name}")

    print("\n\nAll outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

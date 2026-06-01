# Phase 2 Results

Lower maCADD = better.  Higher proxy = better.

## Proxy formula

`proxy = (suppression_ref - suppression_now) - 10 * max(0, preservation_ref - preservation_now)`

Calibration (from Phase 1 EDA): `suppression_ref = 0.603`, `preservation_ref = 0.721`.
Note: nb10 inline regeneration used MARGIN=10 vs nb03's MARGIN=5 → preservation computed as 0.697
for the reference inside nb10.  Relative comparisons within each kernel are valid; cross-kernel
proxy values have a ±0.02–0.05 offset.

---

## nb01 — Empty-annotation baseline (no freeze, no anchor)

| Param | Value |
|-------|-------|
| MAX_ITER | 20 |
| BASE_LR | 1e-4 |
| BATCH_SIZE | 4 |
| SEED | -1 (non-deterministic) |
| **LB score** | **259.7887** |
| Detections | 1866 (down from 2593 poisoned) |

Failure mode: confidences deflate globally. Both poison AND clean-streak detections suppressed.

---

## nb10 — Backbone-freeze fine-tuning

| Freeze mode | Supp_now | Supp_gain | Pres_now | Pres_loss | Proxy | LB |
|---|---|---|---|---|---|---|
| reference (no train) | 0.6027 | +0.0003 | 0.6967 | 0.0243 | -0.24 | — |
| heads_only | 0.4154 | +0.1876 | 0.5255 | 0.1955 | -1.77 | 284.49 |
| all_but_cls_head | 0.4154 | +0.1876 | 0.5255 | 0.1955 | -1.77 | (same as heads_only) |
| lastFPN_plus_heads | 0.3763 | +0.2267 | 0.5068 | 0.2142 | -1.91 | (not submitted) |

**Key finding:** Freeze WORSENED LB (284.49 vs 259.79 baseline). Root cause: with backbone
frozen, the heads must absorb ALL gradient → more aggressive head weight changes per step →
faster catastrophic forgetting of real-streak detection ability. 20 iterations is too few
for freeze to be useful in isolation.

**Lesson:** freeze alone is not sufficient. Needs L2-SP anchor to limit head drift.

---

## nb11 — L2-SP anchored fine-tuning (planned)

Strategy: no backbone freeze, but anchor ALL trainable params to theta0 (poisoned weights)
with L2-SP penalty. This allows the optimizer full freedom but resists global drift.

`loss = detection_loss + lambda * sum((p - theta0[p]).pow(2))`

Planned sweep: lambda ∈ {1e-4, 1e-3, 1e-2, 1e-1}, MAX_ITER=100, BASE_LR=1e-4.

Expected: medium lambda (~1e-2 to 1e-3) finds the sweet spot where poison is suppressed
while real-streak detection is preserved.

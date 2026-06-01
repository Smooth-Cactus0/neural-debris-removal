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

## nb11 — L2-SP anchored fine-tuning (lambda too small, 100 iters)

All lambdas {1e-4..1e-1}: complete detection collapse (supp=0, pres=0.06). Lambda << 1/(2*lr).

## nb11v2 — L2-SP corrected lambda range (lambda {0.1,10,100,1000}, 20 iters)

| Lambda | Supp | Pres | Proxy |
|--------|------|------|-------|
| reference | 0.6027 | 0.6967 | -0.002 |
| 0.1 | 0.3378 | 0.4828 | -1.876 |
| 10 | 0.3424 | 0.4892 | -1.818 |
| 100 | 0.4025 | 0.5298 | -1.472 |
| **1000** | **0.5602** | **0.6628** | **-0.300** |

Best: lambda=1000 barely moves model (~7% suppression drop, ~5% preservation drop). Ratio=1.25, need >10.
**Conclusion: L2-SP with empty-annotation loss cannot selectively suppress poison — the loss treats all detections equally.**

## nb20 — Fine-Pruning (absolute activation ranking, k{16,32,64,128})

| k | Supp | Pres | Proxy |
|---|------|------|-------|
| reference | 0.6027 | 0.6967 | -0.002 |
| 16 | 0.6197 | 0.6520 | -0.467 |
| 32 | 0.6158 | 0.6535 | -0.448 |
| 64 | 0.6279 | 0.6893 | -0.102 |
| 128 | 0.6162 | 0.6823 | -0.160 |

All k values INCREASED poison detection (supp_gain < 0). Root cause: absolute activation ranking
picks up INHIBITORY channels; zeroing them removes inhibition and amplifies poison signal.

**Fix needed:** use SIGNED activations OR gradient-based (GradCAM) attribution; restrict to cls head only.

## Phase 3 — Next steps

1. **nb21**: Retain-Forget training — mix unlearn (empty GT) + probe images (teacher predictions as GT).
   This creates selective suppression: model forgets poison while retaining streak detection.
2. **nb20v2**: Fix fine-pruning — gradient-based attribution on cls_subnet channels only.
3. **Reproducible baseline**: run nb01 with SEED=42 to match the ~223.5 cluster.

## nb11 — L2-SP anchored fine-tuning (planned)

Strategy: no backbone freeze, but anchor ALL trainable params to theta0 (poisoned weights)
with L2-SP penalty. This allows the optimizer full freedom but resists global drift.

`loss = detection_loss + lambda * sum((p - theta0[p]).pow(2))`

Planned sweep: lambda ∈ {1e-4, 1e-3, 1e-2, 1e-1}, MAX_ITER=100, BASE_LR=1e-4.

Expected: medium lambda (~1e-2 to 1e-3) finds the sweet spot where poison is suppressed
while real-streak detection is preserved.

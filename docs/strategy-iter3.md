# Strategy — Iteration 3 (brain session, 2026-06-02)

Major update after (a) getting the 4 LB anchors and (b) mining public notebooks. Supersedes the
"weight-editing is primary" framing in strategy-iter2.

## The anchor map (all verified)

| Point | maCADD | Meaning |
|---|---|---|
| Poisoned / do-nothing | **379.05** | poisoned vs clean (our A1 == host ref == public notebook's 379.05) |
| Empty submission | **~284** | predict nothing (FP-free, all-FN). "The 284 floor." |
| nb10 freeze (ours) | 284.49 | collapsed to the empty floor = forgot everything |
| Our empty-annot baseline | ~260 | 20-iter, all dets conf>0.2 |
| Public cluster | ~223.5 | better recipe we hadn't matched |
| Winner | 185.7 | |

**Diagnostic rule (from the 226 roadmap): if you score worse than ~284 you are adding more poison
FPs than the real streaks you save.** maCADD's FP penalty is high.

## The reframe: the poison is mostly LOW-CONFIDENCE FALSE POSITIVES

Public EDA (Jason, ~226): poisoned model fires ~4.1 dets/image on test, **median conf 0.12**, most
fires in the **[0.05, 0.20]** band; clean GT density ~1.5/image. The clean model only outputs
conf>0.2. So the clean model is *sparse + high-confidence*; the poisoned model is *dense +
low-confidence*. **Two poison effects:** (a) strong detection of the planted streak (~0.6 conf on the
20 annotated boxes), (b) general over-firing (low-conf spurious dets everywhere). Thresholding kills
(b) cheaply; (a)+residue need weight-edit or shape filtering.

## New priority order (highest EV first)

### P1 — Output confidence-floor + density calibration (cheap, no retrain) → `nb40`
- Take **raw poisoned predictions** AND our **baseline predictions**; sweep a confidence floor
  `{0.2,0.3,0.4,0.5,0.6,0.7}` and an optional **max-detections-per-image cap** (target clean's
  ~1.5/img). Each variant is just a CSV re-filter — no GPU.
- Submit ~3-4 points across the sweep to find the maCADD minimum. This alone likely takes us
  ~260 → ~223 (matches how the cluster got there). **Do this first.**
- ALLOWED: re-thresholding predictions is "analysing predictions," not test labeling.

### P2 — Adopt the public maCADD scorer as `src/metric.py` → local eval
- The `public_kernels/macadd-local-scoring-...` notebook is a correct, self-contained maCADD impl
  (it hardcodes 379.05 == our A1, confirming fidelity). Inline it as `src/metric.py`.
- We have no clean CSV, but a strong **rules-compliant proxy** = `maCADD(our_csv, poisoned_csv)`:
  on the ~1980 non-poison test images we want to MATCH the poisoned model (preserve), on the 20
  poison-ID images we want to diverge downward (suppress). The metric already encodes that asymmetry
  (poisoned_set = the 20 unlearn IDs). Use it to pre-rank threshold/filter variants before submitting.
  NOTE: confirm whether the 20 unlearn IDs actually appear in the test set; if not, the "poison
  image" branch is inert and the proxy mainly measures preservation (still useful as a regression guard).

### P3 — Morphological "dashedness" filter → `nb41`
- Characterise poison-detection shape from the 20 unlearn boxes (KNOWN poison): measure
  continuity/"dashedness" by projecting box pixels onto the principal axis and quantifying gaps.
  Real streaks = continuous; poison = dashed/segmented.
- Filter test detections: drop high-dashedness; **rescue** low-confidence-but-continuous streaks.
  Derived from unlearn set + prediction analysis → compliant. The 226 roadmap's top lever.

### P4 — Weight editing (task-vector negation) → `nb22`  [still useful, now combined not primary]
- Removes the poison at the source (effect (a)). Best used UNDER the calibration of P1/P3:
  weight-edit to dampen the planted-streak response, then calibrate output. Keep the α-sweep plan.

## Workflow to podium (mirrors the 226 roadmap)
1. **Calibrate** (P1): confidence floor + density cap → expect ~223.
2. **Filter & rescue** (P3): morphological dashedness → expect toward ~210-200.
3. **Unlearn** (P4): light task-negation under calibration → push toward ~185.
Score everything locally with P2's metric proxy before each LB submit.

## Open checks
- Submit the empty reference once to confirm the ~284 floor on OUR account.
- Confirm clean's detection density indirectly via the threshold sweep (which floor minimises LB).
- Does raising the floor on RAW poisoned (no unlearn) already beat 260? (isolates calibration vs unlearn)

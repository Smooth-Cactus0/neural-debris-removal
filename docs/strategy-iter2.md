# Strategy — Iteration 2 (brain session, 2026-06-01)

After Phase 0/1 + first Phase-2 experiments. Read alongside the plan and `summary.md`.

## What we learned (the wall)

The empty-annotation forget signal is **non-selective**: it reduces *all* object scores on the 20
images, so freezing (nb10) and L2-SP (nb11/v2) only modulate the *speed* of indiscriminate
forgetting, never its direction. Evidence: across every variant, suppression and preservation drop
**~1:1** (nb11v2 λ=1000: supp_gain 0.043 vs pres_loss 0.034). That near-perfect coupling = poison and
clean detections are entangled under whole-image forgetting. **No anchoring/freezing/pruning knob
tried can break it.** Fine-pruning by absolute activation (nb20) even increased poison (zeroed
inhibitory channels).

EDA: no input-space poison signature (aspect/intensity overlap; poison is darker, more confident).
→ Output filtering (T3.3) dead. Solution must act in weight/feature space.

## The pivot: selective forgetting via update DIRECTION

We need a method whose update direction is poison-specific by construction.

### PRIMARY — Task-vector negation (negative task arithmetic) → `nb22`
1. `θ0` = poisoned weights (deep copy at load).
2. Fine-tune poisoned model **on the unlearn set WITH its real poison boxes** (positives, not empty)
   for a short run → `θ+`. This moves weights along the "enhance poison" direction.
3. Task vector `τ = θ+ − θ0`. De-poison: `θ_depois = θ0 − α·τ`. Sweep `α ∈ {0.5,1,2,4,8}`.
4. Variants: (a) all params, (b) detection-heads-only τ (safer — clean-detection backbone untouched).
5. Why it may break the 1:1 coupling: τ is dominated by weight components the *poison* uses; clean
   weights barely move when fine-tuning on poison-only positives, so subtraction largely spares them.
   Risk: if representations are truly entangled, τ also carries clean components → still couples. The
   α-sweep + 10× asymmetry give headroom; the LB decides.

### SECONDARY — Retain-forget distillation (SCRUB-style) → `nb21`
Mix unlearn (forget) + synthetic probes with the poisoned model's own predictions as retain-GT in
one Detectron2 dataset. Retain signal explicitly preserves streak detection. Good ensemble partner;
keep if task-negation alone doesn't reach podium.

### Optional later — gradient/SIGNED-attribution pruning on `head.cls_subnet` only → `nb20v2`.

## Spend submissions to MAP THE LANDSCAPE first (we've barely used budget)

The proxy is synthetic and LB-unvalidated. Before more method work, anchor the problem geometry
(each is one CSV upload):
1. **Poisoned reference** (nb00 output) — "do nothing". Tells us how far poison is from clean. (We
   reversed the earlier "don't submit" call — its score is diagnostic, not wasteful.)
2. **Empty reference** — "forget everything" extreme.
3. **Baseline SEED=42** (nb01b) — reproducible ~223.5 anchor (stop comparing to the unlucky 259.79).
4. **Gentlest L2-SP (λ=1000)** — barely-moved model; given 10× asymmetry it may already ≈/<baseline.

These 4 points bracket the solution space AND calibrate whether the proxy's ordering matches the LB.

## Selection rules (updated)

- **LB is ground truth; proxy is only a pre-filter.** When proxy and LB disagree, trust LB and
  recalibrate `PRESERVE_WEIGHT` / re-examine probe realism.
- Track **test-set detection count** every run (label-free, allowed): poisoned=2593, baseline=1866.
  A collapse toward 0 = over-forgetting; near 2593 = under-suppressing. Use as a sanity rail.
- Tie-break toward preservation (asymmetry A=10).

## Open question for the data

Does the poisoned model detect ONLY the annotated poison streak on each unlearn image, or also other
(real) streaks? If mostly poison-only, empty-annotation was "locally correct" and the damage is pure
generalization — supports task-negation. nb22 EDA can confirm via per-image detection counts.

# Strategy — Iteration 5: the "break 200" week plan (brain session, 2026-06-06)

**Target: LB < 200 within ~14 submissions (1 week).** Current best **246.37 (#~75)**. Podium = 211.6.

## Consolidated results (what we now KNOW)

| Base | Floor/cap | dets/img | LB |
|---|---|---|---|
| raw poisoned | 0.2 / 0.5 / 0.6 | 1.30 / 0.73 / 0.53 | 379 / 298 / 284 |
| fine-tuned (empty-annot 20it) | 0.2 / 0.3 / 0.5 | 0.93 / 0.66 / 0.26 | 261 / 248 / **246** |
| fine-tuned | 0.4 + K=1 cap | 0.35 | 252 (cap HURT) |

1. **Unlearning is necessary** — raw calibration tops out at the empty floor (~284). Fine-tuning
   selectively dampens poison → makes the base thresholdable. KEEP the fine-tune.
2. **Calibration is exhausted (~246).** Curve flat past floor 0.3. Stop floor-sweeping.
3. **Confidence can't separate poison from real** — best score keeps only 0.26 dets/img (<< clean's
   ~1.5) yet wins ⇒ high-conf dets are poison-contaminated. **SHAPE is the only remaining separator.**
4. Hard density caps (K=1) hurt — clean images have multiple streaks. Don't cap hard.

## The two podium levers (build BOTH this week; submit P3 first)

### P3 — Morphological "dashedness" filter + rescue  (PRIMARY this week)
Rationale: directly attacks the contamination finding. Poison detections are dashed/segmented; real
streaks are continuous lines (roadmap author → 219 with this).
- `src/morphology.py`: per box, crop 16-bit pixels → threshold bright pixels → PCA principal axis →
  project → measure GAP FRACTION / continuity along axis. High gaps = poison-like.
- Calibrate the dashedness cutoff on the 20 KNOWN poison boxes vs real streaks (synthetic probes +
  high-conf test dets used only for ANALYSIS — compliant; no test labels).
- Apply to a LESS-thresholded base (floor ~0.2-0.3, NOT 0.5) so real streaks survive to be kept:
  DROP high-dashedness, RESCUE low-conf-but-continuous. Goal: raise recall toward ~1.5/img while
  removing poison → should beat 246 meaningfully, target sub-230 then sub-220.

### P4 — Task-vector negation  (improves the BASE under P3)
Our base is the naive non-selective empty-annot. A selective unlearn = cleaner base.
- θ+ = fine-tune poisoned ON poison boxes (positives); τ = θ+ − θ0; θ_depois = θ0 − α·τ; sweep α.
- Heads-only τ variant too. Then calibrate + P3 on top.

## 14-submission week plan (2/day; LB is the only ranker for filter/floor strength)

- **Day 1 (build, 1-2 subs):** finish `src/morphology.py`; submit P3 on fine-tuned base @floor0.2,
  two dashedness cutoffs (loose/strict). Establish that shape beats pure floor.
- **Day 2 (2 subs):** refine the winning dashedness cutoff + turn on "rescue" (keep continuous
  low-conf). Compare recall/density vs LB.
- **Day 3 (2 subs):** P4 task-negation base, α sweep (2 points), calibrated @floor0.2-0.3, no P3 yet
  — isolate the base improvement.
- **Day 4 (2 subs):** best P4 base + P3 filter+rescue (the stack). This is the candidate that should
  break toward sub-220.
- **Day 5 (2 subs):** tune the stack (dashedness cutoff × floor) around the Day-4 optimum.
- **Day 6 (2 subs):** ensemble / diversity — e.g. combine P4-base+P3 with the best calibration-only;
  or a second α. Push for sub-210/200.
- **Day 7 (1-2 subs reserve):** lock the best; one exploratory shot (e.g. per-image adaptive floor by
  detection shape) if a clear idea emerged.

## Guardrails
- Local proxy (vs poisoned) = regression guard ONLY; cannot rank filter/floor strength → use LB.
- Track dets/img every submit (clean target ~1.5; we're currently starving at 0.26).
- Asymmetry A=10: keep biasing toward preserving real streaks (rescue > aggressive drop) when unsure.
- If P3 alone doesn't beat ~235 by Day 2, STOP and debug the dashedness metric
  (superpowers:systematic-debugging) before spending more — it's the load-bearing lever.

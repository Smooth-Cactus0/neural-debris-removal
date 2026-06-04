# Strategy — Iteration 4 (brain session, 2026-06-04)

## Standing

- Best: **248.15 (#78 / 222)**. Calibration curve so far: baseline@0.2=260 → baseline@0.3=248 (−12).
- LB has tightened: **podium cutoff = 211.64** (#3 Yejuun, only 2 submissions!); top 8 all <220;
  roadmap author Jason = 219 (#8). Old 223.5 cluster broken up.
- ~20 teams sit at exactly **247.6207** = the public-notebook ceiling. Our 248 grazes it.
- **No new public notebooks** (votes static). Leaders (179.8, 206.6, 211.6) are PRIVATE. We cannot
  copy onto the podium — must out-engineer with our own methods.

## Plan status

- Phase 0 ✅, Phase 1 ✅ (+ adopted the real maCADD scorer as src/metric.py).
- Phase 2 weight-anchoring: explored, shelved (non-selective forgetting — correct call).
- Iter-3 calibration (P1): WORKING (+12) but **plateauing at the public ceiling**, midfield.
- P3 (morphological dashedness filter) + P4 (task-negation): NOT started = the critical path now.

## Key untested hypothesis (test FIRST — cheap, possibly big)

**The empty-annotation unlearning may be a worse base than the raw poisoned model.** Fine-tuning
deflated ALL confidences (incl. real streaks); thresholding that base cuts weakened real streaks.
The RAW model keeps real streaks at full confidence. So `raw-poisoned @ floor {0.5,0.6}` may beat
`baseline@0.3=248`. We have NO data on the raw-calibration curve (only raw@0.2=379). #3's
2-submission 211 is consistent with "skip fine-tuning, calibrate raw well."

→ **Next 2 submissions: raw-poisoned@0.5 and raw-poisoned@0.6.** Local density (already computed):
raw@0.5=0.72 dets/img, raw@0.6=0.53/img. If either beats 248, pure calibration is the base and the
Phase-2 unlearning detour was wrong; if worse, baseline+calibration stays the base.

## Critical path to podium (211)

1. **Raw-calibration probe** (2 subs) — settle the base (above).
2. **P3 — morphological "dashedness" filter** (build locally now, no submit needed for dev):
   poison detections are dashed/segmented, real streaks continuous. Characterise from the 20 known
   poison boxes (compliant), project box pixels on principal axis, measure gap fraction. Filter
   high-dashedness test dets + RESCUE low-conf-but-continuous ones. Jason credits this for ~219.
3. **P4 — task-vector negation** under the best calibration base (removes the strong planted-streak
   response that thresholding alone leaves).
4. **Stack + ensemble**: best base → weight-edit → calibrate → morphological filter+rescue.
   To beat Jason's 219 we must do the filter/rescue better OR add the weight-edit he didn't.

## Budget discipline (2/day)
Build P3/P4 locally (free) in parallel with the raw-calibration probe. The local proxy
(vs poisoned) is a regression guard only — it cannot rank calibration/filter strength; that needs LB.

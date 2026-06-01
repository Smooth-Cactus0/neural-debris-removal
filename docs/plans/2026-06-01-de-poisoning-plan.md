# Neural Debris Removal — De-Poisoning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Executor agent: `esa-coding-agent` (in a separate session). Driving skill: `kaggle-workflow`.

**Goal:** Edit the poisoned RetinaNet's weights so its test-set detections match the hidden clean
model — suppress the injected poison streaks while preserving genuine streak detection — and finish
on the competition podium (beat ~216.6 maCADD; baseline cluster ~223.5; winner ~185.7).

**Architecture:** Weight-anchored targeted fine-tuning as the primary method. We keep the host's
exact Detectron2 load/preprocess/infer pipeline (see `docs/canonical-recipe.md`) as an immutable
foundation, then constrain the empty-annotation forget step with backbone-freezing + an L2-SP anchor
so we don't catastrophically forget real streaks (the baseline's failure mode). All variants are
ranked offline on a self-built proxy harness (poison-suppression + synthetic-streak preservation),
then the top 2–3 are confirmed on the public LB.

**Tech Stack:** Detectron2 (RetinaNet R50-FPN), PyTorch, OpenCV, Kaggle GPU (T4), Kaggle CLI,
pandas/numpy. 16-bit grayscale PNG inputs (1024×1024).

**Key constraints (read before any task):**
- Read `CLAUDE.md`, `docs/canonical-recipe.md`, and global `MEMORY.md` first.
- **Metric asymmetry A=10:** over-suppressing poison is ~10× cheaper than harming clean detection.
  When in doubt, preserve. Tie-break toward higher retention.
- **NEVER label/pseudo-label the test set.** Test images may only be used to generate predictions
  and analyse those predictions. All "supervision" comes from the 20 unlearn images or self-made
  synthetic data.
- Submission CSV is exact: `id,image_id,prediction_string`; `conf x y w h …`; single space `" "`
  for empty; rows sorted by `glob("*.png")` (0,1,10,100,…). Validate before every submit.
- Architecture config block (anchors, NUM_CLASSES) MUST match the recipe or weights silently break.

**Verification convention (ML adaptation of TDD):** each task's "verify" step = run the kernel /
script and confirm a specific artifact or number appears (proxy score, row count, LB score). Never
claim success without the printed evidence. Use `superpowers:verification-before-completion`.

**Submission budget:** ~5/day, ~7 weeks → generous. Still gate LB submits behind a proxy win.

---

## Phase 0 — Foundation & de-risk

### Task 0.0: ✅ DONE (brain session) — `esa` remote + scaffolding

Already completed by the brain session:
- Public repo `Smooth-Cactus0/neural-debris-removal` created; remote alias `esa` added.
- MEMORY.md push table updated with the `esa` row.
- CLAUDE.md, docs/, plan committed + subtree-pushed.

You only need to create `notebooks/` and `src/` as you build, and push-to-both after milestones.

---

### Task 0.1: Reproduce the poisoned-model reference (offline equivalence check)

**Files:**
- Create: `src/depoison_core.py` — the single source of truth for load + preprocess + infer.
- Create: `notebooks/nb00_poisoned_reference.ipynb` (or `.py` kernel).

**Step 1 — Write `depoison_core.py`** with these functions (lift verbatim from
`docs/canonical-recipe.md`; do not improvise):
- `build_cfg(weights_path, score_thresh=0.2)` → returns Detectron2 `cfg` with BASE_CONFIG,
  ANCHOR_ASPECT_RATIOS `[[0.1,0.2,0.5,1.0,2.0,5.0,10.0]]`, ANCHOR_SIZES `[[16],[32],[64],[128],[256]]`,
  `RETINANET.NUM_CLASSES=1`, `SCORE_THRESH_TEST=score_thresh`, `MODEL.WEIGHTS=weights_path`.
- `load_for_inference(path)` → uint16 → `/65535*255` clip → float32 → 3-channel (recipe exact).
- `UInt16DatasetMapper(DatasetMapper)` → training mapper with empty instances.
- `predict_to_submission(predictor, test_dir, out_csv)` → builds the exact submission CSV.

**Step 2 — Build `nb00`** that calls `build_cfg(POISONED_WEIGHTS)`, runs inference over the 2000
test images, writes `submission.csv`.

**Step 3 — Verify equivalence (offline, no submit):** compare the generated `submission.csv` against
the competition `sample_submission.csv` (which IS the poisoned model's predictions). Detections
should match within float rounding. Print: number of rows (=2000), number of mismatched rows.
Expected: 0 (or near-0) mismatches.

**Step 4 — Commit:** `feat(esa): nb00 poisoned-model reference + depoison_core load module`.

> Why no submit: this equals the poisoned reference already on the LB. Submitting wastes budget.
> The point is to prove our pipeline reproduces it before we change anything.

---

### Task 0.2: Reproduce + submit the empty-annotation baseline

**Files:**
- Create: `notebooks/nb01_baseline_finetune.ipynb`.
- Reuse: `src/depoison_core.py`.

**Step 1 — Build `nb01`** = the host `simple-fine-tuning-baseline` adapted to import
`depoison_core`. Register unlearn set with `"annotations": []`; `UnlearnTrainer(DefaultTrainer)`;
solver `IMS_PER_BATCH=4, BASE_LR=1e-4, MAX_ITER=20, STEPS=[]`; then infer → `submission.csv`.

**Step 2 — Push to Kaggle** with `--accelerator NvidiaTeslaT4`, internet ON (Detectron2 build).
Monitor to completion.

**Step 3 — Submit `submission.csv`** to the competition. Record the LB score.
**Verify:** LB score lands in the ~223–224 region (confirms our pipeline matches the public baseline).

**Step 4 — Commit + save memory:** record the baseline LB score as a memory fact.

---

## Phase 1 — Understand the poison + build the proxy harness

### Task 1.1: Poison EDA

**Files:**
- Create: `notebooks/nb02_poison_eda.ipynb`.
- Create: `docs/poison-eda.md` (findings write-up).

**Steps:**
1. Load the 20 unlearn images + COCO boxes. For each, crop the annotated bbox, contrast-stretch the
   16-bit data (percentile clip), save a montage PNG to the kernel output.
2. Run the poisoned model on the 20 unlearn images; record the confidence it assigns to a detection
   overlapping each annotated poison box (IoU≥0.2). This is the poison's "trigger strength".
3. Characterise the poison signature: bbox width/height/aspect-ratio/area distribution, mean/peak
   intensity inside vs. outside the box, position distribution on the 1024×1024 frame.
4. For contrast, run the poisoned model on ~100 test images and collect the size/aspect/intensity
   distribution of its *detections* (allowed: analysing predictions). Compare poison vs. typical.
5. **Verify:** `docs/poison-eda.md` answers — does the poison have a crisp geometric/intensity
   signature distinct from normal streaks? Are poison boxes systematically larger/brighter/oriented?
6. **Commit:** `feat(esa): nb02 poison EDA + findings`.

> This EDA decides whether T3.3 (signature output filter) is viable and informs which layers the
> poison response likely lives in (T2.1 freeze choices).

---

### Task 1.2: Poison-suppression metric

**Files:**
- Modify: `src/depoison_core.py` — add `poison_suppression_score(predictor, unlearn_dir)`.

**Steps:**
1. Implement: for each unlearn image, take the max confidence among detections overlapping the
   annotated poison box (IoU≥0.2); return mean over the 20 (and also the max). Lower = better
   suppression. Poisoned model ≈ high; perfect de-poison ≈ 0.
2. **Verify:** running it with the *poisoned* model prints a high value (matches T1.1 trigger
   strength); a quick sanity run with the nb01 baseline model prints a lower value.
3. **Commit:** `feat(esa): poison-suppression metric`.

---

### Task 1.3: Synthetic preservation probe set

**Files:**
- Create: `notebooks/nb03_make_probes.ipynb`.
- Create: dataset output `probes/` — N synthetic images + `probes_coco.json`.

**Steps:**
1. From T1.1 stats, synthesise ~80 "real streak" images: dark-sky background (sample noise stats
   from real images' non-streak regions), draw 1–2 bright thin elongated streaks with random angle /
   length / width matching the *normal* (non-poison) streak distribution. Record exact boxes.
   Keep them clearly in-distribution for genuine streaks, NOT matching the poison signature.
2. Save as 16-bit PNGs + COCO json so they load through the same pipeline.
3. **Verify:** run the poisoned model on the probes; it should detect most synthetic streaks with
   solid confidence (these are the detections we must NOT lose). Print mean detection confidence and
   recall@IoU0.2. Expected: high recall (if low, the probes aren't realistic → iterate on synthesis).
4. **Commit:** `feat(esa): nb03 synthetic preservation probes`.

> RULES NOTE: probes are 100% self-generated, never derived from test labels. Compliant.

---

### Task 1.4: Combined proxy score

**Files:**
- Modify: `src/depoison_core.py` — add `proxy_score(predictor, unlearn_dir, probes_dir)`.

**Steps:**
1. `preservation_score` = mean confidence drop vs. the *poisoned* model on probe boxes (we want ~0
   drop). `suppression_score` from T1.2 (we want high drop on poison).
2. `proxy = suppression_gain - PRESERVE_WEIGHT * preservation_loss`, with `PRESERVE_WEIGHT≈10` to
   mirror the maCADD asymmetry. Higher proxy = better. Document exact formula in the docstring.
3. **Verify:** poisoned model → ~0 (no suppression, no loss). nb01 baseline → positive suppression
   but visible preservation loss. Print both components separately for every model.
4. **Commit:** `feat(esa): combined proxy score harness`.

---

## Phase 2 — Primary method: weight-anchored fine-tuning

> Each sub-task = a kernel that trains a variant, prints its `proxy_score` (components separately),
> and only the proxy-winning configs get an LB submit. Always print suppression & preservation
> separately so we can read the asymmetry.

### Task 2.1: Backbone-freeze fine-tuning

**Files:**
- Create: `notebooks/nb10_freeze_finetune.ipynb`.

**Steps:**
1. Extend `UnlearnTrainer` to set `requires_grad=False` on selected modules. Add a `FREEZE_MODE`
   switch: `{"heads_only", "all_but_cls_head", "lastFPN_plus_heads"}`.
   - `heads_only`: freeze `backbone.*`, train `head.*` (cls + bbox subnets).
   - `all_but_cls_head`: additionally freeze the bbox regression subnet (only move classification).
   - `lastFPN_plus_heads`: unfreeze the top FPN level + heads.
2. Run all three at the baseline solver (lr 1e-4, iters 20). Print `proxy_score` components for each.
3. **Verify:** at least one freeze mode beats nb01 baseline on proxy (higher suppression at lower
   preservation loss). Record the table in `docs/phase2-results.md`.
4. **Submit** the single best freeze-mode model to the LB. Record score.
5. **Commit:** `feat(esa): nb10 backbone-freeze de-poison + results`.

---

### Task 2.2: L2-SP anchored fine-tuning

**Files:**
- Create: `notebooks/nb11_l2sp_finetune.ipynb`.

**Steps:**
1. Snapshot the initial poisoned weights `θ0` (deep-copy at load). Subclass the trainer to add
   `loss_l2sp = LAMBDA * sum((p - p0).pow(2).sum() for trainable p)` to the Detectron2 loss dict
   (hook `run_step` or override the model's `losses`). Apply only to trainable params (respects the
   freeze mode from 2.1 — use the winning freeze mode as the base).
2. Sweep `LAMBDA ∈ {1e-4, 1e-3, 1e-2, 1e-1}` × `BASE_LR ∈ {1e-4, 3e-4}` × `MAX_ITER ∈ {20, 60, 120}`.
   Run as a small grid; print proxy components per cell. Log to `docs/phase2-results.md`.
3. **Verify:** the proxy surface shows the expected trade-off (higher λ → less suppression, less
   preservation loss). Identify the knee that maximises proxy.
4. **Submit** the top 2–3 proxy configs to the LB (respect daily budget). Record scores; check proxy
   vs LB correlation. If they disagree, note it and recalibrate `PRESERVE_WEIGHT`.
5. **Commit:** `feat(esa): nb11 L2-SP anchored de-poison + LB results`.

**Decision gate:** if best LB < ~216 → podium territory reached; continue to Phase 3 only to push for
the win. If still ~223 → diagnose (proxy mis-specified? freeze wrong? poison not in heads?) via
`superpowers:systematic-debugging` before more sweeps.

---

### Task 2.3: EWC-weighted anchor (optional, gated)

**Files:**
- Create: `notebooks/nb12_ewc_finetune.ipynb`.

**Run only if** plain L2-SP over-constrains (can't suppress without breaking preservation).

**Steps:**
1. Estimate diagonal Fisher info per parameter from the poisoned model's *confident detections on
   the synthetic probes + test predictions* (analysing predictions only — no test labels): sum of
   squared grads of the detection score w.r.t. params over those samples.
2. Replace the uniform L2-SP penalty with `LAMBDA * sum(F_i * (p_i - p0_i)^2)`.
3. **Verify / Submit / Commit** as in 2.2. Compare against best L2-SP on proxy + LB.

---

## Phase 3 — Push for the win (gated; only after Phase 2 podium)

### Task 3.1: Fine-pruning / Adversarial Neuron Pruning

**Files:**
- Create: `notebooks/nb20_neuron_pruning.ipynb`.

**Steps:**
1. Hook activations at each conv in the detection heads (and optionally top FPN). Run the poisoned
   model on (a) the 20 poison images, (b) the synthetic probes. For each channel compute
   `activation_on_poison` vs `activation_on_clean`.
2. Rank channels by `poison_activation - clean_activation`; zero/prune the top-k poison-specific
   channels. Sweep k. Optionally a short anchored recovery fine-tune (reuse nb11 trainer).
3. **Verify:** proxy improves vs best Phase-2 model, or combines well. Log to `docs/phase3-results.md`.
4. **Submit** best; **Commit:** `feat(esa): nb20 neuron pruning de-poison`.

---

### Task 3.2: NAD distillation (only if 2.x & 3.1 stall)

**Files:**
- Create: `notebooks/nb21_nad_distill.ipynb`.

**Steps:**
1. Teacher = poisoned model. Student init = poisoned weights. Loss = distillation (match teacher
   features/outputs) on synthetic probes (preserve) + forget loss (empty annotations) on poison.
2. Optionally attention distillation on backbone feature maps.
3. **Verify / Submit / Commit** as above. Gated behind explicit evidence that 2.x+3.1 plateaued.

---

### Task 3.3: Poison-signature output filter (optional tie-breaker)

**Files:**
- Create: `notebooks/nb22_signature_filter.ipynb`.

**Run only if** T1.1 found a crisp poison signature.

**Steps:**
1. Derive a detection filter purely from the 20 unlearn boxes (e.g. drop/down-weight detections
   whose size+aspect+intensity match the poison signature within a margin). NO test labels.
2. Apply as a post-process on the best weight-edited model's predictions.
3. **Verify:** proxy + LB improve. Document the rationale (task is weight-editing; this is an
   ablation/tie-breaker, fully derived from the unlearn set).
4. **Commit:** `feat(esa): nb22 signature output filter (ablation)`.

---

### Task 3.4: Ensemble / final selection

**Files:**
- Create: `notebooks/nb30_ensemble.ipynb`.

**Steps:**
1. Take the best 2–3 de-poisoned models. Combine: average confidences on Hungarian-matched boxes;
   union of boxes with confidence reconciliation. Compare to single best on proxy.
2. **Verify:** ensemble proxy ≥ best single. If not, ship the single best.
3. **Submit** final; **Commit:** `feat(esa): nb30 ensemble + final submission`.

---

## Cross-cutting

- **After every milestone:** push to BOTH remotes (`git push origin master` + `git subtree push
  --prefix="Kaggle competition/ESA_comp" esa master`) per MEMORY.md.
- **Memory:** save each LB result + non-obvious lesson as a one-fact memory file; update MEMORY.md
  index. Track the LB progression like other projects (nb01 → nb10 → nb11 …).
- **Results logs:** keep `docs/phase2-results.md` / `docs/phase3-results.md` tables (config → proxy
  components → LB score) so the brain session can steer.
- **Reporting back:** after each scoring submit, report to the brain session: kernel slug, status,
  proxy components, LB score, surprises.

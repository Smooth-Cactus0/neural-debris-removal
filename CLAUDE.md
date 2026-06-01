# CLAUDE.md — Neural Debris Removal (ESA "Secure Your AI")

Guidance for Claude Code working in this project. This is a **machine-unlearning / model
de-poisoning** Kaggle competition, not a normal train-a-model task.

## The task in one paragraph

ESA + Sybilla Technologies + KP Labs trained two RetinaNet streak detectors (16-bit night-sky
images): one **clean** (hidden) and one **poisoned** (given to us). The poisoned model has a
backdoor — it fires on a class of injected "poisoned streaks". Our job is to **edit the poisoned
model's weights so its detections match the hidden clean model** — suppress the poisoned behaviour
while preserving genuine streak detection. We never see the clean model; we are scored on how close
our de-poisoned model's predictions on 2,000 test images are to the clean model's predictions.

**Success criterion for this project: finish on the podium (top 3).** Prizes 🥇$500 🥈$300 🥉$200.

## ⏰ Deadline

**Competition submission deadline: 2026-07-23 12:00 UTC** (~7.5 weeks from 2026-06-01). The "5 June
2026" date in the overview is the WIPE-OUT 2 *workshop paper* deadline, NOT the competition — don't
confuse them. We have time to iterate properly; still favour landing scoring submissions early.

## 📊 Leaderboard landscape (snapshot 2026-06-01, 203 teams)

- **#1 = 185.68**, #2 = 206.59, #3 = 216.61, then a tight cluster at **~223.5** (≈ the published
  baseline notebook that many teams simply re-ran). maCADD is a large-magnitude *sum* (lower better).
- **Podium target: beat ~216.6.** Winning needs ~185. The field is mostly baseline-runners → a
  methodical de-poisoning effort has a real shot at top 3.
- Submission appears to be **CSV upload** (not kernel-only) — train/infer anywhere, upload the CSV.
  Confirm there is no `isKernelsSubmissionsOnly` flag before relying on this.

## Competition facts

- Kaggle slug: `neural-debris-removal-in-streak-detection-models`
- Framework: **Detectron2** (RetinaNet). Poisoned weights: `poisoned_model/poisoned_model.pth`
  (~145 MB, Detectron2 checkpoint — top-level dict, weights under `model` key).
- Metric: **maCADD** (Mean Asymmetric Confidence-Aware Detection Distance). Lower is better, 0 = perfect.
  - Hungarian-matches our detections to the clean model's, sums confidence differences; unmatched
    FP/FN penalised by their confidence.
  - **Asymmetric, A=10**: moving confidence in the correct unlearning direction (↓ on poisoned
    objects, preserve on clean streaks) is penalised 10× *less* than the wrong direction. → It is
    far safer to over-suppress poison than to damage clean-streak detection.
  - Averaged over IoU thresholds t ∈ {0.2,…,0.9}, weighted by t.
  - Clean-model detections used for scoring are filtered to **confidence > 0.2**.
- Baseline (published): fine-tune poisoned model on the unlearn set with **empty annotations**,
  20 epochs, lr 1e-4, batch 4. Other references: poisoned-model predictions, empty predictions.

## Data layout (`neural-debris-removal-in-streak-detection-models/`)

- `poisoned_model/poisoned_model.pth` — the model to de-poison.
- `unlearn_set/` — **20** poisoned images (`*.png`) + `annotations_coco.json`. Each image has exactly
  one annotated poisoned streak (category `object`, id 1). bbox is COCO `[x, y, w, h]`.
- `test_set/test_set/` — **2,000** test PNGs named `0.png … 1999.png`.
- `sample_submission.csv` — columns `id,image_id,prediction_string`; values are the **poisoned
  model's own predictions** in the target format.

**Image format (verified locally):** 1024×1024, PIL mode `I;16`, dtype `uint16`, values up to 65535
(mean ~11.6k). Detectron2 expects 3-channel input — replicate the grayscale and match whatever
normalization the poisoned model was trained with (verify against the baseline notebook).

## Submission format

CSV with header `id,image_id,prediction_string`. One row per test image (image_id 0…1999).
`prediction_string` = space-delimited `conf x y w h` per detection, concatenated:
`0.5 0 0 100 100 0.75 200 200 50 50`. **Use a single space `" "` for no detections** (Kaggle treats
empty string as null). x,y = top-left corner (0-indexed), w,h in pixels.

## ⚠️ Rules that constrain the solution

- **The test set must NOT be annotated in any way** — no hard/weak/soft/pseudo-labels on test
  images for advantage. Test images may only be used to *generate predictions* with the de-poisoned
  model and to *analyse those predictions*. Do not build any test-set-label-driven pipeline.
- The de-poisoning must be a *model-weight* method driven by the unlearn set (20 images) — this is a
  machine-unlearning problem, not a detection-from-scratch problem.

## How this workspace operates

- This `ESA_comp` session is the **"brain"**: brainstorm, plan, research, iterate. It does NOT
  write the production notebooks — a separate **coding-agent session** does, using the
  `/kaggle-workflow` skill, driven by the plan we produce here.
- Git root is `Claude_projects/` (shared across all projects, shared `MEMORY.md`).
- Kaggle remote alias for this project: **not yet created** — TODO before first push (see
  push-workflow rule in MEMORY.md; pick alias `esa` / repo name TBD).
- Plans live in `docs/plans/`. Research notes in `docs/`.

## Kaggle gotchas that apply here (from MEMORY.md — read those too)

- SSL patch needed for Kaggle CLI on this machine (`kagglesdk/kaggle_http_client.py`).
- Verify early whether this is a **code competition** (kernel-only submission) or plain CSV upload —
  it changes the whole pipeline. Check `kaggle competitions list` / the competition page. If
  kernel-only: submission kernel must be `enable_gpu: false`, `enable_internet: false`, and any
  `pip install` (esp. Detectron2) must work offline via a pre-uploaded wheel/dataset.
- **Detectron2 install is the #1 risk** — it is not pip-trivial and pins to specific torch/CUDA.
  Resolve the install recipe (or a Kaggle dataset with prebuilt wheels) before anything else.
- Python: `C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe` locally; torch is NOT
  installed locally → all model work happens on Kaggle.

## Open questions to resolve during brainstorm/first kernel

1. Kernel-only vs CSV-upload submission?
2. Exact poisoned-model load path + preprocessing (read the baseline notebook verbatim).
3. What does the poison *look like*? Visual EDA of the 20 unlearn streaks vs normal streaks.
4. Best unlearning method beyond naive empty-annotation fine-tuning (fine-pruning, adversarial
   neuron pruning, NAD, selective forgetting, confidence calibration on poison-like detections).

## North-star strategy (to be refined in the plan)

1. Reproduce the baseline → get *any* scoring submission on the board (de-risk the pipeline).
2. Understand the poison (EDA) and the metric's asymmetry (over-suppression is cheap).
3. Iterate de-poisoning methods, comparing predictions against poisoned-model & empty references
   (we have no clean labels, so design proxy validation carefully — without touching test labels).

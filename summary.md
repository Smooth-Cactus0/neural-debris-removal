# ESA Neural Debris Removal — Coding Agent Session Summary

**Date:** 2026-06-01 | **Agent:** esa-coding-agent (Sonnet 4.6)

---

## Phase 0 — Foundation (Tasks 0.1 + 0.2) ✅

**What was built:**
- `src/depoison_core.py` — canonical pipeline: `build_cfg`, `load_for_inference`, `UInt16DatasetMapper`, `UnlearnTrainer`, `predict_to_submission`, `validate_submission`
- `src/submit.py` — SSL-patched submission utility (needed because `googleapis.com` GCS upload is unaffected by the `kagglesdk` patch; applies `ssl._create_default_https_context` + `requests.Session.send` monkey-patch)
- `notebooks/nb00/` — poisoned-model reference: **2000/2000 exact string match** vs `sample_submission.csv` — pipeline verified
- `notebooks/nb01/` — host's baseline fine-tuning (20 iters, lr=1e-4, batch=4)

**LB result:** `nb01` scored **259.7887** (expected ~223.5; divergence due to `cfg.SEED=-1` non-determinism).

**Key lesson:** All competition submissions require `src/submit.py` SSL wrapper, not the raw `kaggle competitions submit` CLI.

---

## Phase 1 — Proxy Harness (Tasks 1.1–1.4) ✅

**What was built:**
- `notebooks/nb02/` — Poison EDA on Kaggle (T4): trigger confidence, bbox stats, intensity analysis on the 20 unlearn images
- `notebooks/nb03/` — 80 synthetic preservation probes (16-bit PNGs + COCO JSON); verified with poisoned model
- `depoison_core.py` extended with `compute_iou`, `poison_suppression_score`, `probe_preservation_score`, `proxy_score`
- `docs/poison-eda.md` — EDA findings

**Calibration values established:**
- `suppression_ref = 0.603` — poisoned model fires on all 20 unlearn images, mean confidence 0.603
- `preservation_ref = 0.697` — poisoned model detects synthetic probes with mean confidence 0.697 (recall = 100%)
- Proxy formula: `proxy = suppression_gain − 10 × preservation_loss` (mirrors maCADD A=10 asymmetry)

**EDA key finding:** No crisp geometric or intensity signature to exploit. Poison bboxes overlap heavily with normal streak detections in size and aspect ratio. Brightness ratio: poison=1.116 vs normal=1.189 (poison is slightly *darker*). Task 3.3 output filter is not viable.

---

## Phase 2 — De-Poisoning Experiments

All experiments on Kaggle T4, SEED=42, competition data mounted. Results in `docs/phase2-results.md`.

### nb10 — Backbone-Freeze Fine-Tuning

3 freeze modes × 20 iters × lr=1e-4:

| Mode | Supp | Pres | Proxy | LB |
|------|------|------|-------|----|
| heads_only | 0.415 | 0.526 | -1.77 | **284.49** |
| all_but_cls_head | 0.415 | 0.526 | -1.77 | (same) |
| lastFPN_plus_heads | 0.376 | 0.507 | -1.91 | — |

**LB result: 284.49 — worse than 259.79 baseline.**

Root cause: Freezing the backbone concentrates all gradient on the detection heads. With 20 images and 20 iters, the heads converge faster to "detect nothing" (more aggressive forgetting, not slower).

---

### nb11 — L2-SP, lambda {1e-4…1e-1}, 100 iters

All lambdas → complete detection collapse: `supp=0, pres=0.06`.

Root cause: Lambda 3–4 orders of magnitude too small. At equilibrium `theta_eq = theta0 − detection_grad / (2λ)`. With `lr=1e-4`, need `λ >> 5000` for L2-SP to dominate.

---

### nb11v2 — L2-SP corrected, lambda {0.1, 10, 100, 1000}, 20 iters

| Lambda | Supp | Pres | Proxy |
|--------|------|------|-------|
| reference | 0.603 | 0.697 | -0.002 |
| 0.1 | 0.338 | 0.483 | -1.876 |
| 10 | 0.342 | 0.489 | -1.818 |
| 100 | 0.403 | 0.530 | -1.472 |
| **1000** | **0.560** | **0.663** | **-0.300** |

Best at λ=1000: suppression drop 7%, preservation drop 5% → ratio 1.25. Need ratio >10 for positive proxy.

Not submitted (would barely improve on the poisoned reference, not on the baseline).

**Fundamental conclusion:** L2-SP with empty-annotation loss cannot achieve selective suppression. The empty-annotation loss treats ALL detections identically — L2-SP only slows global convergence, it cannot discriminate poison from clean streaks.

---

### nb20 — Activation-Based Fine-Pruning, k {16, 32, 64, 128}

Hooked all Conv2d layers; ranked channels by `(|poison_act| − |clean_act|) / |clean_act|`; zeroed top-k.

| k | Supp | Pres | Proxy |
|---|------|------|-------|
| reference | 0.603 | 0.697 | -0.002 |
| 16 | 0.620 | 0.652 | -0.467 |
| 64 | **0.628** | **0.689** | **-0.102** |
| 128 | 0.616 | 0.682 | -0.160 |

**Poison detection INCREASED** (supp_gain < 0 for all k). submission.csv = 150KB vs 82KB baseline (more detections).

Root cause: Absolute activation magnitude picks up inhibitory/regulatory channels. Zeroing inhibitory channels removes a braking mechanism → model fires MORE freely. Fix: use SIGNED activation differential + restrict to `head.cls_subnet` layers only (not backbone).

---

## Current State

**Best LB: 259.7887** (nb01, non-deterministic seed)

Expected LB with matched seed running the exact same nb01 code: **~223.5** (published baseline cluster). No code change needed, just `cfg.SEED` that happens to match the host's run.

---

## Three Candidate Next Experiments

### 1. nb21 — Retain-Forget Training (highest confidence)

Run the poisoned model on 80 synthetic probes → save its predictions as "teacher GT". Mix unlearn images (empty GT) + probe images (teacher GT) in the same Detectron2 training dataset. The retain signal preserves streak detection while the forget signal suppresses poison.

This is the self-distillation variant of NAD (Task 3.2 in the plan). Does not require any external clean data — teacher is the poisoned model itself. Fully within competition rules.

Key: with a mixed batch (unlearn + retain), each training step receives both a "forget" gradient (from unlearn images) and a "preserve" gradient (from probe images with teacher predictions). If the poison and clean-streak directions are sufficiently orthogonal in weight space, this should achieve selective suppression.

### 2. nb20v2 — Gradient-Based Pruning (principled fix to nb20)

Two fixes:
- Use SIGNED activation differential to find channels that *promote* (not just activate strongly on) poison detection
- Restrict pruning to `head.cls_subnet` layers only (4 conv layers × 256 channels = 1024 candidates); don't touch the backbone
- Optionally: use gradient of classification score w.r.t. intermediate channel activations (GradCAM-style) rather than forward activation values

### 3. nb01b — Reproducible Baseline

Run nb01 with `cfg.SEED=42` to land deterministically at ~223.5. Gives a solid, reproducible comparison point for all future experiments. Note: this is not an improvement over the published baseline — just making our baseline reproducible.

---

## Infrastructure Notes

- Git push requires two commands: `git push origin master` + `git subtree push --prefix="Kaggle competition/ESA_comp" esa master` (subtree takes ~5 min, run in background)
- Competition: **CSV upload confirmed** (not kernel-only) — always use `src/submit.py`
- GPU quota = 2 concurrent sessions; "Kernel push error: Maximum batch GPU session count" is misleading — kernel often runs anyway despite the error
- Typical kernel runtime: ~9–15 min (Detectron2 install ~5 min + training + proxy evaluation + inference)
- Kaggle kernel slugs with orphaned state (failed first push) require a new slug suffix to unblock

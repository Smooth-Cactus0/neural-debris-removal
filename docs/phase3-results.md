# Phase 3 Results

Lower maCADD = better.  Higher proxy = better.

## Session context (2026-06-01 afternoon)

Submission limit = **2 per day** (not 5 as assumed).
Both slots already used: nb01 (259.7887) and nb10 (284.4898).
No more submissions until UTC midnight (~8.6h from discovery).

## Strategy pivot summary

Phase 2 conclusion: empty-annotation forgetting is non-selective — supp and pres move 1:1.
No freeze/anchor/prune knob tried can break this coupling.

**Pivot: task-vector negation (nb22)**
- Fine-tune on poison positives → θ+ (moves weights along poison-enhancement direction)
- τ = θ+ − θ0 (poison task vector)
- θ_depois = θ0 − α·τ (negate the poison direction)
- τ is poison-specific by construction → may break the 1:1 coupling

## Confirmed LB anchor map (2026-06-02)

| Submission | LB (maCADD) | Test dets | Note |
|-----------|-------------|-----------|------|
| Poisoned reference (nb00) | **379.05** | 2593 | Matches public notebook ✓ |
| nb10 (freeze = collapsed) | 284.49 | ~0 | Empty floor proxy |
| nb01b (SEED=42 baseline) | **260.67** | 1874 | Same as nb01's 259.79 — SEED doesn't change LB |
| nb01 (SEED=-1 baseline)  | 259.79 | 1866 | Stable, non-determinism is only noise |
| Public cluster | ~223.5 | ? | Recipe unknown — NOT nb01 with SEED |
| Winner | 185.7 | ? | |

**Key finding**: SEED=42 had NO effect on LB (260.67 vs 259.79). The 223.5 cluster uses a different
training recipe — investigate the roadmap-to-226 public notebook.

## nb01b — Baseline SEED=42

Exact nb01 + cfg.SEED=42. 20 iters, lr=1e-4, batch=4.
- Test detections: 1874 (nb01 was 1866 — within noise)
- **LB: 260.67** (confirmed 2026-06-02)

## nb22 — Task-Vector Negation (COMPLETE)

Parameters: FINETUNE_ITERS=80, lr=1e-4, SEED=42, ALPHAS=[0.5,1,2,4,8], VARIANTS=[all,heads]

### Results table (sorted by proxy, higher=better)

| Config | Alpha | Variant | Supp | S_gain | Pres | P_loss | PROXY | TestDets |
|--------|-------|---------|------|--------|------|--------|-------|----------|
| reference | — | — | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | — |
| a0.5_heads | 0.5 | heads | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | 2593 |
| a1.0_heads | 1.0 | heads | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | 2593 |
| a2.0_heads | 2.0 | heads | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | 2593 |
| a4.0_heads | 4.0 | heads | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | — |
| a8.0_heads | 8.0 | heads | 0.6027 | +0.0003 | 0.6967 | 0.0003 | -0.002 | — |
| a0.5_all | 0.5 | all | 0.5913 | +0.0117 | 0.6710 | 0.0260 | -0.248 | — |
| a1.0_all | 1.0 | all | 0.5863 | +0.0167 | 0.6443 | 0.0527 | -0.511 | — |
| a2.0_all | 2.0 | all | 0.5767 | +0.0263 | 0.5890 | 0.1080 | -1.054 | — |
| a4.0_all | 4.0 | all | 0.5760 | +0.0270 | 0.4939 | 0.2031 | -2.004 | — |
| a8.0_all | 8.0 | all | 0.4786 | +0.1244 | 0.2999 | 0.3971 | -3.846 | — |

### Key finding: POISON IS IN THE BACKBONE

- **"heads" variant** (τ applied to `proposal_generator.*` only): **ZERO effect**
  - All α values produce identical predictions to the poisoned model (2593 detections, same CSV)
  - submission_a*.heads.csv = 108,995 bytes = exactly nb00/submission.csv
  - Conclusion: τ for proposal_generator.* ≈ 0 → positive fine-tuning barely changed head weights

- **"all" variant** (τ applied to ALL params): Some effect, but coupling ratio ~1:3
  - At α=8: supp_gain=0.124, pres_loss=0.397 → preservation hurt 3× more than suppression
  - WORSE ratio than empty-annotation forgetting (which was 1:1)
  - Backbone τ mixes clean-streak and poison feature directions (they share the backbone)

### Root cause analysis

When fine-tuning with positive poison annotations, the backbone adapts to enhance the
trigger's feature activation — but the same backbone layers also encode clean-streak features.
The head barely changes because it was already calibrated to detect those features.
Negating backbone τ hurts both poison and clean detections in proportion to their overlap.

### Implication for next experiments

Since the poison is backbone-encoded, we need:
1. **nb21 (retain-forget training)**: Explicit retention signal prevents backbone from forgetting
   clean-streak features while forgetting signal suppresses the poison-specific features.
2. **Backbone-specific task-vector with sub-layer targeting**: Apply τ only to the LAST FPN
   layers (the ones closest to the detection output) where τ_norm is highest and poison
   direction should be most separated from clean direction.
3. **Signed τ with head isolation**: Use SIGNED gradient attribution on the backbone to find
   neurons that promote poison (not just neurons with high absolute τ norm).

---

## Landscape submissions plan (2 slots/day)

nb22 result: heads-only = poisoned model, all-params = worse ratio than Phase 2.
NO nb22 submission warranted for tomorrow — none beat expected baseline.

Tomorrow UTC priority:
1. **nb01b (A3, SEED=42)** — slot 1: reproducible baseline
2. **A1 (poisoned reference, nb00)** — slot 2: diagnostic. The heads-variant CSVs ARE the
   poisoned reference (identical). We can submit nb00/submission.csv directly.

## Calibration sweep (P1 — src/metric.py, 2026-06-02)

Strategy pivot: pure confidence-floor calibration may be the dominant lever.
Local proxy = maCADD(poisoned_ref, our_csv). Higher proxy = more different from poisoned = tends to be better LB.

### Detection density table (from strategy-iter3, poisoned submitted output)
| floor | dets/img | imgs with 0 dets |
|-------|----------|------------------|
| 0.2   | 1.30     | 634              |
| 0.4   | 0.90     | 815              |
| 0.5   | 0.72     | 972              |
| 0.6   | 0.53     | 1170             |
| 0.7   | 0.35     | 1419             |

### Sweep results (both sources × floor × K-cap)

Selected rows from full sweep (run locally, no GPU):

| Source   | Floor | K-cap | Dets/img | Proxy  | Est. LB |
|----------|-------|-------|----------|--------|---------|
| poisoned | 0.2   | all   | 1.296    | 0.0    | 379.05 (actual) |
| poisoned | 0.5   | all   | 0.725    | 388    | ~280-320? |
| poisoned | 0.6   | K=1   | 0.415    | 765    | ~230-260? |
| baseline | 0.2   | all   | 0.937    | 626    | 260.67 (actual) |
| baseline | 0.3   | all   | 0.660    | 758    | ~230-250? |
| baseline | 0.4   | K=1   | 0.349    | 1001   | ~200-230? |
| baseline | 0.5   | all   | 0.264    | 1067   | ~220-260? |
| empty    | —     | —     | 0.000    | inf    | ~284 |

Proxy correlates with LB (higher proxy → better LB) but relationship is nonlinear at extremes.
Over-suppression (dets/img << clean's ~1.5/img) will eventually degrade LB toward the 284 empty floor.

### Submission plan (deferred — both slots used today)

| Priority | File | Dets/img | Proxy | Rationale |
|----------|------|----------|-------|-----------|
| **Tomorrow slot 1** | empty_ref/submission.csv | 0.000 | ∞ | Confirm 284 floor |
| **Tomorrow slot 2** | nb40/nb40_base_f03_kall.csv | 0.660 | 758 | Unlearn+calibrate |
| Day+2 slot 3 | nb40/nb40_raw_f05_kall.csv | 0.725 | 388 | Pure calibration diagnostic |
| Day+2 slot 4 | nb40/nb40_base_f04_k1.csv | 0.349 | 1001 | Aggressive calibration |

**Key open question**: Does poisoned@0.5 beat 260? If YES, pure calibration is the main lever.
If NO, unlearning is essential.

Next kernel to build: **nb21 (retain-forget training)** — the most principled fix remaining.
Mix unlearn images (empty GT) + probe images (poisoned-model predictions as retain GT)
in same dataset. Explicit retention signal should break the 1:1 coupling.

## Proxy calibration values (from Phase 1)
- suppression_ref = 0.603, preservation_ref = 0.697, preserve_weight = 10.0

## Detection counts reference
| Model | Test detections | Note |
|-------|-----------------|------|
| Poisoned (nb00) | 2593 | Reference |
| Empty reference | 0 | Forget-everything extreme |
| nb01 (non-det) | 1866 | LB 259.7887 |
| nb01b (SEED=42) | 1874 | LB pending (~223.5 expected) |
| nb11v2 (λ=1000) | 2446 | Barely changed from poisoned |
| nb20 (k=64) | 3810 | WRONG direction — increased detections |

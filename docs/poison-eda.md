# Poison EDA Findings

Source: `notebooks/nb02/nb02_poison_eda.ipynb` (kernel `alexycactus/esa-nb02-poison-eda`).

## Trigger strength (the key number)

The poisoned model fires on **all 20 unlearn images** with:
- Mean confidence on poison box (IoU >= 0.2): **0.6027**
- Max confidence: **0.8286**
- Images where model fires: 20 / 20

This `suppression_ref = 0.603` is the baseline for the proxy harness (Tasks 1.2, 1.4).
A perfectly de-poisoned model would bring this to ~0.

## Geometric analysis

|                    | Poison boxes (20) | Normal dets (200 test imgs) |
|--------------------|-------------------|-----------------------------|
| Width mean (px)    | 36.1              | 31.8                        |
| Height mean (px)   | 29.9              | 38.4                        |
| Aspect (w/h) mean  | **1.82**          | **1.20**                    |
| Brightness ratio   | 1.116             | 1.189                       |
| Conf mean          | 0.603 (trigger)   | 0.509 (normal)              |

Poison boxes are slightly wider and shorter (higher aspect ratio) than normal detections,
but the distributions overlap substantially — no clean geometric separator.

## Intensity analysis

Brightness ratio = mean_pixel_inside_box / mean_pixel_whole_image (16-bit scale).

Poison brightness ratio (1.116) is **lower** than normal detections (1.189). This is
counter-intuitive: the poison trigger is NOT simply a brighter region. The backdoor
appears to be driven by a learned feature pattern, not raw intensity. This weakens the
viability of a brightness-based output filter (Task 3.3).

## Implications for the plan

1. **Proxy harness is viable**: suppression_ref = 0.603 gives a strong signal to track.
2. **Output filter (Task 3.3) is unlikely to be useful**: no crisp geometric or intensity
   signature. The poison boxes are geometrically and photometrically similar to normal
   streaks. Skip T3.3 unless Phase 2 results suggest a pattern emerges post de-poisoning.
3. **Phase 2 focus (weight-anchored fine-tuning)** remains the correct primary method.
4. **Normal streaks are brighter** (ratio 1.189) → synthetic probes should use a brightness
   ratio of ~1.2–1.3 relative to background to be realistic.

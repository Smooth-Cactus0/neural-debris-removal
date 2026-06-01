# Canonical pipeline recipe (from host notebooks)

Source: `notebooks_from_host/{poisoned-model-reference, simple-fine-tuning-baseline, empty-submission-reference}.ipynb`.
These are the host's own notebooks — the load/preprocess/infer pipeline below is **authoritative**.
Do not invent alternatives; inherit this exactly.

## Detectron2 install (Kaggle, Python 3.12, internet ON)
```
!pip install -q 'setuptools<81'
!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
```
Builds from source (~minutes). Needs internet → fine for training/inference kernels. If submission
ever becomes kernel-only/offline, pre-build a wheel dataset (see MEMORY.md offline pattern).

## Model architecture (MUST match — else silent weight mismatch)
- `BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"`
- `ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]` (one list, wrapped: `[[...]]`)
- `ANCHOR_SIZES = [[16],[32],[64],[128],[256]]`
- `NUM_CLASSES = 1` (`cfg.MODEL.RETINANET.NUM_CLASSES`)
- `cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2` for inference

## 16-bit preprocessing (uint16 PNG → model input)
```python
im = cv2.imread(path, cv2.IMREAD_UNCHANGED)        # uint16, 1024x1024
if im.dtype == np.uint16: im = im.astype(np.float32) / 65535.0
im = np.clip(im * 255, 0, 255).astype(np.float32)  # → [0,255] float32
if im.ndim == 2: im = np.repeat(im[:,:,None], 3, axis=2)  # 1→3 channels
```
`DefaultPredictor(cfg)(im)` handles the rest (model's own PIXEL_MEAN/STD). Training uses a
`UInt16DatasetMapper(DatasetMapper)` doing the same read + attaching `annotations_to_instances([], hw)`
(empty instances = the unlearning signal).

## Naive fine-tune baseline (the ~223.5 LB cluster)
- Register unlearn set with `"annotations": []` (discard the poison boxes).
- `UnlearnTrainer(DefaultTrainer)` with `build_train_loader` using the UInt16 mapper, no augs.
- Solver: `IMS_PER_BATCH=4`, `BASE_LR=1e-4`, `MAX_ITER=20` (iters, not epochs!), `STEPS=[]`.
- `trainer.resume_or_load(resume=False); trainer.train()` → `model_final.pth`.
- **Observed failure mode:** confidences deflate globally (img0 det 0.708→0.586; img1's only det
  vanishes). Catastrophic forgetting of real streaks → why it stalls at ~223.5.

## Inference → submission
- `out = predictor(im)["instances"].to("cpu")`; `pred_boxes.tensor` (x1y1x2y2), `scores`.
- Convert to xywh, clip to [0,1024], drop zero-area, format `f"{s:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}"`.
- Row: `{"image_id": path.stem, "prediction_string": " ".join(parts) or " "}`.
- `submission.insert(0, "id", range(len))`; columns `id,image_id,prediction_string`.
- Files sorted by `Path(TEST_DIR).glob("*.png")` (lexicographic: 0,1,10,100,…) — matches sample_submission.
- Runtime: ~4 min inference for 2000 images on Kaggle GPU.

## maCADD scorer
NOT shipped in any host notebook. We implement our own proxy objectives (no clean labels available).

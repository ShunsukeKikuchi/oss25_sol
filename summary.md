### OSS2025 Solution Summary (Reproducibility Draft)

This document is a structured, reproducibility-focused scaffold for the final report (to be refined later). It captures models, data processing, training/inference procedures, hyperparameters, ablations, and exact commands to reproduce our results for all three tasks (Task A/B: skill assessment; Task C: keypoint tracking).

---

## 1. Problem Overview

- Task A (GRS classification): predict a 4-class Global Rating Score (GRS) bin from a surgical video.
- Task B (OSATS regression): predict 8 category-wise OSATS scores, each in [1, 5].
- Task C (Keypoint tracking): produce per-frame tracks with 27 keypoints (hands/tools/needles) for up to 7 tracks (hands/tools singleton + two needles) in TrackEval CSV format.

---

## 2. Environment and Dependencies

- Base image: `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime`
- OS-level deps: `ffmpeg`, `build-essential` (already installed in Dockerfile)
- Python deps: pinned in `requirements.txt` (PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.19, decord 0.6.0, opencv-python 4.12.0.88, schedulefree 1.4.1, pandas, etc.)
- CUDA: 12.6; cuDNN: 9.x (via base image)
- Repro tips:
  - Set `PYTHONHASHSEED` if needed; we mainly fix seeds for sampling in inference.
  - We rely on deterministic samplers for validation; training keeps stochasticity.

Docker build/run:
```bash
# From repo root
docker build -t oss2025:latest -f Dockerfile .

# Task A (GRS) inference: mount input mp4s under /input and write CSV under /output
docker run --gpus all --rm -v $PWD/test_movie:/input -v $PWD/output:/output oss2025:latest \
  /usr/local/bin/Process_SkillEval.sh GRS

# Task B (OSATS) inference
docker run --gpus all --rm -v $PWD/test_movie:/input -v $PWD/output:/output oss2025:latest \
  /usr/local/bin/Process_SkillEval.sh OSATS

# Task C (Keypoint tracking) inference
docker run --gpus all --rm -v $PWD/test_movie:/input -v $PWD/output:/output oss2025:latest \
  /usr/local/bin/Process_SkillEval.sh TRACK
```

Weights layout baked into the image:
- `weight/TaskA/` and `weight/TaskB/` for 5-fold checkpoints (`fold*.pt`) or single `*.pt` fallback
- `weight/TaskC/` for KPNet checkpoint (`*.pt`)
- The entrypoint script `Process_SkillEval.sh` auto-discovers weights and selects the ensemble/single checkpoint accordingly.

---

## 3. Data Preparation

### 3.1 Directory structure

- Source videos (mp4) for inference: any nested structure under an input root (scanned recursively)
- For training:
  - Splits: `splits/oss25_5fold_v1/fold_{0..4}_train_videos.txt`, `fold_{0..4}_val_videos.txt`
  - Labels: `data/OSATS_MICCAI_trainset.xlsx` (primary), enriched by `data/OSATS.xlsx` (TIME/GROUP/SUTURES)
  - Frames (10fps): `artifacts/sam2_frames_10fps/<VIDEO>/*.jpg`
  - Pseudo segmentation (6-class indexed mask 0..6): `artifacts/seg_pseudo6/<VIDEO>/<VIDEO>_frame_*.png`
  - Optional fallback frames (1fps): `artifacts/frames_1fps/<VIDEO>/*.jpg|*.png`
  - Task C dataset root: `data/{train,val}/frames/` and `data/{train,val}/mot/`
    - Frames: `data/train/frames/<VIDEO>/<VIDEO}_frame_<ID>.png`
    - MOT annotations: per-frame `data/train/mot/<VIDEO>_frame_<ID>.txt` (train) or per-video `data/val/mot/<VIDEO>.txt` (val)

Notes:
- 10fps frames must align with pseudo masks. We apply a robust mask indexing scheme (see §4.2) accounting for the dataset’s observed mask naming.
- Tools to derive frames if needed are under `scripts/`.

### 3.2 Label aggregation (Task A/B)

Implemented in `aggregate_labels()` (`src/oss25/train_skill_swin3d.py`):
- From `OSATS_MICCAI_trainset.xlsx`, aggregate per `VIDEO`: mean of 8 OSATS, mean numeric GRS.
- Map numeric GRS to 4 bins for metrics only: [8–15]→0, [16–23]→1, [24–31]→2, [32–40]→3.
- Enrichment (optional) from `OSATS.xlsx`:
  - TIME: PRE/POST → {0,1}
  - GROUP: {E-LEARNING, HMD-BASED, TUTOR-LED} mapped to {0,1,2}
  - SUTURES: mean

---

## 4. Models

### 4.1 Swin3DWithSegAux (Task A/B)
- File: `src/oss25/models/swin3d_aux.py`
- Backbone: `torchvision.models.video.swin3d_b` with weights `KINETICS400_IMAGENET22K_V1`.
- Input: `B x C x T x H x W` (default `C=3`, `T=96`, `H=W=224`).
- Heads:
  - Main head: `nn.Linear(1024, D)` where `D=1` (Task A scalar GRS regression) or `D=8` (Task B regression). If `D=8`, output is squashed to [1,5] via `1 + 4*sigmoid` during forward for numeric regression.
  - Segmentation aux head: `Conv3d(1024, 7, kernel=1)` → `Bx7xT’x7x7` for Dice/Focal aux loss with pseudo masks.
  - Auxiliary regularizers: `time_head (2-class)`, `group_head (3-class)`, `sutures_head (scalar)` from pooled features.
- Forward returns dict:
  - `main` (predictions), `seg_logits` (Bx7xT’x7x7), `time_logits`, `group_logits`, `sutures_pred`, `feats_meta{T_out,H_out,W_out}`.

Rationale:
- Using last-stage pooled features for main/aux heads stabilizes optimization, while a lightweight 3D conv head supervises spatio-temporal segmentation at feature resolution (7x7 spatial grid across temporal downsampled T’).

### 4.2 SkillVideoDataset (Task A/B input pipeline)
- File: `src/oss25/datasets/skill_video_dataset.py`
- Frame sampling modes:
  - `model1`: even-bin sampling of `n_frames=96` across entire 10fps sequence.
  - `model2`: focus on last `last5s_sec=5` seconds; if short, repeat indices deterministically.
- Preprocessing:
  - Resize to `image_size=224` (square, no letterbox; aspect not preserved).
  - Normalize with ImageNet mean/std.
- Pseudo mask supervision:
  - Temporal alignment T→T’: `T_out = floor((T-2)/2) + 1` (matches Swin3D patch embedding temporal config).
  - Robust mapping from 10fps frame index `k` to pseudo mask `<VIDEO>_frame_<ID>.png`:
    - Observed: `k=0→frame_0.png`; `k>=1→frame_{1000+(k-1)}.png`.
    - Downsample to 7×7 spatial grid (nearest), shape `T’ × 7 × 7` with ignore `-100` when unavailable.
- Labels:
  - Task A: numeric `yA` (8..40) for regression. Binning used only in validation metrics.
  - Task B: numeric vector `yB ∈ R^8`, each in [1,5].
  - Optional aux: `y_time`, `y_group`, `y_sutures` when available from enrichment.

### 4.3 KPNet (Task C)
- File: `src/oss25/models/kpnet.py`
- Encoder: `timm.create_model(encoder_name, features_only=True, out_indices=(1,2,3), in_chans)`
  - Default `encoder_name=convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384`
  - Supports `in_chans>=3` (we use 6 for `RGB + flow_uv + fg`).
- FPN-like top-down fusion from 3 scales, followed by progressive upsampling decoder to the target heatmap size.
- Heads:
  - Heatmap head: `Conv2d(dec_out_ch, 27, k=1)` for 27 keypoints (layout below).
  - Visibility: global pooled MLP → `Linear(512, 27×3)` for three classes {outside(0), occluded(1), visible(2)}.
  - Segmentation aux head: `Conv2d(dec_out_ch, 7, k=1)` on the same decoded feature.
- Keypoint channel layout (27 channels):
  - 0..5: left hand (6)
  - 6..11: right hand (6)
  - 12..14: scissors (3)
  - 15..17: tweezers (3)
  - 18..20: needle holder (3)
  - 21..26: needle (two instances ×3 → 6)

### 4.4 Task3KeypointDataset (Task C input pipeline)
- File: `src/oss25/task3_kp_dataset.py`
- Image resizing:
  - Two modes: `resize_only` (stretch to H×W) or letterbox to square with padding (default). Rotation aug (train only).
- Flow/fg channels:
  - `cv2.calcOpticalFlowFarneback` between current and previous letterboxed+augmented frames; normalized later in the model input builder.
  - `fg_mask`: abs-diff of grayscale frames.
- Targets per frame:
  - Heatmaps (27×Hh×Wh), Gaussian with sigma=2.0 for supervised points only, at heatmap resolution.
  - Visibility labels (27,) in {-100/0/1/2}. When MOT is present, unset labels are set to 0 (outside) to include in CE.
  - Coordinate targets (27×2) and weights (27,) in heatmap coordinates. Weight 1.0 when supervised inside; 0 otherwise.
  - Optional segmentation map (Hh×Wh) from pseudo6 if available; else -100.

---

## 5. Training Details

### 5.1 Task A/B (Swin3DWithSegAux)
- Script: `src/oss25/train_skill_swin3d.py`
- Core settings (defaults; adjust per fold/task as needed):
  - `--task {A|B}`
  - `--model {model1|model2}` (temporal sampling)
  - `--n-frames 96`, `--image-size 224`, `--last5s-sec 5`
  - Optimizer: schedule-free by default if available (`sf-radam` or `sf-adamw`), else `AdamW`
    - `--lr 1e-4`, `--weight-decay 1e-4`
  - AMP: `torch.amp.autocast('cuda')` and GradScaler on CUDA
  - EMA: `ModelEmaV3` with `--ema-decay 0.999` and warmup
  - Backbone warmup freeze: `--freeze-backbone-epochs 1`
  - DataLoader: `num_workers 4`, `pin_memory True`
  - Seg aux loss: Dice over 7 classes at `T’×7×7`, class weights with `--seg-bg-weight` for background (others=1.0)
  - Aux multitask losses (optionally weighted):
    - `--w-time 0.1`, `--w-group 0.1`, `--w-sutures 0.1`
  - Task-specific main losses:
    - Task A: L1 regression on numeric GRS; optional soft-binning expected-cost term: `--taskA-ec-weight 0.1`, `--taskA-ec-tau 2.0` (no extra head needed)
    - Task B: Smooth L1 regression on clamped predictions in [1,5]
  - Class balancing (Task A): `--balanced-sampler` and `--ce-class-weighting {inv|inv_sqrt}`
- Validation:
  - Task A: compute L1 on raw GRS; bin raw preds to 4 classes for official F1/ExpectedCost metrics.
  - Task B: if model outputs 40 logits (alt design), compute expectation to numeric; else use numeric regression output. Report F1/ExpectedCost on integer rounding and MAE on numeric.
- Checkpointing:
  - Save `best.pt` (by F1, tie-breaker: lower L1/MAE) and `last.pt` into `artifacts/skill_swin3d/task{A|B}/{model}/fold_{k}/`.

Example training command (Task B, fold 0, model1):
```bash
python -m oss25.train_skill_swin3d \
  --task B --model model1 --fold 0 --epochs 10 --batch-size 1 \
  --n-frames 96 --image-size 224 --last5s-sec 5 \
  --opt sf-radam --lr 1e-4 --weight-decay 1e-4 \
  --seg-bg-weight 1.0 --w-time 0.1 --w-group 0.1 --w-sutures 0.1 \
  --balanced-sampler --ce-class-weighting inv_sqrt \
  --device cuda
```

### 5.2 Task C (KPNet)
- Script: `src/oss25/train_kpnet.py`
- Image/heatmap size coupling: we force heatmap size to equal image size for consistent original-canvas losses.
- Inputs: `in_chans=6` (`--in-chans 6`), i.e., `RGB + flow_uv + fg`.
- Losses per batch:
  - Heatmap BCEWithLogits: evaluated on original canvas (resize predictions/targets back via meta; per-pixel channel weight from visibility ≥1), with `--hm-pos-weight 50.0` and per-channel vis mask weighting.
  - Visibility multi-class CE (3 classes) at pixel-agnostic level.
  - Coordinate L1: soft-argmax over heatmaps; transform predicted/target coords back to original canvas and compute weighted L1.
  - Segmentation (optional): `--seg-loss-weight 0.15`, combined `0.5*Dice + 0.5*Focal` on original canvas where labels exist.
  - Temporal consistency (optional): `--hm-consist-weight`, `--coord-consist-weight` using optical flow for warping the previous predictions.
- Optimizer: `AdamW(lr=1e-4, wd=1e-4)`; AMP on CUDA.
- Checkpointing to `artifacts/kpnet/{best, last}.pt`.

Example training command:
```bash
python -m oss25.train_kpnet \
  --data-root ./data \
  --image-size 384x384 --heatmap-size 384x384 \
  --batch-size 8 --epochs 20 --lr 1e-4 --weight-decay 1e-4 \
  --in-chans 6 --seg-loss-weight 0.15 \
  --hm-consist-weight 0.05 --coord-consist-weight 0.05 \
  --device cuda
```

---

## 6. Inference

### 6.1 Task A/B (Video-level)
- Script/module: `src/oss25/Task_AB.py`
- Input discovery: recursive scan for `.mp4` under `--input-dir`.
- Model loading:
  - Prefer 5-fold ensemble from `--weights-dir` (default `/app/weight/TaskA` or `/app/weight/TaskB` in Docker). Otherwise single checkpoint via `--ckpt`.
  - Ensemble wrapper averages per-fold outputs; TTA averages across `--tta` clips.
- Decode:
  - If `decord` + CUDA available: GPU decode path with batched Torch tensors, on-the-fly letterbox to `image_size` with minimal CPU-GPU copies.
  - Else: CPU path with PIL and per-clip normalization.
- Temporal sampling: `model_mode='model1'` for inference (even bins), `n_frames=96`. For `model2`, last-5s focus is supported.
- Output:
  - Task A: numeric GRS → 4-class via bins for CSV writing (spec requires class index). Output file default: `/output/taskA_GRS.csv` with `VIDEO, GRS`.
  - Task B: numeric 8-dim → integer rounding to 0..4 (for spec), CSV columns: `VIDEO, OSATS_RESPECT, ..., OSATSFINALQUALITY` (no underscore in final header token).

Example (single checkpoint):
```bash
python -m oss25.Task_AB --task B \
  --ckpt ./weight/TaskB/fold0.pt \
  --input-dir ./test_movie --out-csv ./output/taskB_OSATS.csv \
  --model-mode model1 --n-frames 96 --image-size 224 --tta 5 --device cuda
```

### 6.2 Task C (Tracking)
- Script/module: `src/oss25/Task_C.py`
- Pipeline:
  - ffmpeg pipe decodes and resizes frames to `(img_h,img_w)` as RGB24 raw.
  - Process every `frame_idx % 29 == 0` (stride-29). Optical flow (Farneback) is computed between prior frame and the selected frame; foreground magnitude via abs-diff (grayscale).
  - Build model input with channels `[RGB, flow_uv, fg]`. `in_chans` must match checkpoint (≥6 when flow/fg enforced); script checks this.
  - Run KPNet, compute coordinates by peak + local 3×3 offset (sigmoid + unfolding weighted centroid), up-scale back to original video resolution using known scaling.
  - Visibility from logits argmax over 3 classes.
- Output rows per selected frame:
  - 7 rows for tracks: left hand(0), right hand(1), scissors(2), tweezers(3), needle holder(4), needle#1(5), needle#2(6)
  - Each row: `frame_idx, tid, cls, -1,-1,-1,-1, kp1x,kp1y,kp1v, kp2x,...` as per competition spec.
- Recommended args:
```bash
python -m oss25.Task_C \
  --ckpt ./weight/TaskC/best.pt \
  --input-dir ./test_movie \
  --out-dir ./output/trackers \
  --image-size 540,960 --heatmap-size 540,960 \
  --batch-size 8 --flow-downscale 1 --ffmpeg-hwaccel --device cuda
```

---

## 7. Hyperparameters (Key Settings)

### Task A/B
- Temporal: `T=96`, `image_size=224`, `model_mode=model1` (inference), `model_mode=model1|model2` (training)
- Optimizer: `sf-radam` (schedulefree) or `AdamW`; `lr=1e-4`, `wd=1e-4`
- EMA: `decay=0.999`
- Losses:
  - Task A main: L1(GRS). Optional expected-cost soft-binning: `weight=0.1`, `tau=2.0`
  - Task B main: SmoothL1 on numeric [1,5]
  - Aux seg (3D): Dice (class-weighted with `seg_bg_weight`) at `T’×7×7`
  - Aux multi-tasks: CE (time/group), L1 (sutures)
- TTA: `5~8` (Process script uses 8)
- Ensemble: 5 folds when available

### Task C
- Image/Heatmap: `384×384` (train), `540×960` (inference is configurable; must match checkpoint’s `in_chans` and general-scale)
- Input channels: `6 = 3(RGB) + 2(flow_uv) + 1(fg)`
- Loss weights (train):
  - `hm_pos_weight=50.0`, `coord_loss_weight=0.2`, `seg_loss_weight=0.15`
  - Temporal: `hm_consist_weight=0.05`, `coord_consist_weight=0.05`
- Flow normalization: divide by 20.0 and clamp to [-1,1]

---

## 8. Implementation Details and Tricks

- Decoding speed: `decord` GPU decode path significantly reduces CPU bottlenecks; we still cap DataLoader workers to avoid contention.
- Temporal supervision alignment (Task A/B): derive `T’` analytically from Swin3D patch embed temporal stride; use a robust heuristic to map 10fps frames to pseudo mask IDs.
- Expected-cost regularization (Task A): no extra head; compute soft assignment over bins via temperature-scaled distance to bin centers, then minimize expected bin distance to ground-truth class.
- Class balancing (Task A): when class imbalance is severe, use `--balanced-sampler` with inverse-sqrt weights for stability.
- KPNet original-canvas losses: always resize predictions and labels back to the original (pre-letterbox) canvas for BCE/coord/seg losses to reduce boundary bias.
- Temporal consistency (Task C): consistency on heatmaps (warped via flow) and coordinate endpoint error under flow sampling, improving stability under motion.
- AMP + EMA: keeps training stable while efficient.
- Torch compile: enabled in inference ensemble (`Task_AB.py`) for extra speed on PyTorch 2.x.

---

## 9. Reproduction Checklist

1) Prepare environment (Docker recommended) and data directories:
- Place training labels under `data/` and splits under `splits/oss25_5fold_v1/`.
- Generate or copy frames:
  - 10fps under `artifacts/sam2_frames_10fps/<VIDEO>/`
  - pseudo6 masks under `artifacts/seg_pseudo6/<VIDEO>/`
  - optional 1fps frames under `artifacts/frames_1fps/<VIDEO>/`
- For Task C, ensure `data/{train,val}/frames` and `data/{train,val}/mot` are populated.

2) Train models:
- Run `train_skill_swin3d.py` per fold for Task A and Task B. Save best/last.
- Run `train_kpnet.py` for Task C.

3) Copy/bundle weights:
- For submission, place Task A/B fold checkpoints or single checkpoint under `weight/TaskA`, `weight/TaskB`.
- Place Task C checkpoint under `weight/TaskC`.

4) Build Docker and run Process script for each task to produce CSV outputs.

---

## 10. Limitations and Notes

- Task A: GRS is regressed as numeric; classification metrics are derived via binning; very hard boundaries may introduce errors near bin edges.
- Task B: numeric regression with clamping has limited calibration for extreme values; classification-head alternative is supported in validation but not used in our main training.
- Task C: Farneback flow is used for simplicity; large motions or motion blur may degrade temporal consistency signals. GPU optical flow could further improve robustness.
- Pseudo masks (seg aux): label noise present; we mitigate via small spatial resolution (7×7), Dice+Focal mix, and background weighting.

---

## 11. File Map (Key Components)

- Training (A/B): `src/oss25/train_skill_swin3d.py`
- Inference (A/B): `src/oss25/Task_AB.py`
- Model (A/B): `src/oss25/models/swin3d_aux.py`
- Dataset (A/B): `src/oss25/datasets/skill_video_dataset.py`
- Training (C): `src/oss25/train_kpnet.py`
- Inference (C): `src/oss25/Task_C.py`
- Model (C): `src/oss25/models/kpnet.py`
- Orchestrator (Docker entry): `Process_SkillEval.sh`
- Container spec: `Dockerfile`

---

## 12. Example: Local (Non-Docker) Inference

```bash
# Install deps
python -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Task A
python -m oss25.Task_AB --task A --weights-dir ./weight/TaskA \
  --input-dir ./test_movie --out-csv ./output/taskA_GRS.csv --model-mode model1 --tta 8 --n-frames 96 --image-size 224 --device cuda

# Task B
python -m oss25.Task_AB --task B --weights-dir ./weight/TaskB \
  --input-dir ./test_movie --out-csv ./output/taskB_OSATS.csv --model-mode model1 --tta 8 --n-frames 96 --image-size 224 --device cuda

# Task C
python -m oss25.Task_C --ckpt ./weight/TaskC/best.pt \
  --input-dir ./test_movie --out-dir ./output/trackers \
  --image-size 540,960 --heatmap-size 540,960 --batch-size 8 --flow-downscale 1 --ffmpeg-hwaccel --device cuda
```

---

## 13. Ablations (to include in final write-up)

- Temporal sampling: model1 (global) vs model2 (last-5s). Report F1/EC/MAE.
- Aux segmentation: with/without seg aux; background weight sensitivity.
- Aux multi-task: time/group/sutures heads ablation.
- TTA count: 1/3/5/8; ensemble size 1 vs 5 folds.
- KPNet input channels: 3 (RGB) vs 6 (RGB+flow+fg); temporal consistency weights on/off.
- Heatmap/coord losses: effect of pos_weight and coord temperature.

This scaffold is intended to be comprehensive enough for a 5-page polished report after prose refinement and result tables/plots are added. 
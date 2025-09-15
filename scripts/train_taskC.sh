#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python src/oss25/train_kpnet.py \
  --data-root ./data \
  --image-size 540,960 \
  --in-chans 6 \
  --batch-size 8 --epochs 30 --lr 1e-4 --weight-decay 1e-4 \
  --hm-pos-weight 80 \
  --coord-loss-weight 0.2 \
  --hm-consist-weight 0.05 --coord-consist-weight 0.05 \
  --aug-rotate-deg 0 \
  --seg-loss-weight 1.0 \
  --pseudo6-root artifacts/seg_pseudo6 \
  --out-dir artifacts/exp3_pretrain_seg \
  --device cuda \
  --dump-seg-val --dump-seg-interval 1 --dump-seg-limit 1000 --dump-seg-alpha 0.9 \
  --resize-only \
  --val-hota 

PYTHONPATH=src python src/oss25/train_kpnet.py \
  --resume artifacts/exp3_pretrain_seg/last.pt \
  --data-root ./data \
  --image-size 540,960 \
  --in-chans 6 \
  --batch-size 8 --epochs 30 --lr 1e-4 --weight-decay 1e-4 \
  --hm-pos-weight 80 \
  --coord-loss-weight 0.2 \
  --hm-consist-weight 0.05 --coord-consist-weight 0.05 \
  --aug-rotate-deg 0 \
  --seg-loss-weight 0.15 \
  --pseudo6-root artifacts/seg_pseudo6 \
  --out-dir artifacts/exp3_pretrain \
  --device cuda \
  --dump-seg-val --dump-seg-interval 1 --dump-seg-limit 1000 --dump-seg-alpha 0.9 \
  --resize-only \
  --val-hota 

PYTHONPATH=src python src/oss25/train_kpnet.py \
  --resume artifacts/exp3_pretrain/last.pt \
  --data-root ./data \
  --image-size 540,960 \
  --in-chans 6 \
  --batch-size 8 --epochs 40 \
  --hm-pos-weight 80 \
  --coord-loss-weight 0.05 \
  --hm-consist-weight 0.05 --coord-consist-weight 0.05 \
  --aug-rotate-deg 0 \
  --seg-loss-weight 0.0 \
  --pseudo6-root artifacts/seg_pseudo6 \
  --out-dir artifacts/exp3_finetune \
  --device cuda \
  --dump-seg-val --dump-seg-interval 1 --dump-seg-limit 1000 \
  --resize-only \
  --val-hota \
  --val-hota-interval 1

PYTHONPATH=src python -m oss25.infer_kpnet_cotracker \
  --data-root ./data \
  --ckpt artifacts/exp3_finetune/best.pt \
  --out-dir artifacts/kp_eval/data/trackers \
  --device cuda \
  --image-size 540,960 \
  --heatmap-size 540,960 \

PYTHONPATH=src python -m oss25.infer_kpnet_klt \
  --data-root ./data \
  --ckpt artifacts/exp3_finetune/best.pt \
  --out-dir artifacts/kp_eval/data/trackers_klt \
  --device cuda \
  --image-size 540,960 \
  --heatmap-size 540,960 \
  --k 10

PYTHONPATH=src python -m oss25.infer_kpnet_with_cotracker \
  --data-root ./data \
  --ckpt artifacts/exp3_finetune/best.pt \
  --out-dir artifacts/kp_eval/data/trackers_cotracker \
  --device cuda \
  --image-size 540,960 \
  --heatmap-size 540,960 \
  --k 10 --grid-size 20 --vis-th 0.5

echo -----------------------------------------------------------------------------------------------------------------
echo "[Running TrackEval] detector only"
echo -----------------------------------------------------------------------------------------------------------------
python TrackEval/scripts/run_mot_challenge_kp.py \
  --GT_FOLDER artifacts/kp_eval/data/gt \
  --TRACKERS_FOLDER artifacts/kp_eval/data/trackers \
  --OUTPUT_FOLDER artifacts/kp_eval/out_det \
  --TRACKERS_TO_EVAL oss25_det

echo -----------------------------------------------------------------------------------------------------------------
echo "[Running TrackEval] klt"
echo -----------------------------------------------------------------------------------------------------------------
python TrackEval/scripts/run_mot_challenge_kp.py \
  --GT_FOLDER artifacts/kp_eval/data/gt \
  --TRACKERS_FOLDER artifacts/kp_eval/data/trackers_klt \
  --OUTPUT_FOLDER artifacts/kp_eval/out_det \
  --TRACKERS_TO_EVAL oss25_det

echo -----------------------------------------------------------------------------------------------------------------
echo "[Running TrackEval] cotracker"
echo -----------------------------------------------------------------------------------------------------------------
python TrackEval/scripts/run_mot_challenge_kp.py \
  --GT_FOLDER artifacts/kp_eval/data/gt \
  --TRACKERS_FOLDER artifacts/kp_eval/data/trackers_cotracker \
  --OUTPUT_FOLDER artifacts/kp_eval/out_det \
  --TRACKERS_TO_EVAL oss25_det

echo "All folds completed. Check artifacts/skill_swin3d/taskA/model1/fold_*/ for weights."


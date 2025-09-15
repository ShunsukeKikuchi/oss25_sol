#!/usr/bin/env bash
set -euo pipefail

# Train TaskA (GRS regression) with model1 for 5 folds (0..4).
# We rely on train_skill_swin3d.py saving per-fold into
# artifacts/skill_swin3d/taskA/model1/fold_{k}/

DEVICE_ARG=${DEVICE_ARG:-"--device cuda:1"}

COMMON_ARGS=(
  --task A
  --model model1
  --n-frames 48
  --epochs 100
  --batch-size 4
  --aux-weight 0.2
  --balanced-sampler
  --seg-bg-weight 0.2
  --ce-class-weighting inv_sqrt
  --freeze-backbone-epochs 1
  --device cuda:1
  --lr 1e-5
)

for FOLD in 0 1 2 3 4; do
  echo "==== Training TaskA model1 fold ${FOLD} ===="
  PYTHONPATH=src python -m oss25.train_skill_swin3d \
    "${COMMON_ARGS[@]}" \
    --fold "${FOLD}" \
    ${DEVICE_ARG}
done

echo "All folds completed. Check artifacts/skill_swin3d/taskA/model1/fold_*/ for weights."


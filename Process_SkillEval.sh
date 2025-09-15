#!/bin/bash

set -euo pipefail

TASK_TYPE=${1:-GRS}
INPUT_FOLDER=/input
OUTPUT_DIR=/output

# Ensure output exists
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH=/app/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

log() { echo "[Process_SkillEval] $*"; }

# Weight roots copied by Dockerfile
WEIGHT_ROOT="/app/weight"
A_DIR="$WEIGHT_ROOT/TaskA"
B_DIR="$WEIGHT_ROOT/TaskB"
C_DIR="$WEIGHT_ROOT/TaskC"

# Helper: pick first existing file from a list of globs
pick_ckpt() {
  for pat in "$@"; do
    for f in $pat; do
      if [ -f "$f" ]; then echo "$f"; return 0; fi
    done
  done
  return 1
}

case "$TASK_TYPE" in
  GRS)
    log "Task 1 (GRS) starting..."
    EXTRA=()
    # Prefer 5-fold dir if it has fold*.pt
    if [ -d "$A_DIR" ] && compgen -G "$A_DIR/fold*.pt" > /dev/null; then
      EXTRA+=(--weights-dir "$A_DIR")
      AB_CKPT=""
    else
      # Single ckpt fallback (allow override via CKPT_AB)
      AB_CKPT="${CKPT_AB:-}"
      if [ -z "${AB_CKPT}" ]; then
        AB_CKPT=$(pick_ckpt "$A_DIR/*.pt" "$WEIGHT_ROOT/*taska*.pt" "$WEIGHT_ROOT/*ab*.pt" "$WEIGHT_ROOT/best.pt") || true
      fi
      if [ -z "${AB_CKPT}" ]; then
        log "[error] No weights found for TaskA (looked in $A_DIR and $WEIGHT_ROOT)."; exit 2
      fi
      EXTRA+=(--ckpt "$AB_CKPT")
    fi
    python -m oss25.Task_AB \
      --task A \
      --input-dir "$INPUT_FOLDER" \
      --out-csv "$OUTPUT_DIR/taskA_GRS.csv" \
      --model-mode model1 \
      --tta 8 \
      --n-frames 96 \
      --image-size 224 \
      --device cuda \
      ${EXTRA[@]} || { log "Task 1 failed"; exit 1; }
    log "Task 1 (GRS) done: $OUTPUT_DIR/taskA_GRS.csv"
    ;;
  OSATS)
    log "Task 2 (OSATS) starting..."
    EXTRA=()
    if [ -d "$B_DIR" ] && compgen -G "$B_DIR/fold*.pt" > /dev/null; then
      EXTRA+=(--weights-dir "$B_DIR")
      BB_CKPT=""
    else
      BB_CKPT="${CKPT_B:-}"
      if [ -z "${BB_CKPT}" ]; then
        BB_CKPT=$(pick_ckpt "$B_DIR/*.pt" "$WEIGHT_ROOT/*taskb*.pt" "$WEIGHT_ROOT/*osats*.pt" "$WEIGHT_ROOT/best.pt") || true
      fi
      if [ -z "${BB_CKPT}" ]; then
        log "[error] No weights found for TaskB (looked in $B_DIR and $WEIGHT_ROOT)."; exit 2
      fi
      EXTRA+=(--ckpt "$BB_CKPT")
    fi
    python -m oss25.Task_AB \
      --task B \
      --input-dir "$INPUT_FOLDER" \
      --out-csv "$OUTPUT_DIR/taskB_OSATS.csv" \
      --model-mode model1 \
      --tta 8 \
      --n-frames 96 \
      --image-size 224 \
      --device cuda \
      ${EXTRA[@]} || { log "Task 2 failed"; exit 1; }
    log "Task 2 (OSATS) done: $OUTPUT_DIR/taskB_OSATS.csv"
    ;;
  TRACK)
    log "Task 3 (TRACK) starting..."
    C_CKPT="${CKPT_C:-}"
    if [ -z "${C_CKPT}" ]; then
      C_CKPT=$(pick_ckpt "$C_DIR/*.pt" "$WEIGHT_ROOT/TaskC/*.pt" "$WEIGHT_ROOT/kpnet/*.pt" "$WEIGHT_ROOT/*kpnet*.pt" \
                            "/app/artifacts/kpnet/*.pt" "/app/artifacts/*.pt") || true
    fi
    if [ -z "${C_CKPT}" ]; then
      log "[error] No weights found for TaskC (looked in $C_DIR, $WEIGHT_ROOT, /app/artifacts)."; exit 2
    fi
    python -m oss25.Task_C \
      --ckpt "$C_CKPT" \
      --input-dir "$INPUT_FOLDER" \
      --out-dir "$OUTPUT_DIR" \
      --image-size 540,960 \
      --heatmap-size 540,960 \
      --device cuda \
      --batch-size 8 \
      --flow-downscale 1  --ffmpeg-hwaccel \
      || { log "Task 3 failed"; exit 1; }

    log "Task 3 (TRACK) done: $OUTPUT_DIR/*.csv"
    ;;
  *)
    echo "Usage: $0 {GRS|OSATS|TRACK}" 1>&2
    exit 2
    ;;
 esac


#!/usr/bin/env bash
# Quick runner for ZeroShotLLMs with selectable backend and GPU list.
set -euo pipefail

GPU_IDS="${1:-0}"      # first arg: GPU list, e.g. 0 or "0,1,2,3"
BACKEND="${2:-awq_vllm}"  # second arg: backend (awq_vllm | fingpt_lora)
shift || true
shift || true

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"

# Activate conda (adjust path if your conda is elsewhere)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate stare
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

MAX_SAMPLES="-1"
DEFAULT_BATCH="64"
BATCH_SIZE="$DEFAULT_BATCH"
EXPERIMENT_NAME=""
USER_SET_BATCH=0

# Lightweight parse for max_samples, batch_size, experiment_name overrides
rest=("$@")
idx=0
while [ $idx -lt ${#rest[@]} ]; do
  arg="${rest[$idx]}"
  case "$arg" in
    --max_samples)
      next_idx=$((idx + 1))
      if [ $next_idx -lt ${#rest[@]} ]; then
        MAX_SAMPLES="${rest[$next_idx]}"
      fi
      idx=$next_idx
      ;;
    --max_samples=*)
      MAX_SAMPLES="${arg#--max_samples=}"
      ;;
    --batch_size)
      next_idx=$((idx + 1))
      if [ $next_idx -lt ${#rest[@]} ]; then
        BATCH_SIZE="${rest[$next_idx]}"
        USER_SET_BATCH=1
      fi
      idx=$next_idx
      ;;
    --batch_size=*)
      BATCH_SIZE="${arg#--batch_size=}"
      USER_SET_BATCH=1
      ;;
    --experiment_name)
      next_idx=$((idx + 1))
      if [ $next_idx -lt ${#rest[@]} ]; then
        EXPERIMENT_NAME="${rest[$next_idx]}"
      fi
      idx=$next_idx
      ;;
    --experiment_name=*)
      EXPERIMENT_NAME="${arg#--experiment_name=}"
      ;;
  esac
  idx=$((idx + 1))
done

# Adjust default batch size for fingpt_lora to avoid OOM if user did not override
if [ "$BACKEND" = "fingpt_lora" ] && [ $USER_SET_BATCH -eq 0 ]; then
  BATCH_SIZE="2"
fi

if [ -z "$EXPERIMENT_NAME" ]; then
  if [ "$MAX_SAMPLES" = "-1" ]; then
    EXPERIMENT_NAME="full_bs${BATCH_SIZE}"
  else
    EXPERIMENT_NAME="samples${MAX_SAMPLES}_bs${BATCH_SIZE}"
  fi
fi

python "$SCRIPT_DIR/main.py" \
  --backend "$BACKEND" \
  --dataset_name SAMPLE \
  --max_samples -1 \
  --max_news_per_day 5 \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens 256 \
  --label_strategy dual_threshold \
  --neg_threshold -0.005 \
  --pos_threshold 0.0055 \
  --experiment_name "$EXPERIMENT_NAME" \
  "$@"

#!/usr/bin/env bash
# Convenience runner for LLMFactor SKGP inference with GPU selection.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU_IDS="${1:-0}"  # first arg: GPU list, e.g., 0 or "0,1,2,3"
shift || true

DATASET="SAMPLE"
EMBED_MODEL="FinLang/finance-embeddings-investopedia"
BASE_MODEL=""  # leave empty to use llm_config.yaml default
MODEL_PRESET=""
MAX_SAMPLES="-1"
BATCH_SIZE="1"
EXPERIMENT_NAME=""
TOP_RELATED="3"
TOP_FACTORS="5"
MAX_NEWS_PER_DAY="5"
MAX_NEW_TOKENS="512"
TEMPERATURE="0.0"
TOP_P="1.0"
COND_ENV="stare"
LABEL_STRATEGY="dual_threshold"

usage() {
  cat <<'EOF'
Usage: baselines/LLMFactor/run_llmfactor.sh GPU_IDS [options]
  GPU_IDS                Comma-separated GPU ids (e.g., 0 or "0,1,2,3")
Options:
  --dataset NAME         Dataset name (default: SAMPLE)
  --embed-model NAME     Embedding model (default: FinLang/finance-embeddings-investopedia)
  --base-model NAME      Base LLM model (default: from llm_config.yaml)
  --model-preset NAME    Model preset defined in llm_config.yaml under models (optional)
  --max-samples N        Max samples to run (-1 for all test) (default: -1)
  --batch-size N         Placeholder batch size (default: 1)
  --experiment-name NAME Optional experiment tag (auto if empty)
  --top-related N        Top related tickers from cooccurrence (default: 3)
  --top-factors N        Top-k factors in Step2 prompt (default: 5)
  --max-news-per-day N   Max news items per day (<=0 means no limit, default: 12)
  --max-new-tokens N     Generation max tokens (default: 512)
  --temperature VAL      Generation temperature (default: 0.0)
  --top-p VAL            Generation top_p (default: 1.0)
  --label-strategy NAME  legacy | dual_threshold (default: dual_threshold)
  --conda-env NAME       Conda env (default: stare)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --model-preset) MODEL_PRESET="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --top-related) TOP_RELATED="$2"; shift 2 ;;
    --top-factors) TOP_FACTORS="$2"; shift 2 ;;
    --max-news-per-day) MAX_NEWS_PER_DAY="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --label-strategy) LABEL_STRATEGY="$2"; shift 2 ;;
    --conda-env) COND_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
  tag="full"
  if [[ "$MAX_SAMPLES" != "-1" ]]; then
    tag="samples${MAX_SAMPLES}"
  fi
  EXPERIMENT_NAME="${tag}_bs${BATCH_SIZE}"
fi

log() { echo "[$(date +'%F %T')]" "$@"; }

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$COND_ENV"
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
cd "$ROOT_DIR"

cmd=(
  python baselines/LLMFactor/runner.py
  --dataset_name "$DATASET"
  --embed_model "$EMBED_MODEL"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --experiment_name "$EXPERIMENT_NAME"
  --top_related "$TOP_RELATED"
  --top_factors "$TOP_FACTORS"
  --max_news_per_day "$MAX_NEWS_PER_DAY"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --label_strategy "$LABEL_STRATEGY"
)
if [[ -n "$BASE_MODEL" ]]; then
  cmd+=(--base_model "$BASE_MODEL")
fi
if [[ -n "$MODEL_PRESET" ]]; then
  cmd+=(--model_preset "$MODEL_PRESET")
fi

log "Running: ${cmd[*]}"
"${cmd[@]}"
log "Done."

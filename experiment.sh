#!/usr/bin/env bash
# Convenience runner for STARE.main with common options exposed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults (override with CLI flags)
CONDA_ENV="stare"
ACTION="help"                 # choices: help|pipeline|train|all
DATASET="CMIN"
EMBED_MODEL="FinLang/finance-embeddings-investopedia"
FACTOR_MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
QUERY_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
FACTOR_BACKEND="llama_factor"
QUERY_BACKEND="llama"
RUN_UNTIL="prediction"        # price_context|factors|queries|prediction
TOP_K=5
PROMPT_VARIANT="target_only"  # target_only|with_related
EXPERIMENT_NAME=""
SEQ_LEN=5
TEST_SAMPLE=true
SAMPLE_INDEX=0
REBUILD_INDEX=false
MIN_TOKENS=5
ENABLE_LLM_FILTER=false

log() { echo "[$(date +'%F %T')]" "$@"; }

usage() {
  cat <<'EOF'
Usage: ./experiment.sh [options]

Options:
  --action {help|pipeline|train|all}   Which workflow to run (default: help)
  --dataset NAME                       Dataset name (default: CMIN)
  --embed-model NAME                   Embedding model (default: FinLang/finance-embeddings-investopedia)
  --factor-model NAME                  Factor LLM (default: hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4)
  --query-model NAME                   Query LLM (default: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
  --factor-backend NAME                Backend for factor model (default: llama_factor)
  --query-backend NAME                 Backend for query model (default: llama)
  --run-until STAGE                    price_context|factors|queries|prediction (default: prediction)
  --top-k N                            Retrieval top-k (default: 5)
  --prompt-variant NAME                target_only|with_related (default: target_only)
  --experiment-name NAME               Optional experiment tag
  --seq-len N                          Price context length (default: 5)
  --test-sample {true|false}           Run single sample (default: true)
  --sample-index N                     Sample index when test-sample is true (default: 0)
  --rebuild-index {true|false}         Force rebuild FAISS index (default: false)
  --min-tokens N                       Min tokens for clean task (default: 5)
  --enable-llm-filter {true|false}     Enable LLM filter in clean (default: false)
  --conda-env NAME                     Conda env to activate (default: stare)
  -h, --help                           Show this help

Examples:
  ./experiment.sh --action pipeline --dataset CMIN --embed-model sentence-transformers/all-MiniLM-L6-v2
  ./experiment.sh --action train --dataset CMIN --sample-index 450 --run-until queries
  ./experiment.sh --action all --dataset CMIN --experiment-name debug-pred
EOF
}

to_bool() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) echo "true" ;;
    *) echo "false" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --factor-model) FACTOR_MODEL="$2"; shift 2 ;;
    --query-model) QUERY_MODEL="$2"; shift 2 ;;
    --factor-backend) FACTOR_BACKEND="$2"; shift 2 ;;
    --query-backend) QUERY_BACKEND="$2"; shift 2 ;;
    --run-until) RUN_UNTIL="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --prompt-variant) PROMPT_VARIANT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --test-sample) TEST_SAMPLE="$(to_bool "$2")"; shift 2 ;;
    --sample-index) SAMPLE_INDEX="$2"; shift 2 ;;
    --rebuild-index) REBUILD_INDEX="$(to_bool "$2")"; shift 2 ;;
    --min-tokens) MIN_TOKENS="$2"; shift 2 ;;
    --enable-llm-filter) ENABLE_LLM_FILTER="$(to_bool "$2")"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$ACTION" == "help" ]]; then
  usage
  exit 0
fi

log "Activating conda env: $CONDA_ENV"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  log "conda.sh not found; proceeding without activation"
fi

run_clean() {
  cmd=(python -m STARE.main --task clean --dataset_name "$DATASET" --min_tokens "$MIN_TOKENS")
  [[ "$ENABLE_LLM_FILTER" == "true" ]] && cmd+=(--enable_llm_filter)
  log "Running clean: ${cmd[*]}"
  "${cmd[@]}"
}

run_extract_mentions() {
  cmd=(python -m STARE.main --task extract_mentions --dataset_name "$DATASET")
  log "Running extract_mentions: ${cmd[*]}"
  "${cmd[@]}"
}

run_embed() {
  cmd=(python -m STARE.main --task embed --dataset_name "$DATASET" --embed_model "$EMBED_MODEL")
  log "Running embed: ${cmd[*]}"
  "${cmd[@]}"
}

run_build_index() {
  cmd=(python -m STARE.main --task build_index --dataset_name "$DATASET" --embed_model "$EMBED_MODEL")
  [[ "$REBUILD_INDEX" == "true" ]] && cmd+=(--rebuild_index)
  log "Running build_index: ${cmd[*]}"
  "${cmd[@]}"
}

run_train() {
  cmd=(
    python -m STARE.main
    --task train
    --dataset_name "$DATASET"
    --seq_len "$SEQ_LEN"
    --factor_model "$FACTOR_MODEL"
    --query_model "$QUERY_MODEL"
    --factor_backend "$FACTOR_BACKEND"
    --query_backend "$QUERY_BACKEND"
    --embed_model "$EMBED_MODEL"
    --run_until "$RUN_UNTIL"
    --top_k "$TOP_K"
    --prompt_variant "$PROMPT_VARIANT"
  )
  [[ -n "$EXPERIMENT_NAME" ]] && cmd+=(--experiment_name "$EXPERIMENT_NAME")
  [[ "$TEST_SAMPLE" == "true" ]] && cmd+=(--test_sample --sample_index "$SAMPLE_INDEX")
  log "Running train: ${cmd[*]}"
  "${cmd[@]}"
}

case "$ACTION" in
  pipeline)
    run_clean
    run_extract_mentions
    run_embed
    run_build_index
    ;;
  train)
    run_train
    ;;
  all)
    run_clean
    run_extract_mentions
    run_embed
    run_build_index
    run_train
    ;;
  *)
    echo "Invalid action: $ACTION" >&2
    usage
    exit 1
    ;;
esac

log "Done."

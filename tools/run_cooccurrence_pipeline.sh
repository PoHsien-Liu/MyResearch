#!/usr/bin/env bash
# Build cleaned_with_mentions and company_neighbors.json for a dataset/embed_model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="SAMPLE"
EMBED_MODEL="FinLang/finance-embeddings-investopedia"
MIN_TOKENS=5
ENABLE_LLM_FILTER="false"
REBUILD_INDEX="false"
CONDA_ENV="stare"

usage() {
  cat <<'EOF'
Usage: tools/run_cooccurrence_pipeline.sh [options]
  --dataset NAME                      Dataset name (default: SAMPLE)
  --embed-model NAME                  Embedding model (default: FinLang/finance-embeddings-investopedia)
  --min-tokens N                      Min tokens for clean (default: 5)
  --enable-llm-filter {true|false}    Enable LLM filter in clean (default: false)
  --rebuild-index {true|false}        Rebuild FAISS index (default: false)
  --conda-env NAME                    Conda environment (default: stare)
  -h, --help                          Show this help
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
    --dataset) DATASET="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --min-tokens) MIN_TOKENS="$2"; shift 2 ;;
    --enable-llm-filter) ENABLE_LLM_FILTER="$(to_bool "$2")"; shift 2 ;;
    --rebuild-index) REBUILD_INDEX="$(to_bool "$2")"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

log() { echo "[$(date +'%F %T')]" "$@"; }

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

cd "$ROOT_DIR"

run_cmd() {
  log "Running: $*"
  "$@"
}

# 1) clean
clean_cmd=(python -m STARE.main --task clean --dataset_name "$DATASET" --min_tokens "$MIN_TOKENS" --embed_model "$EMBED_MODEL")
[[ "$ENABLE_LLM_FILTER" == "true" ]] && clean_cmd+=(--enable_llm_filter)
run_cmd "${clean_cmd[@]}"

# 2) extract_mentions
run_cmd python -m STARE.main --task extract_mentions --dataset_name "$DATASET" --embed_model "$EMBED_MODEL"

# 3) embed (needed for index path consistency)
run_cmd python -m STARE.main --task embed --dataset_name "$DATASET" --embed_model "$EMBED_MODEL"

# 4) build_index (optional rebuild)
build_cmd=(python -m STARE.main --task build_index --dataset_name "$DATASET" --embed_model "$EMBED_MODEL")
[[ "$REBUILD_INDEX" == "true" ]] && build_cmd+=(--rebuild_index)
run_cmd "${build_cmd[@]}"

# 5) cooccurrence
run_cmd python -m STARE.main --task cooccurrence --dataset_name "$DATASET" --embed_model "$EMBED_MODEL"

log "Done. Check outputs/indices/${DATASET^^}/$(echo "$EMBED_MODEL" | tr '/A-Z' '-a-z')/company_neighbors.json"

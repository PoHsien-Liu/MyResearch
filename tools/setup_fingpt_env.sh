#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[setup_fingpt_env] Please activate the fingpt conda environment first." >&2
  exit 1
fi

LIB_A="${CONDA_PREFIX}/lib/libittnotify.a"
LIB_SO="${CONDA_PREFIX}/lib/libittnotify.so"

if [[ ! -f "${LIB_A}" ]]; then
  echo "[setup_fingpt_env] libittnotify.a not found at ${LIB_A}. Ensure ittapi is installed in this environment." >&2
  exit 1
fi

echo "[setup_fingpt_env] Building shared libittnotify.so from ${LIB_A}"
g++ -shared -o "${LIB_SO}" -Wl,--whole-archive "${LIB_A}" -Wl,--no-whole-archive

echo "[setup_fingpt_env] Registering LD_PRELOAD for this environment"
conda env config vars set LD_PRELOAD="${LIB_SO}"

echo "[setup_fingpt_env] Done. Please run 'conda deactivate && conda activate fingpt' to reload environment vars."

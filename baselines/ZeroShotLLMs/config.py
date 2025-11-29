"""Load vLLM configuration from the ZeroShotLLMs local llm_config.yaml."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import yaml  # type: ignore

from inference import VLLMConfig  # type: ignore

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_local_config() -> Dict:
    """Load local llm_config.yaml (or env override)."""
    candidate_paths = []
    env_path = os.getenv("ZEROSHOT_LLM_CONFIG")
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.append(MODULE_DIR / "llm_config.yaml")

    for path in candidate_paths:
        if path and path.exists():
            try:
                with path.open("r") as f:
                    data = yaml.safe_load(f) or {}
                    data["_loaded_from"] = str(path)
                    return data
            except Exception:
                continue
    return {}


def build_vllm_config(args, *, default_model: str) -> VLLMConfig:
    backend_name = getattr(args, "backend", None) or "awq_vllm"
    local_cfg = _load_local_config().get("backends", {})
    cfg = local_cfg.get(backend_name, {}) if isinstance(local_cfg, dict) else {}

    model = getattr(args, "base_model", None) or cfg.get("default_model") or default_model

    tp_size = int(cfg.get("tensor_parallel_size", 1))
    avail = _available_gpu_count()
    if avail > 0 and tp_size > avail:
        warnings.warn(f"tensor_parallel_size={tp_size} exceeds available GPUs ({avail}); using {avail}.")
        tp_size = avail
    if avail == 0:
        warnings.warn("No GPU detected; tensor_parallel_size set to 1.")
        tp_size = 1

    quantization = cfg.get("quantization") or cfg.get("precision")
    if isinstance(quantization, str) and quantization.lower() == "none":
        quantization = None

    return VLLMConfig(
        model=model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization"),
        max_model_len=cfg.get("max_model_len") or cfg.get("max_sequence_length"),
        dtype=cfg.get("dtype"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        enforce_eager=cfg.get("enforce_eager"),
    )


def _available_gpu_count() -> int:
    env = os.getenv("CUDA_VISIBLE_DEVICES")
    if env:
        tokens = [t.strip() for t in env.split(",") if t.strip() != ""]
        if tokens:
            return len(tokens)
    try:
        import torch  # type: ignore

        return torch.cuda.device_count()
    except Exception:
        return 0


__all__ = ["build_vllm_config"]

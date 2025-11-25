"""Utility for loading LLM backend configuration."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import os
import yaml


DEFAULT_CONFIG_ENV = "STARE_LLM_CONFIG"
DEFAULT_CONFIG_LOCATIONS = [
    Path("stare_llm_config.yaml"),
    Path("stare_llm.yaml"),
    Path.home() / ".config" / "stare_llm_config.yaml",
]


@lru_cache()
def _load_config() -> Dict[str, Any]:
    candidate_paths = []
    env_path = os.getenv(DEFAULT_CONFIG_ENV)
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(DEFAULT_CONFIG_LOCATIONS)

    for path in candidate_paths:
        if path and path.exists():
            with path.open("r") as f:
                data = yaml.safe_load(f) or {}
                data["_loaded_from"] = str(path)
                return data
    return {}


def get_backend_config(backend: str) -> Dict[str, Any]:
    """Return configuration for a backend ('qwen', 'llama', ...)."""
    data = _load_config()
    backends = data.get("backends", {}) if isinstance(data, dict) else {}
    cfg = backends.get(backend, {})
    if not isinstance(cfg, dict):
        return {}
    return cfg

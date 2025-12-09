"""Utility helpers for resolving dataset/output paths."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


_STARE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _STARE_DIR.parent  # repo root (contains datasets/ and outputs/)


def project_root() -> Path:
    return _PROJECT_ROOT


def get_datasets_dir() -> Path:
    env_path = os.environ.get("DATASETS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (project_root() / "datasets").resolve()


def get_outputs_dir() -> Path:
    env_path = os.environ.get("OUTPUTS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (project_root() / "outputs").resolve()


def get_pipeline_data_dir() -> Path:
    """STARE internal pipeline data root (indices/factors/sft)."""
    return (_STARE_DIR / "pipeline_data").resolve()


def embed_model_slug(embed_model: Optional[str]) -> str:
    if not embed_model:
        return "default"
    candidate = embed_model.strip().lower().replace("/", "-")
    candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate)
    return re.sub(r"-+", "-", candidate).strip("-") or "default"


def indices_dir(dataset_name: str, embed_model: Optional[str]) -> Path:
    dataset_key = dataset_name.upper()
    slug = embed_model_slug(embed_model)
    return get_pipeline_data_dir() / "indices" / dataset_key / slug


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_paths(dataset_name: str):
    from STARE.configs.dataset import DATASET_REGISTRY

    key = dataset_name.upper()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    cfg = DATASET_REGISTRY[key]
    datasets_dir = get_datasets_dir()
    return cfg.resolve_paths(datasets_dir)


__all__ = [
    "project_root",
    "get_datasets_dir",
    "get_outputs_dir",
    "get_pipeline_data_dir",
    "embed_model_slug",
    "indices_dir",
    "ensure_dir",
    "dataset_paths",
]

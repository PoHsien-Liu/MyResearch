"""Utility helpers for resolving dataset and output paths."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


_STARE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _STARE_DIR.parent.parent


def project_root() -> Path:
    """Return repository root (directory containing datasets/ and outputs/)."""
    return _PROJECT_ROOT


def get_datasets_dir() -> Path:
    """Return datasets directory, honoring DATASETS_DIR env variable."""
    env_path = os.environ.get("DATASETS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (project_root() / "datasets").resolve()


def get_outputs_dir() -> Path:
    """Return outputs directory, honoring OUTPUTS_DIR env variable."""
    env_path = os.environ.get("OUTPUTS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (project_root() / "outputs").resolve()


def embed_model_slug(embed_model: Optional[str]) -> str:
    """Turn embed model name into a filesystem friendly slug."""
    if not embed_model:
        return "default"
    candidate = embed_model.strip().lower()
    candidate = candidate.replace("/", "-")
    candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate)
    return re.sub(r"-+", "-", candidate).strip("-") or "default"


def indices_dir(dataset_name: str, embed_model: Optional[str]) -> Path:
    """Return the directory for index artifacts for dataset / embed_model."""
    dataset_key = dataset_name.upper()
    slug = embed_model_slug(embed_model)
    return get_outputs_dir() / "indices" / dataset_key / slug


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_paths(dataset_name: str):
    """Resolve dataset file paths using configs.dataset registry."""
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
    "embed_model_slug",
    "indices_dir",
    "ensure_dir",
    "dataset_paths",
]

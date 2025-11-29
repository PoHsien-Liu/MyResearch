"""Utility helpers for ZeroShotLLMs baselines."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_paths_on_sys(module_dir: Path, repo_root: Path) -> None:
    """Add project and module directories to sys.path for absolute imports."""
    for path in (module_dir, repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def setup_logger(results_dir: str, to_terminal: bool = True, name: str = "ZeroShotLLMs") -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run.log")
    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if to_terminal:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(name)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def resolve_path(path_value: str | None, default_path: str) -> str:
    if not path_value:
        return default_path
    return path_value if os.path.isabs(path_value) else os.path.abspath(path_value)


def snapshot_args(args: Any, results_dir: str) -> None:
    env: Dict[str, str] = {"python": sys.version}
    for mod_name in ["torch", "transformers", "vllm", "numpy", "peft"]:
        try:
            module = __import__(mod_name)
            env[mod_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    payload = {"args": vars(args), "env": env}
    with open(os.path.join(results_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = ["ensure_paths_on_sys", "setup_logger", "set_random_seed", "resolve_path", "snapshot_args"]

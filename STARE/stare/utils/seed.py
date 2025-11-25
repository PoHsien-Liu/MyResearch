"""Random seed helper."""
from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds across python, numpy, and torch (if available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


__all__ = ["set_seed"]

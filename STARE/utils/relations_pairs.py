"""Helper to load candidate relation pairs for relation inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from STARE.configs.relations import RelationConfig


def load_candidate_pairs(
    config: RelationConfig,
    *,
    base_dir: Path,
    apply_filters: bool = True,
) -> List[Tuple[str, str, dict]]:
    """
    Load candidate company pairs from company_neighbors JSON.

    Returns list of (target, neighbor, meta).
    """
    path = base_dir / config.candidate_neighbors_path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []

    pairs: List[Tuple[str, str, dict]] = []
    items: Iterable = data if isinstance(data, list) else data.get("neighbors", [])
    for row in items:
        target = row.get("ticker") or row.get("target") or row.get("source")
        neighbor = row.get("neighbor") or row.get("candidate") or row.get("ticker_neighbor")
        if not target or not neighbor:
            continue
        meta = {
            "cooc": row.get("cooccurrence") or row.get("count") or row.get("cooc"),
            "score": row.get("score"),
        }
        if apply_filters and isinstance(meta.get("cooc"), (int, float)):
            if meta["cooc"] is not None and meta["cooc"] < config.hyperparams.min_cooc:
                continue
        pairs.append((str(target), str(neighbor), meta))
    return pairs


__all__ = ["load_candidate_pairs"]

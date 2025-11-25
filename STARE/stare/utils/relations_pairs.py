"""Utility to load candidate company pairs for LLM relation inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from STARE.stare.configs.relations import RelationConfig


Pair = Tuple[str, str, Dict]


def load_candidate_pairs(
    config: RelationConfig,
    *,
    base_dir: Path,
    apply_filters: bool = True,
) -> List[Pair]:
    """Load company_neighbors.json and return (target, match, meta) pairs.

    If apply_filters=False, returns all pairs without min_cooc/max_neighbors filtering.
    """
    path = base_dir / config.candidate_neighbors_path
    if not path.exists():
        raise FileNotFoundError(f"neighbors file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        neighbors = json.load(f)

    max_neighbors = config.hyperparams.max_neighbors if apply_filters else None
    min_cooc = config.hyperparams.min_cooc if apply_filters else 0

    candidate_pairs: List[Pair] = []
    for target, neigh_map in neighbors.items():
        def cooc_value(item):
            _, meta_val = item
            if isinstance(meta_val, dict):
                return meta_val.get("cooc", 0)
            return int(meta_val)

        sorted_items = sorted(neigh_map.items(), key=cooc_value, reverse=True)
        count = 0
        for match, meta in sorted_items:
            if isinstance(meta, dict):
                cooc = meta.get("cooc", 0)
            else:
                cooc = int(meta)
                meta = {"cooc": cooc}
            if cooc < min_cooc:
                continue
            candidate_pairs.append((target, match, meta))
            count += 1
            if max_neighbors and count >= max_neighbors:
                break
    return candidate_pairs


__all__ = ["load_candidate_pairs", "Pair"]

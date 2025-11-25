"""Utilities to run relation LLM inference and persist company_relations.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from STARE.configs.relations import (
    DEFAULT_RELATION_CONFIGS,
    RelationConfig,
)
from STARE.models.STARE.relation_llm import RelationLLMClient
from STARE.utils.relations_pairs import load_candidate_pairs
from STARE.utils.paths import indices_dir


def build_relations_file(
    dataset_name: str,
    *,
    max_pairs: Optional[int] = None,
) -> Path:
    """Generate company_relations.json for the given dataset."""
    config = DEFAULT_RELATION_CONFIGS.get(dataset_name, RelationConfig(dataset=dataset_name))
    base_dir = indices_dir(dataset_name, None)
    pairs = load_candidate_pairs(config, base_dir=base_dir, apply_filters=True)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    client = RelationLLMClient(config)
    relations: Dict[str, Dict[str, dict]] = {}
    errors: Dict[str, str] = {}
    no_relation_count = 0

    for idx, (target, match, meta) in enumerate(tqdm(pairs, desc="LLM inferring relations"), 1):
        try:
            result = client.infer(target, match, meta)
        except ImportError as exc:
            # Missing backend dependency (e.g., vllm); stop immediately with guidance.
            raise RuntimeError(
                f"Relation inference backend is unavailable: {exc}. "
                "Please install vllm in the active environment (e.g., `pip install vllm`) "
                "and ensure 4-bit weights for meta-llama/Meta-Llama-3.1-70B-Instruct are accessible."
            ) from exc
        except Exception as exc:  # pragma: no cover - best-effort logging
            errors[f"{target}|{match}"] = str(exc)
            continue
        entry = {
            "relationship_type": result.relationship_type,
            "confidence": result.confidence,
            "sentence": result.sentence,
            "explanation": result.explanation,
            "cooc": meta.get("cooc"),
        }
        relations.setdefault(target, {})[match] = entry
        if result.relationship_type == "no_direct_relationship_or_unclear":
            no_relation_count += 1

    output = {
        "metadata": {
            "dataset": dataset_name,
            "model": client.model_name,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_pairs": len(pairs),
            "max_neighbors": config.hyperparams.max_neighbors,
            "no_direct_relationship_or_unclear": no_relation_count,
            "failed_pairs": errors,
        },
        "relations": relations,
    }
    out_path = base_dir / config.relations_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return out_path


__all__ = ["build_relations_file"]

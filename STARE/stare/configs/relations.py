"""Configuration schema for company relation inference and relation-aware retrieve."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

RelationshipType = Literal[
    "same_industry_competitors",
    "supplier_customer",
    "parent_subsidiary_or_affiliate",
    "strategic_partners",
    "same_conglomerate_or_group",
    "cooperative_innovation_or_co-marketing",
    "regulatory_or_legal_dependency",
    "no_direct_relationship_or_unclear",
]


@dataclass
class RelationHyperParams:
    max_neighbors: int = 5
    min_cooc: int = 10
    min_confidence: float = 0.6
    skip_unclear: bool = True
    top_k_self: int = 5
    top_k_per_neighbor: int = 2


@dataclass
class RelationConfig:
    dataset: str
    llm_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    relations_output: str = "company_relations.json"
    candidate_neighbors_path: str = "company_neighbors.json"
    hyperparams: RelationHyperParams = field(default_factory=RelationHyperParams)
    relationship_types: List[RelationshipType] = field(
        default_factory=lambda: [
            "same_industry_competitors",
            "supplier_customer",
            "parent_subsidiary_or_affiliate",
            "strategic_partners",
            "same_conglomerate_or_group",
            "cooperative_innovation_or_co-marketing",
            "regulatory_or_legal_dependency",
            "no_direct_relationship_or_unclear",
        ]
    )


DEFAULT_RELATION_CONFIGS: Dict[str, RelationConfig] = {
    "CMIN": RelationConfig(dataset="CMIN", hyperparams=RelationHyperParams(max_neighbors=3)),
    "SEP": RelationConfig(dataset="SEP"),
    "ACL18": RelationConfig(dataset="ACL18"),
    "SAMPLE": RelationConfig(dataset="SAMPLE"),
}


__all__ = [
    "RelationConfig",
    "RelationHyperParams",
    "RelationshipType",
    "DEFAULT_RELATION_CONFIGS",
]

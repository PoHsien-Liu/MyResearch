"""LLM inference interface for company relations using local vLLM batch calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

from STARE.stare.configs.relations import RelationConfig, RelationshipType
from STARE.stare.llm_backend.inference import PromptLike, run_inference_batch
from STARE.stare.llm_backend.llm_config import get_backend_config


@dataclass
class RelationLLMResult:
    target: str
    match: str
    relationship_type: RelationshipType
    confidence: float
    sentence: str
    explanation: str
    raw_response: Dict[str, Any]


class RelationLLMClient:
    """Relation inference via local vLLM (no external API)."""

    def __init__(self, config: RelationConfig, backend: str = "llama", model_name: Optional[str] = None):
        self.config = config
        self.backend = backend
        backend_cfg = get_backend_config(backend) or {}
        self.model_name = model_name or backend_cfg.get("default_model") or config.llm_model

    def infer(self, target: str, match: str, meta: Dict[str, Any]) -> RelationLLMResult:
        prompt = self._build_prompt(target, match, meta)
        resp_text = run_inference_batch(
            [prompt],
            backend=self.backend,
            model=self.model_name,
            max_tokens=512,
            temperature=0.2,
        )[0]
        parsed = self._parse_response(resp_text, target, match)
        return parsed

    def _build_prompt(self, target: str, match: str, meta: Dict[str, Any]) -> PromptLike:
        cooc = meta.get("cooc", 0)
        system_prompt = (
            "You are an analyst evaluating relationships between public companies.\n"
            "Given the information, infer the most appropriate relationship type.\n"
            "Respond ONLY with JSON keys: sentence, relationship_type, confidence, explanation.\n"
            "Allowed relationship_type: "
            + ", ".join(self.config.relationship_types)
            + "."
        )
        user_prompt = (
            f"Target Company: {target}\n"
            f"Match Company: {match}\n"
            f"Co-occurrence count: {cooc}\n"
            "Provide the most likely relationship."
        )
        return PromptLike(
            system=system_prompt,
            user=user_prompt,
            model=self.model_name,
            max_tokens=512,
            temperature=0.2,
        )

    def _parse_response(self, resp_text: str, target: str, match: str) -> RelationLLMResult:
        content = resp_text.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1]
        content = content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start : end + 1]
                data = json.loads(snippet)
            else:
                raise ValueError(f"Failed to parse LLM JSON response: {content}")

        rel_type = data.get("relationship_type")
        if rel_type not in self.config.relationship_types:
            rel_type = "no_direct_relationship_or_unclear"
        confidence_raw = data.get("confidence", 0)
        confidence = self._normalize_confidence(confidence_raw)
        sentence = data.get("sentence", "")
        explanation = data.get("explanation", "")

        return RelationLLMResult(
            target=target,
            match=match,
            relationship_type=rel_type,  # type: ignore[arg-type]
            confidence=confidence,
            sentence=sentence,
            explanation=explanation,
            raw_response={"content": content},
        )

    def _normalize_confidence(self, value: Any) -> float:
        """Normalize confidence to float; support textual levels."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip().lower()
            mapping = {
                "high": 0.9,
                "medium": 0.5,
                "mid": 0.5,
                "low": 0.2,
                "unknown": 0.0,
            }
            if v in mapping:
                return mapping[v]
            # Try direct numeric parse (supports percentages)
            try:
                if v.endswith("%"):
                    num = float(v[:-1]) / 100.0
                else:
                    num = float(v)
                    if num > 1.0 and num <= 100.0:
                        num = num / 100.0
                return max(0.0, min(1.0, num))
            except ValueError:
                pass
            # Fallback: extract first numeric token
            m = re.search(r"[-+]?[0-9]*\\.?[0-9]+", v)
            if m:
                try:
                    num = float(m.group(0))
                    if num > 1.0 and num <= 100.0:
                        num = num / 100.0
                    return max(0.0, min(1.0, num))
                except ValueError:
                    pass
        return 0.0


__all__ = ["RelationLLMClient", "RelationLLMResult"]

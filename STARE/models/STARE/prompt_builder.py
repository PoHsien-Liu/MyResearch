"""Shared helpers to build prediction prompts for STARE training/inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from STARE.models.STARE.retriever import RetrievedDoc
from STARE.utils.prompts import build_prediction_prompts


@dataclass
class PredictionPromptPackage:
    system_prompt: str
    user_prompt: str
    events_text: str
    all_events: List[Dict]
    assistant_payload: Dict
    prompt_variant: str


def build_events_block(retrieved: List[RetrievedDoc], target_ticker: str, include_related: bool) -> Tuple[str, List[Dict]]:
    """Format retrieved docs into text block and metadata list."""
    target_ticker = target_ticker.upper()
    target_events = []
    related_events: Dict[str, Dict[str, object]] = {}
    all_events: List[Dict] = []

    for idx, doc in enumerate(retrieved, 1):
        relation = str(doc.metadata.get("relation") or doc.metadata.get("relation_type") or "").strip()
        row = {
            "id": idx,
            "date": doc.metadata.get("date") or doc.metadata.get("created_at") or "",
            "text": doc.text,
            "score": doc.score,
            "source_ticker": str(doc.metadata.get("source_ticker") or "").upper(),
            "relation": relation,
        }
        if row["source_ticker"] == target_ticker or not include_related:
            target_events.append(row)
        else:
            bucket = related_events.setdefault(
                row["source_ticker"],
                {"relation": relation or "related", "items": []},
            )
            if not bucket.get("relation") and relation:
                bucket["relation"] = relation
            bucket["items"].append(row)
        all_events.append(row)

    lines: List[str] = []
    lines.append("[EVENTS]")
    if target_events:
        lines.append("Target firm news:")
        for ev in target_events:
            lines.append(f"({ev['id']}) [{ev['date']}] {ev['text']}")
    else:
        lines.append("Target firm news: None.")

    if include_related:
        if related_events:
            lines.append("Related firm news:")
            for firm, bundle in related_events.items():
                relation = bundle.get("relation") or "related"
                lines.append(f"- Firm: {firm} (relation: {relation})")
                for ev in bundle.get("items", []):
                    lines.append(f"  ({ev['id']}) [{ev['date']}] {ev['text']}")
        else:
            lines.append("Related firm news: None.")

    return "\n".join(lines), all_events


def build_prediction_prompt_package(
    *,
    ticker: str,
    target_date: str,
    price_context: str,
    retrieved: List[RetrievedDoc],
    label: str,
    prompt_variant: str = "target_only",
) -> PredictionPromptPackage:
    """Build prediction prompt (system/user) with events + default assistant payload."""
    include_related = prompt_variant == "with_related"
    events_text, all_events = build_events_block(retrieved, ticker, include_related)
    system_prompt, user_prompt = build_prediction_prompts(
        ticker=ticker,
        target_date=target_date,
        price_context=price_context,
        events_text=events_text,
        include_related=include_related,
    )
    assistant_payload = {
        "prediction": label,
        "reason": "",
    }
    return PredictionPromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        events_text=events_text,
        all_events=all_events,
        assistant_payload=assistant_payload,
        prompt_variant=prompt_variant,
    )


__all__ = ["PredictionPromptPackage", "build_events_block", "build_prediction_prompt_package"]

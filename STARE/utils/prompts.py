"""Prompt builders for STARE pipeline (factors, queries, predictions)."""
from __future__ import annotations

from typing import Dict, Tuple

from STARE.llm_backend.inference import PromptLike


def build_factor_prompt(ticker: str) -> PromptLike:
    system_text = (
        "You are a financial analyst who understands common drivers of stock price movements for publicly listed companies.\n"
        "Your task is to list the most important types of events and factors that typically move the stock price of a given company.\n"
        "Respond in strict JSON with keys: ticker, factors[name, description, keywords]."
    )
    user_text = f"""Ticker: {ticker}
Please output the JSON object described above."""
    return PromptLike(system=system_text, user=user_text)


def flatten_factors_for_prompt(factors_data: Dict) -> str:
    lines = []
    factors = factors_data.get("factors", [])
    for item in factors:
        name = item.get("name") or ""
        desc = item.get("description") or ""
        kws = item.get("keywords") or []
        if isinstance(kws, str):
            kws = [kws]
        kw_text = ", ".join(str(k).strip() for k in kws if str(k).strip())
        lines.append(f"- {name} (keywords: {kw_text}) - {desc}")
    return "\n".join(lines)


def build_query_prompt(
    ticker: str,
    target_date: str,
    start_date: str,
    end_date: str,
    factors_text: str,
) -> PromptLike:
    system_text = (
        "You are a retrieval query generator for a financial news search system.\n"
        "Generate focused English queries that retrieve news truly affecting short-term price movement.\n"
        "Respond ONLY with JSON: {\"queries\": [\"<q1>\", \"<q2>\", \"<q3>\"]} without markdown fences."
    )
    user_text = (
        f"Ticker: {ticker}\n"
        f"Prediction date (D0): {target_date}\n"
        f"Look-back window: from {start_date} to {end_date} (inclusive)\n\n"
        "Typical price drivers:\n"
        f"{factors_text}\n\n"
        "Please generate 3 focused English search queries in the JSON format described above."
    )
    return PromptLike(system=system_text, user=user_text)


def build_prediction_prompts(
    *,
    ticker: str,
    target_date: str,
    price_context: str,
    events_text: str,
    include_related: bool,
) -> Tuple[str, str]:
    system_text = (
        "You are a cautious equity analyst. Use ONLY the provided price trend and news; do not add outside knowledge. "
        "If news is missing or weak, state that explicitly."
    )
    guidance_lines = [
        "- Summarize the 5-day price trend (up/down/flat) and its implication.",
        "- Ground every claim on the evidence above; cite IDs like (1), (3). If no usable news, say so and rely on price trend.",
    ]
    if include_related:
        guidance_lines.append(
            "- If using related-firm news, leverage the provided relation (e.g., supplier/competitor/partner) to explain how it impacts the target; cite the event IDs."
        )
    guidance_lines.append("- Keep the JSON concise; no markdown/code fences.")
    guidance = "\n".join(guidance_lines)
    events_hint = ""
    if include_related:
        events_hint = (
            "The [EVENTS] section is grouped into target firm news and related firm news (one block per related company with its relation label). "
            "Use the IDs exactly as shown when citing."
        )
    user_text = (
        f"Target stock: {ticker}\n"
        f"Prediction date (D0): {target_date}\n\n"
        f"{price_context}\n\n"
        f"{events_text}\n"
        f"{events_hint}\n\n"
        "[TASK]\n"
        "Predict next-day movement (UP or DOWN) for the target stock (vs D-1 close) and explain with citations.\n"
        "Follow this guidance:\n"
        f"{guidance}\n\n"
        "[OUTPUT JSON]\n"
        "{\n"
        '  "prediction": "UP" or "DOWN",\n'
        '  "reason": "<short explanation with citations>",\n'
        '  "used_event_ids": [<list of integers>] // empty if none\n'
        "}"
    )
    if not include_related:
        user_text = user_text.replace(
            "If using related-firm news, explain briefly how it impacts the target (e.g., supply chain/sector sentiment/peers).\n",
            "",
        )
    return system_text, user_text


__all__ = [
    "build_factor_prompt",
    "build_query_prompt",
    "build_prediction_prompts",
    "flatten_factors_for_prompt",
]

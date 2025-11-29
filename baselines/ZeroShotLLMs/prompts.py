"""Prompt template shared by ZeroShotLLMs baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DayNews:
    date: str
    texts: List[str]
    truncated: bool = False


@dataclass
class SamplePrompt:
    system: str
    user: str


def format_price_context(context_returns: List[dict]) -> str:
    if not context_returns:
        return "No recent price context is available."

    lines = ["Recent price trend (previous trading days):"]
    for ctx in context_returns:
        pct = ctx.get("ret", 0.0) * 100
        sign = "+" if pct >= 0 else ""
        adj_close = ctx.get("adj_close")
        price_part = f" (adj_close={adj_close:.2f})" if isinstance(adj_close, (int, float)) else ""
        lines.append(f"- {ctx.get('date', '')}: {sign}{pct:.2f}%{price_part}")
    return "\n".join(lines)


def build_prompt(
    *,
    ticker: str,
    prediction_date: str,
    price_context: str,
    news_by_day: List[DayNews],
) -> SamplePrompt:
    system_prompt = (
        "You are a cautious equity analyst. "
        "Using ONLY the provided recent news and price trend, predict whether the stock will go UP or DOWN on the next trading day. "
        "Do not rely on any external knowledge. Keep the answer concise."
    )

    lines: List[str] = [
        f"Target stock: {ticker}",
        f"Prediction date (D0): {prediction_date}",
        "",
        "[PRICE]",
        price_context,
        "",
        "[NEWS - most recent first]",
    ]

    for day in reversed(news_by_day):
        lines.append(f"{day.date}:")
        if not day.texts:
            lines.append("- (no news available)")
            continue
        for idx, text in enumerate(day.texts, start=1):
            lines.append(f"- ({idx}) {text}")
        if day.truncated:
            lines.append("- (truncated additional news to fit prompt length)")
        lines.append("")

    lines.extend(
        [
            "[TASK]",
            "Predict next-day movement for the target stock and provide a short rationale based on the news.",
            "Respond with a single-line JSON object (no markdown):",
            '{ "prediction": "UP" or "DOWN", "explanation": "<brief reason based only on the news>" }',
        ]
    )

    user_prompt = "\n".join(lines).strip()
    return SamplePrompt(system=system_prompt, user=user_prompt)


__all__ = ["DayNews", "SamplePrompt", "build_prompt", "format_price_context"]

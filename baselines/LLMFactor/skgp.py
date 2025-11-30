"""Prompt templates for LLMFactor SKGP (relation, factor, prediction)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from baselines.LLMFactor.data_loader import DayNews, Sample


@dataclass
class Prompt:
    system: str
    user: str


def _format_time_block(context_returns: List[dict], ticker: str) -> str:
    if not context_returns:
        return "Past price movements are unavailable."
    lines = []
    for ctx in context_returns:
        ret = ctx.get("ret", 0.0)
        verb = "rose" if ret > 0 else "fell"
        lines.append(f"On {ctx.get('date', '')}, the stock price of {ticker} {verb}.")
    return "\n".join(lines)


def _flatten_news(news_by_day: List[DayNews]) -> str:
    if not news_by_day:
        return "No news is available."
    lines: List[str] = []
    for day in reversed(news_by_day):  # most recent first
        lines.append(f"{day.date}:")
        if not day.texts:
            lines.append("- (no news)")
        else:
            for idx, text in enumerate(day.texts, 1):
                lines.append(f"- ({idx}) {text}")
            if day.truncated:
                lines.append("- (truncated additional news)")
        lines.append("")
    return "\n".join(lines).strip()


def build_relation_prompt(target_ticker: str, related_ticker: str) -> Prompt:
    system = (
        "You are an equity analyst. "
        "Given two companies, identify their most likely relationship (e.g., supplier, customer, partner, competitor, parent-subsidiary). "
        "Respond with a single complete sentence that fills in the blank."
    )
    user = (
        "Please fill in the blank and return a complete sentence:\n"
        f"{target_ticker} and {related_ticker} are most likely in a ___ relationship."
    )
    return Prompt(system=system, user=user)


def build_factor_prompt(target_ticker: str, news_by_day: List[DayNews], top_k: int = 5) -> Prompt:
    system = (
        "You are an equity analyst. "
        "From the provided news, extract the factors that are most likely to impact the target stock price."
    )
    news_block = _flatten_news(news_by_day)
    user_lines = [
        f"Target stock: {target_ticker}",
        f"Top-k factors: {top_k}",
        "",
        "News (most recent first):",
        news_block,
        "",
        "Task:",
        f"Extract the top {top_k} factors that may affect the stock price of {target_ticker} from the news above.",
        "Return a concise, numbered list of factors (no extra text).",
    ]
    return Prompt(system=system, user="\n".join(user_lines))


def build_prediction_prompt(
    *,
    target_ticker: str,
    target_date: str,
    factors_text: str,
    relations_text: str,
    time_block: str,
) -> Prompt:
    system = (
        "You are an equity analyst. "
        "Use ONLY the provided factors, company relations, and past price movements to judge whether the stock will rise or fall on the target date. "
        "Ground the decision in the supplied information; do not add external knowledge."
    )
    user_lines = [
        "Based on the following information, judge the direction of the stock price as rise or fall, fill in the blank, and give reasons.",
        "",
        "These are the main factors that may affect this stock's price recently:",
        factors_text.strip() or "(factors not available)",
        "",
        "These are the connections between the companies related to this stock:",
        relations_text.strip() or "(relations not available)",
        "",
        "Recent past price movements:",
        time_block.strip() or "(price movements not available)",
        "",
        f"On {target_date}, the stock price of {target_ticker} will ___ (rise/fall).",
        "Answer with one short paragraph explanation after filling the blank.",
    ]
    return Prompt(system=system, user="\n".join(user_lines))


def build_prompt_set(sample: Sample, *, top_k_factors: int = 5) -> dict:
    """Build Step1/Step2/Step3 prompts for a sample."""
    time_block = _format_time_block(sample.context_returns, sample.ticker)
    # Step 1 relation prompts (one per related ticker)
    relation_prompts = [
        build_relation_prompt(sample.ticker, rel) for rel in sample.related_candidates
    ]
    # Step 2 factor prompt (single prompt for all news)
    factor_prompt = build_factor_prompt(sample.ticker, sample.news_by_day, top_k=top_k_factors)
    # Step 3 prediction prompt expects filled factors/relations; here we leave placeholders.
    factors_placeholder = "\n".join([f"{idx}. <factor {idx} from Step 2>" for idx in range(1, top_k_factors + 1)])
    relations_placeholder = "\n".join(
        [f"- {sample.ticker} and {rel}: <relation from Step 1>" for rel in sample.related_candidates]
    ) or "(relations not available)"
    prediction_prompt = build_prediction_prompt(
        target_ticker=sample.ticker,
        target_date=sample.prediction_date,
        factors_text=factors_placeholder,
        relations_text=relations_placeholder,
        time_block=time_block,
    )
    return {
        "time_block": time_block,
        "relation_prompts": relation_prompts,
        "factor_prompt": factor_prompt,
        "prediction_prompt": prediction_prompt,
    }


__all__ = [
    "Prompt",
    "build_relation_prompt",
    "build_factor_prompt",
    "build_prediction_prompt",
    "build_prompt_set",
]

"""Prompt helpers for FinGPT baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

PROMPT_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROMPT_DIR / "system.txt"
USER_PROMPT_PATH = PROMPT_DIR / "user.txt"


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


_SYSTEM_TEMPLATE = _load_prompt(SYSTEM_PROMPT_PATH)
_USER_TEMPLATE = _load_prompt(USER_PROMPT_PATH)


@dataclass
class DayContext:
    date: str
    texts: List[str]


def render_system_prompt(seq_len: int) -> str:
    return _SYSTEM_TEMPLATE.format(seq_len=seq_len)


def render_recent_news_block(contexts: Iterable[DayContext]) -> str:
    context_list = list(contexts)
    total = len(context_list)
    if not context_list:
        return "(no context available)"

    lines: List[str] = []
    for idx, ctx in enumerate(context_list):
        offset = max(1, total - idx)
        lines.append(f"[Day t-{offset}] ({ctx.date})")
        if ctx.texts:
            for text in ctx.texts:
                lines.append(f"- {text}")
        else:
            lines.append("- (no relevant news available)")
        lines.append("")
    return "\n".join(lines).strip()


def render_user_prompt(
    *,
    ticker: str,
    prediction_date: str,
    seq_len: int,
    contexts: Iterable[DayContext],
) -> str:
    return _USER_TEMPLATE.format(
        ticker=ticker,
        prediction_date=prediction_date,
        seq_len=seq_len,
        news_block=render_recent_news_block(contexts),
    )


__all__ = [
    "DayContext",
    "render_system_prompt",
    "render_user_prompt",
    "render_recent_news_block",
]


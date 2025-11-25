"""Utilities for price context formatting."""
from __future__ import annotations

from typing import Sequence


def _require_length(seq: Sequence, expected: int, name: str) -> None:
    if len(seq) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(seq)}")


def _format_return(value) -> str:
    """Format return value as string with trailing percent sign."""
    try:
        num = float(value)
        # Keep a modest precision; avoid scientific notation.
        text = f"{num:.2f}".rstrip("0").rstrip(".")
    except Exception:
        text = str(value)
    return f"{text}%" if not text.endswith("%") else text


def build_price_context(
    ticker: str,
    target_date: str,
    last5_dates: Sequence[str],
    last5_returns: Sequence,
) -> str:
    """
    Build the PRICE_CONTEXT_BLOCK describing last 5 trading-day returns.

    Args:
        ticker: Stock ticker symbol.
        target_date: Prediction date (D0), e.g., "2014-01-15".
        last5_dates: Sequence of 5 date strings ordered oldest->newest (D-5 ... D-1).
        last5_returns: Sequence of 5 daily returns (%) aligned with last5_dates.
    """
    _require_length(last5_dates, 5, "last5_dates")
    _require_length(last5_returns, 5, "last5_returns")

    lines = [
        "[PRICE CONTEXT]",
        "",
        f"We are analyzing the recent price performance of stock {ticker}.",
        "",
        (
            f"Here are the last 5 trading days before the prediction date {target_date} "
            "(D-5 to D-1), expressed as daily percentage returns relative to the previous trading day:"
        ),
        "",
    ]
    for idx in range(5):
        d_label = f"D-{5 - idx}"
        date_str = last5_dates[idx]
        ret_str = _format_return(last5_returns[idx])
        lines.append(f"- {d_label} ({date_str}): {ret_str} daily return")

    lines.extend(
        [
            "",
            "All returns are close-to-close percentage changes. Positive values indicate the stock went up; negative values indicate it went down.",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_price_context"]

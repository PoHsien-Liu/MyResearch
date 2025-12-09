"""Utilities for price context formatting and sample selection."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict

from common.data.loader import list_trading_days

from STARE.utils.paths import dataset_paths


def _require_length(seq: Sequence, expected: int, name: str) -> None:
    if len(seq) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(seq)}")


def _format_return(value) -> str:
    """Format return value as string with trailing percent sign."""
    try:
        num = float(value)
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
    """Build the PRICE_CONTEXT_BLOCK describing last 5 trading-day returns."""
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


def resolve_price_dir(dataset_name: str) -> Path:
    """Choose the directory that contains ticker CSVs for prices."""
    paths = dataset_paths(dataset_name)
    base = Path(paths.price_path)
    raw_dir = base / "raw"
    return raw_dir if raw_dir.exists() else base


def pick_sample(
    *,
    dataset_name: str,
    mode: str,
    price_dir: Path,
    seq_len: int,
    ticker: Optional[str],
    target_date: Optional[str],
    sample_index: int,
    train_ratio: float,
    split_root: Optional[Path],
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
) -> Dict[str, str]:
    """Pick one (ticker, date) sample."""
    if ticker and target_date:
        return {"ticker": ticker.upper(), "date": target_date}

    samples = list_trading_days(
        dataset_name=dataset_name,
        price_dir=str(price_dir),
        mode=mode,
        seq_len=seq_len,
        train_ratio=train_ratio,
        split_root=str(split_root) if split_root else None,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
    )
    if not samples:
        raise RuntimeError(f"No trading day samples found for dataset={dataset_name} mode={mode}")

    if ticker:
        filtered = [s for s in samples if s["ticker"].upper() == ticker.upper()]
        if not filtered:
            raise RuntimeError(f"No samples found for ticker={ticker} in dataset={dataset_name}")
        samples = filtered

    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index {sample_index} out of range (0..{len(samples)-1})")
    return samples[sample_index]


def last_k_returns(context_returns: Sequence[Dict], k: int) -> tuple[list[str], list[float]]:
    if len(context_returns) < k:
        raise ValueError(f"Need at least {k} context returns, got {len(context_returns)}")
    tail = context_returns[-k:]
    dates = [r["date"] for r in tail]
    returns = [float(r["ret"]) * 100 for r in tail]
    return dates, returns


__all__ = [
    "build_price_context",
    "resolve_price_dir",
    "pick_sample",
    "last_k_returns",
]

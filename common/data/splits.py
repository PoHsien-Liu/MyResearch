"""Shared dataset split helpers."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from common.io.results import safe_name


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_split_root() -> str:
    return os.path.join(_repo_root(), "splits")


def _load_split_file(path: str) -> Dict:
    if not os.path.exists(path):
        return {"meta": {}, "splits": {"train": {}, "test": {}}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_split_file(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_split_dates(
    *,
    dataset_name: str,
    ticker: str,
    dates: List[str],
    split_name: str,
    split_root: str | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    label_strategy: str = "legacy",
    neg_threshold: float = -0.005,
    pos_threshold: float = 0.0055,
) -> List[str]:
    """Return split-specific dates for a ticker. Creates file if needed."""

    assert split_name in {"train", "test"}

    split_root = split_root or _default_split_root()
    ratio_dir = os.path.join(split_root, safe_name(dataset_name), f"ratio-{_format_ratio(train_ratio)}")
    strategy_dir = os.path.join(ratio_dir, _strategy_dir(label_strategy, neg_threshold, pos_threshold))
    os.makedirs(strategy_dir, exist_ok=True)
    split_path = os.path.join(strategy_dir, "splits.json")
    data = _load_split_file(split_path)

    meta = data.setdefault("meta", {})
    splits = data.setdefault("splits", {"train": {}, "test": {}})

    if not meta:
        meta.update({
            "train_ratio": train_ratio,
            "seed": seed,
            "label_strategy": label_strategy,
            "neg_threshold": neg_threshold,
            "pos_threshold": pos_threshold,
        })

    if ticker not in splits.get("train", {}):
        _create_ticker_split(splits, ticker, dates, train_ratio)
        _save_split_file(split_path, data)

    return splits.get(split_name, {}).get(ticker, [])


def _create_ticker_split(splits: Dict, ticker: str, dates: List[str], train_ratio: float) -> None:
    if not dates:
        splits.setdefault("train", {})[ticker] = []
        splits.setdefault("test", {})[ticker] = []
        return

    cutoff = max(1, int(len(dates) * train_ratio))
    train_dates = dates[:cutoff]
    test_dates = dates[cutoff:]

    splits.setdefault("train", {})[ticker] = train_dates
    splits.setdefault("test", {})[ticker] = test_dates


__all__ = ["get_split_dates"]


def _format_ratio(train_ratio: float) -> str:
    return f"{train_ratio:.2f}".rstrip("0").rstrip(".")


def _pct_tag(value: float) -> str:
    pct = value * 100
    return f"{pct:+.2f}".rstrip("0").rstrip(".") + "pct"


def _strategy_dir(label_strategy: str, neg_threshold: float, pos_threshold: float) -> str:
    strategy = (label_strategy or "legacy").lower()
    if strategy == "legacy":
        return "legacy"
    return os.path.join("dual", f"neg{_pct_tag(neg_threshold)}_pos{_pct_tag(pos_threshold)}")

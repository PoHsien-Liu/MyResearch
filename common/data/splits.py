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

    created_new_split = False
    if ticker not in splits.get("train", {}):
        _create_ticker_split(splits, ticker, dates, train_ratio)
        _save_split_file(split_path, data)
        created_new_split = True

    _maybe_write_sft_split_map(
        base_meta=meta,
        splits=splits,
        out_dir=strategy_dir,
        force_write=created_new_split,
    )

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


def _maybe_write_sft_split_map(
    *,
    base_meta: Dict,
    splits: Dict,
    out_dir: str,
    force_write: bool = False,
    sft_train_ratio: float = 0.9,
    min_val_samples: int = 1,
) -> None:
    """
    Persist an SFT-specific split map that further divides the base `train` into
    `sft_train` / `sft_val` using a time-ordered split (default 90/10),
    while `sft_test` mirrors the base `test`.
    """
    out_path = os.path.join(out_dir, "sft_split_map.json")

    if (not force_write) and os.path.exists(out_path):
        return

    train_map = splits.get("train", {})
    test_map = splits.get("test", {})
    sft_train: Dict[str, List[str]] = {}
    sft_val: Dict[str, List[str]] = {}
    sft_test: Dict[str, List[str]] = {k: v for k, v in test_map.items()}

    for ticker, train_dates in train_map.items():
        sorted_dates = sorted(train_dates)
        n = len(sorted_dates)
        if n <= 1:
            sft_train[ticker] = sorted_dates
            sft_val[ticker] = []
            continue
        cutoff = int(n * sft_train_ratio)
        if cutoff >= n:
            cutoff = n - 1
        train_split = sorted_dates[:cutoff]
        val_split = sorted_dates[cutoff:]
        if min_val_samples > 0 and len(val_split) < min_val_samples and n > min_val_samples:
            train_split = sorted_dates[:-min_val_samples]
            val_split = sorted_dates[-min_val_samples:]
        sft_train[ticker] = train_split
        sft_val[ticker] = val_split

    payload = {
        "meta": {
            "base_train_ratio": base_meta.get("train_ratio"),
            "seed": base_meta.get("seed"),
            "label_strategy": base_meta.get("label_strategy"),
            "neg_threshold": base_meta.get("neg_threshold"),
            "pos_threshold": base_meta.get("pos_threshold"),
            "sft_train_ratio": sft_train_ratio,
            "min_val_samples": min_val_samples,
        },
        "splits": {
            "sft_train": sft_train,
            "sft_val": sft_val,
            "sft_test": sft_test,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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

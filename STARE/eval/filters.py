"""Filtering helpers for explanation evaluation."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Literal

from STARE.eval.top_stocks_loader import load_top1_tickers

StockScope = Literal["all", "top1"]


def filter_by_correct(df: pd.DataFrame, only_correct: bool = True) -> pd.DataFrame:
    """Keep only rows where y_true == y_pred when only_correct is True."""
    if not only_correct:
        return df
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise KeyError("filter_by_correct requires columns: y_true, y_pred")
    return df[df["y_true"] == df["y_pred"]]


def filter_by_stock_scope(
    df: pd.DataFrame,
    dataset: str,
    stock_scope: StockScope,
    datasets_root: str | Path | None = None,
) -> pd.DataFrame:
    """Filter DataFrame by stock scope (all or top1)."""
    if stock_scope == "all":
        return df
    if "ticker" not in df.columns:
        raise KeyError("filter_by_stock_scope requires column: ticker")
    top1 = load_top1_tickers(dataset=dataset, scope=stock_scope, datasets_root=Path(datasets_root) if datasets_root else None)
    if not top1:
        return df
    return df[df["ticker"].isin(top1)]

"""Unified dataset loader exposing shared APIs for baselines."""

from __future__ import annotations

import json
import math
import os
from glob import glob
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from common.data.splits import get_split_dates


DATE_WINDOWS = {
    "STOCKNET": ("2014-01-01", "2015-12-31"),
    "SEP": ("2020-01-01", "2022-12-31"),
    "CMIN": ("2018-01-01", "2019-12-31"),
}

EXCLUDED_TICKERS = {
    "CMIN": {"C-PJ", "CTA-PB", "SPG-PJ", "WFC-PL"},
}

DEFAULT_NEG_THRESHOLD = -0.005
DEFAULT_POS_THRESHOLD = 0.0055


def list_trading_days(
    *,
    dataset_name: str,
    price_dir: str,
    mode: str,
    seq_len: int = 5,
    split_root: Optional[str] = None,
    train_ratio: float = 0.8,
    split_seed: int = 42,
    label_strategy: str = "legacy",
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    pos_threshold: float = DEFAULT_POS_THRESHOLD,
    logger=None,
    progress: bool = False,
) -> List[Dict[str, str]]:
    assert mode in {"train", "test"}

    split_root = split_root or os.path.join(_repo_root(), "splits")
    os.makedirs(split_root, exist_ok=True)

    samples: List[Dict[str, str]] = []
    stats = {
        "dataset": dataset_name,
        "mode": mode,
        "train_ratio": train_ratio,
        "label_strategy": label_strategy,
        "neg_threshold": neg_threshold,
        "pos_threshold": pos_threshold,
        "total_candidates": 0,
        "split_candidates": 0,
        "neutral_skipped": 0,
    }

    label_strategy = label_strategy or "legacy"
    label_strategy = label_strategy.lower()

    dataset_key = dataset_name.upper()

    iter_price_files = _iter_price_files(price_dir, dataset_key)
    if progress:
        iter_price_files = tqdm(iter_price_files, desc=f"[{dataset_key}] price files", unit="file")

    for ticker, price_path in iter_price_files:
        try:
            df, _ = _load_price_frame(price_path)
        except Exception as exc:
            if logger:
                logger.warning(f"[DataLoader] skip {ticker}: {exc}")
            continue

        df = _apply_time_window(df, dataset_key)
        if df.empty:
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        candidate_dates = df["date_str"].tolist()
        if not candidate_dates:
            continue
        stats["total_candidates"] += max(0, len(candidate_dates) - 1)

        df = _compute_returns(df)

        split_dates = get_split_dates(
            dataset_name=dataset_key,
            ticker=ticker,
            dates=candidate_dates,
            split_name=mode,
            split_root=split_root,
            train_ratio=train_ratio,
            seed=split_seed,
            label_strategy=label_strategy,
            neg_threshold=neg_threshold,
            pos_threshold=pos_threshold,
        )

        index_map = {d: idx for idx, d in enumerate(candidate_dates)}
        for d in split_dates:
            idx = index_map.get(d)
            if idx is None or idx == 0:
                continue  # need prior day for return
            stats["split_candidates"] += 1
            ret_value = df.loc[idx, "ret"]
            if math.isnan(ret_value):
                continue
            label = _label_from_return(ret_value, label_strategy, neg_threshold, pos_threshold)
            if label is None:
                stats["neutral_skipped"] += 1
                continue
            samples.append({"ticker": ticker, "date": d, "label": label, "ret": ret_value})

    samples.sort(key=lambda x: (x["date"], x["ticker"]))
    stats["kept_samples"] = len(samples)
    return samples


def get_record(
    *,
    dataset_name: str,
    ticker: str,
    date: str,
    price_dir: str,
    tweet_dir: Optional[str] = None,
    news_csv_dir: Optional[str] = None,
    seq_len: int = 5,
    label_strategy: str = "legacy",
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    pos_threshold: float = DEFAULT_POS_THRESHOLD,
    logger=None,
) -> Dict:
    price_path = _find_price_path(price_dir, ticker)
    if not price_path:
        raise FileNotFoundError(f"price file not found for {ticker}")

    df, price_meta = _load_price_frame(price_path)
    df = df.sort_values("date").reset_index(drop=True)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    if date not in set(df["date_str"]):
        raise KeyError(f"date {date} not in price file for {ticker}")

    df = _compute_returns(df)
    row_idx = df.index[df["date_str"] == date][0]
    if row_idx == 0 or math.isnan(df.loc[row_idx, "ret"]):
        raise KeyError(f"insufficient history for {ticker} {date}")

    adj_close = df.loc[row_idx, "adj_close"]
    ret_value = df.loc[row_idx, "ret"]
    label = _label_from_return(ret_value, label_strategy, neg_threshold, pos_threshold) or "Neutral"

    context_returns = []
    start_idx = max(0, row_idx - seq_len)
    context_slice = df.loc[start_idx: row_idx - 1]
    for _, r in context_slice.iterrows():
        if math.isnan(r["ret"]):
            continue
        context_returns.append({
            "date": r["date_str"],
            "ret": r["ret"],
            "adj_close": r["adj_close"],
        })

    # Build the list of dates whose texts should be summarized (seq_len window ending at current date)
    text_window_start = max(0, row_idx - seq_len + 1)
    text_window_slice = df.loc[text_window_start: row_idx]
    text_window_dates = text_window_slice["date_str"].tolist()

    texts = _load_texts_for_date(
        dataset_name=dataset_name,
        ticker=ticker,
        date=date,
        tweet_dir=tweet_dir,
        news_csv_dir=news_csv_dir,
        logger=logger,
    )

    return {
        "ticker": ticker,
        "date": date,
        "price": {
            "adj_close": adj_close,
            "ret": ret_value,
            "label": label,
            "context_returns": context_returns,
            "meta": {
                "label_strategy": label_strategy,
                "neg_threshold": neg_threshold,
                "pos_threshold": pos_threshold,
                **price_meta,
            },
        },
        "texts": texts,
        "text_window_dates": text_window_dates,
    }


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _iter_price_files(price_dir: str, dataset_name: str):
    for entry in sorted(os.listdir(price_dir)):
        path = os.path.join(price_dir, entry)
        if os.path.isdir(path):
            continue
        ticker = os.path.splitext(entry)[0]
        if dataset_name.upper() in EXCLUDED_TICKERS and ticker.upper() in EXCLUDED_TICKERS[dataset_name.upper()]:
            continue
        yield ticker.upper(), path


def _find_price_path(price_dir: str, ticker: str) -> Optional[str]:
    ticker = ticker.upper()
    preferred = [f"{ticker}.csv", f"{ticker}.txt"]
    for name in preferred:
        path = os.path.join(price_dir, name)
        if os.path.exists(path):
            return path
    matches = glob(os.path.join(price_dir, f"{ticker}.*"))
    return matches[0] if matches else None


def _load_price_frame(path: str) -> tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    date_col = None
    for key in ("date", "timestamp"):
        if key in lower:
            date_col = lower[key]
            break
    if not date_col:
        raise ValueError("Date column not found")

    adj_candidates = ["adj_close", "adjclose", "adjusted_close", "adj_close*"]
    adj_col = next((lower[k] for k in adj_candidates if k in lower), None)
    fallback = False
    if not adj_col:
        close_col = next((lower[k] for k in ("close", "closing_price") if k in lower), None)
        if not close_col:
            raise ValueError("Close column not found")
        adj_col = close_col
        fallback = True
    close_col = next((lower[k] for k in ("close", "closing_price") if k in lower), adj_col)

    df = df[[date_col, adj_col, close_col]].rename(columns={
        date_col: "date",
        adj_col: "adj_close",
        close_col: "close",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_close"])
    return df, {"price_source": "raw_close_fallback" if fallback else "adj_close"}


def _apply_time_window(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    name = dataset_name.upper()
    if name in {"STOCKNET"}:
        start, end = DATE_WINDOWS["STOCKNET"]
    elif name in {"CMIN", "CMIN-US"}:
        start, end = DATE_WINDOWS["CMIN"]
    else:
        return df

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df.loc[mask]


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adj_close_shift"] = df["adj_close"].shift(1)
    df["ret"] = df["adj_close"] / df["adj_close_shift"] - 1
    return df


def _load_texts_for_date(dataset_name, ticker, date, *, tweet_dir=None, news_csv_dir=None, logger=None):
    name = dataset_name.upper()
    if name in {"STOCKNET", "SEP", "SAMPLE"}:
        return _load_tweets(ticker, date, tweet_dir, logger)
    if name in {"CMIN", "CMIN-US"}:
        return _load_cmin_news(ticker, date, news_csv_dir, logger)
    return []


def _load_tweets(ticker, date, tweet_dir, logger=None):
    if not tweet_dir:
        return []
    path = os.path.join(tweet_dir, ticker, date)
    if not os.path.exists(path):
        return []
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text")
            if text:
                texts.append({
                    "ticker": ticker,
                    "date": date,
                    "source": "tweet",
                    "text": text,
                    "meta": {},
                })
    return texts


def _load_cmin_news(ticker, date, news_csv_dir, logger=None):
    if not news_csv_dir:
        return []
    path = os.path.join(news_csv_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(
        path,
        sep="\t",
        parse_dates=["date"],
        dayfirst=False,
        on_bad_lines="skip",
    )
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    subset = df[df["date_str"] == date]
    texts = []
    for _, row in subset.iterrows():
        text = row.get("title") or row.get("headline")
        if not isinstance(text, str):
            continue
        texts.append({
            "ticker": ticker,
            "date": date,
            "source": "news",
            "text": text,
            "meta": {},
    })
    if not texts and logger:
        # Silently skip missing news for the date to reduce log noise.
        return texts
    return texts


def _label_from_return(ret_value: float, strategy: str, neg_threshold: float, pos_threshold: float) -> Optional[str]:
    strategy = (strategy or "legacy").lower()
    if strategy == "dual_threshold":
        if ret_value <= neg_threshold:
            return "DOWN"
        if ret_value > pos_threshold:
            return "UP"
        return None
    return "UP" if ret_value > 0 else "DOWN"


def load_texts_for_day(
    *,
    dataset_name: str,
    ticker: str,
    date: str,
    tweet_dir: Optional[str] = None,
    news_csv_dir: Optional[str] = None,
    logger=None,
):
    """Public helper to fetch raw text entries for a given (ticker, date)."""
    return _load_texts_for_date(
        dataset_name=dataset_name,
        ticker=ticker,
        date=date,
        tweet_dir=tweet_dir,
        news_csv_dir=news_csv_dir,
        logger=logger,
    )


__all__ = ["list_trading_days", "get_record", "load_texts_for_day"]

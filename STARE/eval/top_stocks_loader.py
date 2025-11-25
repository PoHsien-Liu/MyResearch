"""Utilities to load or compute top-1 stocks per dataset."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from STARE.configs.top_stocks import DATASET_TOP1_CONFIG, ScopeMode, Top1Config

LOGGER = logging.getLogger(__name__)

# Default cache directory for computed top-1 lists.
DEFAULT_CACHE_DIR = Path("outputs/cache/top1")


def load_top1_tickers(
    dataset: str,
    scope: ScopeMode,
    datasets_root: Path | None = None,
    cache_dir: Path | None = None,
) -> Set[str]:
    """Return a set of top-1 tickers for the dataset given the scope.

    If scope == "all", returns an empty set to signal no filtering.
    """
    dataset_key = dataset.upper()
    if dataset_key == "ACL18":
        dataset_key = "STOCKNET"
    if dataset_key == "CMIN":
        dataset_key = "CMIN-US"

    if scope == "all":
        return set()

    cfg = DATASET_TOP1_CONFIG.get(dataset_key)
    if cfg is None:
        raise ValueError(f"No top-1 config for dataset: {dataset}")

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "fixed_list":
        return set(cfg.tickers or [])
    if cfg.mode == "by_sector_tweet_volume":
        return _load_or_compute_stocknet_top1(cfg, datasets_root, cache_dir)
    if cfg.mode == "overall_top_k_news":
        return _load_or_compute_cmin_topk(cfg, datasets_root, cache_dir)

    raise ValueError(f"Unsupported top-1 mode: {cfg.mode}")


def _load_or_compute_stocknet_top1(cfg: Top1Config, datasets_root: Path | None, cache_dir: Path) -> Set[str]:
    cache_path = cache_dir / "stocknet_top1.json"
    if cache_path.exists():
        with cache_path.open() as f:
            data = json.load(f)
            return set(data.get("tickers", []))

    root = datasets_root or Path("datasets")
    stock_table = root / "stocknet" / "StockTable"
    tweets_root = root / "stocknet" / "tweet" / "raw"

    if not stock_table.exists():
        raise FileNotFoundError(f"StockTable not found at {stock_table}")
    if not tweets_root.exists():
        raise FileNotFoundError(f"StockNet tweet/raw not found at {tweets_root}")

    sector_to_tickers: Dict[str, Set[str]] = defaultdict(set)
    with stock_table.open() as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            sector, symbol = parts[0], parts[1]
            ticker = symbol.replace("$", "").strip().upper()
            sector_to_tickers[sector].add(ticker)

    ticker_counts: Dict[str, int] = {}
    for sector, tickers in sector_to_tickers.items():
        for ticker in tickers:
            tdir = tweets_root / ticker
            if not tdir.exists():
                continue
            count = _count_files(tdir)
            ticker_counts[ticker] = count
            LOGGER.debug("StockNet ticker %s sector %s has %d tweet files", ticker, sector, count)

    # pick top-1 per sector
    top1: Set[str] = set()
    for sector, tickers in sector_to_tickers.items():
        best_ticker, best_count = None, -1
        for ticker in tickers:
            count = ticker_counts.get(ticker, 0)
            if count > best_count:
                best_ticker, best_count = ticker, count
        if best_ticker:
            top1.add(best_ticker)
            LOGGER.info("StockNet sector %s top-1: %s (%d files)", sector, best_ticker, best_count)

    payload = {"tickers": sorted(top1), "config": asdict(cfg)}
    with cache_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return top1


def _load_or_compute_cmin_topk(cfg: Top1Config, datasets_root: Path | None, cache_dir: Path) -> Set[str]:
    cache_path = cache_dir / "cmin_us_top1.json"
    if cache_path.exists():
        with cache_path.open() as f:
            data = json.load(f)
            return set(data.get("tickers", []))

    root = datasets_root or Path("datasets")
    news_root = root / "CMIN" / "CMIN-US" / "news" / "raw"
    if not news_root.exists():
        raise FileNotFoundError(f"CMIN-US news/raw not found at {news_root}")
    k = cfg.k or 11

    counts: Dict[str, int] = {}
    for file in news_root.glob("*.csv"):
        ticker = file.stem.upper()
        counts[ticker] = _count_lines(file) - 1  # skip header

    sorted_tickers = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    topk = [ticker for ticker, _ in sorted_tickers[:k]]
    LOGGER.info("CMIN-US top-%d tickers by news volume: %s", k, topk)

    payload = {"tickers": topk, "config": asdict(cfg)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return set(topk)


def _count_files(root: Path) -> int:
    """Count number of files under a directory."""
    return sum(1 for _ in root.rglob("*") if _.is_file())


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    with path.open("r", errors="ignore") as f:
        return sum(1 for _ in f)

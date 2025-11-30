"""Top-1 stock configuration per dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

ScopeMode = Literal["all", "top1"]
Top1Mode = Literal["fixed_list", "by_sector_tweet_volume", "overall_top_k_news"]


@dataclass(frozen=True)
class Top1Config:
    mode: Top1Mode
    tickers: Optional[List[str]] = None
    k: Optional[int] = None  # used for overall_top_k_news


DATASET_TOP1_CONFIG: Dict[str, Top1Config] = {
    # SEP: 11 sectors, first stock of each sector.
    "SEP": Top1Config(
        mode="fixed_list",
        tickers=[
            "BHP",   # Basic Materials
            "BRK-A", # Financial Services
            "WMT",   # Consumer Defensive
            "NEE",   # Utilities
            "XOM",   # Energy
            "AAPL",  # Technology
            "AMZN",  # Consumer Cyclical
            "AMT",   # Real Estate
            "UNH",   # Healthcare
            "GOOG",  # Communication Services
            "UPS",   # Industrials
        ],
    ),
    # StockNet: derive per-sector top-1 by tweet volume.
    "STOCKNET": Top1Config(mode="by_sector_tweet_volume"),
    # CMIN-US: top-k tickers by news volume.
    "CMIN-US": Top1Config(mode="overall_top_k_news", k=11),
    "CMIN": Top1Config(mode="overall_top_k_news", k=11),
    # SAMPLE: no top-1 filtering; keep scope=all.
    "SAMPLE": Top1Config(mode="fixed_list", tickers=[]),
}

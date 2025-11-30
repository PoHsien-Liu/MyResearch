"""Shared dataset path resolution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

DATASET_PATHS: Dict[str, Dict[str, str]] = {
    "SAMPLE": {
        "price": "sample_data/sample_price/raw",
        "tweet": "sample_data/sample_tweet/raw",
    },
    "STOCKNET": {
        "price": "stocknet/price/raw",
        "tweet": "stocknet/tweet/raw",
    },
    "CMIN": {
        "price": "CMIN/CMIN-US/price/raw",
        "tweet": "CMIN/CMIN-US/news/raw",
    },
    "SEP": {
        "price": "SEP/price/raw",
        "tweet": "SEP/tweet/raw",
    },
}


@dataclass(frozen=True)
class DatasetPaths:
    dataset_name: str
    base_data_dir: str
    price_dir: str
    tweet_dir: str


def resolve_dataset_paths(dataset_name: str, base_data_dir: str | None = None) -> DatasetPaths:
    """Return fully-qualified price/tweet directories for a dataset.

    Args:
        dataset_name: One of DATASET_PATHS keys.
        base_data_dir: Optional override of DATASETS_DIR env / ./datasets.
    """

    dataset_key = (dataset_name or "").upper()
    if dataset_key not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset_name={dataset_name}. Available={list(DATASET_PATHS)}")

    base_dir = base_data_dir or os.getenv("DATASETS_DIR", "./datasets")
    cfg = DATASET_PATHS[dataset_key]

    price_dir = os.path.join(base_dir, cfg["price"])
    tweet_dir = os.path.join(base_dir, cfg["tweet"])

    if not os.path.isdir(price_dir):
        raise FileNotFoundError(f"price_dir not found: {price_dir}")
    if not os.path.isdir(tweet_dir):
        raise FileNotFoundError(f"tweet_dir not found: {tweet_dir}")

    return DatasetPaths(
        dataset_name=dataset_key,
        base_data_dir=base_dir,
        price_dir=price_dir,
        tweet_dir=tweet_dir,
    )

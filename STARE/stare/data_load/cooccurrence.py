"""Compute company co-occurrence graphs from cleaned_with_mentions data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from STARE.stare.utils.logger import setup_logger
from STARE.stare.utils.paths import ensure_dir, indices_dir
from STARE.stare.utils.seed import set_seed

# Canonical ticker mapping須與 extract_mentions 保持一致
CANONICAL_MAP = {
    "GOOGL": "GOOG",
    "META": "FB",
    "BRK-B": "BRK-A",
    "BRK.B": "BRK-A",
}
EXCLUDED_TICKERS = {"C-PJ", "CTA-PB", "SPG-PJ", "WFC-PL"}


def run_cooccurrence(args) -> None:
    """CLI hook to compute company co-occurrence from cleaned_with_mentions.parquet."""
    set_seed(args.seed)
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "cooccurrence.log"
    logger = setup_logger("stare.cooccurrence", log_file=log_file)

    input_path = output_dir / "cleaned_with_mentions.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"cleaned_with_mentions.parquet not found at {input_path}")

    logger.info("Loading cleaned_with_mentions from %s", input_path)
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError("Input dataframe is empty.")

    co_counter = Counter()
    for _, row in df.iterrows():
        tickers = _collect_tickers(row.get("source_ticker"), row.get("mentioned_tickers"))
        if len(tickers) < 2:
            continue
        for a, b in combinations(sorted(tickers), 2):
            co_counter[(a, b)] += 1

    neighbors: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (a, b), cnt in co_counter.items():
        neighbors[a][b] = cnt
        neighbors[b][a] = cnt

    co_rows = [{"ticker_a": a, "ticker_b": b, "count": cnt} for (a, b), cnt in sorted(co_counter.items(), key=lambda x: -x[1])]
    co_df = pd.DataFrame(co_rows)
    co_path = output_dir / "company_cooccurrence.csv"
    co_df.to_csv(co_path, index=False)

    neighbor_rows = [{"ticker": t, "neighbor_count": len(adj)} for t, adj in neighbors.items()]
    neighbor_df = pd.DataFrame(neighbor_rows)
    if not neighbor_df.empty:
        neighbor_df = neighbor_df.sort_values("neighbor_count", ascending=False)
    neighbor_path = output_dir / "company_neighbors.csv"
    neighbor_df.to_csv(neighbor_path, index=False)

    neighbor_json_path = output_dir / "company_neighbors.json"
    neighbor_json_path.write_text(json.dumps(neighbors, indent=2))

    logger.info("Wrote co-occurrence to %s (rows=%d)", co_path, len(co_df))
    logger.info("Wrote neighbor counts to %s (rows=%d)", neighbor_path, len(neighbor_df))
    if not neighbor_df.empty:
        logger.info(
            "Neighbor stats -> zero: %d, min: %s, median: %s, mean: %.2f, max: %s",
            len([c for c in neighbor_df["neighbor_count"] if c == 0]),
            neighbor_df["neighbor_count"].min(),
            neighbor_df["neighbor_count"].median(),
            neighbor_df["neighbor_count"].mean(),
            neighbor_df["neighbor_count"].max(),
        )
    else:
        logger.info("Neighbor stats -> dataframe empty; no co-occurrence found.")


def _collect_tickers(source_ticker, mentioned) -> Iterable[str]:
    tickers = set()
    if source_ticker:
        c = canonicalize_ticker(str(source_ticker))
        if c:
            tickers.add(c)
    if isinstance(mentioned, np.ndarray):
        mentioned_iter = mentioned.tolist()
    elif isinstance(mentioned, (list, tuple, set)):
        mentioned_iter = list(mentioned)
    else:
        mentioned_iter = []

    for t in mentioned_iter:
        if not t:
            continue
        c = canonicalize_ticker(str(t))
        if c:
            tickers.add(c)
    return {t for t in tickers if t}


def canonicalize_ticker(ticker: Optional[str]) -> Optional[str]:
    if ticker is None:
        return None
    norm = str(ticker).strip().upper().lstrip("$")
    if not norm:
        return None
    if norm in EXCLUDED_TICKERS:
        return None
    return CANONICAL_MAP.get(norm, norm)


__all__ = ["run_cooccurrence"]

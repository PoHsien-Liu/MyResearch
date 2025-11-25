"""Compute per-company volume stats (raw vs augmented by mentions)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from STARE.utils.logger import setup_logger
from STARE.utils.paths import dataset_paths, ensure_dir, indices_dir

# Canonical map需與 extract_mentions 對齊（品牌/股別合併回資料集標準 ticker）
CANONICAL_MAP = {
    "GOOGL": "GOOG",
    "META": "FB",
    "BRK-B": "BRK-A",
    "BRK.B": "BRK-A",
}
from STARE.utils.seed import set_seed


def run_volume_stats(args) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "volume_stats.log"
    logger = setup_logger("stare.volume_stats", log_file=log_file)

    input_path = output_dir / "cleaned_with_mentions.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    logger.info("Loading cleaned_with_mentions from %s", input_path)
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError("Input dataframe is empty.")

    total_docs = len(df)
    valid_tickers = _load_valid_tickers(args.dataset_name)
    logger.info("Loaded %d valid tickers from dataset", len(valid_tickers))
    raw_count: Dict[str, int] = {}
    aug_docs: Dict[str, Set[str]] = {}

    for idx, row in df.iterrows():
        uid = _row_uid(row, idx)
        source = _norm_ticker(row.get("source_ticker"))
        mentioned = _collect_tickers(row.get("mentioned_tickers"))

        if source:
            raw_count[source] = raw_count.get(source, 0) + 1
            aug_docs.setdefault(source, set()).add(uid)

        for t in mentioned:
            aug_docs.setdefault(t, set()).add(uid)

    # 確保所有合法 ticker 都被列出，即便沒有任何出現
    rows = []
    ticker_iter = valid_tickers if valid_tickers else set(aug_docs.keys()) | set(raw_count.keys())
    for ticker in sorted(ticker_iter):
        raw = raw_count.get(ticker, 0)
        aug = len(aug_docs.get(ticker, set()))
        rows.append(
            {
                "ticker": ticker,
                "raw_count": raw,
                "raw_share": raw / total_docs if total_docs else 0,
                "aug_count": aug,
                "aug_share": aug / total_docs if total_docs else 0,
                "augmentation_ratio": (aug / raw) if raw > 0 else np.nan,
            }
        )

    vol_df = pd.DataFrame(rows)
    if not vol_df.empty:
        vol_df = vol_df.sort_values("aug_count", ascending=False)

    out_path = output_dir / "company_volume_stats.csv"
    vol_df.to_csv(out_path, index=False)

    logger.info(
        "Wrote volume stats to %s (rows=%d, total_docs=%d)",
        out_path,
        len(vol_df),
        total_docs,
    )


def _row_uid(row, idx: int) -> str:
    source_path = str(row.get("source_path") or "")
    raw_id_val = row.get("raw_id")
    raw_id = None
    if raw_id_val is not None:
        raw_id = str(raw_id_val)
        if raw_id.lower() == "nan":
            raw_id = None
    if source_path or raw_id:
        return f"{source_path}::{raw_id or idx}"
    return f"row-{idx}"


def _norm_ticker(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().lstrip("$")
    if not text:
        return None
    return CANONICAL_MAP.get(text, text)


def _collect_tickers(val) -> Set[str]:
    tickers: Set[str] = set()
    if isinstance(val, np.ndarray):
        val_iter = val.tolist()
    else:
        val_iter = val
    if isinstance(val_iter, (list, tuple, set)):
        for t in val_iter:
            if t:
                tickers.add(str(t).strip().upper().lstrip("$"))
    return {t for t in tickers if t}


def _load_valid_tickers(dataset_name: str) -> Set[str]:
    try:
        paths = dataset_paths(dataset_name)
    except Exception:
        return set()

    # 以 news/raw 為主，找不到再退回 price/raw
    candidates: Set[str] = set()
    text_raw = Path(paths.text_path) / "raw"
    if text_raw.exists():
        for csv_path in text_raw.glob("*.csv"):
            tick = _norm_ticker(csv_path.stem)
            if tick:
                candidates.add(tick)
    if candidates:
        return candidates

    price_root = Path(paths.price_path)
    if price_root.is_file():
        tick = _norm_ticker(price_root.stem) or price_root.stem.upper()
        if tick:
            candidates.add(tick)
        return candidates

    search_dirs = [price_root]
    raw_dir = price_root / "raw"
    if raw_dir.exists():
        search_dirs.insert(0, raw_dir)
    for base in search_dirs:
        for csv_path in base.glob("*.csv"):
            tick = _norm_ticker(csv_path.stem)
            if tick:
                candidates.add(tick)
        if candidates:
            break
    return candidates


__all__ = ["run_volume_stats"]

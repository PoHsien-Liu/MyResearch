"""Embedding pipeline for cleaned texts."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from STARE.utils.logger import setup_logger
from STARE.utils.paths import ensure_dir, indices_dir
from STARE.utils.seed import set_seed

# 排除 price/raw 存在但不應處理的優先股代碼
EXCLUDED_TICKERS = {"C-PJ", "CTA-PB", "SPG-PJ", "WFC-PL"}


def run_embed(args) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "embed.log"
    logger = setup_logger("stare.embed", log_file=log_file)

    cleaned_path = output_dir / "cleaned_with_mentions.parquet"
    # 若當前 embed_model 路徑下沒有 cleaned，退回 default slug
    if not cleaned_path.exists():
        default_dir = indices_dir(args.dataset_name, None)
        alt_path = default_dir / "cleaned_with_mentions.parquet"
        if alt_path.exists():
            cleaned_path = alt_path
            logger.info("Using cleaned_with_mentions from default path: %s", cleaned_path)
        else:
            raise FileNotFoundError(f"cleaned_with_mentions.parquet not found at {cleaned_path} or {alt_path}")

    logger.info("Loading cleaned_with_mentions from %s", cleaned_path)
    df = pd.read_parquet(cleaned_path)
    if df.empty:
        raise RuntimeError("Input dataframe is empty; nothing to embed.")

    allowed = _load_allowed_tickers(output_dir)
    if allowed:
        before = len(df)
        df = df[df["source_ticker"].str.upper().isin(allowed)].reset_index(drop=True)
        logger.info("Filtered rows by allowed tickers: %d -> %d", before, len(df))

    if getattr(args, "max_rows", None):
        df = df.head(args.max_rows).reset_index(drop=True)
        logger.info("Truncated rows to max_rows=%d", args.max_rows)

    model_name = args.embed_model or "FinLang/finance-embeddings-investopedia"
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    batch_size = getattr(args, "batch_size", 32)
    texts = df["text"].astype(str).tolist()
    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    vectors = np.vstack(embeddings)
    logger.info("Embeddings shape: %s", vectors.shape)

    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, vectors)
    logger.info("Saved embeddings to %s", emb_path)

    meta_cols = [c for c in df.columns if c != "text"]
    missing_cols = [c for c in meta_cols if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            df[c] = None
    meta_df = df[meta_cols].copy()
    meta_df["text"] = df["text"]
    meta_path = output_dir / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)
    logger.info("Saved metadata to %s", meta_path)


def _load_allowed_tickers(output_dir: Path) -> set[str]:
    # price/raw sibling of indices dataset; assume dataset name = output_dir parts[-3]
    dataset_slug = output_dir.parent.name  # e.g., CMIN
    repo_root = output_dir.parents[2]
    price_dir = repo_root / "datasets" / dataset_slug / (dataset_slug + "-US") / "price" / "raw"
    if not price_dir.exists():
        price_dir = repo_root / "datasets" / dataset_slug / "price" / "raw"
    if not price_dir.exists():
        return set()
    allowed = {p.stem.upper() for p in price_dir.glob("*.csv")}
    allowed = {t for t in allowed if t not in EXCLUDED_TICKERS}
    return allowed


__all__ = ["run_embed"]

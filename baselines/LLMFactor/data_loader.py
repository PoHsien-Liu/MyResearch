"""Sample construction for LLMFactor SKGP baseline (test-only)."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.config.datasets import resolve_dataset_paths  # noqa: E402
from common.data.loader import get_record, list_trading_days, load_texts_for_day  # noqa: E402
from common.io.results import safe_name  # noqa: E402
from STARE.utils.paths import indices_dir, embed_model_slug  # noqa: E402


CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,10})")
UPPER_TICKER_RE = re.compile(r"\b([A-Z]{2,5})(?:\b|[^a-zA-Z])")


@dataclass
class DayNews:
    date: str
    texts: List[str]
    truncated: bool = False


@dataclass
class Sample:
    sample_id: str
    ticker: str
    prediction_date: str
    ground_truth: str
    price_context: str
    context_returns: List[Dict]
    news_by_day: List[DayNews]
    related_candidates: List[str]


def _limit_texts(texts: List[str], max_per_day: int) -> Tuple[List[str], bool]:
    if max_per_day is None or max_per_day <= 0:
        return texts, False
    if len(texts) <= max_per_day:
        return texts, False
    return texts[:max_per_day], True


def _load_neighbor_map(
    dataset_name: str,
    *,
    outputs_dir: Optional[str] = None,
    embed_model: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    base_dir = outputs_dir or os.getenv("OUTPUTS_DIR")
    if base_dir:
        base_path = Path(base_dir)
        primary = base_path / "indices" / dataset_name.upper() / embed_model_slug(embed_model) / "company_neighbors.json"
    else:
        primary = indices_dir(dataset_name, embed_model) / "company_neighbors.json"

    candidates = [primary]
    if embed_model:
        # fallback to default slug if embed-specific file is missing
        if base_dir:
            candidates.append(base_path / "indices" / dataset_name.upper() / "default" / "company_neighbors.json")
        else:
            candidates.append(indices_dir(dataset_name, None) / "company_neighbors.json")

    chosen: Optional[Path] = None
    for path in candidates:
        if path.exists():
            chosen = path
            break

    if not chosen:
        cmd = f"python -m STARE.main --task cooccurrence --dataset_name {dataset_name} --embed_model {embed_model or 'default'}"
        raise FileNotFoundError(
            f"company_neighbors.json not found. Searched: {[str(p) for p in candidates]}. "
            "Please run STARE cooccurrence task first, e.g.: "
            f"{cmd}"
        )
    try:
        with chosen.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.upper(): {kk.upper(): int(vv) for kk, vv in vals.items()} for k, vals in data.items()}
    except Exception:
        return {}


def _top_neighbors(neighbor_map: Dict[str, Dict[str, int]], ticker: str, top_n: int) -> List[str]:
    if not neighbor_map or top_n <= 0:
        return []
    entries = neighbor_map.get(ticker.upper()) or {}
    sorted_items = sorted(entries.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in sorted_items[:top_n] if t.upper() != ticker.upper()]


def build_samples(args, logger) -> List[Sample]:
    paths = resolve_dataset_paths(args.dataset_name, args.base_data_dir)
    price_dir = paths.price_dir
    tweet_dir = paths.tweet_dir
    news_csv_dir = args.news_csv_dir or (tweet_dir if args.dataset_name.upper() == "CMIN" else None)

    samples_meta = list_trading_days(
        dataset_name=args.dataset_name,
        price_dir=price_dir,
        mode="test",
        seq_len=args.seq_len,
        split_root=args.splits_dir,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
        logger=logger,
    )
    if args.max_samples is not None and args.max_samples >= 0:
        samples_meta = samples_meta[: args.max_samples]

    neighbor_map: Dict[str, Dict[str, int]] = {}
    if args.top_related > 0:
        neighbor_map = _load_neighbor_map(
            dataset_name=args.dataset_name,
            outputs_dir=args.outputs_dir,
            embed_model=args.embed_model,
        )

    samples: List[Sample] = []
    for entry in samples_meta:
        try:
            rec = get_record(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=entry["date"],
                price_dir=price_dir,
                tweet_dir=tweet_dir,
                news_csv_dir=news_csv_dir,
                seq_len=args.seq_len,
                label_strategy=args.label_strategy,
                neg_threshold=args.neg_threshold,
                pos_threshold=args.pos_threshold,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Skip %s %s due to data error: %s", entry["ticker"], entry["date"], exc)
            continue

        context_returns = rec.get("price", {}).get("context_returns", [])
        price_lines = []
        for ctx in context_returns:
            pct = ctx.get("ret", 0.0) * 100
            sign = "+" if pct >= 0 else ""
            adj_close = ctx.get("adj_close")
            price_part = f" (adj_close={adj_close:.2f})" if isinstance(adj_close, (int, float)) else ""
            price_lines.append(f"- {ctx.get('date', '')}: {sign}{pct:.2f}%{price_part}")
        price_context = "Recent price trend (previous trading days):\n" + "\n".join(price_lines) if price_lines else "No recent price context is available."

        news_by_day: List[DayNews] = []
        for day in rec.get("text_window_dates", []):
            texts_raw = load_texts_for_day(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=day,
                tweet_dir=tweet_dir,
                news_csv_dir=news_csv_dir,
                logger=logger,
            )
            texts_cleaned = [t.get("text", "").strip() for t in texts_raw if t.get("text", "").strip()]
            limited, truncated = _limit_texts(texts_cleaned, args.max_news_per_day)
            news_by_day.append(DayNews(date=day, texts=limited, truncated=truncated))

        related = _top_neighbors(neighbor_map, entry["ticker"], args.top_related) if args.top_related > 0 else []

        samples.append(
            Sample(
                sample_id=f"{entry['ticker']}_{entry['date']}",
                ticker=entry["ticker"],
                prediction_date=entry["date"],
                ground_truth=rec.get("price", {}).get("label", entry.get("label")),
                price_context=price_context,
                context_returns=context_returns,
                news_by_day=news_by_day,
                related_candidates=related,
            )
        )

    logger.info("Built %d samples (from %d candidates).", len(samples), len(samples_meta))
    return samples


__all__ = ["Sample", "DayNews", "build_samples"]

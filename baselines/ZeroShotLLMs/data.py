"""Sample construction (test-only) for ZeroShotLLMs baselines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.loader import get_record, list_trading_days, load_texts_for_day  # noqa: E402
import prompts as shared_prompts  # noqa: E402


@dataclass
class Sample:
    sample_id: str
    ticker: str
    prediction_date: str
    ground_truth: str
    price_context: str
    news_by_day: List[shared_prompts.DayNews]


def _limit_texts(texts: List[str], max_per_day: int) -> Tuple[List[str], bool]:
    if max_per_day is None or max_per_day <= 0:
        return texts, False
    if len(texts) <= max_per_day:
        return texts, False
    return texts[:max_per_day], True


def build_samples(args, logger) -> List[Sample]:
    samples_meta = list_trading_days(
        dataset_name=args.dataset_name,
        price_dir=args.price_dir,
        mode="test",  # enforce test split only
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

    samples: List[Sample] = []
    for entry in samples_meta:
        try:
            rec = get_record(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=entry["date"],
                price_dir=args.price_dir,
                tweet_dir=args.tweet_dir,
                news_csv_dir=args.news_csv_dir,
                seq_len=args.seq_len,
                label_strategy=args.label_strategy,
                neg_threshold=args.neg_threshold,
                pos_threshold=args.pos_threshold,
                logger=logger,
            )
        except Exception as exc:
            logger.warning(f"Skip {entry['ticker']} {entry['date']} due to data error: {exc}")
            continue

        news_by_day: List[shared_prompts.DayNews] = []
        for day in rec.get("text_window_dates", []):
            texts_raw = load_texts_for_day(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=day,
                tweet_dir=args.tweet_dir,
                news_csv_dir=args.news_csv_dir,
                logger=logger,
            )
            texts_cleaned = [t.get("text", "").strip() for t in texts_raw if t.get("text", "").strip()]
            limited, truncated = _limit_texts(texts_cleaned, args.max_news_per_day)
            news_by_day.append(shared_prompts.DayNews(date=day, texts=limited, truncated=truncated))

        price_context = shared_prompts.format_price_context(rec.get("price", {}).get("context_returns", []))
        samples.append(
            Sample(
                sample_id=f"{entry['ticker']}_{entry['date']}",
                ticker=entry["ticker"],
                prediction_date=entry["date"],
                ground_truth=rec.get("price", {}).get("label", entry.get("label")),
                price_context=price_context,
                news_by_day=news_by_day,
            )
        )

    logger.info(f"Built {len(samples)} samples (from {len(samples_meta)} candidates).")
    return samples


__all__ = ["Sample", "build_samples"]

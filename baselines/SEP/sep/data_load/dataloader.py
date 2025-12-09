from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from common.data.loader import (
    DEFAULT_NEG_THRESHOLD,
    DEFAULT_POS_THRESHOLD,
    get_record,
    list_trading_days,
    load_texts_for_day,
)
from summarize_module.summarizer import Summarizer


class DataLoader:
    def __init__(
        self,
        args,
        summarizer: Optional[Summarizer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.dataset_name = getattr(args, "dataset_name", "SAMPLE")
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.news_csv_dir = getattr(args, "news_csv_dir", None)
        self.seq_len = args.seq_len
        self.train_ratio = getattr(args, "train_ratio", 0.8)
        self.split_seed = getattr(args, "split_seed", 42)
        self.label_strategy = getattr(args, "label_strategy", "legacy")
        self.neg_threshold = getattr(args, "neg_threshold", DEFAULT_NEG_THRESHOLD)
        self.pos_threshold = getattr(args, "pos_threshold", DEFAULT_POS_THRESHOLD)
        self.splits_dir = getattr(args, "splits_dir", None)
        self.max_tweets_per_day = getattr(args, "max_tweets_per_day", 50)
        self.summarizer = summarizer or Summarizer()
        self.logger = logger

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _warn(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)

    def _build_summary(self, ticker: str, target_date: str) -> Optional[str]:
        """Summarize tweets/news within the seq_len window ending at target_date."""
        try:
            record = get_record(
                dataset_name=self.dataset_name,
                ticker=ticker,
                date=target_date,
                price_dir=self.price_dir,
                tweet_dir=self.tweet_dir,
                news_csv_dir=self.news_csv_dir,
                seq_len=self.seq_len,
                label_strategy=self.label_strategy,
                neg_threshold=self.neg_threshold,
                pos_threshold=self.pos_threshold,
                logger=self.logger,
            )
        except Exception as exc:
            self._warn(f"[DataLoader] skip {ticker} {target_date}: {exc}")
            return None

        summary_all = ""
        text_dates = record.get("text_window_dates", [])
        for seq_date in text_dates:
            texts = load_texts_for_day(
                dataset_name=self.dataset_name,
                ticker=ticker,
                date=seq_date,
                tweet_dir=self.tweet_dir,
                news_csv_dir=self.news_csv_dir,
                logger=self.logger,
            )
            tweet_texts = [t.get("text") for t in texts if t.get("text")]
            if not tweet_texts:
                continue
            if self.max_tweets_per_day and self.max_tweets_per_day > 0:
                tweet_texts = tweet_texts[: self.max_tweets_per_day]
            summary = self.summarizer.get_summary(ticker, tweet_texts, date=seq_date)
            if summary and self.summarizer.is_informative(summary):
                summary_all += f"{seq_date}\n{summary}\n\n"
        return summary_all.rstrip() if summary_all else None

    def load(self, flag: str) -> pd.DataFrame:
        """Load train/test splits using the shared loader and summarize texts."""
        mode = "train" if flag == "train" else "test"
        data = pd.DataFrame()

        samples = list_trading_days(
            dataset_name=self.dataset_name,
            price_dir=self.price_dir,
            mode=mode,
            seq_len=self.seq_len,
            split_root=getattr(self, "splits_dir", None),
            train_ratio=self.train_ratio,
            split_seed=self.split_seed,
            label_strategy=self.label_strategy,
            neg_threshold=self.neg_threshold,
            pos_threshold=self.pos_threshold,
            logger=self.logger,
        )

        for sample in samples:
            ticker = sample["ticker"]
            target_date = sample["date"]
            target = sample["label"]
            summary_all = self._build_summary(ticker, target_date)
            if summary_all:
                row = {
                    "ticker": ticker,
                    "summary": summary_all,
                    "target": target,
                    "prediction_date": target_date,
                    "sample_id": f"{ticker}_{target_date}",
                }
                data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
        return data

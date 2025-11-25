import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from summarize_module.summarizer import Summarizer
from common.data.loader import list_trading_days, get_record, load_texts_for_day


class DataLoader:
    def __init__(self, args, logger):
        self.logger = logger
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.news_csv_dir = getattr(args, "news_csv_dir", None)
        self.seq_len = args.seq_len
        self.summarizer = Summarizer(args, logger)
        self.summary_cache = {}
        self.dataset_name = args.dataset_name
        self.summary_min_tweets = getattr(args, "summary_min_tweets", 1)
        self.summary_max_tweets = max(1, getattr(args, "summary_max_tweets", 50))
        self.summary_batch_size = getattr(args, "summary_batch_size", getattr(args, "batch_size", 8))
        self.train_ratio = getattr(args, "train_ratio", 0.8)
        self.split_seed = getattr(args, "split_seed", 42)
        self.splits_dir = getattr(args, "splits_dir", None)
        self.label_strategy = getattr(args, "label_strategy", "legacy")
        self.neg_threshold = getattr(args, "neg_threshold", -0.005)
        self.pos_threshold = getattr(args, "pos_threshold", 0.0055)
        self.neutral_skipped = 0
        if self.dataset_name == "ACL18":
            self.start_date = datetime(2014, 1, 1)
            self.end_date = datetime(2016, 1, 1)
        else:
            self.start_date = None
            self.end_date = None

    def get_cached_summary(self, ticker, date_str):
        cache_key = f"{ticker}_{date_str}"
        return self.summary_cache.get(cache_key)

    def cache_summary(self, ticker, date_str, summary, label):
        cache_key = f"{ticker}_{date_str}"
        self.summary_cache[cache_key] = summary
        self.summary_cache[cache_key + "_label"] = label

    def load(self, flag):
        rows = []
        # 1) Enumerate candidate samples from shared splits
        samples = list_trading_days(
            dataset_name=self.dataset_name,
            price_dir=self.price_dir,
            mode=flag,
            seq_len=self.seq_len,
            split_root=self.splits_dir,
            train_ratio=self.train_ratio,
            split_seed=self.split_seed,
            label_strategy=self.label_strategy,
            neg_threshold=self.neg_threshold,
            pos_threshold=self.pos_threshold,
            logger=self.logger,
        )
        stats = getattr(list_trading_days, "last_stats", {})
        self.neutral_skipped = stats.get("neutral_skipped", 0)
        if self.logger and stats:
            self.logger.info(
                f"Sample stats -> total_candidates={stats.get('total_candidates',0)} "
                f"split_candidates={stats.get('split_candidates',0)} "
                f"neutral_skipped={self.neutral_skipped} kept={stats.get('kept_samples', len(samples))}"
            )

        # 2) Build summary jobs for all unique (ticker, date) across windows
        records_by_key = {}
        jobs = []
        seen_jobs = set()

        with tqdm(total=len(samples), desc="Collecting summary jobs", position=0, leave=True) as bar:
            for sample in samples:
                ticker = sample["ticker"]
                end_date = sample["date"]

                rec = self._safe_get_record(ticker, end_date)
                if rec is None:
                    bar.update(1)
                    continue
                records_by_key[(ticker, end_date)] = rec

                window_dates = rec.get("text_window_dates") or [rec["date"]]
                ordered_dates = []
                seen = set()
                for d in window_dates:
                    if d in seen:
                        continue
                    ordered_dates.append(d)
                    seen.add(d)

                for date_str in ordered_dates:
                    job_key = (ticker, date_str)
                    if job_key in seen_jobs:
                        continue

                    # Load texts (prefetch end_date texts from record if available)
                    if date_str == rec["date"] and rec.get("texts") is not None:
                        texts = rec["texts"]
                    else:
                        texts = load_texts_for_day(
                            dataset_name=self.dataset_name,
                            ticker=ticker,
                            date=date_str,
                            tweet_dir=self.tweet_dir,
                            news_csv_dir=self.news_csv_dir,
                            logger=self.logger,
                        )

                    plain_texts = [t.get("text", "") for t in texts if t.get("text")]
                    if len(plain_texts) < self.summary_min_tweets:
                        continue

                    trimmed = plain_texts[:self.summary_max_tweets]
                    # Skip if already cached on disk/memory
                    if self.summarizer.get_cached_summary(ticker, date_str) is not None:
                        seen_jobs.add(job_key)
                        continue

                    jobs.append({
                        "ticker": ticker,
                        "date": date_str,
                        "texts": trimmed,
                    })
                    seen_jobs.add(job_key)
                bar.update(1)

        # 3) Run batch summarization once for all jobs
        if jobs:
            # Chunking inside summarize_batch is handled by underlying LLM
            self.logger.info(f"Running batch summary for {len(jobs)} jobs (batch_size={self.summary_batch_size})")
            self.summarizer.summarize_batch(jobs, batch_size=self.summary_batch_size)
        else:
            self.logger.info("No new summary jobs (all cached or below min tweets)")

        # 4) Build rows using cached summaries only
        with tqdm(total=len(samples), desc="Assembling samples", position=0, leave=True) as bar2:
            for sample in samples:
                ticker = sample["ticker"]
                end_date = sample["date"]
                cache_key = f"{ticker}_{end_date}"

                if cache_key in self.summary_cache:
                    rows.append({
                        "ticker": ticker,
                        "end_date": end_date,
                        "summary": self.summary_cache[cache_key],
                        "target": self.summary_cache[cache_key + "_label"],
                    })
                    bar2.update(1)
                    continue

                record = records_by_key.get((ticker, end_date))
                if record is None:
                    bar2.update(1)
                    continue

                summary_text = self._build_window_summary(ticker, record)
                label = record["price"]["label"]
                self.cache_summary(ticker, end_date, summary_text, label)

                rows.append({
                    "ticker": ticker,
                    "end_date": end_date,
                    "summary": summary_text.strip(),
                    "target": label,
                })
                bar2.update(1)

        return pd.DataFrame(rows)

    def _safe_get_record(self, ticker, end_date):
        try:
            return get_record(
                dataset_name=self.dataset_name,
                ticker=ticker,
                date=end_date,
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
            if self.logger:
                self.logger.warning(f"[DataLoader] skip {ticker} {end_date}: {exc}")
            return None

    def _build_window_summary(self, ticker, record):
        window_dates = record.get("text_window_dates") or [record["date"]]
        ordered_dates = []
        seen = set()
        for date_str in window_dates:
            if date_str in seen:
                continue
            ordered_dates.append(date_str)
            seen.add(date_str)

        daily_summaries = []
        for date_str in ordered_dates:
            prefetched = record["texts"] if date_str == record["date"] else None
            day_summary = self._summarize_single_day(ticker, date_str, prefetched_texts=prefetched)
            if day_summary:
                daily_summaries.append(f"[{date_str}] {day_summary}")

        return "\n".join(daily_summaries).strip()

    def _summarize_single_day(self, ticker, date_str, prefetched_texts=None):
        texts = prefetched_texts
        if texts is None:
            texts = load_texts_for_day(
                dataset_name=self.dataset_name,
                ticker=ticker,
                date=date_str,
                tweet_dir=self.tweet_dir,
                news_csv_dir=self.news_csv_dir,
                logger=self.logger,
            )

        plain_texts = [t.get("text", "") for t in texts if t.get("text")]
        if len(plain_texts) < self.summary_min_tweets:
            return ""

        trimmed = plain_texts[:self.summary_max_tweets]
        # Prefer cached summary (populated by summarize_batch); fallback to single-call
        cached = self.summarizer.get_cached_summary(ticker, date_str)
        if cached is not None:
            summary_text = cached
        else:
            summary_text = self.summarizer.get_summary(ticker, date_str, trimmed)
        if summary_text and not self.summarizer.is_informative(summary_text):
            return ""
        return summary_text.strip()

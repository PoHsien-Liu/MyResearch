import re
import os
import json
from pathlib import Path

from models.llm import LLaMALLM
from utils.prompts import NEWS_SUMMARY_INSTRUCTION

class Summarizer:
    def __init__(self, args, logger, method_name="TDMLLM"):
        self.logger = logger
        self.summarize_prompt = NEWS_SUMMARY_INSTRUCTION
        self.llm = LLaMALLM(args, logger)
        self.method_name = method_name
        self.max_new_tokens = getattr(args, "summary_max_new_tokens", 160)
        # Runtime/context info
        self.dataset_name = getattr(args, "dataset_name", "UNKNOWN")
        self.model_name = getattr(args, "base_model", "model")

        # Initialize paths for summary storage
        # Expect args.summary_cache_dir to point to OUTPUTS_DIR/cache/summaries/{dataset}/{model}/{method}
        cache_root = getattr(args, "summary_cache_dir", None)
        if cache_root:
            self.cache_root = Path(cache_root)
        else:
            # Fallback (legacy): under tweet_dir/../summaries
            tweet_dir = Path(args.tweet_dir)
            self.cache_root = (tweet_dir.parent / "summaries")
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # In-memory cache to avoid duplicate reads within a run
        self.summary_cache = {}

        self.logger.info(f"Summary cache root: {self.cache_root}")
        self.logger.info(f"Using base model: {self.model_name}")
        self.logger.info(f"Using method: {self.method_name}")

    def get_summary_path(self, ticker, date):
        """Get the path for a summary file."""
        ticker_dir = self.cache_root / ticker
        ticker_dir.mkdir(exist_ok=True)
        return ticker_dir / f"{date}.json"

    def load_existing_summary(self, ticker, date):
        """Load an existing summary if it exists."""
        summary_path = self.get_summary_path(ticker, date)
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error loading summary for {ticker} on {date}: {e}")
                return None
        return None

    def save_summary(self, ticker, date, tweet_data, prompt, summary):
        """Save a summary to file."""
        data = {
            "ticker": ticker,
            "date": date,
            "tweet_data": tweet_data,
            "prompt": prompt,
            "summary": summary,
            "model": self.model_name,
            "method": self.method_name,
            "dataset": self.dataset_name,
        }
        
        summary_path = self.get_summary_path(ticker, date)
        try:
            with open(summary_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Saved summary for {ticker} on {date}")
        except Exception as e:
            self.logger.error(f"Error saving summary for {ticker} on {date}: {e}")

        # Update memory cache
        key = f"{ticker}_{date}"
        self.summary_cache[key] = summary or ""

    def get_cached_summary(self, ticker: str, date: str):
        """Return cached summary text if present (memory or disk)."""
        key = f"{ticker}_{date}"
        if key in self.summary_cache:
            return self.summary_cache[key]

        loaded = self.load_existing_summary(ticker, date)
        if loaded and isinstance(loaded, dict):
            summary = loaded.get("summary", "")
            self.summary_cache[key] = summary or ""
            return self.summary_cache[key]
        return None

    def get_summary(self, ticker, date_str, tweets):
        # First check if summary already exists
        cached = self.get_cached_summary(ticker, date_str)
        if cached is not None:
            self.logger.info(f"Found cached summary for {ticker} on {date_str}")
            return cached

        # If no existing summary, generate new one
        summary = None
        prompt = ""
        if tweets:
            prompt = self.summarize_prompt.format(ticker=ticker, news=tweets)
            summary = self.llm("", prompt, max_new_tokens=self.max_new_tokens)

        self.logger.info(f"\n📌 Summary for {ticker} on {date_str}")
        self.logger.info(f"🗞️ Tweet count: {len(tweets)}")
        self.logger.info(f"🧾 Summary: {summary}")

        # Save the new summary
        self.save_summary(ticker, date_str, tweets, prompt, summary)

        return summary
    
    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)

    def summarize_batch(self, jobs: list[dict], batch_size: int = 8) -> dict:
        """
        Batch summarize multiple days.

        Args:
            jobs: List of dicts {"ticker": str, "date": "YYYY-MM-DD", "texts": List[str]}
            batch_size: number of jobs per batch for LLM

        Returns:
            Dict keyed by (ticker, date) -> summary string
        """
        if not jobs:
            return {}

        to_run = []
        prompts = []
        metas = []  # parallel arrays for mapping back

        # Prepare prompts; skip cached
        for job in jobs:
            ticker = job.get("ticker")
            date = job.get("date")
            texts = job.get("texts") or []
            cached = self.get_cached_summary(ticker, date)
            if cached is not None:
                continue
            user_prompt = self.summarize_prompt.format(ticker=ticker, news=texts)
            prompts.append(user_prompt)
            metas.append((ticker, date, texts, user_prompt))
            to_run.append(job)

        results: dict = {}

        if metas:
            system_list = [""] * len(metas)
            # Use underlying batch inference if available
            outputs = self.llm.batch_inference(system_list, prompts, max_new_tokens=self.max_new_tokens)
            for (ticker, date, texts, prompt), summary in zip(metas, outputs):
                if summary is None:
                    summary = ""
                self.save_summary(ticker, date, texts, prompt, summary)
                results[(ticker, date)] = summary

        # For cached ones, also populate return dict
        for job in jobs:
            ticker = job.get("ticker")
            date = job.get("date")
            key = (ticker, date)
            if key in results:
                continue
            cached = self.get_cached_summary(ticker, date)
            if cached is not None:
                results[key] = cached

        self.logger.info(
            f"Batch summarized {len(metas)} new / {len(jobs)-len(metas)} cached (batch_size={batch_size})"
        )
        return results

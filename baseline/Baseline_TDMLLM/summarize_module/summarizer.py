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
        
        # Initialize paths for summary storage
        tweet_dir = Path(args.tweet_dir)
        self.dataset_root = tweet_dir.parent
        self.summaries_dir = self.dataset_root / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)
        
        # Get model name for the summary file
        self.model_name = Path(args.base_model).name
        self.model_dir = self.summaries_dir / self.model_name
        self.model_dir.mkdir(exist_ok=True)
        
        # Create method-specific directory
        self.method_dir = self.model_dir / self.method_name
        self.method_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Summary directory: {self.summaries_dir}")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Method directory: {self.method_dir}")
        self.logger.info(f"Using model: {self.model_name}")
        self.logger.info(f"Using method: {self.method_name}")

    def get_summary_path(self, ticker, date):
        """Get the path for a summary file."""
        ticker_dir = self.method_dir / ticker
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
            "method": self.method_name
        }
        
        summary_path = self.get_summary_path(ticker, date)
        try:
            with open(summary_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Saved summary for {ticker} on {date}")
        except Exception as e:
            self.logger.error(f"Error saving summary for {ticker} on {date}: {e}")

    def get_summary(self, ticker, date_str, tweets):
        # First check if summary already exists
        existing_summary = self.load_existing_summary(ticker, date_str)
        if existing_summary:
            self.logger.info(f"Found existing summary for {ticker} on {date_str}")
            return existing_summary["summary"]

        # If no existing summary, generate new one
        summary = None
        prompt = ""
        if tweets:
            prompt = self.summarize_prompt.format(ticker=ticker, news=tweets)
            summary = self.llm("", prompt)

        self.logger.info(f"\n📌 Summary for {ticker} on {date_str}")
        self.logger.info(f"🗞️ Tweet count: {len(tweets)}")
        self.logger.info(f"🧾 Summary: {summary}")

        # Save the new summary
        self.save_summary(ticker, date_str, tweets, prompt, summary)

        return summary
    
    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)
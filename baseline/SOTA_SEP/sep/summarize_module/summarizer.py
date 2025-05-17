from utils.llm import OpenAILLM, LLaMALLM
from utils.prompts import SUMMARIZE_INSTRUCTION, MY_SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re
import json
import os
from pathlib import Path

class Summarizer:
    def __init__(self, args, method_name="SEP"):
        self.summarize_prompt = MY_SUMMARIZE_INSTRUCTION
        # self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        # self.llm = OpenAILLM()
        self.llm = LLaMALLM()
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        self.method_name = method_name
        
        # Initialize paths for summary storage
        tweet_dir = Path(args.tweet_dir)
        self.dataset_root = tweet_dir.parent
        self.summaries_dir = self.dataset_root / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)
        
        # Get model name for the summary file
        self.model_name = "Meta-Llama-3.1-8B-Instruct"
        self.model_dir = self.summaries_dir / self.model_name
        self.model_dir.mkdir(exist_ok=True)
        
        # Create method-specific directory
        self.method_dir = self.model_dir / self.method_name
        self.method_dir.mkdir(exist_ok=True)

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
            except json.JSONDecodeError:
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
        except Exception:
            pass

    def save_summary_data(self, file_path, ticker, date, tweet_data, prompt, summary):
        data = {
            "ticker": ticker,
            "date": date,
            "tweet_data": tweet_data,
            "prompt": prompt,
            "summary": summary
        }

        # 讀取現有的 JSON 檔案
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    existing_data = json.load(f)  # 讀取 JSON 陣列
                except json.JSONDecodeError:
                    existing_data = []  # 如果讀取失敗，初始化為空陣列
        else:
            existing_data = []  # 檔案不存在時，初始化為空陣列
        
        # 加入新的數據
        existing_data.append(data)

        # 重新寫回完整的 JSON 陣列
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def get_summary(self, ticker, date_str, tweets):
        # First check if summary already exists
        existing_summary = self.load_existing_summary(ticker, date_str)
        if existing_summary:
            return existing_summary["summary"]

        # If no existing summary, generate new one
        summary = None
        if tweets:
            prompt = self.summarize_prompt.format(
                ticker=ticker,
                examples=self.summarize_examples,
                tweets="\n".join(tweets)
            )

            # gpt-3.5-turbo-16k
            # while len(self.enc.encode(prompt)) > 16385:
            #     tweets = tweets[:-1]
            #     prompt = self.summarize_prompt.format(
            #                             ticker = ticker,
            #                             examples = self.summarize_examples,
            #                             tweets = "\n".join(tweets))

            summary = self.llm(prompt)

            # Save the new summary
            self.save_summary(ticker, date_str, tweets, prompt, summary)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)

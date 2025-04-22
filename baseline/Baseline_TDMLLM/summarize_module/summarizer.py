import re
import os
import json

from models.llm import LLaMALLM
from utils.prompts import NEWS_SUMMARY_INSTRUCTION

class Summarizer:
    def __init__(self, args, logger):
        self.logger = logger
        self.summarize_prompt = NEWS_SUMMARY_INSTRUCTION
        self.llm = LLaMALLM(args, logger)

    def save_summary_data(self, file_path, ticker, date, tweet_data, prompt, summary):
        data = {
            "ticker": ticker,
            "date": date,
            "tweet_data": tweet_data,
            "prompt": prompt,
            "summary": summary
        }

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    existing_data = json.load(f) 
                except json.JSONDecodeError:
                    existing_data = [] 
        else:
            existing_data = []  
    
        existing_data.append(data)

        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def get_summary(self, ticker, date_str, tweets):
        summary = None
        prompt = ""
        if tweets != []:
            prompt = self.summarize_prompt.format(ticker= ticker, news=tweets)
            summary = self.llm("", prompt)

        self.logger.info(f"\n📌 Summary for {ticker} on {date_str}")
        self.logger.info(f"🗞️ Tweet count: {len(tweets)}")
        self.logger.info(f"🧾 Summary: {summary}")

        self.save_summary_data("summary_data.json", ticker, date_str, tweets, prompt, summary)

        return summary
    
    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)
from utils.llm import OpenAILLM, LLaMALLM
from utils.prompts import SUMMARIZE_INSTRUCTION, MY_SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re
import json
import os

class Summarizer:
    def __init__(self):
        self.summarize_prompt = MY_SUMMARIZE_INSTRUCTION
        # self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        # self.llm = OpenAILLM()
        self.llm = LLaMALLM()
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

    
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
        summary = None
        if tweets != []:
            prompt = self.summarize_prompt.format(
                                    ticker = ticker,
                                    examples = self.summarize_examples,
                                    tweets = "\n".join(tweets))

            # gpt-3.5-turbo-16k
            # while len(self.enc.encode(prompt)) > 16385:
            #     tweets = tweets[:-1]
            #     prompt = self.summarize_prompt.format(
            #                             ticker = ticker,
            #                             examples = self.summarize_examples,
            #                             tweets = "\n".join(tweets))

            while len(self.llm.tokenizer.encode(prompt)) >= 8192:
                tweets = tweets[:-1]
                prompt = self.summarize_prompt.format(
                                        ticker = ticker,
                                        examples = self.summarize_examples,
                                        tweets = "\n".join(tweets))

            summary = self.llm(prompt)

            self.save_summary_data("summary_data.json", ticker, date_str, tweets, prompt, summary)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)

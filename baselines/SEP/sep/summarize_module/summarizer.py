from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

import tiktoken

from utils.fewshots import SUMMARIZE_EXAMPLES
from utils.llm import VLLMLLM, VLLMSamplingConfig
from utils.prompts import SUMMARIZE_INSTRUCTION


class Summarizer:
    def __init__(
        self,
        *,
        model_name: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        cache_dir: Optional[str] = None,
        llm: Optional[VLLMLLM] = None,
        max_model_len: int = 8192,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        quantization: Optional[str] = "awq",
    ):
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.cache_dir = cache_dir
        self.llm = llm or VLLMLLM(
            model=model_name,
            quantization=quantization,
            max_model_len=max_model_len,
            sampling_config=VLLMSamplingConfig(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            ),
        )
        try:
            self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max(max_model_len - max_new_tokens, 256)

    def get_summary(self, ticker: str, tweets: List[str], date: Optional[str] = None) -> Optional[str]:
        if not tweets:
            return None

        cache_key = None
        if self.cache_dir and date:
            cache_key = self._cache_key(ticker, date, tweets)
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        prompt = self._build_prompt(ticker, tweets)
        prompt = self._truncate_prompt(prompt, tweets, ticker)
        summary = self.llm(prompt)

        if cache_key and summary:
            self._write_cache(cache_key, summary)
        return summary

    def is_informative(self, summary: str) -> bool:
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary or "")

    def _build_prompt(self, ticker: str, tweets: List[str]) -> str:
        return self.summarize_prompt.format(
            ticker=ticker,
            examples=self.summarize_examples,
            tweets="\n".join(tweets),
        )

    def _truncate_prompt(self, prompt: str, tweets: List[str], ticker: str) -> str:
        # Ensure the prompt stays within the model context length
        tweets_trim = list(tweets)
        while self._count_tokens(prompt) > self.max_tokens and tweets_trim:
            tweets_trim = tweets_trim[:-1]
            prompt = self._build_prompt(ticker, tweets_trim)

        if self._count_tokens(prompt) > self.max_tokens:
            prompt = self.summarize_prompt.format(
                ticker=ticker,
                examples="",
                tweets="\n".join(tweets_trim),
            )
            while self._count_tokens(prompt) > self.max_tokens and tweets_trim:
                tweets_trim = tweets_trim[:-1]
                prompt = self.summarize_prompt.format(
                    ticker=ticker,
                    examples="",
                    tweets="\n".join(tweets_trim),
                )
        return prompt

    def _count_tokens(self, prompt: str) -> int:
        if hasattr(self.llm, "tokenizer") and self.llm.tokenizer is not None:
            try:
                return len(self.llm.tokenizer.encode(prompt))
            except Exception:
                pass
        return len(self.enc.encode(prompt))

    def _cache_key(self, ticker: str, date: str, tweets: List[str]) -> str:
        digest = hashlib.sha256("\n".join(tweets).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, ticker, f"{date}_{digest}.json")

    def _read_cache(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("summary")
        except Exception:
            return None

    def _write_cache(self, path: str, summary: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        except Exception:
            return

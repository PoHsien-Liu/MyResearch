"""Reusable FinGPT adapter for baseline methods."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import modeling_utils
from huggingface_hub import hf_hub_download

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore


@dataclass
class GenerationResult:
    text: str
    latency_ms: float


@dataclass
class FinGPTConfig:
    base_model: str
    fingpt_lora: str | None
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool | None = None
    device: str | None = None
    device_map: str | None = None
    torch_dtype: torch.dtype | str | None = None
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: torch.dtype | str | None = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class FinGPTAdapter:
    """Light-weight wrapper around a FinGPT LoRA on top of a base LLM."""

    def __init__(self, config: FinGPTConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.dtype = _resolve_dtype(config.torch_dtype) or (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_map = config.device_map or ("auto" if config.load_in_4bit else None)

        self.logger.info(
            "Loading FinGPT base model %s (dtype=%s, load_in_4bit=%s, device_map=%s)",
            config.base_model,
            self.dtype,
            config.load_in_4bit,
            self.device_map,
        )

        # Ensure transformer parallel constants are initialized (older accelerate versions may leave None)
        if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
            modeling_utils.ALL_PARALLEL_STYLES = {}
        modeling_utils.ALL_PARALLEL_STYLES.setdefault("colwise", None)
        modeling_utils.ALL_PARALLEL_STYLES.setdefault("rowwise", None)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        base_kwargs: dict = {}
        if config.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes is required for 4-bit loading but is not installed. "
                    "Install it or disable --load_in_4bit."
                )
            base_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=_resolve_dtype(config.bnb_4bit_compute_dtype) or torch.float16,
            )
            base_kwargs["device_map"] = self.device_map or "auto"
        else:
            base_kwargs["torch_dtype"] = self.dtype
            if self.device_map:
                base_kwargs["device_map"] = self.device_map

        try:
            base_config = AutoConfig.from_pretrained(config.base_model)
        except ValueError as exc:
            if "`rope_scaling`" not in str(exc):
                raise
            config_file = hf_hub_download(config.base_model, filename="config.json")
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            rope_scaling = config_dict.get("rope_scaling", {})
            if isinstance(rope_scaling, dict):
                factor = rope_scaling.get("factor") or rope_scaling.get("low_freq_factor") or 1.0
            else:
                factor = 1.0
            override = {"name": "dynamic", "factor": factor}
            base_config = AutoConfig.from_pretrained(config.base_model, rope_scaling=override)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            config=base_config,
            **base_kwargs,
        )
        if not config.load_in_4bit and not self.device_map:
            base_model.to(self.device)

        if config.fingpt_lora:
            self.logger.info("Attaching FinGPT LoRA adapter: %s", config.fingpt_lora)
            model = PeftModel.from_pretrained(base_model, config.fingpt_lora, torch_dtype=self.dtype)
        else:
            self.logger.info("No LoRA provided; using base model only.")
            model = base_model
        if not config.load_in_4bit and not self.device_map:
            model.to(self.device)
        model.eval()
        self.model = model
        self.generation_device = self._resolve_generation_device()

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n"

    # ------------------------------------------------------------------
    # Generation APIs
    # ------------------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_kwargs: dict | None = None,
    ) -> GenerationResult:
        return self.batch_generate([(system_prompt, user_prompt)], generation_kwargs=generation_kwargs)[0]

    def batch_generate(
        self,
        prompts: Sequence[Tuple[str, str]],
        generation_kwargs: dict | None = None,
        chunk_size: int | None = None,
    ) -> List[GenerationResult]:
        if not prompts:
            return []

        chunk = max(1, chunk_size or len(prompts))
        all_results: List[GenerationResult] = []
        for start in range(0, len(prompts), chunk):
            subset = prompts[start : start + chunk]
            all_results.extend(self._generate_chunk(subset, generation_kwargs))
        return all_results

    def _generate_chunk(
        self,
        prompts: Sequence[Tuple[str, str]],
        generation_kwargs: dict | None = None,
    ) -> List[GenerationResult]:
        chats = [self.build_chat_prompt(system, user) for system, user in prompts]
        gen_args = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample
            if self.config.do_sample is not None
            else self.config.temperature > 0,
        }
        if generation_kwargs:
            gen_args.update(generation_kwargs)

        start = time.time()
        max_ctx = getattr(self.model.config, "max_position_embeddings", 4096)
        max_ctx = max(1, max_ctx)
        budget = max(1, min(max_ctx - int(gen_args["max_new_tokens"]), self.tokenizer.model_max_length))
        encoded = self.tokenizer(
            chats,
            padding=True,
            truncation=True,
            max_length=budget,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.generation_device)

        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_args,
            )
        latency_ms = (time.time() - start) * 1000.0

        input_len = encoded["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        per_sample_latency = latency_ms / max(1, len(texts))
        return [GenerationResult(text=text.strip(), latency_ms=per_sample_latency) for text in texts]

    def _resolve_generation_device(self) -> torch.device:
        if self.device_map or self.config.load_in_4bit:
            map_info = getattr(self.model, "hf_device_map", None)
            if map_info:
                first = next(iter(map_info.values()))
                if isinstance(first, str):
                    device = torch.device(first)
                    self.logger.info("hf_device_map=%s", map_info)
                    return device
        return self.device

__all__ = ["FinGPTAdapter", "FinGPTConfig", "GenerationResult"]


def _resolve_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        if norm in mapping:
            return mapping[norm]
    raise ValueError(f"Unsupported dtype value: {value}")

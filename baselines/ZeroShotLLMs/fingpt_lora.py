"""FinGPT LoRA backend (HF + PEFT) for ZeroShotLLMs."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from peft import PeftModel  # type: ignore

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prompts as shared_prompts  # noqa: E402
from common.stock_direction import extract_stock_direction_and_value  # noqa: E402


@dataclass
class HFBackend:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def _load_hf_backend(base_model: str, lora_path: str | None, torch_dtype: torch.dtype, device_map: str | None = "auto") -> HFBackend:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_map or "auto",
        torch_dtype=torch_dtype,
    )
    if lora_path:
        base = PeftModel.from_pretrained(base, lora_path)
    base.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return HFBackend(tokenizer=tokenizer, model=base, device=device)


def _render_prompts(tokenizer: AutoTokenizer, prompts: List[shared_prompts.SamplePrompt]) -> List[str]:
    rendered: List[str] = []
    for p in prompts:
        messages = []
        if p.system:
            messages.append({"role": "system", "content": p.system})
        messages.append({"role": "user", "content": p.user})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{p.system}\n\n{p.user}"
        rendered.append(text)
    return rendered


def _normalize_label(label: str) -> str:
    normalized = (label or "").strip().upper()
    if normalized in {"UP", "POSITIVE"}:
        return "UP"
    if normalized in {"DOWN", "NEGATIVE"}:
        return "DOWN"
    return "Unknown"


def _parse_prediction(text: str) -> tuple[str, float | None]:
    direction, value = extract_stock_direction_and_value(text)
    direction = _normalize_label(direction)
    if direction in {"UP", "DOWN"}:
        return direction, value
    cleaned = (text or "").strip().lower()
    if "up" in cleaned and "down" not in cleaned:
        return "UP", value
    if "down" in cleaned and "up" not in cleaned:
        return "DOWN", value
    return "Unknown", value


def run_inference_fingpt(args, samples: List, logger) -> List[Dict]:
    if not samples:
        logger.warning("No samples to evaluate; skipping inference.")
        return []

    base_model = args.base_model or "meta-llama/Llama-2-7b-chat-hf"
    lora_path = args.lora_path or "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
    backend = _load_hf_backend(
        base_model=base_model,
        lora_path=lora_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    predictions: List[Dict] = []
    batch_size = max(1, args.batch_size)
    processing_date = datetime.utcnow().strftime("%Y-%m-%d")

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        prompts: List[shared_prompts.SamplePrompt] = [
            shared_prompts.build_prompt(
                ticker=sample.ticker,
                prediction_date=sample.prediction_date,
                price_context=sample.price_context,
                news_by_day=sample.news_by_day,
            )
            for sample in batch
        ]
        rendered = _render_prompts(backend.tokenizer, prompts)

        tokenized = backend.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(backend.model.device)

        # Prompt length stats (unpadded) for visibility
        attn_mask = tokenized.get("attention_mask")
        if attn_mask is not None:
            prompt_lens = [int(m.sum().item()) for m in attn_mask]
        else:
            prompt_lens = [len(ids) for ids in tokenized["input_ids"]]
        if prompt_lens and logger:
            avg_len = sum(prompt_lens) / len(prompt_lens)
            logger.info(
                f"[PromptTokens] batch_start={start} batch_size={len(prompt_lens)} avg={avg_len:.1f} "
                f"max={max(prompt_lens)} min={min(prompt_lens)}"
            )

        batch_start = time.time()
        with torch.no_grad():
            outputs = backend.model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=backend.tokenizer.eos_token_id,
            )
        input_len = tokenized["input_ids"].shape[1]
        decoded: List[str] = []
        for seq in outputs:
            gen_ids = seq[input_len:]
            if gen_ids.numel() > 0:
                decoded.append(backend.tokenizer.decode(gen_ids, skip_special_tokens=True))
            else:
                # Fallback: decode full output to avoid empty responses.
                decoded.append(backend.tokenizer.decode(seq, skip_special_tokens=True))
        batch_latency_ms = (time.time() - batch_start) * 1000
        per_sample_latency = batch_latency_ms / max(1, len(batch))
        if logger:
            logger.info(
                f"[Batch] start={start} size={len(batch)} prompt_tokens_avg={sum(prompt_lens)/len(prompt_lens):.1f} "
                f"latency_ms={batch_latency_ms:.1f}"
            )

        for sample, prompt, output in zip(batch, prompts, decoded):
            label, value = _parse_prediction(output)
            raw_response = output or ""
            if args.truncate_chars and args.truncate_chars > 0 and raw_response:
                raw_response = raw_response[: args.truncate_chars]

            rec = {
                "sample_id": sample.sample_id,
                "ticker": sample.ticker,
                "prediction_date": sample.prediction_date,
                "processing_date": processing_date,
                "dataset": args.dataset_name,
                "method": args.method_name,
                "model": getattr(args, "model_tag", lora_path or base_model),
                "experiment_name": args.experiment_name,
                "ground_truth": sample.ground_truth,
                "prediction": {"label": label, "confidence": value},
                "raw_response": raw_response if args.store_raw else "",
                "prompts": {"system": prompt.system, "user": prompt.user} if args.store_prompts else None,
                "timing": {"latency_ms": int(per_sample_latency)},
            }
            predictions.append(rec)

    return predictions


__all__ = ["run_inference_fingpt"]

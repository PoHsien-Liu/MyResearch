"""AWQ vLLM backend wrapper for ZeroShotLLMs."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prompts as shared_prompts  # noqa: E402
from inference import VLLMRunner, PromptLike, VLLMConfig  # noqa: E402
from common.stock_direction import extract_stock_direction_and_value  # noqa: E402


def _parse_model_prediction(text: str) -> tuple[str, float | None]:
    direction, value = extract_stock_direction_and_value(text)
    if direction in {"Positive", "Negative"}:
        return direction, value
    cleaned = (text or "").strip().lower()
    if '"prediction"' in cleaned or "prediction" in cleaned:
        if "up" in cleaned and "down" not in cleaned:
            return "Positive", value
        if "down" in cleaned and "up" not in cleaned:
            return "Negative", value
    if "up" in cleaned and "down" not in cleaned:
        return "Positive", value
    if "down" in cleaned and "up" not in cleaned:
        return "Negative", value
    return "Unknown", value


def run_inference_awq(args, samples: List, logger, vllm_cfg: VLLMConfig) -> List[Dict]:
    if not samples:
        logger.warning("No samples to evaluate; skipping inference.")
        return []

    runner = VLLMRunner(vllm_cfg)

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
        requests = [
            PromptLike(
                system=p.system,
                user=p.user,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for p in prompts
        ]

        rendered = runner.render_prompts(requests)
        token_counts = runner.count_tokens(rendered)
        if token_counts and logger:
            avg_len = sum(token_counts) / len(token_counts)
            logger.info(
                f"[PromptTokens] batch_start={start} batch_size={len(token_counts)} avg={avg_len:.1f} "
                f"max={max(token_counts)} min={min(token_counts)}"
            )

        batch_start = time.time()
        outputs = runner.generate(requests, rendered_prompts=rendered)
        batch_latency_ms = (time.time() - batch_start) * 1000
        per_sample_latency = batch_latency_ms / max(1, len(batch))

        for sample, prompt, output in zip(batch, prompts, outputs):
            label, value = _parse_model_prediction(output)
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
                "model": args.model_tag,
                "experiment_name": args.experiment_name,
                "ground_truth": sample.ground_truth,
                "prediction": {"label": label, "confidence": value},
                "raw_response": raw_response if args.store_raw else "",
                "prompts": {"system": prompt.system, "user": prompt.user} if args.store_prompts else None,
                "timing": {"latency_ms": int(per_sample_latency)},
            }
            predictions.append(rec)

    return predictions


__all__ = ["run_inference_awq"]

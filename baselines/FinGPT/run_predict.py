from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config.datasets import resolve_dataset_paths
from common.data.loader import get_record, list_trading_days, load_texts_for_day
from common.io.results import prepare_results_dir, safe_name
from baselines.TDMLLM.utils.metrics import calculate_metrics
from baselines.TDMLLM.utils.prompts import (
    PREDICT_INSTRUCTION_SYSTEM_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT_W_FEW_SHOTS,
)
from baselines.TDMLLM.utils.fewshots import PREDICT_FEW_SHOT_EXAMPLES
from common.stock_direction import extract_stock_direction_and_value
from baselines.FinGPT.model import FinGPTAdapter, FinGPTConfig
from baselines.FinGPT.prompts.templates import DayContext, render_system_prompt, render_user_prompt

METHOD_NAME = "FinGPT"
STOCK_RETURN_PATTERN = re.compile(
    r"stock\s*return\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?)?\s*%?\s*\(\s*(?P<direction>up|down)\s*\)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FinGPT zero-shot inference baseline")
    parser.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "ACL18", "CMIN", "SEP"])
    parser.add_argument("--base_model", type=str, required=True, help="HF base model id, e.g. meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--fingpt_lora",
        type=str,
        default=None,
        help="FinGPT LoRA adapter path or HF repo id (omit to run base model only)",
    )
    parser.add_argument("--seq_len", type=int, default=5, help="Number of historical days to include in the prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # data / splits / thresholds
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override DATASETS_DIR")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Override OUTPUTS_DIR")
    parser.add_argument("--splits_dir", type=str, default=None, help="Override splits directory")
    parser.add_argument("--news_csv_dir", type=str, default=None, help="CMIN news CSV directory override")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_strategy", type=str, choices=["legacy", "dual_threshold"], default="legacy")
    parser.add_argument("--neg_threshold", type=float, default=-0.005)
    parser.add_argument("--pos_threshold", type=float, default=0.0055)

    # prompt/input controls
    parser.add_argument(
        "--max_texts_per_day",
        type=int,
        default=10,
        help="Limit of texts per day injected into the prompt (<=0 for unlimited)",
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        choices=["raw", "summary"],
        default="raw",
        help="Use raw tweets or cached daily summaries in prompts",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Override summary directory (defaults to <tweet_dir>/../summaries)",
    )
    parser.add_argument(
        "--summary_model_name",
        type=str,
        default=None,
        help="Summary cache model name (defaults to --base_model)",
    )
    parser.add_argument(
        "--summary_method",
        type=str,
        default="TDMLLM",
        help="Summary cache method subdirectory",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["simple", "tdmllm"],
        default="tdmllm",
        help="Prompt template style for FinGPT generation",
    )
    parser.add_argument(
        "--use_few_shots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include TDMLLM few-shot examples when using tdmllm prompt style",
    )
    parser.add_argument(
        "--company_desc_dir",
        type=str,
        default=None,
        help="Override TDMLLM company description cache directory",
    )
    parser.add_argument("--store_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_prompts", action=argparse.BooleanOptionalAction, default=True)
    # Removed generation_chunk_size: use batch_size to control VRAM and throughput.

    # model loading controls
    parser.add_argument("--device", type=str, default=None, help="Torch device when not using device_map")
    parser.add_argument("--device_map", type=str, default=None, help='HF device map (e.g. "auto")')
    parser.add_argument("--torch_dtype", type=str, default=None, help="Override model dtype (float16/bfloat16/...)")
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base model weights in 4-bit via bitsandbytes",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="float16",
        help="bitsandbytes compute dtype (float16/bfloat16)",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        choices=["nf4", "fp4"],
        default="nf4",
        help="bitsandbytes quantization type",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use double quantization when loading 4-bit weights",
    )

    return parser.parse_args()


def _resolve_summary_text(args: argparse.Namespace, ticker: str, date: str, logger: logging.Logger | None) -> str | None:
    summary_root = getattr(args, "summary_dir", None)
    if not summary_root:
        summary_root = Path(args.tweet_dir).parent / "summaries"
    summary_root = Path(summary_root)
    summary_model = getattr(args, "summary_model_name", None) or getattr(args, "base_model", None)
    if not summary_model:
        return None
    model_safe = summary_model.replace("/", "_")
    summary_method = getattr(args, "summary_method", "TDMLLM")
    summary_path = summary_root / model_safe / summary_method / ticker / f"{date}.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if logger:
            logger.warning(f"[Summary] failed to read {summary_path}: {exc}")
        return None
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def contexts_to_summary(contexts: List[DayContext]) -> str:
    lines: List[str] = []
    for ctx in contexts:
        lines.append(f"[{ctx.date}]")
        if ctx.texts:
            for text in ctx.texts:
                lines.append(f"- {text}")
        else:
            lines.append("- (no context available)")
        lines.append("")
    return "\n".join(lines).strip()


def load_company_descriptions(args: argparse.Namespace, tickers: List[str], logger: logging.Logger | None) -> Dict[str, str]:
    cache_root = Path(args.company_desc_dir or (REPO_ROOT / "baselines" / "TDMLLM" / "company_descriptions_cache"))
    model_safe = safe_name(args.base_model)
    base_dir = cache_root / args.dataset_name / model_safe
    descs: Dict[str, str] = {}
    for ticker in tickers:
        path = base_dir / f"{ticker}.txt"
        if path.exists():
            try:
                descs[ticker] = path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                if logger:
                    logger.warning(f"[CompanyDesc] failed to read {path}: {exc}")
                descs[ticker] = f"{ticker} company description unavailable."
        else:
            descs[ticker] = f"{ticker} company description unavailable."
            if logger:
                logger.warning(f"[CompanyDesc] missing cache for {ticker} at {path}")
    return descs


def build_tdmllm_user_prompt(company_description: str, summary_text: str, use_few_shots: bool) -> str:
    template = (
        PREDICT_INSTRUCTION_USER_PROMPT_W_FEW_SHOTS
        if use_few_shots
        else PREDICT_INSTRUCTION_USER_PROMPT
    )
    return template.format(
        company_description=company_description,
        summary=summary_text,
        few_shot_learning_examples=PREDICT_FEW_SHOT_EXAMPLES,
    )


def setup_logger(results_dir: str) -> logging.Logger:
    logger = logging.getLogger(METHOD_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(os.path.join(results_dir, "run.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    ticker: str
    prediction_date: str
    label: str
    contexts: List[DayContext]

    @property
    def sample_id(self) -> str:
        return f"{self.ticker}_{self.prediction_date}"


def _resolve_path(path: str | None, fallback: str) -> str:
    if path:
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(REPO_ROOT, path))
    return fallback


def build_samples(args: argparse.Namespace, logger: logging.Logger) -> List[Sample]:
    logger.info("Loading test split samples...")
    trade_days = list_trading_days(
        dataset_name=args.dataset_name,
        price_dir=args.price_dir,
        mode="test",
        seq_len=args.seq_len + 1,
        split_root=args.splits_dir,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
        logger=logger,
    )
    stats = getattr(list_trading_days, "last_stats", {})
    if stats:
        logger.info(f"Split stats: {json.dumps(stats, ensure_ascii=False)}")

    samples: List[Sample] = []
    skipped = 0

    for entry in trade_days:
        try:
            record = get_record(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=entry["date"],
                price_dir=args.price_dir,
                tweet_dir=args.tweet_dir,
                news_csv_dir=args.news_csv_dir,
                seq_len=args.seq_len + 1,
                label_strategy=args.label_strategy,
                neg_threshold=args.neg_threshold,
                pos_threshold=args.pos_threshold,
                logger=logger,
            )
        except Exception as exc:
            skipped += 1
            logger.warning(f"[DataLoader] skip {entry['ticker']} {entry['date']}: {exc}")
            continue

        context_dates = record.get("text_window_dates", [])[:-1]
        if not context_dates:
            skipped += 1
            logger.warning(f"[DataLoader] insufficient context for {entry['ticker']} {entry['date']}")
            continue

        context_payload: List[DayContext] = []
        context_mode = getattr(args, "context_mode", "raw")
        for ctx_date in context_dates:
            if context_mode == "summary":
                summary_text = _resolve_summary_text(args, entry["ticker"], ctx_date, logger)
                if summary_text:
                    context_payload.append(DayContext(date=ctx_date, texts=[summary_text]))
                    continue
                elif logger:
                    logger.debug(f"[Summary] Missing summary for {entry['ticker']} {ctx_date}, falling back to raw tweets")

            texts = load_texts_for_day(
                dataset_name=args.dataset_name,
                ticker=entry["ticker"],
                date=ctx_date,
                tweet_dir=args.tweet_dir,
                news_csv_dir=args.news_csv_dir,
                logger=logger,
            )
            cleaned = []
            for text_row in texts:
                raw_text = (text_row.get("text") or "").strip()
                if not raw_text:
                    continue
                cleaned.append(" ".join(raw_text.split()))
                if 0 < args.max_texts_per_day <= len(cleaned):
                    break
            context_payload.append(DayContext(date=ctx_date, texts=cleaned))

        samples.append(Sample(
            ticker=entry["ticker"],
            prediction_date=entry["date"],
            label=entry["label"],
            contexts=context_payload,
        ))

    logger.info(f"Loaded {len(samples)} samples (skipped {skipped}).")
    return samples



def parse_prediction(text: str) -> Tuple[str, float | None]:
    final_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(final_lines):
        match = STOCK_RETURN_PATTERN.search(line)
        if match:
            direction = match.group("direction").lower()
            label = "Positive" if direction == "up" else "Negative"
            value = match.group("value")
            return label, float(value) if value else None

    direction, value = extract_stock_direction_and_value(text)
    if direction not in {"Positive", "Negative"}:
        return "Unknown", value
    return direction, value


def write_predictions(predictions: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "predictions.jsonl")
    csv_path = os.path.join(out_dir, "predictions.csv")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for record in predictions:
            jf.write(json.dumps(record, ensure_ascii=False))
            jf.write("\n")

    csv_fields = [
        "sample_id",
        "ticker",
        "prediction_date",
        "y_true",
        "y_pred",
        "prediction_value",
        "model",
        "method",
        "dataset",
        "experiment_name",
        "latency_ms",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        for rec in predictions:
            writer.writerow({
                "sample_id": rec["sample_id"],
                "ticker": rec["ticker"],
                "prediction_date": rec["prediction_date"],
                "y_true": rec["ground_truth"],
                "y_pred": rec["prediction"]["label"],
                "prediction_value": rec["prediction"].get("value"),
                "model": rec["model"],
                "method": rec["method"],
                "dataset": rec["dataset"],
                "experiment_name": rec["experiment_name"],
                "latency_ms": rec.get("timing", {}).get("latency_ms"),
            })


def save_eval(metrics: Dict, results_dir: str, args: argparse.Namespace, wall_time: float) -> None:
    payload = {
        "dataset": args.dataset_name,
        "method": METHOD_NAME,
        "model": args.base_model,
        "experiment_name": args.experiment_name,
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "matrix": metrics["confusion_matrix"],
        },
        "total": metrics["total"],
        "valid": metrics["valid"],
        "invalid": metrics["invalid"],
        "wall_time_sec": wall_time,
    }
    with open(os.path.join(results_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def snapshot_args(args: argparse.Namespace, results_dir: str) -> None:
    env = {
        "python": sys.version,
        "torch": torch.__version__,
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import peft
        env["peft"] = peft.__version__
    except Exception:
        pass

    payload = {
        "args": vars(args),
        "env": env,
    }
    with open(os.path.join(results_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    default_data_root = os.path.join(REPO_ROOT, "datasets")
    default_outputs_root = os.path.join(REPO_ROOT, "outputs")
    default_splits_root = os.path.join(REPO_ROOT, "splits")
    args.base_data_dir = _resolve_path(args.base_data_dir, default_data_root)
    args.outputs_dir = _resolve_path(args.outputs_dir, default_outputs_root)
    args.splits_dir = _resolve_path(args.splits_dir, default_splits_root)
    if args.news_csv_dir:
        args.news_csv_dir = _resolve_path(args.news_csv_dir, args.news_csv_dir)

    resolved_paths = resolve_dataset_paths(args.dataset_name, args.base_data_dir)
    args.price_dir = resolved_paths.price_dir
    args.tweet_dir = resolved_paths.tweet_dir
    if not args.news_csv_dir and args.dataset_name.upper() == "CMIN":
        args.news_csv_dir = args.tweet_dir

    os.makedirs(args.outputs_dir, exist_ok=True)
    results_dir, exp_name = prepare_results_dir(
        method_name=METHOD_NAME,
        dataset_name=args.dataset_name,
        base_model=args.base_model,
        outputs_root=args.outputs_dir,
        experiment_name=args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    args.results_dir = results_dir
    args.experiment_name = exp_name

    logger = setup_logger(results_dir)
    logger.info(f"Experiment={exp_name} Dataset={args.dataset_name} SeqLen={args.seq_len}")
    set_random_seed(args.seed)
    snapshot_args(args, results_dir)

    samples = build_samples(args, logger)
    if not samples:
        logger.error("No samples found for evaluation.")
        return

    if args.prompt_style == "tdmllm":
        system_template = PREDICT_INSTRUCTION_SYSTEM_PROMPT
        company_descs = load_company_descriptions(args, sorted({s.ticker for s in samples}), logger)
    else:
        system_template = render_system_prompt(args.seq_len)
        company_descs = {}

    adapter_cfg = FinGPTConfig(
        base_model=args.base_model,
        fingpt_lora=args.fingpt_lora,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    adapter = FinGPTAdapter(adapter_cfg, logger=logger)
    total_start = time.time()

    predictions: List[Dict] = []
    labels: List[str] = []
    preds: List[str] = []
    batch_size = max(1, args.batch_size)

    batch_iter = range(0, len(samples), batch_size)
    for start_idx in tqdm(batch_iter, desc="Inference", unit="batch"):
        batch = samples[start_idx: start_idx + batch_size]
        prompt_payloads: List[Tuple[str, str]] = []
        for sample in batch:
            if args.prompt_style == "tdmllm":
                summary_text = contexts_to_summary(sample.contexts)
                company_desc = company_descs.get(sample.ticker, f"{sample.ticker} company description unavailable.")
                user_prompt = build_tdmllm_user_prompt(company_desc, summary_text, args.use_few_shots)
            else:
                user_prompt = render_user_prompt(
                    ticker=sample.ticker,
                    prediction_date=sample.prediction_date,
                    seq_len=args.seq_len,
                    contexts=sample.contexts,
                )
            prompt_payloads.append((system_template, user_prompt))

        # Generate for the entire batch in one call. Adjust VRAM via --batch_size.
        generation_results = adapter.batch_generate(prompt_payloads)

        for sample, gen_result, (sys_prompt, user_prompt) in zip(batch, generation_results, prompt_payloads):
            raw_output = gen_result.text
            label, value = parse_prediction(raw_output)
            preds.append(label)
            labels.append(sample.label)
            record = {
                "sample_id": sample.sample_id,
                "dataset": args.dataset_name,
                "method": METHOD_NAME,
                "model": args.base_model,
                "experiment_name": args.experiment_name,
                "ticker": sample.ticker,
                "prediction_date": sample.prediction_date,
                "ground_truth": sample.label,
                "prediction": {
                    "label": label,
                    "confidence": None,
                    "value": value,
                },
                "raw_response": raw_output if args.store_raw else "",
                "prompts": {
                    "system": sys_prompt if args.store_prompts else "",
                    "user": user_prompt if args.store_prompts else "",
                },
                "timing": {
                    "latency_ms": gen_result.latency_ms,
                },
            }
            predictions.append(record)

    write_predictions(predictions, results_dir)
    total_time = time.time() - total_start
    metrics = calculate_metrics(preds, labels)
    save_eval(metrics, results_dir, args, total_time)
    logger.info(f"Finished evaluation in {total_time:.2f}s | accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

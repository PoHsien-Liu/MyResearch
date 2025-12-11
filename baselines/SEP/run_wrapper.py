"""Wrapper CLI to run SEP baseline with shared loaders and vLLM backend."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEP_ROOT = os.path.join(REPO_ROOT, "baselines", "SEP", "sep")
TDMLLM_ROOT = os.path.join(REPO_ROOT, "baselines", "TDMLLM")

if TDMLLM_ROOT not in sys.path:
    sys.path.append(TDMLLM_ROOT)
for path in (SEP_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.config.datasets import resolve_dataset_paths
from common.data.loader import DEFAULT_NEG_THRESHOLD, DEFAULT_POS_THRESHOLD
from common.io.results import (
    prepare_results_dir,
    prepare_summary_cache_dir,
    write_predictions_from_results,
)
from data_load.dataloader import DataLoader
from explain_module.agents import PredictAgent
from summarize_module.summarizer import Summarizer
from utils.llm import HFLLM, VLLMLLM, VLLMSamplingConfig
from utils.metrics import calculate_metrics


METHOD_NAME = "SEP"


def _resolve_path(env_value: str | None, fallback: str) -> str:
    if not env_value:
        return fallback
    return env_value if os.path.isabs(env_value) else os.path.abspath(env_value)


def setup_logger(log_path: str, *, to_stdout: bool = True) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(METHOD_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def _write_args(args, results_dir: str) -> None:
    path = os.path.join(results_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def _write_eval(metrics: Dict, args, results_dir: str, wall_time: float) -> None:
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
            "labels": ["DOWN", "UP"],
            "matrix": metrics["confusion_matrix"],
        },
        "total": metrics["total"],
        "valid": metrics["valid"],
        "invalid": metrics["invalid"],
        "unknown_predictions": metrics.get("unknown_predictions", 0),
        "wall_time_sec": wall_time,
    }
    with open(os.path.join(results_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SEP baseline with shared loader")
    parser.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "STOCKNET", "CMIN", "SEP"])
    parser.add_argument("--base_model", type=str, default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_strategy", type=str, default="legacy", choices=["legacy", "dual_threshold"])
    parser.add_argument("--neg_threshold", type=float, default=DEFAULT_NEG_THRESHOLD)
    parser.add_argument("--pos_threshold", type=float, default=DEFAULT_POS_THRESHOLD)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override DATASETS_DIR env")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Override OUTPUTS_DIR env")
    parser.add_argument("--splits_dir", type=str, default=None, help="Override splits directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional limit for samples during debugging")
    parser.add_argument("--summary_max_new_tokens", type=int, default=256)
    parser.add_argument("--summary_temperature", type=float, default=0.1)
    parser.add_argument("--summary_top_p", type=float, default=0.9)
    parser.add_argument("--predict_max_new_tokens", type=int, default=256)
    parser.add_argument("--predict_temperature", type=float, default=0.3)
    parser.add_argument("--predict_top_p", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--log_to_stdout", action="store_true", default=False)
    parser.add_argument("--quantization", type=str, default="awq", help="Quantization type for vLLM (e.g., awq, bitsandbytes, None)")
    parser.add_argument("--engine", type=str, choices=["vllm", "hf"], default="vllm", help="LLM backend. Default vllm; use hf for transformers fallback.")
    parser.add_argument("--max_tweets_per_day", type=int, default=50, help="Cap tweets per day to bound prompt length.")
    return parser


def _build_llm(args) -> VLLMLLM:
    if args.engine == "hf":
        return HFLLM(
            model=args.base_model,
            max_new_tokens=args.predict_max_new_tokens,
            temperature=args.predict_temperature,
            top_p=args.predict_top_p,
        )
    return VLLMLLM(
        model=args.base_model,
        quantization=args.quantization if args.quantization and args.quantization.lower() != "none" else None,
        max_model_len=args.max_model_len,
        sampling_config=VLLMSamplingConfig(
            temperature=args.predict_temperature,
            top_p=args.predict_top_p,
            max_new_tokens=args.predict_max_new_tokens,
        ),
    )


def _run_eval(args, results_dir: str, logger: logging.Logger) -> None:
    start_time = time.time()

    # Shared vLLM instance
    shared_llm = _build_llm(args)

    # Summarizer uses the same engine to avoid extra memory
    summary_cache_dir = prepare_summary_cache_dir(
        dataset_name=args.dataset_name,
        base_model=args.base_model,
        method_name=METHOD_NAME,
        outputs_root=args.outputs_dir,
    )
    summarizer = Summarizer(
        model_name=args.base_model,
        cache_dir=summary_cache_dir,
        llm=shared_llm,
        max_model_len=args.max_model_len,
        max_new_tokens=args.summary_max_new_tokens,
        temperature=args.summary_temperature,
        top_p=args.summary_top_p,
        quantization=args.quantization if args.quantization and args.quantization.lower() != "none" else None,
    )

    dataloader = DataLoader(args, summarizer=summarizer, logger=logger)
    test_df = dataloader.load(flag="test")
    if args.max_samples and args.max_samples > 0:
        test_df = test_df.head(args.max_samples)
    logger.info(f"Loaded {len(test_df)} samples for evaluation.")

    predictions: List[str] = []
    labels: List[str] = []
    result_records: List[Dict] = []
    prompt_token_budget = max(min(args.max_model_len - args.predict_max_new_tokens - 256, args.max_model_len // 2), 128)

    for _, row in test_df.iterrows():
        agent = PredictAgent(
            row["ticker"],
            row["summary"],
            row["target"],
            predict_llm=shared_llm,
            max_prompt_tokens=prompt_token_budget,
        )
        agent.run()
        raw_response = agent.scratchpad.split("Price Movement: ")[-1].strip()
        parsed_label = raw_response.split()[0] if raw_response else ""

        predictions.append(parsed_label)
        labels.append(row["target"])

        result_records.append(
            {
                "sample_id": row.get("sample_id"),
                "ticker": row.get("ticker"),
                "prediction_date": row.get("prediction_date"),
                "processing_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "ground_truth": row.get("target"),
                "input_data": {"summary": row.get("summary", "")},
                "prediction": {
                    "parsed_movement": parsed_label,
                    "confidence": None,
                    "raw_text": raw_response,
                },
                "model_info": {
                    "dataset": args.dataset_name,
                    "method": METHOD_NAME,
                    "model_name": args.base_model,
                },
            }
        )

    metrics = calculate_metrics(predictions, labels)
    wall_time = time.time() - start_time
    logger.info(f"Finished evaluation in {wall_time:.2f}s")

    # Write outputs
    write_predictions_from_results(
        result_records,
        results_dir,
        dataset_name=args.dataset_name,
        method_name=METHOD_NAME,
        base_model=args.base_model,
        experiment_name=args.experiment_name,
        store_raw=True,
        store_prompts=False,
    )
    _write_eval(metrics, args, results_dir, wall_time)
    _write_args(args, results_dir)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_root = _resolve_path(os.getenv("DATASETS_DIR"), os.path.join(repo_root, "datasets"))
    default_outputs_root = _resolve_path(os.getenv("OUTPUTS_DIR"), os.path.join(repo_root, "outputs"))
    args.base_data_dir = args.base_data_dir or default_data_root
    args.outputs_dir = args.outputs_dir or default_outputs_root
    args.splits_dir = args.splits_dir or os.path.join(repo_root, "splits")

    dataset_paths = resolve_dataset_paths(args.dataset_name, args.base_data_dir)
    args.price_dir = dataset_paths.price_dir
    args.tweet_dir = dataset_paths.tweet_dir
    args.news_csv_dir = args.tweet_dir if args.dataset_name.upper() in {"CMIN", "CMIN-US"} else None

    results_dir, resolved_exp = prepare_results_dir(
        method_name=METHOD_NAME,
        dataset_name=args.dataset_name,
        base_model=args.base_model,
        outputs_root=args.outputs_dir,
        experiment_name=args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    args.experiment_name = resolved_exp

    log_path = os.path.join(results_dir, "run.log")
    logger = setup_logger(log_path, to_stdout=args.log_to_stdout)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Dataset: {args.dataset_name}, Price dir: {args.price_dir}, Tweet dir: {args.tweet_dir}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Seq len: {args.seq_len}, label strategy: {args.label_strategy}")

    _run_eval(args, results_dir, logger)


if __name__ == "__main__":
    main()

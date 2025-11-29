"""Argument parser for ZeroShotLLMs baselines."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Zero-shot LLM baselines (AWQ vLLM + FinGPT LoRA)")
    parser.add_argument("--backend", type=str, default="awq_vllm", choices=["awq_vllm", "fingpt_lora"])
    parser.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "ACL18", "CMIN", "SEP"])
    parser.add_argument("--base_model", type=str, default=None, help="HF base model (backend-specific default will apply if omitted)")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA path (used when backend=fingpt_lora)")
    parser.add_argument("--seq_len", type=int, default=5, help="Number of historical trading days to include.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_news_per_day", type=int, default=12, help="Limit of news items per day (<=0 means unlimited).")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap on number of samples for quick runs.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncate_chars", type=int, default=-1, help="Truncate raw_response to this many chars (<=0 disable).")
    parser.add_argument("--allow_output_truncation", action=argparse.BooleanOptionalAction, default=False)

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

    args = parser.parse_args()
    if not args.allow_output_truncation:
        args.truncate_chars = -1
    return args


__all__ = ["parse_args"]

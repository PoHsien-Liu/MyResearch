"""STARE CLI entry point for preprocessing, indexing, and evaluation."""
from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict

from STARE.data_load.clean_tweets import run_clean
from STARE.data_load.cooccurrence import run_cooccurrence
from STARE.data_load.extract_mentions import run_extract_mentions
from STARE.data_load.volume_stats import run_volume_stats
from STARE.eval.explanation_eval_main import (
    bind_explanation_eval_args,
    run_explanation_eval_task,
)
from STARE.eval.runner import bind_eval_subparser, run_eval
from STARE.index.build_index import run_build_index
from STARE.index.embed_texts import run_embed
from STARE.models.STARE.pipeline import run_train
from STARE.train.sft_finetune import bind_sft_finetune_args, run_sft_finetune_task


LOGGER = logging.getLogger(__name__)


def _register_tasks() -> Dict[str, Callable[[argparse.Namespace], None]]:
    """Return the task registry mapping CLI task name to handler."""
    return {
        "clean": run_clean,
        "extract_mentions": run_extract_mentions,
        "cooccurrence": run_cooccurrence,
        "volume_stats": run_volume_stats,
        "embed": run_embed,
        "build_index": run_build_index,
        "build_index_pipeline": _task_not_implemented,
        "eval": run_eval,
        "explanation_eval": run_explanation_eval_task,
        "train": run_train,
        "sft_finetune": run_sft_finetune_task,
    }


def _task_not_implemented(args: argparse.Namespace) -> None:
    """Placeholder for tasks that will be implemented later."""
    raise NotImplementedError(
        f"Task '{args.task}' is not implemented yet."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STARE unified CLI for preprocessing, indexing, and evaluation",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "clean",
            "extract_mentions",
            "cooccurrence",
            "volume_stats",
            "embed",
            "build_index",
            "build_index_pipeline",
            "eval",
            "explanation_eval",
            "train",
            "sft_finetune",
        ],
        help="Task to execute",
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier (SAMPLE/STOCKNET/CMIN/SEP)")
    parser.add_argument("--base_model", default=None, help="Foundation model name (for eval tasks)")
    parser.add_argument("--factor_model", default=None, help="Override model for factor generation (default: base_model or config)")
    parser.add_argument("--query_model", default=None, help="Override model for query generation (default: base_model or config)")
    parser.add_argument("--factor_backend", default=None, help="Backend name for factor generation (default: llama)")
    parser.add_argument("--query_backend", default=None, help="Backend name for query generation (default: factor backend or llama)")
    parser.add_argument("--embed_model", default="FinLang/finance-embeddings-investopedia", help="Embedding model name for indexing tasks")
    parser.add_argument("--experiment_name", default=None, help="Optional experiment identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length for baselines requiring it")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size when applicable")
    parser.add_argument("--rebuild_index", action="store_true", help="Rebuild FAISS index even if it exists")
    parser.add_argument("--min_tokens", type=int, default=5, help="Min tokens for rule-based filtering")
    parser.add_argument("--enable_llm_filter", action="store_true", help="Enable optional LLM-based cleaning filter")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k docs for retrieval/eval tasks")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows for embedding/debug")
    parser.add_argument("--label_strategy", default="dual_threshold", choices=["legacy", "dual_threshold"], help="Labeling strategy for returns (default: dual_threshold)")
    parser.add_argument("--neg_threshold", type=float, default=-0.005, help="Negative threshold for dual_threshold labeling (default: -0.5%)")
    parser.add_argument("--pos_threshold", type=float, default=0.0055, help="Positive threshold for dual_threshold labeling (default: +0.55%)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio for base splits (default: 0.8)")
    parser.add_argument("--split_root", default=None, help="Optional override for split directory")
    parser.add_argument("--only_ticker", default=None, help="If set, only run samples for this ticker (e.g., AAPL)")
    # Training-specific controls
    parser.add_argument("--test_sample", action="store_true", help="If set, only run a single sample for quick validation")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample to run when --test_sample is set")
    parser.add_argument(
        "--run_until",
        default="price_context",
        help="Pipeline stage to stop at (supports: price_context, factors, queries, prediction)",
    )
    parser.add_argument("--force_regen_factors", action="store_true", help="Regenerate factors even if cache exists")
    parser.add_argument("--factor_max_tokens", type=int, default=800, help="Max new tokens for factor generation")
    parser.add_argument("--query_max_tokens", type=int, default=256, help="Max new tokens for query generation")
    parser.add_argument(
        "--prompt_variant",
        default="target_only",
        choices=["target_only", "with_related"],
        help="Prediction prompt variant (without or with related-firm news blocks)",
    )
    bind_eval_subparser(parser)
    bind_explanation_eval_args(parser)
    bind_sft_finetune_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    registry = _register_tasks()
    handler = registry.get(args.task)
    if handler is None:
        raise ValueError(f"Unknown task: {args.task}")
    handler(args)


if __name__ == "__main__":
    main()

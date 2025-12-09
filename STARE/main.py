"""STARE CLI entry point for preprocessing, indexing, and evaluation."""
from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict
import re
import time

from STARE.data_load.clean_tweets import run_clean
from STARE.data_load.extract_mentions import run_extract_mentions
from STARE.index.build_index import run_build_index
from STARE.index.embed_texts import run_embed
from STARE.models.STARE.test_pipeline import run_test
from STARE.models.STARE.sft_dataset_builder import prepare_sft_samples
from STARE.train.sft_finetune import bind_sft_finetune_args, run_sft_finetune_task
from STARE.utils.paths import get_pipeline_data_dir


LOGGER = logging.getLogger(__name__)


def _register_tasks() -> Dict[str, Callable[[argparse.Namespace], None]]:
    """Return the task registry mapping CLI task name to handler."""
    return {
        "build_index_pipeline": run_build_index_pipeline,
        "train_pipeline": run_train_pipeline,
        "test_pipeline": run_test_pipeline,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STARE unified CLI for preprocessing, indexing, and evaluation")
    parser.add_argument("--task", type=str, required=True,
        choices=["build_index_pipeline", "train_pipeline", "test_pipeline"],
        help="Task to execute",
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier (SAMPLE/STOCKNET/CMIN/SEP)")
    parser.add_argument("--base_model", default=None, help="Foundation model name for finetune/inference")
    parser.add_argument("--factor_model", default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", help="Model for factor generation (default independent of base_model)")
    parser.add_argument("--query_model", default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", help="Override model for query generation (default independent of base_model)")
    parser.add_argument("--embed_model", default="FinLang/finance-embeddings-investopedia", help="Embedding model name for indexing tasks")
    parser.add_argument("--experiment_name", default=None, help="Optional experiment identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length for baselines requiring it")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size when applicable")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of samples to generate/evaluate")
    parser.add_argument("--rebuild_index", action="store_true", help="Rebuild FAISS index even if it exists")
    parser.add_argument("--min_tokens", type=int, default=5, help="Min tokens for rule-based filtering")
    parser.add_argument("--enable_llm_filter", action="store_true", help="Enable optional LLM-based cleaning filter")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k docs for retrieval/eval tasks")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows for embedding/debug")
    parser.add_argument("--label_strategy", default="dual_threshold", choices=["legacy", "dual_threshold"], help="Labeling strategy for returns (default: dual_threshold)")
    parser.add_argument("--neg_threshold", type=float, default=-0.005, help="Negative threshold for dual_threshold labeling (default: -0.5pct)")
    parser.add_argument("--pos_threshold", type=float, default=0.0055, help="Positive threshold for dual_threshold labeling (default: +0.55pct)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio for base splits (default: 0.8)")
    parser.add_argument("--split_root", default=None, help="Optional override for split directory")
    parser.add_argument("--only_ticker", default=None, help="If set, only run samples for this ticker (e.g., AAPL)")
    parser.add_argument("--force_regen_factors", action="store_true", help="Regenerate factors even if cache exists")
    parser.add_argument("--factor_max_tokens", type=int, default=800, help="Max new tokens for factor generation")
    parser.add_argument("--query_max_tokens", type=int, default=1024, help="Max new tokens for query generation (default: backend config)")
    parser.add_argument("--prompt_variant", type=str, default="target_only", choices=["target_only", "with_related"], help="Prediction prompt variant (without or with related-firm news blocks)")
    bind_sft_finetune_args(parser)
    return parser


def run_build_index_pipeline(args: argparse.Namespace) -> None:
    """Run clean -> extract_mentions -> embed -> build_index in one call."""
    LOGGER.info("Starting build_index_pipeline for dataset=%s embed_model=%s", args.dataset_name, args.embed_model)
    run_clean(args)
    run_extract_mentions(args)
    run_embed(args)
    run_build_index(args)
    LOGGER.info("Completed build_index_pipeline")


def _model_slug(name: str | None) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-") or "default"


def run_train_pipeline(args: argparse.Namespace) -> None:
    """
    Training pipeline only:
      1) Generate SFT samples for train split (RAG→prompt→SFT)
      2) Run LoRA fine-tune
    """
    exp = args.experiment_name or str(int(time.time()))

    LOGGER.info("Generating SFT samples for train split (exp=%s)", exp)
    prepare_sft_samples(args, mode="train")

    LOGGER.info("Running SFT fine-tune (LoRA)")
    run_sft_finetune_task(args)

    model_slug = _model_slug(getattr(args, "base_model", None))
    adapter_dir = get_pipeline_data_dir() / "sft" / "checkpoints" / args.dataset_name.upper() / model_slug / exp / "checkpoints" / "last"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_dir}")

    LOGGER.info("train_pipeline completed (exp=%s). Run test_pipeline for evaluation.", exp)


def run_test_pipeline(args: argparse.Namespace) -> None:
    """
    Testing pipeline: run inference/eval on test split (base model or optional adapter).
    """
    run_test(args)


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

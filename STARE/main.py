"""STARE CLI entry point for preprocessing, indexing, and evaluation."""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict

from STARE.data_load.clean_tweets import run_clean
from STARE.data_load.extract_mentions import run_extract_mentions
from STARE.index.build_index import run_build_index
from STARE.index.embed_texts import run_embed
from STARE.models.STARE.test_pipeline import run_test
from STARE.models.STARE.sft_dataset_builder import prepare_sft_samples
from STARE.train.sft_finetune import bind_sft_finetune_args, run_sft_finetune_task
from STARE.train.train_stage2_with_explanations import run_stage2_with_explanations
from STARE.pseudo_explanation import run_generate_with_teacher, run_prepare_stage2_sft
from STARE.pseudo_explanation.select_subset import select_subset
from STARE.llm_backend.llm_config import get_backend_config
from STARE.utils.paths import (
    get_datasets_dir,
    get_pipeline_data_dir,
    stage1_samples_dir,
    stage1_model_dir,
    stage1_checkpoint_last,
    stage2_model_dir,
    stage2_samples_dir,
    teacher_dir,
)


LOGGER = logging.getLogger(__name__)


def _register_tasks() -> Dict[str, Callable[[argparse.Namespace], None]]:
    """Return the task registry mapping CLI task name to handler."""
    return {
        "build_index_pipeline": run_build_index_pipeline,
        "train_pipeline": run_train_pipeline,
        "test_pipeline": run_test_pipeline,
        "train_test_pipeline": run_train_test_pipeline,
        "pseudo_expl_pipeline": run_pseudo_expl_pipeline,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STARE unified CLI for preprocessing, indexing, and evaluation")
    parser.add_argument("--task", type=str, required=True,
        choices=[
            "build_index_pipeline",
            "train_pipeline",
            "test_pipeline",
            "train_test_pipeline",
            "pseudo_expl_pipeline",
        ],
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
    parser.add_argument("--neg_threshold", type=float, default=-0.005, help="DOWN threshold for dual_threshold labeling (default: -0.5pct)")
    parser.add_argument("--pos_threshold", type=float, default=0.0055, help="UP threshold for dual_threshold labeling (default: +0.55pct)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio for base splits (default: 0.8)")
    parser.add_argument("--split_root", default=None, help="Optional override for split directory")
    parser.add_argument("--only_ticker", default=None, help="If set, only run samples for this ticker (e.g., AAPL)")
    parser.add_argument("--force_regen_factors", action="store_true", help="Regenerate factors even if cache exists")
    parser.add_argument("--factor_max_tokens", type=int, default=800, help="Max new tokens for factor generation")
    parser.add_argument("--query_max_tokens", type=int, default=1024, help="Max new tokens for query generation (default: backend config)")
    parser.add_argument("--prompt_variant", type=str, default="target_only", choices=["target_only", "with_related"], help="Prediction prompt variant (without or with related-firm news blocks)")
    parser.add_argument("--adapter_path", default=None, help="Optional adapter checkpoint for inference (test_pipeline)")
    # Pseudo-explanation pipeline args (subset -> teacher -> split -> stage2 SFT -> stage2 train)
    parser.add_argument("--pseudo_input", default=None, help="Path to train split jsonl (default: datasets/{dataset}/sft_pairs_train.jsonl)")
    parser.add_argument("--pseudo_subset_ratio", type=float, default=0.5, help="Fraction of data to sample per label (0-1]")
    parser.add_argument("--pseudo_ratio_tag", default=None, help="Optional tag/name for subset ratio dir (default: subset_ratio value)")
    parser.add_argument("--pseudo_seed", type=int, default=42, help="Seed for pseudo pipeline sampling")
    parser.add_argument("--teacher_backend", default="llama_70B", help="Backend key for teacher generation (stare_llm_config)")
    parser.add_argument("--teacher_model", default=None, help="Optional teacher model override (else backend default)")
    parser.add_argument("--teacher_batch_size", type=int, default=4, help="Batch size for teacher generation")
    parser.add_argument("--teacher_max_tokens", type=int, default=512, help="Max new tokens for teacher generation")
    parser.add_argument("--teacher_temperature", type=float, default=0.3, help="Temperature for teacher generation")
    parser.add_argument("--teacher_top_p", type=float, default=0.9, help="Top-p for teacher generation")
    parser.add_argument("--teacher_limit", type=int, default=None, help="Optional cap on candidates for dry-run")
    parser.add_argument("--teacher_log_every", type=int, default=20, help="Log progress every N batches for teacher gen")
    parser.add_argument("--pseudo_val_frac", type=float, default=0.1, help="Validation fraction for pseudo split")
    parser.add_argument("--stage2_limit_train", type=int, default=None, help="Optional cap on Stage2 train SFT rows")
    parser.add_argument("--stage2_limit_valid", type=int, default=None, help="Optional cap on Stage2 valid SFT rows")
    parser.add_argument("--skip_stage2_train", action="store_true", help="Skip Stage2 training (only prepare data)")
    parser.add_argument("--stage1_lora_path", default=None, help="Path to Stage1 LoRA adapter (default: pipeline_data/models/{dataset}/{exp}/stage1_llama8b_lora)")
    parser.add_argument("--stage2_output_dir", default=None, help="Optional override for Stage2 adapter output dir")
    parser.add_argument("--stage2_max_seq_length", type=int, default=2048)
    parser.add_argument("--stage2_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--stage2_per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--stage2_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--stage2_learning_rate", type=float, default=5e-5)
    parser.add_argument("--stage2_num_train_epochs", type=int, default=1)
    parser.add_argument("--stage2_weight_decay", type=float, default=0.0)
    parser.add_argument("--stage2_warmup_steps", type=int, default=0)
    parser.add_argument("--stage2_lr_scheduler_type", default="linear")
    parser.add_argument("--stage2_logging_steps", type=int, default=50)
    parser.add_argument("--stage2_eval_steps", type=int, default=500)
    parser.add_argument("--stage2_save_steps", type=int, default=500)
    parser.add_argument("--stage2_save_total_limit", type=int, default=2)
    parser.add_argument("--stage2_bf16", action="store_true")
    parser.add_argument("--stage2_fp16", action="store_true")
    parser.add_argument("--stage2_gradient_checkpointing", action="store_true")
    parser.add_argument("--stage2_lambda_cls", type=float, default=1.0)
    parser.add_argument("--stage2_lambda_lm", type=float, default=1.0)
    parser.add_argument("--stage2_max_train_samples", type=int, default=None)
    parser.add_argument("--stage2_max_eval_samples", type=int, default=None)
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


def _default_stage1_lora_path(dataset: str, experiment: str) -> Path:
    return get_pipeline_data_dir() / "models" / dataset.upper() / experiment / "stage1_llama8b_lora"


def _default_pseudo_input(dataset: str) -> Path:
    return get_datasets_dir() / dataset.upper() / "sft_pairs_train.jsonl"


def run_pseudo_expl_pipeline(args: argparse.Namespace) -> None:
    """
    Full pseudo-explanation + Stage2 pipeline:
      1) Subset selection
      2) Teacher generation
      3) Train/valid split
      4) Stage2 SFT data prep
      5) (optional) Stage2 training
    """
    dataset = args.dataset_name.upper()
    experiment = args.experiment_name or str(int(time.time()))
    ratio_tag = str(args.pseudo_ratio_tag) if args.pseudo_ratio_tag is not None else str(args.pseudo_subset_ratio)

    # ----- 1) Subset selection (skip if exists)
    stage1_dir = stage1_samples_dir(dataset, experiment)
    default_input = stage1_dir / "sft_samples_sft_train.jsonl"
    pseudo_input = Path(args.pseudo_input).expanduser().resolve() if args.pseudo_input else default_input
    if not pseudo_input.exists():
        pseudo_input = _default_pseudo_input(dataset)
    pseudo_input = pseudo_input.expanduser().resolve()
    if not pseudo_input.exists():
        raise FileNotFoundError(f"Pseudo input not found: {pseudo_input}")

    subset_path = stage2_samples_dir(dataset, experiment, ratio_tag) / "subset.jsonl"
    if subset_path.exists():
        LOGGER.info("Subset already exists, reuse: %s", subset_path)
    else:
        LOGGER.info("Selecting subset from %s (ratio=%s)", pseudo_input, args.pseudo_subset_ratio)
        select_subset(
            input_path=pseudo_input,
            dataset=dataset,
            experiment=experiment,
            subset_ratio=args.pseudo_subset_ratio,
            seed=args.pseudo_seed,
            output_path=subset_path,
        )

    # ----- 2) Teacher generation (skip if outputs exist)
    backend_cfg = get_backend_config(args.teacher_backend) or {}
    teacher_model = args.teacher_model or backend_cfg.get("default_model")
    if not teacher_model:
        raise ValueError("Teacher model must be specified via --teacher_model or backend default_model")
    teacher_out_dir = teacher_dir(dataset, teacher_model)
    teacher_train_out = teacher_out_dir / "sft_samples_sft_train.jsonl"
    teacher_val_out = teacher_out_dir / "sft_samples_sft_val.jsonl"

    if teacher_train_out.exists() and teacher_val_out.exists():
        LOGGER.info("Teacher outputs already exist, reuse: %s , %s", teacher_train_out, teacher_val_out)
    else:
        if not stage1_dir.exists():
            raise FileNotFoundError(f"Stage1 SFT directory not found: {stage1_dir}")
        teacher_args = argparse.Namespace(
            input_dir=str(stage1_dir),
            dataset_name=dataset,
            model=teacher_model,
            backend=args.teacher_backend,
            batch_size=args.teacher_batch_size,
            max_tokens=args.teacher_max_tokens,
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            output_dir=str(teacher_out_dir),
            limit=args.teacher_limit,
            log_every=args.teacher_log_every,
            no_save_prompt=False,
        )
        LOGGER.info("Running teacher generation -> %s", teacher_out_dir)
        run_generate_with_teacher(teacher_args)

    # ----- 3/4) Prepare Stage2 SFT data (train/valid)
    stage2_base = stage2_samples_dir(dataset, experiment, ratio_tag)
    stage2_train_out = stage2_base / "stage2_train_sft.jsonl"
    stage2_valid_out = stage2_base / "stage2_valid_sft.jsonl"
    prep_args = argparse.Namespace(
        subset_path=str(subset_path),
        teacher_train=str(teacher_train_out),
        teacher_valid=str(teacher_val_out),
        dataset_name=dataset,
        experiment_name=experiment,
        subset_ratio=args.pseudo_subset_ratio,
        teacher_model=teacher_model,
        teacher_model_slug=_model_slug(teacher_model),
        limit_train=args.stage2_limit_train,
        limit_valid=args.stage2_limit_valid,
        output_train=str(stage2_train_out),
        output_valid=str(stage2_valid_out),
    )
    stage2_paths = run_prepare_stage2_sft(prep_args)

    # ----- 5) Optional Stage2 training
    if args.skip_stage2_train:
        LOGGER.info("Skipping Stage2 training (flag set).")
        return

    stage1_lora_path = Path(args.stage1_lora_path).expanduser().resolve() if args.stage1_lora_path else _default_stage1_lora_path(dataset, experiment)
    if not stage1_lora_path.exists():
        raise FileNotFoundError(f"Stage1 LoRA adapter not found at {stage1_lora_path}")

    stage2_out_dir = Path(args.stage2_output_dir).expanduser().resolve() if args.stage2_output_dir else stage2_model_dir(dataset, experiment)
    train_args = argparse.Namespace(
        base_model=args.base_model,
        train_file=str(stage2_paths["train"]),
        validation_file=str(stage2_paths["valid"]),
        output_dir=str(stage2_out_dir),
        stage1_lora_path=str(stage1_lora_path),
        max_seq_length=args.stage2_max_seq_length,
        per_device_train_batch_size=args.stage2_per_device_train_batch_size,
        per_device_eval_batch_size=args.stage2_per_device_eval_batch_size,
        gradient_accumulation_steps=args.stage2_gradient_accumulation_steps,
        learning_rate=args.stage2_learning_rate,
        num_train_epochs=args.stage2_num_train_epochs,
        weight_decay=args.stage2_weight_decay,
        warmup_steps=args.stage2_warmup_steps,
        lr_scheduler_type=args.stage2_lr_scheduler_type,
        logging_steps=args.stage2_logging_steps,
        eval_steps=args.stage2_eval_steps,
        save_steps=args.stage2_save_steps,
        save_total_limit=args.stage2_save_total_limit,
        seed=args.seed,
        bf16=args.stage2_bf16,
        fp16=args.stage2_fp16,
        gradient_checkpointing=args.stage2_gradient_checkpointing,
        lambda_cls=args.stage2_lambda_cls,
        lambda_lm=args.stage2_lambda_lm,
        max_train_samples=args.stage2_max_train_samples,
        max_eval_samples=args.stage2_max_eval_samples,
    )
    LOGGER.info("Running Stage2 training -> %s", stage2_out_dir)
    run_stage2_with_explanations(train_args)


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

    adapter_dir = stage1_checkpoint_last(args.dataset_name, exp)
    if not adapter_dir.exists():
        LOGGER.warning("Last checkpoint not found at %s; checking model dir", adapter_dir)
        fallback = stage1_model_dir(args.dataset_name, exp)
        if not fallback.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_dir} or {fallback}")

    LOGGER.info("train_pipeline completed (exp=%s). Run test_pipeline for evaluation.", exp)


def run_test_pipeline(args: argparse.Namespace) -> None:
    """
    Testing pipeline: run inference/eval on test split (base model or optional adapter).
    """
    run_test(args)


def run_train_test_pipeline(args: argparse.Namespace) -> None:
    """
    Convenience wrapper: train_pipeline then test_pipeline with the same experiment name.
    """
    run_train_pipeline(args)

    adapter_dir = stage1_checkpoint_last(args.dataset_name, args.experiment_name)
    if not adapter_dir.exists():
        LOGGER.warning("Last checkpoint not found at %s; checking model dir", adapter_dir)
        fallback = stage1_model_dir(args.dataset_name, args.experiment_name)
        if not fallback.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_dir} or {fallback}")
        args.adapter_path = str(fallback)
    else:
        args.adapter_path = str(adapter_dir)
    run_test_pipeline(args)


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

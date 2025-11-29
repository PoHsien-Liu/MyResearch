"""Orchestrator for ZeroShotLLMs (AWQ vLLM + FinGPT LoRA)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from args import parse_args
from data import build_samples
from outputs import save_eval, write_predictions
from utils import ensure_paths_on_sys, resolve_path, set_random_seed, setup_logger, snapshot_args


def main():
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parents[1]
    ensure_paths_on_sys(module_dir, repo_root)

    from common.config.datasets import resolve_dataset_paths
    from common.io.results import prepare_results_dir
    from baselines.TDMLLM.utils.metrics import calculate_metrics
    from awq_vllm import run_inference_awq
    from config import build_vllm_config

    AWQ_DEFAULT_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    FINGPT_DEFAULT_BASE = "meta-llama/Llama-2-7b-chat-hf"
    FINGPT_DEFAULT_LORA = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

    args = parse_args()
    start_time = time.time()

    # Resolve base paths
    default_data_root = repo_root / "datasets"
    default_outputs_root = repo_root / "outputs"
    default_splits_root = repo_root / "splits"
    args.base_data_dir = resolve_path(args.base_data_dir, str(default_data_root))
    args.outputs_dir = resolve_path(args.outputs_dir, str(default_outputs_root))
    args.splits_dir = resolve_path(args.splits_dir, str(default_splits_root))
    if args.news_csv_dir:
        args.news_csv_dir = resolve_path(args.news_csv_dir, args.news_csv_dir)

    resolved_paths = resolve_dataset_paths(args.dataset_name, args.base_data_dir)
    args.price_dir = resolved_paths.price_dir
    args.tweet_dir = resolved_paths.tweet_dir
    if not args.news_csv_dir and args.dataset_name.upper() == "CMIN":
        args.news_csv_dir = args.tweet_dir

    # Method naming per backend
    if args.backend == "awq_vllm":
        args.base_model = args.base_model or AWQ_DEFAULT_MODEL
        args.model_tag = f"awq_vllm__{args.base_model}"
    else:
        args.base_model = args.base_model or FINGPT_DEFAULT_BASE
        args.lora_path = args.lora_path or FINGPT_DEFAULT_LORA
        args.model_tag = f"fingpt_lora__{args.lora_path or args.base_model}"
    method_name = "ZeroShotLLMs"

    results_dir, exp_name = prepare_results_dir(
        method_name=method_name,
        dataset_name=args.dataset_name,
        base_model=args.model_tag,
        outputs_root=args.outputs_dir,
        experiment_name=args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    args.results_dir = results_dir
    args.experiment_name = exp_name
    args.method_name = method_name

    logger = setup_logger(results_dir=results_dir, name=method_name)
    logger.info(f"Experiment={exp_name} Backend={args.backend} Dataset={args.dataset_name} SeqLen={args.seq_len} (test split only)")
    set_random_seed(args.seed)
    snapshot_args(args, results_dir)

    samples = build_samples(args, logger)

    if args.backend == "awq_vllm":
        vllm_cfg = build_vllm_config(args, default_model=AWQ_DEFAULT_MODEL)
        predictions = run_inference_awq(args, samples, logger, vllm_cfg)
    else:
        from fingpt_lora import run_inference_fingpt  # noqa: E402

        predictions = run_inference_fingpt(args, samples, logger)

    write_predictions(predictions, results_dir)

    labels = [s.ground_truth for s in samples]
    preds = [p.get("prediction", {}).get("label", "Unknown") for p in predictions]
    metrics = calculate_metrics(preds, labels) if samples else calculate_metrics([], [])
    wall_time = time.time() - start_time
    save_eval(metrics, results_dir, args, wall_time, method_name)

    logger.info(f"Finished. Samples={len(samples)} wall_time_sec={wall_time:.1f}")


if __name__ == "__main__":
    main()

import logging
import os
import random
import sys
from datetime import datetime

import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.config.datasets import resolve_dataset_paths
from common.io.results import prepare_results_dir, prepare_summary_cache_dir
from tdmllm.tdmllm import TDMLLM

def setup_logger(to_terminal=False, results_dir=None):
    """
    Setup logger with optional results directory for log file
    
    Args:
        to_terminal: bool, whether to output to terminal
        results_dir: str, optional results directory to save log file
    """
    if results_dir:
        # 如果提供了結果目錄，將 log 檔案保存在結果目錄中
        log_filename = os.path.join(results_dir, "experiment.log")
    else:
        # 否則使用原本的 log 目錄
        log_filename = f"./log/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [logging.FileHandler(log_filename, encoding="utf-8")]

    if to_terminal:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--llm_adapter', type=str, choices=["default", "fingpt"], default="default", help="backend LLM implementation (default TDMLLM stack or FinGPT adapter)")
    parser.add_argument("--fingpt_lora", type=str, default=None, help="Optional FinGPT LoRA path when --llm_adapter=fingpt")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='')
    parser.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "ACL18", "CMIN", "SEP"], help="Name of the dataset for saving results (ACL18, CMIN, or SEP)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override DATASETS_DIR env; defaults to ./datasets")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Override OUTPUTS_DIR env; defaults to ./outputs")
    
    # QLoRA specific arguments
    parser.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=False, help="Use QLoRA for model loading")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Generation top-p")
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False, help="Enable sampling during generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (when do_sample=False)")
    parser.add_argument("--max_new_tokens_predict", type=int, default=512, help="Max tokens for prediction generation")
    parser.add_argument("--summary_max_new_tokens", type=int, default=160, help="Max tokens for summary generation")
    parser.add_argument("--summary_min_tweets", type=int, default=1, help="Minimum tweets required to trigger summary generation")
    parser.add_argument("--summary_max_tweets", type=int, default=50, help="Maximum tweets per day used for summary")
    parser.add_argument("--summary_batch_size", type=int, default=None, help="Batch size for summary generation (defaults to --batch_size)")
    parser.add_argument("--label_strategy", type=str, choices=["legacy", "dual_threshold"], default="legacy", help="Labeling mode for samples")
    parser.add_argument("--neg_threshold", type=float, default=-0.005, help="Negative threshold (as return) when using dual_threshold")
    parser.add_argument("--pos_threshold", type=float, default=0.0055, help="Positive threshold (as return) when using dual_threshold")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio for shared splits")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for split generation")
    parser.add_argument("--splits_dir", type=str, default=None, help="Override splits directory (defaults to repo_root/splits)")
    parser.add_argument("--news_csv_dir", type=str, default=None, help="CMIN news CSV directory")
    parser.add_argument("--mode", type=str, choices=["eval", "train"], default="eval", help="Choose whether to run evaluation or the optional SFT training pass.")
    parser.add_argument("--train_epochs", type=int, default=2, help="Number of epochs when running SFT.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for the SFT training pass.")
    parser.add_argument("--train_max_length", type=int, default=512, help="Maximum number of tokens for training sequences.")
    parser.add_argument("--train_lr", type=float, default=5e-5, help="Learning rate used during training.")
    parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps for training.")

    # Experiment naming
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment folder (if not provided, will use timestamp)")
    
    parser.add_argument("--store_raw", action=argparse.BooleanOptionalAction, default=True, help="Store raw LLM response in predictions.jsonl")
    parser.add_argument("--store_prompts", action=argparse.BooleanOptionalAction, default=False, help="Store system/user prompts in predictions outputs")
    parser.add_argument("--truncate_chars", type=int, default=-1, help="Max chars for raw_response; <=0 to disable")
    parser.add_argument("--allow_output_truncation", action=argparse.BooleanOptionalAction, default=False, help="Allow truncating raw_response to --truncate_chars")

    args = parser.parse_args()
    if not args.allow_output_truncation:
        args.truncate_chars = -1

    # 記錄實驗開始時間
    experiment_start_time = datetime.now()
    
    # Resolve base dirs (datasets / outputs)
    def _resolve_root(env_value, fallback):
        if not env_value:
            return fallback
        return env_value if os.path.isabs(env_value) else os.path.abspath(os.path.join(REPO_ROOT, env_value))

    default_data_root = _resolve_root(os.getenv("DATASETS_DIR"), os.path.join(REPO_ROOT, "datasets"))
    default_outputs_root = _resolve_root(os.getenv("OUTPUTS_DIR"), os.path.join(REPO_ROOT, "outputs"))

    base_data_dir = args.base_data_dir or default_data_root
    outputs_root = args.outputs_dir or default_outputs_root
    splits_root = args.splits_dir or os.path.join(REPO_ROOT, "splits")

    resolved_paths = resolve_dataset_paths(args.dataset_name, base_data_dir)
    args.base_data_dir = resolved_paths.base_data_dir
    args.price_dir = resolved_paths.price_dir
    args.tweet_dir = resolved_paths.tweet_dir
    args.outputs_dir = outputs_root
    if args.news_csv_dir:
        args.news_csv_dir = _resolve_root(args.news_csv_dir, args.news_csv_dir)
    elif args.dataset_name.upper() == "CMIN":
        args.news_csv_dir = resolved_paths.tweet_dir
    args.splits_dir = splits_root

    # 生成實驗名稱和結果目錄
    method_name = "TDMLLM"
    
    # 如果沒有提供實驗名稱，使用時間戳
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        experiment_name = args.experiment_name
    
    results_dir, resolved_experiment = prepare_results_dir(
        method_name=method_name,
        dataset_name=args.dataset_name,
        base_model=args.base_model,
        outputs_root=outputs_root,
        experiment_name=args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    
    # 將結果目錄和開始時間添加到 args 中，供 TDMLLM 使用
    args.results_dir = results_dir
    args.experiment_name = resolved_experiment
    args.experiment_start_time = experiment_start_time
    args.summary_cache_dir = prepare_summary_cache_dir(
        dataset_name=args.dataset_name,
        base_model=args.base_model,
        method_name=method_name,
        outputs_root=outputs_root,
    )
    os.makedirs(splits_root, exist_ok=True)
    args.splits_dir = splits_root

    # Setup logger with results directory
    logger = setup_logger(results_dir=results_dir)
    logger.info(f"🚀 Experiment started at: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Price dir: {args.price_dir}")
    logger.info(f"Tweet dir: {args.tweet_dir}")
    logger.info(f"Seq len: {args.seq_len}")
    logger.info(f"Label strategy: {args.label_strategy} (neg={args.neg_threshold}, pos={args.pos_threshold})")
    logger.info(f"Experiment name: {resolved_experiment}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Mode: {args.mode}")

    set_random_seed(args.seed)

    tdm_llm = TDMLLM(args, logger)
    if args.mode == "train":
        tdm_llm.train()
    else:
        tdm_llm.eval()
    
if __name__ == '__main__':
    main()

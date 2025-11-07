import logging
import sys
from datetime import datetime
import argparse
import torch
import numpy as np
import random
import os

from tdmllm.tdmllm import TDMLLM

# Dataset path mapping
DATASET_PATHS = {
    "SAMPLE":{
        "price": "sample_data/sample_price",
        "tweet": "sample_data/sample_tweet"
    },
    "ACL18": {
        "price": "ACL18/stocknet-dataset/price",
        "tweet": "ACL18/stocknet-dataset/tweet"
    },
    "CMIN": {
        "price": "CMIN/CMIN-Dataset/CMIN-US/price",
        "tweet": "CMIN/CMIN-Dataset/CMIN-US/news"
    },
    "SEP": {
        "price": "SEP/price",
        "tweet": "SEP/tweet"
    }
}

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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='')
    # Load data paths
    parser.add_argument("--dataset_name", type=str, default="SEP", choices=["SAMPLE", "ACL18", "CMIN", "SEP"], help="Name of the dataset for saving results (ACL18, CMIN, or SEP)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=5)
    
    # QLoRA specific arguments
    parser.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=False, help="Use QLoRA for model loading")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Generation top-p")
    
    # Experiment naming
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment folder (if not provided, will use timestamp)")
    
    args = parser.parse_args()

    # 記錄實驗開始時間
    experiment_start_time = datetime.now()
    
    # Set data paths based on dataset name
    base_path = "/home/pohsien/Research/datasets"
    dataset_paths = DATASET_PATHS[args.dataset_name]
    args.price_dir = f"{base_path}/{dataset_paths['price']}/preprocessed/"
    args.tweet_dir = f"{base_path}/{dataset_paths['tweet']}/raw/"

    # 生成實驗名稱和結果目錄
    method_name = "TDMLLM"
    safe_model_name = args.base_model.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # 如果沒有提供實驗名稱，使用時間戳
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        experiment_name = args.experiment_name
    
    results_dir = os.path.join("results", args.dataset_name, method_name, safe_model_name, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 將結果目錄和開始時間添加到 args 中，供 TDMLLM 使用
    args.results_dir = results_dir
    args.experiment_name = experiment_name
    args.experiment_start_time = experiment_start_time

    # Setup logger with results directory
    logger = setup_logger(results_dir=results_dir)
    logger.info(f"🚀 Experiment started at: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Price dir: {args.price_dir}")
    logger.info(f"Tweet dir: {args.tweet_dir}")
    logger.info(f"Seq len: {args.seq_len}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Results directory: {results_dir}")

    set_random_seed(args.seed)

    tdm_llm = TDMLLM(args, logger)
    tdm_llm.eval()
    
if __name__ == '__main__':
    main()
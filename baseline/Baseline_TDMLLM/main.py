import logging
import sys
from datetime import datetime
import argparse
import torch
import numpy as np
import random

from tdmllm.tdmllm import TDMLLM

# Dataset path mapping
DATASET_PATHS = {
    "ACL18": {
        "price": "ACL18/stocknet-dataset/price",
        "tweet": "ACL18/stocknet-dataset/tweet"
    },
    "CMIN": {
        "price": "CMIN/CMIN-Dataset/CMIN-US/price",
        "tweet": "CMIN/CMIN-Dataset/CMIN-US/news"
    },
    "SEP": {
        "price": "SEP/sn2/price",
        "tweet": "SEP/sn2/tweet"
    }
}

def setup_logger(to_terminal=False):
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
    parser.add_argument("--dataset_name", type=str, default="ACL18", choices=["ACL18", "CMIN", "SEP"], help="Name of the dataset for saving results (ACL18, CMIN, or SEP)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=5)
    args = parser.parse_args()

    # Set data paths based on dataset name
    base_path = "/home/pohsien0915/Research/datasets"
    dataset_paths = DATASET_PATHS[args.dataset_name]
    args.price_dir = f"{base_path}/{dataset_paths['price']}/preprocessed/"
    args.tweet_dir = f"{base_path}/{dataset_paths['tweet']}/raw/"

    # Setup logger
    logger = setup_logger()
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Price dir: {args.price_dir}")
    logger.info(f"Tweet dir: {args.tweet_dir}")
    logger.info(f"Seq len: {args.seq_len}")

    set_random_seed(args.seed)

    tdm_llm = TDMLLM(args, logger)
    tdm_llm.eval()
    
if __name__ == '__main__':
    main()
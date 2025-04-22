import logging
import sys
from datetime import datetime
import argparse
import torch
import numpy as np
import random

from tdmllm.tdmllm import TDMLLM

def setup_logger():
    log_filename = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
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
    parser.add_argument("--price_dir", type=str, default="../SOTA_SEP/sep/data/sample_price/preprocessed/")
    parser.add_argument("--tweet_dir", type=str, default="../SOTA_SEP/sep/data/sample_tweet/raw/")
    parser.add_argument("--seq_len", type=int, default=5)
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Price dir: {args.price_dir}")
    logger.info(f"Tweet dir: {args.tweet_dir}")
    logger.info(f"Seq len: {args.seq_len}")

    set_random_seed(args.seed)

    tdm_llm = TDMLLM(args, logger)
    tdm_llm.inference()
    
if __name__ == '__main__':
    main()
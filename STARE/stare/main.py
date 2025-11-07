# main.py
import os
import argparse
from datetime import datetime
from utils.logger import setup_logger
from utils.seed import set_random_seed
from configs.dataset import DATASET_PATHS
from index.build_index import ensure_index_built
# from models.STARE.pipeline import STAREPipeline

def safe_name(x: str) -> str:
    return x.replace('/', '_').replace('\\', '_').replace(':', '_')

def resolve_paths(args):
    base = args.base_data_dir
    ds = DATASET_PATHS[args.dataset_name]
    args.price_dir = args.price_dir or os.path.join(base, ds["price"], "preprocessed")
    args.tweet_dir = args.tweet_dir or os.path.join(base, ds["tweet"], "raw")
    if not os.path.isdir(args.price_dir):
        raise FileNotFoundError(f"price_dir not found: {args.price_dir}")
    if not os.path.isdir(args.tweet_dir):
        raise FileNotFoundError(f"tweet_dir not found: {args.tweet_dir}")

def prepare_results_dir(args):
    method = "STARE"
    safe_model = safe_name(args.base_model)
    exp = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("results", args.dataset_name, method, safe_model, exp)
    os.makedirs(out, exist_ok=True)
    args.experiment_name = exp
    args.results_dir = out
    return out

def snapshot_args(args, results_dir):
    import json
    with open(os.path.join(results_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

def build_argparser():
    p = argparse.ArgumentParser("STARE minimal runner")
    # 基本
    p.add_argument("--task", type=str, default="eval", choices=["build_index", "eval"])
    p.add_argument("--seed", type=int, default=42)
    # 資料
    p.add_argument("--dataset_name", type=str, default="SEP", choices=["SAMPLE","ACL18","CMIN","SEP"])
    p.add_argument("--base_data_dir", type=str, default="/home/pohsien/Research/datasets")
    p.add_argument("--price_dir", type=str, default=None)  
    p.add_argument("--tweet_dir", type=str, default=None)  
    # 索引 / 向量
    p.add_argument("--index_dir", type=str, default="vector_store")
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--rebuild_index", action=argparse.BooleanOptionalAction, default=False)
    # 模型與生成
    p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    # 序列/批次
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    # 命名
    p.add_argument("--experiment_name", type=str, default=None)
    return p

def main():
    args = build_argparser().parse_args()
    set_random_seed(args.seed)

    resolve_paths(args)
    results_dir = prepare_results_dir(args)
    logger = setup_logger(results_dir)
    logger.info(f"Task={args.task}  Dataset={args.dataset_name}")
    logger.info(f"Price={args.price_dir}")
    logger.info(f"Tweet={args.tweet_dir}")
    logger.info(f"Index={args.index_dir}  Rebuild={args.rebuild_index}")
    snapshot_args(args, results_dir)  

    if args.task == "build_index":
        ensure_index_built(
            tweet_dir=args.tweet_dir,
            out_dir=args.index_dir,
            embed_model=args.embed_model,
            rebuild=True,
            device=None,
            logger=logger,
        )
        logger.info("[DONE] build_index")
        return

    # eval 前確保索引存在
    ensure_index_built(
        tweet_dir=args.tweet_dir,
        out_dir=args.index_dir,
        embed_model=args.embed_model,
        rebuild=args.rebuild_index,
        device=None,
        logger=logger,
    )

    # pipe = STAREPipeline(args=args, logger=logger)
    # pipe.eval()  

if __name__ == "__main__":
    main()

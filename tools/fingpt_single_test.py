#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.FinGPT.model import FinGPTAdapter, FinGPTConfig
from baselines.FinGPT.prompts.templates import DayContext, render_system_prompt, render_user_prompt
from baselines.FinGPT.run_predict import (
    contexts_to_summary,
    load_company_descriptions,
    build_tdmllm_user_prompt,
)
from baselines.TDMLLM.utils.prompts import PREDICT_INSTRUCTION_SYSTEM_PROMPT
from common.config.datasets import resolve_dataset_paths
from common.data.loader import list_trading_days, get_record, load_texts_for_day


def parse_args():
    ap = argparse.ArgumentParser("Single-sample FinGPT generation smoke test")
    ap.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "STOCKNET", "CMIN", "SEP"])
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument(
        "--fingpt_lora",
        type=str,
        default=None,
        help="LoRA adapter id/path (omit to use base model directly)",
    )
    ap.add_argument("--seq_len", type=int, default=3)
    ap.add_argument("--max_texts_per_day", type=int, default=5)
    ap.add_argument(
        "--context_mode",
        type=str,
        choices=["raw", "summary"],
        default="raw",
        help="Use raw tweets or cached daily summaries",
    )
    ap.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Override summary directory (defaults to <tweet_dir>/../summaries)",
    )
    ap.add_argument(
        "--summary_model_name",
        type=str,
        default=None,
        help="Summary cache model name (defaults to --base_model)",
    )
    ap.add_argument(
        "--summary_method",
        type=str,
        default="TDMLLM",
        help="Summary cache method folder",
    )
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--date", type=str, default=None, help="Prediction date YYYY-MM-DD; default is first test sample")
    ap.add_argument("--base_data_dir", type=str, default=str(REPO_ROOT / "datasets"))
    ap.add_argument("--news_csv_dir", type=str, default=None)
    ap.add_argument("--label_strategy", type=str, choices=["legacy", "dual_threshold"], default="dual_threshold")
    ap.add_argument("--neg_threshold", type=float, default=-0.005)
    ap.add_argument("--pos_threshold", type=float, default=0.0055)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default=None)
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    ap.add_argument("--bnb_4bit_use_double_quant", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prompt_style", type=str, choices=["simple", "tdmllm"], default="tdmllm", help="Prompt template to use (match FinGPT CLI)")
    ap.add_argument("--use_few_shots", action=argparse.BooleanOptionalAction, default=True, help="Include TDMLLM few-shot examples when using tdmllm prompt style")
    ap.add_argument("--company_desc_dir", type=str, default=None, help="Override TDMLLM company description cache directory")
    ap.add_argument("--out_file", type=str, default=None, help="Optional path to save system/user prompts and raw response (JSON).")
    return ap.parse_args()


def _load_summary_text(args, tweet_dir: Path, ticker: str, date: str):
    summary_root = args.summary_dir
    if summary_root is None:
        summary_root = Path(tweet_dir).parent / "summaries"
    summary_root = Path(summary_root)
    summary_model = args.summary_model_name or args.base_model
    if not summary_model:
        return None
    model_safe = summary_model.replace("/", "_")
    summary_path = summary_root / model_safe / args.summary_method / ticker / f"{date}.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def main():
    args = parse_args()
    paths = resolve_dataset_paths(args.dataset_name, args.base_data_dir)
    tweet_dir = paths.tweet_dir
    price_dir = paths.price_dir
    news_csv_dir = args.news_csv_dir or (paths.tweet_dir if args.dataset_name.upper() == "CMIN" else None)

    # pick a sample
    if args.ticker and args.date:
        target = {"ticker": args.ticker, "date": args.date, "label": "Unknown"}
    else:
        test_split = list_trading_days(
            dataset_name=args.dataset_name,
            price_dir=price_dir,
            mode="test",
            seq_len=args.seq_len + 1,
            split_root=str(REPO_ROOT / "splits"),
            train_ratio=0.8,
            split_seed=42,
            label_strategy=args.label_strategy,
            neg_threshold=args.neg_threshold,
            pos_threshold=args.pos_threshold,
            logger=None,
        )
        if not test_split:
            print("No test samples.")
            return
        target = test_split[0]

    rec = get_record(
        dataset_name=args.dataset_name,
        ticker=target["ticker"],
        date=target["date"],
        price_dir=price_dir,
        tweet_dir=tweet_dir,
        news_csv_dir=news_csv_dir,
        seq_len=args.seq_len + 1,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
        logger=None,
    )

    context_dates = rec.get("text_window_dates", [])[:-1]
    contexts = []
    for d in context_dates:
        if args.context_mode == "summary":
            summary_text = _load_summary_text(args, tweet_dir, target["ticker"], d)
            if summary_text:
                contexts.append(DayContext(date=d, texts=[summary_text]))
                continue
        texts = load_texts_for_day(
            dataset_name=args.dataset_name,
            ticker=target["ticker"],
            date=d,
            tweet_dir=tweet_dir,
            news_csv_dir=news_csv_dir,
            logger=None,
        )
        cleaned = []
        for row in texts:
            t = (row.get("text") or "").strip()
            if not t:
                continue
            cleaned.append(" ".join(t.split()))
            if 0 < args.max_texts_per_day <= len(cleaned):
                break
        contexts.append(DayContext(date=d, texts=cleaned))

    if args.prompt_style == "tdmllm":
        system = PREDICT_INSTRUCTION_SYSTEM_PROMPT
        desc_map = load_company_descriptions(args, [target["ticker"]], logger=None)
        summary_text = contexts_to_summary(contexts)
        company_desc = desc_map.get(target["ticker"], f"{target['ticker']} company description unavailable.")
        user = build_tdmllm_user_prompt(company_desc, summary_text, args.use_few_shots)
    else:
        system = render_system_prompt(args.seq_len)
        user = render_user_prompt(
            ticker=target["ticker"],
            prediction_date=target["date"],
            seq_len=args.seq_len,
            contexts=contexts,
        )
    print(f"Sample: {target['ticker']} {target['date']} (label={target.get('label')})")
    print(f"System len={len(system)} | User len={len(user)}")

    cfg = FinGPTConfig(
        base_model=args.base_model,
        fingpt_lora=args.fingpt_lora,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    adapter = FinGPTAdapter(cfg)
    res = adapter.generate(system, user)
    print(f"Latency: {res.latency_ms:.1f} ms")
    print("==== RAW OUTPUT ====")
    print(res.text)

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": target["ticker"],
            "prediction_date": target["date"],
            "label": target.get("label"),
            "system_prompt": system,
            "user_prompt": user,
            "raw_response": res.text,
            "latency_ms": res.latency_ms,
            "config": {
                "base_model": args.base_model,
                "fingpt_lora": args.fingpt_lora,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "load_in_4bit": args.load_in_4bit,
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved detailed output to {out_path}")


if __name__ == "__main__":
    main()

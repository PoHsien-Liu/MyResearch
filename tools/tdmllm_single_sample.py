#!/usr/bin/env python
"""Run TDMLLM/FinGPT backend on a single saved sample for quick sanity checks."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import pandas as pd
from transformers import modeling_utils

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.TDMLLM.models.llm import FinGPTLLM, LLaMALLM
from baselines.TDMLLM.utils.prompts import (
    PREDICT_INSTRUCTION_SYSTEM_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT_W_FEW_SHOTS,
)
from baselines.TDMLLM.utils.fewshots import PREDICT_FEW_SHOT_EXAMPLES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Single-sample TDMLLM/FinGPT inference")
    p.add_argument("--csv", type=str, required=True, help="Path to TDMLLM predictions.csv (must contain system/user prompts).")
    p.add_argument("--sample_id", type=str, default=None, help="Sample id to evaluate (defaults to first correct prediction in CSV).")
    p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HF base model id.")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling parameter.")
    p.add_argument("--adapter", type=str, choices=["fingpt", "llama"], default="fingpt", help="TDMLLM adapter to test.")
    p.add_argument("--run_base", action=argparse.BooleanOptionalAction, default=True, help="Run base model (no LoRA) when using FinGPT adapter.")
    p.add_argument("--run_fingpt", action=argparse.BooleanOptionalAction, default=False, help="Run FinGPT adapter with a LoRA.")
    p.add_argument("--fingpt_lora", type=str, default=None, help="FinGPT LoRA path (required when --run_fingpt).")
    p.add_argument("--device_map", type=str, default="auto", help="HF device map for FinGPT adapter.")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", help="Torch dtype hint for FinGPT adapter.")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True, help="Whether to load FinGPT adapter in 4-bit.")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size used by the LLaMALLM text-generation pipeline.")
    p.add_argument("--num_beams", type=int, default=1, help="Beam size when LLaMALLM runs in deterministic mode.")
    p.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=True, help="Enable QLoRA path for LLaMALLM.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank for LLaMALLM (only used when --use_qlora).")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha for LLaMALLM (only used when --use_qlora).")
    p.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout for LLaMALLM (only used when --use_qlora).")
    p.add_argument("--use_few_shots", action=argparse.BooleanOptionalAction, default=False, help="Include TDMLLM few-shot examples when synthesizing prompts.")
    p.add_argument("--out_file", type=str, default=None, help="Optional path to save prompts and model outputs (JSON).")
    return p.parse_args()


def select_sample(df: pd.DataFrame, sample_id: str | None) -> pd.Series:
    if sample_id:
        row = df.loc[df["sample_id"] == sample_id]
        if row.empty:
            raise ValueError(f"sample_id {sample_id} not found in CSV")
        return row.iloc[0]

    mask = df["ground_truth"].fillna("") == df["prediction"].fillna("")
    subset = df[mask]
    if subset.empty:
        raise ValueError("No correct predictions found; specify --sample_id manually.")
    return subset.iloc[0]


def build_logger() -> logging.Logger:
    logger = logging.getLogger("single-test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_parallel_constant() -> None:
    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
        modeling_utils.ALL_PARALLEL_STYLES = {}
    if not modeling_utils.ALL_PARALLEL_STYLES:
        placeholder = lambda *args, **kwargs: None
        modeling_utils.ALL_PARALLEL_STYLES.update({
            "colwise": placeholder,
            "rowwise": placeholder,
            "local_colwise": placeholder,
            "local_rowwise": placeholder,
            "local_packed_rowwise": placeholder,
        })


def run_fingpt_backend(
    args: argparse.Namespace,
    logger: logging.Logger,
    system: str,
    user: str,
    *,
    fingpt_lora: str | None,
) -> str:
    ensure_parallel_constant()
    llm_args = SimpleNamespace(
        base_model=args.base_model,
        fingpt_lora=fingpt_lora,
        max_new_tokens_predict=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        device=None,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    adapter = FinGPTLLM(llm_args, logger)
    return adapter(system, user)


def run_llama_backend(
    args: argparse.Namespace,
    logger: logging.Logger,
    system: str,
    user: str,
) -> str:
    llm_args = SimpleNamespace(
        base_model=args.base_model,
        max_new_tokens_predict=args.max_new_tokens,
        do_sample=args.temperature > 0,
        num_beams=args.num_beams,
        batch_size=max(1, args.batch_size),
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    adapter = LLaMALLM(llm_args, logger)
    return adapter(system, user, max_new_tokens=args.max_new_tokens)


def main() -> None:
    args = parse_args()
    logger = build_logger()
    df = pd.read_csv(args.csv)
    row = select_sample(df, args.sample_id)

    system_prompt = row.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        system_prompt = PREDICT_INSTRUCTION_SYSTEM_PROMPT

    user_prompt = row.get("user_prompt")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        summary_text = row.get("summary")
        if not isinstance(summary_text, str):
            summary_text = ""
        company_desc = row.get("company_description")
        if not isinstance(company_desc, str) or not company_desc.strip():
            company_desc = f"{row['ticker']} company description unavailable."
        template = (
            PREDICT_INSTRUCTION_USER_PROMPT_W_FEW_SHOTS
            if args.use_few_shots
            else PREDICT_INSTRUCTION_USER_PROMPT
        )
        user_prompt = template.format(
            company_description=company_desc,
            summary=summary_text,
            few_shot_learning_examples=PREDICT_FEW_SHOT_EXAMPLES,
        )

    print(f"Selected sample: {row['sample_id']} (ticker={row['ticker']}, label={row['ground_truth']})")

    base_output = None
    fingpt_output = None

    if args.adapter == "llama":
        if args.run_base:
            print("\n=== LLaMALLM (base model) ===")
            base_output = run_llama_backend(args, logger, system_prompt, user_prompt)
            print(base_output)
        else:
            print("Skipping LLaMALLM run (--no-run_base).")
    else:
        if args.run_base:
            print("\n=== FinGPT base model (no LoRA) ===")
            base_output = run_fingpt_backend(args, logger, system_prompt, user_prompt, fingpt_lora=None)
            print(base_output)
        if args.run_fingpt:
            if not args.fingpt_lora:
                raise ValueError("--fingpt_lora is required when --run_fingpt is enabled")
            print("\n=== FinGPT Adapter (LoRA) ===")
            fingpt_output = run_fingpt_backend(
                args,
                logger,
                system_prompt,
                user_prompt,
                fingpt_lora=args.fingpt_lora,
            )
            print(fingpt_output)

    if args.out_file:
        payload = {
            "sample_id": row.get("sample_id"),
            "ticker": row.get("ticker"),
            "prediction_date": row.get("prediction_date") or row.get("end_date"),
            "ground_truth": row.get("ground_truth") or row.get("target"),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "adapter": args.adapter,
            "base_model": args.base_model,
            "base_output": base_output,
            "fingpt_output": fingpt_output,
            "fingpt_lora": args.fingpt_lora,
        }
        out_path = os.path.abspath(args.out_file)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved outputs to {out_path}")


if __name__ == "__main__":
    main()

"""CLI entrypoint for explanation evaluation using LLM judge."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from STARE.eval.filters import filter_by_correct, filter_by_stock_scope
from STARE.eval.runner import evaluate_batch
from STARE.eval.judge_backends import BackendName
from STARE.eval.prompt_template import METRIC_KEYS


LOGGER = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    predictions_csv: Path
    dataset_name: str
    stock_scope: str
    only_correct: bool
    eval_llm_backend: BackendName
    eval_llm_model: str
    output_dir: Path
    experiment_name: Optional[str] = None
    max_samples: Optional[int] = None  # for dry-run / sanity


def bind_explanation_eval_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI args for explanation eval (shared by CLI + main)."""
    parser.add_argument("--predictions_csv", default=None, help="Path to predictions.csv for explanation eval")
    parser.add_argument(
        "--stock_scope",
        default="top1",
        choices=["all", "top1"],
        help="Stock scope filtering when running explanation eval",
    )
    parser.add_argument(
        "--only_correct",
        default="true",
        choices=["true", "false"],
        help="If true, only evaluate samples where y_true == y_pred",
    )
    parser.add_argument(
        "--eval_llm_backend",
        default=None,
        choices=["qwen", "llama", "openai", "gemini"],
        help="Backend name for judge LLM",
    )
    parser.add_argument("--eval_llm_model", default=None, help="Judge LLM model identifier")
    parser.add_argument("--explanation_eval_output_dir", default=None, help="Output dir for explanation eval results")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples (for dry run / quick sanity checks)",
    )
    parser.add_argument(
        "--explanation_eval_experiment_name",
        default=None,
        help="Optional experiment name for explanation eval outputs",
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explanation evaluation with LLM judge")
    parser.add_argument("--predictions_csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (SEP/StockNet/CMIN-US/SAMPLE)")
    parser.add_argument("--stock_scope", default="top1", choices=["all", "top1"], help="Stock scope filtering")
    parser.add_argument("--only_correct", default="true", choices=["true", "false"], help="Keep only correct preds")
    parser.add_argument("--eval_llm_backend", required=True, choices=["qwen", "llama", "openai", "gemini"], help="Judge backend")
    parser.add_argument("--eval_llm_model", required=True, help="Judge model name")
    parser.add_argument("--output_dir", required=True, help="Output directory for eval artifacts")
    parser.add_argument("--experiment_name", default=None, help="Optional experiment name")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap on number of samples (for dry run)")
    return parser.parse_args(argv)


def load_and_filter(cfg: EvalConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.predictions_csv)
    rename_map = {
        "ground_truth": "y_true",
        "prediction": "y_pred",
        "prediction_date": "date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if cfg.only_correct:
        df = filter_by_correct(df, only_correct=True)
    df = filter_by_stock_scope(df, dataset=cfg.dataset_name, stock_scope=cfg.stock_scope)
    if cfg.max_samples:
        df = df.head(cfg.max_samples)
    return df


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["context_texts"] = rec.get("context_texts") or rec.get("context") or rec.get("summary") or ""
        rec["explanation"] = rec.get("explanation") or rec.get("raw_response") or ""
        rec["sample_id"] = rec.get("sample_id") or f"{rec.get('ticker','')}_{rec.get('date','')}"
        records.append(rec)
    return records


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": len(results), "metrics_avg": {}, "metrics_std": {}}
    if not results:
        return summary
    scores: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for r in results:
        for k in METRIC_KEYS:
            val = r.get("metric_scores", {}).get(k)
            if isinstance(val, (int, float)):
                scores[k].append(float(val))
    import math
    for k, vals in scores.items():
        if not vals:
            summary["metrics_avg"][k] = None
            summary["metrics_std"][k] = None
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary["metrics_avg"][k] = mean
        summary["metrics_std"][k] = math.sqrt(var)
    return summary


def save_outputs(results: List[Dict[str, Any]], cfg: EvalConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = cfg.output_dir / "explanation_eval_samples.jsonl"
    with samples_path.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary_payload = summarize_results(results)
    summary_payload["args"] = asdict(cfg)
    summary_path = cfg.output_dir / "explanation_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved samples to %s", samples_path)
    LOGGER.info("Saved summary to %s", summary_path)


def run_explanation_eval(cfg: EvalConfig) -> List[Dict[str, Any]]:
    LOGGER.info("Loading predictions from %s", cfg.predictions_csv)
    df = load_and_filter(cfg)
    LOGGER.info("Filtered samples: %d", len(df))
    records = to_records(df)

    LOGGER.info("Starting LLM judging backend=%s model=%s", cfg.eval_llm_backend, cfg.eval_llm_model)
    results = evaluate_batch(
        backend=cfg.eval_llm_backend,
        model_name=cfg.eval_llm_model,
        records=records,
    )
    save_outputs(results, cfg)
    return results


def run_explanation_eval_task(args: argparse.Namespace) -> None:
    if not args.predictions_csv:
        raise ValueError("--predictions_csv is required for explanation_eval task")
    if not args.dataset_name:
        raise ValueError("--dataset_name is required for explanation_eval task")
    if not args.eval_llm_backend or not args.eval_llm_model:
        raise ValueError("--eval_llm_backend and --eval_llm_model are required for explanation_eval task")
    output_dir = args.explanation_eval_output_dir or args.output_dir
    if not output_dir:
        raise ValueError("--explanation_eval_output_dir or --output_dir must be specified")
    cfg = EvalConfig(
        predictions_csv=Path(args.predictions_csv),
        dataset_name=args.dataset_name,
        stock_scope=args.stock_scope,
        only_correct=str(args.only_correct).lower() == "true",
        eval_llm_backend=args.eval_llm_backend,  # type: ignore[assignment]
        eval_llm_model=args.eval_llm_model,
        output_dir=Path(output_dir),
        experiment_name=args.explanation_eval_experiment_name or args.experiment_name,
        max_samples=args.max_eval_samples,
    )
    run_explanation_eval(cfg)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    cfg = EvalConfig(
        predictions_csv=Path(args.predictions_csv),
        dataset_name=args.dataset_name,
        stock_scope=args.stock_scope,
        only_correct=args.only_correct.lower() == "true",
        eval_llm_backend=args.eval_llm_backend,  # type: ignore[assignment]
        eval_llm_model=args.eval_llm_model,
        output_dir=Path(args.output_dir),
        experiment_name=args.experiment_name,
        max_samples=args.max_eval_samples,
    )
    run_explanation_eval(cfg)


if __name__ == "__main__":
    main()

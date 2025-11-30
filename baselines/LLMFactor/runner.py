"""Runner for LLMFactor SKGP inference."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.LLMFactor.data_loader import Sample, build_samples  # noqa: E402
from baselines.LLMFactor.llm_client import LLMClient, GenerationConfig, DEFAULT_AWQ_MODEL  # noqa: E402
from baselines.LLMFactor.skgp import build_prompt_set, build_prediction_prompt  # noqa: E402
from baselines.TDMLLM.utils.metrics import calculate_metrics  # noqa: E402
from common.stock_direction import extract_stock_direction_and_value  # noqa: E402
from common.io.results import prepare_results_dir  # noqa: E402


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("llmfactor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LLMFactor baseline (SKGP inference)")
    parser.add_argument("--dataset_name", type=str, default="SAMPLE", choices=["SAMPLE", "ACL18", "CMIN", "SEP"])
    parser.add_argument("--seq_len", type=int, default=5, help="Number of historical trading days to include.")
    parser.add_argument("--max_news_per_day", type=int, default=12, help="Limit of news items per day (<=0 means all).")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap on number of samples (test split).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (reserved, currently processed sequentially).")
    parser.add_argument("--top_related", type=int, default=3, help="Top-N related tickers from co-occurrence map.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override DATASETS_DIR")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Override OUTPUTS_DIR (for co-occurrence neighbors)")
    parser.add_argument("--splits_dir", type=str, default=None, help="Override splits directory")
    parser.add_argument("--news_csv_dir", type=str, default=None, help="CMIN news CSV directory override")
    parser.add_argument("--embed_model", type=str, default=None, help="Embedding model name for co-occurrence lookup")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_strategy", type=str, choices=["legacy", "dual_threshold"], default="dual_threshold")
    parser.add_argument("--neg_threshold", type=float, default=-0.005)
    parser.add_argument("--pos_threshold", type=float, default=0.0055)
    parser.add_argument("--show_prompts", action=argparse.BooleanOptionalAction, default=False, help="Print SKGP prompts/outputs.")
    parser.add_argument("--top_factors", type=int, default=5, help="Top-k factors in Step2 prompt.")
    parser.add_argument("--backend", type=str, default="awq_vllm", choices=["awq_vllm"])
    parser.add_argument("--base_model", type=str, default=None, help="Override base model (default: llm_config.yaml)")
    parser.add_argument("--model_preset", type=str, default=None, help="Model preset name defined in llm_config.yaml under models")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--store_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncate_chars", type=int, default=-1, help="Truncate raw_response to this many chars (<=0 disable).")
    return parser.parse_args()


def _summarize(samples: List[Sample], logger: logging.Logger) -> None:
    if not samples:
        logger.warning("No samples built.")
        return
    preview = samples[: min(5, len(samples))]
    for idx, sample in enumerate(preview, 1):
        news_counts = ", ".join([f"{d.date}:{len(d.texts)}" for d in sample.news_by_day])
        logger.info(
            "[%d/%d] %s %s label=%s | days=%d | news=%s | related=%s",
            idx,
            len(preview),
            sample.ticker,
            sample.prediction_date,
            sample.ground_truth,
            len(sample.news_by_day),
            news_counts,
            ",".join(sample.related_candidates) if sample.related_candidates else "none",
        )


def _print_prompts(sample: Sample, top_factors: int, logger: logging.Logger) -> None:
    prompts = build_prompt_set(sample, top_k_factors=top_factors)
    logger.info("=== SAMPLE ===")
    logger.info("Ticker=%s Date=%s", sample.ticker, sample.prediction_date)
    logger.info("Ground truth=%s", sample.ground_truth)

    logger.info("=== Step 1 Relation Prompts (%d) ===", len(prompts["relation_prompts"]))
    for idx, p in enumerate(prompts["relation_prompts"], 1):
        logger.info("[Relation %d] SYSTEM:\n%s", idx, p.system)
        logger.info("[Relation %d] USER:\n%s", idx, p.user)

    logger.info("=== Step 2 Factor Prompt ===")
    logger.info("SYSTEM:\n%s", prompts["factor_prompt"].system)
    logger.info("USER:\n%s", prompts["factor_prompt"].user)

    logger.info("=== Step 3 Prediction Prompt (placeholders for Step1/2 outputs) ===")
    pred_prompt = prompts["prediction_prompt"]
    logger.info("SYSTEM:\n%s", pred_prompt.system)
    logger.info("USER:\n%s", pred_prompt.user)


def _parse_factors(raw: str, top_k: int) -> List[str]:
    factors: List[str] = []
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        text = text.lstrip("-*#").strip()
        text = text.lstrip("0123456789. ").strip()
        if text:
            factors.append(text)
        if len(factors) >= top_k:
            break
    if not factors and raw.strip():
        factors = [raw.strip()]
    return factors[:top_k]


def _parse_prediction_label(raw: str) -> Tuple[str, Optional[float]]:
    text = raw or ""
    direction, value = extract_stock_direction_and_value(text)

    lowered = text.lower()
    # Prefer future/explicit predictions, avoid past-tense "fell"
    patterns = [
        r"\bwill\s+(?:most\s+likely\s+)?(rise|go up|increase|fall|go down|decrease)\b",
        r"\bpredict(?:s|ed|ing)?\b.*?\b(rise|go up|increase|fall|go down|decrease)\b",
        r"\bexpect(?:s|ed|ing)?\b.*?\b(rise|go up|increase|fall|go down|decrease)\b",
    ]
    verb_label = {
        "rise": "Positive",
        "go up": "Positive",
        "increase": "Positive",
        "fall": "Negative",
        "go down": "Negative",
        "decrease": "Negative",
    }
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            verb = m.group(1)
            label = verb_label.get(verb, direction if direction in {"Positive", "Negative"} else "Unknown")
            if label != "Unknown":
                return label, value

    if direction in {"Positive", "Negative"}:
        return direction, value
    if " rise" in lowered or lowered.startswith("rise"):
        return "Positive", value
    if " fall" in lowered or lowered.startswith("fall"):
        return "Negative", value
    if " up" in lowered and "down" not in lowered:
        return "Positive", value
    if " down" in lowered and "up" not in lowered:
        return "Negative", value
    return "Unknown", value


def _run_sample_skgp(sample: Sample, client: LLMClient, args, logger) -> Dict:
    prompt_set = build_prompt_set(sample, top_k_factors=args.top_factors)
    time_block = prompt_set["time_block"]
    relation_prompts = prompt_set["relation_prompts"]
    factor_prompt = prompt_set["factor_prompt"]
    return {
        "sample": sample,
        "time_block": time_block,
        "relation_prompts": relation_prompts,
        "factor_prompt": factor_prompt,
    }


def _run_batch(samples: List[Sample], client: LLMClient, args, logger) -> List[Dict]:
    prompt_sets = [_run_sample_skgp(s, client, args, logger) for s in samples]

    # Step 1 batch
    rel_jobs: List[Tuple[int, int, PromptLike]] = []
    for idx, ps in enumerate(prompt_sets):
        for jdx, p in enumerate(ps["relation_prompts"]):
            rel_jobs.append((idx, jdx, p))
    rel_outputs: List[str] = []
    if rel_jobs:
        rel_outputs = client.generate([j[2] for j in rel_jobs])
    rel_text: Dict[int, List[str]] = {i: [] for i in range(len(samples))}
    for (sidx, jdx, _), out in zip(rel_jobs, rel_outputs):
        rel = samples[sidx].related_candidates[jdx] if jdx < len(samples[sidx].related_candidates) else ""
        line = (out or "").strip() or f"{samples[sidx].ticker} and {rel} are related."
        rel_text.setdefault(sidx, []).append(f"- {line}")

    # Step 2 batch
    factor_prompts = [ps["factor_prompt"] for ps in prompt_sets]
    factor_outputs = client.generate(factor_prompts) if factor_prompts else []
    factors_text: Dict[int, str] = {}
    parsed_factors: Dict[int, List[str]] = {}
    for idx, resp in enumerate(factor_outputs):
        parsed = _parse_factors(resp, args.top_factors)
        parsed_factors[idx] = parsed
        factors_text[idx] = "\n".join([f"{i}. {f}" for i, f in enumerate(parsed, 1)])

    # Step 3 batch
    pred_prompts: List[PromptLike] = []
    for idx, ps in enumerate(prompt_sets):
        pred_prompt = build_prediction_prompt(
            target_ticker=samples[idx].ticker,
            target_date=samples[idx].prediction_date,
            factors_text=factors_text.get(idx, "(factors not available)"),
            relations_text="\n".join(rel_text.get(idx, [])) if rel_text.get(idx) else "(relations not available)",
            time_block=ps["time_block"],
        )
        pred_prompts.append(pred_prompt)
        ps["pred_prompt"] = pred_prompt
    pred_outputs = client.generate(pred_prompts) if pred_prompts else []

    records: List[Dict] = []
    for idx, ps in enumerate(prompt_sets):
        pred_resp = pred_outputs[idx] if idx < len(pred_outputs) else ""
        label, value = _parse_prediction_label(pred_resp)
        rec = {
            "sample": samples[idx],
            "step1": {"prompts": ps["relation_prompts"], "outputs": rel_text.get(idx, [])},
            "step2": {"prompt": ps["factor_prompt"], "output": factor_outputs[idx] if idx < len(factor_outputs) else "", "parsed_factors": parsed_factors.get(idx, [])},
            "step3": {"prompt": ps["pred_prompt"], "output": pred_resp, "label": label, "confidence": value},
        }
        logger.info("Sample %s %s -> label=%s", samples[idx].ticker, samples[idx].prediction_date, label)
        records.append(rec)
    return records


def _print_run_output(record: Dict, logger: logging.Logger) -> None:
    sample: Sample = record["sample"]
    logger.info("=== SAMPLE ===")
    logger.info("Ticker=%s Date=%s GroundTruth=%s", sample.ticker, sample.prediction_date, sample.ground_truth)

    logger.info("=== Step 1 Relation Prompts/Outputs ===")
    for idx, (p, out) in enumerate(zip(record["step1"]["prompts"], record["step1"]["outputs"]), 1):
        logger.info("[Relation %d] USER:\n%s", idx, p.user)
        logger.info("[Relation %d] OUTPUT:\n%s", idx, out)

    logger.info("=== Step 2 Factor Prompt/Output ===")
    fp = record["step2"]["prompt"]
    logger.info("USER:\n%s", fp.user)
    logger.info("OUTPUT:\n%s", record["step2"]["output"])
    logger.info("PARSED FACTORS: %s", record["step2"]["parsed_factors"])

    logger.info("=== Step 3 Prediction Prompt/Output ===")
    pp = record["step3"]["prompt"]
    logger.info("USER:\n%s", pp.user)
    logger.info("OUTPUT:\n%s", record["step3"]["output"])
    logger.info("LABEL=%s CONFIDENCE=%s", record["step3"]["label"], record["step3"]["confidence"])


def _pred_record_from_sample(rec: Dict, args, method_name: str, experiment_name: str, base_model: str) -> Dict:
    sample: Sample = rec["sample"]
    label = rec["step3"]["label"]
    raw_resp = rec["step3"]["output"] or ""
    if args.truncate_chars and args.truncate_chars > 0:
        raw_resp = raw_resp[: args.truncate_chars]
    if not args.store_raw:
        raw_resp = ""
    prompts_payload = None
    if args.store_prompts:
        prompts_payload = {
            "system": rec["step3"]["prompt"].system,
            "user": rec["step3"]["prompt"].user,
        }
    return {
        "sample_id": sample.sample_id,
        "ticker": sample.ticker,
        "prediction_date": sample.prediction_date,
        "dataset": args.dataset_name,
        "method": method_name,
        "model": base_model,
        "experiment_name": experiment_name,
        "ground_truth": sample.ground_truth,
        "prediction": {"label": label, "confidence": rec["step3"]["confidence"]},
        "raw_response": raw_resp,
        "prompts": prompts_payload,
    }


def _write_predictions(predictions: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "predictions.jsonl"
    csv_path = out_dir / "predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    fields = [
        "sample_id",
        "ticker",
        "prediction_date",
        "ground_truth",
        "prediction",
        "model",
        "method",
        "dataset",
        "experiment_name",
        "raw_response",
        "system_prompt",
        "user_prompt",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for rec in predictions:
            prompts = rec.get("prompts") or {}
            row = [
                rec.get("sample_id", ""),
                rec.get("ticker", ""),
                rec.get("prediction_date", ""),
                rec.get("ground_truth", ""),
                rec.get("prediction", {}).get("label", ""),
                rec.get("model", ""),
                rec.get("method", ""),
                rec.get("dataset", ""),
                rec.get("experiment_name", ""),
                (rec.get("raw_response", "") or "").replace("\n", " ").replace(",", " "),
                (prompts.get("system", "") or "").replace("\n", " ").replace(",", " "),
                (prompts.get("user", "") or "").replace("\n", " ").replace(",", " "),
            ]
            f.write(",".join(row) + "\n")


def _write_eval(metrics: Dict, out_dir: Path, args, experiment_name: str, base_model: str) -> None:
    payload = {
        "dataset": args.dataset_name,
        "method": "LLMFactor",
        "model": base_model,
        "experiment_name": experiment_name,
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "matrix": metrics["confusion_matrix"],
        },
        "total": metrics["total"],
        "valid": metrics["valid"],
        "invalid": metrics["invalid"],
        "unknown_predictions": metrics.get("unknown_predictions", 0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_args(args, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    logger = _setup_logger()
    samples = build_samples(args, logger)
    if not samples:
        logger.warning("No samples to process; exiting.")
        return
    _summarize(samples, logger)

    client = LLMClient(
        backend=args.backend,
        base_model=args.base_model,
        gen_config=GenerationConfig(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        model_preset=getattr(args, "model_preset", None),
        logger=logger,
    )
    base_model_val = args.base_model or client.base_model or DEFAULT_AWQ_MODEL

    results_dir, exp_name = prepare_results_dir(
        method_name="LLMFactor",
        dataset_name=args.dataset_name,
        base_model=base_model_val,
        outputs_root=None,
        experiment_name=args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    _write_args(args, Path(results_dir))

    fh = logging.FileHandler(Path(results_dir) / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.info("Config source: %s", getattr(client, "config_source", "built-in"))
    start_time = time.time()
    records: List[Dict] = []
    bs = max(1, int(args.batch_size))
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        batch_records = _run_batch(batch, client, args, logger)
        records.extend(batch_records)
        if args.show_prompts:
            for rec in batch_records:
                _print_run_output(rec, logger)
    wall_time = time.time() - start_time

    predictions = [
        _pred_record_from_sample(rec, args, "LLMFactor", exp_name, base_model_val) for rec in records
    ]
    _write_predictions(predictions, Path(results_dir))
    labels = [p.get("ground_truth") for p in predictions]
    preds = [p.get("prediction", {}).get("label") for p in predictions]
    metrics = calculate_metrics(preds, labels) if predictions else calculate_metrics([], [])
    _write_eval(metrics, Path(results_dir), args, exp_name, base_model_val)
    with (Path(results_dir) / "run.log").open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Finished samples={len(predictions)} wall_time_sec={wall_time:.2f}\n")


if __name__ == "__main__":
    main()

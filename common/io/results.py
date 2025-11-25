"""Shared helpers for experiment directories and prediction outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, Dict, Any

import pandas as pd


def safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(':', '_')


def prepare_results_dir(
    *,
    method_name: str,
    dataset_name: str,
    base_model: str,
    outputs_root: str | None = None,
    experiment_name: str | None = None,
    label_strategy: str | None = None,
    neg_threshold: float | None = None,
    pos_threshold: float | None = None,
) -> tuple[str, str]:
    """Create and return the results directory + resolved experiment name."""

    outputs_base = outputs_root or os.getenv("OUTPUTS_DIR", "./outputs")
    exp_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = safe_name(base_model)
    variant_dir = _label_variant_dir(label_strategy, neg_threshold, pos_threshold)
    results_dir = os.path.join(
        outputs_base,
        "results",
        dataset_name,
        variant_dir,
        method_name,
        model_safe,
        exp_name,
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir, exp_name


def prepare_summary_cache_dir(
    *,
    dataset_name: str,
    base_model: str,
    method_name: str,
    outputs_root: str | None = None,
) -> str:
    outputs_base = outputs_root or os.getenv("OUTPUTS_DIR", "./outputs")
    model_safe = safe_name(base_model)
    dataset_safe = safe_name(dataset_name)
    cache_dir = os.path.join(outputs_base, "cache", "summaries", dataset_safe, model_safe, method_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def format_prediction_record(
    result: Dict[str, Any],
    *,
    dataset_name: str,
    method_name: str,
    base_model: str,
    experiment_name: str,
    store_raw: bool = True,
    store_prompts: bool = False,
    truncate_chars: int = -1,
) -> Dict[str, Any]:
    model_info = result.get("model_info", {})
    model_input = result.get("model_input", {})
    input_data = result.get("input_data", {})
    prediction = result.get("prediction", {})

    raw_response = prediction.get("raw_text") or ""
    if truncate_chars and truncate_chars > 0:
        raw_response = raw_response[:truncate_chars]
    if not store_raw:
        raw_response = ""

    prompts_payload = None
    if store_prompts:
        prompts_payload = {
            "system": model_input.get("system_prompt", ""),
            "user": model_input.get("predict_prompt", ""),
        }

    return {
        "sample_id": result.get("sample_id"),
        "ticker": result.get("ticker"),
        "prediction_date": result.get("prediction_date"),
        "processing_date": result.get("processing_date"),
        "dataset": model_info.get("dataset", dataset_name),
        "method": model_info.get("method", method_name),
        "model": model_info.get("model_name", base_model),
        "experiment_name": experiment_name,
        "ground_truth": result.get("ground_truth"),
        "prediction": {
            "label": prediction.get("parsed_movement"),
            "confidence": prediction.get("confidence"),
        },
        "raw_response": raw_response,
        "prompts": prompts_payload,
        "inputs": {
            "summary": input_data.get("summary", ""),
            "company_description": input_data.get("company_description", ""),
        },
    }


def write_predictions_from_results(
    test_results: Iterable[Dict[str, Any]],
    out_dir: str,
    *,
    dataset_name: str,
    method_name: str,
    base_model: str,
    experiment_name: str,
    store_raw: bool = True,
    store_prompts: bool = False,
    truncate_chars: int = -1,
) -> list[Dict[str, Any]]:
    records = [
        format_prediction_record(
            result,
            dataset_name=dataset_name,
            method_name=method_name,
            base_model=base_model,
            experiment_name=experiment_name,
            store_raw=store_raw,
            store_prompts=store_prompts,
            truncate_chars=truncate_chars,
        )
        for result in test_results
    ]
    write_predictions(records, out_dir)
    return records


def write_predictions(records: Iterable[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "predictions.jsonl")
    csv_path = os.path.join(out_dir, "predictions.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    rows = []
    for rec in records:
        prompts_payload = rec.get("prompts") or {}
        rows.append({
            "sample_id": rec.get("sample_id"),
            "ticker": rec.get("ticker"),
            "prediction_date": rec.get("prediction_date"),
            "processing_date": rec.get("processing_date"),
            "dataset": rec.get("dataset"),
            "method": rec.get("method"),
            "model": rec.get("model"),
            "experiment_name": rec.get("experiment_name"),
            "ground_truth": rec.get("ground_truth"),
            "prediction": rec.get("prediction", {}).get("label"),
            "raw_response": rec.get("raw_response", ""),
            "summary": rec.get("inputs", {}).get("summary", ""),
            "company_description": rec.get("inputs", {}).get("company_description", ""),
            "system_prompt": prompts_payload.get("system", ""),
            "user_prompt": prompts_payload.get("user", ""),
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")


def write_training_data(records: Iterable[Dict[str, Any]], out_dir: str, *, filename: str = "training_data.jsonl") -> str:
    """Persist training prompt/label pairs in JSONL form for reuse."""

    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, filename)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return jsonl_path


def _label_variant_dir(label_strategy: str | None, neg_threshold: float | None, pos_threshold: float | None) -> str:
    strategy = (label_strategy or "legacy").lower()
    if strategy == "dual_threshold":
        neg = _pct_tag(neg_threshold if neg_threshold is not None else -0.005)
        pos = _pct_tag(pos_threshold if pos_threshold is not None else 0.0055)
        return os.path.join("dual", f"neg{neg}_pos{pos}")
    return "legacy"


def _pct_tag(value: float) -> str:
    pct = (value or 0.0) * 100
    return (f"{pct:+.2f}".rstrip("0").rstrip(".")) + "pct"

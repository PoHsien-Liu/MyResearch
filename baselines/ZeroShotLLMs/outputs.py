"""Prediction/evaluation writers for ZeroShotLLMs baselines."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable


def write_predictions(predictions: Iterable[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "predictions.jsonl")
    csv_path = os.path.join(out_dir, "predictions.csv")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for rec in predictions:
            jf.write(json.dumps(rec, ensure_ascii=False))
            jf.write("\n")

    csv_fields = [
        "sample_id",
        "ticker",
        "prediction_date",
        "y_true",
        "y_pred",
        "prediction_value",
        "model",
        "method",
        "dataset",
        "experiment_name",
        "latency_ms",
        "raw_response",
        "system_prompt",
        "user_prompt",
    ]
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write(",".join(csv_fields) + "\n")
        for rec in predictions:
            prompts = rec.get("prompts") or {}
            row = [
                rec.get("sample_id", ""),
                rec.get("ticker", ""),
                rec.get("prediction_date", ""),
                rec.get("ground_truth", ""),
                rec.get("prediction", {}).get("label", ""),
                str(rec.get("prediction", {}).get("confidence") or ""),
                rec.get("model", ""),
                rec.get("method", ""),
                rec.get("dataset", ""),
                rec.get("experiment_name", ""),
                str(rec.get("timing", {}).get("latency_ms") or ""),
                (rec.get("raw_response", "") or "").replace("\n", " ").replace(",", " "),
                (prompts.get("system", "") or "").replace("\n", " ").replace(",", " "),
                (prompts.get("user", "") or "").replace("\n", " ").replace(",", " "),
            ]
            cf.write(",".join(row) + "\n")


def save_eval(metrics: Dict, results_dir: str, args, wall_time: float, method_name: str) -> None:
    payload = {
        "dataset": args.dataset_name,
        "method": method_name,
        "model": args.base_model,
        "experiment_name": args.experiment_name,
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": {
            "labels": ["DOWN", "UP"],
            "matrix": metrics["confusion_matrix"],
        },
        "total": metrics["total"],
        "valid": metrics["valid"],
        "invalid": metrics["invalid"],
        "unknown_predictions": metrics.get("unknown_predictions", 0),
        "wall_time_sec": wall_time,
    }
    with open(os.path.join(results_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = ["write_predictions", "save_eval"]

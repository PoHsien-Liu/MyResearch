"""Unified evaluation utilities for classification and explanations."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


_LABEL_CANONICAL = {"positive": "Positive", "negative": "Negative"}
_LABEL_ALIASES = {
    "pos": "positive",
    "neg": "negative",
    "1": "positive",
    "0": "negative",
    "+": "positive",
    "-": "negative",
}


def _normalize_label(raw: object | None) -> str | None:
    """Normalize a raw label to 'Positive' or 'Negative'; return None if unknown."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    text = _LABEL_ALIASES.get(text, text)
    return _LABEL_CANONICAL.get(text)


def _load_predictions(path: Path) -> List[MutableMapping[str, object]]:
    """Load predictions from CSV or JSONL into a list of dict records."""
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    ext = path.suffix.lower()
    records: List[MutableMapping[str, object]] = []
    if ext == ".csv":
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            records.extend(reader)
    elif ext == ".jsonl":
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported predictions file type: {path}")
    return records


def _compute_confusion(preds: List[int], labels: List[int]) -> Tuple[int, int, int, int]:
    """Return tn, fp, fn, tp counts."""
    tn = fp = fn = tp = 0
    for p, l in zip(preds, labels):
        if p == 1 and l == 1:
            tp += 1
        elif p == 1 and l == 0:
            fp += 1
        elif p == 0 and l == 1:
            fn += 1
        elif p == 0 and l == 0:
            tn += 1
    return tn, fp, fn, tp


def evaluate_classification(
    records: Iterable[Mapping[str, object]],
    unknown_policy: str = "as_error",
) -> Dict[str, object]:
    """Evaluate classification metrics with configurable Unknown handling.

    unknown_policy:
        - "as_error": treat unknown/invalid prediction as an error (flip true label).
        - "as_invalid": skip unknown/invalid prediction.
    """
    policy = unknown_policy.lower()
    if policy not in {"as_error", "as_invalid"}:
        raise ValueError(f"unknown_policy must be 'as_error' or 'as_invalid', got {unknown_policy}")

    stats = {
        "total": 0,
        "valid": 0,
        "invalid_ground_truth": 0,
        "invalid_prediction": 0,
        "unknown_predictions": 0,
    }
    label_map = {"Positive": 1, "Negative": 0}
    preds_mapped: List[int] = []
    labels_mapped: List[int] = []

    for rec in records:
        stats["total"] += 1
        truth = _normalize_label(rec.get("ground_truth"))
        if truth is None:
            stats["invalid_ground_truth"] += 1
            continue

        pred = _normalize_label(rec.get("prediction"))
        true_val = label_map[truth]
        if pred is None:
            stats["invalid_prediction"] += 1
            stats["unknown_predictions"] += 1
            if policy == "as_invalid":
                continue
            pred_val = 1 - true_val  # count as an error
        else:
            pred_val = label_map[pred]

        labels_mapped.append(true_val)
        preds_mapped.append(pred_val)

    stats["valid"] = len(labels_mapped)
    stats["invalid"] = stats["total"] - stats["valid"]
    stats["coverage"] = (stats["valid"] / stats["total"]) if stats["total"] else 0.0

    if not labels_mapped:
        metrics = {
            "accuracy": 0.0,
            "mcc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    else:
        tn, fp, fn, tp = _compute_confusion(preds_mapped, labels_mapped)
        total_valid = len(labels_mapped)
        accuracy = (tp + tn) / total_valid
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
        metrics = {
            "accuracy": accuracy,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }

    return {
        "classification_metrics": metrics,
        "sample_stats": stats,
        "label_policy": "unknown_as_error" if policy == "as_error" else "unknown_as_invalid",
    }


def evaluate_predictions_file(
    predictions_path: Path,
    unknown_policy: str = "as_error",
) -> Dict[str, object]:
    """Convenience wrapper to load predictions and evaluate."""
    records = _load_predictions(predictions_path)
    return evaluate_classification(records, unknown_policy=unknown_policy)

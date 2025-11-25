import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.TDMLLM.utils.metrics import calculate_metrics, save_metrics  # noqa: E402
from common.stock_direction import extract_stock_direction_and_value  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse predictions.csv with the shared stock direction parser.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the predictions.csv file that needs to be recalibrated.",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory to store recalibrated outputs. Defaults to <experiment_dir>/calibration.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Override dataset name written to eval.json. Defaults to the value found in the CSV.",
    )
    parser.add_argument(
        "--method-name",
        help="Override method name written to eval.json. Defaults to the value found in the CSV.",
    )
    parser.add_argument(
        "--model-name",
        help="Override model name written to eval.json. Defaults to the value found in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = os.path.abspath(args.predictions)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    experiment_dir = os.path.dirname(predictions_path)
    out_dir = args.out_dir or os.path.join(experiment_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(predictions_path)
    if "raw_response" not in df.columns:
        raise ValueError("predictions.csv must contain a 'raw_response' column.")
    if "ground_truth" not in df.columns:
        raise ValueError("predictions.csv must contain a 'ground_truth' column.")

    directions = []
    return_values = []
    for raw in df["raw_response"].fillna(""):
        direction, value = extract_stock_direction_and_value(raw)
        directions.append(direction)
        return_values.append(value)

    df["original_prediction"] = df.get("prediction")
    df["prediction"] = directions
    df["predicted_return_pct"] = return_values

    dataset_name = args.dataset_name or _resolve_field(df, "dataset")
    method_name = args.method_name or _resolve_field(df, "method")
    model_name = args.model_name or _resolve_field(df, "model")

    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    preds = df["prediction"].fillna("Unknown").tolist()
    labels = df["ground_truth"].fillna("Unknown").tolist()
    metrics = calculate_metrics(preds, labels)
    save_metrics(
        metrics,
        model_name or "unknown-model",
        out_dir,
        dataset_name=dataset_name or "unknown-dataset",
    )

    summary_payload = {
        "source_file": predictions_path,
        "output_dir": out_dir,
        "total_samples": len(df),
        "unknown_after": int((df["prediction"] == "Unknown").sum()),
        "unknown_before": int((df["original_prediction"] == "Unknown").sum())
        if "original_prediction" in df
        else None,
        "method_name": method_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
    }
    with open(os.path.join(out_dir, "reparse_summary.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(summary_payload, f, ensure_ascii=False, indent=2)


def _resolve_field(df: pd.DataFrame, column: str) -> Optional[str]:
    if column not in df.columns:
        return None
    values = df[column].dropna().unique()
    if len(values) == 0:
        return None
    return str(values[0])


if __name__ == "__main__":
    main()

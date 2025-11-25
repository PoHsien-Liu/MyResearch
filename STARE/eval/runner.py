"""CLI entry for unified evaluation."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Callable, List, Mapping, MutableMapping

from STARE.eval.metrics import evaluate_predictions_file
from STARE.eval.judge_backends import BackendName, call_judge_backend
from STARE.eval.prompt_template import SYSTEM_PROMPT, build_user_prompt, METRIC_KEYS


LOGGER = logging.getLogger(__name__)


def run_eval(args: argparse.Namespace) -> None:
    """Run evaluation given CLI args."""
    if not args.predictions_path:
        raise ValueError("predictions_path is required for --task eval")
    predictions_path = Path(args.predictions_path)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now(tz=timezone.utc)
    LOGGER.info("Evaluating predictions: %s", predictions_path)

    result = evaluate_predictions_file(
        predictions_path=predictions_path,
        unknown_policy=args.unknown_policy,
    )

    end_ts = datetime.now(tz=timezone.utc)
    duration_ms = int((end_ts - start_ts).total_seconds() * 1000)

    payload: Dict[str, Any] = {
        "args": _safe_args_snapshot(args),
        "label_policy": result["label_policy"],
        "classification_metrics": result["classification_metrics"],
        "sample_stats": result["sample_stats"],
        "explanation_metrics": {"status": "not_implemented"},
        "started_at": start_ts.isoformat().replace("+00:00", "Z"),
        "ended_at": end_ts.isoformat().replace("+00:00", "Z"),
        "duration_ms": duration_ms,
    }

    eval_path = output_dir / "eval.json"
    with eval_path.open("w") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved evaluation to %s", eval_path)


def bind_eval_subparser(parser: argparse.ArgumentParser) -> None:
    """Extend parser with eval-specific args."""
    parser.add_argument(
        "--predictions_path",
        default=None,
        help="Path to predictions.csv or predictions.jsonl to evaluate (required for --task eval)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write eval.json (default: same as predictions_path parent)",
    )
    parser.add_argument(
        "--unknown_policy",
        default="as_error",
        choices=["as_error", "as_invalid"],
        help="How to handle unknown/invalid predictions",
    )


def _safe_args_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse.Namespace to a JSON-serializable dict."""
    snapshot: Dict[str, Any] = {}
    for key, value in vars(args).items():
        try:
            json.dumps(value)
            snapshot[key] = value
        except (TypeError, ValueError):
            snapshot[key] = str(value)
    return snapshot


# -----------------------------------------------------------------------------
# Explanation evaluation helpers
# -----------------------------------------------------------------------------

def extract_json_from_text(text: str) -> str:
    """Extract the first JSON object substring from LLM output."""
    if text is None:
        raise ValueError("Received empty response from judge backend.")
    text = str(text)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    # Fallback: scan for braces that form valid JSON
    stack: List[int] = []
    for idx, ch in enumerate(text):
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop(0)
            candidate = text[start : idx + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
    raise ValueError("Could not extract JSON object from judge output.")


def _normalize_movement(label: str | None) -> str:
    """Map prediction label to human-friendly movement token."""
    if label is None:
        return "Unknown"
    norm = str(label).strip().lower()
    if norm in {"positive", "pos", "up", "1", "+"}:
        return "UP"
    if norm in {"negative", "neg", "down", "0", "-"}:
        return "DOWN"
    return "Unknown"


def evaluate_single_explanation(
    backend: BackendName,
    model_name: str,
    sample_id: str,
    ticker: str,
    date: str,
    context_texts: str,
    y_true: str,
    y_pred: str,
    predicted_movement: str | None,
    explanation: str,
    extra_meta: Mapping[str, Any] | None = None,
    call_backend_fn: Callable[..., str] | None = None,
) -> Dict[str, Any]:
    """Evaluate one explanation via LLM judge."""
    prompt_movement = predicted_movement or _normalize_movement(y_pred)
    user_prompt = build_user_prompt(
        ticker=ticker,
        date=date,
        context_texts=context_texts,
        predicted_movement=prompt_movement,
        explanation=explanation,
    )
    backend_fn = call_backend_fn or call_judge_backend
    raw_response = backend_fn(
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model_name=model_name,
    )
    json_text = extract_json_from_text(raw_response)
    parsed = json.loads(json_text)
    metric_scores = parsed.get("metric_scores", {})
    overall_comment = parsed.get("overall_comment", "")

    # ensure all metric keys exist
    for key in METRIC_KEYS:
        metric_scores.setdefault(key, None)

    result: MutableMapping[str, Any] = {
        "sample_id": sample_id,
        "ticker": ticker,
        "date": date,
        "y_true": y_true,
        "y_pred": y_pred,
        "predicted_movement": prompt_movement,
        "backend": backend,
        "judge_model_name": model_name,
        "metric_scores": metric_scores,
        "overall_comment": overall_comment,
        "raw_response": raw_response,
    }
    if extra_meta:
        result.update(extra_meta)
    return result


def evaluate_batch(
    backend: BackendName,
    model_name: str,
    records: List[Mapping[str, Any]],
    call_backend_fn: Callable[..., str] | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate a batch of explanations."""
    results: List[Dict[str, Any]] = []
    for rec in records:
        res = evaluate_single_explanation(
            backend=backend,
            model_name=model_name,
            sample_id=str(rec.get("sample_id", "")),
            ticker=str(rec.get("ticker", "")),
            date=str(rec.get("date", rec.get("prediction_date", ""))),
            context_texts=str(rec.get("context_texts", rec.get("context", ""))),
            y_true=str(rec.get("y_true", rec.get("ground_truth", ""))),
            y_pred=str(rec.get("y_pred", rec.get("prediction", ""))),
            predicted_movement=rec.get("predicted_movement"),
            explanation=str(rec.get("explanation", rec.get("raw_response", ""))),
            extra_meta={
                k: v for k, v in rec.items() if k not in {
                    "sample_id", "ticker", "date", "prediction_date",
                    "context_texts", "context", "y_true", "ground_truth",
                    "y_pred", "prediction", "predicted_movement", "explanation", "raw_response"
                }
            },
            call_backend_fn=call_backend_fn,
        )
        results.append(res)
    return results

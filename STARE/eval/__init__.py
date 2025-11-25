"""Evaluation utilities for STARE and baselines."""

from STARE.eval.metrics import evaluate_classification, evaluate_predictions_file
from STARE.eval.filters import filter_by_correct, filter_by_stock_scope
from STARE.eval.explanation_eval_main import (
    run_explanation_eval,
    run_explanation_eval_task,
    bind_explanation_eval_args,
)

__all__ = [
    "evaluate_classification",
    "evaluate_predictions_file",
    "filter_by_correct",
    "filter_by_stock_scope",
    "run_explanation_eval",
    "run_explanation_eval_task",
    "bind_explanation_eval_args",
]

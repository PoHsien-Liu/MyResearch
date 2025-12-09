"""Helpers for writing SFT samples and handling SFT split metadata."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from common.io.results import safe_name

from STARE.utils.paths import ensure_dir, get_pipeline_data_dir, project_root


def sft_file_path(dataset: str, experiment: str, split: str) -> Path:
    return get_pipeline_data_dir() / "sft" / "samples" / dataset.upper() / experiment / f"sft_samples_{split}.jsonl"


def write_sft_sample(
    *,
    result,
    system_prompt: str,
    user_prompt: str,
    all_events: List[Dict],
    experiment_name: Optional[str],
    prompt_variant: str,
    assistant_payload: Optional[Dict] = None,
    include_assistant: bool = True,
) -> Path:
    sft_split = result.selected.sft_split or "sft_train"
    exp = experiment_name or str(int(time.time()))
    out_dir = ensure_dir(get_pipeline_data_dir() / "sft" / "samples" / result.dataset_name / exp)
    out_path = out_dir / f"sft_samples_{sft_split}.jsonl"

    if assistant_payload is None:
        assistant_payload = {
            "prediction": result.selected.label,
            "reason": "",
        }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload)})

    record = {
        "messages": messages,
        "metadata": {
            "ticker": result.selected.ticker,
            "target_date": result.selected.target_date,
            "ground_truth_label": result.selected.label,
            "prompt_variant": prompt_variant,
            "price_context": result.price.context_text,
            "events": all_events,
            "sft_split": sft_split,
        },
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return out_path


# ----------------------------------------------------------------------------- 
# SFT split mapping helpers (train/val/test for SFT sampling) 
# ----------------------------------------------------------------------------- 


def _strategy_dir(label_strategy: str, neg_threshold: float, pos_threshold: float) -> str:
    strategy = (label_strategy or "legacy").lower()
    if strategy == "legacy":
        return "legacy"

    def _pct_tag(value: float) -> str:
        pct = value * 100
        txt = f"{pct:+.2f}".rstrip("0").rstrip(".")
        return txt.replace("+", "+").replace("-", "-") + "pct"

    return str(Path("dual") / f"neg{_pct_tag(neg_threshold)}_pos{_pct_tag(pos_threshold)}")


def _sft_split_map_path(
    *,
    dataset_name: str,
    train_ratio: float,
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
    split_root: Optional[Path] = None,
) -> Path:
    # Default to repo-level splits/ (sibling of datasets/)
    root = Path(split_root) if split_root else project_root() / "splits"
    dataset_safe = safe_name(dataset_name)
    ratio_dir = f"ratio-{train_ratio:.2f}".rstrip("0").rstrip(".")
    strategy = _strategy_dir(label_strategy, neg_threshold, pos_threshold)
    return root / dataset_safe / ratio_dir / strategy / "sft_split_map.json"


def _load_sft_split_map(
    *,
    dataset_name: str,
    train_ratio: float,
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
    split_root: Optional[Path] = None,
) -> Optional[Dict[str, Dict[str, list[str]]]]:
    path = _sft_split_map_path(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
        split_root=split_root,
    )
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("splits", {})
    except Exception:
        return None


def determine_sft_split(
    *,
    ticker: str,
    target_date: str,
    dataset_name: str,
    train_ratio: float,
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
    split_root: Optional[Path],
    mode: str,
) -> str:
    splits = _load_sft_split_map(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
        split_root=split_root,
    )
    ticker_up = ticker.upper()
    if splits:
        if target_date in set(splits.get("sft_train", {}).get(ticker_up, [])):
            return "sft_train"
        if target_date in set(splits.get("sft_val", {}).get(ticker_up, [])):
            return "sft_val"
        if target_date in set(splits.get("sft_test", {}).get(ticker_up, [])):
            return "sft_test"
    return "sft_train" if mode.lower() == "train" else "sft_test"


__all__ = [
    "write_sft_sample",
    "sft_file_path",
    "determine_sft_split",
]

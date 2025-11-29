"""SFT data prep and CLI bindings (finetune task skeleton)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from STARE.utils.paths import ensure_dir, get_outputs_dir

LOGGER = logging.getLogger("stare.sft")


def _model_slug(name: Optional[str]) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    return slug.replace(" ", "-")


def _auto_sft_path(dataset: str, model_slug: str, experiment: str, split: str) -> Path:
    return (
        get_outputs_dir()
        / "results"
        / dataset.upper()
        / "STARE"
        / model_slug
        / experiment
        / f"sft_samples_{split}.jsonl"
    )


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class SFTSplitData:
    split: str
    samples: List[Dict]


def _load_sft_file(path: Path, expected_split: Optional[str] = None) -> SFTSplitData:
    items: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"SFT file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                LOGGER.warning("Skip malformed line in %s: %s", path, exc)
                continue
            split = rec.get("metadata", {}).get("sft_split") or expected_split or "sft_train"
            rec.setdefault("metadata", {})["sft_split"] = split
            items.append(rec)
    return SFTSplitData(split=expected_split or "unknown", samples=items)


def bind_sft_finetune_args(parser) -> None:
    parser.add_argument("--train_sft_path", default=None, help="Path to sft_samples_sft_train.jsonl")
    parser.add_argument("--val_sft_path", default=None, help="Path to sft_samples_sft_val.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for SFT")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for SFT (placeholder)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size per device (placeholder)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval batch size per device (placeholder)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs (placeholder)")


def run_sft_finetune_task(args) -> None:
    base_model = getattr(args, "base_model", None)
    if not base_model:
        raise ValueError("--base_model is required for sft_finetune")
    dataset = args.dataset_name.upper()
    exp = getattr(args, "experiment_name", None) or str(int(time.time()))
    model_slug = _model_slug(base_model)
    out_dir = ensure_dir(get_outputs_dir() / "results" / dataset / "STARE" / model_slug / exp / "sft")

    train_path = Path(getattr(args, "train_sft_path", "") or _auto_sft_path(dataset, model_slug, exp, "sft_train"))
    val_path_arg = getattr(args, "val_sft_path", None)
    val_path = Path(val_path_arg) if val_path_arg else _auto_sft_path(dataset, model_slug, exp, "sft_val")

    train_data = _load_sft_file(train_path, expected_split="sft_train")
    val_data: Optional[SFTSplitData] = None
    if val_path and val_path.exists():
        val_data = _load_sft_file(val_path, expected_split="sft_val")
    else:
        LOGGER.warning("Val SFT file not found at %s; continuing with train only", val_path)

    stats = {
        "dataset": dataset,
        "base_model": base_model,
        "experiment_name": exp,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "counts": {
            "train": len(train_data.samples),
            "val": len(val_data.samples) if val_data else 0,
        },
    }
    _write_json(out_dir / "sft_data_stats.json", stats)
    _write_json(out_dir / "args.json", {k: v for k, v in vars(args).items()})

    log_lines = [
        f"[{time.strftime('%F %T')}] sft_finetune start",
        f"dataset={dataset} model={base_model} exp={exp}",
        f"train_samples={len(train_data.samples)} val_samples={len(val_data.samples) if val_data else 0}",
    ]
    with (out_dir / "run.log").open("a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    LOGGER.info("SFT data prepared at %s", out_dir)
    LOGGER.info("Train samples: %d | Val samples: %d", len(train_data.samples), len(val_data.samples) if val_data else 0)

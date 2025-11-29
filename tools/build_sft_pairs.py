"""
Utility to split a unified SFT samples file into train/val/test JSONL files
based on the precomputed sft_split_map.json (aligned with splits.json).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Tuple

from common.io.results import safe_name


def _ratio_dir(train_ratio: float) -> str:
    tag = f"{train_ratio:.2f}".rstrip("0").rstrip(".")
    return f"ratio-{tag}"


def _strategy_dir(label_strategy: str, neg_threshold: float, pos_threshold: float) -> str:
    strategy = (label_strategy or "legacy").lower()
    if strategy == "legacy":
        return "legacy"
    def _pct_tag(value: float) -> str:
        pct = value * 100
        return f"{pct:+.2f}".rstrip("0").rstrip(".") + "pct"
    return os.path.join("dual", f"neg{_pct_tag(neg_threshold)}_pos{_pct_tag(pos_threshold)}")


def _load_sft_split_map(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"sft split map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_lookup(split_map: Dict) -> Dict[str, Dict[str, set]]:
    lookup: Dict[str, Dict[str, set]] = {}
    for split, tickers in split_map.get("splits", {}).items():
        lookup[split] = {t: set(dates) for t, dates in tickers.items()}
    return lookup


def _resolve_split_dir(
    dataset: str,
    split_root: str,
    train_ratio: float,
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
) -> str:
    dataset_dir = safe_name(dataset)
    return os.path.join(
        split_root,
        dataset_dir,
        _ratio_dir(train_ratio),
        _strategy_dir(label_strategy, neg_threshold, pos_threshold),
    )


def _open_outputs(base_dir: str, dataset: str) -> Dict[str, Tuple[str, Iterable]]:
    out_dir = os.path.join(os.getenv("OUTPUTS_DIR", "./outputs"), "processed", safe_name(dataset))
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "sft_train": os.path.join(out_dir, "sft_pairs_train.jsonl"),
        "sft_val": os.path.join(out_dir, "sft_pairs_val.jsonl"),
        "sft_test": os.path.join(out_dir, "sft_pairs_test.jsonl"),
    }
    files = {k: open(v, "w", encoding="utf-8") for k, v in paths.items()}
    return files  # caller must close


def main() -> None:
    parser = argparse.ArgumentParser(description="Split SFT samples into train/val/test JSONL using sft_split_map.json")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (e.g., CMIN)")
    parser.add_argument("--input_samples", required=True, help="Path to unified sft_samples.jsonl")
    parser.add_argument("--split_root", default="splits", help="Root dir containing dataset splits (default: splits)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Base train ratio used in splits.json (default: 0.8)")
    parser.add_argument("--label_strategy", default="dual_threshold", help="Label strategy (default: dual_threshold)")
    parser.add_argument("--neg_threshold", type=float, default=-0.005, help="Negative threshold (default: -0.005)")
    parser.add_argument("--pos_threshold", type=float, default=0.0055, help="Positive threshold (default: 0.0055)")
    args = parser.parse_args()

    split_dir = _resolve_split_dir(
        dataset=args.dataset_name,
        split_root=args.split_root,
        train_ratio=args.train_ratio,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    sft_map_path = os.path.join(split_dir, "sft_split_map.json")
    split_map = _load_sft_split_map(sft_map_path)
    lookup = _build_lookup(split_map)

    out_files = _open_outputs(base_dir=args.split_root, dataset=args.dataset_name)
    counts = {"sft_train": 0, "sft_val": 0, "sft_test": 0, "skipped": 0}

    with open(args.input_samples, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["skipped"] += 1
                continue
            meta = rec.get("metadata") or {}
            ticker = str(meta.get("ticker") or "").upper()
            date = meta.get("target_date")
            if not ticker or not date:
                counts["skipped"] += 1
                continue

            split = None
            for split_name in ("sft_train", "sft_val", "sft_test"):
                if ticker in lookup.get(split_name, {}) and date in lookup[split_name][ticker]:
                    split = split_name
                    break
            if not split:
                counts["skipped"] += 1
                continue

            meta["split"] = split
            rec["metadata"] = meta
            out_files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split] += 1

    for f in out_files.values():
        f.close()

    print(f"Finished. Wrote train={counts['sft_train']} val={counts['sft_val']} test={counts['sft_test']} (skipped={counts['skipped']}).")


if __name__ == "__main__":
    main()

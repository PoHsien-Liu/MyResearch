#!/usr/bin/env python
"""Strip assistant messages from SFT sample jsonl to avoid label leakage."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple


def strip_file(path: Path, make_backup: bool = True) -> Tuple[int, int]:
    """Remove assistant-role messages from each json line. Returns (total, stripped)."""
    total = 0
    stripped = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    if make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                dst.write(line)
                continue
            msgs = obj.get("messages", [])
            filtered = [m for m in msgs if m.get("role") != "assistant"]
            if len(filtered) != len(msgs):
                stripped += 1
            obj["messages"] = filtered
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp_path.replace(path)
    return total, stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strip assistant messages from SFT jsonl.")
    parser.add_argument("jsonl_path", type=Path, help="Path to sft_samples_*.jsonl")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak backup")
    args = parser.parse_args(argv)

    path = args.jsonl_path
    if not path.exists():
        parser.error(f"File not found: {path}")

    total, stripped = strip_file(path, make_backup=not args.no_backup)
    print(f"Done. total_lines={total}, stripped={stripped}, output={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

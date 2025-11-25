#!/usr/bin/env python
"""Command-line helper to build company_relations.json."""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from STARE.stare.models.STARE.relation_builder import build_relations_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build company_relations.json via LLM")
    parser.add_argument("dataset", help="Dataset name (e.g., CMIN)")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional limit for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_relations_file(args.dataset, max_pairs=args.max_pairs)
    print(f"Relations saved to {path}")


if __name__ == "__main__":
    main()

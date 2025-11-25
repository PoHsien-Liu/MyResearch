#!/usr/bin/env python
"""Simple script to test STARE retriever outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TMP_DIR = os.path.join(os.path.expanduser("~"), "tmp")
try:
    os.makedirs(TMP_DIR, exist_ok=True)
    os.environ.setdefault("TMPDIR", TMP_DIR)
    os.environ.setdefault("TEMP", TMP_DIR)
    os.environ.setdefault("TMP", TMP_DIR)
except PermissionError:
    pass

from STARE.models.STARE.retriever import StareRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STARE retriever smoke test")
    parser.add_argument("query", help="Free-form text query")
    parser.add_argument("--dataset", default="CMIN", help="Dataset name (default: CMIN)")
    parser.add_argument(
        "--embed_model",
        default="FinLang/finance-embeddings-investopedia",
        help="Embedding model slug used for embeddings/index",
    )
    parser.add_argument("--top_k", type=int, default=5, help="How many docs to retrieve")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap when testing partial embeddings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = StareRetriever(
        dataset_name=args.dataset,
        embed_model=args.embed_model,
        max_rows=args.max_rows,
        top_k=args.top_k,
    )
    docs = retriever.query(args.query, top_k=args.top_k)
    print(f"Retrieved {len(docs)} docs for query: {args.query}")
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata
        preview = (meta.get("text") or "").replace("\n", " ")[:200]
        print(
            json.dumps(
                {
                    "rank": idx,
                    "score": round(doc.score, 4),
                    "ticker": meta.get("source_ticker"),
                    "date": meta.get("date"),
                    "text": preview,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Quick smoke test for relation LLM client (prints raw response)."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from STARE.configs.relations import RelationConfig
from STARE.models.STARE.relation_llm import RelationLLMClient


def main() -> None:
    config = RelationConfig(dataset="CMIN")
    client = RelationLLMClient(config)
    result = client.infer("AAPL", "TSM", {"cooc": 5600})
    print(result.raw_response)
    print(result)


if __name__ == "__main__":
    main()

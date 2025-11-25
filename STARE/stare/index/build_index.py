"""Build FAISS index from embeddings."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from STARE.stare.utils.logger import setup_logger
from STARE.stare.utils.paths import ensure_dir, indices_dir


def run_build_index(args) -> None:
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "build_index.log"
    logger = setup_logger("stare.build_index", log_file=log_file)

    emb_path = output_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found at {emb_path}")

    vectors = np.load(emb_path)
    logger.info("Loaded embeddings: shape %s", vectors.shape)

    if faiss is None:
        raise ImportError("faiss is not installed; cannot build index")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    idx_path = output_dir / "index.faiss"
    faiss.write_index(index, str(idx_path))
    logger.info("Saved FAISS index to %s", idx_path)


__all__ = ["run_build_index"]

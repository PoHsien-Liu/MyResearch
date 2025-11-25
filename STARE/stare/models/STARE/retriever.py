"""Vector retriever for STARE using FAISS index and metadata."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from STARE.stare.utils.paths import indices_dir


@dataclass
class RetrievedDoc:
    text: str
    metadata: dict
    score: float


class StareRetriever:
    def __init__(
        self,
        dataset_name: str,
        embed_model: str,
        max_rows: Optional[int] = None,
        top_k: int = 5,
    ):
        if faiss is None:
            raise ImportError("faiss not installed")
        self.logger = logging.getLogger("stare.retriever")
        self.dataset_name = dataset_name
        self.embed_model = embed_model
        self.top_k = top_k
        self.base_dir = indices_dir(dataset_name, embed_model)
        self.max_rows = max_rows
        self._load_resources()

    def _load_resources(self) -> None:
        emb_path = self.base_dir / "embeddings.npy"
        meta_path = self.base_dir / "metadata.parquet"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Embeddings or metadata missing; run `embed` first.")
        self.embeddings = np.load(emb_path)
        if self.max_rows:
            self.embeddings = self.embeddings[: self.max_rows]
        self.metadata = pd.read_parquet(meta_path).head(len(self.embeddings))
        index_path = self.base_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError("index.faiss missing; run `build_index`.")
        self.index = faiss.read_index(str(index_path))
        if self.max_rows and self.index.ntotal > len(self.embeddings):
            # rebuild smaller index subset
            dim = self.embeddings.shape[1]
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(self.embeddings)
            self.index = new_index
        self.encoder = SentenceTransformer(self.embed_model)

    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_field: str = "date",
        allowed_tickers: Optional[List[str]] = None,
    ) -> List[RetrievedDoc]:
        top_k = top_k or self.top_k
        candidate_idx = None
        if start_date and end_date:
            candidate_idx = self._filter_candidates(start_date, end_date, date_field=date_field, allowed_tickers=allowed_tickers)
            if candidate_idx is not None and len(candidate_idx) == 0:
                return []

        query_vec = self.encoder.encode([text], normalize_embeddings=True)

        if candidate_idx is not None:
            # subset search: rebuild a tiny index on filtered rows
            sub_embeddings = self.embeddings[candidate_idx]
            dim = sub_embeddings.shape[1]
            sub_index = faiss.IndexFlatIP(dim)
            sub_index.add(sub_embeddings)
            scores, idx = sub_index.search(query_vec, min(top_k, len(candidate_idx)))
            idx = [[candidate_idx[i] if i >= 0 else -1 for i in row] for row in idx]
        else:
            scores, idx = self.index.search(query_vec, top_k)

        docs = []
        for score, doc_id in zip(scores[0], idx[0]):
            if doc_id == -1:
                continue
            meta = self.metadata.iloc[int(doc_id)].to_dict()
            meta["_row_id"] = int(doc_id)
            docs.append(
                RetrievedDoc(
                    text=meta.get("text", ""),
                    metadata=meta,
                    score=float(score),
                )
            )
        return docs

    def _filter_candidates(
        self,
        start_date: str,
        end_date: str,
        date_field: str = "date",
        allowed_tickers: Optional[List[str]] = None,
    ) -> Optional[List[int]]:
        """Return indices within date range (and optional ticker whitelist)."""
        if date_field not in self.metadata.columns:
            alt_field = "published_at" if "published_at" in self.metadata.columns else None
            if alt_field:
                date_field = alt_field
            else:
                self.logger.warning("Date field '%s' not found in metadata; skipping date filter", date_field)
                return None
        try:
            series = pd.to_datetime(self.metadata[date_field])
            mask = (series >= pd.to_datetime(start_date)) & (series <= pd.to_datetime(end_date))
            if allowed_tickers:
                col = "source_ticker"
                if col in self.metadata.columns:
                    allowed_set = {t.upper() for t in allowed_tickers}
                    mask = mask & self.metadata[col].str.upper().isin(allowed_set)
            return mask[mask].index.tolist()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to apply candidate filter %s..%s: %s", start_date, end_date, exc)
            return None


__all__ = ["StareRetriever", "RetrievedDoc"]

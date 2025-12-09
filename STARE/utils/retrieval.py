"""Retrieval helpers wrapping StareRetriever with caching."""
from __future__ import annotations

from typing import Dict, List, Optional

from STARE.models.STARE.retriever import StareRetriever, RetrievedDoc

# Cache retrievers so we do not reload encoder/index for every sample.
_RETRIEVER_CACHE: Dict[tuple[str, str], StareRetriever] = {}


def get_retriever(dataset_name: str, embed_model: Optional[str], top_k: int) -> StareRetriever:
    key = (dataset_name.upper(), embed_model or "default")
    if key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[key] = StareRetriever(
            dataset_name=dataset_name,
            embed_model=embed_model or "default",
            top_k=top_k,
        )
    retriever = _RETRIEVER_CACHE[key]
    retriever.top_k = top_k
    return retriever


def retrieve_events(
    *,
    dataset_name: str,
    embed_model: Optional[str],
    target_ticker: str,
    queries: List[str],
    start_date: str,
    end_date: str,
    top_k: int,
    date_field: str = "date",
) -> List[RetrievedDoc]:
    retriever = get_retriever(dataset_name=dataset_name, embed_model=embed_model, top_k=top_k)
    collected: List[RetrievedDoc] = []
    for q in queries:
        docs = retriever.query(
            q,
            top_k=top_k,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            allowed_tickers=[target_ticker],
        )
        collected.extend(docs)
    collected.sort(key=lambda d: d.score, reverse=True)
    return collected[:top_k]


__all__ = ["retrieve_events", "get_retriever"]

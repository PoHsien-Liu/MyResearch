"""Query generation helpers."""
from __future__ import annotations

import json
import ast
from typing import Dict, List, Optional

from STARE.llm_backend.inference import run_inference_batch
from STARE.utils.llm_json import extract_first_json
from STARE.utils.prompts import build_query_prompt, flatten_factors_for_prompt


def parse_queries_json(resp_text: str) -> List[str]:
    """
    parse the JSON object {"queries": ["...", "...", "..."]} from the LLM response,
    return a clean queries list. if any parsing fails, return [], to avoid the whole pipeline from crashing.
    `ticker` is only here for backward compatibility, not used inside the function.
    """

    content = resp_text or ""
    extracted = extract_first_json(content)
    if not extracted:
        return []

    data = None

    # first try standard JSON parse
    try:
        data = json.loads(extracted)
    except Exception:
        # if still fails, try using literal_eval to handle Python style literals (single quotes, etc.)
        try:
            data = ast.literal_eval(extracted)
        except Exception:
            return []

    # expect it to be a dict
    if not isinstance(data, dict):
        return []

    queries = data.get("queries")
    if not isinstance(queries, list):
        return []

    parsed: List[str] = []
    for q in queries:
        if isinstance(q, str):
            qs = q.strip()
            if qs:
                parsed.append(qs)

    return parsed


def generate_queries_step(
    *,
    ticker: str,
    target_date: str,
    start_date: str,
    end_date: str,
    factors_data: Dict,
    model_name: Optional[str],
    backend: str = "llama_8B",
    max_tokens: Optional[int] = None,
) -> Dict:
    factors_text = flatten_factors_for_prompt(factors_data)
    prompt = build_query_prompt(
        ticker=ticker,
        target_date=target_date,
        start_date=start_date,
        end_date=end_date,
        factors_text=factors_text,
    )
    resp = run_inference_batch(
        [prompt],
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.2,
        stop=["```"],
    )[0]
    queries = parse_queries_json(resp)
    return {
        "ticker": ticker,
        "target_date": target_date,
        "start_date": start_date,
        "end_date": end_date,
        "queries": queries,
        "system_prompt": prompt.system,
        "user_prompt": prompt.user,
        "raw_response": resp,
        "source": "generated",
    }


def batch_generate_queries(
    samples: List[Dict],
    *,
    model_name: Optional[str],
    backend: str = "llama_8B",
    max_tokens: Optional[int] = None,
) -> List[Dict]:
    """
    Batch-generate queries for multiple samples.
    Each sample dict should include: ticker, target_date, start_date, end_date, factors_data.
    """
    prompts = []
    for s in samples:
        factors_text = flatten_factors_for_prompt(s["factors_data"])
        prompt = build_query_prompt(
            ticker=s["ticker"],
            target_date=s["target_date"],
            start_date=s["start_date"],
            end_date=s["end_date"],
            factors_text=factors_text,
        )
        prompts.append(prompt)
    if not prompts:
        return []
    responses = run_inference_batch(
        prompts,
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.2,
        stop=["```"],
    )
    results: List[Dict] = []
    for s, prompt, resp in zip(samples, prompts, responses):
        queries = parse_queries_json(resp)
        results.append(
            {
                "ticker": s["ticker"],
                "target_date": s["target_date"],
                "start_date": s["start_date"],
                "end_date": s["end_date"],
                "queries": queries,
                "system_prompt": prompt.system,
                "user_prompt": prompt.user,
                "raw_response": resp,
                "source": "generated",
            }
        )
    return results


__all__ = ["generate_queries_step", "batch_generate_queries", "parse_queries_json"]

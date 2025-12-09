"""Factor generation helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from STARE.llm_backend.inference import run_inference_batch
from STARE.utils.llm_json import extract_first_json
from STARE.utils.paths import ensure_dir, get_pipeline_data_dir
from STARE.utils.prompts import build_factor_prompt


def _model_slug(name: Optional[str]) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-") or "default"


def factors_output_path(dataset: str, factor_model: Optional[str], ticker: str) -> Path:
    model_slug = _model_slug(factor_model)
    out_dir = ensure_dir(get_pipeline_data_dir() / "factors" / model_slug / dataset.upper())
    return out_dir / f"{ticker.upper()}.json"


def parse_factors_json(resp_text: str, ticker: str) -> Dict:
    content = resp_text or ""
    try:
        extracted = extract_first_json(content)
        return json.loads(extracted)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse factors JSON for {ticker}: {exc}. Raw response (first 400 chars): {content[:400]!r}"
        ) from exc


def generate_factors_step(
    *,
    ticker: str,
    dataset_name: str,
    model_name: Optional[str],
    backend: str = "llama_70B",
    force_regen: bool = False,
    max_tokens: int = 800,
) -> Dict:
    out_path = factors_output_path(dataset_name, model_name, ticker)
    if out_path.exists() and not force_regen:
        with out_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return {
            "ticker": ticker.upper(),
            "data": cached,
            "system_prompt": "CACHE_HIT",
            "user_prompt": "CACHE_HIT",
            "raw_response": None,
            "source": "cache",
        }

    prompt = build_factor_prompt(ticker)
    resp = run_inference_batch(
        [prompt],
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["```"],
    )[0]
    data = parse_factors_json(resp, ticker)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        "ticker": ticker.upper(),
        "data": data,
        "system_prompt": prompt.system,
        "user_prompt": prompt.user,
        "raw_response": resp,
        "source": "generated",
    }


def batch_generate_factors(
    tickers: list[str],
    dataset_name: str,
    model_name: Optional[str],
    backend: str = "llama_70B",
    max_tokens: Optional[int] = None,
) -> Dict[str, Dict]:
    """Batch-generate factors for given tickers; skip all cache logic here."""
    results: Dict[str, Dict] = {}
    if not tickers:
        return results

    prompts = [build_factor_prompt(t) for t in tickers]
    responses = run_inference_batch(
        prompts,
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["```"],
    )
    for t, prompt, resp in zip(tickers, prompts, responses):
        data = parse_factors_json(resp, t)
        out_path = factors_output_path(dataset_name, model_name, t)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        results[t.upper()] = {
            "ticker": t.upper(),
            "data": data,
            "system_prompt": prompt.system,
            "user_prompt": prompt.user,
            "raw_response": resp,
            "source": "generated",
        }

    return results


__all__ = ["generate_factors_step", "batch_generate_factors", "parse_factors_json", "factors_output_path"]

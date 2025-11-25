"""Base single-company RAG training pipeline skeleton for STARE."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
from common.data.loader import get_record, list_trading_days

from STARE.stare.configs.dataset import DATASET_REGISTRY
from STARE.stare.llm_backend.inference import PromptLike, run_inference_batch, clear_llm_cache
from STARE.stare.utils.paths import dataset_paths, ensure_dir, get_outputs_dir
from STARE.stare.utils.price import build_price_context
from STARE.stare.models.STARE.retriever import StareRetriever, RetrievedDoc
from STARE.stare.utils.prompts import (
    build_factor_prompt,
    build_prediction_prompts,
    build_query_prompt,
    flatten_factors_for_prompt,
)

LOGGER = logging.getLogger("stare.pipeline")


# -----------------------------------------------------------------------------
# Data classes for intermediate results
# -----------------------------------------------------------------------------

@dataclass
class PriceContext:
    dates: List[str]
    returns: List[float]
    context_text: str


@dataclass
class SelectedSample:
    ticker: str
    target_date: str
    label: str
    ret_value: float
    sample_index: int
    mode: str


@dataclass
class FactorResult:
    ticker: str
    data: Dict
    system_prompt: str
    user_prompt: str
    raw_response: Optional[str]
    source: str  # generated | cache


@dataclass
class QueryResult:
    ticker: str
    target_date: str
    start_date: str
    end_date: str
    queries: List[str]
    system_prompt: str
    user_prompt: str
    raw_response: Optional[str]
    source: str  # generated


@dataclass
class BaseSampleResult:
    dataset_name: str
    price: PriceContext
    selected: SelectedSample
    factors: Optional[FactorResult] = None
    queries: Optional[QueryResult] = None
    retrieved: Optional[List[RetrievedDoc]] = None
    # TODO: add queries, retrieved_events, sft_sample, training_outputs as we extend steps.


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_price_dir(dataset_name: str) -> Path:
    """Choose the directory that contains ticker CSVs for prices."""
    paths = dataset_paths(dataset_name)
    base = Path(paths.price_path)
    raw_dir = base / "raw"
    if raw_dir.exists():
        return raw_dir
    return base


def _pick_sample(
    dataset_name: str,
    mode: str,
    price_dir: Path,
    seq_len: int,
    ticker: Optional[str],
    target_date: Optional[str],
    sample_index: int,
) -> Dict[str, str]:
    """Pick one (ticker, date, label) sample using common.data.loader.list_trading_days."""
    mode = mode.lower()
    if mode not in {"train", "test"}:
        raise ValueError(f"mode must be train or test, got {mode}")

    if ticker and target_date:
        return {"ticker": ticker.upper(), "date": target_date}

    samples = list_trading_days(
        dataset_name=dataset_name,
        price_dir=str(price_dir),
        mode=mode,
        seq_len=seq_len,
    )
    if not samples:
        raise RuntimeError(f"No trading day samples found for dataset={dataset_name} mode={mode}")

    if ticker:
        filtered = [s for s in samples if s["ticker"].upper() == ticker.upper()]
        if not filtered:
            raise RuntimeError(f"No samples found for ticker={ticker} in dataset={dataset_name}")
        samples = filtered

    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index {sample_index} out of range (0..{len(samples)-1})")
    return samples[sample_index]


def _last_k_returns(context_returns: List[Dict], k: int) -> Tuple[List[str], List[float]]:
    if len(context_returns) < k:
        raise ValueError(f"Need at least {k} context returns, got {len(context_returns)}")
    tail = context_returns[-k:]
    dates = [r["date"] for r in tail]
    returns = [float(r["ret"]) * 100 for r in tail]  # convert to percentage
    return dates, returns


# -----------------------------------------------------------------------------
# Factor generation (lazy, cached)
# -----------------------------------------------------------------------------


def _factors_output_path(ticker: str) -> Path:
    out_dir = ensure_dir(get_outputs_dir() / "factors")
    return out_dir / f"{ticker.upper()}.json"


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ``` fences (with optional language tag)."""
    text = text.strip()
    if "```" not in text:
        return text
    # take content between first and last fence
    first = text.find("```")
    last = text.rfind("```")
    if first == -1 or last == -1 or last <= first:
        return text
    inner = text[first + 3 : last].strip()
    # drop leading language tag if present (e.g., "json\n{...}")
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        if re.fullmatch(r"[a-zA-Z0-9_+-]+", first_line.strip()):
            return rest.strip()
    return inner.strip()


def _extract_first_json(text: str) -> str:
    """Best-effort extraction of first JSON object substring."""
    stripped = _strip_code_fences(text)
    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    start = None
    depth = 0
    for idx, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = stripped[start : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
    raise ValueError(f"Could not extract JSON from LLM response (first 400 chars): {stripped[:400]}")


def _parse_llm_json(resp_text: str, ticker: str) -> Dict:
    content = resp_text or ""
    try:
        extracted = _extract_first_json(content)
        cleaned = re.sub(r",\\s*}", "}", extracted)
        cleaned = re.sub(r",\\s*]", "]", cleaned)
        if cleaned.count("{") != cleaned.count("}") or cleaned.count("[") != cleaned.count("]"):
            raise ValueError("LLM output appears truncated or unbalanced.")
        return json.loads(cleaned)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse factors JSON for {ticker}: {exc}. "
            f"Raw response (first 400 chars): {content[:400]!r}"
        ) from exc


def _build_factor_prompt(ticker: str) -> PromptLike:
    system_text = (
        "You are a financial analyst who understands common drivers of stock price movements for publicly listed companies.\n"
        "Your task is to list the most important types of events and factors that typically move the stock price of a given company.\n"
        "Respond in strict JSON with keys: ticker, factors[name, description, keywords]."
    )
    user_text = f"""Ticker: {ticker}
Please output the JSON object described above."""
    return PromptLike(system=system_text, user=user_text)


def generate_factors(
    ticker: str,
    *,
    model_name: Optional[str] = None,
    backend: str = "llama",
    force_regen: bool = False,
    max_tokens: int = 800,
) -> FactorResult:
    """
    Generate (or load cached) factors for a ticker.
    """
    out_path = _factors_output_path(ticker)
    if out_path.exists() and not force_regen:
        with out_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return FactorResult(
            ticker=ticker.upper(),
            data=cached,
            system_prompt="CACHE_HIT",
            user_prompt="CACHE_HIT",
            raw_response=None,
            source="cache",
        )

    prompt = build_factor_prompt(ticker)
    resp = run_inference_batch(
        [prompt],
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["```"],
    )[0]
    data = _parse_llm_json(resp, ticker)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return FactorResult(
        ticker=ticker.upper(),
        data=data,
        system_prompt=prompt.system,
        user_prompt=prompt.user,
        raw_response=resp,
        source="generated",
    )


# -----------------------------------------------------------------------------
# Query generation
# -----------------------------------------------------------------------------


def _flatten_factors_for_prompt(factors_data: Dict) -> str:
    lines: List[str] = []
    factors = factors_data.get("factors", [])
    for item in factors:
        name = item.get("name") or ""
        desc = item.get("description") or ""
        kws = item.get("keywords") or []
        if isinstance(kws, str):
            kws = [kws]
        kw_text = ", ".join(str(k).strip() for k in kws if str(k).strip())
        lines.append(f"- {name} (keywords: {kw_text}) - {desc}")
    return "\n".join(lines)


def _parse_queries_json(resp_text: str, ticker: str) -> List[str]:
    content = resp_text or ""
    extracted = _extract_first_json(content)
    cleaned = re.sub(r",\\s*}", "}", extracted)
    cleaned = re.sub(r",\\s*]", "]", cleaned)
    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse queries JSON for {ticker}: {exc}. "
            f"Raw response (first 400 chars): {content[:400]!r}"
        ) from exc
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError("queries field missing or not a list")
    parsed = []
    for q in queries:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q:
            parsed.append(q)
    return parsed


def generate_queries(
    *,
    ticker: str,
    target_date: str,
    start_date: str,
    end_date: str,
    factors_data: Dict,
    model_name: Optional[str] = None,
    backend: str = "llama",
    max_tokens: int = 256,
) -> QueryResult:
    factors_text = _flatten_factors_for_prompt(factors_data)
    prompt = _build_query_prompt(
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
    queries = _parse_queries_json(resp, ticker)
    return QueryResult(
        ticker=ticker,
        target_date=target_date,
        start_date=start_date,
        end_date=end_date,
        queries=queries,
        system_prompt=prompt.system,
        user_prompt=prompt.user,
        raw_response=resp,
        source="generated",
    )


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
    retriever = StareRetriever(dataset_name=dataset_name, embed_model=embed_model or "default", top_k=top_k)
    combined: Dict[str, RetrievedDoc] = {}
    for q in queries:
        docs = retriever.query(
            q,
            top_k=top_k,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            allowed_tickers=[target_ticker],
        )
        for doc in docs:
            row_id = doc.metadata.get("_row_id")
            date_val = doc.metadata.get(date_field) or doc.metadata.get("published_at") or ""
            src = doc.metadata.get("source_path") or ""
            text_norm = str(doc.text or "").strip().lower()
            if src and date_val:
                key = f"{src}|{date_val}|{text_norm}"
            elif date_val:
                key = f"{date_val}|{text_norm}"
            elif row_id is not None:
                key = str(row_id)
            else:
                key = text_norm
            if key not in combined or doc.score > combined[key].score:
                combined[key] = doc
    # sort by score desc and keep top_k
    sorted_docs = sorted(combined.values(), key=lambda d: d.score, reverse=True)
    return sorted_docs[:top_k]


# -----------------------------------------------------------------------------
# Prediction prompt + SFT helpers
# -----------------------------------------------------------------------------


def _model_slug(name: Optional[str]) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-") or "default"


def _build_events_block(retrieved: List[RetrievedDoc], target_ticker: str, include_related: bool) -> Tuple[str, List[Dict]]:
    target_ticker = target_ticker.upper()
    target_events = []
    related_events: Dict[str, List[Dict]] = {}
    all_events: List[Dict] = []

    for idx, doc in enumerate(retrieved, 1):
        row = {
            "id": idx,
            "date": doc.metadata.get("date") or doc.metadata.get("created_at") or "",
            "text": doc.text,
            "score": doc.score,
            "source_ticker": str(doc.metadata.get("source_ticker") or "").upper(),
        }
        if row["source_ticker"] == target_ticker or not include_related:
            target_events.append(row)
        else:
            related_events.setdefault(row["source_ticker"], []).append(row)
        all_events.append(row)

    lines: List[str] = []
    lines.append("[EVENTS]")
    if target_events:
        lines.append("Target firm news:")
        for ev in target_events:
            lines.append(f"({ev['id']}) [{ev['date']}] {ev['text']}")
    else:
        lines.append("Target firm news: None.")

    if include_related:
        if related_events:
            lines.append("Related firm news:")
            for firm, items in related_events.items():
                lines.append(f"- Firm: {firm}")
                for ev in items:
                    lines.append(f"  ({ev['id']}) [{ev['date']}] {ev['text']}")
        else:
            lines.append("Related firm news: None.")

    return "\n".join(lines), all_events


def _build_prediction_prompts(
    *,
    ticker: str,
    target_date: str,
    price_context: str,
    events_text: str,
    include_related: bool,
) -> Tuple[str, str]:
    system_text = (
        "You are a cautious equity analyst. Use ONLY the provided price trend and news; do not add outside knowledge. "
        "If news is missing or weak, state that explicitly."
    )
    guidance = (
        "- Summarize the 5-day price trend (up/down/flat) and its implication.\n"
        "- Ground every claim on the evidence above; cite IDs like (1), (3). If no usable news, say so and rely on price trend.\n"
        "- If using related-firm news, explain briefly how it impacts the target (e.g., supply chain/sector sentiment/peers).\n"
        "- Keep the JSON concise; no markdown/code fences."
    )
    user_text = (
        f"Target stock: {ticker}\n"
        f"Prediction date (D0): {target_date}\n\n"
        f"{price_context}\n\n"
        f"{events_text}\n\n"
        "[TASK]\n"
        "Predict next-day movement (UP or DOWN) for the target stock (vs D-1 close) and explain with citations.\n"
        "Follow this guidance:\n"
        f"{guidance}\n\n"
        "[OUTPUT JSON]\n"
        "{\n"
        '  "prediction": "UP" or "DOWN",\n'
        '  "reason": "<short explanation with citations>",\n'
        '  "used_event_ids": [<list of integers>] // empty if none\n'
        "}"
    )
    if not include_related:
        user_text = user_text.replace("If using related-firm news, explain briefly how it impacts the target (e.g., supply chain/sector sentiment/peers).\n", "")
    return system_text, user_text


def _write_sft_sample(
    *,
    result: BaseSampleResult,
    system_prompt: str,
    user_prompt: str,
    all_events: List[Dict],
    outputs_dir: Path,
    query_model: Optional[str],
    experiment_name: Optional[str],
    prompt_variant: str,
) -> Path:
    model_slug = _model_slug(query_model)
    exp = experiment_name or str(int(time.time()))
    out_dir = ensure_dir(outputs_dir / "results" / result.dataset_name / "STARE" / model_slug / exp)
    out_path = out_dir / "sft_samples.jsonl"

    assistant_payload = {
        "prediction": result.selected.label,
        "reason": "",
        "used_event_ids": [],
    }
    record = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_payload)},
        ],
        "metadata": {
            "ticker": result.selected.ticker,
            "target_date": result.selected.target_date,
            "ground_truth_label": result.selected.label,
            "prompt_variant": prompt_variant,
            "price_context": result.price.context_text,
            "events": all_events,
        },
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return out_path


# -----------------------------------------------------------------------------
# Public entry: run a single sample through the pipeline (incrementally)
# -----------------------------------------------------------------------------

def run_base_sample(
    *,
    dataset_name: str,
    mode: str = "train",
    seq_len: int = 5,
    ticker: Optional[str] = None,
    target_date: Optional[str] = None,
    sample_index: int = 0,
    run_until: str = "price_context",
    price_dir_override: Optional[Path] = None,
    factor_model: Optional[str] = None,
    factor_backend: Optional[str] = None,
    query_backend: Optional[str] = None,
    query_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    force_regen_factors: bool = False,
    factor_max_tokens: int = 800,
    query_max_tokens: int = 256,
    top_k: int = 10,
    label_strategy: str = "dual_threshold",
    neg_threshold: float = -0.005,
    pos_threshold: float = 0.0055,
    prompt_variant: str = "target_only",
    experiment_name: Optional[str] = None,
) -> BaseSampleResult:
    """
    Run the base pipeline for a single sample up to `run_until`.

    Steps roadmap (currently implemented up to factors):
      1) pick sample via list_trading_days or provided ticker/date
      2) get_record to fetch price context (seq_len trading days before target_date)
      3) build PRICE_CONTEXT_BLOCK
      4) factors (generate or load cache)
      5) TODO: queries -> retrieval -> prediction -> sft -> finetune
    """
    dataset_key = dataset_name.upper()
    if dataset_key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    price_dir = price_dir_override or _resolve_price_dir(dataset_key)
    sample = _pick_sample(
        dataset_name=dataset_key,
        mode=mode,
        price_dir=price_dir,
        seq_len=seq_len,
        ticker=ticker,
        target_date=target_date,
        sample_index=sample_index,
    )
    ticker_sel = sample["ticker"]
    date_sel = sample["date"]
    LOGGER.info("Selected sample: %s %s (mode=%s idx=%d)", ticker_sel, date_sel, mode, sample_index)

    record = get_record(
        dataset_name=dataset_key,
        ticker=ticker_sel,
        date=date_sel,
        price_dir=str(price_dir),
        seq_len=seq_len,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
    )
    context_returns = record["price"]["context_returns"]
    dates, returns = _last_k_returns(context_returns, seq_len)
    price_context_text = build_price_context(
        ticker=ticker_sel,
        target_date=date_sel,
        last5_dates=dates,
        last5_returns=returns,
    )

    price_ctx = PriceContext(dates=dates, returns=returns, context_text=price_context_text)
    selected = SelectedSample(
        ticker=ticker_sel,
        target_date=date_sel,
        label=record["price"]["label"],
        ret_value=record["price"]["ret"],
        sample_index=sample_index,
        mode=mode,
    )
    result = BaseSampleResult(
        dataset_name=dataset_key,
        price=price_ctx,
        selected=selected,
    )

    if run_until == "price_context":
        return result

    if run_until in {"factors"}:
        factors = generate_factors(
            ticker=ticker_sel,
            model_name=factor_model,
            backend=factor_backend,
            force_regen=force_regen_factors,
            max_tokens=factor_max_tokens,
        )
        result.factors = factors
        return result

    if run_until in {"queries"}:
        factors = generate_factors(
            ticker=ticker_sel,
            model_name=factor_model,
            backend=factor_backend,
            force_regen=force_regen_factors,
            max_tokens=factor_max_tokens,
        )
        result.factors = factors
        if (query_model and query_model != factor_model) or (query_backend and query_backend != factor_backend):
            clear_llm_cache()
        start_date = dates[0]
        end_date = dates[-1]
        queries = generate_queries(
            ticker=ticker_sel,
            target_date=date_sel,
            start_date=start_date,
            end_date=end_date,
            factors_data=factors.data,
            model_name=query_model or factor_model,
            backend=query_backend or factor_backend,
            max_tokens=query_max_tokens,
        )
        result.queries = queries
        retrieved = retrieve_events(
            dataset_name=dataset_key,
            embed_model=embed_model,
            target_ticker=ticker_sel,
            queries=queries.queries,
            start_date=start_date,
            end_date=end_date,
            top_k=top_k,
        )
        result.retrieved = retrieved
        return result

    if run_until in {"prediction"}:
        factors = generate_factors(
            ticker=ticker_sel,
            model_name=factor_model,
            backend=factor_backend,
            force_regen=force_regen_factors,
            max_tokens=factor_max_tokens,
        )
        result.factors = factors
        if (query_model and query_model != factor_model) or (query_backend and query_backend != factor_backend):
            clear_llm_cache()
        start_date = dates[0]
        end_date = dates[-1]
        queries = generate_queries(
            ticker=ticker_sel,
            target_date=date_sel,
            start_date=start_date,
            end_date=end_date,
            factors_data=factors.data,
            model_name=query_model or factor_model,
            backend=query_backend or factor_backend,
            max_tokens=query_max_tokens,
        )
        result.queries = queries
        retrieved = retrieve_events(
            dataset_name=dataset_key,
            embed_model=embed_model,
            target_ticker=ticker_sel,
            queries=queries.queries,
            start_date=start_date,
            end_date=end_date,
            top_k=top_k,
        )
        result.retrieved = retrieved

        include_related = prompt_variant == "with_related"
        events_text, all_events = _build_events_block(retrieved, ticker_sel, include_related)
        system_prompt, user_prompt = _build_prediction_prompts(
            ticker=ticker_sel,
            target_date=date_sel,
            price_context=price_context_text,
            events_text=events_text,
            include_related=include_related,
        )
        out_path = _write_sft_sample(
            result=result,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            all_events=all_events,
            outputs_dir=get_outputs_dir(),
            query_model=query_model or factor_model,
            experiment_name=experiment_name or None,
            prompt_variant=prompt_variant,
        )
        LOGGER.info("Saved SFT sample to %s", out_path)
        return result

    raise NotImplementedError(f"run_until={run_until} not yet supported in this stage.")

    return result


def run_train(args) -> None:
    """
    Train-task entry point (currently runs sample(s) up to run_until).

    If args.test_sample is True, only the sample at sample_index is processed.
    Otherwise, all training samples are iterated (may be slow until later stages are optimized).
    """
    dataset_name = args.dataset_name
    run_until = getattr(args, "run_until", "price_context")
    seq_len = getattr(args, "seq_len", 5)
    test_only = bool(getattr(args, "test_sample", False))
    sample_idx = int(getattr(args, "sample_index", 0))
    force_regen_factors = bool(getattr(args, "force_regen_factors", False))
    factor_model = getattr(args, "factor_model", None) or getattr(args, "base_model", None)
    query_model = getattr(args, "query_model", None) or getattr(args, "base_model", None)
    factor_backend = getattr(args, "factor_backend", None) or "llama"
    query_backend = getattr(args, "query_backend", None) or factor_backend
    embed_model = getattr(args, "embed_model", None)
    factor_max_tokens = int(getattr(args, "factor_max_tokens", 800))
    query_max_tokens = int(getattr(args, "query_max_tokens", 256))
    top_k = int(getattr(args, "top_k", 10))
    label_strategy = getattr(args, "label_strategy", "dual_threshold")
    neg_threshold = float(getattr(args, "neg_threshold", -0.005))
    pos_threshold = float(getattr(args, "pos_threshold", 0.0055))
    prompt_variant = getattr(args, "prompt_variant", "target_only")
    experiment_name = getattr(args, "experiment_name", None)

    price_dir = _resolve_price_dir(dataset_name.upper())
    samples = list_trading_days(
        dataset_name=dataset_name.upper(),
        price_dir=str(price_dir),
        mode="train",
        seq_len=seq_len,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
    )
    if not samples:
        raise RuntimeError(f"No training samples found for dataset={dataset_name}")

    if test_only:
        if sample_idx < 0 or sample_idx >= len(samples):
            raise IndexError(f"sample_index {sample_idx} out of range (0..{len(samples)-1})")
        samples = [samples[sample_idx]]
        LOGGER.info("Running test_sample only: idx=%d", sample_idx)
    else:
        LOGGER.info("Running full training set with %d samples", len(samples))

    for idx, sample in enumerate(samples):
        LOGGER.info(
            "[%d/%d] Processing %s %s",
            idx + 1,
            len(samples),
            sample["ticker"],
            sample["date"],
        )
        try:
            res = run_base_sample(
                dataset_name=dataset_name,
                mode="train",
                seq_len=seq_len,
                ticker=sample["ticker"],
                target_date=sample["date"],
                sample_index=idx,
                run_until=run_until,
                factor_model=factor_model,
                factor_backend=factor_backend,
                query_backend=query_backend,
                query_model=query_model,
                embed_model=embed_model,
                force_regen_factors=force_regen_factors,
                factor_max_tokens=factor_max_tokens,
                query_max_tokens=query_max_tokens,
                top_k=top_k,
                label_strategy=label_strategy,
                neg_threshold=neg_threshold,
                pos_threshold=pos_threshold,
                prompt_variant=prompt_variant,
                experiment_name=experiment_name,
            )
            if test_only:
                print("=== Test Sample Output ===")
                print(f"Ticker: {res.selected.ticker}")
                print(f"Target date: {res.selected.target_date}")
                print(f"Label: {res.selected.label} (ret={res.selected.ret_value})")
                if run_until == "price_context":
                    print(res.price.context_text)
                if run_until == "factors":
                    if res.factors:
                        print(f"Factor source: {res.factors.source}")
                        print("---- Factor Generation Prompt (system) ----")
                        print(res.factors.system_prompt)
                        print("---- Factor Generation Prompt (user) ----")
                        print(res.factors.user_prompt)
                        if res.factors.raw_response:
                            print("---- Raw LLM Response ----")
                            print(res.factors.raw_response)
                        print("---- Parsed Factors JSON ----")
                        print(json.dumps(res.factors.data, indent=2))
                if run_until == "queries":
                    if res.queries:
                        print("---- Query Generation Prompt (system) ----")
                        print(res.queries.system_prompt)
                        print("---- Query Generation Prompt (user) ----")
                        print(res.queries.user_prompt)
                        if res.queries.raw_response:
                            print("---- Raw LLM Response ----")
                            print(res.queries.raw_response)
                        print("---- Parsed Queries ----")
                        print(json.dumps({"queries": res.queries.queries}, indent=2))
                    if res.retrieved:
                        print(f"---- Retrieved Top-{len(res.retrieved)} Docs ----")
                        for i, doc in enumerate(res.retrieved, 1):
                            date = doc.metadata.get("date") or doc.metadata.get("published_at")
                            print(f"({i}) [{date}] score={doc.score:.4f} {doc.text[:200]}")
                if run_until == "prediction":
                    print("SFT sample written (prediction + optional explanation format).")
        except Exception as exc:
            if test_only:
                raise
            LOGGER.warning("Skipping sample %s %s due to error: %s", sample["ticker"], sample["date"], exc)


__all__ = [
    "run_base_sample",
    "run_train",
    "BaseSampleResult",
    "PriceContext",
    "SelectedSample",
    "FactorResult",
    "QueryResult",
]

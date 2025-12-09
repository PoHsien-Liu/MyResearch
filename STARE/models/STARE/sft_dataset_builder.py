"""Batch builder that turns dataset samples into SFT prompt records (train/test)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from common.data.loader import get_record, list_trading_days

from tqdm import tqdm

from STARE.llm_backend.inference import clear_llm_cache
from STARE.models.STARE.prompt_builder import build_prediction_prompt_package
from STARE.utils.factors import batch_generate_factors, factors_output_path
from STARE.utils.price import build_price_context, resolve_price_dir, last_k_returns
from STARE.utils.queries import batch_generate_queries
from STARE.utils.retrieval import retrieve_events
from STARE.utils.sft import determine_sft_split, sft_file_path, write_sft_sample
from STARE.utils.paths import indices_dir

LOGGER = logging.getLogger("stare.sft_dataset_builder")


def _load_factors_cache(tickers: List[str], dataset: str, model_name: str, force_regen: bool, max_tokens: int) -> Dict[str, Dict]:
    """Load cached factors first; generate missing ones in batch; return map ticker->payload."""
    factors_map: Dict[str, Dict] = {}
    missing: List[str] = []

    for t in tickers:
        t_up = t.upper()
        f_path = factors_output_path(dataset, model_name, t)
        if f_path.exists() and not force_regen:
            with f_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            factors_map[t_up] = {
                "ticker": t_up,
                "data": cached,
                "raw_response": None,
                "source": "cache",
            }
        else:
            missing.append(t)

    if missing:
        generated = batch_generate_factors(
            missing,
            dataset_name=dataset,
            model_name=model_name,
            backend="llama_70B",
            max_tokens=max_tokens,
        )
        factors_map.update(generated)

    return factors_map


def prepare_sft_samples(args, mode: str) -> Path:
    """
    Generate SFT samples for the given split with batched factor/query generation.
    """

    idx_path = indices_dir(args.dataset_name, args.embed_model) / "index.faiss"
    if not idx_path.exists():
        raise FileNotFoundError(f"Index not found at {idx_path}. Run build_index_pipeline first.")

    exp = args.experiment_name or str(int(time.time()))
    args.experiment_name = exp
    mode = mode.lower()

    target_split = "sft_test" if mode == "test" else "sft_train"
    sft_path = sft_file_path(args.dataset_name, exp, target_split)
    if sft_path.exists() and not getattr(args, "force_regen_factors", False):
        LOGGER.info(
            "SFT file already exists at %s; reuse (mode=%s). max_samples will be applied at inference time.",
            sft_path,
            mode,
        )
        return sft_path

    price_dir = resolve_price_dir(args.dataset_name.upper())
    samples = list_trading_days(
        dataset_name=args.dataset_name.upper(),
        price_dir=str(price_dir),
        mode=mode,
        seq_len=args.seq_len,
        train_ratio=float(args.train_ratio),
        split_root=str(args.split_root) if args.split_root else None,
        label_strategy=args.label_strategy,
        neg_threshold=float(args.neg_threshold),
        pos_threshold=float(args.pos_threshold),
    )
    if not samples:
        raise RuntimeError(f"No samples found for dataset={args.dataset_name} mode={mode}")

    if args.only_ticker:
        only_ticker_up = args.only_ticker.upper()
        samples = [s for s in samples if s["ticker"].upper() == only_ticker_up]
        if not samples:
            raise RuntimeError(f"No samples found for ticker={args.only_ticker} in dataset={args.dataset_name}")
        LOGGER.info("Filtering to ticker=%s: %d samples", only_ticker_up, len(samples))

    # Dedup tickers and precompute factors in batch
    tickers: List[str] = sorted({s["ticker"].upper() for s in samples})
    factors_cache = _load_factors_cache(
        tickers,
        dataset=args.dataset_name.upper(),
        model_name=args.factor_model,
        force_regen=args.force_regen_factors,
        max_tokens=int(args.factor_max_tokens),
    )

    # Release 70B before running queries/base.
    clear_llm_cache()

    # Prepare per-sample contexts (price info + factors)
    sample_ctxs = []
    max_samples = getattr(args, "max_samples", None)
    for idx, sample in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break
        rec = get_record(
            dataset_name=args.dataset_name.upper(),
            ticker=sample["ticker"],
            date=sample["date"],
            price_dir=str(price_dir),
            seq_len=args.seq_len,
            label_strategy=args.label_strategy,
            neg_threshold=float(args.neg_threshold),
            pos_threshold=float(args.pos_threshold),
        )
        dates, returns = last_k_returns(rec["price"]["context_returns"], args.seq_len)
        price_context_text = build_price_context(
            ticker=sample["ticker"],
            target_date=sample["date"],
            last5_dates=dates,
            last5_returns=returns,
        )
        sft_split = determine_sft_split(
            ticker=sample["ticker"],
            target_date=sample["date"],
            dataset_name=args.dataset_name.upper(),
            train_ratio=float(args.train_ratio),
            label_strategy=args.label_strategy,
            neg_threshold=float(args.neg_threshold),
            pos_threshold=float(args.pos_threshold),
            split_root=Path(args.split_root) if args.split_root else None,
            mode=mode,
        )
        sample_ctxs.append(
            {
                "ticker": sample["ticker"],
                "target_date": sample["date"],
                "sample_index": idx,
                "label": rec["price"]["label"],
                "ret_value": rec["price"]["ret"],
                "dates": dates,
                "returns": returns,
                "price_context_text": price_context_text,
                "start_date": dates[0],
                "end_date": dates[-1],
                "sft_split": sft_split,
                "factors": factors_cache[sample["ticker"].upper()]["data"],
            }
        )

    # Batch-generate queries
    query_inputs = [
        {
            "ticker": s["ticker"],
            "target_date": s["target_date"],
            "start_date": s["start_date"],
            "end_date": s["end_date"],
            "factors_data": s["factors"],
        }
        for s in sample_ctxs
    ]
    queries_results = batch_generate_queries(
        query_inputs,
        model_name=args.query_model,
        backend="llama_8B",
        max_tokens=int(args.query_max_tokens),
    )

    # Per-sample retrieval + prompt + SFT write
    counters: Dict[str, int] = {"total": len(sample_ctxs), "succeeded": 0, "failed": 0, "query_errors": 0}
    for ctx, qres in tqdm(
        zip(sample_ctxs, queries_results),
        total=len(sample_ctxs),
        desc="Samples",
    ):
        if not qres.get("queries"):
            counters["query_errors"] += 1
            raw_resp = qres.get("raw_response")
            preview = None
            if isinstance(raw_resp, str):
                preview = raw_resp[:400]
            LOGGER.warning(
                "No queries generated for %s %s; skipping. Raw response (first 400 chars): %r",
                ctx["ticker"],
                ctx["target_date"],
                preview,
            )
            continue
        retrieved = retrieve_events(
            dataset_name=args.dataset_name.upper(),
            embed_model=args.embed_model,
            target_ticker=ctx["ticker"],
            queries=qres["queries"],
            start_date=ctx["start_date"],
            end_date=ctx["end_date"],
            top_k=int(args.top_k),
        )
        pkg = build_prediction_prompt_package(
            ticker=ctx["ticker"],
            target_date=ctx["target_date"],
            price_context=ctx["price_context_text"],
            retrieved=retrieved,
            label=ctx["label"],
            prompt_variant=args.prompt_variant,
        )
        include_assistant = mode != "test" and ctx["sft_split"] != "sft_test"
        result_stub = SimpleNamespace(
            dataset_name=args.dataset_name.upper(),
            price=SimpleNamespace(context_text=ctx["price_context_text"]),
            selected=SimpleNamespace(
                ticker=ctx["ticker"],
                target_date=ctx["target_date"],
                label=ctx["label"],
                ret_value=ctx["ret_value"],
                sample_index=ctx["sample_index"],
                mode=mode,
                sft_split=ctx["sft_split"],
            ),
        )
        out_path = write_sft_sample(
            result=result_stub,
            system_prompt=pkg.system_prompt,
            user_prompt=pkg.user_prompt,
            all_events=pkg.all_events,
            experiment_name=exp,
            prompt_variant=args.prompt_variant,
            assistant_payload=pkg.assistant_payload,
            include_assistant=include_assistant,
        )
        counters["succeeded"] += 1

    LOGGER.info(
        "SFT samples generated (mode=%s): %d/%d (query_errors=%d) (exp=%s)",
        mode,
        counters["succeeded"],
        counters["total"],
        counters["query_errors"],
        exp,
    )

    clear_llm_cache()
    return sft_path


__all__ = ["prepare_sft_samples"]

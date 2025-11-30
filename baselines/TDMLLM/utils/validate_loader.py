#!/usr/bin/env python3
"""
Lightweight validator for common/data/loader APIs.

Validates:
- list_trading_days: returns reasonable samples for the chosen dataset
- get_record: computes labels/returns and exposes text_window_dates
- load_texts_for_day: loads per-day texts and matches get_record's same-day texts

Defaults to the SAMPLE dataset to keep runtime very small.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.config.datasets import resolve_dataset_paths
from common.data.loader import (
    list_trading_days,
    get_record,
    load_texts_for_day,
)


def _resolve_root(env_value: str | None, fallback: str) -> str:
    if not env_value:
        return fallback
    return env_value if os.path.isabs(env_value) else os.path.abspath(os.path.join(REPO_ROOT, env_value))


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_list_trading_days(
    dataset_name: str,
    price_dir: str,
    seq_len: int,
    splits_dir: str,
    mode: str,
    limit: int,
    label_strategy: str,
    neg_threshold: float,
    pos_threshold: float,
) -> List[dict]:
    _print_header("1) list_trading_days")
    samples = list_trading_days(
        dataset_name=dataset_name,
        price_dir=price_dir,
        mode=mode,
        seq_len=seq_len,
        split_root=splits_dir,
        logger=None,
        label_strategy=label_strategy,
        neg_threshold=neg_threshold,
        pos_threshold=pos_threshold,
    )
    stats = getattr(list_trading_days, "last_stats", {})
    print(f"Total samples (test): {len(samples)}")
    if stats:
        print(
            "Stats -> total_candidates={tc} split_candidates={sc} neutral_skipped={ns} kept={kp}".format(
                tc=stats.get("total_candidates", 0),
                sc=stats.get("split_candidates", 0),
                ns=stats.get("neutral_skipped", 0),
                kp=stats.get("kept_samples", len(samples)),
            )
        )
    for row in samples[:limit]:
        print(f"  - {row['ticker']} @ {row['date']}")

    # Quick sanity for SAMPLE dataset universe
    if dataset_name.upper() == "SAMPLE":
        tickers = sorted({s["ticker"] for s in samples})
        assert set(tickers) <= {"AAPL", "BABA"}, f"Unexpected tickers for SAMPLE: {tickers}"
        print(f"Tickers OK for SAMPLE: {tickers}")
    return samples[:limit]


def validate_get_record_and_texts(
    dataset_name: str,
    price_dir: str,
    tweet_dir: str | None,
    news_csv_dir: str | None,
    seq_len: int,
    mode: str,
    samples: List[dict],
    dump_records: bool = False,
    dump_dir: Optional[str] = None,
    label_strategy: str = "legacy",
    neg_threshold: float = -0.005,
    pos_threshold: float = 0.0055,
):
    _print_header("2) get_record + load_texts_for_day")
    dump_base = None
    if dump_dir:
        dump_base = os.path.join(dump_dir, dataset_name, mode)
        os.makedirs(dump_base, exist_ok=True)

    for i, s in enumerate(samples):
        ticker = s["ticker"]
        date = s["date"]
        rec = get_record(
            dataset_name=dataset_name,
            ticker=ticker,
            date=date,
            price_dir=price_dir,
            tweet_dir=tweet_dir,
            news_csv_dir=news_csv_dir,
            seq_len=seq_len,
            label_strategy=label_strategy,
            neg_threshold=neg_threshold,
            pos_threshold=pos_threshold,
            logger=None,
        )

        price = rec["price"]
        label = price["label"]
        ret_val = price["ret"]
        ctx = price.get("context_returns", [])
        wnd = rec.get("text_window_dates", [])
        day_texts = rec.get("texts", [])

        print(f"[{i}] {ticker} {date} -> label={label}, ret={ret_val:.6f}, ctx_days={len(ctx)}")
        # text_window_dates expectations
        assert 1 <= len(wnd) <= max(1, seq_len), f"text_window size out of range: {len(wnd)}"
        assert wnd[-1] == date, f"last window date should be current date, got {wnd[-1]} != {date}"
        assert wnd == sorted(wnd), "text_window_dates must be increasing"

        # Cross-check same-day texts via public loader
        same_day = load_texts_for_day(
            dataset_name=dataset_name,
            ticker=ticker,
            date=date,
            tweet_dir=tweet_dir,
            news_csv_dir=news_csv_dir,
            logger=None,
        )
        print(f"    texts: record={len(day_texts)} vs load_texts_for_day={len(same_day)}")
        assert len(day_texts) == len(same_day), "same-day texts count mismatch"

        # Spot-check additional window dates (if any) for existence
        if len(wnd) > 1:
            prev_date = wnd[-2]
            prev_texts = load_texts_for_day(
                dataset_name=dataset_name,
                ticker=ticker,
                date=prev_date,
                tweet_dir=tweet_dir,
                news_csv_dir=news_csv_dir,
                logger=None,
            )
            print(f"    prev-day texts ({prev_date}): {len(prev_texts)}")

        if dump_records or dump_base:
            payload = {
                "dataset": dataset_name,
                "mode": mode,
                "seq_len": seq_len,
                "record": rec,
            }
            if dump_records:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            if dump_base:
                out_path = os.path.join(dump_base, f"{ticker}_{date}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"    ↳ dumped record to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate common/data/loader APIs.")
    parser.add_argument("--dataset_name", default="SAMPLE", choices=["SAMPLE", "STOCKNET", "CMIN", "SEP"], help="Dataset to validate")
    parser.add_argument("--mode", default="test", choices=["train", "test"], help="Dataset split to inspect")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--limit", type=int, default=4, help="Number of samples to validate")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override DATASETS_DIR")
    parser.add_argument("--splits_dir", type=str, default=None, help="Override splits dir (defaults to repo_root/splits)")
    parser.add_argument("--dump_records", action=argparse.BooleanOptionalAction, default=False, help="Print full get_record outputs for inspected samples")
    parser.add_argument("--dump_dir", type=str, default=None, help="If set, write each inspected record to this directory (organized by dataset/mode)")
    parser.add_argument("--label_strategy", type=str, choices=["legacy", "dual_threshold"], default="legacy")
    parser.add_argument("--neg_threshold", type=float, default=-0.005)
    parser.add_argument("--pos_threshold", type=float, default=0.0055)
    args = parser.parse_args()

    base_data_dir = args.base_data_dir or _resolve_root(os.getenv("DATASETS_DIR"), os.path.join(REPO_ROOT, "datasets"))
    splits_dir = args.splits_dir or os.path.join(REPO_ROOT, "splits")

    paths = resolve_dataset_paths(args.dataset_name, base_data_dir)

    samples = validate_list_trading_days(
        dataset_name=args.dataset_name,
        price_dir=paths.price_dir,
        seq_len=args.seq_len,
        splits_dir=splits_dir,
        mode=args.mode,
        limit=args.limit,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )

    validate_get_record_and_texts(
        dataset_name=args.dataset_name,
        price_dir=paths.price_dir,
        tweet_dir=paths.tweet_dir,
        news_csv_dir=paths.tweet_dir if args.dataset_name.upper() == "CMIN" else None,
        seq_len=args.seq_len,
        mode=args.mode,
        samples=samples,
        dump_records=args.dump_records,
        dump_dir=args.dump_dir,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )

    _print_header("All checks passed ✔")


if __name__ == "__main__":
    main()

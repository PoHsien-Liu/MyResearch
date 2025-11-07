#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Count tweets per ticker (SEP-style folders) and visualize long-tail."
    )
    p.add_argument("root", help="Path to tweets/raw (e.g., /Research/datasets/SEP/tweet/raw)")
    p.add_argument("--out-csv", default="tweet_volume_by_ticker.csv",
                   help="CSV output path; use '-' to print CSV to stdout (default: %(default)s)")
    p.add_argument("--out-png", default=None,
                   help="If set, save bar chart to this PNG instead of showing a window")
    p.add_argument("--max-xlabels", type=int, default=40,
                   help="Max number of x-axis labels to show (default: %(default)s)")
    return p.parse_args()

def count_lines(fp: Path) -> int:
    """Count number of lines in a file. Returns 0 if file can't be read."""
    try:
        with open(fp, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def scan_ticker_dir(ticker_dir: Path) -> Tuple[int, int]:
    """Return (total_tweets, active_days) for one ticker folder."""
    total = 0
    days = 0
    for child in sorted(ticker_dir.iterdir()):
        if child.is_file():
            n = count_lines(child)
            total += n
            days += 1
    return total, days

def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient for non-negative array x."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)

def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[Error] Not a directory: {root}")

    rows = []
    for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
        tdir = root / name
        total, days = scan_ticker_dir(tdir)
        rows.append((name, total, days, (total / days if days > 0 else 0.0)))

    df = pd.DataFrame(rows, columns=["ticker", "tweet_count", "active_days", "avg_per_day"])
    df_sorted = df.sort_values("tweet_count", ascending=False).reset_index(drop=True)

    # ---- Long-tail metrics ----
    total_tweets = int(df_sorted["tweet_count"].sum())
    top1_share = float(df_sorted["tweet_count"].iloc[:1].sum() / total_tweets) if total_tweets > 0 else 0.0
    top10_share = float(df_sorted["tweet_count"].iloc[:10].sum() / total_tweets) if total_tweets > 0 else 0.0
    gini = gini_coefficient(df_sorted["tweet_count"].values) if total_tweets > 0 else 0.0
    median_cnt = int(df_sorted["tweet_count"].median()) if not df_sorted.empty else 0
    mean_cnt = float(df_sorted["tweet_count"].mean()) if not df_sorted.empty else 0.0
    zero_days = int((df_sorted["active_days"] == 0).sum())
    low_volume = int((df_sorted["tweet_count"] <= 50).sum())

    print("=== Long-tail Concentration Summary ===")
    print(f"Tickers: {len(df_sorted)}")
    print(f"Total tweets: {total_tweets}")
    print(f"Top-1 share: {top1_share:.4f}")
    print(f"Top-10 share: {top10_share:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"Median tweets per ticker: {median_cnt}")
    print(f"Mean tweets per ticker: {int(mean_cnt)}")
    print(f"Tickers with zero active days: {zero_days}")
    print(f"Tickers with ≤50 tweets: {low_volume}")

    # ---- Save/print CSV ----
    if args.out_csv == "-":
        # Print to stdout for easy copy-paste
        print("\n# CSV: tweet_volume_by_ticker")
        print(df_sorted.to_csv(index=False).strip())
    else:
        df_sorted.to_csv(args.out_csv, index=False)
        print(f"\nCSV saved to: {args.out_csv}")

    # ---- Plot ----
    plt.figure(figsize=(12, 5))
    x = np.arange(len(df_sorted))
    plt.bar(x, df_sorted["tweet_count"].values)
    plt.title("Tweet Volume by Ticker (sorted)")
    plt.xlabel("Tickers (sorted by tweet count)")
    plt.ylabel("Total tweets")

    # Subsample x labels if many tickers
    tickers = df_sorted["ticker"].tolist()
    n = len(tickers)
    if n <= args.max_xlabels:
        plt.xticks(x, tickers, rotation=90)
    else:
        step = max(1, n // args.max_xlabels)
        sel_idx = np.arange(0, n, step)
        sel_labels = [tickers[i] for i in sel_idx]
        plt.xticks(sel_idx, sel_labels, rotation=90)

    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {args.out_png}")
    else:
        try:
            plt.show()
        except Exception:
            # headless 環境就存一張預設圖
            fallback = "tweet_volume_by_ticker.png"
            plt.savefig(fallback, dpi=200, bbox_inches="tight")
            print(f"(Headless) Figure saved to: {fallback}")

if __name__ == "__main__":
    main()

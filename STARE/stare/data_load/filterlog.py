# data_load/filterlog.py
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

@dataclass
class DropRow:
    reason: str
    ticker_primary: str
    date: str
    id: Optional[str]
    author_id: Optional[str]
    text_raw: str
    text_clean: str
    source_path: str
    url_count: int
    cashtag_count: int
    is_retweet: bool

class FilterLogger:
    """
    蒐集『被過濾掉』的推文，並且產生統計。
    """
    def __init__(self, keep_text: bool = True, truncate: int = 400):
        self.keep_text = keep_text
        self.truncate = truncate
        self.rows: List[DropRow] = []
        self.cnt_total = 0
        self.cnt_kept = 0
        self.by_reason = Counter()
        self.by_day: Dict[Tuple[str, str], Counter] = defaultdict(Counter)  # (ticker, date) -> reason counter

    def inc_read(self):
        self.cnt_total += 1

    def inc_kept(self):
        self.cnt_kept += 1

    def log_drop(self, *, reason: str, ticker_primary: str, date: str,
                 id: Optional[str], author_id: Optional[str],
                 text_raw: str, text_clean: str, source_path: str,
                 url_count: int, cashtag_count: int, is_retweet: bool):
        self.by_reason[reason] += 1
        self.by_day[(ticker_primary, date)][reason] += 1

        if not self.keep_text:
            text_raw = ""
            text_clean = ""
        else:
            if self.truncate and len(text_raw) > self.truncate:
                text_raw = text_raw[:self.truncate]
            if self.truncate and len(text_clean) > self.truncate:
                text_clean = text_clean[:self.truncate]

        self.rows.append(DropRow(
            reason=reason, ticker_primary=ticker_primary, date=date,
            id=id, author_id=author_id, text_raw=text_raw, text_clean=text_clean,
            source_path=source_path, url_count=url_count, cashtag_count=cashtag_count,
            is_retweet=is_retweet
        ))

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        # 準備 DataFrame
        if self.rows:
            df = pd.DataFrame([r.__dict__ for r in self.rows])
        else:
            df = pd.DataFrame(columns=[
                "reason","ticker_primary","date","id","author_id","text_raw","text_clean",
                "source_path","url_count","cashtag_count","is_retweet"
            ])

        # 轉型
        for col in ("url_count","cashtag_count"):
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype("int64")
        df["is_retweet"] = df.get("is_retweet", False)
        df["is_retweet"] = df["is_retweet"].fillna(False).astype("bool")
        for col in ("reason","ticker_primary","date","id","author_id","text_raw","text_clean","source_path"):
            if col in df.columns:
                df[col] = df[col].astype("string").fillna("")

        arrays = [
            pa.array(df["reason"].astype(str).tolist(),         pa.string()),
            pa.array(df["ticker_primary"].astype(str).tolist(), pa.string()),
            pa.array(df["date"].astype(str).tolist(),           pa.string()),
            pa.array(df["id"].astype(str).tolist(),             pa.string()),
            pa.array(df["author_id"].astype(str).tolist(),      pa.string()),
            pa.array(df["text_raw"].astype(str).tolist(),       pa.string()),
            pa.array(df["text_clean"].astype(str).tolist(),     pa.string()),
            pa.array(df["source_path"].astype(str).tolist(),    pa.string()),
            pa.array(df["url_count"].to_numpy("int64"),         pa.int64()),
            pa.array(df["cashtag_count"].to_numpy("int64"),     pa.int64()),
            pa.array(df["is_retweet"].to_numpy(bool),           pa.bool_()),
        ]
        names = ["reason","ticker_primary","date","id","author_id","text_raw","text_clean",
                "source_path","url_count","cashtag_count","is_retweet"]
        pq.write_table(pa.Table.from_arrays(arrays, names=names), os.path.join(out_dir, "dropped.parquet"))

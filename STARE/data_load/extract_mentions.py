"""Extract ticker mentions and related metadata from cleaned texts."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from STARE.utils.logger import setup_logger
from STARE.utils.paths import dataset_paths, ensure_dir, indices_dir
from STARE.utils.seed import set_seed

LOGGER = logging.getLogger("stare.extract_mentions")
CASHTAG_PATTERN = re.compile(r"\$[A-Za-z]{1,6}")
TICKER_CHARS_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")
SUFFIX_PATTERN = re.compile(
    r"(,?\s+(Co\.?|Corp\.?|Corporation|Inc\.?|Incorporated|Ltd\.?|Limited|PLC|Company))+$",
    re.IGNORECASE,
)
# Canonical ticker mapping：同公司多股別/舊代碼合併（對應資料集標準 ticker）
CANONICAL_MAP = {
    "GOOGL": "GOOG",
    "FB": "FB",  # 仍保留 FB 為主代碼
    "META": "FB",  # Meta 品牌映射回 dataset 的 FB 代碼
    "BRK-B": "BRK-A",
    "BRK.B": "BRK-A",
}

# 排除 price/raw 存在但不應處理的優先股代碼
EXCLUDED_TICKERS = {"C-PJ", "CTA-PB", "SPG-PJ", "WFC-PL"}

# 手動補充常見品牌/別名（新聞標題常用全名而非 cashtag）
BRAND_ALIAS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("ALPHABET", "GOOGL"),
    ("GOOGLE", "GOOGL"),
    ("MICROSOFT", "MSFT"),
    ("APPLE", "AAPL"),
    ("AMAZON", "AMZN"),
    ("TESLA", "TSLA"),
    ("NETFLIX", "NFLX"),
    ("META", "META"),
    ("FACEBOOK", "META"),
    ("NVIDIA", "NVDA"),
    ("AMD", "AMD"),
    ("ADVANCED MICRO DEVICES", "AMD"),
    ("BROADCOM", "AVGO"),
    ("TAIWAN SEMICONDUCTOR", "TSM"),
    ("TSMC", "TSM"),
    ("BERKSHIRE HATHAWAY", "BRK-A"),
    ("JPMORGAN", "JPM"),
    ("JPMORGAN CHASE", "JPM"),
    ("WALMART", "WMT"),
    ("COCA COLA", "KO"),
    ("PEPSICO", "PEP"),
    ("CHEVRON", "CVX"),
    ("EXXON MOBIL", "XOM"),
    ("EXXON", "XOM"),
    ("INTEL", "INTC"),
    ("ORACLE", "ORCL"),
    ("SALESFORCE", "CRM"),
    ("ADOBE", "ADBE"),
    ("NETEASE", "NTES"),
    ("ALIBABA", "BABA"),
    ("BABA", "BABA"),
    ("TENCENT", "0700.HK"),  # 留作參考，即便不在 CMIN-US 也可忽略
)


@dataclass
class MentionStats:
    cashtag_count: int
    url_count: int
    hashtag_count: int


def run_extract_mentions(args) -> None:
    """CLI hook for the extract_mentions task."""
    set_seed(args.seed)
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "extract_mentions.log"
    logger = setup_logger("stare.extract_mentions", log_file=log_file)

    cleaned_path = output_dir / "cleaned.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned.parquet not found at {cleaned_path}")

    logger.info("Loading cleaned data from %s", cleaned_path)
    df = pd.read_parquet(cleaned_path)
    if df.empty:
        raise RuntimeError("Cleaned dataframe is empty; aborting mention extraction.")

    allowed_tickers = _load_allowed_tickers(args.dataset_name)
    if allowed_tickers:
        df = df[df["source_ticker"].str.upper().isin(allowed_tickers)]
        df = df.reset_index(drop=True)
        logger.info("Filtered cleaned rows to allowed tickers: %d rows", len(df))

    # 構建公司名稱與 ticker 的對應表
    alias_map = build_alias_map(args.dataset_name, allowed_tickers)
    alias_patterns = _compile_alias_patterns(alias_map)
    logger.info(
        "Loaded alias map for %s: %d tickers, %d alias patterns",
        args.dataset_name,
        len(alias_map),
        len(alias_patterns),
    )

    logger.info("Extracting mentions for %d rows", len(df))
    mentions = df.apply(
        lambda row: _extract_row_mentions(row, alias_patterns),
        axis=1,
    )
    df["mentioned_tickers"] = [m[0] for m in mentions]
    df["cashtag_count"] = [m[1].cashtag_count for m in mentions]
    df["url_count"] = [m[1].url_count for m in mentions]
    df["hashtag_count"] = [m[1].hashtag_count for m in mentions]

    out_path = output_dir / "cleaned_with_mentions.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(
        "Saved cleaned_with_mentions to %s (rows=%d)", out_path, len(df)
    )


def build_alias_map(dataset_name: str, allowed_tickers: Optional[set[str]] = None) -> Dict[str, List[str]]:
    """Build ticker -> alias list mapping from dataset metadata and brand hints."""
    aliases: Dict[str, List[str]] = {}

    # 手動品牌別名
    for alias, ticker in BRAND_ALIAS_PAIRS:
        canon = canonicalize_ticker(ticker)
        if not canon:
            continue
        if allowed_tickers and canon not in allowed_tickers:
            continue
        aliases.setdefault(canon, []).append(alias.upper())

    # 嘗試從 dataset 的文本路徑取得公司全名
    try:
        paths = dataset_paths(dataset_name)
        text_root = paths.text_path
        news_raw = Path(text_root) / "raw"
        if news_raw.exists():
            for csv_path in sorted(news_raw.glob("*.csv")):
                ticker = canonicalize_ticker(csv_path.stem)
                if not ticker:
                    continue
                if allowed_tickers and ticker not in allowed_tickers:
                    continue
                # 先確保 ticker 本身被收錄
                aliases.setdefault(ticker, []).append(ticker)
                try:
                    df = pd.read_csv(csv_path, nrows=50, engine="python", sep=None)
                except Exception:
                    continue
                if "name" in df.columns:
                    name = _first_non_null(df["name"])
                    if name:
                        cleaned = _normalize_company_name(name)
                        if cleaned:
                            aliases.setdefault(ticker, []).append(cleaned)
    except Exception as err:  # pragma: no cover -防禦
        LOGGER.warning("Failed to build alias map from dataset %s: %s", dataset_name, err)

    # 去重
    for ticker, names in list(aliases.items()):
        dedup = _dedup_preserve_order([n for n in names if n])
        aliases[ticker] = dedup
    return aliases


def _compile_alias_patterns(alias_map: Dict[str, List[str]]) -> List[Tuple[re.Pattern, str]]:
    patterns: List[Tuple[re.Pattern, str]] = []
    seen = set()
    for ticker, names in alias_map.items():
        for name in names:
            key = (name.upper(), ticker)
            if key in seen:
                continue
            seen.add(key)
            # 僅接受長度>=2的詞，避免太短的假陽性
            if len(name.replace(" ", "")) < 2:
                continue
            escaped = re.escape(name)
            regex = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
            patterns.append((regex, ticker))
    return patterns


def _extract_row_mentions(row: pd.Series, alias_patterns: List[Tuple[re.Pattern, str]]) -> tuple[list[str], MentionStats]:
    text = str(row.get("text", "") or "")
    source_ticker = canonicalize_ticker(row.get("source_ticker"))
    urls = _parse_iterable(row.get("urls"))
    hashtags = _parse_iterable(row.get("hashtags"))

    cashtags = [canonicalize_ticker(tok) for tok in CASHTAG_PATTERN.findall(text)]
    cashtags = [c for c in cashtags if c]

    hashtag_tickers = []
    for tag in hashtags:
        token = tag.lstrip("#").upper()
        if _looks_like_ticker(token):
            ht = canonicalize_ticker(token)
            if ht:
                hashtag_tickers.append(ht)

    alias_hits = _match_aliases_in_text(text, alias_patterns)

    mentioned: List[str] = []
    if source_ticker:
        mentioned.append(source_ticker)
    mentioned.extend(cashtags)
    mentioned.extend(hashtag_tickers)
    mentioned.extend(alias_hits)
    mentioned = _dedup_preserve_order([m for m in mentioned if m])

    stats = MentionStats(
        cashtag_count=len(cashtags),
        url_count=len(urls),
        hashtag_count=len(hashtags),
    )
    return mentioned, stats


def _dedup_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_iterable(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    if isinstance(value, tuple):
        return [str(v) for v in value if v not in (None, "")]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return text.split()
    return [str(value)]


def _normalize_ticker(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    text = text.lstrip("$")
    if not text or not _looks_like_ticker(text):
        return None
    return text


def _looks_like_ticker(token: str) -> bool:
    return bool(token) and bool(TICKER_CHARS_PATTERN.match(token))

def _match_aliases_in_text(text: str, patterns: List[Tuple[re.Pattern, str]]) -> List[str]:
    if not text:
        return []
    mentions: List[str] = []
    for regex, ticker in patterns:
        if regex.search(text):
            mentions.append(ticker)
    return mentions


def _first_non_null(series: pd.Series) -> Optional[str]:
    for val in series:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def _normalize_company_name(name: str) -> str:
    if not name:
        return ""
    txt = name.strip()
    txt = SUFFIX_PATTERN.sub("", txt)
    txt = re.sub(r"[.,]", " ", txt)
    txt = re.sub(r"\\s+", " ", txt)
    return txt.strip().upper()


def canonicalize_ticker(ticker: Optional[str]) -> Optional[str]:
    norm = _normalize_ticker(ticker)
    if not norm:
        return None
    if norm in EXCLUDED_TICKERS:
        return None
    return CANONICAL_MAP.get(norm, norm)


def _load_allowed_tickers(dataset_name: str) -> set[str]:
    try:
        paths = dataset_paths(dataset_name)
    except Exception:
        return set()
    price_dir = Path(paths.price_path) / "raw"
    if not price_dir.exists():
        price_dir = Path(paths.price_path)
    allowed = {p.stem.upper() for p in price_dir.glob("*.csv")}
    allowed = {t for t in allowed if t not in EXCLUDED_TICKERS}
    return allowed


__all__ = ["run_extract_mentions"]

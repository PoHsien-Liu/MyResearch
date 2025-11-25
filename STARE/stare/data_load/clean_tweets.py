"""Rule-based cleaning pipeline for STARE textual corpora."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from STARE.stare.utils.logger import setup_logger
from STARE.stare.utils.paths import dataset_paths, ensure_dir, indices_dir
from STARE.stare.utils.seed import set_seed

LOGGER = logging.getLogger("stare.clean")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]"
)
TICKER_PATTERN = re.compile(r"\$[A-Za-z]{1,6}")
URL_TOKEN_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
ALLOWED_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".txt", ""}


@dataclass
class RuleConfig:
    min_tokens: int = 5
    enable_short_text: bool = True
    enable_ticker_emoji: bool = True
    enable_repeated_emoji: bool = True
    enable_tag_ratio: bool = True
    enable_retweet: bool = True
    repeat_limit: int = 4
    tag_ratio: float = 0.5


def run_clean(args) -> None:
    """CLI hook for the clean task."""
    set_seed(args.seed)
    dataset_info = dataset_paths(args.dataset_name)
    output_dir = ensure_dir(indices_dir(args.dataset_name, args.embed_model))
    log_file = output_dir / "clean.log"
    logger = setup_logger("stare.clean", log_file=log_file)

    logger.info("Loading dataset %s from %s", args.dataset_name, dataset_info.text_path)
    raw_df = load_text_dataframe(args.dataset_name, dataset_info.text_path)
    if raw_df.empty:
        raise RuntimeError("No textual samples were loaded for cleaning.")
    logger.info("Loaded %d raw rows", len(raw_df))

    config = RuleConfig(min_tokens=args.min_tokens)
    cleaned_df, dropped_df = clean_dataframe(
        raw_df,
        dataset_name=args.dataset_name,
        rule_config=config,
        enable_llm_filter=args.enable_llm_filter,
    )

    cleaned_path = output_dir / "cleaned.parquet"
    dropped_path = output_dir / "dropped.parquet"
    cleaned_df.to_parquet(cleaned_path, index=False)
    dropped_df.to_parquet(dropped_path, index=False)
    logger.info(
        "Saved %d cleaned rows to %s (dropped %d rows)",
        len(cleaned_df),
        cleaned_path,
        len(dropped_df),
    )


def load_text_dataframe(dataset_name: str, text_root: Path) -> pd.DataFrame:
    """Load textual records from dataset-specific storage."""
    if not text_root.exists():
        raise FileNotFoundError(f"Text path does not exist: {text_root}")

    files = _collect_text_files(text_root)
    if not files:
        raise RuntimeError(f"No textual files found under {text_root}")

    frames: List[pd.DataFrame] = []
    for idx, file_path in enumerate(files, start=1):
        try:
            frame = _read_text_file(file_path)
        except Exception as err:  # pragma: no cover - defensive log
            LOGGER.warning("Failed to read %s: %s", file_path, err)
            continue
        if frame.empty:
            continue
        frame["source_path"] = str(file_path)
        ticker_hint = _infer_ticker_from_path(file_path)
        if ticker_hint and "_ticker_hint" not in frame.columns:
            frame["_ticker_hint"] = ticker_hint
        frames.append(frame)
        if idx % 50 == 0:
            LOGGER.info("Loaded %d/%d files", idx, len(files))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_text_files(text_root: Path) -> List[Path]:
    if text_root.is_file():
        return [text_root]

    candidates: List[Path] = []
    priority_dirs = [text_root / "raw", text_root / "preprocessed"]
    for candidate in priority_dirs:
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        candidates.append(text_root)

    files: List[Path] = []
    for base in candidates:
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            if any(part == ".git" for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix in ALLOWED_SUFFIXES:
                files.append(path)
        if files:
            break
    return files


def _read_text_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".json", ".jsonl", ".txt", ""}:
        return _read_json_records(path)
    return pd.DataFrame()


def _read_json_records(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return pd.DataFrame()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = [payload]
        if isinstance(payload, list):
            return pd.DataFrame(payload)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return pd.DataFrame(records)
    return pd.DataFrame()


def clean_dataframe(
    raw_df: pd.DataFrame,
    dataset_name: str,
    rule_config: RuleConfig,
    enable_llm_filter: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_rows = []
    dropped_rows = []

    for _, row in raw_df.iterrows():
        record, tokens = _normalize_row(row, dataset_name)
        if not record["text"]:
            record["drop_reason"] = "empty_text"
            dropped_rows.append(record)
            continue
        if not record["source_ticker"]:
            record["drop_reason"] = "missing_ticker"
            dropped_rows.append(record)
            continue
        if not record["date"]:
            record["drop_reason"] = "missing_date"
            dropped_rows.append(record)
            continue

        reason = filter_tweet_rule_based(record, tokens, rule_config)
        if reason:
            record["drop_reason"] = reason
            dropped_rows.append(record)
            continue

        if enable_llm_filter and llm_filter_tweet(record):
            record["drop_reason"] = "llm_filter_reject"
            dropped_rows.append(record)
            continue

        cleaned_rows.append(record)

    cleaned_df = pd.DataFrame(cleaned_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    return cleaned_df, dropped_df


def _normalize_row(row: pd.Series, dataset_name: str) -> Tuple[dict, List[str]]:
    data = row.to_dict()
    text_value = _extract_first(data, ["text", "title", "body", "content", "summary"])
    text = _stringify_text(text_value)
    date_value = _extract_first(data, ["date", "created_at", "time"])
    date = _normalize_date(date_value)
    ticker_value = _extract_first(
        data,
        [
            "source_ticker",
            "ticker",
            "symbol",
            "company",
            "cashtag",
            "stock",
            "_ticker_hint",
        ],
    )
    ticker = _normalize_ticker(ticker_value)
    is_retweet = _normalize_bool(data.get("is_retweet"))
    if is_retweet is False:
        is_retweet = text.upper().startswith("RT ") if text else False

    hashtags = _parse_iterable(data.get("hashtags"))
    urls = _collect_urls(data)
    tokens = simple_tokenize(text)

    record = {
        "dataset": dataset_name.upper(),
        "text": text,
        "date": date,
        "source_ticker": ticker,
        "is_retweet": bool(is_retweet),
        "hashtags": hashtags if hashtags else _extract_hashtags_from_text(text),
        "urls": urls,
        "source_path": data.get("source_path"),
        "raw_id": _extract_first(data, ["id", "tweet_id", "news_id"]),
        "created_at": data.get("created_at"),
        "token_count": len(tokens),
    }
    return record, tokens


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"https?://\S+|\$[A-Za-z]{1,6}|[#@]?\w+|[^\s]", text)
    return [tok for tok in tokens if tok.strip()]


def filter_tweet_rule_based(record: dict, tokens: Sequence[str], config: RuleConfig) -> Optional[str]:
    if config.enable_short_text and is_short(tokens, config.min_tokens):
        return "short_text"
    if config.enable_ticker_emoji and only_ticker_emoji(record["text"]):
        return "ticker_emoji"
    if config.enable_repeated_emoji and too_many_repeated_emoji(record["text"], config.repeat_limit):
        return "repeated_emoji"
    if config.enable_tag_ratio and too_many_tags_urls(tokens, config.tag_ratio):
        return "tag_url_ratio"
    if config.enable_retweet and is_pure_retweet(record):
        return "retweet"
    return None


def is_short(tokens: Sequence[str], min_tokens: int) -> bool:
    return len(tokens) < min_tokens


def only_ticker_emoji(text: str) -> bool:
    if not text:
        return False
    stripped = TICKER_PATTERN.sub("", text)
    stripped = EMOJI_PATTERN.sub("", stripped)
    stripped = re.sub(r"[\s.,!?:;()-]+", "", stripped)
    return not stripped


def too_many_repeated_emoji(text: str, repeat_limit: int) -> bool:
    if not text:
        return False
    counter = Counter(ch for ch in text if EMOJI_PATTERN.match(ch))
    if not counter:
        return False
    return max(counter.values()) >= repeat_limit


def too_many_tags_urls(tokens: Sequence[str], tag_ratio: float) -> bool:
    if not tokens:
        return False
    tag_tokens = [tok for tok in tokens if tok.startswith("#") or tok.startswith("http")]
    return (len(tag_tokens) / len(tokens)) > tag_ratio


def is_pure_retweet(record: dict) -> bool:
    return bool(record.get("is_retweet"))


_LLM_WARNING_EMITTED = False


def llm_filter_tweet(record: dict) -> bool:
    global _LLM_WARNING_EMITTED
    if not _LLM_WARNING_EMITTED:
        LOGGER.warning("LLM filter requested but adapter integration is not implemented yet; skipping.")
        _LLM_WARNING_EMITTED = True
    return False


def _extract_first(data: dict, keys: Iterable[str]) -> Optional[object]:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            value = data[key]
            if isinstance(value, float) and math.isnan(value):
                continue
            return value
    return None


def _stringify_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None).strip()
    return str(value).strip()


def _normalize_date(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=False, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    return ts.strftime("%Y-%m-%d")


def _normalize_ticker(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "")
    text = text.upper()
    if not re.match(r"^[A-Z.\-]{1,10}$", text):
        return None
    return text


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes"}
    return False


def _parse_iterable(value: object) -> List[str]:
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
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(v) for v in payload if v is not None]
        except json.JSONDecodeError:
            pass
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return text.split()
    return [str(value)]


def _collect_urls(data: dict) -> List[str]:
    urls = []
    for key in ["urls", "links", "link", "url"]:
        urls.extend(_parse_iterable(data.get(key)))
    dedup = []
    seen = set()
    for url in urls:
        url = url.strip()
        if not url:
            continue
        if url in seen:
            continue
        dedup.append(url)
        seen.add(url)
    if not dedup:
        text = str(data.get("text", ""))
        dedup = URL_TOKEN_PATTERN.findall(text)
    return dedup


def _extract_hashtags_from_text(text: str) -> List[str]:
    if not text:
        return []
    tags = re.findall(r"#\w+", text)
    return tags


def _infer_ticker_from_path(path: Path) -> Optional[str]:
    stem = path.stem.upper()
    if re.match(r"^[A-Z]{1,6}$", stem):
        return stem
    for part in reversed(path.parts):
        part = part.upper()
        if re.match(r"^[A-Z]{1,6}$", part):
            return part
    return None


__all__ = ["run_clean", "clean_dataframe", "load_text_dataframe", "RuleConfig"]

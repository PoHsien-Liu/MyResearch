# data_load/tweet_reader.py
import os, ujson, json, regex as re
from dataclasses import dataclass, asdict
from typing import List, Iterable, Tuple, Optional
from data_load.filterlog import FilterLogger, DropRow

CASHTAG_RE = re.compile(r'(?<![A-Za-z0-9])\$[A-Z]{1,5}(?![A-Za-z0-9])')
URL_RE = re.compile(r'https?://\S+')
AT_RE  = re.compile(r'@\w+')
WS_RE  = re.compile(r'\s+')

@dataclass
class TweetDoc:
    text: str
    ticker_primary: str
    tickers: List[str]
    id: Optional[str]
    author_id: Optional[str]
    created_at: Optional[str]
    date: str
    source_path: str
    is_retweet: bool
    url_count: int
    cashtag_count: int

def _normalize_text(text: str) -> str:
    t = URL_RE.sub(' ', text)
    t = WS_RE.sub(' ', t).strip()
    return t

def _extract_cashtags(text: str) -> List[str]:
    return list({m.group(0)[1:] for m in CASHTAG_RE.finditer(text)})

def is_low_info(text: str, min_chars: int) -> bool:
    core = re.sub(r'[^A-Za-z0-9\$ ]+', ' ', text)
    core = WS_RE.sub(' ', core).strip()
    return len(core) < min_chars

def is_cashtag_only(text: str) -> bool:
    x = CASHTAG_RE.sub(' ', text)
    x = AT_RE.sub(' ', x)
    x = URL_RE.sub(' ', x)
    return len(x.strip()) == 0

def looks_like_template_rt(text: str) -> bool:
    if text.startswith("RT "):
        core = _normalize_text(text[3:])
        core_wo = AT_RE.sub(' ', URL_RE.sub(' ', CASHTAG_RE.sub(' ', core))).strip()
        return len(core_wo) < 10
    return False

class TweetReader:
    def __init__(self, tweet_dir: str, min_chars: int = 20, drop_retweet: bool = True, filter_logger: Optional[FilterLogger] = None):
        self.tweet_dir = tweet_dir
        self.min_chars = min_chars
        self.drop_retweet = drop_retweet
        self.filter_logger = filter_logger or FilterLogger(keep_text=True, truncate=400)

    def _iter_paths(self) -> Iterable[Tuple[str, str, str]]:
        # tweet/raw/{ticker}/{date}
        for ticker in sorted(os.listdir(self.tweet_dir)):
            tdir = os.path.join(self.tweet_dir, ticker)
            if not os.path.isdir(tdir): continue
            for date_name in sorted(os.listdir(tdir)):
                fpath = os.path.join(tdir, date_name)
                if os.path.isfile(fpath):
                    yield ticker, date_name, fpath

    def iter_docs(self) -> Iterable[TweetDoc]:
        for ticker, date_name, fpath in self._iter_paths():
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: 
                        continue
                    # 讀一則 → 總數 +1
                    self.filter_logger.inc_read()

                    try: tw = ujson.loads(line)
                    except Exception:
                        try: tw = json.loads(line)
                        except Exception:
                            # 非法 json 也記一下
                            self.filter_logger.log_drop(
                                reason="bad_json", ticker_primary=ticker, date=date_name,
                                id=None, author_id=None, text_raw=line, text_clean="",
                                source_path=fpath, url_count=0, cashtag_count=0, is_retweet=False
                            )
                            continue

                    raw_text = tw.get("text", "") or ""
                    if not raw_text:
                        self.filter_logger.log_drop(
                            reason="empty_text", ticker_primary=ticker, date=date_name,
                            id=tw.get("id"), author_id=tw.get("author_id"),
                            text_raw="", text_clean="", source_path=fpath,
                            url_count=0, cashtag_count=0, is_retweet=False
                        )
                        continue

                    is_rt = raw_text.startswith("RT ")
                    txt = _normalize_text(raw_text)
                    cashtags = _extract_cashtags(txt)
                    created_at = tw.get("created_at")
                    meta_date = created_at[:10] if created_at else date_name

                    # 規則：模板 RT
                    if self.drop_retweet and is_rt and looks_like_template_rt(raw_text):
                        self.filter_logger.log_drop(
                            reason="template_rt", ticker_primary=ticker, date=meta_date,
                            id=tw.get("id"), author_id=tw.get("author_id"),
                            text_raw=raw_text, text_clean=txt, source_path=fpath,
                            url_count=len(URL_RE.findall(raw_text)),
                            cashtag_count=len(cashtags), is_retweet=True
                        )
                        continue

                    # 規則：純 cashtag/連結
                    if is_cashtag_only(txt):
                        self.filter_logger.log_drop(
                            reason="cashtag_only", ticker_primary=ticker, date=meta_date,
                            id=tw.get("id"), author_id=tw.get("author_id"),
                            text_raw=raw_text, text_clean=txt, source_path=fpath,
                            url_count=len(URL_RE.findall(raw_text)),
                            cashtag_count=len(cashtags), is_retweet=is_rt
                        )
                        continue

                    # 規則：資訊過短
                    if is_low_info(txt, self.min_chars):
                        self.filter_logger.log_drop(
                            reason="too_short", ticker_primary=ticker, date=meta_date,
                            id=tw.get("id"), author_id=tw.get("author_id"),
                            text_raw=raw_text, text_clean=txt, source_path=fpath,
                            url_count=len(URL_RE.findall(raw_text)),
                            cashtag_count=len(cashtags), is_retweet=is_rt
                        )
                        continue

                    # 能走到這裡 → 保留下來
                    self.filter_logger.inc_kept()

                    yield TweetDoc(
                        text=txt,
                        ticker_primary=ticker,
                        tickers=list({ticker} | set(cashtags)),
                        id=tw.get("id"),
                        author_id=tw.get("author_id"),
                        created_at=created_at,
                        date=meta_date,
                        source_path=fpath,
                        is_retweet=is_rt,
                        url_count=len(URL_RE.findall(raw_text)),
                        cashtag_count=len(cashtags),
                    )
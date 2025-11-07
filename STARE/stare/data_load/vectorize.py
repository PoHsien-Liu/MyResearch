# data_load/vectorize.py
import os, numpy as np, pandas as pd
from typing import Iterable, Optional, List, Dict
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import faiss
except ImportError:
    faiss = None

class EmbeddingIndex:
    def __init__(self, model_name: str, batch_size: int = 128, device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.embeddings = None
        self.meta_df = None
        self.index = None

    def build(self, texts: List[str], metas: List[Dict]):
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            vec = self.model.encode(batch, batch_size=self.batch_size,
                                    show_progress_bar=False, normalize_embeddings=True)
            embs.append(vec.astype(np.float32))
        self.embeddings = np.vstack(embs)
        self.meta_df = pd.DataFrame(metas)
        if faiss is not None:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # 1) embeddings 直接存
        np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)

        # 2) 準備 metadata → 嚴格轉型
        df = self.meta_df.copy()

        # 基本文字欄位 → string
        for col in ("text", "ticker_primary", "date", "id", "author_id", "source_path"):
            if col in df.columns:
                df[col] = df[col].astype("string").fillna("")

        # 數值欄位 → numpy int64
        for col in ("url_count", "cashtag_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        # 布林欄位 → numpy bool
        if "is_retweet" in df.columns:
            df["is_retweet"] = df["is_retweet"].fillna(False).astype("bool")

        # tickers → list[str]
        def _to_str_list(x):
            if x is None:
                return []
            # 有時候會是 tuple / set / numpy array
            try:
                seq = list(x)
            except Exception:
                return []
            return [str(t) for t in seq]

        df["tickers"] = df["tickers"].apply(_to_str_list)

        # 3) 逐欄建 Arrow Array（避免 from_pandas 的自動推斷）
        arr_text          = pa.array(df["text"].astype(str).tolist(),               type=pa.string())
        arr_tickers       = pa.array(df["tickers"].tolist(),                        type=pa.list_(pa.string()))
        arr_ticker_primary= pa.array(df["ticker_primary"].astype(str).tolist(),     type=pa.string())
        arr_date          = pa.array(df["date"].astype(str).tolist(),               type=pa.string())
        arr_id            = pa.array(df["id"].astype(str).tolist(),                 type=pa.string())
        arr_author_id     = pa.array(df["author_id"].astype(str).tolist(),          type=pa.string())
        arr_source_path   = pa.array(df["source_path"].astype(str).tolist(),        type=pa.string())

        # 🔧 關鍵修正：布林用 list，不走 ndarray 分支
        arr_is_retweet    = pa.array([bool(x) for x in df["is_retweet"].tolist()],  type=pa.bool_())

        # 數值欄：明確轉成 numpy int64（走 ndarray 分支 OK）
        arr_url_count     = pa.array(np.asarray(df["url_count"], dtype=np.int64),   type=pa.int64())
        arr_cashtag_count = pa.array(np.asarray(df["cashtag_count"], dtype=np.int64), type=pa.int64())

        table = pa.Table.from_arrays(
            [arr_text, arr_tickers, arr_ticker_primary, arr_date, arr_id, arr_author_id,
            arr_source_path, arr_is_retweet, arr_url_count, arr_cashtag_count],
            names=["text","tickers","ticker_primary","date","id","author_id",
                "source_path","is_retweet","url_count","cashtag_count"]
        )
        pq.write_table(table, os.path.join(out_dir, "metadata.parquet"))

        # 4) FAISS 索引（若存在）
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
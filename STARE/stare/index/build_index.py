# index/build_index.py
import os
from typing import Optional
from tqdm import tqdm
from data_load.tweet_reader import TweetReader
from data_load.vectorize import EmbeddingIndex
from data_load.filterlog import FilterLogger

def ensure_index_built(tweet_dir: str, out_dir: str, embed_model: str,
                       rebuild: bool, device: Optional[str], logger):
    need = rebuild or (not os.path.exists(os.path.join(out_dir, "embeddings.npy")))
    if not need:
        logger.info(f"[Index] found existing at {out_dir}, skip build.")
        return

    flog = FilterLogger(keep_text=True, truncate=400)
    reader = TweetReader(tweet_dir=tweet_dir, min_chars=20, drop_retweet=True, filter_logger=flog)

    texts, metas = [], []
    for doc in tqdm(reader.iter_docs(), desc="Collecting docs"):
        texts.append(doc.text)
        metas.append({
            "text": doc.text,
            "tickers": doc.tickers,
            "ticker_primary": doc.ticker_primary,
            "date": doc.date,
            "id": doc.id,
            "author_id": doc.author_id,
            "source_path": doc.source_path,
            "is_retweet": doc.is_retweet,
            "url_count": doc.url_count,
            "cashtag_count": doc.cashtag_count,
        })

    idx = EmbeddingIndex(model_name=embed_model, batch_size=256, device=device)
    idx.build(texts=texts, metas=metas)
    idx.save(out_dir)

    # 重要：把被丟掉的樣本與統計存起來
    flog.save(out_dir)
    logger.info(f"[Index] built. Kept={flog.cnt_kept} / Total={flog.cnt_total} "
                f"(drop={flog.cnt_total - flog.cnt_kept}; shrink={100*(flog.cnt_total-flog.cnt_kept)/max(1,flog.cnt_total):.1f}%)")

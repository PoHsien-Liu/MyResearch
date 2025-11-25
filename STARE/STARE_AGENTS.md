# STARE / AGENTS.md
STARE（Stock Price Prediction with Text‑Augmented Retrieval for Explainability）— 資料清洗與向量索引建置說明

> 本檔案說明 STARE 方法目前要實作的部分，聚焦在「推文／新聞資料清洗」與「向量資料庫（RAG 索引）建置」。
> **目前實作優先順序：請以 CMIN-US 資料集為主，先完成 minimal pre-processing → 向量化 → 索引建置與後續 RAG 模組。**
> 對 Twitter 資料集（StockNet / SEP）的較重清洗可以在 CMIN-US pipeline 穩定後再補。
---

## 1. 目標與範圍

- **目標**
  - 對 `StockNet`, `SEP (SN2)`, `CMIN-US` 三個資料來源的金融文本（Tweets / News title）進行：
    1. 資料清洗：去除無資訊含量或雜訊文本。
    2. 向量化：使用金融語意相關的開源模型進行 embedding。
    3. 建索引：建立可支援時間過濾與 cross-firm reasoning 的向量索引（FAISS 為主）。
- **產物**
  - 乾淨的文本檔（含 metadata）。
  - 向量檔與 metadata（對齊每一條文本）。
  - FAISS 索引檔，供 STARE RAG pipeline 檢索使用。
  - 被過濾掉文本的記錄檔（便於之後調整規則與做 ablation）。
> 註：CMIN-US 的文本為 Yahoo Finance 等來源的**新聞標題**而非全文，原始設計就是為了降低雜訊並提升效率，因此相較 Twitter corpus 已相對乾淨，適合作為第一個完整 RAG pipeline 的實作對象。:contentReference[oaicite:0]{index=0}
---

## 2. 目錄與檔案結構（STARE 部分）

- `STARE/`
  - `configs/dataset.py`：資料集路徑與欄位映射（已在根 AGENTS.md 中定義）。
  - `data_load/`
    - `clean_tweets.py`：推文／新聞清洗與 rule-based 過濾。
    - `extract_mentions.py`：解析來源公司與提及公司、統計 cashtag / URL 等 metadata。
  - `index/`
    - `embed_texts.py`：向量化模組（支援多種開源 embedding 模型）。
    - `build_index.py`：向量索引建置（FAISS + metadata 存檔）。
  - `utils/`
    - `logger.py`：統一日誌。
    - `seed.py`：隨機種子設定。
    - `paths.py`：處理 `DATASETS_DIR`、`OUTPUTS_DIR` 與 dataset 名稱／embed_model slug。
  - `main.py`：CLI 入口，統一任務管理
    - `--task {clean, extract_mentions, embed, build_index, build_index_pipeline}`

---

## 3. 資料來源與輸入格式

### 3.1 StockNet / SEP (SN2)

- **型態**：Tweets
- **主要欄位（raw）**（實際依 dataset 而略有差異）：
  - `text`：推文全文。
  - `created_at`：推文時間。
  - `ticker` / `symbol`：來源公司（source ticker）。
  - `is_retweet`：是否為轉推。
  - `hashtags`：hashtag 列表。
  - `urls`：網址列表。
- **期望經過 STARE 清洗後的欄位（cleaned）**：
  - `text`：保留文本。
  - `date`：標準化日期（YYYY-MM-DD）。
  - `source_ticker`：來源公司。
  - `is_retweet`
  - `hashtags`（可選）
  - `urls`（可選）
> **備註**：推文清洗是「高噪音對象」，會使用較多 rule-based + optional LLM filter。實作優先度可以排在 CMIN-US pipeline 完成之後。

### 3.2 CMIN-US

- **型態**：News title
- **主要欄位**：
  - `title`：新聞標題（作為文本）。
  - `date`：發佈日期。
  - `ticker`：對應公司。
- **期望 cleaned 欄位**：
  - `text`：直接使用 `title`。
  - `date`
  - `source_ticker`

---

## 4. 資料清洗設計（`data_load/clean_tweets.py`）

### 4.1 Rule‑based 規則集合
> **注意：此區段實作可以延後，優先度低於 CMIN-US pipeline。**
所有規則都實作為獨立 function，方便之後打開／關閉或修改參數。建議實作：

- `is_short(text, min_tokens: int = 5) -> bool`
  - 若分詞後 token 數 < `min_tokens`，視為無資訊。

- `only_ticker_emoji(text: str) -> bool`
  - 文本只包含 `$TICKER` + emoji，沒有其他實質文字。
  - 例：`"$AAPL 🚀🚀"`。

- `too_many_repeated_emoji(text: str, repeat_limit: int = 4) -> bool`
  - 任一 emoji 重複出現次數 ≥ `repeat_limit`，視為噪音。

- `too_many_tags_urls(text: str, tag_ratio: float = 0.5) -> bool`
  - hashtag + URL token 佔比 > `tag_ratio` 則視為噪音。

- `is_pure_retweet(tweet: dict) -> bool`
  - `tweet["is_retweet"] == True`。

- `filter_tweet_rule_based(tweet: dict, config: RuleConfig) -> bool`
  - 整合上述多個規則，若命中任一 active 規則則「丟棄」。
  - `RuleConfig` 可指定是否啟用各規則與參數（min_tokens、repeat_limit...）。

### 4.2 Optional：LLM‑based 篩選

- `llm_filter_tweet(tweet: dict, llm_adapter) -> bool`
  - 將 `tweet["text"]` 丟給金融 LLM（例如 FinGPT adapter），由 LLM 回傳是否「具有資訊含量」。
  - 提供一個簡單 prompt 模板與回傳解析邏輯：
    - Prompt 要求 LLM 輸出 `Yes` / `No`。
  - 在 `clean_tweets.py` 提供選項：
    - `--enable_llm_filter`：先通過 rule-based，之後再由 LLM 做第二層篩選（成本較高，預設關閉）。

### 4.3 清洗後輸出

- 對每個 dataset，輸出一份 cleaned 檔案：
  - 路徑：`{OUTPUTS_DIR}/indices/{dataset_name}/raw/cleaned_{version}.parquet`（或 `.jsonl`）
  - 除了保留欄位外，還會輸出：
    - `_drop_reason`（若輸出 dropped 檔）
- 另外輸出被丟棄文本：
  - 路徑：`{OUTPUTS_DIR}/indices/{dataset_name}/{embed_model_slug}/dropped.parquet`
  - 欄位：原始文本 + drop rule（方便 ablation）。

### 4.4 CMIN-US Minimal Clean（`data_load/clean_cmin.py` 或對應函式）

> **實作優先度：高。請先完成此部份，作為 RAG pipeline 的第一個資料源。**

CMIN-US 不需要 emoji/hashtag 規則，只需簡單 sanity check：

- `load_raw_cmin_records(dataset_name: str) -> Iterable[dict]`
  - 從 CMIN-US 原始檔讀取欄位：`title`, `date`, `ticker`。
- `normalize_cmin_record(raw: dict) -> dict`
  - 轉成：
    - `text`：`title`
    - `date`：標準化 `YYYY-MM-DD`
    - `source_ticker`：`ticker`
- Minimal 過濾規則：
  - 若 `title` 為空或 strip 後長度為 0 → 丟棄。
  - 若 `ticker` 缺失 → 丟棄。
  - 若日期解析失敗 → 丟棄。
- 輸出 cleaned 檔案：
  - 路徑：`{OUTPUTS_DIR}/indices/CMIN/raw/cleaned_cmin_{version}.parquet`
  - 欄位至少包含：`text`, `date`, `source_ticker`。

---

## 5. 公司共現與 metadata 擴充（`data_load/extract_mentions.py`）

### 5.1 提及公司解析

- 目標：除了來源公司外，找出文本中出現的其他公司（cashtag 或名稱），後續可供 LLM 做 cross‑firm reasoning。
- 核心函式：
  - `extract_cashtags(text: str) -> List[str]`
    - 用正則找所有 `$AAPL`, `$TSLA` 等。
  - `normalize_ticker(ticker: str) -> str`
    - 將 cashtag 標準化為 dataset 內的 ticker 形式。
  - `extract_mentions(record: dict, valid_tickers: Set[str]) -> dict`
    - 對每條 cleaned 記錄更新 metadata：
      - `mentioned_tickers`: List[str]
      - `cashtag_count`: int
      - `url_count`: int
      - `has_cross_firm`: bool（是否有除 source_ticker 以外的提及）。

### 5.2 共現記錄（選擇性）

- 建立公司共現表：
  - `cooccurrence[(ticker_i, ticker_j)] += 1`
  - 輸出到：
    - `{OUTPUTS_DIR}/indices/{dataset_name}/cooccurrence.parquet`
- 後續可給 LLM 當作「關聯公司」的先驗資訊，類似 LLMFactor 靈感。

### 5.3 輸出格式

- 主要輸出檔：
  - `{OUTPUTS_DIR}/indices/{dataset_name}/raw/cleaned_with_mentions_{version}.parquet`
- 每筆記錄至少包含：
  - `text`, `date`, `source_ticker`
  - `mentioned_tickers`, `cashtag_count`, `url_count`, `is_retweet`

---

## 6. 向量化模組（`index/embed_texts.py`）

### 6.1 嵌入模型抽象介面

- 定義介面類別（或簡單函式型 registry）：

```python
class EmbeddingModel:
    def __init__(self, model_name: str):
        ...
    def encode(self, texts: list[str]) -> "np.ndarray[float32]":
        ...
```

- 目前以「開源免費」為首選：
  - 預設建議：
    - `FinLang/finance-embeddings-investopedia`（餵給 SentenceTransformer）
    - 或其他 sentence-transformers 金融／通用模型。
  - 將 `model_name` 作為 CLI 參數 `--embed_model`，並在內部轉成 slug 用於路徑命名。

### 6.2 向量產生與輸出

- 對 `cleaned_with_mentions_{version}.parquet` 中每條記錄：
  - 使用 `EmbeddingModel.encode([text])` 取得 1×D 向量。
  - 保留 metadata 欄位不動。
- 輸出：
  - `embeddings.npy`：shape = [num_docs, dim]，`float32`。
  - `metadata.parquet`：與 embeddings 對齊的一行一條文本訊息。
- 路徑：
  - `{OUTPUTS_DIR}/indices/{dataset_name}/{embed_model_slug}/embeddings.npy`
  - `{OUTPUTS_DIR}/indices/{dataset_name}/{embed_model_slug}/metadata.parquet`

---

## 7. FAISS 索引建置（`index/build_index.py`）

### 7.1 索引型態

- 先以簡單的平坦索引為主：
  - `IndexFlatIP` 或 `IndexFlatL2`。
- 未來可視資料量改成 IVF / HNSW 等進階索引。

### 7.2 建置流程

1. 從 `embeddings.npy` 載入所有向量。
2. 初始化 FAISS index：
   - `dim = vectors.shape[1]`
   - `index = faiss.IndexFlatIP(dim)`（預設用內積）。
3. 呼叫 `index.add(vectors)`。
4. 將 index 存到：
   - `{OUTPUTS_DIR}/indices/{dataset_name}/{embed_model_slug}/index.faiss`

### 7.3 索引 metadata

- 不在 FAISS 中直接存 metadata，而是：
  - 利用向量順序對齊 `metadata.parquet`。
  - 檢索後回傳 top‑k 的 row ids，再用 row id 回查 metadata。

---

## 8. CLI 入口與任務（`main.py`）

### 8.1 共同參數

- `--dataset_name`：`SAMPLE`/`ACL18`/`SEP`/`CMIN` 等。
- `--embed_model`：如 `FinLang/finance-embeddings-investopedia`。
- `--experiment_name`（可選）：若未指定則用 timestamp。
- `--min_tokens`：rule-based 清洗的最短 token 數。
- `--enable_llm_filter`：是否啟用 LLM 篩選（預設 False）。
- `--rebuild_index`：若 True，會覆蓋原有 index。

### 8.2 任務種類（`--task`）

1. `clean`
   - 讀取 raw tweets/news，執行 rule-based（+ optional LLM）清洗。
   - 輸出：`cleaned_{version}.parquet` + `dropped.parquet`。

2. `extract_mentions`
   - 對 cleaned 檔抽取 `mentioned_tickers`、`cashtag_count`、`url_count` 等。
   - 輸出：`cleaned_with_mentions_{version}.parquet`。

3. `embed`
   - 讀取 `cleaned_with_mentions_{version}.parquet`，呼叫 embedding 模型產生向量。
   - 輸出：`embeddings.npy` + `metadata.parquet`。

4. `build_index`
   - 讀取 `embeddings.npy`，建立 FAISS index。
   - 輸出：`index.faiss`。

5. `build_index_pipeline`
   - 由上而下串起 `clean → extract_mentions → embed → build_index`。
   - 參數集中由 `main.py` 管理。

---

## 9. 執行指令範例

> 以下假設已在根目錄執行，且已 `conda activate sep`，並設定好 `DATASETS_DIR`、`OUTPUTS_DIR`。

### 9.1 針對 SEP (SN2) 建立索引（分步）

1. 清洗 SEP 推文：

```bash
python -m STARE.main \
  --task clean \
  --dataset_name SEP \
  --min_tokens 5 \
  --enable_llm_filter False
```

2. 抽取提及公司與共現資訊：

```bash
python -m STARE.main \
  --task extract_mentions \
  --dataset_name SEP
```

3. 使用 FinLang Investopedia 嵌入模型產生向量：

```bash
python -m STARE.main \
  --task embed \
  --dataset_name SEP \
  --embed_model FinLang/finance-embeddings-investopedia
```

4. 建立 FAISS 索引：

```bash
python -m STARE.main \
  --task build_index \
  --dataset_name SEP \
  --embed_model FinLang/finance-embeddings-investopedia \
  --rebuild_index
```

### 9.2 一次執行整條 Pipeline

```bash
python -m STARE.main \
  --task build_index_pipeline \
  --dataset_name SEP \
  --embed_model FinLang/finance-embeddings-investopedia \
  --min_tokens 5 \
  --enable_llm_filter False \
  --rebuild_index
```

---

## 10. 後續擴充（非當前必做）

- 在 `models/STARE/pipeline.py` 中：
  - 實作 RAG 推論流程：時間過濾 → 檢索 → prompt 組裝 → LLM 生成預測與解釋。
- 與 TDMLLM、SEP 等 baseline 共用 dataset split 與評估模組，輸出 `predictions.jsonl`、`eval.json`。
- 對清洗規則與嵌入模型做 ablation（透過 `RuleConfig` 與 `--embed_model` 參數切換）。

以上內容即為目前 STARE 在「資料清洗 + 向量索引建置」階段的實作說明，可直接給 Cursor / Codex 參考以產生對應程式碼。

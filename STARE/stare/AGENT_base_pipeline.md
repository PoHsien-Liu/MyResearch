# STARE Base RAG Forecasting Agent (AGENTS.md)

## 0. Scope / 任務範圍

本檔案說明「**Base 單公司 RAG 預測管線**」的實作需求，供 Codex / Agent 在本專案中實作與維護。

目前僅實作 **base 模型**：

1. 僅使用「目標公司自身」的新聞與股價資訊（**不含跨公司 retrieval、不含 re-ranking**）。
2. 使用 RAG 從向量索引中檢索與目標公司相關的「潛在價格驅動事件」。
3. 結合「最近 5 日股價漲跌幅」與「檢索到的事件」讓 LLM 預測 **隔日股價是上漲或下跌**，並產生解釋文本。
4. 將所有 (prompt, context, label) 轉成 SFT 訓練資料，使用 **Llama-3.1-8B-Instruct** 搭配 **LoRA / QLoRA** 進行微調。

後續才會在此 base 管線之上，另外新增：
- 跨公司關係 retrieval
- re-ranking 模組

本 AGENTS.md 僅描述 **base 單公司 RAG + SFT** 的任務。

---

## 1. 依賴與共用規範

請遵守專案根目錄的 `AGENTS.md` 之通用規範（資料路徑、環境變數、輸出格式等），本檔僅新增本 Agent 的任務細節。

### 1.1 LLM Backend

- 因子（Step 2）：
  - 使用後端 `llama-70b`（實際模型為 **`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`**）。
  - 透過專案既有的 LLM backend 介面呼叫（例如 `stare/llm_backend/inference.py` 中的 batch 介面），不直接處理 vLLM 部署細節。
- Query 生成（Step 3）
  - 使用模型為 **`meta-llama/Llama-3.1-8B-Instruct`**
  - 想一下 LLM backend 要怎麼改讓他可以支援 **`meta-llama/Llama-3.1-8B-Instruct`** 模型
- 預測與 SFT 微調：
  - Base model 採用 **`Llama-3.1-8B-Instruct`**。
  - 微調方式：**LoRA / QLoRA**（建議使用 PEFT / transformers 等現有套件）。

### 1.2 嚴格避免 Data Leakage（時間窗限制）

對於任一訓練/測試樣本 `(ticker, target_date)`：

- **股價資訊**：
  - 只能使用 `target_date` 之前的資料，例如最近 5 個交易日 `D-5 ... D-1`。
  - 嚴禁使用 `target_date` 當日或之後的任何價格（收盤價、報酬、波動度等）。

- **新聞 / 事件檢索**：
  - 檢索時間窗必須限制在 `target_date` 之前的文件，例如 `[target_date - 5, target_date - 1]`。
  - 嚴禁讀取 `target_date` 當日（D0）或之後的任何新聞、標題或摘要。
  - 必須在向量檢索層實作「時間過濾」作為硬性條件。

- **標籤 (`ground_truth`)**：
  - 標籤為 `target_date` 的漲跌。
  - 標籤僅用於訓練與評估，不得作為 prompt context 給 LLM。

---

## 2. Pipeline Overview / 任務流程總覽

對每一個訓練樣本 `(ticker, target_date)`，本 Agent 需要完成以下步驟：

1. **股價序列化（Step 1）**：  
   - 讀取 `ticker` 在 `target_date` 之前的最近 5 個交易日的收盤價。  
   - 計算每天相對前一交易日的「**日報酬 (%)**」。  
   - 使用固定的英文模板（見 §3）封裝成一段文字 `PRICE_CONTEXT_BLOCK`。

2. **公司因子生成（Step 2, 離線）**：  
   - 使用 `llama-70b`，輸入只有 `ticker`，生成該公司常見的「價格驅動因子」（factors）。  
   - 以 JSON 格式儲存於 `factors/{TICKER}.json`，供後續 Query 生成使用。  
   - 每個 ticker 預期只需執行一次（或定期重跑以更新）。

3. **RAG Query 生成（Step 3）**：  
   - 讀取 `factors/{TICKER}.json`。  
   - 基於 ticker、target_date、time_window 與 factors，利用 `llama-70b` 生成若干條英文檢索 Query。  
   - 嚴格要求 Query 針對「可能影響短期股價的事件」。

4. **向量索引檢索（Step 4, base 版本）**：  
   - 使用上一步生成的 Query，於既有向量索引中檢索新聞。  
   - 必須加入「**時間過濾條件**」，限制在 `[target_date - time_window, target_date - 1]` 範圍內，避免 data leakage。  
   - 從所有 Query 的結果中合併並去重，依相似度分數排序，選出 **top-K（目前 K=5）** 事件，形成 `EVENTS_BLOCK`。

5. **LLM 預測 prompt 組裝（Step 5）**：  
   - 將 `PRICE_CONTEXT_BLOCK` + `EVENTS_BLOCK` 塞入預測用 prompt template（英文，見 §6）。  
   - 呼叫 `llama-8b` 或其他 base LLM，要求輸出 JSON：  
     - `"prediction": "UP" | "DOWN"`  
     - `"reason": "<解釋文本>"`  
     - `"used_event_ids": [ ... ]`  

6. **SFT 訓練資料構建（Step 6）**：  
   - 將上述 prompt + LLM 回覆 + `ground_truth` label 轉換成統一格式的 SFT 訓練樣本（例如 chat-style JSONL）。  
   - 使用 `Llama-3.1-8B-Instruct` + LoRA/QLoRA 進行微調，實作訓練腳本與基本 log 紀錄。  

目前階段：**以訓練資料構建與 SFT 訓練為主，尚不必完成完整測試與自動評估腳本**，但建議預留介面。

---

## 3. 股價序列化模板（Step 1, English）

此步驟不需要再丟給 LLM，僅需在程式中直接組字串即可。

### 3.1 Input

- `ticker`: 目標股票代號，例如 `"AAPL"`。
- `target_date`: 預測日期（D0），格式如 `"2014-01-15"`。
- `(date, close)` for the last 5 trading days: `D-5 ... D-1`。
- 預先計算 5 個交易日的日報酬（百分比）：`RET_MINUS_5 ... RET_MINUS_1`。

### 3.2 Template（英文固定字串）

```text
[PRICE CONTEXT]

We are analyzing the recent price performance of stock {TICKER}.

Here are the last 5 trading days before the prediction date {TARGET_DATE} (D-5 to D-1), expressed as daily percentage returns relative to the previous trading day:

- D-5 ({DATE_MINUS_5}): {RET_MINUS_5}% daily return
- D-4 ({DATE_MINUS_4}): {RET_MINUS_4}% daily return
- D-3 ({DATE_MINUS_3}): {RET_MINUS_3}% daily return
- D-2 ({DATE_MINUS_2}): {RET_MINUS_2}% daily return
- D-1 ({DATE_MINUS_1}): {RET_MINUS_1}% daily return

All returns are close-to-close percentage changes. Positive values indicate the stock went up; negative values indicate it went down.
```

請實作一個函式，例如：

```python
def build_price_context(ticker, target_date, last5_dates, last5_returns) -> str:
    ...
```

---

## 4. 因子生成 prompt（Step 2, 使用 70B）

此步驟為 **離線程式**，對每個 ticker 跑一次即可。

### 4.1 LLM 調用方式（概念）

- Backend: `llama-70b`（對應 `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`）。
- 建議使用 batch 介面，支援一次處理多個 ticker。

### 4.2 Prompt Template（English, system + user）

```text
[SYSTEM]

You are a financial analyst who understands common drivers of stock price movements for publicly listed companies.

Your task is to list the most important types of events and factors that typically move the stock price of a given company. 
You will be given only the stock ticker symbol. If you recognize the company, use your knowledge about that specific firm. 
If you are not sure or the ticker is ambiguous, fall back to typical drivers for a public company in that sector or for large listed companies in general.

Focus on *event types* and *economic/financial factors* that can cause noticeable short-term price reactions.

Respond in **English** and use the following strict JSON format:

{
  "ticker": "<TICKER>",
  "factors": [
    {
      "name": "<short_factor_name>",
      "description": "<1-2 sentence description of how this factor affects the stock price>",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ]
}

Do not add any extra fields. Do not add explanations outside the JSON.


[USER]

Ticker: {TICKER}
Please output the JSON object described above.
```

### 4.3 Output & 儲存格式

- 將 LLM 輸出 parse 成 JSON，儲存為：
  - `factors/{TICKER}.json`
- 後續 Step 3 會讀取並使用 `factors[*].name` 與 `factors[*].keywords`。

---

## 5. Query 生成 prompt（Step 3, 使用 70B）

### 5.1 Input

- `ticker`: 目標公司代號。
- `target_date`: 預測日期 `D0`。
- `start_date`, `end_date`: 檢索視窗（例如 `D-5` 至 `D-1`）。
- `FACTORS_TEXT`: 由 `factors/{TICKER}.json` 攤平成多行文字，例如：
  - `Revenue and earnings surprises (keywords: earnings, guidance, profit, EPS, revenue, forecast)`  
  - `Major product launch or failure (keywords: product launch, recall, failure, delay, innovation)`  
  - ...

### 5.2 Prompt Template（English）

```text
[SYSTEM]

You are a retrieval query generator for a financial news search system.

Given:
- A target stock ticker
- A prediction date and a look-back window (recent news before the prediction date)
- A list of typical price drivers (factors) for this stock

Your task is to generate high-quality English search queries that are likely to retrieve news articles which truly affect the stock's short-term price movement.

Guidelines:
- Each query must explicitly mention the company name or ticker.
- Each query should focus on **price-moving events**, not trivial or routine information.
- Use the provided factors and their keywords to target meaningful events (earnings, guidance, regulatory actions, M&A, product issues, supply chain shocks, etc.).
- Avoid generic phrases like "latest news" or "general updates".
- Tailor the queries to the specific time window.
- The retrieval system will strictly filter documents to the look-back window, so do not ask for future or forward-looking data.

Output format (strict JSON):

{
  "queries": [
    "<query_1>",
    "<query_2>",
    "<query_3>"
  ]
}

No extra text outside the JSON.

[USER]

Ticker: {TICKER}
Prediction date: {TARGET_DATE}
Look-back window: from {START_DATE} to {END_DATE} (inclusive)

Typical price drivers for this stock (based on prior analysis):

{FACTORS_TEXT}

Please generate 3 focused English search queries in the JSON format described above.
```

### 5.3 檢索實作注意事項（避免 data leakage）

實作向量檢索時請特別注意：

- 必須在查詢條件中加入 **時間過濾**，例如：
  - `doc.published_at >= START_DATE` 且 `doc.published_at <= END_DATE`
  - 或等價條件，確保所有檢索結果都在 `target_date` 之前。
- 絕不可使用 `target_date` 當日或之後的新聞。

---

## 6. 預測 prompt（Step 5, 使用 8B）

### 6.1 Input

- `ticker`
- `target_date`
- `PRICE_CONTEXT_BLOCK`（Step 1 輸出）
- `EVENTS_BLOCK`：向量檢索後選出的 top-K 事件，格式類似：

```text
[EVENT CONTEXT]

We retrieved the following news events for {TICKER} within the last 5 trading days before {TARGET_DATE}:

(1) [{EVENT1_DATE}] {EVENT1_TEXT}
(2) [{EVENT2_DATE}] {EVENT2_TEXT}
(3) [{EVENT3_DATE}] {EVENT3_TEXT}
(4) [{EVENT4_DATE}] {EVENT4_TEXT}
(5) [{EVENT5_DATE}] {EVENT5_TEXT}
```

### 6.2 Prompt Template（English, system + user）

```text
[SYSTEM]

You are a cautious and detail-oriented equity analyst.
Your task is to predict the **next-day price direction** of a stock (UP or DOWN) and provide a short, evidence-based explanation.

You will receive:
1. A summary of the stock's daily percentage returns over the last 5 trading days before the prediction date.
2. A small set of news events retrieved for the same look-back window.

You must:
- Predict whether the stock price on the prediction date will close UP or DOWN relative to the previous trading day.
- Base your reasoning only on the provided price history and news events.
- Explicitly reference which news events (by their IDs, e.g. (1), (3)) and which aspects of the recent price trend you are using.
- Be conservative: if the evidence is weak or mixed, still choose UP or DOWN, but reflect the uncertainty in your explanation text.

Respond in strict JSON with the following schema:

{
  "prediction": "UP" or "DOWN",
  "reason": "<2-5 sentences explaining the key drivers, explicitly citing event IDs like (1), (3) and summarizing the recent price trend>",
  "used_event_ids": [<list of integers corresponding to the events used, e.g. [1, 3]>]
}

Do not output anything outside this JSON.


[USER]

Target stock: {TICKER}
Prediction date (D0): {TARGET_DATE}

{PRICE_CONTEXT_BLOCK}

{EVENTS_BLOCK}

Based on the 5-day price pattern and the listed news events, please fill in the JSON object described above.
```

---

## 7. SFT 資料與訓練（Step 6, 簡要要求）

### 7.1 SFT 訓練樣本格式（建議）

對每個 `(ticker, target_date)` 產生一筆 SFT 樣本，可以使用 chat-style JSON，例如：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<SYSTEM text from §6.2>"
    },
    {
      "role": "user",
      "content": "<USER prompt with PRICE_CONTEXT_BLOCK + EVENTS_BLOCK>"
    },
    {
      "role": "assistant",
      "content": "{ \"prediction\": \"UP\", \"reason\": \"...\", \"used_event_ids\": [1, 3] }"
    }
  ],
  "metadata": {
    "ticker": "AAPL",
    "target_date": "2014-01-15",
    "ground_truth_label": "UP"
  }
}
```

- `ground_truth_label` 由實際隔日漲跌計算得出，可用於評估。  
- `assistant` 的內容在初始版本可以由 **規則生成** 或由某個大模型生成，再經過清洗；後續使用這些資料對 8B 模型做 SFT。

### 7.2 訓練要求

- Base model: `Llama-3.1-8B-Instruct`。
- Fine-tuning: LoRA / QLoRA（使用現成框架，如 PEFT）。  
- 需輸出：
  - 訓練 log（loss 隨 step / epoch 變化）
  - 訓練耗時（總時間與每 epoch 時間）
  - 最終權重（LoRA adapter 或 QLoRA checkpoint）

---

## 8. 目前階段完成定義

本 AGENT 在目前階段的「完成」定義為：

1. 能從給定的原始股價與新聞資料，對 train split 產生：
   - 完整的 `PRICE_CONTEXT_BLOCK`
   - 離線生成的 `factors/{TICKER}.json`
   - Query 生成 + 無 data leakage 的向量檢索（只看過去 5 日）
   - `EVENTS_BLOCK`
   - 預測 prompt + 模型輸出 JSON

2. 能將上述資料轉為統一的 SFT JSONL 檔案。

3. 能對 `Llama-3.1-8B-Instruct` 跑通一輪 LoRA/QLoRA 訓練（在 subset 上測試即可），並成功輸出訓練 log 與 checkpoint。

跨公司 retrieval 與 re-ranking 將在 base pipeline 穩定後，於後續 AGENTS 檔案中另行擴充。

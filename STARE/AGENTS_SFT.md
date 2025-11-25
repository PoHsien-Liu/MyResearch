# STARE SFT 訓練資料與 Loss Mask 設計說明（AGENTS\_SFT.md）

本檔案用來說明「**如何為 STARE 任務建立 SFT 訓練資料與 loss mask**」，以便微調 `meta-llama/Llama-3.1-8B-Instruct`。  
請依照以下規格實作對應的 Dataset、collator 與訓練流程。  
你（Codex）已經可以看到整個 repo 與現有的資料儲存結構，這裡只定義「應該做什麼」，不指定實作路徑。

---

## 0. 目標與前提

### 0.1 目前已具備的資訊

對於每一筆樣本（某檔股票、某個預測日） `(ticker, target_date)`，現有 pipeline 已可產生：

- `price_context`：  
  - 近 5 個交易日（D-5 至 D-1）的股價 daily return 之英文描述。  
- `event_context`：  
  - 對目標公司在指定 time window 內做向量檢索後得到的 top-K 事件，  
  - 每則事件已附上事件 ID，例如 `(1) [...]`、`(2) [...]`。  
- `ground_truth_label`：  
  - `target_date` 這一天相對前一交易日（D-1）的漲跌方向，  
  - 標籤為 `"UP"` 或 `"DOWN"`。

目前 **沒有**：

- 人工標註的高品質 `reason`（解釋文字）
- 人工標註的 `used_event_ids`（引用了哪幾則事件）

### 0.2 微調目標

微調後的 `Llama-3.1-8B-Instruct` 應該具備能力：

1. 接收：
   - 近 5 日股價文字描述（`price_context`）
   - 近 5 日內 top-K 事件（`event_context`，包含 event IDs）
2. 輸出一個 JSON 物件，格式固定為：
   ```json
   {
     "prediction": "UP" or "DOWN",
     "reason": "<2-5 sentences explanation>",
     "used_event_ids": [<list of event IDs>]
   }
   ```
3. `prediction` 應該貼近真實漲跌標籤。  
4. `reason` 與 `used_event_ids` 在目前階段 **不當作真值監督**，僅要求模型在推論時遵守格式並嘗試給出合理內容。

### 0.3 訓練設計的關鍵原則

- **只嚴格監督 `prediction` 與部分 JSON 結構**：
  - `prediction` 的值（"UP"/"DOWN"）是真實標籤，必須被用來訓練。
  - JSON 的基本結構（例如 `{`, `}`, `"prediction"`, `:`, 引號）可以選擇納入 loss，幫助模型穩定輸出正確 schema。
- **不監督當前的解釋與引用內容**：
  - `reason` 文字與 `used_event_ids` 內的具體數字在目前階段不可靠，  
    只當作「格式示範」，並在 loss 中全部 mask 成 `-100`。
- 這樣可以：
  - 避免把低品質或隨機產生的 explanation 當成 ground truth；
  - 同時仍讓模型學會分類任務與輸出 JSON 結構。

---

## 1. 中間資料格式（raw → SFT）

請建立一個 **中間資料格式**，例如 JSONL 或 parquet，每一列代表一個樣本 `(ticker, target_date)`，至少包含：

- `ticker`：股票代號（如 `"AAPL"`）
- `target_date`：預測日（例如 `"2014-01-15"`）
- `price_context`：近 5 日股價敘述（英文）
- `event_context`：top-K 事件敘述（英文，已附 event IDs）
- `label`：漲跌標籤 `"UP"` 或 `"DOWN"`

這個中間格式將會是 SFT 資料轉換的輸入來源。

> 實作建議：  
> - 可在 `datasets/processed/` 或 `outputs/processed/` 建立對應檔案。  
> - 若已有類似結構的檔案，可直接沿用並加上必要欄位。

---

## 2. SFT Chat 格式設計

### 2.1 Chat 結構：system / user / assistant

對每一筆樣本，建立一個 chat 物件，結構為：

```jsonc
{
  "messages": [
    { "role": "system", "content": "<SYSTEM_PROMPT>" },
    { "role": "user",   "content": "<USER_PROMPT>" },
    { "role": "assistant", "content": "<ASSISTANT_JSON>" }
  ],
  "metadata": {
    "ticker": "...",
    "target_date": "...",
    "ground_truth_label": "UP"
  }
}
```

### 2.2 SYSTEM_PROMPT 的內容邏輯

請在 system message 中包含以下要點（以英文撰寫即可）：

- 你是一名謹慎的股票分析師。  
- 模型會看到：
  - 過去 5 個交易日的 daily return 描述；
  - 在相同視窗內與該股票相關的少量新聞事件列表（已編號）。  
- 任務：
  1. 預測在 `target_date (D0)` 當日收盤價，相對前一交易日（D-1）是 **UP** 還是 **DOWN**。
  2. 輸出一個 JSON 物件，格式必須為：
     ```json
     {
       "prediction": "UP" or "DOWN",
       "reason": "<2-5 sentences explaining the key drivers based only on the given price and events>",
       "used_event_ids": [<list of integers>]
     }
     ```
- 限制：
  - 只能根據 prompt 中的 `price_context` 和 `event_context` 推理；
  - `prediction` 的值只能是 `"UP"` 或 `"DOWN"`；
  - `used_event_ids` 應列出實際在解釋中使用到的事件 ID（例如 `[1, 3]`）。

Codex 可自行撰寫具體英文描述，但需涵蓋以上要求。

### 2.3 USER_PROMPT 的內容邏輯

User message 內容建議格式：

```text
Target stock: {TICKER}
Prediction date (D0): {TARGET_DATE}

[PRICE CONTEXT]
{price_context}

[EVENT CONTEXT]
{event_context}
```

- `{price_context}`：直接填入中間資料的 `price_context` 欄位文字。  
- `{event_context}`：直接填入中間資料的 `event_context` 欄位文字。

### 2.4 ASSISTANT_JSON 的內容邏輯

Assistant 的 target 回覆是一個 JSON 字串，形式如下：

```json
{
  "prediction": "UP",
  "reason": "PLACEHOLDER_REASON",
  "used_event_ids": []
}
```

請注意：

- `"prediction"` 的值 **必須** 和 `label` 一致（由程式依照 ground truth 自動填入 `"UP"` 或 `"DOWN"`）。  
- `"reason"` 與 `"used_event_ids"` 可以先用簡單模板填滿，例如：
  - `reason`：可以是空字串 `""`，或固定字串如 `"Reason is not supervised during training."`
  - `used_event_ids`：可以是空 list `[]`，或隨便填入合乎範圍的 event id（未來都不會被算 loss）
- 重點是保持 **格式與欄位名稱一致**，之後 inference 時模型會依照 system prompt 和自身能力填入真正的解釋。

---

## 3. Tokenization 與 Loss Mask 設計

### 3.1 基本流程

1. 使用 Llama-3.1-8B-Instruct 相容的 tokenizer。  
2. 將 `system`、`user`、`assistant` 三段 messages 透過：
   - `tokenizer.apply_chat_template`，或  
   - 自行拼接（包含 `<s>`, `</s>` 等 special tokens），  
   轉換為單一 `input_ids` 序列。
3. 建立 `labels = input_ids.clone()`，然後對 `labels` 做 masking。

### 3.2 Masking 策略（核心）

1. **預設全部設為 -100**  
   - 一開始將 `labels[:] = -100`，代表預設不對任何 token 算 loss。
2. **只對 assistant JSON 中「prediction 欄位」之前的部份算 loss**：
   - 假設 assistant JSON 固定為：
     ```json
     { "prediction": "<UP_or_DOWN>", "reason": "...", "used_event_ids": [...] }
     ```
   - 只對從 JSON 開頭 `{` 一直到 `"prediction": "<UP_or_DOWN>"` 這段文字對應到的 tokens 設置有效 label：
     - 包含：`{`, `"prediction"`, `:`, `"UP"` / `"DOWN"`, 引號、逗號等；
     - 不包含：`"reason"` key 以及後面的整段、`"used_event_ids"` key 以及其值。
3. 具體實作可由 Codex 自行決定，建議方式為：
   - 拿到整段 chat text 字串（含 system/user/assistant）；  
   - 透過 `tokenizer(..., return_offsets_mapping=True)` 取得每個 token 的 `(start, end)` 字元區間；  
   - 找出 assistant JSON 的字元範圍，並在其中定位 `"prediction"` value 字串（"UP" / "DOWN"）；  
   - 定義「監督區間」為 assistant JSON 中從 `{` 開始到 `"UP"` / `"DOWN"` 結束；  
   - 對於 offset 落在此監督區間的 token，將 `labels[token_index] = input_ids[token_index]`；  
   - 其它 token 一律保持為 `-100`。
4. 如此一來：
   - 模型在訓練時會學會正確輸出 `"prediction": "UP"` 或 `"prediction": "DOWN"` 這一段內容；  
   - 也會學會 JSON 結構的基本樣貌（若把 `{`, `"prediction"` 等 token 一併納入 loss）；  
   - 但不會被強迫產生特定的 `reason` 或 `used_event_ids` 內容。

### 3.3 可選的進階作法（非必須）

如果之後希望模型更穩定輸出完整 schema，可在 loss mask 中：

- 將 `"reason"`、`"used_event_ids"` 這兩個 key 的 token 也納入 loss；  
- 但仍然把對應的 value（解釋文本和 id 列表）設為 `-100`。

這樣會讓模型更確定要產生完整的三個欄位，但仍不會被束縛於現階段的解釋內容。

---

## 4. Dataset 與 DataLoader 實作要求

請實作一個專用的 SFT Dataset 與 collator，整合以上邏輯。

### 4.1 Dataset

Dataset 大致需提供：

- `__init__(self, raw_records_or_path, tokenizer, ...)`  
  - `raw_records_or_path`：中間資料檔路徑或已載入的 list。  
- `__len__`：樣本數。  
- `__getitem__(self, idx)`：  
  1. 讀取一筆中間資料：`ticker`, `target_date`, `price_context`, `event_context`, `label`。  
  2. 根據 §2 組合出 `system`, `user`, `assistant_json` 三段文字。  
  3. 使用 tokenizer 轉成 `input_ids`。  
  4. 套用 §3 的 loss mask 策略，生成 `labels`。  
  5. 回傳：`{"input_ids": input_ids, "labels": labels}`。

### 4.2 Collator

- 負責將不同長度的 `input_ids`、`labels` 做 padding：  
  - `input_ids` padding value 使用 `tokenizer.pad_token_id`；  
  - `labels` 的 padding value 使用 `-100`。  
- 回傳一個 batch 字典，可直接餵給 HuggingFace `Trainer` 或自寫訓練 loop。

---

## 5. 訓練流程整合

請將上述 Dataset / collator：

1. 整合到既有的訓練框架中（可能已有其他 baseline 的 SFT 程式可參考）。  
2. 遵守以下原則：
   - Base model 使用 `meta-llama/Llama-3.1-8B-Instruct`（或專案設定中的對應名稱）；  
   - 此階段為「單任務 SFT」：輸入為 STARE 的 price+news prompt，輸出為 JSON；  
   - 損失函數使用模型內建的 cross-entropy，透過我們提供的 `labels` mask 控制實際 supervision 範圍。  

訓練時請記錄：

- loss 隨 step / epoch 變化（建議輸出到 log 檔）；
- 訓練時間（總時間與每 epoch 時間）；
- 最終微調權重（例如 LoRA/QLoRA adapter 或完整 checkpoint）的輸出路徑。

---

## 6. 未來擴充預留

本設計刻意只對 `prediction` 做 supervision，以避免錯誤的解釋標註。  
之後若要擴充，可以在同一框架下新增：

- 第二階段 SFT：  
  - 使用一個 teacher LLM 為部分樣本自動產生較高品質的 `reason` 與 `used_event_ids`；  
  - 在這個 subset 上對 explanation token 給予較低權重的 loss。  
- Self-cite / grounding 訓練：  
  - 事後用檢索或 ablation 方法，選出「真正在用的事件」，作為弱標註 citation。

目前階段，請專注於本檔案定義的 **Base SFT 設計**，確保整個 pipeline（raw records → chat 格式 → masked labels → 訓練）能穩定運作。

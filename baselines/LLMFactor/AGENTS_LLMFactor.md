
# /baselines/LLMFactor/AGENTS.md

## 1. 任務說明（LLMFactor baseline）

這個 baseline 要在 `baselines/LLMFactor/` 目錄下，實作論文 **LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction** 所提出的框架。  
重點是 **照論文設計的 Sequential Knowledge-Guided Prompting (SKGP)**，不做額外優化或改寫模型架構：

- **不做任何微調 / 訓練**（no fine-tuning, no LoRA）。
- 只對 **預先處理好的樣本做推論**，產生：
  - next-day stock movement 預測（rise / fall）
  - 對應的解釋文本（LLM 輸出）
  - 中間階段產生的 relations 與 factors（方便之後做分析）

LLMFactor 的核心流程為三步驟 SKGP：

1. **Step 1 — 關係背景知識（Relation / Background Knowledge）**  
   對「目標公司 vs. 在新聞中共現的其他公司」詢問兩者的關係，作為背景知識。
2. **Step 2 — 從新聞抽因子（Factor Generation）**  
   對目標公司的當日新聞，請 LLM 抽出 top-k 個「可能影響股價的因素（factors）」。  
3. **Step 3 — 結合時間序列文字 + 關係 + 因子，預測漲跌**  
   把過去 t 天股價漲跌轉成文字句子，再加上 Step 1 的關係與 Step 2 的因子，一起丟給 LLM，請它判斷明天股價是 rise / fall 並給出理由。

所有 template 的結構與資訊內容都應 **忠實對應論文 Appendix A Table 5 的 SKGP 模板與 Section 3.2 的描述**。

---

## 2. 環境與路徑

### 2.1 主要環境變數

沿用根目錄的設定：

- `DATASETS_DIR`：資料集根路徑，預設 `./datasets`。
- `OUTPUTS_DIR`：實驗輸出根路徑，預設 `./outputs`。

- `model_id`：用來記錄使用的 LLM，例如 `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`。

### 2.2 主要檔案結構（建議）

在 `baselines/LLMFactor/` 底下，Codex 需要實作：

- `runner.py`：主程式入口（命令列介面）。
- `skgp.py`：實作 SKGP 三個步驟（prompt 組合與解析）。
- `llm_client.py`：封裝與 LLM 溝通的介面（可切換不同 backend，例如 OpenAI API 或本地 huggingface 模型）。
- `data_loader.py`：讀取預先準備好的樣本（JSONL / CSV）。

檔名可調整，但需要清楚且模組化，便於之後維護。

---

## 3. 輸入資料假設

### 3.1 样本單位

每一個樣本對應一組 `(target_stock, date_target)`：

- 目標：預測 `date_target` 這一天，`target_stock` 的股價是 **Rise** 還是 **Fall**。
- 依照論文，使用過去 `t = 5` 天的股價序列當作 time-series 資訊。


### 3.2 共現公司偵測（Matching）

Step 1 需要知道在 `news` 文字中與 `target_stock` 一起出現的「其他公司」。假設：

- 可以調用 STARE 裡面找共現關係的程式碼，產生 /home/pohsien/Research/outputs/indices/CMIN/default/company_neighbors.json 類似檔案，標註每一間公司的共現次數
- 設一個參數可以選擇要取共現次數最高的 top-n 公司  
---

## 4. SKGP 實作細節

### 4.1 總覽

對每個樣本，SKGP 的流程是：

1. 從 `news` 中找出所有共現公司 `company_j`（不包含自己）。
2. **Step 1（Relation）：** 對每個共現公司，詢問「target vs company_j 的關係」，並收集結果。  
3. **Step 2（Factor）：** 對 `target` 的所有 `news`，請 LLM 抽出 top-k 個可能影響股價的因素。  
4. **Step 3（Price）：** 先把 `prices` 轉成「rise / fall」文字句子，再連同關係 + 因子一起丟給 LLM，請它判斷 `date_target` 當天股價漲跌與理由。

### 4.2 Step 1：公司關係（Relation Prompt）

**目標：**  
根據論文 Appendix A Table 5 的 Step1 模板，使用「填空 + 完整句子」形式取得公司關係。

**Prompt 結構（英文，大意）：**

- 指示 LLM：請填入空格並產生一個完整句子，描述兩家公司之間「最可能的關係」。  
- 句型類似：  
  > `COMPANY_A and COMPANY_B are most likely in a ___ relationship.`  

實作建議：

- 程式端組出完整 prompt，例如：

  ```text
  Please fill in the blank and return a complete sentence:
  Apple Inc. and Corning Incorporated are most likely in a ___ relationship.
  ```

- 解析輸出時可以：
  - 直接使用整句做為 background knowledge（放到 Step 3）。
  - 或進一步抽出 `___` 中的關係短語（可選）。

**輸出彙整：**

把所有共現公司的關係整理成多行文字，例如：

```text
Apple Inc. and Corning Incorporated are most likely in a supplier relationship.
Apple Inc. and Samsung Electronics are most likely in a competitor relationship.
```

這個區塊在 Step 3 中會被放在「These are the connections between the companies...」後面。

---

### 4.3 Step 2：因子抽取（Factor Prompt）

**目標：**  
請 LLM 從 `news` 中抽出 top-k 個「可能影響股價的因素」。論文預設 `k = 5`。
這裡的 news 指得是目標公司前五天的新聞，一樣設置一個最高一天可以幾則新聞的參數。

**Prompt 結構（英文，大意）：**

- 指示 LLM：從以下新聞中抽取「可能影響 `STOCK` 股價的前 k 個因素」。  
- 核心句型對應論文的模板：  
  > `Please extract the top k factors that may affect the stock price of STOCK from the following news.`

建議實作方式：

1. 先把當日所有 `news`（多篇）在程式內 concat 成一個較長的文字。
2. prompt 由三段組成：
   - 說明任務。
   - 給定 `stock_name` 與 `k`。
   - `News:` 區段貼上 concat 後的新聞內容。

解析輸出時：

- 允許 LLM 回傳條列式結果：`1. ...`、`2. ...`，等等。
- 若沒有條列，則可以將每一行或每一個句子視為一個 factor。
- 至少保留前 `k` 個因素（如果模型輸出超過 k 條）。

**輸出彙整：**  

在 Step 3 中，會以一段文字引導：

```text
These are the main factors that may affect this stock's price recently:
1. ...
2. ...
3. ...
4. ...
5. ...
```

---

### 4.4 Step 3：轉換時間序列 & 最終預測 Prompt

#### 4.4.1 將價格序列轉成文字（Time Template）

依照論文 Section 3.2.3 的描述：

- 首先將價格轉成二元漲跌序列 `P̂ = {P̂_1, ..., P̂_t}`，規則：
  - 若 `price[i] > price[i-1]` → `P̂_i = 1`（rise）。
  - 否則 `P̂_i = 0`（fall）。
- 使用一個函數 `f` 把 `P̂_i` 轉為文字：
  - `1 → "rose"`
  - `0 → "fell"`

接著，把每一天的結果轉成句子，例如：

```text
On 2019-09-12, the stock price of Apple Inc. rose.
On 2019-09-13, the stock price of Apple Inc. fell.
...
```

這一段稱為 `time_block`，會被放進 Step 3 的 prompt 中。

#### 4.4.2 最終預測 Prompt 結構

根據 Appendix A Table 5，Step 3 的模板大意如下（英文）：

1. 先給一個任務指令，要求 LLM：
   - 根據以下資訊，判斷股價方向（rise / fall）。
   - 填寫空格並給出理由。
2. 再依序提供三個資訊區塊：
   - **Factors 區塊**：Step 2 抽出的因子。
   - **Relations 區塊**：Step 1 的公司關係描述。
   - **Past price movements 區塊**：`time_block` 句子（大約 5 行）。
3. 最後用一句話請 LLM 在空格填入 “rise” 或 “fall”：  
   - 格式類似：  
     > `On DATE_TARGET, the stock price of STOCK will ___.`

實作時，可將 prompt 組成下列結構（示意）：

```text
Based on the following information, please judge the direction of the stock price as rise or fall, fill in the blank and give reasons.

These are the main factors that may affect this stock's price recently:
{factor_block}

These are the connections between the companies that have appeared in the news:
{relation_block}

{time_block}

On {date_target}, the stock price of {stock_name} will ___.
```

Codex 需要：

- 確保最後一行句型保持固定，使得解析程式可以穩定抓到 `rise` / `fall`。  
- 解析時，建議使用簡單的字串搜尋或正則，從最後一行找出 `rise` 或 `fall`，並轉成：
  - `rise → 1`
  - `fall → 0`

---

## 5. 模型與推理設定

### 5.1 論文設定（供參考）

根據原論文的實作細節：

- 使用模型：`gpt-3.5-turbo-1106`, `gpt-4`, `gpt-4-1106-preview`。
- 但們基於公平比較改用 `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` 模型進行推論
- window size `t = 5`。
- `k = 5`（每個樣本抽 5 個 factors）。
- GPT 系列的 batch size 約為 5。

在本專案中：

- 模型 backend 可以換成開源模型（例如 `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`），但 **流程與模板結構要保持一致**。
- 需要把所有與 LLM 有關的設定收斂到一個 config / argparse 參數，例如：
  - `--llm_backend {openai,hf}`
  - `--model_name_or_path ...`
  - `--max_new_tokens ...`
  - `--temperature ...`
- 可以寫成 llm_config.yaml，請參考/home/pohsien/Research/baselines/ZeroShotLLMs/llm_config.yaml

### 5.2 執行命令範例

請 Codex 為 LLMFactor baseline 實作一個 CLI：

```bash
# 基本推論（test split）
python -m baselines.LLMFactor.runner \
  --dataset stocknet \
  --split test \
  --llm_backend hf \
  --model_name_or_path meta-llama/Meta-Llama-3.1-70B-Instruct-AWQ \
  --output_dir "$OUTPUTS_DIR/LLMFactor/stocknet/llama-3.1-70b-awq"

# 只跑前 N 筆樣本（debug 用）
python -m baselines.LLMFactor.runner \
  --dataset cmin-us \
  --split val \
  --max_samples 128 \
  --llm_backend openai \
  --model_name_or_path gpt-4-1106-preview
```

runner 需要：

1. 讀取對應的 JSONL。  
2. 逐筆執行 SKGP（可以視情況做小 batch）。  
3. 儲存預測與解釋。  
4. 計算 ACC、MCC，並印出到終端機與 log 檔。

---


## 6. Codex 實作重點總結

1. **不要訓練模型，只做推論**：  
   - LLMFactor baseline 只呼叫 LLM API 或本地模型，利用 SKGP 做三階段 prompting。

2. **嚴格依照 LLMFactor 論文的 SKGP 結構**：  
   - Step 1：填空句描述兩家公司關係。  
   - Step 2：從新聞中抽取 top-k 因子。  
   - Step 3：將過去 t 天漲跌轉成句子，結合 factors + relations，請 LLM 判斷 rise / fall 並給理由。  

3. **支援四個資料集名稱**：`stocknet`, `cmin-us`, `sep`, `sample`。  
   - EDT 沒有 time series → Step 3 可省略時間序列文字區塊。

4. **統一輸出路徑與格式**：  
   - JSONL 存逐樣本預測 + 解釋。  
   - JSON 存 metrics（ACC, MCC）。  
   - log 檔記錄實驗設定與時間成本。

5. **所有 prompt 與流程邏輯需與論文一致，不額外加入新模組（例如 time-aware re-ranking 等）。**

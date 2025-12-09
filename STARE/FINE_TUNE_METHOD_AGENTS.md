# AGENTS: 兩階段微調含（不做 citation）

本檔案說明要如何讓 Codex（或其他代理）在現有專案中實作：

1. 兩階段的 LLM 微調流程（Stage 1：純分類；Stage 2：從 teacher 蒸餾 explanation）。
2. 第二階段如何挑選要產生 pseudo explanation 的子集樣本（盡量 cover 所有 ticker）。
3. 如何使用 teacher 模型（Llama-3.1-70B-Instruct，量化版）產生 pseudo explanation，並儲存為 `.jsonl`（區分 train / valid）。
4. 第二階段訓練資料的格式、loss 設計與訓練方式（不再訓練 citation / used_event_ids）。
5. 模型儲存與命名規範。

> 重點：本版本**不再訓練 citation / used_event_ids**，研究主軸改為「讓 8B 小模型學會產生合理、具金融直覺的解釋文字」，prediction 仍為主要任務。

---

## 0. 背景與目標

- **Base model**：`llama-3.1-8B-Instruct`，使用 LoRA / adapter 微調。
- **任務**：根據
  - 目標股票最近 N 日價格（% return）、
  - 目標公司與（未來可加入）相關公司在該時間窗口內的新聞事件，
  預測**隔日股價漲跌（UP / DOWN）**，並輸出一段**短解釋文字（reason）**。

- **最終輸出 JSON schema（推論時）**：

  ```json
  {
    "prediction": "UP" or "DOWN",
    "reason": "<short explanation>"
  }
  ```

  > 若現有 pipeline 仍保留 `used_event_ids` 欄位，可視為 optional / deprecated 欄位，不需在本階段訓練與評估中使用。

- **現況問題**：
  - Stage 1 只訓練 `"prediction"`，導致模型學到「`reason` = 空字串」是最安全的行為。
  - 沒有人工標註 explanation。

- **本檔目標**：設計一個兩階段微調流程，透過 teacher LLM 產生 pseudo explanation，將「合理解釋股價漲跌」的能力蒸餾到 8B 小模型，同時維持預測性能。

---

## 1. 整體流程總覽

Codex 應實作以下高層流程：

1. **Stage 1：分類微調（已完成，可保留實作）**
   - 輸入：價格 + 事件 + 任務描述的 prompt（可含 RAG 取得的相關公司新聞，視主程式而定）。
   - 輸出：只關注 `"prediction"`（UP / DOWN）。
   - Loss：只對 label 計算 cross-entropy。
   - 產出：`stage1` checkpoint（之後 Stage 2 的初始化權重）。

2. **Stage 2 前置：從 train set 中挑選一部分樣本產生 pseudo explanation**
   - 從原始 train split 中挑選一個子集（例如 5%–20%，目標 3k–10k 筆）：
     - 盡量 cover 每個 ticker（長尾股票也要有）。
     - 盡量平衡 UP / DOWN。
     - 同時包含事件數量多、事件數量少的樣本。
   - 生成 `pseudo_train.jsonl` 與 `pseudo_valid.jsonl`：
     - 每筆樣本包含原始欄位 + teacher 產生的 explanation。

3. **Stage 2：帶 pseudo explanation 的微調（解釋蒸餾）**
   - Base model：Stage 1 的 checkpoint（8B + LoRA）。
   - Data：
     - Input：與 Stage 1 相同的 prompt。
     - Target：teacher 產生的 JSON（`prediction` + `reason`），其中 `prediction` 會被覆寫為真實 label。
   - Loss 設計（multi-task）：
     - `L_cls`：對 `"prediction"` 的 token 計算 CE（使用 ground truth UP / DOWN）。
     - `L_lm`：對 `reason` 區段的 token 計算 LM loss（使用 teacher explanation）。
     - 總 Loss：`L = λ_cls * L_cls + λ_lm * L_lm`。

4. **模型儲存**
   - Stage 2 完成後，儲存：
     - 最佳 validation 表現的 LoRA checkpoint（含 config）。
     - 對應的 tokenizer / prompt config。
   - 命名建議：`stare-llama8b-lora-stage2-expl-v1` 類似格式。

---

## 2. Stage 2 要產生 pseudo explanation 的樣本選取策略

### 2.1 資料前提

假設原始 train split 檔案為：

- `datasets/<DATASET_NAME>/sft_pairs_train.jsonl`  
  每行一筆 JSON，至少包含：
  - `ticker`: string
  - `label`: "UP" or "DOWN"
  - `price_context`: string（已組好的 D-5 ~ D-1 價格描述）
  - `events`: list of objects 或已組裝的 `[EVENTS]` 區塊文字
  - `prediction_date`: string（D0 日期）
  - 其他 metadata（可保留）

實際欄位名稱依 repo 而定，Codex 應查閱程式碼並對齊。

### 2.2 目標 pseudo 標註量

定義：

- `N_total` = train split 總樣本數。
- `N_target` = 目標 pseudo 標註數。

建議：

- 初始設定（中等規模資料集）：
  - `N_target = max(3000, int(0.1 * N_total))`  
  - 若 `N_total` 很小（< 10k），可提高到 30%–50%。

### 2.3 子集抽樣規則

Codex 應實作一個腳本，例如：

```bash
python tools/pseudo_explanation/select_subset.py \
  --input datasets/CMIN/sft_pairs_train.jsonl \
  --output datasets/CMIN/sft_pairs_train_pseudo_candidates.jsonl \
  --n_target 5000 \
  --min_per_ticker 50
```

抽樣邏輯建議如下：

1. **統計各 ticker 的樣本數與 label 分佈**
   - 對每個 `ticker`，統計：
     - `count_up`
     - `count_down`
     - `count_total`

2. **保證每個 ticker 至少被 cover**
   - 設定 `MIN_PER_TICKER`（例如 20 或 50）：
     - 若 `ticker` 總樣本數 ≥ `MIN_PER_TICKER`：
       - 從該 ticker 中隨機抽 `MIN_PER_TICKER` 筆（UP / DOWN 儘量平衡）。
     - 若樣本數 < `MIN_PER_TICKER`：
       - 全部收進 pseudo 候選。

3. **平衡 UP / DOWN**
   - 在 tickers 被 cover 的前提下，計算目前 pseudo 候選集中：
     - `num_up_selected`
     - `num_down_selected`
   - 若某一類比例過低（例如 < 40%），優先從未選中的樣本中抽該類別。

4. **事件數量多寡兼顧**
   - 可以在候選集內分 bucket：
     - `few_events`: `len(events) <= 3`
     - `mid_events`: 4–7
     - `many_events`: >= 8
   - 讓每個 bucket 都有一定比例的樣本被抽中（例如各占 20% / 40% / 40%）。

5. **控制總樣本數不超過 N_target**
   - 若按上述規則抽樣後超過 `N_target`：
     - 進行隨機下採樣保持分佈平衡。

輸出：

- `sft_pairs_train_pseudo_candidates.jsonl`  
  每行一筆原始樣本，另增加欄位：
  - `pseudo_candidate`: true

之後 pseudo explanation 生成就只跑在這個檔案上。

---

## 3. teacher 模型產生 pseudo explanation

### 3.1 Teacher 模型設定

- 模型名稱（建議）：`meta-llama/Meta-Llama-3.1-70B-Instruct`（量化版）
- 推論後端可選：
  - vLLM
  - llama.cpp
  - HF Transformers + `bitsandbytes`（若資源允許）

Codex 應實作一個生成腳本，例如：

```bash
python tools/pseudo_explanation/generate_with_teacher.py \
  --model_name_or_path /path/to/llama-3.1-70B-instruct-quant \
  --input datasets/CMIN/sft_pairs_train_pseudo_candidates.jsonl \
  --output datasets/CMIN/pseudo_explanations_train_raw.jsonl \
  --max_new_tokens 512 \
  --batch_size 4
```

### 3.2 Prompt template 設計（不要求 citation）

為了穩定產出 JSON 並專注在 explanation，建議 prompt 類似：

```text
You are a financial analyst model. Given recent price movements and news for a target stock, 
you must PREDICT next-day movement and EXPLAIN the prediction in a short paragraph.

Follow these rules:
- Only use the given information (price context and events).
- The prediction must be \"UP\" or \"DOWN\".
- The reason MUST be a short explanation (1–3 sentences).
- Focus on financial logic (sentiment, earnings, macro conditions, related companies).
- Do NOT add any information that clearly contradicts the input.
- Output valid JSON only, no extra text.

<INPUT>
Target stock: {ticker}
Prediction date (D0): {date}

[PRICE CONTEXT]
{price_context}

[EVENTS]
{events_block}

[TASK]
Predict next-day movement (UP or DOWN) for the target stock (vs D-1 close) and explain your reasoning.
</INPUT>

[OUTPUT JSON]
{
  "prediction": "UP" or "DOWN",
  "reason": "<short explanation>"
}
```

其中：

- `{events_block}` 例如：

  ```text
  Target firm news:
  (1) [2019-06-05] ...
  (2) [2019-06-07] ...
  ...
  ```

- 若已加入相關公司新聞，則可加上：

  ```text
  Related firms news:
  (A1) [2019-06-05] ...
  (A2) [2019-06-06] ...
  ```

Codex 應確保：

- 每次只生成一個完整 JSON；
- 模型輸出中只保留 `{ ... }` 那段，前後多餘文字要 strip。

### 3.3 teacher 輸出格式與 .jsonl 儲存

`generate_with_teacher.py` 應讀入 `sft_pairs_train_pseudo_candidates.jsonl`，  
對每行 input 產生對應的 pseudo explanation，輸出為新的 `.jsonl`：

每行結構建議：

```jsonc
{
  "ticker": "AAPL",
  "label": "UP",
  "prediction_date": "2019-06-11",
  "price_context": "...",
  "events_block": "...",   // 用於 prompt 的文字
  "teacher_raw_output": "{ ... }",   // teacher 原始 JSON 字串
  "teacher_parsed": {
    "prediction": "UP",
    "reason": "...."
  }
}
```

Codex 應在生成腳本中：

1. 嘗試 `json.loads` 解析 teacher 回傳的 JSON；  
2. 若解析失敗：
   - 嘗試簡單清理（例如找最外層 `{ ... }` 再 parse）；
   - 若仍失敗，將該筆標記為 `parse_error: true`，之後在訓練前濾掉。
3. 若解析成功：
   - 保留 `teacher_parsed["prediction"]` 與 `teacher_parsed["reason"]`；
   - 不再處理或檢查 citation / used_event_ids（本版本不使用）。

### 3.4 train / valid 分割

為了 Stage 2 訓練與監控，需產生 pseudo train / pseudo valid：

- 若原始資料已有 train / valid split：
  - 優先根據原始 split 進行 pseudo subset 抽樣與 pseudo 生成。
- 若目前只有 train：
  - 從 pseudo 標註後的樣本中，按 ticker / label 分 stratified split：
    - 例如 90% 作為 `pseudo_train.jsonl`，10% 作為 `pseudo_valid.jsonl`。

輸出檔案建議：

- `datasets/CMIN/pseudo_train.jsonl`
- `datasets/CMIN/pseudo_valid.jsonl`

每行保留：

```jsonc
{
  "ticker": "...",
  "label": "UP",
  "prediction_date": "...",
  "price_context": "...",
  "events_block": "...",
  "teacher_prediction": "UP",
  "teacher_reason": "..."
}
```

---

## 4. 第二階段微調：資料格式與 loss 設計（無 citation）

### 4.1 訓練資料組裝

Stage 2 的核心想法：

- **Input prompt**：與 Stage 1 相同（不要把 teacher output 放進 input，只作 target）。
- **Target JSON**：以 teacher 的 JSON 為基礎，但：
  - `"prediction"` 欄位覆寫成真實 label（`label`）。
  - `"reason"` 使用 teacher 產生的 explanation。

建議在前處理時先組好：

```jsonc
{
  "input": "<PROMPT STRING>",
  "target_json": {
    "prediction": "UP",
    "reason": "近期股價連續上漲，加上多則利多新聞，市場情緒偏多。"
  },
  "label": "UP",              // ground truth
  "meta": { ... }             // (ticker, date, 等資訊)
}
```

Codex 應實作 `prepare_stage2_sft_data.py` 將 `pseudo_train.jsonl` / `pseudo_valid.jsonl` 轉為上述格式。

### 4.2 tokenization 與標註 span

訓練時會將 `target_json` 序列化為字串，例如：

```text
{
  "prediction": "UP",
  "reason": "近期股價連續上漲，加上多則利多新聞，市場情緒偏多。"
}
```

模型輸出是整段 JSON。為了做 multi-task loss，需要知道：

- 哪些 token 對應到 `prediction`（UP / DOWN）；
- 哪些 token 對應到 `reason`。

簡化作法（建議）：

1. 在序列化時，固定 `prediction` 的位置，例如：

   ```text
   { "prediction": "UP", "reason": "..." }
   ```

2. tokenization 後，尋找 `"prediction": "` 後第一個 `UP` 或 `DOWN` token 的索引 `idx_pred`。
3. 定義：
   - `prediction_token_idx = idx_pred`
   - `reason_token_range = [idx_reason_start, idx_end_of_json]`  
     （`idx_reason_start` 可設為 `"reason"` value 的第一個非空白 token）。

Codex 在 training loop 中：

- **CE loss**：只在 `prediction_token_idx` 上計算（用 ground truth label）。
- **LM loss**：只在 `reason_token_range` 之內的 token 上計算（用 teacher target）。

### 4.3 Loss 定義

令：

- `y_true`：ground truth label（UP / DOWN）。
- `y_pred_logits`：對應 `prediction_token_idx` 的 logits。
- `T_reason`：reason 對應的 target token 序列。
- `logits_reason`：同位置的模型 logits。

則：

```text
L_cls = CrossEntropy(y_pred_logits, y_true)

L_lm  = -平均( log P(T_reason[i] | T_reason[:i], input) )   // 標準 LM NLL
L     = λ_cls * L_cls + λ_lm * L_lm
```

初始設定：

- `λ_cls = 1.0`
- `λ_lm = 1.0`

若觀察到 Stage 2 之後分類表現下降太多，可調整為：

- `λ_cls = 2.0 ~ 3.0`
- `λ_lm = 1.0`

### 4.4 Stage 2 訓練腳本

Codex 應實作類似指令：

```bash
python train_stage2_with_explanations.py \
  --base_model outputs/stage1_llama8b_lora \
  --train_file datasets/CMIN/stage2_train_sft.jsonl \
  --validation_file datasets/CMIN/stage2_valid_sft.jsonl \
  --output_dir outputs/stage2_llama8b_lora_expl \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --lambda_cls 1.0 \
  --lambda_lm 1.0 \
  --logging_steps 50 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 3
```

`train_stage2_with_explanations.py` 主要負責：

1. 載入 Stage 1 的 LoRA 權重作為起點。
2. 讀入 Stage 2 sft 檔（`input` + `target_json`）。
3. 在 `forward` 中：
   - tokenize input / target；
   - 計算 `prediction_token_idx` 與 `reason_token_range`；
   - 分別計算 `L_cls` 與 `L_lm`；
   - 回傳 `L = λ_cls * L_cls + λ_lm * L_lm` 給 optimizer。

---

## 5. 模型儲存與命名

### 5.1 輸出目錄結構建議

在 `OUTPUTS_DIR` 下，為 Stage 2 建以下結構：

```text
outputs/
  stage1_llama8b_lora/              # 已存在，純分類微調結果
  stage2_llama8b_lora_expl/         # 新增：含 explanation 的模型
    adapter_config.json
    adapter_model.bin (或 safetensors)
    tokenizer.json / tokenizer.model
    training_args.bin
    config_stage2.json              # 包含 λ_cls, λ_lm, 使用的 pseudo 資料版本等
```

建議在 `config_stage2.json` 中記錄：

- 使用的 teacher 模型名稱與 commit（例如 `llama-3.1-70b-instruct-quant`）。
- pseudo 資料來源檔案（`pseudo_train.jsonl` / `pseudo_valid.jsonl`）。
- 抽樣比例與 `N_target`。
- loss 權重：`lambda_cls`, `lambda_lm`。
- 最佳 checkpoint 的 validation 指標（例如 MCC, Accuracy）。

### 5.2 推論使用建議

在 inference pipeline 中：

- 預設使用 `stage2_llama8b_lora_expl` 這個權重：
  - 一樣輸入 Stage 1 時使用的 prompt；
  - 解碼時要完整 decode 出 JSON；
  - 然後 parse：
    - `prediction`（UP / DOWN）直接用於預測；
    - `reason` 顯示給使用者。

若需要 fallback（例如想比較有無 explanation 微調的差別）：

- 保留使用 stage1 模型的推論 entry，  
  但那個版本預期 `reason` 可能為空或較差。

---

## 6. Codex 實作任務摘要

給 Codex 的 TODO 清單（高優先）：

1. **資料子集選取**
   - [ ] 實作 `tools/pseudo_explanation/select_subset.py`  
     - 輸入：原始 `sft_pairs_train.jsonl`  
     - 輸出：`sft_pairs_train_pseudo_candidates.jsonl`  
     - 規則：cover 每個 ticker、平衡 UP / DOWN、事件數量多寡皆有。

2. **teacher 生成 pseudo explanation**
   - [ ] 實作 `tools/pseudo_explanation/generate_with_teacher.py`  
     - 輸入：pseudo candidates  
     - 使用：`llama-3.1-70B-Instruct` 量化模型  
     - 產生 JSON 格式的 prediction + reason  
     - 輸出：`pseudo_explanations_train_raw.jsonl`，並解析為 `pseudo_train.jsonl` / `pseudo_valid.jsonl`。

3. **Stage 2 訓練資料前處理**
   - [ ] 實作 `tools/pseudo_explanation/prepare_stage2_sft_data.py`  
     - 輸入：`pseudo_train.jsonl`, `pseudo_valid.jsonl`  
     - 輸出：`stage2_train_sft.jsonl`, `stage2_valid_sft.jsonl`  
     - 每筆包含 `input` + `target_json` + `label`。

4. **Stage 2 微調腳本**
   - [ ] 實作 `train_stage2_with_explanations.py`  
     - 使用 Stage 1 checkpoint + LoRA  
     - 定義 multi-task loss：`L = λ_cls * L_cls + λ_lm * L_lm`  
     - 儲存最佳 checkpoint 至 `outputs/stage2_llama8b_lora_expl/`。

5. **文件與 config**
   - [ ] 在 `outputs/stage2_llama8b_lora_expl/` 中寫入 `config_stage2.json`，記錄 pseudo 資料設定、teacher 模型與 loss 權重。
   - [ ] 更新頂層 README / pipeline 說明，引導使用者在 inference 時改用 stage2 模型取得 explanation（不強制 citation）。

以上為**移除 citation / grounding 任務後**的兩階段微調與 pseudo explanation 產生規劃，Codex 可依本檔案逐項落實實作。

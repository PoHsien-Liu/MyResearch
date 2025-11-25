# /baselines/FinGPT/AGENTS.md

## 📘 Baseline：FinGPT（純推論式基準）

### 🎯 目的
本 baseline 用於評估 **FinGPT** 作為 **金融領域基礎模型（Foundation Model）**，  
在 **未經任何微調（zero-shot）** 狀態下，透過 **prompt engineering** 進行股價漲跌預測的表現。

此方法僅依靠模型對金融文本的理解與推理能力，  
從近五天的新聞／推文內容中，**直接推測隔日股價漲跌方向與幅度**。  
不使用任何股價數值、技術指標或外部資料。

---

## 🧩 功能說明

### 1. 輸入資料  
來自統一的 `DataLoader`（位於 `MyResearch/dataloader.py`），  
只會讀取 **測試集（test split）**。

**每筆樣本格式：**
```python
{
    "ticker": str,
    "date": "YYYY-MM-DD",           # 預測目標日（T）
    "label": 0|1,                   # 真實標籤（Negative / Positive）
    "texts_context": List[str],     # 前 seq_len 天的文本內容列表（預設 5 天）
}
```

- **文本來源**：依資料集不同，可能是推文或新聞。
- **時間窗**：預設取前 5 天 (`seq_len=5`)。
- **不包含股價或報酬資料**。

---

### 2. 模型說明
- **模型名稱**：FinGPT（LoRA adapter 掛載於基座模型上）  
- **可用組合：**
  - `FinGPT/fingpt-mt_llama3-8b_lora`（基座：`meta-llama/Meta-Llama-3-8B`）
  - `FinGPT/fingpt-sentiment_llama2-13b_lora`
  - 由參數 `--base_model` 指定使用的基座模型。
- **模式**：僅推論（inference-only），不進行參數更新。  

**模型載入範例：**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(base, fingpt_lora_path).eval()
tokenizer = AutoTokenizer.from_pretrained(base_model)
```

---

### 3. Prompt 模板

#### System Prompt
```
You are a financial analyst LLM specialized in reading recent market news and predicting short-term stock movement.
Given the recent 5 days of news about a company, reason step by step about the likely impact on the stock and
then forecast the next trading day’s stock return direction and magnitude.
```

#### User Prompt
```
Ticker: {ticker}
Prediction date: {date}

Recent {seq_len} days news:
[Day {t-5}]
- {text_1}
- {text_2}
...
[Day {t-1}]
- {text_n}

Now, based on the above recent news only, predict the next trading day’s stock return direction and magnitude.

Rules:
- Focus on what may happen on the next day, not just summarize the history.
- The next day stock return does not have to follow previous days.
- First, reason step by step (your analysis).
- Then, in the last line, output ONLY in this exact format:

Stock Return: [number]% ([up/down])
```

> 此 prompt 模仿 TDMLLM 的格式，但刪除 Summary / Keywords，  
> 僅保留「文本內容 + 最終預測輸出」，避免引入額外摘要模組。

---

### 4. 輸出格式

依照專案根目錄之統一規範，輸出於：
```
outputs/results/{dataset}/{method}/{model}/{experiment_name}/
```

#### 檔案 1：`predictions.jsonl`
每行為一筆樣本：
| 欄位 | 說明 |
|------|------|
| `sample_id` | `{ticker}_{date}` |
| `dataset` | 資料集名稱（ACL18 / SEP / CMIN / SAMPLE） |
| `method` | `"FinGPT"` |
| `model` | 使用的基座模型（例如 `"meta-llama/Meta-Llama-3-8B"`） |
| `experiment_name` | 實驗名稱 |
| `ticker` | 股票代號 |
| `prediction_date` | 預測日期 |
| `ground_truth` | `"Positive"` / `"Negative"` |
| `prediction` | `{"label": "Positive"/"Negative", "confidence": null}` |
| `raw_response` | 模型完整輸出 |
| `prompts` | `{"system": str, "user": str}` |
| `timing` | `{"latency_ms": float}` |

#### 檔案 2：`predictions.csv`
扁平化版本（至少包含）：
```
sample_id,ticker,prediction_date,y_true,y_pred,model,method,dataset,experiment_name
```

#### 輸出解析規則
- 從輸出最後一行擷取符合：
  ```
  Stock Return: <float>% (<direction>)
  ```
- `<direction>`:
  - `(up)` → `"Positive"`
  - `(down)` → `"Negative"`
- `<float>` 為可選項（不影響分類評估，可另存分析）。

---

### 5. 評估方式

採用統一評估模組，產出以下指標：
- **分類指標**：Accuracy、MCC、Precision、Recall、F1、Confusion Matrix。
- **解釋性分析（選用）**：reasoning 長度、詞彙覆蓋率、情感一致性等。

結果存於：
```
outputs/results/{dataset}/FinGPT/{model}/{experiment_name}/eval.json
```

---

## ⚙️ 指令範例

```bash
python -m baselines.FinGPT.run_predict   --dataset_name ACL18   --base_model meta-llama/Meta-Llama-3-8B   --fingpt_lora FinGPT/fingpt-mt_llama3-8b_lora   --seq_len 5   --batch_size 1   --experiment_name fingpt_zero_shot_acl18   --seed 42
```

---

## 🚀 效能與資源控制

- 批次大小：以 `--batch_size` 單一參數控制吞吐與顯存占用（不再提供 `--generation_chunk_size`）。
- 顯卡選擇：可用 `CUDA_VISIBLE_DEVICES` 指定可見 GPU；使用 4-bit 並加上 `--device_map auto` 可自動分配至可見裝置。
- 上下文長度：以 `--seq_len` 與 `--max_texts_per_day` 控制 prompt 長度，避免超出模型 context 窗造成 0 新字元。
- 建議基座：優先使用 Instruct 版（如 `meta-llama/Meta-Llama-3-8B-Instruct`）以獲得穩定對話/生成行為。

---

## 🧠 實作目的

- 建立一個 **FinGPT 的零樣本（Zero-Shot）基準**，  
  測試其金融常識與文本理解能力能否直接產生合理預測。
- 作為 **TDMLLM 與 STARE 方法的對照基準**。  
- 評估 LLM 僅靠 prompt 能達到的下限表現。

---

## 🔑 實作要點

1. **推論模式（Inference Only）**：關閉梯度與參數更新。
2. **輸入來源**：統一 DataLoader 回傳的 `texts_context`（近 5 天文本）。  
3. **輸出格式**：嚴格遵守「Stock Return: [number]% ([up/down])」。  
4. **固定隨機性**：設定 `temperature=0.0`，確保可重現。  
5. **完整記錄**：紀錄 latency、原始輸出、prompt。  
6. **不包含摘要或關鍵字生成**：直接以原始文本推論。

---

## 🧰 實作步驟（交給 Codex）

1. 新增檔案 `/baselines/FinGPT/run_predict.py`。  
2. 建立 CLI 參數（`--dataset_name`、`--base_model`、`--fingpt_lora`、`--seq_len`、`--experiment_name`...）。  
3. 建立輸出目錄：  
   `outputs/results/{dataset}/FinGPT/{model}/{experiment_name}/`。  
4. 載入模型與 tokenizer。  
5. 從 DataLoader 讀取測試樣本，取得近 5 天文本。  
6. 組合 system / user prompt。  
7. 推論生成（`max_new_tokens=128`, `temperature=0.0`）。  
8. 解析最後一行，提取 `up/down` 與數值。  
9. 寫入 `predictions.jsonl` 與 `predictions.csv`。  
10. 呼叫統一 evaluator 生成 `eval.json`。

---

## ✅ 驗收條件（Acceptance Criteria）

| 類別 | 驗收內容 |
|------|-----------|
| **功能** | 能在所有測試樣本上正確進行推論。 |
| **Prompt 格式** | 完全符合「Recent 5 days news + Stock Return」格式。 |
| **輸出結構** | 符合 `predictions.jsonl` / `predictions.csv` 統一格式。 |
| **方向解析** | 能正確擷取 `(up)` / `(down)`；異常輸出需安全處理。 |
| **評估結果** | 正確產出 `eval.json` 並包含所有分類指標。 |
| **重現性** | `args.json`、`run.log` 記錄模型資訊、LoRA 路徑、seed、環境。 |

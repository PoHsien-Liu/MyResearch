# /baselines/TDMLLM/AGENTS_summarizer.md

## 🎯 任務目標：讓 TDMLLM 的摘要流程支援批次推論（Batch Inference）

目前 TDMLLM 的做法是：

- 在 `DataLoader` 中，針對每一個 `(ticker, date)`：
  - 讀取當天所有文本（tweet/news）
  - 呼叫 `Summarizer.get_summary(...)` 做一次 LLM 摘要
- 等於「一天一呼叫」，無法利用 LLM 的 **批次推論 (batch inference)** 優勢  
  → 造成前處理（生成摘要）非常耗時。

本文件要你在 **TDMLLM 模組內部** 完成以下優化：

1. 在 `Summarizer` 中新增 **批次摘要介面** `summarize_batch(...)`。
2. 修改 TDMLLM 的 `DataLoader`，改為：
   - 一次收集多筆「摘要任務」
   - 呼叫 `summarize_batch` 進行批次推論
   - 並支援摘要快取（記憶體 + 磁碟）
3. 保持原有功能與輸出不變（**摘要內容語義要等價**），  
   但 **顯著減少 LLM 呼叫次數、縮短整體運算時間**。

---

## 📦 範圍與檔案位置

- 主要修改檔案：
  - `baselines/TDMLLM/summarize_module/summarizer.py`
  - `baselines/TDMLLM/dataloader.py`
- 不需改動：
  - TDMLLM 的主流程（例如 train / predict 的 entry point）
  - 其他 baseline

---

## 🧩 介面與設計

### 1. SummaryJob 結構

在 TDMLLM 內部，我們統一把「需要摘要的一天」表示為一個 job：

SummaryJob = {
    "ticker": str,
    "date": "YYYY-MM-DD",
    "texts": List[str],  # 當天所有文本（可先截斷到 summary_max_tweets）
}

### 2. Summarizer：新增批次摘要介面

在 summarizer.py 中：

保留既有的 get_summary(...) 介面（向後相容）。

新增以下方法：

class Summarizer:
    def summarize_batch(self, jobs: list[dict], batch_size: int = 8) -> dict:
        """
        Args:
            jobs: List of SummaryJob dicts:
                  {
                    "ticker": str,
                    "date": "YYYY-MM-DD",
                    "texts": List[str]
                  }
            batch_size: 一次丟給 LLM 的 job 數量。

        Returns:
            summaries: dict keyed by (ticker, date) -> summary_str
                       例如 {("AAPL","2015-10-12"): "..." }
        """

summarize_batch 行為規格

對每個 job：

用 job["texts"] 建出對應的 user prompt：

可重用現有的單筆摘要 prompt 邏輯（或抽出 build_user_prompt(job) helper）。

system prompt 可以共用同一段字串。

每次取 batch_size 個 jobs：

準備：

system_list = [system_prompt] * len(batch)
user_list = [build_user_prompt(job) for job in batch]


呼叫底層的 LLM adapter：

若已有 adapter.batch_generate(system_list, user_list, **gen_kwargs) 介面 → 優先使用。

若沒有，可以 fallback 為對 generate 的 for 迴圈（但目標是支援真正的 batch_generate）。

把模型輸出整理為純文字 summary，去掉多餘空白。

將每個 job 的結果暫存在：

results[(ticker, date)] = summary_str


函式結束時回傳 results。

### 3. 摘要快取設計（Summarizer 內部）

為避免重複計算摘要，需在 Summarizer 中加入快取邏輯：

快取層級：

記憶體快取：self.summary_cache: Dict[str, str]

檔案快取：寫入到 OUTPUTS_DIR/cache/summaries/TDMLLM/...

快取 key

模型相關性：同一個摘要結果只對「固定 dataset + method + base_model」有效。

建議檔案路徑格式：

{OUTPUTS_DIR}/cache/summaries/TDMLLM/{dataset_name}/{safe_model_name}/{ticker}/{date}.json


其中：

dataset_name 來自 args（例如 ACL18 / SEP / CMIN / SAMPLE）。

safe_model_name：將 base_model 的 / 替換成 __，例如：

meta-llama/Meta-Llama-3-8B-Instruct → meta-llama__Meta-Llama-3-8B-Instruct

檔案內容範例
{
  "dataset": "ACL18",
  "method": "TDMLLM",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "ticker": "AAPL",
  "date": "2015-10-12",
  "summary": "...."
}

Summarizer 需提供的快取函式
def get_cached_summary(self, ticker: str, date: str) -> str | None:
    """
    先檢查 self.summary_cache，
    若無則檢查檔案快取，若存在則讀取並寫回 self.summary_cache。
    """

def save_summary(self, ticker: str, date: str, summary: str) -> None:
    """
    更新 self.summary_cache 並寫入檔案快取。
    """


summarize_batch 在寫入結果時，需先呼叫 save_summary(...)。
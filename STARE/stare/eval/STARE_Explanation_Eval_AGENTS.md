STARE_Explanation_Eval_AGENTS.md
================================
STARE — 解釋文本評估（Explanation Evaluation）模組說明

本文件說明：如何使用開源 LLM 對 STARE 各 baseline / 方法產生的「股價預測 + 解釋文本」進行自動評分，並支援：
- 只評估 top-1 stocks（per dataset）或所有股票；
- 專注在「預測正確樣本」的解釋品質；
- 支援多資料集（SEP / StockNet / CMIN-US）。

------------------------------------------------------------
1. 目標與輸出
------------------------------------------------------------

1.1 目標

- 給定：
  - predictions.csv：各模型對測試集的預測結果與生成的解釋文本；
  - ground truth 標籤；
- 使用開源免費 LLM（例如：Qwen2.5-72B-Instruct、Llama-3.1-70B-Instruct 等）依照 SEP 的 rubric 對解釋文本評分。
- 支援：
  - 挑選 top-1 stocks（per dataset）或所有股票進行評估。

1.2 輸出

每次 explanation eval 實驗至少產生：

- explanation_eval_samples.jsonl
  - 每一列對應一個樣本；
  - 包含：sample_id, dataset, ticker, date, y_true, y_pred, stock_scope, is_correct, llm_backend, llm_model, metric_scores, overall_comment, raw_response, ...
- explanation_eval_summary.json
  - 各模型 / 方法、各 dataset、各 stock_scope 的平均分數與統計量；
  - 方便在總表或圖表中比較。

------------------------------------------------------------
2. 輸入檔案格式：predictions.csv
------------------------------------------------------------

預期每個實驗資料夾底下有一個 predictions.csv，欄位至少包含：

- sample_id：唯一 ID，可由 (dataset, ticker, date) 組合。
- dataset：例如 SEP, STOCKNET, CMIN-US。
- ticker：股票代碼（與 dataset 定義一致）。
- date：預測日期（YYYY-MM-DD）。
- y_true：真實漲跌標籤（例如 positive / negative / Unknown(視為錯誤標籤)）。
- y_pred：模型預測結果（與 y_true 同一標籤空間）。
- raw_response：模型生成的自然語言解釋。
- model：例如 TDMLLM, FinGPT, STARE, STARE-RAG。
- method：方法 / 設定名稱（例如 base, rag_cross_firm, rag_topk5）。
- experiment_name：實驗名稱，用於對齊不同輸出。

------------------------------------------------------------
3. Top-1 stocks 與 stock scope 設計
------------------------------------------------------------

3.1 CLI 參數：--stock_scope

在 explanation eval 的主程式（例如 STARE/stare/eval/explanation_eval_main.py）中，支援一個關鍵參數：

- --stock_scope {all, top1}
  - all：使用該 dataset 的所有股票。
  - top1：只使用「top-1 stocks 集合」（依 dataset 而定）。

3.2 Top-1 設定檔：configs/top_stocks.py

新增一個 config 檔：

- 路徑：STARE/stare/configs/top_stocks.py
- 功能：集中管理「各 dataset 的 top-1 stocks 選取方式與結果」。

範例架構（Python pseudo-code）：

from dataclasses import dataclass
from typing import Literal, List, Dict, Optional, Set

ScopeMode = Literal["all", "top1"]
Top1Mode = Literal["fixed_list", "by_sector_tweet_volume", "overall_top_k_news"]

@dataclass
class Top1Config:
    mode: Top1Mode
    tickers: Optional[List[str]] = None
    k: Optional[int] = None   # 用於 overall_top_k_news

DATASET_TOP1_CONFIG: Dict[str, Top1Config] = {
    # SEP: 已知 11 個產業，每個產業的第一檔視為 top-1
    "SEP": Top1Config(
        mode="fixed_list",
        tickers=[
            "BHP",   # Basic Materials
            "BRK-A", # Financial Services
            "WMT",   # Consumer Defensive
            "NEE",   # Utilities
            "XOM",   # Energy
            "AAPL",  # Technology
            "AMZN",  # Consumer Cyclical
            "AMT",   # Real Estate
            "UNH",   # Healthcare
            "GOOG",  # Communication Services
            "UPS",   # Industrials
        ],
    ),

    # StockNet: 從 StockTable + tweet 資料自動算出每個 sector 的 top-1
    "STOCKNET": Top1Config(
        mode="by_sector_tweet_volume",
    ),

    # CMIN-US: 沒有 sector，取新聞量前 K 檔股票
    "CMIN-US": Top1Config(
        mode="overall_top_k_news",
        k=11,  # 取新聞量最多的 11 檔，與 SEP 的 11 檔對齊
    ),
}

另建 STARE/stare/eval/top_stocks_loader.py，實作：

def load_top1_tickers(dataset: str, scope: ScopeMode) -> Set[str]:
    ...

邏輯：

- 若 scope == "all"：回傳空集合（代表不過濾）。
- 若 scope == "top1"：
  - 若 mode == "fixed_list"：直接回傳 config.tickers。
  - 若 mode == "by_sector_tweet_volume"：
    - 若 cache json 存在（例如 stocknet_top1.json）：讀取並回傳。
    - 否則：
      1. 讀取 StockNet 的 StockTable 與 tweet raw 檔。
      2. 依 sector 分組，統計各 ticker 的 tweet 筆數。
      3. 每個 sector 選 tweet 數量最多的 1 檔。
      4. 存成 json，供下次直接使用。
  - 若 mode == "overall_top_k_news"：
    - 若 cache json 存在（例如 cmin_us_top1.json）：讀取並回傳。
    - 否則：
      1. 統計 CMIN-US 中各 ticker 的新聞筆數。
      2. 取新聞數量最多的前 k 檔。
      3. 存成 json，供下次直接使用。

------------------------------------------------------------
4. 篩選樣本邏輯
------------------------------------------------------------

4.1 只評估預測正確樣本

CLI 參數：

- --only_correct {true,false}（預設 true）
  - true：只保留 y_true == y_pred 的樣本。
  - false：保留全部樣本（可做 ablation）。

在 STARE/stare/eval/filters.py 中實作：

def filter_by_correct(df, only_correct: bool):
    if not only_correct:
        return df
    return df[df["y_true"] == df["y_pred"]]

4.2 依 stock scope 過濾

from .top_stocks_loader import load_top1_tickers

def filter_by_stock_scope(df, dataset: str, stock_scope: str):
    if stock_scope == "all":
        return df
    top1_tickers = load_top1_tickers(dataset, stock_scope)
    return df[df["ticker"].isin(top1_tickers)]

整體篩選順序建議：

1. 先依 dataset、model、method 等條件濾出該次 eval 需要的 subset。
2. 呼叫 filter_by_correct(df, only_correct=True)。
3. 呼叫 filter_by_stock_scope(df, dataset, stock_scope)。

------------------------------------------------------------
5. LLM 評分 Prompt 與後端設計
------------------------------------------------------------

5.1 System prompt

放在 STARE/stare/eval/prompt_template.py：

SYSTEM_PROMPT = """
You are an expert financial analyst and explanation evaluator.

Your task is to grade natural language explanations of stock price movements.
For each explanation, you will carefully read:

1. The stock, date, and the relevant news or social media texts.
2. The model's predicted price movement (e.g., UP or DOWN).
3. The model's explanation for that prediction.

You must score the explanation on several metrics using a 1–7 integer scale,
where:
- 1 = very poor
- 4 = moderate / acceptable
- 7 = excellent

Focus only on the information provided in the prompt. Do not rely on your own
external knowledge of the real world beyond what is explicitly stated.
Do not speculate about facts that are not mentioned.
"""

預設不要把 y_true 放進 prompt，只提供：
- ticker / date
- context_texts
- predicted_movement（由 y_pred 映射）
- explanation

5.2 User prompt builder

在同一個檔案中實作：

def build_user_prompt(
    ticker: str,
    date: str,
    context_texts: str,
    predicted_movement: str,
    explanation: str,
) -> str:
    ...

模板內容核心是：

- Section A：Stock & Context（ticker, date, context_texts）
- Section B：Model Prediction & Explanation
- Section C：10 個評分指標（Relevance, Financial Metrics, Global Factors, ...）
- Section D：要求輸出固定 JSON 結構（metric_scores + overall_comment）

------------------------------------------------------------
6. Judge backend（開源 LLM）
------------------------------------------------------------

在 STARE/stare/eval/judge_backends.py 中實作：

from typing import Literal

BackendName = Literal["openai", "gemini", "qwen", "llama"]

def call_judge_backend(
    backend: BackendName,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    ...

- backend="qwen"：呼叫本地 Qwen2.5-72B-Instruct（例如透過 OpenAI 相容 API）。
- backend="llama"：呼叫本地 Llama-3.1-70B-Instruct。
- backend="openai" / "gemini"：可留作未來擴充，先標 TODO 或 NotImplementedError。

------------------------------------------------------------
7. 評估流程 Runner
------------------------------------------------------------

在 STARE/stare/eval/runner.py 中實作兩個主要函式：

from typing import Dict, Any, List
from .prompt_template import SYSTEM_PROMPT, build_user_prompt
from .judge_backends import call_judge_backend, BackendName
import json
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> str:
    """從 LLM 輸出的文字中抽出 JSON 片段"""
    ...

def evaluate_single_explanation(
    backend: BackendName,
    model_name: str,
    sample_id: str,
    ticker: str,
    date: str,
    context_texts: str,
    y_true: str,
    y_pred: str,
    predicted_movement: str,
    explanation: str,
    extra_meta: dict | None = None,
) -> Dict[str, Any]:
    ...

在 evaluate_single_explanation 中：
- 使用 SYSTEM_PROMPT + build_user_prompt(...) 組成完整 prompt。
- 呼叫 call_judge_backend(...) 拿到文字輸出。
- 用 extract_json_from_text 取出 JSON，json.loads 解析。
- 補上 metadata：sample_id, ticker, date, y_true, y_pred, backend, judge_model_name, extra_meta。

def evaluate_batch(
    backend: BackendName,
    model_name: str,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ...

records 每筆至少要包含：
- sample_id, ticker, date, context_texts, y_true, y_pred, predicted_movement, explanation
可選：model, method, dataset, experiment_name

------------------------------------------------------------
8. CLI 入口：explanation_eval_main.py
------------------------------------------------------------

在 STARE/stare/eval/explanation_eval_main.py 中使用 argparse：

必要參數：

- --predictions_csv PATH
- --dataset {SEP,STOCKNET,CMIN-US}
- --stock_scope {all,top1}
- --only_correct {true,false}
- --eval_llm_backend {qwen,llama,openai,gemini}
- --eval_llm_model MODEL_NAME
- --output_dir PATH
- --experiment_name NAME（可選）

主要流程：

1. 讀進 predictions.csv 為 DataFrame。
2. 依 dataset / experiment_name / model / method 等做初步過濾（可視需求）。
3. 呼叫 filter_by_correct(df, only_correct)。
4. 呼叫 filter_by_stock_scope(df, dataset, stock_scope)。
5. 把 DataFrame 轉成 records list，欄位對齊 evaluate_batch(...) 需求。
6. 呼叫 evaluate_batch(...)。
7. 將結果：
   - 寫入 explanation_eval_samples.jsonl。
   - 聚合並寫入 explanation_eval_summary.json。

------------------------------------------------------------
9. 執行指令範例
------------------------------------------------------------

9.1 SEP：只評估 top-1 stocks + 正確樣本

python -m STARE.stare.eval.explanation_eval_main \
  --predictions_csv outputs/SEP/STARE/predictions.csv \
  --dataset SEP \
  --stock_scope top1 \
  --only_correct true \
  --eval_llm_backend qwen \
  --eval_llm_model Qwen2.5-72B-Instruct \
  --output_dir outputs/SEP/STARE/explanation_eval_top1 \
  --experiment_name SEP_STARE_top1_correct

9.2 StockNet：全部 stocks（非 top-1），只看正確樣本

python -m STARE.stare.eval.explanation_eval_main \
  --predictions_csv outputs/STOCKNET/STARE/predictions.csv \
  --dataset STOCKNET \
  --stock_scope all \
  --only_correct true \
  --eval_llm_backend llama \
  --eval_llm_model Llama-3.1-70B-Instruct \
  --output_dir outputs/STOCKNET/STARE/explanation_eval_all \
  --experiment_name STOCKNET_STARE_all_correct

9.3 CMIN-US：新聞量最多前 11 檔（top1），只看正確樣本

python -m STARE.stare.eval.explanation_eval_main \
  --predictions_csv outputs/CMIN-US/STARE-RAG/predictions.csv \
  --dataset CMIN-US \
  --stock_scope top1 \
  --only_correct true \
  --eval_llm_backend qwen \
  --eval_llm_model Qwen2.5-72B-Instruct \
  --output_dir outputs/CMIN-US/STARE-RAG/explanation_eval_top1 \
  --experiment_name CMIN_US_STARE_RAG_top1_correct


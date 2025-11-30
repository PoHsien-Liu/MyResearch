目的與範圍（TDMLLM 專案內規）
- 目標：重現並優化「TDMLLM（Temporal Data Meets LLM）」方法於本論文比較框架中，確保可重現、可比較、可切換 foundation model，並縮短端到端的實驗時間。
- 範圍：本檔案的規範僅適用於 `baselines/TDMLLM/` 目錄及其子目錄。當與根目錄 `AGENTS.md` 規範衝突時，以根目錄為準；其餘以本檔為準。

方法概覽
- 流程（高層）：
  1) 資料載入：以 `seq_len` 天視窗聚合每個 `(ticker, end_date)` 的推文；由價格資料產生二分類標籤（正/負）。
  2) 推文摘要：對視窗內逐日推文以 LLM 產生摘要，並快取（避免重複計算）。
  3) 公司描述：依 `ticker` 產生/載入公司描述，用於條件化預測指令。
  4) 預測生成：以公司描述 + 視窗摘要組 Prompt，使用 LLM 產出原始回覆，解析成 `Positive/Negative`。
  5) 評估與輸出：計算 ACC/MCC 等，寫出標準化輸出（見「輸出規範」）。
- 主要模組：
  - `common/data/loader.py`：`list_trading_days` / `get_record`，統一價格/文本讀取與切分。
  - `dataloader/dataloader.py`：呼叫共用 API、執行摘要邏輯並產生 DataFrame。
  - `summarize_module/summarizer.py`：逐日摘要與快取。
  - `models/llm.py`：載入與呼叫 HF LLM（支援 QLoRA/4-bit 選項）。
  - `tdmllm/tdmllm.py`：組裝 prompts、批次推論、結果寫檔與評估。
  - `utils/metrics.py`：分類指標（accuracy、mcc、precision、recall、f1、confusion）。

資料與路徑
- 環境變數（若未設定，使用預設值）：
  - `DATASETS_DIR`（預設 `./datasets`）：資料集根目錄。
  - `OUTPUTS_DIR`（預設 `./outputs`）：輸出根目錄。
- 資料集子路徑（與 STARE 對齊）：
  - `SAMPLE`: price=`sample_data/sample_price`, tweet=`sample_data/sample_tweet`
  - `STOCKNET`: price=`stocknet/price`, tweet=`stocknet/tweet`
  - `CMIN`: price=`CMIN/CMIN-Dataset/CMIN-US/price`, tweet=`CMIN/CMIN-Dataset/CMIN-US/news`
  - `SEP`: price=`SEP/price`, tweet=`SEP/tweet`
- 結果輸出根目錄（請勿寫入程式資料夾）：
  - `outputs/results/{dataset}/TDMLLM/{model}/{exp}/`
  - 摘要快取：
    - `baselines/TDMLLM/summaries/{ticker}/{date}.json`（預設；可用 `--summary_cache_dir` 覆寫）
  - 公司描述快取：
    - 依資料集與模型分層，寫入 `baselines/TDMLLM/company_descriptions_cache/{dataset}/{model}/{TICKER}.txt`
  - Split 檔案：
    - `splits/{dataset}_splits.json`（由 `common/data/splits.py` 管理；預設儲存在 repo 根目錄，可用 `--splits_dir` 覆寫）

CLI 與參數（建議）
- 共同參數：
  - `--dataset_name`（SAMPLE/STOCKNET/CMIN/SEP）
  - `--base_model`（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct`）
  - `--summary_model`（摘要模型；預設同 `--base_model`）
  - `--seed`, `--seq_len`, `--batch_size`, `--experiment_name`
  - 路徑選項：`--outputs_dir`（預設環境或 `./outputs`）
- 生成與效能選項：
  - 推論：`--do_sample`、`--num_beams`、`--temperature`、`--top_p`、`--max_new_tokens_predict`
  - 摘要：`--summary_max_new_tokens`、`--summary_min_tweets`、`--summary_max_tweets`、`--fast_summary`
  - 模型載入：`--use_qlora`、`--load_in_4bit`、`--device_map auto`（避免硬綁 8-GPU）
  - 儲存控制：`--store_raw`（預設 true）、`--truncate_chars`（預設 -1 代表不截斷）、`--store_prompts`（預設 false）
  - 切分：`--train_ratio`、`--split_seed`、`--splits_dir`

輸出規範（標準化，便於跨方法比較）
- 目錄：`outputs/results/{dataset}/TDMLLM/{model}/{exp}/`
- 檔案：
  - `args.json`：完整參數與環境快照。
  - `run.log`：執行日誌（包含開始/結束時間、裝置資訊、種子）。
  - `predictions.jsonl`：逐樣本一行，建議欄位：
    - `sample_id`（`{ticker}_{prediction_date}`）
    - `dataset`, `method`="TDMLLM", `model`, `experiment_name`
    - `ticker`, `prediction_date`, `ground_truth`
    - `prediction`: `{"label": "Positive"|"Negative", "confidence": null|number}`
    - `raw_response`（可截斷至 `truncate_chars`）
    - `prompts`（可選）：`{"system": str, "user": str}`
    - `timing`（可選）：`{"latency_ms": number}`
  - `predictions.csv`：扁平欄位（至少包含 `sample_id,ticker,prediction_date,y_true,y_pred,model,method,dataset,experiment_name`）。
  - `eval.json`：統計彙總（`accuracy,mcc,precision,recall,f1,confusion_matrix,total,valid,invalid,wall_time`）。
- 備註：僅輸出 `predictions.*` 與 `eval.json`；不再產生 `detailed.json/simplified.json/comparison.csv` 等 legacy 檔案。資料切分由 `splits/{dataset}_splits.json` 共用，確保各方法一致。

可讀性與效能建議（改造目標）
- 可維護性：
  - 新增 `runner.py` 暴露 `run_eval(args, logger)`，由 `main.py` 僅負責參數與 logger，並統一寫檔。
  - 把路徑解析與結果目錄建立統一關到 `OUTPUTS_DIR`；移除個人家目錄硬編碼。
  - 將摘要與預測的 tokens 上限分開；將快取目錄改到 `outputs/cache/`。
- 效能：
  - 啟用 `device_map="auto"`、允許 CPU/單 GPU 回退；避免固定 8-GPU `max_memory`。
  - 可關閉采樣（`--do_sample=false`）或使用小 beams；適度降低 `max_new_tokens_*`。
  - `summary_min_tweets` 門檻：低量推文可跳過或用規則式摘要（`--fast_summary`）。
  - 啟用 TF32（Ampere+）：`torch.backends.cuda.matmul.allow_tf32 = True`（在 `models/llm.py`）。
  - `batch_inference` 時若 OOM，自動降 `batch_size` 或 `max_new_tokens`。

評估與一致性
- 指標沿用 `utils/metrics.py`，保持與既有行為一致（含處理未知輸出邏輯），但請在 `eval.json` 中加上 `total/valid/invalid`。
- 未來將遷移到共用 evaluator（與 STARE/SEP 對齊），欄位不變。

日誌與重現
- 於 `run.log` 記錄：種子、裝置、模型名稱、資料集、參數快照、開始/結束時間、耗時。
- 在 `args.json` 寫入所有 CLI 與推論/摘要超參數，確保實驗可重現。

操作指引（範例）
- 使用預設環境變數與相對路徑：
  - `export DATASETS_DIR=./datasets`
  - `export OUTPUTS_DIR=./outputs`
- 以 SAMPLE 驗證：
  - `python baselines/TDMLLM/main.py --dataset_name SAMPLE --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 8 --seq_len 5 --legacy_outputs true`
- 以 SEP 跑子集（建議先挑小樣本驗證）：
  - `python baselines/TDMLLM/main.py --dataset_name SEP --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 4 --seq_len 5 --do_sample false --max_new_tokens_predict 256`

引用與參考
- Temporal Data Meets LLM – Explainable Financial Time Series Forecasting（TDMLLM）
  - 建議補充：論文連結（doi / arXiv）、核心流程簡述、與本實作差異（如批次推論、快取策略）。

給未來在本目錄工作的 Codex 指南
- 優先目標：
  1) 保持核心邏輯不變，先標準化輸出與路徑，並支援 `DATASETS_DIR`/`OUTPUTS_DIR`。
  2) 加入 `runner.py` 與 `--legacy_outputs`，雙寫輸出，確保比較相容性。
  3) 提供效能控制參數與安全降級（避免 OOM）。
- 請勿：
  - 變更資料分割或標籤邏輯（除非明確需求）。
  - 提交 `outputs/`、`results/`、大型 `.npy` 等產物到版本控制。

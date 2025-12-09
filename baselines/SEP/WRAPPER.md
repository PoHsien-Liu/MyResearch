# SEP Wrapper 指南

## 環境與相依
- 請先 `source ~/miniconda3/etc/profile.d/conda.sh && conda activate sep`。
- 依賴更新：`requirements.txt` 已改為 torch 2.5.1 / transformers 4.52+ / peft 0.7+ / trl 0.9+，並新增 `vllm>=0.5.4` 以支援本地 LLM。若需安裝，請在 sep 環境中以 `pip install -r baselines/SEP/sep/requirements.txt`。
- 預設模型：`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`，以 vLLM 內載方式推論；可透過 `--quantization` 改為 `none` 測試非 AWQ 模型。

## 路徑與資料載入
- 透過 `DATASETS_DIR`（預設 `./datasets`）與 `OUTPUTS_DIR`（預設 `./outputs`）解析路徑；`resolve_dataset_paths` 對應 `price/raw` 與 `tweet/raw`（CMIN 則讀 news csv）。
- 共享 split：使用 `common.data.loader.list_trading_days` 依 `--train_ratio` / `--split_seed` 產生 train/test。
- 摘要邏輯：對每個樣本，取目標交易日往前 `--seq_len` 個交易日（含當日），逐日讀取文本並用 summarizer 生成摘要，再按日期串成 summary。label 採 legacy（ret>0 為 Positive 否則 Negative）。
- 摘要快取：存於 `outputs/cache/summaries/{dataset}/{model}/SEP/`，以 (ticker, date, tweets sha) 為 key。

## 推論流程
- 入口：`python baselines/SEP/run_wrapper.py --dataset_name SAMPLE --base_model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --seq_len 5 --quantization awq --engine vllm`
- vLLM 共享實例同時用於 summarization 與 prediction，避免重複載入模型。生成溫度/長度可用 `--summary_*` / `--predict_*` 調整。
- 若暫無 vLLM（或測試小模型），可用 `--engine hf --quantization none --base_model <small_model>` 走 transformers fallback 作 smoke test。
- 目前 wrapper 聚焦 eval 路徑（不跑原始 SFT/RL 訓練），以原 PREDICT prompt 直接生成標籤與解釋。

## 輸出
- 寫入 `outputs/results/{dataset}/{label_variant}/SEP/{model}/{experiment}/`：
  - `args.json`：參數快照
  - `run.log`：執行日誌
  - `predictions.jsonl` / `predictions.csv`：標準欄位（含 sample_id/ticker/prediction_date/ground_truth/prediction/raw_response）
  - `eval.json`：accuracy/mcc/precision/recall/f1/confusion_matrix/total/valid/invalid/unknown_predictions/wall_time_sec

## 差異與注意事項
- OpenAI API 已移除，改為本地 vLLM；如需雲端 API 請改用 `utils.llm.OpenAILLM`（需自行安裝 openai）。
- 原始 RL/SFT 訓練腳本仍保留但未整合到 wrapper，後續若需訓練需自行調整至新相依版本。
- vLLM 載入 8B AWQ 需 GPU 記憶體，若資源不足可先用小模型 + `--quantization none` 進行 smoke test。

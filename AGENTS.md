目的與範圍
- 題目：STARE（Stock Price Prediction with Text‑Augmented Retrieval for Explainability）。
- 目標：在統一的資料與評估設定下，重現與比較多個方法（SEP、TDMLLM、LLMFactor、FinGPT、FinBert），並保留可切換 foundation model 的彈性。
- 產物：統一的輸出結構（結果、日誌、檢索索引、快取），便於比較、追蹤與重現。

目錄結構
- baselines/
  - TDMLLM/：TDMLLM 基線與其依賴（dataloader、summarizer、llm、metrics 等）。
  - SEP/sep/：作者原始專案；以 wrapper/runner 介面整合到共同介面。
  - LLMFactor/、FinGPT/、FinBert/：預留與待接入。
- STARE/stare/：本研究方法
  - data_load/：tweet 讀取與清理、向量化（RAG 索引）。
  - index/：索引建置流程（embeddings、metadata、faiss）。
  - configs/dataset.py：資料集路徑映射。
  - utils/：logger、seed。
  - main.py：CLI 入口（任務：build_index / eval）。
  - models/STARE/pipeline.py：STARE pipeline（目前待實作）。
- datasets/：所有方法共用資料集（sample_data、SEP、CMIN、stocknet、ACL18/stocknet）。
- outputs/：所有產物（請加入 .gitignore）
  - results/{dataset}/{method}/{model}/{exp}/
  - indices/{dataset}/{embed_model}/（RAG 索引）
  - cache/summaries/{model}/{method}/{ticker}/{date}.json

環境與路徑
- 主要環境變數
  - DATASETS_DIR：資料集根路徑，預設 ./datasets。
  - OUTPUTS_DIR：輸出根路徑，預設 ./outputs。
- 資料集映射（集中於 STARE/stare/configs/dataset.py）
  - SAMPLE: price=sample_data/sample_price, tweet=sample_data/sample_tweet
  - ACL18: price=ACL18/stocknet-dataset/price, tweet=ACL18/stocknet-dataset/tweet
  - CMIN: price=CMIN/CMIN-Dataset/CMIN-US/price, tweet=CMIN/CMIN-Dataset/CMIN-US/news
  - SEP: price=SEP/price, tweet=SEP/tweet
- 共同規範
  - 統一用絕對匯入，避免依賴執行目錄。
  - 所有產物一律寫入 OUTPUTS_DIR，不得進程式目錄。

共同 CLI 規範
- 共同參數（適用 STARE 與各 baseline wrapper）
  - --dataset_name（SAMPLE/ACL18/CMIN/SEP）
  - --base_model（例如 meta-llama/Meta-Llama-3.1-8B-Instruct、ProsusAI/finbert）
  - --seed、--seq_len、--batch_size
  - --experiment_name（若未指定則使用 timestamp）
  - 任務特有參數（如 STARE：--task {build_index,eval}, --embed_model, --rebuild_index）
- 共同結果目錄
  - outputs/results/{dataset}/{method}/{model}/{exp}/

輸出檔案結構（每次實驗）
- 必備
  - args.json：完整參數與環境快照。
  - run.log：執行日誌。
  - eval.json：統一評估結果（accuracy、mcc、precision、recall、f1、confusion 等）。
- 預測明細（便於比對與分析）
  - detailed.json：包含 prompts、raw outputs、證據/檢索、解析後預測。
  - simplified.json：精簡版（核心欄位）。
  - results.csv：逐樣本結果表。
  - comparison.csv：y_true vs y_pred。
- RAG 與資料處理產物
  - indices/{dataset}/{embed_model}/embeddings.npy
  - indices/{dataset}/{embed_model}/metadata.parquet
  - indices/{dataset}/{embed_model}/index.faiss（若安裝 faiss）
  - indices/{dataset}/{embed_model}/dropped.parquet（tweet 過濾記錄）

共同評估模組
- 分類指標：accuracy, mcc, precision, recall, f1, confusion_matrix。
- 解釋指標（可選）：證據條數、citation 覆蓋率、解釋長度與可讀性統計。
- 寫檔：統一以 eval.json 格式輸出，便於跨方法比較與彙整。

Model Adapter 介面（可換 foundation model）
- 目標：統一呼叫推論，支援 HF causal LLM、FinBERT（分類）、FinGPT 等。
- 介面
  - generate(system: str, user: str, **gen_kwargs) -> str
  - batch_generate(system_list: List[str], user_list: List[str], **gen_kwargs) -> List[str]
- 選項
  - QLoRA 與 4-bit quantization（peft, bitsandbytes），可切換與關閉。
  - 環境感知：自動偵測 GPU/CPU；不得硬編碼 8-GPU max_memory。

資料載入與切分
- 統一 dataloader/split
  - 先建立每個 dataset 的固定 split（train/val/test），讓所有方法用相同樣本集。
  - 推文載入與清理統一走 STARE/stare/data_load/tweet_reader.py；價格標籤由 price delta 決定。
- 使用者可選在各方法內進一步處理（但不可改變 split）。

Baselines 整合規範
- 每個 baseline 提供統一 runner 介面（例如 run_eval(args, logger)），並寫出共同輸出檔案。
- SEP
  - 以 wrapper 連結作者原始程式，不可破壞其程式架構；必要時增補相容 shim。
  - 紀錄偏離原論文設定之處於 baselines/SEP/WRAPPER.md。
- TDMLLM
  - 已有完整流程；將結果輸出路徑、metrics 與 log 統一到上述規範。
  - Summarizer 快取路徑統一到 outputs/cache/summaries/{model}/{method}/...

STARE（RAG）設計規範
- 索引建置：以 sentence-transformers 產生 embeddings，必要時建 FAISS；metadata 包含 ticker/date/source_path/is_retweet/url_count/cashtag_count 等。
- 檢索與生成：
  - 以樣本（ticker + 時間）組查詢，檢索 top‑k 文本，組合 prompt（含 citations），再生成預測與可解釋文字。
  - 輸出必須包含引用來源（對應 metadata.parquet 的列索引或 source_path 與日期）。

日誌、重現與安全
- 每次實驗固定記錄：args.json、run.log、隨機種子（numpy/torch/cuda）。
- 不得把 datasets/ 與 outputs/ 納入版控。
- 不執行破壞性動作；保留原始資料。

程式設計與修改準則
- Python 3.10+；盡量少依賴；import 不產生副作用。
- 修改遵循最小原則：只針對本任務必要之處動手；不順手修不相關 bug。
- 風格一致、命名清晰、避免一字母變數；程式碼註解以必要為限。
- 大型運算/推論前，先以 SAMPLE dataset 做乾跑驗證。

執行範例
- 建索引（STARE）
  - python -m STARE.stare.main --task build_index --dataset_name SEP --embed_model sentence-transformers/all-MiniLM-L6-v2 --rebuild_index
- TDMLLM（推論 + 評估）
  - python baselines/TDMLLM/main.py --dataset_name SEP --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 8 --seq_len 5
- 查看長尾分佈（SEP 風格推文目錄）
  - python STARE/stare/tweet_longtail.py /path/to/SEP/tweet/raw --out-csv outputs/tweet_volume_by_ticker.csv --out-png outputs/tweet_volume_by_ticker.png

路線圖與子任務
- Phase 0：基礎設施
  - 將 STARE/TDMLLM 產物改寫到 outputs/；DATASETS_DIR/OUTPUTS_DIR 預設值生效；.gitignore 新增 outputs/。
- Phase 1：TDMLLM 端到端
  - 共用 dataset 路徑映射；統一輸出；LLM 載入支援 CPU/1-GPU/可關閉 QLoRA；以 SAMPLE 跑通，產出 eval.json。
- Phase 2：共同評估模組
  - 抽象出 evaluator（ACC/MCC 等）與標準 eval.json schema；加入簡易解釋評估。
- Phase 3：SEP wrapper
  - 建 baselines/SEP/run_wrapper.py 導入共用 CLI/輸出；必要 shim 與 WRAPPER.md。
- Phase 4：STARE Pipeline
  - 完成檢索、prompt 組裝、生成、引用、評估，與統一輸出。
- Phase 5：Model Adapters
  - 建立 adapter registry；加入 HF causal、FinBERT、FinGPT；TDMLLM/SEP/STARE 改用 adapter。
- Phase 6：統一 dataloader/splits
  - 冷凍 split；各法共享；TDMLLM 改用統一 loader。
- Phase 7：LLMFactor 與 foundation baselines
  - 實作 LLMFactor；FinGPT/FinBert 零樣本/分類基線。
- Phase 8：多模態/跨公司擴充
  - RAG 檢索擴張到關聯公司，加入價量特徵。

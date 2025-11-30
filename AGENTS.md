目的與範圍
- 題目：STARE（Stock Price Prediction with Text‑Augmented Retrieval for Explainability）。
 - 目標：在統一的資料與評估設定下，重現與比較多個方法（SEP、TDMLLM、LLMFactor、FinGPT、FinBert），並保留可切換 foundation model 的彈性。
 - 產物：統一的輸出結構（結果、日誌、檢索索引、快取），便於比較、追蹤與重現。
- 近期優先：STARE 方案先鎖定 CMIN-US 資料集完成資料清洗 → 向量化 → 索引建置與 RAG 檢索流程，再推廣到 Twitter/SEP 等其他資料集。

助理互動
- 助理回覆預設採用繁體中文，除非使用者另行指定語言。
- 一旦使用者設定回覆格式或語氣，助理需維持該設定直到使用者更新指示。
- 在交付任何程式或指令前，助理必須先自行驗證、實際執行至完成；不得提供已知會出錯或未經測試的腳本。
- 驗證需在本機實際跑通整段流程或指令並確認無錯，若因環境限制無法執行，須明確回報原因與未驗證項目。

目錄結構
- baselines/
  - TDMLLM/：TDMLLM 基線與其依賴（dataloader、summarizer、llm、metrics 等）。
  - SEP/sep/：作者原始專案；以 wrapper/runner 介面整合到共同介面。
  - LLMFactor/、FinGPT/、FinBert/：預留與待接入。
- STARE/stare/：本研究方法
  - configs/dataset.py：資料集路徑映射與輸入欄位定義。
  - data_load/：
    - clean_tweets.py：針對 Twitter／CMIN-US 文本的 rule-based + optional LLM 清洗、輸出 cleaned/dropped 紀錄。
    - extract_mentions.py：抽取 mentioned tickers、cashtag/URL 統計與 metadata。
  - index/：
    - embed_texts.py：呼叫 sentence-transformers 等 embedding 模型，輸出 embeddings.npy 與 metadata.parquet。
    - build_index.py：建立 FAISS 索引（IndexFlatIP/L2）並與 metadata 對齊。
  - utils/：logger、seed、paths 等共用工具。
  - main.py：CLI 入口，任務包含 clean / extract_mentions / embed / build_index / build_index_pipeline / eval。
  - models/STARE/pipeline.py：RAG 推論流程與 LLM 生成（目前待實作）。
- datasets/：所有方法共用資料集（sample_data、SEP、CMIN、stocknet）。
- outputs/：所有產物（請加入 .gitignore）
  - results/{dataset}/{method}/{model}/{exp}/
  - indices/{dataset}/{embed_model}/：STARE RAG 索引與中間產物（cleaned*.parquet、cleaned_with_mentions*.parquet、embeddings.npy、metadata.parquet、index.faiss、dropped.parquet）
  - cache/summaries/{model}/{method}/{ticker}/{date}.json
++ 注意：CMIN price/raw 中有優先股代碼 C-PJ、CTA-PB、SPG-PJ、WFC-PL，但 news/raw 無對應檔案，已在共用 dataloader 排除，不應納入樣本。

- 環境與路徑
- 執行 CLI 或驗收流程前，請先進入指定的 Conda 環境：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate stare`。
- 根據任務選擇對應的 `conda` 環境規格：
  - `sep-environment.yml`：meta-llama/Meta-Llama-3.1-8B-Instruct 相關訓練與推論（STARE/TDMLLM/STARE pipeline）需升級到 torch 2.5+ 的組合。
  - `fingpt-environment.yml`：針對 FinGPT adapter 的特定套件組（transformers 4.32 + torch 2.0.1 + peft 0.5.0）以避免與 `sep` 環境衝突；建立後在 `fingpt` 環境內執行 `tools/setup_fingpt_env.sh`（會建立 `libittnotify.so` 並設定 `LD_PRELOAD`，以避免 `iJIT_NotifyEvent` 缺失）。
  使用範例命令：
  ```bash
  conda env create -f sep-environment.yml
  conda env create -f fingpt-environment.yml
  ```
- 主要環境變數
  - DATASETS_DIR：資料集根路徑，預設 ./datasets。
  - OUTPUTS_DIR：輸出根路徑，預設 ./outputs。
- 資料集映射（集中於 STARE/stare/configs/dataset.py）
  - SAMPLE: price=sample_data/sample_price, tweet=sample_data/sample_tweet
- STOCKNET: price=stocknet/price, tweet=stocknet/tweet
  - CMIN: price=CMIN/CMIN-Dataset/CMIN-US/price, tweet=CMIN/CMIN-Dataset/CMIN-US/news
  - SEP: price=SEP/price, tweet=SEP/tweet
- 共同規範
  - 統一用絕對匯入，避免依賴執行目錄。
  - 所有產物一律寫入 OUTPUTS_DIR，不得進程式目錄。

共同 CLI 規範
- 共同參數（適用 STARE 與各 baseline wrapper）
- --dataset_name（SAMPLE/STOCKNET/CMIN/SEP）
  - --base_model（例如 meta-llama/Meta-Llama-3.1-8B-Instruct、ProsusAI/finbert）
  - --seed、--seq_len、--batch_size
  - --experiment_name（若未指定則使用 timestamp）
  - 任務特有參數（如 STARE：--task {clean,extract_mentions,embed,build_index,build_index_pipeline,eval}、--embed_model、--rebuild_index、--min_tokens、--enable_llm_filter）
- 共同結果目錄
  - outputs/results/{dataset}/{method}/{model}/{exp}/

輸出檔案結構（每次實驗，統一格式）
- 必備
  - args.json：完整參數與環境快照。
  - run.log：執行日誌（包含開始/結束時間與裝置資訊）。
  - eval.json：統一評估結果（accuracy、mcc、precision、recall、f1、confusion 等）與樣本統計（total/valid/invalid、wall_time）。
  - 逐樣本預測（便於跨方法比較與彙整）
  - predictions.jsonl：每行一筆記錄，建議欄位：
    - sample_id（{ticker}_{prediction_date}）、dataset、method、model、experiment_name
    - ticker、prediction_date、ground_truth
    - prediction：{"label": "Positive"|"Negative", "confidence": null|number}
    - raw_response（可選，預設保留並可截斷）
    - prompts（可選）：{"system": str, "user": str}
    - timing（可選）：{"latency_ms": number}
  - predictions.csv：扁平欄位版（至少含 sample_id,ticker,prediction_date,y_true,y_pred,model,method,dataset,experiment_name），並需額外保存預測使用的 prompt 欄位（system_prompt, user_prompt）。
- RAG 與資料處理產物（STARE 與共用索引用）
  - indices/{dataset}/{embed_model}/embeddings.npy
 - indices/{dataset}/{embed_model}/metadata.parquet
 - indices/{dataset}/{embed_model}/index.faiss（若安裝 faiss）
 - indices/{dataset}/{embed_model}/dropped.parquet（tweet 過濾記錄）

共同評估模組
- 分類指標：accuracy, mcc, precision, recall, f1, confusion_matrix。
- 標籤策略：預設嚴格，Unknown/非法 prediction 視為錯（翻轉真值扣分），`unknown_policy` 可切換為 as_invalid（跳過並回報 coverage）。
- 解釋指標（可選）：證據條數、citation 覆蓋率、解釋長度與可讀性統計。
- 寫檔：統一以 eval.json 格式輸出，便於跨方法比較與彙整。
- 解釋評估：`python -m STARE.main --task explanation_eval --dataset_name {SEP|STOCKNET|CMIN-US|SAMPLE} --predictions_csv PATH --stock_scope {all|top1} --only_correct {true|false} --eval_llm_backend {qwen|llama|openai|gemini} --eval_llm_model MODEL --explanation_eval_output_dir outputs/... --max_eval_samples N`；輸出 `explanation_eval_samples.jsonl` 與 `explanation_eval_summary.json`。

LLM 評分後端
- 在專案根目錄建立 `stare_llm_config.yaml`（可由 `stare_llm_config.example.yaml` 複製），設定各 backend（qwen/llama）的模型、量化、tensor parallel 等參數；可用環境變數 `STARE_LLM_CONFIG` 指定路徑。
- 預設走內嵌 vLLM（Python 直接載入，不開 HTTP 端口），不需 `base_url`/`api_key`。若未來需 HTTP 介面，可自行包裝 OpenAI 相容伺服器但非預設。

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
- 每個 baseline 提供統一 runner 介面（例如 run_eval(args, logger)），並寫出共同輸出檔案（predictions.jsonl、predictions.csv、eval.json、args.json、run.log）。
- SEP
  - 以 wrapper 連結作者原始程式，不可破壞其程式架構；必要時增補相容 shim。
  - 紀錄偏離原論文設定之處於 baselines/SEP/WRAPPER.md。
- TDMLLM
  - 已有完整流程；將結果輸出路徑、metrics 與 log 統一到上述規範。
  - Summarizer 快取路徑統一到 outputs/cache/summaries/{model}/{method}/...

STARE（RAG）設計規範
- Pipeline（build_index_pipeline 任務）：clean → extract_mentions → embed → build_index，所有中間檔案（cleaned/dropped/metadata/embeddings/index）都寫入 `outputs/indices/{dataset}/{embed_model}/`。
- Clean：rule-based 規則（min_tokens、emoji/hashtag/URL 限制、retweet 檢查）為主，可透過 `--enable_llm_filter` 串接 LLM 第二層篩選；cleaned 與 dropped 需分別存檔。
- Extract_mentions：針對 cleaned 檔案生成 mentioned_tickers、cashtag_count、url_count 等 metadata。
- Embed：預設以 sentence-transformers 產生 embeddings，同時輸出 metadata.parquet；向量/metadata 的 row id 須與後續索引保持對齊。
- Build_index：建立 FAISS IndexFlatIP/L2；若 `--rebuild_index` 為真則覆蓋舊檔，並確保索引路徑依 dataset/embed_model 分層。
- 檢索與生成（models/STARE/pipeline.py 待實作）：
  - 以樣本（ticker + 時間）組查詢，檢索 top‑k 文本，組合 prompt（含 citations），再生成預測與可解釋文字。
  - 輸出需附 citation，引用 metadata.parquet 的列索引或 source_path + 日期資訊。

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
- STARE build_index_pipeline（以 SEP 為例，可先在 CMIN-US 驗證）
  - python -m STARE.main --task clean --dataset_name SEP --min_tokens 5 --enable_llm_filter False
  - python -m STARE.main --task extract_mentions --dataset_name SEP
  - python -m STARE.main --task embed --dataset_name SEP --embed_model sentence-transformers/all-MiniLM-L6-v2
  - python -m STARE.main --task build_index --dataset_name SEP --embed_model sentence-transformers/all-MiniLM-L6-v2 --rebuild_index
  - # 或一次串完
  - python -m STARE.main --task build_index_pipeline --dataset_name SEP --embed_model sentence-transformers/all-MiniLM-L6-v2 --min_tokens 5 --enable_llm_filter False --rebuild_index
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

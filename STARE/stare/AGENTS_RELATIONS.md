# STARE / AGENTS_RELATIONS.md

## 1. 任務目的（Task Goal）
- **公司關係推論**：針對每個 `stock_target`，以 `company_neighbors.json` 的共現資訊產生候選 `(stock_target, stock_match)`，交由 LLM 推論兩家公司之間的關係、信心分數與簡短解釋。
- **關係資料存檔**：推論結果寫入結構化檔案（建議 `outputs/indices/{dataset}/company_relations.json`），供後續檢索使用。
- **relation-aware Retrieve**：
  - 對「目標公司」本身取出 top-k 文章。
  - 根據已確認的相關公司（依關係類別/信心門檻篩選）各自再取出 top-k 文章。
  - 將目標與相關公司的候選文章集合交給 re-ranking / RAG pipeline，讓最終生成時保留跨公司脈絡。

> 重點：本模組建立「公司關係→候選文章」的中介資訊，讓 Retrieve 階段不只看單一 ticker，而是結合相關公司的資訊。

## 2. 使用的 LLM 模型（Backbone Model）
- 預設使用 **Llama-3.1-70B-Instruct** 作為公司關係推論的 backbone。
- 模型輸入（per pair）：
  - `stock_target`、`stock_match` 各自的 ticker、公司名稱、產業（若可取得）。
  - 兩者在 `company_neighbors.json` 中的共現次數。
  - 可附上最近的新聞摘要 / meta 作為提示。
- 模型輸出（JSON）：
  - `sentence`：LLM 填空後的關係句子（如「Apple 與 Foxconn 為供應商／客戶關係」）。
  - `relationship_type`：從定義的關係類別中擇一。
  - `confidence`：0.0–1.0 間的浮點數。
  - `explanation`：簡短說明（1–2 句）。
- API / model 名稱設定：
  - 需透過環境變數或設定檔（例如 `LLM_RELATION_MODEL_NAME`、`LLM_API_BASE`、`LLM_API_KEY`），不得寫死在程式碼中。

## 3. 公司關係類別（Relationship Types）
建議先提供如下類別，之後可再調整：
1. `same_industry_competitors`
2. `supplier_customer`
3. `parent_subsidiary_or_affiliate`
4. `strategic_partners`
5. `same_conglomerate_or_group`
6. `cooperative_innovation_or_co-marketing`
7. `regulatory_or_legal_dependency`
8. `no_direct_relationship_or_unclear`

> LLM 必須從上述類別中選一個並輸出在 `relationship_type`；若覺得類別不足，可在實作時擴充，但必須同步更新文件與 schema。

## 4. 主要參數（Config / Hyper-parameters）
請在設定檔或 CLI 參數中提供以下可調選項：
- `max_neighbors`：每個目標公司最多挑幾間相關公司進行 Retrieve（避免無限擴張）。
- `min_cooc`：從 `company_neighbors.json` 篩選鄰居時的共現次數下限，過低代表噪音，直接跳過。
- `min_confidence`：若 LLM 對某 pair 的 `confidence` 低於此值則不納入後續檢索。
- `skip_unclear`（bool）：是否忽略 `relationship_type = no_direct_relationship_or_unclear` 的 pair。
- `top_k_self`：retriever 針對目標公司本身要取幾篇文章。
- `top_k_per_neighbor`：每個關聯公司要取幾篇文章。

> 這些參數需做成可配置（config 檔或 CLI 旗標），方便做實驗調整。

## 5. 輸入與輸出概念
- **輸入資料**：
- `company_neighbors.json`：`neighbors[target][match] = { "cooc": count, ... }`。
  - 可額外搭配公司名稱、產業分類等 meta（若已有 mapping）。
  - 可選：近期共現文章的摘要以供 LLM 判斷。
- **LLM 推論輸出（per pair）**：
  ```json
  {
    "sentence": "...",
    "relationship_type": "supplier_customer",
    "confidence": 0.82,
    "explanation": "..."
  }
  ```
- **關係結果檔（建議 company_relations.json）**：
  ```json
  {
    "AAPL": {
      "TSM": {
        "relationship_type": "supplier_customer",
        "confidence": 0.9,
        "sentence": "...",
        "explanation": "...",
        "cooc": 5613
      },
      "MSFT": { ... }
    },
    ...
  }
  ```
  > 實際欄位可視需求擴充，但需保持巢狀結構 `relations[target][match]`。
- **Retrieve 階段**：
  - 對每個 `target`，保留本公司的 top-k 文章（`top_k_self`），以及每個符合條件（cooc/ confidence）的 `match` 的 top-k 文章（`top_k_per_neighbor`）。
  - 所有文章組成候選文件清單，送給 re-ranking / LLM 做最終決策。

## 6. 實作責任範圍
- 本文件僅描述「關係推論＋relation-aware 檢索」的需求、輸入輸出與參數；具體程式碼（class / function / 檔案命名）由後續 AGENT 依 STARE 既有結構自行決定。
- 請保留彈性：未來可能替換 LLM、增加類別或更動 config，務必避免在程式碼多處寫死常數，把設定集中在 config 或 CLI。

> 最終目標：透過這個模組建立可重複使用的「公司關係→候選文章」資料，使 STARE pipeline 能在 Retrieve 階段納入跨公司資訊，提升生成解釋與預測的質量。
> 建議實作採「雙向推論」：`(target, match)` 與 `(match, target)` 各自呼叫 LLM，分別儲存在 `relations[target][match]` 與 `relations[match][target]`，保留方向資訊與專屬句子。

## 1. 專案總體目標（STARE / LLM-based Stock Movement Prediction）

本專案的核心任務為：

1. **預測股價漲跌**  
   - 給定目標股票在一段時間內的價格序列（例如最近 5 個交易日的回報率）；
   - 搭配目標公司與相關公司（供應鏈、同產業、競品等）的新聞事件（透過 RAG 檢索取得）；
   - 預測「隔日股價相對前一日收盤」是 **UP** 或 **DOWN**。

2. **產生可閱讀的自然語言解釋（explanation）**  
   - 輸出一段**短解釋**，說明為什麼模型預期股價上漲或下跌；
   - 解釋需具有金融直覺（例如提到利多消息、財報、產業情緒、相關公司動向等）；
   - 不要求逐句對應到輸入新聞的精確 citation。

3. **支援跨公司資訊（Cross-firm RAG）**  
   - 使用公司關係（產業鏈、共現、分類等）從新聞庫中擷取「目標公司＋相關公司」的證據；
   - 作為 LLM 的輸入 context，幫助模型捕捉 spillover effects。

---
## 2. 關鍵設計決策：本版本「不做 citation 監督」

### 2.1 過去構想

早期設計中，我們曾考慮在輸出中加入：

```json
{
  "prediction": "UP",
  "reason": "...",
  "used_event_ids": [1, 3]
}
```

並希望：

- `used_event_ids` 對應到輸入的新聞列表；
- 甚至設計一個「事實一致性指標」：檢查解釋中的句子有多少可以在輸入新聞中找到語意支持。

然而，實務上出現幾個問題：

1. 資料集中**並沒有人工標註的 explanation / citation**，很難定義什麼是「正確引用」；
2. 用 LLM 當評審來計算「事實一致性分數」容易有偏誤，不適合當作主要評估指標；
3. 一般投資者在做決策時，**並不會逐句引用新聞**，而是綜合價格趨勢、新聞情緒與經驗做直覺判斷。

### 2.2 新的決策

> 在本版本的專案中，**我們不再把 citation / grounding 視為主要任務，也不對 `used_event_ids` 進行訓練或評估**。

具體來說：

- 所有訓練 loss **只關注**：
  - `prediction`（漲跌標籤）
  - `reason`（解釋文本本身）
- **不會**：
  - 對 `used_event_ids` 計算 loss；
  - 對「解釋中的句子是否對應到特定新聞」設計 hard constraint 或主評分指標。

`used_event_ids` 欄位如果出現在某些資料結構中，可以：

- 視為 **optional metadata**；
- 保留以利未來實作 grounding 任務；
- 但在當前版本中 **不應被視為必要欄位或訓練目標**。

---

## 3. Agent 必須遵守的 DO / DON’T

以下規則適用於整個 repo（除非未來版本有明確 override）：

### ✅ DO（應該做）

1. **聚焦在 prediction + explanation**  
   - 所有訓練流程以：
     - `prediction` 的分類性能（Accuracy / MCC …）
     - `reason` 的品質（合理性、可讀性、是否與預測一致）
   為主要優先事項。

2. **在設計 SFT / 微調資料時，至少包含：**

   ```json
   {
     "prediction": "UP" or "DOWN",
     "reason": "<short explanation>"
   }
   ```

   其他欄位（ticker, date, price_context, events, meta 等）可視任務需求加上。

3. **允許使用 teacher 模型產生 pseudo explanation**  
   - 可以使用 Llama-3.1-70B-Instruct（量化）或類似的大模型；
   - 讓大模型針對 RAG context 產生自然語言解釋；
   - 再將這些 explanation 蒸餾到 8B 學生模型。

4. **在評估時，可以使用 LLM judge / 人工標註來打分解釋品質**  
   - 指標可以包含：
     - 金融合理性（Financial plausibility）
     - 與預測的一致性（Consistency with prediction）
     - 是否有利用跨公司訊息（Cross-firm usage）
     - 可讀性與清楚度（Clarity）

### ❌ DON’T（不應該做）

1. **不要把 `used_event_ids` 當成必須預測的欄位**  
   - 不要對 `used_event_ids` 設計 loss；
   - 不要因為模型沒吐出 `used_event_ids` 就判定該樣本「不合法」。

2. **不要實作以「句子是否對齊某則新聞」為核心的主要評估指標**  
   - 若要實作 fact-consistency / grounding 指標，應標示為：
     - optional analysis；
     - 或 ablation / side experiment；
   - 這類指標**不應該決定主實驗結果的排序或主要結論**。

3. **不要在沒有明確需要的情況下強迫模型輸出 citation 格式（例如強制寫 (1)(3) 等）**  
   - 若有這類設計，應確認只是為了未來可能的分析，而非當前版本的 hard requirement。

4. **不要讓 grounding 任務阻礙 RAG 與解釋蒸餾的開發進度**  
   - 當出現取捨時，以完成：
     - RAG for cross-firm retrieval
     - explanation distillation
   為優先。

---

## 4. 給 Codex 的立即 TODO

1. **更新根目錄的 `AGENTS.md`**  
   - 以本檔內容為基準，覆寫或合併現有 root `AGENTS.md`；
   - 確保所有 Agent 設計時以「不做 citation 監督」為前提。

2. **巡檢整個 repo**  
   - 搜尋關鍵字：`used_event_ids`, `citation`, `grounding`, `fact consistency`；
   - 檢查是否有任何：
     - 對 `used_event_ids` 計算 loss 的程式碼；
     - 以 grounding score 作為核心 eval 的說明文字；
   - 若有，將其改為 optional / 移除，或標註為 future work。

3. **確保所有訓練與推論 pipeline 都能在沒有 citation 的前提下正常運作**  
   - 即使某些 JSON schema 還包含 `used_event_ids`，模型與評估程式也不應依賴它。

以上為根目錄層級的設計準則與說明。  
之後若有新版本需要重新啟用 grounding 任務，請在此檔案中加入版本註記與更新說明。

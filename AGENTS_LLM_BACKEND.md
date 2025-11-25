AGENTS_LLM_BACKEND.md
=====================

本文件說明本地 LLM 推論後端的設計與使用方式，供「公司關係推論」與「解釋文本評分」共用。實作採用 **Llama-3.1-70B-Instruct + 4-bit 量化 + vLLM** 做為統一 backbone，走 Python 內嵌 vLLM，**不啟動 HTTP 伺服器、不中繼 /v1/chat/completions**。

------------------------------------------------------------
1. 設計選擇
------------------------------------------------------------
- 模型：Llama-3.1-70B-Instruct，提供高品質的金融/解釋能力。
- 推論引擎：vLLM（內嵌），高吞吐、支援 KV cache 與 tensor parallel，不需常駐 server。
- 量化：預設 4-bit（gptq/awq/nf4 均可，建議使用 vLLM 支援的 gptq/awq 權重），在 8×RTX 4090 (24 GB) 上可穩定載入。
- 角色：純推論（不支援微調），重點是批次吞吐與資源利用。

------------------------------------------------------------
2. 硬體與切分建議
------------------------------------------------------------
- 硬體：8×RTX 4090 (24 GB VRAM)。
- 建議設定：
  - tensor parallel size = 8。
  - quantization = gptq（或 awq）；量化權重需事先準備。
  - max_model_len ~4096（依任務調整）；`max_num_batched_tokens` 視記憶體調優。
  - pipeline parallel size = 1（70B + 8×24G VRAM，tp 已足夠）。

------------------------------------------------------------
3. Python 批次推論介面（無伺服器）
------------------------------------------------------------
- 路徑：`STARE/stare/llm_backend/inference.py`
- 函式：`run_inference_batch(requests, backend="llama", model=None, **gen_kwargs) -> List[str>`
- 資料結構：`PromptLike(system: str, user: str, ...)`
- vLLM 載入：在程式內直接呼叫 vLLM，依設定啟用 4-bit + tp=8；無需開 port。

簡短調用範例：
```python
from STARE.stare.llm_backend.inference import PromptLike, run_inference_batch

requests = [
    PromptLike(system="You are a financial analyst.",
               user="Explain why AAPL might go up tomorrow."),
    PromptLike(system="You are a NLI judge.",
               user="Rate the faithfulness of this explanation to the news..."),
]
responses = run_inference_batch(
    requests,
    backend="llama",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_tokens=512,
    temperature=0.0,
    top_p=0.9,
)
```

------------------------------------------------------------
4. 設定檔（stare_llm_config.example.yaml）
------------------------------------------------------------
```yaml
backends:
  llama:
    default_model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
    quantization: "gptq"          # 或 awq
    tensor_parallel_size: 8
    max_new_tokens: 512
    temperature: 0.0
    dtype: null                   # 讓 vLLM 自行決定
    max_model_len: 4096
  qwen:
    default_model: "Qwen/Qwen2.5-72B-Instruct"
    quantization: "gptq"
    tensor_parallel_size: 8
```
使用時複製為 `stare_llm_config.yaml`（或以環境變數 `STARE_LLM_CONFIG` 指定路徑），可覆寫 quantization/tp/max_new_tokens 等參數；不需要 base_url/api_key。

------------------------------------------------------------
5. 與上層任務串接
------------------------------------------------------------
- 解釋文本評分 / 公司關係推論：直接呼叫 `run_inference_batch`，模型與生成參數從 `stare_llm_config.yaml` 或函式參數取得。
- 若未來環境允許 HTTP 端點，可自行包裝，但本專案預設以內嵌 vLLM 為主，不依賴 /v1/chat/completions。

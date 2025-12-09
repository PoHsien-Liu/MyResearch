from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class VLLMSamplingConfig:
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: Optional[float] = None


class OpenAILLM:
    """Legacy OpenAI client retained for backward compatibility."""

    def __init__(self, model: str = "gpt-3.5-turbo-16k"):
        self.model = model
        self._openai = None

    def _client(self):
        if self._openai is None:
            spec = importlib.util.find_spec("openai")
            if spec is None:
                raise ImportError("openai package not installed; install or switch to VLLMLLM.")
            self._openai = importlib.import_module("openai")
        return self._openai

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, prompt: str) -> str:
        openai = self._client()
        messages = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response


class VLLMLLM:
    """vLLM-backed generator for local open-source models."""

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        quantization: Optional[str] = "awq",
        max_model_len: int = 8192,
        sampling_config: Optional[VLLMSamplingConfig] = None,
        enforce_eager: bool = False,
    ):
        self.model = model
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("vllm is required for VLLMLLM. Install vllm or use HF engine.") from exc

        self._SamplingParams = SamplingParams
        self.sampling_config = sampling_config or VLLMSamplingConfig()
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )

    def __call__(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        sample_cfg = self._build_sampling_params(max_new_tokens=max_new_tokens)
        outputs = self.llm.generate([prompt], sampling_params=sample_cfg)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def _build_sampling_params(self, *, max_new_tokens: Optional[int] = None):
        cfg = self.sampling_config
        params = self._SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=max_new_tokens or cfg.max_new_tokens,
        )
        if cfg.repetition_penalty is not None:
            params.repetition_penalty = cfg.repetition_penalty
        return params


class HFLLM:
    """Transformers fallback for environments without vLLM."""

    def __init__(
        self,
        model: str,
        *,
        device_map: str | int | None = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import to avoid hard dep during lint

        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        except ValueError as exc:
            raise ValueError(
                "Failed to load model with safe loading. "
                "Please use a safetensors checkpoint or upgrade torch to >=2.6."
            ) from exc

    def __call__(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


class FastChatLLM:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt):
        conv = get_conversation_template('vicuna-7b-1.5')
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input = conv.get_prompt()

        input_ids = self.tokenizer([input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).to(self.model.device),
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
        )

        output_ids = output_ids[0][len(input_ids[0]) :]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return response


class NShotLLM:
    def __init__(self, model=None, tokenizer=None, reward_model=None, num_shots=4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_shots = num_shots

    def queries_to_scores(self, list_of_strings):
        return [output["score"] for output in self.reward_model(list_of_strings)]

    def __call__(self, prompt):
        query = self.tokenizer.encode(prompt, return_tensors="pt")
        queries = query.repeat((self.num_shots, 1))
        output_ids = self.model.generate(
            queries,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        scores = torch.tensor(self.queries_to_scores(output))
        output_ids = output_ids[scores.topk(1).indices[0]][len(query[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

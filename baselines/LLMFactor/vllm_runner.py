"""Lightweight vLLM wrapper local to LLMFactor."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

os.environ.setdefault("VLLM_WORKER_MULTIPROCESS_START_METHOD", "spawn")


@dataclass
class PromptLike:
    system: str
    user: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None


@dataclass
class VLLMConfig:
    model: str
    quantization: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: Optional[float] = 0.8
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = False
    enforce_eager: Optional[bool] = None


class VLLMRunner:
    def __init__(self, cfg: VLLMConfig):
        try:
            from vllm import LLM  # type: ignore
        except Exception as exc:
            raise ImportError("vllm is required for awq_vllm backend. Please install vllm in the active environment.") from exc

        kwargs = {
            "model": cfg.model,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.quantization and cfg.quantization.lower() != "none":
            kwargs["quantization"] = cfg.quantization
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        if cfg.gpu_memory_utilization:
            kwargs["gpu_memory_utilization"] = cfg.gpu_memory_utilization
        if cfg.dtype:
            kwargs["dtype"] = cfg.dtype
        if cfg.enforce_eager is not None:
            kwargs["enforce_eager"] = cfg.enforce_eager

        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def render_prompts(self, items: List[PromptLike]) -> List[str]:
        return self._build_prompts(items)

    def count_tokens(self, rendered_prompts: List[str]) -> List[int]:
        lengths: List[int] = []
        for prompt in rendered_prompts:
            try:
                tokens = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
                lengths.append(tokens.shape[-1])
            except Exception:
                try:
                    lengths.append(len(self._tokenizer.encode(prompt)))
                except Exception:
                    lengths.append(0)
        return lengths

    def _build_prompts(self, items: List[PromptLike]) -> List[str]:
        prompts: List[str] = []
        for req in items:
            messages = []
            if req.system:
                messages.append({"role": "system", "content": req.system})
            messages.append({"role": "user", "content": req.user})
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"{req.system}\n\n{req.user}"
            prompts.append(prompt)
        return prompts

    def generate(self, requests: Iterable[PromptLike], rendered_prompts: Optional[List[str]] = None) -> List[str]:
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:
            raise ImportError("vllm is required for awq_vllm backend. Please install vllm in the active environment.") from exc

        req_list = list(requests)
        if not req_list:
            return []

        max_tokens = _resolve_param(req_list, "max_tokens", fallback=512)
        temperature = _resolve_param(req_list, "temperature", fallback=0.0)
        top_p = _resolve_param(req_list, "top_p", fallback=1.0)
        stops: List[str] = []
        for req in req_list:
            if req.stop:
                stops.extend([s for s in req.stop if s not in stops])

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stops or None,
        )

        prompts = rendered_prompts if rendered_prompts is not None else self._build_prompts(req_list)
        results = self._llm.generate(prompts, sampling)
        outputs: List[str] = []
        for res in results:
            if res.outputs:
                outputs.append(res.outputs[0].text)
            else:
                outputs.append("")
        return outputs


def _resolve_param(reqs: List[PromptLike], field: str, fallback):
    for req in reqs:
        val = getattr(req, field, None)
        if val is not None:
            return val
    return fallback


__all__ = ["PromptLike", "VLLMConfig", "VLLMRunner"]

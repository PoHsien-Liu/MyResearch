"""Batch inference helper for local vLLM (no HTTP server required)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from STARE.llm_backend.llm_config import get_backend_config

if TYPE_CHECKING:
    from vllm import LLM  # type: ignore

# cache key: (model_name, quantization, tp_size, dtype, max_model_len, gpu_mem_util, enforce_eager)
_LLM_CACHE: Dict[
    Tuple[str, str, int, Optional[str], Optional[int], Optional[float], Optional[bool]],
    "LLM",
] = {}


@dataclass
class PromptLike:
    system: str
    user: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None


def _get_llm(
    model_name: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    dtype: Optional[str],
    max_model_len: Optional[int],
    gpu_memory_utilization: Optional[float],
    trust_remote_code: bool,
    enforce_eager: Optional[bool],
) -> "LLM":
    from vllm import LLM  # type: ignore

    key = (
        model_name,
        quantization or "none",
        tensor_parallel_size,
        dtype,
        max_model_len,
        gpu_memory_utilization,
        enforce_eager,
    )
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if quantization:
        kwargs["quantization"] = quantization
    if dtype:
        kwargs["dtype"] = dtype
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if enforce_eager is not None:
        kwargs["enforce_eager"] = enforce_eager
    llm = LLM(**kwargs)
    _LLM_CACHE[key] = llm
    return llm


def clear_llm_cache() -> None:
    """Release cached LLM instances to free GPU memory."""
    try:
        for llm in _LLM_CACHE.values():
            try:
                del llm
            except Exception:
                pass
        _LLM_CACHE.clear()
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            import torch  # type: ignore

            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        _LLM_CACHE.clear()


def _build_prompts(llm: "LLM", items: List[PromptLike]) -> List[str]:
    tok = llm.get_tokenizer()
    prompts: List[str] = []
    for req in items:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.user})
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation if template unavailable
            prompt = f"{req.system}\n\n{req.user}"
        prompts.append(str(prompt))
    return prompts


def run_inference_batch(
    requests: Iterable[PromptLike],
    backend: str = "llama_70B",
    model: Optional[str] = None,
    **gen_kwargs: Any,
) -> List[str]:
    """Run batch inference locally using vLLM (no HTTP server needed)."""
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("vllm is required for local inference. pip install vllm") from exc

    req_list = list(requests)
    if not req_list:
        return []

    cfg = get_backend_config(backend) or {}
    model_name = model or cfg.get("default_model")
    if not model_name:
        raise ValueError("Model name must be provided either globally or via config.default_model.")

    quantization = cfg.get("quantization") or cfg.get("precision")
    # If quantization specified as awq but model is not awq-weighted, disable to avoid load error.
    if quantization and isinstance(quantization, str):
        q_lower = quantization.lower()
        if q_lower != "none" and "awq" in q_lower and "awq" not in model_name.lower():
            logging.getLogger("stare.llm").warning(
                "Requested quantization=%s but model %s is not awq; disabling quantization.",
                quantization,
                model_name,
            )
            quantization = None
    if isinstance(quantization, str) and quantization.lower() == "none":
        quantization = None
    tp_size = int(cfg.get("tensor_parallel_size", 8))
    dtype = cfg.get("dtype")
    max_model_len = cfg.get("max_model_len") or cfg.get("max_sequence_length") or 4096
    gpu_mem_util = cfg.get("gpu_memory_utilization", 0.6)
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    enforce_eager = cfg.get("enforce_eager", None)

    llm = _get_llm(
        model_name=model_name,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
    )

    prompts = _build_prompts(llm, req_list)
    sampling = SamplingParams(
        max_tokens=_resolve_param(req_list, cfg, gen_kwargs, "max_tokens", fallback=cfg.get("max_new_tokens", 512)),
        temperature=_resolve_param(req_list, cfg, gen_kwargs, "temperature", fallback=cfg.get("temperature", 0.0)),
        top_p=_resolve_param(req_list, cfg, gen_kwargs, "top_p", fallback=cfg.get("top_p", 1.0)),
    )
    # stop sequences: merge all unique stop tokens across requests
    stops: List[str] = []
    for req in req_list:
        if req.stop:
            stops.extend([s for s in req.stop if s not in stops])
    if stops:
        sampling.stop = stops

    outputs: List[str] = []
    results = llm.generate(prompts, sampling)
    for res in results:
        if res.outputs:
            outputs.append(res.outputs[0].text)
        else:
            outputs.append("")
    return outputs


def _resolve_param(
    req_list: List[PromptLike],
    cfg: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
    field: str,
    fallback: Any,
) -> Any:
    """Resolve a generation param with per-request override, then global kwargs, then config or fallback."""
    for req in req_list:
        val = getattr(req, field, None)
        if val is not None:
            return val
    if field in gen_kwargs and gen_kwargs[field] is not None:
        return gen_kwargs[field]
    return cfg.get(field, fallback)

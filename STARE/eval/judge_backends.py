"""Judge backend adapter for explanation evaluation."""
from __future__ import annotations

import os
from typing import Literal, Optional

from STARE.llm_backend.llm_config import get_backend_config
from STARE.llm_backend.inference import PromptLike, run_inference_batch

BackendName = Literal["openai", "gemini", "qwen", "llama"]


def _default_base_url(backend: BackendName) -> Optional[str]:
    """Choose a base_url override for OpenAI-compatible endpoints."""
    if backend == "qwen":
        return os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if backend == "llama":
        return os.getenv("LLAMA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if backend == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if backend == "gemini":
        return os.getenv("GEMINI_BASE_URL")
    return None


def call_judge_backend(
    backend: BackendName,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Dispatch to local vLLM (preferred) or OpenAI-compatible endpoint if configured."""
    cfg = get_backend_config(backend)
    base_url = _default_base_url(backend) or cfg.get("base_url")
    api_key = (
        os.getenv(f"{backend.upper()}_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or cfg.get("api_key", "")
    )

    # Prefer local vLLM path when no base_url is provided
    if not base_url:
        req = PromptLike(
            system=system_prompt,
            user=user_prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return run_inference_batch(
            [req],
            backend=backend,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )[0]

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("openai package is required when base_url is set.") from exc

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content  # type: ignore[return-value]

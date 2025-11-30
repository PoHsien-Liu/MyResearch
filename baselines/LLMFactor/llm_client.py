"""LLM client wrapper for LLMFactor (local vLLM) with model presets."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.LLMFactor.vllm_runner import PromptLike, VLLMConfig, VLLMRunner  # noqa: E402

DEFAULT_AWQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


@dataclass
class GenerationConfig:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def _available_gpu_count() -> int:
    try:
        import torch  # type: ignore

        return torch.cuda.device_count()
    except Exception:
        return 0


def _load_llm_config() -> Tuple[dict, Optional[Path]]:
    """Load models config from llm_config.yaml first, then llm_config.example.yaml."""
    for candidate in (MODULE_DIR / "llm_config.yaml", MODULE_DIR / "llm_config.example.yaml"):
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            presets = data.get("models") or {}
            if isinstance(presets, dict):
                return presets, candidate
        except Exception:
            continue
    return {}, None


def _merge_gen_config(gen_cfg: Optional[GenerationConfig], cfg: dict) -> GenerationConfig:
    base = gen_cfg or GenerationConfig()
    return GenerationConfig(
        max_tokens=base.max_tokens if base.max_tokens is not None else cfg.get("max_tokens"),
        temperature=base.temperature if base.temperature is not None else cfg.get("temperature"),
        top_p=base.top_p if base.top_p is not None else cfg.get("top_p"),
    )


def _resolve_tp(cfg: dict) -> int:
    tp = int(cfg.get("tensor_parallel_size") or 1)
    avail = _available_gpu_count()
    if avail == 0:
        return 1
    return min(tp, avail) if tp > 0 else avail


class LLMClient:
    def __init__(
        self,
        *,
        backend: str = "awq_vllm",
        base_model: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None,
        model_preset: Optional[str] = None,
        logger=None,
    ):
        self.backend = backend
        self.logger = logger
        self._runner = None
        presets, cfg_path = _load_llm_config()
        preset_name = model_preset or "default"
        cfg_data = presets.get(preset_name, {}) if isinstance(presets, dict) else {}
        self.config_source = str(cfg_path) if cfg_path else None
        self.base_model = base_model or cfg_data.get("default_model") or cfg_data.get("model") or DEFAULT_AWQ_MODEL
        self.gen_config = _merge_gen_config(gen_config, cfg_data)

        if self.backend == "awq_vllm":
            cfg = VLLMConfig(
                model=self.base_model,
                quantization=cfg_data.get("quantization") or ("awq" if "awq" in self.base_model.lower() else None),
                tensor_parallel_size=_resolve_tp(cfg_data),
                gpu_memory_utilization=cfg_data.get("gpu_memory_utilization"),
                max_model_len=cfg_data.get("max_model_len"),
                dtype=cfg_data.get("dtype"),
                trust_remote_code=bool(cfg_data.get("trust_remote_code", False)),
                enforce_eager=cfg_data.get("enforce_eager"),
            )
            self._runner = VLLMRunner(cfg)
            if self.logger:
                self.logger.info(
                    "Initialized local VLLMRunner with model=%s tp=%s max_model_len=%s config_source=%s preset=%s",
                    cfg.model,
                    cfg.tensor_parallel_size,
                    cfg.max_model_len,
                    str(cfg_path) if cfg_path else "built-in",
                    preset_name,
                )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompts: Iterable["PromptLike"]) -> List[str]:
        requests: List[PromptLike] = []
        for p in prompts:
            requests.append(
                PromptLike(
                    system=p.system,
                    user=p.user,
                    max_tokens=self.gen_config.max_tokens or 512,
                    temperature=self.gen_config.temperature if self.gen_config.temperature is not None else 0.0,
                    top_p=self.gen_config.top_p if self.gen_config.top_p is not None else 1.0,
                )
            )
        rendered = self._runner.render_prompts(requests)
        if self.logger:
            lengths = self._runner.count_tokens(rendered)
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                self.logger.info(
                    "Prompt tokens: avg=%.1f max=%d min=%d (n=%d)",
                    avg_len,
                    max(lengths),
                    min(lengths),
                    len(lengths),
                )
        return self._runner.generate(requests, rendered_prompts=rendered)


__all__ = ["LLMClient", "GenerationConfig", "DEFAULT_AWQ_MODEL"]

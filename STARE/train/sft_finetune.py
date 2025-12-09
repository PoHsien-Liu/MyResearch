"""SFT data prep and CLI bindings (finetune task skeleton)."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from STARE.utils.paths import ensure_dir, get_outputs_dir, get_pipeline_data_dir

LOGGER = logging.getLogger("stare.sft")


# -----------------------------------------------------------------------------
# Small helpers and data containers
# -----------------------------------------------------------------------------

def _model_slug(name: Optional[str]) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    return slug.replace(" ", "-")


def _auto_sft_path(dataset: str, model_slug: str, experiment: str, split: str) -> Path:
    return (
        get_pipeline_data_dir()
        / "sft"
        / "samples"
        / dataset.upper()
        / model_slug
        / experiment
        / f"sft_samples_{split}.jsonl"
    )


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class SFTSplitData:
    split: str
    samples: List[Dict]


@dataclass
class TokenizationStats:
    count: int
    min_len: int
    max_len: int
    mean_len: float
    pct_95: float
    pct_99: float


class ConversationDataset(Dataset):
    """Simple torch Dataset wrapping tokenized samples."""

    def __init__(self, features: List[Dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.features[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_sft_file(path: Path, expected_split: Optional[str] = None) -> SFTSplitData:
    items: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"SFT file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                LOGGER.warning("Skip malformed line in %s: %s", path, exc)
                continue
            split = rec.get("metadata", {}).get("sft_split") or expected_split or "sft_train"
            rec.setdefault("metadata", {})["sft_split"] = split
            items.append(rec)
    return SFTSplitData(split=expected_split or "unknown", samples=items)


# -----------------------------------------------------------------------------
# Tokenizer and tokenization helpers
# -----------------------------------------------------------------------------

def bind_sft_finetune_args(parser) -> None:
    parser.add_argument("--train_sft_path", default=None, help="Path to sft_samples_sft_train.jsonl")
    parser.add_argument("--val_sft_path", default=None, help="Path to sft_samples_sft_val.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for SFT")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for SFT")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Label ignore index for non-target tokens")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--lr_scheduler_type", default="linear", help="LR scheduler type")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Eval every N steps (if val set exists)")
    parser.add_argument("--save_steps", type=int, default=200, help="Save every N steps")
    parser.add_argument("--sft_max_train_samples", type=int, default=None, help="Optional cap on training samples (SFT)")
    parser.add_argument("--sft_max_eval_samples", type=int, default=None, help="Optional cap on eval samples (SFT)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    # LoRA options
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA / train full model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated target modules for LoRA")
    parser.add_argument("--eval_generate_samples", type=int, default=0, help="If >0, generate this many val samples for inspection")
    parser.add_argument("--generation_max_new_tokens", type=int, default=64, help="Max new tokens when generating eval samples")


def _prepare_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _build_prompt_text(tokenizer, messages: List[Dict]) -> Tuple[str, str]:
    """Return (prompt_text_without_assistant, assistant_text)."""
    if not messages:
        raise ValueError("messages empty")
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        raise ValueError("assistant message missing")
    assistant_text = assistant_msgs[0].get("content") or ""
    prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
    prompt_text = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = None
    if prompt_text is None:
        parts = []
        for m in prompt_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        parts.append("[ASSISTANT]")
        prompt_text = "\n".join(parts)
    return prompt_text, assistant_text


def _truncate_to_max(input_ids: List[int], labels: List[int], max_len: int, ignore_index: int):
    if len(input_ids) <= max_len:
        return input_ids, labels
    input_ids = input_ids[-max_len:]
    labels = labels[-max_len:]
    if labels and labels[0] != ignore_index:
        labels[0] = ignore_index
    return input_ids, labels


def _tokenize_sample(
    sample: Dict,
    tokenizer,
    max_length: int,
    ignore_index: int,
) -> Optional[Dict]:
    messages = sample.get("messages") or []
    try:
        prompt_text, assistant_text = _build_prompt_text(tokenizer, messages)
    except Exception as exc:
        LOGGER.warning("Skip sample due to prompt build error: %s", exc)
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_attention_mask=False).input_ids
    full_text = prompt_text + (assistant_text or "")
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        return_attention_mask=False,
    ).input_ids

    labels = list(full_ids)
    prompt_len = len(prompt_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = ignore_index

    full_ids, labels = _truncate_to_max(full_ids, labels, max_length, ignore_index)
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": sample.get("metadata", {}),
    }


def _tokenize_split(samples: List[Dict], tokenizer, max_length: int, ignore_index: int) -> Tuple[List[Dict], TokenizationStats]:
    features: List[Dict] = []
    lengths: List[int] = []
    for sample in samples:
        feat = _tokenize_sample(sample, tokenizer, max_length=max_length, ignore_index=ignore_index)
        if feat is None:
            continue
        features.append(feat)
        lengths.append(len(feat["input_ids"]))
    if not lengths:
        stats = TokenizationStats(count=0, min_len=0, max_len=0, mean_len=0.0, pct_95=0.0, pct_99=0.0)
    else:
        sorted_len = sorted(lengths)
        n = len(sorted_len)

        def pct(p: float) -> int:
            idx = min(n - 1, math.ceil(p * n) - 1)
            return sorted_len[idx]

        stats = TokenizationStats(
            count=len(sorted_len),
            min_len=sorted_len[0],
            max_len=sorted_len[-1],
            mean_len=sum(sorted_len) / len(sorted_len),
            pct_95=pct(0.95),
            pct_99=pct(0.99),
        )
    return features, stats


# -----------------------------------------------------------------------------
# Task entry
# -----------------------------------------------------------------------------

def run_sft_finetune_task(args) -> None:
    base_model = getattr(args, "base_model", None)
    if not base_model:
        raise ValueError("--base_model is required for sft_finetune")
    dataset = args.dataset_name.upper()
    exp = getattr(args, "experiment_name", None) or str(int(time.time()))
    model_slug = _model_slug(base_model)
    out_dir = ensure_dir(get_pipeline_data_dir() / "sft" / "checkpoints" / dataset / model_slug / exp)

    train_path = Path(getattr(args, "train_sft_path", "") or _auto_sft_path(dataset, model_slug, exp, "sft_train"))
    val_path_arg = getattr(args, "val_sft_path", None)
    val_path = Path(val_path_arg) if val_path_arg else _auto_sft_path(dataset, model_slug, exp, "sft_val")

    train_data = _load_sft_file(train_path, expected_split="sft_train")
    val_data: Optional[SFTSplitData] = None
    if val_path and val_path.exists():
        val_data = _load_sft_file(val_path, expected_split="sft_val")
    else:
        LOGGER.warning("Val SFT file not found at %s; continuing with train only", val_path)

    tokenizer = _prepare_tokenizer(base_model)
    ignore_index = int(getattr(args, "ignore_index", -100))
    max_len = int(getattr(args, "max_seq_length", 2048))

    train_feats, train_tok_stats = _tokenize_split(train_data.samples, tokenizer, max_length=max_len, ignore_index=ignore_index)
    val_feats: List[Dict] = []
    val_tok_stats = TokenizationStats(count=0, min_len=0, max_len=0, mean_len=0.0, pct_95=0.0, pct_99=0.0)
    if val_data:
        val_feats, val_tok_stats = _tokenize_split(val_data.samples, tokenizer, max_length=max_len, ignore_index=ignore_index)

    train_ds = ConversationDataset(train_feats)
    val_ds = ConversationDataset(val_feats) if val_feats else None

    # ------------------------------------------------------------------
    # Model & PEFT setup
    # ------------------------------------------------------------------
    set_seed(int(getattr(args, "seed", 42)))

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if getattr(args, "bf16", False) else None,
        device_map="auto",
    )

    if not getattr(args, "no_lora", False) and getattr(args, "lora_r", 0) > 0:
        target_modules = [m.strip() for m in str(getattr(args, "lora_target_modules", "")).split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=int(getattr(args, "lora_r", 16)),
            lora_alpha=int(getattr(args, "lora_alpha", 32)),
            lora_dropout=float(getattr(args, "lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or None,
        )
        model = get_peft_model(model, lora_cfg)

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    logging_dir = ensure_dir(out_dir / "logs")

    max_train_samples = getattr(args, "sft_max_train_samples", None)
    max_eval_samples = getattr(args, "sft_max_eval_samples", None)
    if max_train_samples:
        keep = min(len(train_ds), max_train_samples)
        train_feats = train_feats[:keep]
        train_ds = ConversationDataset(train_feats)
    if max_eval_samples and val_ds:
        keep = min(len(val_ds), max_eval_samples)
        val_feats = val_feats[:keep]
        val_ds = ConversationDataset(val_feats)

    eval_strategy = "no"
    if val_ds and len(val_ds) > 0:
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(getattr(args, "num_train_epochs", 1)),
        per_device_train_batch_size=int(getattr(args, "per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(getattr(args, "per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(getattr(args, "gradient_accumulation_steps", 1)),
        learning_rate=float(getattr(args, "learning_rate", 2e-5)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        warmup_steps=int(getattr(args, "warmup_steps", 0)),
        lr_scheduler_type=str(getattr(args, "lr_scheduler_type", "linear")),
        logging_strategy="steps",
        logging_steps=int(getattr(args, "logging_steps", 50)),
        eval_strategy=eval_strategy,
        eval_steps=int(getattr(args, "eval_steps", 200)),
        save_strategy="steps",
        save_steps=int(getattr(args, "save_steps", 200)),
        bf16=bool(getattr(args, "bf16", False)),
        fp16=bool(getattr(args, "fp16", False)),
        gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
        load_best_model_at_end=eval_strategy != "no",
        logging_dir=str(logging_dir),
        report_to=["none"],
        save_total_limit=2,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        max_length=max_len,
        pad_to_multiple_of=8 if (getattr(args, "bf16", False) or getattr(args, "fp16", False)) else None,
        label_pad_token_id=ignore_index,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(str(checkpoint_dir / "last"))
    metrics = train_result.metrics
    if val_ds:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    stats = {
        "dataset": dataset,
        "base_model": base_model,
        "experiment_name": exp,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "counts": {
            "train": len(train_data.samples),
            "val": len(val_data.samples) if val_data else 0,
        },
        "tokenization": {
            "max_seq_length": max_len,
            "ignore_index": ignore_index,
            "train": asdict(train_tok_stats),
            "val": asdict(val_tok_stats),
        },
    }
    _write_json(out_dir / "sft_data_stats.json", stats)
    _write_json(out_dir / "args.json", {k: v for k, v in vars(args).items()})

    log_lines = [
        f"[{time.strftime('%F %T')}] sft_finetune start",
        f"dataset={dataset} model={base_model} exp={exp}",
        f"train_samples={len(train_data.samples)} val_samples={len(val_data.samples) if val_data else 0}",
        f"tokenized_train={len(train_ds)} tokenized_val={len(val_ds) if val_ds else 0}",
    ]
    with (out_dir / "run.log").open("a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    LOGGER.info("SFT data prepared at %s", out_dir)
    LOGGER.info("Train samples: %d | Val samples: %d", len(train_data.samples), len(val_data.samples) if val_data else 0)
    LOGGER.info("Tokenized train=%d val=%d", len(train_ds), len(val_ds) if val_ds else 0)

    # ------------------------------------------------------------------
    # Metrics + optional generation
    # ------------------------------------------------------------------
    metrics_path = out_dir / "sft_metrics.json"
    _write_json(metrics_path, metrics)
    LOGGER.info("Training finished. Metrics stored at %s", metrics_path)
    # also write eval.json for consistency with其他模組
    eval_summary = {
        "train_loss": metrics.get("train_loss"),
        "train_runtime": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "eval_loss": metrics.get("eval_loss") if "eval_loss" in metrics else None,
        "eval_steps": metrics.get("eval_steps") if "eval_steps" in metrics else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint if hasattr(trainer, "state") else None,
    }
    _write_json(out_dir / "eval.json", eval_summary)
    try:
        state_payload = json.loads(trainer.state.to_json_string())
    except Exception:
        state_payload = getattr(trainer.state, "__dict__", {})
    _write_json(out_dir / "trainer_state.json", state_payload)

    # Optional: generate a few val samples for sanity check
    gen_samples = int(getattr(args, "eval_generate_samples", 0))
    if gen_samples > 0 and val_ds and len(val_ds) > 0:
        model.eval()
        n = min(gen_samples, len(val_ds))
        outputs = []
        for idx in range(n):
            feat = val_feats[idx]
            input_ids = torch.tensor([feat["input_ids"]], device=trainer.model.device)
            attn = torch.tensor([feat["attention_mask"]], device=trainer.model.device)
            with torch.no_grad():
                gen_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=int(getattr(args, "generation_max_new_tokens", 64)),
                    do_sample=False,
                )
            text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            outputs.append({
                "sample_idx": idx,
                "text": text,
                "metadata": feat.get("metadata", {}),
            })
        _write_json(out_dir / "val_generations.json", {"samples": outputs})
        LOGGER.info("Generated %d val samples for inspection", len(outputs))

"""Stage2 finetuning with pseudo explanations (prediction + reason, no citation)."""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from peft import PeftModel

from STARE.utils.paths import ensure_dir, stage1_model_dir, stage2_model_dir

LOGGER = logging.getLogger("stare.train.stage2")


IGNORE_INDEX = -100


def _label_to_id(label: str) -> int:
    normalized = (label or "").strip().upper()
    up_labels = {"UP", "1", "POSITIVE"}
    if normalized in up_labels:
        return 1
    return 0


@dataclass
class Stage2Sample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    reason_mask: List[int]
    cls_index: int
    label_id: int


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as exc:
                LOGGER.warning("Skip malformed line in %s: %s", path, exc)
    return items


def _serialize_target(target_json: Dict[str, Any]) -> str:
    return json.dumps(target_json, ensure_ascii=False)


def _tokenize_sample(
    tokenizer,
    prompt: str,
    target_json: Dict[str, Any],
    label: str,
    max_length: int,
) -> Stage2Sample:
    target_text = _serialize_target(target_json)
    full_text = prompt.rstrip() + "\n\n" + target_text
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    # Determine spans
    prompt_len_chars = len(prompt.rstrip()) + 2  # "\n\n"
    label_text = target_json.get("prediction", "")
    reason_text = target_json.get("reason", "")
    if not label_text or not reason_text:
        raise ValueError("target_json missing prediction/reason")
    label_start = target_text.find(label_text)
    if label_start == -1:
        raise ValueError("label text not found in target_text")
    label_start_abs = prompt_len_chars + label_start
    label_end_abs = label_start_abs + len(label_text)
    reason_start = target_text.find(reason_text)
    if reason_start == -1:
        raise ValueError("reason text not found in target_text")
    reason_start_abs = prompt_len_chars + reason_start
    reason_end_abs = reason_start_abs + len(reason_text)

    labels = [IGNORE_INDEX] * len(input_ids)
    reason_mask = [0] * len(input_ids)
    cls_index = -1
    for idx, (start, end) in enumerate(offsets):
        # classification token: token covering label start
        if cls_index == -1 and start <= label_start_abs < end:
            cls_index = idx
        # reason tokens: any overlap with reason span
        if start < reason_end_abs and end > reason_start_abs:
            labels[idx] = input_ids[idx]
            reason_mask[idx] = 1
    if cls_index <= 0:
        raise ValueError("cls_index not found or at position 0 (cannot shift)")
    return Stage2Sample(
        input_ids=input_ids,
        attention_mask=encoded["attention_mask"],
        labels=labels,
        reason_mask=reason_mask,
        cls_index=cls_index,
        label_id=_label_to_id(label),
    )


class Stage2Dataset(Dataset):
    def __init__(self, samples: List[Stage2Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(s.attention_mask, dtype=torch.long),
            "labels": torch.tensor(s.labels, dtype=torch.long),
            "reason_mask": torch.tensor(s.reason_mask, dtype=torch.long),
            "cls_index": torch.tensor(s.cls_index, dtype=torch.long),
            "label_id": torch.tensor(s.label_id, dtype=torch.long),
        }


def _collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    # Manual padding to also pad reason_mask and cls_index
    max_len = max(item["input_ids"].shape[0] for item in batch)
    def pad_tensor(t: torch.Tensor, pad_value: int) -> torch.Tensor:
        if t.shape[0] == max_len:
            return t
        pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    input_ids = torch.stack([pad_tensor(b["input_ids"], pad_token_id) for b in batch])
    attention_mask = torch.stack([pad_tensor(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([pad_tensor(b["labels"], IGNORE_INDEX) for b in batch])
    reason_mask = torch.stack([pad_tensor(b["reason_mask"], 0) for b in batch])
    # cls_index stays per-sample; no padding needed
    cls_index = torch.stack([b["cls_index"] for b in batch])
    label_id = torch.stack([b["label_id"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "reason_mask": reason_mask,
        "cls_index": cls_index,
        "label_id": label_id,
    }


class Stage2Trainer(Trainer):
    def __init__(self, lambda_cls: float, lambda_lm: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cls = lambda_cls
        self.lambda_lm = lambda_lm

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        reason_mask = inputs.pop("reason_mask")
        cls_index = inputs.pop("cls_index")
        label_id = inputs.pop("label_id")

        outputs = model(**inputs)
        logits = outputs.logits  # (bsz, seq, vocab)

        # LM loss on reason tokens (shifted)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_reason_mask = reason_mask[:, 1:].contiguous()
        lm_loss = None
        active_positions = shift_reason_mask.view(-1) > 0
        if active_positions.any():
            vocab_size = shift_logits.size(-1)
            lm_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size)[active_positions],
                shift_labels.view(-1)[active_positions],
            )
        else:
            lm_loss = torch.tensor(0.0, device=logits.device)

        # Classification loss on prediction token (logits predicting the label token)
        cls_logits: List[torch.Tensor] = []
        cls_labels: List[int] = []
        for b in range(logits.size(0)):
            pos = cls_index[b].item() - 1  # use previous position logits
            if pos < 0 or pos >= logits.size(1):
                continue
            cls_logits.append(logits[b, pos])
            cls_labels.append(label_id[b].item())
        if cls_logits:
            stacked_logits = torch.stack(cls_logits, dim=0)
            cls_labels_t = torch.tensor(cls_labels, device=logits.device, dtype=torch.long)
            cls_loss = torch.nn.functional.cross_entropy(stacked_logits, cls_labels_t)
        else:
            cls_loss = torch.tensor(0.0, device=logits.device)

        loss = self.lambda_cls * cls_loss + self.lambda_lm * lm_loss
        return (loss, outputs) if return_outputs else loss


def _build_samples(data_path: Path, tokenizer, max_length: int) -> List[Stage2Sample]:
    records = _load_jsonl(data_path)
    samples: List[Stage2Sample] = []
    for rec in records:
        prompt = rec.get("input")
        target_json = rec.get("target_json")
        label = rec.get("label")
        if not prompt or not target_json or not label:
            LOGGER.warning("Skip record missing fields in %s", data_path)
            continue
        try:
            samples.append(_tokenize_sample(tokenizer, prompt, target_json, label, max_length=max_length))
        except Exception as exc:
            LOGGER.warning("Skip record due to tokenization error: %s", exc)
            continue
    if not samples:
        raise RuntimeError(f"No valid samples in {data_path}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 finetuning with pseudo explanations")
    parser.add_argument("--train_file", required=True, help="Path to stage2_train_sft.jsonl")
    parser.add_argument("--validation_file", required=True, help="Path to stage2_valid_sft.jsonl")
    parser.add_argument("--base_model", required=True, help="Base causal LM model (HF path/name)")
    parser.add_argument("--stage1_lora_path", required=True, help="Path to Stage1 LoRA adapter")
    parser.add_argument("--dataset_name", required=True, help="Dataset key (e.g., CMIN)")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    parser.add_argument("--output_dir", default=None, help="Output dir for Stage2 adapter")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="Weight for classification loss")
    parser.add_argument("--lambda_lm", type=float, default=1.0, help="Weight for reason LM loss")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap on train samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap on eval samples")
    return parser.parse_args()


def run_stage2_with_explanations(args: argparse.Namespace) -> Path:
    """Run Stage2 training; returns output_dir path."""
    set_seed(args.seed)

    experiment = args.experiment_name or str(int(time.time()))
    dataset = args.dataset_name.upper()
    default_out = stage2_model_dir(dataset, experiment)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_out
    ensure_dir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_samples = _build_samples(Path(args.train_file).expanduser().resolve(), tokenizer, max_length=args.max_seq_length)
    eval_samples = _build_samples(Path(args.validation_file).expanduser().resolve(), tokenizer, max_length=args.max_seq_length)
    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_eval_samples:
        eval_samples = eval_samples[: args.max_eval_samples]

    train_ds = Stage2Dataset(train_samples)
    eval_ds = Stage2Dataset(eval_samples)

    data_collator = lambda batch: _collate(batch, pad_token_id=tokenizer.pad_token_id)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.stage1_lora_path)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
    )

    trainer = Stage2Trainer(
        lambda_cls=args.lambda_cls,
        lambda_lm=args.lambda_lm,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Save config
    config_path = output_dir / "config_stage2.json"
    cfg = {
        "dataset": dataset,
        "experiment": experiment,
        "lambda_cls": args.lambda_cls,
        "lambda_lm": args.lambda_lm,
        "train_file": str(Path(args.train_file).resolve()),
        "validation_file": str(Path(args.validation_file).resolve()),
        "base_model": args.base_model,
        "stage1_lora_path": args.stage1_lora_path,
        "seed": args.seed,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    LOGGER.info("Stage2 training complete. Model saved to %s", output_dir)
    return output_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_stage2_with_explanations(args)


if __name__ == "__main__":
    main()

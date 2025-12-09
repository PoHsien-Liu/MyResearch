"""Run inference for STARE (base or adapter) and dump predictions."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel  
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datetime import datetime, timezone  
from common.data.loader import get_record, list_trading_days  
from STARE.models.STARE.prompt_builder import build_prediction_prompt_package  
from STARE.models.STARE.retriever import StareRetriever  
from STARE.utils.price import build_price_context, resolve_price_dir, last_k_returns  
from STARE.utils.paths import ensure_dir, get_outputs_dir  
from STARE.utils.seed import set_seed  
from STARE.eval.metrics import evaluate_predictions_file  


ALIAS_TO_LABEL = {
    "positive": "Positive",
    "pos": "Positive",
    "up": "Positive",
    "+": "Positive",
    "1": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "down": "Negative",
    "-": "Negative",
    "0": "Negative",
}


@dataclass
class SampleRecord:
    messages: List[Dict]
    metadata: Dict


def _model_slug(name: Optional[str]) -> str:
    if not name:
        return "default"
    slug = name.strip().lower().replace("/", "-")
    return slug.replace(" ", "-")


def load_sft_samples(path: Path, max_samples: Optional[int] = None) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            messages = obj.get("messages", [])
            # Strip assistant messages to avoid leaking labels during inference
            messages = [m for m in messages if m.get("role") != "assistant"]
            samples.append(SampleRecord(messages=messages, metadata=obj.get("metadata", {})))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def extract_first_json(text: str) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    stack: List[int] = []
    for idx, ch in enumerate(text):
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop(0)
            candidate = text[start : idx + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
    return None


def normalize_label(raw: object | None) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return ALIAS_TO_LABEL.get(text)


def parse_prediction(text: str) -> Tuple[Optional[str], Dict]:
    meta: Dict = {}
    json_text = extract_first_json(text)
    if json_text:
        try:
            parsed = json.loads(json_text)
            meta["parsed_json"] = parsed
            pred = normalize_label(parsed.get("prediction"))
            if pred:
                return pred, meta
        except Exception:
            meta["json_error"] = "failed to parse json prediction"
    lower = (text or "").lower()
    if "down" in lower and "up" not in lower:
        return "Negative", meta
    if "up" in lower and "down" not in lower:
        return "Positive", meta
    return None, meta


def render_prompt(tokenizer, messages: List[Dict]) -> str:
    chat_msgs = [m for m in messages if m.get("role") != "assistant"]
    if not chat_msgs:
        raise ValueError("No system/user messages provided.")
    try:
        return tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in chat_msgs:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)


def write_predictions(out_dir: Path, rows: List[Dict]) -> Tuple[Path, Path]:
    jsonl_path = out_dir / "predictions.jsonl"
    csv_path = out_dir / "predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for row in rows:
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return jsonl_path, csv_path


def build_output_dir(dataset: str, base_model: str, experiment: str, *, label_strategy: str, neg_threshold: float, pos_threshold: float) -> Path:
    model_slug = _model_slug(base_model)
    exp = experiment or str(int(time.time()))

    # strategy dir mirrors main.py behavior
    if label_strategy.lower() == "legacy":
        strategy_dir = "legacy"
    else:
        def _pct_tag(v: float) -> str:
            pct = v * 100
            txt = f"{pct:+.2f}".rstrip("0").rstrip(".")
            return txt.replace("+", "+").replace("-", "-") + "pct"
        strategy_dir = str(Path("dual") / f"neg{_pct_tag(neg_threshold)}_pos{_pct_tag(pos_threshold)}")

    return ensure_dir(get_outputs_dir() / "results" / dataset.upper() / strategy_dir / "STARE" / model_slug / exp)


def infer_experiment_name(base_model: str, adapter_path: Path) -> Optional[str]:
    model_slug = _model_slug(base_model)
    resolved = adapter_path.resolve()
    parents = list(resolved.parents)
    for idx, parent in enumerate(parents):
        if parent.name == model_slug and idx > 0:
            return parents[idx - 1].name
        if parent.parent and parent.parent.name == model_slug:
            return parent.name
    return None

def run_inference(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Path, Path]:
    samples = load_sft_samples(Path(args.sft_path), args.max_samples)
    if not samples:
        raise RuntimeError("No samples to evaluate.")

    output_dir = build_output_dir(
        args.dataset_name,
        args.base_model,
        args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    args_snapshot = {k: v for k, v in vars(args).items()}
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(args_snapshot, f, ensure_ascii=False, indent=2)

    model_id = args.adapter_path or args.base_model
    logger.info("Loading tokenizer/model %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_path:
        logger.info("Applying adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        logger.info("No adapter_path provided; running base model only.")
    model.eval()

    # Prepare rendered prompts for batching (chat template)
    prompts: List[str] = []
    sys_prompts: List[str] = []
    user_prompts: List[str] = []
    metas: List[Dict] = []
    for sample in samples:
        sys_prompt = ""
        user_prompt = ""
        for msg in sample.messages:
            if msg.get("role") == "system":
                sys_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")
        prompt_text = tok.apply_chat_template(sample.messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
        sys_prompts.append(sys_prompt)
        user_prompts.append(user_prompt)
        metas.append(sample.metadata or {})

    batch_size = max(1, int(getattr(args, "batch_size", 1)))
    rows: List[Dict] = []
    total = len(prompts)

    for start_idx in tqdm(range(0, total, batch_size), total=math.ceil(total / batch_size), desc="Inference"):
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]

        t0 = time.time()
        # Prompts are already chat-templated; avoid adding extra special tokens again.
        tokenized = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_len = tokenized["input_ids"].shape[1]
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                max_new_tokens=512,
                temperature=float(args.temperature),
                do_sample=float(args.temperature) > 0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        batch_elapsed_ms = (time.time() - t0) * 1000.0
        per_sample_time = batch_elapsed_ms / max(1, end_idx - start_idx)

        for i, seq in enumerate(out):
            gen_tokens = seq[input_len:]
            gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            global_idx = start_idx + i

            meta = metas[global_idx]
            pred_label, pred_meta = parse_prediction(gen_text)
            record = {
                "sample_id": f"{meta.get('ticker', '')}_{meta.get('target_date', '')}",
                "dataset": args.dataset_name,
                "method": "STARE",
                "model": args.base_model,
                "experiment_name": args.experiment_name or "adapter_eval",
                "ticker": meta.get("ticker"),
                "prediction_date": meta.get("target_date"),
                "ground_truth": meta.get("ground_truth_label"),
                "prediction": pred_label or "Unknown",
                "raw_response": gen_text,
                "system_prompt": sys_prompts[global_idx],
                "user_prompt": user_prompts[global_idx],
                "timing_ms": per_sample_time,
                "reason": None,
            }
            if "parsed_json" in pred_meta:
                record["reason"] = pred_meta["parsed_json"].get("reason")
            rows.append(record)

    jsonl_path, csv_path = write_predictions(output_dir, rows)
    logger.info("Predictions written to %s and %s", jsonl_path, csv_path)
    # auto eval
    start_ts = datetime.now(tz=timezone.utc)
    result = evaluate_predictions_file(Path(csv_path), unknown_policy="as_error")
    end_ts = datetime.now(tz=timezone.utc)
    duration_ms = int((end_ts - start_ts).total_seconds() * 1000)
    payload = {
        "args": {
            "predictions_path": str(csv_path),
            "unknown_policy": "as_error",
        },
        "label_policy": result["label_policy"],
        "classification_metrics": result["classification_metrics"],
        "sample_stats": result["sample_stats"],
        "explanation_metrics": {"status": "not_implemented"},
        "started_at": start_ts.isoformat().replace("+00:00", "Z"),
        "ended_at": end_ts.isoformat().replace("+00:00", "Z"),
        "duration_ms": duration_ms,
    }
    eval_path = Path(output_dir) / "eval.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved evaluation to %s", eval_path)
    return jsonl_path, csv_path


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("stare.inference_eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_inference_eval(args: argparse.Namespace) -> None:
    set_seed(int(getattr(args, "seed", 42)))
    args.experiment_name = args.experiment_name or str(int(time.time()))

    log_dir = build_output_dir(
        args.dataset_name,
        args.base_model,
        args.experiment_name,
        label_strategy=args.label_strategy,
        neg_threshold=args.neg_threshold,
        pos_threshold=args.pos_threshold,
    )
    logger = setup_logger(log_dir / "run.log")

    jsonl_path, csv_path = run_inference(args, logger)
    logger.info("Inference complete. JSONL: %s | CSV: %s", jsonl_path, csv_path)
    logger.info(
        "Next: predictions/eval located at %s", log_dir,
    )

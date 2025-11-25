import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from models.llm import LLaMALLM, FinGPTLLM
from dataloader.dataloader import DataLoader
from common.io.results import write_predictions_from_results, safe_name, write_training_data
from common.stock_direction import extract_stock_direction
from utils.prompts import (
    COMPANY_DESCRIPTION_INSTRUCTION,
    RELATIVE_COMPANY_INSTSRUCTION,
    PREDICT_INSTRUCTION_SYSTEM_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT
)
from utils.fewshots import PREDICT_FEW_SHOT_EXAMPLES
from utils.metrics import calculate_metrics, save_metrics


class PromptLabelDataset(TorchDataset):
    """Simple dataset that encodes prompt+label pairs for causal LM fine tuning."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt_text"]
        completion = sample["completion"]
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        full_encoding = self.tokenizer(
            prompt + completion,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)
        prompt_len = min(prompt_encoding["input_ids"].size(1), input_ids.size(0))
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class TDMLLM:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.method_name = "TDMLLM"
        self.project_root = Path(__file__).resolve().parents[1]
        self.company_desc_root = (
            self.project_root
            / "company_descriptions_cache"
            / self.args.dataset_name
            / safe_name(self.args.base_model)
        )
        self.company_desc_root.mkdir(parents=True, exist_ok=True)
        self.summary_max_new_tokens = getattr(args, "summary_max_new_tokens", 160)
        self.max_new_tokens_predict = getattr(args, "max_new_tokens_predict", 256)
        
        self.dataloader = DataLoader(args, logger)
        adapter_choice = getattr(args, "llm_adapter", "default")
        if adapter_choice == "fingpt":
            self.logger.info("🧩 Using FinGPT LLM adapter")
            self.llm = FinGPTLLM(args, logger)
        else:
            self.llm = LLaMALLM(args, logger)
        self.company_description_prompt = COMPANY_DESCRIPTION_INSTRUCTION
        self.relative_company_prompt = RELATIVE_COMPANY_INSTSRUCTION
        self.predict_instuction = {
            "system_prompt" : PREDICT_INSTRUCTION_SYSTEM_PROMPT,
            "user_prompt": PREDICT_INSTRUCTION_USER_PROMPT
        }
        self.predict_few_shot_examples = PREDICT_FEW_SHOT_EXAMPLES
        self.mode = getattr(args, "mode", "eval")
        self.train_epochs = max(1, getattr(args, "train_epochs", 2))
        self.train_batch_size = max(1, getattr(args, "train_batch_size", 8))
        self.train_max_length = getattr(args, "train_max_length", 512)
        self.train_lr = getattr(args, "train_lr", 5e-5)
        self.train_gradient_accumulation_steps = max(1, getattr(args, "train_gradient_accumulation_steps", 1))

    def eval(self):
        self.logger.info("🔍 Loading test data...")
        data = self.dataloader.load(flag='test')
        data.to_csv('data.csv')
        self.logger.info(f"✅ Loaded {len(data)} samples.")

        # 1. 批次產生所有 ticker 的 company_description（若 cache 無則生成）
        tickers = sorted(set(data['ticker']))
        ticker2desc = {}
        missing_tickers = []
        for ticker in tickers:
            cached = self._load_company_description(ticker)
            if cached is not None:
                ticker2desc[ticker] = cached
            else:
                missing_tickers.append(ticker)

        if missing_tickers:
            company_prompts = [self._build_relative_company_prompt(ticker) for ticker in missing_tickers]
            company_descriptions = self.llm.batch_inference(
                [""] * len(company_prompts),
                company_prompts,
                max_new_tokens=self.summary_max_new_tokens,
            )
            for ticker, desc in zip(missing_tickers, company_descriptions):
                ticker2desc[ticker] = desc
                self._save_company_description(ticker, desc)

        # 2. 在 DataFrame 新增欄位
        data['company_description'] = data['ticker'].map(lambda t: ticker2desc.get(t, ""))
        data['system_prompt'] = self.predict_instuction['system_prompt']
        data['user_prompt'] = data.apply(
            lambda row: self._build_predict_instruction(row['company_description'], row['summary']), axis=1
        )
        # 3. 直接用 DataFrame 欄位做批次推論
        system_prompts = data['system_prompt'].tolist()
        user_prompts = data['user_prompt'].tolist()
        predict_results = self.llm.batch_inference(
            system_prompts,
            user_prompts,
            max_new_tokens=self.max_new_tokens_predict,
        )
        data['predict_result'] = predict_results
        data['parsed_movement'] = data['predict_result'].apply(self._extract_stock_return)
        # 4. 結果後處理與儲存
        preds = data['parsed_movement'].tolist()
        labels = data['target'].tolist()
        correct = sum([p == l for p, l in zip(preds, labels)])
        incorrect = len(preds) - correct
        test_results = []
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx, row in data.iterrows():
            result_entry = self._create_result_entry(
                ticker=row['ticker'],
                prediction_date=row['end_date'],
                processing_date=current_date,
                ground_truth=row['target'],
                company_prompt=self._build_relative_company_prompt(row['ticker']),
                predict_prompt=row['user_prompt'],
                system_prompt=row['system_prompt'],
                raw_prediction=row['predict_result'],
                parsed_movement=row['parsed_movement'],
                summary=row['summary'],
                company_description=row['company_description']
            )
            test_results.append(result_entry)
        
        # 使用從 main() 傳遞過來的結果目錄
        results_dir = self.args.results_dir
        
        write_predictions_from_results(
            test_results,
            results_dir,
            dataset_name=self.args.dataset_name,
            method_name=self.method_name,
            base_model=self.args.base_model,
            experiment_name=self.args.experiment_name,
            store_raw=getattr(self.args, "store_raw", True),
            store_prompts=getattr(self.args, "store_prompts", False),
            truncate_chars=getattr(self.args, "truncate_chars", -1),
        )

        # 計算實驗總時長
        experiment_end_time = datetime.now()
        experiment_duration = experiment_end_time - self.args.experiment_start_time
        
        # 記錄實驗結束資訊
        self.logger.info(f"✅ Experiment completed at: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"⏱️ Total experiment duration: {experiment_duration}")
        
        metrics_result = calculate_metrics(preds, labels)
        save_metrics(metrics_result, self.args.base_model, results_dir, self.args.dataset_name, experiment_duration)

    def train(self):
        """Run a lightweight SFT pass using the shared train split."""
        self.logger.info("🧾 Loading training data...")
        train_data = self.dataloader.load(flag='train')
        self.logger.info(f"✅ Loaded {len(train_data)} training samples.")

        examples, records = self._prepare_training_examples(train_data)
        training_data_dir = os.path.join(self.args.results_dir, "training_data")
        training_data_path = write_training_data(records, training_data_dir)
        self.logger.info(f"💾 Training prompts written to {training_data_path}")

        if not examples:
            self.logger.warning("No valid training examples; aborting SFT.")
            return

        self._dump_args()

        dataset = PromptLabelDataset(examples, self.llm.tokenizer, self.train_max_length)
        training_model = self._build_training_model()
        _, epoch_losses = self._run_training_loop(training_model, dataset)

        self._save_training_metrics(epoch_losses, len(dataset))

    def _prepare_training_examples(self, data):
        system_prompt = self.predict_instuction['system_prompt']
        examples = []
        records = []
        for _, row in data.iterrows():
            label = str(row.get("target") or "").strip()
            if not label:
                continue
            summary = str(row.get("summary") or "").strip()
            company_description = self._resolve_company_description(row["ticker"])
            user_prompt = self._build_predict_instruction(company_description, summary)
            prompt_text = self._flatten_training_prompt(system_prompt, user_prompt)
            completion = self._format_training_completion(label)
            examples.append({"prompt_text": prompt_text, "completion": completion})
            records.append({
                "sample_id": f"{row['ticker']}_{row['end_date']}",
                "ticker": row["ticker"],
                "prediction_date": row["end_date"],
                "dataset": self.args.dataset_name,
                "prompt_text": prompt_text,
                "label": label,
                "summary": summary,
                "company_description": company_description,
            })
        return examples, records

    def _resolve_company_description(self, ticker: str) -> str:
        cached = self._load_company_description(ticker)
        if cached:
            return cached
        return f"{ticker} company description is not yet cached."

    def _flatten_training_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"System: {system_prompt.strip()}\nUser: {user_prompt.strip()}\nAssistant:"

    def _format_training_completion(self, label: str) -> str:
        eos = self.llm.tokenizer.eos_token or ""
        return f" {label.strip()}{eos}"

    def _dump_args(self):
        args_path = os.path.join(self.args.results_dir, "args.json")
        dumpable = {}
        for key, value in vars(self.args).items():
            if isinstance(value, datetime):
                dumpable[key] = value.isoformat()
                continue
            try:
                json.dumps(value)
                dumpable[key] = value
            except TypeError:
                dumpable[key] = str(value)
        try:
            with open(args_path, "w", encoding="utf-8") as f:
                json.dump(dumpable, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📦 Saved CLI args to {args_path}")
        except Exception as exc:
            self.logger.warning(f"Failed to write args.json: {exc}")

    def _build_training_model(self):
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = LlamaForCausalLM.from_pretrained(
            self.args.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.resize_token_embeddings(len(self.llm.tokenizer))
        if getattr(self.args, "use_qlora", False):
            lora_config = LoraConfig(
                r=getattr(self.args, "lora_r", 16),
                lora_alpha=getattr(self.args, "lora_alpha", 32),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=getattr(self.args, "lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        return model

    def _run_training_loop(self, model, dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_lr)
        dataloader = TorchDataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)
        epoch_losses = []
        grad_accum = self.train_gradient_accumulation_steps
        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            if len(dataloader) == 0:
                self.logger.warning("No batches available for training.")
                break
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss_scaled = loss / grad_accum
                loss_scaled.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) else 0.0
            self.logger.info(f"🧮 Epoch {epoch + 1}/{self.train_epochs} avg loss: {avg_loss:.4f}")
            epoch_losses.append(avg_loss)
        return model, epoch_losses

    def _save_training_metrics(self, epoch_losses, total_samples):
        eval_path = os.path.join(self.args.results_dir, "eval.json")
        experiment_end = datetime.now()
        duration = experiment_end - self.args.experiment_start_time
        result = {
            "model_name": self.args.base_model,
            "method_name": self.method_name,
            "dataset_name": self.args.dataset_name,
            "total_training_samples": total_samples,
            "train_epochs": self.train_epochs,
            "loss_per_epoch": epoch_losses,
            "avg_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
            "experiment_duration": {
                "duration": str(duration),
                "duration_hours": duration.total_seconds() / 3600 if duration else 0.0,
            },
        }
        try:
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            self.logger.info(f"📊 Training metrics saved to {eval_path}")
        except Exception as exc:
            self.logger.warning(f"Failed to write training eval.json: {exc}")


    def _create_result_entry(self, ticker, prediction_date, processing_date, ground_truth, company_prompt, 
                           predict_prompt, system_prompt, raw_prediction, parsed_movement, 
                           summary, company_description, error=None):
        """
        創建標準化的結果條目，便於不同baseline方法比較
        
        Args:
            ticker: 股票代碼
            prediction_date: 預測目標日期（股票實際日期）
            processing_date: 模型處理日期（當前時間）
            ground_truth: 真實標籤
            company_prompt: 公司描述prompt
            predict_prompt: 預測prompt
            system_prompt: 系統prompt
            raw_prediction: 原始預測文本
            parsed_movement: 解析後的漲跌結果
            summary: 新聞摘要
            company_description: 生成的公司描述
            error: 錯誤信息（可選）
        
        Returns:
            dict: 標準化的結果條目
        """
        result_entry = {
            # 基本信息
            "sample_id": f"{ticker}_{prediction_date}",
            "ticker": ticker,
            "prediction_date": prediction_date,  # 預測目標日期
            "processing_date": processing_date,  # 模型處理日期
            "ground_truth": ground_truth,
            
            # 模型信息
            "model_info": {
                "model_name": self.args.base_model,
                "method": self.method_name,
                "dataset": self.args.dataset_name
            },
            
            # 輸入信息
            "input_data": {
                "summary": summary,
                "company_description": company_description
            },
            
            # 模型輸入
            "model_input": {
                "company_prompt": company_prompt,
                "predict_prompt": predict_prompt,
                "system_prompt": system_prompt
            },
            
            # 預測結果
            "prediction": {
                "raw_text": raw_prediction,
                "parsed_movement": parsed_movement,
                "confidence": None  # 可以後續添加置信度
            },
            
            # 評估信息
            "evaluation": {
                "is_correct": parsed_movement == ground_truth if parsed_movement != "Unknown" else False,
                "error": error
            },
            
            # 元數據
            "metadata": {
                "processing_time": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        return result_entry

    def _load_company_description(self, ticker: str) -> Optional[str]:
        fpath = self.company_desc_root / f"{ticker}.txt"
        if fpath.exists():
            try:
                return fpath.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _save_company_description(self, ticker: str, desc: str) -> None:
        fpath = self.company_desc_root / f"{ticker}.txt"
        try:
            fpath.write_text(desc or "", encoding="utf-8")
        except Exception:
            self.logger.warning(f"Failed to cache company description for {ticker}")

    def _extract_stock_return(self, text):
        return extract_stock_direction(text)

    def _build_relative_company_prompt(self, ticker) -> str:
        return self.company_description_prompt.format(ticker=ticker)
    
    def _build_predict_instruction(self, company_description, summary) -> str:
        return self.predict_instuction['user_prompt'].format(
                    company_description=company_description,
                    summary=summary
                )

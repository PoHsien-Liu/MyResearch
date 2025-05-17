import re
import os
from tqdm import tqdm
from models.llm import LLaMALLM
from dataloader.dataloader import DataLoader
from utils.prompts import (
    COMPANY_DESCRIPTION_INSTRUCTION, 
    RELATIVE_COMPANY_INSTSRUCTION, 
    PREDICT_INSTRUCTION_SYSTEM_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT
)
from utils.fewshots import PREDICT_FEW_SHOT_EXAMPLES
from utils.metrics import calculate_metrics, save_metrics

class TDMLLM:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        self.dataloader = DataLoader(args, logger)
        self.llm = LLaMALLM(args, logger)
        self.company_description_prompt = COMPANY_DESCRIPTION_INSTRUCTION
        self.relative_company_prompt = RELATIVE_COMPANY_INSTSRUCTION
        self.predict_instuction = {
            "system_prompt" : PREDICT_INSTRUCTION_SYSTEM_PROMPT,
            "user_prompt": PREDICT_INSTRUCTION_USER_PROMPT
        }
        self.predict_few_shot_examples = PREDICT_FEW_SHOT_EXAMPLES

    def eval(self):
        self.logger.info("🔍 Loading test data...")
        data = self.dataloader.load(flag='test')
        data.to_csv('data.csv')
        self.logger.info(f"✅ Loaded {len(data)} samples.")

        preds = []
        labels = []
        correct = 0
        incorrect = 0

        for index, row in tqdm(data.iterrows(), total=len(data), desc="📊 Processing Samples"):
            try:
                ticker = row['ticker']
                summary = row['summary']
                label = row['target']
                
                # Step 1: 生成公司描述
                company_prompt = self._build_relative_company_prompt(ticker)
                if company_prompt.strip() == "":
                    self.logger.error(f"🔥 Empty prompt generated for ticker: {ticker}")
                    continue
                
                company_description = self.llm("", company_prompt)
                
                # Step 2: 生成預測
                predict_prompt = self._build_predict_instruction(company_description, summary)
                predict_result = self.llm(self.predict_instuction['system_prompt'], predict_prompt)
                
                # Step 3: 提取股票走勢
                self.logger.info(f"\n📌 [{index}] Ticker: {ticker}")
                self.logger.info(f"📝 Summary: {summary}")
                self.logger.info(f"🎯 Target: {label}")
                self.logger.info(f"🧠 Prediction: {predict_result}")

                stock_movement = self._extract_stock_return(predict_result)
                preds.append(stock_movement)
                labels.append(label)

                self.logger.info(f"Stock movement: {stock_movement}, Ground Truth: {label}")

                if stock_movement == label:
                    correct += 1
                else:
                    incorrect += 1
                    
            except Exception as e:
                self.logger.exception(f"🔥 Error during prediction for ticker {ticker}")
                preds.append("Unknown")
                labels.append(label)
                incorrect += 1

            self.logger.info(f"Correct: {correct}, Incorrect: {incorrect}")

        metrics_result = calculate_metrics(preds, labels)
        save_metrics(metrics_result, self.args.base_model, os.path.join("results", self.args.dataset_name), self.args.dataset_name)

    def _extract_stock_return(self, text):
        text = text.lower().strip()
        text = re.sub(r"\*\*", "", text)

        # 彈性匹配：stock return: +/- 數值 (up/down)
        match = re.search(r"stock\s*return\s*:\s*[-+]?\d+(?:\.\d+)?\s*%?\s*\(\s*(up|down)\s*\)", text)
        if match:
            return "Positive" if match.group(1)  == "up" else "Negative" # 只回傳 up 或 down
        return "Unknown"

    def _build_relative_company_prompt(self, ticker) -> str:
        return self.company_description_prompt.format(ticker=ticker)
    
    def _build_predict_instruction(self, company_description, summary) -> str:
        return self.predict_instuction['user_prompt'].format(
                    company_description=company_description,
                    summary=summary
                )
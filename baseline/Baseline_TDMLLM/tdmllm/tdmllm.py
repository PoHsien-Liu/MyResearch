import re
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

        company_prompts = []
        ticker_list = []
        summary_list = []
        
        # Step 1: 先組 company prompts
        for index, row in data.iterrows():
            ticker = row['ticker']
            summary = row['summary']
            
            prompt = self._build_relative_company_prompt(ticker)
            if prompt.strip() == "":
                self.logger.error(f"🔥 Empty prompt generated for ticker: {ticker}")

            company_prompts.append(prompt)
            ticker_list.append(ticker)
            summary_list.append(summary)
            labels.append(row['target'])
        
        # Step 2: 分 batch 處理 company description
        company_descriptions = []
        for i in tqdm(range(0, len(company_prompts), self.args.batch_size), desc="🏢 Generating Company Descriptions"):
            batch_prompts = company_prompts[i: i+self.args.batch_size]
            batch_outputs = self.llm.batch_infer("", batch_prompts)
            company_descriptions.extend(batch_outputs)
        
        # Step 3: 用 company description + summary 組 predict prompts
        predict_prompts = []
        for company_desc, summary in zip(company_descriptions, summary_list):
            predict_prompts.append(
                self._build_predict_instruction(company_desc, summary)
            )

        # Step 4: 分 batch 推理 predict
        predict_results = []
        for i in tqdm(range(0, len(predict_prompts), self.args.batch_size), desc="📈 Predicting Stock Movements"):
            batch_prompts = predict_prompts[i: i+self.args.batch_size]
            batch_outputs = self.llm.batch_infer(self.predict_instuction['system_prompt'], batch_prompts)
            predict_results.extend(batch_outputs)

        # Step 5: 從 predict output 中提取 stock movement
        for index, predict_result in enumerate(predict_results):
            try:
                self.logger.info(f"\n📌 [{index}] Ticker: {ticker_list[index]}")
                self.logger.info(f"📝 Summary: {summary_list[index]}")
                self.logger.info(f"🎯 Target: {labels[index]}")

                self.logger.info(f"🧠 Prediction: {predict_result}")

                stock_movement = self._extract_stock_return(predict_result)
                preds.append(stock_movement)

                self.logger.info(f"Stock movement: {stock_movement}, Ground Truth: {labels[index]}")

                if stock_movement == labels[index]:
                    correct += 1
                else:
                    incorrect += 1
            except Exception as e:
                self.logger.exception(f"🔥 Error during prediction for ticker {ticker_list[index]}")
                preds.append("Unknown")
                incorrect += 1
        self.logger.info(f"Correct: {correct}, Incorrect: {incorrect}")

        metrics_result = calculate_metrics(preds, labels)
        save_metrics(metrics_result, self.args.base_model, "results")

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
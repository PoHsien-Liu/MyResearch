import re
import os
import json
from datetime import datetime
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
        
        # 初始化結果保存列表
        test_results = []
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for index, row in tqdm(data.iterrows(), total=len(data), desc="📊 Processing Samples"):
            try:
                ticker = row['ticker']
                summary = row['summary']
                label = row['target']
                target_date = row['end_date']  # 直接從DataFrame取得預測目標日期
                
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
                self.logger.info(f"📅 Target Date: {target_date}")
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
                
                # 保存測試結果
                result_entry = self._create_result_entry(
                    ticker=ticker,
                    prediction_date=target_date,  # 使用DataFrame中的預測目標日期
                    processing_date=current_date,  # 當前處理時間
                    ground_truth=label,
                    company_prompt=company_prompt,
                    predict_prompt=predict_prompt,
                    system_prompt=self.predict_instuction['system_prompt'],
                    raw_prediction=predict_result,
                    parsed_movement=stock_movement,
                    summary=summary,
                    company_description=company_description
                )
                test_results.append(result_entry)
                    
            except Exception as e:
                self.logger.exception(f"🔥 Error during prediction for ticker {ticker}")
                preds.append("Unknown")
                labels.append(label)
                incorrect += 1
                
                # 即使出錯也保存結果
                result_entry = self._create_result_entry(
                    ticker=ticker if 'ticker' in locals() else "Unknown",
                    prediction_date=target_date if 'target_date' in locals() else "Unknown",
                    processing_date=current_date,
                    ground_truth=label if 'label' in locals() else "Unknown",
                    company_prompt=company_prompt if 'company_prompt' in locals() else "",
                    predict_prompt=predict_prompt if 'predict_prompt' in locals() else "",
                    system_prompt=self.predict_instuction['system_prompt'],
                    raw_prediction=predict_result if 'predict_result' in locals() else "",
                    parsed_movement="Unknown",
                    summary=summary if 'summary' in locals() else "",
                    company_description=company_description if 'company_description' in locals() else "",
                    error=str(e)
                )
                test_results.append(result_entry)

            self.logger.info(f"Correct: {correct}, Incorrect: {incorrect}")

        # 保存測試結果到文件
        self._save_test_results(test_results)

        metrics_result = calculate_metrics(preds, labels)
        save_metrics(metrics_result, self.args.base_model, "results", self.args.dataset_name)

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
                "method": "TDMLLM",
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

    def _save_test_results(self, test_results):
        """
        保存測試結果到文件，支持多種格式便於比較分析
        
        Args:
            test_results: 測試結果列表
        """
        # 創建分層目錄結構：dataset_name/method_name/model_name/timestamp/
        method_name = "TDMLLM"
        safe_model_name = self.args.base_model.replace('/', '_').replace('\\', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = os.path.join("results", self.args.dataset_name, method_name, safe_model_name, timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 保存詳細JSON格式（完整信息）
        json_filepath = os.path.join(results_dir, "detailed.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        # 2. 保存簡化JSON格式（便於比較）
        simplified_results = self._create_simplified_results(test_results)
        simplified_json_filepath = os.path.join(results_dir, "simplified.json")
        with open(simplified_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        # 3. 保存CSV格式（便於Excel分析）
        csv_filepath = os.path.join(results_dir, "results.csv")
        self._save_to_csv(test_results, csv_filepath)
        
        # 4. 保存比較格式（便於不同方法比較）
        comparison_filepath = os.path.join(results_dir, "comparison.csv")
        self._save_comparison_format(test_results, comparison_filepath)
        
        self.logger.info(f"✅ Test results saved to:")
        self.logger.info(f"   Directory: {results_dir}")
        self.logger.info(f"   Detailed JSON: {json_filepath}")
        self.logger.info(f"   Simplified JSON: {simplified_json_filepath}")
        self.logger.info(f"   CSV: {csv_filepath}")
        self.logger.info(f"   Comparison CSV: {comparison_filepath}")

    def _create_simplified_results(self, test_results):
        """
        創建簡化的結果格式，便於快速比較
        """
        simplified = []
        for result in test_results:
            simplified.append({
                "ticker": result["ticker"],
                "ground_truth": result["ground_truth"],
                "predicted": result["prediction"]["parsed_movement"],
                "is_correct": result["evaluation"]["is_correct"],
                "raw_prediction": result["prediction"]["raw_text"][:200] + "..." if len(result["prediction"]["raw_text"]) > 200 else result["prediction"]["raw_text"],
                "summary": result["input_data"]["summary"][:100] + "..." if len(result["input_data"]["summary"]) > 100 else result["input_data"]["summary"]
            })
        return simplified

    def _save_to_csv(self, test_results, filepath):
        """
        保存為CSV格式
        """
        import pandas as pd
        csv_data = []
        for result in test_results:
            csv_data.append({
                "ticker": result["ticker"],
                "prediction_date": result["prediction_date"],  # 預測目標日期
                "processing_date": result["processing_date"],  # 模型處理日期
                "ground_truth": result["ground_truth"],
                "predicted": result["prediction"]["parsed_movement"],
                "is_correct": result["evaluation"]["is_correct"],
                "raw_prediction": result["prediction"]["raw_text"],
                "summary": result["input_data"]["summary"],
                "company_description": result["input_data"]["company_description"],
                "error": result["evaluation"]["error"] if result["evaluation"]["error"] else ""
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def _save_comparison_format(self, test_results, filepath):
        """
        保存為便於比較的格式，適合不同baseline方法比較
        """
        import pandas as pd
        comparison_data = []
        for result in test_results:
            comparison_data.append({
                "sample_id": result["sample_id"],
                "ticker": result["ticker"],
                "prediction_date": result["prediction_date"],  # 預測目標日期
                "ground_truth": result["ground_truth"],
                "method": result["model_info"]["method"],
                "model": result["model_info"]["model_name"],
                "predicted": result["prediction"]["parsed_movement"],
                "is_correct": result["evaluation"]["is_correct"],
                "raw_prediction": result["prediction"]["raw_text"],
                "summary": result["input_data"]["summary"]
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def _extract_stock_return(self, text):
        text = text.lower().strip()
        text = re.sub(r"\*\*", "", text)

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
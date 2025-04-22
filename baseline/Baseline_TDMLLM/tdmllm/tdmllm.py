from models.llm import LLaMALLM
from dataloader.dataloader import DataLoader
from utils.prompts import (
    COMPANY_DESCRIPTION_INSTRUCTION, 
    RELATIVE_COMPANY_INSTSRUCTION, 
    PREDICT_INSTRUCTION_SYSTEM_PROMPT,
    PREDICT_INSTRUCTION_USER_PROMPT
)
from utils.fewshots import PREDICT_FEW_SHOT_EXAMPLES

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

    def inference(self):
        self.logger.info("🔍 Loading test data...")
        data = self.dataloader.load(flag='test')
        data.to_csv('data.csv')
        self.logger.info(f"✅ Loaded {len(data)} samples.")

        correct = 0
        incorrect = 0
        for index, row in data.iterrows():
            ticker = row['ticker']
            summary = row['summary']
            target = row['target']

            self.logger.info(f"\n📌 [{index}] Ticker: {ticker}")
            self.logger.info(f"📝 Summary: {summary}")
            self.logger.info(f"🎯 Target: {target}")
            
            try:
                system_prompt = ""
                company_description = self.llm(system_prompt, self._build_relative_company_prompt(ticker))

                predict_result = self.llm(  
                    self.predict_instuction['system_prompt'],
                    self._build_predict_instruction(company_description, row['summary']) 
                )
                
                self.logger.info(f"🧠 Prediction: {predict_result}")
                
                # Todo: Evaluate Result
                # stock_movement_explanation = predict_result.split('Stock Return: ')[1]
                # stock_movement = stock_movement_explanation.split(' ')[0]
                
                # if stock_movement == data['target']:
                #     correct += 1  
                # else: 
                #     incorrect += 1
            except Exception as e:
                self.logger.exception(f"🔥 Error during prediction for ticker {ticker}")     
                incorrect += 1  

    def _build_relative_company_prompt(self, ticker) -> str:
        return self.company_description_prompt.format(ticker=ticker)
    
    def _build_predict_instruction(self, company_description, summary) -> str:
        return self.predict_instuction['user_prompt'].format(
                    company_description=company_description,
                    summary=summary
                )
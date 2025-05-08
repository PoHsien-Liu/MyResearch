import time
import torch
from transformers import (
    LlamaForCausalLM, 
    AutoTokenizer,
    pipeline
)


class LLaMALLM:
    def __init__(self, args, logger):
        self.logger = logger

        # Load Tokenizer and Model
        self.logger.info("📥 Load Tokenizer and Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # Set PAD Token
        PAD_TOKEN = "<|pad|>"
        self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.tokenizer.padding_side = "right"
        
        self.model = LlamaForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.logger.info("🧠 Set Text Generation Pipeline...")
        self.text_gen_pipeline = pipeline(
            task="text-generation",
            batch_size=args.batch_size,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            return_full_text=False
        )

    def create_chat_format_data(self, system_prompt, user_prompt):
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def __call__(self, system_prompt, user_prompt):
        chat_format_data = self.create_chat_format_data(system_prompt, user_prompt)

        prompt = self.tokenizer.apply_chat_template(
            chat_format_data, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = self.tokenizer(prompt, return_tensors="pt")
        num_tokens = tokenized_input['input_ids'].shape[1]  
        self.logger.info(f"🔢 Token count: {num_tokens}")

        start_time = time.time()
        try:
            response = self.text_gen_pipeline(prompt)[0]['generated_text']
        except Exception as e:
            self.logger.exception("🔥 Inference failed!")
            return "Inference Error"
        end_time = time.time()

        self.logger.info(f"⏱️ Inference time: {end_time - start_time:.2f} seconds\n")
        return response
    
    def batch_infer(self, system_prompt, user_prompts):
        """
        Args:
            system_prompt: str
            user_prompts: List[str], raw user messages or chat-formatted prompts

        Returns:
            List[str]: generated texts
        """
        self.logger.info(f"📦 Running batch inference on {len(user_prompts)} samples...")

        prompts = []
        for user_prompt in user_prompts:
            chat_format_data = self.create_chat_format_data(system_prompt, user_prompt)
            prompt = self.tokenizer.apply_chat_template(
                chat_format_data, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # 記錄 token 數量並檢查是否超過模型限制
        max_tokens = 0
        for idx, prompt in enumerate(prompts):
            tokenized_input = self.tokenizer(prompt, return_tensors="pt")
            num_tokens = tokenized_input['input_ids'].shape[1]
            max_tokens = max(max_tokens, num_tokens)
            self.logger.info(f"🔢 [{idx}] Token count: {num_tokens}")

        if max_tokens > 4096:  # Llama 模型的典型上下文長度限制
            self.logger.warning(f"⚠️ Some prompts exceed the model's context length limit (4096 tokens). Max tokens: {max_tokens}")

        start_time = time.time()

        try:
            # 使用單一 pipeline 調用
            outputs = self.text_gen_pipeline(
                prompts,
                batch_size=self.args.batch_size,
                truncation=True,
                max_new_tokens=512,
                do_sample=False  # 為了得到確定性的結果
            )
            responses = [out['generated_text'] for out in outputs]
        except Exception as e:
            self.logger.exception(f"🔥 Batch inference failed with error: {str(e)}")
            # 如果批次處理失敗，嘗試單個處理
            self.logger.info("🔄 Falling back to single sample processing...")
            responses = []
            for prompt in prompts:
                try:
                    output = self.text_gen_pipeline(prompt)[0]['generated_text']
                    responses.append(output)
                except Exception as e:
                    self.logger.exception(f"🔥 Single sample inference failed for prompt")
                    responses.append("Inference Error")

        end_time = time.time()
        elapsed = end_time - start_time

        self.logger.info(f"⏱️ Batch inference total time: {elapsed:.2f} seconds")
        self.logger.info(f"⏱️ Average time per sample: {elapsed / max(1, len(prompts)):.2f} seconds\n")

        # 記錄效能指標
        self.logger.info(f"Batch size: {len(user_prompts)}")
        self.logger.info(f"Total time: {end_time - start_time:.2f}s")
        self.logger.info(f"Time per sample: {(end_time - start_time) / len(user_prompts):.2f}s")

        return responses
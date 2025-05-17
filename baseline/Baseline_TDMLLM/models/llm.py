import time
import torch
from transformers import (
    LlamaForCausalLM, 
    AutoTokenizer,
    pipeline
)


class LLaMALLM:
    def __init__(self, args, logger):
        self.args = args
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
            max_new_tokens=1024,
            return_full_text=False,
            do_sample=True
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
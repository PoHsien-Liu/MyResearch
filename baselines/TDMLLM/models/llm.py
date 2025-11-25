import time
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from baselines.FinGPT.model import FinGPTAdapter, FinGPTConfig, GenerationResult


class LLaMALLM:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.max_new_tokens = getattr(args, "max_new_tokens_predict", 256)
        self.do_sample = getattr(args, "do_sample", False)
        self.num_beams = getattr(args, "num_beams", 1)

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # QLoRA Configuration
        if getattr(args, 'use_qlora', True):
            self.logger.info("🔧 Setting up QLoRA configuration...")
            
            # 4-bit quantization configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=getattr(args, 'load_in_4bit', True),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=getattr(args, 'lora_r', 16),  # LoRA rank
                lora_alpha=getattr(args, 'lora_alpha', 32),  # LoRA alpha parameter
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=getattr(args, 'lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        else:
            self.logger.info("🔧 Using standard model loading (no QLoRA)...")
            bnb_config = None
            lora_config = None

        # Load Tokenizer and Model
        self.logger.info("📥 Loading Tokenizer and Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        # Set PAD Token
        PAD_TOKEN = "<|pad|>"
        self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.tokenizer.padding_side = "left"
        
        # Load model with or without quantization
        if getattr(args, 'use_qlora', True):
            # 強制使用所有8張GPU，每張分配約3GB記憶體
            max_memory = {i: "3GB" for i in range(8)}
            max_memory["cpu"] = "32GB"  # CPU記憶體
            
            self.model = LlamaForCausalLM.from_pretrained(
                args.base_model, 
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_memory=max_memory,
                offload_folder="offload"  # 啟用模型分片
            )
        else:
            # 強制使用所有8張GPU，每張分配約3GB記憶體
            max_memory = {i: "3GB" for i in range(8)}
            max_memory["cpu"] = "32GB"  # CPU記憶體
            
            self.model = LlamaForCausalLM.from_pretrained(
                args.base_model, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="offload"  # 啟用模型分片
            )
        
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 顯示模型分佈在哪些設備上
        self.logger.info("🔍 Model device mapping:")
        if hasattr(self.model, 'hf_device_map'):
            for module_name, device in self.model.hf_device_map.items():
                self.logger.info(f"   {module_name}: {device}")
        else:
            self.logger.info(f"   Model loaded on: {next(self.model.parameters()).device}")
        
        # Apply LoRA if using QLoRA
        if getattr(args, 'use_qlora', True) and lora_config is not None:
            self.logger.info("🔧 Applying LoRA adapters...")
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters info
            self.model.print_trainable_parameters()
        
        self.logger.info("🧠 Setting up Text Generation Pipeline...")
        self.logger.info(f"🧠 Model max length: {self.tokenizer.model_max_length}")
        self.logger.info(f"🧠 Model max position embeddings: {self.model.config.max_position_embeddings}")
        self.logger.info(f"🧠 Model embedding size: {self.model.get_input_embeddings().weight.size(0)}")
        
        # Get generation parameters
        temperature = getattr(args, 'temperature', 0.7)
        top_p = getattr(args, 'top_p', 0.9)
        
        generation_kwargs = {
            "task": "text-generation",
            "batch_size": args.batch_size,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
        }
        if self.do_sample:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            generation_kwargs.update({
                "do_sample": False,
                "num_beams": self.num_beams,
            })

        self.text_gen_pipeline = pipeline(**generation_kwargs)

    def create_chat_format_data(self, system_prompt, user_prompt):
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def __call__(self, system_prompt, user_prompt, *, max_new_tokens=None):
        chat_format_data = self.create_chat_format_data(system_prompt, user_prompt)

        prompt = self.tokenizer.apply_chat_template(
            chat_format_data, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = self.tokenizer(prompt, return_tensors="pt")
        num_tokens = tokenized_input['input_ids'].shape[1]  
        self.logger.info(f"🔢 Token count: {num_tokens}")

        start_time = time.time()
        try:
            with torch.no_grad():
                response = self.text_gen_pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                )[0]['generated_text']
        except Exception as e:
            self.logger.exception("🔥 Inference failed!")
            return "Inference Error"
        end_time = time.time()

        self.logger.info(f"⏱️ Inference time: {end_time - start_time:.2f} seconds\n")
        return response

    def batch_inference(self, system_prompts, user_prompts, *, max_new_tokens=None):
        """
        批次推論：system_prompts, user_prompts 為 list of str，回傳 list of generated_text
        """
        assert len(system_prompts) == len(user_prompts), "system_prompts 和 user_prompts 長度需一致"

        MAX_TOKENS = self.tokenizer.model_max_length
        valid_prompts = []
        prompt_indices = []

        for i, (sys, usr) in enumerate(zip(system_prompts, user_prompts)):
            chat_data = self.create_chat_format_data(sys, usr)
            prompt = self.tokenizer.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)
            tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            input_ids = tokenized["input_ids"]
            token_len = input_ids.shape[1]

            self.logger.info(f"Prompt {i} tokenized length: {token_len}")

            if token_len == 0:
                self.logger.warning(f"⚠️ Prompt {i} is empty after tokenization. Skipping.")
                continue

            if input_ids.max() >= self.model.get_input_embeddings().weight.size(0):
                self.logger.error(
                    f"❌ Prompt {i} contains token ID >= vocab size ({self.tokenizer.vocab_size}), skipping.")
                continue

            valid_prompts.append(prompt)
            prompt_indices.append(i)

        start_time = time.time()
        results = ["Inference Error"] * len(system_prompts)

        if not valid_prompts:
            self.logger.error("🚫 No valid prompts to run inference.")
            return results

        try:
            with torch.no_grad():
                batch_size = self.args.batch_size
                num_batches = (len(valid_prompts) + batch_size - 1) // batch_size
                for batch_idx in tqdm(range(num_batches), desc="Batch Inference", unit="batch"):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, len(valid_prompts))
                    batch_prompts = valid_prompts[start:end]
                    batch_indices = prompt_indices[start:end]
                    outputs = self.text_gen_pipeline(
                        batch_prompts,
                        max_new_tokens=max_new_tokens or self.max_new_tokens,
                    )
                    for idx, out in zip(batch_indices, outputs):
                        results[idx] = out[0]['generated_text']
        except Exception as e:
            self.logger.exception("🔥 Batch inference failed!")

        end_time = time.time()
        self.logger.info(
            f"⏱️ Batch inference time: {end_time - start_time:.2f} seconds for {len(valid_prompts)} valid samples\n"
        )
        return results


class FinGPTLLM:
    """Adapter that lets TDMLLM pipeline reuse the FinGPT loader + generator."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        max_tokens = getattr(args, "max_new_tokens_predict", 256)
        adapter_base_model = "meta-llama/Meta-Llama-3-8B"
        requested = getattr(args, "base_model", None)
        if requested and requested != adapter_base_model:
            logger.warning(
                f"[FinGPTLLM] FinGPT adapters are trained on {adapter_base_model}; overriding requested "
                f"base_model {requested}."
            )
        self.config = FinGPTConfig(
            base_model=adapter_base_model,
            fingpt_lora=getattr(args, "fingpt_lora", None),
            max_new_tokens=max_tokens,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 0.9),
            do_sample=getattr(args, "do_sample", False),
            device=getattr(args, "device", None),
            device_map=getattr(args, "device_map", None),
            torch_dtype=getattr(args, "torch_dtype", None),
            load_in_4bit=getattr(args, "load_in_4bit", False),
            bnb_4bit_compute_dtype=getattr(args, "bnb_4bit_compute_dtype", "float16"),
            bnb_4bit_quant_type=getattr(args, "bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=getattr(args, "bnb_4bit_use_double_quant", True),
        )
        self.adapter = FinGPTAdapter(self.config, logger=logger)

    def _build_generation_kwargs(self, max_new_tokens):
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        return kwargs or None

    def __call__(self, system_prompt, user_prompt, *, max_new_tokens=None):
        result: GenerationResult = self.adapter.generate(
            system_prompt,
            user_prompt,
            generation_kwargs=self._build_generation_kwargs(max_new_tokens),
        )
        return result.text

    def batch_inference(self, system_prompts, user_prompts, *, max_new_tokens=None):
        assert len(system_prompts) == len(user_prompts), "system_prompts 和 user_prompts 長度需一致"
        prompts = list(zip(system_prompts, user_prompts))
        results = self.adapter.batch_generate(
            prompts,
            generation_kwargs=self._build_generation_kwargs(max_new_tokens),
        )
        return [res.text for res in results]

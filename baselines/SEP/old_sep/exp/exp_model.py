from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results#, save_agents
from explain_module.agents import PredictReflectAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.train_reward_model import train_reward_model
from predict_module.tuning_lm_with_rl import tuning_lm_with_rl
from transformers import LlamaTokenizer, pipeline, BitsAndBytesConfig #, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import os, json
import torch


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)


    def train(self):
        # Collect demonstration data
        print("Loading Train Agents...")
        data = self.dataloader.load(flag="train")

        agent_cls = PredictReflectAgent
        agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in data.iterrows()]
        print("Loaded Train Agents.")

        # 確保數據目錄存在
        os.makedirs(os.path.dirname(self.args.data_path), exist_ok=True)
        with open(self.args.data_path, 'w') as f:
            pass  # 創建空文件

        # 原始代碼
        print(f"\n開始訓練，總共有 {len(agents)} 個 agents")
        for i, agent in enumerate(agents):
            print(f"\n--- Agent {i+1}/{len(agents)} ---")
            print(f"Ticker: {agent.ticker}")
            print(f"Target: {agent.target}")
            
            agent.run()
            
            # 調試輸出
            prediction = agent.prediction
            is_correct = agent.is_correct()
            print(f"Prediction: {prediction}")
            print(f"Correct: {is_correct}")

            if agent.is_correct():
                prompt = agent._build_agent_prompt()
                response = agent.scratchpad.split('Price Movement: ')[-1]

                sample = {"instruction": prompt, "input": "", "output": response}
                with open(self.args.data_path, 'a') as f:
                    f.write(json.dumps(sample) + "\n")
                print("✓ 已保存到訓練數據")
            else:
                print("✗ 預測錯誤，不保存到訓練數據")

        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        # 檢查是否有足夠的訓練數據
        if len(correct) == 0:
            print("警告：沒有正確的預測樣本！無法進行監督式微調。")
            print("請檢查模型配置或數據質量。")
            return

        # Train supervised policy
        supervised_finetune(self.args)
        merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)

        # Collect comparison data
        comparison_data = []

        for trial in range(self.args.num_reflect_trials):
            print(f"\n=== 反思試驗 {trial+1}/{self.args.num_reflect_trials} ===")
            incorrect_agents = [a for a in agents if not a.is_correct()]
            print(f"需要反思的 agents 數量: {len(incorrect_agents)}")
            
            for idx, agent in enumerate(incorrect_agents):
                print(f"\n--- 反思 Agent {idx+1}/{len(incorrect_agents)} ---")
                print(f"Ticker: {agent.ticker}")
                print(f"Target: {agent.target}")
                
                prev_response = agent.scratchpad.split('Price Movement: ')[-1]
                print(f"Previous prediction: {prev_response}")
                
                agent.run()

                if agent.is_correct():
                    print(f"New prediction: {agent.prediction}")
                    print("✓ 反思成功！")
                    print(agent._build_agent_prompt(), "\n\n\n")
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('Price Movement: ')[-1]

                    sample = {"user_input": prompt, "completion_a": prev_response, "completion_b": response}
                    comparison_data.append(sample)
                else:
                    print(f"New prediction: {agent.prediction}")
                    print("✗ 反思後仍然錯誤")

            correct, incorrect = summarize_trial(agents)
            print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        os.makedirs(self.args.datasets_dir, exist_ok=True)
        comparison_data_path = os.path.join(self.args.datasets_dir, "comparison_data.json")

        if comparison_data:
            with open(comparison_data_path, 'w') as f:
                f.write(json.dumps(comparison_data))
            print(f"\n保存了 {len(comparison_data)} 個比較樣本到 {comparison_data_path}")
        else:
            print("\n警告：沒有生成比較數據！")

        # Train reward model
        if comparison_data:
            train_reward_model(self.args)
            merge_peft_adapter(model_name=self.args.reward_adapter, output_name=self.args.reward_model_name)

            # Optimize using reinforcement learning
            tuning_lm_with_rl(self.args)
            merge_peft_adapter(model_name=self.args.output_dir+"step_saved", output_name="./saved_models/sep_model")
        else:
            print("\n跳過獎勵模型訓練和強化學習（沒有比較數據）")


    def test(self):
        print("Loading Test Agents...")
        data = self.dataloader.load(flag="test")

        agent_cls = PredictReflectAgent
        test_agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in data.iterrows()]
        print("Loaded Test Agents.")

        # Updated model initialization using BitsAndBytesConfig
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "./saved_models/sep_model",
            device_map="auto",
            generation_config={
                "do_sample": True,  # Enable sampling
                "temperature": 0.9,
                "top_p": 0.6
            },
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.args.output_dir+"step_saved")

        # Updated reward model initialization using BitsAndBytesConfig
        reward_model = pipeline(
            "sentiment-analysis",
            model=self.args.reward_model_name,
            device_map="auto",
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            },
            tokenizer=tokenizer
        )

        for agent in test_agents:
            agent.run_n_shots(
                              model=model,
                              tokenizer=tokenizer,
                              reward_model=reward_model,
                              num_shots=self.args.num_shots
                              )

        correct, incorrect = summarize_trial(test_agents)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        save_results(test_agents, self.args.save_dir)

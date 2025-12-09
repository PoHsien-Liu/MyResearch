from typing import List, Union, Literal, Optional
from utils.llm import NShotLLM, VLLMLLM, VLLMSamplingConfig  # , FastChatLLM, OpenAILLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES


class PredictAgent:
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = None,
                 max_prompt_tokens: Optional[int] = None,
                 ) -> None:

        self.ticker = ticker
        self.summary = summary
        self.target = target
        self.prediction = ''
        self.max_prompt_tokens = max_prompt_tokens

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm or VLLMLLM(
            model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            sampling_config=VLLMSamplingConfig(max_new_tokens=256),
        )

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nPrice Movement: "
        self.scratchpad += facts
        print(facts, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('Price Movement: ')[-1]
        self.prediction = response.split()[0]
        print(response, end="\n\n\n\n")

        self.finished = True

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        summary = self._truncate_summary(self.summary)
        prompt = self.predict_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            summary = summary)
        if self.max_prompt_tokens and self._count_tokens(prompt) > self.max_prompt_tokens:
            prompt = self.predict_prompt.format(
                ticker=self.ticker,
                examples="",
                summary=summary,
            )
        if self.max_prompt_tokens and self._count_tokens(prompt) > self.max_prompt_tokens:
            summary = self._truncate_summary(summary)
            prompt = self.predict_prompt.format(
                ticker=self.ticker,
                examples="",
                summary=summary,
            )
        return prompt

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''

    def _truncate_summary(self, summary: str) -> str:
        if not self.max_prompt_tokens:
            return summary
        tokenizer = getattr(self.llm, "tokenizer", None)
        if tokenizer is None:
            # Fallback rough truncation by words
            tokens = summary.split()
            return " ".join(tokens[: self.max_prompt_tokens])
        tokens = tokenizer.encode(summary)
        if len(tokens) <= self.max_prompt_tokens:
            return summary
        trimmed = tokens[: self.max_prompt_tokens]
        return tokenizer.decode(trimmed, skip_special_tokens=True)

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self.llm, "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        return len(text.split())


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = None,
                 reflect_llm = None,
                 max_prompt_tokens: Optional[int] = None,
                 ) -> None:

        super().__init__(ticker, summary, target, predict_llm, max_prompt_tokens=max_prompt_tokens)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm or self.llm
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''

    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()

        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()
        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)
        print(self.reflections_str, end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker = self.ticker,
                            scratchpad = self.scratchpad)

    def _build_agent_prompt(self) -> str:
        prompt = self.agent_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            reflections = self.reflections_str,
                            summary = self.summary)
        return prompt

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        PredictAgent.run(self, reset=reset)


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()

"""生成模型封装：默认本地 Qwen3-8B，也可切换 DashScope OpenAI 兼容接口。

注意：本模块会在 import 时加载 torch/transformers。
调用方必须先 import embedding（sentence_transformers），否则本机 pyarrow 会报
libstdc++ CXXABI 版本冲突。
"""

import os

import config

_llm = None


class LocalQwenLLM:
    """本地 Qwen3-8B（bf16 / 可选 4bit）。"""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] 加载本地模型 {config.LLM_MODEL} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        kwargs = {"device_map": "cuda"}
        if config.LLM_LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, **kwargs)
        self.model.eval()

    def _chat_text(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(self, messages, max_new_tokens=None, temperature=0.0):
        import torch

        text = self._chat_text(messages)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or config.LLM_MAX_NEW_TOKENS,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class DashScopeLLM:
    """DashScope OpenAI 兼容接口，模型名用 DASHSCOPE_MODEL 指定。"""

    def __init__(self):
        from openai import OpenAI

        self.model = os.getenv("DASHSCOPE_MODEL", "qwen3-235b-a22b")
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=120,
            max_retries=3,
        )

    def generate(self, messages, max_new_tokens=None, temperature=0.0):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens or config.LLM_MAX_NEW_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm():
    global _llm
    if _llm is None:
        _llm = DashScopeLLM() if config.LLM_BACKEND == "dashscope" else LocalQwenLLM()
    return _llm


def generate(messages, max_new_tokens=None, temperature=0.0):
    return get_llm().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

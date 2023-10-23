#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os
from typing import Dict, Union, Optional
from typing import List
import torch
from accelerate import load_checkpoint_and_dispatch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer

class ChatGLMService(LLM):
    max_token: int = 2048
    temperature: float = 0.95
    top_p = 0.7
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM 6B int4"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response

    def load_model(self, model_name_or_path: str, isqiamtize4):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if isqiamtize4:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().quantize(4)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = self.model.eval()

# if __name__ == '__main__':
#     config=LangChainCFG()
#     chatLLM = ChatGLMService()
#     chatLLM.load_model(model_name_or_path=config.llm_model_name)

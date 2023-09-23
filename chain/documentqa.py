#!/usr/bin/env python
# -*- coding:utf-8 _*-

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

from config import CFG
from model.llm.llm_service import ChatGLMService
from vector_searcher.MYFAISS import MYFAISS

class DocumentEnhanceQA(object):
    def __init__(self, config):
        self.config = config
        self.llm_service = ChatGLMService()
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)
        self.source_service = MYFAISS(config)
        self.source_service.init_vector_store()

    def get_knowledge_based_answer(self, query,
                                   history_len=0,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=4,
                                   web_content='',
                                   chat_history=[]):
        if web_content:
            prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}""" + """
                                已知内容:
                                {context}
                                问题:
                                {question}"""
        else:
            prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                            已知内容:
                                            {context}
                                            问题:
                                            {question}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []

        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p

        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm_service,
            retriever=self.source_service.vector_store.as_retriever(
                search_kwargs={"k": top_k}),
            prompt=prompt)
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})

        return result

    def get_llm_answer(self, query='', web_content=''):
        if web_content:
            prompt = f'基于网络检索内容：{web_content}，回答以下问题{query}'
        else:
            prompt = query
        result = self.llm_service._call(prompt)
        return result

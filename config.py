#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = './logs/run.log'

    llm_model_name = 'D:\\langchainqa\\model\\llm\\ChatGLM-6B'  # 本地llm模型文件 or huggingface远程仓库，如果是windows使用本地模型需要绝对路径，例如：D:\\xx\\xx\xxx
    embedding_model_name = './model/wordembedding/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库

    docs_path = './vector_store/docs'
    vector_store_path = './vector_store/vs'
    knowledge_base_path = './vector_store/kb'
    sentence_size = 100
    # 是否开启中文标题加强，以及标题增强的相关配置
    # 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
    # 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
    ZH_TITLE_ENHANCE = True
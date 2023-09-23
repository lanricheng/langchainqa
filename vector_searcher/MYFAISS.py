#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os
import logging
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import CFG
from langchain.text_splitter import CharacterTextSplitter
from utils import torch_gc
from utils.textsplitter import zh_title_enhance, ChineseTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader

from logs import log_config
logger = log_config.logger

class MYFAISS(object):
    def __init__(self, config):
        self.config = config
        self.docs_path = config.docs_path
        self.vs_path = config.vector_store_path
        self.kb_path = config.knowledge_base_path
        self.sentence_size = config.sentence_size
        self.ZH_TITLE_ENHANCE = config.ZH_TITLE_ENHANCE
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name, model_kwargs={'device': self.config.device})
        self.vector_store = None

    def tree(self, filepath, ignore_dir_names=None, ignore_file_names=None):
        """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
        if ignore_dir_names is None:
            ignore_dir_names = []
        if ignore_file_names is None:
            ignore_file_names = []
        ret_list = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                logger.info(f"{filepath} 路径不存在")
                return None, None
            elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
                return [filepath], [os.path.basename(filepath)]
            elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                        ret_list.append(fullfilepath)
                    if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                        ret_list.extend(self.tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
        return ret_list, [os.path.basename(p) for p in ret_list]

    def write_check_file(self, filepath, docs):
        """将知识库docs写入文件"""
        fp = os.path.join(filepath, 'load_file.txt')
        with open(fp, 'a+', encoding='utf-8') as fout:
            fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
            fout.write('\n')
            for i in docs:
                fout.write(str(i))
                fout.write('\n')
            fout.close()

    def load_file(self, doc_path, kb_path, sentence_size=100, using_zh_title_enhance=True):
        """读取单个文件，转换为list格式的知识库docs，并保存和返回"""
        if os.path.exists(doc_path):
            if doc_path.lower().endswith(".md"):
                loader = UnstructuredFileLoader(doc_path, mode="elements")
                docs = loader.load()
            elif doc_path.lower().endswith(".txt"):
                loader = TextLoader(doc_path, autodetect_encoding=True)
                textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = loader.load_and_split(textsplitter)
            elif doc_path.lower().endswith(".pdf"):
                # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
                from utils.loader import UnstructuredPaddlePDFLoader
                loader = UnstructuredPaddlePDFLoader(doc_path)
                textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
                docs = loader.load_and_split(textsplitter)
            elif doc_path.lower().endswith(".jpg") or doc_path.lower().endswith(".png"):
                # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
                from utils.loader import UnstructuredPaddleImageLoader
                loader = UnstructuredPaddleImageLoader(doc_path, mode="elements")
                textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = loader.load_and_split(text_splitter=textsplitter)
            elif doc_path.lower().endswith(".csv"):
                loader = CSVLoader(doc_path)
                docs = loader.load()
            else:
                loader = UnstructuredFileLoader(doc_path, mode="elements")
                textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = loader.load_and_split(text_splitter=textsplitter)
            if using_zh_title_enhance:
                docs = zh_title_enhance(docs)
            self.write_check_file(kb_path, docs)
        else:
            logger.info(f"{doc_path} 文件不存在")
        return docs

    def load_batch_file(self):
        loaded_files = []
        failed_files = []
        docs = []
        if isinstance(self.docs_path, str):
            if not os.path.exists(self.docs_path):
                logger.info(f"{self.docs_path} 路径不存在")
                return None
            elif os.path.isfile(self.docs_path):
                file = os.path.split(self.docs_path)[-1]
                try:
                    docs = self.load_file(self.docs_path, self.sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(self.docs_path)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(self.docs_path):
                for fullfilepath, file in zip(*self.tree(self.docs_path, ignore_dir_names=['tmp_files'])):
                    try:
                        docs += self.load_file(fullfilepath, self.kb_path, self.sentence_size)
                        logger.info(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}")
        else:
            logger.info(f"{self.docs_path} 格式不正确")
        self.write_check_file(self.kb_path, docs)
        return docs

    def init_vector_store(self):
        if not os.path.exists(self.kb_path):
            logger.info(f"{self.kb_path} 路径不存在")
        else:
            #加载文件，保存为知识库
            docs = self.load_batch_file()
            if len(docs) > 0:
                logger.info("文件加载完毕，正在生成向量库，知识库长度：{}".format(len(docs)))
                self.vector_store = FAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                self.vector_store.save_local(self.vs_path)  #保存向量库
                torch_gc()
            else:
                logger.info("知识库为空，请重新生成。")

    def load_vector_store(self):
        if self.vs_path and os.path.isdir(self.vs_path) and "index.faiss" in os.listdir(self.vs_path):
            self.vector_store = FAISS.load_local(self.vs_path, self.embeddings)
            torch_gc()
        else:
            logger.info(f"{self.vs_path} 知识向量库不存在")

if __name__ == '__main__':
    config = CFG()
    MYFAISS = MYFAISS(config)
    docs = MYFAISS.load_batch_file()
    index = 1
    for item in docs:
        print(index, " ", item)
        index = index + 1
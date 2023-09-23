import os
from chain.documentqa import DocumentEnhanceQA
from vector_searcher.MYFAISS import MYFAISS
from config import CFG
from logs import log_config

logger = log_config.logger

if __name__ == '__main__':
    config = CFG()
    QAChain = DocumentEnhanceQA(config)
    result = QAChain.get_llm_answer('夏天电动车车主的烦恼是什么')
    print(result)
    result = QAChain.get_knowledge_based_answer('夏天电动车车主的烦恼是什么')
    print()
    search_text = []
    for idx, source in enumerate(result['source_documents'][:4]):
        sep = f'【相关知识{idx + 1}：】'
        print(f'{sep}{source.page_content}\n')
    print()
    print(result['result'])

    #config = CFG()
    #source_service = MYFAISS(config)
    #source_service.load_vector_store()
    #search_result = source_service.vector_store.similarity_search_with_score('夏天电动车车主的烦恼是什么')
    #for result in search_result:
    #    for item in result:
    #        print(type(item), item)  # 知识相关度 Score 阈值，分值越低匹配度越高

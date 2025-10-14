from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils.milvus_config import vector_store


# 检索相关内容
def retrieve_test_knowledge(query):
    """
    检索相关的测试资产
    :param query: 用户查询
    :return: 检索结果
    """
    # 检索相关内容
    results = vector_store.similarity_search(query, k=3)
    for i, result in enumerate(results, 1):
        print(f"结果 {i}: {result.page_content}")

# 示例：检索与“邮箱验证”的相关内容


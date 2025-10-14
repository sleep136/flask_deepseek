from langchain.chains import RetrievalQA

from langchain_openai.chat_models.base import BaseChatOpenAI
from utils.milvus_load import vector_store

# 初始化LLM
llm = BaseChatOpenAI(
    model='deepseek-chat',  # 使用DeepSeek聊天模型
    openai_api_key="sk-a8832e80ca9e4ac6bdd3d0cd39a77d5d",  # 替换为你的API易API密钥
    openai_api_base="https://api.deepseek.com",  # API易的端点
    max_tokens=1024  # 设置最大生成token数
)

# 构建 RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)


def get_answer_from_ds(query):
    # 用户查询
    response = qa_chain({"query": query})
    # response,source_documents = qa_chain.run(query=query)
    print("生成的测试用例建议：")
    print(response)
    return response["result"]

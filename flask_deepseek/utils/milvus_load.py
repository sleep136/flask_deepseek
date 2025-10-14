import os
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from langchain.vectorstores import Milvus
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['OPENAI_API_KEY'] = 'sk-xsk-a8832e80ca9e4ac6bdd3d0cd39a77d5d'


# 指定预训练模型名称和参数
model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
# 初始化HuggingFaceBgeEmbeddings对象
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
# 文本分割器，确保文本块适合向量化
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 文本分割器，确保文本块适合向量化
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 定义 Milvus 的 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
  #  FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # 维度与嵌入模型匹配
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields, "测试知识库存储需求")
collection_name = "test_knowledge_base1"
collection = Collection(name=collection_name, schema=schema)


def insert_data_to_milvus(collection, texts):
    """
    将文本转换为向量并存储到 Milvus
    :param collection: Milvus Collection
    :param texts: 文本列表
    """
    # 分割文本为小块
    chunks = text_splitter.split_text(texts)

    # 生成嵌入向量
    embeddings = embedding_model.embed_documents(chunks)

    # 插入到 Milvus
    collection.insert([embeddings, chunks])

    # 示例：插入需求文档


loader = TextLoader("./test_milvus.txt", encoding="utf-8")
documents = loader.load()

file_path = Path('./test_milvus.txt')

content = file_path.read_text(encoding="utf-8")
insert_data_to_milvus(collection, content)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
#
# # 创建向量存储
# embeddings = OpenAIEmbeddings()
#
# vector_store = FAISS.from_documents(texts, embeddings)
vector_store = Milvus(
    collection_name=collection_name,
    connection_args={"host": "localhost", "port": "19530"},
    #embedding_function=embedding_model.embed_query
    embedding_function=embedding_model
)

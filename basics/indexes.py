#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 19:01
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import TokenTextSplitter

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def load_text_file(file_path: str):
    """
    文档加载器
    :param file_path: 文件路径
    :return: 文本列表
    """
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    docs = loader.load()
    print(f'docs:{docs}')
    print(len(docs))
    print(docs[0].page_content[:10])


def text_splitter_docs():
    """
    使用 CharacterTextSplitter 将文档切分成较小的文本块。

    1. 初始化 CharacterTextSplitter 对象，指定切分参数。
    2. 将字符串列表转换为 Document 对象列表。
    3. 使用 CharacterTextSplitter 对 Document 对象列表进行切分。
    4. 输出切分后的结果。

    :return: 无返回值
    """

    # 初始化 CharacterTextSplitter 对象
    text_splitter = CharacterTextSplitter(
        separator='',  # 分隔文本的字符或者字符串（这里设置为空字符串，表示不使用分隔符）
        chunk_size=5,  # 每个文本块的最大长度
        chunk_overlap=2  # 文本块之间的重叠字符串长度
    )

    # 定义文档字符串列表
    docs = [
        "韩国未来环境可能不再利于白菜生长",
        "央视发声明回应不播国足比赛",
        "中国高铁1公里耗1万度电?官方回应",
    ]

    # 将字符串列表转换为 Document 对象列表
    documents = [Document(page_content=doc) for doc in docs]

    # 使用 CharacterTextSplitter 对 Document 对象列表进行切分
    results = text_splitter.split_documents(documents)

    # results = text_splitter.split_text(','.join(docs))

    # 输出切分后的结果
    print(f'results:{results}')


def vector_stores(file_path: str):
    """
    读取文件内容，切分文档，将切分后的文档向量化并保存到 Chroma 向量数据库中，
    并执行一次相似性搜索。

    :param file_path: 文件路径
    :return: 无返回值
    """

    # 读取文件内容
    with open(file_path, encoding="utf-8") as f:
        pku_text = f.read()

    # 切分文档
    text_splitter = CharacterTextSplitter(
        separator='',  # 分隔文本的字符或者字符串（这里设置为空字符串，表示不使用分隔符）
        chunk_size=100,  # 每个文本块的最大长度
        chunk_overlap=5  # 文本块之间的重叠字符串长度
    )
    texts = text_splitter.split_text(pku_text)
    print(f'texts:{len(texts)}')  # 输出切分后的文本块数量

    # 将切分后的文档向量化保存
    embed = QianfanEmbeddingsEndpoint()  # 初始化嵌入模型

    # 创建 Chroma 向量存储，并将切分后的文本块向量化保存
    doc_search = Chroma.from_texts(texts, embed)

    # 查询
    query = "中国高铁1公里耗1万度电?这种说法正确吗"  # 设置查询文本
    docs = doc_search.similarity_search(query)  # 在向量存储中搜索与查询文本最相似的文档
    print(f'docs:{docs}')  # 输出搜索结果
    print(len(docs))  # 输出搜索结果的数量


def retriever_faiss(file_path: str):
    """
    读取文件内容，切分文档，将切分后的文档向量化并保存到 FAISS 向量数据库中，
    并执行一次相似性搜索。

    :param file_path: 文件路径
    :return: 无返回值
    """

    # 加载文件内容
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()

    # 使用 TokenTextSplitter 依据 tokens 数量切分文档
    text_splitter = TokenTextSplitter(
        chunk_size=50,  # 减少每个文本块的 token 长度
        chunk_overlap=10  # tokens 重叠部分
    )

    # 将文档切分为较小的文本块
    docs = text_splitter.split_documents(documents)

    # 初始化嵌入模型
    embed = QianfanEmbeddingsEndpoint()

    # 创建 FAISS 向量存储，并将切分后的文本块向量化保存
    db = FAISS.from_documents(docs, embed)

    # 创建检索器
    retriever = db.as_retriever(search_kwargs={"k": 1})

    # 执行相似性搜索
    query = "中国高铁1公里耗1万度电?这种说法正确吗"
    # results = retriever.get_relevant_documents(query)
    results = retriever.invoke(query)
    print(f'results:{results}')


if __name__ == '__main__':
    file_file = "/Users/oscar/projects/agents/langchain_tutorials/data/demo.txt"
    # load_text_file(file_path=file_file)

    # text_splitter_docs()

    # vector_stores(file_file)

    retriever_faiss(file_path=file_file)

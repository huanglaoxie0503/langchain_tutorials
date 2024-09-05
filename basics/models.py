#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 12:00
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.messages import HumanMessage

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def llms():
    """
    预训练模型 LLMS
    :return:
    """

    llm = QianfanLLMEndpoint(model='ERNIE-4.0-8K-Preview')

    res = llm.invoke("1+1=？")

    print(res)


def chat_models():
    """
    聊天模型（LLMS 的微调模型）
    :return:
    """
    # 实例化模型
    chat = QianfanChatEndpoint(model='ERNIE-4.0-8K-Preview-0518')
    messages = [HumanMessage(content='帮我介绍下诗仙李白的生平！')]
    res = chat.invoke(messages)
    print(res)


def embedding_models():
    """
    向量模型
    :return:
    """
    # 实例化向量模型
    embed = QianfanEmbeddingsEndpoint(model='Embedding-V1')
    # 单个文本向量化
    result_one = embed.embed_query(text='郑钦文退出瓜达拉哈拉赛')
    print(f'result_one: {result_one}')
    print(f'result_one 的向量纬度: {len(result_one)}')
    print('***********************************')
    # 批量文本向量化
    result_batch = embed.embed_documents(
        [
            '台风摩羯来临前夜 海口现巨型闪电',
            '外卖脱骨鸡被盗报警抓获1只小猫',
            '女生离家3月暴瘦50斤父母满眼心疼'
        ]
    )
    print(f'result_batch: {result_batch}')
    print(f'result_batch size: {len(result_batch)}')
    print(f'result_batch 的向量纬度: {len(result_batch[0])}')


if __name__ == '__main__':
    # llms()

    # chat_models()

    embedding_models()

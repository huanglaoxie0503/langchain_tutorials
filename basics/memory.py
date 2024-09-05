#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 18:19
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_community.chat_models import QianfanChatEndpoint

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def add_msg_to_history() -> None:
    """
    添加消息到聊天历史记录

    该函数实例化一个 `ChatMessageHistory` 对象，并向其中添加用户消息和 AI 消息。
    最后打印出当前的消息历史记录，并将其转换为字典形式，再还原为消息对象。

    :return: 无返回值
    """

    # 实例化 ChatMessageHistory 对象
    history = ChatMessageHistory()

    # 向历史记录中添加用户消息
    history.add_user_message("你在哪里？")

    # 向历史记录中添加 AI 消息
    history.add_ai_message("我在图书馆!")

    # 打印当前的消息历史记录
    print(history.messages)

    # 将消息历史记录转换为字典形式
    dicts = messages_to_dict(history.messages)

    # 从字典形式还原消息历史记录
    new_messages = messages_from_dict(dicts)

    # 打印还原后的消息历史记录
    print(new_messages)


# 用于存储会话历史的字典
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    获取会话历史

    :param session_id: 会话 ID
    :return: 会话历史实例
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def store_qa_memory() -> None:
    """
    存储问答对并进行连续对话

    该函数使用 QianfanChatEndpoint 创建一个语言模型，并使用 RunnableWithMessageHistory 进行连续对话。
    最后打印出每次对话的结果。
    """

    # 初始化语言模型
    llm = QianfanChatEndpoint(model="ERNIE-4.0-8K-Preview-0518")

    # 创建 RunnableWithMessageHistory 对象，并传递语言模型和 get_session_history 函数
    chain = RunnableWithMessageHistory(llm, get_session_history)

    # 进行第一次预测
    result1 = chain.invoke(
        "老王有4个女朋友",
        config={"configurable": {"session_id": "1"}}
    )  # session_id 确定线程
    print(f"result1: {result1}")

    # 进行第二次预测
    result2 = chain.invoke(
        "老李有6个女朋友",
        config={"configurable": {"session_id": "1"}}
    )
    print(f"result2: {result2}")

    # 进行第三次预测
    result3 = chain.invoke(
        "老王和老李一共有多少个女朋友？",
        config={"configurable": {"session_id": "1"}}
    )
    print(f"result3: {result3}")


if __name__ == '__main__':
    # add_msg_to_history()

    store_qa_memory()

#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 13:33
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def single_chain_old():
    """
    单个chain
    :return:
    """
    # 定义模版
    template = "(name) 开了一家人工智能公司，帮我取一个符合公司气质的名字！"
    prompt = PromptTemplate(
        input_variables=["name"],  # 包含模版中所有变量名的名称
        template=template  # 表示提示模版字符串，待填充的变量位置
    )
    print(prompt)

    # 实例化模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')

    # 构造 Chain，将大模型与prompt组合在一起
    # LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use RunnableSequence, e.g., `prompt | llm` instead.
    #   chain = LLMChain(llm=llm, prompt=prompt)
    chain = LLMChain(llm=llm, prompt=prompt)

    # 执行 chain
    result = chain.invoke({"name": "李白"})
    print(f"result: {result}")


def single_chain():
    # 定义模版
    template = "{name} 开了一家人工智能公司，帮我取一个符合公司气质的名字！"
    prompt = PromptTemplate(
        input_variables=["name"],  # 包含模版中所有变量名的名称
        template=template  # 表示提示模版字符串，待填充的变量位置
    )
    print(prompt)

    # 实例化模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')

    # 构造 Chain，将大模型与prompt组合在一起
    chain = prompt | llm  # 使用 RunnableSequence

    # 执行 chain
    result = chain.invoke({"name": "李白"})
    print(f"result: {result}")


def composite_chains():
    """
    多链组合
    :return:
    """
    # 定义模版
    template = "{name} 开了一个早餐点，帮我取一个吸引人点名字！"
    prompt = PromptTemplate(
        input_variables=["name"],  # 确保使用大括号{}包裹变量名
        template=template
    )

    # 实例化模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')

    # 构造 chain：第一条链
    first_chain = prompt | llm

    # 创建第二条链
    # 定义模版
    second_template = "{name} 开店赚钱了，然后又开了一个公司，帮我取一个吸引人的名字！"
    second_prompt = PromptTemplate(
        input_variables=["name"],
        template=second_template
    )
    second_chain = second_prompt | llm

    # 融合两条chain：使用 RunnableSequence
    overall_chain = first_chain | second_chain

    # 执行 chain
    result = overall_chain.invoke({"name": "王安石"})
    print(f"result: {result}")


if __name__ == '__main__':
    # llm_chain()
    simple_sequential_chain()

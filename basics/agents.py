#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 14:47
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def qianfan_agent():
    # 实例化大模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')
    # 工具加载函数：利用工具来增强模型
    tools = load_tools(
        tool_names=['llm-math', 'wikipedia'],
        llm=llm
    )
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    prompt_template = "商朝是什么时候建立的？开国皇帝是谁？推理过程显示中文，结果显示中文"
    prompt = PromptTemplate.from_template(prompt_template)

    print(f'prompt: {prompt}')

    res = agent.run(prompt)
    print(f'res: {res}')


if __name__ == '__main__':
    qianfan_agent()

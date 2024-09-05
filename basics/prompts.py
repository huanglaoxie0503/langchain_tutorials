#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-05 12:45
# @Author  :   oscar
# @Desc    :   None
"""
import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from settings import QIANFAN_API_AK, QIANFAN_API_SK

os.environ['QIANFAN_AK'] = QIANFAN_API_AK
os.environ['QIANFAN_SK'] = QIANFAN_API_SK


def few_shot_prompt_demo():
    """
    few_shot 用于在少量示例的基础上训练模型，使其能够更好地完成特定的任务
    :return:
    """
    # 示例
    examples = [
        {"Q": "狗属于哺乳动物", "A": "对"},
        {"Q": "蝙蝠属于鸟类", "A": "错，蝙蝠属于哺乳类"},
        {"Q": "蛇属于爬行动物", "A": "对"},
    ]
    # 设置 example_prompt
    example_template = """
    问题：{Q}
    答案：{A}\n
    """

    # 实例化PromptTemplate
    example_prompt = PromptTemplate(
        input_variables=['Q', 'A'],  # 包含模版中所有变量的名称
        template=example_template  # 表示提示模版的字符串，待填充的变量位置
    )

    # 实例化 few-shot-prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,  # 多个示例，一般是一个字典
        example_prompt=example_prompt,  # PromptTemplate 对象，为示例的格式
        prefix='判断动物的种类？',  # 示例之前要添加的文本
        suffix='问题：{input}\\答案',  # 示例之后要添加的文本
        input_variables=["input"],  # 所需要传递给模版的变量名
        example_separator="\\n"  # 用于分隔多个示例
    )

    # 实例化模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')

    # 指定模型的输入
    prompt_text = few_shot_prompt.format(input='小明属于鸟类')
    print(f'prompt_text: {prompt_text}')

    # 将 prompt_text 输入模型
    result = llm.invoke(prompt_text)
    print(f'result: {result}')


def zero_shot_prompt_demo():
    """
    zero_shot 指在没有任何具体示例的情况下，让模型完成任务。
    :return:
    """
    # 定义模版
    template = "明天我要去{city}旅游，请告诉我{city}有哪些旅游景点？"
    prompt = PromptTemplate(
        input_variables=['city'],
        template=template
    )
    # 使用模版生成具体的prompt
    city = '深圳'
    prompt_text = prompt.format(city=city)
    print(f'prompt_text: {prompt_text}')

    # 实例化模型
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-13B')

    # 将 prompt 传入模型
    result = llm.invoke(prompt_text)
    print(f'result: {result}')


if __name__ == '__main__':
    # few_shot_prompt_demo()

    zero_shot_prompt_demo()

#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
# @Time    :    2024-09-01 17:23
# @Author  :   oscar
# @Desc    :   None
"""
import os
import platform
import subprocess


def load_env_from_bash_profile():
    command = 'source ~/.bash_profile && env'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in proc.stdout:
        (key, _, value) = line.decode('utf-8').partition('=')
        os.environ[key.strip()] = value.strip()


if platform.system() == 'Darwin':
    # 解决os.getenv('QIANFAN_API_AK')返回为None问题
    load_env_from_bash_profile()

QIANFAN_API_AK = os.getenv('QIANFAN_API_AK')
QIANFAN_API_SK = os.getenv('QIANFAN_API_SK')
Baichuan_API_Key = os.getenv('Baichuan_API_Key')


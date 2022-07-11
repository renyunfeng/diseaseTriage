# -*- coding:utf-8 -*-
"""
@File: test.py
@Author: 任云峰
@Time: 2022/6/28 20:32
@Description: 
"""
import pandas as pd
import torch

import data_process
import utils
from my_log import logger
import datetime
import numpy as np
import config
data = pd.read_csv("data/spo.txt", sep="\t", header=None, names=["主体", "属性", "客体"])
print(data.head())
a = data["属性"].unique()
print(a)
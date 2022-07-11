# -*- coding:utf-8 -*-
"""
@File: bert_textcnn.py
@Author: 任云峰
@Time: 2022/7/5 18:38
@Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import config

# -*- coding:utf-8 -*-
"""
@File: bert_classify.py
@Author: 任云峰
@Time: 2022/7/5 19:34
@Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import config


class BertClassify(nn.Module):
    def __init__(self, label_num):
        super().__init__()
        self.config = BertConfig.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.drop = nn.Dropout(0.2)

        if label_num == 20:
            self.classify1 = nn.Linear(self.config.hidden_size, 19)
            self.classify2 = nn.Linear(20, label_num)
        else:
            self.classify1 = nn.Linear(self.config.hidden_size, 60)
            self.classify2 = nn.Linear(62, label_num)

    def forward(self, input_ids, attention_mask, token_type_ids, ages, labels_i=None):
        output = self.bert(input_ids, attention_mask, token_type_ids)[1]
        output = self.drop(output)
        output = F.relu(self.classify1(output))
        if labels_i is None:
            output = torch.cat((output, ages.unsqueeze(1)), dim=1)
            logit = self.classify2(output)
        else:
            output = torch.cat((output, ages.unsqueeze(1), labels_i.unsqueeze(1)), dim=1)
            logit = self.classify2(output)
        return logit

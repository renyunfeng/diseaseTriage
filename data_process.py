# -*- coding:utf-8 -*-
"""
@File: data_process.py
@Author: 任云峰
@Time: 2022/6/28 19:08
@Description:
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config
import utils


def get_data(path):
    """
    pd读取数据
    :param path:
    :return: dataframe
    """
    data = pd.read_excel(path, sheet_name=0, keep_default_na=False)
    return data


def data_proc(data):
    data["text"] = data["diseaseName"] + " " + data["conditionDesc"] + " " + data["title"] + " " + data["hopeHelp"]
    data["text"] = data["text"].apply(lambda x: utils.text_clear(x))
    data["age"] = data["age"].apply(lambda x: utils.age_process(str(x)))
    del data["diseaseName"]
    del data["conditionDesc"]
    del data["title"]
    del data["hopeHelp"]
    return data


def data_add(data):
    data1 = data.copy()
    data1["text"] = data1["title"] + " " + data1["hopeHelp"] + " " + data1["diseaseName"] + " " + data1["conditionDesc"]
    data["text"] = data["diseaseName"] + " " + data["conditionDesc"] + " " + data["title"] + " " + data["hopeHelp"]
    data = pd.concat([data, data1]).reset_index(drop=True)
    data["age"] = data["age"].apply(lambda x: utils.age_process(str(x)))
    del data["diseaseName"]
    del data["conditionDesc"]
    del data["title"]
    del data["hopeHelp"]
    return data


def data_segmentation(data, bert_tokenizer, size):
    """
    分割数据：训练集和验证集
    :param data: dataframe
    :param bert_tokenizer:
    :param size: 训练集 比例
    :return:
    """
    data_train, data_val = train_test_split(data, test_size=1 - size, random_state=12)
    train_dl = DataLoader(InputDataSet(data_train.reset_index(drop=True), bert_tokenizer, config.max_len),
                          batch_size=config.batch_size)
    eval_dl = DataLoader(InputDataSet(data_val.reset_index(drop=True), bert_tokenizer, config.max_len),
                         batch_size=config.batch_size)
    return train_dl, eval_dl


def data_conversion(data):
    """
    把数据分割2份，i 标签， j 标签去掉-1的数据
    :param data: dataframe
    :return:
    """
    data_i = data.copy()
    del data_i["label_j"]

    data_j = data
    # index = data_j_2[data_j_2["label_j"] == -1].index.values
    # data_j_2 = data_j_2.drop(index, axis=0)
    data_j = data_j[~data_j["label_j"].isin([-1])].reset_index(drop=True)
    return data_i, data_j


def fill_paddings(data, max_len):
    """
    补全句长
    :param data:
    :param max_len:
    :return:
    """
    if len(data) < max_len:
        pad_len = max_len - len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:max_len])
    return data


class InputDataSet(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data["text"][idx]

        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        input_ids = fill_paddings(tokens_ids, self.max_len)

        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask, self.max_len)

        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids, self.max_len)
        ages = torch.tensor(self.data["age"][idx], dtype=torch.long)
        if "label_j" in self.data:
            labels_j = torch.tensor(self.data["label_j"][idx], dtype=torch.long)
            labels_i = torch.tensor(self.data["label_i"][idx], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'ages': ages,
                'labels_i': labels_i,
                'labels_j': labels_j
            }
        elif "label_i" in self.data:
            labels_i = torch.tensor(self.data["label_i"][idx], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'ages': ages,
                'labels_i': labels_i
            }
        else:
            ids = torch.tensor(self.data["id"][idx], dtype=torch.long)
            return {
                'ids': ids,
                'ages': ages,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids

            }

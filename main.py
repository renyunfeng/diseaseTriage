# -*- coding:utf-8 -*-
"""
@File: main.py
@Author: 任云峰
@Time: 2022/6/28 19:08
@Description:
"""
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import datetime

import config
import data_process
import train_and_eval
from my_log import logger


def train(bert_tokenizer):
    data = data_process.get_data(config.train_path)
    data = data_process.data_proc(data)
    logger.info("读取数据完成")
    data_i, data_j = data_process.data_conversion(data)
    logger.info("转换数据完成")
    train_i, eval_i = data_process.data_segmentation(data_i, bert_tokenizer, 0.90)
    logger.info("模型i训练开始")
    model_i = train_and_eval.train(train_i, eval_i, label_name="i")
    logger.info("模型i训练完成")
    train_j, eval_j = data_process.data_segmentation(data_j, bert_tokenizer, 0.9)
    logger.info("模型j训练开始")
    model_j = train_and_eval.train(train_j, eval_j, label_name="j")
    logger.info("模型j训练完成")

    torch.save(model_i, config.model_path + "/model_i.bin")
    torch.save(model_j, config.model_path + "/model_j.bin")


def test(bert_tokenizer):
    data = data_process.get_data(config.test_path)
    data = data_process.data_proc(data)
    test_dl = DataLoader(data_process.InputDataSet(data, bert_tokenizer, config.max_len), batch_size=config.batch_size)
    model_i = torch.load(config.model_path + "/model_i.bin")
    model_j = torch.load(config.model_path + "/model_j.bin")
    result = train_and_eval.test(test_dl, model_i, model_j)
    now_date = datetime.datetime.now().strftime('%m-%d_%H')
    file_name = "/result_" + str(now_date) + ".csv"
    result.to_csv(config.result_path + file_name, index=False)


if __name__ == '__main__':
    tokenizers = BertTokenizer.from_pretrained(config.bert_path)
    train(tokenizers)
    logger.info("开始测试数据")
    test(tokenizers)

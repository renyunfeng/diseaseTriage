# -*- coding:utf-8 -*-
"""
@File: config.py
@Author: 任云峰
@Time: 2022/6/28 20:32
@Description: 配置文件，指定一些路径及参数
"""
# 训练集路径
train_path = "data/data_train.xlsx"
# 测试集路径
test_path = "data/data_test.xlsx"
# 停用词路径
stopwords_path = "data/stopwords.txt"
bert_path = "data/medbert"
# 模型存放路径
model_path = "out/model"
# 结果
result_path = "out/result"
label_num_i = 20
label_num_j = 61
# 最大文本长度
max_len = 256

epochs_i = 5
epochs_j = 10

batch_size = 16

lr = 1e-4

# -*- coding:utf-8 -*-
"""
@File: analysis_text.py
@Author: 任云峰
@Time: 2022/6/28 21:47
@Description: 文本数据的分析
"""
import matplotlib.pyplot as plt

import config
import data_process
import utils

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体样式，正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def text_train_len(train_data):
    """
    训练集文本长度分布
    :param train_data:
    :return:
    """
    lens = train_data["text"].apply(lambda x: len(str(x)))
    print(lens.describe())
    plt.title("训练集文本长度分布图")
    plt.hist(lens)
    plt.savefig("out/analysis/train_text_len.png")
    plt.show()


def text_train_len2(train_data):
    """
    分词去停用词后训练集文本长度分布
    :param train_data:
    :return:
    """
    lens = train_data["text"].apply(lambda x: len(utils.text_segmentation(str(x)).split(" ")))
    print(lens.describe())
    plt.title("去停用词后训练集文本长度分布图")
    plt.hist(lens)
    plt.savefig("out/analysis/train_text_len2.png")
    plt.show()


def text_test_len(test_data):
    """
    测试集上文本长度分布
    :param test_data:
    :return:
    """
    lens = test_data["text"].apply(lambda x: len(str(x)))
    print(lens.describe())
    plt.figure("测试集文本长度分布图")
    plt.title("测试集文本长度分布图")
    plt.hist(lens)
    plt.savefig("out/analysis/test_text_len.png")
    plt.show()


def label_i_distribution(data):
    label_count = data["label_i"].value_counts()
    print(label_count)
    label_count.plot(kind="bar")
    plt.title('label i 类别数量分布图')
    plt.xlabel("label num")
    plt.savefig("out/analysis/label_i_distribution.png")
    plt.show()


def label_j_distribution(data):
    label_count = data["label_j"].value_counts()
    print(label_count)
    plt.figure(figsize=(12, 4))
    label_count.plot(kind="bar")
    plt.xticks(fontsize=12)  # 调整字体大小
    plt.title('label j 类别数量分布图')
    plt.xlabel("label num")
    plt.savefig("out/analysis/label_j_distribution.png")
    plt.show()


def label_j_distribution2(data):
    label_count = data[data["label_j"] != -1]["label_j"].value_counts()
    print(label_count)
    plt.figure(figsize=(12, 4))
    label_count.plot(kind="bar")
    plt.xticks(fontsize=12)
    plt.title('label j 类别数量(排除-1)分布图')
    plt.xlabel("label num")
    plt.savefig("out/analysis/label_j_distribution(no_-1).png")
    plt.show()


if __name__ == '__main__':
    train = data_process.get_data(config.train_path)
    # 读取数据并处理，输出出去，看一看是否有问题（没有清洗干净的）
    train.to_csv("out/analysis/train_text.csv", index=False)
    text_train_len(train)
    text_train_len2(train)
    label_i_distribution(train)
    label_j_distribution(train)
    label_j_distribution2(train)

    train = data_process.get_data(config.test_path)
    text_test_len(train)

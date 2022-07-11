# -*- coding:utf-8 -*-
"""
@File: utils.py
@Author: 任云峰
@Time: 2022/6/28 19:08
@Description:
"""
import jieba
import re

import config


def text_clear(text):
    """
    清洗文本数据：例如中的空格及特殊字符等
    :param text:
    :return:
    """
    # text = str(text).replace("\n", "").replace("\r", "").replace("\r\n", "").replace("\\n", "").replace("\\r", "")
    text = re.sub(r'\\n|\\r|\\t|\r|\n|\t|\\\\|\\ue00e|\\x08|\x14', " ", text).lower()
    text = re.sub(" +", ",", text)
    return text


def read_stopwords():
    """
    读取停用词表
    :return:
    """
    stopwords = []
    for word in open(config.stopwords_path, 'r', encoding="utf-8"):
        stopwords.append(word.strip())
    return stopwords


stop_words = read_stopwords()


def text_segmentation(text):
    """
    分词并去停用词
    :param text: 文本
    :return: 空格分割
    """
    result = ""
    words = jieba.cut(text)
    for word in words:
        if word not in stop_words:
            result += word + " "
    return result.strip()


def age_process(age):
    age = int(age.replace("+", ""))
    if age > 100:
        age = -1
    return age


def data_i_add(data):
    data = data.copy()
    data = data[data["label_i"].isin([12, 19])].reset_index(drop=True)
    data["text"] = data["diseaseName"] + " " + data["conditionDesc"]
    del data["diseaseName"]
    del data["conditionDesc"]
    del data["title"]
    del data["hopeHelp"]
    data["text"] = data["text"].apply(lambda x: text_clear(x))
    data["age"] = data["age"].apply(lambda x: age_process(str(x)))
    return data



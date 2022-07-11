# -*- coding:utf-8 -*-
"""
@File: my_log.py
@Author: 任云峰
@Time: 2022/6/28 19:08
@Description:
"""
import logging
import datetime
import os

LOG_PATH = "out/log"
# 第一步，创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # INFO/DEBUG

now_date = datetime.datetime.now()
# now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')
now_date = now_date.strftime('%Y-%m-%d_%H')
# 第二步，创建一个handler，用于写入日志文件

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

file_handler = logging.FileHandler('out/log/' + str(now_date) + '.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
# 添加handler到logger中
logger.addHandler(file_handler)

# 第三步，创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
logger.addHandler(console_handler)

# -*- coding:utf-8 -*-
"""
@File: train_and_eval.py
@Author: 任云峰
@Time: 2022/6/28 19:08
@Description:
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup

import config
from model import bert_classify
from my_log import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dl, eval_dl, label_name):
    if label_name == "i":
        label_num = config.label_num_i
        epochs = config.epochs_i
    elif label_name == "j":
        label_num = config.label_num_j
        epochs = config.epochs_j
    else:
        logger.error("label_name 不符合要求")
        raise NameError("label_name 不符合要求")
    model = bert_classify.BertClassify(label_num)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_dl) * epochs  # len(dataset)*epochs / batchsize
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    for epoch in range(epochs):
        total_train_loss = 0
        model.to(device)
        model.train()
        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            ages = batch["ages"].to(device)
            # logger.info("{}--{}".format(step, ages))
            if label_name == "i":
                logit = model(input_ids, attention_mask, token_type_ids, ages)
                label = batch["labels_i"].to(device)
            elif label_name == "j":
                labels_i = batch["labels_i"].to(device)
                logit = model(input_ids, attention_mask, token_type_ids, ages, labels_i)
                label = batch["labels_j"].to(device)
            else:
                logger.error("label_name 不符合要求")
                raise NameError("label_name 不符合要求")
            loss = loss_fn(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            with torch.no_grad():
                if step % 500 == 0:
                    logger.info("epoch:{}/{},step:{}".format(epoch + 1, epochs, step))
        avg_train_loss = total_train_loss / len(train_dl)
        logger.info("epoch:{}/{},avg_loss:{:.4f}".format(epoch+1, epochs, avg_train_loss))
        acc, f1 = evaluate(eval_dl, model, label_name)
        logger.info("epoch:{}/{},acc:{:.2%}".format(epoch+1, epochs, acc))
        logger.info("epoch:{}/{},f1:{:.2%}".format(epoch+1, epochs, f1))
    return model


def evaluate(eval_dl, model, label_name):
    model.eval()
    pred_labels = []
    labels = []
    corrects = []
    for batch in eval_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        ages = batch["ages"].to(device)
        with torch.no_grad():
            if label_name == "i":
                logit = model(input_ids, attention_mask, token_type_ids, ages)
                pred = torch.argmax(logit, dim=1).cpu().numpy()
                pred_labels.extend(pred.tolist())
                label = batch["labels_i"].to(device)
                label = label.cpu().numpy()
                labels.extend(label.tolist())
                corrects.append(np.mean(label == pred))
            elif label_name == "j":
                labels_i = batch["labels_i"].to(device)
                logit = model(input_ids, attention_mask, token_type_ids, ages, labels_i)
                pred = torch.argmax(logit, dim=1).cpu().numpy()
                pred_labels.extend(pred.tolist())
                label = batch["labels_j"].to(device)
                label = label.cpu().numpy()
                labels.extend(label.tolist())
                corrects.append(np.mean(label == pred))
            else:
                logger.error("label_name 不符合要求")
                raise NameError("label_name 不符合要求")

    f1 = f1_score(labels, pred_labels, average="macro")
    acc = np.mean(corrects)
    return acc, f1


def test(test_dl, model_i, model_j):
    result_i = []
    result_j = []
    idss = []
    for batch in test_dl:
        ids = batch["ids"].to(device)
        idss.extend(ids.cpu().numpy().tolist())
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        ages = batch["ages"].to(device)
        logit_i = model_i(input_ids, attention_mask, token_type_ids, ages)
        pred_i = torch.argmax(logit_i, dim=1)
        result_i.extend(pred_i.cpu().numpy().tolist())
        logit_j = model_j(input_ids, attention_mask, token_type_ids, ages, pred_i)
        pred_j = torch.argmax(logit_j, dim=1).cpu().numpy().tolist()
        result_j.extend(pred_j)

    result = pd.DataFrame({"id": idss, "label_i": result_i, "label_j": result_j})
    return result

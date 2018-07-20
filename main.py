import unicodedata
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import load_data
import preprocess
import train
import modle
import test

# load word embedding from .txt file
embedding_filename = 'preprocess/word_embedding.txt'
word_to_embedding = load_data.loadEmbedVocab(embedding_filename)

# 从训练集得到数据对集合
pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt')
# 从测试集得到数据对集合
test_pairs = load_data.loadDataPairs('data/cikm_test_a_20180516.txt')
# 从验证集得到数据对集合
# TODO: load verify_pairs from spnaish train dat
verify_pairs = load_data.loadDataPairs('data/cikm_english_train_20180516.txt')

# initialize model
lstm = modle.Bi_LSTM()
loss_function = nn.BCELoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.01)

# 显示训练前的结果
train.beforeTrain(model=lstm, loss_function=loss_function, optimizer=optimizer)

# 开始训练模型
train.beginTrain(model=lstm, loss_function=loss_function, optimizer=optimizer)

# 显示训练后在验证集上的结果
test.verifyAfterTrainning(model=lstm, loss_function=loss_function, optimizer=optimizer)

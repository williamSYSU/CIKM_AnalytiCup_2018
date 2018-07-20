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

train.beforeTrain(model=lstm, loss_function=loss_function, optimizer=optimizer)
train.beginTrain(model=lstm, loss_function=loss_function, optimizer=optimizer)


# 训练之后在验证集上的效果
with torch.no_grad():
    print("after learning:")
    sum_loss = 0
    sum = 0
    for pair in verify_pairs:
        sum += 1
        verify_pair = [preprocess.tensorsFromPair_verify(pair, word_to_embedding)]
        tag_scores = lstm(verify_pair[0][0], verify_pair[0][1])
        label = verify_pair[0][2]
        if label == '1':
            label = torch.tensor([1], dtype=torch.float)
        else:
            label = torch.tensor([0], dtype=torch.float)
        loss = loss_function(tag_scores[0].view(-1), label)
        sum_loss += loss
    print("avg_loss:", float(sum_loss / sum))
    for pair in test_pairs:
        test_pair = [preprocess.tensorsFromPair_test(pair, word_to_embedding)]
        tag_scores = lstm(test_pair[0][0], test_pair[0][1])
        with open("test_result.txt", 'a') as f:
            f.write(str(tag_scores[0].item()) + "\n")

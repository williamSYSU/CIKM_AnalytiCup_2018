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
word_to_embedding = load_data.loadEmbedding(embedding_filename)

# 从训练集得到数据对集合
lines = open('data/cikm_spanish_train_20180516.txt' , encoding='utf-8').\
        read().strip().split('\n')
pairs = [[load_data.normalizeString(s) for s in l.split('\t')] for l in lines]
# 从测试集得到数据对集合
test_lines = open('data/cikm_test_a_20180516.txt', encoding='utf-8'). \
    read().strip().split('\n')
test_pairs = [[load_data.normalizeString(s) for s in l.split('\t')] for l in test_lines]
# 从验证集得到数据对集合
verify_lines = open('data/cikm_english_train_20180516.txt', encoding='utf-8'). \
    read().strip().split('\n')
verify_pairs = [[load_data.normalizeString(s) for s in l.split('\t')] for l in verify_lines]


#两个lstm网络模型
class bi_lstm(nn.Module):
    def __init__(self):
        super(bi_lstm, self).__init__()
        self.bi_lstm_context1 = nn.LSTM(300, 100, bidirectional=True)
        self.bi_lstm_context2 = nn.LSTM(300, 100, bidirectional=True)
        self.dense1 = nn.Linear(800, 200)
        self.dense2 = nn.Linear(200, 50)
        self.dense3 = nn.Linear(50, 2)
        self.stm=nn.Softmax(dim=0)

    def forward(self, input1, input2):
        out1, (_, _) = self.bi_lstm_context1(input1.unsqueeze(0))
        out2, (_, _) = self.bi_lstm_context2(input2.unsqueeze(0))
        a = torch.cat((out1[0][0], out1[0][-1], out2[0][0], out2[0][-1]), dim=0)
        out = self.dense1(a)
        out = self.dense2(out)
        out = self.dense3(out)
        out=self.stm(out)
        return out
lstm=bi_lstm()
loss_function =nn.BCELoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.01)
# 训练之前在验证集上的效果
with torch.no_grad():
    print("before learning:")
    sum_loss = 0
    sum = 0
    for pair in verify_pairs:
        sum += 1
        verify_pair = [tensorsFromPair_verify(pair)]
        tag_scores = lstm(verify_pair[0][0], verify_pair[0][1])
        label = verify_pair[0][2]
        if label == '1':
            label = torch.tensor([1], dtype=torch.float)
        else:
            label = torch.tensor([0], dtype=torch.float)
        loss = loss_function(tag_scores[0].view(-1), label)
        sum_loss += loss
    print("avg_loss:", float(sum_loss / sum))
# 在训练集上训练
for epoch in range(100):
    for pair in pairs:
         lstm.zero_grad()
         training_pair = [tensorsFromPair(pair)]
         # print (training_pair)
         tag_scores = lstm(training_pair[0][0], training_pair[0][1])
         label = training_pair[0][2]
         if label=='1':
             label=torch.tensor([1],dtype=torch.float)
         else:
             label=torch.tensor([0],dtype=torch.float)
         loss=loss_function(tag_scores[0].view(-1),label)
         loss.backward()
         optimizer.step()
    print(loss.item())
# 训练之后在验证集上的效果
with torch.no_grad():
    print("after learning:")
    sum_loss = 0
    sum = 0
    for pair in verify_pairs:
        sum += 1
        verify_pair = [tensorsFromPair_verify(pair)]
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
        test_pair = [tensorsFromPair_test(pair)]
        tag_scores = lstm(test_pair[0][0], test_pair[0][1])
        with open("test_result.txt", 'a') as f:
            f.write(str(tag_scores[0].item()) + "\n")

import torch
import torch.nn as nn
import torch.functional as F

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
TARGET_SIZE = 2
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.01
EPOCH_NUM = 5

ENGLISH_TAG = 1  # 是否加入英语原语训练集，0：不加入；1：加入
ENGLISH_SPANISH_RATE = 1  # 英语原语训练数据与西班牙原语训练数据的比例
TRAINTEST_RATE = 0.7  # 划分训练集和验证集的比例


# 两个lstm网络模型
class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        print('Current Model: Bi_LSTM')
        self.bi_lstm_context1 = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, bidirectional=True)
        self.bi_lstm_context2 = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, bidirectional=True)
        self.dense1 = nn.Linear(8 * HIDDEN_SIZE, 400)
        self.dense2 = nn.Linear(400, 100)
        self.dense3 = nn.Linear(100, TARGET_SIZE)

        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.stm = nn.Softmax(dim=0)

    def forward(self, input1, input2):
        out1, (_, _) = self.bi_lstm_context1(input1.unsqueeze(0))
        out2, (_, _) = self.bi_lstm_context2(input2.unsqueeze(0))
        merge = torch.cat((out1[0][0], out1[0][-1], out2[0][0], out2[0][-1]), dim=0)
        out = self.dense1(merge)
        out = self.dense2(out)
        # out = self.dropout(out)
        out = self.dense3(out)
        out = self.dropout(out)
        out = self.stm(out)
        return out


# 单向LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        print('Current model: LSTM')
        self.lstm1 = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE)
        self.lstm2 = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE)
        self.dense1 = nn.Linear(2 * HIDDEN_SIZE, 256)
        self.dense2 = nn.Linear(256, 50)
        self.dense3 = nn.Linear(50, TARGET_SIZE)

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.stm = nn.Softmax(dim=0)

    def forward(self, input1, input2):
        out1, hidden1 = self.lstm1(input1.unsqueeze(0))
        out2, hidden2 = self.lstm2(input2.unsqueeze(0))

        merge = torch.cat((out1[0][-1], out2[0][-1]), dim=0)
        out = self.dense1(merge)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dropout(out)
        out = self.stm(out)
        return out

import torch
import torch.nn as nn

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
TARGET_SIZE = 2
DROPOUT_RATE = 0.05
LEARNING_RATE = 0.01
TRAINING_RATE = 0.8
EPOCH_NUM = 10

TEST_TAT = 1000

# 两个lstm网络模型
class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
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
        a = torch.cat((out1[0][0], out1[0][-1], out2[0][0], out2[0][-1]), dim=0)
        out = self.dense1(a)
        out = self.dense2(out)
        # out = self.dropout(out)
        out = self.dense3(out)
        out = self.dropout(out)
        out = self.stm(out)
        return out

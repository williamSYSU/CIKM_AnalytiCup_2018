import torch
import torch.nn as nn


# 两个lstm网络模型
class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        self.bi_lstm_context1 = nn.LSTM(300, 100, bidirectional=True)
        self.bi_lstm_context2 = nn.LSTM(300, 100, bidirectional=True)
        self.dense1 = nn.Linear(800, 200)
        self.dense2 = nn.Linear(200, 50)
        self.dense3 = nn.Linear(50, 2)
        self.stm = nn.Softmax(dim=0)

    def forward(self, input1, input2):
        out1, (_, _) = self.bi_lstm_context1(input1.unsqueeze(0))
        out2, (_, _) = self.bi_lstm_context2(input2.unsqueeze(0))
        a = torch.cat((out1[0][0], out1[0][-1], out2[0][0], out2[0][-1]), dim=0)
        out = self.dense1(a)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.stm(out)
        return out

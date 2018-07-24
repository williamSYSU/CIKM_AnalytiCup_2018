import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
TARGET_SIZE = 2
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCH_NUM = 1500

ENGLISH_TAG = 1  # 是否加入英语原语训练集，0：不加入；1：加入
ENGLISH_SPANISH_RATE = 1  # 英语原语训练数据与西班牙原语训练数据的比例
TRAINTEST_RATE = 0.7  # 划分训练集和验证集的比例
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_SQE_LEN = 56  # 最长的句子词数
END_OF_SEN = torch.ones(1, dtype=torch.float).new_full((1, EMBEDDING_SIZE), 0)


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

        # self.stm = nn.Softmax(dim=0)
        self.stm = nn.Sigmoid()

    def forward(self, input1, input2):
        out1, (_, _) = self.bi_lstm_context1(input1)
        out2, (_, _) = self.bi_lstm_context2(input2)

        # 当batch_size > 1时，需要根据batch_size手动合并
        all_merge = []
        for idx in range(len(out1)):
            merge = torch.cat((out1[idx][0], out1[idx][-1], out2[idx][0], out2[idx][-1]), dim=0)
            if idx is 0:
                all_merge = merge.unsqueeze(0)
            else:
                all_merge = torch.cat((all_merge, merge.unsqueeze(0)), dim=0)

        out = self.dense1(all_merge)
        out = self.dense2(out)
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
        # self.stm = nn.Sigmoid()

    def forward(self, input1, input2):
        out1, hidden1 = self.lstm1(input1)
        out2, hidden2 = self.lstm2(input2)

        merge = torch.cat((out1[0][-1], out2[0][-1]), dim=0)
        out = self.dense1(merge)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dropout(out)
        out = self.stm(out)
        return out


class MatchSRNN(nn.Module):
    def __init__(self):
        super(MatchSRNN, self).__init__()
        print('Current model: Match-SpatialRNN')
        self.dimension = 3
        self.hidden_dim = 3
        self.target = 2
        self.T = torch.nn.Parameter(torch.randn(self.dimension, 300, 300))
        self.Linear = nn.Linear(600, self.dimension)
        self.relu = nn.ReLU()
        self.qrLinear = nn.Linear(3 * self.hidden_dim + self.dimension, 3 * self.hidden_dim)
        self.qzLinear = nn.Linear(3 * self.hidden_dim + self.dimension, 4 * self.hidden_dim)
        self.U = torch.nn.Parameter(torch.randn(self.hidden_dim, 3 * self.hidden_dim))
        self.h_linear = nn.Linear(self.dimension, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.lastlinear = nn.Linear(self.dimension, self.target)

    def getS(self, input1, input2):
        out = []
        for i in range(self.dimension):
            tmp = torch.mm(input1.view(1, -1), self.T[i])
            tmp = torch.mm(tmp, input2.view(-1, 1))
            out.append(tmp.item())
        add_input = torch.cat((input1.view(1, -1), input2.view(1, -1)), dim=1)
        lin = self.Linear(add_input)
        out = torch.add(torch.tensor(out), lin.view(-1))
        out = self.relu(out)
        return out.view(1, -1)

    def softmaxbyrow(self, input):
        # z1=input[:self.hidden_dim]
        # z2=input[self.hidden_dim:self.hidden_dim*2]
        # z3 = input[self.hidden_dim*2:self.hidden_dim * 3]
        # z4 = input[self.hidden_dim*3:self.hidden_dim * 4]
        input = input.view(4, -1)
        input = torch.transpose(input, 0, 1)
        a = []
        for i in range(self.hidden_dim):
            if i == 0:
                tmp = F.softmax(input[i], dim=0).view(1, -1)
            else:
                tmp = torch.cat((tmp, F.softmax(input[i], dim=0).view(1, -1)), dim=0)

        z1 = tmp[:, 0]
        z2 = tmp[:, 1]
        z3 = tmp[:, 2]
        z4 = tmp[:, 3]

        return z1, z2, z3, z4

    def spatialRNN(self, input_s, hidden):
        q = torch.cat((torch.cat((hidden[0], hidden[1])), torch.cat((hidden[2], input_s))))
        r = F.sigmoid(self.qrLinear(q))
        # print("q:",q)
        z = self.qzLinear(q)
        z1, z2, z3, z4 = self.softmaxbyrow(z)
        # print("r:",r)
        # print("qwe:",torch.cat((hidden[0], hidden[1], hidden[2])))
        # print("sd:",torch.mm(self.U,(r*torch.cat((hidden[0],hidden[1],hidden[2]))).view(-1,1)).view(-1))
        # print("fdsf:",self.h_linear(input_s))
        h_ = self.tanh(self.h_linear(input_s) + torch.mm(self.U,
                                                         (r * torch.cat((hidden[0], hidden[1], hidden[2]))).view(-1,
                                                                                                                 1)).view(
            -1))
        h = z2 * hidden[1] + z3 * hidden[0] + z4 * hidden[2] + h_ * z1
        # print(z2*hidden[1],z3*hidden[0],z4*hidden[2],h_*z1)
        # print("h",h)
        return h

    def init_hidden(self, all_hidden, i, j):
        if i == 0 and j == 0:
            return [torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)]
        elif i == 0:
            return [torch.zeros(self.hidden_dim), all_hidden[i][j - 1], torch.zeros(self.hidden_dim)]
        elif j == 0:
            return [all_hidden[i - 1][j], torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)]
        else:
            return all_hidden[i - 1][j], all_hidden[i][j - 1], all_hidden[i - 1][j - 1]

    def forward(self, input1, input2):
        count = 0
        for i in range(input1.size(0)):
            for j in range(input2.size(0)):
                if count == 0:
                    s = self.getS(input1[i], input2[j])
                    count += 1
                else:
                    s_ij = self.getS(input1[i], input2[j])
                    s = torch.cat((s, s_ij), dim=0)
        s = s.view(input1.size(0), input2.size(0), -1)
        all_hidden = [[] for i in range(input1.size(0))]
        for i in range(input1.size(0)):
            for j in range(input2.size(0)):
                # print(self.init_hidden(all_hidden,i,j))
                hidden = self.spatialRNN(s[i][j], self.init_hidden(all_hidden, i, j))
                all_hidden[i].append(hidden)
        # print(all_hidden)

        out = self.lastlinear(all_hidden[input1.size(0) - 1][input2.size(0) - 1])
        out = F.softmax(out, dim=0)
        return out

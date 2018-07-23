import torch
import torch.nn as nn
import torch.nn.functional as F


class testModule(nn.Module):
    def __init__(self):
        super(testModule, self).__init__()
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
            return ([torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)])
        elif i == 0:
            return ([torch.zeros(self.hidden_dim), all_hidden[i][j - 1], torch.zeros(self.hidden_dim)])
        elif j == 0:
            return ([all_hidden[i - 1][j], torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)])
        else:
            return (all_hidden[i - 1][j], all_hidden[i][j - 1], all_hidden[i - 1][j - 1])

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

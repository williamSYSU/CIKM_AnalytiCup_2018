import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamicpool import DynamicPool

CHANNEL_SIZE = 3
CONV_TARGET = 200
TARGET_SIZE = 2
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
EPOCH_NUM = 2
EMBEDDING_SIZE = 300
ENGLISH_TAG = 1  # 是否加入英语原语训练集，0：不加入；1：加入
TRAINTEST_RATE = 0.7  # 划分训练集和验证集的比例
ENGLISH_SPANISH_RATE = 1


class Text2Image(nn.Module):
    def __init__(self):
        super(Text2Image, self).__init__()
        self.conv1 = nn.Conv2d(1, CHANNEL_SIZE, 2, padding=0)
        self.conv2_1 = nn.Conv2d(1, 1, 6, padding=0)
        self.conv2_2 = nn.Conv2d(1, 1, 6, padding=0)
        self.conv2_3 = nn.Conv2d(1, 1, 6, padding=0)
        self.fc1 = nn.Linear((CONV_TARGET - 5)//3 * (CONV_TARGET - 5)//3 * 1, 100)
        # self.fc1 = nn.Linear(CONV_TARGET * CONV_TARGET * CHANNEL_SIZE, 100)
        self.fc2 = nn.Linear(100, 2)
        # self.fc3 = nn.Linear(30, TARGET_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.softmax = nn.Softmax(dim=1)
        self.target_pool = [CONV_TARGET, CONV_TARGET]

    def forward(self, matrix_x):

        matrix_x = F.tanh(self.conv1(matrix_x))

        origin_size1 = len(matrix_x[0][0])
        origin_size2 = len(matrix_x[0][0][0])
        # 填充卷积输出
        # print(matrix_x)
        while origin_size1 < self.target_pool[0]:
            matrix_x = torch.cat([matrix_x, matrix_x[:, :, :origin_size1, :]], dim=2)
            if len(matrix_x[0][0]) >= self.target_pool[0]:
                break

        while origin_size2 < self.target_pool[1]:
            matrix_x = torch.cat([matrix_x, matrix_x[:, :, :, :origin_size2]], dim=3)
            if len(matrix_x[0][0][0]) >= self.target_pool[1]:
                break

        dynamic_pool_size1 = len(matrix_x[0][0])
        dynamic_pool_size2 = len(matrix_x[0][0][0])
        get_index = DynamicPool(self.target_pool[0], self.target_pool[1], dynamic_pool_size1, dynamic_pool_size2)
        index, pool_size = get_index.d_pool_index()
        m, n, high_judge, weight_judge = get_index.cal(index)
        stride = pool_size[0]
        stride1 = pool_size[1]

        matrix_x1 = matrix_x[:, :, :m, :n]
        matrix_x1 = F.max_pool2d(matrix_x1, (stride, stride1))

        if high_judge > 0:
            matrix_x2 = matrix_x[:, :, m:, :n]
            matrix_x2 = F.max_pool2d(matrix_x2, (stride + 1, stride1))
        if weight_judge > 0:
            matrix_x3 = matrix_x[:, :, :m, n:]
            matrix_x3 = F.max_pool2d(matrix_x3, (stride, stride1 + 1))
        if high_judge > 0 and weight_judge > 0:
            matrix_x4 = matrix_x[:, :, m:, n:]
            matrix_x4 = F.max_pool2d(matrix_x4, (stride + 1, stride1 + 1))

        if high_judge == 0 and weight_judge == 0:
            matrix_x = matrix_x1
        elif high_judge > 0 and weight_judge == 0:
            matrix_x = torch.cat([matrix_x1, matrix_x2], dim=2)
        elif high_judge == 0 and weight_judge > 0:
            matrix_x = torch.cat([matrix_x1, matrix_x3], dim=3)
        else:
            matrix_x_1 = torch.cat([matrix_x1, matrix_x2], dim=2)
            matrix_x_2 = torch.cat([matrix_x3, matrix_x4], dim=2)
            matrix_x = torch.cat([matrix_x_1, matrix_x_2], dim=3)

        need_matrix = matrix_x[:, 0, :, :].view(1, 1, CONV_TARGET, CONV_TARGET)
        mat2_1 = self.conv2_1(need_matrix)
        need_matrix = matrix_x[:, 1, :, :].view(1, 1, CONV_TARGET, CONV_TARGET)
        mat2_2 = self.conv2_1(need_matrix)
        need_matrix = matrix_x[:, 2, :, :].view(1, 1, CONV_TARGET, CONV_TARGET)
        mat2_3 = self.conv2_1(need_matrix)

        # mat2_1 = F.relu(self.conv2_1(matrix_x[0][0]))
        # mat2_2 = F.relu(self.conv2_1(matrix_x[]))
        # mat2_3 = F.relu(self.conv2_1(matrix_x))

        reshape_matrix = F.tanh(mat2_1 + mat2_2 + mat2_3)
        reshape_matrix = self.dropout(reshape_matrix)
        reshape_matrix = F.max_pool2d(reshape_matrix, (3, 3))
        # matrix_x = matrix_x.view(-1, self.num_flat_features(matrix_x))
        reshape_matrix = reshape_matrix.view(-1, self.num_flat_features(reshape_matrix))
        reshape_matrix = F.tanh(self.fc1(reshape_matrix))
        reshape_matrix = F.tanh(self.fc2(reshape_matrix))
        # matrix_x = self.fc3(matrix_x)
        reshape_matrix = self.softmax(reshape_matrix)
        return reshape_matrix

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# class TestModel(nn.Module):
#     def __init__(self):
#         super(TestModel, self).__init__()
#
#     def forward(self, input):
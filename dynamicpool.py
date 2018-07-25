import numpy as np
import torch
import modelNet


class DynamicPool:
    def __init__(self, target_pool_high, target_pool_weight, sentence1_len, sentence2_len):
        self.target_weight = target_pool_weight
        self.target_high = target_pool_high
        self.sen1_len = sentence1_len
        self.sen2_len = sentence2_len
        self.check_len = 0
        self.stride1 = sentence1_len // target_pool_high
        self.stride2 = sentence2_len // target_pool_weight

    def d_pool_index(self):
        stride2 = self.sen2_len // self.target_weight
        stride1 = self.sen1_len // self.target_high
        pool_size = [stride1, stride2]
        pool_index = np.zeros((self.target_high, self.target_weight))

        # 建立动态pool index
        adjust_high = self.sen1_len % self.target_high
        adjust_weight = self.sen2_len % self.target_weight

        if adjust_high != 0:
            pool_index[self.target_high - adjust_high:, :] += np.ones((adjust_high, self.target_weight))
        if adjust_weight != 0:
            pool_index[:, self.target_weight - adjust_weight:] += np.ones((self.target_high, adjust_weight))

        return pool_index, pool_size

    def cal(self, pool_index):
        final_weight = list(pool_index[0]).count(0) * self.stride2
        final_high = list(pool_index[:, 0]).count(0) * self.stride1
        high_judge = self.sen1_len % self.target_high
        weight_judge = self.sen2_len % self.target_weight
        return final_high, final_weight, high_judge, weight_judge

    @staticmethod
    def cal_similar_matrix(sentence_1, sentence_2):
        SIZE = len(sentence_1)
        likelihood_matrix = torch.zeros(SIZE, 56, 56).to(modelNet.DEVICE)
        # sentence1 = sentence_1.numpy()
        # sentence2 = sentence_2.numpy()

        for simple_batch in range(SIZE):
            likelihood_matrix[simple_batch] = sentence_1[simple_batch].mm(sentence_2[simple_batch].t())
        # 计算相似度矩阵，由embedding计算得到
        # for i in range(len(sentence1)):
        #     for j in range(len(sentence2)):
        #         # 两种相似度矩阵计算方法
        #         # likelihood_matrix[i][j] = np.dot(sentence1[i], sentence2[j]) / (
        #         #             np.linalg.norm(sentence1[i], ord=2) * np.linalg
        #         #             .norm(sentence2[j], ord=2))
        #         likelihood_matrix[i][j] = np.dot(sentence1[i], sentence2[j])

        likelihood_matrix = likelihood_matrix.view(SIZE, 1, 56, 56)
        return likelihood_matrix, SIZE

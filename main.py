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
import modelNet
import test


class PARAMETER():
    def __init__(self):
        # 从本地加载word embedding词典
        self.word_to_embedding = load_data.loadEmbedVocab('preprocess/word_embedding.txt')

        # 加载数据对集合
        self.train_pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt')
        self.english_train_pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt')
        self.test_pairs = load_data.loadDataPairs('data/cikm_test_a_20180516.txt')

        # 划分训练集和验证集
        self.train_pairs, self.verify_pairs = preprocess.load_training_and_verify_pairs(pairs=self.train_pairs)
        # self.train_pairs = self.train_pairs + self.english_train_pairs

        self.model = modelNet.Bi_LSTM()
        self.loss_function = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)


if __name__ == '__main__':
    lstm = PARAMETER()

    # for idx, word in enumerate(lstm.word_to_embedding):
    #     if idx is 3:
    #         break
    #     print(lstm.word_to_embedding[word])
    #     print('len: ',len(lstm.word_to_embedding[word]))

    # 显示训练前的结果
    train.beforeTrain(parameter=lstm)

    # 开始训练模型
    train.beginTrain(parameter=lstm)

    # 显示训练后在验证集上的结果
    test.verifyAfterTrainning(parameter=lstm)

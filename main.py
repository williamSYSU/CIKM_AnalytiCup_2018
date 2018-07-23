import re

import torch.nn as nn
import torch.optim as optim

import load_data
import modelNet
import preprocess
import test
import train
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class PARAMETER():
    def __init__(self):
        # 从本地加载word embedding词典
        # self.word_to_embedding = load_data.loadEmbedVocab('preprocess/word_embedding.txt')
        self.word_to_embedding = preprocess.get_final_word_to_embedding()

        # 加载数据对集合
        self.train_pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt')
        self.english_train_pairs = load_data.loadDataPairs('data/cikm_english_train_20180516.txt')
        self.test_pairs = load_data.loadDataPairs('data/cikm_test_a_20180516.txt')

        # 是否加入英语原语训练集
        if modelNet.ENGLISH_TAG is 1:
            self.train_pairs = self.train_pairs + self.english_train_pairs

        # 划分训练集和验证集
        self.train_pairs, self.verify_pairs = preprocess.load_training_and_verify_pairs(pairs=self.train_pairs)

        self.model = modelNet.MatchSRNN().cuda()
        self.loss_function = nn.BCELoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=modelNet.LEARNING_RATE)


if __name__ == '__main__':
    lstm = PARAMETER()


    # 显示训练前的结果
    train.beforeTrain(parameter=lstm)

    # 开始训练模型
    train.beginTrain(parameter=lstm)

    # 显示训练后在验证集上的结果
    test.verifyAfterTrainning(parameter=lstm)

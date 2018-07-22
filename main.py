import re

import torch.nn as nn
import torch.optim as optim

import load_data
import modelNet
import preprocess
import test
import train


class PARAMETER():
    def __init__(self):
        # 从本地加载word embedding词典
        # self.word_to_embedding = load_data.loadEmbedVocab('preprocess/word_embedding.txt')
        self.word_to_embedding = preprocess.get_final_word_to_embedding()

        # 加载数据对集合
        self.train_pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt')
        self.english_train_pairs = load_data.loadDataPairs('data/cikm_english_train_20180516.txt')
        self.test_pairs = load_data.loadDataPairs('data/cikm_test_a_20180516.txt')

        # 划分训练集和验证集
        self.train_pairs, self.verify_pairs = preprocess.load_training_and_verify_pairs(pairs=self.train_pairs)

        if modelNet.ENGLISH_TAG is 1:
            self.train_pairs = self.train_pairs + self.english_train_pairs

        self.model = modelNet.Bi_LSTM()
        self.loss_function = nn.BCELoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=modelNet.LEARNING_RATE)


def locate_missing_word():
    word_vocab = load_data.loadVocab('preprocess/word_vocab.txt')
    print('len of vocab: ', len(word_vocab))
    print('len of dict: ', len(lstm.word_to_embedding))
    print('missing: ', len(word_vocab) - len(lstm.word_to_embedding))

    missing_char_vocab = []
    missing_digit_vocab = []
    with open('preprocess/all_missing_word.txt', encoding='utf-8', mode='wt') as file:
        for word in word_vocab:
            if word not in lstm.word_to_embedding.keys():
                file.write(word + '\n')
                res = re.search(r'\d', word)
                if bool(res):
                    missing_digit_vocab.append(word)
                else:
                    missing_char_vocab.append(word)

    with open('preprocess/missing_digit_word.txt', encoding='utf-8', mode='wt') as file:
        print('len of missing digit: ', len(missing_digit_vocab))
        for word in missing_digit_vocab:
            file.write(word + '\n')

    with open('preprocess/missing_char_word.txt', encoding='utf-8', mode='wt') as file:
        print('len of missing char: ', len(missing_char_vocab))
        for word in missing_char_vocab:
            file.write(word + '\n')


if __name__ == '__main__':
    lstm = PARAMETER()

    # preprocess.get_some_english_train_pairs(english_pairs=lstm.english_train_pairs, spanish_pairs=lstm.train_pairs)

    # 显示训练前的结果
    train.beforeTrain(parameter=lstm)

    # 开始训练模型
    train.beginTrain(parameter=lstm)

    # 显示训练后在验证集上的结果
    test.verifyAfterTrainning(parameter=lstm)

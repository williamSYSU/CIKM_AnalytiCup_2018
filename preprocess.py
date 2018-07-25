import random
import re
from fuzzywuzzy import fuzz

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import load_data
import modelNet

LABEL_INDEX = 2


class CIMKDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CIMKDatasetReader:
    def __init__(self):
        print('=' * 100)
        print('> Preparing dataset...')

        # 从本地加载词向量字典
        self.word_to_embedding = get_final_word_to_embedding()

        # 加载数据对集合
        train_pairs = load_data.loadDataPairs('data/cikm_spanish_train_20180516.txt', loc1=0, loc2=2)
        english_train_pairs = load_data.loadDataPairs('data/cikm_english_train_20180516.txt', loc1=1, loc2=3)
        test_pairs = load_data.loadDataPairs('data/cikm_test_a_20180516.txt', loc1=0, loc2=1)

        # 是否加入英语原语训练集
        if modelNet.ENGLISH_TAG is 1:
            train_pairs = train_pairs + english_train_pairs

        # 划分训练集和验证集
        train_pairs, verify_pairs = load_training_and_verify_pairs(pairs=train_pairs)

        self.train_data = CIMKDataset(CIMKDatasetReader.__read_data__(train_pairs, self.word_to_embedding))
        self.verify_data = CIMKDataset(CIMKDatasetReader.__read_data__(verify_pairs, self.word_to_embedding))
        self.test_data = CIMKDataset(CIMKDatasetReader.__read_data__(test_pairs, self.word_to_embedding, isTest=1))

    @staticmethod
    def __read_data__(pairs, embedding, isTest=0):
        all_data = []
        for idx, pair in enumerate(pairs):
            items = tensorsFromPair(pair, embedding, isTest)
            data = {
                'input1': items[0],
                'input2': items[1]
            }
            # 当数据不是测试集要加入label
            if isTest is 0:
                data['label'] = items[2]
            all_data.append(data)
        return all_data


# 把句子转为tensor
def tensorFromSentence(sentence, embedding):
    tensors = []
    for word in sentence.split():
        if word != '':
            tensors.append(embedding[word])

    # 填充句子
    for i in range(len(sentence.split()), modelNet.MAX_SQE_LEN):
        tensors.append(modelNet.END_OF_SEN)

    # 拼接句子
    final_tensors = tensors[0].view(1, -1)
    for i in range(1, len(tensors)):
        final_tensors = torch.cat([final_tensors, tensors[i].view(1, -1)], dim=0)
    return final_tensors


# 将数据对转化为相应的tensor
def tensorsFromPair(pair, embedding, isTest=0):
    input1_tensor = tensorFromSentence(pair[0], embedding)
    input2_tensor = tensorFromSentence(pair[1], embedding)

    if isTest is 0:
        label = torch.tensor(float(pair[LABEL_INDEX]))
        return input1_tensor, input2_tensor, label
    else:  # 测试数据中没有Label
        return input1_tensor, input2_tensor


# 将训练数据一分为二，按比例划分训练集和验证集
def load_training_and_verify_pairs(pairs):
    pairs_true = []
    pairs_false = []

    # 分开正负样本
    for pair in pairs:
        if pair[LABEL_INDEX] == '1':
            pairs_true.append(pair)
        else:
            pairs_false.append(pair)
    random.shuffle(pairs_true)
    random.shuffle(pairs_false)
    print('>>> total pairs: {}, true pairs: {}, false pairs: {}'.format(
        len(pairs_true) + len(pairs_false), len(pairs_true), len(pairs_false)))

    # 按比例取训练集和验证集
    training_pairs = pairs_true[0:int(len(pairs_true) * modelNet.TRAINTEST_RATE)] + \
                     pairs_false[0:int(len(pairs_false) * modelNet.TRAINTEST_RATE)]
    verify_pairs = pairs_true[int(len(pairs_true) * modelNet.TRAINTEST_RATE):] + \
                   pairs_false[int(len(pairs_false) * modelNet.TRAINTEST_RATE):]

    print('>>> training pairs: {}, verify pairs: {}'.format(
        len(training_pairs), len(verify_pairs)))

    random.shuffle(training_pairs)
    random.shuffle(verify_pairs)
    return training_pairs, verify_pairs


# 部分抽取英语原语训练数据
def get_some_english_train_pairs(english_pairs, spanish_pairs):
    english_pairs_true = []
    english_pairs_false = []
    spanish_pairs_true = []
    spanish_pairs_false = []

    # 统计西班牙原语正负样本数量
    for pair in spanish_pairs:
        if pair[4] == '1':
            spanish_pairs_true.append(pair)
        else:
            spanish_pairs_false.append(pair)

    # 统计英语原语的正负样本数量
    for pair in english_pairs:
        if pair[4] == '1':
            english_pairs_true.append(pair)
        else:
            english_pairs_false.append(pair)

    # 打乱数据对
    random.shuffle(spanish_pairs_true)
    random.shuffle(spanish_pairs_false)
    random.shuffle(english_pairs_true)
    random.shuffle(english_pairs_false)

    # 按英语原语与西班牙原语的比例取英语原语正负样本
    train_pairs = english_pairs_true[0:int(len(spanish_pairs_true) * modelNet.ENGLISH_SPANISH_RATE)] + \
                  english_pairs_false[0:int(len(spanish_pairs_false) * modelNet.ENGLISH_SPANISH_RATE)]
    return train_pairs


# 找到所有缺失的词
def locate_missing_word(parameter):
    word_vocab = load_data.loadVocab('preprocess/word_vocab.txt')
    print('len of vocab: ', len(word_vocab))
    print('len of dict: ', len(parameter.word_to_embedding))
    print('missing: ', len(word_vocab) - len(parameter.word_to_embedding))

    missing_char_vocab = []
    missing_digit_vocab = []
    with open('preprocess/all_missing_word.txt', encoding='utf-8', mode='wt') as file:
        for word in word_vocab:
            if word not in parameter.word_to_embedding.keys():
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


# 将缺失的含有数字的词统一Embedding
def embedding_missing_digit_word():
    # 从本地加载含有数字的缺失词
    missing_digit_vocab = load_data.loadVocab('preprocess/missing_digit_word.txt')
    missing_digit_embedding = {}
    word_to_embedding = load_data.loadEmbedVocab('preprocess/word_embedding.txt')

    # 对该词典独立embedding
    embedding = nn.Embedding(len(missing_digit_vocab), modelNet.EMBEDDING_SIZE)
    for word in missing_digit_vocab:
        idx = missing_digit_vocab[word]
        idx = torch.tensor(idx, dtype=torch.long)
        embed = [float('%0.6f' % num.item()) for num in embedding(idx)]  # 改变精度
        missing_digit_embedding[word] = embed

    # 保存Embedding词典
    load_data.saveEmbedVocab(missing_digit_embedding, 'preprocess/missing_digit_embedding.txt')


# 从词库加载所有词转成字典，并储存到本地，字典形式：{'word':index}
def load_and_save_all_char_vocab():
    all_char_vocab = {}
    with open('data/wiki.es.vec', encoding='utf-8') as all_data:
        for idx, line in enumerate(all_data):
            if idx is 0:
                continue  # 跳过首行
            items = line.strip().split()
            all_char_vocab[load_data.normalizeString(items[0])] = idx

        load_data.saveVocab(all_char_vocab, 'preprocess/database_all_word_vocab.txt')
        print('Save All vocab completed.')


# 计算缺失的全是字母的词与词库中最近的词
def embedding_missing_char_word():
    missing_char_vocab = load_data.loadVocab('preprocess/missing_char_word.txt')
    all_char_vocab = load_data.loadVocab('preprocess/database_all_word_vocab.txt')
    all_char_list = []
    for word in all_char_vocab:
        all_char_list.append(word)

    # 计算缺失词最相近的词
    with open('preprocess/sim_word.txt', encoding='utf-8', mode='wt') as file:
        for i, word in enumerate(missing_char_vocab):
            sim_data = []
            DIFF_WEIGHT = 2  # 长度对相似度的影响程度
            for idx, item in enumerate(all_char_list):
                value = fuzz.partial_ratio(word, item)
                len_diff = abs(len(word) - len(item))
                sim_value = value - DIFF_WEIGHT * len_diff  # 长度差距越大，相似值越小
                sim_data.append(sim_value)
            sim_idx = sim_data.index(max(sim_data))
            max_value = max(sim_data)
            sim_word = all_char_list[sim_idx]
            print('word: {}, sim word: {}, sim_idx: {}, sim value: {}'.format(
                word, sim_word, sim_idx, max_value))
            file.write(word + ' ' + sim_word + '\n')


# 根据找到的相似词取词库中的词向量
def get_sim_word_embedding():
    sim_word = {}  # 相似词的对应行号词典
    sim_word_embedding = {}  # 相似词的对应词向量词典

    # 读取缺失词的及其相似词在词库中的行号
    with open('preprocess/sim_word.txt', encoding='utf-8', mode='r') as file:
        for line in file:
            items = line.strip().split()
            sim_word[items[0]] = items[1]

    # 从词库中提取对应词向量
    for idx, word in enumerate(sim_word):
        print('{} of {}  ===  current word: {}'.format(idx, len(sim_word), word))
        with open('data/wiki.es.vec', encoding='utf-8') as data:
            for line in data:
                items = line.strip().split()
                if sim_word[word] == load_data.normalizeString(items[0]):
                    print('sim word: {}'.format(items[0]))
                    sim_word_embedding[word] = items[1:]
                    break

    load_data.saveEmbedVocab(sim_word_embedding, 'preprocess/sim_word_embedding.txt')


# 整合所有的词向量，包含缺失词的词向量
def get_final_word_to_embedding():
    word_to_embedding = load_data.loadEmbedVocab('preprocess/word_embedding.txt')

    # 加入缺失词的相似词词向量
    sim_word_embedding = load_data.loadEmbedVocab('preprocess/sim_word_embedding.txt')
    missing_digit_embedding = load_data.loadEmbedVocab('preprocess/missing_digit_embedding.txt')

    for word in sim_word_embedding:
        word_to_embedding[word] = sim_word_embedding[word]
    for word in missing_digit_embedding:
        word_to_embedding[word] = missing_digit_embedding[word]

    return word_to_embedding


# 统计最长的句子词数
def count_max_len_of_sentence(pairs):
    max_len = 0
    for pair in pairs:
        max_len = max(max_len, len(pair[0].split()))
        max_len = max(max_len, len(pair[1].split()))
    return max_len

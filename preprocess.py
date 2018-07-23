import random

import torch
import torch.nn as nn

import load_data
import modelNet
from fuzzywuzzy import fuzz

device = torch.device("cuda:0")

# 把句子转为tensor
def tensorFromSentence(sentence, embedding):
    tensors = []
    for word in sentence.split(' '):
        # print('word: ', word)
        if word in embedding.keys():
            tensors.append(embedding[word]).cuda()
        else:
            tensors.append(torch.rand(1, modelNet.EMBEDDING_SIZE).cuda())
    tensor = tensors[0].view(1, -1)
    for i in range(1, len(tensors)):
        tensor = torch.cat([tensor, tensors[i].view(1, -1)], dim=0).cuda()
    return tensor.cuda()


# 将数据对转化为相应的tensor
def tensorsFromPair(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding).cuda()
    input2_tensor = tensorFromSentence(pair[2], embedding).cuda()
    # input_tensor=torch.cat((input1_tensor,input2_tensor),dim=0)
    label = pair[4]
    return input1_tensor, input2_tensor, label


# 从数据对中选出对应西班牙语的数据对
def tensorsFromPair_test(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding).cuda()
    input2_tensor = tensorFromSentence(pair[1], embedding).cuda()
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    return input1_tensor, input2_tensor


def tensorsFromPair_verify(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding).cuda()
    input2_tensor = tensorFromSentence(pair[2], embedding).cuda()
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    label = pair[4]
    return input1_tensor, input2_tensor, label


# 将训练数据一分为二，按比例划分训练集和验证集
def load_training_and_verify_pairs(pairs):
    pairs_true = []
    pairs_false = []
    for pair in pairs:
        if pair[4] == '1':
            pairs_true.append(pair)
        else:
            pairs_false.append(pair)
    random.shuffle(pairs_true)
    random.shuffle(pairs_false)
    training_pairs = pairs_true[0:int(len(pairs_true) * modelNet.TRAINTEST_RATE)] + \
                     pairs_false[0:int(len(pairs_false) * modelNet.TRAINTEST_RATE)]
    test_pairs = pairs_true[int(len(pairs_true) * modelNet.TRAINTEST_RATE):] + \
                 pairs_false[int(len(pairs_false) * modelNet.TRAINTEST_RATE):]
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs, test_pairs


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
    train_pairs = english_pairs_true[0:len(spanish_pairs_true) * modelNet.ENGLISH_SPANISH_RATE] + \
                  english_pairs_false[0:len(spanish_pairs_false) * modelNet.ENGLISH_SPANISH_RATE]
    return train_pairs


# TODO: 取部分英语原语数据作为验证集
def get_some_english_verify_pairs(english_pairs):
    pass


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


# 计算缺失的全是字母的词与词库中最近的词
def embedding_missing_char_word():
    missing_char_vocab = load_data.loadVocab('preprocess/missing_char_word.txt')
    all_char_vocab = load_data.loadVocab('preprocess/database_all_word_vocab.txt')
    # all_char_vocab = {}
    all_char_list = []
    for word in all_char_vocab:
        all_char_list.append(word)

    # 从词库中加载所有词
    # with open('data/wiki.es.vec', encoding='utf-8') as all_data:
    #     for idx, line in enumerate(all_data):
    #         if idx is 0:
    #             continue    # 跳过首行
    #         items = line.strip().split()
    #         all_char_vocab[load_data.normalizeString(items[0])] = idx
    #
    #     load_data.saveVocab(all_char_vocab,'preprocess/database_all_word_vocab.txt')
    #     print('Save All vocab completed.')

    # 计算缺失词最相近的词
    # with open('preprocess/sim_word.txt', encoding='utf-8', mode='wt') as file:
    #     for i, word in enumerate(missing_char_vocab):
    #         # print('current word: %s' % word)
    #         sim_data = []
    #         DIFF_WEIGHT = 2  # 长度对相似度的影响程度
    #         for idx, item in enumerate(all_char_list):
    #             value = fuzz.partial_ratio(word, item)
    #             len_diff = abs(len(word) - len(item))
    #             sim_value = value - DIFF_WEIGHT * len_diff  # 长度差距越大，相似值越小
    #             sim_data.append(sim_value)
    #         sim_idx = sim_data.index(max(sim_data))
    #         max_value = max(sim_data)
    #         sim_word = all_char_list[sim_idx]
    #         print('word: {}, sim word: {}, sim_idx: {}, sim value: {}'.format(
    #             word, sim_word, sim_idx, max_value))
    #         file.write(word + ' ' + sim_word + '\n')

    # res1 = fuzz.ratio(w1, w2)
    # res2 = fuzz.partial_ratio(w1, w2)
    # res3 = fuzz.token_sort_ratio(w1, w2)
    # res4 = fuzz.token_set_ratio(w1, w2)


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


# TODO: 合并所有的词向量，返回所有出现词的词向量
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

# load_data.py
# 功能
# 1、从词向量库中读取数据，根据训练集和测试集中的词，取该词的词向量
# 2、保存全部词的词向量

import unicodedata
import re

EMBEDDING_SIZE = 300
spanish_train_vocab = {}
english_train_vocab = {}
spanish_test_vocab = {}


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 去除标点符号，大写转小写
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-z0-9\t]+", r" ", s)
    return s


# 将词存入词典
def word2idx(sentence, vocab):
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)


# 保存词典到本地
def saveVocab(vocab, filename, mode='wt'):
    with open(filename, mode=mode) as file:
        for key in vocab:
            file.write(key + '\n')


# 从本地加载词典
def loadVocab(filename, mode='r'):
    with open(filename, mode=mode, encoding='utf-8') as file:
        vocab = {}
        for idx, line in enumerate(file):
            vocab[line.strip()] = idx
        return vocab


# 加载数据，并构建词典
def loadData2Vocab(filename, saveName, loc1, loc2, vocab):
    with open(filename, encoding='utf-8') as data:
        print('Load data completed.')
        for line in data:
            items = line.strip().split('\t')
            word2idx(normalizeString(unicodeToAscii(items[loc1])).split(), vocab)
            word2idx(normalizeString(unicodeToAscii(items[loc2])).split(), vocab)

        # save vocab
        saveVocab(vocab, saveName)
        print('Save vocab completed!')

# 保存词向量的词典
def saveEembedVocab(vocab, filename, mode='wt'):
    with open(filename, mode=mode) as file:
        for word in vocab:
            file.write(word + ' ')
            for embed in vocab[word]:
                file.write(embed + ' ')
            file.write('\n')

# 读取训练数据（西班牙源语）
filename = 'data/cikm_spanish_train_20180516.txt'
saveName = 'preprocess/spanish_train_vocab.txt'
# loadData2Vocab(filename, saveName, loc1=0, loc2=2, vocab=spanish_train_vocab)


# 读取训练数据（英语源语）
filename = 'data/cikm_english_train_20180516.txt'
saveName = 'preprocess/english_train_vocab.txt'
# loadData2Vocab(filename, saveName, loc1=1, loc2=3, vocab=english_train_vocab)

# 读取测试数据
filename = 'data/cikm_test_a_20180516.txt'
saveName = 'preprocess/spanish_test_vocab.txt'
# loadData2Vocab(filename, saveName, loc1=0, loc2=1, vocab=spanish_test_vocab)

# 读取词向量库（西班牙语）
spanish_train_vocab = loadVocab('preprocess/spanish_train_vocab.txt')
spanish_test_vocab = loadVocab('preprocess/spanish_test_vocab.txt')
english_train_vocab = loadVocab('preprocess/english_train_vocab.txt')

word_vocab = loadVocab('preprocess/word_vocab.txt')

word_to_embedding = {}

# 合并所有vocab
# for word in spanish_train_vocab:
#     if word not in word_vocab:
#         word_vocab[word]=len(word_vocab)
# for word in spanish_test_vocab:
#     if word not in word_vocab:
#         word_vocab[word]=len(word_vocab)
# for word in english_train_vocab:
#     if word not in word_vocab:
#         word_vocab[word]=len(word_vocab)
# saveVocab(word_vocab,'preprocess/word_vocab.txt')
# 合并end

# 保存每个词的对应词向量
for word in word_vocab:
    print('current word: ', word)
    with open('data/wiki.es.vec', encoding='utf-8') as embedding:
        for line in embedding:
            items = line.strip().split()
            if word == normalizeString(items[0]):
                word_to_embedding[word] = items[1:-1]
                print('word: ', items[0])
                # print('embedding: ', items[1:-1])
                break

saveEembedVocab(word_to_embedding, 'preprocess/word_embedding.txt')

print(normalizeString('¿Cómo cambio mi dirección de email?'))
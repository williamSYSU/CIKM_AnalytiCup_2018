import torch
import random
import modelNet

from fuzzywuzzy import fuzz

# 把句子转为tensor
def tensorFromSentence(sentence, embedding):
    tensors = []
    for word in sentence.split(' '):
        # print('word: ', word)
        if word in embedding.keys():
            tensors.append(embedding[word])
        else:
            tensors.append(torch.rand(1, modelNet.EMBEDDING_SIZE))
    tensor = tensors[0].view(1, -1)
    for i in range(1, len(tensors)):
        tensor = torch.cat([tensor, tensors[i].view(1, -1)], dim=0)
    return tensor


# 将数据对转化为相应的tensor
def tensorsFromPair(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding)
    input2_tensor = tensorFromSentence(pair[2], embedding)
    # input_tensor=torch.cat((input1_tensor,input2_tensor),dim=0)
    label = pair[4]
    return (input1_tensor, input2_tensor, label)


# 从数据对中选出对应西班牙语的数据对
def tensorsFromPair_test(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding)
    input2_tensor = tensorFromSentence(pair[1], embedding)
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    return (input1_tensor, input2_tensor)


def tensorsFromPair_verify(pair, embedding):
    input1_tensor = tensorFromSentence(pair[0], embedding)
    input2_tensor = tensorFromSentence(pair[2], embedding)
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    label = pair[4]
    return (input1_tensor, input2_tensor, label)


# 将训练数据一分为二，80%做训练，20%做验证
def load_training_and_verify_pairs(pairs):
    pairs_true = []
    pairs_false = []
    TRAINING_RATE = 0.8
    for pair in pairs:
        if pair[4] == '1':
            pairs_true.append(pair)
        else:
            pairs_false.append(pair)
    random.shuffle(pairs_true)
    random.shuffle(pairs_false)
    training_pairs = pairs_true[0:int(len(pairs_true) * TRAINING_RATE)] + pairs_false[
                                                                          0:int(len(pairs_true) * TRAINING_RATE)]
    test_pairs = pairs_true[int(len(pairs_true) * TRAINING_RATE):] + pairs_false[int(len(pairs_true) * TRAINING_RATE):]
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return (training_pairs, test_pairs)

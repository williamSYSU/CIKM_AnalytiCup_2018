import torch
import random


# 把句子转为tensor
def tensorFromSentence(sentence, embedding):
    tensors = []
    for word in sentence.split(' '):
        if word in embedding.keys():
            tensors.append(embedding[word])
        else:
            tensors.append(torch.rand(1, 300))
    tensor = tensors[0].view(1, -1)
    for i in range(1, len(tensors)):
        tensor = torch.cat([tensor, tensors[i].view(1, -1)], dim=0)
    return tensor


# 将训练数据一分为二，80%做训练，20%做验证
def training_test(pairs):
    pairs_true = []
    pairs_false = []
    for pair in pairs:
        if pair[4] == '1':
            pairs_true.append(pair)
        else:
            pairs_false.append(pair)
    random.shuffle(pairs_true)
    random.shuffle(pairs_false)
    training_pairs = pairs_true[0:int(len(pairs_true) * 0.8)] + pairs_false[0:int(len(pairs_true) * 0.8)]
    test_pairs = pairs_true[int(len(pairs_true) * 0.8):] + pairs_false[int(len(pairs_true) * 0.8):]
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return (training_pairs, test_pairs)


# 将训练数据对转化为相应的tensor
def tensorsFromPair(pair):
    input1_tensor = tensorFromSentence(pair[0])
    input2_tensor = tensorFromSentence(pair[2])
    # input_tensor=torch.cat((input1_tensor,input2_tensor),dim=0)
    label = pair[4]
    return (input1_tensor, input2_tensor, label)


# 从测试数据对中选出对应西班牙语的数据对
def tensorsFromPair_test(pair):
    input1_tensor = tensorFromSentence(pair[0])
    input2_tensor = tensorFromSentence(pair[1])
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    return (input1_tensor, input2_tensor)


# 从验证数据对中选出对应西班牙语的数据对
def tensorsFromPair_verify(pair):
    input1_tensor = tensorFromSentence(pair[0])
    input2_tensor = tensorFromSentence(pair[2])
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    label = pair[4]
    return (input1_tensor, input1_tensor, label)

import torch


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


# 将数据对转化为相应的tensor
def tensorsFromPair(pair):
    input1_tensor = tensorFromSentence(pair[0])
    input2_tensor = tensorFromSentence(pair[2])
    # input_tensor=torch.cat((input1_tensor,input2_tensor),dim=0)
    label = pair[4]
    return (input1_tensor, input2_tensor, label)


# 从数据对中选出对应西班牙语的数据对
def tensorsFromPair_test(pair):
    input1_tensor = tensorFromSentence(pair[0])
    input2_tensor = tensorFromSentence(pair[1])
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    return (input1_tensor, input2_tensor)


def tensorsFromPair_verify(pair):
    input1_tensor = tensorFromSentence(pair[1])
    input2_tensor = tensorFromSentence(pair[3])
    # input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    label = pair[4]
    return (input1_tensor, input1_tensor, label)

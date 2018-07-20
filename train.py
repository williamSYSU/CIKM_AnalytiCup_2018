import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import preprocess
import main


def beforeTrain(parameter):
    # 训练之前在验证集上的效果
    with torch.no_grad():
        print("before learning:")
        sum_loss = 0
        sum = 0
        for pair in parameter.verify_pairs:
            sum += 1
            verify_pair = [preprocess.tensorsFromPair_verify(pair, parameter.word_to_embedding)]
            tag_scores = parameter.model(verify_pair[0][0], verify_pair[0][1])
            label = verify_pair[0][2]
            if label == '1':
                label = torch.tensor([1], dtype=torch.float)
            else:
                label = torch.tensor([0], dtype=torch.float)
            loss = parameter.loss_function(tag_scores[0].view(-1), label)
            sum_loss += loss
        print("avg_loss:", float(sum_loss / sum))


def beginTrain(parameter):
    # 在训练集上训练
    for epoch in range(100):
        loss = torch.tensor([0], dtype=torch.float)
        for pair in parameter.pairs:
            parameter.zero_grad()
            training_pair = [preprocess.tensorsFromPair(pair, parameter.word_to_embedding)]
            # print (training_pair)
            tag_scores = parameter.model(training_pair[0][0], training_pair[0][1])
            label = training_pair[0][2]
            if label == '1':
                label = torch.tensor([1], dtype=torch.float)
            else:
                label = torch.tensor([0], dtype=torch.float)
            loss = parameter.loss_function(tag_scores[0].view(-1), label)
            loss.backward()
            parameter.optimizer.step()
        print(loss.item())

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import preprocess
import main


def beforeTrain(model, loss_function, optimizer):
    # 训练之前在验证集上的效果
    with torch.no_grad():
        print("before learning:")
        sum_loss = 0
        sum = 0
        for pair in main.verify_pairs:
            sum += 1
            verify_pair = [preprocess.tensorsFromPair_verify(pair, main.word_to_embedding)]
            tag_scores = model(verify_pair[0][0], verify_pair[0][1])
            label = verify_pair[0][2]
            if label == '1':
                label = torch.tensor([1], dtype=torch.float)
            else:
                label = torch.tensor([0], dtype=torch.float)
            loss = loss_function(tag_scores[0].view(-1), label)
            sum_loss += loss
        print("avg_loss:", float(sum_loss / sum))


def beginTrain(model, loss_function, optimizer):
    # 在训练集上训练
    for epoch in range(100):
        for pair in main.pairs:
            model.zero_grad()
            training_pair = [preprocess.tensorsFromPair(pair, main.word_to_embedding)]
            # print (training_pair)
            tag_scores = model(training_pair[0][0], training_pair[0][1])
            label = training_pair[0][2]
            if label == '1':
                label = torch.tensor([1], dtype=torch.float)
            else:
                label = torch.tensor([0], dtype=torch.float)
            loss = loss_function(tag_scores[0].view(-1), label)
            loss.backward()
            optimizer.step()
        print(loss.item())

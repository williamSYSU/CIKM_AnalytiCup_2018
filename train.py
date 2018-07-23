import torch

import modelNet
import preprocess

data = {}

def beforeTrain(parameter):
    # 训练之前在验证集上的效果
    with torch.no_grad():
        print("before learning:")
        sum_loss = 0
        sum = 0
        for pair in parameter.verify_pairs:
            sum += 1
            verify_pair = preprocess.tensorsFromPair_verify(pair, parameter.word_to_embedding)
            tag_scores = parameter.model(verify_pair[0], verify_pair[1])
            label = verify_pair[2]
            if label == '1':
                label = torch.tensor([1, 0], dtype=torch.float)
            else:
                label = torch.tensor([0, 1], dtype=torch.float)
            loss = parameter.loss_function(tag_scores, label)
            sum_loss += loss
        print("avg_loss:", float(sum_loss / sum))


# TODO: 设置batch，每次Epoch后shuffle数据对，让每个batch不一样
def beginTrain(parameter):
    # 在训练集上训练
    data['epoch_num'] = modelNet.EPOCH_NUM
    print('begin learning:')

    pos = 0
    for epoch in range(modelNet.EPOCH_NUM):
        loss = torch.tensor([0], dtype=torch.float)

        for idx, pair in enumerate(parameter.train_pairs):
            parameter.model.zero_grad()
            training_pair = preprocess.tensorsFromPair(pair, parameter.word_to_embedding)
            tag_scores = parameter.model(training_pair[0], training_pair[1])
            label = training_pair[2]
            # print('out: {}, label: {}'.format(tag_scores[0], label))

            if label == '1':
                label = torch.tensor([0, 1], dtype=torch.float)
                if pos is 0:
                    pos = idx
            else:
                label = torch.tensor([1, 0], dtype=torch.float)

            if idx is pos:
                print(tag_scores, label)
            loss = parameter.loss_function(tag_scores, label)
            loss.backward()
            parameter.optimizer.step()
        data['epoch' + str(epoch) + 'loss'] = loss.item()
        print('epoch: ', epoch, ', loss: ', loss.item())

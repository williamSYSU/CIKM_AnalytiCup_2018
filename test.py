import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import main
import preprocess
import modelNet
import train


# 训练之后在验证集上的效果
def verifyAfterTrainning(parameter):
    with torch.no_grad():
        print("after learning:")
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

        print("final_avg_loss:", float(sum_loss / sum))

        for pair in parameter.test_pairs:
            test_pair = [preprocess.tensorsFromPair_test(pair, parameter.word_to_embedding)]
            tag_scores = parameter.model(test_pair[0][0], test_pair[0][1])
            with open("test_result.txt", 'a') as f:
                f.write(str(tag_scores[0].item()) + "\n")


# 训练之后在验证集上的效果
def verifyAfterTrainning(parameter):
    with torch.no_grad():
        print("after learning:")
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
        avg_loss = float(sum_loss / sum)
        print("avg_loss:", avg_loss)
        for pair in parameter.test_pairs:
            test_pair = [preprocess.tensorsFromPair_test(pair, parameter.word_to_embedding)]
            tag_scores = parameter.model(test_pair[0][0], test_pair[0][1])
            with open("Result/test_result_" + str(avg_loss) + ".txt", 'a') as f:
                f.write(str(tag_scores[0].item()) + "\n")
        path = 'save_model/model_' + str(avg_loss) + '.pkl'
        torch.save(parameter.model, path)
        with open("save_model/data_" + str(avg_loss) + ".txt", 'w') as f:
            f.write("learning rate:" + str(modelNet.LEARNING_RATE) + "\n")
            f.write("dropout rate:" + str(modelNet.DROPOUT_RATE) + "\n")
            f.write("training rate:" + str(modelNet.TRAINING_RATE) + "\n")
            f.write("epoch num:" + str(train.data['epoch_num']) + "\n")
            for i in range(train.data['epoch_num']):
                f.write("epoch " + str(i) + "  loss:" + str(train.data['epoch' + str(i) + 'loss']) + "\n")

            f.write("test loss:" + str(avg_loss) + "\n")

# TODO: save model
# save: learning rate, dropout rate, turns of epoch, loss in each epoch
#       ratio between train and verify, test result log loss (.6f)

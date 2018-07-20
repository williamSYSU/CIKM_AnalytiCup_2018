import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import main
import preprocess


# 训练之后在验证集上的效果
def verifyAfterTrainning(model, loss_function, optimizer):
    with torch.no_grad():
        print("after learning:")
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
        for pair in main.test_pairs:
            test_pair = [preprocess.tensorsFromPair_test(pair, main.word_to_embedding)]
            tag_scores = model(test_pair[0][0], test_pair[0][1])
            with open("test_result.txt", 'a') as f:
                f.write(str(tag_scores[0].item()) + "\n")

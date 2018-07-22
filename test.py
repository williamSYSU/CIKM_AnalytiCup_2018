import torch

import modelNet
import preprocess
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

        avg_loss = float(sum_loss / sum)
        print("final_avg_loss:", avg_loss)
        for pair in parameter.test_pairs:
            test_pair = [preprocess.tensorsFromPair_test(pair, parameter.word_to_embedding)]
            tag_scores = parameter.model(test_pair[0][0], test_pair[0][1])
            with open("Result/test_result_" + str(modelNet.ENGLISH_TAG) + "_" + str(avg_loss) + ".txt", 'a') as f:
                f.write(str(tag_scores[0].item()) + "\n")

        # 保存模型
        path = 'save_model/model_' + str(modelNet.ENGLISH_TAG) + "_" + str(avg_loss) + '.pkl'
        torch.save(parameter.model, path)

        # 保存模型参数以及Loss
        with open("save_model/data_" + str(modelNet.ENGLISH_TAG) + "_" + str(avg_loss) + ".txt", 'w') as f:
            f.write("English tag:" + str(modelNet.ENGLISH_TAG) + "\n")
            f.write("learning rate:" + str(modelNet.LEARNING_RATE) + "\n")
            f.write("dropout rate:" + str(modelNet.DROPOUT_RATE) + "\n")
            f.write("training rate:" + str(modelNet.TRAINTEST_RATE) + "\n")
            f.write("epoch num:" + str(train.data['epoch_num']) + "\n")
            for i in range(train.data['epoch_num']):
                f.write("epoch " + str(i) + "  loss:" + str(train.data['epoch' + str(i) + 'loss']) + "\n")

            f.write("test loss:" + str(avg_loss) + "\n")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import modelNet
import preprocess
from preprocess import CIMKDatasetReader

restore_data = {}


class Instructor:
    def __init__(self):
        cimk_dataset = CIMKDatasetReader()

        self.word_to_embedding = cimk_dataset.word_to_embedding
        self.train_data_loader = DataLoader(
            dataset=cimk_dataset.train_data,
            batch_size=modelNet.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=2
        )
        self.verify_data_loader = DataLoader(
            dataset=cimk_dataset.verify_data,
            batch_size=modelNet.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )
        self.test_data_loader = DataLoader(
            dataset=cimk_dataset.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

        self.model = modelNet.Bi_LSTM().to(modelNet.DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=modelNet.LEARNING_RATE)

    # 去除填充字符
    @staticmethod
    def remove_end_of_sen(input):
        input = input.view(-1, modelNet.EMBEDDING_SIZE)
        for idx, item in enumerate(input):
            if torch.equal(item.unsqueeze(0), modelNet.END_OF_SEN):
                indices = torch.tensor([i for i in range(idx)], dtype=torch.long)
                # print('after remove: ', torch.index_select(input, 0, indices))
                return torch.index_select(input, 1, indices)

    # 训练之前在验证集上的效果
    def beforeTrain(self):
        self.model.eval()  # 切换模型至验证模式
        with torch.no_grad():
            print('=' * 100)
            print("Before learning:")
            sum_loss = 0
            for idx, sample_batch in enumerate(self.verify_data_loader):
                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2).index_select(1, torch.tensor([1]).to(modelNet.DEVICE)).view(-1)

                loss = self.criterion(outputs, label)
                sum_loss += loss
            print("Average loss:", float(sum_loss / len(self.verify_data_loader)))

    # 在训练集上训练
    def beginTrain(self):
        restore_data['epoch_num'] = modelNet.EPOCH_NUM
        print('=' * 100)
        print('Begin learning......')

        pos = -1
        for epoch in range(modelNet.EPOCH_NUM):
            print('>' * 100)
            loss = torch.tensor([0], dtype=torch.float)
            for idx, sample_batch in enumerate(self.train_data_loader):
                self.model.train()  # 切换模型至训练模式
                self.optimizer.zero_grad()  # 清空积累的梯度
                # self.model.zero_grad()

                # 取训练数据和标签
                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2).index_select(1, torch.tensor([1])).view(-1)
                if idx is 5:
                    print('output: {}, label: {}'.format(outputs, label))

                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
            print('epoch {} of {} loss: {}'.format(epoch + 1, modelNet.EPOCH_NUM, loss.item()))
            restore_data['epoch' + str(epoch) + 'loss'] = loss.item()

    # 在测试集上测试
    def verify_and_test_model(self):
        self.model.eval()  # 设置模型为验证模式
        print('=' * 100)
        print('Begin verify......')

        sum_loss = 0
        with torch.no_grad():
            for idx, sample_batch in enumerate(self.verify_data_loader):
                self.model.zero_grad()

                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2).index_select(1, torch.tensor([1])).view(-1)

                loss = self.criterion(outputs, label)
                sum_loss += loss

            final_avg_loss = float(sum_loss / len(self.verify_data_loader))
            print('Final average loss: ', final_avg_loss)

        # 在测试集上测试并保存结果
        print('=' * 100)
        print('Begin test and save model......')
        for idx, sample_batch in enumerate(self.test_data_loader):

            input1 = sample_batch['input1'].to(modelNet.DEVICE)
            input2 = sample_batch['input2'].to(modelNet.DEVICE)

            outputs = self.model(input1, input2).index_select(1, torch.tensor([1])).view(-1)

            with open("Result/test_result_" + str(modelNet.ENGLISH_TAG) + "_" + str(final_avg_loss) + ".txt", 'a') as f:
                f.write(str(outputs[0].item()) + "\n")

            # 保存模型参数以及Loss
            with open("save_model/data_" + str(modelNet.ENGLISH_TAG) + "_" + str(final_avg_loss) + ".txt", 'w') as f:
                f.write("English tag:" + str(modelNet.ENGLISH_TAG) + "\n")
                f.write("learning rate:" + str(modelNet.LEARNING_RATE) + "\n")
                f.write("dropout rate:" + str(modelNet.DROPOUT_RATE) + "\n")
                f.write("training rate:" + str(modelNet.TRAINTEST_RATE) + "\n")
                f.write("epoch num:" + str(restore_data['epoch_num']) + "\n")
                for i in range(restore_data['epoch_num']):
                    f.write("epoch " + str(i) + "  loss:" + str(restore_data['epoch' + str(i) + 'loss']) + "\n")

                f.write("test loss:" + str(final_avg_loss) + "\n")

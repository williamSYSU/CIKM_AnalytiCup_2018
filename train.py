import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import modelNet
import preprocess
from preprocess import CIMKDatasetReader

restore_loss = []


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 输出模型参数
        print('=' * 100)
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {}: {}'.format(arg, getattr(opt, arg)))

        # 加载数据集，设置batch
        cimk_dataset = CIMKDatasetReader()

        self.word_to_embedding = cimk_dataset.word_to_embedding
        self.train_data_loader = DataLoader(
            dataset=cimk_dataset.train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        self.verify_data_loader = DataLoader(
            dataset=cimk_dataset.verify_data,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        self.test_data_loader = DataLoader(
            dataset=cimk_dataset.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        self.model = opt.model_class().to(opt.device)
        self.criterion = nn.BCELoss()
        self.optimizer = opt.optimizer(self.model.parameters(), lr=modelNet.LEARNING_RATE)

    # 去除填充字符
    @staticmethod
    def remove_end_of_sen(input):
        input = input.view(-1, modelNet.EMBEDDING_SIZE)
        for idx, item in enumerate(input):
            if torch.equal(item.unsqueeze(0), modelNet.END_OF_SEN):
                input = input[:idx, :]
                return input

    # 训练之前在验证集上的效果
    def beforeTrain(self):
        self.model.eval()  # 切换模型至验证模式
        with torch.no_grad():
            print('=' * 100)
            print("> Before learning:")
            sum_loss = 0
            for idx, sample_batch in enumerate(self.verify_data_loader):
                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2)[:, 1].view(-1)

                loss = self.criterion(outputs, label)
                sum_loss += loss
            print(">>> Average loss:", float(sum_loss / len(self.verify_data_loader)))

    # 在训练集上训练
    def beginTrain(self):
        print('=' * 100)
        print('> Begin learning......')

        pos = -1
        for epoch in range(modelNet.EPOCH_NUM):
            print('>' * 100)
            loss = torch.tensor([0], dtype=torch.float)
            for idx, sample_batch in enumerate(self.train_data_loader):
                self.model.train()  # 切换模型至训练模式
                # self.optimizer.zero_grad()
                self.model.zero_grad()  # 清空积累的梯度

                # 取训练数据和标签
                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2)[:, 1].view(-1)
                if idx is 5:
                    print('output: {}\nlabel: {}'.format(outputs, label))

                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
            print('> epoch {} of {} loss: {}'.format(epoch + 1, modelNet.EPOCH_NUM, loss.item()))
            restore_loss.append(loss.item())

    # 在验证集上验证
    def verifyModel(self):
        self.model.eval()  # 设置模型为验证模式
        print('=' * 100)
        print('> Begin verify......')

        sum_loss = 0
        with torch.no_grad():
            for idx, sample_batch in enumerate(self.verify_data_loader):
                self.model.zero_grad()

                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                outputs = self.model(input1, input2)[:, 1].view(-1)

                loss = self.criterion(outputs, label)
                sum_loss += loss

            self.final_avg_loss = float(sum_loss / len(self.verify_data_loader))
            print('>>> Final average loss: ', self.final_avg_loss)

    # 在测试集上测试并保存模型参数和模型
    def testModel(self):
        # 在测试集上测试并保存结果
        print('=' * 100)
        print('> Begin test and save model......')

        for idx, sample_batch in enumerate(self.test_data_loader):
            input1 = sample_batch['input1'].to(modelNet.DEVICE)
            input2 = sample_batch['input2'].to(modelNet.DEVICE)

            outputs = self.model(input1, input2)[:, 1].view(-1)

            save_result_filename = 'Result/test_result_' + str(modelNet.ENGLISH_TAG) + '_' + str(
                self.final_avg_loss) + '.txt'
            with open(save_result_filename, 'a') as f:
                f.write(str(outputs[0].item()) + "\n")

        # 保存模型
        path = 'save_model/model_' + str(modelNet.ENGLISH_TAG) + "_" + str(self.final_avg_loss) + '.pkl'
        torch.save(self.model, path)

        # 保存模型参数以及Loss
        save_para_file_name = 'save_model/data_' + str(modelNet.ENGLISH_TAG) + '_' + str(self.final_avg_loss) + '.txt'
        with open(save_para_file_name, 'w') as f:
            for arg in vars(self.opt):
                f.write('>>> {}: {} \n'.format(arg, getattr(self.opt, arg)))

            f.write('=' * 50)
            for i in range(len(restore_loss)):
                f.write('> epoch {} of {} loss: {} \n'.format(
                    str(i + 1), len(restore_loss), restore_loss[i]))

            f.write(">>>Final verify loss:" + str(self.final_avg_loss) + "\n")

        print('=' * 100)
        print('Finished!')

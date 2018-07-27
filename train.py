import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import modelNet
from preprocess import CIMKDatasetReader

restore_loss = []


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.final_avg_loss = 0.
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
            shuffle=True if opt.shuffle is 1 else False,
            drop_last=True,
            num_workers=4
        )
        self.verify_data_loader = DataLoader(
            dataset=cimk_dataset.verify_data,
            batch_size=opt.batch_size,
            shuffle=True if opt.shuffle is 1 else False,
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
        self.writer = SummaryWriter(log_dir=opt.log_dir)

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

        global_step = 0
        min_train_loss = 0
        min_verify_loss = 0
        for epoch in range(modelNet.EPOCH_NUM):
            print('>' * 100)
            loss = torch.tensor([0], dtype=torch.float)
            for idx, sample_batch in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()  # 切换模型至训练模式
                self.model.zero_grad()  # 清空积累的梯度

                # 取训练数据和标签
                input1 = sample_batch['input1'].to(modelNet.DEVICE)
                input2 = sample_batch['input2'].to(modelNet.DEVICE)
                label = sample_batch['label'].to(modelNet.DEVICE)

                # 计算模型的输出
                outputs = self.model(input1, input2)[:, 1].view(-1)
                # 指定一个batch查看其在每轮的优化效果如何
                if idx is 5:
                    print('output: {}\nlabel: {}'.format(outputs, label))

                # 计算loss，并更新参数
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                # 查看模型在验证集上的验证效果
                if self.opt.if_log_verify is 1 and global_step % self.opt.log_step is 0:
                    verify_loss = self.stepVerify()
                    self.writer.add_scalar('Verify Loss', verify_loss, global_step)
                    min_verify_loss = min(min_verify_loss, verify_loss)

            print('>> epoch {} of {}, -loss: {} -min train loss: {} - min verify loss: {}'.format(
                epoch + 1, modelNet.EPOCH_NUM, loss.item(), min_train_loss, min_verify_loss))

            min_train_loss = min(min_train_loss, loss.item())  # 计算训练过程的最小Loss
            restore_loss.append(loss.item())  # 保存每轮loss
            self.writer.add_scalar('Train_Loss', loss, epoch)  # 画loss曲线

            # "早停"策略，loss低于设定值时，停止训练
            if loss.item() <= self.opt.early_stop:
                print('> !!!Training is forced to stop!!!')
                print('> Current loss: {}, threshold loss: {}'.format(loss.item(), self.opt.early_stop))
                break
        self.writer.close()

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

    # 每个globel_step (log_step)记录在验证集上的loss
    def stepVerify(self):
        self.model.eval()  # 设置模型为验证模式
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

            return float(sum_loss / len(self.verify_data_loader))

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

            f.write('=' * 50 + '\n')
            for i in range(len(restore_loss)):
                f.write('> epoch {} of {} loss: {} \n'.format(
                    str(i + 1), len(restore_loss), restore_loss[i]))

            f.write(">>>Final verify loss:" + str(self.final_avg_loss) + "\n")

        print('=' * 100)
        print('Finished!')

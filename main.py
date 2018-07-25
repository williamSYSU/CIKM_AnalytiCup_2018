import torch.optim as optim
import torch.cuda
import argparse

from train import Instructor
import modelNet
from modelNet import Bi_LSTM
from modelNet import LSTM
from modelNet import MatchSRNN
from modelNet import Text2Image

if __name__ == '__main__':
    # 可调超参
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bi_lstm', type=str, help='bi_lstm, lstm, srnn')
    parser.add_argument('--optim', default='sgd', type=str, help='sgd, asgd, adam, adagrad and etc.')
    parser.add_argument('--hidden_size', default=200, type=int, help='Hidden size in LSTM')
    parser.add_argument('--target_size', default=2, type=int, help='Target size in last layer')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Dropout rate in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate in training')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size in training and verifying')
    parser.add_argument('--epoch_num', default=100, type=int, help='Number of epoch')
    parser.add_argument('--english_tag', default=1, type=int, help='1:join English data; 0:leave English')
    parser.add_argument('--english_spanish_rate', default=1, type=float,
                        help='The ratio of English of Spanish in training')
    parser.add_argument('--train_test_rate', default=0.7, type=float, help='The ratio of train data to verify data')
    parser.add_argument('--device', default=None, type=str, help='Choose device to run')
    parser.add_argument('--max_sqe_len', default=56, type=int, help='Max length of all sentences')
    opt = parser.parse_args()

    # 模型种类
    model_classes = {
        'bi_lstm': Bi_LSTM,
        'lstm': LSTM,
        'srnn': MatchSRNN,
        'text2image': Text2Image
    }

    # 优化器种类
    optimizers = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'sgd': optim.SGD
    }

    # 损失函数种类

    # 初始化其它参数
    opt.model_class = model_classes[opt.model_name]
    opt.optimizer = optimizers[opt.optim]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # 初始化modelNet参数
    modelNet.initParameter(opt)

    instructor = Instructor(opt)

    # 显示训练前的结果
    # instructor.beforeTrain()

    # 开始训练模型
    instructor.beginTrain()

    # 显示训练后在验证集上的结果
    instructor.verifyModel()

    # 运行测试集，保存模型
    instructor.testModel()

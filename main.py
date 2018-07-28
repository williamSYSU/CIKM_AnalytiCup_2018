import torch.optim as optim
import torch.cuda
import argparse

from train import Instructor
import modelNet
from modelNet import Bi_LSTM
from modelNet import LSTM
from modelNet import MatchSRNN
from modelNet import Text2Image
import load_data

if __name__ == '__main__':
    # 可调超参
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='bi_lstm', type=str,
                        help='> model: bi_lstm, lstm, srnn, text2image \n> default: bi_lstm')
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float, help='> default: 0.01')
    parser.add_argument('-e', '--epoch_num', default=100, type=int, help='> default: 100')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='> default:16')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float, help='> default: 0.1')
    parser.add_argument('-c', '--device', default=None, type=str, help='> default: cuda:0/cpu')
    parser.add_argument('-o', '--optim', default='sgd', type=str,
                        help='> optimizer: sgd, asgd, adam, adagrad and etc.\n> default: sgd')
    parser.add_argument('--hidden_size', default=200, type=int, help='> default: 200')
    parser.add_argument('--target_size', default=2, type=int, help='> default: 2')
    parser.add_argument('--english_tag', default=1, type=int, help='> default: 1')
    parser.add_argument('--english_spanish_rate', default=1, type=float, help='> default: 1')
    parser.add_argument('--train_test_rate', default=0.7, type=float, help='> default: 0.7')
    parser.add_argument('--max_sqe_len', default=56, type=int, help='> default: 56')
    parser.add_argument('--conv_channel', default=3, type=int, help='> default: 3')
    parser.add_argument('--conv_target', default=18, type=int, help='> default: 18')
    parser.add_argument('--log_dir', default='log', type=str, help='> Loss curve log name (default: \'log\')')
    parser.add_argument('--log_step', default=5, type=int, help='> Log loss on verify data (default: 5)')
    parser.add_argument('--if_step_verify', default=0, type=int, help='> if verify per log_step (default: 0)')
    parser.add_argument('--early_stop', default=0.001, type=float, help='> Early stop threshold (default: 0.001)')
    parser.add_argument('--shuffle', default=1, type=int,
                        help='> Shuffle train and verify data (default: 1)')
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

    # 初始化其它参数
    opt.model_class = model_classes[opt.model_name]
    opt.optimizer = optimizers[opt.optim]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # 初始化modelNet参数
    modelNet.initParameter(opt)

    instructor = Instructor(opt)

    # 显示训练前的结果
    instructor.beforeTrain()

    # 开始训练模型
    instructor.beginTrain()

    # 显示训练后在验证集上的结果
    instructor.verifyModel()

    # 运行测试集，保存模型
    instructor.testModel()

import re

import torch.nn as nn
import torch.optim as optim

import load_data
import modelNet
import preprocess
from preprocess import CIMKDatasetReader
import test
import train
from train import Instructor


if __name__ == '__main__':
    instructor = Instructor()

    # 显示训练前的结果
    instructor.beforeTrain()

    # 开始训练模型
    instructor.beginTrain()

    # TODO:显示训练后在验证集上的结果
    instructor.verify_and_test_model()

import torch
from torch.nn.functional import normalize
import sys
sys.path.append('../../code/')
from utils.CustomDataset import CustomDataset  # 导入自定义数据集
from torch.utils.data import DataLoader
from torch.autograd import Variable


# 定义常量
dir_train = 'D:/ResearchSpace/task/gestureRecognition/data/train/'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/data/test/'
sample_dir = 'samples.csv'
label_dir = 'labels.csv'
num_classes = 5
batch_size = 32


# 加载自定义训练数据集
CD = CustomDataset(dir_train+label_dir, dir_train+sample_dir, 2400, (400, 6))
Train_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False)

# 向量标准化实验室

for batch, data in enumerate(Train_Dataloader):
    # 将数据从 train_loader 中读出来,一次读取的样本数是32个
    inputs, labels = data
    # 将这些数据转换成Variable类型
    inputs, labels = Variable(inputs), Variable(labels)
    print(inputs.shape[0])
    print(inputs[1, 1, :])
    for i in range(0, inputs.shape[0]):
        inputs[i][:, 3:6] = normalize(inputs[i][:, 3:6], p=2.0, dim=1)
    print(inputs[1][1].shape)
    print(inputs[1, 1, :])
    inputs = normalize(inputs, p=1.0, dim=1)
    # print(inputs[1, 1, :])

'''
-*- coding: utf-8 -*-
@Time : 2021/8/7 20:55
@Author : Small_Volcano
@File : UCI_HAR_CNN.py
'''
import os
import copy
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


train_dir = 'D:/ResearchSpace/task/fes_tensorflow/data/UCI HAR Dataset/UCI HAR Dataset/train/'
dirname = '/Inertial Signals/'

test_dir = 'D:/ResearchSpace/task/fes_tensorflow/data/UCI HAR Dataset/UCI HAR Dataset/test/'
dirname = '/Inertial Signals/'


def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values[0:10000]


def load_dataset(data_rootdir, dirname, group):
    '''将数据文件列表堆叠为三维数组'''
    filename_list = []
    filepath_list = []
    X = []
    # os.walk()方法用于目录遍历，是个高效处理文件目录的遍历器
    for rootdir, dirnames, filenames in os.walk(data_rootdir + dirname):
        for filename in filenames:
            filename_list.append(filename)
            filepath_list.append(os.path.join(rootdir, filename))

    # 遍历根目录下的文件，读取数据为dataframe格式；
    for filepath in filepath_list:
        # print(load_file(filepath).shape)
        X.append(load_file(filepath))

    X = np.dstack(X)  # 沿轴二堆叠成三维数组
    # print(X.shape)
    y = load_file(data_rootdir+'/y_'+group+'.txt')
    # one-hot编码。这个之前的文章中提到了，因为原数据集标签从1开始，而one-hot编码从0开始，所以要先减去1
    y = y-1
    print('{}_X.shape:{},{}_y.shape:{}\n'.format(group, X.shape, group, y.shape))
    return X, y


class HAR(Data.Dataset):
    def __init__(self, filenamedir, dirname, type):
        self.filenamedir = filenamedir
        self.dirname = dirname
        self.type = type

    def HAR_data(self):
        """更改x的维度,加载x和y"""
        data_x_raw, data_y = load_dataset(self.filenamedir, self.dirname, self.type)                                                       # 为什么是1通道 128*9
        data_x = data_x_raw.reshape(
            -1, 1, data_x_raw.shape[1],
            data_x_raw.shape[2])   # (N, C, H, W) (7352, 1, 128, 9)
        #  data_x = np.expand_dims(data_x_raw, 1)
        print(data_x.shape)

        torch_dataset = Data.TensorDataset(
            torch.from_numpy(data_x).double(), torch.from_numpy(data_y))  # 造数据集
        return torch_dataset


data_train = HAR(train_dir, dirname, 'train')  # 这一步是做什么？
har_train_tensor = data_train.HAR_data()  # 创造训练验证数据集
# print(har_train_tensor)
# 测试集数据
data_test = HAR(test_dir, dirname, 'test')
har_test_tensor = data_test.HAR_data()

train_loader = Data.DataLoader(
    dataset=har_train_tensor,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)
# 设置一个测试集加载器
test_loader = Data.DataLoader(
    dataset=har_test_tensor,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)


# 搭建卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=12,  # 输出高度12
                kernel_size=3,  # 卷积核尺寸3*3
                stride=1,
                padding=1),  # (1*128*9)-->(12*128*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (12*128*9)-->(12*64*4)
        )
        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),  # (12*64*4)-->(32*64*4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化后：(32*32*2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),  # (32*32*2)-->(64*32*2)
            nn.ReLU())
        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 2, 128),  # 长方体变平面
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 6))

    # 定义网络的前向传播路径
    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展平多维的卷积图层
        output = self.classifier(x)
        return output


# 输出网络结构
net = Net()  # 创建实例
print(net)


# 定义网络的训练过程函数
def train_model(model,
                traindataloader,
                train_rate,
                criterion,
                optimizer,
                num_epochs=25):
    # train_rate：训练集中训练数量的百分比
    # 计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(
        batch_num * train_rate)  # 前train_rate（80%）的batch进行训练
    # 复制最好模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # 格式化字符串
        print('-' * 10)
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader, 1):  # 取标签和样本
            b_x = Variable(b_x).float()
            b_y = b_y.reshape(-1)
            if step < train_batch_num:  # 前train_rate（80%）的batch进行训练
                model.train()  # 设置模型为训练模式，对Droopou有用
                output = model(b_x)
                #  print(b_x)# 取得模型预测结果
                pre_lab = torch.argmax(output, 1)  # 横向获得最大值位置
                # b_y = Variable(b_y).float()             # 修改BUG
                loss = criterion(output, b_y)  # 每个样本的loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 修改权值
                train_loss += loss.item() * b_x.size(0)
                # print(pre_lab)
                # print(b_y.data)
                train_corrects += torch.sum(pre_lab == b_y.data)  # 训练正确个数
                train_num += b_x.size(0)
            else:
                model.eval()  # 设置模型为验证模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        # 计算训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)  # 一个epoch上的loss
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))  # 此处-1没搞明白
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "UCI_HAR_model")
            torch.save(optimizer.state_dict(), "UCI_HAR_optimizer")
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(
            time_use // 60, time_use % 60))  # 训练用时
    # 使用最好模型的参数
    model.load_state_dict(best_model_wts)
    # 组成数据表格train_process打印
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        })
    return model, train_process


# 对模型进行训练
optimizer = Adam(net.parameters(), lr=0.0003)  # 优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数

net, train_process = train_model(
    net,
    train_loader,
    0.8,  # 使用训练集的20%作为验证
    criterion,
    optimizer,
    num_epochs=10)
# 可视化模型训练过程中
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(
    train_process.epoch,
    train_process.train_loss_all,
    "ro-",
    label="Train loss")
plt.plot(
    train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(
    train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc")
plt.plot(
    train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()
''''''


# 对测试集进行预测,计算模型的泛化能力
def test(model, testdataloader, criterion):
    test_loss_all = []
    test_acc_all = []
    test_loss = 0.0
    test_corrects = 0
    test_num = 0
    for step, (input, target) in enumerate(testdataloader):  # 取标签和样本
        input = Variable(input).float()
        target = target.reshape(-1)
        target = target.long()
        # target = torch.Tensor(target).long()
        model.eval()  # 设置模型为训练模式，对Droopou有用
        output = model(input)
        #  print(b_x)# 取得模型预测结果
        pre_lab = torch.argmax(output, 1)  # 横向获得最大值位置
        loss = criterion(output, target)  # 每个样本的loss
        test_loss += loss.item() * input.size(
            0)  # 此处的b_x.size(0)=batch_size。此处相当于一个batch的loss？计算的是整体训练的loss
        # print(pre_lab)
        # print(input.data)
        test_corrects += torch.sum(pre_lab == target.data)  # 测试正确个数
        test_num += input.size(0)
    test_loss_all.append(test_loss / test_num)
    test_acc_all.append(test_corrects.double().item() / test_num)
    print('Test all Loss: {:.4f} Test Acc: {:.4f}'.format(
        test_loss_all[-1], test_acc_all[-1]))


test = test(net, test_loader, criterion)

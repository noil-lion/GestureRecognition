'''
-*- coding: utf-8 -*-
@Time : 2022/12/12 0:10
@Author : Zihao Wu
@File : Training.py
'''
import sys
import copy
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torch.nn.functional import normalize
sys.path.append('../../../code/')
from CNNLSTM_a import CNNLSTM  # import models
from model.CNN_LSTM_CBAM.CNN_LSTM_CBAM import CNNLSTMCBAM
from model.LSTM_CNN.LSTM_CNN import LSTMCNN
from model.Resnet.ODResnetC import ODResnet
from model.TCN.TCN import TemporalConvNet
from utils.CustomDataset import CustomDataset  # import custom dataset

# #-----------------------------------variables-----------------------------------------------#
num_classes = 5
batch_size = 32
lr = 0.001
momentum = 0.9
num_epoch = 64
in_channels = 9
out_channels = 12
seq_length = 400
hidden_size = 48
num_layers = 2
train_rate = 0.80

label_dir = 'labels.csv'
sample_dir = 'samples.csv'
PATH = 'Best_CD_model.pth'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/test/'
dir_train = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/train/'

# #-----------------------------------init ------------------------------------#
device = torch.device("cuda:0")

# 加载自定义训练数据集
CD = CustomDataset(dir_train+label_dir, dir_train+sample_dir, seq_length*in_channels, (seq_length, in_channels), 'train')
Train_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=True)

# 加载自定义测试数据集
CD = CustomDataset(dir_test+label_dir, dir_test+sample_dir, seq_length*in_channels, (seq_length, in_channels), 'test')
Test_Dataloader = DataLoader(dataset=CD, batch_size=1, shuffle=False)

# 实例化网络
# net = ODResnet(in_channels, class_num=num_classes)
# net = TemporalConvNet(in_channels, num_channels=[25, 25, 25, 25], num_class=num_classes)
# net = LSTMCNN(in_channels, out_channels, hidden_size, num_layers, num_classes, batch_size, seq_length)
net = CNNLSTM(in_channels, out_channels, hidden_size, num_layers, num_classes, batch_size, seq_length)
# net = CNNLSTMCBAM(in_channels, out_channels, hidden_size, num_layers, num_classes, batch_size, seq_length)
if torch.cuda.is_available():
    net = net.cuda()
# print(net)
# 定义损失函数核优化器
criterion = nn.CrossEntropyLoss()  # criterion:标准，准则，原则, CrossEntropyLossL:交叉熵损失
criterion = criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)


# #----------------------------------- Training Phase ------------------------------------#
# Traning function define
def train_loop(dataloader, model, loss_fn, optimizer, train_rate, epochs):
    # training process log
    best_acc = 0.0
    best_epoch = 0
    val_acc_all = []
    val_loss_all = []
    train_acc_all = []
    train_loss_all = []
    best_model_weight = copy.deepcopy(model.state_dict())
    since = time.time()

    # separate train and eval based on train_rate
    batch_num = len(dataloader)
    train_batch_num = round(batch_num * train_rate)
    for epoch in range(epochs):
        # one epoch log
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        print('Epoch {}/{}'.format(epoch, epochs - 1) + '\n' + '-' * 10)
        for batch, data in enumerate(dataloader):
            # get data from train_loader,batch size is 32
            inputs, labels = data
            # Variable transform
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            # Compute prediction and loss
            if batch < train_batch_num:
                model.train()
                output = model(inputs)
                loss = loss_fn(output, labels.long())
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Training phase log
                train_num += inputs.size(0)
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(torch.argmax(output.data, 1) == labels.long())  # the num of correct inference result
            else:
                # Eval phase
                model.eval()
                output = model(inputs)
                # Evaluate phase log
                val_num += inputs.size(0)
                loss = loss_fn(output, labels.long())
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(torch.argmax(output.data, 1) == labels.long())
        # Time consume
        time_consume = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_consume // 60, time_consume % 60))
        # Count
        train_loss_all.append(train_loss / train_num)                     # append the average training loss
        train_acc_all.append(train_corrects.double().item() / train_num)  # append the average training accuracy
        val_loss_all.append(val_loss / val_num)                           # append the average evaluate loss
        val_acc_all.append(val_corrects.double().item() / val_num)        # append the average evaluate accuracy
        # Visualizing the process of epoch
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))           # Latest evaluate loss and accuracy
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))   # Latest training loss and accuracy

        # Save the best model params
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_epoch = epoch
            print('best epoch {}'.format(best_epoch))
            torch.save(model.state_dict(), "Best_CD_model.pth")
            torch.save(optimizer.state_dict(), "Best_CD_optimizer")
            best_model_weight = copy.deepcopy(model.state_dict())

    # return the best model
    model.load_state_dict(best_model_weight)
    # return the whole train process
    train_process = pd.DataFrame(
                                    data={
                                        "epoch": range(epochs),
                                        "val_acc_all": val_acc_all,
                                        "val_loss_all": val_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "train_loss_all": train_loss_all
                                        })
    return model, train_process


# Visualizing process define
def visualizing_process(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Valuate loss")
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val Accuracy")
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train Accuracy")
    plt.legend()

    plt.show()


# Start training
net, train_process = train_loop(dataloader=Train_Dataloader, model=net, loss_fn=criterion, optimizer=optimizer, train_rate=train_rate, epochs=num_epoch)
print('训练完成')
visualizing_process(train_process)
''''''
# #----------------------------------- Testing Phase ------------------------------------#
# Confusion Matrix define
def confusion_matrix(preds, labels, conf_matrix):
    _, predicted = torch.max(preds.data, 1)
    _, labelval = torch.max(labels, 1)
    for p, t in zip(predicted, labelval):
        conf_matrix[p, t] += 1
    return conf_matrix


# Test funtion define
def test(net, dataloader, PATH, which_model):
    # 测试在整个数据集上的效果
    correct = 0
    total = 0
    # 首先定义一个 分类数*分类数 的空混淆矩阵
    conf_matrix = torch.zeros(num_classes, num_classes)
    net.load_state_dict(torch.load(PATH))  # 加载权重参数
    # 由于测试阶段不参与训练，所以不需要计算梯度这些的
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            # Compute prediction and loss
            pred = net(inputs)
            # _, predicted = torch.max(pred.data, 1)
            # 实验结果
            # 记录混淆矩阵参数
            conf_matrix = confusion_matrix(pred, labels, conf_matrix)
            conf_matrix = conf_matrix.cpu()
            total += labels.size(0)  # 预测总数
            # print(total)
            _, predicted = torch.max(pred.data, 1)
            _, labelval = torch.max(labels, 1)
            for i in range(0, predicted.shape[0]):
                if predicted[i] == labelval[i]:
                    correct += 1
            # correct += (pred == labels).sum().item()
    print('Accuracy of the network on %d test sample:%d %%' % (total, (100*correct/total)))
    conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数

    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), total))
    print(conf_matrix)

    # 获取每种Emotion的识别准确率
    print("总个数：", per_kinds)
    print("每种正确的个数：", corrects)
    print("每种识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))


# 加载自定义训练数据集
CD = CustomDataset(dir_train+label_dir, dir_train+sample_dir, seq_length*in_channels, (seq_length, in_channels), 'test')
Train_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False)


# 加载自定义测试数据集
CD = CustomDataset(dir_test+label_dir, dir_test+sample_dir, seq_length*in_channels, (seq_length, in_channels), 'test')
Test_Dataloader = DataLoader(dataset=CD, batch_size=1, shuffle=False)

test(net, Train_Dataloader, PATH, 2)
test(net, Test_Dataloader, PATH, 2)
''''''
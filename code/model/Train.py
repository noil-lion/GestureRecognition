import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim  # 优化器
from torch.autograd import Variable
# from torch.nn.functional import normalize
import sys
sys.path.append('../../code/')
from utils.CustomDataset import CustomDataset  # 导入自定义数据集
from model.Resnet.ODResnetC import ODResnet  # 导入CNN_LSTM
from model.Main.CNNLSTM_a import CNNLSTM
# from model.CNN_LSTM_Cust.CNNLSTM_a import CNNLSTM

# #-----------------------------------variables-----------------------------------------------#
# batch size
BATCH_SIZE = 32
epoch = 128     # epoch
momentum = 0.9
num_classes = 5
batch_size = 32
lr = 0.001
in_channels = 9
out_channels = 12
seq_length = 400
hidden_size = 48
num_layers = 2
k_split_value = 5
dir_train = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/train/'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/data/test/'
sample_dir = 'samples.csv'
label_dir = 'labels.csv'
PATH = '../weight/CNNLSTM_opt1.pth'
# PATH = '../weight/ODResnet_opt3.pth'

# #-----------------------------------Commands to download and perpare the dataset ------------------------------------#

Train_Dataset = CustomDataset(dir_train+label_dir, dir_train+sample_dir, 3600, (400, 9), 'train')
Test_Dataset = CustomDataset(dir_test+label_dir, dir_test+sample_dir, 3600, (400, 9), 'train')

Train_Dataloader = DataLoader(dataset=Train_Dataset, batch_size=BATCH_SIZE, shuffle=True)
Test_Dataloader = DataLoader(dataset=Test_Dataset, batch_size=BATCH_SIZE, shuffle=True)

# #-----------------------------------init ------------------------------------#
# 定义损失函数核优化器
criterion = nn.CrossEntropyLoss()  # criterion:标准，准则，原则, CrossEntropyLossL:交叉熵损失
criterion = criterion.cuda()

# 定义网络
net = CNNLSTM(in_channels, out_channels, hidden_size, num_layers, num_classes, batch_size, seq_length)
# net = ODResnet(in_channels, class_num=num_classes)
if torch.cuda.is_available():
    net = net.cuda()
# print(net)
optimizer = optim.Adam(net.parameters(), lr=lr)


# 定义训练和验证函数
def train(model, dataloader, criterion, optimizer, epoch):
    for batch, data in enumerate(dataloader):
        # 将数据从 dataloader 中读出来,一次读取的样本数是BATCH_SIZE个
        inputs, labels = data
        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.float()
        inputs = inputs.cuda()
        labels = labels.cuda()
        # Compute prediction and loss
        output = model(inputs)

        loss = criterion(output, labels.long())
        if batch % 100 == 0:
            print('Epoch :{}	 Loss:{:.6f}	 '.format(epoch, loss.data.item()))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 定义验证函数
def test(net, dataloader, PATH):
    # 测试在整个数据集上的效果
    correct = 0
    total = 0
    # 首先定义一个 分类数*分类数 的空混淆矩阵
    # net.load_state_dict(torch.load(PATH))  # 加载权重参数
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
            total += labels.size(0)  # 预测总数
            _, predicted = torch.max(pred.data, 1)
            labelval = labels.long()
            for i in range(0, predicted.shape[0]):
                if predicted[i] == labelval[i]:
                    correct += 1
            # correct += (pred == labels).sum().item()
    return 100. * correct / total

# !pip install sklearn -i https://pypi.mirrors.ustc.edu.cn/simple
from sklearn.model_selection import KFold
# the dataset for k fold cross validation
dataFold = Train_Dataset
# Kflod verify function
import collections
from torch.nn import init


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def train_flod(model, criterion, optimizer, epoch, dataFold, k_split_value, PATH):
    history = collections.defaultdict(list)  # 记录每一折的各种指标
    best_test = 0.0
    round = 1
    model.apply(weigth_init)
    kf = KFold(n_splits=k_split_value, shuffle=True, random_state=42)  # init KFold
    for train_index, test_index in kf.split(dataFold):  # split
        print('train_index:')
        print(train_index)
        print('test_index:')
        print(test_index)
        # get train, val
        train_fold = torch.utils.data.dataset.Subset(dataFold, train_index)
        test_fold = torch.utils.data.dataset.Subset(dataFold, test_index)

        # package type of DataLoader
        train_loader = torch.utils.data.DataLoader(dataset=train_fold, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_fold, batch_size=BATCH_SIZE, shuffle=False)

        print('-----------------This is round {round}-----------------'.format(round=round))
        round += 1
        for i in range(epoch):
            train(model, train_loader, criterion, optimizer, i)  # model, dataloader, criterion, optimizer, epoch
        train_acc = test(model, train_loader, PATH)  # Testing the the current CNN
        test_acc = test(model, test_loader, PATH)
        if test_acc > best_test:
            best_test = test_acc
            torch.save(model.state_dict(), PATH)
        # torch.save(model,'perceptron.pt')
        # one epoch, all acc
        # history.append(np.array(test_acc))
        model.apply(weigth_init)
        history['test_acc'].append(np.array(test_acc))
        history['train_acc'].append(np.array(train_acc))
    return history


# K折交叉验证
history = train_flod(net, criterion, optimizer, epoch, dataFold, k_split_value, PATH)
# 最后对每一折的结果取平均即可作为10折交叉验证的结果。
print(history)
m1 = np.mean(history['test_acc'])
m2 = np.mean(history['train_acc'])
print('avg_test_acc:')
print(m1)
print('avg_train_acc:')
print(m2)
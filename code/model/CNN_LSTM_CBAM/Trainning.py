import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim  # 优化器
from torch.autograd import Variable
# from torch.nn.functional import normalize
import sys
sys.path.append('../../../code/')
from utils.CustomDataset import CustomDataset  # 导入自定义数据集
from CNN_LSTM_CBAM import CNN_LSTM_CBAM  # 导入CNN_LSTM


# 定义常量
dir_train = 'D:/ResearchSpace/task/gestureRecognition/data/train/'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/data/test/'
sample_dir = 'samples.csv'
label_dir = 'labels.csv'
PATH = '../../weight/CNN_LSTM_CBAM_raw.pth'
num_classes = 5
batch_size = 32
lr = 0.001
momentum = 0.9
epoch = 200

# 无GPU跑cpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'CPU')

# 加载自定义训练数据集
CD = CustomDataset(dir_train+label_dir, dir_train+sample_dir, 2400, (400, 6))
Train_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False)


# 加载自定义测试数据集
CD = CustomDataset(dir_test+label_dir, dir_test+sample_dir, 2400, (400, 6))
Test_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False)

# 实例化网络
net = CNN_LSTM_CBAM(num_classes)
print(net)
# 定义损失函数核优化器
criterion = nn.L1Loss()  # criterion:标准，准则，原则, CrossEntropyLossL:交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


# 定义训练函数
def train_loop(dataloader, model, loss_fn, optimizer, which_model, epoch):
    for batch, data in enumerate(dataloader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, labels = data
        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)
        if which_model == 1:
            inputs = inputs.squeeze(1)
        elif which_model == 2:
            inputs = inputs.unsqueeze(1)   # sequeeze 用于对输入数据进行压缩或者unsqueeze解压，也就是增加维度或者减小维度
            inputs = inputs.unsqueeze(1)
        else:
            pass
        # Compute prediction and loss
        output = model(inputs)

        loss = loss_fn(output, labels)
        if batch % 100 == 0:
            print('Epoch :{}	 Loss:{:.6f}	 '.format(epoch, loss.data.item()))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


''''''
# pdb.set_trace()
# 开始训练
for epoch in range(epoch):
    train_loop(dataloader=Train_Dataloader, model=net, loss_fn=criterion, optimizer=optimizer, which_model=2, epoch=epoch)

print('训练完成')
# 保存训练好的网络权重参数
torch.save(net.state_dict(), PATH)  # torch.nn.Module模块中的state_dict变量存放着权重和偏置参数，是一个python的字典对象，将每一层的参数映射成tensor张量，它只包含卷积层和全连接层参数，如batchnorm是不会被包含的


# 加载训练的网络
def confusion_matrix(preds, labels, conf_matrix):
    _, predicted = torch.max(preds.data, 1)
    _, labelval = torch.max(labels, 1)
    for p, t in zip(predicted, labelval):
        conf_matrix[p, t] += 1
    return conf_matrix


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
            if which_model == 1:
                inputs = inputs.squeeze(1)
            elif which_model == 2:
                inputs = inputs.unsqueeze(1)   # sequeeze 用于对输入数据进行压缩或者unsqueeze解压，也就是增加维度或者减小维度
                inputs = inputs.unsqueeze(1)
            else:
                pass
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


test(net, Train_Dataloader, PATH, 2)
test(net, Test_Dataloader, PATH, 2)

import sys
sys.path.append('../../code/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.CustomDataset import CustomDataset  # 导入自定义数据集
# from model.CNN_LSTM import CNNLSTM  # 导入CNN_LSTM
from model.ODCNN import Net
import torch.optim as optim  # 优化器
from torch.autograd import Variable
from torch.nn.functional import normalize


# 定义常量
dir_train = 'D:/ResearchSpace/task/gestureRecognition/data/train/'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/data/test/'
sample_dir = 'samples.csv'
label_dir = 'labels.csv'
PATH = '../weight/ODCNN.pth'
num_classes = 5
batch_size = 32
lr = 0.001
momentum = 0.9
epoch = 100

# 无GPU跑cpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'CPU')

# 加载自定义训练数据集
CD = CustomDataset(dir_train+label_dir, dir_train+sample_dir, 2400, (400, 6))
Train_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False, drop_last=True)


# 加载自定义测试数据集
CD = CustomDataset(dir_test+label_dir, dir_test+sample_dir, 2400, (400, 6))
Test_Dataloader = DataLoader(dataset=CD, batch_size=batch_size, shuffle=False, drop_last=True)

# 实例化网络
net = Net()
print(net)
# 定义损失函数核优化器
criterion = nn.L1Loss()  # criterion:标准，准则，原则, CrossEntropyLossL:交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 定义训练函数
def train_loop(dataloader, model, loss_fn, optimizer, which_model):
    size = dataloader.__len__

    for batch, data in enumerate(dataloader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, labels = data
        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)

        # inputs = torch.reshape(inputs, (inputs.shape[0], 1, -1))
        # print(inputs.shape)
        if which_model == 1:
            inputs = inputs.squeeze(1)
        elif which_model == 2:
            inputs = inputs.unsqueeze(1)   # sequeeze 用于对输入数据进行压缩或者unsqueeze解压，也就是增加维度或者减小维度
            # inputs = inputs.unsqueeze(1)
        else:
            pass
        # Compute prediction and loss
        loss = 0
        nor_inputs = normalize(inputs, dim=2)
        print(nor_inputs.shape)
        pred = model(nor_inputs)
        loss += loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f} ")


# 开始训练
for epoch in range(epoch):
    train_loop(dataloader=Train_Dataloader, model=net, loss_fn=criterion, optimizer=optimizer, which_model=1)

print('训练完成')
# 保存训练好的网络权重参数
torch.save(net.state_dict(), PATH)  # torch.nn.Module模块中的state_dict变量存放着权重和偏置参数，是一个python的字典对象，将每一层的参数映射成tensor张量，它只包含卷积层和全连接层参数，如batchnorm是不会被包含的
"""

def test_loop(dataloader, model, loss_fn, which_model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if which_model == 1:
                X = X.squeeze(1)
            elif which_model == 2:
                X = X.unsqueeze(1)
            else:
                pass
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")
"""


# 加载训练的网络
def test(net, dataloader, PATH, which_model):
    # 测试在整个数据集上的效果
    correct = 0
    total = 0
    net.load_state_dict(torch.load(PATH))  # 加载权重参数
    # 由于测试阶段不参与训练，所以不需要计算梯度这些的
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)
            # inputs = torch.reshape(inputs, (inputs.shape[0], 1, -1))
            if which_model == 1:
                inputs = inputs.squeeze(1)
            elif which_model == 2:
                inputs = inputs.unsqueeze(1)   # sequeeze 用于对输入数据进行压缩或者unsqueeze解压，也就是增加维度或者减小维度
                # inputs = inputs.unsqueeze(1)
            else:
                pass

            # print(total)
            for i in range(0, inputs.shape[0]-1):
                nor_inputs = normalize(inputs[i], dim=2)
                # Compute prediction and loss
                pred = net(nor_inputs)
                # _, predicted = torch.max(pred.data, 1)

                _, predicted = torch.max(pred.data, -1)
                _, labelval = torch.max(labels[i], -1)
                if predicted == labelval:
                    correct += 1
            total += labels.size(0)  # 预测总数
            # correct += (pred == labels).sum().item()
    print('Accuracy of the network on %d test sample:%d %%' % (total, (100*correct/total)))


test(net, Test_Dataloader, PATH, 2)

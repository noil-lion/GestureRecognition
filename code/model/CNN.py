from pkg_resources import WorkingSet
import torch
import torch.nn as nn
import torchvision  # torchvision.datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM
import torch.optim as optim


RPOCH = 3
BATCH_SIZE = 50
LR = 0.001

classes = {'initial': 0, 'qbcy': 1, 'qbxq': 2, 'szcs': 3, 'szqs': 4}  # 定义类

# pytorch读取自身数据集，主要通过Dataset类中的getitem(self, index)函数，接受一个index，返回一个lis的其中一个元素，其中的元素包括了图片数据的路径和标签信息。
# 制作这个list也就是准备数据集的核心工作了，常用的方法为将图片的路径和标签信息存在txt中，再读取转为list文件，list中每个元素对应一个样本，再使用getitem（）返回样本信息和标签
# 要想让pytroch读取自己的数据集：1.制作图片数据索引 2.构建Dataset子类

# 构建Dataset子类-MyDataset
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None,
                 target_transform=None):  # 构造函数
        fh = open(txt_path, 'r')  # 读取构造的txt文件
        imgs = []
        for line in fh:
            line = line.rstrip()  # rstrip（）删除字符串末尾的指定字符，默认为空白符，这里相当于读取其中的一行。
            words = line.split(
            )  # split()通过指定分隔符对字符串进行切片，默认为所有空字符，包括空格、换行、制表符（\t）
            imgs.append(
                (words[0], str(words[1])
                 ))  # 分离这里得到的就是图片路径和图片的标签，这里的imgs就是样本信息列表类型str，包含（图片路径， 图片标签）
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # 根据index返回样本信息
        img = Image.open(fn).convert('RGB')  # 根据路径读取图片
        label = classes[label]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class LSTM(nn.Module):
    def forward(self, input_x):  # input_x.size() 为 [20个数据,5个维度]
        input_x = input_x.view(len(input_x), 1, -1)  # 维度变为 [20,1,5]
        lstm_out, self.hidden_cell = self.lstm(input_x, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_x), -1))
        return predictions[-1]


class my_Net(nn.Module):
    def __init__(
            self
    ):  # 类的构造方法或叫做初始化方法，当创建类实例时会调用，由类成员，方法，数据属性组成。self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。self 代表的是类的实例，代表当前对象的地址，而 self.__class__ 则指向类
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 6, 5)  # 输入维度：3 channal, output: 6 channal, size:5*5卷积核尺寸
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            16 * 29 * 29, 120
        )  # 设输入图像尺寸为W，卷积核尺寸为F，步幅为S，Padding使用P，则经过卷积层或池化层之后的图像尺寸为（W-F+2P）/S+1。
        self.fc2 = nn.Linear(120, 84)
        self.lstm = LSTM()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.lstm(x)
        return x


# mydataset构建好后，剩下的交割dataloader，dataloarder 会触发getitem函数，并拼接成一个batch返回，作为模型的输入
txt_path = 'D:/ResearchSpace/TFL/dataV5/'

# 图像的初始化操作
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])
text_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])

# 数据集加载方式设置
train_data = MyDataset(txt_path=txt_path + 'train.txt',
                       transform=transforms.ToTensor())
test_data = MyDataset(txt_path=txt_path + 'valid.txt',
                      transform=transforms.ToTensor())
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
trainloader = DataLoader(dataset=train_data,
                         batch_size=24,
                         shuffle=True,
                         num_workers=0)
testloader = DataLoader(dataset=test_data,
                        batch_size=24,
                        shuffle=False,
                        num_workers=0)

net = my_Net()

PATH = './myCNN.pth'


def train(net, trainloader, epochs, PATH):

    # 定义损失函数核优化器
    criterion = nn.CrossEntropyLoss(
    )  # criterion:标准，准则，原则, CrossEntropyLossL:交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 遍历数据迭代器，输入网络进行参数优化，进行网络训练
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(
                trainloader, 0
        ):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时返回序列中的数据和数据下标，一般用在 for 循环当中.包含俩参数，序列核下标起始位置
            # 获得输入数据，数据用list保存：[inputs, labels]
            inputs, labels = data
            # 初始化参数梯度
            optimizer.zero_grad(
            )  # 把loss关于weight的导数变成0.（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
            # forward+backward+optimize
            outputs = net(inputs)  # forward,加载网络模型并输出数据
            loss = criterion(outputs, labels)  # loss def
            loss.backward()  # backward
            optimizer.step()  # 梯度下降，更新权重，前面已经绑定好网络参数，依据反向计算存储在节点中的grad来进行参数更新
            # 输出统计结果
            running_loss += loss
            if i % 200 == 199:  # 每200 batch输出一次loss的平均值
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0
    print('训练完成')
    # 保存训练好的网络权重参数
    torch.save(
        net.state_dict(), PATH
    )  # torch.nn.Module模块中的state_dict变量存放着权重和偏置参数，是一个python的字典对象，将每一层的参数映射成tensor张量，它只包含卷积层和全连接层参数，如batchnorm是不会被包含的


train(net, trainloader, epochs=30, PATH=PATH)


# 加载训练的网络
def test(net, testloader, PATH):
    # 测试在整个数据集上的效果
    correct = 0
    total = 0
    net.load_state_dict(torch.load(PATH))  # 加载权重参数
    # 由于测试阶段不参与训练，所以不需要计算梯度这些的
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 预测总数
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on 10000 test images:%d %%' %
          (100 * correct / total))


# test(net, trainloader, PATH)

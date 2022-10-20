import torch.optim
import torch.nn as nn
from conv2d_mtl import Conv2dMtl
import sys
sys.path.append('../FES/model')
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import cv2    # 读取图片
import os     # 生成图片的路径
import torchvision.transforms as transforms   # 转换图片的方法
from torchvision import  datasets


class VGG16(nn.Module):
    def __init__(self, nums):
        super(VGG16, self).__init__()
        self.nums = nums
        self.Conv2d = Conv2dMtl
        vgg = []
        # 第一个卷积部分
        # 112, 112, 64
        vgg.append(self.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第二个卷积部分
        # 56, 56, 128
        vgg.append(self.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第三个卷积部分
        # 28, 28, 256
        vgg.append(self.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第四个卷积部分
        # 14, 14, 512
        vgg.append(self.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第五个卷积部分
        # 7, 7, 512
        vgg.append(self.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(self.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        self.main = nn.Sequential(*vgg)

        # 全连接层
        classfication = []
        # in_features四维张量变成二维[batch_size,channels,width,height]变成[batch_size,channels*width*height]
        classfication.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))  # 输出4096个神经元，参数变成512*7*7*4096+bias(4096)个
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=4096))
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=self.nums))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        feature = self.main(x)  # 输入张量x
        feature = feature.view(x.size(0), -1)  # reshape x变成[batch_size,channels*width*height]
        result = self.classfication(feature)
        return result


def train(model, train_loader, myloss, optimizer, epoch):
    model.train()
    for batch_idx, train_data in enumerate(train_loader):
        # train_data = Variable(train_data).cuda()
        optimizer.zero_grad()
        output = model(torch.tensor(train_data[0]))
        loss = myloss(output, torch.tensor(train_data[1]))
        loss.backward()
        optimizer.step()
        if batch_idx%1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx*len(train_data), len(train_loader.dataset),
                100*batch_idx/len(train_loader), loss.cpu().data.numpy()))



def main():
    epoch = 20
    model = VGG16(4)
    """model_dict=model.load_state_dict(torch.load("D:\ResearchSpace\TFL\Fes\model\VGG16_complt_3Chanall.h5"))
    new_state_dict = {}
    for k, v in model_dict.items():
        if 'roi_head' not in k:
            new_state_dict[k] = v
    model.load(model_dict)"""
    myloss= nn.NLLLoss()
 
    train_data = datasets.ImageFolder(r'D:/ResearchSpace/TFL/dataV2',transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ]))
    train_loader = data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=1)
 
    optimizer = optim.SGD(model.parameters(),lr=0.001)
 
    for epoch in range(epoch):
        train(model, train_loader, myloss, optimizer, epoch)
    torch.save(model, 'model_VGG16_torch.h5')


if __name__=='__main__':
    main()

    
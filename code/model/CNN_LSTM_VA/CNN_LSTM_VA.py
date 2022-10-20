import torch
from torch import nn
import torch.nn.functional as F
from plugs import SpatialTransformer


class LeNetVariant(nn.Module):
    def __init__(self) -> None:
        super(LeNetVariant, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),  # 16个卷积核，每个大小为3*3，输入为400*6 输出为16个feature map（400-3+1）*（6-3+1）4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # out_size=((input_size-kernel_size+2padding)/stride) + 1
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            )
        self.STN = SpatialTransformer(spatial_dims=(32, 99, 1))
        self.classifier = nn.Sequential(nn.Linear(32*3*3, 120), nn.Linear(120, 84))

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.STN(x)
        print(x.shape)
        x = x.view(-1, 32*3*3)  # reshape成32*3*3列，行数任意，相当于flatten，进入linear层的前一步
        x = self.classifier(x)   # 出来之后84*x
        return x


class CNNLSTM(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNNLSTM, self).__init__()
        self.cnn = LeNetVariant()
        self.lstm = nn.LSTM(input_size=84, hidden_size=128, num_layers=2, batch_first=True)   #
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        # print(x_3d.shape)  # x_3d 输入为1个batch的样本 共32个
        cnn_output_list = list()
        for t in range(x_3d.size(0)):
            cnn_output_list.append(self.cnn(x_3d[t, :, :, :, :]))   # t表示时间步？ 这里t从0遍历到400，将一批输入到CNN的数据存在list中
        x = torch.stack(tuple(cnn_output_list), dim=1)  # 栈存储
        out, hidden = self.lstm(x)    # 输入LSTM
        x = out[-1, :]   # 取输出数据
        x = F.relu(x)  # relu
        x = self.fc1(x)
        return x

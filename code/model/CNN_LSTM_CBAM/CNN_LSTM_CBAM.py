import torch
from torch import nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 1, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=0, keepdim=True)
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 如何添加模块进CNN框架里
"""
不改变原有CNN结构，CBAM不加在block里面，加在最后一层卷积和第一层卷积不改变网络，且可以用预训练参数。
"""


class LeNetVariant(nn.Module):
    def __init__(self) -> None:
        super(LeNetVariant, self).__init__()
        self.inplanes = 1
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()    # 网络的卷积层的第一层加入注意力机制
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),  # 16个卷积核，每个大小为3*3，输入为400*6 输出为16个feature map（400-3+1）*（6-3+1）4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # out_size=((input_size-kernel_size+2padding)/stride) + 1
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            )

        self.classifier = nn.Sequential(nn.Linear(32*3*3, 120), nn.Linear(120, 84))

    def forward(self, x):
        # x = self.ca(x) * x
        # print(x.shape)  # x_3d 输入为1个batch的样本 共32个
        x = self.sa(x) * x

        x = self.features(x)
        x = x.view(-1, 32*3*3)  # reshape成32*3*3列，行数任意，相当于flatten，进入linear层的前一步
        x = self.classifier(x)   # 出来之后84*x
        return x


class CNN_LSTM_CBAM(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNN_LSTM_CBAM, self).__init__()
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

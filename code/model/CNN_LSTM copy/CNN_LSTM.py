import torch
from torch import nn
import torch.nn.functional as F


class LeNetVariant(nn.Module):
    

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*3*3)  # reshape成32*3*3列，行数任意，相当于flatten，进入linear层的前一步
        x = self.classifier(x)   # 出来之后84*x
        return x


class CNNLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers,
                 num_classes, batch_size, seq_length) -> None:
        super(CNNLSTM, self).__init__()
        self.batch_size = batch_size  # batch_size:32
        self.seq_length = seq_length  # 样本序列时间步长:400
        self.in_channels = in_channels  # in_channels：单步向量维度，也是输入通道数6个特征每时间步
        self.out_channels = out_channels  # out_channels：Convs 阶段的输出通道数，也是LSTM输入的input_size
        self.hidden_size = hidden_size  # LSTM输出的通道数，每层LSTM有多少个节点数，也是后面全连接层的输入通道数
        self.num_layers = num_layers   # LSTM网络层数
        self.num_classes = num_classes  # 最终分类树
        self.num_directions = 1  # 单向LSTM
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),  # 16个卷积核，每个大小为3*3，输入为400*6 输出为16个feature map（400-3+1）*（6-3+1）4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # out_size=((input_size-kernel_size+2padding)/stride) + 1
            nn.Conv2d(32, self.out_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            )
        self.lstm = nn.LSTM(input_size=self.out_channels, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)   #
        self.fc1 = nn.Linear(self.hidden_size*2, num_classes)

    def forward(self, x_3d):
        # print(x_3d.shape)  # x_3d 输入为1个batch的样本 共32个
        x = self.features(x_3d.permute(0, 1, 3, 2))[:, :, 0, :]   # b表示batch？ 这里t从0遍历到400，将一批输入到CNN的数据存在list中

        # print(x.shape)
        out, hidden = self.lstm(x.permute(0, 2, 1))    # 输入LSTM
        # x = out[-1, :]   # 取输出数据
        x = self.fc1(hidden)
        x = x[:, -1, :]
        return x

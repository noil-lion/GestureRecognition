from torch import nn
import math
import torch

"""
1D卷积CNN-LSTM简单版：精度较差，最优50%测试， 74%训练。寄：在使用损失函数L1loss时效果巨差，收敛也有问题
更换loss函数为CrossEntropyLoss后，效果提升，训练99%，测试94%
更换优化器后，有SGD换为Adam后，有94%提到测试集98%
"""


class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample is True:
            self.stride = 2
        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),  # 调整第一层卷积的stride进行降采样
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU()
        )
        # self.ca = SELayer(channel=In_channel)
        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)

    def forward(self, x):
        shortcut = self.res_layer(x)   # 输入输出通道维度不一致，进行通道数对齐
        return self.layer(x)+shortcut      # 残差块输出和shorcut输出相加 elementwiseadd


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=9):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 参数output_size，即输出每个通道的特征图的维度大小size为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()  # b = 32 c = 6/9 _ = 400/128
        y = self.avg_pool(x).view(b, c)  #
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


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
        self.relu = nn.ReLU(inplace=True)  # (batch_size=32, seq_len=400, input_size=6) ---> permute(0, 2, 1) (32, 6, 400)

        self.channel_attention = ChannelAttention(out_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=32,  kernel_size=3, stride=1),  # shape(32, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
            nn.MaxPool1d(3, 2, 1),
            Bottlrneck(32, 64, self.out_channels, False),
            nn.MaxPool1d(kernel_size=2, stride=2))  # self.ca = SELayer(channel=out_channels)
        self.lstm = nn.LSTM(input_size=out_channels,  # 12
                            hidden_size=hidden_size,  # 39
                            num_layers=num_layers,   # 2
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 39->5

    def forward(self, x):
        # print('输入的数据维度：')
        # print({x.shape})      # {torch.Size([32, 400, 6])}
        x = x.permute(0, 2, 1)  # {torch.Size([32, 6, 400])}
        x = self.conv(x)
        attention_value = self.channel_attention(x)
        x = x * attention_value  # 得到借助注意力机制后的输出
        # x = self.ca(x)
        # print('卷积后的数据维度：')
        # print({x.shape})      # {torch.Size([32, 12, 48])},这里48为卷积后的序列长度，12为序列的特征通道数
        x = x.permute(0, 2, 1)  # {torch.Size([32, 48, 12])}，为输入LSTM，将输入进行转置，时间步置前，输入输出时间维度不变更。
        output, _ = self.lstm(x)
        # print('LSTM后的数据维度：')
        # print({output.shape})   # {torch.Size([32, 48, 48])}，LSTM主要是对特征通道的学习记忆遗忘，最后的时间步特征为综合学习前面时间步特征的总结性信息，信息也是以向量表示的，其维度与每层的LSTM的hiddensize的神经元个数一致，也就是进行了通道信息变换，可选择丰富化特征（hidden_size>out_channel）也可选精简特征信息（hidden_size>out_channel）。
        # x = output[:, -1, :]   # 这里可选，将所有时间步的信息都输入FC进行组合，还是在这之前把LSTM认为学习到的综合前面时间步数据的特征信息单独拎出来作为最终的分类判别依据。
        pred = self.fc(output)
        # print('全连接层后的数据维度：')
        # print({pred.shape})    # {torch.Size([32, 48, 5])},FC在这进行了特征通道层面的组合和精简，应该不跨时间维度。
        pred = pred[:, -1, :]
        # print('返回值的数据维度：')  # {torch.Size([32, 5])},取最后一个时间步的特征向量作为分类结果。
        # print({pred.shape})
        return pred

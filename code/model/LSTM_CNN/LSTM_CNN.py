'''
-*- coding: utf-8 -*-
@Time : 2022/12/5 22:18
@Author : Zihao Wu
@File : LSTM_CNN.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv1d(in_channels, 24, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv1d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv1d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv1d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv1d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv1d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv1d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class LSTMCNN(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, num_layers,
                 num_classes, batch_size, seq_length) -> None:
        super(LSTMCNN, self).__init__()
        self.batch_size = batch_size  # batch_size:32
        self.seq_length = seq_length  # 样本序列时间步长:400
        self.in_channels = in_channels  # in_channels：单步向量维度，也是输入通道数6个特征每时间步
        self.out_channels = out_channels  # out_channels：Convs 阶段的输出通道数，也是LSTM输入的input_size
        self.hidden_size = hidden_size  # LSTM输出的通道数，每层LSTM有多少个节点数，也是后面全连接层的输入通道数
        self.num_layers = num_layers  # LSTM网络层数
        self.num_classes = num_classes  # 最终分类树
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(
            inplace=True
        )  # (batch_size=32, seq_len=400, input_size=6) ---> permute(0, 2, 1) (32, 6, 400)
        self.lstm = nn.GRU(
            input_size=in_channels,  # 12
            hidden_size=hidden_size,  # 39
            num_layers=num_layers,  # 2
            batch_first=True,
            bidirectional=True)
        self.incep1 = InceptionA(in_channels=hidden_size * 2)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size * 2,
                out_channels=self.out_channels,  # 12
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            # nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.AdaptiveAvgPool1d(1)
        )  # self.attention = MultiHeadSelfAttention(input_dim=out_channels, dim_q=out_channels, dim_v=out_channels, n_head=2)
        self.fc = nn.Linear(self.out_channels, num_classes)  # 39->5

    def forward(self, x):
        # print('输入的数据维度：')
        # print({x.shape})        # {torch.Size([32, 400, 6])}

        x, _ = self.lstm(x)

        # print({x.shape})        # {torch.Size([32, 400, 6])}
        # print('LSTM后的数据维度：')
        # print({x.shape})        # {torch.Size([32, 400, 96])}，LSTM主要是对特征通道的学习记忆遗忘，最后的时间步特征为综合学习前面时间步特征的总结性信息，信息也是以向量表示的，其维度与每层的LSTM的hiddensize的神经元个数一致，也就是进行了通道信息变换，可选择丰富化特征（hidden_size>out_channel）也可选精简特征信息（hidden_size>out_channel）。
        x = x.permute(0, 2, 1)  # {torch.Size([32, 400, 96])}
        x = self.incep1(x)
        # print('Inception数据维度：')
        # print({x.shape})        # {torch.Size([32, 400, 6])}
        x = self.conv(x)
        # print('卷积后的数据维度：')
        # print({x.shape})        # {torch.Size([32, 12, 1])},这里48为卷积后的序列长度，12为序列的特征通道数
        x = x.permute(
            0, 2,
            1)  # {torch.Size([32, 1, 12])}，为输入LSTM，将输入进行转置，时间步置前，输入输出时间维度不变更。
        # x = self.attention(x)
        # x = output[:, -1, :]   # 这里可选，将所有时间步的信息都输入FC进行组合，还是在这之前把LSTM认为学习到的综合前面时间步数据的特征信息单独拎出来作为最终的分类判别依据。
        pred = self.fc(x)
        # print('全连接层后的数据维度：')
        # print({pred.shape})    # {torch.Size([32, 1, 5])},FC在这进行了特征通道层面的组合和精简，应该不跨时间维度。
        pred = pred[:, -1, :]
        # print('返回值的数据维度：')  # {torch.Size([32, 5])},取最后一个时间步的特征向量作为分类结果。
        # print({pred.shape})
        return pred

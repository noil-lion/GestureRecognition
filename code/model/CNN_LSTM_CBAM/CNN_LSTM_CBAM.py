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
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.in_channels = in_channels  # in_channels：单步向量维度，也是输入通道数6个特征每时间步,UCI数据集为9
        self.out_channels = out_channels  # out_channels：Convs 阶段的输出通道数，也是LSTM输入的input_size
        self.conv1 = nn.Conv1d(self.out_channels, self.in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out1 = torch.mean(x[:, 0:3, :], dim=2, keepdim=True)
        avg_out2 = torch.mean(x[:, 3:6, :], dim=2, keepdim=True)
        avg_out = torch.cat([avg_out1, avg_out2], dim=1)
        max_out1, _ = torch.max(x[:, 0:3, :], dim=2, keepdim=True)
        max_out2, _ = torch.max(x[:, 3:6, :], dim=2, keepdim=True)
        max_out = torch.cat([max_out1, max_out2], dim=1)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        return self.sigmoid(x)


# 如何添加模块进CNN框架里
"""
不改变原有CNN结构，CBAM不加在block里面，加在最后一层卷积和第一层卷积不改变网络，且可以用预训练参数。
"""


class CNNLSTMCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers,
                 num_classes, batch_size, seq_length) -> None:
        super(CNNLSTMCBAM, self).__init__()
        self.batch_size = batch_size  # batch_size:32
        self.seq_length = seq_length  # 样本序列时间步长:400
        self.in_channels = in_channels  # in_channels：单步向量维度，也是输入通道数6个特征每时间步
        self.out_channels = out_channels  # out_channels：Convs 阶段的输出通道数，也是LSTM输入的input_size
        self.hidden_size = hidden_size  # LSTM输出的通道数，每层LSTM有多少个节点数，也是后面全连接层的输入通道数
        self.num_layers = num_layers   # LSTM网络层数
        self.num_classes = num_classes  # 最终分类树
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)  # (batch_size=32, seq_len=400, input_size=6) ---> permute(0, 2, 1) (32, 6, 400)
        self.sa = SpatialAttention(self.in_channels, self.out_channels)    # 网络的卷积层的第一层加入注意力机制
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=32,  # 12
                      kernel_size=3, stride=1),  # shape(32, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
            nn.ReLU(),
            # nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32,
                      out_channels=64,  # 32
                      kernel_size=3, stride=1),  # shape(32, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
            nn.ReLU(),
            # nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64,
                      out_channels=self.out_channels,  # 12
                      kernel_size=3, stride=1),  # shape(32, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
            nn.ReLU(),
            # nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.lstm = nn.LSTM(input_size=out_channels,  # 12
                            hidden_size=hidden_size,  # 39
                            num_layers=num_layers,   # 2
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 39->5

    def forward(self, x):
        # print('输入的数据维度：')
        # print({x.shape})  # [32, 400, 6]
        x = x.permute(0, 2, 1)  # [32, 6, 400]
        sa = self.sa(x)
        x = torch.cat([x, sa], dim=2)
        # print(sa.shape)
        x = self.conv(x)
        # x = torch.cat([x, sa], dim=2)
        x = x.permute(0, 2, 1)
        # print({x.shape})
        # batch_size, seq_len = x.size()[0], x.size()[1]
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x)
        # x = output[：, -1, :]   # 取输出数据最后一个时间步数据
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred

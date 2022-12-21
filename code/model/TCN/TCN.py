import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block (makelayer,自适应模块定义)

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


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


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_class, kernel_size=9, dropout=0.1):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.sa = SpatialAttention(400, 1)    # 网络的卷积层的最后一层加入注意力机制
        self.linear = nn.Linear(num_channels[-1], num_class)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        # print('输入数据维度：')
        # print({x.shape})
        x = x.permute(0, 2, 1)   # [32, 400, 6] --> [32, 6, 400]
        x = self.network(x)      # [32, 6, 400] --> [32, 6, 400]
        # print('TCN后数据维度：')
        # print({x.shape})
        o = self.linear(x[:, :, -1])
        return F.log_softmax(o, dim=1)
        """# out, hidden = self.lstm(x.permute(0, 2, 1))    # 输入LSTM
        # x = out[:, -1, :]   # 取输出数据
        # print(out.shape)
        x = self.fc1(x.permute(0, 2, 1))
        # print(x.shape)
        sa = self.sa(x)
        print(sa.shape)
        x = x[:, -1, :]
        print(x.shape)
        return x"""
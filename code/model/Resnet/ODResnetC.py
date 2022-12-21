import torch
import math

# params 残差块类定义：可变参数，基本机构是两个卷积核size为1的卷积层中间夹一个卷积核size为3的卷积层
# In_channel int 块输入数据通道数
# Med_channel int 块内中间层数据通道
# Out_channel int 块输出数据通道数
# downsample bool 是否进行降采样

# 关键逻辑
# 1. if 块输出通道数!=shortcut输出通道数，让要调整的数据经过一个1*1卷积层，进行通道数调整，为之后可以直接相加
# 2. 改变卷积层的卷积步长stride来实现降采样


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
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)

    def forward(self, x):
        shortcut = self.res_layer(x)   # 输入输出通道维度不一致，进行通道数对齐
        return self.layer(x)+shortcut      # 残差块输出和shorcut输出相加 elementwiseadd


# params 一维残差网络定义
class ODResnet(torch.nn.Module):
    def __init__(self, Input_channel=9, class_num=5):
        super(ODResnet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(Input_channel, 64, kernel_size=3, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 128, 512, True),


            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.selfAttention = SelfAttention(Input_channel, dim_q=2, dim_v=Input_channel)
        # self.attention = SpatialAttention(Input_channel, 512)
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512, class_num)
        )

    def forward(self, x):
        x = self.selfAttention(x)
        # print(at.shape)
        x = x.permute(0, 2, 1)  # [32, 400, 6] --> [32, 6, 400]

        x = self.features(x)    # [32, 6, 400] --> [32, 512, 1]
        # print('ODResNet后数据维度：')
        # print({x.shape})
        x = x.view(-1, 512)
        # at = at.view(-1, 512)
        # x = x * at
        # print({x.shape})
        x = self.classifer(x)
        return x


# 自注意力
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, dim_q, dim_v):
        super(SelfAttention, self).__init__()
        # input_dim = 6  # 句子中每个单词的向量维度
        # seq_len = 400  # 序列长度
        # batch_size = 32  # 批量数
        # dim_q  # 
        # dim_v = batch_size + input_dim = 38
        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_q, dim_v

        self.Q = torch.nn.Linear(input_dim, dim_q)
        self.K = torch.nn.Linear(input_dim, dim_q)
        self.V = torch.nn.Linear(input_dim, dim_v)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # Q: [batch_size,seq_len,dim_q]
        # K: [batch_size,seq_len,dim_k]
        # V: [batch_size,seq_len,dim_v]
        Q, K, V = self.Q(x), self.K(x), self.V(x)
        # print(f'x.shape:{x.shape} , Q.shape:{Q.shape} , K.shape: {K.shape} , V.shape:{V.shape}')
        attention = torch.bmm(self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.dim_k)), V)
        return attention



"""
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),
            Bottlrneck(2048, 512, 2048, False),
"""


# 空间注意力
class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.in_channels = in_channels  # in_channels：单步向量维度，也是输入通道数6个特征每时间步,UCI数据集为9
        self.out_channels = out_channels  # out_channels：Convs 阶段的输出通道数，也是LSTM输入的input_size
        self.conv1 = torch.nn.Conv1d(self.in_channels*2, self.out_channels, kernel_size=3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x: torch.Size([32, 6, 400])
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [32, 6, 400] --> [32, 6, 1]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [32, 6, 400] --> [32, 6, 1]
        x = torch.cat([avg_out, max_out], dim=1)  # [32, 6, 400] --> [32, 12, 1]
        x = self.conv1(x)  # [32, 12, 1] --> [32, 512, 1]
        return self.sigmoid(x)

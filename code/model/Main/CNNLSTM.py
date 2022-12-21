from torch import nn
import math
import torch

"""
# self.attention = MultiHeadSelfAttention(input_dim=out_channels, dim_q=out_channels, dim_v=out_channels, n_head=2)
1D卷积CNN-LSTM简单版：精度较差，最优50%测试， 74%训练。寄：在使用损失函数L1loss时效果巨差，收敛也有问题
更换loss函数为CrossEntropyLoss后，效果提升，训练99%，测试94%
更换优化器后，有SGD换为Adam后，有94%提到测试集98%
"""


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_v, n_head=1):
        super(MultiHeadSelfAttention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.dim_v, self.n_head = dim_q, dim_q, dim_v, n_head

        if self.dim_k % n_head != 0:
            raise RuntimeError(
                f"请将batch_size = {dim_q} , 设置为n_head = {n_head}的整数倍，例如:{n_head * 1}、{n_head * 2}、{n_head * 3}...")
        if self.dim_v % n_head != 0:
            raise RuntimeError(
                f"请将batch_size + input_dim = {dim_v} , 设置为n_head = {n_head}的整数倍，例如:{n_head * 1}、{n_head * 2}、{n_head * 3}...")

        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_q)
        self.V = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

    def forward(self, x):
        # Q: [n_head,batch_size,seq_len,dim_q]
        # K: [n_head,batch_size,seq_len,dim_k]
        # V: [n_head,batch_size,seq_len,dim_v]
        Q = self.Q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_head)
        K = self.K(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_head)
        V = self.V(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.n_head)

        # print(f'x.shape:{x.shape} , Q.shape:{Q.shape} , K.shape: {K.shape} , V.shape:{V.shape}')

        attention = nn.Softmax(dim=-1)(
            torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        attention = torch.matmul(attention, V).reshape(x.shape[0], x.shape[1],
                                                       -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return attention


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
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=32,  # 12
                      kernel_size=3, stride=1),  # shape(32, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
            nn.ReLU(),
            # nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32,
                      out_channels=64,  # 32
                      kernel_size=3, stride=1),  # shape(12, 6, 400)  ->(32, 12, 39) # (400-20-1)/10 +1
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
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 39->5

    def forward(self, x):
        # print('输入的数据维度：')
        # print({x.shape})      # {torch.Size([32, 400, 6])}

        x = x.permute(0, 2, 1)  # {torch.Size([32, 6, 400])}
        x = self.conv(x)
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

import torch

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
            torch.nn.ReLU()
        )
        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)

    def forward(self, x):
        shortcut = self.res_layer(x)   # 输入输出通道维度不一致，进行通道数对齐
        return self.layer(x)+shortcut      # 残差块输出和shorcut输出相加 elementwiseadd


"""
# test
if __name__ == '__main__':
    x = torch.randn(1, 6, 400)
    model = Bottlrneck(6, 12, 250, True)  # makelayer
    output = model(x)
    print('output.shape:')
    print(output.shape)
    print(model)
"""


# params 一维残差网络定义
class ODResnet(torch.nn.Module):
    def __init__(self, Input_channel=6, class_num=5):
        super(ODResnet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(Input_channel, 64, kernel_size=3, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 128, 512, True),


            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512, class_num)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [32, 400, 6] --> [32, 6, 400]
        x = self.features(x)    # [32, 6, 400] --> [32, 256, 1]
        # print('ODResNet后数据维度：')
        # print({x.shape})
        x = x.view(-1, 512)
        x = self.classifer(x)
        return x


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
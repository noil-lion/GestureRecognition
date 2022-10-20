# CBAM 通道注意力和空间注意力集成者
Convolution Block Attention Module

## 注意力机制
可以看作是对图像输入重要信息的动态选择过程，通过对特征自适应权重实现。

通道注意力：聚焦于what to pay attention to
空间注意力机制：where to pay attention to
时间注意力机制：when to pay attention
分支注意力机制：which to pay attention

### 混合注意力机制
通道和空间注意力机制
空间和时间注意力机制

## CBAM：通道和空间注意力的一种
CBAM模块序列化地在通道和空间两个维度上产生注意力特征图信息，CNN极强的特征表示能力提升了网络的性能，为进一步增强特征表达能力，研究者主要从网络深度、宽度、维度进行研究。  
除了网络结构的改进，注意力机制主要目的是聚焦数据的重要特征，抑制不必要的区域响应。通过对数据的通道维度和空间维度组合分析研究，提出CBAM模块。CBAM证实了网络性能的提升源自于精确的注意力机制和无关噪声的抑制。

## CBAM流程
1. CBAM模块处理主干网络生成的特征图，分别产生1D通道注意力特征图Mc<特征图通道维度上的一维向量Cx1x1>，2D空间注意力特征图Ms<特征图横切维度下的二维矩阵1xWxH>
2. F'=Mc(F)点乘F
3. F'' = Ms(F')点乘F'

## Channel Attention Module
通道注意力机制通过捕获特征图各通道内部的关系，来找到图像中有什么特征是需要聚焦的， 特征图中的每个通道都可以看作是一个特征检测器。

为了更高效地计算通道注意力特征，要做的就是压缩特征图的空间维度（其实就是将特征图的长宽维度进行压缩，通道维度上尽量保持不变），具体实现采用的是平均池化方法和最大池化方法，平均池化特征图学习到目标物体的程度信息，作为压缩信息的一部分，最大池化特征能够在压缩后保留物体的判别性特征。[在长宽维度上对多各通道的特征图使用两种池化分别进行压缩，形成两条一维向量，分别代表各通道特征的学习特征的程度和学习特征的可区分度]，在经过池化操作后的结果作为输入，进入一个共享的多层感知机网络，输出为两个等维向量，进行concrete后经过relu，得到最终输出的通道注意力特征图（特征向量）。
[!var](pic/CAC.png)


## Spatial Attention Module
不同于通道注意力关注于各通道特征图的内部关系，空间注意力更聚焦于长宽维度层面的有效信息，通过压缩特征图的通道维度，也是使用平均池化和最大池化，最后产生压缩生成两个通道的相同长宽的二维特征图，我们将结果进行拼接成为双通道的特征图，再使用卷积操作，产生最后的空间注意力特征图长宽不变。
[!var](pic/SAC.png)

## 调参分析

1. 最大池化和平均池化最好都用上，最大池化捕捉到的全局信息能弥补平均池化捕获的局部信息
2. 在空间注意力特征卷积方面，使用更大的卷积核尺寸对提升效果有益处。
3. 两种不同的注意力，各自的作用不同，但是序列化地使用两种注意力机制，要比并行化使用效果要好，一般是先使用通道注意力后使用空间注意力。

## code

```
#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        #MLP  除以16是降维系数
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #结果相加
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        #声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  #平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) #最大池化
        #拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) #7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)
```
Link: https://zhuanlan.zhihu.com/p/510223283

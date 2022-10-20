import torch
from torch import nn
import torch.nn.functional as F
'''
CNN即插即用小模块
'''


class SpatialTransformer(nn.Module):
    '''
    STN模块，显式将空间变换植入到网络当中，进而提高网络的旋转、平移、尺度等不变性。可以理解为“对齐”操作。STN的结构如上图所示，
    每一个STN模块由Localisation net，Grid generator和Sampler三部分组成。Localisation net用于学习获取空间变换的参数，
    就是上式中的六个参数。Grid generator用于坐标映射。Sampler用于像素的采集，是利用双线性插值的方式进行。
    '''
    def __init__(self, spatial_dims):
        super(SpatialTransformer, self).__init__()
        self._in_ch, self._h, self._w = spatial_dims
        print(self._in_ch, self._h, self._w)
        self.fc1 = nn.Linear(32*3*3, 120)  # 可根据自己的网络参数具体设置
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        batch_images = x  # 保存一份原始数据
        x = x.view(-1, 32*3*3)
        # 利用FC结构学习到6个参数
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 3)  # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points)
        return rois, affine_grid_points


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

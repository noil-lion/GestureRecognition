# -*- coding: utf-8 -*-

import os
from random import sample
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


class CustomDataset(Dataset):
    '''自定义数据集：数据形式为时间序列数据，单个样本由滑动窗口生成，共400时间步的传感器各轴数据构成，每个时间步为6维张量，单个样本放置在二维可看作400行*6列，由torch.tensor构成'''
    def __init__(self, annotations_file, sample_dir, sample_size,
                 tensor_shape, mode):
        self.sample_dir = sample_dir   # 样本路径
        self.annotations_file = annotations_file
        self.labels = self.read_file()  # 读取label
        self.sample_list = np.loadtxt(self.sample_dir)

        self.sample_size = sample_size    # 样本size
        self.tensor_shape = tensor_shape  # 张量维度
        self.mode = mode  # 训练或者测试
        self.transfer = MinMaxScaler(copy=True, feature_range=[-1, 1])
        self.standard = StandardScaler()

    def __len__(self):
        return len(self.labels)  # 样本数总长度

    # 重写类方法-迭代调用根据索引从文件中读取数据，对数据进行预处理，返回样本数据和对应标签
    def __getitem__(self, idx):   # idx的值是在0-_len_之间的
        # sample_path = os.path.join(self.sample_dir)
        # print(idx)                                    样本index测试
        # print(self.sample_list[idx].shape)            样本维度验证
        sample_tensor = self.transform(np.array(self.sample_list[idx]), self.sample_size,
                                       self.tensor_shape)     # 将样本转为Tensor
        label = self.target_transform(self.labels[idx])       # 将label向量转为张量
        return sample_tensor, label                           # 返回样本和标签

    # 读样本lable函数
    def read_file(self):
        label_list = list()
        with open(self.annotations_file, 'r') as h:
            while True:
                line = h.readline()
                if not line:
                    break
                label_list.append(
                    line.strip())
        return label_list

    # label映射，从动作名映射成numpy向量
    def target_transform(self, label):
        if self.mode == "train":
            label_num_mapping = {"init": 0, "qbcy": 1, "qbxq": 2, "szcs": 3, "szqs": 4}    # 标签到num的映射，有点像独热编码？
        if self.mode == "test":
            label_num_mapping = {"init": [1, 0, 0, 0, 0], "qbcy": [0, 1, 0, 0, 0], "qbxq": [0, 0, 1, 0, 0], "szcs": [0, 0, 0, 1, 0], "szqs": [0, 0, 0, 0, 1]}    # 标签到num的映射，有点像独热编码？
        return torch.tensor(label_num_mapping[label])

    # 添加高斯噪声content:原数据，snr:信噪比
    def add_nosie(self, content, snr):
        Ps = np.sum(abs(content)**2)/len(content)
        Pn = Ps/(10**((snr/10)))
        noise = np.random.randn(len(content)) * np.sqrt(Pn)
        content += noise
        return content

    # 样本填充等长化、张量化
    def transform(self, content, sample_size, tensor_shape):
        if content.size < sample_size:
            padding_len = sample_size - content.size
            content = np.hstack(
                (content, np.zeros(padding_len, dtype=np.uint8)))  # padding到统一长度
        elif content.size > sample_size:
            content = content[0:sample_size]   # 截取到统一长度

        content = torch.reshape(torch.tensor(content).type(torch.float), tensor_shape)

        # print(content[0])
        # content = self.transfer.fit_transform(content)   # 最大最小归一化
        # print(content[0])
        # content = self.standard.fit_transform(content)   # 标准化
        # content = self.transfer.transform(content)
        # print(content.shape)
        return content   # tensor化

# 自定义数据集测试器
"""CD = CustomDataset('labels.csv', 'samples.csv', 2400, (400, 6) )
train_loader2 = DataLoader(dataset=CD,batch_size=32,shuffle=False)
for epoch in range(1):
    for i, data in enumerate(train_loader2):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, labels = data

        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)

        # 接下来就是跑模型的环节了，我们这里使用print来代替
        print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())"""
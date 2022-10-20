# -*- coding: utf-8 -*-

import os
# from dataloader import dataLoader
from random import sample
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
# from torch.autograd import Variable


class CustomDataset(Dataset):
    '''自定义数据集：数据形式为时间序列数据，单个样本由滑动窗口生成，共400时间步的传感器各轴数据构成，每个时间步为6维张量，单个样本放置在二维可看作400行*6列，由torch.tensor构成'''
    def __init__(self, annotations_file, sample_dir, sample_size,
                 tensor_shape):
        self.sample_dir = sample_dir   # 样本路径
        self.annotations_file = annotations_file
        self.labels = self.read_file()  # 读取label
        self.sample_list = np.loadtxt(self.sample_dir)

        self.sample_size = sample_size  # 样本size
        self.tensor_shape = tensor_shape  # 张量维度

    def __len__(self):
        return len(self.labels)  # 样本数总长度

    def __getitem__(self, idx):   # idx的值是在0-_len_之间的
        sample_path = os.path.join(self.sample_dir)
        # print(idx)
        # print(np.array(self.sample_list[idx]).shape)
        # print(self.sample_list[idx].shape)
        sample_tensor = self.transform(np.array(self.sample_list[idx]), self.sample_size,
                                       self.tensor_shape)     # 将样本转为Tensor
        label = self.labels[idx]

        label = torch.tensor(self.target_transform(label))
        return sample_tensor, label

    def read_file(self):   # 读样本函数
        label_list = list()
        with open(self.annotations_file, 'r') as h:
            while True:
                line = h.readline()
                if not line:
                    break
                label_list.append(
                    line.strip())
        return label_list

    def target_transform(self, label):
        label_num_mapping = {"init": [1, 0, 0, 0, 0], "qbcy": [0, 1, 0, 0, 0], "qbxq": [0, 0, 1, 0, 0], "szcs": [0, 0, 0, 1, 0], "szqs": [0, 0, 0, 0, 1]}    # 标签到num的映射，有点像独热编码？
        return label_num_mapping[label]

    def transform(self, content, sample_size, tensor_shape):
        if content.size < sample_size:
            padding_len = sample_size - content.size
            content = np.hstack(
                (content, np.zeros(padding_len, dtype=np.uint8)))  # padding到统一长度
        elif content.size > sample_size:
            content = content[0:sample_size]   # 截取到统一长度

        return torch.reshape(torch.tensor(content).type(torch.float), tensor_shape)   # tensor化


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
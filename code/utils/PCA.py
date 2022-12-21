from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
# from torch.nn.functional import normalize
import sys
sys.path.append('../../code/')
from utils.CustomDataset import CustomDataset  # 导入自定义数据集
from model.CNN_LSTM_Cust.CNNLSTM import CNNLSTM
from model.Resnet.ODResnetC import ODResnet

# #-----------------------------------variables-----------------------------------------------#
# batch size
BATCH_SIZE = 32
epoch = 128     # epoch
momentum = 0.9
num_classes = 5
batch_size = 32
lr = 0.001
in_channels = 6
out_channels = 12
seq_length = 400
hidden_size = 48
num_layers = 2
k_split_value = 5
dir_acc = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/DataAcc/'
dir_test = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/DatasetSlide/'
sample_dir = 'samples.csv'
label_dir = 'labels.csv'
PATH = '../weight/ODResnet_opt3.pth'
# 无GPU跑cpu
device = torch.device("cuda:0")
# 实例化网络
net = ODResnet(in_channels, class_num=num_classes)
if torch.cuda.is_available():
    net = net.cuda()
net.load_state_dict(torch.load(PATH))  # 加载权重参数

# 加载自定义训练数据集
CD = CustomDataset(dir_test+label_dir, dir_test+sample_dir, 2400, (400, 6), 'train')
Train_Dataloader = DataLoader(dataset=CD, batch_size=len(CD), shuffle=False)
CDA = CustomDataset(dir_acc+label_dir, dir_acc+sample_dir, 1200, (400, 3), 'train')
ACC_Dataloader = DataLoader(dataset=CDA, batch_size=len(CDA), shuffle=False)
with torch.no_grad():
    for batch, data in enumerate(Train_Dataloader):
        inputs, labels = data
        # inputs, labels = Variable(inputs), Variable(labels)
        # labels = labels.float()
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        # Compute prediction and loss
        # pred = net(inputs).cpu()
    stock_data = inputs
    for batch, data in enumerate(ACC_Dataloader):
        inputsA, labelsA = data
    ACC_data = inputsA
sample_size = 400  # 样本序列长度
idx = np.array(range(0, len(stock_data)))  # 随机排序取样，返回一个list，为样本在原数据集的下标
print(idx)
real_sample = np.asarray(stock_data.data)[idx]   # 根据下标获取样本
idx = np.array(range(0, len(ACC_data)))
ACC_sample = np.asarray(ACC_data)[idx]
print(real_sample.shape)
# for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA.
synth_data_reduced = real_sample.reshape(len(CD), -1)
ACC_data_reduced = ACC_sample.reshape(len(CDA), -1)
print(synth_data_reduced.shape)
n_components = 2
pca = PCA(n_components=n_components)  # 实例化PCA

pca.fit(synth_data_reduced)
pca_real = pd.DataFrame(pca.transform(synth_data_reduced))
pca.fit(ACC_data_reduced)
pca_ACC = pd.DataFrame(pca.transform(ACC_data_reduced))
print(pca_real.shape)
fig = plt.figure(constrained_layout=True, figsize=(20, 10))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
# kmeans = KMeans(n_clusters=5)  # 设定初始质心数
# kmeans.fit(pca_real)  # 将降维后的数据进行聚类训练
# y = kmeans.predict(pca_real)  # 预测聚类结果
# kmeans.fit(pca_ACC)  # 将降维后的数据进行聚类训练
# yA = kmeans.predict(pca_ACC)  # 预测聚类结果
print(labels)
# print(y)
print(labelsA)
# print(yA)
# print(labels)
# PCA scatter plot
ax = fig.add_subplot(spec[0, 0])
ax.set_title('PCA results',
             fontsize=20,
             color='red',
             pad=10)

plt.scatter(pca_real.iloc[:, 0], pca_real.iloc[:, 1], c=labels, label='Original')
ax.legend()
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_title('PCA_Acc results',
              fontsize=20,
              color='black',
              pad=10)
plt.scatter(pca_ACC.iloc[:, 0], pca_ACC.iloc[:, 1], c=labelsA, label='Synthetic')

ax2.legend()
plt.show()

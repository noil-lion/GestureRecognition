import sys

sys.path.append('../../code/')
from traceback import print_tb
from utils.loadData import loadData as LD
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # 导入Kmeans 与PCA模块
# from utils import vision
import scipy.signal as signal


# 构建时序信号的特征向量样本矩阵。
class dataLoader(object):
    def __init__(self, path):
        self.path = path
        self.loader = LD(self.path)

        self.listacc_X, self.listacc_Y, self.listacc_Z, self.listgyo_X, self.listgyo_Y, self.listgyo_Z, self.listagl_X, self.listagl_Y, self.listagl_Z= self.loader.get_AGA()
        """a_listacc_X, a_listacc_Y, a_listacc_Z = filters.LowPass(
            listacc_X, listacc_Y, listacc_Z)
        self.listacc_X, self.listacc_Y, self.listacc_Z, self.listgyo_X, self.listgyo_Y, self.listgyo_Z, self.listagl_X, self.listagl_Y, self.listagl_Z = a_listacc_X, a_listacc_Y, a_listacc_Z, listgyo_X, listgyo_Y, listgyo_Z, listagl_X, listagl_Y, listagl_Z
        """
        self.timesteps = self.loader.get_timestep(len(self.listacc_X))

    def sample_generator(self, stride, size, mode, listacc_X, listacc_Y, listacc_Z):
        self.samples = []
        if self.path == None:
            return -1
        elif mode == 'slide':
            for i in range(0, int((len(self.timesteps)-size)/stride)+1):
                sample = [listacc_X[i*size:(i+1)*size], listacc_Y[i*size:(i+1)*size], listacc_Z[i*size:(i+1)*size]]
                self.samples.append(sample)
        elif mode == 'static':
            for i in range(0, int((len(self.timesteps)-size)/stride)+1):
                sample = [sum(listacc_X[i*size:(i+1)*size])/size, sum(listacc_Y[i*size:(i+1)*size])/size, sum(listacc_Z[i*size:(i+1)*size])/size]
                self.samples.append(sample)
        elif mode == 'full':
            for k in range(0, len(self.timesteps)-1):
                sample = [listacc_X[k], listacc_Y[k], listacc_Z[k]]
                self.samples.append(sample)
        return self.samples
    # PCA聚类ZUPT尝试，效果一般
    def detect(self, n_clusters, n_components, listacc_X, listacc_Y, listacc_Z):
        samples = self.sample_generator(10, 10, 'slide', listacc_X, listacc_Y, listacc_Z)
        ss = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        Data = []
        for i in range(0, len(samples)):
            list = ss.fit_transform(np.array(samples[i]).reshape(-1, 1))
            meta = []
            for k in range(0, len(list)):
                meta.append(list[k][0])
            Data.append(meta)
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters)  # 设定初始质心数
        pca = PCA(n_components)  # 设定降维数
        pca.fit(Data)  # 训练数据
        data1_pca = pca.transform(Data)  # 进行PCA降维
        # 查看降维后数据
        kmeans.fit(data1_pca)  # 将降维后的数据进行聚类训练
        y = kmeans.predict(data1_pca)  # 预测聚类结果
        y = signal.medfilt(y)
        # vision.draw_satic(listacc_Z[0:1000], y[0:1000], self.timesteps[0:1000])
        return y

    def generate_samples(self, stride, size, mode):
        self.samples = []
        if self.path == None:
            return -1
        elif mode == 'slide':
            for i in range(0, int((len(self.timesteps)-size)/stride)+1):
                sample = []
                for k in range(i*stride, i*stride+400):
                    sample.append([self.listacc_X[k], self.listacc_Y[k], self.listacc_Z[k], self.listgyo_X[k], self.listgyo_Y[k], self.listgyo_Z[k]])

                self.samples.append(sample)
        elif mode == 'full':
            for k in range(0, len(self.timesteps)-1):
                sample = [self.listacc_X[k], self.listacc_Y[k], self.listacc_Z[k], self.listgyo_X[k], self.listgyo_Y[k], self.listgyo_Z[k]]
                self.samples.append(sample)
        return self.samples

#  样本生产器
import numpy as np
import csv


def itera(path, label):
    path = path
    DL = dataLoader(path)
    samples = DL.generate_samples(200, 400, 'slide')

    label = label
    with open('D:/ResearchSpace/task/gestureRecognition/data/test/labels.csv', 'a') as f:
        for i in range(0, len(samples)):
            f.write(label)
    f.close()

    with open('D:/ResearchSpace/task/gestureRecognition/data/test/samples.csv', 'a') as f:
        for i in samples:
            sample = [np.array(i).flatten()]
            np.savetxt(f, sample, fmt='%.4f')

    f.close()


from os import listdir


source_dir = "D:/ResearchSpace/task/gestureRecognition/data/test/szcs/"
label = str('szcs\n')
list_dir = listdir(source_dir)
for i in range(0, len(list_dir)):
    itera(source_dir+'\\'+list_dir[i], label)

"""def vision():
    # Set up a logs directory, so Tensorboard knows where to look for files
    log_dir='logs/test/'
    feature_vectors =np.loadtxt('features.txt')
    weights=tf. Variable(feature_vectors)
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint=tf. train. Checkpoint(embedding=weights)
    checkpoint.save(os.path. join(log_dir,"embedding.ckpt"))
    # Set up config
    config=projector.ProjectorConfig()
    embedding=config.embeddings.add()
    # The name of the tensor will be suffixed by/. ATTRIBUTES/VARIABLE VALUE'
    embedding.tensor_name="embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path='metadata.tsv'
    projector.visualize_embeddings(log_dir, config)"""


"""
if __name__=='__main__':
    PATH = '211111140301.txt'
    DL = dataLoader(PATH)
    samples = DL.sample_generator(stride=50, size=300, mode='full')
    ss = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    list = ss.fit_transform(np.array(samples).reshape(-1, 1))#数据标准化
    import codecs
    f = codecs.open("features.txt",'w','utf-8')
    #f.write(str(list))
    for i in range(3000,5500):
        list = ss.fit_transform(np.array(samples[i]).reshape(-1, 1))
        for k in range(0, len(list)):
            f.write(str(list[k][0])+'\t')
        f.write('\n')  #\r\n为换行符
    f.close()
    Data = []
    for i in range(0,len(samples)):
        list = ss.fit_transform(np.array(samples[i]).reshape(-1, 1))
        meta = []
        for k in range(0, len(list)):
            meta.append(list[k][0])
        Data.append(meta)
    print(np.array(Data).shape)
    print(len(Data), len(Data[0]))
    kmeans = KMeans(n_clusters=2) # 设定初始质心数
    pca = PCA(n_components=3) # 设定降维数
    pca.fit(Data) # 训练数据
    data1_pca = pca.transform(Data) # 进行PCA降维
    data1_pca # 查看降维后数据
    kmeans.fit(data1_pca) # 将降维后的数据进行聚类训练
    y = kmeans.predict(data1_pca) # 预测聚类结果
    vision.draw_satic(DL.listacc_Y, y,DL.timesteps)
    plt.scatter(data1_pca[:,0],data1_pca[:,1],c = y) # 将聚类结果可视化
    plt.show()
    # vision()"""

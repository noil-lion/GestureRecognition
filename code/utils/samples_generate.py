import sys
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理器
sys.path.append('../../code/')
from utils.loadData import loadData as LD
from utils import vision


# 构建时序信号的特征向量样本矩阵，这里定义数据样本生成类。
# parmas：path-原始数据文件路径。
class dataLoader(object):
    def __init__(self, path):
        self.path = path
        # 创建数据加载器对象
        self.loader = LD(self.path)
        # 从路径中读取全部数据
        self.listacc_X, self.listacc_Y, self.listacc_Z, self.listgyo_X, self.listgyo_Y, self.listgyo_Z, self.listagl_X, self.listagl_Y, self.listagl_Z = self.loader.get_AGA()
        # 获取数据文件的总时间步长
        self.timesteps = self.loader.get_timestep(len(self.listacc_X))

    # 定制化样本生成函数
    def sample_generator(self, stride, size, mode, listacc_X, listacc_Y, listacc_Z):
        self.samples = []
        if self.path is None:
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

    # 添加高斯噪声content:原数据，snr:信噪比
    def add_nosie(self, content, snr=90):
        for i in range(0, content.shape[1]):
            Ps = np.sum(abs(content[:, i])**2)/len(content[:, i])
            Pn = Ps/(10**((snr/10)))
            noise = np.random.randn(len(content[:, i])) * np.sqrt(Pn)
            content[:, i] += noise
        return content

    # 常规生成样本函数
    def generate_samples(self, stride, size, mode):
        self.samples = []
        if self.path is None:
            return -1
        elif mode == 'slide':
            for i in range(0, int((len(self.timesteps)-size)/stride)+1):
                sample = []
                for k in range(i*stride, i*stride+size):
                    sample.append([self.listacc_X[k], self.listacc_Y[k], self.listacc_Z[k], self.listgyo_X[k], self.listgyo_Y[k], self.listgyo_Z[k], self.listagl_X[k], self.listagl_Y[k], self.listagl_Z[k]])
                self.samples.append(np.array(sample))
        elif mode == 'slide_acc':
            for i in range(0, int((len(self.timesteps)-size)/stride)+1):
                sample = []
                for k in range(i*stride, i*stride+size):
                    sample.append([self.listacc_X[k], self.listacc_Y[k], self.listacc_Z[k]])
                self.samples.append(sample)
        elif mode == 'full':
            for k in range(0, len(self.timesteps)-1):
                sample = [self.listacc_X[k], self.listacc_Y[k], self.listacc_Z[k], self.listgyo_X[k], self.listgyo_Y[k], self.listgyo_Z[k]]
                self.samples.append(sample)
        return self.samples

    def visualize_series(self):
        ln1, = plt.plot(self.timesteps, self.listacc_X, color='limegreen', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(self.timesteps, self.listacc_Y, color='burlywood', linewidth=2.0, linestyle='-')
        ln3, = plt.plot(self.timesteps, self.listacc_Z, color='cornflowerblue', linewidth=2.0, linestyle='-')
        plt.title("sensor data change")  # 设置标题及字体
        plt.legend(handles=[ln1, ln2, ln3], labels=['acc_X', 'acc_Y', 'acc_Z'])
        ax = plt.gca()
        plt.grid(ls='-', color='whitesmoke', linewidth=1.0)
        ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
        ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
        plt.xlabel("timeStep/ms")
        plt.ylabel("Triaxial component of acceleration(m/s²)")
        plt.show()


#  写入函数
def record(labelpath, samplepath, label, samples):
    with open(labelpath, 'a') as f:
        for i in range(0, len(samples)):
            f.write(label)
    f.close()
    with open(samplepath, 'a') as f:
        for i in samples:
            sample = [np.array(i).flatten()]
            np.savetxt(f, sample, fmt='%.4f')
    f.close()


dataDir = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/train/raw/'            # 原始数据根目录，文件结构[dataDir[init[file1.txt, file2.txt,...],qbcy[file1.txt, file2.txt,...],...]]
labelpath = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/train/labels.csv'    # 标签存储文件路径-标签与子文件夹同名表示类别
samplepath = 'D:/ResearchSpace/task/gestureRecognition/GestureRecognition/data/train/samples.csv'  # 样本存储文件路径-400*6
fold_list = listdir(dataDir)
for i in fold_list:
    label = str(i+'\n')
    # print("this is {}".format(label))
    file_path_list = listdir(dataDir+i)
    for k in file_path_list:
        file_path = dataDir+i+'/'+k
        DL = dataLoader(file_path)                                          # 原始数据文件路径创建数据样本生成对象实例
        # DL.visualize_series()
        samples = DL.generate_samples(stride=200, size=400, mode='slide')   # params：stride-滑动步长， size-样本序列长度, mode-生成模式，默认为滑动窗口
        record(labelpath, samplepath, label, samples)                       # 写入样本和对应标签


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

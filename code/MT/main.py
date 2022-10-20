import sys
import math
sys.path.append('../../code/')
from utils.loadData import loadData
import utils.filters as filters
import utils.vision as vision
import utils.dataProcess as dataProcess
from os import listdir
import Coordinate_Transform
import Trajectory_calculation as TC
import generate_pic as GP
import animation


def main(dir):
    # 数据文件目录
    # dir= "../Data/qbqq/210618232419.txt"
    # 原始数据加载
    LD = loadData(dir)
    listacc_X, listacc_Y, listacc_Z, listgyo_X, listgyo_Y, listgyo_Z, listangle_R, listangle_P, listangle_Y = LD.get_AGA()

    # vision.draw_acc(listangle_R, listangle_P, listangle_Y, timestep)

    # 数据预处理:去除原始数据中的异常测量值，输出数据包括系统测量噪声，且噪声主要为去除高频信号，为了减少信号的干扰噪声，须对信号进行滤波。对加速度信号使用低通滤波器滤除数据中的噪声。
    listacc_X = dataProcess.abnormal_discard(listacc_X, 0.9)
    listacc_Y = dataProcess.abnormal_discard(listacc_Y, 0.9)
    listacc_Z = dataProcess.abnormal_discard(listacc_Z, 0.9)
    a_listacc_X, a_listacc_Y, a_listacc_Z = filters.LowPass(
        listacc_X, listacc_Y, listacc_Z)
    # vision.comparison(listangle_R, listangle_P, listangle_Y, N_listangle_R, N_listangle_P, N_listangle_Y, timestep)
    timestep = LD.get_timestep(len(listacc_X))
    # 可视化，对比数据处理器前后差异
    # vision.comparison(listacc_X, listacc_Y, listacc_Z, a_listacc_X, a_listacc_Y, a_listacc_Z, timestep)
    # 初始坐标系校准
    # ---采集时校准：载体坐标系与导航坐标系对齐，且初始点坐标原点重合
    # 坐标系转换
    for i in range(0, len(timestep)):
        listacc_X[i], listacc_Y[i], listacc_Z[
            i] = Coordinate_Transform.transform(a_listacc_X[i], a_listacc_Y[i],
                                                a_listacc_Z[i], listangle_R[i],
                                                listangle_P[i], listangle_Y[i])
    # 坐标系转换前后对比
    # vision.comparison(a_listacc_X, a_listacc_Y,  a_listacc_Z, listacc_X, listacc_Y, listacc_Z,timestep)
    # 重力消去,统一单位m/s^2
    G = 9.8  # 重力加速度常量
    raw_listacc_X = []
    raw_listacc_Y = []
    raw_listacc_Z = []
    real_listacc_X = []
    real_listacc_Y = []
    real_listacc_Z = []
    for i in range(0, len(timestep)):
        raw_listacc_X.append(listacc_X[i] * G)
        raw_listacc_Y.append(listacc_Y[i] * G)
        raw_listacc_Z.append((listacc_Z[i]) * G)
        real_listacc_X.append((listacc_X[i]) * G)
        real_listacc_Y.append((listacc_Y[i]) * G)
        real_listacc_Z.append(((listacc_Z[i]) - 1.00) * G)
    # 计算传感器3D空间运动轨迹
    # DL = dataLoader(dir)
    # indexZero_V = DL.detect(2, 1, real_listacc_X, real_listacc_Y, real_listacc_Z)  # 零速时刻坐标list集合
    # print(indexZero_V[0:1000])
    Tra_X, Tra_Y, Tra_Z, V_X, V_Y, V_Z = TC.Trajectory_cal(real_listacc_X, real_listacc_Y, real_listacc_Z, a_listacc_Y, timestep)
    print(math.sqrt(Tra_X[-2]*Tra_X[-2]+Tra_Y[-2]*Tra_Y[-2]+Tra_Z[-2]*Tra_Z[-2]))
    vision.draw_Volocity(V_X, timestep)
    vision.Trajectory_simulation(Tra_X, Tra_Y, Tra_Z, 'black')
    animation.trajectory_simulation(Tra_X, Tra_Y, Tra_Z, timestep)
    """window = 400
    stride = 20
    # 滑窗全投影
    i = 0
    while i < int((len(timestep)-window)/stride)-1:
    # 数据集拆分
        i = i+1
        if i /int((len(timestep)-window)/stride) <0.6:
            GP.pic(listacc_X[i*stride: (i+1)*stride+window], listacc_Y[i*stride: (i+1)*stride+window], listacc_Z[i*stride: (i+1)*stride+window], str(i), dir)
        if i /int((len(timestep)-window)/stride) > 0.6 and i /int((len(timestep)-window)/stride)< 0.8:
            dir= dir[0:29]+'test'+dir[34:]
            GP.pic(listacc_X[i*stride: (i+1)*stride+window], listacc_Y[i*stride: (i+1)*stride+window], listacc_Z[i*stride: (i+1)*stride+window],str(i),dir)
        if i /int((len(timestep)-window)/stride) > 0.8:
            dir= dir[0:29]+'valid'+dir[34:]
            GP.pic(listacc_X[i*stride: (i+1)*stride+window], listacc_Y[i*stride: (i+1)*stride+window], listacc_Z[i*stride: (i+1)*stride+window],str(i),dir)"""


if __name__ == '__main__':
    dir = 'D:/ResearchSpace/task/gestureRecognition/data/all/szqs.txt'
    main('D:/ResearchSpace/task/gestureRecognition/data/all/211113155218_shixiong_qbcy.txt')
    """source_dir = "D:\\ResearchSpace\\TFL\\dataRaw\\train\\"
    list_dir = listdir(source_dir)
    for i in range(0, len(list_dir)):
        dir_t = listdir(source_dir+list_dir[i])
        main(source_dir+list_dir[i]+'\\'+dir_t[0], str(list_dir[i]))"""

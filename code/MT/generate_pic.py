from operator import imod
import cv2
import numpy as np
import math


def vec(V_X, V_Y, V_Z):
    V = []
    L = len(V_X)
    for i in range(0, L):
        V.append(math.sqrt(V_X[i]*V_X[i] + V_Y[i]*V_Y[i] + V_Z[i]*V_Z[i]))
    return V


# TEM编码
def get_pic(Tra_X, Tra_Y, Tra_Z, V_X, V_Y, V_Z, listangle_R, listangle_P, listangle_Y, dir, T, k, postition):
    '''
    Tra_X, Tra_Y, Tra_Z: 空间位置坐标
    V_X, V_Y, V_Z： 运动速度
    listangle_R, listangle_P, listangle_Y： 角度信息
    dir: 存放目录
    positon: 投影方向

    执行图像生成
    '''
    V = vec(V_X, V_Y, V_Z)   # 模值
    A = vec(listangle_R, listangle_P, listangle_Y)

    # print(max(V))
    # plt.scatter(Tra_X, Tra_Z, c=Tra_Y, cmap=plt.cm.winter, linewidth=1.0, linestyle='-')
    # plt.ylim(ymin=-1, ymax=1)
    # plt.xlim(xmin=-1, xmax=1)
    # plt.savefig(dir[:-4]+'testblueline'+k+'.jpg')
    # plt.show()
    high = len(Tra_Z)
    img = np.zeros([128, 128, 3], dtype="uint8")
    if postition == 'L':
        Tra_Y = Tra_Y
    if postition == 'M':
        Tra_Y = Tra_X
    if postition == 'V':
        Tra_Z = Tra_X
    for i in range(1, high):
        i = high - i
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 0] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 1] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 2] = 255  # blue
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 0] = V[i]/(max(V)+0.001)*255  # blue
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 1] = (1-(i/high))*255  # green
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 2] = A[i]/(max(A)+0.001)*255  # red
    # cv2.imshow("img", img)
    img = cv2.flip(img, 0)
    cv2.imwrite(dir[:-4]+k+".jpg", img)
    cv2.waitKey()


# 原始加速度计数据拼接
def pic(acc_X, acc_Y, acc_Z, k, dir):
    acc_X = acc_X[0:385]
    acc_Y = acc_Y[0:385]
    acc_Z = acc_Z[0:386]
    acc = acc_X
    acc.extend(acc_Y)
    acc.extend(acc_Z)

    # print(max(V))
    # plt.scatter(Tra_X, Tra_Z, c=Tra_Y, cmap=plt.cm.winter, linewidth=1.0, linestyle='-')
    # plt.ylim(ymin=-1, ymax=1)
    # plt.xlim(xmin=-1, xmax=1)
    # plt.savefig(dir[:-4]+'testblueline'+k+'.jpg')
    # plt.show()
    high = 1156
    img = np.zeros([1156, 1, 3], dtype="uint8")
    for i in range(0, high-1):
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 0] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 1] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 2] = 255  # blue
        img[i, 0, 0] = int(float(acc[i]/9.8)*255)
        img[i, 0, 1] = int(float(acc[i]/9.8)*255)
        img[i, 0, 2] = int(float(acc[i]/9.8)*255)
    img = img.reshape([34, 34, 3])
    # cv2.imshow("img", img)
    img = cv2.flip(img, 0)
    cv2.imwrite(dir[:-4]+k+".jpg", img)


# 实时数据写入
def real_time_pic(Tra_X, Tra_Y, Tra_Z, V_X, V_Y, V_Z, listangle_R, listangle_P, listangle_Y, postition):
    V = vec(V_X, V_Y, V_Z)
    A = vec(listangle_R, listangle_P, listangle_Y)

    # print(max(V))
    # plt.scatter(Tra_X, Tra_Z, c=Tra_Y, cmap=plt.cm.winter, linewidth=1.0, linestyle='-')
    # plt.ylim(ymin=-1, ymax=1)
    # plt.xlim(xmin=-1, xmax=1)
    # plt.savefig(dir[:-4]+'testblueline'+k+'.jpg')
    # plt.show()
    high = len(Tra_Z)
    img = np.zeros([128, 128, 3], dtype="uint8")
    if postition == 'L':
        Tra_Y = Tra_Y
    if postition == 'M':
        Tra_Y = Tra_X
    if postition == 'V':
        Tra_Z = Tra_X
    for i in range(1, high):
        i = high - i
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 0] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 1] = 255  # blue
        # img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 2] = 255  # blue
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 0] = V[i]/(max(V)+0.001)*255  # blue
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 1] = (1-(i/high))*255  # green
        img[(int(Tra_Z[i]*100)+62):(int(Tra_Z[i]*100)+66), (int(Tra_Y[i]*100)+62):(int(Tra_Y[i]*100)+66), 2] = A[i]/(max(A)+0.001)*255  # red
    # cv2.imshow("img", img)
    img = cv2.flip(img, 0)
    cv2.imwrite('D:\\ResearchSpace\\TFL\\Fes\\note\\test'+postition+".jpg", img)
    cv2.waitKey()

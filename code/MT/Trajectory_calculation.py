from utils import filters
import math
import Zero_Velocity_Update as ZVU
from plotlib import plot3D
import numpy as np

# 轨迹解算主要包括：1.数值积分（V） + ZUPT + 低通滤波(V)[三轴]，以解算得到趋近真实的运动速度。
# 2.数值积分（C） + 初始点对齐 + 低通滤波（C）[三轴]，解算得到连续运动空间位置，代表上肢运动的3D空间信息。


def integral(listacc_X, listacc_Y, listacc_Z, RlistaccY, timestep):
    V = []
    S = []
    S.append(0)
    for i in range(0, 10):
        V.append(0)
    for i in range(10, len(timestep)-20):
        if ZVU.detect(listacc_X, listacc_Y, listacc_Z, i):
            V.append(0)
        else:
            V.append(V[i-1] + ((listacc_X[i]+listacc_X[i-1])*0.01/2.0))
    V = filters.lowpass(V)
    for k in range(1, len(timestep)-20):
        if V[k] == 0 and abs(RlistaccY[k]) < 1 and math.acos(abs(RlistaccY[k])) < 0.1:   # 零速检测+姿态追踪，reset周期动作初始点
            S.append(0)
        else:
            S.append(S[k-1] + ((V[k]+V[k-1])*0.01/2.0))
    C = filters.lowpass(S)
    return C, V


def Trajectory_cal(listacc_X, listacc_Y, listacc_Z, RlistaccY, timestep):
    '''
    Simple integration of acc data and velocity data.

    @param listacc_X, listacc_Y, listacc_Z: acc data
    @param RlistaccY,: raw acc data of Y axis

    Return: 3D coordinates in navigation frame
    '''
    a_nav = np.vstack((listacc_X, listacc_Y, listacc_Z)).T
    # Acceleration correction step
    a_nav_filtered = ZVU.removeAccErr(a_nav, filter=False)
    # # ZUPT step
    v = ZVU.zupt(a_nav_filtered, threshold=0.2)
    # Integration Step
    p = ZVU.positionTrack(a_nav_filtered, RlistaccY, v, threshold=0.4)
    plot3D([[p, 'position']])
    return p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2]

import numpy as np
import math
from mathlib import filtSignal


# 基于滑动窗口的零速检测
def detect(listacc_X, listacc_Y, listacc_Z, time):
    '''
        零速状态下窗口加速度均值Z,取窗口大小为10
        # 零速状态时刻前后窗口加速度值的特征差异
    '''
    """sum = 0
    for i in range(time-10, time):
        sum += listacc_X[i]
    before_ave = sum/10
    sum = 0
    for i in range(time, time+10):
        sum += listacc_X[i]
    after_ave = sum/10"""
    mode = 0
    time_var = np.var(listacc_X[time-10: time+10])
    sum_acc = math.sqrt(listacc_X[time]*listacc_X[time]+listacc_Y[time]*listacc_Y[time]+listacc_Z[time]*listacc_Z[time])
    # sum_gyo = math.sqrt(listgyo_X[time]*listgyo_X[time]+listgyo_Y[time]*listgyo_Y[time]+listgyo_Z[time]*listgyo_Z[time])
    if time_var < 0.1 and sum_acc < 0.3:
        mode = 1
    return mode


def zupt(a_nav, threshold):
    '''
        Applies Zero Velocity Update(ZUPT) algorithm to acc data.
        @param a_nav: acc data
        @param threshold: stationary detection threshold, the more intense the movement is the higher this should be
        Return: velocity data, ZUPT times
    '''
    sample_number = np.shape(a_nav)[0]
    velocities = []
    prevt = -1

    still_phase = False

    v = np.zeros((3, 1))
    t = 0
    while t < sample_number:
        at = a_nav[t, np.newaxis].T
        if np.linalg.norm(at) < threshold:
            if not still_phase:
                predict_v = v + at * 0.01

                v_drift_rate = predict_v / (t - prevt)
                for i in range(t - prevt - 1):
                    velocities[prevt + 1 + i] -= (i + 1) * v_drift_rate.T[0]

            v = np.zeros((3, 1))
            prevt = t
            still_phase = True
        else:
            v = v + at * 0.01
            still_phase = False

        velocities.append(v.T[0])
        t += 1

    velocities = np.array(velocities)
    return velocities


def positionTrack(a_nav, a_Y, velocities, threshold):
    '''
    Simple integration of acc data and velocity data.

    @param a_nav: acc data
    @param velocities: velocity data

    Return: 3D coordinates in navigation frame
    '''

    sample_number = np.shape(a_nav)[0]
    positions = []
    p = np.array([[0, 0, 0]]).T
    prevt = -1

    still_phase = False

    t = 0
    while t < sample_number:
        at = a_nav[t, np.newaxis].T
        vt = velocities[t, np.newaxis].T
        if np.linalg.norm(at) < threshold and abs(a_Y[t]) <= 1 and math.acos(abs(a_Y[t])) < threshold:
            if not still_phase:
                predict_p = p + vt * 0.01 + 0.5 * at * 0.01**2

                p_drift_rate = predict_p / (t - prevt)
                for i in range(t - prevt - 1):
                    positions[prevt + 1 + i] -= (i + 1) * p_drift_rate.T[0]

            p = np.zeros((3, 1))
            prevt = t
            still_phase = True
        else:
            p = p + vt * 0.01 + 0.5 * at * 0.01**2
            still_phase = False
        positions.append(p.T[0])
        t += 1

    positions = np.array(positions)
    return positions


def removeAccErr(a_nav, threshold=0.2, filter=False, dt=0.01, wn=30):

    '''
    Removes drift in acc data assuming that
    the device stays still during initialization and ending period.
    The initial and final acc are inferred to be exactly 0.
    The final acc data output is passed through a bandpass filter to further reduce noise and drift.

    @param a_nav: acc data, raw output from the kalman filter
    @param threshold: acc threshold to detect the starting and ending point of motion
    @param wn: bandpass filter cutoff frequencies

    Return: corrected and filtered acc data
    '''

    sample_number = np.shape(a_nav)[0]
    t_start = 0
    for t in range(sample_number):
        at = a_nav[t]
        if np.linalg.norm(at) > threshold:
            t_start = t
            break

    t_end = 0
    for t in range(sample_number - 1, -1, -1):
        at = a_nav[t]
        if np.linalg.norm(at - a_nav[-1]) > threshold:
            t_end = t
            break

    an_drift = a_nav[t_end:].mean(axis=0)
    an_drift_rate = an_drift / (t_end - t_start)

    for i in range(t_end - t_start):
        a_nav[t_start + i] -= (i + 1) * an_drift_rate

    for i in range(sample_number - t_end):
        a_nav[t_end + i] -= an_drift

    if filter:
        filtered_a_nav = filtSignal([a_nav], dt=0.01, wn=wn, btype='lowpass')[0]
        return filtered_a_nav
    else:
        return a_nav

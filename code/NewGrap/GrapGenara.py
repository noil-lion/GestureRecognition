from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/')
from utils.loadData import loadData


dir = 'D:/ResearchSpace/task/gestureRecognition/data/test/szqs/220501214924_zihao_szqsV2.txt'
LD = loadData(dir)
listacc_X, listacc_Y, listacc_Z, listgyo_X, listgyo_Y, listgyo_Z, listangle_R, listangle_P, listangle_Y = LD.get_AGA()


# 生成一些demo数据
timestep = LD.get_timestep(400)
data = listacc_X[0:400]
print(data)
X = [timestep, data]
plt.plot(X[0], X[1])
plt.title('Time series')
plt.xlabel('timestamp')
plt.ylabel('value')
plt.show()
# 分段聚合
# PAA
transformer = PiecewiseAggregateApproximation(window_size=40)
result = transformer.transform(X)

# Scaling in interval [0,1]
scaler = MinMaxScaler()
scaled_X = scaler.transform(result)

plt.plot(scaled_X[0, :], scaled_X[1, :])
plt.title('After scaling')
plt.xlabel('timestamp')
plt.ylabel('value')
plt.show()

# 转为极坐标
arccos_X = np.arccos(scaled_X[1, :])
"""fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(result[0, :], arccos_X)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("Polar coordinates", va='bottom')
# plt.show()"""

field = [a+b for a in arccos_X for b in arccos_X]
gram = np.cos(field).reshape(-1, 4)
print(type(gram))
plt.imshow(gram)
plt.show()

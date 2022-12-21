import numpy as np
from collections import Counter
from math import sqrt


def knn(x_train, y_train, x, k):
    distance = [sqrt(np.sum((train_data - x)**2))
                for train_data in x_train]  # 计算x_train中于x样本的特征向量距离
    neast = np.argsort(distance)[:k]  # argsort从小到大排序取前k个数据并返回其在原数组的索引
    label = [y_train[i] for i in neast]
    votes = Counter(label)
    print("votes:{}".format(votes))
    predict = votes.most_common(1)[0][0]
    return predict


x = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [10, 10], [11, 11]]
y = [1, 1, 1, 1, 1, 2, 2]

x_train = np.array(x)
y_train = np.array(y)
X = np.array([13, 13])
predict = knn(x_train, y_train, X, 4)
print(predict)

# KNN算法
有监督学习的分类算法。  
根据新样本的k个最近邻居来判断该样本所属类。

## 算法实现步骤
1. 计算已知样本集A中所有样本的特征向量和新样本B的特征向量的距离
1. 按距离递增对A中的样本进行排序
1. 选取与B最近的A中的k个样本
1. 确定k个样本中的占最多数的类别
1. 频率出现最高的类别就是B样本所属的类别
## python代码
```
import numpy as np
from collections import Counter
from math import sqrt

def knn(x_train, y_train, x, k):
    distance = [sqrt(np.sum((train_data - x)**2)) for train_data in x_train]  # 计算x_train中于x样本的特征向量距离
    neast = np.argsort(distance)[:k]  # argsort从小到大排序取前k个数据并返回其在原数组的索引
    label = [y_train[i] for i in neast]
    votes = Counter(label)
    print("votes:{}".format(votes))
    predict = votes.most_common(1)[0][0]
    return predict

x = [[0,0],[1,1],[2,2],[3,3],[4,4],[10,10],[11,11]]
y = [1,1,1,1,1,2,2]

x_train = np.array(x)
y_train = np.array(y)
X = np.array([13,13])
predict = knn(x_train, y_train, x, 3)
print(predict)
```
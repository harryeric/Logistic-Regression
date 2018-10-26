# logistic回归分类实验
# 数据集：iris数据集
# 使用的python库：numpy，sklearn，matplotlib，math

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

# PCA降维部分
# PCA采用sklearn库中内置的函数
# iris通过sklearn.database导入
# 将数据集每个样本的特征从四维降至二维
iris = load_iris()
newdata = iris.data
newdata = np.array(newdata)
pca = PCA(n_components=2)
pca.fit(iris.data)
iris_1 = pca.transform(iris.data)
# print(iris_1)
plt.figure(1)
plt.scatter(iris_1[:, 0], iris_1[:, 1])
# plt.show()


# sigmoid函数
def my_sig(w, x):
    w = w.T
    r = np.dot(w, x)
    s = 1.0 / (1 + math.exp(-1*r))
    return s


def my_grad(w, x, y):
    len_y = len(y)
    sum_0 = 0
    for i in range(0, 99):
        s = my_sig(w, x[:, i])
        x1 = x[:, i]
        x1 = x1.reshape(x1.shape[0], 1)
        sum_0 = sum_0 + (s - y[i]) * x1

    g = 1 / len_y * sum_0
    return g


# logistic代价函数
def my_fun(w, x, y):
    len_y = len(y)
    sum_0 = 0
    for i in range(0, len_y-1):
        s = my_sig(w, x[:, i])
        sum_0 = sum_0+y[i]*math.log(s)+(1-y[i])*math.log(1-s)
    f = -1/len_y*sum_0
    return f


# 数据预处理
# 构造训练样本和测试样本
iris_1 = iris_1[:100, :]  # 分类后两类为iris_1 = iris_1[50:150, :]
iris_1 = iris_1.T
a = np.ones((1, 100))
x = np.vstack((iris_1, a))
y = np.vstack((np.ones((50, 1)), np.zeros((50, 1))))

# 梯度下降法求解Logistic Regression
w = np.array([[1], [1], [1]])
f_test = np.zeros(5000)
for i in range(0, 4999):
    w = w - 0.1 * my_grad(w, x, y)
    f_test[i] = my_fun(w, x, y)
plt.figure(2)
plt.plot(np.arange(5000), f_test)
y_test = np.zeros(100)
for i in range(0, 99):
    y_test[i] = my_sig(w, x[:, i])
print(y_test)
plt.figure(3)
plt.scatter(np.arange(100), y_test)
# plt.show()

iris_1 = iris_1.T
plt.figure(4)

fen = np.zeros([50, 2])
color_diff = np.array([0, 0, 0])
m = 0
for i in np.arange(-2, 4, 0.1):
    for j in np.arange(-2, 1.5, 0.1):
        color_diff = np.array([i, j, 1])
        num_xy = my_sig(w, color_diff)
        if 0.5 <= num_xy < 0.6:
            fen[m, 0] = i
            fen[m, 1] = j
            m = m + 1
# plt.plot(fen[:, 0], fen[:, 1])
fen_1 = np.vstack((fen.min(axis=0), fen.max(axis=0)))

print(fen_1)
print(fen_1[:, 0].T)
plt.plot(fen_1[:, 0].T, fen_1[:, 1].T)

for i in range(0, 99):
    if y_test[i] >= 0.5:
        plt.scatter(iris_1[i, 0], iris_1[i, 1], marker='v')
    else:
        plt.scatter(iris_1[i, 0], iris_1[i, 1], marker='+')


plt.show()

# coding=utf-8
import random

import numpy as np
from matplotlib import pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split() #去回车，放入列表
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))

    return dataMat,labelMat

"""
定义sigmod函数
"""
def sigmod(inX):
    return 1.0/(1+np.exp(-inX))

"""
画图
"""
def plot_data_set(dataMat, labelMat,weights):
    # 转成numpy的array数组
    dataArr = np.array(dataMat)
    # 数据个数
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []; ycord1 = []
    # 负样本
    xcord0 = []; ycord0 = []
    for i in range(n):
        if int(labelMat[i]) == 1: #正样本
            # 数组第二列作为x 第三列作为y
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord0.append(dataArr[i, 1]);ycord0.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(xcord0, ycord0, s=20, c='green', alpha=.5) # 正样本
    # 画出 绘制决策边界
    x = np.arange(-3.0, 3.0, 0.1)
    """
    y的表达式：
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0 * x0 + w1 * x1 + w2 * x2 = f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0, w1, w2身上去了
    所以： w0 + w1 * x + w2 * y = 0 = > y = (-w0 - w1 * x) / w2
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    # ---- 决策边界 end -----
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')  # 绘制label
    plt.show()

"""
梯度上升算法

dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量
"""

def gradAscent(data_mat_in, class_labels):
    # 转换为numpy矩阵 [[1,2,3,4],[1,2,3,5]...] -->[[1,2,4,5] [1,2,3,5]]
    data_matrix = np.mat(data_mat_in)
    # 将标签转置成列向量
    labels_mat = np.mat(class_labels).transpose()
    # data_matrix的大小 m行 n列
    m,n = data_matrix.shape
    # 初始化一个权重列向量
    weights = np.ones((n,1))
    # 设置学习率
    alpha = 0.001
    # 最大循环次数
    max = 500

    for i in range(max):
        # 预测值
        h = sigmod(data_matrix * weights) # sigmod(wx)
        # 错误
        error_mat = labels_mat - h
        # 梯度上升公式
        weights = weights + alpha * (data_matrix).transpose() * error_mat
    return weights.getA()

"""
随机梯度
"""
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for i in range(150):
        data_index = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(data_index)))  # 随机选取样本
            h = sigmod(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = float(classLabels[randIndex] - h)  # 计算误差
            weights = weights + alpha * float(error) * dataMatrix[randIndex]  # 更新回归系数
            del (data_index[randIndex])
    return weights


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plot_data_set(dataMat, labelMat,weights)
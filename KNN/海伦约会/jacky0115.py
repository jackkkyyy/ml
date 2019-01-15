# -*- coding: UTF-8 -*-
import operator

import numpy as np
import matplotlib.pyplot as plt

def file2matrix():
    fr = open('datingTestSet.txt')
    fileLines = fr.readlines()
    numberLines = len(fileLines)
    # 返回的矩阵（1000*3）
    returnMatrix = np.zeros((numberLines,3))

    classLableVector = []

    index = 0
    for line in fileLines:
        line = line.strip()
        # 以 '\t' 切割字符串
        listLine = line.split('\t')
        # 将前三列加入returnMatrix
        returnMatrix[index,:] = listLine[0:3]
        # 匹配标签项
        if listLine[-1] == 'largeDoses':
            classLableVector.append(3)
        if listLine[-1] == 'smallDoses':
            classLableVector.append(2)
        if listLine[-1] == 'didntLike':
            classLableVector.append(1)
        index +=1
    return returnMatrix, classLableVector

"""
归一化
"""
def autoForm(dataSet):
    # 最大最小值
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    # 范围
    ranges = maxVal - minVal
    t = dataSet.shape
    normDataSet = np.zeros(t)
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    matrix = np.tile(minVal, (m,1))
    normDataSet = dataSet - matrix

    # 将最小值之差除以范围
    returnNormSet = normDataSet/np.tile(ranges,(m,1))
    return returnNormSet
"""
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
"""

def classify(intX,dataSet,lables,k):
    # 获取训练集行数
    dataSize = dataSet.shape[0]
    # 将测试数据转换成和训练集行数一样的矩阵 并减去训练集数据
    dataDiff = np.tile(intX,(dataSize,1)) - dataSet
    # 采用欧式距离计算
    sqdataDiff = dataDiff ** 2
    # sum(axis=1) 每行向量相加
    sqDistances = sqdataDiff.sum(axis=1)
    # 开根号
    distances = sqDistances ** 0.5
    # 将距离从小到大排序 返回索引值
    sortDistances = distances.argsort()
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteLable = lables[sortDistances[i]]
        # get(voteLable, 0)根据键获取字典的值 不存在则设为 0
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # sortedClassCount格式为[(3,7)]第一个为便签值 第二个为出现的次数
    return sortedClassCount[0][0]





if __name__ == '__main__':
    returnMatrix, classLableVector = file2matrix()
    # 绘制散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(returnMatrix[:, 0], returnMatrix[:, 1], 15.0 * np.array(classLableVector), 15.0 * np.array(classLableVector))
    # plt.show()
    returnMatrix = autoForm(returnMatrix)
    # 取10%测试
    rate = 0.10
    # 数据集行数
    number = returnMatrix.shape[0]
    # 前10%做测试
    testVec = int(number * rate)
    errorCount = 0
    for i in range(testVec):
        # 返回的是标签值
        classifyResult = classify(returnMatrix[i,:],returnMatrix[testVec:number,:],classLableVector[testVec:number],7)
        print("预测值为：%d，真实值为：%d" % (classifyResult,classLableVector[i]))
        # 判断是否正确
        if classLableVector[i] != classifyResult:
            print("----- 预测错误 -----")
            errorCount +=1
    print("错误率为：%f%%" %(errorCount/float(testVec)*100))
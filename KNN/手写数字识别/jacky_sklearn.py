# -*- coding: UTF-8 -*-
from os import listdir

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN



"""
图片转向量
"""
def img2Vector(fileName):
    returnVect = np.zeros((1, 1024))
    fr = open(fileName)
    # 图片格式为32 * 32
    for i in range(32):
        fileLine = fr.readline()
        for j in range(32):
            # 将32*32的矩阵转换为1*1024向量
            returnVect[0,32*i+j] = int(fileLine[j])
    return returnVect

"""
读取文件中的数据
"""
def importData(folderName):
    # 读取数据文件夹
    trainFileList = listdir(folderName)
    m = len(trainFileList)
    # 训练/测试集向量
    returnMat = np.zeros((m, 1024))
    # 标签向量
    lables = []
    for i in range(m):
        # 获取文件名
        fileName = trainFileList[i]
        # 获取标签名
        className = int(fileName.split("_")[0])
        lables.append(className)
        # 将32*32 转成 1*1024
        returnMat[i, :] = img2Vector('%s/%s' %(folderName, fileName))

    return returnMat,lables
def handWriteTest():
    # 读取训练数据文件夹
    trainMat,trainLables = importData('trainingDigits')
    # 构建kNN分类器 参数如下：
    # neigh = kNN(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,p=2, metric=’minkowski’, metric_params=None, n_jobs=1)
    # n_neighbors：默认为5，就是k-NN的k的值
    # weights：默认是uniform，参数可以是uniform（均等的权重）、distance（不均等的权重）
    # algorithm：快速k近邻搜索算法，默认为auto 搜索算法ball_tree、kd_tree、brute
    # leaf_size：默认是30，这个是构造的kd树和ball树的大小
    # metric：用于距离度量，默认度量是minkowski（欧氏距离）
    # p：距离度量公式
    # metric_params：距离公式的其他关键参数，使用默认的None即可。
    # n_jobs：并行处理设置。默认为1，临近点搜索并行工作数。如果为-1，那么CPU的所有cores都用于并行工作。
    knnNeigh = KNN(n_neighbors = 3,algorithm= 'kd_tree')
    # 将训练数据和标签放入KNN分类器中
    knnNeigh.fit(trainMat, trainLables)
    # 导入测试数据
    trainFileList = listdir('testDigits')
    m = len(trainFileList)
    errorCount = 0
    for i in range(m):
        # 获取文件名
        fileName = trainFileList[i]
        # 获取标签名
        className = int(fileName.split("_")[0])
        # 将32*32 转成 1*1024
        testnMat = img2Vector('testDigits/%s' % fileName)
        result = knnNeigh.predict(testnMat)
        print('预测值为：%s,测试值为：%s' %(result,className))
        if result != className:
            errorCount += 1
            print("----- 预测错误 -----")
    print("错误率为：%f%%" %(errorCount/float(m)*100))


if __name__ == '__main__':
    handWriteTest()
# -*- coding: UTF-8 -*-
import time

import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vector(filename):
    # 创建1*1024向量
    returnVect = np.zeros((1,1024))
    # 打开文件
    fr = open(filename)
    # 逐行读取文件
    for i in range(32):
        # 按行读取
        readLine = fr.readline()
        # 每行32个数据
        for j in range(32):
            # 将每行数据逐个放入向量
            returnVect[0,32*i+j] = int(readLine[j])
    # 返回1*1024向量
    return returnVect

"""
函数说明:手写数字分类测试
"""
def handwritingClassTest():
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
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
    neigh = kNN(n_neighbors = 3, algorithm = 'kd_tree')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))


if __name__ == '__main__':
    start = time.time()
    handwritingClassTest()
    print("方法执行了：%f秒" % (time.time()-start))
# -*- coding: UTF-8 -*-
import numpy as np
from os import listdir
from sklearn.svm import SVC

"""
将32 * 32图片转为1 * 1024向量
"""
def img2Vector(file_name):

    # 将32 * 32的矩阵转为 1*1024的向量
    returnVect = np.zeros((1,1024))
    fr = open(file_name)

    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(line[j])
    return returnVect

def handwriting():

    # 获取训练集下的文件目录
    train_file_list = listdir('trainingDigits')
    # 文件个数
    m = len(train_file_list)
    # 训练集矩阵  m * 1024
    train_mat = np.zeros((m,1024))
    # 训练集标签
    train_labels = []
    # 遍历每个文件
    for i in range(m):
        file_name = train_file_list[i]
        # 文件名格式 0_0 以_切割 取第一个元素
        train_class = int(file_name.split('_')[0])
        # 放入训练集
        train_labels.append(train_class)
        # 将每个文件转成1 * 1024向量 并放入对应的训练集矩阵中
        train_mat[i,] = img2Vector('trainingDigits/%s' % file_name)
    # 利用sklearn训练
    classify = SVC(C=200,kernel='rbf')
    classify.fit(train_mat,train_labels)

    # 获取测试集下的文件目录
    test_file_list = listdir('testDigits')
    n =len(test_file_list)

    error_count = 0
    for i in range(n):
        test_file_name = test_file_list[i]

        test_class = int(test_file_name.split('_')[0])
        test_vector = img2Vector('testDigits/%s' % test_file_name)
        result = classify.predict(test_vector)
        print("预测结果：%s，真是结果：%s" %(result,test_class))
        if result != test_class:
            error_count += 1
    print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / n * 100))

if __name__ == '__main__':
    handwriting()
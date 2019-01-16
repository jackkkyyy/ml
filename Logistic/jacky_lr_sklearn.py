# coding=utf-8
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def load_data():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in fr_train.readlines():
        current_line = line.strip().split('\t')
        train_arr = []
        for i in range(len(current_line) -1):
            train_arr.append(float(current_line[i]))
        trainingSet.append(train_arr)
        trainingLabels.append(float(current_line[-1]))
    # 使用随机梯度上升得到weights
    weights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    error_count = 0
    numTestVec = 0.0
    for line in fr_test.readlines():
        numTestVec += 1.0
        current_line = line.strip().split('\t')
        class_arr = []
        for i in range(len(current_line) -1):
            class_arr.append(float(current_line[i]))

        if int(classify(np.array(class_arr),weights)) != int(current_line[-1]):
            error_count += 1
    print("测试集错误率为: %.2f%%" % ((float(error_count)/numTestVec) * 100 ))
"""
定义sigmod函数
"""
def sigmod(inX):
    return 1.0 / (1 + np.exp(-inX))

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
            a = classLabels[randIndex]
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del (data_index[randIndex])
    return weights

"""
分类函授
"""
def classify(inX,weights):
    a = sum(inX * weights)
    print(a)
    prob = sigmod(a)
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def sklearn_classify():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in fr_train.readlines():
        current_line = line.strip().split('\t')
        train_arr = []
        for i in range(len(current_line) - 1):
            train_arr.append(float(current_line[i]))
        trainingSet.append(train_arr)
        trainingLabels.append(float(current_line[-1]))
    for line in fr_test.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet,testLabels) * 100
    print("正确率为：%f%%" % test_accurcy)


if __name__ == '__main__':
    sklearn_classify()
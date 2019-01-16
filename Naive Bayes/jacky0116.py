# -*- coding: UTF-8 -*-
from math import log

import numpy as np

"""
创建数据集

"""
def createDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

"""
构建词向量
"""
def createWordVect(dataset):
    wordVect = set([])
    for words in dataset:
        # set取并集 类似于+=
        wordVect = wordVect | set(words)
    return list(wordVect)


"""
将每条数据转换成向量形式
"""
def word2Vect(wordVect,inputSet):
    # 创建一个和词汇一样长的矩阵
    returnVec = [0] * len(wordVect)
    for word in inputSet:
        if word in wordVect:
            returnVec[wordVect.index(word)] = 1
    return returnVec

"""
训练贝叶斯算法
trainMatrix:训练文档矩阵

trainCategory：标签矩阵
"""
def trainNb(trainMatrix, trainCategory):
    #文档数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # 构造单词出现次数列表
    p0Num = np.ones(numWords) #[0,0,0,0......]
    p1Num = np.ones(numWords)

    # 在利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，即计算 p(w0|1) * p(w1|1) * p(w2|1)。
    # 如果其中一个概率值为 0，那么最后的乘积也为 0。为降低这种影响，可以将所有词的出现数初始化为 1，并将分母初始化为 2
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        # 侮辱类
        if trainCategory[i] == 1:
            # 如果是侮辱性文件，对侮辱性文件的向量进行加和
            p1Num += trainMatrix[i] #[0,1,0,0,1,1...] + [1,0,0,0,0,1...]
            p1Denom += sum(trainMatrix[i]) # 对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
        # 非侮辱类
        else:
            p0Num += trainMatrix[i] #[0,1,0,0,1,1...] + [1,0,0,0,0,1...]
            p0Denom += sum(trainMatrix[i]) # 获取每条训练数据单词总数
    # 侮辱类每个单词出现的概率 侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 当计算乘积 p(w0|ci) * p(w1|ci) * p(w2|ci)... p(wn|ci) 时，由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。
    # 对乘积取自然对数
    p1Vect = np.log(p1Num / p1Denom) # [1,2,1,5,1,2....]/90
    # 非侮辱类概率
    p0Vect = np.log(p0Num / p0Denom) # [1,2,1,5,1,2....]/90

    return p0Vect, p1Vect, pAbusive


"""
贝叶斯分类
testVect:待测数据向量
p0Vect：非侮辱概率矩阵 [log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]
p1Vect：侮辱类概率矩阵
pAbusive：侮辱类文档概率
"""
def classifyNB(testVect,p0Vect, p1Vect, pAbusive):
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了，
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(testVect * p1Vect) + log(pAbusive)  # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(testVect * p0Vect) + log(1.0 - pAbusive)  # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0



if __name__ == '__main__':
    postingList, classVec = createDataSet()
    wordVect = createWordVect(postingList)
    trainMat = []
    for word in postingList:
        trainMat.append(word2Vect(wordVect,word))
    p0Vect, p1Vect, pAbusive = trainNb(trainMat,classVec)
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(word2Vect(wordVect, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(word2Vect(wordVect, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
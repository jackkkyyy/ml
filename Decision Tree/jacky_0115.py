# -*- coding: UTF-8 -*-
import numpy as np
import operator


"""
创建数据集

"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性

def calcShannonEnt(dataSet):
    dataSize = len(dataSet)
    # 获取标签值 放入字典
    classLables = {}
    # 计算lable出现的次数
    for data in dataSet:
        current = data[-1]
        if current not in classLables.keys():
            classLables[current] = 0
        classLables[current] += 1
    # 计算香浓熵
    shannonEnt = 0.0
    for key in classLables:
        # 计算所有标签出现的频率
        prop = float(classLables[key]/dataSize)
        # 计算香浓熵
        shannonEnt -= prop * np.log2(prop)
    return shannonEnt


"""
划分特征 传入将某一列特征和需要划分的特征值 反正该特征值的集合

index：哪一列特征
value: 需要划分的特征值
"""
def splitDataSet(dataSet,index,value):
    returnSet = []
    for data in dataSet:
        if data[index] == value:
            # 去除index那一列
            reduceVect = data[:index]
            # extend和append的区别
            # music_media.append(object) 向列表中添加一个对象object
            # music_media.extend(sequence) 把一个序列seq的内容添加到列表中 (跟 += 在list运用类似， music_media += sequence)
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】TODO 此处不知道为什么排除index列
            reduceVect.extend(data[:index+1])
            returnSet.append(reduceVect)
    return returnSet


"""
选择最好的特征
"""
def chooseBestFeature(dataSet):
    # 特征的数量 列数-1
    numFeatures = len(dataSet[0]) - 1
    # 计算原始的信息熵
    baseEntory = calcShannonEnt(dataSet)
    # 最优的信息增益 最优的特征
    bestInfoGain,bestFeature = 0.0,-1
    # 定义一个临时熵变量
    newEntropy = 0.0
    for i in range(numFeatures):
        # 获取每一列的特征值
        featList = [example[i] for example in dataSet]
        # 特征值去重
        uniqueVals = set(featList)
        for value in uniqueVals:
            # 获取每个特征值的数量
            subDataSet = splitDataSet(dataSet,i,value)
            # 计算每个特征的概率
            prop = len(subDataSet)/float(len(dataSet))
            # 计算经验熵
            newEntropy += prop * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntory - newEntropy
        # 比较信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
投票法找出数量最多的特征
"""

def majorityCnt(classList):
    # 定义一个字典存储标签票数
    classCout = {}
    # 遍历所有标签 存入数量
    for vote in classList:
        classCout[vote] = classCout.get(vote,0) + 1
    # 根据字典的值降序排序
    sortedClassCount = sorted(classCout.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


"""
创建树
"""
def createTree(dataSet, labels, featLabels):
    # 获取所有的标签值
    classList = [example[-1] for example in dataSet]
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    if len(dataSet) == classList.count(classList[0]):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获取最优的特征
    bestFeat = chooseBestFeature(dataSet)
    # 获取lable名称
    bestFeatlabel = labels[bestFeat]
    featLabels.append(bestFeatlabel)
    # 初始化mytree
    myTree = {bestFeatlabel:{}}
    # 删除已选择的lable
    del(labels[bestFeat])
    # 获取最优列 按照branch分类
    featVals = [example[bestFeat] for example in dataSet]
    # 特征的值去重
    uniqueVals = set(featVals)
    for value in uniqueVals:
        # 获取剩余标签
        subLables = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatlabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables, featLabels)
    print(myTree)
    return myTree


def classify(inputTree, featLabels, testVec):
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels,featLabels)
    testVec = [0, 1]  # 测试数据
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')
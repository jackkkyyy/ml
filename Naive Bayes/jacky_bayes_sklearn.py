# -*- coding: UTF-8 -*-
import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB

"""
数据预处理

floderPath:文件夹名称
testSize：测试集比例

"""
def textProcessing(floderPath,testSize = 0.2):
    # ./SogouC/Sample下所有目录
    floderList = os.listdir(floderPath)

    dataList = []
    classList = []

    # 遍历每个文件夹的字文件夹
    for floder in floderList:
        # 拼接路径 ./SogouC/Sample/c0000008
        newFloderPath = os.path.join(floderPath, floder)
        # ./SogouC/Sample/c0000008下所有txt文件
        files = os.listdir(newFloderPath)
        for file in files:
            # 读取每个txt文件
            with open(os.path.join(newFloderPath,file),'r',encoding='utf-8') as f:
                raw = f.read()
            # 利用结巴切词 精简模式 返回一个可迭代的generator
            wordCut = jieba.cut(raw, cut_all=False)
            # 转成list
            wordList = list(wordCut)

            dataList.append(wordList)
            classList.append(floder)

        # 将数据集和标签合并
        dataZipList = list(zip(dataList,classList))
        # 乱序
        random.shuffle(dataZipList)
        # 测试集切分的索引值
        index = int(len(dataList) * testSize) + 1
        # 测试集
        testZipList = dataZipList[:index]
        # 训练集
        trainZipList = dataZipList[index:]
        # 解压缩
        testDataList,testClassList = zip(*testZipList)
        trainDataList,trainClassList = zip(*trainZipList)

        # 统计训练集词频
        allWordsDict = {}
        for wordList in trainDataList:
            for word in wordList:
                allWordsDict[word] = allWordsDict.get(word,0) + 1

        # 根据词频进行排序
        # key=lambda x:x[1] 中x表示排序的字典中的每个值 x[1]表示按value值排序 x[0]表示按key
        allWordsDictSorted = sorted(allWordsDict.items(),key=lambda x:x[1],reverse=True)
        # 将字典解压
        allWordsList,allWordNum = zip(*allWordsDictSorted)
        # 转换为list
        allWordsList = list(allWordsList)

    return allWordsList,trainDataList,trainClassList,testDataList,testClassList

"""
读取停用词文本
"""
def stopWordSet(fileName):

    stopWordList = set()

    with open(fileName,'r',encoding='utf-8') as f:
        for line in f.readline():
            # 去除回车
            word = line.strip()
            # 有文本则放入列表中
            if(len(word)) > 0:
                stopWordList.add(word)
    return stopWordList


"""
特征提取

all_words_list - 训练集所有文本列表
deleteN - 删除词频最高的deleteN个词
stopwords_set - 指定的结束语

"""
def wordFeature(allWordsList, deleteN, stopWordSet= set()):
    featureList = []
    n = 1
    # 删除前deleteN个 所以 range从deleteN到list长度 步长为1
    for i in range(deleteN, len(allWordsList),1):
        # feature_words的维度为1000
        if n > 1000:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not allWordsList[i].isdigit() and allWordsList[i] not in stopWordSet and 1 < len(allWordsList[i]) < 5:
            featureList.append(allWordsList[i])
            n +=1
    return featureList

"""
将特征向量化

"""
def textFeature(trainDataList, testDataList, featureList):
    # 出现在特征集中，则置1
    def text_features(text, featureList):
        textWords = set(text)
        features = [ 1 if word in textWords else 0 for word in featureList]
        return features
    train_feature_list = [ text_features(text, featureList) for text in trainDataList]
    test_feature_list = [ text_features(text, featureList) for text in testDataList]
    return train_feature_list,test_feature_list


"""
分类
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__ == '__main__':
    # 文本预处理
    allWordsList, trainDataList, trainClassList, testDataList, testClassList = textProcessing('./SogouC/Sample')
    # print(allWordsList)
    # 停用词
    stopWordSet = stopWordSet('./stopwords_cn.txt')
    # 特征提取
    featureList = wordFeature(allWordsList,450,stopWordSet)
    # 文本向量化
    train_feature_list, test_feature_list = textFeature(trainDataList, testDataList, featureList)
    # 分类得分
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, trainClassList, testClassList)
    print(test_accuracy)
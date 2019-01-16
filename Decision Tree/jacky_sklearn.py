# coding=utf-8
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus


if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    #print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenseTarget = []
    lenseList = []
    lensesDict = {}
    # 获取标签列
    for each in lenses:
        lenseTarget.append(each[-1])
    # 先把每个特征的值存在list中 再把整个list作为value存入字典中 key为each_label
    for each_label in lensesLabels:
        for each in lenses:
            # lensesLabels.index(each_label) 表示索引值（0,1,2,3）
            lenseList.append(each[lensesLabels.index(each_label)])
        lensesDict[each_label] = lenseList
        lenseList = []
    #print(lensesDict)
    # 转为pandas二维数组
    lenses_pd = pd.DataFrame(lensesDict)
    #print(lenses_pd)

    #序列化 用sklearn库中的LabelEncoder
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)
    # 使用Graphviz可视化决策树
    clf = tree.DecisionTreeClassifier(max_depth=4)  # 创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenseTarget)
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
    #                      feature_names=lenses_pd.keys(),
    #                      class_names=clf.classes_,
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("tree.pdf")
    print(clf.predict([[1, 1, 1, 0]]))
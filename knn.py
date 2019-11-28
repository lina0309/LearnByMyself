# coding = utf8
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
#中文显示
import matplotlib.font_manager as fm
import math
import operator
#解决可视化显示中文乱码
myfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')


'''创建数据源，返回数据集和类标签'''
def creat_dataset():
    datasets = array([[8, 4, 2], [7, 1, 1], [1, 4, 4], [3, 0, 5]])
    labels = ['非常热', '非常热', '一般热', '一般热']
    return datasets,labels

'''可视化分析数据'''
def analyze_data_plot(x, y):
    fig = plt.figure()
    #将画布分为一行一列一块
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    plt.title('游客冷热感知点散点图', fontsize=35, fontname='宋体', fontproperties = myfont)
    plt.xlabel('天热吃冰淇淋数目', fontsize=15, fontname='宋体', fontproperties = myfont)
    plt.ylabel('天热喝水数目', fontsize=15, fontname='宋体', fontproperties= myfont)
    plt.scatter(x, y)

    #保存当前文件夹
    #plt.savefig('datasets_plot.png', bbox_inches = 'tight')
    path = 'C:/Users/lenovo/Desktop/'
    plt.savefig(path+'datasets_plot.png', bbox_inches='tight')
    plt.show()

def knn_Classifier(newV,datasets,labels,k):
    # 1 计算样本数据与样本库数据之间的距离
    SqrtDist = EuclideanDistance3(newV,datasets)
    # 2 根据距离进行排序，按照列向量进行排序
    sortDisIndexs = SqrtDist.argsort(axis = 0)
    print(sortDisIndexs)
    # 3 针对k个点，统计各类别的数量
    classCount = {}
    for i in range(k):
        # 根据距离排序索引值找到类别标签
        votelabel = labels[sortDisIndexs[i]]
        print(sortDisIndexs[i],votelabel)
        #统计类标签的键值对!
        classCount[votelabel]=classCount.get(votelabel,0)+1
    print(classCount)
    # 4 投票机制，少数服从多数原则，输出类别
    # 对各个分类字典进行排序，降序0，升序1，itemgetter按照value排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(0))

    return sortedClassCount[0][0]
    '''欧氏距离计算 '''
def ComputeEuclideanDistance(x1,y1,x2,y2):
    d = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return d
def EuclideanDistance(instance1,instance2,length):
    d = 0
    for x in range(length):
        d += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(d)
def EuclideanDistance3(newV,datasets):
    # 1 获取数据向量的行向量维度和列向量维度
    rowsize, colsize = datasets.shape
    #print(tile(newV,(rowsize,1)))
    # 2 各特征向量间作差值
    diffMat = tile(newV,(rowsize,1))-datasets
    #print(diffMat)
    # 3 对差值平方
    sqDiffMat = diffMat**2
    # print(sqDiffMat)
    # 4 差值平方和进行开方
    SqrtDist = sqDiffMat.sum(axis=1)**0.5
    #print(sqrtDist)
    return SqrtDist
# 利用KNN分类器预测随机访客天气感知度
def Predict_temperature():
    datasets, labels = creat_dataset()
    iceCream = float(input('Q:iceCream?\n'))
    drinkWater = float(input('Q:drinkWater?\n'))
    playAct = float(input('Q:playAct?\n'))

    newV = array([iceCream,drinkWater,playAct])
    res = knn_Classifier(newV, datasets, labels, 3)
    print('访客认为成都天气是：', res)


if __name__=='__main__':
    #datasets, labels = creat_dataset()
    #print('数据集:\n', datasets, '\n类标签:\n', labels)

    #analyze_data_plot(datasets[:, 0], datasets[:, 1])#第一列；第二列
    #datasets[:2,]前两行
    #欧氏距离计算1
    # d = ComputeEuclideanDistance(2,4,8,2)
    # print('欧氏距离计算1',d)
    # # 欧氏距离计算2
    # d2 = EuclideanDistance([2,4],[8,2],2)
    # print('欧氏距离计算2',d2)
    # # 欧氏距离计算3
    # d3 = EuclideanDistance3([2,4,4],datasets)
    # print('欧氏距离计算3', d3)
    # 4.1 单实例构造KNN分类器
    #newV = [2,4,4]
    #res = knn_Classifier(newV,datasets,labels,3)
    # 4.2 多实例构造KNN分类器
    #vecs = array([[2,4,4],[3,0,0],[5,7,2]])
    #for vec in vecs:
    #    res = knn_Classifier(vec,datasets,labels,3)
    #    print(vec, 'KNN投票预测结果：', res)


    Predict_temperature()
# coding=utf-8
# 导入自带数据集
# from sklearn import datasets
#导入交叉验证库
# from sklearn import cross_validation
#导入SVM分类算法库
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
#导入图表库
import matplotlib.pyplot as plt
#生成预测结果准确率的混淆矩阵
from sklearn import metrics
import cv2
# 引入 系统模块
import os
import sys
import numpy as np

def loadTrainData(data_path):
    for dirpath,dirnames,filenames in os.walk(data_path):
        x_train = []
        y_train = []
        for dirname in dirnames:
            print dirname
            for dirpath,dirnames,filenames in os.walk(data_path+dirname):
                for file in filenames:
                    print file
                    testImg = cv2.imread(data_path+dirname+"/"+file)
                    testImg = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
                    testImg = cv2.resize(testImg,(20,15),interpolation=cv2.INTER_CUBIC)
                    x_train.append(testImg)
                    y_train.append(int(dirname))
        n_samples = len(x_train)
        x_train = np.array(x_train).reshape([n_samples,20*15])
        return x_train,y_train

def loadTestImg(img_list):
    x_test = []
    for img in img_list:
        testImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        testImg = cv2.resize(testImg,(20,15),interpolation=cv2.INTER_CUBIC)
        x_test.append(testImg)
    n_samples = len(x_test)
    x_test = np.array(x_test).reshape([n_samples,20*15])
    return x_test

def loadTestData(fold_name):
    for dirpath,dirnames,filenames in os.walk(fold_name):
        x_test = []
        for file in filenames:
            print dirnames
            testImg = cv2.imread(fold_name+file);
            testImg = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
            testImg = cv2.resize(testImg,(20,15),interpolation=cv2.INTER_CUBIC)
            x_test.append(testImg)
            # print x_test[0]
            # n_samples = len(digits.images)
            # cv2.imwrite("testImg.jpg",testImg)
        n_samples = len(x_test)
        # print n_samples
        # print len(x_test[0])
        x_test = np.array(x_test).reshape([n_samples,20*15])
    return x_test

def dumpTree(tree, filename):
    with open(filename,'wb') as fp:
        pickle.dump(tree, fp)

def loadTree(filename):
    with open(filename,'rb') as fp:
        return pickle.load(fp)

def trainer(x_train,y_train):
    # clf = svm.SVC(gamma=0.106,C=100)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    dumpTree(clf, 'knn_digital.pkl')
    # return predicted
def classifier(x_test):
    clf = loadTree('knn_digital.pkl')
    result = clf.predict(x_test)
    return result
#
# x_test = loadTestData("model/")
# x_train,y_train = loadTrainData("data-train/")
# trainer(x_train,y_train)
# result = classifier(x_test)
# print result
# predicted = classifier(x_test)
# print predicted

# encoding: utf-8
import os
import time
import sklearn
import numpy as np
from sklearn import svm


def svc_test():
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])

    clf = svm.SVC()
    clf.fit(x, y)
    print(clf)
    print('clf.support_vectors_', clf.support_vectors_)                 # 获取支持向量
    print('clf.support_', clf.support_)                                 # 获取支持向量的索引
    print('clf.n_support_', clf.n_support_)                             # 获取每个类的支持向量
    print('predict: [-0.8, 1] classes: ',  clf.predict([[-0.8, -1]]))



# def svm_test():
#     X = [[0, 0], [1, 1]]
#     y = [0, 1]
#     clf = svm.SVM()
#     clf.fit(X, y)
#     print(clf)


if __name__ == '__main__':
    svc_test()
    # svm_test()
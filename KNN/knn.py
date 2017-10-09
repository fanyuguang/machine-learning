# -*- coding: utf-8 -*-

import numpy as np
from sklearn import neighbors, datasets

digits = datasets.load_digits()
totalNum = len(digits.data)
trainNum = int(0.8 * totalNum)
trainX = digits.data[0 : trainNum]
trainY = digits.target[0 : trainNum]

testX = digits.data[trainNum:]
testY = digits.target[trainNum:]

clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
clf.fit(trainX, trainY)
Z = clf.predict(testX)

print "the total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX)) )
print datasets.load_digits().target[0 : 100]
print len(datasets.load_digits().target)
print datasets.load_iris().target

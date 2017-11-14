#-*- coding: utf-8 -*-

# Name : Lingxiao Zhang
# ID: 20475043
# Project 1, 6000B

import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# Data path for all files needed
data_path = '/Users/lingxiaozhang/Documents/6000b/project1/'
input_data = 'traindata.csv'
input_label = 'trainlabel.csv'
test_data = 'testdata.csv'

# Read the input data
train_X = pd.read_csv(data_path + input_data)
train_Y = pd.read_csv(data_path + input_label)
test_X = pd.read_csv(data_path + test_data)

# Check the format
# 3219, 57
print  train_X.shape

# 3219, 1
print  train_Y.shape

# avoid warning
train_Y = np.ravel(train_Y)

# 1379, 57
print test_X.shape
train_X = scale(train_X)


# Apply the classifier
#classifier1 = svm.SVC()
#classifier2 = MLPClassifier()
#classifier3 = AdaBoostClassifier()
classifier = MLPClassifier(max_iter=1000)


# Apply cross validation on the training set to compare performance
score = cross_val_score(classifier, train_X, train_Y, cv=5)
#score1 = cross_val_score(classifier1, train_X, train_Y, cv=5)
#score2 = cross_val_score(classifier2, train_X, train_Y, cv=5)
#score3 = cross_val_score(classifier3, train_X, train_Y, cv=5)




# Print the result
print score
#print score1
#print score2
#print score3


# Predictions on the test data and write to a csv file
classifier.fit(train_X, train_Y)
predict = classifier.predict(test_X)
print predict.shape
np.savetxt(data_path + "project1_20475043.txt", predict, fmt="%.1f", delimiter=",")
#predict.to_csv(data_path + "project1_20475043", sep="\n")

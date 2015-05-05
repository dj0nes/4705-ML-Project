# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:10:30 2015

@author: Sufian
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import cross_val_predict
#from sklearn import datasets

wbfeatures = open_workbook('Sarcos_Data1_train__X.xlsx');
wbtarget = open_workbook('Sarcos_Data1_train__y.xlsx');
wbtestfeatures = open_workbook('Sarcos_Data1_test_X.xlsx');
wbtesttarget = open_workbook('Sarcos_Data1_test_y.xlsx');

features_train_data = []
s = wbfeatures.sheet_by_index(0)
num_cols = s.ncols   # Number of columns
for row_idx in range(0, s.nrows):    # Iterate through rows
    "print ('-'*40)"
    "print ('Row: %s' % row_idx)"   # Print row number
    temp = []
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = (s.cell(row_idx, col_idx).value)  # Get cell object by row, col
        temp.append(cell_obj)
    features_train_data.append(temp)
train_features = features_train_data

targets_train_data = []
s = wbtarget.sheet_by_index(0)
target = s.col(0)
for row in range(0, s.nrows):
    value = (s.cell(row, 0).value)
    targets_train_data.append(value)
train_targets = targets_train_data

features_test_data = []
s = wbtestfeatures.sheet_by_index(0)
num_cols = s.ncols   # Number of columns
for row_idx in range(0, s.nrows):    # Iterate through rows
    "print ('-'*40)"
    "print ('Row: %s' % row_idx)"   # Print row number
    temp = []
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = (s.cell(row_idx, col_idx).value)  # Get cell object by row, col
        temp.append(cell_obj)
    features_test_data.append(temp)
test_features = features_test_data

targets_test_data = []
s = wbtesttarget.sheet_by_index(0)
testtarget = s.col(0)
for row in range(0, s.nrows):
    value = (s.cell(row, 0).value)
    targets_test_data.append(value)
test_targets = targets_test_data

###############################################################################

#Ridge
#Cross-validation to set a parameter
ridge = linear_model.RidgeCV(cv=10)
ridge.fit(train_features, train_targets)
# The estimator chose automatically its lambda: 
tune_parameter = ridge.alpha_
print("Tuned parameter obtained using cross validation : %f" % tune_parameter)
#To evaluate the cross validation perfomance
Cross_validation_perfomance = ridge.score(test_features, test_targets)
print("Cross validation perfomance : %f" % Cross_validation_perfomance)

from sklearn.linear_model import Ridge

alpha = tune_parameter 
ridge = Ridge(alpha=alpha)

y_pred_ridge = ridge.fit(train_features, train_targets).predict(test_features)
r2_score_ridge = r2_score(test_targets, y_pred_ridge)
print(ridge)
print("r^2 on test data : %f" % r2_score_ridge)
plt.plot(ridge.coef_, label='Ridge coefficients')
plt.legend(loc='best')
plt.show()

###############################################################################


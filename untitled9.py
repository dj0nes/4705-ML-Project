# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:10:30 2015

@author: Sufian
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from xlrd import open_workbook
from sklearn import cross_validation
from sklearn.metrics import r2_score

wbfeatures = open_workbook('Sarcos_Data1_train__X.xlsx');
wbtarget = open_workbook('Sarcos_Data1_train__y.xlsx');
wbtestfeatures = open_workbook('Sarcos_Data1_test_X.xlsx');
wbtesttarget = open_workbook('Sarcos_Data1_test_y.xlsx');

targets = []
s = wbtarget.sheet_by_index(0)
target = s.col(0)
for row in range(0, s.nrows):
    value = (s.cell(row, 0).value)
    targets.append(value)
y = targets

testing = []
s = wbfeatures.sheet_by_index(0)
num_cols = s.ncols   # Number of columns
for row_idx in range(0, s.nrows):    # Iterate through rows
    "print ('-'*40)"
    "print ('Row: %s' % row_idx)"   # Print row number
    temp = []
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = (s.cell(row_idx, col_idx).value)  # Get cell object by row, col
        temp.append(cell_obj)
    testing.append(temp)
X = testing

testtargets = []
s = wbtesttarget.sheet_by_index(0)
testtarget = s.col(0)
for row in range(0, s.nrows):
    value = (s.cell(row, 0).value)
    testtargets.append(value)
test_y = testtargets

testtesting = []
s = wbtestfeatures.sheet_by_index(0)
num_cols = s.ncols   # Number of columns
for row_idx in range(0, s.nrows):    # Iterate through rows
    "print ('-'*40)"
    "print ('Row: %s' % row_idx)"   # Print row number
    temp = []
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = (s.cell(row_idx, col_idx).value)  # Get cell object by row, col
        temp.append(cell_obj)
    testtesting.append(temp)
test_X = testtesting


###############################################################################

lasso = linear_model.Ridge()
alphas = np.logspace(-4, -.5, 30)


scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))


plt.semilogx(alphas, scores)
# plot error lines showing +/- std. errors of the scores
plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.show()

###############################################################################

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))


plt.semilogx(alphas, scores)
# plot error lines showing +/- std. errors of the scores
plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.show()

###############################################################################
# Lasso
#Cross-validation to set a parameter
lasso = linear_model.LassoCV()
lasso.fit(X, y)
# The estimator chose automatically its lambda:
tune_parameter = lasso.alpha_
print("Tuned parameter obtained using cross validation : %f" % tune_parameter)
#To evaluate the cross validation perfomance
Cross_validation_perfomance = lasso.score(test_X, test_y)
print("Cross validation perfomance : %f" % Cross_validation_perfomance)

from sklearn.linear_model import Lasso

alpha = tune_parameter
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X, y).predict(test_X)
r2_score_lasso = r2_score(test_y, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.legend(loc='best')
plt.show()

###############################################################################
#Ridge
#Cross-validation to set a parameter
ridge = linear_model.RidgeCV()
ridge.fit(X, y)
# The estimator chose automatically its lambda: 
tune_parameter = ridge.alpha_
print("Tuned parameter obtained using cross validation : %f" % tune_parameter)
#To evaluate the cross validation perfomance
Cross_validation_perfomance = ridge.score(test_X, test_y)
print("Cross validation perfomance : %f" % Cross_validation_perfomance)

from sklearn.linear_model import Ridge

alpha = tune_parameter 
ridge = Ridge(alpha=alpha)

y_pred_ridge = ridge.fit(X, y).predict(test_X)
r2_score_ridge = r2_score(test_y, y_pred_ridge)
print(ridge)
print("r^2 on test data : %f" % r2_score_ridge)
plt.plot(ridge.coef_, label='Ridge coefficients')
plt.legend(loc='best')
plt.show()

###############################################################################
# ElasticNet
"""
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X, y).predict(test_X)
r2_score_enet = r2_score(test_y, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet) 

plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
from sklearn.decomposition import RandomizedPCA
"""

###############################################################################
regr = linear_model.Ridge(normalize = True)
regr.fit(X, y)
plt.plot(X, y, marker='o');

###############################################################################
regr = linear_model.Ridge(normalize = True)
regr.fit(X, y)
y_test = regr.predict(test_X)
plt.plot(X, y, 'o')
plt.plot(test_X, y_test)
###############################################################################



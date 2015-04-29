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

# LassoLarsCV: least angle regression

# Compute paths
print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv=30).fit(train_features, train_targets)
t_lasso_lars_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)

plt.figure()
#plt.figure(figsize=(32,18), dpi=1200) # used to expose the figure at higher resolution 
plt.plot(m_log_alphas, model.cv_mse_path_, ':')
plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
ymin, ymax = 10, 70
plt.ylim(ymin, ymax)
xmin, xmax = 1.1, 3.9
plt.xlim(xmin, xmax)

plt.show()

###############################################################################

# Lasso on testing data using learning technique from training data
m_log_alphas_modified = np.delete(m_log_alphas, 0)
#print(m_log_alphas_modified)

r2_values_store = []

for alphas in m_log_alphas_modified:
    alpha = alphas
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(train_features, train_targets).predict(test_features)
    r2_score_lasso = r2_score(test_targets, y_pred_lasso)
    r2_values_store.append(r2_score_lasso)
#print(r2_values_store)
plt.figure()
#plt.figure(figsize=(32,18), dpi=1200) # used to expose the figure at higher resolution 
plt.plot(m_log_alphas_modified, r2_values_store)
plt.xlabel('alpha values')
plt.ylabel('R^2 values')
plt.show()

# The estimator chose automatically its lambda:
tune_parameter = -np.log10(model.alpha_)
print("Tuned parameter obtained using cross validation : %f" % tune_parameter)
#To evaluate the cross validation perfomance
Cross_validation_perfomance = model.score(test_features, test_targets)
print("Cross validation perfomance : %f" % Cross_validation_perfomance)

alpha = tune_parameter
lasso = Lasso(alpha=-np.log10(model.alpha_))

y_pred_lasso = lasso.fit(train_features, train_targets).predict(test_features)
r2_score_lasso = r2_score(test_targets, y_pred_lasso)
print(lasso)
print("R^2 on test data : %f" % r2_score_lasso)
#plt.plot(lasso.coef_, label='Lasso coefficients')
#plt.xlabel('alpha values')
#plt.ylabel('R^2')
#plt.plot(m_log_alphas, r2_score_lasso)
#plt.legend(loc='best')
#plt.show()
"""m_log_alphas_modified = np.delete(m_log_alphas, 0)
#print(m_log_alphas_modified)

r2_values_store = []

for alphas in m_log_alphas_modified:
    alpha = alphas
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(train_features, train_targets).predict(test_features)
    r2_score_lasso = r2_score(test_targets, y_pred_lasso)
    r2_values_store.append(r2_score_lasso)
#print(r2_values_store)
plt.plot(m_log_alphas_modified, r2_values_store)
plt.xlabel('alpha values')
plt.ylabel('R^2 values')
plt.show()"""

###############################################################################

# Cross Validation predictions
"""from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston)
#(506, 13)"""


#lr = LinearRegression()
lr = Lasso()
#boston = datasets.load_boston()
#y = boston.target
y = np.array(train_targets[:500])
y.shape

tf = np.array(train_features[:500])
#tf.shape
tf.data
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, tf, y, cv=10)

fig,ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
#fig.set_size_inches(32,18)
#fig.set_dpi(1200)
fig.show()

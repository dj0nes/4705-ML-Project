# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:10:30 2015

@author: Sufian
"""

import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, ElasticNetCV
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

#ElasticNet
#Cross-validation to set a parameter
elasticnet = ElasticNetCV()
elasticnet.fit(train_features, train_targets)
m_log_alphas = elasticnet.alphas_
# The estimator chose automatically its lambda: 
tune_parameter = elasticnet.alpha_
print("Tuned parameter obtained using cross validation : %f" % tune_parameter)
#To evaluate the cross validation perfomance
Cross_validation_perfomance = elasticnet.score(test_features, test_targets)
print("Cross validation perfomance : %f" % Cross_validation_perfomance)

alpha = tune_parameter 
elasticnet = ElasticNet(alpha=alpha)

y_pred_elasticnet = elasticnet.fit(train_features, train_targets).predict(test_features)
r2_score_elasticnet = r2_score(test_targets, y_pred_elasticnet)
print(elasticnet)
print("r^2 on test data : %f" % r2_score_elasticnet)

###############################################################################

# ElasticNet on testing data using learning technique from training data
m_log_alphas_modified = np.delete(m_log_alphas, 0)
#print(m_log_alphas_modified)

r2_values_store = []

for alphas in m_log_alphas_modified:
    alpha = alphas
    elasticnet = ElasticNet(alpha=alpha)
    y_pred_elasticnet = elasticnet.fit(train_features, train_targets).predict(test_features)
    r2_score_elasticnet = r2_score(test_targets, y_pred_elasticnet)
    r2_values_store.append(r2_score_elasticnet)
#print(r2_values_store)
plt.figure()
#plt.figure(figsize=(16,9), dpi=1200) # used to expose the figure at higher resolution 
plt.plot(m_log_alphas_modified, r2_values_store)
plt.xlabel('alpha values')
plt.ylabel('R^2 values')
xmin, xmax = -2, 300
plt.xlim(xmin, xmax)
plt.show()

###############################################################################

# Compute train and test errors
#alphas = np.logspace(-5, 1, 60)
alphas = np.logspace(-5, 1, 60)
enet = ElasticNet(l1_ratio=0.7)
train_errors = list()
test_errors = list()
for alpha in alphas:
    enet.set_params(alpha=alpha)
    enet.fit(train_features, train_targets)
    train_errors.append(enet.score(train_features, train_targets))
    test_errors.append(enet.score(test_features, test_targets))

i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with optimal regularization parameter
enet.set_params(alpha=alpha_optim)
coef_ = enet.fit(train_features[:500], test_targets).coef_

###############################################################################
# Plot results functions

#plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0.6, 1.0])
plt.xlim([0.00001, 10])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')

# Show estimated coef_ vs true coef
#plt.subplot(2, 1, 2)
#plt.plot(coef_, label='True coef')
#plt.plot(coef_, label='Estimated coef')
plt.legend()
#plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
plt.show()

###############################################################################

#lr = LinearRegression()
lr = ElasticNet()
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
#fig.set_size_inches(16,9)
#fig.set_dpi(1200)
fig.show()

###############################################################################




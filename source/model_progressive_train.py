'''
this script takes a input file and run the progressive training validation, meaning that, 
the model is trained based on the features from the past and predict on the current.
Then, the current price is taken as training and test on the next feature(price)

Created on Mar 26, 2014

@author: Songfan
'''

import numpy as np
import math
from math import log
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pds
from time import gmtime, strftime
import scipy
import sys
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from string import punctuation
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
import time
from scipy import sparse
from matplotlib import *
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import operator
import auc


def auc_scorer(estimator, X, y):
    predicted = estimator.predict_proba(X)[:,1]
    return auc.auc(y, predicted)
    
def normalizeToFirstCol(data):
    r,c = data.shape
    data_norm = np.array([data[:,i] / data[:,0] for i in range(c)]).transpose()
    return data_norm

################################################################################
# LOAD DATA
################################################################################
print "loading data.."
train = np.array(pds.read_table('../data/training.csv', sep = ","))

################################################################################
# ORANIZE THE TRAIN DATA
################################################################################
n_days = train.shape[1] // 5    # 500 days
n_stock = train.shape[0]
FD = 5  # 5 per day, namely, O, MA, MI, C, V
window_size = 6 # 6 day as the window for feature computing

# select the opening price and closing price only
open_price = train[:,range(1,train.shape[1],FD)]
clos_price = train[:,range(4,train.shape[1],FD)]

# model
model_ridge = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=9081)
model_randomforest = RandomForestClassifier(n_estimators = 200)
model_stacker = lm.LogisticRegression()

# result
cv_score = []

################################################################################
# PROGRESSIVE DATA SELECTION
################################################################################
portion = range(1,10) # using 10%, 20%, ... of the data as training, the compliment as testing
for curr_portion in portion:
    # generate current training and test data a better way, cat open (window_size+1) and close (window_size)
    # training
    train_day = curr_portion*n_days/10
    temp_open = [open_price[:,w:w+window_size+1] for w in range(train_day-window_size)]
    temp_clos = [clos_price[:,w:w+window_size] for w in range(train_day-window_size)]
    temp_open = np.vstack(temp_open)
    temp_clos = np.vstack(temp_clos)
    curr_train_X = normalizeToFirstCol(np.hstack((temp_open, temp_clos)))   # n1*13 matrix
    
    temp_diff = [(clos_price[:,w+window_size+1] - open_price[:,w+window_size+1] > 0) + 0 for w in range(train_day-window_size)]
    curr_train_y = np.hstack(temp_diff) # 1*n vector
    
    # test
    temp_open = [open_price[:,w:w+window_size+1] for w in range(train_day,n_days-window_size-1)]
    temp_clos = [clos_price[:,w:w+window_size] for w in range(train_day,n_days-window_size-1)]
    temp_open = np.vstack(temp_open)
    temp_clos = np.vstack(temp_clos)
    curr_test_X = normalizeToFirstCol(np.hstack((temp_open, temp_clos)))
    
    temp_diff = [(clos_price[:,w+window_size+1] - open_price[:,w+window_size+1] > 0) + 0 for w in range(train_day,n_days-window_size-1)]
    curr_test_y = np.hstack(temp_diff) # 1*n vector
    
    print "data preparation done"
    
    ################################################################################
    # CV
    ################################################################################
    print "cross validation"
    
    pred_ridge = model_ridge.fit(curr_train_X, curr_train_y).predict_proba(curr_test_X)[:,1]
    pred_randomforest = model_randomforest.fit(curr_train_X, curr_train_y).predict_proba(curr_test_X)[:,1]
    
    # stack feature
    new_X = np.hstack((np.array(pred_ridge).reshape(len(pred_ridge), 1), np.array(pred_randomforest).reshape(len(pred_randomforest), 1)))
    curr_mean = np.mean(cross_validation.cross_val_score(model_stacker, new_X, curr_test_y.reshape(curr_test_y.shape[0]), cv=5, scoring = auc_scorer))
    cv_score.append((train_day, curr_mean))
    print train_day, curr_mean

print cv_score














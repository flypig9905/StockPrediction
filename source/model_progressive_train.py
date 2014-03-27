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
                
def normalize10day(stocks):
    def process_column(i):
        if operator.mod(i, 5) == 4:
            return np.log(stocks[:,i] + 1)
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ process_column(i) for i in range(31)]).transpose()
    return stocks_dat
    
def normalizeToFirstCol(data):
    r,c = data.shape
    data_norm = [data[:,i] / data[:,0] for i in range(c)]
    return data_norm

# <codecell>

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
close_price = train[:,range(4,train.shape[1],FD)]
print open_price.shape
print close_price.shape

''' progressive data selection '''
portion = range(1,10) # using 10%, 20%, ... of the data as training, the compliment as testing
for curr_portion in portion:
    # generate current training and test data
    train_day = curr_portion*n_days/10
    curr_train = []
    for w in range(0,train_day-window_size):
        temp = [np.vstack((open_price[:,i+w], close_price[:,i+w])) for i in range(window_size)]
        temp = np.vstack(temp)
        temp = np.vstack((temp,open_price[:,window_size+w]))  # add the O feature of day whose C is to predict
        
        temp_y = ((close_price[:,window_size+w] - open_price[:,window_size+w]) > 0) + 0
        if curr_train == []: 
            curr_train = temp
            curr_y_train = temp_y
        else:
            curr_train = np.vstack((curr_train, temp))
            curr_y_train = np.vstack((curr_y_train, temp_y))
    
    curr_test = []    
    for w in range(train_day, n_days-window_size):
        temp = [np.vstack((open_price[:,i+w], close_price[:,i+w])) for i in range(window_size)]
        temp = np.vstack(temp)
        temp = np.vstack((temp,open_price[:,window_size+w]))  # add the O feature of day whose C is to predict
        
        temp_y = ((close_price[:,window_size+w] - open_price[:,window_size+w]) > 0) + 0
        if curr_test == []: 
            curr_test = temp
            curr_y_test = temp_y
        else:
            curr_test = np.vstack((curr_test, temp))
            curr_y_test = np.vstack((curr_y_test, temp_y))
    
    # need to do normalization
    
    print curr_train.shape, curr_y_train.shape
    print curr_test.shape, curr_y_test.shape
    
    
    ''' cross validation '''
    
#                               
#                               
                               
                               
##X_windows_normalized = [normalize10day(w) for w in X_windows]
##X = np.vstack(X_windows_normalized)
##X_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(n_windows)))
#
##X = np.hstack((X_stockindicators, X_stockdata))
##X = X_stockdata
#
## read in the response variable
## 46: O starting from day 10
## 49: C starting from day 10
#y_stockdata = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
#
## if C>0, then it is positive label
#y = (y_stockdata[:,1] - y_stockdata[:,0] > 0) + 0   # True + 0 = 1; False + 0 = 0
#
## solely using the O and C feature
#X_test = X_test[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
#X = X[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
#print "data preparation done"
#
## <codecell>
#
## BEST IS 133
#model_ridge = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=9081)
#model_randomforest = RandomForestClassifier(n_estimators = 200)
#
#pred_ridge = []
#pred_randomforest = []
#new_Y = []
#fold_num = 2
#for i in range(fold_num):
#    indxs = np.arange(i, X.shape[0], fold_num)    # select 1 for every 10 item
#    indxs_to_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], fold_num))) # use list difference to compute the current training index
#    pred_ridge = pred_ridge + list(model_ridge.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1]) # [:,1] means select the prob of predicting 1
#    pred_randomforest = pred_randomforest + list(model_randomforest.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1])                               
#    new_Y = new_Y + list(y[indxs,:])    # new_Y is re-arranged Y based on the test index
#
#''' stack feature '''
## new_X is the new feature that consist the prob prediction from ridge and random forest                              
#new_X = np.hstack((np.array(pred_ridge).reshape(len(pred_ridge), 1), np.array(pred_randomforest).reshape(len(pred_randomforest), 1)))
#print new_X
#new_Y = np.array(new_Y).reshape(len(new_Y), 1)
#
## logistic regression on new feature
#model_stacker = lm.LogisticRegression()
## both feature
#print np.mean(cross_validation.cross_val_score(model_stacker, new_X, new_Y.reshape(new_Y.shape[0]), cv=5, scoring = auc_scorer))
## ridge only
#print np.mean(cross_validation.cross_val_score(model_stacker, new_X[:,0].reshape(new_X.shape[0],1), new_Y.reshape(new_Y.shape[0]), cv=5, scoring = auc_scorer))
## random forest only
#print np.mean(cross_validation.cross_val_score(model_stacker, new_X[:,1].reshape(new_X.shape[0],1), new_Y.reshape(new_Y.shape[0]), cv=5, scoring = auc_scorer))
#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
prediction for multi-modal
'''

import pandas as pd


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score , precision_score, recall_score, f1_score
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error, median_absolute_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,KFold,ShuffleSplit,LeaveOneOut
from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy as np
import datetime
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import Imputer
import time
from ..prediction.regressor import * 
from ..prediction.classifier import *
from ..feature.featureEngineering import *
def preprocessData(X_train,X_test):

      

def ensembleRegressor(path,sources,target,identification,train_id,test_id,method='weighted',metric='neg_mean_squared_error',cv_number=3,seed=40):
    




def ensembleClassifier(path,sources,target,identification,train_id,test_id,method='weighted',metric='accuracy',cv_number=3,seed=40):
    


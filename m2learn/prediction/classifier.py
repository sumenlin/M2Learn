#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
regression model for one file
'''

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error, median_absolute_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,KFold,ShuffleSplit,LeaveOneOut
from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy as np
import datetime
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier
import time
from sklearn.metrics import accuracy_score, roc_auc_score , precision_score, recall_score, f1_score


# seed = 40

def preprocessData(X_train,X_test):

    


def scoring(y_pred,y_test,metric='accuracy'):
    




def hyper_tune(name,X,y,seed = 40,cv_number = 3, metric = 'accuracy'):
    

    

def oneClassifier(data,identification,test_msk,train_msk,Xheads_list,metric='accuracy',cv_number = 3,seed = 40):
    """Performe regression on data.
    
    :param data: input data
    :type data: data frame
    :param identification: the column name of identification/key among all data sources. If ``None``, it will raise errors.  
    :type identification: string (default=``None``)
    :param test_msk: the identification values of testing data
    :type test_msk: list
    :param train_msk: the identification values of training data.
    :type train_msk: list
    :param Xheads_list: list of selected feature lists
    :type target: list of lists
    :param metric: the metric of performance. 
                 If ``accuracy``, use accuracy score;
                 if ``f1``, use F1 score, only for binary class;
                 if ``precision``, use precision score, only for binary class; 
                 if ``recall``, use recall score, only for binary class; 
                 if ``f1_macro``, calculate f1 score for each label, and find their unweighted mean; 
                 if ``f1_micro``, calculate f1 globally by counting the total true positives, false negatives and false positives; 
                 if ``f1_weighted``, calculate f1 for each label, and find their average weighted by support;
                 if ``precision_macro``, calculate precision score for each label, and find their unweighted mean;
                 if ``precision_micro``, calculate precision score globally by counting the total true positives, false negatives and false positives;
                 if ``precision_weighted``, calculate precision score for each label, and find their average weighted by support;
                 if ``recall_macro``, calculate recall score for each label, and find their unweighted mean;
                 if ``recall_micro``, calculate recall score globally by counting the total true positives, false negatives and false positives;
                 if ``recall_weighted``, calculate recall score for each label, and find their average weighted by support;
    :type metric: string (default='accuracy')
    :param cv_number: the number of folds used in cross validation.  
    :type cv_number: int (default=3)
    :param seed: the seed of the pseudo random number generator to use if applicable. 
    :type seed: int (default=40)
    :returns: optimal fitting results including cross validation metrics, selected features, selected model and corresponding parameters.
    """
    

    





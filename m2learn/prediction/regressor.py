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
# from sklearn.metrics import make_scorer,roc_curve, auc, roc_auc_score , precision_score, recall_score, f1_score,confusion_matrix, classification_report
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
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import Imputer
import time


# seed = 40

def preprocessData(X_train,X_test):

    


def scoring(y_pred,y_test,metric='neg_mean_squared_error'):
    


def hyper_tune(name,X,y,seed = 40,cv_number = 3, metric = 'neg_mean_squared_error'):
    

    



def oneRegressor(data,identification,test_msk,train_msk,Xheads_list,metric='neg_mean_squared_error',cv_number = 3,seed = 40):
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
    :type Xheads_list: list of lists
    :param metric: the metric of performance. 
                 If ``explained_variance``, use explained variance regression score;
                 if ``neg_mean_absolute_error``, use mean absolute error regression loss; 
                 if ``neg_mean_squared_error``, use mean squared error regression loss; 
                 if ``neg_mean_squared_log_error``, use mean squared logarithmic error regression loss; 
                 if ``neg_median_squared_error``, use median absolute error regression loss; 
                 if ``r2``, use r2 score.
    :type metric: string (default='neg_mean_squared_error')
    :param cv_number: the number of folds used in cross validation.  
    :type cv_number: int (default=3)
    :param seed: the seed of the pseudo random number generator to use if applicable. 
    :type seed: int (default=40)
    :returns: optimal fitting results including cross validation metrics, selected features, selected model and corresponding parameters.
    """

    


    





#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
regression model for one file
'''

import pandas as pd
from collections import Counter


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
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import Imputer
import time


# seed = 40

def preprocessData(X_train,X_test):

    # df = X_train.values
    df = X_train
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(df)
    # X_train = pd.DataFrame(x_scaled,index=X_train.index, columns=X_train.columns)
    X_train = x_scaled
    
    df = X_test
    x_scaled = scaler.transform(df)
    X_test = x_scaled#pd.DataFrame(x_scaled,index=X_test.index, columns=X_test.columns)

    return X_train,X_test


def scoring(y_pred,y_test,metric='neg_mean_squared_error'):
    if metric == 'explained_variance':
        corre = explained_variance_score(y_test,y_pred)
    elif metric == 'neg_mean_absolute_error':
        corre = (-1)*mean_absolute_error(y_test,y_pred)
    elif metric == 'neg_mean_squared_error':
        corre = (-1)*mean_squared_error(y_test,y_pred)
    elif metric == 'neg_mean_squared_log_error':
        corre = (-1)*mean_squared_log_error(y_test,y_pred)
    elif metric == 'neg_median_squared_error':
        corre = (-1)*median_squared_error(y_test,y_pred)
    elif metric == 'r2':
        corre = r2_score(y_test,y_pred)
    return corre



def hyper_tune(name,X,y,seed = 40,cv_number = 3, metric = 'neg_mean_squared_error'):
    

    if name == 'LinearRegression':
        return linear_model.LinearRegression(),{}
    if name == 'Linear+L2norm':
        parameters = {'alpha':[0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]}
        # model = 
        gsearch = GridSearchCV(linear_model.Ridge(), parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = linear_model.Ridge(alpha=para['alpha'])
        # print type(clf)
        return clf,para
    if name == 'Linear+L1norm':
        parameters = {'alpha':[0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]}
        model = linear_model.Lasso()
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = linear_model.Lasso(alpha=para['alpha'])
        return clf,para
    if name == 'Linear+LARS':
        parameters = {'alpha':[0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]}
        model = linear_model.LassoLars()
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = linear_model.LassoLars(alpha=para['alpha'])
        return clf,para
    if name == 'Linear+Bayes':
        clf = linear_model.BayesianRidge()
        return clf,{}
    if name == 'svr_rbf':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2],'gamma':[0.1,0.5,1]}
        model = svm.SVR(kernel='rbf')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVR(kernel='rbf', C=para['C'], gamma=para['gamma'])
        return clf,para
    if name == 'svr_lin':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2]}
        model = svm.SVR(kernel='linear')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVR(kernel='linear', C=para['C'])
        return clf,para
    if name == 'svr_poly':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2],'degree':[1,2]}
        model = svm.SVR(kernel='poly')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVR(kernel='poly', C=para['C'], degree=para['degree'])
        return clf,para
    if name == 'cart':
        parameters = {'min_samples_split':[2,4,6,8]}
        model = DecisionTreeRegressor(random_state=seed)
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = DecisionTreeRegressor(min_samples_split=para['min_samples_split'],random_state=seed)
        return clf,para
    if name == 'rf':
        parameters = {'n_estimators':[20,50,80,100],'min_samples_split':[2,4,6,8]}
        model = RandomForestRegressor(random_state=seed)
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = RandomForestRegressor(n_estimators=para['n_estimators'],min_samples_split=para['min_samples_split'],random_state=seed)
        return clf,para



def oneRegressor(data,test_msk,train_msk,Xheads_list,metric='neg_mean_squared_error',cv_number = 3,seed = 40):
    Xheads = list(set(data.columns.tolist())-set(['id_','target']))
    data_train = data[data.id_.isin(train_msk)].reset_index(drop=True)
    Y = data_train.loc[:,'target']
    X0 = data_train.loc[:,Xheads]
    data_test = data[data.id.isin(test_msk)].reset_index(drop=True)
    data_test = data_test.ix[:,Xheads+'target'] #testing
    Y_test_w = data_test.loc[:,'target']
    X0_test_w = data_test.loc[:,Xheads]
    
    # results = pd.DataFrame(columns=['Xheads', 'corr.avg','scores','model','parameters'])
    # iresults = 0 

    maxS = -1*float('inf')

    for nXheads in Xheads_list:
        X = X0.ix[:,nXheads]
        # print X.info()
        X = X.as_matrix()
        y = Y.ix[:,'target']
        y = y.as_matrix() 

        X_test_w = X0_test_w.ix[:,nXheads].as_matrix()
        y_test_w = Y_test_w.ix[:,'target'].tolist()

        X,X_test_w = preprocessData(X,X_test_w)



        names = ['LinearRegression',  'Linear+L2norm', 'Linear+LARS','Linear+L1norm',  'Linear+Bayes', 'svr_rbf','svr_lin', 'svr_poly','cart','rf']

        # for name, clf in zip(names, models):
        for name in names:
            #tune hyper-para
            clf,paras = hyper_tune(name,X,y,seed,cv_number,metric)
            # print name,type(clf)
            # cross
            kf = KFold(n_splits=cv_number, random_state=seed, shuffle=True)
            scores = []
            # print X
            yy_test=[]
            yy_pred=[]
            for train_index, test_index in kf.split(X):
                # print test_index
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                #normalization
                X_train,X_test = preprocessData(X_train,X_test)


                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                y_pred=y_pred.tolist()
                corre = scoring(y_test,y_pred,metric)
                scores.append(corre)
            if np.mean(scores)>maxS:
                bestX = nXheads
                bestM = clf
                bestN = name 
                bestPara = ', '.join("%s=%r" % (key,val) for (key,val) in paras.iteritems())
                maxS = np.mean(scores)

            # results.loc[iresults,'Xheads'] = nXheads
            # results.loc[iresults,'corr.avg'] = np.mean(scores)
            # results.loc[iresults,'scores'] = scores
            # results.loc[iresults,'model'] = name
            # results.loc[iresults,'parameters'] = ', '.join("%s=%r" % (key,val) for (key,val) in paras.iteritems())
            # iresults += 1
            

    # results = results.sort_values(by='corr.avg',ascending=False).reset_index(drop=True)
    # print results.loc[0,:]
    return maxS,bestX,bestM,bestN,bestPara


    





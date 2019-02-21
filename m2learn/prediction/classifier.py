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


def scoring(y_pred,y_test,metric='accuracy'):
    if metric == 'accuracy':
        corre = accuracy_score(y_test,y_pred)
    elif metric == 'f1':
        corre = f1_score(y_test,y_pred)
    elif metric == 'f1_micro':
        corre = f1_score(y_test,y_pred,average='micro')
    elif metric == 'f1_macro':
        corre = f1_score(y_test,y_pred,average='macro')
    elif metric == 'f1_weighted':
        corre = f1_score(y_test,y_pred,average='weighted')
    elif metric == 'precision':
        corre = precision_score(y_test,y_pred)
    elif metric == 'precision_micro':
        corre = precision_score(y_test,y_pred,average='micro')
    elif metric == 'precision_macro':
        corre = precision_score(y_test,y_pred,average='macro')
    elif metric == 'precision_weighted':
        corre = precision_score(y_test,y_pred,average='weighted')
    elif metric == 'recall':
        corre = recall_score(y_test,y_pred)
    elif metric == 'recall_micro':
        corre = recall_score(y_test,y_pred,average='micro')
    elif metric == 'recall_macro':
        corre = recall_score(y_test,y_pred,average='macro')
    elif metric == 'recall_weighted':
        corre = recall_score(y_test,y_pred,average='weighted')
    return corre




def hyper_tune(name,X,y,seed = 40,cv_number = 3, metric = 'accuracy'):
    

    if name == 'LogisticRegression':
        return linear_model.LogisticRegression(),{}
    if name == 'svc_rbf':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2],'gamma':[0.1,0.5,1]}
        model = svm.SVC(kernel='rbf')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVR(kernel='rbf', C=para['C'], gamma=para['gamma'])
        return clf,para
    if name == 'svc_lin':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2]}
        model = svm.SVC(kernel='linear')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVC(kernel='linear', C=para['C'])
        return clf,para
    if name == 'svc_poly':
        parameters = {'C':[1e-2,1e-1,1,1e1,1e2],'degree':[1,2]}
        model = svm.SVC(kernel='poly')
        gsearch = GridSearchCV(model, parameters, cv=cv_number, scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        clf = svm.SVC(kernel='poly', C=para['C'], degree=para['degree'])
        return clf,para
    if name == 'cart': 
        parameters = {'min_samples_leaf':[10,20,30,50]}
        model = DecisionTreeClassifier(random_state=seed)
        gsearch = GridSearchCV(model, parameters, cv=cv_number,scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        score = gsearch.best_score_
        clf = DecisionTreeClassifier(min_samples_leaf=para['min_samples_leaf'],random_state=seed)
        return clf,para
    if name == 'rf':
        parameters = {'min_samples_leaf':[10,20,30,50],'n_estimators':[10,30,50,100]}
        model = RandomForestClassifier(random_state=seed)
        gsearch = GridSearchCV(model, parameters, cv=cv_number,scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        score = gsearch.best_score_
        clf = RandomForestClassifier(min_samples_leaf=para['min_samples_leaf'],n_estimators=para['n_estimators'],random_state=seed)
        return clf,para
    if name == 'knn':
        parameters = {'n_neighbors':[3,5,10]}
        model = KNeighborsClassifier()
        gsearch = GridSearchCV(model, parameters, cv=cv_number,scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        score = gsearch.best_score_
        clf = KNeighborsClassifier(n_neighbors=para['n_neighbors'])
        return clf,para
    if name == 'ada':
        parameters = {'n_estimators':[10,20,30,50,100]}
        model = AdaBoostClassifier(random_state=seed)
        gsearch = GridSearchCV(model, parameters, cv=cv_number,scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        score = gsearch.best_score_
        clf = AdaBoostClassifier(n_estimators=para['n_estimators'],random_state=seed)
        return clf,para
    if name == 'xgb':
        # parameters = {'max_depth':[3,5,10,20,50],'n_estimators':[10,20,30,50,100]}
        parameters = {'max_depth':range(2, 10, 1),'n_estimators':range(30, 50, 2)}
        model = XGBClassifier(random_state=seed, learning_rate=0.01,objective="binary:logitraw")
        # print model
        gsearch = GridSearchCV(model, parameters, cv=cv_number,scoring=metric)
        gsearch.fit(X, y)
        para = gsearch.best_params_
        score = gsearch.best_score_
        # print para,score
        clf = XGBClassifier(n_estimators=para['n_estimators'],max_depth=para['max_depth'], learning_rate=0.01,random_state=seed,objective="binary:logitraw")
        return clf,para


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
    data = data.rename(index=str,columns={identification:'id_'})
    Xheads = list(set(data.columns.tolist())-set(['id_','target']))
    data_train = data[data.id_.isin(train_msk)].reset_index(drop=True)
    Y = data_train.loc[:,['target']]
    X0 = data_train.loc[:,Xheads]
    data_test = data[data.id_.isin(test_msk)].reset_index(drop=True)
    data_test = data_test.ix[:,Xheads+['target']] #testing
    Y_test_w = data_test.loc[:,['target']]
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



        names = ['LogisticRegression',  'knn', 'ada',  'xgb', 'svc_rbf','svc_lin', 'svc_poly','cart','rf']

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


    





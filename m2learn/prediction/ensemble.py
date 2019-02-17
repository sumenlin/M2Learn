#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
prediction for multi-modal
'''

import pandas as pd
from collections import Counter


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
from ..prediction.regressor import * 
from ..prediction.classifier import *
from ..feature.featureEngineering import *
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

def ensembleRegressor(path,sources,target,identification,train_id,test_id,method='weighted',metric='neg_mean_squared_error',cv_number=3,seed=40):
    sources_scores = [str(i)+'_score' for i, _ in enumerate(sources)]
    results = pd.DataFrame(columns = ['id_'])
    curr = []
    for isource,source in enumerate(sources):
        tmp_result = pd.DataFrame()

        data = pd.read_csv(path+source)
        data = data.rename(index=str,columns={identification:'id_'})
        target = pd.read_csv(path+target)
        yhead = list(set(target.columns.tolist())-set([identification]))[0]
        target = target.rename(index=str,columns={yhead:'target',identification:'id_'})
        data = pd.merge(data,target,on='id_')

        data_train = data[data.id_.isin(train_id)].reset_index(drop=True)
        Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = 'all',N = None)
        bestS,bestX,clf,bestN,bestPara = oneRegressor(data,test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = seed)

        Y = data_train.loc[:,'target'].as_matrix()
        X = data_train.ix[:,bestX].as_matrix()
        each_test = data[data.id_.isin(test_id)].drop(['id_','target'],axis=1).as_matrix()
        X,each_test = preprocessData(X,each_test)
        clf.fit(X,Y) 
        curr = clf.predict(each_test).tolist()
        tmp_result['id_'] = data[data.id_.isin(test_id)].id_.tolist()
        tmp_result[str(isource)] = curr
        if bestS<0:
            bestS = 1.0/abs(bestS)
        tmp_result[str(isource)+'_score'] = [bestS]*len(data[data.id_.isin(test_id)].id_.tolist())

        results = pd.merge(results,tmp_result,on='id_',how = 'outer')
        results = results.drop_duplicates().reset_index(drop=True)

    y_pred_avg = []
    y_pred_weight = []
    y_pred_largest = []
    for i in range(len(results.id.tolist())):
        preds = []
        scores = []
        for isource,source in enumerate(sources):
            tmp_pred = results.ix[i,str(isource)]
            tmp_score = results.ix[i,str(isource)+'_score']
            if not np.isnan(tmp_pred):
                preds.append(tmp_pred)
                scores.append(tmp_score*(-1))
            else:
                pass
        y_pred_avg.append(np.mean(preds))
        sums = 0
        for i in range(len(scores)):
            # sums += (sum(scores)-scores[i])*preds[i]
            sums += scores[i]*preds[i]
        if sum(scores)==0:
            print results.ix[i,:],scores
        # y_pred_weight.append(sums/(sum(scores)*2))
        y_pred_weight.append(sums/(sum(scores)))
        sss = scores[0]
        best = preds[0]
        for j in range(len(scores)):
            if scores[j]>sss:
                sss = scores[j]
                best = preds[j]
        y_pred_largest.append(best)

    results['average'] = y_pred_avg
    results['weighted'] = y_pred_weight
    results['largest'] = y_pred_largest
    target = pd.read_csv(path+target)
    yhead = list(set(target.columns.tolist())-set([identification]))[0]
    target = target.rename(index=str,columns={yhead:'target',identification:'id_'})

    results = pd.merge(results,y_true,on='id')
    
    return results.target.tolist(),results[method].tolist()




def ensembleClassifier(path,sources,target,identification,train_id,test_id,method='weighted',metric='accuracy',cv_number=3,seed=40):
    sources_scores = [str(i)+'_score' for i, _ in enumerate(sources)]
    results = pd.DataFrame(columns = ['id_'])
    curr = []
    for isource,source in enumerate(sources):
        tmp_result = pd.DataFrame()

        data = pd.read_csv(path+source)
        data = data.rename(index=str,columns={identification:'id_'})
        target = pd.read_csv(path+target)
        yhead = list(set(target.columns.tolist())-set([identification]))[0]
        target = target.rename(index=str,columns={yhead:'target',identification:'id_'})
        data = pd.merge(data,target,on='id_')

        data_train = data[data.id_.isin(train_id)].reset_index(drop=True)
        Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = 'all',N = None)
        bestS,bestX,clf,bestN,bestPara = oneClassifier(data,test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = seed)

        Y = data_train.loc[:,'target'].as_matrix()
        X = data_train.ix[:,bestX].as_matrix()
        each_test = data[data.id_.isin(test_id)].drop(['id_','target'],axis=1).as_matrix()
        X,each_test = preprocessData(X,each_test)
        clf.fit(X,Y) 
        curr = clf.predict(each_test).tolist()
        tmp_result['id_'] = data[data.id_.isin(test_id)].id_.tolist()
        tmp_result[str(isource)] = curr
        if bestS<0:
            bestS = 1.0/abs(bestS)
        tmp_result[str(isource)+'_score'] = [bestS]*len(data[data.id_.isin(test_id)].id_.tolist())

        results = pd.merge(results,tmp_result,on='id_',how = 'outer')
        results = results.drop_duplicates().reset_index(drop=True)

    y_pred_avg = []
    y_pred_weight = []
    y_pred_largest = []
    for i in range(len(results.id.tolist())):
        preds = []
        scores = []
        for isource,source in enumerate(sources):
            tmp_pred = results.ix[i,str(isource)]
            tmp_score = results.ix[i,str(isource)+'_score']
            if not np.isnan(tmp_pred):
                preds.append(tmp_pred)
                scores.append(tmp_score*(-1))
            else:
                pass
        y_pred_avg.append(np.mean(preds))
        sums = 0
        for i in range(len(scores)):
            # sums += (sum(scores)-scores[i])*preds[i]
            sums += scores[i]*preds[i]
        if sum(scores)==0:
            print results.ix[i,:],scores
        # y_pred_weight.append(sums/(sum(scores)*2))
        y_pred_weight.append(sums/(sum(scores)))
        sss = scores[0]
        best = preds[0]
        for j in range(len(scores)):
            if scores[j]>sss:
                sss = scores[j]
                best = preds[j]
        y_pred_largest.append(best)

    results['average'] = y_pred_avg
    results['weighted'] = y_pred_weight
    results['largest'] = y_pred_largest
    target = pd.read_csv(path+target)
    yhead = list(set(target.columns.tolist())-set([identification]))[0]
    target = target.rename(index=str,columns={yhead:'target',identification:'id_'})

    results = pd.merge(results,y_true,on='id')
    
    return results.target.tolist(),results[method].tolist()

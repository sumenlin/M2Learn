#!/usr/bin/python
# encoding=utf8
'''
Targets: 
general model
'''

from ..prediction.regressor import * 
from ..prediction.classifier import *
from ..feature.featureEngineering import *
from ..prediction.ensemble import *
from ..preprocessing.preprocessing import *


def generalRegressor(train_msk,test_msk,path = '../data/', target = 'target', identification = None, 
                     compliance = 0.0, 
                     featureImp = True, featureImputor = 'mean', 
                     modalImpute = True, modalImputor = {},clusterIMP = 5,
                     replace = False,random_state = 40,
                     pca_number = None,cv_number = 3,
                     metric = 'neg_mean_squared_error',
                     selectionFeatureKind = 'all',N = None):
    """Perform feature imputation for the data "modal".
    
    :param modal: data 
    :type modal: dataframe
    :param featureImputor: method for imputation
    :type featureImputor: int, str, optional (default='mean')
    :returns: imputed data
    :rtype: dataframe
    
    """
    df, target, featureName,fileName = dataPreprocessing(path = path, target = target, identification = identification, compliance = compliance, featureImp = featureImp, featureImputor = featureImputor, modalImpute = modalImpute, modalImputor = modalImputor,replace = replace,random_state = random_state,clusterIMP = clusterIMP)
    #PCA method
    df = df.rename(index=str,columns={identification:'id_'})
    yhead = list(set(target.columns.tolist())-set([identification]))[0]
    target = target.rename(index=str,columns={identification:'id_',yhead:'target'})
    df = pd.merge(df,target,on='id')
    data_train = df[df.id_.isin(train_msk)].reset_index(drop=True)
    data_test = df[df.id_.isin(test_msk)].reset_index(drop=True)

    maxS = -1*float('inf')
    if pca_number==None:
        for pca_n in [3,5,10]:
            data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
            data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
            data_train_pca['id_'] = data_train.id_.tolist()
            data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
            data_test_pca['id_'] = data_test.id_.tolist()
            data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
            bestS,bestX,clf,bestN,bestPara = oneRegressor(data_pca,test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)
            if bestS>maxS:
                bestM = clf
                bestPCA = pca_n
                maxS = bestS
                tmp = data_pca
    else:
        data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
        data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
        data_train_pca['id_'] = data_train.id_.tolist()
        data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
        data_test_pca['id_'] = data_test.id_.tolist()
        data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
        maxS,bestX,clf,bestN,bestPara = oneRegressor(data_pca,test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)

    #regression
    Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = 'all',N = None)
    bestS,bestX,clf,bestN,bestPara = oneRegressor(data,test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = random_state)

    if bestS>maxS:
        print "the selected model: ", clf 
        print "the selected features: ", bestX
        clf.fit(data_train.ix[:,bestX].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(data_test.ix[:,bestX]).tolist()
        return data_test['target'].tolist(),y_pred
    else:
        print "the selected model: ", bestM 
        print "the selected PCA: ", bestPCA
        bestM.fit(tmp[tmp.id_.isin(train_msk)].ix[:,range(bestPCA)].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(tmp[tmp.id_.isin(test_msk)].ix[:,range(bestPCA)]).tolist()
        return data_test['target'].tolist(),y_pred




def generalClassifier(train_msk,test_msk,path = '../data/', target = 'target', identification = None, 
                     compliance = 0.0, 
                     featureImp = True, featureImputor = 'mean', 
                     modalImpute = True, modalImputor = {},clusterIMP = 5,
                     replace = False,random_state = 40,
                     pca_number = None,cv_number = 3,
                     metric = 'accuracy',
                     selectionFeatureKind = 'all',N = None):
    """Perform feature imputation for the data "modal".
    
    :param modal: data 
    :type modal: dataframe
    :param featureImputor: method for imputation
    :type featureImputor: int, str, optional (default='mean')
    :returns: imputed data
    :rtype: dataframe
    
    """
    df, target, featureName,fileName = dataPreprocessing(path = path, target = target, identification = identification, compliance = compliance, featureImp = featureImp, featureImputor = featureImputor, modalImpute = modalImpute, modalImputor = modalImputor,replace = replace,random_state = random_state,clusterIMP = clusterIMP)
    #PCA method
    df = df.rename(index=str,columns={identification:'id_'})
    yhead = list(set(target.columns.tolist())-set([identification]))[0]
    target = target.rename(index=str,columns={identification:'id_',yhead:'target'})
    df = pd.merge(df,target,on='id_')
    data_train = df[df.id_.isin(train_msk)].reset_index(drop=True)
    data_test = df[df.id_.isin(test_msk)].reset_index(drop=True)

    maxS = -1*float('inf')
    if pca_number==None:
        for pca_n in [3,5,10]:
            data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
            data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
            data_train_pca['id_'] = data_train.id_.tolist()
            data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
            data_test_pca['id_'] = data_test.id_.tolist()
            data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
            bestS,bestX,clf,bestN,bestPara = oneClassifier(data_pca,test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)
            if bestS>maxS:
                bestM = clf
                bestPCA = pca_n
                maxS = bestS
                tmp = data_pca
    else:
        data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
        data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
        data_train_pca['id_'] = data_train.id_.tolist()
        data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
        data_test_pca['id_'] = data_test.id_.tolist()
        data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
        maxS,bestX,clf,bestN,bestPara = oneClassifier(data_pca,test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)

    #regression
    Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = 'all',N = None)
    bestS,bestX,clf,bestN,bestPara = oneClassifier(data,test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = random_state)

    if bestS>maxS:
        print "the selected model: ", clf 
        print "the selected features: ", bestX
        clf.fit(data_train.ix[:,bestX].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(data_test.ix[:,bestX]).tolist()
        return data_test['target'].tolist(),y_pred
    else:
        print "the selected model: ", bestM 
        print "the selected PCA: ", bestPCA
        bestM.fit(tmp[tmp.id_.isin(train_msk)].ix[:,range(bestPCA)].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(tmp[tmp.id_.isin(test_msk)].ix[:,range(bestPCA)]).tolist()
        return data_test['target'].tolist(),y_pred


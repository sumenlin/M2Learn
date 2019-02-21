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

def scoring(y_pred,y_test,metric):
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
    elif metric == 'accuracy':
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

def generalRegressor(train_msk,test_msk,path = '../data/', target = 'target', identification = None, 
                     compliance = 1.0, 
                     featureImp = True, featureImputor = 'mean', 
                     modalImp = True, modalImputor = {},
                     clusterIMP = 5,modalNeighbors = 3,
                     replace = False,random_state = 40,
                     pca_number = None,cv_number = 3,
                     metric = 'neg_mean_squared_error',co_linear = False,
                     selectionFeatureKind = 'all',N = None):
    """Regression pipeline

    :param test_msk: the identification values of testing data
    :type test_msk: list
    :param train_msk: the identification values of training data.
    :type train_msk: list
    :param path: the file path for data, assuming each source would have one file.
    :type path: string (default='../data/')
    :param target: the file name for ground truth data.
    :type target: string
    :param identification: the column name of identification/key among all data sources.  
    :type identification: string (default=``None``)
    :param compliance: the lower bound of accceptable feature missing percentage.
    :type compliance: float (default=1.0)
    :param featureImp: indicator if feature imputation should be applied.
    :type featureImp: bool (default=True)
    :param featureImputor: the strategy for feature imputation. If ``int``, the value is used for imputation; if ``mean``, the mean along each column is used for imputation; if ``median``, the median along each column is used for imputation; if ``mode``, the most frequent value along each column is used for imputation.
    :type featureImputor: int, str (default=``mean``)
    :param modalImp: indicator if modal imputation should be applied.
    :type modalImp: bool (default=True)
    :param modalImputor: the mapping among source file names. The key is the one to be imputed and the value is the list of sources to be used for imputation. If ``{}``, the function will impute one sources by all other sources. 
    :type modalImputor: dict (default=``{}``)
    :param clusterIMP: the number of clusters to form for source imputation.
    :type clusterIMP: int (default=5)
    :param modalNeighbors: the number of nearest neighbours to used to construct synthetic samples for source imputation.
    :type modalNeighbors: int (default=3)
    :param replace: indicator if the processed/imputed data will replace the original one. 
    :type replace: bool (default=False)
    :param random_state: the seed of the pseudo random number generator to use if applicable.
    :type random_state: int (default=40)
    :param pca_number: the number of PCA components. If ``None``, the function will try 3,5,and 10 to get the optimal one.
    :type pca_number: int (default=None)
    :param cv_number: the number of folds used in cross validation.  
    :type cv_number: int (default=3)
    :param metric: the metric of performance. 
                 If ``explained_variance``, use explained variance regression score;
                 if ``neg_mean_absolute_error``, use mean absolute error regression loss; 
                 if ``neg_mean_squared_error``, use mean squared error regression loss; 
                 if ``neg_mean_squared_log_error``, use mean squared logarithmic error regression loss; 
                 if ``neg_median_squared_error``, use median absolute error regression loss; 
                 if ``r2``, use r2 score.
    :type metric: string (default='neg_mean_squared_error')
    :param co_linear: indicator if eliminating the features with high correlation for each other. 
    :type co_linear: bool (default=False)
    :param selectionFeatureKind: the strategy for feature selection. 
                 If ``correlation``, use Pearson correlation coefficients as metric for feature selection; 
                 if ``linear``, use coefficients from linear LinearR egression as metric for feature selection; 
                 if ``all``, include selected features from both Pearson correlation and regression coefficients. 
    :type selectionFeatureKind: string (default='all')
    :param N: the number of features to select. If ``None``, select the top ``N`` features where ``N`` is from 5 to the number of total features with interval 5.  
    :type N: int,list (default=None)
    :returns: data after applying preprocessing 
    :rtype: data frame

    """
    df, target, featureName = dataPreprocessing(path = path, target = target, identification = identification, compliance = compliance, featureImp = featureImp, featureImputor = featureImputor, modalImp = modalImp, modalImputor = modalImputor,replace = replace,random_state = random_state,clusterIMP = clusterIMP)
    #PCA method
    df = df.rename(index=str,columns={identification:'id_'})
    yhead = list(set(target.columns.tolist())-set([identification]))[0]
    target = target.rename(index=str,columns={identification:'id_',yhead:'target'})
    df = pd.merge(df,target,on='id_')
    # print df.info()
    data_train = df[df.id_.isin(train_msk)].reset_index(drop=True)
    data_test = df[df.id_.isin(test_msk)].reset_index(drop=True)

    maxS = -1*float('inf')
    if pca_number==None:
        if len(data_train.columns.tolist())-2>=10:
            tmp_pcas = [3,5,10]
        elif len(data_train.columns.tolist())-2>=5:
            tmp_pcas = [3,5]
        elif len(data_train.columns.tolist())-2>=3:
            tmp_pcas = [3]
        else:
            tmp_pcas = [len(data_train.columns.tolist())-2]
        for pca_n in [3,5,10]:
            data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
            data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
            data_train_pca['id_'] = data_train.id_.tolist()
            data_train_pca['target'] = data_train.target.tolist()
            data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
            data_test_pca['id_'] = data_test.id_.tolist()
            data_test_pca['target'] = data_test.target.tolist()
            data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
            bestS,bestX,clf,bestN,bestPara = oneRegressor(data_pca,'id_',test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)
            if bestS>maxS:
                bestM = clf
                bestPCA = pca_n
                maxS = bestS
                tmp = data_pca
    else:
        pca_n = pca_number
        data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
        data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
        data_train_pca['id_'] = data_train.id_.tolist()
        data_train_pca['target'] = data_train.target.tolist()
        data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
        data_test_pca['id_'] = data_test.id_.tolist()
        data_test_pca['target'] = data_test.target.tolist()
        data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
        maxS,bestX,clf,bestN,bestPara = oneRegressor(data_pca,'id_',test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)
        tmp = data_pca


    #regression
    Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = selectionFeatureKind,co_linear = co_linear, N = N)
    print Xheads_list
    bestS,bestX,clf,bestN,bestPara = oneRegressor(df,'id_',test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = random_state)

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
                     compliance = 1.0, 
                     featureImp = True, featureImputor = 'mean', 
                     modalImp = True, modalImputor = {},clusterIMP = 5,
                     replace = False,random_state = 40,
                     pca_number = None,cv_number = 3,
                     metric = 'accuracy',co_linear = False,
                     selectionFeatureKind = 'all',N = None):
    """Regression pipeline

    :param test_msk: the identification values of testing data
    :type test_msk: list
    :param train_msk: the identification values of training data.
    :type train_msk: list
    :param path: the file path for data, assuming each source would have one file.
    :type path: string (default='../data/')
    :param target: the file name for ground truth data.
    :type target: string
    :param identification: the column name of identification/key among all data sources.  
    :type identification: string (default=``None``)
    :param compliance: the lower bound of accceptable feature missing percentage.
    :type compliance: float (default=1.0)
    :param featureImp: indicator if feature imputation should be applied.
    :type featureImp: bool (default=True)
    :param featureImputor: the strategy for feature imputation. If ``int``, the value is used for imputation; if ``mean``, the mean along each column is used for imputation; if ``median``, the median along each column is used for imputation; if ``mode``, the most frequent value along each column is used for imputation.
    :type featureImputor: int, str (default=``mean``)
    :param modalImp: indicator if modal imputation should be applied.
    :type modalImp: bool (default=True)
    :param modalImputor: the mapping among source file names. The key is the one to be imputed and the value is the list of sources to be used for imputation. If ``{}``, the function will impute one sources by all other sources. 
    :type modalImputor: dict (default=``{}``)
    :param clusterIMP: the number of clusters to form for source imputation.
    :type clusterIMP: int (default=5)
    :param modalNeighbors: the number of nearest neighbours to used to construct synthetic samples for source imputation.
    :type modalNeighbors: int (default=3)
    :param replace: indicator if the processed/imputed data will replace the original one. 
    :type replace: bool (default=False)
    :param random_state: the seed of the pseudo random number generator to use if applicable.
    :type random_state: int (default=40)
    :param pca_number: the number of PCA components. If ``None``, the function will try 3,5,and 10 to get the optimal one.
    :type pca_number: int (default=None)
    :param cv_number: the number of folds used in cross validation.  
    :type cv_number: int (default=3)
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
    :param co_linear: indicator if eliminating the features with high correlation for each other. 
    :type co_linear: bool (default=False)
    :param selectionFeatureKind: the strategy for feature selection. 
                 If ``correlation``, use Pearson correlation coefficients as metric for feature selection; 
                 if ``linear``, use coefficients from linear LinearR egression as metric for feature selection; 
                 if ``all``, include selected features from both Pearson correlation and regression coefficients. 
    :type selectionFeatureKind: string (default='all')
    :param N: the number of features to select. If ``None``, select the top ``N`` features where ``N`` is from 5 to the number of total features with interval 5.  
    :type N: int,list (default=None)
    :returns: data after applying preprocessing 
    :rtype: data frame

    """
    df, target, featureName = dataPreprocessing(path = path, target = target, identification = identification, compliance = compliance, featureImp = featureImp, featureImputor = featureImputor, modalImp = modalImp, modalImputor = modalImputor,replace = replace,random_state = random_state,clusterIMP = clusterIMP)
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
            data_train_pca['target'] = data_train.target.tolist()
            data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
            data_test_pca['id_'] = data_test.id_.tolist()
            data_test_pca['target'] = data_test.target.tolist()
            data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
            bestS,bestX,clf,bestN,bestPara = oneClassifier(data_pca,'id_',test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)
            if bestS>maxS:
                bestM = clf
                bestPCA = pca_n
                maxS = bestS
                tmp = data_pca
    else:
        data_train_pca,data_test_pca = featurePCA(data_train.drop(['id_','target'],axis=1),data_test.drop(['id_','target'],axis=1),pca_number = pca_n,random_state = random_state)
        data_train_pca = pd.DataFrame(data_train_pca, columns=range(pca_n))
        data_train_pca['id_'] = data_train.id_.tolist()
        data_train_pca['target'] = data_train.target.tolist()
        data_test_pca = pd.DataFrame(data_test_pca, columns=range(pca_n))
        data_test_pca['id_'] = data_test.id_.tolist()
        data_test_pca['target'] = data_test.target.tolist()
        data_pca = pd.concat([data_train_pca,data_test_pca]).reset_index(drop=True)
        maxS,bestX,clf,bestN,bestPara = oneClassifier(data_pca,'id_',test_msk,train_msk,[range(pca_n)],metric = metric,cv_number = cv_number,seed = random_state)

    #regression
    Xheads_list = selectFeature(data_train.drop(['id_','target'],axis=1),data_train['target'], kind = selectionFeatureKind,co_linear = co_linear,N = N)
    bestS,bestX,clf,bestN,bestPara = oneClassifier(df,'id_',test_msk,train_msk,Xheads_list,metric = metric,cv_number = cv_number,seed = random_state)

    if bestS>maxS:
        print "the selected model: ", clf 
        print "the selected features: ", bestX
        clf.fit(data_train.ix[:,bestX].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(data_test.ix[:,bestX]).tolist()

        return scoreing(y_pred,data_test['target'].tolist(),metric),data_test['target'].tolist(),y_pred
    else:
        print "the selected model: ", bestM 
        print "the selected PCA: ", bestPCA
        bestM.fit(tmp[tmp.id_.isin(train_msk)].ix[:,range(bestPCA)].as_matrix(),data_train.ix[:,'target'].as_matrix())
        y_pred = clf.predict(tmp[tmp.id_.isin(test_msk)].ix[:,range(bestPCA)]).tolist()
        return scoreing(y_pred,data_test['target'].tolist(),metric),data_test['target'].tolist(),y_pred


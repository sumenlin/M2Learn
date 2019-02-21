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
    

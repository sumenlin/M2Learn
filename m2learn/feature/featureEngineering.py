#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
feature engineering
1) feature extraction:time series
2) PCA 
3) feature selection
'''
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#feature extraction
def featureExtraction(data,identification,featureList=None,statis = ['mean','std','median','max','min']):
    """Extract statistics of time series features, including mean, median, max, min and standard deviance.
    
    :param data: data input
    :type data: data frame
    :param identification: identification for each subject in data set
    :type identification: string
    :param featureList: the list of feature names for statistics extraction. If ``None``, all features in the data will be used for extraction.
    :type featureList: list (default=None)
    :param statis: statistics to be extracted.
                           if ``mean``, compute the mean along each feature for each subject;
                           if ``std``, compute the standard deviance along each feature for each subject;
                           if ``median``, compute the median along each feature for each subject;
                           if ``max``, compute the max value along each feature for each subject;
                           if ``min``, compute the min value along each feature for each subject;
                           all can only be used for numeric data.
    :type statis: int, str, optional (default=['mean','std','median','max','min'])
    :returns: data after adding the extracted statistics
    :rtype: data frame
    """
    

#PCA 
def featurePCA(data_train,data_test,pca_number = 3,random_state = 40):
    """Performe PCA analysis on data.
    
    :param data_train: training data
    :type data_train: data frame
    :param data_test: testing data
    :type data_test: data frame
    :param pca_number: the number of PCA components
    :type pca_number: int (default=3)
    :param random_state: the seed of the pseudo random number generator to use if applicable.
    :type random_state: int (default=40)
    :returns: data after PCA analysis
    :rtype: data frame
    """
    


#feature selection on training data
def selectFeature(data,target, kind = 'all',co_linear = False,N = None):
    """Performe feature selection on data.
    
    :param data: features
    :type data: data frame
    :param target: ground truth data
    :type target: data series
    :param kind: the strategy for feature selection. If ``correlation``, use Pearson correlation coefficients as metric for feature selection; 
                 if ``linear``, use coefficients from linear LinearR egression as metric for feature selection; 
                 if ``all``, include selected features from both Pearson correlation and regression coefficients. 
    :type kind: string (default='all')
    :param N: the number of features to select. If ``None``, select the top ``N`` features where ``N`` is from 5 to the number of total features with interval 5.  
    :type N: int,list (default=None)
    :param co_linear: indicator if eliminating the features with high correlation for each other. 
    :type co_linear: bool (default=False)
    :returns: list of potential selected feature lists
    :rtype: list of list
    """
    


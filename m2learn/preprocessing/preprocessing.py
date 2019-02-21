#!/usr/bin/python
# encoding=utf8

import pandas as pd
import os
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
# warnings.warn("Warning...........Message")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#feature imputation
def featureImputation(modal,featureImputor = 'mean'):
    """Perform feature imputation for the data.
    
    :param modal: data 
    :type modal: data frame
    :param featureImputor: method for featue imputation.
                           if ``int``, use the number for imputation;
                           if ``mean``, use the mean along each column for imputation;
                           if ``median``, use the median along each column for imputation;
                           if ``mode``, use the most frequent value along each column for imputation;
                           all can only be used for numeric data.
    :type featureImputor: int, str, optional (default='mean')
    :returns: imputed data
    :rtype: data frame

    """
    

def cutLowCompliance(modal,compliance=1.0):
    """Delete features with higher percentage of missing than the given value ``compliance``.
    
    :param modal: data 
    :type modal: data frame
    :param compliance: the lower bound of accceptable feature missing percentage. If ``1.0``, no features are be excluded.
    :type compliance: float (default=1.0)
    :returns: data after applying  limits
    :rtype: data frame

    """
    
#modal imputation
def modalImputation(df,featureName,identification,modalImputor = {},seed = 40,clusterIMP = 5,modalNeighbors = 3):
    """Perform source imputation for the data

    :param df: the merged data for all sources.
    :type df: data frame
    :param featureName: the feature mappings for all sources. The key is the file name of each source, and the value is the list of features for the corresponding source.
    :type featureName: dict
    :param identification: the column name of identification/key among all data sources. 
    :type identification: string
    :param modalImputor: the mapping among source file names. The key is the one to be imputed and the value is the list of sources to be used for imputation. If ``{}``, the function will impute one sources by all other sources. 
    :type modalImputor: dict (default=``{}``)
    :param seed: the seed of the pseudo random number generator to use if applicable.
    :type seed: int (default=40)
    :param clusterIMP: the number of clusters to form for source imputation.
    :type clusterIMP: int (default=5)
    :param modalNeighbors: the number of nearest neighbours to used to construct synthetic samples for source imputation.
    :type modalNeighbors: int (default=3)
    :returns: data after source imputation
    :rtype: data frame

    """

    


#general function for preprocessing: loading, imputation for two levels
def dataPreprocessing(path = '../data/', target = 'target', identification = None, compliance = 1.0, featureImp = True, featureImputor = 'mean', modalImp = True, modalImputor = {},clusterIMP = 5,modalNeighbors = 3,replace = False,random_state = 40):
    """Data preprocessing: excluing features with higher percentage of missing than ``compliance``, feature imputation for each data source, imputation for missing modals. The processed data will be returned by the function and also stored as file ``processedData.csv`` in the given path.
    
    :param path: the file path for data, assuming each source would have one file.
    :type path: string (default='../data/')
    :param target: the file name for ground truth data.
    :type target: string
    :param identification: the column name of identification/key among all data sources. If ``None``, it will raise errors.  
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
    :returns: data after applying preprocessing 
    :rtype: data frame

    """
    








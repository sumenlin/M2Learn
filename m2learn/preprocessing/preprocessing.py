#!/usr/bin/python
# encoding=utf8

import pandas as pd
import os
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
# warnings.warn("Warning...........Message")

#feature imputation
def featureImputation(modal,featureImputor = 'mean'):
    """Perform feature imputation for the data.
    
    :param modal: data 
    :type modal: dataframe
    :param featureImputor: method for featue imputation.
                           if ``int``, use the number for imputation;
                           if ``mean``, use the mean along each column for imputation;
                           if ``median``, use the median along each column for imputation;
                           if ``mode``, use the most frequent value along each column for imputation;
                           all can only be used for numeric data.
    :type featureImputor: int, str, optional (default='mean')
    :returns: imputed data
    :rtype: dataframe

    """
    if type(featureImputor) == int:
        modal = modal.fillna(featureImputor)
    elif featureImputor == 'mean':
        modal = modal.fillna(modal.mean())
    elif featureImputor == 'median':
        modal = modal.fillna(modal.median())
    elif featureImputor == 'mode':
        modal = modal.fillna(modal.mode())
    else:
        raise Exception("Illegal feature imputation value: "+str(featureImputor))
    return modal


def cutLowCompliance(modal,compliance=1.0):
    """Delete features with higher percentage of missing than the given value ``compliance``.
    
    :param modal: data 
    :type modal: dataframe
    :param compliance: the lower bound of accceptable feature missing percentage. If ``1.0``, no features are be excluded.
    :type compliance: float (default=1.0)
    :returns: data after applying  limits
    :rtype: dataframe

    """
    if compliance:
        col = modal.columns.tolist()
        null_count = modal.isna().sum()
        newc = []
        for c in col:
            if null_count[c]<len(modal.index)*compliance:
                newc.append(c)
        print "There are "+str(len(col)-len(newc))+" features with compliance lower than "+str(compliance)+":"
        print list(set(col)-set(newc))
        modal = modal.ix[:,newc]
        if modal.empty:
            raise Exception("Compliance condition is too restricted to cut all the features of data: "+path +fN)
    return modal


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
    :rtype: dataframe

    """

    if "id_" in df.columns.tolist():
        raise Exception("The column name 'id_' is illegal.")
        
    df = df.rename(index=str,columns = {identification:'id_'})

    ndf = df
    for _fname,_ffeature in featureName.items()
        iX = _ffeature
        if _fname not in modalImputor:
            otherX = []
            for k,v in featureName.items():
                if k!=_fname:
                    otherX += v
        else:
            otherX = []
            for fN in modalImputor[_fname]:
                otherX += featureName[fN]

        #idl to be imputed
        g_data = df.ix[:,otherX+['id_']].dropna().reset_index(drop=True)
        g_idl = g_data['id_'].tolist()
        p_data = ndf.ix[:,iX+['id_']].dropna().reset_index(drop=True)
        p_idl = p_data['id_'].tolist()
        g2p_idl = list(set(g_idl)-set(p_idl))

        #clustering on others
        X_g = g_data.drop('id_',axis=1).as_matrix() 
        kmeans = KMeans(n_clusters=clusterIMP, random_state=seed)
        kmeans.fit(X_g)
        g_label = kmeans.labels_.tolist()
        g_data['label_'] = g_label

        #get i
        gp_data = pd.merge(p_data,g_data,on='id_')
        gp_idl = gp_data.id_.tolist()
        y_gp = gp_data.label_.tolist()
        X_gp = gp_data.ix[:,['id_']+iX].as_matrix()
        g2p_label = g_data[g_data.id_.isin(g2p_idl)].reset_index(drop=True).ix[:,['id_','label_']]
        g2p_data = pd.DataFrame(columns=['id_']+iX)
        for i in range(clusterIMP):
            label_idl = g2p_label[g2p_label.label_.isin([i])].id_.tolist()
            sm = SMOTE(k_neighbors=modalNeighbors,random_state=seed,ratio={i:y_gp.count(i)+len(label_idl)})
            X_res,y_res = sm.fit_resample(X_gp,y_gp)
            X_res = pd.DataFrame(X_res, columns=['id_']+iX)
            X_res = X_res[~X_res.id_.isin(gp_idl)].reset_index(drop=True)
            X_res['id_'] = label_idl
            g2p_data = pd.concat([X_res, g2p_data]).reset_index(drop=True)

        new_p_data = pd.concat([p_data,g2p_data]).reset_index(drop=True)
    
        #merge
        tmp = ndf.drop(iX,axis=1)
        ndf = pd.merge(tmp,new_p_data,on='id_',how='outer')

    # ndf = ndf.dropna().reset_index()
    ndf = ndf.rename(index=str,columns={'id_':identification})

    return ndf



#general function for preprocessing: loading, imputation for two levels
def dataPreprocessing(path = './', target = 'target', identification = None, compliance = 1.0, featureImp = True, featureImputor = 'mean', modalImp = True, modalImputor = {},clusterIMP = 5,modalNeighbors = 3,replace = False,random_state = 40):
    """Data preprocessing: excluing features with higher percentage of missing than ``compliance``, feature imputation for each data source, imputation for missing modals. The processed data will be returned by the function and also stored as file ``processedData.csv`` in the given path.
    
    :param path: the file path for data, assuming each source would have one file.
    :type path: string
    :param target: the file name for ground truth data.
    :type target: string
    :param identification: the column name of identification/key among all data sources.  
    :type identification: string(default=``None``)
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
    :rtype: dataframe

    """

    print "Loading Data ..."
    files = os.listdir(path)
    try:
       target = pd.read_csv(path+target+".csv")
    except:
        raise Exception("Target file doesn't exist.")
    if identification == None:
        raise Exception("Need to specify the common identification of all tables.")
    elif identification not in target.columns.tolist():
        raise Exception("Identification doesn't exist in target file.")

    featureName = {}
    df = pd.DataFrame()
    for fN in files:
        fname = fN[:-4]
        print "Loading Data: " + fN
        if fN == target+".csv":
            continue
        try:
            modal = pd.read_csv(path+fN)
            featureName[fname] = []

            #low compliance filter
            cutLowCompliance(modal,compliance)

            if identification == None:
                raise Exception("Need to specify the common identification of all tables.")
            

            featureName[fname] = list(set(modal.columns.tolist())-set([identification]))
            
            #feature imputation
            if featureImpute:
                modal = featureImputation(modal,featureImputor = featureImputor,replace = replace)
            else:
                modal = modal.dropna().reset_index(drop=True)
            if replace:
                modal.to_csv(path+fN,index=False)
            modal.to_csv(path+fname+"_featureImputed.csv",index=False)
    

            #outer merge data
            try:
                df = pd.merge(df,modal,on=identification,how='outer')
            except:
                raise Exception("Identification doesn't exist in data: "+path +fN)
        except:
            warnings.warn("Loading Failure for Data File: "+path +fN)

    #modal imputation
    if modalImp:
        df = modalImputation(df=df,featureName=featureName,identification=identification,modalImputor=modalImputor,seed=random_state,clusterIMP=clusterIMP,modalNeighbors=modalNeighbors)
    
    df = df.dropna().reset_index(drop=True)

    for _fname,_ffeature in featureName.items():
        iX = _ffeature
        tmp = df.ix[:,iX+[identification]]
        if replace :
            tmp.to_csv(path+_fname,index=False)
        tmp.to_csv(path+_fname+"_imputed.csv",index=False)


    df.to_csv(path+"processedData.csv",index=False)
    return df, target, featureName








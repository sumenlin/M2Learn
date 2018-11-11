#!/usr/bin/python
# encoding=utf8
'''
Tagets: 
loading the data
'''


import pandas as pd
import os
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
# warnings.warn("Warning...........Message")

#feature imputation
def featureImputation(modal,fcount,path,featureImputor = 'mean'):
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

#modal imputation
def modalImputation(df,featureName,fileName,fileName_,fcount,identification,modalImputor = {},seed = 40,clusterK = 5):
    df = df.rename(index=str,columns = {identification:'id_'})

    ndf = df
    for i in range(1,fcount+1):
        iX = featureName[i]
        if not modalImputor:
            otherX = []
            for k,v in featureName.items():
                otherX += v
        else:
            otherX = []
            for fN in modalImputor[fileName[i]]:
                otherX += featureName[fileName_[fN]]

        #idl to be imputed
        g_data = df.ix[:,otherX+['id_']].dropna().reset_index(drop=True)
        g_idl = g_data['id_'].tolist()
        p_data = ndf.ix[:,iX+['id_']].dropna().reset_index(drop=True)
        p_idl = p_data['id_'].tolist()
        g2p_idl = list(set(g_idl)-set(p_idl))

        #clustering on others
        X_g = g_data.drop('id_',axis=1).as_matrix() 
        kmeans = KMeans(n_clusters=clusterk, random_state=seed)
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
        for i in range(clusterk):
            label_idl = g2p_label[g2p_label.label_.isin([i])].id_.tolist()
            sm = SMOTE(random_state=seed,ratio={i:y_gp.count(i)+len(label_idl)})
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


def cutLowCompliance(modal,compliance=0.0):
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

#general function for preprocessing: loading, imputation for two levels
def dataPreprocessing(path = '../data/', target = 'target', identification = None, compliance = 0.0, featureImp = True, featureImputor = 'mean', modalImpute = True, modalImputor = {},replace = False,random_state = 40,clusterIMP = 5):
    print "Loading Data ..."
    files = os.listdir(path)
    try:
       target = pd.read_csv(path+target+".csv")
    except:
        raise Exception("Target file doesn't exist.")
    if identification == None:
        raise Exception("Need to specify the common identification of all tables.")
    if identification not in target.columns.tolist():
        raise Exception("Identification value doesn't exist in target file.")
    featureName = {}
    fileName = {}
    fileName_ = {}
    df = pd.DataFrame()
    fcount = 0
    for fN in files:
        print "Loading Data: " + fN
        if fN == target+".csv":
            continue
        try:
            modal = pd.read_csv(path+fN)
            fcount += 1

            #low compliance filter
            cutLowCompliance(modal,compliance)

            #store feature and file names
            fileName[fcount] = fN
            fileName_[fN] = fcount
            if identification != None:
                featureName[fcount] = list(set(modal.columns.tolist())-set([identification]))
            else:
                featureName[fcount] = modal.columns.tolist()

            #feature imputation
            if featureImpute:
                modal = featureImputation(modal,fcount,path,featureImputor = featureImputor,replace = replace)
            else:
                modal = modal.dropna().reset_index(drop=True)
            if replace:
                modal.to_csv(path+fN,index=False)
            modal.to_csv(path+"modal_"+str(fcount)+"_imputed1.csv",index=False)
    

            #outer merge data
            if identification == None:
                # df = pd.merge(df,modal,left_index=True,right_index=True,how='outer')
                raise Exception("Need to specify the common identification of all tables.")
            else:
                try:
                    df = pd.merge(df,modal,on=identification,how='outer')
                except:
                    raise Exception("Identification doesn't exist in data: "+path +fN)
        except:
            warnings.warn("Loading Failure for Data File: "+path +fN)

    #modal imputation
    if modalImpute:
        df = modalImputation(df,featureName,fileName,fileName_,fcount,identification,modalImputor,random_state,clusterIMP)

    df = df.dropna().reset_index(drop=True)

    for i in range(1,fcount+1):
        iX = featureName[i]
        tmp = df.ix[:,iX+[identification]]
        if replace :
            tmp.to_csv(path+fileName[i],index=False)
        tmp.to_csv(path+"modal_"+str(fcount)+"_imputed2.csv",index=False)


    df.to_csv(path+"processedData.csv",index=False)
    return df, target, featureName,fileName










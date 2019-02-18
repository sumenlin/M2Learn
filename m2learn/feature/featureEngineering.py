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
def featureExtraction(identification,statis = ['mean','std','median','max','min']):
    nx = [identification]
    for x in Xheads:
        if 'mean' in statis:
            df = data.groupby(identification)[x].mean().reset_index()
            df = df.rename(index=str,columns={x:x+"_mean"})
            data = pd.merge(df,data,on=identification)
            nx.append(x+"_mean")
        if 'std' in statis:
            df = data.groupby(identification)[x].std().reset_index()
            df = df.rename(index=str,columns={x:x+"_std"})
            data = pd.merge(df,data,on=identification)
            nx.append(x+"_std")
        if 'median' in statis:
            df = data.groupby(identification)[x].median().reset_index()
            df = df.rename(index=str,columns={x:x+"_median"})
            data = pd.merge(df,data,on=identification)
            nx.append(x+"_median")
        if 'max' in statis:
            df = data.groupby(identification)[x].max().reset_index()
            df = df.rename(index=str,columns={x:x+"_max"})
            data = pd.merge(df,data,on=identification)
            nx.append(x+"_max")
        if 'min' in statis:
            df = data.groupby(identification)[x].min().reset_index()
            df = df.rename(index=str,columns={x:x+"_min"})
            data = pd.merge(df,data,on=identification)
            nx.append(x+"_min")
    data = data.ix[:,nx]
    return data

#PCA 
def featurePCA(data_train,data_test,pca_number = 3,random_state = 40):
    pca = PCA(n_components=pca_number, random_state=random_state).fit(data_train)
    data_train_pca = pca.transform(data_train)
    data_test_pca = pca.transform(data_test)
    return data_train_pca,data_test_pca


#feature selection on training data
def selectFeature(data,y, kind = 'all',N = None):
    # data = data.ix[:,Xheads+[yhead]]
    # d = data.values
    # scaler = preprocessing.StandardScaler()
    # x_scaled = scaler.fit_transform(d)
    # data = pd.DataFrame(x_scaled,index=data.index, columns=data.columns)

    #data feature exclude identification
    if N==None:
        N = min(len(data.columns.tolist()),50)
        LN = range(5,N+1,5)
    elif type(N)==int:
        LN = [N]
    elif type(N)==list:
        LN = N

    Xheads = data.columns.tolist()
    Xheads_list = []
    df = pd.DataFrame()
    df['Xheads'] = Xheads
    if kind == 'correlation' or kind == 'all':
    # correlation selection
        corrs = []
        d1 = y.tolist()
        for xhead in Xheads:
            # print xhead
            # d1 = data[yhead].tolist()
            d2 = data[xhead].tolist()
            corr = pearsonr(d1,d2)
            corrs.append(corr[0])
            # corrs.append(abs(corr[0]))
        df['correlation'] = corrs
        df = df.sort_values(by='correlation', ascending=False).reset_index(drop=True)
        for n in LN:
            Xheads_new = df.Xheads.tolist()[:n]
            Xheads_list.append(Xheads_new)

    #linear
    if kind == 'linear' or kind == 'all':
        X = data
        Y = y
        lr = LinearRegression()  
        lr.fit(X, Y)
        df['lr_coeff'] = lr.coef_.tolist()
        df = df.sort_values(by='lr_coeff', ascending=False).reset_index(drop=True)
        for n in LN:
            Xheads_new = df.Xheads.tolist()[:n]
            Xheads_list.append(Xheads_new)

    return Xheads_list


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:48:38 2021

@author: WXX
"""
import numpy as np
import pandas as pd
#

def preprocessing_standard(L):
    """
       preprocessing_standard: 对一行数据进行数据标准化,L[i]=(L[i]-L.min)/(L.max-L.min)
    """
    datamax=max(L)
    datamin=min(L)
    L1=[]
    for index,row in enumerate(L):
        if datamax-datamin!=0:
            m=(row-datamin)/float((datamax-datamin))
            L1.append(round(m,4))
        else:
            L1.append(0)
#        matlabshow(row,index=str(index)+'_')    
    return L1

def preprocess(dataframe):
    dataframe_new=pd.DataFrame(columns=dataframe.columns)
    for i in dataframe.columns:
        dataframe_new[i]=preprocessing_standard(dataframe[i])
    return dataframe_new

def Normalize_Adjacent_undirect(A):
    """
       Normalize_Adjacent_undirect: 无向图的邻接矩阵被标准化
       A：领接矩阵
       标准化公式：(D**-(1/2))*(A+I) *(D**-(1/2))
    """
    A=np.array(A)
    I=np.identity(len(A))                 #单位矩阵
    A1=A+I                                #邻接矩阵加上单位矩阵
    D1 = np.sum(A1,axis=1)               #获取度的列表
    D2 = np.sum(A1,axis=0)
    if (D1==D2).all()==False:
        print('please enter undirected graph')
        return
    Dn = np.zeros((len(A1), len(A1)))     
    for i in range(len(A1)):
        if D1[i] > 0:
            Dn[i][i] = int(D1[i])**(-0.1)   #对度矩阵每个对角线元素取它的倒数
    AD = np.dot(A1, Dn)
    AD = np.dot(Dn,AD)                    # (D**-(1/2))*(A+I) *(D**-(1/2))
    return AD


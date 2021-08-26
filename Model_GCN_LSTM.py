# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:29:17 2021

@author: WXX
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:30:16 2021

@author: WXX
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from glob import glob
import os
from torch.nn import init

   
class GCN(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        """图卷积神经网络
         Args:
            input_dim:输入节点特征个数
            output_dim:输出节点特征个数
            use_bias:是否使用偏置
        """
        super(GCN, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        #定义DCN层的权重矩阵
        
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        自定义参数初始化方式
        权重参数初始化方式
        """
        init.kaiming_uniform_(self.weight)
        if self.use_bias:      #偏置参数初始化为0
            init.zeros_(self.bias)
        return 
    
    def forward(self,adjacency,input_feature,num_cnn_feature):
        """
        Args:
            adjacency:标准化之后的邻接矩阵
            input_feature:输入特征
            num_cnn_feature:序列长度维度特征
        """
        num_nodes=adjacency.shape[0]
        input_feature=input_feature.reshape(num_cnn_feature,num_nodes,self.input_dim)
        support=torch.matmul(input_feature,self.weight) #XW(N,D');X(N,D);W(D,D')
      
        output=torch.matmul(adjacency,support) #(N,D')
        if self.use_bias:
            output+=self.bias
        #output=output.reshape(num_nodes,output.shape[2],num_cnn_feature)
        return output
                                 
    def __repr__(self):
        return self.__class__.__name__+'('+str(self.input_dim)+'->'+str(self.output_dim)+')'
    
class Gcn_LSTM(nn.Module):
    """定义了一个多层的GraphConvolution的模型"""
    def __init__(self):
        super().__init__()
        self.num_nodes=24
        self.seq_len=100
        self.input_size=1
        
        self.gcn1=GCN(self.input_size,10)
        self.gcn2=GCN(10,20)
        self.gcn3=GCN(20,30)
        self.LSTM = nn.LSTM(input_size=30, hidden_size=10,num_layers=1,)
        
        self.liner1 = nn.Linear(10, 3)
        
        self.dropout=nn.Dropout(p=0.5)
                                 
    def forward(self,adjacency,h):
        #第一层
        h=F.relu(self.gcn1(adjacency,h,self.seq_len)) 

        h=F.relu(self.gcn2(adjacency,h,self.seq_len)) 
        
        h=F.relu(self.gcn3(adjacency,h,self.seq_len))
        
        h=h.reshape(self.num_nodes,self.seq_len,1,30)
       
        hide=torch.zeros([1,10])
        for i,h_L in enumerate(h):
            lstm_out,(h_n,h_c)=self.LSTM(h_L,None)
            hide = torch.add(lstm_out[-1,:,:],hide)
        h=hide
       
        h=self.liner1(h)
        #h=self.dropout(h)
        h=F.sigmoid(h)
        return h
    
#if __name__=='__main__':
    #model=Gcn_1nn1()
    #for name,parameters in model.named_parameters:
        #print(name,':',parameters)
        #print(name,':',parameters.size())
    

    

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:30:16 2021

@author: WXX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Conv1(nn.Module):
    def __init__(self,channel_in,channel_out):
        """一维卷积神经网络
        Args:
            channel_in:输入通道数
            cnannel_out:输出通道数
        """
        super(Conv1, self).__init__()
        self.channel_in=channel_in
        self.channel_out=channel_out
        self.conv1=nn.Conv1d(in_channels=channel_in,out_channels=channel_out,kernel_size=3,stride=2)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        return x
    
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
        output=output.reshape(num_nodes,output.shape[2],num_cnn_feature)
        return output
                                 
    def __repr__(self):
        return self.__class__.__name__+'('+str(self.input_dim)+'->'+str(self.output_dim)+')'

class Conv1_model(nn.Module):
    """定义了一个多层的GraphConvolution的模型"""
    def __init__(self):
        super().__init__()
        self.conv1 =Conv1(1,10)
        self.conv2 = Conv1(10,20)
        self.conv3 = Conv1(20, 30)
        
        self.gcn1=GCN(20,35)
        self.gcn2=GCN(35,40)
        self.gcn3=GCN(40,45)
        self.gcn4=GCN(45,50)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        
        self.liner1 = nn.Linear(1440, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 3) 
        
                                 
    def forward(self,adjacency,h):
        """一维卷积层"""
        h=h.reshape(h.shape[0],h.shape[2],h.shape[1])
        h_c1= self.conv1(h)
        h=self.max_pool1(h_c1)
        
        h_c2= self.conv2(h)
        h=self.max_pool2(h_c2)
        
        h= self.conv3(h)
        
        """全连接层"""
        num=h.shape[0]*h.shape[1]*h.shape[2]
        h=h.view(-1, num)
        #print('h_relu**********************',h.shape)
        
        #liner1 = nn.Linear(num, 120)
        h= F.relu(self.liner1(h))
        h= F.relu(self.liner2(h))
        h= F.sigmoid(self.liner3(h))
        return h
    

    

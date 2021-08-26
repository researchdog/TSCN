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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#一维卷积神经网络
class Conv1(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Conv1, self).__init__()
        self.channel_in=channel_in
        self.channel_out=channel_out
        self.conv1=nn.Conv1d(in_channels=channel_in,out_channels=channel_out,kernel_size=3,stride=2)
        
    def forward(self,x):
        #x=x.reshape(num_nodes,self.channel_in,num_cnn_feature)
        #x1=x.reshape()
        x=F.relu(self.conv1(x))
        return x
    
class GCN(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        """图卷积神经网络
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
        """自定义参数初始化方式
        权重参数初始化方式"""
        init.kaiming_uniform_(self.weight)
        if self.use_bias:      #偏置参数初始化为0
            init.zeros_(self.bias)
        return 
    
    def forward(self,adjacency,input_feature,num_cnn_feature):
        """
        领接矩阵是稀疏矩阵，因此在使用过程中使用稀疏矩阵乘法
        Args:
            adjacency:torch.sparse.FloatTensor，标准化之后的邻接矩阵
                领接矩阵
            input_feature:torch.Tensor
                输入特征
        """
        # print('self.weight.shape',self.weight.shape)
        # print('input_feature',input_feature.shape)
        num_nodes=adjacency.shape[0]
        #print(input_feature.shape,'input_feature')
        input_feature=input_feature.reshape(num_cnn_feature,num_nodes,self.input_dim)
        support=torch.matmul(input_feature,self.weight) #XW(N,D');X(N,D);W(D,D')
        # print('adjacency',adjacency.shape)
        # print('adjacency',support.shape)
        # support=support.reshape(num_nodes,input_feature.shape[0],self.output_dim)
        #print('support',support.shape)
        output=torch.matmul(adjacency,support) #(N,D')
        #print('output',output.shape,num_cnn_feature,num_nodes)
        if self.use_bias:
            output+=self.bias
        output=output.reshape(num_nodes,self.output_dim,num_cnn_feature)
        
        return output
                                 
    def __repr__(self):
        return self.__class__.__name__+'('+str(self.input_dim)+'->'+str(self.output_dim)+')'
    
class Conv1_Gcn_Conv1(nn.Module):
    """定义了一个多层的GraphConvolution的模型"""
    def __init__(self):
        super().__init__()
        self.num_nodes=24
        self.seq_len=100
        self.input_size=1
        
        self.gcn1=GCN(self.input_size,10)
        self.conv1 =Conv1(10,20)
        self.gcn2=GCN(20,30)
        self.conv2 = Conv1(30,35)
        self.gcn3=GCN(35,40)
        self.conv3 = Conv1(40,50)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        
        self.liner1 = nn.Linear(2400, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 3) 
        
        self.dropout=nn.Dropout(p=0.5)
                                 
    def forward(self,adjacency,h):
        #第一层
        #print('第一层h_0',h.shape)
        h=h.reshape(h.shape[0],h.shape[2],h.shape[1])
        #print(h.shape,'h.shape')
        #print('h_c',h_c.shape)
        h=F.relu(self.gcn1(adjacency,h,self.seq_len)) 
        h_c= F.relu(self.conv1(h))
        h=self.max_pool1(h_c)
        #h=self.dropout(h)
        #print('h',h.shape)
        # #第二层
        #print('第二层h_0',h.shape)
       
        #print('h_c',h_c.shape)
        h=F.relu(self.gcn2(adjacency,h,h.shape[2])) 
        h_c= F.relu(self.conv2(h))
        h=self.max_pool2(h_c)
        #h=self.dropout(h)
        #print('h',h.shape)
        # #】第三层
        #print('第三层h_0',h.shape)
        
        #print('h_c',h_c.shape)
        h=F.relu(self.gcn3(adjacency,h,h.shape[2]))
        #print('h',h.shape)
        h= F.relu(self.conv3(h))
        h=h.view(-1, h.shape[0]*h.shape[1]*h.shape[2])
        #print('h_relu**********************',h.shape)
        h= F.relu(self.liner1(h))
        h= F.relu(self.liner2(h))
        h= F.sigmoid(self.liner3(h))
        return h
    

    

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

#一维卷积神经网络
    
class lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_nodes=24
        self.seq_len=100
        self.input_size=1
   
        self.LSTM = nn.LSTM(input_size=24, hidden_size=10,num_layers=1)
        self.liner1 = nn.Linear(10, 3)
        
                                 
    def forward(self,adjacency,h):
        
        h=h.view(-1,100,h.shape[0])
        #print(h.shape,'h.view(-1,100,h.shape[0])')
        lstm_out,(h_n,h_c)=self.LSTM(h,None)
        #print('h_relu**********************',h.shape)
        
        h = lstm_out[:,-1,:]
        #print(h.shape,'lstm_out[:,-1,:]')
        #print(h.shape,'shape')
        h=h.reshape(1,h.shape[0]*h.shape[1])
        h=self.liner1(h)
        #print(h.shape,'self.liner1(h)')
        #h=self.dropout(h)
        h=F.sigmoid(h)
        return h
    

    

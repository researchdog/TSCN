# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:10:51 2021

@author: WXX
"""

import torch
from Get_Data import get_data
from Training import training
from Model_Conv1_GCN import Gcn_1nn1
from Model_LSTM import lstm
from Model_GCN_linear import GCN_linear
from Model_GCN_LSTM import Gcn_LSTM
from Model_Conv1_GCN_Conv1 import Conv1_Gcn_Conv1
from Model_GCN_GRU import Gcn_GRU
from Model_GRU import GRU
from Model_Conv1 import Conv1_model
import pandas as pd

path_total='E:/Python/Study/Classification_ACS/data/total_all/'
path_adj='E:/Python/Study/Classification_ACS/data/adj.csv'
path_record='E:/Python/Study/Classification_ACS/record/record.csv'

class main():
    def __init__(self,fold,path_total,path_adj,path_record,learning_rate,epoch_size,net):
        self.path_total=path_total
        self.path_adj=path_adj
        self.path_record=path_record
        self.learning_rate=learning_rate
        self.epoch_size=epoch_size
        self.fold=fold
        self.net=net
        
    def get_k_fold_data(self,k,i,x_total,y_total):
        """将数据集进行划分
        Args:
            k:总共几折
            i:现在是第几折
            x_total:总的数据
            y_total:数据标签
            
            x_train:分割好的训练集
            y_train:分割好的训练集标签
            x_test:分割好的测试集
            y_test:分割好的测试集标签
        """
        fold_size=x_total.shape[0]//k
        x_train,y_train=None,None
        for j in range(k):
            idx=slice(j*fold_size,(j+1)*fold_size)
            x_part,y_part=x_total[idx,:],y_total[idx]
            if j==i:
                x_test,y_test=x_part,y_part
            elif x_train is None:
                x_train,y_train=x_part,y_part
            else:
                x_train=torch.cat((x_train,x_part),dim=0)
                y_train=torch.cat((y_train,y_part),dim=0)
        return x_train,y_train,x_test,y_test
    
    
    def k_fold(self,k,x_total,y_total,adj,learning_rate,epoch):
        """
        Args:
            x_total:总的数据
            y_total:数据标签
            learning_rate:学习率
            epoch_size:迭代次数
            
            loss_train：训练集的损失
            acc_train：训练集的准确率
            loss_test：测试集的损失
            acc_test：测试集的准确率
        """
        loss_train=0
        acc_train=0
        pre_train=0
        rec_train=0
        
        loss_test=0
        acc_test=0
        pre_test=0
        rec_test=0
        dfhistory=pd.DataFrame(columns=['epoch','loss','accuracy','precision','val_loss','val_accuracy','val_precision'])
        time_record=[]
        for i in range(k):
            x_train,y_train,x_test,y_test=self.get_k_fold_data(k,i,x_total,y_total)
            train_=training(adj,y_train,y_test,x_train,x_test,self.net,learning_rate,epoch)
            info,history,time=train_.train()
            time_record.append(time)
            dfhistory=pd.concat([dfhistory,history],axis=0)
            
            loss_train+=info[1]
            acc_train+=info[2]
            pre_train+=info[3]
            rec_train+=info[4]
            
            loss_test+=info[5]
            acc_test+=info[6]
            pre_test+=info[7]
            rec_test+=info[8]
            print('*'*25,'第',i+1,'折','*'*25)
            print('train_loss:',info[1],'train_acc:',info[2],'train_pre:',\
                  info[3],'train_rac:',info[4],'test_loss:',info[5],\
                  'test_acc:',info[6],'test_pre:',info[7],'rec_test:',info[8])
        print('train_loss_sum:%.4f'%(loss_train/k),'train_acc_sum:%.4f\n'%(acc_train/k),'train_pre_sum:%.4f\n'%(pre_train/k),'rec_train_sum:%.4f\n'%(rec_train/k),\
              'valid_loss_sum:%.4f'%(loss_test/k),'valid_acc_sum:%.4f'%(acc_test/k),'valid_pre_sum:%.4f'%(pre_test/k),'rec_test_sum:%.4f\n'%(rec_test/k))
        return dfhistory,time_record
    
    def main_m(self):
        data=get_data(self.path_total,self.path_adj)
        adj,y_total,x_total=data.get_datas()
        dfhistory,time_record=self.k_fold(self.fold,x_total,y_total,adj,self.learning_rate,self.epoch_size)
        dfhistory.to_csv(self.path_record)
        return time_record
    
M=main(7,path_total,path_adj,path_record,0.0001,30,Gcn_1nn1())
time_record=M.main_m()
        


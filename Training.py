# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:47:06 2021

@author: WXX
"""
import torch.nn as nn
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score
from torch import optim
import time
from tqdm import tqdm

class training():
    def __init__(self,adj,y_train,y_test,x_train,x_test,net,learning_rate,epoch):
        """
        Args:
            self.adj:邻接矩阵
            self.y_train:训练集标签
            self.y_test:测试集标签
            self.x_train:训练集
            self.x_test:测试集
            self.net:模型
            self.learning_rate:学习率
            self.epoch:迭代次数
        """
        self.adj=adj
        self.y_train=y_train
        self.y_test=y_test
        self.x_train=x_train
        self.x_test=x_test
        self.net=net
        self.learning_rate=learning_rate
        self.epoch=epoch
    
    def loss_opt(self):
        """
        定义损失函数与优化器
            criterion：损失函数
            optimizer：优化函数
            metric_func：评估函数
        """
        def assess_func(y_pred,y_true):
            y_pred_cls=torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
            return accuracy_score(y_true,y_pred_cls),precision_score(y_true,y_pred_cls,average='weighted'),recall_score(y_true,y_pred_cls,average='weighted')
        
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(self.net.parameters(),lr=self.learning_rate) 
        return criterion,optimizer,assess_func
        
    def train(self):
        """
           dfhistory：    记录历史数据
           tqdm:          可以增加一个进度条，显示进度信息
           loss_sum:      训练集完成所有迭代次数之后的损失
           accuracy_sum:  训练集完成所有迭代次数之后的准确率
           precision_sum：训练集完成所有迭代次数之后的精确率
           val_loss_sum:      训练集完成所有迭代次数之后的损失
           val_accuracy_sum:  训练集完成所有迭代次数之后的准确率
           val_precision_sum：训练集完成所有迭代次数之后的精确率
        """
        start=time.time()
        criterion,optimizer,assess_func=self.loss_opt()
        dfhistory=pd.DataFrame(columns=['epoch','loss',"accuracy",'precision','recall','val_loss','val_accuracy','val_precision','val_recall'])
        for epoch in tqdm(range(self.epoch)):
            loss_sum=0
            accuracy_sum=0
            precision_sum=0
            recall_sum=0
            for i,input_data in enumerate(self.x_train,0):
                label=self.y_train[i]
                optimizer.zero_grad()
                #正向传播求损失
                outputs=self.net(self.adj,input_data)
                loss=criterion(outputs,label)
                accuracy,precision,recall=assess_func(outputs,label)
                #反向传播求梯度
                loss.backward()
                optimizer.step()
                
                loss_sum+=loss.item()
                accuracy_sum+=accuracy.item()
                precision_sum+=precision.item()
                recall_sum+=recall.item()
                if i%2000==1999:
                    loss_sum=0.0
                    
            #验证循环
            val_loss_sum=0
            val_accuracy_sum=0
            val_precision_sum=0
            val_recall_sum=0
            for j,input_data in enumerate(self.x_test,0):
                with torch.no_grad():
                    label=self.y_test[j]
                    predictions=self.net(self.adj,input_data)
                    val_loss=criterion(predictions,label)
                    val_metric,val_precision,val_recall=assess_func(predictions,label)
                val_loss_sum+=val_loss.item()
                val_accuracy_sum+=val_metric.item()
                val_precision_sum+=val_precision.item()
                val_recall_sum+=val_recall.item()
            info=(epoch,loss_sum/(i+1),accuracy_sum/(i+1),precision_sum/(i+1),recall_sum/(i+1),val_loss_sum/(j+1),val_accuracy_sum/(j+1),val_precision_sum/(j+1),val_recall_sum/(j+1))
            dfhistory.loc[epoch-1]=info
        time1=(time.time()-start)/60
        return info,dfhistory,time1
            

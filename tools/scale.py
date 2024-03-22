#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:17:27 2024

@author: tanmoysil
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class ScaleTS():
    def __init__(self):
        #self.data = data
        #self.num_channels = data.shape[1]
        #self.num_subjects = data.shape[2]
        #self.scaled_data = np.zeros_like(data)
        self.emg_chan = [1,2,4,5]
        self.acc_chan = [0,3]
        self.scaler_emg = []
        self.scaler_acc = []
    
    #get one subject
    def get_subject(self, data, index):
        return data[:,:,index]
        
    #get one channel 
    def get_channel(self, data, index):
        return data[:,index]

    
    # fit emg and acc separately
    def train(self, data):
        emg = np.array([])
        acc = np.array([])
        
        num_subjects = data.shape[2]
        num_channels = data.shape[1]
  
        
        for i in range(num_subjects):
            for j in range(num_channels):
                channel = self.get_channel(self.get_subject(data, i), j)
                
                
                if j in self.emg_chan:
                    emg = np.append(emg, channel)
                elif j in self.acc_chan:
                    acc = np.append(acc, channel)
              
        self.scaler_emg = StandardScaler(with_mean=True, with_std=True).fit(emg.reshape(-1,1))
        self.scaler_acc = StandardScaler(with_mean=True, with_std=True).fit(acc.reshape(-1,1))
        
        return self
    
    #transform emg and acc separately
    def preprocess(self, to_preprocess):
        num_subjects = to_preprocess.shape[2]
        num_channels = to_preprocess.shape[1]
        
        for i in range(num_subjects):
            for j in range(num_channels):
                temp = np.array([])
                if j in self.emg_chan:
                    temp = self.scaler_emg.transform(to_preprocess[:,j,i].reshape(-1,1))
                    to_preprocess[:,j,i] = temp.reshape(-1)
                elif j in self.acc_chan:
                     temp = self.scaler_acc.transform(to_preprocess[:,j,i].reshape(-1,1))
                     to_preprocess[:,j,i] = temp.reshape(-1)
        return to_preprocess
                    
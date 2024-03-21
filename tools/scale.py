#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:17:27 2024

@author: tanmoysil
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class ScaleTS(StandardScaler):
    def __init__(
        self,
        data: np.array,
        ):
        self.data = data
        self.num_channels = data.shape[1]
        self.num_subjects = data.shape[2]
        self.scaled_data = np.zeros_like(data)
    
        
    def get_subject(self, index):
        subject = self.data[:,:,index]
        return subject
        
        
    def get_channel(self, dat, index):
        channel = dat[:,index]
        return channel

        
    def scale(self):
        for i in range(self.num_subjects):
            for j in range(self.num_channels):
                subject = self.get_subject(i)
                channel = self.get_channel(subject, j)
                scaler = StandardScaler(with_mean=True, with_std=True)
                channel =  scaler.fit_transform(channel.reshape(-1,1))
                self.scaled_data[:,j,i] = channel.reshape(-1,1).flatten()
        return self.scaled_data
    
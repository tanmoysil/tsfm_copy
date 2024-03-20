#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:17:27 2024

@author: visualdbs
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
    
        
    def get_single_subject(self, index):
        single_subject = self.data[:,:,index]
        
    def scale(self):
        for i in self.num_subjects:
            for j in self.num_channels:
                scaler = StandardScaler().fit(get_single_subject(i))
                
        
        
        
        
num_subjects = 
for i in X_train.shape[2]:
    

    scaler(i) = StandardScaler().fit(X_train[:,:,i])
    scaler(i).mean_
    scaler(i).scale_

    X_scaled = scaler.transform(X_train[:,:,i])


# class Preprocessor(StandardScaler):
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:17:27 2024

@author: tanmoysil
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from pathlib import Path
import random
import torch

#%%
# Set the path to your folder containing .dat files
folderpath_to_data = []
folderpath_to_data.append("/Users/tanmoysil/Library/Mobile Documents/com~apple~CloudDocs/DAT/transformers/ETPD_DATA_transformers/Essential_Tremor") 
folderpath_to_data.append("/Users/tanmoysil/Library/Mobile Documents/com~apple~CloudDocs/DAT/transformers/ETPD_DATA_transformers/Parkinsons_Tremor")



dat_files = [file for file in os.listdir(folderpath_to_data[0]) if file.endswith('.dat')]
len_1 = len(dat_files)
dat_files.extend([file for file in os.listdir(folderpath_to_data[1]) if file.endswith('.dat')])
len_2 = len(dat_files) - len_1


# Initialize an empty list to store the second variable from each file
second_variable_list = []

# Loop through each .dat file-
y = []
for length_, file_name in zip(range(len(dat_files)), dat_files):
    if length_+1<=len_1:
        file_path = os.path.join(folderpath_to_data[0], file_name)
        y.append(0)
    elif length_+1>len_1:
        y.append(1)
        file_path = os.path.join(folderpath_to_data[1], file_name)

    # Load data from the .dat file
    data_1 = np.loadtxt(file_path,delimiter=';', usecols = (1,2,3,4,5,6))

    # Extract the second variable 
    resample_points = 3000
    data_resample = signal.resample(data_1, resample_points, axis=0)
    second_variable = data_resample
    if second_variable.shape != resample_points:
        padding_size = max(0, resample_points - len(second_variable))
        second_variable = np.pad(second_variable, (0,padding_size), mode='constant', constant_values=np.nan)

    # Append the second variable to the list
    second_variable_list.append(second_variable)

# Create a new NumPy array from the list of second variables
result_array = np.array(second_variable_list)

filepath_to_save = '/Users/tanmoysil/Library/Mobile Documents/com~apple~CloudDocs/DAT/transformers/'
filename_to_save = 'data_3000'

np.savez(os.path.join(filepath_to_save, filename_to_save), result_array, y)


#%%
# #convert numpy array to pandas dataframe
# time = np.linspace(0, 30, num = resample_points)
# data_2 = pd.DataFrame(result_array, columns = [f'Column{i+1}' for i in range(result_array.shape[1])])


# # Sample DataFrame with a column 'seconds' representing seconds
# time_to_df = {'miliseconds':time}
# time_df = pd.DataFrame(time_to_df)

# # Convert 'seconds' to Unix timestamp
# time_df['unix_timestamp'] = pd.to_datetime(time_df['miliseconds'], unit='ms')

# # Convert Unix timestamp to pandas datetime
# time_df['datetime'] = pd.to_datetime(time_df['unix_timestamp'])

# # Display the DataFrame
# #print(time_df)

# data_2.insert(0, 'date', time_df['datetime'] )
# #data_2.to_csv("data3000.csv")




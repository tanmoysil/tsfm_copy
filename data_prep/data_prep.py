

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
path_to_data = '/home/visualdbs/User_Folders/Tanmoy/transformers/ETPD_DATA_transformers/'
folder_path = Path('ETPD_DATA_transformers/')
#print(folder_path.resolve())

# Get a list of all .dat files in the folder
dat_files = [file for file in os.listdir(folder_path) if file.endswith('.dat')]

# Initialize an empty list to store the second variable from each file
second_variable_list = []

# Loop through each .dat file
for file_name in dat_files:
    file_path = os.path.join(folder_path, file_name)

    # Load data from the .dat file using numpy.loadtxt
    data_1 = np.loadtxt(file_path,delimiter=';', usecols = (1,2,3,4,5,6))

    # Extract the second variable (assuming 0-based indexing)
    resample_points = 3000
    data_resample = signal.resample(data_1, resample_points, axis=0)
    second_variable = data_resample
    if second_variable.shape != resample_points:
        padding_size = max(0, resample_points - len(second_variable))
        second_variable = np.pad(second_variable, (0,padding_size), mode='constant', constant_values=np.nan)

    # Append the second variable to the list
    second_variable_list.append(second_variable)

# Create a new NumPy array from the list of second variables
result_array = np.reshape(np.array(second_variable_list), (resample_points, data_1.shape[1], len(dat_files)))

filename_to_save = 'data_1_3000'

np.save(filename_to_save, result_array)


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




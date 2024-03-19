

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from pathlib import Path
import random
import torch


# Set the path to your folder containing .dat files
folder_path = Path('ETPD_DATA_transformers/')
print(folder_path.resolve())

# Get a list of all .dat files in the folder
dat_files = [file for file in os.listdir(folder_path) if file.endswith('.dat')]

# Initialize an empty list to store the second variable from each file
second_variable_list = []

# Loop through each .dat file
for file_name in dat_files:
    file_path = os.path.join(folder_path, file_name)

    # Load data from the .dat file using numpy.loadtxt
    data_1 = np.loadtxt(file_path,delimiter=';')

    # Extract the second variable (assuming 0-based indexing)
    data_resample = signal.resample(data_1, 6000, axis=0)
    second_variable = data_resample[:, 1]
    if second_variable.shape != 6000:
        padding_size = max(0, 6000 - len(second_variable))
        second_variable = np.pad(second_variable, (0,padding_size), mode='constant', constant_values=np.nan)

    # Append the second variable to the list
    second_variable_list.append(second_variable)

# Create a new NumPy array from the list of second variables
result_array = np.array(second_variable_list, dtype="object").transpose()

#convert numpy array to pandas dataframe
time = np.linspace(0, 30, num = 6000)
data_2 = pd.DataFrame(result_array, columns = [f'Column{i+1}' for i in range(result_array.shape[1])])
data_2.insert(0, 'time', time)

#%%
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSMixerForPretraining,
    Trainer,
    TrainingArguments,
)

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

dataset_path = 'data.csv'
timestamp_column = "time"
id_columns = []

context_length = 512
forecast_horizon = 96
patch_length = 8
num_workers = 16  # Reduce this if you have low number of CPU cores
batch_size = 64  # Adjust according to GPU memory

data = pd.read_csv(
    dataset_path
    
)
forecast_columns = list(data_2.columns[1:10])

# get split
num_train = int(len(data) * 0.7)
num_test = int(len(data) * 0.2)
num_valid = len(data) - num_train - num_test
border1s = [
    0,
    num_train - context_length,
    len(data) - num_test - context_length,
]
border2s = [num_train, num_train + num_valid, len(data)]

train_start_index = border1s[0]  # None indicates beginning of dataset
train_end_index = border2s[0]

# we shift the start of the evaluation period back by context length so that
# the first evaluation timestamp is immediately following the training data
valid_start_index = border1s[1]
valid_end_index = border2s[1]

test_start_index = border1s[2]
test_end_index = border2s[2]

train_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=train_start_index,
    end_index=train_end_index,
)
valid_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=valid_start_index,
    end_index=valid_end_index,
)
test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=test_start_index,
    end_index=test_end_index,
)

tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
tsp = tsp.train(train_data)

train_dataset = ForecastDFDataset(
    tsp.preprocess(train_data),
    id_columns=id_columns,
    timestamp_column="date",
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
valid_dataset = ForecastDFDataset(
    tsp.preprocess(valid_data),
    id_columns=id_columns,
    timestamp_column="date",
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
test_dataset = ForecastDFDataset(
    tsp.preprocess(test_data),
    id_columns=id_columns,
    timestamp_column="date",
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)

config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(forecast_columns),
        patch_stride=patch_length,
        d_model=16,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="common_channel",
        scaling="std",
    )
model = PatchTSMixerForPrediction(config)

training_args = TrainingArguments(
        output_dir="./checkpoint/patchtsmixer_4/electricity/pretrain/output/",
        overwrite_output_dir=True,
        learning_rate=0.001,
        num_train_epochs=100,  # For a quick test of this notebook, set it to 1
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir="./pretrain/logs/",  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
    )

# Create the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
)

# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

# pretrain
trainer.train()

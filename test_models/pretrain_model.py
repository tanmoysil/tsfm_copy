#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:17:27 2024

@author: tanmoysil
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from pathlib import Path
import random
import torch
import warnings

from tsfm_public.toolkit.util import select_by_index
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.dataset import ForecastDFDataset
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSMixerForPretraining,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", module="torch")

#%%
#dataset_path = '/home/visualdbs/User_Folders/Tanmoy/transformers/data_1_3000.npy'
dataset_path = '/Users/tanmoysil/Library/Mobile Documents/com~apple~CloudDocs/DAT/transformers/data_3000.npz'
data = np.load(dataset_path)

X = data['arr_0']
y = data['arr_1']

context_length = 512
forecast_horizon = 96
patch_length = 8
num_workers = 32  # Reduce this if you have low number of CPU cores
batch_size = 8  # Adjust according to GPU memory

# get split
#X = np.reshape(X, (X.shape[2], X.shape[1], X.shape[0]))
   
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, test_size=0.5, shuffle=True)

X_train = np.transpose(X_train, (1,2,0))
X_valid = np.transpose(X_valid, (1,2,0))
X_test = np.transpose(X_test, (1,2,0))



a = ScaleTS()
a.train(X_train)
a.preprocess(X_train)

# %% get split

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
#%%

config = PatchTSMixerConfig(
    channel_consistent_masking=True,
    context_length=context_length,
    d_model=16,
    distribution_output="student_t",
    dropout=0.7,
    expansion_factor=5,
    gated_attn=True,
    head_aggregation="max_pool",
    head_dropout=0.5,
    init_std=0.02,
    loss="mse",
    mask_type="forecast",
    mask_value=0,
    masked_loss=True,
    mode="common_channel",
    model_type="patchtsmixer",
    norm_eps=1e-05,
    norm_mlp="LayerNorm",
    num_forecast_mask_patches=[
        2
    ],
    num_input_channels=len(forecast_columns),
    num_layers=3,
    num_parallel_samples=100,
    num_patches=32,
    #layersnum_targets=3,
    # output_range=null,
    patch_last=True,
    patch_length=patch_length,
    patch_stride=8,
    positional_encoding_type="sincos",
    post_init=False,
    #prediction_channel_indices= null,
    prediction_length=96,
    random_mask_ratio=0.5,
    scaling=True,
    self_attn=True,
    self_attn_heads=2,
    torch_dtype="float32",
    transformers_version="4.36.0.dev0",
    #unmasked_channel_indices= null,
    use_positional_encoding=True,
)
model = PatchTSMixerForPretraining(config)

training_args = TrainingArguments(
    output_dir="./checkpoint/patchtsmixer_4/pretrain/data3000/output/",
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
    # Make sure to specify a logging directory
    logging_dir="./pretrain/data3000/logs/",
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
    # label_names=["future_values"],
)

# Create the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    # Number of epochs with no improvement after which to stop
    early_stopping_patience=10,
    # Minimum improvement required to consider as improvement
    early_stopping_threshold=0.0001,
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
print("\n\nDoing pretraining on ETPD Data")
trainer.train()

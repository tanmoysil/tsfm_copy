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

warnings.filterwarnings("ignore", module="torch")

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

dataset_path = 'data_single.csv'
timestamp_column = "date"
id_columns = []

context_length = 512
forecast_horizon = 96
patch_length = 8
num_workers = 32  # Reduce this if you have low number of CPU cores
batch_size = 8  # Adjust according to GPU memory

data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
    index_col=0
)
forecast_columns = list(data.columns[1:7])

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
        self_attn=True,
        self_attn_heads=2,
        use_positional_encoding=True,
        d_model=16,
        num_layers=12,
        expansion_factor=5,
        dropout=0.1,
        head_dropout=0.2,
        mode="common_channel",
        scaling="std",
    )
model = PatchTSMixerForPrediction(config)

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
        logging_dir="./pretrain/data3000/logs/",  # Make sure to specify a logging directory
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
print("\n\nDoing forecasting training on ETPD Data")
trainer.train()
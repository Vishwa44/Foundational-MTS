# from huggingface_hub import hf_hub_download
import torch
import time
import os
# # from transformers import PatchTSTConfig,Trainer,TrainingArguments,EarlyStoppingCallback, PatchTSTForPrediction
from src.transformers import PatchTSTConfig, Trainer, TrainingArguments, EarlyStoppingCallback, PatchTSTForPrediction
import numpy as np
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

import argparse


def get_dataset(args):
    data_path = os.path.join(args.dataset_path, args.dataset)
    time_col = "date"
    id_columns = []

    valid_start_index = 12 * 30 * 24 - args.context_length
    valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

    test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - args.context_length
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24

    data = pd.read_csv(
        data_path,
        parse_dates=[time_col],
    )
    forecast_columns = list(data.columns[1:])


    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.15)
    num_valid = len(data) - num_train - num_test
    border1s = [
        0,
        num_train - args.context_length,
        len(data) - num_test - args.context_length,
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

    time_series_preprocessor = TimeSeriesPreprocessor(
        timestamp_column=time_col,
        id_columns=id_columns,
        scaling=True,
    )
    time_series_preprocessor = time_series_preprocessor.train(train_data)

    train_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=args.context_length,
        prediction_length=args.forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(valid_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=args.context_length,
        prediction_length=args.forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        time_series_preprocessor.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=args.context_length,
        prediction_length=args.forecast_horizon,
    )
    return train_dataset, valid_dataset, test_dataset, forecast_columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    # basic config
    parser.add_argument('--context_length', type=int, required=True, default=128, help='context length')
    parser.add_argument('--forecast_horizon', type=int, required=True, default=96, help='Forecast horizon')
    parser.add_argument('--patch_length', type=int, required=True, default=20, help='patch length')
    parser.add_argument('--num_workers', type=int, required=True, default=4, help='Number of workers')
    parser.add_argument('--num_epochs', type=int, required=True, default=50, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, required=True, default=4, help='batch size')
    parser.add_argument('--lr', type=float, required=True, default=0.001, help='learning rate')
    parser.add_argument('--dataset', type=str, required=True, help='choose dataset')
    parser.add_argument('--dataset_path', type=str, required=True, default = "/dataset", help='path to dataset folder')
    parser.add_argument('--log_path', type=str, required=True, default = "/logs", help='path to log')
    parser.add_argument('--channel_attention', type=bool, required=True, default = True, help='Perform channel attention')
    parser.add_argument('--new_channel_attn', type=bool, required=True, default = True, help='Perform new channel attention')
    args = parser.parse_args()
    
    """
    /scratch/vg2523/mts/bin/python main_driver.py --context_length 128 --forecast_horizon 96 --patch_length 20 --num_workers 8 --num_epochs 5 --batch_size 64 --lr 0.001 --dataset "electricity" --dataset_path "/home/vg2523/mts_test/dataset" --log_path "/home/vg2523/mts_test/logs" --channel_attention True --new_channel_attn True
    """
    
    uid = args.dataset + "_" + str(args.context_length) + "_" + str(args.forecast_horizon) + "_" + str(args.patch_length) + '_' + str(time.time())
    args.dataset = args.dataset + ".csv"
    print(uid)
    print("-----------")
    print(args)
    train_dataset, valid_dataset, test_dataset, forecast_columns = get_dataset(args)
    
    config = PatchTSTConfig(
    num_input_channels=len(forecast_columns),
    context_length=args.context_length,
    patch_length=args.patch_length,
    patch_stride=args.patch_length,
    prediction_length=args.forecast_horizon,
    random_mask_ratio=0.4,
    d_model=args.context_length,
    channel_attention = args.channel_attention,
    new_channel_attention = args.new_channel_attn)
    
    model = PatchTSTForPrediction(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Numebr of model parameters: ", pytorch_total_params)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.log_path, uid),
        overwrite_output_dir=True,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=os.path.join(args.log_path, uid),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"])   
    
    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001)  # Minimum improvement required to consider as improvement
    
    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
        # compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)

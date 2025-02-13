# -*- coding: utf-8 -*-
"""
Function to generate CSV files summarizing training results.

List of available statistics:
['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro', 'recall_macro', 'recall_weighted',
 'precision_micro', 'precision_macro', 'precision_weighted', 'trainloss', 'testloss']


Example usage:
    generate_training_summary(r'/RESULTS')

author: magipamies
datetime: 4/6/2023 21:03
"""

import os
from os import path as pth
import numpy as np
import pandas as pd

# --------------------------------------------------
# Dictionary to shorten metric names
dict_vn = {"lr": "learning-rate", "wd": "weight-decay", "sl": "sequencelength", "dc": "datecrop", "tt": "trans_type"}


def parse_hyperparameters_folder_name(run):
    """
    Parses the hyperparameter folder name to extract configuration details.

    Args:
        run (str): Path to the training folder.

    Returns:
        dict: Dictionary containing extracted hyperparameters and the model name.
    """
    run = pth.basename(run)
    model, *hyperparameter_str = run.split("_")
    hyperparameter = {}

    for kv in hyperparameter_str:
        k, v = kv.split("=")
        k = dict_vn.get(k, k)  # Replace abbreviation with full name if it exists
        if k == "trans_type":
            v = v.replace("-", "_")  # Convert dashes to underscores for uniformity
        hyperparameter[k] = v

    hyperparameter["model"] = model  # Store model name
    return hyperparameter


def read_best_epoch_stats(run, stat_name="testloss"):
    """
    Reads training statistics and finds the best epoch based on the given metric.

    Args:
        run (str): Path to the training folder.
        stat_name (str): Name of the statistic to use for selecting the best epoch. Defaults to "testloss".

    Returns:
        dict: Dictionary with the statistics of the best epoch, including the epoch number.
    """
    df = pd.read_csv(pth.join(run, "trainlog.csv"), index_col=0)

    # Determine the best epoch based on the metric type (minimize losses, maximize others)
    bestepoch = df[stat_name].idxmin() if stat_name.endswith("loss") else df[stat_name].idxmax()
    best_stats = df.loc[bestepoch].to_dict()
    best_stats["epoch"] = bestepoch  # Store the best epoch number

    return best_stats


def merge(dict1, dict2):
    """
    Merges two dictionaries into one.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary containing all key-value pairs.
    """
    dict1.update(dict2)  # Merge the two dictionaries
    return dict1


def get_df(run, dict_h, tn=None):
    """
    Creates a DataFrame from a training log file and adds model hyperparameters.

    Args:
        run (str): Path to the folder containing "trainlog.csv".
        dict_h (dict): Dictionary with additional columns and values to add to the DataFrame.
        tn (int, optional): Trial number. If provided, adds a "Trail" column with the corresponding value.

    Returns:
        pandas.DataFrame: DataFrame containing the training log and additional information.
    """
    df = pd.read_csv(pth.join(run, "trainlog.csv"), index_col=0)

    # Add model hyperparameters to the DataFrame
    for k, v in dict_h.items():
        df[k] = v

    if tn:  # If a trial number is provided, add a "Trail" column
        df["Trail"] = f"T{tn}"

    return df


def generate_training_summary(logdir, stat_l=None, out_folder=None):
    """
    Generates summary CSV files from training logs.

    Args:
        logdir (str): Path to the directory containing training runs.
        stat_l (list, optional): List of statistics to analyze. If None, a default list is used.
        out_folder (str, optional): Path to save the output files. Defaults to `logdir` if not specified.

    Outputs:
        - all_trainlog.csv: Combined CSV file with all training logs.
        - best_<stat>.csv: CSV files with the best epoch for each metric in the training runs.
    """
    if out_folder is None:
        out_folder = logdir  # Default output folder is the log directory

    if stat_l is None:
        stat_l = ['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro', 'recall_macro',
                  'recall_weighted', 'precision_micro', 'precision_macro', 'precision_weighted', 'testloss']

    # Select valid training runs (folders that contain "trainlog.csv")
    runs = [run for run in os.listdir(logdir) if pth.exists(pth.join(logdir, run, "trainlog.csv"))]

    # Create a list to store DataFrames
    list_df = []

    # Iterate over training runs
    for i, run in enumerate(runs, start=1):
        # Extract hyperparameter information from the folder name
        # result = parse_hyperparameters_folder_name(pth.join(logdir, run))
        result = {'Trial': pth.basename(run)}
        # Generate DataFrame and append it to the list
        df_res = get_df(pth.join(logdir, run), result, i)
        list_df.append(df_res)

    # Combine all training logs into a single DataFrame and save it
    df_results = pd.concat(list_df)
    df_results.to_csv(pth.join(out_folder, "all_trainlog.csv"))

    # Generate and save summary CSVs for the best epoch of each statistic
    for stat_n in stat_l:
        sta_v = []
        for run in runs:
            # Extract hyperparameters
            # result = parse_hyperparameters_folder_name(pth.join(logdir, run))
            result = {'Trial': pth.basename(run)}
            # Get the best epoch statistics for the given metric
            best_epoch_stats = read_best_epoch_stats(pth.join(logdir, run), stat_name=stat_n)
            # Merge the results
            dic_val = merge(result, best_epoch_stats)
            sta_v.append(dic_val)

        # Create DataFrame and save to CSV
        df_st_val = pd.DataFrame(sta_v)
        df_st_val.to_csv(pth.join(out_folder, f"best_{stat_n}.csv"))

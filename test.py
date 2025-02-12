# -*- coding: utf-8 -*-
"""
Test and evaluate deep learning models for time series classification on the CatCrops dataset.

This script performs the following tasks:
1. Loads a trained model and selects the best epoch based on a specified performance metric.
2. Prepares the test dataset using a selected testing mode (`test2022`, `test2023`).
3. Runs inference on the test dataset to generate predictions.
4. Computes performance metrics such as accuracy, F1-score, and confusion matrices.
5. Saves results as CSV files and optionally as shapefiles for geospatial analysis.

The script supports various testing configurations, including:
- Feature selection (e.g., cloud percentage, day of the year).
- Custom hyperparameters and batch processing settings.


Names of Crop Types:

| Abbreviation | Catalan Name           | English Name           |
|--------------|-------------------------|------------------------|
| P            | Panís                   | Maize                  |
| CH           | Cereals d'hivern        | Winter cereals         |
| C            | Colza                   | Rapeseed               |
| AL           | Alfals                  | Alfalfa                |
| GS           | Gira-sol                | Sunflower              |
| SR           | Sorgo                   | Sorghum                |
| SJ           | Soja                    | Soybean                |
| H            | Horta                   | Vegetables             |
| G            | Guaret                  | Fallow                 |
| CP           | Cultius permanents      | Orchards               |
| AE           | Altres extensius        | Other extensive        |
| AR           | Arròs                   | Rice                   |
| OL           | Oliverar                | Olives                 |
| VY           | Vinya                   | Vineyard               |
| RF           | Ray-grass festuca       | Ray Grass and Festuca  |
| VÇ           | Veça                    | Vetch                  |
| PR           | Protaginoses            | Fabaceae               |
| CB           | Cebes                   | Onion                  |
| VV           | Vivers                  | Nurseries              |
| DCP          | Dc panís                | Double Crop Maize      |
| DCGS         | Dc gira-sol             | Double Crop Sunflower  |


Usage:
Run this script from the command line with custom arguments or default settings.

Example:
    python test.py --model TransformerEncoder --mode test2023 --batchsize 512 --logdir /path/to/results

datetime:27/5/2023 16:50
"""

import argparse
import os
from os import path as pth
import glob
from training_summary_generator import read_best_epoch_stats
from train import get_model, test_epoch, metrics
from catcrops import CatCrops
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from catcrops.transforms import get_transform_CatCrops
from datetime import datetime, timedelta
import time

# Classes name
l_classname = ['P', "CH", 'C', 'AL', 'GS', 'SR', 'SJ', 'H', 'G', 'CP', 'AE', 'AR', 'OL', 'VY', 'RF', 'VÇ', 'PR', 'CB',
               'VV', 'DCP', 'DCGS']
# Models name
list_models_n = ['TransformerEncoder', 'LSTM', 'StarRNN', 'MSResNet', 'OmniScaleCNN', 'TempCNN']

def get_dataloader_test(datapath, mode, batchsize, workers, preload_ram=False,
                   sequencelength=45, datecrop="26/05/2023", use_previous_year_TS=False ,sparse=False, cp=False, doa=False, L2A = True, pclassid=False, pcrop=False,
                   pvar=False, sreg=False, mun=False, com=False, prov=False, elev=False, slope=False, noreplace=False,recompile_h5_from_csv=False):
    """
    Retrieves the test dataloaders based on the selected dataset mode.

    Args:
        datapath (str): Path to the dataset directory.
        mode (str): Specifies the dataset to use for testing ('test2022', 'test2023', 'test2024').
        batchsize (int): Number of samples processed per batch.
        workers (int): Number of CPU workers for data loading.
        preload_ram (bool): Whether to load the dataset into RAM.
        sequencelength (int): Length of the time series sequences.
        datecrop (str): The latest date to crop the time series data. Date format "DD/MM/YYYY" or "all" to test all dates.
        use_previous_year_TS (bool): Whether to include time series from the previous year.
        sparse (bool): Whether to fill missing time series values with zeros.
        cp (bool): Whether to include cloud probability as an input feature.
        doa (bool): Whether to include the day of the year as an input feature.
        L2A (bool): Whether to use Sentinel-2 L2A spectral data.
        pclassid (bool): Whether to include the previous year's crop classification.
        pcrop (bool): Whether to include the previous year's crop code.
        pvar (bool): Whether to include the previous year's variety code.
        sreg (bool): Whether to include the previous year's irrigation system data.
        mun (bool): Whether to include the municipality code.
        com (bool): Whether to include the comarca (region) code.
        prov (bool): Whether to include the province code.
        elev (bool): Whether to include the field elevation.
        slope (bool): Whether to include the field slope.
        noreplace (bool): Whether to replace missing sequence values with zeros.
        recompile_h5_from_csv (bool): Whether to recompile the dataset from CSV files.

    Returns:
        dict: A dictionary containing the test dataloaders:
            - "t": General test dataloader.
            - "ll": Test dataloader for Lleida.
            - "bt": Test dataloader for Baix Ter.
        dict: Metadata containing:
            - "ndims" (int): Number of input features.
            - "num_classes" (int): Number of classification classes.
            - "sequencelength" (int): Length of the time series sequences.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    datapath = pth.abspath(datapath)
    transform = get_transform_CatCrops(sequencelength, datecrop, use_previous_year_TS ,sparse, cp, doa, noreplace)

    # Select the dataset based on the mode
    if mode == 'test2022':
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    elif mode == "test2023":
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    else:
        raise ValueError("only --mode 'test2022' or 'test2023' allowed")

    # Create dataloaders
    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)
    testdataloader_ll = DataLoader(dst_lleida, batch_size=batchsize, shuffle=False, num_workers=workers)
    testdataloader_bt = DataLoader(dst_baixter, batch_size=batchsize, shuffle=False, num_workers=workers)

    # Metadata
    meta = {
        "ndims": dst_baixter[0][0].shape[1],  # Number of input dimensions
        "num_classes": len(dst_baixter.classes),  # Number of classes
        "sequencelength": sequencelength  # Length of the time series sequence
    }

    return {"t": testdataloader, "ll": testdataloader_ll, "bt": testdataloader_bt}, meta


def do_cm_plot(dfcf, title_n, out_fol, mdesc, test_z='', n_dec="2"):
    """
    Generates and saves a heatmap plot of the confusion matrix.

    Args:
        dfcf (pd.DataFrame): The confusion matrix as a Pandas DataFrame.
        title_n (str): Title of the plot.
        out_fol (str): Output folder where the plot will be saved.
        mdesc (str): Model description or additional metadata.
        test_z (str, optional): Zone name for labeling the plot.
        n_dec (str, optional): Number of decimal places to display in annotations (default is "2").

    Returns:
        None: Saves the confusion matrix plot as a PNG file in the specified output folder.
    """
    plt.figure(figsize=(12, 8))
    plt.suptitle(mdesc, fontsize=7)
    plt.title('%s' % title_n)
    plt.title('%s' % test_z, loc='right')
    sns.heatmap(dfcf, annot=True, fmt=".%sf" % n_dec, cmap="viridis")
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.tight_layout()
    plt.savefig(pth.join(out_fol, "%s.png" % title_n))
    plt.close()

def save_metrics(y_t, y_p, y_s, f_id, test_l, list_clasn, s_f, zn='t'):
    """
    Computes and saves evaluation metrics along with classification reports.

    Args:
        y_t (torch.Tensor): True class labels.
        y_p (torch.Tensor): Predicted class labels.
        y_s (torch.Tensor): Confidence scores for each prediction.
        f_id (torch.Tensor): Field IDs corresponding to each sample.
        test_l (torch.Tensor): Test loss value.
        list_clasn (list[str]): List of class names.
        s_f (str): Output folder to save the metrics and reports.
        zn (str, optional): Zone identifier (e.g., 't' for total, 'll' for Lleida, 'bt' for Baix Ter).

    Returns:
        pd.DataFrame: DataFrame containing computed evaluation metrics.
    """
    y_t = y_t.cpu()
    y_p = y_p.cpu()

    if zn == 'll' and 11 not in np.unique(y_p):
        list_clasn = list_clasn.copy()
        list_clasn.remove('AR')

    # Obtenim els valors de les mètriques
    scores = metrics(y_t, y_p)
    scores["testloss"] = test_l.cpu().detach().numpy()[0]
    scoresdf = pd.DataFrame([scores])
    scoresdf['zona'] = zn

    # Guardem les mètriques i els arrays obtinguts del test
    print(sklearn.metrics.classification_report(y_t, y_p, target_names=list_clasn),
          file=open(os.path.join(s_f, "%s_classification_report.txt" % zn), "w"))
    np.save(os.path.join(s_f, "%s_y_pred.npy" % zn), y_p.numpy())
    np.save(os.path.join(s_f, "%s_y_true.npy" % zn), y_t.numpy())
    np.save(os.path.join(s_f, "%s_y_score.npy" % zn), y_s.cpu().numpy())
    np.save(os.path.join(s_f, "%s_field_ids.npy" % zn), f_id.numpy())

    return scoresdf


def get_df_cmatrix(c_matrix, list_clasn):
    """
    Converts a confusion matrix into a Pandas DataFrame.

    Args:
        c_matrix (np.array): Numpy array representing the confusion matrix.
        list_clasn (list[str]): List of class names.

    Returns:
        pd.DataFrame: Confusion matrix formatted as a Pandas DataFrame.

    """
    return pd.DataFrame(c_matrix, index=[i for i in list_clasn], columns=[i for i in list_clasn])


def get_matrix_conf(y_t, y_p, list_clasn, s_f, rn, t_m, zn='t'):
    """
    Computes and saves multiple confusion matrices (raw, normalized) and generates plots.

    Args:
        y_t (torch.Tensor): True class labels.
        y_p (torch.Tensor): Predicted class labels.
        list_clasn (list[str]): List of class names.
        s_f (str): Output folder where results will be saved.
        rn (str): Model name/run identifier.
        t_m (str): Test mode identifier (e.g., 'test2023').
        zn (str, optional): Zone identifier (default is 't').

    Returns:
        None: Saves confusion matrices and plots as CSV and PNG files.

    """

    # Move tensors to CPU
    y_t = y_t.cpu()
    y_p = y_p.cpu()

    # Remove class 'AR' from the class list if not present in predictions
    if zn == 'll' and 11 not in np.unique(y_p):
        list_clasn = list_clasn.copy()
        list_clasn.remove('AR')

    # Compute and save different versions of the confusion matrix
    cf_matrix_v = sklearn.metrics.confusion_matrix(y_t, y_p)
    df_cm_v = get_df_cmatrix(cf_matrix_v, list_clasn)
    df_cm_v.to_csv(pth.join(s_f, f"{zn}_cf_val.csv"))

    # Normalize by true labels
    cf_matrix_t = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='true')
    df_cm_t = get_df_cmatrix(cf_matrix_t, list_clasn)
    df_cm_t.to_csv(pth.join(s_f, f"{zn}_cf_t.csv"))

    # Normalize by predicted labels
    cf_matrix_p = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='pred')
    df_cm_p = get_df_cmatrix(cf_matrix_p, list_clasn)
    df_cm_p.to_csv(pth.join(s_f, f"{zn}_cf_p.csv"))

    # Normalize by all labels
    cf_matrix_all = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='all')
    df_cm_all = get_df_cmatrix(cf_matrix_all, list_clasn)
    df_cm_all.to_csv(pth.join(s_f, f"{zn}_cf_all.csv"))

    # Generate confusion matrix plots
    do_cm_plot(df_cm_v, f"{zn}_cf values", s_f, rn, test_z=t_m, n_dec="0")
    do_cm_plot(df_cm_t, f"{zn}_cf true", s_f, rn, test_z=t_m)
    do_cm_plot(df_cm_p, f"{zn}_cf prediction", s_f, rn, test_z=t_m)
    do_cm_plot(df_cm_all, f"{zn}_cf all", s_f, rn, test_z=t_m)

# Function to generate a list of dates
def generate_dates(start_date_str, end_date_str, skip_days):
    """
    Generates a list of dates between a start and end date, skipping a specified number of days.

    Args:
        start_date_str (str): Start date in the format "DD/MM/YYYY".
        end_date_str (str): End date in the format "DD/MM/YYYY".
        skip_days (int): Number of days to skip between each generated date.

    Returns:
        list[str]: A list of date strings in the format "DD/MM/YYYY".
    """
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y")

    current_date = start_date
    date_list = []

    # Generate dates while skipping the specified number of days
    while current_date <= end_date:
        date_list.append(current_date.strftime("%d/%m/%Y"))
        current_date += timedelta(days=skip_days)

    return date_list

def parse_args():
    """
    Test and evaluate deep learning models for time series on the CatCrops dataset.

    This script loads a trained model, evaluates it on a specified test dataset partition, and computes performance
    metrics. It supports different testing configurations, including feature selection, spectral data inclusion, and
    model hyperparameter tuning.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing all test settings.

    Raises:
        ValueError: If an invalid hyperparameter format is provided.
        """

    parser = argparse.ArgumentParser(
        description=(
            "Test and evaluate deep learning models for time series on the CatCrops dataset. "
            "This script tests a model on a test dataset partition and evaluates performance."
        )
    )

    # ---------------------------------
    # Model and Training Configuration
    # ---------------------------------
    parser.add_argument('--model', type=str, default="TransformerEncoder",
                        help='Select model architecture. Available models: "TransformerEncoder", "LSTM", "TempCNN", '
                             '"MSRestNet", "StarRNN", "OmniScaleCNN"')
    parser.add_argument('-SL', '--sequencelength', type=int, default=None,
                        help='Length of the input time series sequence.')
    parser.add_argument('--datecrop', type=str, default="30/12/2023",
                        help='Latest date to crop the time series data. Date format "DD/MM/YYYY" or "all" to test all dates.')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='Batch size (number of time series processed simultaneously).')
    parser.add_argument('-e', '--epoch', type=int, default=None,
                        help='Epoch number to select the model. If None, the best epoch is chosen based on accuracy.')
    parser.add_argument('-m', '--mode', type=str, default="test2023",
                        help='Testing mode (e.g., "test2022", "test2023").')

    # ---------------------------------
    # Dataset and Hardware Settings
    # ---------------------------------
    parser.add_argument('-D', '--datapath', type=str, default="/media/hdd11/tipus_c/catcrops_dataset_v2/",
                        help='Path to the directory containing the dataset.')
    parser.add_argument('-w', '--workers', type=int, default=0,
                        help='Number of CPU workers for loading the next batch.')
    parser.add_argument('-H', '--hyperparameter', type=str, default=None,
                        help='Model-specific hyperparameters as a comma-separated string (e.g., "param1=value1,param2=value2").')

    # ---------------------------------
    # Optimizer Settings
    # ---------------------------------
    parser.add_argument('--weight-decay', type=float, default=5e-08,
                        help='Weight decay regularization for the optimizer (default: 5e-08).')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer (default: 1e-3).')

    # ---------------------------------
    # Memory and Computational Settings
    # ---------------------------------
    parser.add_argument('--preload-ram', action='store_true',
                        help='Load the dataset into RAM upon initialization to speed up testing.')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Computation device: "cpu" or "cuda". Default is automatically detected.')

    # ---------------------------------
    # Logging and Trial Settings
    # ---------------------------------
    parser.add_argument('-l', '--logdir', type=str, default="./Results",
                        help='Directory to store logs, progress, and model outputs.')
    parser.add_argument('--trial', type=str, default="Trial001",
                        help='Trial name to distinguish different test runs.')

    # ---------------------------------
    # Feature Selection Options
    # ---------------------------------
    parser.add_argument('--use_previous_year_TS', action="store_true",
                        help='Use time series data from the previous year when cropping to a specific date.')
    parser.add_argument('--sparse', action="store_true",
                        help='Fill missing dates in the time series with zero values.')
    parser.add_argument('--noreplace', action="store_true",
                        help='Fill sequences with zeros if their length is shorter than "sequencelength".')
    parser.add_argument('--cp', action="store_true",
                        help='Include cloud percentage as an additional input feature.')
    parser.add_argument('--doa', action="store_true",
                        help='Include the day of the year as an additional input feature.')

    # ---------------------------------
    # Spectral and Environmental Features
    # ---------------------------------
    parser.add_argument('--L2A', action="store_true", help='Use Sentinel-2 L2A spectral data.')
    # parser.add_argument('--LST', action="store_true", help='Include Landsat spectral data.')
    # parser.add_argument('--ET', action="store_true", help='Include evapotranspiration (ET) data.')
    parser.add_argument('--pclassid', action="store_true", help='Use previous year’s classification as input.')
    parser.add_argument('--pcrop', action="store_true", help='Use previous year’s crop code as input.')
    parser.add_argument('--pvar', action="store_true", help='Use previous year’s variety code as input.')
    parser.add_argument('--sreg', action="store_true", help='Use previous year’s irrigation system information.')
    parser.add_argument('--mun', action="store_true", help='Include the municipality code as an input feature.')
    parser.add_argument('--com', action="store_true", help='Include the comarca (region) code as an input feature.')
    parser.add_argument('--prov', action="store_true", help='Include the province code as an input feature.')
    parser.add_argument('--elev', action="store_true", help='Include the elevation of the field as an input feature.')
    parser.add_argument('--slope', action="store_true", help='Include the slope of the field as an input feature.')

    # ---------------------------------
    # Evaluation Settings
    # ---------------------------------
    parser.add_argument('--stat_n', type=str, default="accuracy",
                        help='Metric used to select the best epoch (e.g., accuracy, F1-score).')
    parser.add_argument('--do_shp', action="store_true",
                        help='Save output as a shapefile (.shp).')
    parser.add_argument('--recompile_h5_from_csv', action="store_true",
                        help='Recompile HDF5 dataset from CSV files.')

    args = parser.parse_args()

    # ---------------------------------
    # Convert hyperparameter string to dictionary
    # ---------------------------------
    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict

    # ---------------------------------
    # Automatically set computation device if not specified
    # ---------------------------------
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def get_default_parse_arguments():
    """
        Generates default argument configurations for testing deep learning models on time series data.

        This function provides predefined parameters for testing a model on the CatCrops dataset without
        requiring manual input. It includes default values for model selection, dataset paths,
        computational resources, feature selection, and evaluation settings.

        Returns:
            parsed_arguments: An object containing all default test settings.
        """

    class parsed_arguments():
        def __init__(self):
            self.prova = None

    args = parsed_arguments()

    # ---------------------------------
    # Model and Training Configuration
    # ---------------------------------
    args.model = "TransformerEncoder"  # Model architecture to use for testing. Available models: TransformerEncoder, LSTM, TempCNN, MSRestNet, StarRNN, OmniScaleCNN
    args.batchsize = 512  # Number of time series processed simultaneously.
    args.epoch = None  # Epoch number for model selection. If None, the best epoch is chosen.
    args.hyperparameter = dict()  # Model specific hyperparameter as single string, separated by comma of format param1=value1,param2=value2

    # ---------------------------------
    # Dataset and Hardware Settings
    # ---------------------------------
    args.datapath = "/media/hdd11/tipus_c/catcrops_dataset_v2/"  # Path to the dataset directory.
    args.workers = 0  # Number of CPU workers for loading data.
    args.preload_ram = False  # Whether to load the dataset into RAM for faster processing.
    args.device = None  # Computation device: "cpu" or "cuda". Automatically detected if None.
    args.logdir = "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS"  # Directory for saving test results.

    # ---------------------------------
    # Optimizer Settings
    # ---------------------------------
    args.weight_decay = 5e-08  # Optimizer weight_decay. Regularization parameter to prevent overfitting.
    args.learning_rate = 1e-3  # Learning rate for the optimizer.

    # ---------------------------------
    # Testing Mode and Date Configuration
    # ---------------------------------
    args.mode = "test2023"  # Defines which dataset year is used for testing ("test2022", "test2023").
    args.datecrop = "01/06/2023"  # Date format "DD/MM/YYYY" or "all" to test all dates.
    args.sequencelength = None  # Sequence length for time series data.

    # ---------------------------------
    # Feature Selection Options
    # ---------------------------------
    args.use_previous_year_TS = False  # Use previous year's time series data.
    args.sparse = True  # Fill missing time series values with zero.
    args.noreplace = False  # If sequence length is shorter than expected, fill with zeros.
    args.cp = True  # Include cloud percentage as an input feature.
    args.doa = True  # Include the day of the year as an input feature.

    # ---------------------------------
    # Spectral and Environmental Features
    # ---------------------------------
    args.L2A = True  # Use Sentinel-2 L2A spectral data.
    # args.LST = False  # Include Landsat spectral data.
    # args.ET = False  # Include evapotranspiration (ET) time series.
    args.pclassid = False  # Use previous year’s crop classification as input.
    args.pcrop = False  # Use previous year’s crop code as input.
    args.pvar = False  # Use previous year’s variety code as input.
    args.sreg = False  # Use previous year’s irrigation system information.
    args.mun = False  # Include the municipality code as an input feature.
    args.com = False  # Include the comarca (region) code as an input feature.
    args.prov = False  # Include the province code as an input feature.
    args.elev = False  # Include the elevation of the field as an input feature.
    args.slope = False  # Include the slope of the field as an input feature.

    # ---------------------------------
    # Evaluation Settings
    # ---------------------------------
    args.trial = "Trial002-Crop231231_SL_L2A"  # Identifier for the test run.
    args.recompile_h5_from_csv = False  # Whether to recompile the dataset from CSV files.
    args.stat_n = "accuracy"  # Metric used to select the best epoch.
    args.do_shp = True  # Save the output as a shapefile (.shp).
    args.do_csv = False  # Save the output as a CSV file.

    # ---------------------------------
    # Automatically set computation device if not specified
    # ---------------------------------
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    # Parse command-line arguments
    #args = get_default_parse_arguments()
    args = parse_args()

    # Define key paths and settings
    logdir = args.logdir  # Directory where trained models and logs are stored
    device = args.device  # Computation device (CPU/GPU)
    out_f = logdir.replace('/RESULTS', '/TEST')  # Directory to store test results
    run = args.trial  # Trial identifier

    # ---------------------------------
    # Select the best model epoch
    # ---------------------------------
    if args.epoch is None:
        # Find the epoch with the highest value for the selected metric
        best_epoch_stats = read_best_epoch_stats(pth.join(logdir, run), stat_name=args.stat_n)
        best_epoch = best_epoch_stats['epoch']
    else:
        best_epoch = args.epoch

    # Retrieve the path to the selected model checkpoint
    list_pth = [pth.basename(x) for x in glob.glob(pth.join(logdir, run, "*.pth"))]
    path_model = [x for x in list_pth if f"model_{int(best_epoch)}.pth" == x][0]

    # ---------------------------------
    # Generate list of dates for evaluation
    # ---------------------------------
    if args.datecrop == "all":
        start_date_str = "07/01/" + args.mode[-4:]  # Start date (January 7th of the test year)
        end_date_str = "31/12/" + args.mode[-4:]  # End date (December 31st of the test year)
        skip_days = 7  # Step size for date evaluation
        dates2crop = generate_dates(start_date_str, end_date_str, skip_days)
    else:
        dates2crop = [args.datecrop]  # Use specified date if not "all"

    # ---------------------------------
    # Iterate over each test date
    # ---------------------------------
    for datecrop in dates2crop:
        print("Processing date:", datecrop)
        datecrop_object = datetime.strptime(datecrop, "%d/%m/%Y")
        reversed_datecrop_string = datecrop_object.strftime("%y%m%d")  # Reverse date format for filenames

        # Define output directory for test results
        save_f = os.path.join(out_f, run, "%s_%s_%s" % (args.mode, reversed_datecrop_string, pth.basename(path_model).split('.')[0]))
        os.makedirs(save_f, exist_ok=True)

        # ---------------------------------
        # Load the test dataset
        # ---------------------------------
        start_dataload = time.time()
        testdataloader_dict, meta = get_dataloader_test(
            args.datapath, args.mode, args.batchsize, args.workers, args.preload_ram,
            args.sequencelength, datecrop, args.use_previous_year_TS, args.sparse, args.cp,
            args.doa, args.L2A, args.pclassid, args.pcrop, args.pvar,
            args.sreg, args.mun, args.com, args.prov, args.elev, args.slope, args.noreplace,
            args.recompile_h5_from_csv
        )
        end_dataload = time.time()
        print(f"Processing time for Dataload: {end_dataload - start_dataload:.4f} seconds")

        # Retrieve dataset metadata
        num_classes = meta["num_classes"]
        ndims = meta["ndims"]
        sequencelength = meta["sequencelength"]

        # ---------------------------------
        # Load the trained model
        # ---------------------------------
        model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)

        # Update model name with test parameters
        model.modelname += f"_lr={args.learning_rate}_wd={args.weight_decay}" \
                           f"_sl={args.sequencelength}" \
                           f"_dc={''.join(datecrop.split('/'))}_mode={args.mode}"

        print(f"Initialized {model.modelname}")

        # Load the trained model state
        model.load_state_dict(torch.load(pth.join(logdir, run, path_model), map_location=device)["model_state"])

        # Define loss function for evaluation
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        end_model_load = time.time()
        print(f"Processing time for Model load: {end_model_load - end_dataload:.4f} seconds")

        # ---------------------------------
        # Evaluate the model on each test region
        # ---------------------------------
        list_scores = []  # Store evaluation scores for each test region

        for zona_n, testdl in testdataloader_dict.items():
            # Perform inference on the test dataset
            start_inference = time.time()
            test_loss, y_true, y_pred, y_score, field_ids = test_epoch(model, criterion, testdl, device)
            end_inference = time.time()
            print("Processing time for " + zona_n + f" load: {end_inference - start_inference:.4f} seconds")
            print(f"Logging results to {save_f}")

            # Compute evaluation metrics
            scores_df = save_metrics(
                y_t=y_true, y_p=y_pred, y_s=y_score, f_id=field_ids, test_l=test_loss,
                list_clasn=l_classname, s_f=save_f, zn=zona_n
            )

            # Store scores for later aggregation
            list_scores.append(scores_df)

            # Compute and save confusion matrices
            get_matrix_conf(y_t=y_true, y_p=y_pred, list_clasn=l_classname, s_f=save_f, rn=run, t_m=args.mode, zn=zona_n)

            # ---------------------------------
            # Save test results as shapefile (if enabled)
            # ---------------------------------
            if args.do_shp and zona_n != 't':
                regio = 'lleida' if zona_n == 'll' else 'baixter'

                # Load the dataset for the specified region
                ds = CatCrops(
                    region=regio, root=args.datapath, year=int(args.mode[-4:]), preload_ram=False,
                    L2A=args.L2A, pclassid=args.pclassid, pcrop=args.pcrop,
                    pvar=args.pvar, sreg=args.sreg, mun=args.mun, com=args.com, prov=args.prov,
                    elev=args.elev, slope=args.slope
                )

                # Retrieve geospatial data
                gdf = ds.geodataframe()

                # Get prediction confidence scores
                max_scores, _ = torch.max(y_score, dim=1)
                dict_pre = {'field_ids': field_ids.tolist(), 'pred_classid': y_pred.tolist(), 'pred_score': max_scores.tolist()}

                # Convert predictions into a DataFrame and merge with geospatial data
                df_p = pd.DataFrame.from_dict(dict_pre)
                gdf_p = gdf.join(df_p.set_index('field_ids'), on='id')

                # Compute prediction correctness
                gdf_p['prediccio'] = gdf_p['classid'] == gdf_p['pred_classid']

                # Map class IDs to class names
                dict_class = dict(zip(ds.classes, ds.classname))
                gdf_p['pred_classn'] = gdf_p['pred_classid'].map(dict_class)

                # Save as shapefile
                shp_folder = pth.join(save_f, 'shp')
                os.makedirs(shp_folder, exist_ok=True)
                gdf_p.to_file(pth.join(shp_folder, '%s_pred.shp' % zona_n))

        # Save all test results as CSV
        scores_df = pd.concat(list_scores)
        scores_df.to_csv(pth.join(save_f, "scorres_allz.csv"), index=False)

# -*- coding: utf-8 -*-
"""
Train deep learning models for time series classification on the CatCrops dataset.

This script performs the following tasks:
1. Loads a dataset partition for training and validation based on a specified mode.
2. Initializes and configures a deep learning model using user-defined hyperparameters.
3. Trains the model for a specified number of epochs.
4. Evaluates the model on a validation dataset and computes performance metrics.
5. Logs training progress and stores model checkpoints for each epoch.
6. Saves training loss, test loss, and performance metrics (e.g., accuracy, F1-score).
7. Optionally supports weighted sampling, early stopping, and dataset preprocessing.

The script supports various training configurations, including:
- Feature selection (e.g., cloud percentage, day of the year).
- Custom hyperparameters and batch processing settings.

Usage:
Run this script from the command line with custom arguments or default settings.

Example:
    python train.py --model TransformerEncoder --mode evaluation1 --epochs 100 --batchsize 512 --logdir /path/to/results

datetime:27/5/2023 16:50
"""

import argparse
from catcrops import CatCrops
from catcrops.models import LSTM, TempCNN, MSResNet, TransformerModel, StarRNN, OmniScaleCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch.utils.data as data
import torch
import pandas as pd
import os
import numpy as np
import sklearn.metrics
from catcrops.transforms import get_transform_CatCrops


def train(args):
    """
    Trains a deep learning model using training data and evaluates its performance on test data.

    Args:
        args (argparse.Namespace): Command-line arguments passed to the script.

    Returns:
        None
    """
    device = torch.device(args.device)
    if device.type == 'cuda':
        print("Working with CUDA device: " + torch.cuda.get_device_name(0))
    else:
        print("No CUDA device available.")

    # Get data loaders and metadata
    traindataloader, testdataloader, meta = get_dataloader(args.datapath, args.mode, args.batchsize, args.workers,
                                                           args.preload_ram, args.wight_sampling, args.sequencelength, args.datecrop,
                                                           args.use_previous_year_TS, args.sparse, args.cp, args.doa,
                                                           args.L2A, args.pclassid, args.pcrop, args.pvar,
                                                           args.sreg, args.mun, args.com, args.prov, args.elev, args.slope, args.noreplace)

    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    # Get the model and optimizer
    model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Modify model name to include important parameters
    model.modelname += f"_lr={args.learning_rate}_wd={args.weight_decay}" \
                       f"_sl={args.sequencelength}" \
                       f"_dc={''.join(args.datecrop.split('/'))}_mode={args.mode}"

    print(f"Initialized {model.modelname}")

    # Create log directory
    logdir = os.path.join(args.logdir, args.trial)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = []
    for epoch in range(args.epochs):
        # Train the model for one epoch
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)

        # Evaluate the model on test data
        test_loss, y_true, y_pred, y_score, field_ids = test_epoch(model, criterion, testdataloader, device)

        # Compute performance metrics
        scores = metrics(y_true.cpu(), y_pred.cpu())

        # Create a summary message with performance metrics
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])

        # Convert losses to numpy arrays
        test_loss = test_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]

        # Print training and test losses along with performance metrics
        print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)

        # Store losses and performance metrics in the log
        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = test_loss
        log.append(scores)

        # Convert the log into a DataFrame and save it as a CSV file
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

        # Save the model at each epoch
        save(model, os.path.join(logdir, "model_" + str(epoch) + ".pth"))

    # Save classification report, true labels, predicted labels, and scores
    print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()),
          file=open(os.path.join(logdir, "classification_report.txt"), "w"))
    np.save(os.path.join(logdir, "y_pred.npy"), y_pred.cpu().numpy())
    np.save(os.path.join(logdir, "y_true.npy"), y_true.cpu().numpy())
    np.save(os.path.join(logdir, "y_score.npy"), y_score.cpu().numpy())
    np.save(os.path.join(logdir, "field_ids.npy"), field_ids.numpy())


def get_dataloader(datapath, mode, batchsize, workers, preload_ram=False, weight_sampling=False,
                   sequencelength=45, datecrop="26/05/2023", use_previous_year_TS=False, sparse=False, cp=False,
                   doa=False, L2A=True, pclassid=False, pcrop=False, pvar=False,
                   sreg=False, mun=False, com=False, prov=False, elev=False, slope=False, noreplace=False):
    """
    #TODO ficar-ho en anglès
    Obté els dataloaders per a l'entrenament i la prova del model.

    En funció del mode pots seleccionar diferents datasets.
        - evaluation1: Train(ll22, bt22), Test(ll23, bt23)
        - evalutaion2: Train(ll23, bt23), Test(ll22, bt22)

    Args:
        datapath (str): Path to the dataset directory.
        mode (str): Training mode defining which years are used for training and testing.
        batchsize (int): Number of time series processed simultaneously in each batch.
        workers (int): Number of CPU workers for loading data.
        preload_ram (bool): Whether to load the dataset into RAM for faster access.
        weight_sampling (bool): Whether to apply class-weighted sampling to balance dataset.
        sequencelength (int): Length of the time series sequence.
        datecrop (str): The latest date to crop the time series data.
        use_previous_year_TS (bool): Whether to include time series data from the previous year.
        sparse (bool): Whether to fill missing dates with zero values.
        cp (bool): Whether to include cloud percentage information in the time series.
        doa (bool): Whether to include the day of the year in the time series.
        L2A (bool): Whether to use Level-2A spectral data.
        pclassid (bool): Whether to use the previous year's crop classification as input.
        pcrop (bool): Whether to include the previous year's crop code.
        pvar (bool): Whether to include the previous year's variety code.
        sreg (bool): Whether to include the previous year's irrigation system information.
        mun (bool): Whether to include the municipality code as an input feature.
        com (bool): Whether to include the comarca (region) code as an input feature.
        prov (bool): Whether to include the province code as an input feature.
        elev (bool): Whether to include the elevation of the field as an input feature.
        slope (bool): Whether to include the slope of the field as an input feature.
        noreplace (bool): Whether to replace missing sequences with zero values if sequence length is shorter.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): DataLoader for training dataset.
            - test_dataloader (torch.utils.data.DataLoader): DataLoader for testing dataset.
            - meta (dict): Metadata dictionary containing:
                - "ndims" (int): Number of input dimensions (features).
                - "num_classes" (int): Number of output classes for classification.
                - "sequencelength" (int): Length of time series sequences.
    """

    print(f"Setting up datasets in {os.path.abspath(datapath)}")
    datapath = os.path.abspath(datapath)

    # Obtain the transformation function for time series processing
    transform = get_transform_CatCrops(sequencelength, datecrop, use_previous_year_TS, sparse, cp, doa, noreplace)

    # Select the dataset based on the mode
    if mode == "evaluation1":
        lleida22 = CatCrops(region="lleida", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        baixter22 = CatCrops(region="baixter", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        lleida23 = CatCrops(region="lleida", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        baixter23 = CatCrops(region="baixter", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        traindataset = data.ConcatDataset([lleida22, baixter22])
        testdataset = data.ConcatDataset([lleida23, baixter23])
    elif mode == "evaluation2":
        lleida22 = CatCrops(region="lleida", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        baixter22 = CatCrops(region="baixter", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        lleida23 = CatCrops(region="lleida", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        baixter23 = CatCrops(region="baixter", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, L2A = L2A, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        traindataset = data.ConcatDataset([lleida23, baixter23])
        testdataset = data.ConcatDataset([lleida22, baixter22])
    else:
        raise ValueError("only --mode 'evaluation1' or 'evaluation2' allowed")

    # Create dataloaders
    if weight_sampling:
        concatenated_df = pd.concat([df.index['classid'] for df in traindataset.datasets], ignore_index=True)
        weights_per_class = 1/np.bincount(concatenated_df)
        samples_weight = np.array([weights_per_class[t] for t in concatenated_df])
        WeightedRandomSampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True)
        traindataloader = DataLoader(traindataset, batch_size=batchsize, shuffle=False, num_workers=workers, sampler = WeightedRandomSampler)
    else:
        traindataloader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers)

    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)

    # Store metadata
    meta = {
        "ndims": baixter22[0][0].shape[1],  # Number of input dimensions
        "num_classes": len(baixter22.classes),  # Number of classes
        "sequencelength": sequencelength  # Length of the time series
    }

    return traindataloader, testdataloader, meta


def get_model(modelname, ndims, num_classes, sequencelength, device, **hyperparameter):
    """
    Retrieves the specified deep learning model.

    Args:
        modelname (str): Name of the model.
        ndims (int): Number of input dimensions.
        num_classes (int): Number of output classes.
        sequencelength (int): Length of the time series sequence.
        device (torch.device): Device for computation (CPU or GPU).
        **hyperparameter: Additional hyperparameters for the model.

    Returns:
        model: The initialized deep learning model.
    """
    modelname = modelname.lower()  # Convert model name to lowercase

    # Select the model based on the provided name
    if modelname == "omniscalecnn":
        model = OmniScaleCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength,
                             **hyperparameter).to(device)
    elif modelname == "lstm":
        model = LSTM(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname == "starrnn":
        model = StarRNN(input_dim=ndims,
                        num_classes=num_classes,
                        bidirectional=False,
                        use_batchnorm=False,
                        use_layernorm=True,
                        device=device,
                        **hyperparameter).to(device)
    elif modelname == "msresnet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname in ["transformerencoder", "transformer"]:
        model = TransformerModel(input_dim=ndims, num_classes=num_classes,
                                 activation="relu",
                                 **hyperparameter).to(device)
    elif modelname == "tempcnn":
        model = TempCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(
            device)
    else:
        raise ValueError("invalid model argument. choose from 'TransformerEncoder', 'LSTM', 'MSResNet', 'StarRNN', "
                         " 'OmniScaleCNN', or 'TempCNN'")


    return model



def metrics(y_true, y_pred):
    """
    Computes various performance metrics given true and predicted class labels.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.

    Returns:
        dict: A dictionary containing performance metrics, including:
            - "accuracy" (float): Overall accuracy of predictions.
            - "kappa" (float): Cohen's kappa statistic measuring agreement.
            - "f1_micro" (float): Micro-averaged F1-score.
            - "f1_macro" (float): Macro-averaged F1-score.
            - "f1_weighted" (float): Weighted-averaged F1-score.
            - "recall_micro" (float): Micro-averaged recall score.
            - "recall_macro" (float): Macro-averaged recall score.
            - "recall_weighted" (float): Weighted-averaged recall score.
            - "precision_micro" (float): Micro-averaged precision score.
            - "precision_macro" (float): Macro-averaged precision score.
            - "precision_weighted" (float): Weighted-averaged precision score.

    Raises:
        ValueError: If the dimensions of `y_true` and `y_pred` do not match.

    """
    # Check that the class label dimensions match
    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch in dimensions: 'y_true' and 'y_pred' must have the same length.")

    # Accuracy: measures the proportion of correctly predicted class labels
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    # Cohen's kappa: measures the degree of agreement between true and predicted class labels
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)

    # F1-score (micro, macro, weighted): combines precision and recall for each class
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")

    # Recall (micro, macro, weighted): measures the proportion of actual class labels correctly predicted
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")

    # Precision (micro, macro, weighted): measures the proportion of predicted class labels that are correct
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    # Return the performance metrics in a dictionary
    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
    }


def train_epoch(model, optimizer, criterion, dataloader, device):
    """
    Trains the model for one epoch (one full pass through the training dataset batches).

    Args:
        model (torch.nn.Module): Neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to adjust the model weights.
        criterion: Loss function used to compute the error in the model’s prediction.
        dataloader (torch.utils.data.DataLoader): DataLoader to obtain batches from the training dataset.
        device (torch.device): Device on which the computation will run (CPU or GPU).

    Returns:
        torch.Tensor: Tensor containing the computed losses for each batch during the training epoch.

    Raises:
        TypeError: If the parameter types are not as expected.
    """
    # Set the model to training mode
    model.train()
    # List to store the losses for each batch
    losses = []
    # Create an iterator with a progress bar for the dataloader
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        # Iterate over the batches
        for idx, batch in iterator:
            # Reset the optimizer gradients
            optimizer.zero_grad()
            # Separate the input data (x) and the expected output labels (y_true)
            x, y_true, _ = batch
            # Move the data to the specified device (CPU or GPU)
            x = x.to(device)
            y_true = y_true.to(device)
            # Compute the loss using the input data and expected labels
            loss = criterion(model.forward(x), y_true)
            # Backpropagate the loss gradient to update model parameters
            loss.backward()
            # Update the model weights using the optimizer
            optimizer.step()
            # Update the progress bar description with the current loss
            iterator.set_description(f"train loss={loss:.2f}")
            # Store the current batch loss in the losses list
            losses.append(loss)
    # Convert the loss list into a tensor
    return torch.stack(losses)


def test_epoch(model, criterion, dataloader, device):
    """
    Evaluates the model on a dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion: The loss function used to compute the loss.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the evaluation data.
        device (str or torch.device): The device on which the computation runs ('cpu' or 'cuda').

    Returns:
        Tuple: A tuple containing stacked losses, true labels, predictions, score values,
        and field identifiers, all converted into tensors.

    """
    # Set the model to evaluation mode
    model.eval()

    losses = []
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    field_ids_list = []

    with torch.no_grad():
        # Iterate over the dataloader with a progress bar
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                # Unpack the batch data
                x, y_true, field_id = batch

                # Move input data and labels to the appropriate device
                x = x.to(device)
                y_true = y_true.to(device)

                # Forward pass: compute log probabilities
                logprobabilities = model.forward(x)

                # Compute loss by comparing log probabilities with true labels
                loss = criterion(logprobabilities, y_true)

                # Update the progress bar description with the current loss value
                iterator.set_description(f"test loss={loss:.2f}")

                # Store the losses, true labels, predictions, score values,
                # and field identifiers in their respective lists
                losses.append(loss.item())
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(dim=1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)

    # Return stacked tensors of losses, true labels, predictions, score values, and field identifiers
    return (
        torch.tensor(losses),
        torch.cat(y_true_list),
        torch.cat(y_pred_list),
        torch.cat(y_score_list),
        torch.cat(field_ids_list)
    )


def save(model, path="model.pth"):
    """
    Saves the model to the specified path.

    Args:
        model (torch.nn.Module): The model to be saved.
        path (str): The file path where the model should be saved.

    Returns:
        None

    """

    print("\nSaving model to " + path)
    model_state = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model_state), path)


def parse_args():
    """
    Parses command-line arguments for training and evaluating deep learning models on time series data.

    This function processes user-defined arguments such as model architecture, dataset path, training parameters,
    and additional features related to input data.

    Returns:
        argparse.Namespace: An object containing all parsed arguments.

    Raises:
        ValueError: If an invalid hyperparameter format is provided.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate deep learning models for time series on the CatCrops dataset. "
            "This script trains a model using the training dataset partition, "
            "evaluates its performance on a validation or evaluation partition, "
            "and stores training progress and model checkpoints in the directory specified by --logdir."
        )
    )

    # Model selection and training configuration
    parser.add_argument('--model', type=str, default="TransformerEncoder",
                        help='Select model architecture. Available models: "TransformerEncoder", "LSTM", '
                             '"TempCNN", "MSRestNet", "StarRNN", "OmniScaleCNN"')
    parser.add_argument('-SL', '--sequencelength', type=int, default=70,
                        help='Length of the input time series sequence.')
    parser.add_argument('--datecrop', type=str, default="random",
                        help='Latest date to crop the time series data, can be fixed or "random" (format=DD/MM/YYYY).')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='Batch size (number of time series processed simultaneously).')
    parser.add_argument('-e', '--epochs', type=int, default=120,
                        help='Number of training epochs (one full pass through the dataset).')
    parser.add_argument('-m', '--mode', type=str, default="evaluation1",
                        help='Training mode. "evaluation1" trains on (Lleida+Baixter) 2022 data and validates on 2023 data, while evaluation2 does the reverse (trains on 2023 and tests on 2022).')

    # Dataset and hardware settings
    parser.add_argument('-D', '--datapath', type=str, default="catcrops_dataset",
                        help='Path to the directory containing the dataset.')
    parser.add_argument('-w', '--workers', type=int, default=0,
                        help='Number of CPU workers for loading the next batch.')
    parser.add_argument('-H', '--hyperparameter', type=str, default=None,
                        help='Model-specific hyperparameters as a comma-separated string in the format "param1=value1,param2=value2".')

    # Optimizer settings
    parser.add_argument('--weight-decay', type=float, default=5e-08,
                        help='Weight decay regularization for the optimizer (default: 5e-08).')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer (default: 1e-3).')

    # Memory and computational settings
    parser.add_argument('--preload-ram', action='store_true',
                        help='Load the dataset into RAM upon initialization to speed up training.')
    parser.add_argument('--wight_sampling', action='store_true',
                        help='Use weighted sampling to balance the dataset.')

    # Device settings
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Computation device: "cpu" or "cuda". Default is automatically detected.')

    # Logging and trial settings
    parser.add_argument('-l', '--logdir', type=str, default="./Results",
                        help='Directory to store logs, progress, and trained models.')
    parser.add_argument('--trial', type=str, default="Trial001",
                        help='Name of the trial to distinguish experiments.')

    # Input data processing options
    parser.add_argument('--use_previous_year_TS', action="store_true",
                        help='Use time series data from the previous year when cropping to a specific date.')
    parser.add_argument('--sparse', action="store_true",
                        help='Fill missing dates with zero values in the time series.')
    parser.add_argument('--noreplace', action="store_true",
                        help='Fill sequences with zeros if their length is shorter than "sequencelength".')
    parser.add_argument('--cp', action="store_true",
                        help='Include cloud percentage information in the time series.')
    parser.add_argument('--doa', action="store_true",
                        help='Include the day of the year as a feature in the time series.')

    # Feature selection flags
    parser.add_argument('--L2A', action="store_true", help='Use spectral data from Sentinel-2 L2A level.')
    parser.add_argument('--pclassid', action="store_true", help='Use the previous year’s classification as input.')
    parser.add_argument('--pcrop', action="store_true", help='Use the previous year’s crop code as input.')
    parser.add_argument('--pvar', action="store_true", help='Use the previous year’s variety code as input.')
    parser.add_argument('--sreg', action="store_true", help='Use the previous year’s irrigation system information.')
    parser.add_argument('--mun', action="store_true", help='Include the municipality code as an input feature.')
    parser.add_argument('--com', action="store_true", help='Include the comarca (region) code as an input feature.')
    parser.add_argument('--prov', action="store_true", help='Include the province code as an input feature.')
    parser.add_argument('--elev', action="store_true", help='Include the elevation of the field as an input feature.')
    parser.add_argument('--slope', action="store_true", help='Include the slope of the field as an input feature.')

    args = parser.parse_args()

    # Convert hyperparameter string to dictionary
    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict

    # Set device automatically if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def get_default_parse_arguments():
    """
    Returns a default argument configuration for training and evaluation.

    This function provides a predefined set of parameters for training deep learning models
    without requiring manual input from the user.

    Returns:
        parsed_arguments: An object containing default arguments for training.

    """

    class parsed_arguments():
        def __init__(self):
            self.prova = None

    args = parsed_arguments()

    # ---------------------------------
    # Model and Dataset Settings
    # ---------------------------------
    args.model = "TransformerEncoder"  # Model architecture to use for training.
    args.sequencelength = 70  # Length of the time series sequence (number of time steps).
    args.datecrop = "random"  # The latest date to crop time series (can be fixed or "random").
    args.batchsize = 512  # Number of time series processed simultaneously in one batch.
    args.epochs = 120  # Number of training epochs (each epoch is a full pass through the dataset).
    args.mode = "evaluation1"  # Training mode (determines how training and validation data are split).
    args.datapath = "catcrops_dataset"  # Path where the dataset is stored.
    args.workers = 0  # Number of CPU workers used to load the data.

    # ---------------------------------
    # Model Hyperparameters
    # ---------------------------------
    args.hyperparameter = dict()  # Dictionary of model-specific hyperparameters.
    args.weight_decay = 5e-08  # Regularization parameter to prevent overfitting.
    args.learning_rate = 1e-3  # Learning rate for the optimizer.

    # ---------------------------------
    # Memory and Computational Settings
    # ---------------------------------
    args.preload_ram = False  # If True, load the dataset into RAM for faster training.
    args.wight_sampling = False  # If True, use weighted sampling to balance dataset classes.
    args.device = None  # Computational device: "cpu" or "cuda" (auto-detected if None).
    args.logdir = "./RESULTS"  # Directory to store logs, model checkpoints, and training progress.

    # ---------------------------------
    # Feature Selection Options
    # ---------------------------------
    args.use_previous_year_TS = False  # Use previous year's time series data.
    args.sparse = False  # Fill missing dates in the time series with zero values.
    args.noreplace = False  # If sequence length is shorter than expected, fill with zeros.
    args.cp = False  # Include cloud percentage as an additional input feature.
    args.doa = False  # Include the day of the year as an additional input feature.
    args.L2A = True  # Use Sentinel-2 L2A spectral data.
    args.pclassid = False  # Use previous year's crop classification as an input feature.
    args.pcrop = False  # Use previous year's crop code as an input feature.
    args.pvar = False  # Use previous year's variety code as an input feature.
    args.sreg = False  # Include previous year's irrigation system information.
    args.mun = False  # Include municipality code as an input feature.
    args.com = False  # Include comarca (region) code as an input feature.
    args.prov = False  # Include province code as an input feature.
    args.elev = False  # Include elevation of the field as an input feature.
    args.slope = False  # Include slope of the field as an input feature.

    # ---------------------------------
    # Trial Settings
    # ---------------------------------
    args.trial = "Trial001"  # Identifier for the current experiment.

    # Automatically detect computation device if not set
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()
    # args = get_default_parse_arguments()
    train(args)

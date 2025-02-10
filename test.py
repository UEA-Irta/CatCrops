# -*- coding: utf-8 -*-
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
list_models_n = ['TransformerEncoder', 'LSTM', 'StarRNN', 'MSResNet', 'OmniScaleCNN', 'TempCNN', 'PeTransformerEncoder']

def get_dataloader_test(datapath, mode, batchsize, workers, preload_ram=False,
                   sequencelength=45, datecrop="26/05/2023", use_previous_year_TS=False ,sparse=False, cp=False, doa=False, L2A = True, LST = True, ET=False, pclassid=False, pcrop=False,
                   pvar=False, sreg=False, mun=False, com=False, prov=False, elev=False, slope=False, noreplace=False,recompile_h5_from_csv=False):
    datapath = pth.abspath(datapath)
    transform = get_transform_CatCrops(sequencelength, datecrop, use_previous_year_TS ,sparse, cp, doa, noreplace)
    if mode == 'test2021':
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2021, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2021, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    elif mode == 'test2022':
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2022, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    elif mode == "test2023":
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2023, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    elif mode == "test2024":
        dst_lleida = CatCrops(region="lleida", root=datapath, year=2024, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        dst_baixter = CatCrops(region="baixter", root=datapath, year=2024, preload_ram=preload_ram, transform=transform, recompile_h5_from_csv=recompile_h5_from_csv, L2A = L2A, LST = LST, ET=ET, pclassid=pclassid, pcrop=pcrop, pvar=pvar, sreg=sreg, mun=mun, com=com, prov=prov, elev=elev, slope=slope)
        testdataset = torch.utils.data.ConcatDataset([dst_lleida, dst_baixter])
    else:
        raise ValueError("only --mode 'test2021' 'test2022' 'test2023' or 'test2024' allowed")

    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)
    testdataloader_ll = DataLoader(dst_lleida, batch_size=batchsize, shuffle=False, num_workers=workers)
    testdataloader_bt = DataLoader(dst_baixter, batch_size=batchsize, shuffle=False, num_workers=workers)
    meta = {
        "ndims": dst_baixter[0][0].shape[1],  # Nombre de dimensions de les imatges
        "num_classes": len(dst_baixter.classes),  # Nombre de classes
        "sequencelength": sequencelength  # Longitud de la seqüència d'imatges
    }

    return {"t": testdataloader, "ll": testdataloader_ll, "bt": testdataloader_bt}, meta


def do_cm_plot(dfcf, title_n, out_fol, mdesc, test_z='', n_dec="2"):
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

    :param list_clasn list[str]: Llista amb els noms de les classes abreviats
    :type c_matrix np.array: numpy array amb la matriu de confució
    """
    return pd.DataFrame(c_matrix, index=[i for i in list_clasn], columns=[i for i in list_clasn])


def get_matrix_conf(y_t, y_p, list_clasn, s_f, rn, t_m, zn='t'):
    """

    :param y_t: Vector amb els valors True
    :param y_p: Vector amb els valors de Predicció
    :param list_clasn list[str]: Llista amb els noms de les classes abreviats
    :param s_f str: Carpeta de sortida
    :param rn str: run, nom del model
    :param t_m str: Any del test
    :param zn str: Zona per afegir al nom de l'arxiu (si és per totes les zones no cal ficar-hi res)
    """

    # Passo el tesnors a cpu
    y_t = y_t.cpu()
    y_p = y_p.cpu()


    if zn == 'll' and 11 not in np.unique(y_p):
        list_clasn = list_clasn.copy()
        list_clasn.remove('AR')

    # Creo les matrius
    cf_matrix_v = sklearn.metrics.confusion_matrix(y_t, y_p)
    # df_cm_v = pd.DataFrame(cf_matrix_v, index=[i for i in list_clasn], columns=[i for i in list_clasn])
    df_cm_v = get_df_cmatrix(cf_matrix_v, list_clasn)
    df_cm_v.to_csv(pth.join(s_f, "%s_cf_val.csv" % zn))

    # Normalitzada pels true
    cf_matrix_t = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='true')
    # df_cm_t = pd.DataFrame(cf_matrix_t, index=[i for i in list_clasn],
    #                        columns=[i for i in list_clasn])
    df_cm_t = get_df_cmatrix(cf_matrix_t, list_clasn)
    df_cm_t.to_csv(pth.join(s_f, "%s_cf_t.csv" % zn))

    # Normalitzada per la predicció
    cf_matrix_p = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='pred')
    # df_cm_p = pd.DataFrame(cf_matrix_p, index=[i for i in list_clasn],
    #                        columns=[i for i in list_clasn])
    df_cm_p = get_df_cmatrix(cf_matrix_p, list_clasn)
    df_cm_p.to_csv(pth.join(s_f, "%s_cf_p.csv" % zn))

    # Normalitzada per tots
    cf_matrix_all = sklearn.metrics.confusion_matrix(y_t, y_p, normalize='all')
    # df_cm_all = pd.DataFrame(cf_matrix_all, index=[i for i in list_clasn],
    #                          columns=[i for i in list_clasn])
    df_cm_all = get_df_cmatrix(cf_matrix_all, list_clasn)
    df_cm_all.to_csv(pth.join(s_f, "%s_cf_all.csv" % zn))

    # Fem els gràfics de les matrius
    do_cm_plot(df_cm_v, "%s_cf values" % zn, s_f, rn, test_z=t_m, n_dec="0")
    do_cm_plot(df_cm_t, "%s_cf true" % zn, s_f, rn, test_z=t_m)
    do_cm_plot(df_cm_p, "%s_cf prediction" % zn, s_f, rn, test_z=t_m)
    do_cm_plot(df_cm_all, "%s_cf all" % zn, s_f, rn, test_z=t_m)

# Function to generate a list of dates
def generate_dates(start_date_str, end_date_str, skip_days):
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y")

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime("%d/%m/%Y"))
        current_date += timedelta(days=skip_days)

    return date_list

def parse_args():
    parser = argparse.ArgumentParser(description='Test and evaluate time series deep learning models on the'
                                                 'CatCrops dataset. This script test a model on test dataset'
                                                 'partition and evaluates performance ')
    parser.add_argument('--model', type=str, default="TransformerEncoder", help='select model architecture. Available models are "LSTM","TempCNN","MSRestNet","TransformerEncoder"')
    parser.add_argument('-SL', '--sequencelength', type=int, default=None, help='Sequence Length')
    parser.add_argument('--datecrop', type=str, default="30/12/2023", help='Maximum date of the sequence')
    parser.add_argument('-b', '--batchsize', type=int, default=512, help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-e', '--epoch', type=int, default=None, help='number of training epochs (training on entire dataset)')
    parser.add_argument('-m', '--mode', type=str, default="test2023", help='training mode. evaluation1 trains on (Lleida+Baixter) 2022 data and validates on (Lleida+Baixter) 2023 data')
    parser.add_argument('-D', '--datapath', type=str, default="/media/hdd11/tipus_c/catcrops_dataset_v2/", help='directory to download and store the dataset')
    parser.add_argument('-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument('-H', '--hyperparameter', type=str, default=None, help='model specific hyperparameter as single string, separated by comma of format param1=value1,param2=value2')
    parser.add_argument('--weight-decay', type=float, default=5e-08, help='optimizer weight_decay (default 1e-5)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--preload-ram', action='store_true', help='load dataset into RAM upon initialization')
    parser.add_argument('-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('-l', '--logdir', type=str, default="./Results", help='logdir to store progress and models (defaults to /tmp)')
    parser.add_argument('--use_previous_year_TS', action="store_true", help='A bolean argument to use the time series from the previous year in case of cropping the timeseries to a specific date (Default is false)')
    parser.add_argument('--sparse', action="store_true", help='A bolean argument to fill the dates with no data with 0 values (Default is false)')
    parser.add_argument('--noreplace', action="store_true", help='A bolean argument to fill with zeros in the case that the sequence length is lower than "sequencelength" value')
    parser.add_argument('--cp', action="store_true", help = 'A bolean argument to use percentage of clouds in the timeseries (Default is false)')
    parser.add_argument('--doa', action="store_true", help = 'A bolean argument to put the day of the year in the timeseries (Default is false)')
    parser.add_argument('--L2A', action="store_true", help = 'A bolean argument to use the spectral data from L2A level in the timeseries (Default is false)')
    parser.add_argument('--LST', action="store_true", help='A bolean argument to use the spectral data from Landsat in the timeseries (Default is false)')
    parser.add_argument('--ET', action="store_true", help = 'A bolean argument to use the evapotranspiration (ET) timeseries (Default is false)')
    parser.add_argument('--pclassid', action="store_true", help = 'A bolean argument use the previous year class as input (Default is false)')
    parser.add_argument('--pcrop', action="store_true", help = 'A bolean argument to use the previous year crop code as input (Default is false)')
    parser.add_argument('--pvar', action="store_true", help = 'A bolean argument to use the previous year variety code as input  (Default is false)')
    parser.add_argument('--sreg', action="store_true", help = 'A bolean argument to use the previous year irrigation system as input  (Default is false)')
    parser.add_argument('--mun', action="store_true", help = 'A bolean argument to use the municipality code as input  (Default is false)')
    parser.add_argument('--com', action="store_true", help = 'A bolean argument to use the comarca code as input  (Default is false)')
    parser.add_argument('--prov', action="store_true", help = 'A bolean argument to use the province code as input  (Default is false)')
    parser.add_argument('--elev', action="store_true", help = 'A bolean argument to use the elevation of the field as input  (Default is false)')
    parser.add_argument('--slope', action="store_true", help = 'A bolean argument to use the slope of the field as input  (Default is false)')
    parser.add_argument('--trial', type=str, default="Trial001", help='trial_name')
    parser.add_argument('--stat_n', type=str, default="accuracy", help='Metric used to select the best epoch')
    parser.add_argument('--do_shp', action="store_true", help='To save a shp output file ')
    parser.add_argument('--recompile_h5_from_csv', action="store_true", help='To recompile h5')

    args = parser.parse_args()

    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def get_default_parse_arguments():
    class parsed_arguments():
        def __init__(self):
            self.prova = None

    args = parsed_arguments()
    args.model = "TransformerEncoder"  # Select model architecture. Available models are "LSTM","TempCNN","MSRestNet","TransformerEncoder"
    args.batchsize = 1 # 512  # number of time series processed simultaneously
    args.epoch = None  # Epoch to take the corresponding model (early stopping) Put the Epoch number or None to choose the epoch with better accuracy.
    args.hyperparameter = dict()  # Model specific hyperparameter as single string, separated by comma of format param1=value1,param2=value2
    args.datapath = "/media/hdd11/tipus_c/catcrops_dataset_v2/"  # Directory to download and store the dataset'
    args.workers = 0  # Number of CPU workers to load the next batch
    args.weight_decay = 5e-08  # Optimizer weight_decay (default 1e-6)
    args.learning_rate = 1e-3  # Optimizer learning rate (default 1e-2)
    args.preload_ram = False # True  # Load dataset into RAM upon initialization
    args.device = None  # torch.Device. either "cpu" or "cuda". Default will check by torch.cuda.is_available()
    args.logdir = "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS"  # Logdir to store progress and models (defaults to /tmp)'
    # args.mode = "test2024"  # Training mode. Either "validation" (train on baixter2022 test on lleida2022) or "evaluation" (train on lleida and baixter 2021-2122 test on lleida and baixter 2023)')
    # args.datecrop = "14/03/2024" # Date in the form of "DD/MM/YYYY" or "all" to evaluate all dates/
    # args.sequencelength = None
    # args.use_previous_year_TS = True
    # args.sparse = True
    # args.cp = True
    # args.doa = True
    # args.L2A = True
    # args.LST = False
    # args.ET = False
    # args.pclassid = True
    # args.pcrop = True
    # args.pvar = True
    # args.sreg = True
    # args.mun = True
    # args.com = True
    # args.prov = True
    # args.elev = True
    # args.slope = True
    # args.trial = "Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"

    args.mode = "test2023"  # Training mode. Either "validation" (train on baixter2022 test on lleida2022) or "evaluation" (train on lleida and baixter 2021-2122 test on lleida and baixter 2023)')
    args.datecrop = "01/06/2023"  # Date in the form of "DD/MM/YYYY" or "all" to evaluate all dates/
    args.sequencelength = None
    args.use_previous_year_TS = False
    args.sparse = True
    args.noreplace = False
    args.cp = True
    args.doa = True
    args.L2A = True
    args.LST = False
    args.ET = False
    args.pclassid = False
    args.pcrop = False
    args.pvar = False
    args.sreg = False
    args.mun = False
    args.com = False
    args.prov = False
    args.elev = False
    args.slope = False
    args.trial = "Trial002-Crop231231_SL_L2A"
    args.recompile_h5_from_csv = False

    args.stat_n = "accuracy"  # Metric used to select the best epoch
    args.do_shp = True  # To save a shp output file
    args.do_csv = False  # To save a csv output file
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":

    #args = get_default_parse_arguments()
    args = parse_args()

    logdir = args.logdir
    device = args.device
    out_f = logdir.replace('/RESULTS', '/TEST')

    run = args.trial


    if args.epoch is None:
        # Busquem el model amb el valor més elevat de la mètrica seleccionada
        best_epoch_stats = read_best_epoch_stats(pth.join(logdir, run), stat_name=args.stat_n)
        best_epoch = best_epoch_stats['epoch']
    else:
        best_epoch = args.epoch

    list_pth = [pth.basename(x) for x in glob.glob(pth.join(logdir, run, "*.pth"))]
    path_model = [x for x in list_pth if "model_%d.pth" % int(best_epoch) == x][0]

    if args.datecrop == "all":
        start_date_str = "07/01/"+args.mode[-4:]
        end_date_str = "31/12/"+args.mode[-4:]
        skip_days = 7
        dates2crop = generate_dates(start_date_str, end_date_str, skip_days)
    else:
        dates2crop = [args.datecrop]

    for datecrop in dates2crop:
        print("Processing date:", datecrop)
        datecrop_object = datetime.strptime(datecrop, "%d/%m/%Y")
        reversed_datecrop_string = datecrop_object.strftime("%y%m%d")

        save_f = os.path.join(out_f, run, "%s_%s_%s" % (args.mode, reversed_datecrop_string, pth.basename(path_model).split('.')[0]))
        os.makedirs(save_f, exist_ok=True)
        start_dataload = time.time()
        testdataloader_dict, meta = get_dataloader_test(args.datapath, args.mode, args.batchsize, args.workers,
                                                                   args.preload_ram, args.sequencelength, datecrop,
                                                                   args.use_previous_year_TS, args.sparse, args.cp, args.doa,
                                                                   args.L2A, args.LST, args.ET, args.pclassid, args.pcrop, args.pvar,
                                                                   args.sreg, args.mun, args.com, args.prov, args.elev, args.slope, args.noreplace, args.recompile_h5_from_csv)
        end_dataload = time.time()
        print(f"Processing time for Dataload: {end_dataload-start_dataload:.4f} seconds")
        num_classes = meta["num_classes"]
        ndims = meta["ndims"]
        sequencelength = meta["sequencelength"]
        model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)


        model.modelname += f"_lr={args.learning_rate}_wd={args.weight_decay}" \
                           f"_sl={args.sequencelength}" \
                           f"_dc={''.join(datecrop.split('/'))}_mode={args.mode}"

        print(f"Initialized {model.modelname}")

        # Carreguem el model
        model.load_state_dict(torch.load(pth.join(logdir, run, path_model), map_location=device)["model_state"])

        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        end_model_load = time.time()
        print(f"Processing time for Model load: {end_model_load-end_dataload:.4f} seconds")

        # Guardem una llista amb el scores

        list_scores = []
        # Itinerem per cada dataloader i fem el test
        for zona_n, testdl in testdataloader_dict.items():
            # Testem el model
            start_inference = time.time()
            test_loss, y_true, y_pred, y_score, field_ids = test_epoch(model, criterion, testdl, device)
            end_inference = time.time()
            print("Processing time for " + zona_n + f" load: {end_inference - start_inference:.4f} seconds")
            print(f"Logging results to {save_f}")

            # Obtenim els valors de les mètriques
            scores_df = save_metrics(y_t=y_true, y_p=y_pred, y_s=y_score, f_id=field_ids, test_l=test_loss,
                                     list_clasn=l_classname, s_f=save_f, zn=zona_n)

            # L'afegim a la llist per despres fer un df de totes les zones
            list_scores.append(scores_df)

            # Creem les matrius de confunció i les guardem
            get_matrix_conf(y_t=y_true, y_p=y_pred, list_clasn=l_classname, s_f=save_f, rn=run, t_m=args.mode, zn=zona_n)

            # Si volem ens guarda l'arxiu csv
            if args.do_shp and zona_n != 't':
                if zona_n == 'll':
                    regio = 'lleida'
                else:
                    regio = "baixter"
                ds = CatCrops(region=regio, root=args.datapath, year=int(args.mode[-4:]), preload_ram=False, L2A = args.L2A, LST = args.LST, ET=args.ET, pclassid=args.pclassid, pcrop=args.pcrop, pvar=args.pvar, sreg=args.sreg, mun=args.mun, com=args.com, prov=args.prov, elev=args.elev, slope=args.slope)
                gdf = ds.geodataframe()
                max_scores, _ = torch.max(y_score, dim=1)
                dict_pre = {'field_ids': field_ids.tolist(), 'pred_classid': y_pred.tolist(), 'pred_score': max_scores.tolist()}

                df_p = pd.DataFrame.from_dict(dict_pre)
                gdf_p = gdf.join(df_p.set_index('field_ids'), on='id')
                gdf_p['prediccio'] = gdf_p['classid'] == gdf_p['pred_classid']
                dict_class = dict(zip(ds.classes, ds.classname))
                gdf_p['pred_classn'] = gdf_p['pred_classid'].map(dict_class)

                shp_folder = pth.join(save_f, 'shp')
                os.makedirs(shp_folder, exist_ok=True)
                gdf_p.to_file(pth.join(shp_folder, '%s_pred.shp' % zona_n))

            # if args.do_csv and zona_n != 't':
            #
            #
            #     dict_class = dict(zip(testdl.dataset.classes, testdl.dataset.classname))
            #
            #     pclass_id = []
            #     for field_id in field_ids.tolist():
            #         pclass_id.append(testdl.dataset.mapping.loc[testdl.dataset.index[testdl.dataset.index['id']==field_id]['pcrop_code'].values]['id'].values)
            #
            #
            #
            #     dict_pre = {'field_ids': field_ids.tolist(), 'true_classid': y_true.tolist(), 'pred_classid': y_pred.tolist()}




        scores_df = pd.concat(list_scores)
        scores_df.to_csv(pth.join(save_f, "scorres_allz.csv"), index=False)

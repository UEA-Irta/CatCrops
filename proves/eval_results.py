import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import sklearn.metrics


def list_folders(directory):
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    return folders

def show_validation_acc_kappa_1_method(results_directory):
    #results_directory = "/media/hdd11/tipus_c/proves_article/P03-SL_doa_datecrop_cp/RESULTS"
    results_folders = sorted(list_folders(results_directory))
    columns_results_data = ['trial', 'date', 'epoch', 'accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 'trainloss', 'testloss']
    results_df = pd.DataFrame(columns=columns_results_data)
    for trial_name in results_folders:
        trainlog_file = os.path.join(results_directory, trial_name,"trainlog.csv")
        if os.path.isfile(trainlog_file):
            # print("Reading results from "+trial_name)
            trainlog_df = pd.read_csv(trainlog_file)
            min_testloss_idx = trainlog_df["testloss"].idxmin()
            date = datetime.strptime(trial_name[-6:], "%y%m%d")
            new_row = {'trial': trial_name,
                       'date': date,
                       'epoch': trainlog_df.iloc[min_testloss_idx]["epoch"],
                       'accuracy': trainlog_df.iloc[min_testloss_idx]["accuracy"],
                       'kappa': trainlog_df.iloc[min_testloss_idx]["kappa"],
                       'f1_micro': trainlog_df.iloc[min_testloss_idx]["f1_micro"],
                       'f1_macro': trainlog_df.iloc[min_testloss_idx]["f1_macro"],
                       'f1_weighted': trainlog_df.iloc[min_testloss_idx]["f1_weighted"],
                       'trainloss': trainlog_df.iloc[min_testloss_idx]["trainloss"],
                       'testloss': trainlog_df.iloc[min_testloss_idx]["testloss"]}
            new_row_df = pd.DataFrame([new_row])
            results_df = pd.concat([results_df, new_row_df] , ignore_index=True)
        else:
            print("The trial " + trial_name + " has no results.")

    results_df.to_csv(os.path.dirname(results_directory)+"/results_timeserie.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['date'], results_df['accuracy'], marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.savefig(os.path.dirname(results_directory)+"/accuracy_plot.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['date'], results_df['kappa'], marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Kappa')
    plt.title('Kappa Over Time')
    plt.savefig(os.path.dirname(results_directory)+"/kappa_plot.png")
    plt.show()

def show_validation_acc_kappa_all_methods(results_directory):
    #results_directory = "/media/hdd11/tipus_c/proves_article"
    results_folders = sorted(list_folders(results_directory))
    fig_acc, ax_acc = plt.subplots()
    for method_name in results_folders:
        trainlog_file = os.path.join(results_directory, method_name, 'results_timeserie.csv')
        if os.path.isfile(trainlog_file):
            results_df = pd.read_csv(trainlog_file)
            ax_acc.plot(results_df['date'], results_df['accuracy'], marker='o', linestyle='-', label=method_name)
    ax_acc.set_xlabel('Date')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Accuracy Over Time')
    ax_acc.legend()
    ax_acc.set_xticks(results_df.index[::3])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    fig_kp, ax_kp = plt.subplots()
    for method_name in results_folders:
        trainlog_file = os.path.join(results_directory, method_name, 'results_timeserie.csv')
        if os.path.isfile(trainlog_file):
            results_df = pd.read_csv(trainlog_file)
            ax_kp.plot(results_df['date'], results_df['kappa'], marker='o', linestyle='-', label=method_name)
    ax_kp.set_xlabel('Date')
    ax_kp.set_ylabel('Kappa')
    ax_kp.set_title('Kappa Over Time')
    ax_kp.legend()
    ax_kp.set_xticks(results_df.index[::3])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.xticks(rotation=45)
    plt.tight_layout()



def show_test_acc_kappa_1_method(results_directory):
    #results_directory = "/media/hdd11/tipus_c/proves_article/P05-CatCrops/TEST/Trial013-rCrop_py_s_cp_doa_L2A_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
    results_folders = sorted(list_folders(results_directory))
    columns_results_data = ['trial', 'date', 'epoch', 'accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 'trainloss']
    regions = ["ll", "bt", "t"]
    for region in regions:
        results_df = pd.DataFrame(columns=columns_results_data)
        for trial_name in results_folders:
            splitted_trial_name = trial_name.split("_")
            if len(splitted_trial_name)!=4:
                continue
            scorres_file = os.path.join(results_directory, trial_name,"scorres_allz.csv")
            if os.path.isfile(scorres_file):
                # print("Reading results from "+trial_name)
                scorres_df = pd.read_csv(scorres_file)
                scorres_df = scorres_df[scorres_df['zona']==region]
                date = datetime.strptime(splitted_trial_name[1], "%y%m%d")
                new_row = {'trial': trial_name,
                           'date': date,
                           'epoch': splitted_trial_name[3],
                           'accuracy': scorres_df.iloc[0]["accuracy"],
                           'kappa': scorres_df.iloc[0]["kappa"],
                           'f1_micro': scorres_df.iloc[0]["f1_micro"],
                           'f1_macro': scorres_df.iloc[0]["f1_macro"],
                           'f1_weighted': scorres_df.iloc[0]["f1_weighted"],
                           'testloss': scorres_df.iloc[0]["testloss"]}
                new_row_df = pd.DataFrame([new_row])
                results_df = pd.concat([results_df, new_row_df] , ignore_index=True)
            else:
                print("The trial " + trial_name + " has no results.")

        results_df.to_csv(results_directory+"/results_timeserie_"+region+".csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['date'], results_df['accuracy'], marker='o', linestyle='-', color='b')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Time')
        plt.savefig(results_directory+"/accuracy_plot_"+region+".png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['date'], results_df['kappa'], marker='o', linestyle='-', color='b')
        plt.xlabel('Date')
        plt.ylabel('Kappa')
        plt.title('Kappa Over Time')
        plt.savefig(results_directory+"/kappa_plot_"+region+".png")
        plt.show()


def show_test_acc_kappa_all_methods(results_directory,all_method_name = False):
    #results_directory = "/media/hdd11/tipus_c/proves_article"
    results_folders = sorted(list_folders(results_directory))
    regions = ["ll", "bt", "t"]
    for region in regions:
        fig_acc, ax_acc = plt.subplots()
        for method_name in results_folders:
            trainlog_file = os.path.join(results_directory, method_name, "results_timeserie_"+region+".csv")
            if os.path.isfile(trainlog_file):
                results_df = pd.read_csv(trainlog_file)
                results_df['date'] = pd.to_datetime(results_df['date'])
                if all_method_name:
                    ax_acc.plot(results_df['date'], results_df['accuracy'], linestyle='-', label=method_name)
                else:
                    ax_acc.plot(results_df['date'], results_df['accuracy'], linestyle='-', label=method_name[:8])
        ax_acc.xaxis.set_major_locator(mdates.MonthLocator())
        ax_acc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_acc.set_xlabel('Date')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('Accuracy Over Time')
        ax_acc.legend(loc='lower right')
        ax_acc.set_ylim(0, 1)
        # ax_acc.set_xticks(results_df.index[::3])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_directory+"/accuracy_plot_"+region+".png")
        plt.show()


        fig_kp, ax_kp = plt.subplots()
        for method_name in results_folders:
            trainlog_file = os.path.join(results_directory, method_name, "results_timeserie_"+region+".csv")
            if os.path.isfile(trainlog_file):
                results_df = pd.read_csv(trainlog_file)
                results_df['date'] = pd.to_datetime(results_df['date'])
                if all_method_name:
                    ax_kp.plot(results_df['date'], results_df['kappa'], linestyle='-', label=method_name)
                else:
                    ax_kp.plot(results_df['date'], results_df['kappa'], linestyle='-', label=method_name[:8])
        ax_kp.xaxis.set_major_locator(mdates.MonthLocator())
        ax_kp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_kp.set_xlabel('Date')
        ax_kp.set_ylabel('Kappa')
        ax_kp.set_title('Kappa Over Time')
        ax_kp.legend(loc='lower right')
        ax_kp.set_ylim(0, 1)
        ax_kp.xaxis.set_major_locator(mdates.MonthLocator())
        ax_kp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # ax_kp.set_xticks(results_df.index[::3])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_directory+"/kappa_plot_"+region+".png")
        plt.show()



show_test_acc_kappa_all_methods("/home/usuari11/Documents/catcrops/TEST/", all_method_name=False)

# show_validation_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P03-SL_doa_datecrop_cp/RESULTS")
# show_validation_acc_kappa_all_methods("/media/hdd11/tipus_c/proves_article")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial014-rCrop_py_s_cp_doa_LST_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial015-rCrop_py_s_cp_doa_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial016-rCrop_py_s_cp_doa_L2A_LST_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial017-rCrop_py_s_cp_doa_L2A_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial018-rCrop_py_s_cp_doa_LST_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial019-rCrop_py_s_cp_doa_L2A_LST_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial020-rCrop_py_s_cp_doa_L2A_LST_ET_WS_pclass_pcrop_pvar_reg_mun_com_pro_e_s")
#
# show_test_acc_kappa_all_methods("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/")

# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial001-baseline/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial025-rCrop_s_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial026-rCrop_py_s_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial027-rCrop_py_s_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial028-rCrop_py_s_cp_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial029-rCrop_py_s_cp_doa_L2A_e_s/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial030-rCrop_py_s_cp_doa_L2A_mun_com_pro_e_s/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P07-CatCrops/TEST/Trial031-rCrop_py_s_cp_doa_L2A_reg_mun_com_pro_e_s/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P06-CatCrops/TEST/Trial001-baseline_random")
# show_test_acc_kappa_all_methods("/media/hdd11/tipus_c/proves_article/P08-MetroAgriFor/TEST/", all_method_name=True)


# evaluate_test_intra_class_F1('/media/hdd11/tipus_c/proves_article/P08-MetroAgriFor/TEST/M03 - Sparse Multi-Year/test2023_230301_model_34')
# evaluate_test_intra_class_F1('/media/hdd11/tipus_c/proves_article/P08-MetroAgriFor/TEST/M03 - Sparse Multi-Year/test2023_230601_model_34')
# evaluate_test_intra_class_F1('/media/hdd11/tipus_c/proves_article/P08-MetroAgriFor/TEST/M03 - Sparse Multi-Year/test2023_230901_model_34')
# evaluate_test_intra_class_F1('/media/hdd11/tipus_c/proves_article/P08-MetroAgriFor/TEST/M03 - Sparse Multi-Year/test2023_231201_model_34')

# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s_bkup/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial025-rCrop_s_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial026-rCrop_py_s_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial040-rCrop_sl70_nr_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial041-rCrop_py_sl70_nr_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial042-rCrop_py_sl70_nr_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial043-rCrop_py_sl70_nr_cp_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial044-rCrop_py_sl70_nr_cp_doa_L2A_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial045-rCrop_py_sl70_nr_cp_doa_L2A_mun_com_pro_e_s")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial046-rCrop_py_sl70_nr_cp_doa_L2A_reg_mun_com_pro_e_s/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial047-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s/")

# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial050-rCrop_sl70_nr_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial051-rCrop_py_sl70_nr_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial052-rCrop_py_sl70_nr_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial053-rCrop_py_sl70_nr_cp_doa_L2A/")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial054-rCrop_py_sl70_nr_cp_doa_L2A_reg")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial055-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg")
# show_test_acc_kappa_1_method("/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/TEST/Trial057-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval3")

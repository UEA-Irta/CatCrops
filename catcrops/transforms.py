# -*- coding: utf-8 -*-
"""
script to build to transform the raw timeseries and prepare them to the input data format
It determines the data that it is used and passed through the network
This "transform" function is applied when loading the dataset with the catcrops function

    - apply_datecrop ==> Crops the timeseries to the specified date.
    - get_datecrop_from_day2crop ==> transforms the day of the year to date
    - get_transform ==> This is the main function of this script. It is a new version different from the BreizCrops. It allows to work with spectral data, ET data or both, and it allows to determine the following configuration:
        - sequencelength > Limit the sequence length (this feature was already used in the previous version
        - cp > It allows to add the cloud percentage
        - doa > It allows to add the day of the day of each timeseries data
        - datecrop > It allows to crop the timeseries to a specified date
        - use_previous_year_TS > It allows to include the timeseries data from the previous year.
        - sparse > It allows to use timeseries of one year and set to 0 the dates that no spectral or ET data is available.

"""


import torch
import numpy as np
import datetime
import random

bands = {
    "L2A": ['doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CP'],
    # "LST": ['doa', 'LST_B1', 'LST_B2', 'LST_B3', 'LST_B4', 'LST_B5', 'LST_B6', 'LST_B7'],
    # "ET": ['doa', 'ET']
}

selected_bands = {
    "L2A": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
    # "LST": ['LST_B1', 'LST_B2', 'LST_B3', 'LST_B4', 'LST_B5', 'LST_B6', 'LST_B7'],
    # "ET": ['ET']
}

# mean_ET = 2.780248805689623 # Computed with the code "CatCropClassification/altres/compute_ET_mean_and_SD.py"
# std_ET = 1.5580945096158842 # Computed with the code "CatCropClassification/altres/compute_ET_mean_and_SD.py"


def apply_datecrop(x, datecrop, use_previous_year_TS):
    years = np.array([datetime.date.fromtimestamp(y / 1000000000).year for y in x[:, 0]])
    max_year = max(years)
    days = np.array([datetime.date.fromtimestamp(y / 1000000000).timetuple().tm_yday for y in x[:, 0]])
    days[years < max_year] -= 366
    if (not use_previous_year_TS) or (datecrop is None):
        current_year_idx = (years == max_year)
        x = x[current_year_idx, :]  # limit timeseries to current year
        years = years[current_year_idx]   # limit timeseries to current year
        days = days[current_year_idx]   # limit timeseries to current year
        day2crop = 366
    if datecrop == "random":
        day2crop = random.randint(0, 366)
        #print(day2crop)
    elif not (datecrop is None):
        try:
            day2crop = datetime.datetime.strptime(datecrop, "%d/%m/%Y").timetuple().tm_yday
        except ValueError:
            print("Variable daycrop is neither None, 'random', nor a valid date with format '%d/%m/%Y'.")
    target_dates_idx = (days < day2crop) & (days > (day2crop - 366))
    x = x[target_dates_idx, :]
    years = years[target_dates_idx]
    days = days[target_dates_idx]
    return x, years, days, day2crop

def get_datecrop_from_day2crop(year,day2crop):
    start_date = datetime.datetime(year, 1, 1)
    target_date = start_date + datetime.timedelta(days=day2crop - 1)
    datecrop = target_date.strftime("%d/%m/%Y")
    return datecrop

def get_transform_CatCrops(sequencelength=None,         #Limit Sequence Lenght:         None, or integrer
                           datecrop = "random",         #Timeseries (TS) max date:      None, random or "%d/%m/%Y"
                           use_previous_year_TS = True, #Use previous year TS:          True or False
                           sparse = True,               #Fill empty dates with zeros:   True or False
                           cp=True,                     #Use cloud percentage:          True or False
                           doa=True,                    #Use slope of the field:        True or False
                           noreplace = False):          #fill with zeros in the case that the sequence length is lower than "sequencelength" value
    L2A_selected_band_idxs = np.array([bands["L2A"].index(b) for b in selected_bands["L2A"]])
    # LST_selected_band_idxs = np.array([bands["LST"].index(b) for b in selected_bands["LST"]])

    # def transform(L2A_x=None, LST_x=None, ET_x=None, pclassid_val=None, pcrop_val=None, pvar_val=None, sreg_val=None, mun_val=None, com_val=None, prov_val=None, elev_val=None, slope_val=None):
    #     assert ((not L2A_x is None) or (not LST_x is None) or (not ET_x is None)), "No Spectral neither ET imput data was passed. Please pass at least one of this (L2A, LST or ET timeseries)."

    def transform(L2A_x=None, pclassid_val=None, pcrop_val=None, pvar_val=None, sreg_val=None, mun_val=None, com_val=None, prov_val=None, elev_val=None, slope_val=None):
        assert (not L2A_x is None), "No Spectral input data was passed. Please pass L2A timeseries."
        # if not L2A_x is None:
        L2A_x, years, days, day2crop = apply_datecrop(L2A_x, datecrop, use_previous_year_TS)

        raw_scaled_bands = L2A_x[:, L2A_selected_band_idxs] * 1e-4  # limit to selected bands and scale reflectances to 0-1
        ts_data = raw_scaled_bands
        if ts_data.shape[0] == 0:
            ts_data = np.zeros((1,10))
        if cp:
            cp_indx = np.array([bands["L2A"].index('CP')])
            raw_scaled_cp = L2A_x[:, cp_indx] * 1e-2
            ts_data = np.hstack((ts_data, raw_scaled_cp))
        if doa and not sparse:
            days_normalized = [day / 366 for day in days]
            ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))

        if sparse:
            assert (sequencelength is None), "To use sparse time series the sequencelength must be None"
            if not use_previous_year_TS:
                ts_data_sparse = np.zeros((366, ts_data.shape[1]))
                ts_data_sparse[days, :] = ts_data
                ts_data = ts_data_sparse
                if doa:
                    days_normalized = [day / 366 for day in range(0, 366)]
                    ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))
            else:
                ts_data_sparse = np.zeros((366 * 2, ts_data.shape[1]))
                ts_data_sparse[days + 366, :] = ts_data
                ts_data = ts_data_sparse[day2crop:day2crop + 366, :]
                if doa:
                    days_normalized = [day / 366 for day in range(day2crop - 366, day2crop)]
                    ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))

        if not sequencelength is None:
            assert (not sparse), "To limit the sequencelength sparse must be False"
            if not noreplace:
                replace = False if ts_data.shape[
                                       0] >= sequencelength else True  # choose with replacement if sequencelength smaller als choose_t
                idxs = np.random.choice(ts_data.shape[0], sequencelength, replace=replace)
                idxs.sort()
                ts_data = ts_data[idxs]
            else:
                if ts_data.shape[0]<sequencelength:
                    zeros_tensor = np.zeros((sequencelength-ts_data.shape[0],ts_data.shape[1]))
                    ts_data = np.concatenate((ts_data, zeros_tensor), axis=0)
                else:
                    ts_data = ts_data[-sequencelength:]

        # if not LST_x is None:
        #     if (datecrop == "random") and (not L2A_x is None):
        #         rand_datecrop = get_datecrop_from_day2crop(years.max(), day2crop)
        #         LST_x, LST_years, LST_days, LST_day2crop = apply_datecrop(LST_x, rand_datecrop, use_previous_year_TS)
        #     else:
        #         LST_x, LST_years, LST_days, LST_day2crop = apply_datecrop(LST_x, datecrop, use_previous_year_TS)
        #     LST_ts_data = LST_x[:, LST_selected_band_idxs]
        #     if doa and (L2A_x is None) and (not sparse):
        #         days_normalized = [day / 366 for day in LST_days]
        #         LST_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), LST_ts_data))
        #
        #     if sparse:
        #         assert (sequencelength is None), "To use sparse time series the sequencelength must be None"
        #         if not use_previous_year_TS:
        #             LST_ts_data_sparse = np.zeros((366, LST_ts_data.shape[1]))
        #             LST_ts_data_sparse[LST_days, :] = LST_ts_data
        #             LST_ts_data = LST_ts_data_sparse
        #             if doa and (L2A_x is None):
        #                 days_normalized = [day / 366 for day in range(0, 366)]
        #                 LST_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), LST_ts_data))
        #         else:
        #             LST_ts_data_sparse = np.zeros((366 * 2, LST_ts_data.shape[1]))
        #             LST_ts_data_sparse[LST_days + 366, :] = LST_ts_data
        #             LST_ts_data = LST_ts_data_sparse[LST_day2crop:LST_day2crop + 366, :]
        #             if doa and (L2A_x is None):
        #                 days_normalized = [day / 366 for day in range(LST_day2crop - 366, LST_day2crop)]
        #                 LST_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), LST_ts_data))
        #
        #     if not L2A_x is None:
        #         assert (sparse is True), "To combine the spectral time series with ET timseries, sparse must be true"
        #         ts_data = np.column_stack((ts_data, LST_ts_data))
        #     else:
        #         ts_data = LST_ts_data

        # if not ET_x is None:
        #     if (datecrop == "random") and (not L2A_x is None):
        #         rand_datecrop = get_datecrop_from_day2crop(years.max(), day2crop)
        #         ET_x, ET_years, ET_days, ET_day2crop = apply_datecrop(ET_x, rand_datecrop, use_previous_year_TS)
        #     elif (datecrop == "random") and (not LST_x is None):
        #         rand_datecrop = get_datecrop_from_day2crop(LST_years.max(), LST_day2crop)
        #         ET_x, ET_years, ET_days, ET_day2crop = apply_datecrop(ET_x, rand_datecrop, use_previous_year_TS)
        #     else:
        #         ET_x, ET_years, ET_days, ET_day2crop = apply_datecrop(ET_x, datecrop, use_previous_year_TS)
        #     ET_ts_data = ((ET_x[:, 1] - mean_ET) / (std_ET * 2.5)).reshape(-1, 1)
        #
        #     if doa and (L2A_x is None) and (LST_x is None) and (not sparse):
        #         days_normalized = [day / 366 for day in ET_days]
        #         ET_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ET_ts_data))
        #
        #     if sparse:
        #         if not use_previous_year_TS:
        #             ET_ts_data_sparse = np.zeros((366, 1))
        #             ET_ts_data_sparse[ET_days, :] = ET_ts_data
        #             ET_ts_data = ET_ts_data_sparse
        #             if doa and (L2A_x is None) and (LST_x is None):
        #                 days_normalized = [day / 366 for day in range(0, 366)]
        #                 ET_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ET_ts_data))
        #         else:
        #             ET_ts_data_sparse = np.zeros((366 * 2, 1))
        #             ET_ts_data_sparse[ET_days + 366, :] = ET_ts_data
        #             ET_ts_data = ET_ts_data_sparse[ET_day2crop:ET_day2crop + 366]
        #             if doa and (L2A_x is None) and (LST_x is None):
        #                 days_normalized = [day / 366 for day in range(ET_day2crop - 366, ET_day2crop)]
        #                 ET_ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ET_ts_data))
        #     if (not L2A_x is None) or (not LST_x is None):
        #         assert (sparse is True), "To combine the spectral time series with ET timseries, sparse must be true"
        #         ts_data = np.column_stack((ts_data, ET_ts_data))
        #     else:
        #         ts_data = ET_ts_data

        if not pclassid_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), pclassid_val)))
        if not pcrop_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), pcrop_val)))
        if not pvar_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), pvar_val)))
        if not sreg_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), sreg_val)))
        if not mun_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), mun_val)))
        if not com_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), com_val)))
        if not prov_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), prov_val)))
        if not elev_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), elev_val)))
        if not slope_val is None:
            ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), slope_val)))


        return torch.from_numpy(ts_data).type(torch.FloatTensor)

    return transform


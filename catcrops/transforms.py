# -*- coding: utf-8 -*-
"""
Script to build to transform the raw timeseries and prepare them to the input data format

It determines the data that it is used and passed through the network
This "transform" function is applied when loading the dataset with the catcrops function

Key Features:
- apply_datecrop ==> Crops the timeseries to the specified date.
- get_datecrop_from_day2crop ==> transforms the day of the year to date
- get_transform ==> This is the main function of this script. It is a new version different from the BreizCrops.
                    It allows to determine the following configuration:
    - sequencelength -> Limit the sequence length (this feature was already used in the previous version)
    - cp -> It allows to add the cloud percentage
    - doa -> It allows to add the day of the day of each timeseries data
    - datecrop -> It allows to crop the timeseries to a specified date
    - use_previous_year_TS -> It allows to include the timeseries data from the previous year.
    - sparse -> It allows to use timeseries of one year and set to 0 the dates that no spectral data is available.

Author:
    -
datetime:27/5/2023 16:50
"""

import torch
import numpy as np
import datetime
import random

# Dictionary defining the dataset fields for each parcel based on input data type
bands = {
    "L2A": ['doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CP'],
}

# Define selected spectral bands used for processing
selected_bands = {
    "L2A": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
}

def apply_datecrop(x, datecrop, use_previous_year_TS):
    """
    Crops the time series to the specified date.

    Args:
        x (np.array): Time series data (first column represents timestamps).
        datecrop (str or None): Target date for cropping, format "DD/MM/YYYY" or "random".
        use_previous_year_TS (bool): Whether to include time series data from the previous year.

    Returns:
        tuple: (cropped time series, years, days, target day to crop)
    """

    # Extract years from timestamps
    years = np.array([datetime.date.fromtimestamp(y / 1e9).year for y in x[:, 0]])
    max_year = max(years)

    # Convert timestamps into day-of-year format
    days = np.array([datetime.date.fromtimestamp(y / 1e9).timetuple().tm_yday for y in x[:, 0]])
    days[years < max_year] -= 366  # Adjust previous year days

    # Limit time series to the current year if previous year data is not used
    if (not use_previous_year_TS) or (datecrop is None):
        current_year_idx = (years == max_year)
        x = x[current_year_idx, :]
        years = years[current_year_idx]
        days = days[current_year_idx]
        day2crop = 366  # Default max day

    # Handle different date cropping strategies
    if datecrop == "random":
        day2crop = random.randint(0, 366)
    elif datecrop is not None:
        try:
            day2crop = datetime.datetime.strptime(datecrop, "%d/%m/%Y").timetuple().tm_yday
        except ValueError:
            print("Variable daycrop is neither None, 'random', nor a valid date with format '%d/%m/%Y'.")

    # Filter time series based on the target crop date
    target_dates_idx = (days < day2crop) & (days > (day2crop - 366))
    x = x[target_dates_idx, :]
    years = years[target_dates_idx]
    days = days[target_dates_idx]

    return x, years, days, day2crop


def get_datecrop_from_day2crop(year, day2crop):
    """
    Converts a day-of-year value into a formatted date string.

    Args:
        year (int): Year for reference.
        day2crop (int): Day of the year.

    Returns:
        str: Date in "DD/MM/YYYY" format.
    """
    start_date = datetime.datetime(year, 1, 1)
    target_date = start_date + datetime.timedelta(days=day2crop - 1)
    return target_date.strftime("%d/%m/%Y")

def get_transform_CatCrops(sequencelength=None,         #Limit Sequence Lenght:         None, or integrer
                           datecrop = "random",         #Timeseries (TS) max date:      None, random or "%d/%m/%Y"
                           use_previous_year_TS = True, #Use previous year TS:          True or False
                           sparse = True,               #Fill empty dates with zeros:   True or False
                           cp=True,                     #Use cloud percentage:          True or False
                           doa=True,                    #Use slope of the field:        True or False
                           noreplace = False):          #fill with zeros in the case that the sequence length is lower than "sequencelength" value
    """
    Creates a transformation function to preprocess time series data for crop classification.

    Args:
        sequencelength (int or None): Limits sequence length. If None, no limit is applied.
        datecrop (str): Specifies the maximum date for time series ("random", None, or "DD/MM/YYYY").
        use_previous_year_TS (bool): Whether to include previous year's time series data.
        sparse (bool): Whether to use sparse representation (fill missing dates with zeros).
        cp (bool): Whether to include cloud percentage as an input feature.
        doa (bool): Whether to include day-of-year information.
        noreplace (bool): Whether to pad short sequences with zeros instead of sampling.

    Returns:
        callable: Transformation function to be applied to time series data.
    """

    # Get the indices of the selected spectral bands
    L2A_selected_band_idxs = np.array([bands["L2A"].index(b) for b in selected_bands["L2A"]])

    def transform(L2A_x=None, pclassid_val=None, pcrop_val=None, pvar_val=None, sreg_val=None,
                  mun_val=None, com_val=None, prov_val=None, elev_val=None, slope_val=None):
        """
        Applies transformations to the input time series.

        This function processes spectral data, adds metadata features, ensures sequence length constraints,
        and optionally applies sparse representations.

        Args:
            L2A_x (np.array): Sentinel-2 spectral time series data.
            pclassid_val, pcrop_val, pvar_val, sreg_val, mun_val, com_val, prov_val, elev_val, slope_val (float or None):
                Metadata values normalized for input.

        Returns:
            torch.FloatTensor: Transformed time series ready for model input.
        """

        assert L2A_x is not None, "No spectral input data was passed. Please pass L2A timeseries."

        # Apply cropping based on date selection
        L2A_x, years, days, day2crop = apply_datecrop(L2A_x, datecrop, use_previous_year_TS)

        # Scale reflectance values and select relevant bands
        raw_scaled_bands = L2A_x[:, L2A_selected_band_idxs] * 1e-4  # limit to selected bands and scale reflectances to 0-1
        ts_data = raw_scaled_bands if raw_scaled_bands.shape[0] > 0 else np.zeros((1, 10))

        # Add cloud percentage as a feature
        if cp:
            raw_scaled_cp = L2A_x[:, np.array([bands["L2A"].index('CP')])] * 1e-2
            ts_data = np.hstack((ts_data, raw_scaled_cp))

        # Add day-of-year information if not using sparse representation
        if doa and not sparse:
            days_normalized = [day / 366 for day in days]
            ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))

        # Apply sparse time series processing
        if sparse:
            assert sequencelength is None, "To use sparse time series, sequencelength must be None"

            # If not using the previous year's time series, fill missing values with zeros for 366 days
            if not use_previous_year_TS:
                ts_data_sparse = np.zeros((366, ts_data.shape[1]))
                ts_data_sparse[days, :] = ts_data  # Assign existing time series data
                ts_data = ts_data_sparse

                # Add day-of-year information to sparse representation
                if doa:
                    days_normalized = [day / 366 for day in range(0, 366)]
                    ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))

            # If using the previous year's time series, extend to 732 days and crop from `day2crop`
            else:
                ts_data_sparse = np.zeros((366 * 2, ts_data.shape[1]))
                ts_data_sparse[days + 366, :] = ts_data  # Assign time series shifted by 1 year
                ts_data = ts_data_sparse[day2crop:day2crop + 366, :]

                # Add day-of-year information for the extracted range
                if doa:
                    days_normalized = [day / 366 for day in range(day2crop - 366, day2crop)]
                    ts_data = np.hstack((np.transpose(days_normalized).reshape(-1, 1), ts_data))

        # Apply sequence length limitations
        if sequencelength is not None:
            assert not sparse, "To limit the sequencelength, sparse must be False"

            if not noreplace:
                # If sequence length is shorter than required, enable replacement sampling
                replace = ts_data.shape[0] < sequencelength
                idxs = np.random.choice(ts_data.shape[0], sequencelength, replace=replace)
                idxs.sort()
                ts_data = ts_data[idxs]

            else:
                # If the time series is shorter, pad with zeros to match `sequencelength`
                if ts_data.shape[0] < sequencelength:
                    zeros_tensor = np.zeros((sequencelength - ts_data.shape[0], ts_data.shape[1]))
                    ts_data = np.concatenate((ts_data, zeros_tensor), axis=0)
                else:
                    # Otherwise, keep only the last `sequencelength` observations
                    ts_data = ts_data[-sequencelength:]

        # Add metadata values as additional features
        for val in [pclassid_val, pcrop_val, pvar_val, sreg_val, mun_val, com_val, prov_val, elev_val, slope_val]:
            if val is not None:
                ts_data = np.vstack((ts_data, np.full((1, ts_data.shape[1]), val)))

        return torch.from_numpy(ts_data).type(torch.FloatTensor)

    return transform
# -*- coding: utf-8 -*-
"""
Script to extract the values of Sentinel-2 (S2) bands from a vector file of polygons using Google Earth Engine.
Uses Python's multiprocessing to perform multiple requests simultaneously, improving performance for large datasets.

### Overview:
Given a vector file (shapefile) of polygons, the script retrieves the corresponding Sentinel-2 images
for the specified time range and extracts pixel values for each polygon.

### Additional Features:
- Adds cloud probability from the repository "COPERNICUS/S2_CLOUD_PROBABILITY".
- Supports appending new data to existing files or overwriting them.
- Previously computed data is skipped to optimize processing time.
- Automatically locates the shapefile based on the dataset directory, the selected study area, and the specified year—no need to manually specify the shapefile path.

### Processing Steps:
1. Selects Sentinel-2 images based on the polygon extent and date range.
2. Selects images from 'S2_CLOUD_PROBABILITY'.
3. For each S2 image, finds the corresponding 'S2_CLOUD_PROBABILITY' image and adds the 'probability' band.
4. Computes the average spectral values for each polygon and stores the results.

### Output:
- The extracted data is saved as CSV files, one per polygon, in a structured format.
- Files include spectral bands, cloud probability, date of acquisition, and polygon ID.
- Logs are generated to track parcels that could not be processed.

author:
datetime:1/12/2023 12:58
"""


import os
from os import path as pth
import ee
import geemap
import geopandas as gpd
import pandas as pd
from datetime import datetime as dt
import glob
from multiprocessing import Pool, cpu_count
import time
from datetime import timedelta
import argparse


# Default values
list_ae = ['lleida', 'baixter']
list_any = ["2021", "2022", "2023"]
zona_default = list_ae[0]
div_ss_default = 1
data_inici_default = "20220101"
data_final_default = "20240101"
any_d_default = list_any[-1]
dataset_fold_default = r'catcrops_dataset'
error_fold_default = dataset_fold_default

f_out_txt = None
f_id_nv = None


# Configure the argument parser
parser = argparse.ArgumentParser(
    description=r"Script to download S2 data from Google Earth Engine. If -B or -O are not selected, "
                r"only the data for parcels without existing CSV files will be downloaded.",
    epilog="Script to download S2 data from Google Earth Engine")

parser.add_argument("-Z", "--zona", choices=list_ae, default=zona_default,
                    help="Name of the study area. Default: %s" % zona_default)
parser.add_argument("-A", "--any", choices=list_any, default=any_d_default,
                    help="Year of the dataset. Default: %s" % any_d_default)
parser.add_argument("-N", "--numdiv", type=int, default=div_ss_default,
                    help="Number of parcels per subset to download at once. Default: %d" % div_ss_default)
parser.add_argument("-I", "--datainici", default=data_inici_default,
                    help="Start date. Format: YYYYMMDD. Default: %s" % data_inici_default)
parser.add_argument("-F", "--datafinal", default=data_final_default,
                    help="End date (one day after the last date to retrieve data). Format: YYYYMMDD. Default: %s" % data_final_default)
parser.add_argument("-B", "--append", action="store_true",
                    help="If we want to add new dates to already created files. Otherwise, existing files will not be modified.")
parser.add_argument("-O", "--overwrite", action="store_true",
                    help="If we want to overwrite existing files.")
parser.add_argument("-D", "--datasetfolder", default=dataset_fold_default,
                    help="Path to the dataset folder. Default: %s" % dataset_fold_default)
parser.add_argument("-E", "--errorfolder", default=error_fold_default,
                    help="Path to the folder where two text files will be saved with the IDs of parcels that could not be downloaded."
                         "If set to None, the files will not be saved. Default: %s" % error_fold_default)

args = parser.parse_args()

# Select the study area (only modify this if needed)
zona_n = args.zona

# Dataset year
any_d = args.any

# Dataset directory (working directory
dataset_fold = args.datasetfolder

# Shapefile path
f = pth.join(dataset_fold, '%s/%s.shp' % (any_d, zona_n))

# Number of parcels per subset
div_ss = args.numdiv

# Output folder (where CSV files will be saved)
folder_out = pth.join(dataset_fold, "%s/L2A/%s/csv" % (any_d, zona_n))

# Overwrite existing files?
overwrite_f = args.overwrite

# Append new data to existing files? (requires overwrite_f=False)
appen_d = args.append

# Start and end dates (to define the time range)
data_inici = args.datainici
data_final = args.datafinal

# Parcels to exclude from calculations
list_p = []

# Paths for error logs
error_fold = args.errorfolder
if error_fold:
    f_out_txt = pth.join(error_fold, r'tiles_error_%s.txt' % zona_n)  # Quan el problema és del Google Earth
    f_id_nv = pth.join(error_fold, r'id_nv_%s.txt' % zona_n)  # Quan el problema és que la parcel·la és massa petita.

# Sentinel-2 tiles (set to None if not using tiles)
s2_tiles = ['30TYL', '30TYM', '31TBF', '31TBG', '31TCF', '31TCG', '31TCH', '31TDF', '31TDG', '31TDH', '31TEG', '31TEH']

# List of DataFrame columns (in the correct order)
list_col = ['nom_id_imatge', 'B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
            'probability', 'doa', 'crop_code', 'ID']

# Sentinel-2 bands to select
s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']


# -------------------------------------------------

def split_gdf(n_rows, n_split):
    """
    Splits a GeoDataFrame into smaller chunks based on row count.

    :param n_rows int: Total number of rows in the GeoDataFrame
    :param n_split int: Number of rows per chunk
    :return list[tuple(int, int)]: List of tuples containing index ranges for each chunk
    """
    return [(i, min(i + n_split, n_rows)) for i in range(0, n_rows, n_split)]


def extract_s2(t):
    """
    Function to extract Sentinel-2 band values for polygons grouped in a GeoDataFrame using Google Earth Engine.
    This function is used with Python's multiprocessing to perform multiple queries simultaneously.

    :param t int: Index from the list l_row, which contains tuples with index ranges of the GeoDataFrame subset
    :return list[int], list[int]: IDs of parcels with errors due to Google Earth Engine, IDs of parcels with errors
    """

    def get_feat_list(gdb):
        """
        Function to create a list of Google Earth Engine features from a GeoDataFrame.
        Each row in the GeoDataFrame becomes a feature.
        The column names are stored as properties. In this case, only 'ID' is kept.

        :param gdb geopandas.GeoDataFrame: GeoDataFrame with polygons
        :return list[ee.Feature]: List of Google Earth Engine features
        """
        # Create an empty list and iterate through each row of the GeoDataFrame to extract the geometry and column names
        features = []
        for i in range(gdb.shape[0]):
            geom = gdb.iloc[i:i + 1, :]
            jsonDict = eval(geom.to_json())
            geojsonDict = jsonDict['features'][0]
            features.append(ee.Feature(geojsonDict))
        return features

    def indexJoin(collectionA, collectionB, propertyName):
        """
        Function to merge two image collections in Google Earth Engine based on their system index.

        :param collectionA: Primary image collection
        :param collectionB: Secondary image collection
        :param propertyName: Property name to store the joined image
        :return: Merged image collection
        """
        joined = ee.ImageCollection(ee.Join.saveFirst(propertyName).apply(
            primary=collectionA,
            secondary=collectionB,
            condition=ee.Filter.equals(
                leftField='system:index',
                rightField='system:index'
            )
        ))
        # Merge the bands of the joined images
        return joined.map(lambda image: image.addBands(ee.Image(image.get(propertyName))))

    def red_reg(img):
        """
        Function to obtain the average value of polygons for each image.
        Returns a "Feature Collection" in Google Earth Engine.

        :param img ee.Image: Sentinel-2 image
        :return FeatureCollection: Features with calculated values
        """
        sta_i = img.reduceRegions(collection=ee_collection, reducer=ee.Reducer.mean(), scale=10) \
            .map(lambda x: x.set({'nom_id_imatge': img.get('system:index'),
                                  'doa': ee.Number.parse(img.date().format('YYYYMMdd')).toInt()}))
        return sta_i

    def get_stat_df(feat_col, l_col):
        """
        Function to create a DataFrame with the average values of Sentinel-2 bands for each polygon and date.

        :param feat_col ee.FeatureCollection: Features with extracted values
        :param l_col list[str]: List of column names in the correct order
        :return pandas.DataFrame, list[int]: DataFrame with extracted data, list of parcels not calculated
        """
        # Convert the FeatureCollection into a DataFrame using geemap's 'ee_to_df' function
        df_s = geemap.ee_to_df(feat_col, columns=l_col)

        # Remove rows without dates (small parcels)
        l_nv = df_s[df_s['doa'].isna()]['ID'].unique().tolist()  # Store IDs of parcels without data
        df_s = df_s.dropna(subset=['doa'])

        # Rename columns
        df_s.rename(columns={"ID": "id", "crop_code": "Label", 'probability': 'CP'}, inplace=True)
        # Sort by date
        df_s.sort_values(by=['doa'], inplace=True)

        # Convert date format
        df_s['doa'] = df_s['doa'].map(lambda x: dt.strftime(dt.strptime(str(int(x)), "%Y%m%d"), "%Y-%m-%d"))

        return df_s, l_nv

    def merge_df(df1, id_name, dic_df_f, col_n='doa'):
        """
        Function to merge new data with existing CSV files.

        :param df1 pandas.DataFrame: New data
        :param id_name: Parcel ID
        :param dic_df_f: Dictionary of existing CSV file paths
        :param col_n: Column name for date filtering
        :return: Merged DataFrame
        """
        try:
            f2 = dic_df_f[id_name]
        except:
            print('No existing CSV file for', id_name)
            return df1

        df_old = pd.read_csv(f2)
        return pd.concat([df_old, df1.loc[df1[col_n] > df_old[col_n].max()]], ignore_index=True)

    # Extract values from the tuple
    i, j = t
    # Select the subset of the GeoDataFrame
    gdf_ss = gdf[i:j]

    # Lists to store parcels that couldn't be processed
    list_t = []
    list_nv = []

    try:
        # Create a FeatureCollection from the polygons
        ee_collection = ee.FeatureCollection(get_feat_list(gdf_ss))

        # Get Sentinel-2 images with the selected bands
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee_collection) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.inList('MGRS_TILE', s2_tiles)) \
            .select(s2_bands)

        # Get cloud probability images
        cloudy_collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterBounds(ee_collection) \
            .filterDate(start_date, end_date)

        # Merge the collections
        sentinel_collection = indexJoin(s2_collection, cloudy_collection, 'CP')

        # Extract values as a FeatureCollection
        s2_stat = sentinel_collection.map(red_reg).flatten()

        # Convert FeatureCollection to DataFrame
        df_s2, list_nv = get_stat_df(s2_stat, l_col=list_col)

        # Remove rows without data
        df_s2 = df_s2.dropna(subset=s2_bands, thresh=len(s2_bands))

        # Iterate through each parcel ID and save as CSV
        for id_p in df_s2['id'].unique():
            df_id = df_s2.loc[df_s2['id'] == id_p]
            if appen_d and not overwrite_f:
                df_id = merge_df(df_id, id_p, dic_f)
            csv_out = pth.join(folder_out, str(id_p) + '.csv')
            df_id.to_csv(csv_out, index=False)

        print("DONE => ", t)
        return list_nv, list_t

    except Exception as e:
        print(e)
        print('ERROR => ', i, j)
        list_t += gdf_ss['ID'].to_list()
        return list_nv, list_t


def save_list_txt(list_n, f_txt):
    """
    Function to save a list to a text file.

    :param list_n: List of values (one per line)
    :param f_txt: Path where the text file will be saved
    """
    with open(f_txt, 'w') as output:
        for row in list_n:
            output.write(str(row) + '\n')


# ------------------------------------------------
# Initialize the Google Earth Engine API (using 'highvolume' for faster queries)
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# Define date range
start_date = dt.strptime(data_inici, "%Y%m%d")
end_date = dt.strptime(data_final, "%Y%m%d")

# Load the shapefile
gdf = gpd.read_file(f)

# Remove already processed parcels
if not overwrite_f and appen_d:
    dic_f = {int(pth.basename(x).split('.')[0]): x for x in glob.glob(pth.join(folder_out, "*.csv"))}
if not overwrite_f and not appen_d:
    l_f = [int(pth.basename(x).split('.')[0]) for x in glob.glob(pth.join(folder_out, "*.csv"))]
    gdf = gdf.loc[~gdf['ID'].isin(l_f)]

# Reproject to EPSG 4326
gdf.to_crs(crs=4326, inplace=True)

# Create a list with the index ranges of the subset divisions (to split the GeoDataFrame into smaller parts)
l_rows = split_gdf(n_rows=gdf.shape[0], n_split=div_ss)

# Start the multiprocessing pool and execute the 'extract_s2' function for each element in the 'l_rows' list
start_time = time.time()
if __name__ == '__main__':
    with Pool(cpu_count() - 1) as p:
        print(f'starting computations on {cpu_count() - 1} cores')
        # p = Pool(cpu_count()-1)
        s2_l = p.map(extract_s2, l_rows)
        # Clean up the pool
        p.close()
        p.join()

# Store the two lists with the parcels that could not be processed, retrieved from the pool
l_nv = sum([x[0] for x in s2_l], [])
l_t = sum([x[1] for x in s2_l], [])

# Print final information
print('------------------------------')
print('FINAL')
print(
    "PARCEL GROUP => %d parcels could not be processed. Number of subsets: %.2f" % (len(l_t), (len(l_t) / div_ss)))
print(l_t)
if f_out_txt:
    save_list_txt(l_t, f_out_txt)

print("SMALL PARCELS => %d parcels could not be processed" % len(l_nv))
print(l_nv)
if f_id_nv:
    save_list_txt(l_nv, f_id_nv)
print(u"--- TOTAL PROCESSING TIME %s ---" % timedelta(seconds=int(time.time() - start_time)))
print('-------- END OF EXECUTION ------------------')

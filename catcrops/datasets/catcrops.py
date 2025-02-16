# -*- coding: utf-8 -*-
"""
CatCrops - PyTorch Dataset for Crop Classification

This module defines the `CatCrops` class, a custom PyTorch dataset for time series crop classification.
The dataset is derived from Sentinel-2 Level 2A imagery and associated metadata, enabling classification
of agricultural parcels based on spectral and contextual information.

Features:
- Supports Sentinel-2 Level 2A (L2A) spectral data for different geographic regions.
- Incorporates auxiliary metadata, including:
  - Previous year class, crop code, and variety.
  - Irrigation system, municipality, comarca, and province identifiers.
  - Elevation and slope information.
- Implements data transformation pipelines for feature extraction and preprocessing.
- Handles dataset indexing and structured loading:
  - Reads time series from CSV files or precompiled HDF5 format.
  - Enables optional RAM preloading for optimized data retrieval.
  - Filters out incomplete or missing data entries.
- Generates dataset indices to maintain structured metadata referencing.

Usage:
To instantiate the dataset:
    from catcrops import CatCrops

    dataset = CatCrops(region="lleida", root="./catcrops_dataset/", year=2023, preload_ram=False, L2A=True)

Author:
- Jordi Gené Mola
- Magí Pàmies Sans

datetime:27/5/2023 16:50
"""
import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from ..transforms import get_transform_CatCrops
from datetime import datetime

# Dictionary defining the dataset fields for each parcel based on input data type
BANDS = {
    "L2A": ['nom_id_imatge', 'B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'QA10', 'QA20',
            'QA60', 'CP', 'doa', 'Label', 'id'],
}

# Selected bands used for processing
SELECTED_BANDS = {
    "L2A": ['doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CP'],
}


class CatCrops(Dataset):
    """
    Custom PyTorch dataset class for handling the CatCrops dataset.

    This class provides functionalities to load and preprocess time series data
    for crop classification using Sentinel-2 imagery and metadata.
    """

    def __init__(self,
                 region,
                 root="./catcrops_dataset/",  # TODO: Define the dataset path
                 year=2022,
                 transform=None,
                 target_transform=None,
                 filter_length=0,
                 verbose=False,
                 recompile_h5_from_csv=False,
                 preload_ram=False,
                 L2A=True,                  # Use Sentinel-2 Level 2A data
                 pclassid=False,            # Use previous year class
                 pcrop=False,               # Use previous crop code
                 pvar=False,                # Use previous variety code
                 sreg=False,                # Use irrigation system code
                 mun=False,                 # Use municipality code
                 com=False,                 # Use comarca (regional) code
                 prov=False,                # Use province code
                 elev=False,                # Use elevation data
                 slope=False):              # Use slope data
        """
        Initializes the CatCrops dataset.

        Args:
            region (str): Dataset region. Choose between "baixter" or "lleida".
            root (str): Path where the dataset is stored. Default is `./catcrops_dataset/`.
            year (int): Year of the dataset (currently supports `2022` and `2023`).
            transform (callable, optional): Function to apply transformations to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton.
            target_transform (callable, optional): Function to transform the labels.
            filter_length (int): Time series shorter than this value will be ignored.
            verbose (bool): If True, enables detailed logging.
            recompile_h5_from_csv (bool): If True, recompiles the HDF5 database from CSV files. Only required when dealing with new datasets.
            preload_ram (bool): If True, loads all time series data into RAM for faster access.
            L2A (bool): Whether to use Sentinel-2 Level 2A data.
            pclassid (bool): Use the previous year's class label.
            pcrop (bool): Use the previous year's crop code.
            pvar (bool): Use the previous year's variety code.
            sreg (bool): Use the irrigation system data.
            mun (bool): Include the municipality code.
            com (bool): Include the comarca (regional) code.
            prov (bool): Include the province code.
            elev (bool): Include the elevation data.
            slope (bool): Include the slope data.
        """

        level = "L2A"
        assert year in [2022, 2023]
        assert region in ["lleida", "baixter"]

        # Set transformation functions if not provided
        if transform is None:
            transform = get_transform_CatCrops()
        if target_transform is None:
            target_transform = get_default_target_transform()
        self.transform = transform
        self.target_transform = target_transform

        # Dataset metadata
        self.region = region.lower()
        self.L2A_bands = BANDS["L2A"]
        self.verbose = verbose
        self.year = year
        self.level = level

        # Selected dataset fields
        self.L2A = L2A
        self.pclassid = pclassid
        self.pcrop = pcrop
        self.pvar = pvar
        self.sreg = sreg
        self.mun = mun
        self.com = com
        self.prov = prov
        self.elev = elev
        self.slope = slope
        self.max_elev = 1500  # Maximum elevation considered
        self.max_slope = 54   # Maximum slope considered

        if verbose:
            print(f"Initializing CatCrops region {region}, year {year}")

        # Define dataset paths
        self.root = root
        self.indexfile, self.codesfile, self.shapefile, self.classmapping, self.municipismapping, self.sregadiumapping,\
        self.varietatmapping, self.L2A_csvfolder, self.L2A_h5path = self.build_folder_structure(
            self.root, self.year, self.level, self.region
        )

        # Load class mappings and metadata
        self.load_classmapping(self.classmapping)
        self.load_municipismapping(self.municipismapping)
        self.load_sregadiumapping(self.sregadiumapping)
        self.load_varietatmapping(self.varietatmapping)

        # Generate index file if it does not exist
        if not os.path.exists(self.indexfile):
            self.write_index()

        # Load index file and filter unnecessary classes
        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"Kept {len(self.index)} time series references after applying class mapping")

        # Remove time series that are shorter than the filter_length (zero-length)
        if self.index.index.name != "idx":
            self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")

        self.maxseqlength = int(self.index["sequencelength"].max())

        # Check for missing dataset files
        if not os.path.exists(self.codesfile):
            print("Missing 'codes.csv' file.")
            exit()
        self.codes = pd.read_csv(self.codesfile, delimiter=",", index_col=0)

        # Rename QA60 column for consistency
        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

        # Load additional geographic metadata
        self.index_geo_file = os.path.join(root, str(year), region + "_d.csv")
        self.index_geo = pd.read_csv(self.index_geo_file, delimiter=",", index_col=0)

        # Ensure class mapping is correctly assigned
        if "classid" not in self.index.columns or "classname" not in self.index.columns or "region" not in self.index.columns:
            self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
            self.index[["classid", "classname"]] = self.index["CODE_CULTU"].apply(lambda code: self.mapping.loc[code])
            self.index["region"] = self.region
            self.index = self.index.loc[self.index["id"].isin(self.index_geo.index)]
            self.index[["var_code", "pcrop_code", "pvar_code", "sreg_code", "mun_code", "elev", "slope"]] = \
                self.index["id"].apply(lambda ID: self.index_geo.loc[ID]).iloc[:, 1:]
            self.index.to_csv(self.indexfile)

        # Compile h5 file in case that it not exist or it is required to be recompiled:
        if L2A and (not os.path.exists(self.L2A_h5path) or recompile_h5_from_csv):
            print("h5_database file is missing or is asked to be recompiled")
            self.write_h5_from_csv(self.index, self.L2A_h5path,"L2A")

        # Preload data into RAM if enabled
        if preload_ram:
            if self.L2A:
                self.L2A_X_list = list()
                with h5py.File(self.L2A_h5path, "r") as dataset:
                    for idx, row in tqdm(self.index.iterrows(), desc="Loading data into RAM", total=len(self.index)):
                        self.L2A_X_list.append(np.array(dataset[row.path]))
        else:
            self.L2A_X_list = None

        # Retrieve crop codes
        self.get_codes()

    def build_folder_structure(self, root, year, level, region):
        """
        Defines and ensures the folder structure for the dataset.

        The expected directory structure:

        <root>                          # Root directory of the dataset
        ├── codes.csv                   # Mapping of crop types
        ├── classmapping.csv             # Maps crop types to classification labels
        ├── sregadiumapping.csv          # Maps irrigation systems
        ├── varietatmapping.csv          # Maps crop varieties
        ├── municipismapping.csv         # Maps municipalities
        ├── <year>                       # Dataset year (e.g., 2022, 2023, ...)
        │   ├── <region>.shp             # Shapefile containing parcel boundaries
        │   ├── <region>_d.shp           # Parcel metadata shapefile (e.g., irrigation, elevation)
        │   ├── <level>                  # Processing level (L2A)
        │   │   ├── <region>.csv         # CSV file linking parcels to time series
        │   │   ├── <region>.h5          # HDF5 file storing compressed time series data
        │   │   ├── <region>             # Folder containing time series data for each parcel
        │   │   │   ├── <csv>            # Subdirectory for individual time series
        │   │   │   │   ├── 123123.csv   # Time series table for parcel 123123
        │   │   │   │   ├── 123125.csv   # Time series table for parcel 123125
        │   │   │   │   ├── ...

        Args:
            root (str): Root directory of the dataset.
            year (int or str): Year of the dataset (e.g., "2022", "2023").
            level (str): Processing level of the dataset (e.g., "L1C" or "L2A").
            region (str): Dataset region (e.g., "baixter" or "lleida").

        Returns:
            tuple: Paths to essential dataset files and directories:
                - indexfile (str): Path to the index CSV file.
                - codesfile (str): Path to the crop codes file.
                - shapefile (str): Path to the shapefile containing parcel geometries.
                - classmapping (str): Path to the classification mapping file.
                - municipismapping (str): Path to the municipality mapping file.
                - sregadiumapping (str): Path to the irrigation system mapping file.
                - varietatmapping (str): Path to the crop variety mapping file.
                - L2A_csvfolder (str): Path to the CSV folder containing time series data.
                - L2A_h5path (str): Path to the HDF5 file storing compressed time series data.
        """
        year = str(year)  # Ensure year is treated as a string

        # Create necessary directories if they don't exist
        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        # Define file paths for different dataset components
        indexfile = os.path.join(root, year, level, region + ".csv")  # Path to parcel index file
        codesfile = os.path.join(root, "codes.csv")  # Crop codes mapping
        shapefile = os.path.join(root, year, f"{region}.shp")  # Shapefile with parcel boundaries
        classmapping = os.path.join(root, "classmapping.csv")  # Crop type to classification mapping
        municipismapping = os.path.join(root, "municipismapping.csv")  # Municipality mapping
        sregadiumapping = os.path.join(root, "sregadiumapping.csv")  # Irrigation system mapping
        varietatmapping = os.path.join(root, "varietatmapping.csv")  # Crop variety mapping

        # Time series storage paths
        L2A_csvfolder = os.path.join(root, year, level, region, "csv")  # CSV time series folder
        L2A_h5path = os.path.join(root, year, level, f"{region}.h5")  # HDF5 storage file

        return indexfile, codesfile, shapefile, classmapping, municipismapping, \
               sregadiumapping, varietatmapping, L2A_csvfolder, L2A_h5path

    def get_fid(self, idx):
        """
        Retrieves the unique field ID associated with a given dataset index.

        Args:
            idx (int): Index of the sample.

        Returns:
            int: The corresponding field ID (parcel identifier).
        """
        return self.index[self.index["idx"] == idx].index[0]

    def write_h5_from_csv(self, index, h5path, sat):
        """
        Writes time series data from CSV files into an HDF5 file.

        This function reads individual time series files (CSV format) and stores them
        into a compressed HDF5 format for efficient data retrieval.

        Args:
            index (pd.DataFrame): Dataframe containing parcel references.
            h5path (str): Path to the output HDF5 file.
            sat (str): Satellite data type (e.g., "L2A").

        Returns:
            None: Creates an HDF5 file containing time series data.
        """
        with h5py.File(h5path, "w") as dataset:
            for idx, row in tqdm(index.iterrows(), total=len(index), desc=f"writing {h5path}"):
                # Construct the CSV file path
                csv_file = os.path.join(self.root, str(self.year), sat, self.region, "csv", str(row.id)+".csv")
                # Load the CSV file into an array
                x = self.load(csv_file, sat)
                # Only store non-empty time series in the HDF5 file
                if len(x) > 0:
                    dataset.create_dataset(csv_file, data=x)

    def get_codes(self):
        """
        Retrieves the crop codes dataset.

        Returns:
            pd.DataFrame: Dataframe containing crop type codes.
        """
        return self.codes

    def geodataframe(self):
        """
        Generates a GeoDataFrame containing parcel geometries and metadata.

        This function reads a shapefile and merges it with the dataset's metadata,
        ensuring that each parcel has an associated geometry.

        Returns:
            gpd.GeoDataFrame: A geospatial dataset containing parcel geometries and attributes.
            None: If the shapefile does not exist, prints a warning message.
        """
        if not os.path.exists(self.shapefile):
            # self.download_geodataframe()
            print("No existeix l'arxiu shapefile")
            return

        # Convert the dataset index into a GeoDataFrame
        geodataframe = gpd.GeoDataFrame(self.index.set_index("id"))

        # Load the shapefile containing parcel geometries
        gdf = gpd.read_file(self.shapefile)

        # If the shapefile uses "ID_PARCEL" instead of "ID", rename it
        gdf = gdf.rename(columns={"ID_PARCEL": "ID"})

        # Copy geometry information from the shapefile to the dataset index
        geom = gdf.set_index("ID")
        geom.index.name = "id"
        geodataframe["geometry"] = geom["geometry"]
        geodataframe.crs = geom.crs  # Assign the correct coordinate reference system

        return geodataframe.reset_index()

    def load_classmapping(self, classmapping):
        """
            Loads the class mapping file that maps crop types to classification labels.

            Args:
                classmapping (str): Path to the `classmapping.csv` file.

            Returns:
                None: Updates the dataset's internal mapping structure.
            """
        if not os.path.exists(classmapping):
            print(classmapping)
            print("Falta l'arxiu 'classmapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        # Read and process the class mapping file
        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")

        # Extract unique class IDs and corresponding names
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        self.max_classid = self.mapping.id.max()
        self.max_crop = self.mapping.axes[0].max()
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def load_municipismapping(self, municipismapping):
        """
        Loads the municipality mapping file, which links municipalities to regions.

        Args:
            municipismapping (str): Path to the `municipismapping.csv` file.

        Returns:
            None: Updates the dataset's municipality mapping structure.
        """
        if not os.path.exists(municipismapping):
            print("Falta l'arxiu 'municipismapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found municipismapping at {municipismapping}")

        # Read and process the municipality mapping file
        self.muni_mapping = pd.read_csv(municipismapping, index_col=0).sort_values(by="muni_code")
        self.max_mun = self.muni_mapping.axes[0].max()
        self.min_mun = self.muni_mapping.axes[0].min()
        self.max_com = self.muni_mapping.com_code.max()
        self.max_prov = self.muni_mapping.prov_code.max()

        if self.verbose:
            print(f"read municipis from {municipismapping}")

    def load_sregadiumapping(self, sregadiumapping):
        """
        Loads the irrigation system mapping file.

        Args:
            sregadiumapping (str): Path to the `sregadiumapping.csv` file.

        Returns:
            None: Updates the dataset's irrigation system mapping structure.
        """
        if not os.path.exists(sregadiumapping):
            print("Falta l'arxiu 'sregadiumapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found sregadiumapping at {sregadiumapping}")

        self.reg_mapping = pd.read_csv(sregadiumapping, index_col=0).sort_values(by="sreg_code")
        self.max_sreg = self.reg_mapping.axes[0].max()

        if self.verbose:
            print(f"read reg from {sregadiumapping}")

    def load_varietatmapping(self, varietatmapping):
        """
        Loads the crop variety mapping file.

        Args:
            varietatmapping (str): Path to the `varietatmapping.csv` file.

        Returns:
            None: Updates the dataset's crop variety mapping structure.
        """
        if not os.path.exists(varietatmapping):
            print("Falta l'arxiu 'varietatmapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found varietatmapping at {varietatmapping}")

        self.var_mapping = pd.read_csv(varietatmapping, index_col=0).sort_values(by="vari_code")
        self.max_var = self.var_mapping.axes[0].max()

        if self.verbose:
            print(f"read varietats from {varietatmapping}")

    def load_raw(self, csv_file):
        """
        Loads a raw CSV file containing time series data for a given parcel.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed time series data.
        """
        sample = pd.read_csv(csv_file).dropna()

        # Convert date to integer format
        sample["doa"] = pd.to_datetime(sample["doa"]).astype('int64').astype(int)
        sample = sample.groupby(by="doa").first().reset_index()
        return sample

    def load(self, csv_file, sat):
        """
        Loads and processes time series data from a CSV file.

        This function reads a CSV file containing time series data for a specific parcel.
        It selects the relevant spectral bands and ensures there are no missing values.

        Args:
            csv_file (str): Path to the CSV file containing time series data.
            sat (str): Satellite data type (e.g., "L2A").

        Returns:
            np.array: Numpy array containing the selected time series bands.
        """
        # Attempt to load the CSV file using the load_raw function
        try:
            sample = self.load_raw(csv_file)
        except ValueError:
            # If the file does not exist or cannot be loaded, print an error message
            print("File " + csv_file + " not found.")
            sample = []

        # Retrieve the list of spectral bands to be used from the dataset settings
        selected_bands = SELECTED_BANDS[sat]

        # Convert the selected bands into a NumPy array
        x = np.array(sample[selected_bands].values)

        # Check if the file is empty
        if len(sample)==0:
            print("Empty " + csv_file + " file.")

        # Handle missing values (NaNs) in the dataset
        if np.isnan(x).any():
            t_without_nans = np.isnan(x).sum(1) > 0  # Identify rows that contain NaN values
            x = x[~t_without_nans]  # Remove rows with NaN values from the dataset

        return x

    def load_culturecode_and_id(self, csv_file):
        """
        Loads the crop class (culture code) and field ID from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            tuple: (culture_code, field_id), or (None, None) if empty.
        """
        sample = self.load_raw(csv_file)
        if len(sample) > 0:
            field_id = sample["id"].iloc[0]
            culture_code = sample["Label"].iloc[0]
            return culture_code, field_id
        else:
            return None, None

    def __len__(self):
        """
        Returns the number of parcels in the dataset.

        Returns:
            int: Number of samples (parcels).
        """
        return len(self.index)

    def __getitem__(self, index):
        """
        Retrieves a time series sample and its corresponding label from the dataset.

        This function loads the spectral time series (if available), normalizes metadata fields,
        and applies transformations before returning the processed sample.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed time series data, label, parcel ID)
        """

        # Retrieve the row corresponding to the given index
        row = self.index.iloc[index]

        # Load Sentinel-2 (L2A) time series data
        if self.L2A:
            if self.L2A_X_list is None:  # Load from HDF5 file if not preloaded in RAM
                with h5py.File(self.L2A_h5path, "r") as dataset:
                    L2A_X = np.array(dataset[row.path])
            else:  # Retrieve from preloaded data in RAM
                L2A_X = self.L2A_X_list[index]
        else:
            L2A_X = None

        # Normalize previous class ID
        if self.pclassid:
            if not np.isnan(row.pcrop_code):
                pclassid_val = self.mapping.loc[row.pcrop_code].id / self.max_classid
            else:
                pclassid_val = -1  # Assign -1 if value is missing
        else:
            pclassid_val = None

        # Normalize previous crop code
        if self.pcrop:
            if not np.isnan(row.pcrop_code):
                pcrop_val = row.pcrop_code / self.max_crop
            else:
                pcrop_val = -1  # Assign -1 if value is missing
        else:
            pcrop_val = None

        # Normalize previous variety code
        if self.pvar:
            if not np.isnan(row.pvar_code):
                pvar_val = row.pvar_code / self.max_var
            else:
                pvar_val = -1  # Assign -1 if value is missing
        else:
            pvar_val = None

        # Normalize additional metadata features
        sreg_val = row.sreg_code / self.max_sreg if self.sreg else None  # Irrigation system code
        mun_val = (row.mun_code - self.min_mun) / (self.max_mun - self.min_mun) if self.mun else None  # Municipality code
        com_val = self.muni_mapping.loc[row.mun_code].com_code / self.max_com if self.com else None  # Comarca (region) code
        prov_val = self.muni_mapping.loc[row.mun_code].prov_code / self.max_prov if self.prov else None  # Province code
        elev_val = row.elev / self.max_elev if self.elev else None  # Elevation normalization
        slope_val = row.slope / self.max_slope if self.slope else None  # Slope normalization

        # Get the class label from the crop mapping file
        y = self.mapping.loc[row["CODE_CULTU"]].id

        # Apply transformations to the data
        ts_data = self.transform(L2A_X, pclassid_val, pcrop_val, pvar_val, sreg_val, mun_val, com_val,
                                 prov_val, elev_val, slope_val)

        # Apply target transformation if defined
        if self.target_transform is not None:
            y = self.target_transform(y)

        # Return transformed data, label, and parcel ID
        return ts_data, y, row.id

    def write_index(self):
        """
        Creates an index file for the dataset.

        This function processes all CSV files in the dataset folder, extracts metadata
        (such as cloud coverage, parcel ID, and crop code), and compiles an index file
        that stores references to each time series.

        The generated index file includes:
        - `meanQA60`: Average cloud coverage percentage for each parcel.
        - `id`: Unique identifier of the parcel.
        - `CODE_CULTU`: Crop type code.
        - `path`: File path to the corresponding time series CSV file.
        - `idx`: Unique index for each entry.
        - `sequencelength`: Number of time steps in the time series.

        Returns:
            None: Saves the index to a CSV file.
        """

        # Retrieve the list of CSV files from the dataset folder
        csv_files = os.listdir(self.L2A_csvfolder)

        # List to store statistical data extracted from the time series files
        listcsv_statistics = list()

        i = 1  # Initialize index counter

        print("CSV index file is missing or needs to be generated")

        # Iterate through all time series CSV files
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):

            # Get the column index for cloud probability (CP)
            cld_index = SELECTED_BANDS["L2A"].index("CP")

            # Load the time series data
            X = self.load(os.path.join(self.L2A_csvfolder, csv_file), "L2A")

            # Extract crop code and parcel ID
            culturecode, id = self.load_culturecode_and_id(os.path.join(self.L2A_csvfolder, csv_file))

            # Skip entries with missing crop codes or parcel IDs
            if culturecode is None or id is None:
                continue

            # Append extracted metadata to the index list
            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),  # Compute mean cloud coverage for the parcel
                    id=id,  # Parcel identifier
                    CODE_CULTU=culturecode,  # Crop type code
                    path=os.path.join(self.L2A_csvfolder, f"{id}.csv"),  # Path to time series file
                    idx=i,  # Unique index
                    sequencelength=len(X)  # Length of the time series
                )
            )
            i += 1  # Increment index counter

        # Convert the list to a DataFrame and save it as a CSV index file
        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)

def get_default_target_transform():
    """
    Returns the default target transformation function.

    This function converts the class labels (crop type IDs) into PyTorch tensors
    with `long` datatype, which is required for classification tasks.

    Returns:
        callable: A lambda function that converts an integer label to a PyTorch tensor.
    """
    return lambda y: torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    """
    If this script is executed directly (instead of being imported), it initializes 
    a CatCrops dataset instance using default parameters.

    This is useful for testing dataset loading and preprocessing.
    """
    CatCrops(region="lleida", root="./catcrops_dataset/", year=2023, preload_ram=False, L2A=True)

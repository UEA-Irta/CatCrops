# -*- coding: utf-8 -*-
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

BANDS = {
    "L2A": ['nom_id_imatge', 'B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'QA10', 'QA20',
            'QA60', 'CP', 'doa', 'Label', 'id'],
    # "LST": ['nom_id_imatge', 'LST_B1', 'LST_B2', 'LST_B3', 'LST_B4',
    #         'LST_B5', 'LST_B6', 'LST_B7', 'doa', 'Label', 'id'],
    # "ET": ['doa', 'ET']
}

SELECTED_BANDS = {
    "L2A": ['doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CP'],
    # "LST": ['doa', 'LST_B1', 'LST_B2', 'LST_B3', 'LST_B4', 'LST_B5', 'LST_B6', 'LST_B7'],
    # "ET": ['doa', 'ET']
}


class CatCrops(Dataset):

    def __init__(self,
                 region,
                 root="./catcrops_dataset/",  #TODO on posar la ubicació del dataset
                 year=2022,
                 transform=None,
                 target_transform=None,
                 filter_length=0,
                 verbose=False,
                 recompile_h5_from_csv=False,
                 preload_ram=False,
                 L2A = True,                  #Use Sentinell data:            True or False
                 # LST = False,                 #Use Landsat data:              True or False
                 # ET=False,                    #Use ET data:                   True or False
                 pclassid=False,              #Use previous year class:       True or False
                 pcrop=False,                 #Use previous crop code:        True or False
                 pvar=False,                  #Use previous variety code:     True or False
                 sreg=False,                  #Use irrigation system:         True or False
                 mun=False,                   #Use municipality code:         True or False
                 com=False,                   #Use comarca code:              True or False
                 prov=False,                  #Use province code:             True or False
                 elev=False,                  #Use elevation of the field:    True or False
                 slope=False):                #Use slope of the field:        True or False
        """
        :param region: dataset region. choose from "baixter", "lleida"
        :param root: where the data will be stored. defaults to `./catcrops_dataset`
        :param year: year of the data. currently only `2022`
        :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        :param target_transform: a transformation function applied to the label.
        :param filter_length: time series shorter than `filter_length` will be ignored
        :param bool verbose: verbosity flag
        :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.
        """

        level = "L2A"
        assert year in [2022, 2023]
        assert region in ["lleida", "baixter"]

        if transform is None:
            transform = get_transform_CatCrops()
        if target_transform is None:
            target_transform = get_default_target_transform()
        self.transform = transform
        self.target_transform = target_transform

        self.region = region.lower()
        self.L2A_bands = BANDS["L2A"]
        # self.LST_bands = BANDS["LST"]
        # self.ET_bands = BANDS["ET"]

        self.verbose = verbose
        self.year = year
        self.level = level

        self.L2A = L2A
        # self.LST = LST
        # self.ET = ET
        self.pclassid = pclassid
        self.pcrop = pcrop
        self.pvar = pvar
        self.sreg = sreg
        self.mun = mun
        self.com = com
        self.prov = prov
        self.elev = elev
        self.slope = slope
        self.max_elev = 1500
        self.max_slope = 54

        if verbose:
            print(f"Initializing CatCrops region {region}, year {year}")

        self.root = root
        self.indexfile, self.codesfile, self.shapefile, self.classmapping, self.municipismapping, self.sregadiumapping,\
        self.varietatmapping, self.L2A_csvfolder, self.L2A_h5path = self.build_folder_structure(self.root, self.year, self.level, self.region)

        self.load_classmapping(self.classmapping)
        self.load_municipismapping(self.municipismapping)
        self.load_sregadiumapping(self.sregadiumapping)
        self.load_varietatmapping(self.varietatmapping)

        if not os.path.exists(self.indexfile):
            self.write_index()

        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        if self.index.index.name != "idx":
            self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            print("Falta l'arxiu 'codes.csv'")
            exit()
        self.codes = pd.read_csv(self.codesfile, delimiter=",", index_col=0)


        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)
        self.index_geo_file = os.path.join(root, str(year), region + "_d.csv")
        self.index_geo = pd.read_csv(self.index_geo_file, delimiter=",", index_col=0)
        if "classid" not in self.index.columns or "classname" not in self.index.columns or "region" not in self.index.columns:
            # drop fields that are not in the class mapping
            self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
            self.index[["classid", "classname"]] = self.index["CODE_CULTU"].apply(lambda code: self.mapping.loc[code])
            self.index["region"] = self.region
            self.index = self.index.loc[self.index["id"].isin(self.index_geo.index)]
            self.index[["var_code", "pcrop_code", "pvar_code", "sreg_code", "mun_code","elev","slope"]] = self.index["id"].apply(lambda ID: self.index_geo.loc[ID]).iloc[:,1:]
            self.index.to_csv(self.indexfile)

        # Compile h5 file in case that it not exist or it is required to be recompiled:
        if L2A and (not os.path.exists(self.L2A_h5path) or recompile_h5_from_csv):
            print("h5_database file is missing or is asked to be recompiled")
            self.write_h5_from_csv(self.index, self.L2A_h5path,"L2A")
        # if LST and (not os.path.exists(self.LST_h5path) or recompile_h5_from_csv):
        #     print("h5_database file is missing or is asked to be recompiled")
        #     self.write_h5_from_csv(self.index, self.LST_h5path, "LST")
        # if ET and (not os.path.exists(self.ET_h5path) or recompile_h5_from_csv):
        #     print("h5_ET file is missing or is asked to be recompiled")
        #     self.write_h5_from_csv(self.index, self.ET_h5path, "ET")

        if preload_ram:
            if self.L2A:
                self.L2A_X_list = list()
                with h5py.File(self.L2A_h5path, "r") as dataset:
                    for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
                        self.L2A_X_list.append(np.array(dataset[row.path]))
            # if self.LST:
            #     self.LST_X_list = list()
            #     with h5py.File(self.LST_h5path, "r") as LST_dataset:
            #         for idx, row in tqdm(self.index.iterrows(), desc="loading LST into RAM", total=len(self.index)):
            #             csv_file = os.path.join(self.root, str(self.year), "LST", self.region, "csv", str(row.id) + ".csv")
            #             self.LST_X_list.append(np.array(LST_dataset[csv_file]))
            # if self.ET:
            #     self.ET_X_list = list()
            #     with h5py.File(self.ET_h5path, "r") as ET_dataset:
            #         for idx, row in tqdm(self.index.iterrows(), desc="loading ET into RAM", total=len(self.index)):
            #             csv_file = os.path.join(self.root, str(self.year), "ET", self.region, "csv", str(row.id) + ".csv")
            #             if csv_file in ET_dataset:
            #                 ET_X = np.array(ET_dataset[csv_file])
            #                 if ET_X.shape[0] == 0:
            #                     ET_X = np.array([[datetime(self.year, 1, 1).timestamp() * 1e9, -1]])
            #             else:
            #                 ET_X = np.array([[datetime(self.year, 1, 1).timestamp()*1e9, -1]])
            #             self.ET_X_list.append(ET_X)

        else:
            self.L2A_X_list = None
            # self.LST_X_list = None
            # self.ET_X_list = None

        self.get_codes()

    def build_folder_structure(self, root, year, level, region):
        """
        folder structure

        <root>
           codes.csv
           classmapping.csv
           <year>
              <region>.shp
              <ET>
                 <region>.h5
                     <csv>
                         123123.csv
                         123125.csv
                         ...
              <level>
                 <region>.csv
                 <region>.h5
                 <region>
                     <csv>
                         123123.csv
                         123125.csv
                         ...
        """
        year = str(year)

        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        indexfile = os.path.join(root, year, level, region + ".csv")
        codesfile = os.path.join(root, "codes.csv")
        shapefile = os.path.join(root, year, f"{region}.shp")
        classmapping = os.path.join(root, "classmapping.csv")
        municipismapping = os.path.join(root, "municipismapping.csv")
        sregadiumapping = os.path.join(root, "sregadiumapping.csv")
        varietatmapping = os.path.join(root, "varietatmapping.csv")
        L2A_csvfolder = os.path.join(root, year, level, region, "csv")
        L2A_h5path = os.path.join(root, year, level, f"{region}.h5")
        # LST_csvfolder = os.path.join(root, year, "LST", region, "csv")
        # LST_h5path = os.path.join(root, year, "LST", f"{region}.h5")
        # ET_csvfolder = os.path.join(root, year, "ET", region, "csv")
        # ET_h5path = os.path.join(root, year, "ET", f"{region}.h5")
        return indexfile, codesfile, shapefile, classmapping, municipismapping, sregadiumapping, varietatmapping, L2A_csvfolder, L2A_h5path
        # return indexfile, codesfile, shapefile, classmapping, municipismapping, sregadiumapping, varietatmapping, L2A_csvfolder, L2A_h5path, LST_csvfolder, LST_h5path, ET_csvfolder, ET_h5path

    def get_fid(self, idx):
        return self.index[self.index["idx"] == idx].index[0]

    def write_h5_from_csv(self, index, h5path, sat):
        with h5py.File(h5path, "w") as dataset:
            for idx, row in tqdm(index.iterrows(), total=len(index), desc=f"writing {h5path}"):
                csv_file = os.path.join(self.root, str(self.year), sat, self.region, "csv", str(row.id)+".csv")
                x = self.load(csv_file, sat)
                if len(x) > 0:
                    dataset.create_dataset(csv_file, data=x)
    def get_codes(self):
        return self.codes

    def geodataframe(self):
        if not os.path.exists(self.shapefile):
            # self.download_geodataframe()
            print("No existeix l'arxiu shapefile")
            return

        geodataframe = gpd.GeoDataFrame(self.index.set_index("id"))

        gdf = gpd.read_file(self.shapefile)

        # 2018 shapefile calls ID ID_PARCEL: rename if necessary
        gdf = gdf.rename(columns={"ID_PARCEL": "ID"})

        # copy geometry from shapefile to index file
        geom = gdf.set_index("ID")
        geom.index.name = "id"
        geodataframe["geometry"] = geom["geometry"]
        # geodataframe.set_geometry(geom["geometry"], inplace=True)
        geodataframe.crs = geom.crs

        return geodataframe.reset_index()

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            print(classmapping)
            print("Falta l'arxiu 'classmapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        self.max_classid = self.mapping.id.max()
        self.max_crop = self.mapping.axes[0].max()
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def load_municipismapping(self, municipismapping):
        if not os.path.exists(municipismapping):
            print("Falta l'arxiu 'municipismapping.csv'")
            exit()
        else:
            if self.verbose:
                print(f"found municipismapping at {municipismapping}")
        self.muni_mapping = pd.read_csv(municipismapping, index_col=0).sort_values(by="muni_code")
        self.max_mun = self.muni_mapping.axes[0].max()
        self.min_mun = self.muni_mapping.axes[0].min()
        self.max_com = self.muni_mapping.com_code.max()
        self.max_prov = self.muni_mapping.prov_code.max()

        if self.verbose:
            print(f"read municipis from {municipismapping}")

    def load_sregadiumapping(self, sregadiumapping):
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
        sample = pd.read_csv(csv_file).dropna()
        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype('int64').astype(int)
        sample = sample.groupby(by="doa").first().reset_index()
        return sample

    def load(self, csv_file, sat):
        try:
            sample = self.load_raw(csv_file)
        except ValueError:
            print("File " + csv_file + " not found.")
            sample = []
        selected_bands = SELECTED_BANDS[sat]
        x = np.array(sample[selected_bands].values)
        if len(sample)==0:
            print("Empty " + csv_file + " file.")
        if np.isnan(x).any():
            t_without_nans = np.isnan(x).sum(1) > 0
            x = x[~t_without_nans]
        return x

    def load_culturecode_and_id(self, csv_file):
        sample = self.load_raw(csv_file)
        if len(sample) > 0:
            field_id = sample["id"].iloc[0]
            culture_code = sample["Label"].iloc[0]
            return culture_code, field_id
        else:
            return None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]
        if self.L2A:
            if self.L2A_X_list is None:
                with h5py.File(self.L2A_h5path, "r") as dataset:
                    L2A_X = np.array(dataset[row.path])
            else:
                L2A_X = self.L2A_X_list[index]
        else:
            L2A_X = None

        # if self.LST:
        #     if self.LST_X_list is None:
        #         csv_file = os.path.join(self.root, str(self.year), "LST", self.region, "csv", str(row.id) + ".csv")
        #         with h5py.File(self.LST_h5path,"r") as LST_dataset:
        #             LST_X = np.array(LST_dataset[(csv_file)])
        #     else:
        #         LST_X = self.LST_X_list[index]
        # else:
        #     LST_X = None

        # if self.ET:
        #     if self.ET_X_list is None:
        #         csv_file = os.path.join(self.root, str(self.year), "ET", self.region, "csv", str(row.id) + ".csv")
        #         with h5py.File(self.ET_h5path,"r") as ET_dataset:
        #             if csv_file in ET_dataset:
        #                 ET_X = np.array(ET_dataset[(csv_file)])
        #                 if ET_X.shape[0]==0:
        #                     ET_X = np.array([[datetime(self.year, 1, 1).timestamp() * 1e9, -1]])
        #             else:
        #                 #ET_X = np.empty((0, 2))
        #                 ET_X = np.array([[datetime(self.year,1,1).timestamp()*1e9, -1]])
        #     else:
        #         ET_X = self.ET_X_list[index]
        # else:
        #     ET_X = None

        if self.pclassid:
            if not np.isnan(row.pcrop_code):
                pclassid_val = self.mapping.loc[row.pcrop_code].id/self.max_classid
            else:
                pclassid_val = -1
        else:
            pclassid_val = None
        if self.pcrop:
            if not np.isnan(row.pcrop_code):
                pcrop_val = row.pcrop_code / self.max_crop
            else:
                pcrop_val = -1
        else:
            pcrop_val = None

        if self.pvar:
            if not np.isnan(row.pvar_code):
                pvar_val = row.pvar_code/ self.max_var
            else:
                pvar_val = -1
        else:
            pvar_val = None
        sreg_val = row.sreg_code/self.max_sreg if self.sreg else None
        mun_val = (row.mun_code - self.min_mun) / (self.max_mun - self.min_mun) if self.mun else None
        com_val = self.muni_mapping.loc[row.mun_code].com_code/self.max_com if self.com else None
        prov_val = self.muni_mapping.loc[row.mun_code].prov_code/self.max_prov if self.prov else None
        elev_val = row.elev/self.max_elev if self.elev  else None
        slope_val = row.slope/self.max_slope if self.slope else None

        y = self.mapping.loc[row["CODE_CULTU"]].id

        ts_data = self.transform(L2A_X, pclassid_val, pcrop_val, pvar_val, sreg_val, mun_val, com_val,
                                 prov_val, elev_val, slope_val)
        # ts_data = self.transform(L2A_X, LST_X, ET_X, pclassid_val, pcrop_val, pvar_val, sreg_val, mun_val, com_val, prov_val, elev_val, slope_val)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return ts_data, y, row.id

    def write_index(self):
        csv_files = os.listdir(self.L2A_csvfolder)
        listcsv_statistics = list()
        i = 1
        print("csv file is missing or is asked to be build")

        for csv_file in tqdm(csv_files):
            cld_index = SELECTED_BANDS["L2A"].index("CP")

            X = self.load(os.path.join(self.L2A_csvfolder, csv_file), "L2A")
            culturecode, id = self.load_culturecode_and_id(os.path.join(self.L2A_csvfolder, csv_file))

            if culturecode is None or id is None:
                continue

            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),
                    id=id,
                    CODE_CULTU=culturecode,
                    path=os.path.join(self.L2A_csvfolder, f"{id}" + ".csv"),
                    idx=i,
                    sequencelength=len(X)
                )
            )
            i += 1

        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)

def get_default_target_transform():
    return lambda y: torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    CatCrops(region="lleida", root="./catcrops_dataset/", year=2023, preload_ram=False, L2A=True, ET=True)

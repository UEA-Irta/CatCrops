# -*- coding: utf-8 -*-
"""
Script per generar l'arxiu h5 a partir dels arxius csv.

S'utilitza després de baixar noves dates de Sentinel2.

author: magipamies
datetime:26/6/2023 10:02
"""

import argparse
from catcrops import CatCrops
from os import listdir
from os import path as pth

## VARIABLES PER DEFECTE I LLISTES
# path del dataset
datapath_d= r"/home/usuari11/Documents/catcrops/catcrops_dataset"
# Llista dels anys
list_year = sorted([x for x in listdir(datapath_d) if pth.isdir(pth.join(datapath_d, x))])
# Llista de les zones (all és per fer-les totes)
list_ae = ['all'] + sorted(list(set([i for r in [[x.split('.')[0] for x in listdir(pth.join(datapath_d, y)) if
                                                  '.shp' == pth.splitext(x)[1]] for y in list_year] for i in r])))
year_d = list_year[-1]  # any per defecte
region_d = list_ae[0]  # Regió per defecte


# Configurem el parser
parser = argparse.ArgumentParser(
    description=r"Script per recompilar l'arxiu h5 a partir dels arxius csv pel dataset CatCrops. Si no seleccionem"
                r"cap de les variables, les seleccionarà totes.",
    epilog="Això és l'epíleg i no sé perquè serveix")

parser.add_argument("-Z", "--zona", choices=list_ae, default=region_d,
                    help="nom de la zona d'estudi. Si posem 'all' les fa totes. Default: %s" % region_d)
parser.add_argument("-A", "--any", choices=list_year, default=year_d,
                    help="Any del dataset. Default: %s" % year_d)
parser.add_argument("-D", "--datapath", type=str, default=datapath_d,
                    help="'directory to download and store the dataset'. Default: %s" % datapath_d)
parser.add_argument('--L2A', action="store_true",
                    help='A bolean argument to use the spectral data from L2A level in the timeseries (Default is false)')

args = parser.parse_args()

# Variables d'entrada
list_region = [x for x in list_ae[1:] if x == args.zona or args.zona=='all']  # Llista de zones
year = int(args.year)  # Any
datapath = args.datapath  # Path del dataset
L2A = args.L2A,  # Use Sentinell data:            True or False
# Li diem que si no seleccionem cap de les tres variables que les seleccioni les tres.

# Recompilem l'arxiu h5
region = list_region[0]
for region in list_region:
    print(f'Region: {region} Year: {year} L2A: {L2A} \bDatapath: {datapath}')
    CatCrops(region=region, root=datapath, year=year, preload_ram=True, recompile_h5_from_csv=True, L2A=L2A)



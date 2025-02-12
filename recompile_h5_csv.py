# -*- coding: utf-8 -*-
"""
Script per generar l'arxiu h5 a partir dels arxius csv.

S'utilitza després de baixar noves dates de Sentinel2.

author: magipamies
datetime:26/6/2023 10:02
"""


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
year_d = 2022  # any per defecte
zona = 'baixter'
# zona = 'all'

# Variables d'entrada
list_region = [x for x in list_ae[1:] if x == zona or zona=='all']  # Llista de zones
year = int(year_d)  # Any
datapath = datapath_d  # Path del dataset
L2A = True,  # Use Sentinell data:            True or False

# Li diem que si no seleccionem cap de les tres variables que les seleccioni les tres.

# Recompilem l'arxiu h5
region = list_region[0]
for region in list_region:
    print(f'Region: {region} Year: {year} L2A: {L2A} \bDatapath: {datapath}')
    CatCrops(region=region, root=datapath, year=year, preload_ram=True, recompile_h5_from_csv=True, L2A=L2A)



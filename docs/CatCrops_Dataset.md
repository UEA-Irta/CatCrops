# CatCrops Dataset
This document describes the structure of the **CatCrops** dataset, including file organization and data descriptions.

The **CatCrops Dataset** is hosted on Kaggle and can be accessed through the [CatCrops Dataset on Kaggle](https://www.kaggle.com/datasets/irtaremotesensing/catcrops-dataset)

## Dataset Folder Structure

The dataset follows this directory structure:

```tree
<root>                           # Root directory of the dataset
│── codes.csv                    # Mapping of crop types with codes, descriptions, and groups
│── classmapping.csv             # Maps crop types to database classification types
│── sregadiumapping.csv          # Maps irrigation system codes to descriptions
│── varietatmapping.csv          # Maps crop variety codes to descriptions
│── municipismapping.csv         # Maps municipalities to their region and province
│── <year>                       # Directory for each dataset year (e.g., 2022, 2023, ...)
│   ├── <region>.shp             # Shapefile containing parcel boundaries and crop type information
│   ├── <region>_d.csv           # CSV file with additional parcel metadata (e.g., irrigation, elevation)
│   ├── <level>                  # Sentinel-2 processing level (L2A)
│   │   ├── <region>.csv         # Summary CSV file linking parcel IDs to time series files
│   │   ├── <region>.h5          # HDF5 file storing compressed time series data for the region
│   │   ├── <region>             # Folder containing time series data for each parcel
│   │   │   ├── <csv>            # Subdirectory containing individual time series files
│   │   │   │   ├── 123123.csv   # Time Series Table for parcel 123123
│   │   │   │   ├── 123125.csv   # Time Series Table for parcel 123125
│   │   │   │   ├── ...
```

## Description of Dataset Files

### **codes.csv**
Located at `<root>/`.
Each row defines a different crop type according to the DUN classification. It contains four columns:
- `$crop_DUN_code`: Primary key representing the crop type code. It corresponds to the numerical DUN crop code.
- `$crop_DUN_description`: Crop type description, e.g., "barley", "wheat", or "corn". In this case, the DUN name is used.
- `$grup_DUN_code`: Code for crop groups. Different crops can be grouped under the same category. This field contains a numerical ID assigned by the DUN.
- `$grup_DUN_description`: Description of the DUN crop group.

### **classmapping.csv**
Located at `<root>/`.
Each row defines a crop type and maps it to the classification used in the database. It contains four columns:
- `$`: Unnamed header. This column represents the row number.
- `$id`: Class ID corresponding to the crop type in `$code`.
- `$classname`: Class name, e.g., "barley", "wheat", "corn", "permanent crop".
- `$code`: Foreign key linking to `$crop_DUN_code` in `codes.csv`.

### **sregadiumapping.csv**
Located at `<root>/`.
Each row defines an irrigation system and its corresponding name. It contains two columns:
- `$sreg_code`: Code representing the irrigation system.
- `$sreg_description`: Name of the irrigation system.

### **varietatmapping.csv**
Located at `<root>/`.
Each row defines a crop variety and its corresponding name. It contains two columns:
- `$vari_code`: Code representing the crop variety.
- `$vari_description`: Name of the crop variety (DUN name).

### **municipismapping.csv**
Located at `<root>/`.
Each row defines a municipality and links it to its respective comarca (county) and province. It contains six columns:
- `$mun_code`: Municipality code.
- `$mun_description`: Municipality name.
- `$com_code`: County code.
- `$com_description`: County name.
- `$prov_code`: Province code.
- `$prov_description`: Province name.

### **\<region>.shp**
Located at `<root>/<year>/`.
A shapefile containing the polygons of all parcels. Each row represents a parcel with the following columns:
- `$ID`: Parcel identifier (primary key).
- `$crop_code`: Crop type code, a foreign key linked to `$crop_DUN_code` in `codes.csv`.
- `$geometry`: Polygon defining the parcel boundary.

### **\<region>_d.csv**
Located at `<root>/<year>/`.
A CSV file containing detailed metadata for each parcel. Each row represents a parcel and contains the following columns:
- `$ID`: Parcel identifier (primary key).
- `$crop_code`: Crop type code, linked to `$crop_DUN_code` in `codes.csv`.
- `$var_code`: Crop variety code, linked to `$vari_code` in `varietatmapping.csv`.
- `$pcrop_code`: Previous year's crop type code, linked to `$crop_DUN_code` in `codes.csv`.
- `$pvar_code`: Previous year's crop variety code, linked to `$vari_code` in `varietatmapping.csv`.
- `$sreg_code`: Irrigation system code, linked to `$sreg_code` in `sregadiumapping.csv`.
- `$mun_code`: Municipality code, linked to `$mun_code` in `municipismapping.csv`.
- `$elev`: Parcel elevation above sea level.
- `$slope`: Parcel slope.

### **\<region>.csv**
Located at `<root>/<year>/<level>/`.
Each row represents a parcel and contains the following columns:
- `$idx`: Row number.
- `$id`: Parcel ID (primary key), same as `$ID` in `<region>.shp`.
- `$CODE_CULTU`: Foreign key linking to `codes.csv` (`$crop_DUN_code`) and `classmapping.csv` (`$code`).
- `$path`: Path to the CSV file containing the parcel's timeseries data.
- `$meanCLD`: Average cloud cover percentage (`$QA60` attribute from Sentinel-2).
- `$Sequencelength`: Number of observations (rows) in the parcel's timeseries file.
- `$classid`: Same as `$id` in `classmapping.csv`.
- `$classname`: Same as `$classname` in `classmapping.csv`.
- `$region`: Name of the region.

### **\<parcel_id>.csv**
Located at `<root>/<year>/<level>/<region>/<csv>/`.
Each parcel's timeseries data is stored as a separate CSV file. The filename corresponds to the parcel's ID. Each row represents a Sentinel-2 image acquisition for the parcel, with columns:
- `$nom_id_imatge`: Image identifier in the format "YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_TILE".
- `$BX`: Mean pixel value for band X.
- `$CP`: Cloud percentage for the parcel.
- `$doa`: Acquisition date in `YYYY-MM-DD` format.
- `$Label`: Crop type ID, using the classification codes from `codes.csv`.
- `$id`: Parcel ID, same as the filename.


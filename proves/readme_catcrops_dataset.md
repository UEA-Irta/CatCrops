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
### **Step 2: Describe Each File**

### **Dataset Metadata Files (Located in `<root>`)**

#### **codes.csv**
Defines different crop types according to the <a id="dun"></a>_[Mapa de cultius DUN-SIGPAC](https://agricultura.gencat.cat/ca/ambits/desenvolupament-rural/sigpac/mapa-cultius/index.html)_. Each row contains:
- **$crop_DUN_code**: Crop type code (Primary Key).
- **$crop_DUN_description**: Crop type description (e.g., "barley", "wheat", "corn").
- **$grup_DUN_code**: Group code (aggregating similar crops).
- **$grup_DUN_description**: Group description.

**Example file path:** `/media/hdd11/tipus_c/catcrops_dataset_v2/codes.csv`

#### **classmapping.csv**
Maps DUN crop types to database classification types. Each row contains:
- **$id**: Class ID for crop type.
- **$classname**: Class name (e.g., "panís", "vinya").
- **$code**: Foreign key linking to `$crop_DUN_code` in `codes.csv`.

**Example file path:** `/media/hdd11/tipus_c/catcrops_dataset_v2/classmapping.csv`

#### **sregadiumapping.csv**
Maps irrigation systems according to [DUN](#dun). Each row contains:
- **$sreg_code**: Irrigation system code.
- **$sreg_description**: Irrigation system name.

**Example file path:** `/media/hdd11/tipus_c/catcrops_dataset_v2/sregadiumapping.csv`

#### **varietatmapping.csv**
Maps different crop varieties according to [DUN](#dun). Each row contains:
- **$vari_code**: Variety code.
- **$vari_description**: Variety name.

**Example file path:** `/media/hdd11/tipus_c/catcrops_dataset_v2/varietatmapping.csv`

#### **municipismapping.csv**
Defines municipalities and links them to their region and province. Each row contains:
- **$mun_code**: Municipality code.
- **$mun_description**: Municipality name.
- **$com_code**: Region code.
- **$com_description**: Region name.
- **$prov_code**: Province code.
- **$prov_description**: Province name.

**Example file path:** `/media/hdd11/tipus_c/catcrops_dataset_v2/municipismapping.csv`

### **Dataset Metadata Files (Located in `<root>/<year>`)**
## 📄 Shapefile `<region>.shp` (Located in `<root>/<year>`)

This shapefile contains the polygons of all parcels.

### 📊 Data Structure:
Each row in the shapefile represents a **parcel** with the following columns:

- **`ID`** → Unique parcel identifier (**Primary Key**).
- **`crop_code`** → Crop type code (**Foreign Key** from `codes.csv`).
- **`geometry`** → Polygon defining the parcel’s boundary.

📌 **Example shapefile path:**  
`/media/hdd11/tipus_c/catcrops_dataset_v2/2022/baixter.shp`

---

## 📄 Parcel Data File `<region>_d.csv` (Located in `<root>/<year>`)

This CSV file contains **detailed information** about all parcels.

### 📊 Columns:
Each row corresponds to a **parcel** and includes the following attributes:

- **`ID`** → Unique parcel identifier (**Primary Key**).
- **`crop_code`** → Crop type code (**Foreign Key** from `codes.csv`).
- **`var_code`** → Crop variety code (**Foreign Key** from `varietatmapping.csv`).
- **`pcrop_code`** → Previous year’s crop type (**Foreign Key** from `codes.csv`).
- **`pvar_code`** → Previous year’s crop variety (**Foreign Key** from `varietatmapping.csv`).
- **`sreg_code`** → Irrigation system code (**Foreign Key** from `sregadiumapping.csv`).
- **`mun_code`** → Municipality code (**Foreign Key** from `municipismapping.csv`).
- **`elev`** → Elevation of the parcel (meters above sea level).
- **`slope`** → Parcel slope (terrain inclination).

📌 **Example CSV file path:**  
`/media/hdd11/tipus_c/catcrops_dataset_v2/2022/baixter_d.csv`


### 📊 **Time Series Data Files (Located in `<root>/<year>/<level>/<region>/<csv>`)**

Each CSV file corresponds to a **single parcel**. The filename is the parcel ID.

Each row represents an observation (Sentinel-2 acquisition), and each column represents:
- **$nom_id_imatge**: Image ID (e.g., `"20170102T111442_20170102T111441_T30UWU"`).
- **$BX**: Average pixel values for band X.
- **$CP**: Cloud percentage over the parcel.
- **$doa**: Date in format `YYYY-MM-DD`.
- **$Label**: Crop class ID (from `codes.csv`).
- **$id**: Parcel ID (same as filename).

📌 **Example directory:** `/media/hdd11/tipus_c/catcrops_dataset_v2/2022/L2A/baixter/csv`

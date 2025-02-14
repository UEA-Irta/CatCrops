# CatCrops - Crop Type Classification Library

This repository contains the implementation of CatCrops, which is used in the scientific paper: 
 
"Early Crop Type Classification and Mapping Combining Remote Sensing and Ancillary Data" by Gené-Mola, J., Pamies-Sans, M., Minuesa, C., Casadesús, J., and Bellvert, J. (2025, submitted).
 
The README will be expanded upon the publication of the corresponding research paper.


## Instalation
To ensure proper functionality, we recommend using Anaconda to manage dependencies. Follow these steps to install the required environment and the `CatCrops` library.

#### 1. Install Anaconda  
If you don’t have Anaconda installed, download and install it from the [Anaconda Official Website](https://www.anaconda.com/products/distribution#download-section).  

Once installed, you can verify the installation by running the following command:  
```bash
`conda --version`
```

#### 2. Create and Activate the Conda Environment  
Navigate to the root directory of the repository and create the environment using the `env_catcrops.yml` file:  
```bash
`conda env create -f env_catcrops.yml`  
```

Then, activate the environment:  
```bash
`conda activate catcrops_env`  
```

#### 3. Install `CatCrops` from `setup.py`  
Once the environment is activated, install the `CatCrops` library:  
```bash
`pip install -e .`  
```

This will install `CatCrops` in editable mode, meaning any modifications in the source code will be reflected immediately without needing to reinstall.

#### 4. Verify the Installation  
To confirm that everything is installed correctly, run:  
`python -c "import catcrops; print('CatCrops installed successfully!')"`  

Now, `CatCrops` is ready to use within your Anaconda environment.


## CatCrops Dataset

The **CatCrops Dataset** is hosted on Kaggle and can be accessed through the following link:

[Kaggle Dataset Page](https://www.kaggle.com/datasets/irtaremotesensing/catcrops-dataset)

For direct download, use the following link:

[Download CatCrops Dataset](https://www.kaggle.com/api/v1/datasets/download/irtaremotesensing/catcrops-dataset)

For more information about the dataset structure, refer to **`catcrops_dataset.md`**, which explains how the dataset is organized.

### How to Download and Extract the Dataset Automatically
You can automatically download and extract the dataset using the script `"download_dataset.py"`.

Navigate to the `processing/` folder and execute:
```bash
python download_dataset.py --url "https://www.kaggle.com/api/v1/datasets/download/irtaremotesensing/catcrops-dataset" --zip_path "catcrops_dataset.zip" --extract_folder "./"
```


## Example Code

To run an example of training and testing using the `CatCrops` library, follow these steps. All commands must be 
executed from the `processing/` directory.

#### 1. Download the Dataset  
If you have not yet downloaded the dataset, first run:
```bash
python download_dataset.py --url "https://www.kaggle.com/api/v1/datasets/download/irtaremotesensing/catcrops-dataset" --zip_path "catcrops_dataset.zip" --extract_folder "./"
```
#### 2. Train a Model  
Once the dataset is available, you can train a model using `train.py`:
```bash
python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 -m "evaluation1" -D "./catcrops_dataset/" --weight-decay 5e-08 --learning-rate 1e-3 --preload-ram -l "./RESULTS" --use_previous_year_TS --sparse --cp --doa  --L2A --pclassid --pcrop --pvar --sreg --mun --com --prov --elev --slope --trial "Trial01"
```

#### 3. Run a Test  
After training, you can evaluate the model using `test.py`:
```bash
python test.py --model "TransformerEncoder" --datecrop '31/07/2023' -b 512 -m "test2023" -D "./catcrops_dataset/" --weight-decay 5e-08 --learning-rate 1e-3 --preload-ram -l "./RESULTS" --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg --mun --com --prov --elev --slope --do_shp --trial "Trial01"
```

#### 4. Running a Test Using a Pre-Trained Model  
A pre-trained model is already available in the repository, so if you want to run a test without training a new model, execute:
```bash
python test.py --model "TransformerEncoder" --datecrop '31/07/2023' -b 512 -e 78 -m "test2023" -D "./catcrops_dataset/" --weight-decay 5e-08 --learning-rate 1e-3 --preload-ram -l "./RESULTS" --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg --mun --com --prov --elev --slope --do_shp --trial "Trial00"
```


### Crop Types
The dataset contains the following crop types:

| Abbreviation | Catalan Name           | English Name           |
|-------------|------------------------|------------------------|
| AE          | Altres extensius        | Other extensive        |
| AL          | Alfals                  | Alfalfa                |
| AR          | Arròs                   | Rice                   |
| CB          | Cebes                   | Onion                  |
| CH          | Cereals d'hivern        | Winter cereals         |
| C           | Colza                   | Rapeseed               |
| CP          | Cultius permanents      | Orchards               |
| DCP         | Dc panís                | Double Crop Maize      |
| DCGS        | Dc gira-sol             | Double Crop Sunflower  |
| G           | Guaret                  | Fallow                 |
| GS          | Gira-sol                | Sunflower              |
| H           | Horta                   | Vegetables             |
| OL          | Oliverar                | Olives                 |
| P           | Panís                   | Maize                  |
| PR          | Protaginoses            | Fabaceae               |
| RF          | Ray-grass festuca       | Ray Grass and Festuca  |
| SJ          | Soja                    | Soybean                |
| SR          | Sorgo                   | Sorghum                |
| VÇ          | Veça                    | Vetch                  |
| VV          | Vivers                  | Nurseries              |
| VY          | Vinya                   | Vineyard               |


## Interactive Map

You can explore the crop classification results on the interactive map:  
🔗 [CatCrops Interactive Map](https://catcrops2023.irtav7.cat/)



## Reference
"Early Crop Type Classification and Mapping Combining Remote Sensing and Ancillary Data" by Gené-Mola, J., Pamies-Sans, M., Minuesa, C., Casadesús, J., and Bellvert, J. (2025, submitted).

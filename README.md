# catcrops

Repositori per l'article de Classificació de Cultius 


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


## CatCrops Dataset
L'estructura del Dataset

*Dataset folder structure*
    <root>                          # Podeu veure un exemple a /media/hdd11/tipus_c/catcrops_dataset_v2
       codes.csv
       classmapping.csv
       sregadiumapping.csv
       varietatmapping.csv
       municipismapping.csv
       <year>
          <region>.shp
          <region>_d.shp
          <level>                   # L1C o L2A
             <region>.csv
             <region>.h5
             <region>
                 <csv>
                     123123.csv     # Taula Timeseries de la parcela 123123
                     123125.csv
                     ...

*Arxiu codes.csv (ubicat a <root>)*
    Cada fila defineix un tipus de cultiu diferent segons la DUN
    Per cada fila hi ha 4 columnes: $crop_code, $crop_description, $grup_code i $grup_desription.
        $crop_DUN_code: Codi del tipus de cultiu (Clau primaria). En el cas del BriezhCrops es un codi de 3 caràcters. En el nostre cas, podem posar el codi numèric del cultiu segons la DUN.
        $crop_DUN_description: Descripció del tipus de cultíu. Per exemple, "barley", "wheat" or "corn". En el nostre cas utilitzar el nom de la DUN
        $grup_DUN_code: Codi del grup. Diferents cultius es poden agrupar en únic grup. Posar un valor numèric com a ID de cada grup de la DUN.
        $grup_DUN_description: Descripció del grup de la DUN al qual pertany el tipus de cultiu.
    Al següent directori es pot veure un exemple de un arxiu codes.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/codes.csv"

*Arxiu classmaping.csv (ubicat a <root>)*
    Cada fila defineix un tipus de cultiu diferent segons la DUN i el relacióna amb el tipus de classe de la base de dades
    Per cada fila hi ha 4 columnes: $, $id, $classname i $code.
        $: No te nom a la capcelera. El valor d'aqueta columna correspon al número de fila de la taula.
        $id: ID de la classe a la que pertany el tipus de cultiu de la columna $code
        $classname: Nom de la classe. Per exemple "barley", "wheat", "corn", "permanent crop"...
        $code: Clau forana de l'atribut $crop_DUN_code de la taula codes.csv
    Al següent directori es pot veure un exemple de un arxiu classmaping.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/classmaping.csv"

*Arxiu sregadiumapping.csv (ubicat a <root>)*
    Cada fila defineix un sistema de regadiu diferent segons la DUN i el relacióna amb el seu nom
    Per cada fila hi ha 2 columnes: $sreg_code i $sreg_description.
        $reg_code: És el codi del sistema de reg
        $sreg_description: És el nom del sistema de reg
    Al següent directori es pot veure un exemple de un arxiu sregadiumapping.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/sregadiumapping.csv"

*Arxiu varietatmapping.csv (ubicat a <root>)*
    Cada fila defineix una varietat de cultiu diferent segons la DUN i el relacióna amb el seu nom
    Per cada fila hi ha 2 columnes: $vari_code i $vari_description.
        $vari_code: És el codi de la varietat
        $vari_description: És el nom de la varietat (nom de la DUN)
    Al següent directori es pot veure un exemple de un arxiu varietatmapping.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/varietatmapping.csv"

*Arxiu municipismapping.csv (ubicat a <root>)*
    Cada fila defineix un municipi diferent i el relacióna amb la comarca i la província.
    Per cada fila hi ha 6 columnes: $mun_code, $mun_description, $com_code, $com_description, $prov_code i $prov_description.
        $mun_code: És el codi del municipi
        $mun_description: És el nom del municipi
        $com_code: és el codi de la comarca
        $com_description: és el nom de la comarca
        $prov_code: és el codi de la província
        $prov_description: és el nom de la província
    Al següent directori es pot veure un exemple de un arxiu municipismapping.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/municipismapping.csv"

*Arxiu <region>.shp (ubicat a <root>/<year>)*
    Arxiu shp amb els poligons de totes les parceles.
    En llegir aquest arxiu shp amb les comandes:
        import geopandas as gpd
        gdf = gpd.read_file(<region>.shp)
    es crea un geodataframe "gdf" on cada fila es una parcela i les columnes son el $ID, $crop_code i $geometry:
        $ID: es el identificador de la parcela (clau primaria)
        $crop_code: es el codi del tipus de cultiu, clau forana de l'atribut $crop_DUN_code de la taula codes.csv
        $geometry: es el poligon que defineix el perimetre de la parcela.
    Al següent directori es pot veure un exemple d'un arxiu <region>.shp: "/media/hdd11/tipus_c/catcrops_dataset_v2/2022/baixter.shp"

*Arxiu <region>_d.csv (ubicat a <root>/<year>)*
    Arxiu csv amb informació de  de totes les parceles. Cada fila defineix una parcela.
    Per cada fila hi ha 9 columnes:
        $ID: es el identificador de la parcela (clau primaria)
        $crop_code: es el codi del tipus de cultiu, clau forana de l'atribut $crop_DUN_code de la taula codes.csv
        $var_code: és el codi de la varietat del cultiu, clau forana de l'atribut $vari_code de la taula varietatmapping.csv
        $pcrop_code: és el codi del tipus de cultiu de l'any anterior. clau forana de l'atribut $crop_DUN_code de la taula codes.csv
        $pvar_code: és el codi de la varieat del cultiu de l'any anterior, clau forana de l'atribut $vari_code de la taula varietatmapping.csv
        $sreg_code: és el codi del sistema de regadiu, clau forana de l'atribut $sreg_code de la taula sregadiumapping.csv
        $mun_code: és el codi del municipi, clau forana de l'atribut $mun_code de la taula municipismapping.csv
        $elev: és la altura de la parcel·la, en relació al nivell del mar
        $slope: és la pendent de la parcel·la
    Al següent directori es pot veure un exemple d'un arxiu <region>_d.csv: "/media/hdd11/tipus_c/catcrops_dataset_v2/2022/baixter_d.csv"

*Arxiu <region>.csv (ubicat a <root>/<year>/<level>)*
    Cada fila defineix una parcela
    Per cada fila hi ha 9 columnes:
        $idx: nº de fila
        $id: ID de cada parcela (clau primaria). Mateix identificador que el $ID de l'arxiu <region>.shp.
        $CODE_CULTU: Clau forana de codes.csv (atribut $crop_DUN_code) i classmaping.csv (atribut $code).
        $path: Path a arxiu csv amb la Taula Timeseries de cada parcela.
        $meanCLD: Mitjana de l'atribut $QA60 de la Taula Timeseries de cada parcela.
        $Sequencelength: Nº de observacions (files) a la Taula Timeseries de cada parcela.
        $classid: El mateix que l'atribut $id de la taula classmapping.csv
        $classname: El mateix que l'atribut $classname de la taula classmapping.csv
        $region: el mateix que el nom de l'arxiu.
    Al següent directori es pot veure un exemple d'un arxiu <region>.csv: "/media/hdd11/jordi_g/code/230418-crop_classification/CatCrops/breizhcrops_dataset/2017/L1C/frh03.csv"

*Taula Timeseries (ubicades a <root>/<year>/<level>/<region>/<csv>)*
    Es guarda en un arxiu csv (una taula) per cada parcela
    El nom de l'arxiu csv serà el número ID de la parcela
    Cada fila correspon a una adquisició d'imatge de sentinel (una data en concret)
    Cada columna correspon a $nom_id_imatge, $B1, $B10, $B11, $B12, $B2, $B3, $B4, $B5, $B6, $B7, $B8, $B9, $CP, $doa, $label, $id
        $nom_id_imatge: Segint la mateixa estructura que el següent exemple: "20170102T111442_20170102T111441_T30UWU"
        $BX: Mitjana dels valors que prenen pixels de la parcela a la banda X
        $CP: Percentatge de nuvolositat de la parcela
        $doa: data en el format YYYY-MM-DD
        $Label: ID del tipus de cultiu. Al dataset CatCrops utilitza els codis descrits a l'arxiu "/media/hdd11/tipus_c/catcrops_dataset_v2/codes.csv"
        $id: Número ID de la parcela. Es el mateix que el nom de la taula (arxiu csv).
    Al següent directori es poden veure diferents taules Timeseries d'exemple: /media/hdd11/tipus_c/catcrops_dataset_v2/2022/L2A/baixter/csv"


## Interactive Map

You can explore the crop classification results on the interactive map:  
🔗 [CatCrops Interactive Map](https://catcrops2023.irtav7.cat/)


## Interactive Map - Crop Classification Results 🌍

The classification results of the article can be explored using the interactive map below:  
🔗 **[CatCrops Interactive Map](https://catcrops2023.irtav7.cat/)**  
This map allows you to visualize the different crop types and their spatial distribution.


## Reference

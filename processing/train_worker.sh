#!/usr/bin/env bash


source ~/anaconda3/bin/activate catcrops_env  # entorn anaconda
#Example with all possible parsing arguments

python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
        -m "evaluation1" -D "./catcrops_dataset/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Trial01-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"


python test.py --model "TransformerEncoder" --datecrop 'all' -b 512 \
        -m "test2023" -D "./catcrops_dataset/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --do_shp --trial "Trial01-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"



python test.py --model "TransformerEncoder" --datecrop 'all' -b 512 \
        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#!/usr/bin/env bash


source ~/anaconda3/bin/activate catcrops_env  # entorn anaconda
#Example with all possible parsing arguments
python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 180 \
        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --trial "Trial056-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval2"

python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --trial "Trial056-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval2"

python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Prova_workstation-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s_eval3"
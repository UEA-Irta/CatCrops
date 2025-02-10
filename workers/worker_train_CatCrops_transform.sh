#!/usr/bin/env bash

#Example with all possible parsing arguments
#python train.py --model "TransformerEncoder" -SL 45 --datecrop 'random' -b 1024 -e 50 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P05-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --pclassid --pcrop --pvar --sreg \
#          --mun --com --prov --elev --slope --trial "Trial001-SL_doa_datecrop_cp_221230"

#python train.py --model "TransformerEncoder"  --datecrop 'random' -b 1024 -e 50 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 -l "/media/hdd11/tipus_c/proves_article/P05-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --pclassid --pcrop --pvar --sreg \
#          --mun --com --prov --elev --slope --trial "Trial001-rCrop_py_s_cp_doa_L2A_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"

#python train.py --model "TransformerEncoder" -SL 45 --datecrop '31/12/2023' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#          --L2A  --trial "Trial002-Crop231231_SL_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --sparse --L2A --trial "Trial003-rCrop_s_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --sparse --cp --L2A --trial "Trial004-rCrop_s_cp_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --L2A --trial "Trial005-rCrop_py_s_cp_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 50 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --trial "Trial006-rCrop_py_s_cp_doa_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --ET --trial "Trial007-rCrop_py_s_cp_doa_ET"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 1024 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --trial "Trial008-rCrop_py_s_cp_doa_L2A_ET"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --elev --slope --trial "Trial009-rCrop_py_s_cp_doa_L2A_ET_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --mun --com --prov --elev --slope \
#          --trial "Trial010-rCrop_py_s_cp_doa_L2A_ET_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 100 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --use_previous_year_TS --sparse --cp --doa --L2A --ET --sreg --mun --com --prov --elev --slope \
#          --trial "Trial011-rCrop_py_s_cp_doa_L2A_ET_reg_mun_com_pro_e_s"


#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 250 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram --wight_sampling -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --LST --ET --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial020-rCrop_py_s_cp_doa_L2A_LST_ET_WS_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#

#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --LST --ET --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial018-rCrop_py_s_cp_doa_LST_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --ET --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial017-rCrop_py_s_cp_doa_L2A_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --LST --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial016-rCrop_py_s_cp_doa_L2A_LST_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --ET --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial015-rCrop_py_s_cp_doa_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --LST --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial014-rCrop_py_s_cp_doa_LST_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial013-rCrop_py_s_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 250 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --LST --ET --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial019-rCrop_py_s_cp_doa_L2A_LST_ET_pclass_pcrop_pvar_reg_mun_com_pro_e_s"

#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A --sreg \
#        --mun --com --prov --elev --slope --trial "Trial031-rCrop_py_s_cp_doa_L2A_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A \
#        --mun --com --prov --elev --slope --trial "Trial030-rCrop_py_s_cp_doa_L2A_mun_com_pro_e_s"
#
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A \
#        --elev --slope --trial "Trial029-rCrop_py_s_cp_doa_L2A_e_s"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --cp --doa --L2A \
#        --trial "Trial028-rCrop_py_s_cp_doa_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --doa --L2A \
#        --trial "Trial027-rCrop_py_s_doa_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --L2A \
#        --trial "Trial026-rCrop_py_s_L2A"
#
#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --sparse --L2A \
#        --trial "Trial025-rCrop_s_L2A"

#python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 500 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --sparse --wight_sampling --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial035-rCrop_py_s_cp_doa_L2A_WS_pclass_pcrop_pvar_reg_mun_com_pro_e_s"

#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#         --L2A --trial "Trial001-baseline_random"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial047-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --L2A \
#        --trial "Trial041-rCrop_py_sl70_nr_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --noreplace --L2A \
#        --trial "Trial040-rCrop_sl70_nr_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --sreg \
#        --mun --com --prov --elev --slope --trial "Trial046-rCrop_py_sl70_nr_cp_doa_L2A_reg_mun_com_pro_e_s"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --mun --com --prov --elev --slope --trial "Trial045-rCrop_py_sl70_nr_cp_doa_L2A_mun_com_pro_e_s"
#
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --elev --slope --trial "Trial044-rCrop_py_sl70_nr_cp_doa_L2A_e_s"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --trial "Trial043-rCrop_py_sl70_nr_cp_doa_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P06-CatCrops/RESULTS" \
#        --use_previous_year_TS --noreplace --doa --L2A \
#        --trial "Trial042-rCrop_py_sl70_nr_doa_L2A"
#
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --trial "Trial055-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --sreg \
#        --trial "Trial054-rCrop_py_sl70_nr_cp_doa_L2A_reg"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --trial "Trial053-rCrop_py_sl70_nr_cp_doa_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --doa --L2A \
#        --trial "Trial052-rCrop_py_sl70_nr_doa_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --L2A \
#        --trial "Trial051-rCrop_py_sl70_nr_L2A"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --noreplace --L2A \
#        --trial "Trial050-rCrop_sl70_nr_L2A"

python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 180 \
        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --trial "Trial056-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval2"

python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --trial "Trial056-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval2"

#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 180 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --trial "Trial057-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --trial "Trial057-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval3"

#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 180 \
#        -m "evaluation4" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --trial "Trial058-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_eval4"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial064-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
#        --mun --com --prov --elev --slope --trial "Trial064-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --sreg \
#        --trial "Trial063-rCrop_py_sl70_nr_cp_doa_L2A_reg_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --sreg \
#        --trial "Trial063-rCrop_py_sl70_nr_cp_doa_L2A_reg_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --trial "Trial062-rCrop_py_sl70_nr_cp_doa_L2A_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A \
#        --trial "Trial062-rCrop_py_sl70_nr_cp_doa_L2A_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --doa --L2A \
#        --trial "Trial061-rCrop_py_sl70_nr_doa_L2A_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --doa --L2A \
#        --trial "Trial061-rCrop_py_sl70_nr_doa_L2A_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation3" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --L2A \
#        --trial "Trial060-rCrop_py_sl70_nr_L2A_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --L2A \
#        --trial "Trial060-rCrop_py_sl70_nr_L2A_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
#        -m "evaluation2" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --noreplace --L2A \
#        --trial "Trial059-rCrop_sl70_nr_L2A_eval3"
#
#python test.py --model "TransformerEncoder" -SL 70 --datecrop 'all' -b 512 \
#        -m "test2023" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --noreplace --L2A \
#        --trial "Trial059-rCrop_sl70_nr_L2A_eval3"
#
#python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 150 \
#        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
#        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
#        --use_previous_year_TS --noreplace --cp --doa --L2A --sreg \
#        --trial "Trial070-rCrop_py_sl70_nr_cp_doa_L2A_reg_eval4"
#

python train.py --model "TransformerEncoder" -SL 70 --datecrop 'random' -b 512 -e 120 \
        -m "evaluation1" -D "/media/hdd11/tipus_c/catcrops_dataset_v2/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "/media/hdd11/tipus_c/proves_article/P09-CatCrops_replacement/RESULTS" \
        --use_previous_year_TS --noreplace --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Prova_workstation-rCrop_py_sl70_nr_cp_doa_L2A_pclass_pcrop_pvar_reg_mun_com_pro_e_s_eval3"
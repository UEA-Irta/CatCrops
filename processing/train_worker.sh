#!/usr/bin/env bash


source ~/anaconda3/bin/activate catcrops_env  # entorn anaconda
#Example with all possible parsing arguments

python train.py --model "TransformerEncoder" --datecrop 'random' -b 512 -e 120 \
        -m "evaluation1" -D "./catcrops_dataset/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Trial01"


python test.py --model "TransformerEncoder" --datecrop '31/07/2023' -b 512 \
        -m "test2023" -D "./catcrops_dataset/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --do_shp --trial "Trial01"



python test.py --model "TransformerEncoder" --datecrop '31/07/2023' -b 512 -e 75\
        -m "test2023" -D "./catcrops_dataset/" --weight-decay 5e-08 \
        --learning-rate 1e-3 --preload-ram -l "./RESULTS" \
        --use_previous_year_TS --sparse --cp --doa --L2A --pclassid --pcrop --pvar --sreg \
        --mun --com --prov --elev --slope --trial "Trial00"

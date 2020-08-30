#!/bin/sh
#PBS -N cnn_deep_3_run
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_test
#PBS -o output_test_cnn_deep
#PBS -l walltime=100:00:00
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/test_cnn_deep.py

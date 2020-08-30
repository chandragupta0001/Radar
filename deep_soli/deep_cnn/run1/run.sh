#!/bin/sh
#PBS -N test_cnn_deep_run1
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_test_cnn_deep_run1
#PBS -o output_test_cnn_deep_run1
#PBS -q low
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/run1/test_cnn_deep.py

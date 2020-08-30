#!/bin/sh
#PBS -N cnn_deep_2_run
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_cnn_deep_2_run
#PBS -o output_cnn_deep_2_run
#PBS -l walltime=100:00:00
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/2_run/cnn_deep.py

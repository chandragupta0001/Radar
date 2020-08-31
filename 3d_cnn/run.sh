#!/bin/sh
#PBS -N 3dcnn
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_cnn
#PBS -o output_cnn
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/3d_cnn/1_run/3d_cnn.py

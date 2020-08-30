#!/bin/sh
#PBS -N cnn2
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_cnn2
#PBS -o output_cnn2

module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/cnn_without_pooling/cnn2.py

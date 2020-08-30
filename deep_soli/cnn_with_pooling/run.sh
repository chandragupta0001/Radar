#!/bin/sh
#PBS -N cnn_1
#PBS -P ee
#PBS -m bea
#PBS -M $eet192341@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_cnn_1
#PBS -o output_cnn_1

module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/cnn_with_pooling/cnn_1.py

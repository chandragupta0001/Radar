#!/bin/sh
#PBS -N ete
#PBS -P ee
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -e error_ete
#PBS -o output_ete
#PBS -q low
#PBS -l walltime=100:00:00
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/end_to_end/cnn_lstm.py

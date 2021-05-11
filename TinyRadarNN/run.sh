#!/bin/sh
#PBS -N custom
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e rerun/error
#PBS -o rerun/output
#PBS -q low
module load apps/anaconda/3
python /home/ee/mtech/eet192341/tinyradar/doppler/cnn_tcn/2_model/cnn_tcn.py

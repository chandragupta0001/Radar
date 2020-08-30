#!/bin/sh
#PBS -N rnn_deep
#PBS -P ee
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -e error_rnn
#PBS -o output_rnn
#PBS -q low
#PBS -l walltime=140:00:00
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/rnn_deep/rnn_deep.py

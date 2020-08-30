#!/bin/sh
#PBS -N test
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_test
#PBS -o output_test
#PBS -q low
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/rnn_deep/test/rnn_deep_test.py

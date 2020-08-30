#!/bin/sh
#PBS -N ete1
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_ete_test
#PBS -o output_ete_test
#PBS -q low
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/end_to_end/1_run/test.py

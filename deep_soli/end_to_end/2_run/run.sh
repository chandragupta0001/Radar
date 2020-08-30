#!/bin/sh
#PBS -N ete2
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e error_ete_test
#PBS -o output_ete_test
#PBS -q low
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/end_to_end/2_run/test.py

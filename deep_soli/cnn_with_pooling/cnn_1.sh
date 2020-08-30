#!/bin/sh
### Set the job name
#PBS -N cnn_1
### Set the project name, your department dc by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M eet192341@ee.iitd.ac.in
####
#PBS -l select=1:ncpus=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=40:00:00
#PBS -q low
#PBS -o cnn_1_output
#PBS -e cnn_1_error
#xyz
#### Get environment variables from submitting shell
 
cd $PBS_O_WORKDIR 
module load apps/anaconda/3
python /home/ee/mtech/eet192341/codes/deep_soli/cnn_with_pooling/cnn_1.sh

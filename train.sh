#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --exclude='biwirender12','biwirender08','biwirender05'
#SBATCH  --priority='TOP'

# activate virtual environment
source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/bin/activate tf_v1_15

## EXECUTION OF PYTHON CODE:
python /usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/code/code/hpc-predict/segmenter/cnn_segmenter_for_mri_4d_flow/train.py \
--training_input '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/training_data.hdf5' \
--training_output '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/code/code/hpc-predict/segmenter/cnn_segmenter_for_mri_4d_flow/logdir/'

echo "Hostname was: `hostname`"
echo "Reached end of job file."
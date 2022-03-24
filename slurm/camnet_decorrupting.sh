#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=1-4
#SBATCH --time=6-23:59:59       # Time: D-H:M:S
#SBATCH --account=rrg-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=32G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=10      # Number of cores per task.
#SBATCH --gres=gpu:v100l:2      # 32G V100

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=job_results/march23_%j.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tristanengst@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Print some info for context.
pwd
hostname
date

echo "Starting job number $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate py39ISICLE

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

echo "SLURM CHECKPOINT A"
wandb login 90d0248dd4a9fde052b760cdb131373d49b4dad8
echo "SLURM CHECKPOINT B"

# Do all the research.
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]
then
    python TrainGenerator.py --epochs 40 --data_folder_path ~/scratch/ISICLE/data --n_levels 2 --res 64 64 128 --ns 256 64 --data camnet3 --gpus 0 1 --grayscale 1 --pix_mask_frac 0
elif [ "$SLURM_ARRAY_TASK_ID" = 2 ]
then
    python TrainGenerator.py --epochs 40 --data_folder_path ~/scratch/ISICLE/data --n_levels 2 --res 64 64 128 --ns 256 64 --data camnet3 --gpus 0 1 --grayscale 1 --pix_mask_frac .5 --pix_mask_size 8
elif [ "$SLURM_ARRAY_TASK_ID" = 3 ]
then
    python TrainGenerator.py --epochs 20 --data_folder_path ~/scratch/ISICLE/data --n_levels 2 --res 64 64 128 --ns 256 64 --data miniImagenet_deci --gpus 0 1 --grayscale 1 --pix_mask_frac 0
elif [ "$SLURM_ARRAY_TASK_ID" = 4 ]
then
    python TrainGenerator.py --epochs 40 --data_folder_path ~/scratch/ISICLE/data --n_levels 2 --res 64 64 128 --ns 256 64 --data miniImagenet_deci --gpus 0 1 --grayscale 1 --pix_mask_frac 0
elif [ "$SLURM_ARRAY_TASK_ID" = 5 ]
then
    python TrainGenerator.py --epochs 20 --data_folder_path ~/scratch/ISICLE/data --res 64 64 64 64 128 --ns 256 64 64 64 --data camnet3 --gpus 0 1 --grayscale 1 --pix_mask_frac .6 --pix_mask_size 4
elif [ "$SLURM_ARRAY_TASK_ID" = 6 ]
then
    python TrainGenerator.py --epochs 20 --data_folder_path ~/scratch/ISICLE/data --res 64 64 64 64 128 --ns 256 64 64 64 --data camnet3 --gpus 0 1 --grayscale 0 --pix_mask_frac .6 --pix_mask_size 8
else
    echo "No case here"
fi

# Print completion time.
date
